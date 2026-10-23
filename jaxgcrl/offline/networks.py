# Adapted from OGBench (https://github.com/seohongpark/ogbench),
# Copyright (c) 2024 OGBench Authors, MIT License.
# Modified for this repository in 2026.
"""OGBench reference networks and their hyperbolic counterparts (shape suffixes: B batch, A action, D latent, E ensemble)."""

import jax
import jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers import HyperPPFeatureScaling

from jaxgcrl import hyperbolic as hyp
from jaxgcrl.online.networks import euclidean_energy


def default_init(scale=1.0):
    return nnx.initializers.variance_scaling(scale, "fan_avg", "uniform")


class OGBMLP(nnx.Module):
    """Reference MLP: Dense -> GELU -> (LayerNorm) per activated layer."""

    def __init__(self, in_features, hidden_dims, *, activate_final=False, layer_norm=False, rngs: nnx.Rngs):
        dims = (in_features, *hidden_dims)
        self.layers = nnx.List(
            [nnx.Linear(dims[i], dims[i + 1], kernel_init=default_init(), rngs=rngs) for i in range(len(hidden_dims))]
        )
        self.activate_final = activate_final
        num_activated = len(hidden_dims) - 1 + (1 if activate_final else 0)
        self.layer_norms = (
            nnx.List([nnx.LayerNorm(dims[i + 1], rngs=rngs) for i in range(num_activated)]) if layer_norm else None
        )

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = jax.nn.gelu(x)
                if self.layer_norms is not None:
                    x = self.layer_norms[i](x)
        return x


def gaussian_log_prob(means_BA, stds_BA, actions_BA):
    return jax.scipy.stats.norm.logpdf(actions_BA, means_BA, stds_BA).sum(axis=-1)


def gaussian_sample(means_BA, stds_BA, key):
    return means_BA + stds_BA * jax.random.normal(key, means_BA.shape)


class GCActor(nnx.Module):
    """Goal-conditioned Gaussian actor returning (mean, std); actions are clipped, not squashed."""

    def __init__(
        self,
        obs_dim,
        goal_dim,
        action_dim,
        hidden_dims=(512, 512, 512),
        *,
        log_std_min=-5.0,
        log_std_max=2.0,
        const_std=True,
        final_fc_init_scale=1e-2,
        rngs: nnx.Rngs,
    ):
        trunk_in = obs_dim + goal_dim
        self.actor_net = OGBMLP(trunk_in, hidden_dims, activate_final=True, rngs=rngs)
        self.mean_net = nnx.Linear(hidden_dims[-1], action_dim, kernel_init=default_init(final_fc_init_scale), rngs=rngs)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.const_std = const_std
        if not const_std:
            self.log_stds = nnx.Param(jnp.zeros((action_dim,), dtype=jnp.float32))

    def features(self, x):
        return self.actor_net(x)

    def __call__(self, observations_BO, goals_BG=None, temperature=1.0):
        inputs_BI = observations_BO if goals_BG is None else jnp.concatenate([observations_BO, goals_BG], axis=-1)
        means_BA = self.mean_net(self.features(inputs_BI))
        log_stds = jnp.zeros_like(means_BA) if self.const_std else self.log_stds[...]
        stds = jnp.exp(jnp.clip(log_stds, self.log_std_min, self.log_std_max)) * temperature
        return means_BA, stds


class HypGCActor(GCActor):
    """GCActor whose features pass through Hyper++ -> expmap_0 -> HTC layers -> logmap_0."""

    def __init__(
        self,
        obs_dim,
        goal_dim,
        action_dim,
        hidden_dims=(512, 512, 512),
        *,
        repr_dim=64,
        depth=1,
        curvature=1.0,
        manifold_float64=False,
        log_std_min=-5.0,
        log_std_max=2.0,
        const_std=True,
        final_fc_init_scale=1e-2,
        rngs: nnx.Rngs,
    ):
        trunk_in = obs_dim + goal_dim
        self.actor_net = OGBMLP(trunk_in, (*hidden_dims, repr_dim), rngs=rngs)
        self.curvature = float(curvature)
        self.manifold = Hyperboloid(dtype=jnp.float64 if manifold_float64 else jnp.float32)
        self.bridge = HyperPPFeatureScaling(repr_dim, alpha=0.95, rngs=rngs)
        self.hyp_layers = nnx.List(
            [hyp.HypLinear("hyperboloid", self.manifold, repr_dim, repr_dim, rngs=rngs) for _ in range(depth)]
        )
        self.mean_net = nnx.Linear(repr_dim, action_dim, kernel_init=default_init(final_fc_init_scale), rngs=rngs)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.const_std = const_std
        if not const_std:
            self.log_stds = nnx.Param(jnp.zeros((action_dim,)))
        hyp.ensure_params_f32(self)

    def features(self, x):
        c = self.curvature
        h = self.actor_net(x)
        batch_shape = h.shape[:-1]
        h_NW = self.bridge(h.reshape(-1, h.shape[-1]), c=c)
        p_NWp1 = hyp.project_to_manifold(h_NW, "hyperboloid", self.manifold, c)
        p_NWp1 = hyp.apply_hyp_stack("hyperboloid", self.hyp_layers, p_NWp1, c)
        v_NWp1 = jax.vmap(self.manifold.logmap_0, in_axes=(0, None))(p_NWp1, c)
        return v_NWp1[..., 1:].reshape(*batch_shape, -1)


class HypGCEncoder(nnx.Module):
    """OGBench MLP trunk -> expmap_0 -> hyp_depth hyperbolic linear layers."""

    def __init__(self, in_features, hidden_dims, latent_dim, *, layer_norm, manifold_name, manifold, hyp_depth, rngs):
        self.trunk = OGBMLP(in_features, (*hidden_dims, latent_dim), layer_norm=layer_norm, rngs=rngs)
        self.manifold_name = manifold_name
        self.manifold = manifold
        self.hyp_layers = nnx.List(
            [hyp.HypLinear(self.manifold_name, manifold, latent_dim, latent_dim, rngs=rngs) for _ in range(hyp_depth)]
        )

    def __call__(self, x, c):
        h = self.trunk(x)
        h = hyp.project_to_manifold(h, self.manifold_name, self.manifold, c)
        return hyp.apply_hyp_stack(self.manifold_name, self.hyp_layers, h, c)


class GCBilinearCritic(nnx.Module):
    """Ensembled contrastive critic Q_e(s, a, g) = energy(phi_e(s, a), psi_e(g)) + bias."""

    def __init__(
        self,
        obs_dim,
        goal_dim,
        action_dim,
        hidden_dims=(512, 512, 512),
        latent_dim=512,
        *,
        layer_norm=True,
        num_ensembles=2,
        energy_fn_name="dot",
        use_logit_bias=False,
        rngs: nnx.Rngs,
    ):
        phi_in = obs_dim + action_dim
        psi_in = goal_dim
        self.phis = nnx.List([self._encoder(phi_in, hidden_dims, latent_dim, layer_norm, rngs) for _ in range(num_ensembles)])
        self.psis = nnx.List([self._encoder(psi_in, hidden_dims, latent_dim, layer_norm, rngs) for _ in range(num_ensembles)])
        self.latent_dim = latent_dim
        self.energy_fn_name = energy_fn_name
        self.logit_bias = nnx.Param(jnp.zeros(())) if use_logit_bias else None

    def _encoder(self, in_features, hidden_dims, latent_dim, layer_norm, rngs):
        return OGBMLP(in_features, (*hidden_dims, latent_dim), layer_norm=layer_norm, rngs=rngs)

    def _bias(self):
        return self.logit_bias[...] if self.logit_bias is not None else 0.0

    def _embed(self, encoder, x):
        return encoder(x)

    def encode(self, observations_BO, goals_BG, actions_BA):
        phi_in = jnp.concatenate([observations_BO, actions_BA], axis=-1)
        phi_EBD = jnp.stack([self._embed(phi, phi_in) for phi in self.phis])
        psi_EBD = jnp.stack([self._embed(psi, goals_BG) for psi in self.psis])
        return phi_EBD, psi_EBD

    def diagonal(self, phi_EBD, psi_EBD):
        if self.energy_fn_name == "dot":
            return (phi_EBD * psi_EBD).sum(axis=-1) / jnp.sqrt(self.latent_dim)
        return euclidean_energy(self.energy_fn_name, phi_EBD, psi_EBD)

    def pairwise(self, phi_EBD, psi_EBD):
        if self.energy_fn_name == "dot":
            return jnp.einsum("ebd,ecd->ebc", phi_EBD, psi_EBD) / jnp.sqrt(self.latent_dim)
        return euclidean_energy(self.energy_fn_name, phi_EBD[:, :, None, :], psi_EBD[:, None, :, :])

    def __call__(self, observations_BO, goals_BG, actions_BA, info=False):
        phi_EBD, psi_EBD = self.encode(observations_BO, goals_BG, actions_BA)
        v_EB = self.diagonal(phi_EBD, psi_EBD) + self._bias()
        return (v_EB, phi_EBD, psi_EBD) if info else v_EB

    def pairwise_logits(self, phi_EBD, psi_EBD):
        return self.pairwise(phi_EBD, psi_EBD) + self._bias()


class HypGCBilinearCritic(GCBilinearCritic):
    """GCBilinearCritic with hyperbolic encoders and a geodesic or Minkowski energy at fixed curvature."""

    def __init__(
        self,
        obs_dim,
        goal_dim,
        action_dim,
        hidden_dims=(512, 512, 512),
        latent_dim=512,
        *,
        layer_norm=True,
        num_ensembles=2,
        manifold="hyperboloid",
        curvature=1.0,
        hyp_energy_fn="geodesic",
        temperature=1.0,
        hyp_depth=1,
        manifold_float64=False,
        use_logit_bias=True,
        rngs: nnx.Rngs,
    ):
        self.curvature = float(curvature)
        self.manifold = hyp.make_manifold(manifold, manifold_float64)
        self.hyp_energy_fn = hyp_energy_fn
        self.temperature = temperature
        self._hyp_kwargs = dict(manifold_name=manifold, manifold=self.manifold, hyp_depth=hyp_depth)
        super().__init__(
            obs_dim,
            goal_dim,
            action_dim,
            hidden_dims,
            latent_dim,
            layer_norm=layer_norm,
            num_ensembles=num_ensembles,
            use_logit_bias=use_logit_bias,
            rngs=rngs,
        )
        del self._hyp_kwargs
        hyp.ensure_params_f32(self)

    def _encoder(self, in_features, hidden_dims, latent_dim, layer_norm, rngs):
        return HypGCEncoder(in_features, hidden_dims, latent_dim, layer_norm=layer_norm, **self._hyp_kwargs, rngs=rngs)

    def _embed(self, encoder, x):
        return encoder(x, self.curvature)

    def diagonal(self, phi_EBD, psi_EBD):
        return jnp.stack(
            [
                hyp.diagonal_energy(
                    self.hyp_energy_fn, phi_EBD[e], psi_EBD[e], self.manifold, self.curvature, self.temperature
                )
                for e in range(len(self.phis))
            ]
        )

    def pairwise(self, phi_EBD, psi_EBD):
        return jnp.stack(
            [
                hyp.pairwise_energy(
                    self.hyp_energy_fn, phi_EBD[e], psi_EBD[e], self.manifold, self.curvature, self.temperature
                )
                for e in range(len(self.phis))
            ]
        )
