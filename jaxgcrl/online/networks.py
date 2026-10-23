# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL) and
# scaling-crl (https://github.com/wang-kevin3290/scaling-crl), Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
"""Deep residual networks ("1000 Layer Networks for Self-Supervised RL", arXiv:2503.14858) with optional hyperbolic
critic heads (B batch, W width, A action)."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import variance_scaling, zeros

from jaxgcrl import hyperbolic as hyp

# Dense layers per residual block; network depth must be a multiple of it.
REPEATS_PER_BLOCK = 4

_LECUN_UNIFORM = variance_scaling(1 / 3, "fan_in", "uniform")


def _dense(in_features, out_features, rngs):
    return nnx.Linear(in_features, out_features, kernel_init=_LECUN_UNIFORM, bias_init=zeros, rngs=rngs)


def _activation(use_relu: bool):
    return jax.nn.relu if use_relu else jax.nn.swish


class ResidualBlock(nnx.Module):
    """REPEATS_PER_BLOCK x [Dense -> LayerNorm -> act], then a skip connection."""

    def __init__(self, width: int, use_relu: bool = False, *, rngs: nnx.Rngs):
        self.use_relu = use_relu
        layers, norms = [], []
        for _ in range(REPEATS_PER_BLOCK):
            layers.append(_dense(width, width, rngs))
            norms.append(nnx.LayerNorm(width, rngs=rngs))
        self.layers = nnx.List(layers)
        self.norms = nnx.List(norms)

    def __call__(self, x_BW):
        act = _activation(self.use_relu)
        identity_BW = x_BW
        for layer, norm in zip(self.layers, self.norms, strict=True):
            x_BW = act(norm(layer(x_BW)))
        return x_BW + identity_BW


class ResidualTrunk(nnx.Module):
    """Input projection followed by network_depth // 4 residual blocks."""

    def __init__(self, in_features: int, width: int, depth: int, use_relu: bool, *, rngs: nnx.Rngs):
        assert depth % REPEATS_PER_BLOCK == 0, f"depth must be divisible by {REPEATS_PER_BLOCK} (got {depth})"
        self.use_relu = use_relu
        self.in_proj = _dense(in_features, width, rngs)
        self.in_norm = nnx.LayerNorm(width, rngs=rngs)
        self.blocks = nnx.List([ResidualBlock(width, use_relu, rngs=rngs) for _ in range(depth // REPEATS_PER_BLOCK)])

    def __call__(self, x):
        x_BW = _activation(self.use_relu)(self.in_norm(self.in_proj(x)))
        for block in self.blocks:
            x_BW = block(x_BW)
        return x_BW


class Actor(nnx.Module):
    """Residual Gaussian actor returning (mean, log_std); tanh squashing happens in the callers."""

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, in_features, action_size, width=256, depth=4, use_relu=False, *, rngs: nnx.Rngs):
        self.trunk = ResidualTrunk(in_features, width, depth, use_relu, rngs=rngs)
        self.mean_layer = _dense(width, action_size, rngs)
        self.log_std_layer = _dense(width, action_size, rngs)

    def __call__(self, x):
        x_BW = self.trunk(x)
        log_std_BA = jax.nn.tanh(self.log_std_layer(x_BW))
        log_std_BA = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std_BA + 1)
        return self.mean_layer(x_BW), log_std_BA


class Encoder(nnx.Module):
    """Residual trunk followed by a linear projection to repr_dim."""

    def __init__(self, in_features, repr_dim=64, width=256, depth=4, use_relu=False, *, rngs: nnx.Rngs):
        self.trunk = ResidualTrunk(in_features, width, depth, use_relu, rngs=rngs)
        self.output_layer = _dense(width, repr_dim, rngs)

    def __call__(self, x):
        return self.output_layer(self.trunk(x))


class HypEncoder(nnx.Module):
    """Residual trunk -> Dense -> expmap_0 -> hyp_depth hyperbolic linear layers."""

    def __init__(
        self,
        in_features,
        repr_dim=64,
        width=256,
        depth=4,
        use_relu=False,
        *,
        manifold="hyperboloid",
        hyp_depth=1,
        manifold_float64=False,
        rngs: nnx.Rngs,
    ):
        self.manifold_name = manifold
        self.manifold = hyp.make_manifold(manifold, manifold_float64)
        self.trunk = ResidualTrunk(in_features, width, depth, use_relu, rngs=rngs)
        self.output_layer = _dense(width, repr_dim, rngs)
        self.hyp_layers = nnx.List(
            [hyp.HypLinear(manifold, self.manifold, repr_dim, repr_dim, rngs=rngs) for _ in range(hyp_depth)]
        )
        hyp.ensure_params_f32(self)

    def __call__(self, x, c):
        h = hyp.project_to_manifold(self.output_layer(self.trunk(x)), self.manifold_name, self.manifold, c)
        return hyp.apply_hyp_stack(self.manifold_name, self.hyp_layers, h, c)


class Critic(nnx.Module):
    """Contrastive critic with Euclidean energy e(phi(s, a), psi(g))."""

    def __init__(self, sa_size, goal_size, repr_dim=64, width=256, depth=4, use_relu=False, *, energy_fn, rngs):
        self.energy_fn = energy_fn
        self.sa_encoder = Encoder(sa_size, repr_dim, width, depth, use_relu, rngs=rngs)
        self.g_encoder = Encoder(goal_size, repr_dim, width, depth, use_relu, rngs=rngs)

    def encode(self, sa, g):
        return self.sa_encoder(sa), self.g_encoder(g)

    def pairwise(self, phi_BD, psi_BD):
        return euclidean_energy(self.energy_fn, phi_BD[:, None, :], psi_BD[None, :, :])

    def diagonal(self, phi_BD, psi_BD):
        return euclidean_energy(self.energy_fn, phi_BD, psi_BD)


class HypCritic(nnx.Module):
    """Contrastive critic whose encoders end in a hyperbolic head at fixed curvature."""

    def __init__(
        self,
        sa_size,
        goal_size,
        repr_dim=64,
        width=256,
        depth=4,
        use_relu=False,
        *,
        manifold,
        curvature,
        hyp_energy_fn,
        temperature,
        hyp_depth,
        manifold_float64,
        rngs,
    ):
        self.curvature = float(curvature)
        self.hyp_energy_fn = hyp_energy_fn
        self.temperature = temperature
        self.manifold = hyp.make_manifold(manifold, manifold_float64)
        kwargs = dict(manifold=manifold, hyp_depth=hyp_depth, manifold_float64=manifold_float64, rngs=rngs)
        self.sa_encoder = HypEncoder(sa_size, repr_dim, width, depth, use_relu, **kwargs)
        self.g_encoder = HypEncoder(goal_size, repr_dim, width, depth, use_relu, **kwargs)

    def encode(self, sa, g):
        return self.sa_encoder(sa, self.curvature), self.g_encoder(g, self.curvature)

    def pairwise(self, phi, psi):
        return hyp.pairwise_energy(self.hyp_energy_fn, phi, psi, self.manifold, self.curvature, self.temperature)

    def diagonal(self, phi, psi):
        return hyp.diagonal_energy(self.hyp_energy_fn, phi, psi, self.manifold, self.curvature, self.temperature)


class Alpha(nnx.Module):
    """Learnable SAC entropy coefficient (stored as log alpha)."""

    def __init__(self, init: float = 0.0):
        self.log_alpha = nnx.Param(jnp.asarray(init, dtype=jnp.float32))


def euclidean_energy(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    if name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    if name == "dot":
        return jnp.sum(x * y, axis=-1)
    raise ValueError(f"Unknown energy function: {name}")
