"""Hyperbolic building blocks shared by the online and offline agents (shape suffixes: B batch, D dim)."""

import jax
import jax.numpy as jnp
from flax import nnx
from hyperbolix.manifolds import Hyperboloid, Poincare
from hyperbolix.manifolds.hyperboloid import VERSION_SMOOTHENED
from hyperbolix.nn_layers import HTCLinear, HypLinearPoincarePP
from hyperbolix.nn_layers.hyperboloid_activations import hrc_relu
from hyperbolix.nn_layers.poincare_activations import poincare_relu

MANIFOLDS = {"poincare": Poincare, "hyperboloid": Hyperboloid}


def make_manifold(name: str, float64: bool = False):
    if name not in MANIFOLDS:
        raise ValueError(f"Unknown manifold: {name}. Must be one of {tuple(MANIFOLDS)}.")
    return MANIFOLDS[name](dtype=jnp.float64 if float64 else jnp.float32)


def ensure_params_f32(module: nnx.Module):
    """Keep parameters in float32 even when manifold ops run in float64."""
    state = nnx.state(module, nnx.Param)
    nnx.update(module, jax.tree.map(lambda p: p.astype(jnp.float32), state))


def cast_grads_f32(grads):
    nnx.update(grads, jax.tree.map(lambda g: g.astype(jnp.float32), nnx.state(grads)))


def project_to_manifold(h_BD, manifold_name, manifold, c):
    """expmap_0 of Euclidean tangent vectors; hyperboloid output is ambient (B, D+1)."""
    if manifold_name == "hyperboloid":
        zeros_B1 = jnp.zeros((h_BD.shape[0], 1), dtype=h_BD.dtype)
        h_BD = jnp.concatenate([zeros_B1, h_BD], axis=-1)
    return jax.vmap(manifold.expmap_0, in_axes=(0, None))(h_BD, c)


class HypLinear(nnx.Module):
    """Hyperbolic linear layer: HTC on the hyperboloid, Hyper++ on the Poincare ball."""

    def __init__(self, manifold_name, manifold, in_spatial, out_spatial, *, rngs: nnx.Rngs):
        self.manifold_name = manifold_name
        if manifold_name == "poincare":
            self.layer = HypLinearPoincarePP(manifold, in_spatial, out_spatial, rngs=rngs)
        else:
            self.layer = HTCLinear(in_spatial + 1, out_spatial, rngs=rngs)

    def __call__(self, x, c):
        if self.manifold_name == "poincare":
            return self.layer(x, c=c)
        return self.layer(x, c, c)


def apply_hyp_stack(manifold_name, layers, h, c):
    """Apply hyperbolic linear layers with a manifold ReLU between consecutive layers."""
    relu = poincare_relu if manifold_name == "poincare" else (lambda x, c: hrc_relu(x, c_in=c, c_out=c))
    h = layers[0](h, c)
    for layer in layers[1:]:
        h = layer(relu(h, c), c)
    return h


def _dist_kwargs(manifold) -> dict:
    return {"version_idx": VERSION_SMOOTHENED} if isinstance(manifold, Hyperboloid) else {}


def _minkowski_metric(x):
    return jnp.ones(x.shape[-1], dtype=x.dtype).at[0].set(-1.0)


def pairwise_energy(name, x_BD, y_BD, manifold, c, temperature):
    """(B, B) logits: -d(x_i, y_j)/T (geodesic) or <x_i, y_j>_L/T (minkowski, hyperboloid only)."""
    if name == "geodesic":
        kwargs = _dist_kwargs(manifold)
        dist_fn = jax.vmap(jax.vmap(lambda x, y: manifold.dist(x, y, c, **kwargs), in_axes=(None, 0)), in_axes=(0, None))
        return -dist_fn(x_BD, y_BD) / temperature
    if name == "minkowski":
        return ((x_BD * _minkowski_metric(x_BD)) @ y_BD.T) / temperature
    raise ValueError(f"Unknown hyperbolic energy function: {name}")


def diagonal_energy(name, x_BD, y_BD, manifold, c, temperature):
    """(B,) energies of matched pairs."""
    if name == "geodesic":
        kwargs = _dist_kwargs(manifold)
        return -jax.vmap(lambda x, y: manifold.dist(x, y, c, **kwargs))(x_BD, y_BD) / temperature
    if name == "minkowski":
        return jnp.sum(x_BD * _minkowski_metric(x_BD) * y_BD, axis=-1) / temperature
    raise ValueError(f"Unknown hyperbolic energy function: {name}")
