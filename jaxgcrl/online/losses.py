# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
"""Contrastive critic and SAC actor updates, shared by the Euclidean and hyperbolic critics."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jaxgcrl.hyperbolic import cast_grads_f32


def contrastive_loss(name, logits_BB):
    """Contrastive objective on the (B, B) logits whose diagonal holds the positive pairs."""
    diag_B = jnp.diag(logits_BB)
    if name == "fwd_infonce":
        return -jnp.mean(diag_B - jax.nn.logsumexp(logits_BB, axis=1))
    if name == "bwd_infonce":
        return -jnp.mean(diag_B - jax.nn.logsumexp(logits_BB, axis=0))
    if name == "sym_infonce":
        return -jnp.mean(2 * diag_B - jax.nn.logsumexp(logits_BB, axis=1) - jax.nn.logsumexp(logits_BB, axis=0))
    if name == "binary_nce":
        eye_BB = jnp.eye(logits_BB.shape[0], dtype=logits_BB.dtype)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits_BB, labels=eye_BB))
    raise ValueError(f"Unknown contrastive loss function: {name}")


def update_actor_and_alpha(ctx, actor, actor_opt, critic, alpha, alpha_opt, transitions, rngs):
    noise_key = rngs.noise()

    def actor_loss(actor):
        state = transitions.observation[:, : ctx["state_size"]]
        goal = transitions.extras["future_state"][:, ctx["goal_indices"]]
        means, log_stds = actor(jnp.concatenate([state, goal], axis=1))
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
        action = jax.nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)

        sa_repr, g_repr = critic.encode(jnp.concatenate([state, action], axis=-1), goal)
        qf_pi = critic.diagonal(sa_repr, g_repr)
        loss = jnp.mean(jnp.exp(alpha.log_alpha[...]) * log_prob - qf_pi)
        return loss, log_prob

    (actor_loss_val, log_prob), actor_grads = nnx.value_and_grad(actor_loss, has_aux=True)(actor)
    if ctx["manifold_float64"]:
        cast_grads_f32(actor_grads)
    actor_opt.update(actor, actor_grads)

    def alpha_loss_fn(alpha):
        return jnp.exp(alpha.log_alpha[...]) * jnp.mean(jax.lax.stop_gradient(-log_prob - ctx["target_entropy"]))

    alpha_loss_val, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(alpha)
    if ctx["fixed_alpha"] is None:
        alpha_opt.update(alpha, alpha_grads)

    return {
        "entropy": jnp.mean(-log_prob),
        "actor_loss": actor_loss_val,
        "alpha_loss": alpha_loss_val,
        "log_alpha": alpha.log_alpha[...],
    }


def update_critic(ctx, critic, critic_opt, transitions):
    def critic_loss(critic):
        state = transitions.observation[:, : ctx["state_size"]]
        goal = transitions.observation[:, ctx["state_size"] :]
        sa_repr, g_repr = critic.encode(jnp.concatenate([state, transitions.action], axis=-1), goal)
        logits = critic.pairwise(sa_repr, g_repr)
        loss = contrastive_loss(ctx["contrastive_loss_fn"], logits)
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += ctx["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)
        correct = jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0])
        return loss, (logsumexp, correct)

    (loss, (logsumexp, correct)), grads = nnx.value_and_grad(critic_loss, has_aux=True)(critic)
    if ctx["manifold_float64"]:
        cast_grads_f32(grads)
    critic_opt.update(critic, grads)

    return {
        "categorical_accuracy": jnp.mean(correct),
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }
