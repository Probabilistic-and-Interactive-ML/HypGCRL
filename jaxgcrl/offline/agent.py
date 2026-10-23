# Adapted from OGBench (https://github.com/seohongpark/ogbench),
# Copyright (c) 2024 OGBench Authors, MIT License.
# Modified for this repository in 2026.
"""Offline contrastive RL learner, ported from OGBench (shape suffixes: B batch, A action, D latent, E ensemble)."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jaxgcrl.hyperbolic import cast_grads_f32
from jaxgcrl.offline.networks import GCActor, HypGCActor, gaussian_log_prob, gaussian_sample
from jaxgcrl.online.losses import contrastive_loss


def _critic_loss_and_info(cfg, critic, batch):
    v_EB, phi_EBD, psi_EBD = critic(batch["observations"], batch["value_goals"], batch["actions"], info=True)
    logits_EBB = critic.pairwise_logits(phi_EBD, psi_EBD)
    loss = jax.vmap(lambda logits_BB: contrastive_loss(cfg.contrastive_loss, logits_BB))(logits_EBB).mean()
    if cfg.logsumexp_penalty_coeff > 0.0:
        logsumexp_EB = jax.nn.logsumexp(logits_EBB + 1e-6, axis=-1)
        loss = loss + cfg.logsumexp_penalty_coeff * jnp.mean(logsumexp_EB**2)

    eye_BB = jnp.eye(logits_EBB.shape[-1], dtype=logits_EBB.dtype)
    logits_BB = logits_EBB.mean(axis=0)
    info = {
        "contrastive_loss": loss,
        "v_mean": jnp.exp(v_EB).mean(),
        "binary_accuracy": jnp.mean((logits_BB > 0) == eye_BB),
        "categorical_accuracy": jnp.mean(jnp.argmax(logits_BB, axis=1) == jnp.argmax(eye_BB, axis=1)),
        "logits_pos": jnp.sum(logits_BB * eye_BB) / jnp.sum(eye_BB),
        "logits_neg": jnp.sum(logits_BB * (1 - eye_BB)) / jnp.sum(1 - eye_BB),
    }
    return loss, info


def _actor_loss_and_info(cfg, actor, critic, batch, key):
    """DDPG+BC: maximize the normalized Q of the policy action plus a BC log-likelihood term."""
    means_BA, stds_BA = actor(batch["observations"], batch["actor_goals"])
    if cfg.const_std:
        q_actions_BA = jnp.clip(means_BA, -1, 1)
    else:
        q_actions_BA = jnp.clip(gaussian_sample(means_BA, stds_BA, key), -1, 1)
    q_B = jnp.min(critic(batch["observations"], batch["actor_goals"], q_actions_BA), axis=0)

    q_loss = -q_B.mean() / jax.lax.stop_gradient(jnp.abs(q_B).mean() + 1e-6)
    log_prob_B = gaussian_log_prob(means_BA, stds_BA, batch["actions"])
    bc_loss = -(cfg.alpha * log_prob_B).mean()
    actor_loss = q_loss + bc_loss
    info = {"actor_loss": actor_loss, "q_loss": q_loss, "bc_loss": bc_loss, "q_mean": q_B.mean(), "std": jnp.mean(stds_BA)}
    return actor_loss, info


def _grad_step(module, opt, loss_fn, cast_grads):
    (_, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(module)
    if cast_grads:
        cast_grads_f32(grads)
    opt.update(module, grads)
    return info


def _merge(critic_info, actor_info):
    return {**{f"critic/{k}": v for k, v in critic_info.items()}, **{f"actor/{k}": v for k, v in actor_info.items()}}


def make_update_fn(cfg):
    cast_grads = getattr(cfg, "manifold_float64", False)

    def update(actor, actor_opt, critic, critic_opt, batch, key):
        # The actor is updated first so it sees the pre-update critic (joint update of the reference).
        actor_info = _grad_step(actor, actor_opt, lambda a: _actor_loss_and_info(cfg, a, critic, batch, key), cast_grads)
        critic_info = _grad_step(critic, critic_opt, lambda c: _critic_loss_and_info(cfg, c, batch), cast_grads)
        return _merge(critic_info, actor_info)

    return update


def make_eval_loss_fn(cfg):
    def eval_losses(actor, critic, batch, key):
        _, critic_info = _critic_loss_and_info(cfg, critic, batch)
        _, actor_info = _actor_loss_and_info(cfg, actor, critic, batch, key)
        return _merge(critic_info, actor_info)

    return eval_losses


def _sample_actions(actor, observations, goals, seed, temperature):
    means, stds = actor(observations, goals, temperature=temperature)
    return jnp.clip(gaussian_sample(means, stds, seed), -1, 1)


class CRLLearner:
    """Holds actor, critic and their optimizers; exposes jitted update and action sampling."""

    sample_fn = staticmethod(_sample_actions)

    def __init__(self, config, actor, critic, seed):
        self.config = config
        self.actor = actor
        self.critic = critic
        self.actor_opt = nnx.Optimizer(actor, optax.adam(learning_rate=config.lr), wrt=nnx.Param)
        self.critic_opt = nnx.Optimizer(critic, optax.adam(learning_rate=config.lr), wrt=nnx.Param)
        self._rng = jax.random.PRNGKey(seed)
        self._dummy_key = jax.random.PRNGKey(0)
        self._update = nnx.cached_partial(nnx.jit(make_update_fn(config)), actor, self.actor_opt, critic, self.critic_opt)
        self._eval_losses = nnx.cached_partial(nnx.jit(make_eval_loss_fn(config)), actor, critic)
        self._sample = nnx.cached_partial(nnx.jit(_sample_actions), actor)

    def sample_modules(self):
        return (self.actor,)

    def update(self, batch):
        if self.config.const_std:
            key = self._dummy_key
        else:
            self._rng, key = jax.random.split(self._rng)
        return self._update(batch, key)

    def eval_losses(self, batch):
        return self._eval_losses(batch, self._dummy_key)

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        return self._sample(observations, goals, seed, temperature)

    @classmethod
    def create(cls, seed, example_batch, config):
        """Build the critic and then the actor from a shared RNG stream."""
        rngs = nnx.Rngs(seed)
        obs_dim = int(example_batch["observations"].shape[-1])
        goal_dim = int(example_batch["actor_goals"].shape[-1])
        action_dim = example_batch["actions"].shape[-1]
        critic = config.build_critic(obs_dim, goal_dim, action_dim, rngs=rngs)
        if config.hyp_actor:
            actor = HypGCActor(
                obs_dim,
                goal_dim,
                action_dim,
                config.actor_hidden_dims,
                repr_dim=config.hyp_actor_repr_dim,
                depth=config.hyp_actor_depth,
                curvature=config.hyp_actor_curvature,
                manifold_float64=getattr(config, "manifold_float64", False),
                const_std=config.const_std,
                rngs=rngs,
            )
        else:
            actor = GCActor(obs_dim, goal_dim, action_dim, config.actor_hidden_dims, const_std=config.const_std, rngs=rngs)
        return cls(config, actor, critic, seed)
