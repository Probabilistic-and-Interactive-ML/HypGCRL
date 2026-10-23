# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
import time

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from flax import nnx

EPISODE_METRICS = ("reward", "success", "success_easy", "success_hard", "dist", "distance_from_origin")


def fix_reset_info_dtypes(state):
    """Align Brax reset() info dtypes with step() under jax_enable_x64 (no-op otherwise)."""
    for k in ("truncation", "episode_done"):
        if k in state.info:
            state.info[k] = state.info[k].astype(jnp.float32)
    if "eval_metrics" in state.info:
        em = state.info["eval_metrics"]
        steps_dtype = state.info["steps"].dtype if "steps" in state.info else em.episode_steps.dtype
        state.info["eval_metrics"] = em.replace(episode_steps=em.episode_steps.astype(steps_dtype))
    return state


class ActorEvaluator:
    """Runs one deterministic episode per eval env and aggregates the episode metrics."""

    def __init__(self, actor_step, eval_env, num_eval_envs, episode_length, key):
        self._key = key
        self._eval_walltime = 0.0
        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(actor, key):
            state = fix_reset_info_dtypes(eval_env.reset(jax.random.split(key, num_eval_envs)))

            def step(state, _):
                return actor_step(actor, eval_env, state)[0], ()

            return jax.lax.scan(step, state, (), length=episode_length)[0]

        self._generate_eval_unroll = nnx.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self, actor, training_metrics):
        self._key, unroll_key = jax.random.split(self._key)
        t = time.time()
        eval_metrics = self._generate_eval_unroll(actor, unroll_key).info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        # Second split keeps the evaluation key stream identical to the one used for the reported runs.
        self._key, _ = jax.random.split(self._key)

        episode = eval_metrics.episode_metrics
        metrics = {f"eval/episode_{name}": np.mean(episode[name]) for name in EPISODE_METRICS if name in episode}
        if "success" in episode:
            metrics["eval/success_rate"] = float(np.mean(episode["success"] > 0.0))
            metrics["eval/time_at_goal_frac"] = float(np.mean(episode["success"] / np.maximum(eval_metrics.episode_steps, 1)))
        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime += epoch_eval_time
        return {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}
