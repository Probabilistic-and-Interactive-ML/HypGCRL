# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
import jax
from brax.envs import PipelineEnv, State, Wrapper
from jax import numpy as jnp


class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["traj_id"] = jnp.zeros(rng.shape[:-1], dtype=jnp.float32)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info.keys():
            traj_id = state.info["traj_id"] + jnp.where(state.info["steps"], 0, 1)
        else:
            traj_id = state.info["traj_id"]
        state = self.env.step(state, action)
        state.info["traj_id"] = traj_id
        return state


def _cast_info_float32(info: dict) -> dict:
    """Cast float64 info values to float32 (recursive for nested dicts)."""
    out = {}
    for k, v in info.items():
        if isinstance(v, dict):
            out[k] = _cast_info_float32(v)
        elif hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating):
            out[k] = v.astype(jnp.float32)
        else:
            out[k] = v
    return out


class Float32InfoWrapper(Wrapper):
    """Cast float64 info fields created by Brax training wrappers to float32 to avoid scan dtype mismatches."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        return state.replace(info=_cast_info_float32(state.info))

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        return state.replace(info=_cast_info_float32(state.info))
