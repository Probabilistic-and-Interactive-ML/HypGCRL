# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
import os

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

# Based on the Ant environment from Brax (https://github.com/google/brax).


class Ant(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        dense_reward: bool = False,
        randomize_start=False,
        goal_distance=10,
        **kwargs,
    ):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "ant.xml")
        sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear)))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self.dense_reward = dense_reward
        self.state_dim = 29
        self.goal_indices = jnp.array([0, 1], dtype=jnp.int32)
        self.goal_reach_thresh = 0.5
        self.goal_distance = goal_distance
        self.randomize_start = randomize_start

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd
        rng, target = self._random_target(rng)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        if self.randomize_start:
            _, start_delta = self._random_target(rng)
            start = target + start_delta
            q = q.at[:2].set(start)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3, dtype=jnp.float32)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        old_obs = self._get_obs(pipeline_state0)
        old_dist = jnp.linalg.norm(old_obs[:2] - old_obs[-2:])
        obs = self._get_obs(pipeline_state)
        dist = jnp.linalg.norm(obs[:2] - obs[-2:])
        vel_to_target = (old_dist - dist) / self.dt
        success = jnp.array(dist < self.goal_reach_thresh, dtype=jnp.float32)
        success_easy = jnp.array(dist < 2.0, dtype=jnp.float32)

        if self.dense_reward:
            reward = 10 * vel_to_target + healthy_reward - ctrl_cost - contact_cost
        else:
            reward = success
        done = jnp.float32(1.0 - is_healthy) if self._terminate_when_unhealthy else jnp.float32(0.0)

        state.metrics.update(
            reward_forward=jnp.float32(forward_reward),
            reward_survive=jnp.float32(healthy_reward),
            reward_ctrl=jnp.float32(-ctrl_cost),
            reward_contact=jnp.float32(-contact_cost),
            x_position=jnp.float32(pipeline_state.x.pos[0, 0]),
            y_position=jnp.float32(pipeline_state.x.pos[0, 1]),
            distance_from_origin=jnp.float32(math.safe_norm(pipeline_state.x.pos[0])),
            x_velocity=jnp.float32(velocity[0]),
            y_velocity=jnp.float32(velocity[1]),
            forward_reward=jnp.float32(forward_reward),
            dist=jnp.float32(dist),
            success=success,
            success_easy=success_easy,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=jnp.float32(reward), done=done)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        # remove target q, qd
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jnp.concatenate([qpos, qvel, target_pos]).astype(jnp.float32)

    def _random_target(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, _rng1, rng2 = jax.random.split(rng, 3)
        ang = jnp.pi * 2.0 * jax.random.uniform(rng2)
        target_x = self.goal_distance * jnp.cos(ang)
        target_y = self.goal_distance * jnp.sin(ang)
        return rng, jnp.array([target_x, target_y])
