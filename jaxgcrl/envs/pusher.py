# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
import jax
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jnp

# Based on the Pusher environment from Brax (https://github.com/google/brax).


class Pusher(PipelineEnv):
    def __init__(self, backend="generalized", kind="easy", dense_reward: bool = False, **kwargs):
        path = epath.resource_path("brax") / "envs/assets/pusher.xml"
        sys = mjcf.load(path)
        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.001})

            sys = sys.replace(actuator=sys.actuator.replace(gear=jnp.array([20.0] * sys.act_size())))
            n_frames = 50

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        # tips_arm is fused with r_wrist_roll_link, so use its parent r_wrist_flex_link.
        self._tips_arm_idx = self.sys.link_names.index("r_wrist_flex_link")
        self._object_idx = self.sys.link_names.index("object")
        self._goal_idx = self.sys.link_names.index("goal")
        self.kind = kind
        self.dense_reward = dense_reward
        self.state_dim = 20
        self.goal_indices = jnp.array([10, 11, 12], dtype=jnp.int32)
        self.goal_reach_thresh = 0.1

    def reset(self, rng: jax.Array) -> State:
        qpos = self.sys.init_q

        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        cylinder_pos = jnp.concatenate(
            [
                jax.random.uniform(rng, (1,), minval=-0.3, maxval=-1e-6),
                jax.random.uniform(rng1, (1,), minval=-0.2, maxval=0.2),
            ]
        )

        # randomly place the goal depending on env kind
        if self.kind == "hard":
            goal_pos = jnp.concatenate(
                [
                    jax.random.uniform(rng2, (1,), minval=-0.65, maxval=0.35),
                    jax.random.uniform(rng3, (1,), minval=-0.55, maxval=0.45),
                ]
            )
        elif self.kind == "easy":
            goal_pos = jnp.concatenate(
                [
                    jax.random.uniform(rng2, (1,), minval=-0.3, maxval=-1e-6) - 0.25,
                    jax.random.uniform(rng3, (1,), minval=-0.2, maxval=0.2),
                ]
            )

        # constrain minimum distance of object to goal
        norm = math.safe_norm(cylinder_pos - goal_pos)
        scale = jnp.where(norm < 0.17, 0.17 / norm, 1.0)
        cylinder_pos *= scale
        qpos = qpos.at[-4:].set(jnp.concatenate([cylinder_pos, goal_pos]))

        qvel = jax.random.uniform(rng4, (self.sys.qd_size(),), minval=-0.005, maxval=0.005)
        qvel = qvel.at[-4:].set(0.0)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3, dtype=jnp.float32)
        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
            "reward_near": zero,
            "success": zero,
            "success_hard": zero,
            "dist": zero,
        }

        state = State(pipeline_state, obs, reward, done, metrics)

        return state

    def step(self, state: State, action: jax.Array) -> State:
        assert state.pipeline_state is not None
        x_i = state.pipeline_state.x.vmap().do(base.Transform.create(pos=self.sys.link.inertia.transform.pos))
        vec_1 = x_i.pos[self._object_idx] - x_i.pos[self._tips_arm_idx]
        vec_2 = x_i.pos[self._object_idx] - x_i.pos[self._goal_idx]

        obj_to_goal_dist = math.safe_norm(vec_2)

        reward_near = -math.safe_norm(vec_1)
        reward_dist = -obj_to_goal_dist
        reward_ctrl = -jnp.square(action).sum()

        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        obs = self._get_obs(pipeline_state)
        success = jnp.array(obj_to_goal_dist < self.goal_reach_thresh, dtype=jnp.float32)

        if self.dense_reward:
            reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        else:
            reward = success

        state.metrics.update(
            reward_near=jnp.float32(reward_near),
            reward_dist=jnp.float32(reward_dist),
            reward_ctrl=jnp.float32(reward_ctrl),
            success=success,
            success_hard=jnp.array(obj_to_goal_dist < 0.05, dtype=jnp.float32),
            # Cast to float32 so metrics stay float32 under jax_enable_x64.
            dist=jnp.float32(obj_to_goal_dist),
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=jnp.float32(reward))

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observes pusher body position and velocities."""
        x_i = pipeline_state.x.vmap().do(base.Transform.create(pos=self.sys.link.inertia.transform.pos))
        return jnp.concatenate(
            [
                # state
                pipeline_state.q[:7],  # Rotations of arm joints [7, ]
                x_i.pos[self._tips_arm_idx],  # Arm tip position [3, ]
                x_i.pos[self._object_idx],  # Movable object position [3, ]
                pipeline_state.qd[:7],  # Rotational velocities of arm joints [7, ]
                # goal
                x_i.pos[self._goal_idx],  # Goal object position [3, ]
            ]
        ).astype(jnp.float32)


# Debug variant of Pusher: move the arm tip to a target position, as in Reacher.
class PusherReacher(PipelineEnv):
    def __init__(self, backend="generalized", **kwargs):
        path = epath.resource_path("brax") / "envs/assets/pusher.xml"
        sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.001})
            sys = sys.replace(actuator=sys.actuator.replace(gear=jnp.array([20.0] * sys.act_size())))
            n_frames = 50

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        # tips_arm is fused with r_wrist_roll_link, so use its parent r_wrist_flex_link.
        self._tips_arm_idx = self.sys.link_names.index("r_wrist_flex_link")
        self._object_idx = self.sys.link_names.index("object")
        self._goal_idx = self.sys.link_names.index("goal")

        self.state_dim = 17
        self.goal_indices = jnp.array([14, 15, 16], dtype=jnp.int32)
        self.goal_reach_thresh = 0.1

    def reset(self, rng: jax.Array) -> State:
        qpos = self.sys.init_q

        rng, _rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        cylinder_pos = jnp.concatenate(
            [
                jnp.array([1.0]),
                jnp.array([1.0]),
            ]
        )

        # randomly place the goal depending on env kind
        goal_pos = jnp.concatenate(
            [
                jax.random.uniform(rng2, (1,), minval=-0.3, maxval=-1e-6),
                jax.random.uniform(rng3, (1,), minval=-0.2, maxval=0.2),
            ]
        )

        # constrain minimum distance of object to goal
        norm = math.safe_norm(cylinder_pos - goal_pos)
        scale = jnp.where(norm < 0.17, 0.17 / norm, 1.0)
        cylinder_pos *= scale
        qpos = qpos.at[-4:].set(jnp.concatenate([cylinder_pos, goal_pos]))

        qvel = jax.random.uniform(rng4, (self.sys.qd_size(),), minval=-0.005, maxval=0.005)
        qvel = qvel.at[-4:].set(0.0)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3, dtype=jnp.float32)
        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
            "reward_near": zero,
            "success": zero,
            "success_hard": zero,
            "dist": zero,
        }

        state = State(pipeline_state, obs, reward, done, metrics)

        return state

    def step(self, state: State, action: jax.Array) -> State:
        assert state.pipeline_state is not None
        x_i = state.pipeline_state.x.vmap().do(base.Transform.create(pos=self.sys.link.inertia.transform.pos))
        x_i.pos[self._object_idx] - x_i.pos[self._tips_arm_idx]
        x_i.pos[self._object_idx] - x_i.pos[self._goal_idx]

        arm_to_goal_dist = math.safe_norm(x_i.pos[self._goal_idx] - x_i.pos[self._tips_arm_idx])

        reward_dist = -arm_to_goal_dist
        reward = reward_dist

        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        obs = self._get_obs(pipeline_state)
        state.metrics.update(
            reward_near=jnp.float32(0.0),
            reward_dist=jnp.float32(reward_dist),
            reward_ctrl=jnp.float32(0.0),
            success=jnp.array(arm_to_goal_dist < self.goal_reach_thresh, dtype=jnp.float32),
            success_hard=jnp.array(arm_to_goal_dist < 0.05, dtype=jnp.float32),
            dist=jnp.float32(arm_to_goal_dist),
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=jnp.float32(reward))

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observes pusher body position and velocities."""
        x_i = pipeline_state.x.vmap().do(base.Transform.create(pos=self.sys.link.inertia.transform.pos))
        return jnp.concatenate(
            [
                # state
                pipeline_state.q[:7],  # Rotations of arm joints [7, ]
                pipeline_state.qd[:7],  # Rotational velocities of arm joints [7, ]
                x_i.pos[self._tips_arm_idx],  # Arm tip position [3, ]
                # goal
                x_i.pos[self._goal_idx],
            ]
        ).astype(jnp.float32)
