# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
import jax
from brax import base
from jax import numpy as jnp

from .arm_envs import ArmEnvs

"""
Grasp: Close fingers on opposite sides of a cube.
- Observation space: 24-dim obs + 7-dim goal.
- Action space:      5-dim, each element in [-1, 1], corresponding to target angles
                     for joints 1, 2, 4, 6, and finger closedness.

See _get_obs() and ArmEnvs._convert_action() for details.
"""


class ArmGrasp(ArmEnvs):
    def __init__(self, cube_noise_scale=0.3, **kwargs):
        super().__init__(**kwargs)
        # Only read at reset time, so overriding after super().__init__ is safe.
        self.cube_noise_scale = cube_noise_scale

    def _get_xml_path(self):
        return "envs/assets/panda_grasp.xml"

    @property
    def action_size(self) -> int:
        return 5  # Override default (actuator count)

    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "arm_grasp"
        self.episode_length = 100

        self.goal_indices = jnp.array(
            [16, 17, 18, 19, 20, 21, 22], dtype=jnp.int32
        )  # Left and right fingertip positions, and fingertip distance
        self.completion_goal_indices = jnp.array([16, 17, 18, 19, 20, 21, 22], dtype=jnp.int32)  # Identical
        self.state_dim = 24

        self.arm_noise_scale = 0
        self.cube_noise_scale = 0.3
        self.goal_reach_thresh = 0.05

    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        arm_q_default = jnp.array(
            [1.571, 0.742, 0, -1.571, 0, 3.054, 1.449, 0.04, 0.04, 0, 0]
        )  # Start closer to the relevant area
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)

        q = jnp.concatenate([cube_q_xy, cube_q_remaining, target_q, arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd

    def _get_initial_goal(self, pipeline_state: base.State, rng):
        # The fingertip goal positions will constantly adjust to be next to the cube (logic handled in arm_envs:step)
        cube_pos = pipeline_state.q[:3]
        left_fingertip_goal_pos = cube_pos + jnp.array([0.0375, 0, 0])
        right_fingertip_goal_pos = cube_pos + jnp.array([-0.0375, 0, 0])
        gripper_openness_goal = jnp.array(
            [0.075]
        )  # The cube itself is 0.06 wide, but we want the centers of the fingertips to be slightly farther apart

        goal = jnp.concatenate([left_fingertip_goal_pos, right_fingertip_goal_pos, gripper_openness_goal])
        return goal

    def _compute_goal_completion(self, obs, goal):
        # Success: fingertip midpoint near the cube and gripper closed enough (only success_hard is a true grip).
        cube_pos = obs[:3]
        left_fingertip_pos = obs[16:19]
        right_fingertip_pos = obs[19:22]
        fingertip_midpoint = (left_fingertip_pos + right_fingertip_pos) / 2
        cube_to_fingertip_midpoint_dist = jnp.linalg.norm(cube_pos - fingertip_midpoint)

        gripper_openness = obs[22]
        # The goal is 7-dim; index 6 is the gripper openness.
        goal_gripper_openness = goal[6]
        gripper_openness_difference = jnp.linalg.norm(gripper_openness - goal_gripper_openness)

        success = jnp.array(
            jnp.all(
                jnp.array(
                    [
                        cube_to_fingertip_midpoint_dist < self.goal_reach_thresh,
                        gripper_openness_difference < 0.02,
                    ]
                )
            ),
            dtype=jnp.float32,
        )
        success_easy = jnp.array(
            jnp.all(
                jnp.array(
                    [
                        cube_to_fingertip_midpoint_dist < 0.15,
                        gripper_openness_difference < 0.05,
                    ]
                )
            ),
            dtype=jnp.float32,
        )
        success_hard = jnp.array(
            jnp.all(
                jnp.array(
                    [
                        cube_to_fingertip_midpoint_dist < 0.02,
                        gripper_openness_difference < 0.005,
                    ]
                )
            ),
            dtype=jnp.float32,
        )
        return success, success_easy, success_hard, cube_to_fingertip_midpoint_dist

    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(
            goal[:3]
        )  # Only set the position, not orientation (we set the left marker and the right marker is automatically offset)
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state

    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """Observation (24-dim): cube pos, joint angles, EEF pos/vel, fingertips, gripper force; goal is fingertips."""
        q_indices = jnp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20], dtype=jnp.int32)
        q_subset = pipeline_state.q[q_indices]

        # Cube is 0, goal marker is 1, links 1-7 are 2-8; the end-effector is merged with link 7.
        eef_index = 8
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]

        left_fingertip_index = 10  # Left finger is 9, fingertip is 10
        left_fingertip_x_pos = pipeline_state.x.pos[left_fingertip_index]
        right_fingertip_index = 12  # Right finger is 11, fingertip is 12
        right_fingertip_x_pos = pipeline_state.x.pos[right_fingertip_index]
        fingertip_distance = jnp.linalg.norm(right_fingertip_x_pos - left_fingertip_x_pos)[
            None
        ]  # [None] expands dims from 0 to 1

        # Index -4 and -2 to get the fingers, rather than fingertips
        gripper_force = (
            (pipeline_state.qfrc_actuator[jnp.array([-4, -2], dtype=jnp.int32)]).mean(keepdims=True) * 0.1
        )  # Normalize it from range [-20, 20] to [-2, 2]

        return jnp.concatenate(
            [
                q_subset,
                eef_x_pos,
                eef_xd_vel,
                left_fingertip_x_pos,
                right_fingertip_x_pos,
                fingertip_distance,
                gripper_force,
                goal,
            ]
        ).astype(jnp.float32)

    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(14, 21, dtype=jnp.int32)
        return pipeline_state.q[q_indices]
