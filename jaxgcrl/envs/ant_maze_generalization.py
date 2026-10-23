# Adapted from scaling-crl (https://github.com/wang-kevin3290/scaling-crl),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
"""AntMaze variant sampling (start, goal) pairs at configured corridor distances, adapted from scaling-crl."""

import jax
from brax.envs.base import State
from jax import numpy as jnp

from .ant_maze import AntMaze
from .maze_layouts import GOAL, RESET

R = RESET
G = GOAL

# Generalization layouts: same walls as train layouts; the corridor from R to G defines forward distances.
U_MAZE_GEN = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

U2_MAZE_GEN = [
    [1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1],
    [1, G, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
]

U3_MAZE_GEN = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, G, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

U4_MAZE_GEN = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, R, 1, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

U5_MAZE_GEN = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, R, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, G, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

GEN_MAZE_LAYOUTS = {
    "u_maze": U_MAZE_GEN,
    "u2_maze": U2_MAZE_GEN,
    "u3_maze": U3_MAZE_GEN,
    "u4_maze": U4_MAZE_GEN,
    "u5_maze": U5_MAZE_GEN,
}


def get_forward_path(maze_layout):
    """Walk the single non-branching corridor from the R cell to the G cell; returns [(i, j), ...]."""
    start, end = None, None
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            if maze_layout[i][j] == RESET:
                start = (i, j)
            elif maze_layout[i][j] == GOAL:
                end = (i, j)

    di = [0, 1, 0, -1]
    dj = [1, 0, -1, 0]
    prev = None
    curr = start
    path = []
    while curr != end:
        path.append(curr)
        for direction in range(4):
            nxt = (curr[0] + di[direction], curr[1] + dj[direction])
            if maze_layout[nxt[0]][nxt[1]] == 1:
                continue
            if nxt == prev:
                continue
            prev, curr = curr, nxt
            break
        else:
            raise ValueError("Corridor walk got stuck; layout is not a simple path.")
    path.append(end)
    return path


def parse_generalization_config(generalization_config):
    """Parse a config like "1f2f3f" into forward distances [1, 2, 3]."""
    tokens = generalization_config.split("f")
    if tokens[-1] != "" or len(tokens) < 2 or not all(t.isdigit() and int(t) > 0 for t in tokens[:-1]):
        raise ValueError(f"Malformed generalization config: {generalization_config!r} (expected e.g. '1f' or '1f2f3f')")
    return [int(t) for t in tokens[:-1]]


def build_start_goal_pairs(maze_layout, generalization_config):
    """All (start, goal) cell pairs at the configured distances, with weights uniform per distance."""
    forward_path = get_forward_path(maze_layout)
    distances = parse_generalization_config(generalization_config)
    for d in distances:
        if d >= len(forward_path):
            raise ValueError(f"Distance {d} exceeds forward-path length {len(forward_path) - 1} of this maze.")

    pairs = []
    weights = []
    for d in distances:
        n_pairs = len(forward_path) - d
        for i in range(n_pairs):
            pairs.append((forward_path[i], forward_path[i + d]))
            weights.append(1.0 / (len(distances) * n_pairs))
    return pairs, weights


class AntMazeGeneralization(AntMaze):
    def __init__(
        self,
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        generalization_config="1f",
        **kwargs,
    ):
        if maze_layout_name not in GEN_MAZE_LAYOUTS:
            raise ValueError(f"Unknown generalization maze layout: {maze_layout_name} (supported: {sorted(GEN_MAZE_LAYOUTS)})")
        # Gen layouts share walls with the train layouts, so AntMaze builds the correct XML.
        super().__init__(maze_layout_name=maze_layout_name, maze_size_scaling=maze_size_scaling, **kwargs)

        pairs, weights = build_start_goal_pairs(GEN_MAZE_LAYOUTS[maze_layout_name], generalization_config)
        self.generalization_config = generalization_config
        # (P, 2, 2): P pairs of [start_xy, goal_xy], pre-scaled to world coordinates
        self.start_goal_pairs = jnp.array(pairs, dtype=jnp.float32) * maze_size_scaling
        self.start_goal_weights = jnp.array(weights, dtype=jnp.float32)  # (P,), sums to 1

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment, sampling a joint (start, goal) pair."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        # Joint start/goal sample (unlike AntMaze's independent start and goal draws)
        idx = jax.random.choice(rng2, self.start_goal_pairs.shape[0], p=self.start_goal_weights)
        pair = self.start_goal_pairs[idx]  # (2, 2)
        q = q.at[:2].set(pair[0])
        q = q.at[-2:].set(pair[1])
        qd = qd.at[-2:].set(0)

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
        return State(pipeline_state, obs, reward, done, metrics)
