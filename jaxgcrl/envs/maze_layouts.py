# Adapted from scaling-crl (https://github.com/wang-kevin3290/scaling-crl),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
"""Maze layouts (1 = wall, 0 = free, R = reset cell, G = goal candidate), adapted from scaling-crl and D4RL."""

RESET = R = "r"
GOAL = G = "g"


U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_SINGLE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL_1F2F3F4F5F = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, 0, G, G, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL_1F2F3F4F = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, 0, 0, G, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL_1F2F3F = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL_5F6F = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, G, 0, 1],
    [1, 1, 1, 1, 1],
]

U2_MAZE = [
    [1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, 1],
    [1, 1, 1, 1, G, 1],
    [1, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1],
]

U2_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1],
    [1, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1],
]

U3_MAZE = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, G, 1],
    [1, 1, 1, 1, 1, G, 1],
    [1, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

U3_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

U3_MAZE_SINGLE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, G, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

U4_MAZE = [
    [1, 1, 1, 1, 1],
    [1, G, G, G, 1],
    [1, R, 1, G, 1],
    [1, 1, 1, G, 1],
    [1, G, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U4_MAZE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, R, 1, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 1, 0, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U5_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, G, G, G, G, G, G, 1],
    [1, R, 1, 1, 1, 1, G, 1],
    [1, 1, 1, 1, 1, 1, G, 1],
    [1, G, 1, 1, 1, 1, G, 1],
    [1, G, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

U5_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, R, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, G, 1, 1, 1, 1, G, 1],
    [1, G, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

U5_MAZE_SINGLE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, R, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, G, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

U6_MAZE = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, G, G, G, G, G, 1],
    [1, R, 1, 1, 1, G, 1],
    [1, 1, 1, 1, 1, G, 1],
    [1, G, 1, 1, 1, G, 1],
    [1, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

U6_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, R, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, G, 1, 1, 1, G, 1],
    [1, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

U7_MAZE = [
    [1, 1, 1, 1, 1, 1],
    [1, G, G, G, G, 1],
    [1, R, 1, 1, G, 1],
    [1, 1, 1, 1, G, 1],
    [1, G, 1, 1, G, 1],
    [1, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1],
]

U7_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, R, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1],
    [1, G, 1, 1, G, 1],
    [1, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1],
]

BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, 1, 1, G, G, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, 1, G, G, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, G, 1, G, G, 1, G, 1],
    [1, G, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, G, G, 1],
    [1, 0, 0, 1, 0, 0, G, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, G, 0, 1, G, 1],
    [1, 0, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

HARDEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, 1, G, G, G, G, G, 1],
    [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
    [1, G, G, G, G, G, G, 1, G, G, G, 1],
    [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
    [1, G, G, 1, G, 1, G, G, G, G, G, 1],
    [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

MAZE_LAYOUTS = {
    "u_maze": U_MAZE,
    "u_maze_eval": U_MAZE_EVAL,
    "u_maze_single_eval": U_MAZE_SINGLE_EVAL,
    "u_maze_eval_1f2f3f4f5f": U_MAZE_EVAL_1F2F3F4F5F,
    "u_maze_eval_1f2f3f4f": U_MAZE_EVAL_1F2F3F4F,
    "u_maze_eval_1f2f3f": U_MAZE_EVAL_1F2F3F,
    "u_maze_eval_5f6f": U_MAZE_EVAL_5F6F,
    "u2_maze": U2_MAZE,
    "u2_maze_eval": U2_MAZE_EVAL,
    "u3_maze": U3_MAZE,
    "u3_maze_eval": U3_MAZE_EVAL,
    "u3_maze_single_eval": U3_MAZE_SINGLE_EVAL,
    "u4_maze": U4_MAZE,
    "u4_maze_eval": U4_MAZE_EVAL,
    "u5_maze": U5_MAZE,
    "u5_maze_eval": U5_MAZE_EVAL,
    "u5_maze_single_eval": U5_MAZE_SINGLE_EVAL,
    "u6_maze": U6_MAZE,
    "u6_maze_eval": U6_MAZE_EVAL,
    "u7_maze": U7_MAZE,
    "u7_maze_eval": U7_MAZE_EVAL,
    "big_maze": BIG_MAZE,
    "big_maze_eval": BIG_MAZE_EVAL,
    "hardest_maze": HARDEST_MAZE,
}
