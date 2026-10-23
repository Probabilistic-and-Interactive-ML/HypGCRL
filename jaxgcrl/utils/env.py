# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
from jaxgcrl.envs.ant import Ant
from jaxgcrl.envs.ant_ball import AntBall
from jaxgcrl.envs.ant_ball_maze import AntBallMaze
from jaxgcrl.envs.ant_maze import AntMaze
from jaxgcrl.envs.ant_maze_generalization import AntMazeGeneralization
from jaxgcrl.envs.ant_push import AntPush
from jaxgcrl.envs.half_cheetah import Halfcheetah
from jaxgcrl.envs.humanoid import Humanoid
from jaxgcrl.envs.humanoid_maze import HumanoidMaze
from jaxgcrl.envs.manipulation.arm_binpick_easy import ArmBinpickEasy
from jaxgcrl.envs.manipulation.arm_binpick_easy_EEF import ArmBinpickEasyEEF
from jaxgcrl.envs.manipulation.arm_binpick_hard import ArmBinpickHard
from jaxgcrl.envs.manipulation.arm_grasp import ArmGrasp
from jaxgcrl.envs.manipulation.arm_push_easy import ArmPushEasy
from jaxgcrl.envs.manipulation.arm_push_hard import ArmPushHard
from jaxgcrl.envs.manipulation.arm_reach import ArmReach
from jaxgcrl.envs.maze_layouts import MAZE_LAYOUTS
from jaxgcrl.envs.pusher import Pusher, PusherReacher
from jaxgcrl.envs.pusher2 import Pusher2
from jaxgcrl.envs.reacher import Reacher
from jaxgcrl.envs.simple_maze import SimpleMaze

# Every maze layout is available for AntMaze and SimpleMaze.
_ANT_MAZE_ENVS = tuple(f"ant_{layout}" for layout in MAZE_LAYOUTS)
_SIMPLE_MAZE_ENVS = tuple(f"simple_{layout}" for layout in MAZE_LAYOUTS)
# Generalization variants of ant_u_maze with distance-restricted start/goal pairs.
_ANT_MAZE_GEN_CONFIGS = ("1f", "2f", "3f", "4f", "5f", "1f2f", "1f2f3f", "1f2f3f4f", "1f2f3f4f5f")
_ANT_MAZE_GEN_ENVS = tuple(f"ant_u_maze_gen_{config}" for config in _ANT_MAZE_GEN_CONFIGS)

legal_envs = (
    "ant",
    "ant_random_start",
    "ant_ball",
    "ant_push",
    "humanoid",
    # Humanoid at physics substep 0.003 s, the effective setting of "1000 Layer Networks for Self-Supervised RL".
    "humanoid_dt003",
    "reacher",
    "cheetah",
    "pusher_easy",
    "pusher_hard",
    "pusher_reacher",
    "pusher2",
    "arm_reach",
    "arm_grasp",
    # arm_grasp with a custom cube-start noise scale (default 0.3).
    "arm_grasp_0.1",
    "arm_grasp_0.5",
    "arm_push_easy",
    "arm_push_hard",
    "arm_binpick_easy",
    "arm_binpick_easy_EEF",
    "arm_binpick_hard",
    "ant_ball_u_maze",
    "ant_ball_big_maze",
    *_ANT_MAZE_ENVS,
    *_ANT_MAZE_GEN_ENVS,
    "humanoid_u_maze",
    "humanoid_big_maze",
    "humanoid_hardest_maze",
    *_SIMPLE_MAZE_ENVS,
)

# Evaluate on the far goals of the training layout unless --eval_env is given, as in
# "1000 Layer Networks for Self-Supervised RL" (arXiv:2503.14858).
DEFAULT_EVAL_ENVS = {
    "ant_u4_maze": "ant_u4_maze_eval",
    "ant_u5_maze": "ant_u5_maze_eval",
}


def create_env(env_name: str, backend: str | None = None):
    """Instantiate an environment by name."""
    if env_name == "reacher":
        env = Reacher(backend=backend or "generalized")
    elif env_name == "ant":
        env = Ant(backend=backend or "spring")
    elif env_name == "ant_random_start":
        env = Ant(backend=backend or "spring", randomize_start=True)
    elif env_name == "ant_ball":
        env = AntBall(backend=backend or "spring")
    elif env_name == "ant_push":
        # Only stable with the mjx backend.
        assert backend == "mjx" or backend is None
        env = AntPush(backend=backend or "mjx")
    elif "maze" in env_name:
        if "ant_ball" in env_name:
            env = AntBallMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        elif "_gen_" in env_name:
            gen_idx = env_name.index("_gen_")
            env = AntMazeGeneralization(
                backend=backend or "spring",
                maze_layout_name=env_name[4:gen_idx],
                generalization_config=env_name[gen_idx + 5 :],
            )
        elif "ant" in env_name:
            env = AntMaze(backend=backend or "spring", maze_layout_name=env_name[4:])
        elif "humanoid" in env_name:
            env = HumanoidMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        else:
            env = SimpleMaze(backend=backend or "spring", maze_layout_name=env_name[7:])
    elif env_name == "cheetah":
        env = Halfcheetah(backend=backend or "mjx")
    elif env_name == "pusher_easy":
        env = Pusher(backend=backend or "generalized", kind="easy")
    elif env_name == "pusher_hard":
        env = Pusher(backend=backend or "generalized", kind="hard")
    elif env_name == "pusher_reacher":
        env = PusherReacher(backend=backend or "generalized")
    elif env_name == "pusher2":
        env = Pusher2(backend=backend or "generalized")
    elif env_name == "humanoid":
        env = Humanoid(backend=backend or "spring")
    elif env_name == "humanoid_dt003":
        env = Humanoid(backend=backend or "spring", spring_timestep=0.003)
    elif env_name == "arm_reach":
        env = ArmReach(backend=backend or "mjx")
    elif env_name == "arm_grasp" or env_name.startswith("arm_grasp_"):
        cube_noise_scale = float(env_name[len("arm_grasp_") :]) if env_name != "arm_grasp" else 0.3
        env = ArmGrasp(backend=backend or "mjx", cube_noise_scale=cube_noise_scale)
    elif env_name == "arm_push_easy":
        env = ArmPushEasy(backend=backend or "mjx")
    elif env_name == "arm_push_hard":
        env = ArmPushHard(backend=backend or "mjx")
    elif env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=backend or "mjx")
    elif env_name == "arm_binpick_easy_EEF":
        env = ArmBinpickEasyEEF(backend=backend or "mjx")
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend=backend or "mjx")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env
