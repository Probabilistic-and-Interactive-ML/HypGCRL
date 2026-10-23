# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
"""Online run and agent configurations (defaults follow Table 7 of "1000 Layer Networks for Self-Supervised RL",
arXiv:2503.14858)."""

from typing import Annotated, Literal

import tyro
from flax import nnx
from flax.struct import dataclass

from jaxgcrl.online.networks import REPEATS_PER_BLOCK, Critic, HypCritic
from jaxgcrl.utils.env import legal_envs


@dataclass
class RunConfig:
    env: Literal[legal_envs]
    total_env_steps: int = 100_000_000
    episode_length: int = 1000
    # Evaluate on a different environment than the training one. None uses the far-goal layout
    # for ant_u4_maze and ant_u5_maze (DEFAULT_EVAL_ENVS) and the training environment otherwise.
    eval_env: Literal[legal_envs] | None = None
    num_envs: int = 512
    num_eval_envs: int = 32
    num_evals: int = 200
    seed: int = 0
    backend: Literal["mjx", "spring", "positional", "generalized"] | None = None
    exp_name: str = "run"
    log_wandb: bool = False
    wandb_project_name: str = "hypgcrl-online"
    wandb_group: str | None = None
    wandb_mode: Literal["online", "offline"] = "online"
    # Render a rollout every N evaluations (0 = never).
    visualization_interval: int = 0


@dataclass
class _ContrastiveAgent:
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 512
    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    # Sample->update sweeps per collected unroll.
    train_step_multiplier: int = 1
    # Minibatches per sweep (800 of 999 = UTD 1:40); None uses all of them.
    num_sgd_batches_per_step: int | None = 800
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    actor_network_width: int = 256
    critic_network_width: int = 256
    # Number of Dense layers in the residual blocks (multiple of 4).
    actor_depth: int = 4
    critic_depth: int = 4
    use_relu: bool = False
    repr_dim: int = 64
    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "sym_infonce"
    # Fix the SAC entropy coefficient instead of tuning it.
    fixed_alpha: float | None = None

    manifold_float64 = False

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )
        for name in ("actor_depth", "critic_depth"):
            assert getattr(self, name) % REPEATS_PER_BLOCK == 0, f"{name} must be divisible by {REPEATS_PER_BLOCK}"


@dataclass
class CRL(_ContrastiveAgent):
    """Deep contrastive RL with a Euclidean critic."""

    energy_fn: Literal["norm", "l2", "dot"] = "norm"

    def build_critic(self, sa_size, goal_size, rngs: nnx.Rngs):
        return Critic(
            sa_size,
            goal_size,
            self.repr_dim,
            self.critic_network_width,
            self.critic_depth,
            self.use_relu,
            energy_fn=self.energy_fn,
            rngs=rngs,
        )


@dataclass
class HypCRL(_ContrastiveAgent):
    """Deep contrastive RL whose critic encoders end in a hyperbolic head."""

    manifold: Literal["poincare", "hyperboloid"] = "poincare"
    curvature: float = 1.0
    hyp_energy_fn: Literal["geodesic", "minkowski"] = "geodesic"
    temperature: float = 1.0
    # Number of hyperbolic linear layers in the head.
    hyp_depth: int = 1
    manifold_float64: bool = False

    def check_config(self, config):
        super().check_config(config)
        if self.hyp_energy_fn == "minkowski" and self.manifold != "hyperboloid":
            raise ValueError("The Minkowski energy requires --manifold hyperboloid.")
        if self.hyp_depth < 1:
            raise ValueError(f"hyp_depth must be >= 1 (got {self.hyp_depth})")

    def build_critic(self, sa_size, goal_size, rngs: nnx.Rngs):
        return HypCritic(
            sa_size,
            goal_size,
            self.repr_dim,
            self.critic_network_width,
            self.critic_depth,
            self.use_relu,
            manifold=self.manifold,
            curvature=self.curvature,
            hyp_energy_fn=self.hyp_energy_fn,
            temperature=self.temperature,
            hyp_depth=self.hyp_depth,
            manifold_float64=self.manifold_float64,
            rngs=rngs,
        )


AgentConfig = Annotated[CRL, tyro.conf.subcommand(name="crl")] | Annotated[HypCRL, tyro.conf.subcommand(name="hyp-crl")]


@dataclass
class Config:
    agent: AgentConfig
    run: RunConfig
