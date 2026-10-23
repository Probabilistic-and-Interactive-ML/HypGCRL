"""Offline (OGBench) run and agent configurations; defaults follow the OGBench CRL reference."""

import dataclasses
from typing import Annotated, Literal

import tyro
from flax.struct import dataclass

from jaxgcrl.offline.networks import GCBilinearCritic, HypGCBilinearCritic

# Per-env fields left at None are filled from the env name (see env_defaults.py).
PER_ENV_FIELDS = ("discount", "alpha", "actor_p_trajgoal", "actor_p_randomgoal")


@dataclass
class OfflineRunConfig:
    env_name: str
    dataset_dir: str = "~/.ogbench/data"
    train_steps: int = 1_000_000
    log_interval: int = 5_000
    eval_interval: int = 100_000
    save_interval: int = 1_000_000
    # Number of evaluation tasks (None = all 5).
    eval_tasks: int | None = None
    eval_episodes: int = 20
    eval_temperature: float = 0.0
    eval_gaussian: float | None = None
    eval_on_cpu: bool = True
    seed: int = 0
    exp_name: str = "offline"
    log_wandb: bool = False
    wandb_project_name: str = "hypgcrl-offline"
    wandb_group: str | None = None
    wandb_mode: Literal["online", "offline"] = "online"


@dataclass
class _OfflineCRLBase:
    lr: float = 3e-4
    batch_size: int = 1024
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512)
    value_hidden_dims: tuple[int, ...] = (512, 512, 512)
    latent_dim: int = 512
    layer_norm: bool = True
    discount: float | None = None
    # DDPG+BC behavior-cloning coefficient.
    alpha: float | None = None
    const_std: bool = True
    contrastive_loss: Literal["binary_nce", "fwd_infonce", "sym_infonce", "bwd_infonce"] = "binary_nce"
    logsumexp_penalty_coeff: float = 0.0
    # Hyperbolic actor: MLP -> Hyper++ -> expmap_0 -> HTC layers -> logmap_0 -> Gaussian head.
    hyp_actor: bool = False
    hyp_actor_depth: int = 1
    hyp_actor_curvature: float = 1.0
    hyp_actor_repr_dim: int = 64
    value_p_curgoal: float = 0.0
    value_p_trajgoal: float = 1.0
    value_p_randomgoal: float = 0.0
    value_geom_sample: bool = True
    actor_p_curgoal: float = 0.0
    actor_p_trajgoal: float | None = None
    actor_p_randomgoal: float | None = None
    actor_geom_sample: bool = False
    gc_negative: bool = False

    def check_config(self):
        pass

    def dataset_config(self) -> dict:
        unresolved = [f for f in PER_ENV_FIELDS if getattr(self, f) is None]
        if unresolved:
            raise ValueError(f"Unresolved per-env fields {unresolved}; call env_defaults.resolve_env_defaults first.")
        return dataclasses.asdict(self)

    def make_learner(self, seed: int, example_batch: dict):
        from jaxgcrl.offline.agent import CRLLearner

        self.check_config()
        self.dataset_config()
        if example_batch["observations"].ndim != 2:
            raise ValueError("Only state-based OGBench environments are supported.")
        return CRLLearner.create(seed, example_batch, self)


@dataclass
class OfflineCRL(_OfflineCRLBase):
    """Offline contrastive RL with a Euclidean critic."""

    energy_fn: Literal["dot", "norm", "l2"] = "dot"
    use_logit_bias: bool = False

    def build_critic(self, obs_dim, goal_dim, action_dim, rngs):
        return GCBilinearCritic(
            obs_dim,
            goal_dim,
            action_dim,
            self.value_hidden_dims,
            self.latent_dim,
            layer_norm=self.layer_norm,
            energy_fn_name=self.energy_fn,
            use_logit_bias=self.use_logit_bias,
            rngs=rngs,
        )


@dataclass
class OfflineHypCRL(_OfflineCRLBase):
    """Offline contrastive RL with a hyperbolic critic."""

    use_logit_bias: bool = True
    manifold: Literal["poincare", "hyperboloid"] = "hyperboloid"
    curvature: float = 1.0
    hyp_energy_fn: Literal["geodesic", "minkowski"] = "geodesic"
    temperature: float = 1.0
    hyp_depth: int = 1
    manifold_float64: bool = False

    def check_config(self):
        if self.hyp_energy_fn == "minkowski" and self.manifold != "hyperboloid":
            raise ValueError("--hyp_energy_fn minkowski requires --manifold hyperboloid")
        if self.hyp_depth < 1:
            raise ValueError(f"hyp_depth must be >= 1 (got {self.hyp_depth})")

    def build_critic(self, obs_dim, goal_dim, action_dim, rngs):
        return HypGCBilinearCritic(
            obs_dim,
            goal_dim,
            action_dim,
            self.value_hidden_dims,
            self.latent_dim,
            layer_norm=self.layer_norm,
            manifold=self.manifold,
            curvature=self.curvature,
            hyp_energy_fn=self.hyp_energy_fn,
            temperature=self.temperature,
            hyp_depth=self.hyp_depth,
            manifold_float64=self.manifold_float64,
            use_logit_bias=self.use_logit_bias,
            rngs=rngs,
        )


OfflineAgentConfig = (
    Annotated[OfflineCRL, tyro.conf.subcommand(name="crl")] | Annotated[OfflineHypCRL, tyro.conf.subcommand(name="hyp-crl")]
)


@dataclass
class OfflineConfig:
    agent: OfflineAgentConfig
    run: OfflineRunConfig
