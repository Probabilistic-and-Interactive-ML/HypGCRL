# Adapted from OGBench (https://github.com/seohongpark/ogbench),
# Copyright (c) 2024 OGBench Authors, MIT License.
# Modified for this repository in 2026.
"""Offline goal-conditioned RL on OGBench: `jaxgcrl-offline --env_name <env> {crl,hyp-crl} [agent flags]`."""

import dataclasses
import os
import pickle
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tyro
import wandb
from flax import nnx

from jaxgcrl.offline.config import OfflineConfig
from jaxgcrl.offline.datasets import GCDataset
from jaxgcrl.offline.env_defaults import resolve_env_defaults
from jaxgcrl.offline.env_utils import make_env_and_datasets
from jaxgcrl.offline.evaluation import evaluate
from jaxgcrl.utils.logger import CsvLogger, setup_wandb


def _save_checkpoint(path, learner):
    with open(path, "wb") as f:
        pickle.dump({"actor": nnx.state(learner.actor), "critic": nnx.state(learner.critic)}, f)


class _CpuActorSampler:
    """Runs policy queries on CPU during evaluation."""

    def __init__(self, learner):
        cpu = jax.devices("cpu")[0]
        modules = []
        for module in learner.sample_modules():
            graphdef, state = nnx.split(module)
            modules.append(nnx.merge(graphdef, jax.device_put(state, cpu)))
        self._sample = nnx.cached_partial(nnx.jit(learner.sample_fn), *modules)

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        return self._sample(observations, goals, seed, temperature)


def main(config: OfflineConfig):
    agent_cfg, run = config.agent, config.run
    agent_cfg.check_config()
    agent_cfg, filled = resolve_env_defaults(agent_cfg, run.env_name)
    if filled:
        print(f"[env defaults] {run.env_name}: " + " ".join(f"{k}={v}" for k, v in filled.items()))
    if getattr(agent_cfg, "manifold_float64", False):
        jax.config.update("jax_enable_x64", True)

    run_dir = os.path.join("./runs", f"run_{run.exp_name}_s_{run.seed}")
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)

    env, train_dataset, val_dataset = make_env_and_datasets(run.env_name, dataset_dir=run.dataset_dir)
    dataset_cfg = agent_cfg.dataset_config()
    train_dataset = GCDataset(train_dataset, dataset_cfg)
    val_dataset = GCDataset(val_dataset, dataset_cfg) if val_dataset is not None else None

    # Seed after dataset construction, as in the reference implementation.
    random.seed(run.seed)
    np.random.seed(run.seed)

    example_batch = train_dataset.sample(1)
    with open(os.path.join(run_dir, "args.pkl"), "wb") as f:
        pickle.dump({"agent": agent_cfg, "run": run}, f)
    learner = agent_cfg.make_learner(run.seed, example_batch)

    wandb_config = {"agent": dataclasses.asdict(agent_cfg), "run": dataclasses.asdict(run)}
    setup_wandb(
        run.log_wandb, run.wandb_project_name, run.wandb_group, f"{run.exp_name}_s_{run.seed}", run.wandb_mode, wandb_config
    )
    train_logger = CsvLogger(os.path.join(run_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(run_dir, "eval.csv"))
    first_time = last_time = time.time()

    for i in range(1, run.train_steps + 1):
        batch = train_dataset.sample(agent_cfg.batch_size)
        update_info = learner.update(batch)

        if i % run.log_interval == 0:
            train_metrics = {f"training/{k}": float(v) for k, v in update_info.items()}
            nonfinite = [k for k, v in train_metrics.items() if not np.isfinite(v)]
            if nonfinite:
                raise RuntimeError(f"Non-finite training metrics at step {i}: {nonfinite}")
            if val_dataset is not None:
                val_info = learner.eval_losses(val_dataset.sample(agent_cfg.batch_size))
                train_metrics.update({f"validation/{k}": float(v) for k, v in val_info.items()})
            train_metrics["time/epoch_time"] = (time.time() - last_time) / run.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)
            print(
                f"[{i}/{run.train_steps}] actor_loss={train_metrics['training/actor/actor_loss']:.4f}"
                f" critic_loss={train_metrics['training/critic/contrastive_loss']:.4f}"
            )

        if i == 1 or i % run.eval_interval == 0:
            eval_agent = _CpuActorSampler(learner) if run.eval_on_cpu else learner
            eval_metrics, overall = {}, defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
            num_tasks = run.eval_tasks if run.eval_tasks is not None else len(task_infos)
            for task_id in range(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]["task_name"]
                eval_info, _, _ = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=dataset_cfg,
                    num_eval_episodes=run.eval_episodes,
                    num_video_episodes=0,
                    eval_temperature=run.eval_temperature,
                    eval_gaussian=run.eval_gaussian,
                )
                if "success" in eval_info:
                    eval_metrics[f"evaluation/{task_name}_success"] = eval_info["success"]
                    overall["success"].append(eval_info["success"])
            for k, v in overall.items():
                eval_metrics[f"evaluation/overall_{k}"] = float(np.mean(v))
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
            print(f"[{i}] eval: " + " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in sorted(eval_metrics.items())))

        if i % run.save_interval == 0 and i < run.train_steps:
            _save_checkpoint(os.path.join(run_dir, "ckpt", f"step_{i}.pkl"), learner)

    _save_checkpoint(os.path.join(run_dir, "ckpt", "final.pkl"), learner)
    train_logger.close()
    eval_logger.close()
    wandb.finish()


def cli():
    tyro.cli(
        main,
        config=(tyro.conf.OmitArgPrefixes, tyro.conf.OmitSubcommandPrefixes, tyro.conf.ConsolidateSubcommandArgs),
    )


if __name__ == "__main__":
    cli()
