# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
"""Online goal-conditioned RL: `python run.py --env <env> {crl,hyp-crl} [agent flags]`."""

import dataclasses
import logging
import os
import pickle

import jax
import tyro
import wandb
from flax import nnx

from jaxgcrl.online.config import Config
from jaxgcrl.online.train import train
from jaxgcrl.utils.env import DEFAULT_EVAL_ENVS, create_env
from jaxgcrl.utils.logger import MetricsRecorder, setup_wandb


def main(config: Config):
    if config.run.eval_env is None and config.run.env in DEFAULT_EVAL_ENVS:
        config = config.replace(run=config.run.replace(eval_env=DEFAULT_EVAL_ENVS[config.run.env]))
    agent, run = config.agent, config.run
    if agent.manifold_float64:
        jax.config.update("jax_enable_x64", True)
    logging.basicConfig(format="%(asctime)s %(levelname)-8s| %(message)s", level=logging.INFO)

    run_dir = f"./runs/run_{run.exp_name}_s_{run.seed}"
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    with open(f"{run_dir}/args.pkl", "wb") as f:
        pickle.dump(vars(config), f)

    info = {**dataclasses.asdict(run), **dataclasses.asdict(agent), "agent": type(agent).__name__}
    setup_wandb(run.log_wandb, run.wandb_project_name, run.wandb_group, run.exp_name, run.wandb_mode, info)

    env = create_env(run.env, backend=run.backend)
    eval_env = create_env(run.eval_env, backend=run.backend) if run.eval_env else env
    recorder = MetricsRecorder(run_dir, run.exp_name)
    _, params, _ = train(agent, run, env, eval_env, progress_fn=recorder.progress)

    with open(f"{run_dir}/ckpt/final.pkl", "wb") as f:
        pickle.dump(tuple(nnx.state(m) for m in params), f)
    wandb.finish()


def cli():
    tyro.cli(
        main,
        config=(tyro.conf.OmitArgPrefixes, tyro.conf.OmitSubcommandPrefixes, tyro.conf.ConsolidateSubcommandArgs),
    )


if __name__ == "__main__":
    cli()
