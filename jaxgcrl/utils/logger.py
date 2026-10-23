# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL),
# Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
import logging
import math
import os

import jax
import wandb
from brax.io import html


class CsvLogger:
    """Append-only CSV logger whose header is fixed by the first logged row."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None

    def log(self, row, step):
        row = {"step": step, **row}
        if self.file is None:
            self.file = open(self.path, "w")
            self.header = list(row.keys())
            self.file.write(",".join(self.header) + "\n")
        self.file.write(",".join(str(row.get(k, "")) for k in self.header) + "\n")
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def setup_wandb(enabled, project, group, name, mode, config):
    """Initialize wandb, falling back to offline and then disabled mode on failure."""
    if not enabled:
        return wandb.init(mode="disabled")
    first = os.environ.get("WANDB_MODE", "").strip().lower() or mode
    for attempt in dict.fromkeys((first, mode, "offline", "disabled")):
        try:
            return wandb.init(project=project, group=group, name=name, mode=attempt, config=config)
        except Exception:
            logging.exception("wandb.init(mode=%r) failed", attempt)
            wandb.teardown()
    return None


class MetricsRecorder:
    """Logs scalar metrics to wandb and to <run_dir>/metrics.csv after each evaluation."""

    def __init__(self, run_dir, exp_name):
        self.run_dir = run_dir
        self.exp_name = exp_name
        self.csv = CsvLogger(os.path.join(run_dir, "metrics.csv"))

    def progress(self, num_steps, metrics, make_policy, params, env, do_render=False):
        metrics = {k: float(v) for k, v in metrics.items()}
        nan = [k for k, v in metrics.items() if math.isnan(v)]
        if nan:
            raise RuntimeError(f"NaN metrics at step {num_steps}: {nan}")
        if do_render:
            render(make_policy, params, env, self.run_dir, self.exp_name, num_steps)
        wandb.log(metrics, step=num_steps)
        self.csv.log(metrics, num_steps)


def render(make_policy, params, env, run_dir, exp_name, num_steps):
    """Roll out the deterministic policy and save/log an HTML visualization."""
    policy = jax.jit(make_policy(params))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key)
    state = reset(rng=subkey)
    rollout = []
    for i in range(5000):
        rollout.append(state.pipeline_state)
        key, subkey = jax.random.split(key)
        action, _ = policy(state.obs[None], subkey)
        state = step(state, jax.numpy.tanh(action[0]))
        if i % 1000 == 0:
            key, subkey = jax.random.split(key)
            state = reset(rng=subkey)
    page = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)
    with open(os.path.join(run_dir, f"{exp_name}_{num_steps}.html"), "w") as f:
        f.write(page)
    wandb.log({"render": wandb.Html(page)})
