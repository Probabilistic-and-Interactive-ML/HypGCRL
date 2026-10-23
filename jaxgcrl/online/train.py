# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL) and
# scaling-crl (https://github.com/wang-kevin3290/scaling-crl), Copyright 2023 FLAIR, Apache License 2.0.
# Modified for this repository in 2026.
"""Online training loop: collect, relabel, update contrastive critic and SAC actor, evaluate (B batch, L unroll, N envs)."""

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs
from flax import nnx

from jaxgcrl.envs.wrappers import Float32InfoWrapper, TrajectoryIdWrapper
from jaxgcrl.online.losses import update_actor_and_alpha, update_critic
from jaxgcrl.online.networks import Actor, Alpha
from jaxgcrl.utils.evaluator import ActorEvaluator, fix_reset_info_dtypes
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_batch(buffer_config, transition, sample_key):
    """Relabel one trajectory window with geometrically discounted future goals (hindsight relabeling)."""
    gamma, state_size, goal_indices = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate([transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0)
    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len, dtype=jnp.float32) * 1e-5
    goal_index = jax.random.categorical(sample_key, jnp.log(probs))

    future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state": state,
        "future_state": future_state,
    }
    return transition._replace(
        observation=jnp.squeeze(new_obs),
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def _prefill_steps(agent, config):
    """Prefill enough rows for the sampler to draw one full episode-length window."""
    min_rows = max(agent.min_replay_size, config.episode_length)
    assert agent.max_replay_size >= config.episode_length, "max_replay_size must be >= episode_length"
    return min_rows * config.num_envs, -(-min_rows // agent.unroll_length)


def _actor_step(actor, env, env_state, noise_key=None):
    means, log_stds = actor(env_state.obs)
    if noise_key is not None:
        means = means + jnp.exp(log_stds) * jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
    actions = jax.nn.tanh(means)
    nstate = env.step(env_state, actions.astype(env_state.pipeline_state.q.dtype))
    return nstate, Transition(
        observation=env_state.obs,
        action=actions.astype(jnp.float32),
        reward=nstate.reward,
        discount=1 - nstate.done,
        extras={"state_extras": {x: nstate.info[x] for x in ("truncation", "traj_id")}},
    )


def train(
    agent,
    config,
    train_env: envs.Env,
    eval_env: envs.Env,
    progress_fn: Callable[..., None] = lambda *args, **kwargs: None,
):
    """Train ``agent`` (a CRL / HypCRL config) and return ``(make_policy, params, metrics)``."""
    agent.check_config(config)
    unwrapped_env = train_env

    def wrap(env):
        env = envs.training.wrap(TrajectoryIdWrapper(env), episode_length=config.episode_length, action_repeat=1)
        return Float32InfoWrapper(env) if agent.manifold_float64 else env

    train_env, eval_env = wrap(train_env), wrap(eval_env)

    env_steps_per_actor_step = config.num_envs * agent.unroll_length
    num_prefill_env_steps, num_prefill_actor_steps = _prefill_steps(agent, config)
    num_training_steps_per_epoch = int(
        np.ceil((config.total_env_steps - num_prefill_env_steps) / (config.num_evals * env_steps_per_actor_step))
    )
    assert num_training_steps_per_epoch > 0, "total_env_steps too small for given num_envs and episode_length"

    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    buffer_key, eval_env_key, env_key, actor_key, critic_key, noise_key, sampling_key, perm_key = jax.random.split(key, 8)

    env_state = fix_reset_info_dtypes(jax.jit(train_env.reset)(jax.random.split(env_key, config.num_envs)))
    train_env.step = jax.jit(train_env.step)

    action_size = train_env.action_size
    state_size = train_env.state_dim
    goal_indices = tuple(int(i) for i in train_env.goal_indices)
    obs_size = state_size + len(goal_indices)
    assert obs_size == train_env.observation_size, f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"

    actor = Actor(
        obs_size, action_size, agent.actor_network_width, agent.actor_depth, agent.use_relu, rngs=nnx.Rngs(actor_key)
    )
    critic = agent.build_critic(state_size + action_size, len(goal_indices), nnx.Rngs(critic_key))
    alpha = Alpha(init=float(np.log(agent.fixed_alpha)) if agent.fixed_alpha is not None else 0.0)
    actor_opt = nnx.Optimizer(actor, optax.adam(agent.policy_lr), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, optax.adam(agent.critic_lr), wrt=nnx.Param)
    alpha_opt = nnx.Optimizer(alpha, optax.adam(agent.alpha_lr), wrt=nnx.Param)
    rngs = nnx.Rngs(noise=noise_key, sampling=sampling_key, permutation=perm_key)

    ctx = dict(
        state_size=state_size,
        goal_indices=goal_indices,
        target_entropy=-0.5 * action_size,
        contrastive_loss_fn=agent.contrastive_loss_fn,
        logsumexp_penalty_coeff=agent.logsumexp_penalty_coeff,
        fixed_alpha=agent.fixed_alpha,
        manifold_float64=agent.manifold_float64,
    )
    buffer_config = (agent.discounting, state_size, goal_indices)

    dummy_transition = Transition(
        observation=jnp.zeros((obs_size,), dtype=jnp.float32),
        action=jnp.zeros((action_size,), dtype=jnp.float32),
        reward=jnp.float32(0.0),
        discount=jnp.float32(0.0),
        extras={"state_extras": {"truncation": jnp.float32(0.0), "traj_id": jnp.float32(0.0)}},
    )
    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=agent.max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=agent.batch_size,
        num_envs=config.num_envs,
        episode_length=config.episode_length,
    )
    replay_buffer.insert_internal = jax.jit(replay_buffer.insert_internal)
    replay_buffer.sample_internal = jax.jit(replay_buffer.sample_internal)
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    def get_experience(env_state, buffer_state, rngs, actor):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def collect_step(carry, _):
            env_state, rngs = carry
            env_state, transition = _actor_step(actor, train_env, env_state, rngs.noise())
            success_N = env_state.metrics["success"] if "success" in env_state.metrics else env_state.reward
            return (env_state, rngs), (transition, success_N)

        (env_state, _), (data, success_LN) = collect_step((env_state, rngs), jnp.arange(agent.unroll_length))
        return env_state, replay_buffer.insert(buffer_state, data), jnp.mean(success_LN)

    @nnx.jit
    def prefill(actor, env_state, buffer_state, rngs):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step(carry, _):
            env_state, buffer_state, rngs = carry
            env_state, buffer_state, _ = get_experience(env_state, buffer_state, rngs, actor)
            return (env_state, buffer_state, rngs), ()

        (env_state, buffer_state, _), () = step((env_state, buffer_state, rngs), jnp.arange(num_prefill_actor_steps))
        return env_state, buffer_state

    # Each sweep runs a random subset of num_sgd_batches of the available minibatches.
    num_minibatches = config.num_envs * (config.episode_length - 1) // agent.batch_size
    num_sgd_batches = min(agent.num_sgd_batches_per_step or num_minibatches, num_minibatches)
    sgd_batch_key = jax.random.fold_in(perm_key, 1)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def update_networks(carry, transitions):
        actor, actor_opt, critic, critic_opt, alpha, alpha_opt, rngs = carry
        actor_metrics = update_actor_and_alpha(ctx, actor, actor_opt, critic, alpha, alpha_opt, transitions, rngs)
        critic_metrics = update_critic(ctx, critic, critic_opt, transitions)
        return (actor, actor_opt, critic, critic_opt, alpha, alpha_opt, rngs), {**actor_metrics, **critic_metrics}

    log_every = max(1, num_training_steps_per_epoch // 10)

    def _log_step(step, critic_loss, actor_loss, cat_acc, entropy):
        print(
            f"  [step {int(step):>5d}/{num_training_steps_per_epoch}]  critic_loss={float(critic_loss):.4f}"
            f"  actor_loss={float(actor_loss):.4f}  cat_acc={float(cat_acc):.3f}  entropy={float(entropy):.3f}"
        )

    @nnx.jit
    def training_epoch(actor, actor_opt, critic, critic_opt, alpha, alpha_opt, env_state, buffer_state, rngs):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def epoch_step(carry, step_idx):
            actor, actor_opt, critic, critic_opt, alpha, alpha_opt, env_state, buffer_state, rngs = carry
            env_state, buffer_state, success_frac = get_experience(env_state, buffer_state, rngs, actor)

            for sweep_idx in range(agent.train_step_multiplier):
                buffer_state, transitions = replay_buffer.sample(buffer_state)
                batch_keys = jax.random.split(rngs.sampling(), transitions.observation.shape[0])
                transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(buffer_config, transitions, batch_keys)
                transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:]), order="F"), transitions)
                permutation = jax.random.permutation(rngs.permutation(), len(transitions.observation))
                transitions = jax.tree_util.tree_map(lambda x, p=permutation: x[p], transitions)
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, agent.batch_size, *x.shape[1:])), transitions
                )
                if num_sgd_batches < num_minibatches:
                    subset_key = jax.random.fold_in(sgd_batch_key, step_idx * agent.train_step_multiplier + sweep_idx)
                    selected = jax.random.permutation(subset_key, num_minibatches)[:num_sgd_batches]
                    transitions = jax.tree_util.tree_map(lambda x, s=selected: x[s], transitions)

                (actor, actor_opt, critic, critic_opt, alpha, alpha_opt, rngs), metrics = update_networks(
                    (actor, actor_opt, critic, critic_opt, alpha, alpha_opt, rngs), transitions
                )

            metrics = {**jax.tree_util.tree_map(jnp.mean, metrics), "success_frac": success_frac}
            nnx.cond(
                step_idx % log_every == 0,
                lambda: jax.debug.callback(
                    _log_step,
                    step_idx,
                    metrics["critic_loss"],
                    metrics["actor_loss"],
                    metrics["categorical_accuracy"],
                    metrics["entropy"],
                ),
                lambda: None,
            )
            return (actor, actor_opt, critic, critic_opt, alpha, alpha_opt, env_state, buffer_state, rngs), metrics

        carry, metrics = epoch_step(
            (actor, actor_opt, critic, critic_opt, alpha, alpha_opt, env_state, buffer_state, rngs),
            jnp.arange(num_training_steps_per_epoch),
        )
        env_state, buffer_state = carry[6], carry[7]
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return env_state, buffer_state, metrics

    print(f"Prefilling replay buffer ({num_prefill_actor_steps} actor steps, {num_prefill_env_steps:,} env steps)...")
    env_state, buffer_state = prefill(actor, env_state, buffer_state, rngs)
    jax.block_until_ready(buffer_state)

    evaluator = ActorEvaluator(
        lambda actor, env, state: _actor_step(actor, env, state),
        eval_env,
        num_eval_envs=config.num_eval_envs,
        episode_length=config.episode_length,
        key=eval_env_key,
    )

    def make_policy(actor):
        return lambda obs, rng: actor(obs)

    print(f"Training {type(agent).__name__} on {config.env} | {num_training_steps_per_epoch} steps/epoch x {config.num_evals}")
    training_walltime = 0.0
    for epoch in range(config.num_evals):
        t = time.time()
        env_state, buffer_state, metrics = training_epoch(
            actor, actor_opt, critic, critic_opt, alpha, alpha_opt, env_state, buffer_state, rngs
        )
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x).block_until_ready(), metrics)
        epoch_time = time.time() - t
        training_walltime += epoch_time
        current_step = int((num_prefill_actor_steps + (epoch + 1) * num_training_steps_per_epoch) * env_steps_per_actor_step)
        sps = env_steps_per_actor_step * num_training_steps_per_epoch / epoch_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": current_step,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        metrics = evaluator.run_evaluation(actor, metrics)
        print(
            f"Epoch {epoch + 1}/{config.num_evals} | {current_step:,} steps | SPS={sps:,.0f}"
            f" | success={float(metrics.get('eval/success_rate', 0.0)):.3f}"
            f" | steps_at_goal={float(metrics.get('eval/episode_success', 0.0)):.1f}"
        )
        do_render = config.visualization_interval > 0 and epoch % config.visualization_interval == 0
        progress_fn(current_step, metrics, make_policy, actor, unwrapped_env, do_render=do_render)

    logging.info("total steps: %s", current_step)
    return make_policy, (alpha, actor, critic), metrics
