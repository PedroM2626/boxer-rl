"""Symmetric self-play plumbing: one shared policy drives both boxers.

Brax's training stack is single-agent, and its ``EpisodeWrapper`` assumes
``reward`` and ``done`` share a leading shape. A duel breaks that in two ways at
once: there are two learners per physics world, and the round ends for both of
them together. So the bookkeeping is done here at the world level, and the agent
axis is folded into the batch only at the very end, where Brax sees an ordinary
single-agent env with twice as many learners.

Because both corners are driven by the same network and see the same
egocentric observation, this is symmetric self-play against the current policy
rather than a centralised controller that could arrange the outcome.
"""

from __future__ import annotations

from typing import Any, Dict

from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import AutoResetWrapper, VmapWrapper
import jax
from jax import numpy as jp

from boxer.env import NUM_AGENTS


class RoundEpisode(Wrapper):
    """Episode bookkeeping for a shared round.

    ``done`` stays per-world because both boxers are reset together; ``reward``
    keeps its agent axis. Everything the training wrappers log is reduced to a
    per-world scalar so the shapes downstream stay uniform.
    """

    def __init__(self, env: Env, episode_length: int):
        super().__init__(env)
        self.episode_length = episode_length

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        world = rng.shape[:-1]
        zeros = jp.zeros(world)
        state.info["steps"] = zeros
        state.info["truncation"] = zeros
        state.info["time_out"] = zeros
        state.info["episode_done"] = zeros
        episode = {"sum_reward": zeros, "length": zeros}
        episode["sum_reward_red"] = zeros
        episode["sum_reward_blue"] = zeros
        for name in state.metrics:
            episode[name] = jp.zeros(world)
        state.info["episode_metrics"] = episode
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        steps = state.info["steps"] + 1
        timeout = steps >= self.episode_length
        # PPO bootstraps the value on a timeout, so a round that merely ran out
        # of clock must not look like a knockout to the critic.
        state.info["time_out"] = jp.where(
            timeout & (state.done == 0), 1.0, jp.zeros_like(state.done)
        )
        done = jp.where(timeout, jp.ones_like(state.done), state.done)
        state.info["truncation"] = jp.where(
            timeout, 1.0 - state.done, jp.zeros_like(state.done)
        )
        state.info["steps"] = steps

        fresh = 1.0 - state.info["episode_done"]
        episode = state.info["episode_metrics"]
        reward = state.reward
        episode["sum_reward"] = episode["sum_reward"] * fresh + jp.mean(reward, axis=-1)
        episode["sum_reward_red"] = episode["sum_reward_red"] * fresh + reward[..., 0]
        episode["sum_reward_blue"] = episode["sum_reward_blue"] * fresh + reward[..., 1]
        episode["length"] = episode["length"] * fresh + 1.0
        for name, value in state.metrics.items():
            episode[name] = episode[name] * fresh + value
        state.info["episode_done"] = done
        return state.replace(done=done)


class SelfPlay(Wrapper):
    """Folds the agent axis into the batch axis.

    Input and output are exact inverses: per-agent fields are reshaped between
    ``(worlds, 2, ...)`` and ``(worlds * 2, ...)``, per-world fields are spread
    and re-gathered. ``pipeline_state`` keeps its world-level batch, which is
    what lets the reset wrapper upstream still replace the shared physics state.
    """

    def __init__(self, env: Env, observation_dim: int):
        super().__init__(env)
        self._obs_dim = observation_dim

    @property
    def observation_size(self) -> int:
        return self._obs_dim

    def reset(self, rng: jax.Array) -> State:
        keys = rng.reshape(-1, NUM_AGENTS, rng.shape[-1])[:, 0]
        return _unfold(self.env.reset(keys), self._obs_dim)

    def step(self, state: State, action: jax.Array) -> State:
        inner = _fold(state, self._obs_dim)
        action = action.reshape(-1, NUM_AGENTS, action.shape[-1])
        return _unfold(self.env.step(inner, action), self._obs_dim)


def _spread(x: jax.Array) -> jax.Array:
    """(worlds, ...) -> (worlds * 2, ...) by repeating each world twice."""
    return jp.repeat(x, NUM_AGENTS, axis=0)


def _gather(x: jax.Array) -> jax.Array:
    """(worlds * 2, ...) -> (worlds, ...), the inverse of :func:`_spread`."""
    return x.reshape(-1, NUM_AGENTS, *x.shape[1:])[:, 0]


# Info fields that describe the round rather than one boxer. Everything else in
# ``info`` (the physics state, the pending action) keeps its world-level shape.
_ROUND_INFO = ("steps", "truncation", "time_out", "episode_done")


def _unfold(state: State, obs_dim: int) -> State:
    info: Dict[str, Any] = {}
    for key, value in state.info.items():
        if key == "first_obs":
            info[key] = value.reshape(-1, obs_dim)
        elif key == "episode_metrics":
            info[key] = {name: _spread(v) for name, v in value.items()}
        elif key in _ROUND_INFO:
            info[key] = _spread(value)
        else:
            info[key] = value
    return state.replace(
        obs=state.obs.reshape(-1, obs_dim),
        reward=state.reward.reshape(-1),
        done=_spread(state.done),
        metrics={name: _spread(v) for name, v in state.metrics.items()},
        info=info,
    )


def _fold(state: State, obs_dim: int) -> State:
    info: Dict[str, Any] = {}
    for key, value in state.info.items():
        if key == "first_obs":
            info[key] = value.reshape(-1, NUM_AGENTS, obs_dim)
        elif key == "episode_metrics":
            info[key] = {name: _gather(v) for name, v in value.items()}
        elif key in _ROUND_INFO:
            info[key] = _gather(value)
        else:
            info[key] = value
    return state.replace(
        obs=state.obs.reshape(-1, NUM_AGENTS, obs_dim),
        reward=state.reward.reshape(-1, NUM_AGENTS),
        done=_gather(state.done),
        metrics={name: _gather(v) for name, v in state.metrics.items()},
        info=info,
    )


class FrozenOpponent(Wrapper):
    """Drives one corner with a fixed policy and exposes the other as the learner.

    After this wrapper the duel is single-agent again -- one learner per physics
    world, against an opponent that does not move under it. That is the
    "self-play without MARL" baseline, and it is what makes an opponent-difficulty
    curriculum and a win-rate measurement possible at all.

    The opponent acts on the *same* observation the learner acts on, so the two
    corners stay simultaneous rather than one getting to react first.

    ``opponent`` must be deterministic. That is what "frozen" means here, and it
    also sidesteps JAX's key batching: ``jax.random.split`` refuses a stacked key
    array, so sampling per world would need the opponent to run inside the env's
    vmap rather than alongside the learner's batched observations.
    """

    def __init__(self, env: Env, opponent, learner_side: int = 0):
        super().__init__(env)
        self._opponent = opponent
        self._side = learner_side

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    def _project(self, state: State) -> State:
        # Brax hands the state this returns straight back into the next step, and
        # the opponent needs the *other* corner's observation, so the full
        # per-agent obs travels along in info. Projecting it away made the second
        # step index a 2-D array as if it still had the agent axis, which silently
        # worked for an opponent that only reads ``obs.shape[0]`` -- like
        # ``passive`` -- and exploded for one that reads its observation.
        state = state.replace(info={**state.info, "both_obs": state.obs})
        return state.replace(
            obs=state.obs[:, self._side],
            reward=state.reward[:, self._side],
        )

    def reset(self, rng: jax.Array) -> State:
        return self._project(self.env.reset(rng))

    def step(self, state: State, action: jax.Array) -> State:
        opponent_action = self._opponent(
            state.info["both_obs"][:, 1 - self._side], state.pipeline_state.time, None
        )
        order = [action, opponent_action] if self._side == 0 else [opponent_action, action]
        return self._project(self.env.step(state, jp.stack(order, axis=1)))


def wrap(env: Env, episode_length: int, observation_dim: int) -> Env:
    """Wraps an unbatched duel into the batched single-agent env Brax expects."""
    env = VmapWrapper(env)
    env = RoundEpisode(env, episode_length)
    env = AutoResetWrapper(env)
    return SelfPlay(env, observation_dim)


def wrap_frozen(env: Env, opponent, episode_length: int, learner_side: int = 0) -> Env:
    """Wraps an unbatched duel so Brax sees one learner per physics world."""
    env = VmapWrapper(env)
    env = RoundEpisode(env, episode_length)
    env = AutoResetWrapper(env)
    return FrozenOpponent(env, opponent, learner_side)
