"""Gymnasium wrappers that turn the raw emulator into something PPO can learn.

Frame skipping and the 84x84 grayscale resize are the standard Atari
preprocessing, so we reuse Stable-Baselines3's tested versions rather than
rewriting them:

- ``MaxAndSkip`` is SB3's ``MaxAndSkipEnv``: repeat each action for k frames,
  sum the rewards, and return the pixel-wise max of the last two frames. The
  max removes flicker, where a sprite is drawn only on alternate frames.
- ``WarpFrame`` is SB3's ``WarpFrame``: grayscale, resize to 84x84 with
  shape (84, 84, 1). Colour carries little information for platforming, and
  a smaller image makes the CNN cheaper.

Frame stacking is left to SB3's ``VecFrameStack``, which does it once for the
whole vectorised batch.

The reward is ours: ``ProgressReward``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv as MaxAndSkip
from stable_baselines3.common.atari_wrappers import WarpFrame

from mario_ai.config import RewardConfig

__all__ = ["GameState", "MaxAndSkip", "ProgressReward", "RandomNoopStart", "WarpFrame"]


@dataclass(frozen=True)
class GameState:
    """The few facts about the game the reward needs, read after each step.

    This is deliberately separate from how the facts are obtained (RAM
    addresses, emulator variables), so the reward logic can be tested with
    made-up states and the RAM reading can change without touching it.
    """

    x: int  # Mario's horizontal position in the level, in pixels
    dying: bool  # the death animation has started
    cleared: bool  # the level has been completed


class ProgressReward(gym.Wrapper):
    """Rewards reaching new ground, and ends episodes that go nowhere.

    The emulator's built-in reward is replaced entirely. Per agent step:

        progress_scale * (new pixels beyond the best x so far) - time_penalty

    Only *new* ground counts. Rewarding plain movement to the right would let
    the agent farm reward by walking left and right over the same spot.

    Terminal events add to that step's reward:
      - dying:    -death_penalty, terminated
      - cleared:  +clear_reward, terminated
      - no new max x for stall_steps steps: truncated (no extra penalty;
        the time penalty has already been charging for the wait)

    Wrap this *outside* the frame skip, so that one step here is one agent
    decision and stall_steps counts decisions, not frames.

    ``read_state`` is injected rather than built in so tests can feed scripted
    states, and so the RAM details live in one place elsewhere.
    """

    def __init__(
        self,
        env: gym.Env,
        read_state: Callable[[gym.Env], GameState],
        config: RewardConfig | None = None,
    ):
        super().__init__(env)
        self.read_state = read_state
        self.config = config or RewardConfig()
        self._reset_stats(x=0)

    def _reset_stats(self, x: int) -> None:
        self.max_x = x
        self.steps_since_progress = 0
        # Summed reward components, so TensorBoard can show *why* an
        # episode's return was what it was, not just the total.
        self.totals = {"progress": 0.0, "time": 0.0, "death": 0.0, "clear": 0.0}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Start from wherever Mario spawns, so the spawn x is not "progress".
        self._reset_stats(x=self.read_state(self.env).x)
        return obs, info

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        state = self.read_state(self.env)
        cfg = self.config

        gained = max(0, state.x - self.max_x)
        if gained > 0:
            self.max_x = state.x
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        parts = {
            "progress": cfg.progress_scale * gained,
            "time": -cfg.time_penalty,
            "death": -cfg.death_penalty if state.dying else 0.0,
            "clear": cfg.clear_reward if state.cleared else 0.0,
        }
        for name, value in parts.items():
            self.totals[name] += value

        stalled = self.steps_since_progress >= cfg.stall_steps
        terminated = terminated or state.dying or state.cleared
        truncated = truncated or stalled

        # Running episode stats on every step. On the final step they are the
        # episode's totals, which is what a logging callback reads.
        info = dict(info)
        info["max_x"] = self.max_x
        info["cleared"] = state.cleared
        info["died"] = state.dying
        info["stalled"] = stalled
        info["reward_parts"] = dict(self.totals)

        return obs, sum(parts.values()), terminated, truncated, info


class RandomNoopStart(gym.Wrapper):
    """Waits a random number of steps after reset before the agent takes over.

    The emulator is perfectly deterministic: the same inputs give the same
    game, frame for frame. Without some randomness a policy could memorise one
    exact button sequence instead of learning to react to what it sees, and
    every evaluation episode would be an identical replay, so "clear rate"
    would always be 0% or 100%. Shifting the start by a few frames moves the
    enemies relative to Mario, which is enough to make episodes differ.

    (SB3's NoopResetEnv does the same thing but requires the Atari-only
    ``get_action_meanings`` API, which stable-retro doesn't have.)
    """

    def __init__(self, env: gym.Env, max_noops: int = 30, noop_action=0):
        super().__init__(env)
        self.max_noops = max_noops
        self.noop_action = noop_action
        # Our own generator: gym.Wrapper.np_random belongs to the inner env,
        # which isn't guaranteed to reseed it on reset(seed=...).
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, **kwargs):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, **kwargs)
        for _ in range(int(self.rng.integers(0, self.max_noops + 1))):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info
