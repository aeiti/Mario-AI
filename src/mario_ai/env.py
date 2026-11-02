"""Builds the environment the agent trains and plays in.

The raw stable-retro env gives 224x256 RGB frames, 12-button actions, and a
score-based reward. Each wrapper changes one thing, innermost first:

    RetroEnv          the emulator itself, 60 frames per second
    Discretizer       pick 1 of 10 button combos instead of 12 buttons
    MaxAndSkip        hold each choice for 4 frames: 15 decisions per second
    RandomNoopStart   start after a random short wait, so episodes differ
    ProgressReward    our reward; end the episode on death, clear, or stall
    WarpFrame         84x84 grayscale observation for the CNN

Frame stacking (so the agent can see motion) happens later, across the whole
vectorised batch, with SB3's VecFrameStack.
"""

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import stable_retro as retro

from mario_ai import ram
from mario_ai.actions import Discretizer
from mario_ai.config import TrainConfig
from mario_ai.viewer import LiveViewer
from mario_ai.wrappers import GameState, MaxAndSkip, ProgressReward, RandomNoopStart, WarpFrame


def read_state(env: gym.Env) -> GameState:
    r = env.unwrapped.get_ram()
    return GameState(x=ram.player_x(r), dying=ram.is_dying(r), cleared=ram.level_cleared(r))


class FrameRecorder(gym.Wrapper):
    """Keeps every full-colour emulator frame of the episode, for videos.

    Sits directly on the emulator, below the frame skip, so videos play at the
    real 60 fps rather than the agent's 15.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.frames: list[np.ndarray] = []
        # The previous episode's frames. A VecEnv resets the env the moment an
        # episode ends, so by the time anyone asks, `frames` is the new episode.
        self.last_frames: list[np.ndarray] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.frames:
            self.last_frames = self.frames
        self.frames = [obs]
        return obs, info

    def step(self, action):
        obs, *rest = self.env.step(action)
        self.frames.append(obs)
        return (obs, *rest)


def make_env(
    config: TrainConfig,
    state: str | None = None,
    record: bool = False,
    on_frame: Callable[[np.ndarray], None] | None = None,
    live_view: bool = False,
) -> gym.Env:
    """One wrapped environment.

    ``on_frame`` is called with every RGB frame (for watch.py); ``live_view``
    opens a window showing this env as it trains.
    """
    env = retro.make(
        config.game,
        state=state or config.state,
        render_mode="rgb_array",
        # Our reward and termination replace the built-in scenario entirely.
        use_restricted_actions=retro.Actions.ALL,
    )
    if record:
        env = FrameRecorder(env)
    if on_frame:
        env = FrameCallback(env, on_frame)
    env = Discretizer(env)
    env = MaxAndSkip(env, skip=config.frame_skip)
    env = RandomNoopStart(env, max_noops=config.max_noops)
    env = ProgressReward(env, read_state=read_state, config=config.reward)
    if live_view:
        env = LiveViewer(env)
    env = WarpFrame(env)
    return env


class FrameCallback(gym.Wrapper):
    def __init__(self, env: gym.Env, on_frame: Callable[[np.ndarray], None]):
        super().__init__(env)
        self.on_frame = on_frame

    def step(self, action):
        obs, *rest = self.env.step(action)
        self.on_frame(obs)
        return (obs, *rest)
