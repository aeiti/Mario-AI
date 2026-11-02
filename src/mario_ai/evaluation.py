"""Vectorised envs for training, and a single-episode runner for eval and watching."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from mario_ai.config import TrainConfig
from mario_ai.env import FrameRecorder, make_env

# Per-episode info keys that VecMonitor copies into its episode summary.
EPISODE_KEYS = ("max_x", "cleared", "died", "stalled")


def make_training_env(config: TrainConfig, seed: int = 0, live_view: bool = False) -> VecFrameStack:
    """n_envs emulators, each in its own process.

    A separate process per emulator is required, not just faster: stable-retro
    allows only one emulator instance per process. With ``live_view``, env 0
    shows its play in a window.
    """
    venv = SubprocVecEnv([lambda rank=i: make_env(config, live_view=live_view and rank == 0) for i in range(config.n_envs)])
    venv.seed(seed)  # env i gets seed + i at its first reset
    venv = VecMonitor(venv, info_keywords=EPISODE_KEYS)
    return VecFrameStack(venv, n_stack=config.frame_stack)


@dataclass
class EpisodeResult:
    seed: int
    max_x: int
    cleared: bool
    died: bool
    stalled: bool
    total_reward: float
    steps: int
    frames: list[np.ndarray] = field(default_factory=list, repr=False)


def play_episode(
    model: BaseAlgorithm,
    config: TrainConfig,
    seed: int,
    deterministic: bool = True,
    record: bool = False,
    on_frame: Callable[[np.ndarray], None] | None = None,
    max_steps: int = 20_000,
) -> EpisodeResult:
    """Plays one episode in this process. Records RGB frames if ``record``.

    The env is wrapped exactly as in training (including frame stacking), so
    the policy sees the same kind of observation it learned from.
    """
    venv = VecFrameStack(
        DummyVecEnv([lambda: make_env(config, record=record, on_frame=on_frame)]),
        n_stack=config.frame_stack,
    )
    venv.seed(seed)
    obs = venv.reset()
    total, steps = 0.0, 0
    for steps in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = venv.step(action)
        total += float(reward[0])
        if done[0]:
            break
    info = infos[0]
    frames = []
    if record:
        # The VecEnv auto-reset has already started a new episode, so the
        # finished one is in FrameRecorder.last_frames.
        recorder = venv.venv.envs[0]
        while not isinstance(recorder, FrameRecorder):
            recorder = recorder.env
        frames = recorder.last_frames
    venv.close()
    return EpisodeResult(
        seed=seed,
        max_x=int(info["max_x"]),
        cleared=bool(info["cleared"]),
        died=bool(info["died"]),
        stalled=bool(info["stalled"]),
        total_reward=total,
        steps=steps,
        frames=frames,
    )
