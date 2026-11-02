"""Vectorised envs for training, and single-episode runners for eval and watching."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from mario_ai.config import TrainConfig
from mario_ai.env import FrameRecorder, make_env

# Per-episode info keys that VecMonitor copies into its episode summary.
EPISODE_KEYS = ("max_x", "cleared", "died", "stalled")


def make_training_env(config: TrainConfig, seed: int = 0, live_feed: str | None = None) -> VecFrameStack:
    """n_envs emulators, each in its own process.

    A separate process per emulator is required, not just faster: stable-retro
    allows only one emulator instance per process. With ``live_feed`` (a
    viewer.LiveView's name), env 0 streams its frames to the live window.
    """
    venv = SubprocVecEnv([lambda rank=i: make_env(config, live_feed=live_feed if rank == 0 else None) for i in range(config.n_envs)])
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


def eval_worker_init() -> None:
    # Each evaluation worker plays one game on one CPU core. Torch would
    # otherwise start a thread per core in every worker and fight the
    # training emulators for the CPU.
    import torch

    torch.set_num_threads(1)


def evaluate_snapshot(snapshot: Path, config: TrainConfig, seed: int, video: Path | None) -> EpisodeResult:
    """Runs in an evaluation worker process: load a saved network, play one episode.

    Writes the episode to ``video`` if given, and returns the result without
    its frames (sending ~170 KB per frame back to the trainer would be waste).
    """
    model = PPO.load(snapshot, device="cpu")
    result = play_episode(model, config, seed=seed, record=video is not None)
    if video is not None:
        imageio.mimwrite(video, result.frames, fps=60, macro_block_size=16)
    result.frames = []
    return result
