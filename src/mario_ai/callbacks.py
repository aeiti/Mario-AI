"""Training callbacks: TensorBoard episode stats, an attempt log, and periodic evaluation with video."""

import csv
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from mario_ai.config import TrainConfig
from mario_ai.evaluation import eval_worker_init, evaluate_snapshot


class EpisodeStatsCallback(BaseCallback):
    """Logs why episodes ended and how far Mario got, averaged per rollout.

    SB3 already logs the mean return (rollout/ep_rew_mean). These add the
    numbers that say what the return means: distance, clear/death/stall rates,
    and each reward component separately.
    """

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.logger.record_mean("episode/max_x", info["max_x"])
            self.logger.record_mean("episode/clear_rate", float(info["cleared"]))
            self.logger.record_mean("episode/death_rate", float(info["died"]))
            self.logger.record_mean("episode/stall_rate", float(info["stalled"]))
            for name, value in info["reward_parts"].items():
                self.logger.record_mean(f"reward/{name}", value)
        return True


class EvalVideoCallback(BaseCallback):
    """Every ``eval_freq`` transitions: test the agent in the background, log it, save a video.

    Evaluation runs in a pool of separate processes, one episode each, while
    training carries on; nothing waits for it. The current network is saved
    to a snapshot file, each worker loads it and plays one episode, and the
    results are logged when they come back (a minute or so later).

    Episodes use the deterministic policy (always the most likely action), so
    they show what the agent has learned, without exploration noise. The
    first episode always starts from the same seed and is saved as an MP4, so
    consecutive videos compare like with like. The best snapshot so far is
    kept as ``best.zip``.
    """

    def __init__(self, config: TrainConfig, video_dir: Path, checkpoint_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.video_dir = video_dir
        self.checkpoint_dir = checkpoint_dir
        self.best = (-1.0, -1.0)  # (clear rate, mean max_x), compared in that order
        self.next_eval = 0
        self.pending = None  # (timesteps, snapshot path, futures) of the running evaluation

    def _on_training_start(self) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)
        # "spawn" gives each worker a clean interpreter (no inherited emulator
        # or torch state); workers are reused, so the startup cost is paid once.
        self.pool = ProcessPoolExecutor(
            max_workers=self.config.n_eval_episodes,
            mp_context=mp.get_context("spawn"),
            initializer=eval_worker_init,
        )

    def _on_step(self) -> bool:
        if self.pending and all(f.done() for f in self.pending[2]):
            self.collect()
        # Start a new evaluation only when the previous one has finished, so
        # evaluations can't pile up if they're ever slower than training.
        if self.num_timesteps >= self.next_eval and self.pending is None:
            self.next_eval = self.num_timesteps + self.config.eval_freq
            self.launch()
        return True

    def launch(self) -> None:
        steps = self.num_timesteps
        snapshot = self.checkpoint_dir / f"eval_snapshot_{steps}.zip"
        self.model.save(snapshot)
        video = self.video_dir / f"step_{steps:09d}.mp4"
        futures = [
            self.pool.submit(evaluate_snapshot, snapshot, self.config, 1000 + i, video if i == 0 else None)
            for i in range(self.config.n_eval_episodes)
        ]
        self.pending = (steps, snapshot, futures)

    def collect(self) -> None:
        steps, snapshot, futures = self.pending
        self.pending = None
        results = [f.result() for f in futures]
        clear_rate = float(np.mean([r.cleared for r in results]))
        mean_x = float(np.mean([r.max_x for r in results]))
        # Logged at the current step, a little after the snapshot was taken;
        # eval/snapshot_timesteps records exactly which network was tested.
        self.logger.record("eval/clear_rate", clear_rate)
        self.logger.record("eval/max_x", mean_x)
        self.logger.record("eval/max_x_best_episode", max(r.max_x for r in results))
        self.logger.record("eval/snapshot_timesteps", steps)

        if (clear_rate, mean_x) > self.best:
            self.best = (clear_rate, mean_x)
            snapshot.replace(self.checkpoint_dir / "best.zip")
        else:
            snapshot.unlink()
        if self.verbose:
            print(f"eval of step {steps:,}: clear {clear_rate:.0%}, max_x {mean_x:.0f} -> step_{steps:09d}.mp4")

    def _on_training_end(self) -> None:
        if self.pending:
            for f in self.pending[2]:
                f.result()
            self.collect()
        self.pool.shutdown()


class AttemptLogCallback(BaseCallback):
    """Appends one CSV row per finished attempt, from every env.

    This is the raw material for scripts/progress.py: how far each attempt
    got and how it ended, in the order they happened.
    """

    FIELDS = ("attempt", "timesteps", "env", "max_x", "outcome", "reward")

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.attempt = 0

    def _on_training_start(self) -> None:
        resuming = self.path.exists()
        if resuming:
            with self.path.open() as f:
                self.attempt = max(sum(1 for _ in f) - 1, 0)
        self.file = self.path.open("a", newline="")
        self.writer = csv.writer(self.file)
        if not resuming:
            self.writer.writerow(self.FIELDS)

    def _on_step(self) -> bool:
        for env_index, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if not done:
                continue
            self.attempt += 1
            outcome = "cleared" if info["cleared"] else "died" if info["died"] else "stalled"
            reward = info.get("episode", {}).get("r", float("nan"))
            self.writer.writerow((self.attempt, self.num_timesteps, env_index, info["max_x"], outcome, round(reward, 2)))
        self.file.flush()  # so progress.py can read it while training runs
        return True

    def _on_training_end(self) -> None:
        self.file.close()


class PhaseTimerCallback(BaseCallback):
    """Logs how long each training round spends playing vs. learning.

    PPO alternates two phases: a rollout (all emulators play n_steps each, in
    parallel) and an update (the network trains on what they collected, while
    the emulators wait). Knowing the split says where speed-ups would help:
    more emulator processes shorten the rollout; more threads or the GPU
    shorten the update.
    """

    def _on_rollout_start(self) -> None:
        now = time.perf_counter()
        if hasattr(self, "rollout_end"):
            self.logger.record("time/update_seconds", now - self.rollout_end)
        self.rollout_start = now

    def _on_rollout_end(self) -> None:
        self.rollout_end = time.perf_counter()
        self.logger.record("time/rollout_seconds", self.rollout_end - self.rollout_start)

    def _on_step(self) -> bool:
        return True
