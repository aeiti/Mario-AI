"""Training callbacks: TensorBoard episode stats, an attempt log, and periodic evaluation with video."""

import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from mario_ai.config import TrainConfig
from mario_ai.evaluation import play_episode


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
    """Every ``eval_freq`` transitions: play a few episodes, log them, save a video.

    Episodes use the deterministic policy (always the most likely action), so
    they show what the agent has learned, without exploration noise. The
    first episode is saved as an MP4 so progress can be watched over time.
    The model with the best evaluation so far is saved as ``best.zip``.
    """

    def __init__(self, config: TrainConfig, video_dir: Path, checkpoint_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.video_dir = video_dir
        self.checkpoint_dir = checkpoint_dir
        self.best = (-1.0, -1.0)  # (clear rate, mean max_x), compared in that order
        self.next_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval = self.num_timesteps + self.config.eval_freq
            self.evaluate()
        return True

    def evaluate(self) -> None:
        results = [
            play_episode(self.model, self.config, seed=1000 + i, record=(i == 0))
            for i in range(self.config.n_eval_episodes)
        ]
        clear_rate = float(np.mean([r.cleared for r in results]))
        mean_x = float(np.mean([r.max_x for r in results]))
        self.logger.record("eval/clear_rate", clear_rate)
        self.logger.record("eval/max_x", mean_x)
        self.logger.record("eval/max_x_best_episode", max(r.max_x for r in results))

        self.video_dir.mkdir(parents=True, exist_ok=True)
        video = self.video_dir / f"step_{self.num_timesteps:09d}.mp4"
        imageio.mimwrite(video, results[0].frames, fps=60, macro_block_size=16)

        if (clear_rate, mean_x) > self.best:
            self.best = (clear_rate, mean_x)
            self.model.save(self.checkpoint_dir / "best")
        if self.verbose:
            print(f"eval @ {self.num_timesteps:,}: clear {clear_rate:.0%}, max_x {mean_x:.0f} -> {video.name}")


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
