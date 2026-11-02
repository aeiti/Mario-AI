"""Train a PPO agent on Yoshi's Island 1.

    uv run scripts/train.py --run yi1 --watch            # full run, with a live window
    uv run scripts/train.py --run yi1                    # full run, headless
    uv run scripts/train.py --run smoke --timesteps 50000
    uv run scripts/train.py --run yi1 --resume checkpoints/yi1/ppo_2000000_steps.zip

Outputs, per run name:
    runs/<run>/          TensorBoard logs:  uv run tensorboard --logdir runs
    checkpoints/<run>/   periodic saves, best.zip (best evaluation so far), attempts.csv
    videos/<run>/        an MP4 of one evaluation episode every eval_freq steps
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from mario_ai.callbacks import (
    AttemptLogCallback,
    EpisodeStatsCallback,
    EvalVideoCallback,
    PhaseTimerCallback,
    TrainingControlCallback,
)
from mario_ai.config import TrainConfig, linear_schedule
from mario_ai.evaluation import make_training_env
from mario_ai.viewer import LiveView

ROOT = Path(__file__).resolve().parent.parent


def default_device() -> str:
    # The network update is ~17x faster on the Apple GPU than the CPU (0.28 s
    # vs 4.8 s per round here), and the emulators sit idle while it runs.
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="name for this run's logs, checkpoints and videos")
    parser.add_argument("--timesteps", type=int, help="override TrainConfig.total_timesteps")
    parser.add_argument("--n-envs", type=int, help="override TrainConfig.n_envs")
    parser.add_argument("--eval-freq", type=int, help="override TrainConfig.eval_freq (0 disables evaluation)")
    parser.add_argument("--resume", type=Path, help="checkpoint .zip to continue from")
    parser.add_argument("--device", default=default_device(), help="torch device (default: mps when available)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch", action="store_true", help="live window on one env, with Pause/Save/Stop controls")
    args = parser.parse_args()

    config = TrainConfig()
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.n_envs:
        config.n_envs = args.n_envs
    if args.eval_freq is not None:
        config.eval_freq = args.eval_freq

    log_dir = ROOT / "runs"
    checkpoint_dir = ROOT / "checkpoints" / args.run
    video_dir = ROOT / "videos" / args.run
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))

    live = LiveView() if args.watch else None
    env = make_training_env(config, seed=args.seed, live_feed=live.name if live else None)

    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device, tensorboard_log=str(log_dir))
    else:
        model = PPO(
            "CnnPolicy",
            env,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            learning_rate=linear_schedule(config.learning_rate),
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            tensorboard_log=str(log_dir),
            device=args.device,
            seed=args.seed,
            verbose=1,
        )

    callbacks = [
        PhaseTimerCallback(),
        EpisodeStatsCallback(),
        AttemptLogCallback(checkpoint_dir / "attempts.csv"),
        # SB3 counts checkpoint frequency in calls to the vectorised env,
        # and each call is n_envs transitions.
        CheckpointCallback(save_freq=max(config.checkpoint_freq // config.n_envs, 1), save_path=str(checkpoint_dir), name_prefix="ppo"),
    ]
    if live:
        callbacks.append(TrainingControlCallback(live, checkpoint_dir))
    if config.eval_freq:
        callbacks.append(EvalVideoCallback(config, video_dir=video_dir, checkpoint_dir=checkpoint_dir, verbose=1))
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name=args.run,
            reset_num_timesteps=not args.resume,
        )
    finally:
        # Ctrl-C still leaves a usable model behind.
        model.save(checkpoint_dir / "final")
        env.close()
        if live:
            live.close()


if __name__ == "__main__":
    main()
