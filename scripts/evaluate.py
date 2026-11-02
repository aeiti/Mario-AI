"""Measure a checkpoint: clear rate and distance over many episodes.

    uv run scripts/evaluate.py checkpoints/yi1/best.zip --episodes 20

Each episode uses a different seed (a different random wait before the agent
takes over), so the episodes are genuinely different. The milestone for this
project is a clear rate of at least 90% over 20 episodes.
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mario_ai.config import TrainConfig
from mario_ai.evaluation import play_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic policy")
    parser.add_argument("--target", type=float, default=0.9, help="clear rate that counts as a pass")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    config = TrainConfig()
    results = []
    for seed in range(args.episodes):
        r = play_episode(model, config, seed=seed, deterministic=not args.stochastic)
        how = "CLEARED" if r.cleared else "died" if r.died else "stalled"
        print(f"seed {seed:3d}: {how:8s} max_x {r.max_x:5d}  steps {r.steps:5d}  reward {r.total_reward:7.1f}")
        results.append(r)

    clear_rate = np.mean([r.cleared for r in results])
    print(f"\nclear rate {clear_rate:.0%} ({sum(r.cleared for r in results)}/{len(results)})")
    print(f"mean max_x {np.mean([r.max_x for r in results]):.0f}")
    passed = clear_rate >= args.target
    print("PASS" if passed else f"below target ({args.target:.0%})")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
