"""Watch a trained agent play, in a window, at real speed.

    uv run scripts/watch.py checkpoints/yi1/best.zip
    uv run scripts/watch.py checkpoints/yi1/best.zip --episodes 5 --speed 2

Unlike the live view during training, this uses the deterministic policy
(the agent's single favourite action each step), so it shows what the agent
has actually learned. Each episode starts after a different random wait, so
repeated episodes aren't identical replays.
"""

import argparse
import time
from pathlib import Path

import cv2
from stable_baselines3 import PPO

from mario_ai.config import TrainConfig
from mario_ai.evaluation import play_episode

WINDOW = "Mario-AI: watch"


class RealTimeWindow:
    """Shows every emulator frame, sleeping as needed to hold 60 fps x speed."""

    def __init__(self, speed: float, scale: int = 3):
        self.frame_time = 1 / (60 * speed)
        self.scale = scale
        self.next_frame = time.perf_counter()

    def __call__(self, frame) -> None:
        img = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WINDOW, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.pollKey()
        self.next_frame += self.frame_time
        delay = self.next_frame - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_frame = time.perf_counter()  # fell behind; don't try to catch up


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    parser.add_argument("--stochastic", action="store_true", help="sample actions, as during training")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    config = TrainConfig()
    window = RealTimeWindow(args.speed)
    for i in range(args.episodes):
        result = play_episode(model, config, seed=i, deterministic=not args.stochastic, on_frame=window)
        how = "CLEARED" if result.cleared else "died" if result.died else "stalled"
        print(f"episode {i + 1}: {how} at x={result.max_x} after {result.steps} steps")
        time.sleep(1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
