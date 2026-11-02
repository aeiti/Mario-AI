"""Stitch a run's evaluation videos into one "progress reel".

    uv run scripts/reel.py yi1      # writes videos/yi1/reel.mp4

Every evaluation starts from the same seed, so each clip is the same attempt
replayed by a more-trained agent. Watching them back to back shows exactly
what it learned between checkpoints. Each clip is captioned with how many
training steps the agent had at the time.
"""

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent


def caption(frame, text: str):
    frame = frame.copy()
    cv2.putText(frame, text, (9, 217), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  # shadow
    cv2.putText(frame, text, (8, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--speed", type=float, default=2.0, help="playback speed-up (default 2x)")
    args = parser.parse_args()

    video_dir = ROOT / "videos" / args.run
    clips = sorted(video_dir.glob("step_*.mp4"))
    if not clips:
        raise SystemExit(f"no evaluation videos in {video_dir}")

    out = video_dir / "reel.mp4"
    with imageio.get_writer(out, fps=60 * args.speed, macro_block_size=16) as writer:
        for clip in clips:
            steps = int(clip.stem.split("_")[1])
            label = f"after {steps:,} training steps"
            for frame in imageio.get_reader(clip):
                writer.append_data(caption(frame, label))
    print(f"{out}  ({len(clips)} clips)")


if __name__ == "__main__":
    main()
