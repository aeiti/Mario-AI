"""Check the RAM signals the reward depends on, headlessly.

Before training trusts an address in mario_ai/ram.py, this script makes the
game produce each event and prints what RAM says:

    uv run scripts/explore.py death   # run right into the first Rex
    uv run scripts/explore.py goal    # warp forward with star power to the goal tape

Add --frames DIR to save a PNG at the moment of the event.

To play the level yourself with the keyboard instead, use stable-retro's player:

    uv run python -m stable_retro.examples.interactive --game SuperMarioWorld-Snes-v0 --state YoshiIsland1
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import stable_retro as retro

from mario_ai import ram

GAME = "SuperMarioWorld-Snes-v0"
WRAM = 0x7E0000  # stable-retro addresses memory by absolute SNES address


def press(env, *names: str) -> np.ndarray:
    action = np.zeros(len(env.unwrapped.buttons), dtype=np.int8)
    for name in names:
        action[env.unwrapped.buttons.index(name)] = 1
    return action


def status(t: int, r: np.ndarray) -> str:
    return (
        f"t={t:5d} x={ram.player_x(r):5d} y={ram.u16(r, ram.PLAYER_Y):4d} "
        f"anim={r[ram.PLAYER_ANIMATION]:3d} end_timer={r[ram.END_LEVEL_TIMER]:3d} "
        f"mode={r[ram.GAME_MODE]:3d} lives={r[ram.LIVES]}"
    )


def save(frames: Path | None, name: str, obs: np.ndarray) -> None:
    if frames:
        frames.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frames / f"{name}.png"), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))


def death(env, frames: Path | None) -> bool:
    env.reset()
    for t in range(1, 1200):
        obs, *_ = env.step(press(env, "RIGHT"))
        r = env.get_ram()
        if t % 20 == 0 or ram.is_dying(r):
            print(status(t, r))
        if ram.is_dying(r):
            save(frames, "death", obs)
            return True
    return False


def goal(env, frames: Path | None) -> bool:
    """Warp right 128 px every 2 s, with star power so enemies can't kill Mario.

    Writing RAM like this is only for testing the goal signal; training never
    touches RAM.
    """
    env.reset()
    mem = env.unwrapped.data.memory
    for t in range(1, 8000):
        mem.assign(WRAM + ram.STAR_TIMER, "|u1", 0xFF)
        if t % 120 == 0:
            mem.assign(WRAM + ram.PLAYER_X, "<u2", ram.player_x(env.get_ram()) + 128)
            mem.assign(WRAM + ram.PLAYER_Y, "<u2", 0x60)  # drop in from above, clear of walls
        obs, *_ = env.step(press(env, "RIGHT", "Y"))
        r = env.get_ram()
        if t % 300 == 0 or ram.level_cleared(r) or ram.is_dying(r):
            print(status(t, r))
        if ram.level_cleared(r):
            save(frames, "goal", obs)
            return True
        if ram.is_dying(r):
            return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("event", choices=["death", "goal"])
    parser.add_argument("--state", default="YoshiIsland1")
    parser.add_argument("--frames", type=Path)
    args = parser.parse_args()

    env = retro.make(GAME, state=args.state, render_mode="rgb_array")
    ok = {"death": death, "goal": goal}[args.event](env, args.frames)
    env.close()
    print(f"{args.event}: {'detected' if ok else 'NOT detected'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
