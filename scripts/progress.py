"""Chart how the agent improves, attempt by attempt.

    uv run scripts/progress.py yi1           # write checkpoints/yi1/progress.png
    uv run scripts/progress.py yi1 --live    # a window that refreshes while training runs

Top panel: how far every attempt got (one dot each), with a rolling average.
Bottom panel: where attempts *end* along the level, over the course of
training. Each row is a slice of attempts; dark cells are where most of them
ended. The dark band is the obstacle the agent is currently stuck on, and you
can watch it move right as the agent learns to get past it.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
GOAL_X = 4825  # measured with scripts/explore.py goal

# Validated palette (dataviz reference instance, light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
OUTCOMES = {  # fixed order: the colour follows the outcome, never its count
    "died": ("#2a78d6", "o", 6),
    "stalled": ("#eb6834", "o", 6),
    "cleared": ("#1baf7a", "*", 40),  # star + size: not colour alone
}
BLUES = LinearSegmentedColormap.from_list("blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {
        "attempt": np.array([int(r["attempt"]) for r in rows]),
        "timesteps": np.array([int(r["timesteps"]) for r in rows]),
        "max_x": np.array([int(r["max_x"]) for r in rows]),
        "outcome": np.array([r["outcome"] for r in rows]),
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        window = max(len(values), 1)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def draw(fig, data: dict[str, np.ndarray], run: str) -> None:
    fig.clear()
    top, bottom = fig.subplots(2, 1, height_ratios=[1, 1.1])
    n = len(data["attempt"])

    # --- How far each attempt got ---------------------------------------
    for outcome, (color, marker, size) in OUTCOMES.items():
        mask = data["outcome"] == outcome
        count = int(mask.sum())
        top.scatter(
            data["attempt"][mask], data["max_x"][mask],
            s=size, c=color, marker=marker, alpha=0.35 if outcome != "cleared" else 0.9,
            linewidths=0, label=f"{outcome} ({count:,})", zorder=2,
        )
    window = max(n // 50, 20)
    avg = rolling_mean(data["max_x"].astype(float), window)
    if len(avg):
        top.plot(data["attempt"][len(data["attempt"]) - len(avg):], avg, color=INK, lw=2, label=f"average of last {window}", zorder=3)
    top.axhline(GOAL_X, color=INK_2, lw=1, ls="--", zorder=1)
    top.text(1, GOAL_X, " goal tape", color=INK_2, va="bottom", fontsize=9, transform=top.get_yaxis_transform())
    top.set_ylim(0, GOAL_X * 1.22)  # headroom above the goal line for the legend
    top.set_xlim(0, max(n, 1) * 1.01)
    top.set_xlabel("attempt", color=INK_2)
    top.set_ylabel("furthest x reached (px)", color=INK_2)
    top.set_title("How far each attempt got", loc="left", color=INK, fontsize=12)
    legend = top.legend(loc="upper left", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    # --- Where attempts end, over training ---------------------------------
    bin_px = 64
    x_bins = np.arange(0, GOAL_X + bin_px, bin_px)
    n_rows = int(np.clip(n // 100, 1, 40))
    slices = np.array_split(np.arange(n), n_rows)
    grid = np.zeros((n_rows, len(x_bins) - 1))
    for row, idx in enumerate(slices):
        if len(idx):
            counts, _ = np.histogram(np.clip(data["max_x"][idx], 0, GOAL_X - 1), bins=x_bins)
            grid[row] = counts / len(idx)  # share of this slice's attempts
    bottom.imshow(
        grid, aspect="auto", origin="lower", cmap=BLUES, interpolation="nearest",
        extent=(0, x_bins[-1], 0, n), vmin=0, vmax=max(grid.max(), 0.05),
    )
    bottom.axvline(GOAL_X, color=INK_2, lw=1, ls="--")
    bottom.set_xlabel("position in the level (px)", color=INK_2)
    bottom.set_ylabel("attempt", color=INK_2)
    bottom.set_title("Where attempts end (darker = more attempts ended here)", loc="left", color=INK, fontsize=12)

    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    top.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    steps = int(data["timesteps"][-1]) if n else 0
    clears = int((data["outcome"] == "cleared").sum())
    fig.suptitle(f"{run}: {n:,} attempts, {steps:,} steps, best x {int(data['max_x'].max()) if n else 0:,}, {clears:,} clears",
                 x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--live", action="store_true", help="keep a window open and refresh it")
    parser.add_argument("--every", type=float, default=10.0, help="seconds between refreshes with --live")
    args = parser.parse_args()

    path = ROOT / "checkpoints" / args.run / "attempts.csv"
    fig = plt.figure(figsize=(11, 8), facecolor=SURFACE)
    if not args.live:
        draw(fig, load(path), args.run)
        out = path.with_name("progress.png")
        fig.savefig(out, dpi=120, facecolor=SURFACE)
        print(out)
        return

    plt.ion()
    while plt.fignum_exists(fig.number):
        data = load(path) if path.exists() else None
        if data is not None and len(data["attempt"]):
            draw(fig, data, args.run)
        plt.pause(args.every)


if __name__ == "__main__":
    main()
