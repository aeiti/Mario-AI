# Mario-AI

A reinforcement-learning agent that teaches itself to play **Super Mario World** (SNES) from the pixels on screen. The first milestone is clearing **Yoshi's Island 1** reliably.

It uses [stable-retro](https://github.com/Farama-Foundation/stable-retro) to run the game and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) PPO to learn. [docs/how-it-works.md](docs/how-it-works.md) explains every piece in plain English.

## Setup

You need [uv](https://docs.astral.sh/uv/) and your own copy of the US ROM, `Super Mario World (U) [!].smc`. The ROM is never committed.

```bash
uv sync
uv run scripts/import_rom.py "/path/to/Super Mario World (U) [!].smc"
uv run pytest
```

`import_rom.py` strips the 512-byte copier header if there is one, checks the SHA-1, and installs the ROM into stable-retro.

> **Apple Silicon note:** stable-retro 1.0.1 on PyPI builds its SNES core with the wrong byte order on ARM64, so the game crashes to a black screen. `pyproject.toml` therefore pins stable-retro to a commit on `main` that includes the fix (Farama-Foundation/stable-retro#281). The first `uv sync` compiles it from source, which takes about 6 minutes. Switch back to the PyPI release once a version after 1.0.1 is out.

## Train and watch it learn

```bash
uv run scripts/train.py --run yi1 --watch       # trains; --watch opens a live window on one emulator
uv run scripts/progress.py yi1 --live           # live chart of every attempt (separate terminal)
uv run tensorboard --logdir runs                # dashboards at http://localhost:6006
```

There are four ways to watch it improve:

| What | Where | Shows |
|---|---|---|
| Live window | `--watch` | One of the 16 training emulators, played back smoothly by its own process so it never freezes. The agent is exploring, so expect flailing early on. |
| Attempt chart | `scripts/progress.py` | How far every single attempt got, and *where* attempts end. The dark band in the lower panel is the obstacle it's stuck on; watch it move right. |
| Eval videos | `videos/<run>/step_*.mp4` | Every 200k steps, the agent's best behaviour (no exploration) from the same starting point. |
| Progress reel | `scripts/reel.py <run>` | All the eval videos stitched together, each captioned with its training step. |

### Controls

The live window has buttons under the game, and matching keys:

| Button | Key | Does |
|---|---|---|
| Pause / Resume | **P** or Space | Freezes training (all emulators wait) until pressed again |
| Save | **S** | Saves `checkpoints/<run>/manual_<steps>.zip` right away, even while paused |
| Stop | **Q**, twice within 3 s | Ends training cleanly and saves `final.zip`. The second press is a guard against stray clicks. |

Without `--watch`, use **Ctrl-C** in the terminal to stop (it also saves `final.zip`). Resume any saved model with:

```bash
uv run scripts/train.py --run yi1 --watch --resume checkpoints/yi1/final.zip
```

Resuming continues the step count, the attempt log, the TensorBoard graphs and the videos, and remembers the best evaluation so far, so `best.zip` is only replaced by a better one.

TensorBoard's `episode/` and `eval/` sections show distance, clear rate, death rate and each reward component. `time/` shows how long each round spends playing vs. learning.

### Where the work happens

Training keeps the whole machine busy, in separate processes:

| Process | Does |
|---|---|
| 16 emulator workers | play the game in parallel, one per CPU core (stable-retro allows one emulator per process) |
| trainer (main) | picks actions for all 16 at once, and updates the network on the **Apple GPU (MPS)**, which is about 17× faster than the CPU here |
| 5 evaluation workers | test a snapshot of the network in the background every 200k steps, one episode each, so training never waits |
| viewer | draws the live window from shared memory at 60 fps |

That's about 1,000 steps/s on an M4 Max (5M steps in under 1½ hours), up from about 210 with 12 CPU-only workers and evaluation inline.

## Use a trained agent

```bash
uv run scripts/watch.py checkpoints/yi1/best.zip            # watch it play at real speed
uv run scripts/evaluate.py checkpoints/yi1/best.zip         # clear rate over 20 episodes
```

## Tools

```bash
uv run scripts/explore.py death     # check the RAM signals the reward relies on
uv run scripts/explore.py goal
uv run python -m stable_retro.examples.interactive --game SuperMarioWorld-Snes-v0 --state YoshiIsland1   # play it yourself
```

## Layout

```
src/mario_ai/
  env.py         make_env(): the emulator plus the wrapper stack
  wrappers.py    frame skip, random start, the progress reward
  actions.py     the 10 button combos the agent chooses from
  ram.py         RAM addresses for position, death and the goal tape (each one verified)
  config.py      every tunable number
  evaluation.py  vectorised training envs; play one episode
  callbacks.py   TensorBoard stats, the attempt log, eval videos
  viewer.py      the live training window (its own process, fed through shared memory)
scripts/         import_rom, train, watch, evaluate, progress, reel, explore
tests/           wrapper unit tests + real-emulator smoke tests
```
