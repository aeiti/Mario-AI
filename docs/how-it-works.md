# How it works

This is a walk through the whole system: what the agent sees, what it can do, how it's rewarded, and how it learns. Every file mentioned is short and commented.

## The loop

Reinforcement learning is trial and error with a score:

1. The agent looks at the screen (the **observation**).
2. It picks a button combo (the **action**).
3. The game advances, and the agent gets a number saying how good that was (the **reward**).
4. Repeat until Mario dies, clears the level, or gets stuck. That's one **episode**, or attempt.

Nobody tells the agent how to play. It starts out pressing buttons at random and gradually does more of whatever earned reward. **PPO** (Proximal Policy Optimization) is the algorithm that turns "that went well" into small, safe adjustments to a neural network.

## What the agent sees

The emulator produces 256×224 colour frames, 60 per second. `env.py` stacks wrappers on the raw game, each one changing a single thing:

- **Frame skip** (`MaxAndSkip`). The agent makes a decision every 4 frames, and the action is held in between. That gives 15 decisions per second, which is plenty for platforming and makes learning 4× cheaper. It also takes the pixel-wise max of the last two frames, because the SNES sometimes draws a sprite only on alternate frames.
- **Grayscale, 84×84** (`WarpFrame`). Colour adds little, and a small image keeps the network fast.
- **Frame stacking** (`VecFrameStack`, 4 frames). A single still can't show which way an enemy is moving. Four in a row can.

The network is SB3's `CnnPolicy`, the "Nature DQN" convolutional net: three conv layers, then two output heads. The **policy head** gives a probability for each action. The **value head** guesses how much reward is still to come from this moment.

## What the agent can do

The SNES pad has 12 buttons, so there are 4,096 combinations, nearly all of them useless. `actions.py` offers 10 a human would actually press: noop, right, right+run, right+jump, right+run+jump, right+spin, left, left+jump, down, and jump. A small menu is the single biggest speed-up for learning. In SMW, B is jump, A is spin jump (it bounces off most enemies), and Y is run.

## The reward

The game's own score is a poor teacher (coins don't get you to the goal), so `wrappers.ProgressReward` replaces it. Each step:

- **+0.1 per pixel of new ground.** Only progress past the furthest point reached *this attempt* counts. Paying for plain rightward movement lets an agent farm reward by walking back and forth over the same spot; other projects hit exactly that.
- **−0.01 per step,** so dawdling costs a little.
- **−25 and the episode ends** when Mario starts dying.
- **+50 and the episode ends** when he breaks the goal tape.
- **The episode ends** if he makes no new progress for 500 steps (about 33 seconds).

Yoshi's Island 1 is about 4,825 px long, so a full clear earns roughly 480 + 50.

### Where those signals come from: RAM

The agent only ever sees pixels. The *reward*, though, reads the game's memory, because that's the reliable way to know Mario's position. `ram.py` names the addresses, and each one was confirmed by making the event happen in the emulator (`scripts/explore.py`):

| Signal | Address | Meaning |
|---|---|---|
| X position | `$94–$95` | 16-bit pixels from the level start (16 at spawn, ~4825 at the goal) |
| Dying | `$71 == 9` | the death animation has started. It's set the same frame for enemies, pits, lava and time-up. |
| Goal tape | `$1493 != 0` | the end-level timer jumps to 255 on the exact frame Mario touches the tape |

We end the episode on `$71`, not when the lives counter drops. Lives only change about 3 seconds later, after the death animation, and the agent would learn nothing from those 48 extra uncontrollable steps.

## Why a random start

The emulator is perfectly deterministic: the same buttons produce the same game, frame for frame. Left alone, an agent could memorise one exact button sequence, and every evaluation would be an identical replay. `RandomNoopStart` waits 0–30 steps before the agent takes control. That shifts every enemy relative to Mario, so the agent has to react to what it sees.

## How PPO learns

Training runs 16 copies of the game in parallel, one per CPU process (stable-retro allows only one emulator per process). The cycle:

1. **Collect.** Each copy plays 128 steps with the current network, sampling actions from its probabilities. That's 2,048 steps per round, taking about 1.6 s.
2. **Score.** For each step, work out the **advantage**: how much better things went than the value head expected. This uses GAE with γ = 0.99, which counts rewards up to roughly 100 steps (about 7 seconds) into the future.
3. **Update.** On the GPU, about 0.3 s. Nudge the network to make high-advantage actions more likely and low-advantage ones less likely. PPO's *clip* (0.1) caps how far the policy can move in one update, which keeps it from wrecking itself on one lucky or unlucky batch. An *entropy bonus* (0.01) keeps it exploring rather than committing too early.
4. Repeat for millions of steps.

The hyperparameters in `config.py` are the standard SB3 settings for Atari games, a good starting point for pixel-based platformers. Another project reached a 95–100% clear rate on this level after about 5M steps with similar settings, so that's a reasonable budget expectation.

## Watching it learn

- The **live window** (`--watch`) shows training env 0. Its actions are *sampled*, so it explores and looks messier than the agent really is. The emulator records one complete attempt at a time into shared memory, and a separate viewer process plays it back at real speed, so the window keeps moving even while training pauses to update the network. Training runs faster than real time, so the attempts in between are skipped.
- **Eval videos** (made by background worker processes, so training doesn't stop) and `watch.py` use the *deterministic* policy (always the most likely action): what the agent knows, minus the exploration.
- The **attempt chart** (`progress.py`) is the clearest picture of learning. In the lower panel, dark bands mark where attempts end. Early on they pile up at the first Rex. As the agent learns, the band jumps right, obstacle by obstacle, until the attempts reach the goal tape.

## Glossary

- **Episode / attempt:** one life, from the start until death, clear, or stall.
- **Timestep / step:** one agent decision, which is 4 frames.
- **Policy:** the network's mapping from screen to action probabilities.
- **Deterministic vs stochastic:** always picking the top action, versus sampling by probability.
- **Checkpoint:** a saved network you can watch or resume from.
