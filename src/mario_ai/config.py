"""Every tunable number in one place.

Two dataclasses: what the agent is rewarded for (RewardConfig) and how PPO
trains (TrainConfig). Keeping them here means an experiment is "change a
config", not "hunt through the code for a magic number".
"""

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """Shapes the reward in ``wrappers.ProgressReward``.

    Scale reasoning: at full run Mario moves about 3 px per frame, so about
    12 px per agent step (4 frames per step). A per-step reward of order 1 is
    a comfortable range for PPO's value function, which suggests 0.1 per px.
    Yoshi's Island 1 is roughly 4,800 px long, so finishing earns about 480
    from progress alone.
    """

    # Reward per pixel of *new* rightmost progress. See the class docstring.
    progress_scale: float = 0.1

    # Charged on the step Mario starts dying. 25 equals about 250 px of
    # progress: big enough that dying is clearly bad, but small next to the
    # level, so the agent is not scared into standing still at the start.
    death_penalty: float = 25.0

    # Paid once on clearing the level. Progress already rewards getting close;
    # this extra lump tells the agent that actually *finishing* matters, about
    # 10% on top of the progress reward for a full run.
    clear_reward: float = 50.0

    # Charged every agent step, so that dawdling costs something. Kept tiny on
    # purpose: a stalled episode costs at most time_penalty * stall_steps = 5,
    # far below death_penalty, so suicide is never a way to "save time".
    time_penalty: float = 0.01

    # End the episode (truncated) after this many agent steps without a new
    # max x. 500 steps is about 33 seconds of game time at 15 steps/s: long
    # enough to back up and retry a jump, short enough that a policy stuck
    # against a wall does not burn the rollout buffer.
    stall_steps: int = 500


@dataclass
class TrainConfig:
    """PPO settings, starting from the standard SB3 Atari recipe.

    These are the rl-zoo hyperparameters for Atari PPO, which work well on
    84x84 grayscale frame-stacked pixel games. Change them one at a time.
    """

    game: str = "SuperMarioWorld-Snes-v0"
    state: str = "YoshiIsland1"

    # Environment pipeline
    n_envs: int = 8
    frame_skip: int = 4
    frame_stack: int = 4
    # Up to this many agent steps of standing still after reset; see RandomNoopStart.
    max_noops: int = 30

    # PPO. One rollout is n_envs * n_steps = 1024 transitions, split into
    # 4 minibatches of 256 and replayed for n_epochs passes.
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 2.5e-4  # decayed linearly to 0, see linear_schedule
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    total_timesteps: int = 10_000_000

    # Callbacks. SB3's CheckpointCallback and EvalCallback count calls to the
    # vectorised env, not transitions, so divide by n_envs when passing these:
    # e.g. save_freq=max(checkpoint_freq // n_envs, 1).
    checkpoint_freq: int = 500_000  # transitions between saved models
    eval_freq: int = 100_000  # transitions between evaluation runs (each makes one video)
    n_eval_episodes: int = 5

    reward: RewardConfig = field(default_factory=RewardConfig)


def linear_schedule(initial_value: float):
    """A learning rate that falls linearly from initial_value to 0.

    SB3 calls the schedule with ``progress_remaining``, which goes from 1 at
    the start of training to 0 at the end. Decaying the step size lets early
    training move fast and late training settle instead of oscillating.
    """

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule
