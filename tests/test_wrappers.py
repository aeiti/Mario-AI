import gymnasium as gym
import numpy as np
import pytest

from mario_ai.config import RewardConfig
from mario_ai.wrappers import GameState, MaxAndSkip, ProgressReward, WarpFrame

# Round numbers so expected rewards can be written out exactly.
CFG = RewardConfig(
    progress_scale=0.5, death_penalty=10.0, clear_reward=100.0, time_penalty=1.0, stall_steps=3
)


def state(x, dying=False, cleared=False):
    return GameState(x=x, dying=dying, cleared=cleared)


class ScriptedEnv(gym.Env):
    """Plays back a fixed list of GameStates, one per step, starting after reset.

    It also returns an env reward of 1000 per step, which ProgressReward must
    ignore, and never ends an episode by itself.
    """

    observation_space = gym.spaces.Box(0, 255, (1,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, start, script):
        self.start = start
        self.script = script

    def reset(self, *, seed=None, options=None):
        self.state = self.start
        self._upcoming = iter(self.script)
        return np.zeros(1, dtype=np.uint8), {"from_env": True}

    def step(self, action):
        self.state = next(self._upcoming)
        return np.zeros(1, dtype=np.uint8), 1000.0, False, False, {"from_env": True}


def make(script, start=state(100), config=CFG):
    env = ProgressReward(ScriptedEnv(start, script), read_state=lambda e: e.state, config=config)
    env.reset()
    return env


def run(env, n):
    return [env.step(0) for _ in range(n)]


# --- ProgressReward ---------------------------------------------------------


def test_reward_is_scaled_new_progress_minus_time_penalty():
    env = make([state(110)])
    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == pytest.approx(0.5 * 10 - 1.0)  # 4.0
    assert (terminated, truncated) == (False, False)


def test_spawn_position_is_not_progress():
    # Reset sets max_x to the spawn x (100), so staying there earns nothing.
    env = make([state(100)])
    assert env.step(0)[1] == pytest.approx(-1.0)


def test_only_new_max_x_is_rewarded():
    # Forward to 120, back to 90, forward again to the old max, then past it.
    env = make([state(120), state(90), state(120), state(130)])
    rewards = [r for _, r, *_ in run(env, 4)]
    assert rewards == pytest.approx([0.5 * 20 - 1, -1.0, -1.0, 0.5 * 10 - 1])


def test_death_terminates_with_penalty():
    env = make([state(100, dying=True)])
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == pytest.approx(-10.0 - 1.0)
    assert terminated is True and truncated is False
    assert info["died"] is True and info["cleared"] is False


def test_clear_terminates_with_bonus():
    env = make([state(104, cleared=True)])
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == pytest.approx(0.5 * 4 + 100.0 - 1.0)
    assert terminated is True and truncated is False
    assert info["cleared"] is True and info["died"] is False


def test_stall_truncates_at_exactly_stall_steps():
    env = make([state(100)] * 3)
    truncated = [t for *_, t, _ in run(env, 3)]
    assert truncated == [False, False, True]


def test_new_progress_resets_the_stall_counter():
    # Two idle steps, progress, then it takes three more idle steps to stall.
    env = make([state(100), state(100), state(101), state(101), state(101), state(101)])
    results = run(env, 6)
    assert [t for *_, t, _ in results] == [False, False, False, False, False, True]
    assert results[-1][4]["stalled"] is True


def test_info_carries_episode_stats_and_keeps_env_info():
    env = make([state(120), state(110), state(110, dying=True)])
    *_, info = run(env, 3)[-1]
    assert info["from_env"] is True
    assert info["max_x"] == 120
    assert info["died"] is True
    assert info["stalled"] is False
    assert info["reward_parts"] == pytest.approx(
        {"progress": 10.0, "time": -3.0, "death": -10.0, "clear": 0.0}
    )


def test_reset_clears_the_stats():
    env = ProgressReward(
        ScriptedEnv(state(100), [state(150), state(150)]), read_state=lambda e: e.state, config=CFG
    )
    env.reset()
    run(env, 2)
    assert env.max_x == 150 and env.steps_since_progress == 1

    env.reset()
    assert env.max_x == 100
    assert env.steps_since_progress == 0
    assert env.totals == {"progress": 0.0, "time": 0.0, "death": 0.0, "clear": 0.0}
    # And the next episode is rewarded from the spawn again, not from 150.
    _, reward, *_ , info = env.step(0)
    assert reward == pytest.approx(0.5 * 50 - 1)
    assert info["reward_parts"]["progress"] == pytest.approx(25.0)


# --- MaxAndSkip and WarpFrame -----------------------------------------------


class FrameCounterEnv(gym.Env):
    """Returns frame number n as an image filled with value n, reward 1 per frame."""

    observation_space = gym.spaces.Box(0, 255, (2, 2, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(3)

    def __init__(self, end_on_frame=None):
        self.end_on_frame = end_on_frame

    def reset(self, *, seed=None, options=None):
        self.frame = 0
        self.actions = []
        return np.zeros((2, 2, 3), dtype=np.uint8), {}

    def step(self, action):
        self.frame += 1
        self.actions.append(action)
        obs = np.full((2, 2, 3), self.frame * 10, dtype=np.uint8)
        obs[0, 0] = 255 if self.frame == 3 else obs[0, 0]  # a "flickering" pixel
        terminated = self.frame == self.end_on_frame
        return obs, 1.0, terminated, False, {}


def test_max_and_skip_repeats_the_action_and_sums_rewards():
    inner = FrameCounterEnv()
    env = MaxAndSkip(inner, skip=4)
    env.reset()
    obs, reward, terminated, _, _ = env.step(2)
    assert inner.actions == [2, 2, 2, 2]
    assert reward == 4.0
    assert terminated is False
    # Pixel-wise max of frames 3 and 4: the flicker pixel from frame 3 survives.
    assert obs[0, 0, 0] == 255
    assert obs[1, 1, 0] == 40


def test_max_and_skip_stops_early_when_the_episode_ends():
    inner = FrameCounterEnv(end_on_frame=2)
    env = MaxAndSkip(inner, skip=4)
    env.reset()
    _, reward, terminated, _, _ = env.step(1)
    assert inner.actions == [1, 1]
    assert reward == 2.0
    assert terminated is True


def test_warp_frame_makes_84x84_grayscale():
    env = WarpFrame(FrameCounterEnv())
    env.reset()
    obs, *_ = env.step(0)
    assert env.observation_space.shape == (84, 84, 1)
    assert obs.shape == (84, 84, 1)


# --- RandomNoopStart --------------------------------------------------------


class CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (1,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.actions = []
        return np.zeros(1, dtype=np.uint8), {}

    def step(self, action):
        self.actions.append(action)
        return np.zeros(1, dtype=np.uint8), 0.0, False, False, {}


def test_random_noop_start_presses_only_noop_and_varies_with_seed():
    from mario_ai.wrappers import RandomNoopStart

    env = RandomNoopStart(CountingEnv(), max_noops=30, noop_action=0)
    counts = set()
    for seed in range(20):
        env.reset(seed=seed)
        assert all(a == 0 for a in env.unwrapped.actions)
        assert len(env.unwrapped.actions) <= 30
        counts.add(len(env.unwrapped.actions))
    assert len(counts) > 5  # different seeds give different start delays

    env.reset(seed=3)
    first = len(env.unwrapped.actions)
    env.reset(seed=3)
    assert len(env.unwrapped.actions) == first  # same seed, same delay
