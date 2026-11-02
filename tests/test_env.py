"""Smoke test against the real emulator. Skipped until the ROM is imported."""

import numpy as np
import pytest
import stable_retro as retro

from mario_ai.config import TrainConfig

CONFIG = TrainConfig()

try:
    retro.data.get_romfile_path(CONFIG.game)
    HAVE_ROM = True
except FileNotFoundError:
    HAVE_ROM = False

pytestmark = pytest.mark.skipif(not HAVE_ROM, reason="run scripts/import_rom.py first")


def test_env_steps_and_reports_progress():
    from mario_ai.env import make_env

    env = make_env(CONFIG)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 1) and obs.dtype == np.uint8
    assert env.action_space.n == 10

    # Hold right+run+jump (action 4) for 2 seconds: bouncing clears the first
    # Rex, so Mario must move well right without the episode ending, and the
    # reward must be positive.
    total = 0.0
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(4)
        total += reward
        assert not (terminated or truncated)
    assert info["max_x"] > 150
    assert total > 0
    env.close()


def test_walking_into_the_first_rex_ends_the_episode_with_the_death_penalty():
    from mario_ai.env import make_env

    env = make_env(CONFIG)
    env.reset(seed=0)
    for _ in range(300):  # "right" (action 1) walks into the first Rex within a few seconds
        _, reward, terminated, truncated, info = env.step(1)
        if terminated or truncated:
            break
    assert terminated and info["died"]
    assert info["reward_parts"]["death"] == -CONFIG.reward.death_penalty
    env.close()
