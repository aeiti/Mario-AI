import gymnasium as gym
import numpy as np
import pytest

from mario_ai.actions import ACTIONS, Discretizer

SNES_BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]


class ButtonEnv(gym.Env):
    """Stands in for a retro env: exposes `buttons` and records the last action."""

    def __init__(self, buttons):
        self.buttons = buttons
        self.action_space = gym.spaces.MultiBinary(len(buttons))
        self.observation_space = gym.spaces.Box(0, 255, (1,), dtype=np.uint8)
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.uint8), {}

    def step(self, action):
        self.last_action = action
        return np.zeros(1, dtype=np.uint8), 0.0, False, False, {}


def press(action_name, buttons=SNES_BUTTONS):
    env = ButtonEnv(buttons)
    wrapped = Discretizer(env)
    wrapped.step(wrapped.action_names.index(action_name))
    return env.last_action


def test_action_space_is_one_index_per_combo():
    assert Discretizer(ButtonEnv(SNES_BUTTONS)).action_space == gym.spaces.Discrete(10)
    assert len(ACTIONS) == 10


@pytest.mark.parametrize(
    "name, expected",
    [
        #                 B  Y  SE ST U  D  L  R  A  X  L  R
        ("noop",           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("right",          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ("right+run",      [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ("right+jump",     [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ("right+run+jump", [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ("right+spin",     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]),
        ("left",           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        ("left+jump",      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        ("down",           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        ("jump",           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_each_combo_presses_exactly_its_buttons(name, expected):
    np.testing.assert_array_equal(press(name), expected)


def test_button_order_comes_from_the_env_not_a_hardcoded_list():
    # Same buttons, reversed order: "right+jump" must follow the env's layout.
    reversed_buttons = list(reversed(SNES_BUTTONS))
    pressed = press("right+jump", reversed_buttons)
    held = {b for b, p in zip(reversed_buttons, pressed) if p}
    assert held == {"RIGHT", "B"}
    assert pressed[reversed_buttons.index("B")] == 1
    assert pressed[SNES_BUTTONS.index("B")] == 0  # index 0 in SNES order is "R" here


def test_mutating_a_returned_action_does_not_change_the_table():
    wrapped = Discretizer(ButtonEnv(SNES_BUTTONS))
    wrapped.action(0)[:] = 1
    np.testing.assert_array_equal(wrapped.action(0), np.zeros(12))
