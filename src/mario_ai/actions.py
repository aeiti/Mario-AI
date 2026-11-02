"""A small, hand-picked action set for Super Mario World.

The SNES pad has 12 buttons, so the raw action space is 2**12 = 4096 button
combinations. Almost all of them are useless (SELECT + L + DOWN + ...), and a
policy that has to discover that by trial and error learns very slowly. So we
give PPO a short menu of combos a human would actually press, and translate
the chosen menu index back into buttons.

SMW controls, for reference: B = jump, A = spin jump, Y = run (hold it while
moving to speed up).
"""

import gymnasium as gym
import numpy as np

# Each entry is (name, buttons held). The name is only for humans: logs,
# debugging, and watching a replay.
ACTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("noop", ()),
    ("right", ("RIGHT",)),
    ("right+run", ("RIGHT", "Y")),
    ("right+jump", ("RIGHT", "B")),
    ("right+run+jump", ("RIGHT", "Y", "B")),
    ("right+spin", ("RIGHT", "A")),
    ("left", ("LEFT",)),
    ("left+jump", ("LEFT", "B")),
    ("down", ("DOWN",)),
    ("jump", ("B",)),
]


class Discretizer(gym.ActionWrapper):
    """Lets the agent pick a combo by index instead of pressing 12 buttons.

    The button order is read from the emulator (``env.unwrapped.buttons``)
    rather than hard-coded, so a different core or button layout cannot
    silently map "jump" onto the wrong button.
    """

    def __init__(self, env: gym.Env, actions=ACTIONS):
        super().__init__(env)
        buttons = list(env.unwrapped.buttons)
        self.action_names = [name for name, _ in actions]

        # Precompute one boolean button array per combo, so step() is a lookup.
        self._button_arrays = []
        for _, combo in actions:
            pressed = np.zeros(len(buttons), dtype=np.int8)
            for button in combo:
                pressed[buttons.index(button)] = 1  # ValueError on a typo: good
            self._button_arrays.append(pressed)

        self.action_space = gym.spaces.Discrete(len(actions))

    def action(self, action):
        # Copy so a caller that mutates the array cannot corrupt our table.
        return self._button_arrays[int(action)].copy()
