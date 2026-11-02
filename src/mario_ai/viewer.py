"""A live window onto one training emulator.

During training, PPO samples actions at random in proportion to how much it
likes them, so this shows the agent *exploring*: early on Mario flails, and
over time the runs get longer and more purposeful. (The eval videos show the
deterministic policy instead, i.e. what it would do if it stopped exploring.)
"""

import cv2
import gymnasium as gym
import numpy as np

WINDOW = "Mario-AI: live training (env 0)"


class LiveViewer(gym.Wrapper):
    """Shows each agent step's frame, with episode stats drawn on top.

    Sits above ProgressReward so it can read max_x and how episodes ended,
    and below WarpFrame so it still gets the full-colour frame. The window is
    opened lazily, inside whichever process runs this env.
    """

    def __init__(self, env: gym.Env, scale: int = 3):
        super().__init__(env)
        self.scale = scale
        self.episode = 0
        self.best_x = 0
        self.clears = 0
        self.last_result = ""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.best_x = max(self.best_x, info["max_x"])
        if terminated or truncated:
            self.clears += info["cleared"]
            how = "CLEARED" if info["cleared"] else "died" if info["died"] else "stalled" if info["stalled"] else "ended"
            self.last_result = f"last: {how} at x={info['max_x']}"
        self.show(obs, info)
        return obs, reward, terminated, truncated, info

    def show(self, frame: np.ndarray, info: dict) -> None:
        img = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lines = [
            f"episode {self.episode}   x {info['max_x']}   best {self.best_x} / ~4825",
            f"clears {self.clears}   {self.last_result}",
        ]
        for i, text in enumerate(lines):
            y = img.shape[0] - 40 + 26 * i
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, img)
        # pollKey lets the window repaint without blocking. waitKey(1) would wait
        # ~11 ms on macOS, and since all envs step in lockstep, that slows every one.
        cv2.pollKey()
