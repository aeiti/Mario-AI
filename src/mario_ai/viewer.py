"""A live window onto one training emulator, drawn by its own process.

During training, PPO samples actions at random in proportion to how much it
likes them, so this shows the agent *exploring*: early on Mario flails, and
over time the runs get longer and more purposeful. (The eval videos show the
deterministic policy instead, i.e. what it would do if it stopped exploring.)

Why a separate process: training stops all the emulators every few seconds
while PPO updates the network, and for longer during evaluations. A window
drawn by an emulator process would freeze then, and macOS would mark it as
not responding. So the emulator only copies frames into a shared-memory ring
buffer (fast, never blocks), and a viewer process plays them back at a steady
60 fps, keeping the window responsive even while training is paused.

    train.py ──starts──> LiveView ──starts──> viewer process (the window)
                            │ shared memory ▲
    env 0 ── FrameFeed ─────┘ writes frames  └── reads at 60 fps
"""

import multiprocessing as mp
import time
from multiprocessing import shared_memory

import cv2
import gymnasium as gym
import numpy as np

WINDOW = "Mario-AI: live training (env 0)"
HEIGHT, WIDTH = 224, 256
SLOTS = 1200  # 20 s of frames at 60 fps
TEXT_BYTES = 120  # overlay text stored with each frame, so stats stay in sync
HEADER = 8  # one uint64: total frames written so far
FRAME_BYTES = HEIGHT * WIDTH * 3
SIZE = HEADER + SLOTS * (FRAME_BYTES + TEXT_BYTES)


def _views(buf):
    """numpy views of the shared buffer: write counter, frames, overlay texts."""
    count = np.ndarray((1,), dtype=np.uint64, buffer=buf, offset=0)
    frames = np.ndarray((SLOTS, HEIGHT, WIDTH, 3), dtype=np.uint8, buffer=buf, offset=HEADER)
    texts = np.ndarray((SLOTS, TEXT_BYTES), dtype=np.uint8, buffer=buf, offset=HEADER + SLOTS * FRAME_BYTES)
    return count, frames, texts


class LiveView:
    """Owns the shared memory and the viewer process. Used by train.py."""

    def __init__(self):
        self.shm = shared_memory.SharedMemory(create=True, size=SIZE)
        _views(self.shm.buf)[0][0] = 0
        self.process = mp.get_context("spawn").Process(target=_viewer_main, args=(self.shm.name,), daemon=True)
        self.process.start()

    @property
    def name(self) -> str:
        return self.shm.name

    def close(self) -> None:
        self.process.terminate()
        self.process.join(timeout=2)
        self.shm.close()
        self.shm.unlink()


class FrameFeed:
    """The writing end, living in the emulator process. Never blocks."""

    def __init__(self, name: str):
        self.name = name
        self.text = ""
        self._shm = None

    def _open(self):
        # Child processes share the trainer's resource tracker, so attaching
        # here doesn't change who cleans up: LiveView.close() unlinks it.
        self._shm = shared_memory.SharedMemory(name=self.name)
        self.count, self.frames, self.texts = _views(self._shm.buf)

    def push(self, frame: np.ndarray) -> None:
        if self._shm is None:
            self._open()
        n = int(self.count[0])
        slot = n % SLOTS
        self.frames[slot] = frame
        encoded = self.text.encode()[:TEXT_BYTES]
        self.texts[slot] = 0
        self.texts[slot, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        self.count[0] = n + 1  # publish only after the slot is fully written


class FeedFrames(gym.Wrapper):
    """Sits directly on the emulator, below the frame skip, so all 60 fps reach the feed."""

    def __init__(self, env: gym.Env, feed: FrameFeed):
        super().__init__(env)
        self.feed = feed

    def step(self, action):
        obs, *rest = self.env.step(action)
        self.feed.push(obs)
        return (obs, *rest)


class FeedStats(gym.Wrapper):
    """Sits above ProgressReward and keeps the feed's overlay text current."""

    def __init__(self, env: gym.Env, feed: FrameFeed):
        super().__init__(env)
        self.feed = feed
        self.episode = 0
        self.best_x = 0
        self.clears = 0
        self.last = ""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.best_x = max(self.best_x, info["max_x"])
        if terminated or truncated:
            self.clears += info["cleared"]
            how = "CLEARED" if info["cleared"] else "died" if info["died"] else "stalled"
            self.last = f"last: {how} at x={info['max_x']}"
        self.feed.text = (
            f"attempt {self.episode}  x {info['max_x']}  best {self.best_x}/4825"
            f"|clears {self.clears}  {self.last}"
        )
        return obs, reward, terminated, truncated, info


def _draw_text(img: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        y = img.shape[0] - 14 - 26 * (len(lines) - 1 - i)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def _viewer_main(name: str, scale: int = 3) -> None:
    """The viewer process: play frames from the ring buffer at a steady pace.

    Emulator 0 produces frames in bursts (a few seconds of fast play, then a
    pause while PPO updates), slightly faster than 60 fps on average. So we
    play from a buffer: normally at 60 fps, a little faster when the backlog
    grows, skipping ahead if it ever gets too far behind, and showing a
    "waiting" note while the buffer is empty.
    """
    shm = shared_memory.SharedMemory(name=name)
    count, frames, texts = _views(shm.buf)
    read = 0
    last = np.zeros((HEIGHT * scale, WIDTH * scale, 3), dtype=np.uint8)
    lines = ["waiting for the first frames..."]
    tick = 1 / 60
    next_tick = time.perf_counter()
    try:
        while True:
            written = int(count[0])
            backlog = written - read
            if backlog > SLOTS - 120:  # about to be overwritten: jump ahead
                read = written - 60
                backlog = 60
            if backlog > 0:
                # 1 frame per tick normally; up to 3 when ~10 s behind.
                step = 1 + min(backlog // 300, 2)
                read += step
                slot = (read - 1) % SLOTS
                img = cv2.resize(frames[slot], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                last = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                text = bytes(texts[slot]).rstrip(b"\0").decode(errors="ignore")
                lines = text.split("|")
                shown = last.copy()
            else:
                shown = (last * 0.5).astype(np.uint8)  # dim the last frame while paused
                lines = lines[:2] + ["(training is updating the network...)"]
            _draw_text(shown, lines[:3])
            cv2.imshow(WINDOW, shown)
            cv2.pollKey()
            next_tick += tick
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    finally:
        shm.close()
