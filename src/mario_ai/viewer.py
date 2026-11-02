"""A live window onto one training emulator, with controls, drawn by its own process.

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

The same shared memory carries the controls. The window's buttons and keys
set flags the trainer checks every step, and the trainer writes back its
state and step count for the status bar:

    train.py ──starts──> LiveView ──starts──> viewer process (the window)
       ▲  pause/save/stop flags │ shared memory ▲      │
       └────────────────────────┤               │      │ clicks, keys
                                │ state, steps  └──────┘
    env 0 ── FrameFeed ─────────┘ frames
"""

import multiprocessing as mp
import time
from multiprocessing import shared_memory

import cv2
import gymnasium as gym
import numpy as np

WINDOW = "Mario-AI: live training (env 0)"
HEIGHT, WIDTH = 224, 256
SCALE = 3
SLOTS = 600  # 10 s of frames at 60 fps (~100 MB)
TEXT_BYTES = 120  # overlay text stored with each frame, so stats stay in sync
FRAME_BYTES = HEIGHT * WIDTH * 3

# Header: a few uint64 fields, then a status message from the trainer.
FIELDS = (
    "frames_written",  # emulator 0 -> viewer
    "paused",          # viewer -> trainer: 1 while the user wants a pause
    "save_requests",   # viewer -> trainer: incremented once per Save press
    "stop_requested",  # viewer -> trainer: 1 once the user confirms Stop
    "trainer_state",   # trainer -> viewer: RUNNING / PAUSED / STOPPING
    "timesteps",       # trainer -> viewer
)
RUNNING, PAUSED, STOPPING = 0, 1, 2
STATUS_BYTES = 128
HEADER = 8 * len(FIELDS) + STATUS_BYTES
SIZE = HEADER + SLOTS * (FRAME_BYTES + TEXT_BYTES)


class Shared:
    """Named access to the shared-memory layout, from whichever process."""

    def __init__(self, buf):
        self._fields = np.ndarray((len(FIELDS),), dtype=np.uint64, buffer=buf, offset=0)
        self._status = np.ndarray((STATUS_BYTES,), dtype=np.uint8, buffer=buf, offset=8 * len(FIELDS))
        self.frames = np.ndarray((SLOTS, HEIGHT, WIDTH, 3), dtype=np.uint8, buffer=buf, offset=HEADER)
        self.texts = np.ndarray((SLOTS, TEXT_BYTES), dtype=np.uint8, buffer=buf, offset=HEADER + SLOTS * FRAME_BYTES)

    def __getitem__(self, field: str) -> int:
        return int(self._fields[FIELDS.index(field)])

    def __setitem__(self, field: str, value: int) -> None:
        self._fields[FIELDS.index(field)] = value

    @property
    def status(self) -> str:
        return bytes(self._status).rstrip(b"\0").decode(errors="ignore")

    @status.setter
    def status(self, text: str) -> None:
        encoded = text.encode()[:STATUS_BYTES]
        self._status[:] = 0
        self._status[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)


class LiveView:
    """Owns the shared memory and the viewer process; the trainer's side of the controls."""

    def __init__(self):
        self.shm = shared_memory.SharedMemory(create=True, size=SIZE)
        self.shared = Shared(self.shm.buf)
        for field in FIELDS:
            self.shared[field] = 0
        self.shared.status = ""
        self._saves_handled = 0
        self.process = mp.get_context("spawn").Process(target=_viewer_main, args=(self.shm.name,), daemon=True)
        self.process.start()

    @property
    def name(self) -> str:
        return self.shm.name

    # Read by the trainer every step.
    def pause_requested(self) -> bool:
        return bool(self.shared["paused"])

    def stop_requested(self) -> bool:
        return bool(self.shared["stop_requested"])

    def take_save_request(self) -> bool:
        """True once per Save press."""
        if self.shared["save_requests"] > self._saves_handled:
            self._saves_handled = self.shared["save_requests"]
            return True
        return False

    # Written by the trainer, shown in the window's status bar.
    def report(self, timesteps: int, state: int = RUNNING, status: str | None = None) -> None:
        self.shared["timesteps"] = timesteps
        self.shared["trainer_state"] = state
        if status is not None:
            self.shared.status = status

    def close(self) -> None:
        self.process.terminate()
        self.process.join(timeout=2)
        del self.shared  # release the numpy views before closing the buffer
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
        self.shared = Shared(self._shm.buf)

    def push(self, frame: np.ndarray) -> None:
        if self._shm is None:
            self._open()
        n = self.shared["frames_written"]
        slot = n % SLOTS
        self.shared.frames[slot] = frame
        encoded = self.text.encode()[:TEXT_BYTES]
        self.shared.texts[slot] = 0
        self.shared.texts[slot, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        self.shared["frames_written"] = n + 1  # publish only after the slot is fully written


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


# --- The viewer process ------------------------------------------------------

BAR = 60  # height of the control bar under the game, in pixels
BUTTONS = {  # name: (x, width); all buttons share the bar's y range
    "pause": (10, 160),
    "save": (180, 130),
    "stop": (320, 150),
}
CONFIRM_SECONDS = 3


def button_at(x: int, y: int) -> str | None:
    """Which button a click at window position (x, y) hit, if any."""
    top = HEIGHT * SCALE + 10
    if not top <= y <= top + BAR - 20:
        return None
    for name, (bx, width) in BUTTONS.items():
        if bx <= x <= bx + width:
            return name
    return None


def _text(img, text, org, scale=0.7, color=(255, 255, 255)):
    # A drop shadow keeps the text readable on any background. (A thick black
    # outline would be simpler, but OpenCV 5 spaces thick text differently,
    # so the outline doesn't line up with the letters.)
    x, y = org
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compose(game: np.ndarray, overlay: list[str], labels: dict[str, str], hover: str | None, status: list[str]) -> np.ndarray:
    """The full window: game frame with overlay text, then the control bar."""
    h, w = game.shape[:2]
    img = np.zeros((h + BAR, w, 3), dtype=np.uint8)
    img[:h] = game
    for i, line in enumerate(overlay):
        _text(img, line, (10, h - 14 - 26 * (len(overlay) - 1 - i)))

    img[h:] = (25, 26, 26)
    top = h + 10
    for name, (x, width) in BUTTONS.items():
        fill = (80, 80, 76) if name == hover else (56, 56, 53)
        if name == "stop" and labels[name].startswith("Confirm"):
            fill = (40, 40, 160)  # red-ish while waiting for confirmation
        cv2.rectangle(img, (x, top), (x + width, top + BAR - 20), fill, -1, cv2.LINE_AA)
        cv2.putText(img, labels[name], (x + 12, top + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 239, 236), 1, cv2.LINE_AA)
    for i, line in enumerate(status[:2]):
        cv2.putText(img, line, (490, top + 14 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (195, 194, 183), 1, cv2.LINE_AA)
    return img


class Controls:
    """Turns clicks and keys into flags in shared memory."""

    def __init__(self, shared: Shared):
        self.shared = shared
        self.hover = None
        self.stop_armed_until = 0.0  # Stop needs a second press within a few seconds

    def press(self, name: str) -> None:
        if name == "pause":
            self.shared["paused"] = 0 if self.shared["paused"] else 1
        elif name == "save":
            self.shared["save_requests"] = self.shared["save_requests"] + 1
        elif name == "stop":
            if time.monotonic() < self.stop_armed_until:
                self.shared["stop_requested"] = 1
                self.shared["paused"] = 0  # let a paused trainer reach the stop
            else:
                self.stop_armed_until = time.monotonic() + CONFIRM_SECONDS

    def key(self, code: int) -> None:
        key = chr(code & 0xFF).lower() if code >= 0 else ""
        action = {"p": "pause", " ": "pause", "s": "save", "q": "stop"}.get(key)
        if action:
            self.press(action)

    def mouse(self, event, x, y, flags, param) -> None:
        self.hover = button_at(x, y)
        if event == cv2.EVENT_LBUTTONUP and self.hover:
            self.press(self.hover)

    def labels(self) -> dict[str, str]:
        if self.shared["stop_requested"]:
            stop = "Stopping..."
        elif time.monotonic() < self.stop_armed_until:
            stop = "Confirm: Q"
        else:
            stop = "[Q] Stop"
        return {
            "pause": "[P] Resume" if self.shared["paused"] else "[P] Pause",
            "save": "[S] Save",
            "stop": stop,
        }


def status_lines(shared: Shared) -> list[str]:
    state = shared["trainer_state"]
    word = {RUNNING: "running", PAUSED: "PAUSED", STOPPING: "stopping"}.get(state, "")
    if shared["paused"] and state == RUNNING:
        word = "pausing..."
    return [f"{word}   {shared['timesteps']:,} steps", shared.status]


def _viewer_main(name: str) -> None:
    """The viewer process: play frames from the ring buffer at a steady pace.

    Emulator 0 produces frames in bursts (a few seconds of fast play, then a
    pause while PPO updates), slightly faster than 60 fps on average. So we
    play from a buffer: normally at 60 fps, a little faster when the backlog
    grows, skipping ahead if it ever gets too far behind, and showing the last
    frame dimmed while the buffer is empty.
    """
    shm = shared_memory.SharedMemory(name=name)
    shared = Shared(shm.buf)
    controls = Controls(shared)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, controls.mouse)

    read = 0
    last = np.zeros((HEIGHT * SCALE, WIDTH * SCALE, 3), dtype=np.uint8)
    overlay = ["waiting for the first frames..."]
    tick = 1 / 60
    next_tick = time.perf_counter()
    try:
        while True:
            written = shared["frames_written"]
            backlog = written - read
            if backlog > SLOTS - 120:  # about to be overwritten: jump ahead
                read = written - 60
                backlog = 60
            if backlog > 0:
                # 1 frame per tick normally; up to 3 when ~10 s behind.
                read += 1 + min(backlog // 300, 2)
                slot = (read - 1) % SLOTS
                img = cv2.resize(shared.frames[slot], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
                last = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                overlay = bytes(shared.texts[slot]).rstrip(b"\0").decode(errors="ignore").split("|")
                game = last
            else:
                game = (last * 0.5).astype(np.uint8)  # dim the last frame while nothing new arrives
                note = "PAUSED" if shared["trainer_state"] == PAUSED else "(training is updating the network...)"
                overlay = overlay[:2] + [note]
            cv2.imshow(WINDOW, compose(game, overlay[:3], controls.labels(), controls.hover, status_lines(shared)))
            controls.key(cv2.pollKey())
            next_tick += tick
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    finally:
        del shared, controls
        shm.close()
