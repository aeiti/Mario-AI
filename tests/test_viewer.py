"""The live window's controls, without opening a window."""

import mario_ai.viewer as viewer
from mario_ai.viewer import SCALE, HEIGHT, Controls, button_at

BAR_Y = HEIGHT * SCALE + 30  # middle of the control bar


def test_clicks_map_to_buttons_and_misses_map_to_nothing():
    assert button_at(50, BAR_Y) == "pause"
    assert button_at(200, BAR_Y) == "save"
    assert button_at(400, BAR_Y) == "stop"
    assert button_at(600, BAR_Y) is None  # the status text area
    assert button_at(50, 100) is None  # on the game itself


def make_controls():
    shared = {"paused": 0, "save_requests": 0, "stop_requested": 0}
    return Controls(shared), shared


def test_pause_toggles_and_space_works_too():
    controls, shared = make_controls()
    controls.key(ord("p"))
    assert shared["paused"] == 1
    controls.key(ord(" "))
    assert shared["paused"] == 0


def test_each_save_press_is_counted():
    controls, shared = make_controls()
    controls.key(ord("s"))
    controls.press("save")
    assert shared["save_requests"] == 2


def test_stop_needs_a_second_press_within_the_confirm_window(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(viewer.time, "monotonic", lambda: now[0])
    controls, shared = make_controls()

    controls.key(ord("q"))
    assert shared["stop_requested"] == 0
    assert controls.labels()["stop"] == "Confirm: Q"

    now[0] += viewer.CONFIRM_SECONDS + 1  # too slow: the first press expires
    controls.key(ord("q"))
    assert shared["stop_requested"] == 0

    now[0] += 1
    shared["paused"] = 1
    controls.key(ord("Q"))  # second press in time, capital letters too
    assert shared["stop_requested"] == 1
    assert shared["paused"] == 0  # a paused trainer must wake up to stop


def test_no_key_does_nothing():
    controls, shared = make_controls()
    controls.key(-1)  # what pollKey returns when nothing was pressed
    assert shared == {"paused": 0, "save_requests": 0, "stop_requested": 0}


def test_feed_records_whole_attempts_into_free_clips_only():
    import numpy as np
    from multiprocessing import shared_memory
    from mario_ai.viewer import CLIP_FRAMES, FREE, PLAYING, READY, RECORDING, SIZE, FrameFeed, Shared

    shm = shared_memory.SharedMemory(create=True, size=SIZE)
    try:
        shared = Shared(shm.buf)
        shared["clip0_state"] = shared["clip1_state"] = FREE
        feed = FrameFeed(shm.name)
        frame = lambda v: np.full((viewer.HEIGHT, viewer.WIDTH, 3), v, dtype=np.uint8)

        def attempt(value, frames):
            feed.new_attempt()
            for _ in range(frames):
                feed.push(frame(value))

        attempt(1, 3)  # recorded into clip 0
        attempt(2, 5)  # clip 0 handed over; recorded into clip 1
        assert shared["clip0_state"] == READY and shared["clip0_length"] == 3
        attempt(3, 4)  # clip 1 handed over; no clip free, so attempt 3 is skipped
        assert shared["clip1_state"] == READY and shared["clip1_length"] == 5
        assert feed.clip is None

        shared["clip0_state"] = PLAYING  # the viewer plays clip 0, then frees it
        shared["clip0_state"] = FREE
        attempt(4, 2)  # recorded into clip 0, replacing attempt 1
        assert shared["clip0_state"] == RECORDING
        assert shared.frames[0, 0, 0, 0] == 4 and shared.frames[CLIP_FRAMES, 0, 0, 0] == 2

        feed.new_attempt()
        assert shared["clip0_length"] == 2
        del shared, feed
    finally:
        shm.close()
        shm.unlink()
