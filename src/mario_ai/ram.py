"""Named Super Mario World RAM addresses, and helpers to read them.

The agent only *sees* pixels. RAM is used for the things pixels can't cheaply
tell us: how far right Mario is (the progress reward), whether he is dying,
and whether he broke the goal tape. Each address below was checked live in
the emulator (see scripts/explore.py), not just copied from a RAM map.

Addresses are offsets into the SNES work RAM ($7E0000) — the array
`env.get_ram()` returns. Reference: SMW Central RAM map,
https://www.smwcentral.net/?p=memorymap&game=smw&region=ram
"""

import numpy as np

# Player X position in the level, in pixels, 16-bit little-endian.
# Yoshi's Island 1 starts at 16; the goal tape is at about 4825.
PLAYER_X = 0x94
# Player Y position, 16-bit; grows downward. Standing on YI1's ground ≈ 352.
PLAYER_Y = 0x96
# Player animation state. 0x09 = the death animation has started.
PLAYER_ANIMATION = 0x71
DYING = 0x09
# End-level timer. Becomes 255 on the frame Mario breaks the goal tape,
# then counts down to 0 while the "course clear" sequence plays.
END_LEVEL_TIMER = 0x1493
# Game mode. 0x14 (20) = playing a level.
GAME_MODE = 0x100
IN_LEVEL = 0x14
# Translated level number. Yoshi's Island 1 = 0x29, Yoshi's Island 2 = 0x2A.
LEVEL_NUMBER = 0x13BF
LIVES = 0xDBE
# Star-power timer. Only used by explore.py's goal test, to survive enemies.
STAR_TIMER = 0x1490


def u16(ram: np.ndarray, addr: int) -> int:
    # Cast before shifting: `ram` is uint8, so `ram[a + 1] << 8` would wrap
    # back into 0..255 and silently drop the high byte.
    return int(ram[addr]) | int(ram[addr + 1]) << 8


def player_x(ram: np.ndarray) -> int:
    return u16(ram, PLAYER_X)


def is_dying(ram: np.ndarray) -> bool:
    return int(ram[PLAYER_ANIMATION]) == DYING


def level_cleared(ram: np.ndarray) -> bool:
    return int(ram[END_LEVEL_TIMER]) != 0
