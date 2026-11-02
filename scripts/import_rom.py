"""Install a Super Mario World ROM into stable-retro.

Many SNES dumps (like "Super Mario World (U) [!].smc") carry a 512-byte copier
header left over from old backup devices. stable-retro identifies games by the
SHA-1 of the bare ROM, so we strip the header, check the hash, and copy the
result into retro's game data folder. The ROM itself is never committed.

    uv run scripts/import_rom.py "/path/to/Super Mario World (U) [!].smc"
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import stable_retro as retro

GAME = "SuperMarioWorld-Snes-v0"
COPIER_HEADER = 512


def strip_header(data: bytes) -> bytes:
    # A real SNES ROM is a multiple of 32 KiB; a copier header adds exactly 512 bytes.
    if len(data) % 0x8000 == COPIER_HEADER:
        return data[COPIER_HEADER:]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rom", type=Path)
    args = parser.parse_args()

    rom = strip_header(args.rom.read_bytes())
    sha = hashlib.sha1(rom).hexdigest()
    expected = Path(retro.data.get_file_path(GAME, "rom.sha")).read_text().split()[0]
    if sha != expected:
        print(f"SHA-1 mismatch: got {sha}, stable-retro expects {expected}.", file=sys.stderr)
        print("This needs the US release, Super Mario World (U).", file=sys.stderr)
        return 1

    dest = Path(retro.data.get_file_path(GAME, "rom.sha")).with_name("rom.sfc")
    dest.write_bytes(rom)
    print(f"Installed {GAME} -> {dest}")

    # Keep a header-free copy in roms/ (gitignored) for reference.
    local = Path(__file__).resolve().parent.parent / "roms" / "smw.sfc"
    local.parent.mkdir(exist_ok=True)
    shutil.copyfile(dest, local)
    return 0


if __name__ == "__main__":
    sys.exit(main())
