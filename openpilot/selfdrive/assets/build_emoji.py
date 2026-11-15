#!/usr/bin/env python3
# Builds emoji.npz from Noto Color Emoji's 72px PNGs and region flags: uv run --with pillow build_emoji.py
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

NOTO_EMOJI = "https://github.com/googlefonts/noto-emoji.git"
COMMIT = "99701e07d3d1fe1bdbcdb14964547f0fd592966f"
OUT = Path(__file__).parent / "emoji.npz"


def main():
  with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp)
    git = ["git", "-C", tmp]
    subprocess.run(["git", "init", "-q", tmp], check=True)
    subprocess.run([*git, "sparse-checkout", "set", "2D/png/72", "third_party/region-flags/png"], check=True)
    subprocess.run([*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", NOTO_EMOJI, COMMIT], check=True)
    subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True)

    arrays = {}
    for f in sorted((src / "2D/png/72").glob("emoji_u*.png")):
      key = f.stem.removeprefix("emoji_u")
      codepoints = [int(c, 16) for c in key.split("_")]
      # digits, # and * are always text, and only the default skin tone ships
      if len(codepoints) == 1 and codepoints[0] < 0x2000 and codepoints[0] not in (0xA9, 0xAE):
        continue
      if any(0x1F3FB <= c <= 0x1F3FF for c in codepoints):
        continue
      arrays[key] = np.asarray(Image.open(f).convert("RGBA"))

    # flags are pairs of regional indicators, one per letter of the country code
    for f in sorted((src / "third_party/region-flags/png").glob("??.png")):
      img = Image.open(f).convert("RGBA")
      img = img.resize((72, max(1, round(72 * img.height / img.width))), Image.Resampling.LANCZOS)
      arrays["_".join(f"{0x1F1E6 + ord(c) - ord('A'):x}" for c in f.stem)] = np.asarray(img)

  np.savez_compressed(OUT, **arrays)
  print(f"{len(arrays)} emoji, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
  main()
