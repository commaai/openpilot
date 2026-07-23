"""Locate the FFmpeg tools bundled with the openpilot Python environment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil


def executable(name: str) -> str | None:
  try:
    import ffmpeg as ffmpeg_package
  except ImportError:
    pass
  else:
    package_dir = getattr(ffmpeg_package, "DIR", None)
    if package_dir is not None:
      candidate = Path(package_dir) / "bin" / name
      if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
  return shutil.which(name)
