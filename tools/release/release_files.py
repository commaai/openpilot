#!/usr/bin/env python3
import os
import re
import subprocess
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))

blacklist = [
  ".git/",
  ".venv/",
  ".github/workflows/",

  "matlab.*.md",

  # no LFS or submodules in release
  ".lfsconfig",
  ".gitattributes",
  ".git$",
  ".gitmodules",
]

# gets you through the blacklist
whitelist: list[str] = [
]

if __name__ == "__main__":
  tracked_files = subprocess.check_output(["git", "ls-files", "-z", "--recurse-submodules"], cwd=ROOT).split(b"\0")
  for tracked_file in tracked_files:
    if not tracked_file:
      continue

    rf = os.fsdecode(tracked_file)
    if not os.getenv("INCLUDE_BIG_MODEL") and rf.startswith("openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"):
      continue
    blacklisted = any(re.search(p, rf) for p in blacklist)
    whitelisted = any(re.search(p, rf) for p in whitelist)
    if blacklisted and not whitelisted:
      continue

    sys.stdout.buffer.write(tracked_file + b"\0")
