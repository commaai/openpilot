#!/usr/bin/env python3
import os
import re
from pathlib import Path

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = HERE + "/.."

blacklist = [
  ".git/",

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
  for f in Path(ROOT).rglob("**/*"):
    if not (f.is_file() or f.is_symlink()):
      continue

    rf = str(f.relative_to(ROOT))
    blacklisted = any(re.search(p, rf) for p in blacklist)
    whitelisted = any(re.search(p, rf) for p in whitelist)
    if blacklisted and not whitelisted:
      continue

    print(rf)
