#!/usr/bin/env python3
import os
import re
import glob
from pathlib import Path

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = HERE + "/.."

# blacklisting is for two purposes:
# - minimizing release download size
# - keeping the diff readable
blacklist = [
  "body/STL",
  "tools/cabana/",
  "opendbc/generator",
  "third_party/acados/acados_template/gnsf",

  ".git$",  # for submodules
  ".git/",
  ".github/",
  ".devcontainer/",
  "Darwin/",
  ".vscode",
]

if __name__ == "__main__":
  for f in Path(ROOT).glob("**/*"):
    if not (f.is_file() or f.is_symlink()):
      continue
    #if any(x in str(f).split('/') for x in blacklisted_dirs):
    #  continue

    rf = str(f.relative_to(ROOT))
    if any(re.search(pattern, rf) for pattern in blacklist):
      continue

    assert " " not in rf, rf
    print(rf)
