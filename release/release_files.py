#!/usr/bin/env python3
import os
import re
from pathlib import Path

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = HERE + "/.."

# blacklisting is for two purposes:
# - minimizing release download size
# - keeping the diff readable
blacklist = [
  "^scripts/",
  "body/STL/",
  "tools/cabana/",
  "panda/examples/",
  "opendbc/generator/",

  "^tools/",
  "^tinygrad_repo/",

  "matlab.*.md",

  ".git$",  # for submodules
  ".git/",
  ".github/",
  ".devcontainer/",
  "Darwin/",
  ".vscode",
]

# gets you through the blacklist
whitelist = [
  "tools/lib/",

  "tinygrad_repo/openpilot/compile2.py",
  "tinygrad_repo/extra/onnx.py",
  "tinygrad_repo/extra/onnx_ops.py",
  "tinygrad_repo/extra/thneed.py",
  "tinygrad_repo/extra/utils.py",
  "tinygrad_repo/tinygrad/codegen/kernel.py",
  "tinygrad_repo/tinygrad/codegen/linearizer.py",
  "tinygrad_repo/tinygrad/features/image.py",
  "tinygrad_repo/tinygrad/features/search.py",
  "tinygrad_repo/tinygrad/nn/*",
  "tinygrad_repo/tinygrad/renderer/cstyle.py",
  "tinygrad_repo/tinygrad/renderer/opencl.py",
  "tinygrad_repo/tinygrad/runtime/lib.py",
  "tinygrad_repo/tinygrad/runtime/ops_cpu.py",
  "tinygrad_repo/tinygrad/runtime/ops_disk.py",
  "tinygrad_repo/tinygrad/runtime/ops_gpu.py",
  "tinygrad_repo/tinygrad/shape/*",
  "tinygrad_repo/tinygrad/*.py",
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
