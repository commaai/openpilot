#!/usr/bin/env python3
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def git(*args, check=True):
  return subprocess.run(["git", *args], capture_output=True, check=check).stdout.decode().strip()


ROOT, GIT_DIR = map(Path, git("rev-parse", "--path-format=absolute", "--show-toplevel", "--git-common-dir").splitlines())
HELPER = GIT_DIR / "lfs-setup.py"
SHIM = GIT_DIR / "git-lfs"
STOCK_HOOK = '#!/bin/sh\nexec "$(git rev-parse --git-common-dir)/git-lfs" pre-push "$@"\n'
CUSTOM_HOOK = '#!/bin/sh\nexec "$(git rev-parse --git-common-dir)/lfs.py" pre-push "$@"\n'


def stock_binary():
  for root in (ROOT, GIT_DIR.parent):
    for path in root.glob(".venv/lib/python*/site-packages/git_lfs/bin/git-lfs"):
      if os.access(path, os.X_OK):
        return str(path)
  binary = shutil.which("git-lfs")
  if binary and Path(binary).resolve() != SHIM.resolve():
    return binary
  raise RuntimeError("git-lfs is missing; run tools/op.sh setup")


def remove_stock_hooks(directory):
  for name in ("pre-push", "post-checkout", "post-commit", "post-merge"):
    path = directory / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024:
      continue
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    commands = {f'git lfs {name} "$@"'}
    if name == "pre-push":
      commands.update(('git lfs push --stdin $*', 'git lfs push --stdin "$@"'))
      if lines and re.fullmatch(r'exec "[^"]*/git-lfs" pre-push "\$@"', lines[-1]):
        commands.add(lines[-1])
    has_guard = len(lines) == 3 and lines[1].startswith("command -v git-lfs ")
    if lines[:1] == ["#!/bin/sh"] and lines[-1] in commands and (len(lines) == 2 or has_guard):
      path.unlink()


def install_stock():
  stock_binary()
  hook = Path(git("rev-parse", "--path-format=absolute", "--git-path", "hooks/pre-push"))
  remove_stock_hooks(hook.parent)
  if hook.is_file() and hook.read_text(errors="replace") not in (STOCK_HOOK, CUSTOM_HOOK):
    raise RuntimeError(f"refusing to replace custom hook {hook}")
  SHIM.write_text('#!/bin/sh\nexec "$(git rev-parse --git-common-dir)/lfs-setup.py" stock "$@"\n')
  SHIM.chmod(0o755)
  git("config", "--worktree", "--remove-section", "filter.lfs", check=False)
  command = '"$(git rev-parse --git-common-dir)/git-lfs"'
  for name, args in (("clean", "clean -- %f"), ("smudge", "smudge -- %f"), ("process", "filter-process")):
    git("config", "--local", f"filter.lfs.{name}", f"{command} {args}")
  git("config", "--local", "filter.lfs.required", "true")
  hook.parent.mkdir(parents=True, exist_ok=True)
  hook.write_text(STOCK_HOOK)
  hook.chmod(0o755)
  (GIT_DIR / "lfs.py").unlink(missing_ok=True)


def main():
  command, *args = sys.argv[1:]
  if command == "stock":
    os.execv(stock_binary(), ["git-lfs", *args])
  if Path(__file__).resolve() != HELPER:
    shutil.copyfile(__file__, HELPER)
    HELPER.chmod(0o755)
  if command == "git":
    # Read the new checkout without invoking the previous version's filters or hooks.
    options = [f"filter.lfs.{name}=" for name in ("process", "clean", "smudge")]
    options += ["filter.lfs.required=false", "core.hooksPath=/dev/null"]
    os.execvp("git", ["git", *(arg for option in options for arg in ("-c", option)), *args])
  custom = ROOT / "lfs.py"
  if custom.is_file():
    os.execv(custom, [str(custom), command, *args])
  if command in ("install", "pull"):
    install_stock()
  if command != "install":
    os.execv(stock_binary(), ["git-lfs", command, *args])


if __name__ == "__main__":
  main()
