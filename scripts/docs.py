"""Wrapper around zensical that materializes symlinks in docs/ before build."""
from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def _materialize(docs: Path) -> dict[Path, str]:
  originals: dict[Path, str] = {}
  for link in docs.rglob("*"):
    if not link.is_symlink():
      continue
    target = link.resolve()
    if not target.is_file():
      continue
    originals[link] = os.readlink(link)
    link.unlink()
    shutil.copy2(target, link)
  return originals


def _restore(originals: dict[Path, str]) -> None:
  for link, target in originals.items():
    link.unlink(missing_ok=True)
    os.symlink(target, link)


def _raise_interrupt(*_):
  raise KeyboardInterrupt


def main() -> None:
  signal.signal(signal.SIGTERM, _raise_interrupt)
  originals = _materialize(DOCS_DIR)
  try:
    from zensical.main import cli
    cli()
  finally:
    _restore(originals)


if __name__ == "__main__":
  main()
