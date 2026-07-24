"""
  wrapper that materializes symlinks in docs/ before build

  we can delete this once zensical supports symlinks:
  https://github.com/zensical/backlog/issues/55
"""
import os
import shutil
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
SITE_DIR = REPO_ROOT / "docs_site"
sys.path.insert(0, str(REPO_ROOT))
# Local docs build helpers live under docs/ so they stay near the content
# source. The wrapper prunes them from docs_site/ after build.
sys.path.insert(0, str(DOCS_DIR))


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


def _prune_site_output() -> None:
  shutil.rmtree(SITE_DIR / "ext", ignore_errors=True)


def main() -> None:
  signal.signal(signal.SIGTERM, _raise_interrupt)
  originals = _materialize(DOCS_DIR)
  try:
    from zensical.main import cli
    cli(standalone_mode=False)
    if len(sys.argv) > 1 and sys.argv[1] == "build":
      _prune_site_output()
  finally:
    _restore(originals)


if __name__ == "__main__":
  main()
