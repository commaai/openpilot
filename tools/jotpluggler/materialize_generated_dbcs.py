#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from opendbc import get_generated_dbcs


def main() -> int:
  parser = argparse.ArgumentParser(description="Materialize generated opendbc DBCs for JotPlugger")
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  for existing in out_dir.glob("*.dbc"):
    existing.unlink()

  for name, content in sorted(get_generated_dbcs().items()):
    (out_dir / f"{name}.dbc").write_text(content)

  (out_dir / ".stamp").write_text("ok\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
