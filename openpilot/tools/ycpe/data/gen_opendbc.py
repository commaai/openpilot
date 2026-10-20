#!/usr/bin/env python3
"""Writes opendbc.json, car fingerprint -> its dbc_dict ({"pt": DBC name, "radar": ...}). With a directory argument, also
writes every DBC there, generated ones included, for dbc.test.mjs (not into data/opendbc/: next to this script it would shadow the opendbc package)."""
import json
import sys
from pathlib import Path

from opendbc.car.fingerprints import MIGRATION
from opendbc.car.values import PLATFORMS
from opendbc.dbc.generator.generator import generate_all

HERE = Path(__file__).resolve().parent
SRC = Path(generate_all.__code__.co_filename).parents[1]


def fingerprints() -> dict[str, dict[str, str]]:
  out = {p.name: dict(sorted(p.config.dbc_dict.items())) for p in PLATFORMS.values() if p.config.dbc_dict}
  out.update({old: out[new] for old, new in MIGRATION.items() if new in out})
  return dict(sorted(out.items()))


if __name__ == "__main__":
  (HERE / "opendbc.json").write_text(json.dumps(fingerprints(), indent=1) + "\n")
  if len(sys.argv) > 1:
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=True)
    dbcs = {f.stem: f.read_text() for f in SRC.glob("*.dbc")} | generate_all()
    for name, text in dbcs.items():
      (out / f"{name}.dbc").write_text(text)
    print(f"{len(dbcs)} DBCs, {sum(map(len, dbcs.values())) / 1e6:.1f} MB")
