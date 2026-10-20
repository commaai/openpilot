#!/usr/bin/env python3
"""Writes opendbc.json, car fingerprint -> DBC name. With a directory argument, also writes every DBC there, generated
ones included, for dbc.test.mjs (not into data/opendbc/: next to this script it would shadow the opendbc package)."""
import json
import sys
from pathlib import Path

from opendbc.car import Bus
from opendbc.car.fingerprints import MIGRATION
from opendbc.car.values import PLATFORMS
from opendbc.dbc.generator.generator import generate_all

HERE = Path(__file__).resolve().parent
SRC = Path(generate_all.__code__.co_filename).parents[1]


def fingerprints() -> dict[str, str]:
  # as cabana's generate_dbc_json.py: the powertrain DBC, else main, else party
  out = {}
  for p in PLATFORMS.values():
    dbcs = p.config.dbc_dict
    bus = next((b for b in (Bus.pt, Bus.main, Bus.party) if b in dbcs), None)
    if bus is not None:
      out[p.name] = dbcs[bus]
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
