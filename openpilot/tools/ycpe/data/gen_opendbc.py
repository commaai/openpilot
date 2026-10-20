#!/usr/bin/env python3
"""Writes opendbc.json (car fingerprint -> DBC name, plus every DBC name) and the DBC files themselves, generated
ones included, into opendbc/ (not committed; deploys run this)."""
import json
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
  dbcs = {f.stem: f.read_text() for f in SRC.glob("*.dbc")} | generate_all()
  out = HERE / "opendbc"
  out.mkdir(exist_ok=True)
  (out / ".gitignore").write_text("*\n")
  for name, text in dbcs.items():
    (out / f"{name}.dbc").write_text(text)
  (HERE / "opendbc.json").write_text(json.dumps({"fingerprints": fingerprints(), "files": sorted(dbcs)}, indent=1) + "\n")
  print(f"{len(dbcs)} DBCs, {sum(map(len, dbcs.values())) / 1e6:.1f} MB")
