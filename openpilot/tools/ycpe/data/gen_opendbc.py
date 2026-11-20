#!/usr/bin/env python3
"""Writes opendbc.json, car fingerprint -> its dbc_dict ({"pt": DBC name, "radar": ...}). With a directory argument, also
writes every DBC there, generated ones included, for dbc.test.mjs (not into data/opendbc/: next to this script it would shadow the opendbc package)."""
import json
import sys
from pathlib import Path

from opendbc.car.values import PLATFORMS
from opendbc.dbc.generator.generator import generate_all

HERE = Path(__file__).resolve().parent
SRC = Path(generate_all.__code__.co_filename).parents[1]


def fingerprints() -> dict[str, dict[str, str]]:
  return dict(sorted((p.name, dict(sorted(p.config.dbc_dict.items()))) for p in PLATFORMS.values() if p.config.dbc_dict))


if __name__ == "__main__":
  # one car per line
  (HERE / "opendbc.json").write_text("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in fingerprints().items()) + "\n}\n")
  if len(sys.argv) > 1:
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=True)
    dbcs = {f.stem: f.read_text() for f in SRC.glob("*.dbc")} | generate_all()
    for name, text in dbcs.items():
      (out / f"{name}.dbc").write_text(text)
    print(f"{len(dbcs)} DBCs, {sum(map(len, dbcs.values())) / 1e6:.1f} MB")
