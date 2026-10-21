#!/usr/bin/env python3
"""Writes data/migrate.json: the opendbc and openpilot tables that data/migrate.js needs from migration.py."""
import json
from pathlib import Path

import numpy as np

from opendbc.car.fingerprints import MIGRATION
from openpilot.selfdrive.modeld.constants import ModelConstants

# fill_xyz_poly's polyfit is linear in the samples: coefficients = polyfit @ samples
t = np.array(ModelConstants.T_IDXS)
polyfit = np.polynomial.polynomial.polyfit(t, np.eye(len(t)), deg=ModelConstants.POLY_PATH_DEGREE)

if __name__ == "__main__":
  out = {"fingerprints": MIGRATION, "polyfit": polyfit.tolist()}
  path = Path(__file__).parents[2] / "data" / "migrate.json"
  path.write_text(json.dumps(out, separators=(",", ":")))
  print(f"{path}: {path.stat().st_size // 1024} KB")
