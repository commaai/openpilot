#!/usr/bin/env python3
"""Writes data/migrate.json: the opendbc and openpilot tables that data/migrate.js needs from migration.py."""
import json
from pathlib import Path

import numpy as np

from opendbc.car.fingerprints import MIGRATION
from opendbc.car.toyota.values import EPS_SCALE, ToyotaSafetyFlags
from opendbc.car.ford.values import CAR as FORD, FordFlags, FordSafetyFlags
from opendbc.car.hyundai.values import HyundaiSafetyFlags
from opendbc.car.gm.values import GMSafetyFlags
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.system.manager.process_config import managed_processes

# same as migrate_pandaStates
safety_param = {
  "TOYOTA_PRIUS": EPS_SCALE["TOYOTA_PRIUS"] | ToyotaSafetyFlags.STOCK_LONGITUDINAL,
  "TOYOTA_RAV4": EPS_SCALE["TOYOTA_RAV4"] | ToyotaSafetyFlags.ALT_BRAKE,
  "KIA_EV6": HyundaiSafetyFlags.EV_GAS | HyundaiSafetyFlags.CANFD_LKA_STEER_MSG,
  "CHEVROLET_VOLT": GMSafetyFlags.EV,
  "CHEVROLET_BOLT_EUV": GMSafetyFlags.EV | GMSafetyFlags.HW_CAM,
}
safety_param |= dict.fromkeys({p for p in FORD if not (p.config.flags & FordFlags.CANFD)}, FordSafetyFlags.LONG_CONTROL)

# fill_xyz_poly's polyfit is linear in the samples: coefficients = polyfit @ samples
t = np.array(ModelConstants.T_IDXS)
polyfit = np.polynomial.polynomial.polyfit(t, np.eye(len(t)), deg=ModelConstants.POLY_PATH_DEGREE)

if __name__ == "__main__":
  out = {
    "fingerprints": MIGRATION,
    "safetyParam": {str(k): int(v) for k, v in sorted(safety_param.items())},
    "processes": list(managed_processes),
    "polyfit": polyfit.tolist(),
  }
  path = Path(__file__).parents[2] / "data" / "migrate.json"
  path.write_text(json.dumps(out, separators=(",", ":")))
  print(f"{path}: {path.stat().st_size // 1024} KB")
