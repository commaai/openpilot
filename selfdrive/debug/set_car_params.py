#!/usr/bin/env python3
import sys

from cereal import car
from openpilot.common.params import Params
from openpilot.tools.lib.route import Route
from openpilot.tools.lib.logreader import LogReader

if __name__ == "__main__":
  CP = None
  if len(sys.argv) > 1:
    r = Route(sys.argv[1])
    cps = [m for m in LogReader(r.qlog_paths()[0]) if m.which() == 'carParams']
    CP = cps[0].carParams.as_builder()
  else:
    CP = car.CarParams.new_message()
    CP.openpilotLongitudinalControl = True
    CP.alphaLongitudinalAvailable = False

  cp_bytes = CP.to_bytes()
  for p in ("CarParams", "CarParamsCache", "CarParamsPersistent"):
    Params().put(p, cp_bytes)
