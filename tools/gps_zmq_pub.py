#!/usr/bin/env python3
"""GPS/desire ZMQ publisher: runs on the comma device, reads live fixes and
lateral-desire state off the existing openpilot message bus, and republishes
compact JSON summaries over a plain ZMQ PUB socket so a laptop can subscribe
(see gps_rerun_viewer.py / zmq_debug_viewer.py) without needing cereal/capnp
installed locally. Read-only: does not touch the GNSS receiver, the car, or
any openpilot process -- it only subscribes to messages already flowing, and
runs the *actual* on-device DesireHelper/NavDesireInjector classes against
that bus to reconstruct the desire modeld computes (which is itself never
published to any log/topic).

Topics published: "gps" (fix/accuracy/local displacement/ublox C-N0),
"desire" (final commanded desire + raw blinker/blindspot state), and
"parknav" (raw parkNavSignal fields -- alive/valid even when parknavd isn't
running, so you can see it's dead rather than guessing).

Run on the device:

    cd /data/openpilot
    PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 tools/gps_zmq_pub.py"""
import json
import math
import time

import zmq

from openpilot.cereal.messaging import SubMaster
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.modeld.park_nav import NavDesireInjector

PORT = 5555
EARTH_RADIUS_M = 6378137.0

# Signed direction for a log.Desire name: -1 = left, 0 = none, +1 = right.
DESIRE_SIGN = {
  "none": 0,
  "turnLeft": -1,
  "turnRight": 1,
  "laneChangeLeft": -1,
  "laneChangeRight": 1,
  "keepLeft": -1,
  "keepRight": 1,
}


def main():
  sm = SubMaster(
    ["gpsLocationExternal", "ubloxGnss", "carState", "carControl", "parkNavSignal"],
    poll="gpsLocationExternal",
  )
  desire_helper = DesireHelper()
  nav_injector = NavDesireInjector()

  ctx = zmq.Context()
  sock = ctx.socket(zmq.PUB)
  sock.bind(f"tcp://*:{PORT}")

  origin = None
  max_displacement = 0.0
  ublox_seen = False

  print(f"gps_zmq_pub: publishing on tcp://*:{PORT} topics=gps,desire,parknav", flush=True)

  while True:
    sm.update(1000)

    if sm.seen["ubloxGnss"]:
      ublox_seen = True

    ublox_stats = None
    if sm.seen["ubloxGnss"]:
      ub = sm["ubloxGnss"]
      if ub.which() == "measurementReport":
        report = ub.measurementReport
        cnos = [m.cno for m in report.measurements if m.cno > 0]
        ublox_stats = {
          "avgCno": (sum(cnos) / len(cnos)) if cnos else None,
          "maxCno": max(cnos) if cnos else None,
          "numMeas": int(report.numMeas),
        }

    if sm.seen["carState"] and sm.seen["carControl"]:
      cs = sm["carState"]
      cc = sm["carControl"]
      nav_signal = sm["parkNavSignal"]
      nav_alive = sm.alive["parkNavSignal"]

      desire_helper.update(cs, cc.latActive)
      final_desire = str(nav_injector.update(nav_signal, nav_alive, cs, cc.latActive, cs.vEgo, desire_helper.desire))
      blinker_desire = str(desire_helper.desire)
      blinker_signed = -1 if (cs.leftBlinker and not cs.rightBlinker) else (1 if (cs.rightBlinker and not cs.leftBlinker) else 0)

      t = time.time()
      desire_msg = {
        "t": t,
        "desire": final_desire,
        "desireSigned": DESIRE_SIGN.get(final_desire, 0),
        "blinkerDesire": blinker_desire,
        "blinkerDesireSigned": DESIRE_SIGN.get(blinker_desire, 0),
        "blinkerSigned": blinker_signed,
        "leftBlinker": bool(cs.leftBlinker),
        "rightBlinker": bool(cs.rightBlinker),
        "leftBlindspot": bool(cs.leftBlindspot),
        "rightBlindspot": bool(cs.rightBlindspot),
        "latActive": bool(cc.latActive),
        "vEgo": float(cs.vEgo),
      }
      sock.send_multipart([b"desire", json.dumps(desire_msg).encode()])

      parknav_msg = {
        "t": t,
        "alive": bool(nav_alive),
        "valid": bool(nav_signal.valid) if sm.seen["parkNavSignal"] else False,
        "relBearing": float(nav_signal.relBearing) if sm.seen["parkNavSignal"] else None,
        "lateralOffset": float(nav_signal.lateralOffset) if sm.seen["parkNavSignal"] else None,
        "forwardDist": float(nav_signal.forwardDist) if sm.seen["parkNavSignal"] else None,
        "totalDist": float(nav_signal.totalDist) if sm.seen["parkNavSignal"] else None,
      }
      sock.send_multipart([b"parknav", json.dumps(parknav_msg).encode()])

    if not sm.updated["gpsLocationExternal"]:
      continue

    gps = sm["gpsLocationExternal"]

    east = north = displacement = None
    if gps.hasFix:
      if origin is None:
        origin = (gps.latitude, gps.longitude)
      lat0, lon0 = origin
      north = math.radians(gps.latitude - lat0) * EARTH_RADIUS_M
      east = math.radians(gps.longitude - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
      displacement = math.hypot(east, north)
      max_displacement = max(max_displacement, displacement)

    msg = {
      "t": time.time(),
      "hasFix": bool(gps.hasFix),
      "ubloxAvailable": ublox_seen,
      "satelliteCount": int(gps.satelliteCount),
      "horizontalAccuracy": float(gps.horizontalAccuracy),
      "verticalAccuracy": float(gps.verticalAccuracy),
      "speed": float(gps.speed),
      "ublox": ublox_stats,
      "latitude": gps.latitude,
      "longitude": gps.longitude,
      "east": east,
      "north": north,
      "displacement": displacement,
      "maxDisplacement": max_displacement if displacement is not None else None,
    }
    sock.send_multipart([b"gps", json.dumps(msg).encode()])


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass
