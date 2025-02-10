#!/usr/bin/env python3
import time
import cereal.messaging as messaging

if __name__ == "__main__":
  sm = messaging.SubMaster(['ubloxGnss', 'gpsLocationExternal'])

  while 1:
    ug = sm['ubloxGnss']
    gle = sm['gpsLocationExternal']

    try:
      cnos = sorted(m.cno for m in ug.measurementReport.measurements)
      print(f"Sats: {ug.measurementReport.numMeas}   Accuracy: {gle.horizontalAccuracy:.2f} m   cnos", cnos)
    except Exception:
      pass
    sm.update()
    time.sleep(0.1)
