#!/usr/bin/env python3
import time
import cereal.messaging as messaging

if __name__ == "__main__":
  sm = messaging.SubMaster(['ubloxGnss', 'gpsLocationExternal'])

  while 1:
    ug = sm['ubloxGnss']
    gle = sm['gpsLocationExternal']

    try:
      cnos = []
      for m in ug.measurementReport.measurements:
        cnos.append(m.cno)
      print(f"Sats: {ug.measurementReport.numMeas}   Accuracy: {gle.horizontalAccuracy:.2f} m   cnos", sorted(cnos))
    except Exception:
      pass
    sm.update()
    time.sleep(0.1)
