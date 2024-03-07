#!/usr/bin/env python3
import time
import datetime

from cereal import log
import cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog
from openpilot.system.qcomgpsd.qcomgpsd import at_cmd, wait_for_modem

"""
AT+CGPSGPOS=1
response: '$GNGGA,220212.00,3245.09188,N,11711.76362,W,1,06,24.54,0.0,M,,M,,*77'

AT+CGPSGPOS=2
response: '$GNGSA,A,3,06,17,19,22,,,,,,,,,14.11,8.95,10.91,1*01
$GNGSA,A,3,29,26,,,,,,,,,,,14.11,8.95,10.91,4*03'

AT+CGPSGPOS=3
response: '$GPGSV,3,1,11,06,55,047,22,19,29,053,20,22,19,115,14,05,01,177,,0*68
$GPGSV,3,2,11,11,77,156,23,12,47,322,17,17,08,066,10,20,25,151,,0*6D
$GPGSV,3,3,11,24,44,232,,25,16,312,,29,02,260,,0*5D'

AT+CGPSGPOS=4
response: '$GBGSV,1,1,03,26,75,242,20,29,19,049,16,35,,,24,0*7D'

AT+CGPSGPOS=5
response: '$GNRMC,220216.00,A,3245.09531,N,11711.76043,W,,,070324,,,A,V*20'
"""

def main():
  wait_for_modem("AT+CGPS?")
  print("got modem")

  # enable GPS
  cmds = [
    #"AT+CGPS=0",
    "AT+GPSPORT=1",
    "AT+CGPS=1",
  ]
  for c in cmds:
    at_cmd(c)

  pm = messaging.PubMaster(['gpsLocation'])
  while True:
    time.sleep(1)
    try:
      # TODO: read from streaming AT port instead of polling
      out = at_cmd("AT+CGPS?")
      nmea_sentences = out.split("'")[1].splitlines()
      gnrmc_sentences = [l for l in nmea_sentences if l.startswith('$GNRMC')]
      if len(gnrmc_sentences) == 0:
        print(f"no GNRMC:\n{out}\n")
        continue

      gnrmc = gnrmc_sentences[-1].split(",")
      print(gnrmc_sentences[-1], gnrmc)
      assert gnrmc[0] == "$GNRMC"

      if gnrmc.count('') > 5:
        print("no fix :(")
        continue

      msg = messaging.new_message('gpsLocation', valid=True)
      gps = msg.gpsLocation
      gps.latitude = (float(gnrmc[3][:2]) + (float(gnrmc[3][2:]) / 60)) * (1 if gnrmc[4] == "N" else -1)
      gps.longitude = (float(gnrmc[5][:3]) + (float(gnrmc[5][3:]) / 60)) * (1 if gnrmc[6] == "E" else -1)

      date = gnrmc[9][:6]
      dt = datetime.datetime.strptime(f"{date} {gnrmc[1]}", '%d%m%y %H%M%S.%f')
      gps.unixTimestampMillis = dt.timestamp()*1e3

      #gps.hasFix = gnrmc[1] == 'A'
      gps.flags = 1 if gnrmc[1] == 'A' else 0

      # TODO: make our own source
      gps.source = log.GpsLocationData.SensorSource.qcomdiag

      """
      gps.altitude = report["q_FltFinalPosAlt"]
      gps.speed = math.sqrt(sum([x**2 for x in vNED]))
      gps.bearingDeg = report["q_FltHeadingRad"] * 180/math.pi

      gps.vNED = vNED
      gps.verticalAccuracy = report["q_FltVdop"]
      gps.bearingAccuracyDeg = report["q_FltHeadingUncRad"] * 180/math.pi if (report["q_FltHeadingUncRad"] != 0) else 180
      gps.speedAccuracy = math.sqrt(sum([x**2 for x in vNEDsigma]))
      """

      pm.send('gpsLocation', msg)

    except Exception:
      cloudlog.exception("gps.issue")


if __name__ == "__main__":
  main()
