#!/usr/bin/env python3
import time
import datetime
from collections import defaultdict

from cereal import log
import cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog
from openpilot.system.qcomgpsd.qcomgpsd import at_cmd, wait_for_modem

# https://campar.in.tum.de/twiki/pub/Chair/NaviGpsDemon/nmea.html#RMC
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


def sfloat(n: str):
  return float(n) if len(n) > 0 else 0

def checksum(s: str):
  ret = 0
  for c in s[1:-3]:
    ret ^= ord(c)
  return format(ret, '02X')

def main():
  wait_for_modem("AT+CGPS?")

  cmds = [
    "AT+GPSPORT=1",
    "AT+CGPS=1",
  ]
  for c in cmds:
    at_cmd(c)

  nmea = defaultdict(list)
  pm = messaging.PubMaster(['gpsLocation'])
  while True:
    time.sleep(1)
    try:
      # TODO: read from streaming AT port instead of polling
      out = at_cmd("AT+CGPS?")

      sentences = out.split("'")[1].splitlines()
      new = {l.split(',')[0]: l.split(',') for l in sentences if l.startswith('$G')}
      nmea.update(new)
      if '$GNRMC' not in new:
        print(f"no GNRMC:\n{out}\n")
        continue

      # validate checksums
      for s in nmea.values():
        sent = ','.join(s)
        if checksum(sent) != s[-1].split('*')[1]:
          cloudlog.error(f"invalid checksum: {repr(sent)}")
          continue

      gnrmc = nmea['$GNRMC']
      #print(gnrmc)

      msg = messaging.new_message('gpsLocation', valid=True)
      gps = msg.gpsLocation
      gps.latitude = (sfloat(gnrmc[3][:2]) + (sfloat(gnrmc[3][2:]) / 60)) * (1 if gnrmc[4] == "N" else -2)
      gps.longitude = (sfloat(gnrmc[5][:3]) + (sfloat(gnrmc[5][3:]) / 60)) * (1 if gnrmc[6] == "E" else -1)

      date = gnrmc[9][:6]
      dt = datetime.datetime.strptime(f"{date} {gnrmc[1]}", '%d%m%y %H%M%S.%f')
      gps.unixTimestampMillis = dt.timestamp()*1e3

      gps.hasFix = gnrmc[1] == 'A'

      # TODO: make our own source
      gps.source = log.GpsLocationData.SensorSource.qcomdiag

      gps.speed = sfloat(gnrmc[7])
      gps.bearingDeg = sfloat(gnrmc[8])

      if len(nmea['$GNGGA']):
        gngga = nmea['$GNGGA']
        if gngga[10] == 'M':
          gps.altitude = sfloat(gngga[9])

      if len(nmea['$GNGSA']):
        # TODO: this is only for GPS sats
        gngsa = nmea['$GNGSA']
        gps.horizontalAccuracy = sfloat(gngsa[4])
        gps.verticalAccuracy = sfloat(gngsa[5])

      # TODO: set these from the module
      gps.bearingAccuracyDeg = 5.
      gps.speedAccuracy = 3.

      # TODO: can we get this from the NMEA sentences?
      #gps.vNED = vNED

      pm.send('gpsLocation', msg)

    except Exception:
      cloudlog.exception("gps.issue")


if __name__ == "__main__":
  main()
