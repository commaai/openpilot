#!/usr/bin/env python3
import os
import time
import traceback
import serial
import datetime
import numpy as np
from collections import defaultdict

from cereal import log
import cereal.messaging as messaging
from openpilot.common.retry import retry
from openpilot.common.swaglog import cloudlog
from openpilot.system.qcomgpsd.qcomgpsd import at_cmd, wait_for_modem


def sfloat(n: str):
  return float(n) if len(n) > 0 else 0

def checksum(s: str):
  ret = 0
  for c in s[1:-3]:
    ret ^= ord(c)
  return format(ret, '02X')

class Unicore:
  def __init__(self):
    self.s = serial.Serial('/dev/ttyHS0', 115200)
    self.s.timeout = 1
    self.s.writeTimeout = 1
    self.s.newline = b'\r\n'

    self.s.flush()
    self.s.reset_input_buffer()
    self.s.reset_output_buffer()
    self.s.read(2048)

  def send(self, cmd):
    self.s.write(cmd.encode('utf8') + b'\r')
    resp = self.s.read(2048)
    print(len(resp), cmd, "\n", resp)
    assert b"OK" in resp

  def recv(self):
    return self.s.readline()

def build_msg(state):
  """
    NMEA sentences:
      https://campar.in.tum.de/twiki/pub/Chair/NaviGpsDemon/nmea.html#RMC
    NAV messages:
      https://www.unicorecomm.com/assets/upload/file/UFirebird_Standard_Positioning_Products_Protocol_Specification_CH.pdf
  """

  msg = messaging.new_message('gpsLocation', valid=True)
  gps = msg.gpsLocation

  gnrmc = state['$GNRMC']
  gps.hasFix = gnrmc[1] == 'A'
  gps.source = log.GpsLocationData.SensorSource.unicore
  gps.latitude = (sfloat(gnrmc[3][:2]) + (sfloat(gnrmc[3][2:]) / 60)) * (1 if gnrmc[4] == "N" else -1)
  gps.longitude = (sfloat(gnrmc[5][:3]) + (sfloat(gnrmc[5][3:]) / 60)) * (1 if gnrmc[6] == "E" else -1)

  try:
    date = gnrmc[9][:6]
    dt = datetime.datetime.strptime(f"{date} {gnrmc[1]}", '%d%m%y %H%M%S.%f')
    gps.unixTimestampMillis = dt.timestamp()*1e3
  except Exception:
    pass

  gps.bearingDeg = sfloat(gnrmc[8])

  if len(state['$GNGGA']):
    gngga = state['$GNGGA']
    if gngga[10] == 'M':
      gps.altitude = sfloat(gngga[9])

  if len(state['$GNGSA']):
    gngsa = state['$GNGSA']
    gps.horizontalAccuracy = sfloat(gngsa[4])
    gps.verticalAccuracy = sfloat(gngsa[5])

  #if len(state['$NAVACC']):
  #  # $NAVVEL,264415000,5,3,0.375,0.141,-0.735,-65.450*2A
  #  navacc = state['$NAVACC']
  #  gps.horizontalAccuracy = sfloat(navacc[3])
  #  gps.speedAccuracy = sfloat(navacc[4])
  #  gps.bearingAccuracyDeg = sfloat(navacc[5])

  if len(state['$NAVVEL']):
    # $NAVVEL,264415000,5,3,0.375,0.141,-0.735,-65.450*2A
    navvel = state['$NAVVEL']
    vECEF = [
      sfloat(navvel[4]),
      sfloat(navvel[5]),
      sfloat(navvel[6]),
    ]

    lat = np.radians(gps.latitude)
    lon = np.radians(gps.longitude)
    R = np.array([
      [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
      [-np.sin(lat) * np.sin(lon), np.cos(lon), -np.cos(lat) * np.sin(lon)],
      [np.cos(lat), 0, -np.sin(lat)]
    ])

    vNED = [float(x) for x in R.dot(vECEF)]
    gps.vNED = vNED
    gps.speed = np.linalg.norm(vNED)

  # TODO: set these from the module
  gps.bearingAccuracyDeg = 5.
  gps.speedAccuracy = 3.

  return msg


@retry(attempts=10, delay=0.1)
def setup(u):
  at_cmd('AT+CGPS=0')
  at_cmd('AT+CGPS=1')
  time.sleep(1.0)

  # setup NAVXXX outputs
  for i in range(4):
    u.send(f"$CFGMSG,1,{i},1")
  for i in (1, 3):
    u.send(f"$CFGMSG,3,{i},1")

  # 10Hz NAV outputs
  u.send("$CFGNAV,100,100,1000")


def main():
  wait_for_modem("AT+CGPS?")

  u = Unicore()
  setup(u)

  state = defaultdict(list)
  pm = messaging.PubMaster(['gpsLocation'])
  while True:
    try:
      msg = u.recv().decode('utf8').strip()
      if "DEBUG" in os.environ:
        print(repr(msg))

      if len(msg) > 0:
        if checksum(msg) != msg.split('*')[1]:
          cloudlog.error(f"invalid checksum: {repr(msg)}")
          continue

        k = msg.split(',')[0]
        state[k] = msg.split(',')
        if '$GNRMC' not in msg:
          continue

        pm.send('gpsLocation', build_msg(state))
    except Exception:
      traceback.print_exc()
      cloudlog.exception("gps.issue")


if __name__ == "__main__":
  main()
