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
from openpilot.system.hardware import PC
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
    tty = "USB0" if PC else "HS0"
    self.s = serial.Serial('/dev/tty' + tty, 115200)
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
    return resp

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
  gps.source = log.GpsLocationData.SensorSource.unicore

  if "USE_NMEA" in os.environ:
    gnrmc = state['$GNRMC']
    gps.hasFix = gnrmc[1] == 'A'
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

  gnrmc = state['$GNRMC']
  gps.bearingDeg = sfloat(gnrmc[8])

  # $NAVPOS,0,1,1,-2454130.135,-4775892.719,3431018.827,32.752570,-117.196723,195.880771*2A
  navpos = state['$NAVPOS']
  gps.hasFix = sfloat(navpos[3]) >= 1
  gps.latitude = sfloat(navpos[7])
  gps.longitude = sfloat(navpos[8])
  gps.altitude = sfloat(navpos[9])

  try:
    navtime = state['$NAVTIME']
    # TODO needs update if there is another leap second, after june 2024?
    dt_timestamp = (datetime.datetime(1980, 1, 6, 0, 0, 0, 0, datetime.UTC) +
                    datetime.timedelta(weeks=int(sfloat(navtime[1]))) +
                    datetime.timedelta(seconds=(sfloat(navtime[2]) - 18)))
    gps.unixTimestampMillis = int(dt_timestamp.timestamp()*1e3)
    print(datetime.datetime.now(datetime.UTC) - dt_timestamp)
  except Exception as e:
    print(str(e))

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

    # ECEF velocity -> NED
    lat = np.radians(gps.latitude)
    lon = np.radians(gps.longitude)
    R = np.array([
      [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
      [-np.sin(lat) * np.sin(lon), np.cos(lon), -np.cos(lat) * np.sin(lon)],
      [np.cos(lat), 0, -np.sin(lat)]
    ])
    vNED = [float(x) for x in R.dot(vECEF)]

    gps.vNED = vNED
    gps.speed = float(np.linalg.norm(vNED))

  # TODO: set these from the module
  gps.bearingAccuracyDeg = 5.
  gps.speedAccuracy = 3.
  gps.horizontalAccuracy = 1.
  gps.verticalAccuracy = 1.

  if "DEBUG" in os.environ:
    print(gps)

  return msg


@retry(attempts=10, delay=0.1)
def setup(u):
  if "SKIP_SETUP" in os.environ:
    return

  if not PC:
    wait_for_modem("AT+CGPS?")
    #at_cmd('AT+CGPS=0')
    at_cmd('AT+CGPS=1')
  time.sleep(1.0)

  # reset with cold start
  u.send("$RESET,0,hff")

  # NMEA outputs
  #for i in range(8):
  #  u.send(f"$CFGMSG,0,{i},1")

  # setup NAVXXX outputs
  for i in range(4):
    u.send(f"$CFGMSG,1,{i},1")

  # atenna status outputs
  for i in (1, 3):
    u.send(f"$CFGMSG,3,{i},1")

  # 10Hz NAV outputs
  #u.send("$CFGNAV,100,100,1000")

  # AGPS
  now = datetime.datetime.utcnow()
  if now.year == 2024 and now.month == 3:
    u.send(f"$AIDTIME,{now.year},{now.month},{now.day},{now.hour},{now.minute},{now.second},{int(now.microsecond/1000)}")

  # dynamic config
  static = 0 # 0 for portable
  u.send(f"$CFGDYN,2,{static},0")

  # elevation output is altitude
  u.send("$CFGGEOID,1")

  # debug info
  #u.send("$CFGPRT,1")
  #u.send("$CFGPRT,2")
  #u.send("$PDTINFO")


def main():
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

        # update state
        k = msg.split(',')[0]
        state[k] = msg.split('*')[0].split(',')

        # publish on new position
        if '$NAVPOS' in msg:
          pm.send('gpsLocation', build_msg(state))
    except Exception:
      traceback.print_exc()
      cloudlog.exception("gps.issue")


if __name__ == "__main__":
  main()
