import os
import sys
from dataclasses import dataclass, fields
from subprocess import check_output, CalledProcessError
from time import sleep
from typing import NoReturn

DEBUG = int(os.environ.get("DEBUG", "0"))

@dataclass
class GnssClockNmeaPort:
  # flags bit mask:
  # 0x01 = leap_seconds valid
  # 0x02 = time_uncertainty_ns valid
  # 0x04 = full_bias_ns valid
  # 0x08 = bias_ns valid
  # 0x10 = bias_uncertainty_ns valid
  # 0x20 = drift_nsps valid
  # 0x40 = drift_uncertainty_nsps valid
  flags: int
  leap_seconds: int
  time_ns: int
  time_uncertainty_ns: int # 1-sigma
  full_bias_ns: int
  bias_ns: float
  bias_uncertainty_ns: float # 1-sigma
  drift_nsps: float
  drift_uncertainty_nsps: float # 1-sigma

  def __post_init__(self):
    for field in fields(self):
      val = getattr(self, field.name)
      setattr(self, field.name, field.type(val) if val else None)

@dataclass
class GnssMeasNmeaPort:
  messageCount: int
  messageNum: int
  svCount: int
  # constellation enum:
  # 1 = GPS
  # 2 = SBAS
  # 3 = GLONASS
  # 4 = QZSS
  # 5 = BEIDOU
  # 6 = GALILEO
  constellation: int
  svId: int
  flags: int # always zero
  time_offset_ns: int
  # state bit mask:
  # 0x0001 = CODE LOCK
  # 0x0002 = BIT SYNC
  # 0x0004 = SUBFRAME SYNC
  # 0x0008 = TIME OF WEEK DECODED
  # 0x0010 = MSEC AMBIGUOUS
  # 0x0020 = SYMBOL SYNC
  # 0x0040 = GLONASS STRING SYNC
  # 0x0080 = GLONASS TIME OF DAY DECODED
  # 0x0100 = BEIDOU D2 BIT SYNC
  # 0x0200 = BEIDOU D2 SUBFRAME SYNC
  # 0x0400 = GALILEO E1BC CODE LOCK
  # 0x0800 = GALILEO E1C 2ND CODE LOCK
  # 0x1000 = GALILEO E1B PAGE SYNC
  # 0x2000 = GALILEO E1B PAGE SYNC
  state: int
  time_of_week_ns: int
  time_of_week_uncertainty_ns: int # 1-sigma
  carrier_to_noise_ratio: float
  pseudorange_rate: float
  pseudorange_rate_uncertainty: float # 1-sigma

  def __post_init__(self):
    for field in fields(self):
      val = getattr(self, field.name)
      setattr(self, field.name, field.type(val) if val else None)

def nmea_checksum_ok(s):
  checksum = 0
  for i, c in enumerate(s[1:]):
    if c == "*":
      if i != len(s) - 4: # should be 3rd to last character
        print("ERROR: NMEA string does not have checksum delimiter in correct location:", s)
        return False
      break
    checksum ^= ord(c)
  else:
    print("ERROR: NMEA string does not have checksum delimiter:", s)
    return False

  return True

def process_nmea_port_messages(device:str="/dev/ttyUSB1") -> NoReturn:
  while True:
    try:
      with open(device) as nmeaport:
        for line in nmeaport:
          line = line.strip()
          if DEBUG:
            print(line)
          if not line.startswith("$"): # all NMEA messages start with $
            continue
          if not nmea_checksum_ok(line):
            continue

          fields = line.split(",")
          match fields[0]:
            case "$GNCLK":
              # fields at end are reserved (not used)
              gnss_clock = GnssClockNmeaPort(*fields[1:10]) # type: ignore[arg-type]
              print(gnss_clock)
            case "$GNMEAS":
              # fields at end are reserved (not used)
              gnss_meas = GnssMeasNmeaPort(*fields[1:14]) # type: ignore[arg-type]
              print(gnss_meas)
    except Exception as e:
      print(e)
      sleep(1)

def main() -> NoReturn:
  from openpilot.common.gpio import gpio_init, gpio_set
  from openpilot.system.hardware.tici.pins import GPIO
  from openpilot.system.qcomgpsd.qcomgpsd import at_cmd

  try:
    check_output(["pidof", "qcomgpsd"])
    print("qcomgpsd is running, please kill openpilot before running this script! (aborted)")
    sys.exit(1)
  except CalledProcessError as e:
    if e.returncode != 1: # 1 == no process found (boardd not running)
      raise e

  print("power up antenna ...")
  gpio_init(GPIO.GNSS_PWR_EN, True)
  gpio_set(GPIO.GNSS_PWR_EN, True)

  if b"+QGPS: 0" not in (at_cmd("AT+QGPS?") or b""):
    print("stop location tracking ...")
    at_cmd("AT+QGPSEND")

  if b'+QGPSCFG: "outport",usbnmea' not in (at_cmd('AT+QGPSCFG="outport"') or b""):
    print("configure outport ...")
    at_cmd('AT+QGPSCFG="outport","usbnmea"') # usbnmea = /dev/ttyUSB1

  if b'+QGPSCFG: "gnssrawdata",3,0' not in (at_cmd('AT+QGPSCFG="gnssrawdata"') or b""):
    print("configure gnssrawdata ...")
    # AT+QGPSCFG="gnssrawdata",<constellation-mask>,<port>'
    # <constellation-mask> values:
    # 0x01 = GPS
    # 0x02 = GLONASS
    # 0x04 = BEIDOU
    # 0x08 = GALILEO
    # 0x10 = QZSS
    # <port> values:
    # 0 = NMEA port
    # 1 = AT port
    at_cmd('AT+QGPSCFG="gnssrawdata",3,0') # enable all constellations, output data to NMEA port
    print("rebooting ...")
    at_cmd('AT+CFUN=1,1')
    print("re-run this script when it is back up")
    sys.exit(2)

  print("starting location tracking ...")
  at_cmd("AT+QGPS=1")

  process_nmea_port_messages()

if __name__ == "__main__":
  main()
