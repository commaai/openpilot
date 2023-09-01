import os
from dataclasses import dataclass, fields
from time import sleep
from typing import NoReturn

DEBUG = int(os.environ.get("DEBUG", "0"))

@dataclass
class GnssClock:
  flags: int
  leap_seconds: int
  time_ns: int
  time_uncertainty_ns: int
  full_bias_ns: int
  bias_ns: float
  bias_uncertainty_ns: float
  drift_nsps: float
  drift_uncertainty_nsps: float

  def __post_init__(self):
    for field in fields(self):
      val = getattr(self, field.name)
      setattr(self, field.name, field.type(val) if val else None)

@dataclass
class GnssMeas:
  messageCount: int
  messageNum: int
  svCount: int
  constellation: int
  svId: int
  flags: int
  time_offset_ns: int
  state: int
  time_of_week_ns: int
  time_of_week_uncertainty_ns: int
  carrier_to_noise_dbhz: float
  pseudorange_rate: float
  pseudorange_rate_uncertainty: float

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

def main(device:str="/dev/ttyUSB1") -> NoReturn:
  while True:
    try:
      with open(device, "r") as nmeaport:
        for line in nmeaport:
          line = line.strip()
          if DEBUG: print(line)
          if not nmea_checksum_ok(line):
            continue

          fields = line.split(",")
          match fields[0]:
            case "$GNCLK":
              # fields at end are reserved (not used)
              gnss_clock = GnssClock(*fields[1:10])
              print(gnss_clock)
            case "$GNMEAS":
              # fields at end are reserved (not used)
              gnss_meas = GnssMeas(*fields[1:14])
              print(gnss_meas)
    except Exception as e:
      print(e)
      sleep(1)

if __name__ == "__main__":
  main()
