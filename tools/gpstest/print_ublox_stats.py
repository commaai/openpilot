import time
from datetime import datetime
from collections import defaultdict
import cereal.messaging as messaging

MAX_SATS = 150

def print_stats(start_time, ephem_stats, sat_meas, meas_cnt):
  print(f"----- {datetime.now().strftime('%H:%M:%S.%f')}, Runtime: {datetime.now() - start_time}")
  for sat in range(len(ephem_stats)):
    if ephem_stats[sat] != 0:
      print(f"{sat}: {round(ephem_stats[sat]/meas_cnt, 2)} ephem/min")

  print("Measurements (in last minute):")
  for sat in range(len(sat_meas)):
    if sat_meas[sat] != 0:
      print(f"{sat}: {sat_meas[sat]}")


def print_data(ephem_data):
  print(f"----- {datetime.now().strftime('%H:%M:%S.%f')}")
  for sat in ephem_data:
    print(f"{sat}: {ephem_data[sat]}")


def main():
  ephem_stats = [0]*MAX_SATS
  sat_meas = [0]*MAX_SATS
  meas_cnt = 0

  # recv of ephemeris per minute
  ephem_data = defaultdict(int)

  sm = messaging.SubMaster(["ubloxGnss", "ubloxRaw"])
  last_log = time.monotonic()
  last_log_10s = time.monotonic()
  start_time = datetime.now()

  total_ephems_per_min = []
  diff_ephems_per_min = []

  # TODO: add locktime check, to alert for when we are expecting a ephemeris
  #       but dont get one

  while True:
    sm.update()

    #if sm.updated["ubloxRaw"]:
    #  continue

    if not sm.updated["ubloxGnss"]:
      continue

    # check ubluxGnss message
    msg = sm['ubloxGnss']

    if msg.which() == "measurementReport":
      m = time.monotonic()
      # add satellite count numbers
      for m in msg.measurementReport.measurements:
        if m.gnssId == 6 and m.svId == 255:
          continue
        if m.gnssId == 6:
          sat_meas[m.svId+100] += 1
        else:
          sat_meas[m.svId] += 1

    if msg.which() == "ephemeris":
      ephem_data[msg.ephemeris.svId] += 1

    if (time.monotonic() - last_log_10s) > 10:
      print_data(ephem_data)
      last_log_10s = time.monotonic()

    if (time.monotonic() - last_log) > 60:
      for sat in ephem_data:
        ephem_stats[sat] += ephem_data[sat]
      meas_cnt += 1
      diff_ephems_per_min.append(len(ephem_data))
      total_ephems_per_min.append(sum(n for n in ephem_data.values()))

      ephem_data = defaultdict(int)
      print_stats(start_time, ephem_stats, sat_meas, meas_cnt)
      print(f"Total     e/min: {sum(total_ephems_per_min)/len(total_ephems_per_min)}")
      print(f"Different e/min: {sum(diff_ephems_per_min)/len(diff_ephems_per_min)}")
      sat_meas = [0]*MAX_SATS
      last_log = time.monotonic()

if __name__ == "__main__":
  main()

