#!/usr/bin/env python3
import os
import numpy as np

import gmplot

from laikad import Laikad
from tools.lib.logreader import LogReader
from tools.lib.route import Route
from common.transformations.coordinates import ecef2geodetic

GMPLOT_FILE = "/tmp/gm_laikad.html"


def run_laikad(raw_gnss, ublox_available):

  use_internet = True #"LAIKAD_NO_INTERNET" not in os.environ
  laikad = Laikad(save_ephemeris=True, auto_fetch_orbits=use_internet, use_qcom=not ublox_available)

  # store produced messages from laikad
  p_msgs = []
  e_msgs = []

  for msg in raw_gnss:
    gnss_msg = getattr(msg, "ubloxGnss" if ublox_available else "qcomGnss")

    # TODO: Understand and use remaining unknown constellations
    if gnss_msg.which() == "drMeasurementReport":
      if getattr(gnss_msg, gnss_msg.which()).source not in ['glonass', 'gps', 'beidou', 'sbas']:
        continue

      if getattr(gnss_msg, gnss_msg.which()).gpsWeek > np.iinfo(np.int16).max:
        # gpsWeek 65535 is received rarely from quectel, this cannot be
        # passed to GnssMeasurements's gpsWeek (Int16)
        continue

      tmp = laikad.process_gnss_msg(gnss_msg, msg.logMonoTime, block=False)
      if tmp is None:
        print("result is None")
        continue

      m = tmp
      if m is not None:
        p_msgs.append(m)

  return p_msgs, e_msgs


def prep_and_call_laikad():

  # TODO: improve for running on miniray
  route_name = "018654717bc93d7d|2022-11-16--17-12-30"
  #route_name = "018654717bc93d7d|2022-11-15--19-11-44"
  r = Route(route_name)

  ublox_available = False
  raw_ublox_messages = []
  raw_qcom_messages = []

  for i, seg in enumerate(r.segments[5:15]):
    print(f"segment processing: {seg}")

    print(f"processing: {i}/{len(r.segments)}")
    if seg.log_path is None:
        print("WARNING: segment path NONE")
        continue

    lr = LogReader(seg.log_path)
    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
    for msg in all_msgs:

      if msg.which() == "qcomGnss":
        raw_qcom_messages.append(msg)

      if msg.which() == "ubloxGnss":
        raw_ublox_messages.append(msg)

    if len(raw_ublox_messages) != 0:
      ublox_available = True

  print(f"{ublox_available} {len(raw_qcom_messages)}")

  # run laikad
  if ublox_available:
    p_msgs, e_info = run_laikad(raw_ublox_messages, ublox_available)
  else:
    p_msgs, e_info = run_laikad(raw_qcom_messages, ublox_available)

  return p_msgs, e_info

# make public for ipython analysis
p_msgs = []
e_info = []
geo_pos = []
def main():
  global p_msgs, geo_pos, e_info

  p_msgs, e_info = prep_and_call_laikad()

  # post process results
  for r in p_msgs:
    ecef_pos = list(r.gnssMeasurements.positionECEF.value)
    geo_pos.append(ecef2geodetic(ecef_pos))

  # draw map for debugging purposes (office start location)
  gmap = gmplot.GoogleMapPlotter(32.7841634, -117.128145, 15)
  for gp in geo_pos:
    gmap.marker(gp[0], gp[1], color='red')

  gmap.draw(f"{GMPLOT_FILE}")
  print(f"Results: file://{GMPLOT_FILE}")

if __name__ == "__main__":
  main()
