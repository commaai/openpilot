#!/usr/bin/env python3
import unittest

import numpy as np
from cereal import messaging

from laika import AstroDog
from selfdrive.locationd.laikad import process_ublox_msg


class TestLaikad(unittest.TestCase):
  def test_ublox_processing(self):
    dog = AstroDog()

    path = 'ublox_gnss_msgs_demo_segment'
    raw_ublox = np.load(path, allow_pickle=True)

    pm = messaging.PubMaster(['GnssMeasurements'])

    for msg in raw_ublox:
      process_ublox_msg(msg, dog, pm)

if __name__ == "__main__":
  unittest.main()
# struct gnssMeasurements {
#   # Lat,long and alt for debugging purposes.
#   # Longitude and latitude in degrees.
#   latitude @1 :Float64;
#   longitude @2 :Float64;
#   # In meters above the WGS 84 reference ellipsoid.
#   altitude @3 :Float64;
#
#
#
# }
