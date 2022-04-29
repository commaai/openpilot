#!/usr/bin/env python3
import unittest

import numpy as np

from cereal import messaging
from laika import AstroDog
from laika.raw_gnss import read_raw_ublox
from selfdrive.locationd.laikad import process_ublox_msg


class TestLaikad(unittest.TestCase):
  def test_ublox_processing(self):
    # todo verify more than just no errors
    # todo remove usage ublox_gnss_msgs_demo_segment and use logreader
    dog = AstroDog()
    pm = messaging.PubMaster(['gnssMeasurements'])

    path = 'ublox_gnss_msgs_demo_segment'  # 60 seconds segment
    raw_ublox, raw_ublox_t = np.load(path, allow_pickle=True).values()

    for i, msg in enumerate(raw_ublox):
      if msg.which == 'measurementReport':
        report = msg.measurementReport
        if len(report.measurements) > 0:
          read_raw_ublox(report)
      process_ublox_msg(msg, dog, pm, raw_ublox_t[i])


if __name__ == "__main__":
  unittest.main()
