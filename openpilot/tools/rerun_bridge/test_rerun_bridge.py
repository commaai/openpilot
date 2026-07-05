#!/usr/bin/env python3

import os
import unittest

import rerun as rr

from openpilot.tools.rerun_bridge.extract import EventExtractor, finalize_series
from openpilot.tools.rerun_bridge.ingest import DEMO_ROUTE, ingest_route
from openpilot.tools.lib.logreader import LogReader, ReadMode


class RerunBridgeTest(unittest.TestCase):
  def test_extract_demo_route(self):
    lr = LogReader(DEMO_ROUTE, default_mode=ReadMode.QLOG)
    store = EventExtractor().process_events(list(lr))
    series = finalize_series(store)
    self.assertIn("/carState/vEgo", series)
    self.assertGreater(len(series["/carState/vEgo"][0]), 1000)

  def test_smoke_ingest(self):
    out = "/tmp/rerun_bridge_smoke_test.rrd"
    rr.init("openpilot", recording_id="rerun_bridge_smoke", spawn=False)
    rr.save(out)
    stats = ingest_route(
      rr,
      DEMO_ROUTE,
      layout_name="tuning",
      include_video=False,
      include_can=False,
    )
    self.assertGreater(stats.messages, 100000)
    self.assertGreater(stats.series_paths, 500)
    self.assertGreater(stats.gps_points, 100)
    self.assertTrue(os.path.exists(out))
    self.assertGreater(os.path.getsize(out), 100_000)


if __name__ == "__main__":
  unittest.main()