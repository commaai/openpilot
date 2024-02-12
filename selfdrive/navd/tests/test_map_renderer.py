#!/usr/bin/env python3
import time
import numpy as np
import os
import pytest
import unittest
import requests
import threading
import http.server
import cereal.messaging as messaging

from typing import Any
from cereal.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.mock.generators import LLK_DECIMATION, LOCATION1, LOCATION2, generate_liveLocationKalman
from openpilot.selfdrive.test.helpers import with_processes

CACHE_PATH = "/data/mbgl-cache-navd.db"

RENDER_FRAMES = 15
DEFAULT_ITERATIONS = RENDER_FRAMES * LLK_DECIMATION
LOCATION1_REPEATED = [LOCATION1] * DEFAULT_ITERATIONS
LOCATION2_REPEATED = [LOCATION2] * DEFAULT_ITERATIONS


class MapBoxInternetDisabledRequestHandler(http.server.BaseHTTPRequestHandler):
  INTERNET_ACTIVE = True

  def do_GET(self):
    if not self.INTERNET_ACTIVE:
      self.send_response(500)
      self.end_headers()
      return

    url = f'https://api.mapbox.com{self.path}'

    headers = dict(self.headers)
    headers["Host"] = "api.mapbox.com"

    r = requests.get(url, headers=headers, timeout=5)

    self.send_response(r.status_code)
    self.end_headers()
    self.wfile.write(r.content)

  def log_message(self, *args: Any) -> None:
    return

  def log_error(self, *args: Any) -> None:
    return


class MapBoxInternetDisabledServer(threading.Thread):
  def run(self):
    self.server = http.server.HTTPServer(("127.0.0.1", 0), MapBoxInternetDisabledRequestHandler)
    self.port = self.server.server_port
    self.server.serve_forever()

  def stop(self):
    self.server.shutdown()

  def disable_internet(self):
    MapBoxInternetDisabledRequestHandler.INTERNET_ACTIVE = False

  def enable_internet(self):
    MapBoxInternetDisabledRequestHandler.INTERNET_ACTIVE = True


class TestMapRenderer(unittest.TestCase):
  server: MapBoxInternetDisabledServer

  @classmethod
  def setUpClass(cls):
    assert "MAPBOX_TOKEN" in os.environ
    cls.original_token = os.environ["MAPBOX_TOKEN"]
    cls.server = MapBoxInternetDisabledServer()
    cls.server.start()
    time.sleep(0.5) # wait for server to startup

  @classmethod
  def tearDownClass(cls) -> None:
    cls.server.stop()

  def setUp(self):
    self.server.enable_internet()
    os.environ['MAPS_HOST'] = f'http://localhost:{self.server.port}'

    self.sm = messaging.SubMaster(['mapRenderState'])
    self.pm = messaging.PubMaster(['liveLocationKalman'])
    self.vipc = VisionIpcClient("navd", VisionStreamType.VISION_STREAM_MAP, True)

    if os.path.exists(CACHE_PATH):
      os.remove(CACHE_PATH)

  def _setup_test(self):
    assert self.pm.wait_for_readers_to_update("liveLocationKalman", 10)

    time.sleep(0.5)

    assert VisionIpcClient.available_streams("navd", False) == {VisionStreamType.VISION_STREAM_MAP, }
    assert self.vipc.connect(False)
    self.vipc.recv()

  def _run_test(self, expect_valid, locations=LOCATION1_REPEATED):
    starting_frame_id = None

    render_times = []

    # run test
    prev_frame_id = -1
    for i, location in enumerate(locations):
      frame_expected = (i+1) % LLK_DECIMATION == 0

      if self.sm.logMonoTime['mapRenderState'] == 0:
        prev_valid = False
        prev_frame_id = -1
      else:
        prev_valid = self.sm.valid['mapRenderState']
        prev_frame_id = self.sm['mapRenderState'].frameId

      if starting_frame_id is None:
        starting_frame_id = prev_frame_id

      llk = generate_liveLocationKalman(location)
      self.pm.send("liveLocationKalman", llk)
      self.pm.wait_for_readers_to_update("liveLocationKalman", 10)
      self.sm.update(1000 if frame_expected else 0)
      assert self.sm.updated['mapRenderState'] == frame_expected, "renderer running at wrong frequency"

      if not frame_expected:
        continue

      frames_since_test_start = self.sm['mapRenderState'].frameId - starting_frame_id

      # give a few frames to switch from valid to invalid, or vice versa
      invalid_and_not_previously_valid = (expect_valid and not self.sm.valid['mapRenderState'] and not prev_valid)
      valid_and_not_previously_invalid = (not expect_valid and self.sm.valid['mapRenderState'] and prev_valid)

      if (invalid_and_not_previously_valid or valid_and_not_previously_invalid) and frames_since_test_start < 5:
        continue

      # check output
      assert self.sm.valid['mapRenderState'] == expect_valid
      assert self.sm['mapRenderState'].frameId == (prev_frame_id + 1)
      assert self.sm['mapRenderState'].locationMonoTime == llk.logMonoTime
      if not expect_valid:
        assert self.sm['mapRenderState'].renderTime == 0.
      else:
        assert 0. < self.sm['mapRenderState'].renderTime < 0.1
        render_times.append(self.sm['mapRenderState'].renderTime)

      # check vision ipc output
      assert self.vipc.recv() is not None
      assert self.vipc.valid == expect_valid
      assert self.vipc.timestamp_sof == llk.logMonoTime
      assert self.vipc.frame_id == self.sm['mapRenderState'].frameId

    assert frames_since_test_start >= RENDER_FRAMES

    return render_times

  @with_processes(["mapsd"])
  def test_with_internet(self):
    self._setup_test()
    self._run_test(True)

  @with_processes(["mapsd"])
  def test_with_no_internet(self):
    self.server.disable_internet()
    self._setup_test()
    self._run_test(False)

  @with_processes(["mapsd"])
  @pytest.mark.skip(reason="slow, flaky, and unlikely to break")
  def test_recover_from_no_internet(self):
    self._setup_test()
    self._run_test(True)

    self.server.disable_internet()

    # change locations to force mapsd to refetch
    self._run_test(False, LOCATION2_REPEATED)

    self.server.enable_internet()
    self._run_test(True, LOCATION2_REPEATED)

  @with_processes(["mapsd"])
  @pytest.mark.tici
  def test_render_time_distribution(self):
    self._setup_test()
    # from location1 -> location2 and back
    locations = np.array([*np.linspace(LOCATION1, LOCATION2, 2000), *np.linspace(LOCATION2, LOCATION1, 2000)]).tolist()

    render_times = self._run_test(True, locations)

    _min = np.min(render_times)
    _max = np.max(render_times)
    _mean = np.mean(render_times)
    _median = np.median(render_times)
    _stddev = np.std(render_times)

    print(f"Stats: min: {_min}, max: {_max}, mean: {_mean}, median: {_median}, stddev: {_stddev}, count: {len(render_times)}")

    def assert_stat(stat, nominal, tol=0.3):
      tol = (nominal / (1+tol)), (nominal * (1+tol))
      self.assertTrue(tol[0] < stat < tol[1], f"{stat} not in tolerance {tol}")

    assert_stat(_mean,   0.030)
    assert_stat(_median, 0.027)
    assert_stat(_stddev, 0.0078)

    self.assertLess(_max, 0.065)
    self.assertGreater(_min, 0.015)


if __name__ == "__main__":
  unittest.main()
