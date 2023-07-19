#!/usr/bin/env python3
import os
import unittest
import time

import cereal.messaging as messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType
from selfdrive.manager.process_config import managed_processes

LLK_DECIMATION = 10
CACHE_PATH = "/data/mbgl-cache-navd.db"

def gen_llk():
  msg = messaging.new_message('liveLocationKalman')
  msg.liveLocationKalman.positionGeodetic = {'value': [32.7174, -117.16277, 0], 'std': [0., 0., 0.], 'valid': True}
  msg.liveLocationKalman.calibratedOrientationNED = {'value': [0., 0., 0.], 'std': [0., 0., 0.], 'valid': True}
  msg.liveLocationKalman.status = 'valid'
  return msg

import requests
import threading
import http.server

class MapBoxInternetDisabledRequestHandler(http.server.BaseHTTPRequestHandler):
  INTERNET_ACTIVE = True

  def do_GET(self):
    url = f'https://api.mapbox.com{self.path}'

    print(url, self.INTERNET_ACTIVE)

    if self.INTERNET_ACTIVE:
      headers = dict(self.headers)
      headers["Host"] = "api.mapbox.com"
      
      r = requests.get(url, headers=headers, timeout=5)

      if r.status_code != 200:
        print(r.content)
        print(r.headers)
        print(headers)

      self.send_response(r.status_code)
      for key in r.headers:
        self.send_header(key, r.headers[key])

      self.end_headers()
      self.wfile.write(r.content)

    else:
      self.send_response(404, "404 page not found (internet disabled)")


class MapBoxInternetDisabledServer(threading.Thread):
  def run(self):
    self.server = http.server.HTTPServer(("127.0.0.1", 5000), MapBoxInternetDisabledRequestHandler)
    self.server.serve_forever()
  
  def stop(self):
    self.server.shutdown()


class TestMapRenderer(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    assert "MAPBOX_TOKEN" in os.environ

    cls.server = MapBoxInternetDisabledServer()
    cls.server.start()
    time.sleep(1) # wait for server to be setup
  
  @classmethod
  def tearDownClass(cls) -> None:
    cls.server.stop()

  def setUp(self):
    self.sm = messaging.SubMaster(['mapRenderState'])
    self.pm = messaging.PubMaster(['liveLocationKalman'])
    self.vipc = VisionIpcClient("navd", VisionStreamType.VISION_STREAM_MAP, True)

    if os.path.exists(CACHE_PATH):
      os.remove(CACHE_PATH)
  
  def tearDown(self):
    managed_processes['mapsd'].stop()
  
  def _setup_test(self):
    # start + sync up
    os.environ['MAPS_HOST'] = 'http://localhost:5000'

    managed_processes['mapsd'].start()

    assert self.pm.wait_for_readers_to_update("liveLocationKalman", 10)

    assert VisionIpcClient.available_streams("navd", False) == {VisionStreamType.VISION_STREAM_MAP, }
    assert self.vipc.connect(False)
    self.vipc.recv()
  
  def _run_test(self, expect_valid):
    self._setup_test()
    self.__run_test(expect_valid)

  def __run_test(self, expect_valid):
    # run test
    prev_frame_id = -1
    for i in range(30*LLK_DECIMATION):
      frame_expected = (i+1) % LLK_DECIMATION == 0

      if self.sm.logMonoTime['mapRenderState'] == 0:
        prev_valid = False
        prev_frame_id = -1
      else:
        prev_frame_id = self.sm['mapRenderState'].frameId
        prev_valid = self.sm.valid['mapRenderState']

      llk = gen_llk()
      self.pm.send("liveLocationKalman", llk)
      self.pm.wait_for_readers_to_update("liveLocationKalman", 10)
      self.sm.update(1000 if frame_expected else 0)
      assert self.sm.updated['mapRenderState'] == frame_expected, "renderer running at wrong frequency"

      if not frame_expected:

        continue

      # give a few frames to go valid
      if expect_valid and not self.sm.valid['mapRenderState'] and not prev_valid and self.sm['mapRenderState'].frameId < 5:
        continue

      # check output
      assert self.sm.valid['mapRenderState'] == expect_valid
      assert self.sm['mapRenderState'].frameId == (prev_frame_id + 1)
      assert self.sm['mapRenderState'].locationMonoTime == llk.logMonoTime
      if not expect_valid:
        assert self.sm['mapRenderState'].renderTime == 0.
      else:
        assert 0. < self.sm['mapRenderState'].renderTime < 0.1

      # check vision ipc output
      assert self.vipc.recv() is not None
      assert self.vipc.valid == expect_valid
      assert self.vipc.timestamp_sof == llk.logMonoTime
      assert self.vipc.frame_id == self.sm['mapRenderState'].frameId
  
  def disable_internet(self):
    MapBoxInternetDisabledRequestHandler.INTERNET_ACTIVE = False
  
  def enable_internet(self):
    MapBoxInternetDisabledRequestHandler.INTERNET_ACTIVE = True

  def test_with_internet(self):
    self.enable_internet()
    self._run_test(True)

  def test_with_no_internet(self):
    try:
      self.disable_internet()
      self._run_test(False)
    finally:
      self.enable_internet()
    
  def test_recover_from_no_internet(self):
    self.enable_internet()
    self._setup_test()

    self.__run_test(True)

    self.disable_internet()
    time.sleep(2)
    self.__run_test(False)

    self.enable_internet()
    time.sleep(2)
    self.__run_test(True)

if __name__ == "__main__":
  unittest.main()
