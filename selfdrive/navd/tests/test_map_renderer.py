#!/usr/bin/env python3
import os
import unittest

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


class TestMapRenderer(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    assert "MAPBOX_TOKEN" in os.environ

  def setUp(self):
    self.sm = messaging.SubMaster(['mapRenderState'])
    self.pm = messaging.PubMaster(['liveLocationKalman'])
    self.vipc = VisionIpcClient("navd", VisionStreamType.VISION_STREAM_MAP, True)

    if os.path.exists(CACHE_PATH):
      os.remove(CACHE_PATH)

  def tearDown(self):
    managed_processes['mapsd'].stop()

  def _run_test(self, expect_valid):
    # start + sync up
    managed_processes['mapsd'].start()
    assert self.pm.wait_for_readers_to_update("liveLocationKalman", 10)

    assert VisionIpcClient.available_streams("navd", False) == {VisionStreamType.VISION_STREAM_MAP, }
    assert self.vipc.connect(False)
    self.vipc.recv()

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

  def test_with_internet(self):
    self._run_test(True)

  def test_with_no_internet(self):
    token = os.environ['MAPBOX_TOKEN']
    try:
      os.environ['MAPBOX_TOKEN'] = 'invalid_token'
      self._run_test(False)
    finally:
      os.environ['MAPBOX_TOKEN'] = token

if __name__ == "__main__":
  unittest.main()
