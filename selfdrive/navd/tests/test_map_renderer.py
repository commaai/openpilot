#!/usr/bin/env python3
import os
import unittest

import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes

LLK_DECIMATION = 10


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

  def tearDown(self):
    managed_processes['mapsd'].stop()

  def _run_test(self, valid):
    # start + sync up
    managed_processes['mapsd'].start()
    for _ in range(100):
      self.pm.send("liveLocationKalman", gen_llk())
      self.sm.update(100)
      if self.sm.updated['mapRenderState']:
        break
    assert self.sm.updated['mapRenderState'], "renderer didn't start"

    # run test
    for i in range(20*LLK_DECIMATION):
      prev_frame_id = self.sm['mapRenderState'].frameId

      llk = gen_llk()
      self.pm.send("liveLocationKalman", llk)
      self.sm.update(200)

      if (i+1) % LLK_DECIMATION != 0:
        assert not self.sm.updated['mapRenderState'], "renderer running at wrong frequency"
        continue

      assert self.sm.updated['mapRenderState']

      # check output
      assert self.sm.valid['mapRenderState'] == valid
      assert 0. < self.sm['mapRenderState'].renderTime < 0.1
      assert self.sm['mapRenderState'].frameId == (prev_frame_id + 1)
      assert self.sm['mapRenderState'].locationMonoTime == llk.logMonoTime

  def test_with_internet(self):
    self._run_test(True)

  def test_no_internet(self):
    token = os.environ['MAPBOX_TOKEN']
    try:
      os.environ['MAPBOX_TOKEN'] = "notatoken"
      self._run_test(False)
    finally:
      os.environ['MAPBOX_TOKEN'] = token


if __name__ == "__main__":
  unittest.main()
