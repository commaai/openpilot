#!/usr/bin/env python3
import os
import time
import unittest

import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes


def gen_llk():
  msg = messaging.new_message('liveLocationKalman')
  msg.liveLocationKalman.positionGeodetic = {'value': [32.7174, -117.16277, 0], 'std': [0., 0., 0.], 'valid': True}
  msg.liveLocationKalman.calibratedOrientationNED = {'value': [0., 0., 0.], 'std': [0., 0., 0.], 'valid': True}
  msg.liveLocationKalman.status = 'valid'
  return msg


class TestMapRenderer(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ['MAP_RENDER_TEST_MODE'] = '1'
    cls.pm = messaging.PubMaster(['liveLocationKalman'])

  def tearDown(self):
    managed_processes['mapsd'].stop()

  # send LLK
  def test_frequency(self):
    managed_processes['mapsd'].start()
    time.sleep(1)

    # get everything connected
