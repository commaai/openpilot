import json
import os
import subprocess
import unittest

import zmq

from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware.hw import Paths


class TestSwaglog(unittest.TestCase):
  def test_cpp_log_reaches_swaglog_socket(self):
    executable = os.path.join(BASEDIR, "openpilot/common/tests/swaglog_test")
    if not os.path.exists(executable):
      self.skipTest("optional native swaglog fixture was not built")

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.bind(Paths.swaglog_ipc())
    try:
      env = os.environ.copy()
      env.update(MANAGER_DAEMON="swaglog_test", DONGLE_ID="test_dongle_id", CLEAN="1")
      subprocess.run([executable], check=True, env=env)

      raw = socket.recv()
      self.assertEqual(raw[0], 10)  # CLOUDLOG_DEBUG
      message = json.loads(raw[1:])
      self.assertEqual(message["levelnum"], 10)
      self.assertEqual(message["msg"], "python-e2e-cpp-log")
      self.assertEqual(message["funcname"], "main")
      self.assertTrue(message["filename"].endswith("swaglog_test.cc"))
      self.assertEqual(message["ctx"]["daemon"], "swaglog_test")
      self.assertEqual(message["ctx"]["dongle_id"], "test_dongle_id")
      self.assertFalse(message["ctx"]["dirty"])
    finally:
      socket.close()
      context.term()
