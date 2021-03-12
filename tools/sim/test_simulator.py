#!/usr/bin/env python3
import os
# import signal
# import subprocess
# import time
import unittest

from common.basedir import BASEDIR
# from common.timeout import Timeout

class TestSimulator(unittest.TestCase):

  def test_install(self):
    exit_code = os.system(os.path.join(BASEDIR, "tools/sim/install_carla.sh"))
    self.assertEqual(exit_code, 0)

  # def test_run(self):

  #   # p_server = subprocess.Popen(f'{os.path.join(BASEDIR, "tools/sim/start_carla.sh")}', stderr=subprocess.PIPE, shell=True, start_new_session=True)
  #   p_openpilot = subprocess.Popen(f'{os.path.join(BASEDIR, "tools/sim/start_openpilot_docker.sh")}')

  #   with Timeout(60):
  #     while 1:
  #       out = p_server.stderr.readline().decode("utf-8")
  #       if out != '':
  #         print(out)


  #   os.killpg(os.getpgid(p_server.pid), signal.SIGTERM)

if __name__ == "__main__":
  unittest.main()
