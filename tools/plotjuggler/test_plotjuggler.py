#!/usr/bin/env python3
import os
import random
import subprocess
import time
import unittest

from common.basedir import BASEDIR
from selfdrive.test.openpilotci import get_url
from selfdrive.test.test_car_models import routes

class TestPlotJuggler(unittest.TestCase):

  def test_install(self):
    exit_code = os.system(os.path.join(BASEDIR, "tools/plotjuggler/install.sh"))
    self.assertEqual(exit_code, 0)

  def test_run(self):

    test_url = get_url(random.choice(list(routes.keys())), 0)

    p = subprocess.Popen(f'{os.path.join(BASEDIR, "tools/plotjuggler/juggle.py")} "{test_url}"', stderr=subprocess.PIPE, shell=True)

    exit_code = 1
    state = "parsing"

    while True:
      if state == "parsing":
        output = p.stderr.readline()
        if output == b'Done\n':
          state = "waiting"
          start = time.time()
        continue

      if time.time() - start >= 15.0:
        exit_code = 0
        break

    p.kill()
    self.assertEqual(exit_code, 0)

if __name__ == "__main__":
  unittest.main()
