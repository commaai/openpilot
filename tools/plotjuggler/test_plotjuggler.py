#!/usr/bin/env python3
import os
# import random
import signal
import subprocess
import time
import unittest

from common.basedir import BASEDIR
from common.timeout import Timeout
# from selfdrive.test.openpilotci import get_url
# from selfdrive.test.test_car_models import routes

class TestPlotJuggler(unittest.TestCase):

  def test_install(self):
    os.system(f'cd {os.path.join(BASEDIR, "tools/plotjuggler/install.sh")}')
    exit_code = os.system(os.path.join(BASEDIR, "tools/plotjuggler/install.sh"))
    self.assertEqual(exit_code, 0)

  def test_run(self):

    #test_url = get_url(random.choice(list(routes.keys())), 0)
    test_url = "https://commadataci.blob.core.windows.net/openpilotci/ffccc77938ddbc44/2021-01-04--16-55-41/0/rlog.bz2"

    p = subprocess.Popen(f'QT_QPA_PLATFORM=offscreen {os.path.join(BASEDIR, "tools/plotjuggler/juggle.py")} \
    "{test_url}" --bin_path={os.path.join(BASEDIR, "tools/plotjuggler/bin")} ', stderr=subprocess.PIPE ,shell=True,
    start_new_session=True)

    with Timeout(60):
      while True:
          output = p.stderr.readline()
          if output == b'Done\n':
            break

    time.sleep(15)
    self.assertEqual(p.poll(), None)
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)

if __name__ == "__main__":
  unittest.main()
