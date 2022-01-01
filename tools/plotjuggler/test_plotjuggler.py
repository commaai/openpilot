#!/usr/bin/env python3
import os
import signal
import subprocess
import time
import unittest

from common.basedir import BASEDIR
from common.timeout import Timeout
from tools.plotjuggler.juggle import install

class TestPlotJuggler(unittest.TestCase):

  def test_demo(self):
    install()

    pj = os.path.join(BASEDIR, "tools/plotjuggler/juggle.py")
    p = subprocess.Popen(f'QT_QPA_PLATFORM=offscreen {pj} --demo None 1 --qlog',
                         stderr=subprocess.PIPE, shell=True, start_new_session=True)

    # Wait for "Done reading Rlog data" signal from the plugin
    output = "\n"
    with Timeout(180, error_msg=output):
      while output.splitlines()[-1] != "Done reading Rlog data":
        output += p.stderr.readline().decode("utf-8")

    # ensure plotjuggler didn't crash after exiting the plugin
    time.sleep(15)
    self.assertEqual(p.poll(), None)
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)

if __name__ == "__main__":
  unittest.main()
