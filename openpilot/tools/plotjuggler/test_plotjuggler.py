import os
import glob
import shutil
import signal
import subprocess
import sys
import time
import unittest

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.basedir import BASEDIR
from openpilot.common.timeout import Timeout
from openpilot.tools.plotjuggler.juggle import DEMO_ROUTE, install

PJ_DIR = os.path.join(BASEDIR, "openpilot/tools/plotjuggler")


class TestPlotJuggler(OpenpilotTestCase):

  def _run_plotjuggler(self, args=""):
    pj = os.path.join(PJ_DIR, "juggle.py")
    cmd = f'QT_QPA_PLATFORM=offscreen {sys.executable} {pj} "{DEMO_ROUTE}/:2" {args}'
    with subprocess.Popen(cmd, stderr=subprocess.PIPE, shell=True, start_new_session=True) as p:
      # Wait for "Done reading Rlog data" signal from the plugin
      output = "\n"
      with Timeout(180, error_msg=output):
        while output.splitlines()[-1] != "Done reading Rlog data":
          line = p.stderr.readline().decode("utf-8")
          if not line and p.poll() is not None:
            break
          output += line

      # ensure plotjuggler didn't crash after exiting the plugin
      time.sleep(2)
      assert p.poll() is None, f"PlotJuggler crashed:\n{output}"
      os.killpg(os.getpgid(p.pid), signal.SIGTERM)

      assert "Raw file read failed" not in output

  @unittest.skipIf(not shutil.which('qmake'), "Qt not installed")
  def test_demo(self):
    install()
    self._run_plotjuggler()

  def test_layouts(self, subtests):
    has_qt = bool(shutil.which('qmake'))
    if has_qt:
      install()

    bad_strings = (
      # if a previously loaded file is defined,
      # PJ will throw a warning when loading the layout
      "fileInfo",
      "previouslyLoaded_Datafiles",
    )
    for fn in glob.glob(os.path.join(PJ_DIR, "layouts/*")):
      name = os.path.basename(fn)
      with subtests.test(layout=name):
        with open(fn) as f:
          layout = f.read()
          violations = [s for s in bad_strings if s in layout]
          assert len(violations) == 0, f"These should be stripped out of the layout: {str(violations)}"

        if has_qt:
          self._run_plotjuggler(f'--layout "{fn}"')
