import os
import glob
import signal
import subprocess
import time

from openpilot.common.basedir import BASEDIR
from openpilot.common.timeout import Timeout
from openpilot.tools.plotjuggler.juggle import DEMO_ROUTE, install

PJ_DIR = os.path.join(BASEDIR, "tools/plotjuggler")

class TestPlotJuggler:

  def test_demo(self):
    install()

    pj = os.path.join(PJ_DIR, "juggle.py")
    with subprocess.Popen(f'QT_QPA_PLATFORM=offscreen {pj} "{DEMO_ROUTE}/:2"',
                           stderr=subprocess.PIPE, shell=True, start_new_session=True) as p:
      # Wait for "Done reading Rlog data" signal from the plugin
      output = "\n"
      with Timeout(180, error_msg=output):
        while output.splitlines()[-1] != "Done reading Rlog data":
          output += p.stderr.readline().decode("utf-8")

      # ensure plotjuggler didn't crash after exiting the plugin
      time.sleep(2)
      assert p.poll() is None
      os.killpg(os.getpgid(p.pid), signal.SIGTERM)

      assert "Raw file read failed" not in output

  # TODO: also test that layouts successfully load
  def test_layouts(self, subtests):
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
