import os
import subprocess

from openpilot.common.basedir import BASEDIR
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase


NATIVE_TESTS = (
  "openpilot/common/tests/test_common",
  "openpilot/selfdrive/pandad/tests/test_pandad_canprotocol",
  "openpilot/system/loggerd/tests/test_logger",
  "openpilot/tools/cabana/tests/test_cabana",
  "openpilot/tools/cabana/tests/test_dbc_core",
  "openpilot/tools/replay/tests/test_replay",
)


class TestNative(OpenpilotTestCase):
  @parameterized.expand(NATIVE_TESTS)
  def test_native(self, executable):
    path = os.path.join(BASEDIR, executable)
    if not os.path.exists(path):
      self.skipTest(f"optional native test was not built: {executable}")
    subprocess.run([path], check=True)
