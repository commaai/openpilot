import os
import subprocess

from openpilot.common.basedir import BASEDIR
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase


NATIVE_TESTS = (
  "openpilot/common/tests/test_swaglog",
  "openpilot/selfdrive/pandad/tests/test_pandad_canprotocol",
  "openpilot/tools/cabana/tests/test_dbc_core",
)


class TestNative(OpenpilotTestCase):
  @parameterized.expand(NATIVE_TESTS)
  def test_native(self, executable):
    path = os.path.join(BASEDIR, executable)
    if not os.path.exists(path):
      self.skipTest(f"optional native test was not built: {executable}")
    subprocess.run([path], check=True)
