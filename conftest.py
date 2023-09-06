import os
import pytest
import shutil

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.system.hardware.hw import Paths


@pytest.fixture(scope="function", autouse=True)
def global_setup_and_teardown():
  # setup a clean environment for each test
  with OpenpilotPrefix():
    os.makedirs(Paths.log_root())
    yield
    shutil.rmtree(Paths.log_root(), ignore_errors=True)
