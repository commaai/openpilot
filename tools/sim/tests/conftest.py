import os
import pytest
import shutil

from openpilot.selfdrive.car.tests.test_models import CI

def pytest_addoption(parser):
  parser.addoption("--test_duration", action="store", default=60, type=int, help="Seconds to run metadrive drive")

@pytest.fixture
def test_duration(request):
  return request.config.getoption("--test_duration")

@pytest.fixture(scope="session", autouse=True)
def setup_preserve_logs():
  home = os.getenv("HOME")
  op_prefix = os.getenv("OPENPILOT_PREFIX", "")
  rel_log_path = f".comma{op_prefix}/media/0/realdata"
  log_path = os.path.join(home, rel_log_path)
  yield log_path
  if CI and os.path.exists(log_path):
    shutil.rmtree(log_path)
