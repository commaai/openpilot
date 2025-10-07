import pytest
import multiprocessing

def pytest_configure(config):
  multiprocessing.set_start_method("spawn", force=True)

def pytest_addoption(parser):
  parser.addoption("--test_duration", action="store", default=60, type=int, help="Seconds to run metadrive drive")

@pytest.fixture
def test_duration(request):
  return request.config.getoption("--test_duration")
