import pytest

def pytest_addoption(parser):
  parser.addoption("--track_size", action="store", default=60, type=int, help="Set the TRACK_SIZE for tests")

@pytest.fixture
def track_size(request):
  return request.config.getoption("--track_size")
