import pytest

def pytest_addoption(parser):
  parser.addoption("--time_done", action="store", default=60, type=int, help="Seconds to run metadrive drive")

@pytest.fixture
def time_done(request):
  return request.config.getoption("--time_done")
