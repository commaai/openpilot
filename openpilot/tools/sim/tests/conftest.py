import pytest

def pytest_addoption(parser):
  parser.addoption("--test_duration", action="store", default=60, type=int, help="Seconds to run metadrive drive")

@pytest.fixture
def test_duration(request):
  return request.config.getoption("--test_duration")
