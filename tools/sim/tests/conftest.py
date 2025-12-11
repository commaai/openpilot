import pytest

def pytest_addoption(parser):
  try:
    parser.addoption("--test_duration", action="store", default=60, type=int, help="Seconds to run metadrive drive")
  except ValueError:
    pass # already added

@pytest.fixture
def test_duration(request):
  return request.config.getoption("--test_duration")
