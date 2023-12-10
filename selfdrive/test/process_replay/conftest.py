import pytest

from openpilot.selfdrive.test.process_replay.helpers import ALL_PROCS
from openpilot.selfdrive.test.process_replay.test_processes import ALL_CARS


def pytest_addoption(parser: pytest.Parser):
  parser.addoption("--whitelist-procs", type=str, nargs="*", default=ALL_PROCS,
                      help="Whitelist given processes from the test (e.g. controlsd)")
  parser.addoption("--whitelist-cars", type=str, nargs="*", default=ALL_CARS,
                      help="Whitelist given cars from the test (e.g. HONDA)")
  parser.addoption("--blacklist-procs", type=str, nargs="*", default=[],
                      help="Blacklist given processes from the test (e.g. controlsd)")
  parser.addoption("--blacklist-cars", type=str, nargs="*", default=[],
                      help="Blacklist given cars from the test (e.g. HONDA)")
  parser.addoption("--ignore-fields", type=str, nargs="*", default=[],
                      help="Extra fields or msgs to ignore (e.g. carState.events)")
  parser.addoption("--ignore-msgs", type=str, nargs="*", default=[],
                      help="Msgs to ignore (e.g. onroadEvents)")
  parser.addoption("--update-refs", action="store_true",
                      help="Updates reference logs using current commit")
  parser.addoption("--upload-only", action="store_true",
                      help="Skips testing processes and uploads logs from previous test run")
  parser.addoption("--long-diff", action="store_true",
                      help="Outputs diff in long format")


@pytest.fixture(scope="class", autouse=True)
def process_replay_test_arguments(request):
  if hasattr(request.cls, "segment"): # check if a subclass of TestProcessReplayBase
    request.cls.tested_procs = list(set(request.config.getoption("--whitelist-procs")) - set(request.config.getoption("--blacklist-procs")))
    request.cls.tested_cars = list({c.upper() for c in set(request.config.getoption("--whitelist-cars")) - set(request.config.getoption("--blacklist-cars"))})
    request.cls.ignore_fields = request.config.getoption("--ignore-fields")
    request.cls.ignore_msgs = request.config.getoption("--ignore-msgs")
    request.cls.upload_only = request.config.getoption("--upload-only")
    request.cls.update_refs = request.config.getoption("--update-refs")
    request.cls.long_diff = request.config.getoption("--long-diff")
