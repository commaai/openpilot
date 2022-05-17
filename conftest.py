def pytest_addoption(parser):
  parser.addoption("--whitelist-procs", type=str, nargs="*", default=[],
                      help="Whitelist given processes from the test (e.g. controlsd)")
  parser.addoption("--whitelist-cars", type=str, nargs="*", default=[],
                      help="Whitelist given cars from the test (e.g. HONDA)")
  parser.addoption("--blacklist-procs", type=str, nargs="*", default=[],
                      help="Blacklist given processes from the test (e.g. controlsd)")
  parser.addoption("--blacklist-cars", type=str, nargs="*", default=[],
                      help="Blacklist given cars from the test (e.g. HONDA)")
  parser.addoption("--ignore-fields", type=str, nargs="*", default=[],
                      help="Extra fields or msgs to ignore (e.g. carState.events)")
  parser.addoption("--ignore-msgs", type=str, nargs="*", default=[],
                      help="Msgs to ignore (e.g. carEvents)")
  parser.addoption("--update-refs", action="store_true",
                      help="Updates reference logs using current commit")
  parser.addoption("--upload-only", action="store_true",
                      help="Skips testing processes and uploads logs from previous test run")
