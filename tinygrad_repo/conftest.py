import os, pytest, faulthandler

stderr_fd_key = pytest.StashKey[int]()
def pytest_configure(config):
  # dup stderr so that we can still write to it in faulthandler even if stderr would be captured
  config.stash[stderr_fd_key] = fd = os.dup(2)
  config.add_cleanup(lambda: os.close(fd))

@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
  faulthandler.dump_traceback_later(int(os.getenv("TEST_TIMEOUT", 120)), file=item.config.stash[stderr_fd_key], exit=True)
  try: yield
  finally: faulthandler.cancel_dump_traceback_later()
