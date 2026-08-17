import os, pytest, signal, threading

@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
  t = threading.Timer(int(os.getenv("TEST_TIMEOUT", 300)), os.kill, args=(os.getpid(), signal.SIGABRT))
  t.start()
  try: yield
  finally:
    t.cancel()
    t.join()
