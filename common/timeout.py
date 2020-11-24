import signal

class TimeoutException(Exception):
  pass

class Timeout:
  """
  Timeout context manager.
  For example this code will raise a TimeoutException:
  with Timeout(seconds=5, error_msg="Sleep was too long"):
    time.sleep(10)
  """
  def __init__(self, seconds, error_msg=None):
    if error_msg is None:
      error_msg = 'Timed out after {} seconds'.format(seconds)
    self.seconds = seconds
    self.error_msg = error_msg

  def handle_timeout(self, signume, frame):
    raise TimeoutException(self.error_msg)

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)

  def __exit__(self, exc_type, exc_val, exc_tb):
    signal.alarm(0)
