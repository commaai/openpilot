import os
import subprocess
from common.basedir import BASEDIR


class Spinner():
  def __init__(self):
    self.spinner_proc = subprocess.Popen(["./spinner"],
                                         stdin=subprocess.PIPE,
                                         cwd=os.path.join(BASEDIR, "selfdrive", "ui", "spinner"),
                                         close_fds=True)

  def __enter__(self):
    return self

  def update(self, spinner_text):
    self.spinner_proc.stdin.write(spinner_text.encode('utf8') + b"\n")
    self.spinner_proc.stdin.flush()

  def close(self):
    if self.spinner_proc is not None:
      self.spinner_proc.stdin.close()
      self.spinner_proc.terminate()
      self.spinner_proc = None

  def __del__(self):
    self.close()

  def __exit__(self, type, value, traceback):
    self.close()


class FakeSpinner():
  def __init__(self):
    pass

  def __enter__(self):
    return self

  def update(self, _):
    pass

  def __exit__(self, type, value, traceback):
    pass


if __name__ == "__main__":
  import time
  with Spinner() as s:
    s.update("Spinner text")
    time.sleep(5.0)
  print("gone")
  time.sleep(5.0)
