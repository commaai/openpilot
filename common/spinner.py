import os
import subprocess
from common.basedir import BASEDIR


class Spinner():
  def __init__(self):
    try:
      self.spinner_proc = subprocess.Popen(["./spinner"],
                                           stdin=subprocess.PIPE,
                                           cwd=os.path.join(BASEDIR, "selfdrive", "ui", "spinner"),
                                           close_fds=True)
    except OSError:
      self.spinner_proc = None

  def __enter__(self):
    return self

  def update(self, spinner_text, spinner_status=None):
    if self.spinner_proc is not None:
      if spinner_status is not None:
        spinner_text += ",{}".format(spinner_status)
      self.spinner_proc.stdin.write(spinner_text.encode('utf8') + b"\n")
      try:
        self.spinner_proc.stdin.flush()
      except BrokenPipeError:
        pass

  def close(self):
    if self.spinner_proc is not None:
      try:
        self.spinner_proc.stdin.close()
      except BrokenPipeError:
        pass
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

  def update(self, _, __):
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
