import os
import subprocess
from openpilot.common.basedir import BASEDIR


class Spinner():
  def __init__(self):
    try:
      self.spinner_proc = subprocess.Popen(["./spinner"],
                                           stdin=subprocess.PIPE,
                                           cwd=os.path.join(BASEDIR, "selfdrive", "ui"),
                                           close_fds=True)
    except OSError:
      self.spinner_proc = None

  def __enter__(self):
    return self

  def update(self, spinner_text: str):
    if self.spinner_proc is not None:
      self.spinner_proc.stdin.write(spinner_text.encode('utf8') + b"\n")
      try:
        self.spinner_proc.stdin.flush()
      except BrokenPipeError:
        pass

  def update_progress(self, cur: float, total: float):
    self.update(str(round(100 * cur / total)))

  def close(self):
    if self.spinner_proc is not None:
      self.spinner_proc.kill()
      try:
        self.spinner_proc.communicate(timeout=2.)
      except subprocess.TimeoutExpired:
        print("WARNING: failed to kill spinner")
      self.spinner_proc = None

  def __del__(self):
    self.close()

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()


if __name__ == "__main__":
  import time
  with Spinner() as s:
    s.update("Spinner text")
    time.sleep(5.0)
  print("gone")
  time.sleep(5.0)
