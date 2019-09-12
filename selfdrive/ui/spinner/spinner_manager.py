import os
import subprocess
from common.basedir import BASEDIR

class Spinner(object):
  def __enter__(self):
    self.spinner_proc = subprocess.Popen(["./spinner"], stdin=subprocess.PIPE,
      cwd=os.path.join(BASEDIR, "selfdrive", "ui", "spinner"),
      close_fds=True)
    return self

  def update(self, spinner_text):
    if os.getenv("PREPAREONLY") is None:
      self.spinner_proc.stdin.write(spinner_text + "\n")

  def __exit__(self, type, value, traceback):
    self.spinner_proc.stdin.close()
    self.spinner_proc.terminate()