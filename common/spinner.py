import os
import subprocess
from common.basedir import BASEDIR

class Spinner():
  def __enter__(self):
    self.spinner_proc = subprocess.Popen(["./spinner"],
                                         stdin=subprocess.PIPE,
                                         cwd=os.path.join(BASEDIR, "selfdrive", "ui", "spinner"),
                                         close_fds=True)
    return self

  def update(self, spinner_text):
    self.spinner_proc.stdin.write(spinner_text.encode('utf8') + b"\n")
    self.spinner_proc.stdin.flush()

  def __exit__(self, type, value, traceback):
    self.spinner_proc.stdin.close()
    self.spinner_proc.terminate()



if __name__ == "__main__":
  import time
  with Spinner() as s:
    s.update("Spinner text")
    time.sleep(5.0)

