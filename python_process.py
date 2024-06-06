import os
import subprocess
import sys
import signal
import time

from openpilot.common.basedir import BASEDIR


def signal_handler(sig, frame):
  print('python process got signal', sig)
  # time.sleep(5)
  # sys.exit()



def main():
  print('pandad!')

  # SystemExit on sigterm
  # signal.signal(signal.SIGTERM, signal_handler)

  while True:
    os.chdir(os.path.join(BASEDIR))
    subprocess.run(["python", "cpp_process.py"], check=True)
