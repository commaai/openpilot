#!/usr/bin/env python3
import os
import sys
from urllib.request import urlopen
import subprocess
import importlib.util

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from common.basedir import BASEDIR
from common.spinner import Spinner

OPSPLINE_SPEC = importlib.util.find_spec('opspline')
TOTAL_PIP_STEPS = 24
MAX_BUILD_PROGRESS = 100


def wait_for_internet_connection():
  while True:
    try:
      _ = urlopen('https://www.google.com/', timeout=10)
      return
    except Exception:
      pass


def install_dep(spinner):
  wait_for_internet_connection()

  # mount system rw so apt and pip can do its thing
  subprocess.check_call(['mount', '-o', 'rw,remount', '/system'])

  # Run preparation script for pip installation
  subprocess.check_call(['sh', './install_gfortran.sh'], cwd=os.path.join(BASEDIR, 'installer/custom/'))

  # install pip from git
  package = 'git+https://github.com/move-fast/opspline.git@master'
  # pip = subprocess.check_call([sys.executable, "-m", "pip", "install", "-v", package], stderr=subprocess.PIPE)
  pip = subprocess.Popen([sys.executable, "-m", "pip", "install", "-v", package], stdout=subprocess.PIPE)

  # Read progress from pip and update spinner
  steps = 0
  while True:
    output = pip.stdout.readline()
    if pip.poll() is not None:
      break
    if output:
      steps += 1
      spinner.update_progress(MAX_BUILD_PROGRESS * min(1., steps / TOTAL_PIP_STEPS), 100.)
      print(output.decode('utf8', 'replace'))


if __name__ == "__main__" and OPSPLINE_SPEC is None:
  spinner = Spinner()
  spinner.update_progress(0, 100)
  install_dep(spinner)
