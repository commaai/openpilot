#!/usr/bin/env python3
import os
import sys
import errno
import shutil
import time
from common.basedir import BASEDIR
from urllib.request import urlopen
from glob import glob
import subprocess
import importlib.util

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from common.spinner import Spinner


sys.path.append(os.path.join(BASEDIR, "third_party"))
OPSPLINE_SPEC = importlib.util.find_spec('scipy')
OVERPY_SPEC = importlib.util.find_spec('overpy')
MAX_BUILD_PROGRESS = 100
TMP_DIR = '/data/tmp'
THIRD_PARTY_DIR = '/data/openpilot/third_party'


def wait_for_internet_connection(return_on_failure=False):
  retries = 0
  while True:
    try:
      _ = urlopen('https://www.google.com/', timeout=10)
      return True
    except Exception as e:
      print(f'Wait for internet failed: {e}')
      if return_on_failure and retries == 15:
        return False
      retries += 1
      time.sleep(2)  # Wait for 2 seconds before retrying


def install_dep(spinner):
  wait_for_internet_connection()

  TOTAL_PIP_STEPS = 2986

  try:
    os.makedirs(TMP_DIR)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
  my_env = os.environ.copy()
  my_env['TMPDIR'] = TMP_DIR

  pip_target = [f'--target={THIRD_PARTY_DIR}']
  packages = []
  if OPSPLINE_SPEC is None:
    packages.append('scipy==1.7.1')
  if OVERPY_SPEC is None:
    packages.append('overpy==0.6')

  pip = subprocess.Popen([sys.executable, "-m", "pip", "install", "-v"] + pip_target + packages,
                          stdout=subprocess.PIPE, env=my_env)

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

  shutil.rmtree(TMP_DIR)
  os.unsetenv('TMPDIR')

  # remove numpy installed to THIRD_PARTY_DIR since numpy is already present in the AGNOS image
  if OPSPLINE_SPEC is None:
    for directory in glob(f'{THIRD_PARTY_DIR}/numpy*'):
      shutil.rmtree(directory)
    shutil.rmtree(f'{THIRD_PARTY_DIR}/bin')


if __name__ == "__main__" and (OPSPLINE_SPEC is None or OVERPY_SPEC is None):
  spinner = Spinner()
  spinner.update_progress(0, 100)
  install_dep(spinner)
