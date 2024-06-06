import os
import subprocess

from openpilot.common.basedir import BASEDIR


def main():
  print('pandad!')

  while True:
    os.chdir(os.path.join(BASEDIR))
    subprocess.run(["python", "cpp_process.py"], check=True)
