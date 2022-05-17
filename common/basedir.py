import os
from pathlib import Path

from selfdrive.hardware import PC

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

if PC:
  PERSIST = os.path.join(str(Path.home()), ".comma", "persist")
else:
  PERSIST = "/persist"

if PC:
  PARAMS = os.path.join(str(Path.home()), ".comma", "params")
else:
  PARAMS = "/data/params"
