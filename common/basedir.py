import os
from pathlib import Path

from selfdrive.hardware import PC

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

if PC:
  PERSIST = os.path.join(BASEDIR, "persist")
  PARAMS = os.path.join(BASEDIR, "persist", "params")
else:
  PERSIST = "/persist"
  PARAMS = "/data/params"
