import os
from pathlib import Path

from selfdrive.hardware import PC, JETSON

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

if PC or JETSON:
  PERSIST = os.path.join(str(Path.home()), ".comma", "persist")
else:
  PERSIST = "/persist"
