import os
from pathlib import Path

from system.hardware import PC

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

if PC:
  PERSIST = os.path.join(str(Path.home()), ".comma", "persist")
else:
  PERSIST = "/persist"
