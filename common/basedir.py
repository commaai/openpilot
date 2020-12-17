import os
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

from selfdrive.hardware import PC
if PC:
  PERSIST = os.path.join(BASEDIR, "persist")
else:
  PERSIST = "/persist"
