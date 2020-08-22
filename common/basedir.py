import os
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

from common.android import ANDROID
if ANDROID:
  PERSIST = "/persist"
  PARAMS = "/data/params"
else:
  PERSIST = os.path.join(BASEDIR, "persist")
  PARAMS = os.path.join(BASEDIR, "persist", "params")
