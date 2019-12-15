#!/usr/bin/env python3
import os

assert os.system("make") == 0
from common.basedir import BASEDIR

os.environ["ADSP_LIBRARY_PATH"] = os.path.join(BASEDIR, "selfdrive/visiond/dsp")
os.execv("./visiond", ["visiond"])
