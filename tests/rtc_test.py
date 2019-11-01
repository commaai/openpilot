#!/usr/bin/env python
import os
import sys
import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda

if __name__ == "__main__":
  p = Panda()
  
  p.set_datetime(datetime.datetime.now())
  print(p.get_datetime())
