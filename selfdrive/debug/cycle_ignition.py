#!/usr/bin/env python3
import time
from common.params import Params

if __name__ == "__main__":
  ign = False
  params = Params()
  while True:
    ign = not ign
    params.put_bool("IgnitionOverride", ign)
    time.sleep(30)

