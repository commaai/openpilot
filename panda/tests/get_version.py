#!/usr/bin/env python
from panda import Panda

if __name__ == "__main__":
  for p in Panda.list():
    pp = Panda(p)
    print("%s: %s" % (pp.get_serial()[0], pp.get_version()))


