#!/usr/bin/env python3
from panda import PandaJungle

if __name__ == "__main__":
  for p in PandaJungle.list():
    pp = PandaJungle(p)
    print(f"{pp.get_serial()[0]}: {pp.get_version()}")


