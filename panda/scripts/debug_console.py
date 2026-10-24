#!/usr/bin/env python3

import os
import sys
import time
import codecs

from panda import Panda

setcolor = ["\033[1;32;40m", "\033[1;31;40m"]
unsetcolor = "\033[00m"

claim = os.getenv("CLAIM") is not None
no_color = os.getenv("NO_COLOR") is not None
no_reconnect = os.getenv("NO_RECONNECT") is not None

if __name__ == "__main__":
  while True:
    try:
      serials = Panda.list()
      if os.getenv("SERIAL"):
        serials = [x for x in serials if x == os.getenv("SERIAL")]

      pandas = [Panda(x, claim=claim) for x in serials]
      decoders = [codecs.getincrementaldecoder('utf-8')() for _ in pandas]

      if not len(pandas):
        print("no pandas found")
        if no_reconnect:
          sys.exit(0)
        time.sleep(1)
        continue

      while True:
        for i, panda in enumerate(pandas):
          while True:
            ret = panda.debug_read()
            if len(ret) > 0:
              decoded = decoders[i].decode(ret)
              if no_color:
                sys.stdout.write(decoded)
              else:
                sys.stdout.write(setcolor[i] + decoded + unsetcolor)
              sys.stdout.flush()
            else:
              break
          time.sleep(0.01)
    except KeyboardInterrupt:
      break
    except Exception:
      print("panda disconnected!")
      time.sleep(0.5)
