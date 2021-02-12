#!/usr/bin/env python3

import sys
import subprocess
from tools.lib.route import Route
from tools.lib.url_file import URLFile

def juggle_segment(route_name, segment_nr):
  
  r = Route(route_name)
  lp = r.log_paths()[segment_nr]

  if lp is None:
    print("This segment does not exist, please try a different one")
    return

  uf = URLFile(lp)
  
  subprocess.call(f"bin/plotjuggler -d {uf.name}", shell=True)


if __name__ == "__main__":
  juggle_segment(sys.argv[1], int(sys.argv[2]))
