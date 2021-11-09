#!/usr/bin/env python3
import json
import sys

from common.params import Params

if __name__ == "__main__":
  coords = sys.argv[1].split("/@")[-1].split("/")[0].split(",")
  dest = {"latitude": float(coords[0]), "longitude": float(coords[1])}
  Params().put("NavDestination", json.dumps(dest))
