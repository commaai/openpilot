#!/usr/bin/env python3
import sys

from common.params import Params
from tools.lib.route import Route
from tools.lib.logreader import LogReader

if __name__ == "__main__":
  r = Route(sys.argv[1])
  cp = [m for m in LogReader(r.qlog_paths()[0]) if m.which() == 'carParams']
  Params().put("CarParams", cp[0].carParams.as_builder().to_bytes())
