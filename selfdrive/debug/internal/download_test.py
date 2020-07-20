#!/usr/bin/env python3

import os
import time

os.environ["COMMA_PARALLEL_DOWNLOADS"] = "1"

from tools.lib.auth_config import get_token
from tools.lib.api import CommaApi
from tools.lib.filereader import FileReader


route_name = "1ea81d35a5792cef|2020-07-09--15-27-25"

t = time.time()
api = CommaApi(get_token())
route_files = api.get('v1/route/' + route_name + '/files')

log_url = route_files['logs'][0]
# log_url = route_files['cameras'][0]

dt = time.time() - t
print(f"Getting api response took {dt:.2f} s")

for threads in range(17):
  t = time.time()
  with FileReader(log_url) as f:
    os.environ["COMMA_PARALLEL_DOWNLOADS"] = str(threads)
    contents = f.read()
    length = len(contents)

  dt = time.time() - t
  kbps = length / dt / 1024
  print(f"{threads} threads - Dowloading file took {dt:.2f} s  - {kbps:.2f} kbps")
