#!/usr/bin/env python3
import bz2
import os
import sys

import dictdiffer
if "CI" in os.environ:
  tqdm = lambda x: x
else:
  from tqdm import tqdm

from tools.lib.logreader import LogReader

def save_log(dest, log_msgs):
  dat = ""
  for msg in log_msgs:
    dat += msg.as_builder().to_bytes()
  dat = bz2.compress(dat)

  with open(dest, "w") as f:
   f.write(dat)

def remove_ignored_fields(msg, ignore):
  msg = msg.as_builder() 
  for key, val in ignore:
    attr = msg
    keys = key.split(".")
    if msg.which() not in key and len(keys) > 1:
      continue
    
    for k in keys[:-1]:
      try:
        attr = getattr(msg, k)
      except:
        break
    else:
      setattr(attr, keys[-1], val)
  return msg.as_reader()

def compare_logs(log1, log2, ignore=[]):
  assert len(log1) == len(log2), "logs are not same length"

  ignore_fields = [k for k, v in ignore]
  diff = []
  for msg1, msg2 in tqdm(zip(log1, log2)):
    assert msg1.which() == msg2.which(), "msgs not aligned between logs"

    msg1_bytes = remove_ignored_fields(msg1, ignore).as_builder().to_bytes()
    msg2_bytes = remove_ignored_fields(msg2, ignore).as_builder().to_bytes()

    if msg1_bytes != msg2_bytes:
      msg1_dict = msg1.to_dict(verbose=True)
      msg2_dict = msg2.to_dict(verbose=True)
      dd = dictdiffer.diff(msg1_dict, msg2_dict, ignore=ignore_fields, tolerance=0)
      diff.extend(dd)
  return diff

if __name__ == "__main__":
  log1 = list(LogReader(sys.argv[1]))
  log2 = list(LogReader(sys.argv[2]))
  compare_logs(log1, log2, sys.argv[3:])
