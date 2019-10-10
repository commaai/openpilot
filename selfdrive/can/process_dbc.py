#!/usr/bin/env python3
from __future__ import print_function
import os
import glob
import sys

import jinja2

from collections import Counter
from common.dbc import dbc

def main():
  if len(sys.argv) != 3:
    print("usage: %s dbc_directory output_directory" % (sys.argv[0],))
    sys.exit(0)

  dbc_dir = sys.argv[1]
  out_dir = sys.argv[2]

  template_fn = os.path.join(os.path.dirname(__file__), "dbc_template.cc")
  template_mtime = os.path.getmtime(template_fn)

  this_file_mtime = os.path.getmtime(__file__)

  with open(template_fn, "r") as template_f:
    template = jinja2.Template(template_f.read(), trim_blocks=True, lstrip_blocks=True)

  for dbc_path in glob.iglob(os.path.join(dbc_dir, "*.dbc")):
    dbc_mtime = os.path.getmtime(dbc_path)
    dbc_fn = os.path.split(dbc_path)[1]
    dbc_name = os.path.splitext(dbc_fn)[0]
    can_dbc = dbc(dbc_path)
    out_fn = os.path.join(os.path.dirname(__file__), out_dir, dbc_name + ".cc")
    if os.path.exists(out_fn):
      out_mtime = os.path.getmtime(out_fn)
    else:
      out_mtime = 0

    if dbc_mtime < out_mtime and template_mtime < out_mtime and this_file_mtime < out_mtime:
      continue #skip output is newer than template and dbc

    msgs = [(address, msg_name, msg_size, sorted(msg_sigs, key=lambda s: s.name not in (b"COUNTER", b"CHECKSUM"))) # process counter and checksums first
            for address, ((msg_name, msg_size), msg_sigs) in sorted(can_dbc.msgs.items()) if msg_sigs]

    def_vals = {a: set(b) for a,b in can_dbc.def_vals.items()} #remove duplicates
    def_vals = [(address, sig) for address, sig in sorted(def_vals.items())]

    if can_dbc.name.startswith("honda") or can_dbc.name.startswith("acura"):
      checksum_type = "honda"
      checksum_size = 4
    elif can_dbc.name.startswith("toyota") or can_dbc.name.startswith("lexus"):
      checksum_type = "toyota"
      checksum_size = 8
    else:
      checksum_type = None

    for address, msg_name, msg_size, sigs in msgs:
      for sig in sigs:
        if checksum_type is not None and sig.name == b"CHECKSUM":
          if sig.size != checksum_size:
            sys.exit("CHECKSUM is not %d bits longs %s" % (checksum_size, msg_name))
          if checksum_type == "honda" and sig.start_bit % 8 != 3:
            sys.exit("CHECKSUM starts at wrong bit %s" % msg_name)
          if checksum_type == "toyota" and sig.start_bit % 8 != 7:
            sys.exit("CHECKSUM starts at wrong bit %s" % msg_name)
        if checksum_type == "honda" and sig.name == b"COUNTER":
          if sig.size != 2:
            sys.exit("COUNTER is not 2 bits longs %s" % msg_name)
          if sig.start_bit % 8 != 5:
            sys.exit("COUNTER starts at wrong bit %s" % msg_name)
        if address in [0x200, 0x201]:
          if sig.name == b"COUNTER_PEDAL" and sig.size != 4:
            sys.exit("PEDAL COUNTER is not 4 bits longs %s" % msg_name)
          if sig.name == b"CHECKSUM_PEDAL" and sig.size != 8:
            sys.exit("PEDAL CHECKSUM is not 8 bits longs %s" % msg_name)

    # Fail on duplicate message names
    c = Counter([msg_name for address, msg_name, msg_size, sigs in msgs])
    for name, count in c.items():
      if count > 1:
        sys.exit("Duplicate message name in DBC file %s" % name)

    parser_code = template.render(dbc=can_dbc, checksum_type=checksum_type, msgs=msgs, def_vals=def_vals, len=len)


    with open(out_fn, "w") as out_f:
      out_f.write(parser_code)

if __name__ == '__main__':
  main()
