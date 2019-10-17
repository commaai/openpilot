#!/usr/bin/env python3
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

    msgs = [(address, msg_name, msg_size, sorted(msg_sigs, key=lambda s: s.name not in ("COUNTER", "CHECKSUM"))) # process counter and checksums first
            for address, ((msg_name, msg_size), msg_sigs) in sorted(can_dbc.msgs.items()) if msg_sigs]

    def_vals = {a: set(b) for a,b in can_dbc.def_vals.items()} #remove duplicates
    def_vals = [(address, sig) for address, sig in sorted(def_vals.items())]

    if can_dbc.name.startswith("honda") or can_dbc.name.startswith("acura"):
      car_type = "honda"
      car_is_little_endian = False
      checksum_size = 4
      counter_size = 2
    elif can_dbc.name.startswith("toyota") or can_dbc.name.startswith("lexus"):
      car_type = "toyota"
      car_is_little_endian = False
      checksum_size = 8
      counter_size = None
    elif can_dbc.name.startswith("vw") or can_dbc.name.startswith("volkswagen") or can_dbc.name.startswith("audi") or can_dbc.name.startswith ("seat") or can_dbc.name.startswith("skoda"):
      car_type = "volkswagen"
      car_is_little_endian = True
      checksum_size = 8
      counter_size = 4
    else:
      car_type = None
      car_is_little_endian = None
      checksum_size = None
      counter_size = None

    for address, msg_name, msg_size, sigs in msgs:
      dbc_msg_name = dbc_name + " " + msg_name
      for sig in sigs:

        if sig.name == "CHECKSUM" and checksum_size is not None:
          if sig.size != checksum_size:
            sys.exit("%s: CHECKSUM is not %d bits long" % (dbc_msg_name, checksum_size))
          if car_is_little_endian:
            if not sig.is_little_endian:
              sys.exit("%s: CHECKSUM is not little endian" % dbc_msg_name)
            if sig.start_bit % sig.size != 0:
              sys.exit("%s: CHECKSUM start bit is misaligned" % dbc_msg_name)
          else:
            if sig.is_little_endian:
              sys.exit("%s: CHECKSUM is not big endian" % dbc_msg_name)
            if (sig.start_bit - sig.size + 1) % sig.size != 0:
              sys.exit("%s: CHECKSUM start bit is misaligned" % dbc_msg_name)

        elif sig.name == "COUNTER" and counter_size is not None:
          if sig.size != counter_size:
            sys.exit("%s: COUNTER is not %d bits long" % (dbc_msg_name, counter_size))
          if car_is_little_endian:
            if not sig.is_little_endian:
              sys.exit("%s: COUNTER is not little endian" % dbc_msg_name)
            if sig.start_bit % sig.size != 0:
              sys.exit("%s: COUNTER starts at wrong bit" % dbc_msg_name)
          else:
            if sig.is_little_endian:
              sys.exit("%s: COUNTER is not big endian" % dbc_msg_name)
            if (sig.start_bit - sig.size + 1) % sig.size != 0:
              sys.exit("%s: COUNTER start bit is misaligned" % dbc_msg_name)

        elif address in [0x200, 0x201]:
          if sig.name == "COUNTER_PEDAL" and sig.size != 4:
            sys.exit("%s: PEDAL COUNTER is not 4 bits long" % dbc_msg_name)
          if sig.name == "CHECKSUM_PEDAL" and sig.size != 8:
            sys.exit("%s: PEDAL CHECKSUM is not 8 bits long" % dbc_msg_name)

    # Fail on duplicate message names
    c = Counter([msg_name for address, msg_name, msg_size, sigs in msgs])
    for name, count in c.items():
      if count > 1:
        sys.exit("%s: Duplicate message name in DBC file %s" % (dbc_name, name))

    parser_code = template.render(dbc=can_dbc, checksum_type=car_type, msgs=msgs, def_vals=def_vals, len=len)


    with open(out_fn, "w") as out_f:
      out_f.write(parser_code)

if __name__ == '__main__':
  main()
