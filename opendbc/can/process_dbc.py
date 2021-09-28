#!/usr/bin/env python3
from __future__ import print_function
import os
import sys

import jinja2

from collections import Counter
from opendbc.can.dbc import dbc

def process(in_fn, out_fn):
  dbc_name = os.path.split(out_fn)[-1].replace('.cc', '')
  # print("processing %s: %s -> %s" % (dbc_name, in_fn, out_fn))

  template_fn = os.path.join(os.path.dirname(__file__), "dbc_template.cc")

  with open(template_fn, "r") as template_f:
    template = jinja2.Template(template_f.read(), trim_blocks=True, lstrip_blocks=True)

  can_dbc = dbc(in_fn)

  # process counter and checksums first
  msgs = [(address, msg_name, msg_size, sorted(msg_sigs, key=lambda s: s.name not in ("COUNTER", "CHECKSUM")))
          for address, ((msg_name, msg_size), msg_sigs) in sorted(can_dbc.msgs.items()) if msg_sigs]

  def_vals = {a: sorted(set(b)) for a, b in can_dbc.def_vals.items()}  # remove duplicates
  def_vals = sorted(def_vals.items())

  if can_dbc.name.startswith(("honda_", "acura_")):
    checksum_type = "honda"
    checksum_size = 4
    counter_size = 2
    checksum_start_bit = 3
    counter_start_bit = 5
    little_endian = False
  elif can_dbc.name.startswith(("toyota_", "lexus_")):
    checksum_type = "toyota"
    checksum_size = 8
    counter_size = None
    checksum_start_bit = 7
    counter_start_bit = None
    little_endian = False
  elif can_dbc.name.startswith(("vw_", "volkswagen_", "audi_", "seat_", "skoda_")):
    checksum_type = "volkswagen"
    checksum_size = 8
    counter_size = 4
    checksum_start_bit = 0
    counter_start_bit = 0
    little_endian = True
  elif can_dbc.name.startswith(("subaru_global_")):
    checksum_type = "subaru"
    checksum_size = 8
    counter_size = None
    checksum_start_bit = 0
    counter_start_bit = None
    little_endian = True
  elif can_dbc.name.startswith(("chrysler_", "stellantis_")):
    checksum_type = "chrysler"
    checksum_size = 8
    counter_size = None
    checksum_start_bit = 7
    counter_start_bit = None
    little_endian = False
  else:
    checksum_type = None
    checksum_size = None
    counter_size = None
    checksum_start_bit = None
    counter_start_bit = None
    little_endian = None

  # sanity checks on expected COUNTER and CHECKSUM rules, as packer and parser auto-compute those signals
  for address, msg_name, _, sigs in msgs:
    dbc_msg_name = dbc_name + " " + msg_name
    for sig in sigs:
      if checksum_type is not None:
        # checksum rules
        if sig.name == "CHECKSUM":
          if sig.size != checksum_size:
            sys.exit("%s: CHECKSUM is not %d bits long" % (dbc_msg_name, checksum_size))
          if sig.start_bit % 8 != checksum_start_bit:
            sys.exit("%s: CHECKSUM starts at wrong bit" % dbc_msg_name)
          if little_endian != sig.is_little_endian:
            sys.exit("%s: CHECKSUM has wrong endianness" % dbc_msg_name)
        # counter rules
        if sig.name == "COUNTER":
          if counter_size is not None and sig.size != counter_size:
            sys.exit("%s: COUNTER is not %d bits long" % (dbc_msg_name, counter_size))
          if counter_start_bit is not None and sig.start_bit % 8 != counter_start_bit:
            print(counter_start_bit, sig.start_bit)
            sys.exit("%s: COUNTER starts at wrong bit" % dbc_msg_name)
          if little_endian != sig.is_little_endian:
            sys.exit("%s: COUNTER has wrong endianness" % dbc_msg_name)
      # pedal rules
      if address in [0x200, 0x201]:
        if sig.name == "COUNTER_PEDAL" and sig.size != 4:
          sys.exit("%s: PEDAL COUNTER is not 4 bits long" % dbc_msg_name)
        if sig.name == "CHECKSUM_PEDAL" and sig.size != 8:
          sys.exit("%s: PEDAL CHECKSUM is not 8 bits long" % dbc_msg_name)

  # Fail on duplicate message names
  c = Counter([msg_name for address, msg_name, msg_size, sigs in msgs])
  for name, count in c.items():
    if count > 1:
      sys.exit("%s: Duplicate message name in DBC file %s" % (dbc_name, name))

  parser_code = template.render(dbc=can_dbc, checksum_type=checksum_type, msgs=msgs, def_vals=def_vals, len=len)

  with open(out_fn, "a+") as out_f:
    out_f.seek(0)
    if out_f.read() != parser_code:
      out_f.seek(0)
      out_f.truncate()
      out_f.write(parser_code)

def main():
  if len(sys.argv) != 3:
    print("usage: %s dbc_directory output_filename" % (sys.argv[0],))
    sys.exit(0)

  dbc_dir = sys.argv[1]
  out_fn = sys.argv[2]

  dbc_name = os.path.split(out_fn)[-1].replace('.cc', '')
  in_fn = os.path.join(dbc_dir, dbc_name + '.dbc')

  process(in_fn, out_fn)


if __name__ == '__main__':
  main()
