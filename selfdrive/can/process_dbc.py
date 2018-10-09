#!/usr/bin/env python
import os
import sys

import jinja2

from collections import Counter
from common.dbc import dbc

if len(sys.argv) != 3:
  print "usage: %s dbc_path struct_path" % (sys.argv[0],)
  sys.exit(0)

dbc_fn = sys.argv[1]
out_fn = sys.argv[2]

template_fn = os.path.join(os.path.dirname(__file__), "dbc_template.cc")

can_dbc = dbc(dbc_fn)

with open(template_fn, "r") as template_f:
  template = jinja2.Template(template_f.read(), trim_blocks=True, lstrip_blocks=True)

msgs = [(address, msg_name, msg_size, sorted(msg_sigs, key=lambda s: s.name not in ("COUNTER", "CHECKSUM"))) # process counter and checksums first
        for address, ((msg_name, msg_size), msg_sigs) in sorted(can_dbc.msgs.iteritems()) if msg_sigs]

def_vals = {a: set(b) for a,b in can_dbc.def_vals.items()} #remove duplicates
def_vals = [(address, sig) for address, sig in sorted(def_vals.iteritems())]

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
    if checksum_type is not None and sig.name == "CHECKSUM":
      if sig.size != checksum_size:
        sys.exit("CHECKSUM is not %d bits longs %s" % (checksum_size, msg_name))
      if checksum_type == "honda" and sig.start_bit % 8 != 3:
        sys.exit("CHECKSUM starts at wrong bit %s" % msg_name)
      if checksum_type == "toyota" and sig.start_bit % 8 != 7:
        sys.exit("CHECKSUM starts at wrong bit %s" % msg_name)
    if checksum_type == "honda" and sig.name == "COUNTER":
      if sig.size != 2:
        sys.exit("COUNTER is not 2 bits longs %s" % msg_name)
      if sig.start_bit % 8 != 5:
        sys.exit("COUNTER starts at wrong bit %s" % msg_name)


# Fail on duplicate message names
c = Counter([msg_name for address, msg_name, msg_size, sigs in msgs])
for name, count in c.items():
  if count > 1:
    sys.exit("Duplicate message name in DBC file %s" % name)

parser_code = template.render(dbc=can_dbc, checksum_type=checksum_type, msgs=msgs, def_vals=def_vals, len=len)

with open(out_fn, "w") as out_f:
  out_f.write(parser_code)
