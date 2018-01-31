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

if can_dbc.name.startswith("honda") or can_dbc.name.startswith("acura"):
  checksum_type = "honda"
elif can_dbc.name.startswith("toyota"):
  checksum_type = "toyota"
else:
  checksum_type = None


# Fail on duplicate messgae names
c = Counter([msg_name for address, msg_name, msg_size, sigs in msgs])
for name, count in c.items():
  if count > 1:
    sys.exit("Duplicate message name in DBC file %s" % name)

parser_code = template.render(dbc=can_dbc, checksum_type=checksum_type, msgs=msgs, len=len)

with open(out_fn, "w") as out_f:
  out_f.write(parser_code)
