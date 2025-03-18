#!/usr/bin/env python3
import sys
from hexdump import hexdump
from macholib import MachO
from tinygrad.helpers import getenv
def get_macho(fn):
  # mod to make the header okay
  # MH_CIGAM_64 is good
  dat = open(fn, "rb").read()
  dat = b"\xcf\xfa\xed\xfe"+dat[4:]
  from tempfile import NamedTemporaryFile
  with NamedTemporaryFile(delete=False) as f:
    f.write(dat)
    f.close()
  return MachO.MachO(f.name)

a = get_macho("model.hwx.golden")

# load commands
for c in a.headers[0].commands:
  print("command", c[0], c[1])
  if c[0].cmd == 4:
    hexdump(c[2])
    pass
  if c[0].cmd == 6:
    print("name:", c[2].decode('utf-8'))
  if c[0].cmd == 8:
    print(c[2].decode('utf-8'))
  if c[0].cmd == 25:
    for section in c[2]:
      print(section.segname.strip(b'\0'), section.sectname.strip(b'\0'), hex(section.addr), hex(section.size), "@", hex(c[1].fileoff))
      #print(dir(section))
      if c[1].filesize > 0:
        if len(section.section_data) < 0x100:
          hexdump(section.section_data)
        else:
          print("in file, not dumping 0x%x" % len(section.section_data))

# this parser is wrong (fixed with 64-bit one)
from macholib import SymbolTable
sym = SymbolTable.SymbolTable(a)

syms = {}
for l in sym.nlists:
  print(l)
  if l[0].n_value != 0:
    syms[l[1]] = l[0].n_value

for k,v in syms.items():
  print(k, hex(v))


# **** document what we know ***
from ane import ANE_Struct, ANE
ane = ANE()

aneb = set()
for typ, num, nam in ANE_Struct:
  ltyp = {"u32": 4, "u16": 2, "u8": 1}[typ]
  for l in range(num, num+ltyp):
    aneb.add(l)

# we understand these too
for l in range(0x34, 0xF4):
  aneb.add(l)

from termcolor import colored
def compare(x, y):
  ss = []
  ln = []
  ln2 = []

  ll = (max(len(x), len(y)) + 0xF)//0x10 * 0x10

  highlight = False
  next_highlight = 0x2b
  for i in range(ll+1):
    if i == next_highlight:
      highlight = True
      if i < len(y):
        next_highlight += y[i]+8
      else:
        next_highlight = None
    else:
      highlight = False
    a = "%02X" % x[i] if i < len(x) else "--", \
        "%02X" % y[i] if i < len(y) else "--"
    def fj(x):
      ss = []
      for i in range(0, 0x10, 4):
        ss.append(' '.join(x[i:i+4]))
      return '  '.join(ss)

    if i!=0 and i%0x10 == 0:
      ss.append("%8X: " % (i-0x10)+fj(ln)+"  |  "+fj(ln2)+"\n")
      ln = []
      ln2 = []
    if a[0] != a[1] and a[0] != "--" and a[1] != "--":
      ln.append(colored(a[0], 'green'))
      ln2.append(colored(a[1], 'red'))
    else:
      if highlight:
        ln.append(colored(a[0], 'yellow'))
        ln2.append(colored(a[1], 'yellow'))
      else:
        if i in aneb:
          ln.append(colored(a[0], 'white'))
          ln2.append(colored(a[1], 'white'))
        else:
          ln.append(a[0])
          ln2.append(a[1])
  return ''.join(ss)

import json
aneregs = dict(json.load(open("aneregs.json")))
g = get_macho("model.hwx.golden" if len(sys.argv) < 2 else sys.argv[1])
f1 = g.headers[0].commands[1][2][0].section_data
f2 = a.headers[0].commands[1][2][0].section_data
for i in range(0, len(f2), 0x300):
  print("===== op %d =====" % (i//0x300))
  if len(f1) < 0x300:
    c1, c2 = f1, f2[i:i+0x300]
  else:
    c1, c2 = f1[i:i+0x300], f2[i:i+0x300]
  dbg1 = ane.debug(c1, 16)
  dbg2 = ane.debug(c2, 16)
  if getenv("PRINTALL"):
    for k in dbg2:
      if k in aneregs:
        rr = aneregs[k] if k in aneregs else (-1,-1,-1)
        print("0x%3x %d %2d" % tuple(rr), k, dbg1[k], "->", dbg2[k])
  else:
    for k in dbg1:
      if dbg1[k] != dbg2[k]:
        rr = aneregs[k] if k in aneregs else (-1,-1,-1)
        print("0x%3x %d %2d" % tuple(rr), k, dbg1[k], "->", dbg2[k])

  print(compare(c1, c2))
#open("/tmp/data.section", "wb").write(f2)
#print(compare(open("model.hwx.golden", "rb").read(), open("model.hwx", "rb").read()))
