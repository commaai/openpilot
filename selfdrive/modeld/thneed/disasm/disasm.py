#!/usr/bin/env python3
import struct
from hexdump import hexdump
from termcolor import colored

def parse_packets(dat):
  ret = []
  ptr = 0
  while ptr < len(dat):
    ll = struct.unpack("Q", dat[ptr:ptr+8])[0]
    ptr += 8
    ldat = dat[ptr:ptr+ll]
    ptr += ll
    ret.append(ldat)
  return ret

def parse_cmd_packet(dat):
  hexdump(dat)
  ptr = 0


regs = {}
for l in open("../include/a5xx.xml.h").read().split("\n"):
  if l.startswith("#define REG"):
    _, rr, aa = l.split()
    regs[int(aa, 16)] = rr

ops = {}
for l in open("../include/adreno_pm4.xml.h").read().split("\n")[134:233]:
  rr, _, aa = l.strip(",").split()
  ops[int(aa)] = rr


for i in range(34):
  pkts1 = parse_packets(open("../runs/run_1_%d" % i, "rb").read())
  pkts2 = parse_packets(open("../runs/run_2_%d" % i, "rb").read())
  assert pkts1[0] == pkts2[0]
  assert pkts1[2] == pkts2[2]
  print("parsing packet with len %x" % len(pkts1[1]))
  prt = []
  k = 0
  while k*4 < len(pkts1[1]):
    o1 = struct.unpack("I", pkts1[1][k*4:k*4+4])[0]
    o2 = struct.unpack("I", pkts2[1][k*4:k*4+4])[0]
    assert o1 == o2
 
    pktsize = 1
    if (o1 >> 28) == 7:
      pkttype = 7
      pktsize += o1 & 0x3FFF
    elif (o1 >> 28) == 4:
      pkttype = 4
      pktsize += o1 & 0x7F
    else:
      assert False

    prt.append("%d size: %2d -- " % (pkttype, pktsize))

    prtsize = 0

    for _ in range(pktsize):
      o1 = struct.unpack("I", pkts1[1][k*4:k*4+4])[0]
      o2 = struct.unpack("I", pkts2[1][k*4:k*4+4])[0]
      if o1 == o2:
        prt.append("%08X " % o1)
        prtsize += 1
      else:
        prt.append(colored("%08X " % o1, 'red'))
        prt.append(colored("%08X " % o2, 'green'))
        prtsize += 2
      k += 1

    prt.append(" "*(9*(10-prtsize)))

    if pkttype == 7:
      op = (o1>>16) & 0x7F
      prt.append("-- op: %s" % (ops[op] if op in ops else ("%5X" % op)))

    if pkttype == 4:
      op = (o1>>8) & 0x7FFFF
      prt.append("-- reg: %s" % (regs[op] if op in regs else ("%5X" % op)))

    prt.append("\n")
  print(''.join(prt))
    

  #hexdump(pkts1[1])
  #hexdump(pkts2[1])


