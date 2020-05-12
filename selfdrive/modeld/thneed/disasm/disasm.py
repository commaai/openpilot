#!/usr/bin/env python3
import struct
from hexdump import hexdump

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

dat = open("../runs/run_3_0", "rb").read()
pkts = parse_packets(dat)
parse_cmd_packet(pkts[0])

