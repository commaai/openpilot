#!/usr/bin/env python
import os
import sys
import struct
import hashlib
from Crypto.PublicKey import RSA

rsa = RSA.importKey(open(sys.argv[3]).read())

with open(sys.argv[1]) as f:
  dat = f.read()

print "signing", len(dat), "bytes"

with open(sys.argv[2], "wb") as f:
  if os.getenv("SETLEN") is not None:
    x = struct.pack("I", len(dat)) + dat[4:]
    # mock signature of dat[4:]
    dd = hashlib.sha1(dat[4:]).digest()
  else:
    x = dat
    dd = hashlib.sha1(dat).digest()
  print "hash:",dd.encode("hex")
  dd = "\x00\x01" + "\xff"*0x69 + "\x00" + dd
  rsa_out = pow(int(dd.encode("hex"), 16), rsa.d, rsa.n)
  sig = (hex(rsa_out)[2:-1].rjust(0x100, '0')).decode("hex")
  x += sig
  f.write(x)

