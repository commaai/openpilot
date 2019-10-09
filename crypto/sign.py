#!/usr/bin/env python3
import os
import sys
import struct
import hashlib
from Crypto.PublicKey import RSA
import binascii

rsa = RSA.importKey(open(sys.argv[3]).read())

with open(sys.argv[1], "rb") as f:
  dat = f.read()

print("signing", len(dat), "bytes")

with open(sys.argv[2], "wb") as f:
  if os.getenv("SETLEN") is not None:
    x = struct.pack("I", len(dat)) + dat[4:]
    # mock signature of dat[4:]
    dd = hashlib.sha1(dat[4:]).digest()
  else:
    x = dat
    dd = hashlib.sha1(dat).digest()

  print("hash:", str(binascii.hexlify(dd), "utf-8"))
  dd = b"\x00\x01" + b"\xff"*0x69 + b"\x00" + dd
  rsa_out = pow(int.from_bytes(dd, byteorder='big', signed=False), rsa.d, rsa.n)
  sig = (hex(rsa_out)[2:].rjust(0x100, '0'))
  x += binascii.unhexlify(sig)
  f.write(x)

