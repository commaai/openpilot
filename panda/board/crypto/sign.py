#!/usr/bin/env python3
import os
import sys
import struct
import hashlib
import binascii
import base64


def read_der(dat):
  length = dat[1]
  offset = 2
  if length & 0x80:
    length_bytes = length & 0x7f
    length = int.from_bytes(dat[offset:offset + length_bytes], "big")
    offset += length_bytes
  return dat[offset:offset + length], dat[offset + length:]

# increment this to make new hardware not run old versions
VERSION = 2

if __name__ == "__main__":
  with open(sys.argv[3], "rb") as k:
    pem = b"".join(k.read().splitlines()[1:-1])
  der, _ = read_der(base64.b64decode(pem))
  values = []
  while der:
    value, der = read_der(der)
    values.append(int.from_bytes(value, "big"))
  _, rsa_n, _, rsa_d = values[:4]

  with open(sys.argv[1], "rb") as f:
    dat = f.read()

  print("signing", len(dat), "bytes")

  with open(sys.argv[2], "wb") as f:
    if os.getenv("SETLEN") is not None:
      # add the version at the end
      dat += b"VERS" + struct.pack("I", VERSION)
      # add the length at the beginning
      x = struct.pack("I", len(dat)) + dat[4:]
      # mock signature of dat[4:]
      dd = hashlib.sha1(dat[4:]).digest()
    else:
      x = dat
      dd = hashlib.sha1(dat).digest()

    print("hash:", str(binascii.hexlify(dd), "utf-8"))
    dd = b"\x00\x01" + b"\xff" * 0x69 + b"\x00" + dd
    rsa_out = pow(int.from_bytes(dd, byteorder='big', signed=False), rsa_d, rsa_n)
    sig = (hex(rsa_out)[2:].rjust(0x100, '0'))
    x += binascii.unhexlify(sig)
    f.write(x)
