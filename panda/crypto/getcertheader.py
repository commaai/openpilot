#!/usr/bin/env python3
import sys
import struct
from Crypto.PublicKey import RSA

def egcd(a, b):
  if a == 0:
    return (b, 0, 1)
  else:
    g, y, x = egcd(b % a, a)
    return (g, x - (b // a) * y, y)

def modinv(a, m):
  g, x, y = egcd(a, m)
  if g != 1:
    raise Exception('modular inverse does not exist')
  else:
    return x % m

def to_c_string(x):
  mod = (hex(x)[2:-1].rjust(0x100, '0'))
  hh = ''.join('\\x'+mod[i:i+2] for i in range(0, 0x100, 2))
  return hh

def to_c_uint32(x):
  nums = []
  for i in range(0x20):
    nums.append(x%(2**32))
    x //= (2**32)
  return "{"+'U,'.join(map(str, nums))+"U}"

for fn in sys.argv[1:]:
  rsa = RSA.importKey(open(fn).read())
  rr = pow(2**1024, 2, rsa.n)
  n0inv = 2**32 - modinv(rsa.n, 2**32)

  cname = fn.split("/")[-1].split(".")[0] + "_rsa_key"

  print('RSAPublicKey '+cname+' = {.len = 0x20,')
  print('  .n0inv = %dU,' % n0inv)
  print('  .n = %s,' % to_c_uint32(rsa.n))
  print('  .rr = %s,' % to_c_uint32(rr))
  print('  .exponent = %d,' % rsa.e)
  print('};')


