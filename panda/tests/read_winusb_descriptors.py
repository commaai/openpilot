#!/usr/bin/env python3
# type: ignore
from panda import Panda
from hexdump import hexdump

DEBUG = False

if __name__ == "__main__":
  p = Panda()

  length = p._handle.controlRead(Panda.REQUEST_IN, 0x06, 3 << 8 | 238, 0, 1)
  print('Microsoft OS String Descriptor')
  dat = p._handle.controlRead(Panda.REQUEST_IN, 0x06, 3 << 8 | 238, 0, length[0])
  if DEBUG:
    print(f'LEN: {hex(length[0])}')
  hexdump("".join(map(chr, dat)))

  ms_vendor_code = dat[16]
  if DEBUG:
    print(f'MS_VENDOR_CODE: {hex(length[0])}')

  print('\nMicrosoft Compatible ID Feature Descriptor')
  length = p._handle.controlRead(Panda.REQUEST_IN, ms_vendor_code, 0, 4, 1)
  if DEBUG:
    print(f'LEN: {hex(length[0])}')
  dat = p._handle.controlRead(Panda.REQUEST_IN, ms_vendor_code, 0, 4, length[0])
  hexdump("".join(map(chr, dat)))

  print('\nMicrosoft Extended Properties Feature Descriptor')
  length = p._handle.controlRead(Panda.REQUEST_IN, ms_vendor_code, 0, 5, 1)
  if DEBUG:
    print(f'LEN: {hex(length[0])}')
  dat = p._handle.controlRead(Panda.REQUEST_IN, ms_vendor_code, 0, 5, length[0])
  hexdump("".join(map(chr, dat)))
