#!/usr/bin/env python3
from panda import Panda
from hexdump import hexdump

DEBUG = False

if __name__ == "__main__":
  p = Panda()

  len = p._handle.controlRead(Panda.REQUEST_IN, 0x06, 3 << 8 | 238, 0, 1)
  print('Microsoft OS String Descriptor')
  dat = p._handle.controlRead(Panda.REQUEST_IN, 0x06, 3 << 8 | 238, 0, len[0])
  if DEBUG: print('LEN: {}'.format(hex(len[0])))
  hexdump("".join(map(chr, dat)))

  ms_vendor_code = dat[16]
  if DEBUG: print('MS_VENDOR_CODE: {}'.format(hex(len[0])))

  print('\nMicrosoft Compatible ID Feature Descriptor')
  len = p._handle.controlRead(Panda.REQUEST_IN, ms_vendor_code, 0, 4, 1)
  if DEBUG: print('LEN: {}'.format(hex(len[0])))
  dat = p._handle.controlRead(Panda.REQUEST_IN, ms_vendor_code, 0, 4, len[0])
  hexdump("".join(map(chr, dat)))

  print('\nMicrosoft Extended Properties Feature Descriptor')
  len = p._handle.controlRead(Panda.REQUEST_IN, ms_vendor_code, 0, 5, 1)
  if DEBUG: print('LEN: {}'.format(hex(len[0])))
  dat = p._handle.controlRead(Panda.REQUEST_IN, ms_vendor_code, 0, 5, len[0])
  hexdump("".join(map(chr, dat)))
