#!/usr/bin/env python3
import random
import unittest

from panda import pack_can_buffer, unpack_can_buffer, DLC_TO_LEN

class PandaTestPackUnpack(unittest.TestCase):
  def test_panda_lib_pack_unpack(self):
    overflow_buf = b''

    to_pack = []
    for _ in range(10000):
      address = random.randint(1, (1 << 29) - 1)
      data = bytes([random.getrandbits(8) for _ in range(DLC_TO_LEN[random.randrange(0, len(DLC_TO_LEN))])])
      to_pack.append((address, data, 0))

    packed = pack_can_buffer(to_pack)
    unpacked = []
    for dat in packed:
      msgs, overflow_buf = unpack_can_buffer(overflow_buf + dat)
      unpacked.extend(msgs)

    self.assertEqual(unpacked, to_pack)

if __name__ == "__main__":
  unittest.main()
