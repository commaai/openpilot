#!/usr/bin/env python3
import random
import unittest
from panda import pack_can_buffer, unpack_can_buffer


class PandaTestPackUnpack(unittest.TestCase):
  def test_panda_lib_pack_unpack(self):
    to_pack = []
    for _ in range(10000):
      address = random.randint(1, 0x1FFFFFFF)
      data = bytes([random.getrandbits(8) for _ in range(random.randrange(1, 9))])
      to_pack.append((address, 0, data, 0))

    packed = pack_can_buffer(to_pack)
    unpacked = []
    for dat in packed:
      unpacked.extend(unpack_can_buffer(dat))

    assert unpacked == to_pack

if __name__ == "__main__":
  unittest.main()
