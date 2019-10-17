import random
import numpy as np

import selfdrive.boardd.tests.boardd_old as boardd_old
import selfdrive.boardd.boardd as boardd

from common.realtime import sec_since_boot
from cereal import log
import unittest


def generate_random_can_data_list():
  can_list = []
  cnt = random.randint(1, 64)
  for j in range(cnt):
    can_data = np.random.bytes(random.randint(1, 8))
    can_list.append([random.randint(0, 128), random.randint(0, 128), can_data, random.randint(0, 128)])
  return can_list, cnt


class TestBoarddApiMethods(unittest.TestCase):
  def test_correctness(self):
    for i in range(1000):
      can_list, _ = generate_random_can_data_list()

      # Sendcan
      # Old API
      m_old = boardd_old.can_list_to_can_capnp(can_list, 'sendcan').to_bytes()
      # new API
      m = boardd.can_list_to_can_capnp(can_list, 'sendcan')

      ev_old = log.Event.from_bytes(m_old)
      ev = log.Event.from_bytes(m)

      self.assertEqual(ev_old.which(), ev.which())
      self.assertEqual(len(ev.sendcan), len(ev_old.sendcan))
      for i in range(len(ev.sendcan)):
        attrs = ['address', 'busTime', 'dat', 'src']
        for attr in attrs:
          self.assertEqual(getattr(ev.sendcan[i], attr, 'new'), getattr(ev_old.sendcan[i], attr, 'old'))

      # Can
      m_old = boardd_old.can_list_to_can_capnp(can_list, 'can').to_bytes()
      # new API
      m = boardd.can_list_to_can_capnp(can_list, 'can')

      ev_old = log.Event.from_bytes(m_old)
      ev = log.Event.from_bytes(m)
      self.assertEqual(ev_old.which(), ev.which())
      self.assertEqual(len(ev.can), len(ev_old.can))
      for i in range(len(ev.can)):
        attrs = ['address', 'busTime', 'dat', 'src']
        for attr in attrs:
          self.assertEqual(getattr(ev.can[i], attr, 'new'), getattr(ev_old.can[i], attr, 'old'))

  def test_performance(self):
    can_list, cnt = generate_random_can_data_list()
    recursions = 1000

    n1 = sec_since_boot()
    for i in range(recursions):
      boardd_old.can_list_to_can_capnp(can_list, 'sendcan').to_bytes()
    n2 = sec_since_boot()
    elapsed_old = n2 - n1

    # print('Old API, elapsed time: {} secs'.format(elapsed_old))
    n1 = sec_since_boot()
    for i in range(recursions):
      boardd.can_list_to_can_capnp(can_list)
    n2 = sec_since_boot()
    elapsed_new = n2 - n1
    # print('New API, elapsed time: {} secs'.format(elapsed_new))
    self.assertTrue(elapsed_new < elapsed_old / 2)


if __name__ == '__main__':
    unittest.main()
