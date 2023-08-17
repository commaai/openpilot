#!/usr/bin/env python3
from parameterized import parameterized
import unittest

from cereal import log, messaging
from selfdrive.car.car_helpers import can_fingerprint
from selfdrive.car.fingerprints import _FINGERPRINTS as FINGERPRINTS


class TestCanFingerprint(unittest.TestCase):
  @parameterized.expand([(c, f) for c, f in FINGERPRINTS.items()])
  def test_can_fingerprint(self, car_model, fingerprints):
    # Tests online fingerprinting function on offline fingerprints
    for fingerprint in fingerprints:  # can have multiple fingerprints for each platform
      can = messaging.new_message('can', 1)
      can.can = [log.CanData(address=address, dat=b'\x00' * length)
                 for address, length in fingerprint.items()]

      fingerprint_iter = iter([can])
      empty_can = messaging.new_message('can', 0)
      car_fingerprint, finger = can_fingerprint(lambda: next(fingerprint_iter, empty_can))  # noqa: B023

      self.assertEqual(car_fingerprint, car_model)


if __name__ == "__main__":
  unittest.main()
