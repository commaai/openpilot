#!/usr/bin/env python3
from parameterized import parameterized
import unittest

from cereal import log, messaging
from openpilot.selfdrive.car.car_helpers import FRAME_FINGERPRINT, can_fingerprint
from openpilot.selfdrive.car.fingerprints import _FINGERPRINTS as FINGERPRINTS


class TestCanFingerprint(unittest.TestCase):
  @parameterized.expand(list(FINGERPRINTS.items()))
  def test_can_fingerprint(self, car_model, fingerprints):
    """Tests online fingerprinting function on offline fingerprints"""

    for fingerprint in fingerprints:  # can have multiple fingerprints for each platform
      can = messaging.new_message('can', 1)
      can.can = [log.CanData(address=address, dat=b'\x00' * length, src=src)
                 for address, length in fingerprint.items() for src in (0, 1)]

      fingerprint_iter = iter([can])
      empty_can = messaging.new_message('can', 0)
      car_fingerprint, finger = can_fingerprint(lambda: next(fingerprint_iter, empty_can))  # noqa: B023

      self.assertEqual(car_fingerprint, car_model)
      self.assertEqual(finger[0], fingerprint)
      self.assertEqual(finger[1], fingerprint)
      self.assertEqual(finger[2], {})

  def test_timing(self):
    # just pick any CAN fingerprinting car
    car_model = 'CHEVROLET BOLT EUV 2022'
    fingerprint = FINGERPRINTS[car_model][0]

    cases = []

    # case 1 - one match, make sure we keep going for 100 frames
    can = messaging.new_message('can', 1)
    can.can = [log.CanData(address=address, dat=b'\x00' * length, src=src)
               for address, length in fingerprint.items() for src in (0, 1)]
    cases.append((FRAME_FINGERPRINT, car_model, can))

    # case 2 - no matches, make sure we keep going for 100 frames
    can = messaging.new_message('can', 1)
    can.can = [log.CanData(address=1, dat=b'\x00' * 1, src=src) for src in (0, 1)]  # uncommon address
    cases.append((FRAME_FINGERPRINT, None, can))

    # case 3 - multiple matches, make sure we keep going for 200 frames to try to eliminate some
    can = messaging.new_message('can', 1)
    can.can = [log.CanData(address=2016, dat=b'\x00' * 8, src=src) for src in (0, 1)]  # common address
    cases.append((FRAME_FINGERPRINT * 2, None, can))

    for expected_frames, car_model, can in cases:
      with self.subTest(expected_frames=expected_frames, car_model=car_model):
        frames = 0

        def test():
          nonlocal frames
          frames += 1
          return can  # noqa: B023

        car_fingerprint, _ = can_fingerprint(test)
        self.assertEqual(car_fingerprint, car_model)
        self.assertEqual(frames, expected_frames + 2)  # TODO: fix extra frames


if __name__ == "__main__":
  unittest.main()
