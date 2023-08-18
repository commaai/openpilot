#!/usr/bin/env python3
from parameterized import parameterized
import unittest

from cereal import log, messaging
from selfdrive.car.car_helpers import FRAME_FINGERPRINT, can_fingerprint
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
      self.assertEqual(finger[0], fingerprint)  # we only fake bus 0

  def test_timing(self):
    # just pick any CAN fingerprinting car
    car_model = 'CHEVROLET BOLT EUV 2022'
    fingerprint = FINGERPRINTS[car_model][0]

    can = messaging.new_message('can', 1)
    can.can = [log.CanData(address=address, dat=b'\x00' * length)
               for address, length in fingerprint.items()]

    frames = 0

    def test():
      nonlocal frames
      frames += 1
      return can

    car_fingerprint, _ = can_fingerprint(test)
    # if one match, make sure we keep going for 100 frames
    self.assertEqual(frames, FRAME_FINGERPRINT + 2)  # TODO: not sure why weird offset

    # test 2 - no matches
    can = messaging.new_message('can', 1)
    can.can = [log.CanData(address=1, dat=b'\x00' * 1, src=src) for src in (0, 1)]  # weird address

    frames = 0

    def test():
      nonlocal frames
      frames += 1
      return can

    # print(fingerprint)
    print(car_model, car_fingerprint, frames)
    car_fingerprint, _ = can_fingerprint(test)
    # if no matches, make sure we keep going for 100 frames
    self.assertEqual(frames, FRAME_FINGERPRINT + 2)  # TODO: not sure why weird offset

    # test 3 - multiple matches
    can = messaging.new_message('can', 1)
    can.can = [log.CanData(address=2016, dat=b'\x00' * 8)]  # common address

    frames = 0

    def test():
      nonlocal frames
      frames += 1
      return can

    # print(fingerprint)
    print(car_model, car_fingerprint, frames)
    car_fingerprint, _ = can_fingerprint(test)
    # if multiple matches, make sure we keep going for 200 frames to try to eliminate some
    self.assertEqual(frames, FRAME_FINGERPRINT * 2 + 2)  # TODO: not sure why weird offset


if __name__ == "__main__":
  unittest.main()
