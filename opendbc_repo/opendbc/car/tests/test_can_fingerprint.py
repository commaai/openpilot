from parameterized import parameterized

from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import FRAME_FINGERPRINT, can_fingerprint
from opendbc.car.fingerprints import _FINGERPRINTS as FINGERPRINTS


class TestCanFingerprint:
  @parameterized.expand(list(FINGERPRINTS.items()))
  def test_can_fingerprint(self, car_model, fingerprints):
    """Tests online fingerprinting function on offline fingerprints"""

    for fingerprint in fingerprints:  # can have multiple fingerprints for each platform
      can = [CanData(address=address, dat=b'\x00' * length, src=src)
             for address, length in fingerprint.items() for src in (0, 1)]

      fingerprint_iter = iter([can])
      car_fingerprint, finger = can_fingerprint(lambda **kwargs: [next(fingerprint_iter, [])])  # noqa: B023

      assert car_fingerprint == car_model
      assert finger[0] == fingerprint
      assert finger[1] == fingerprint
      assert finger[2] == {}

  def test_timing(self, subtests):
    # just pick any CAN fingerprinting car
    car_model = "CHEVROLET_BOLT_EUV"
    fingerprint = FINGERPRINTS[car_model][0]

    cases = []

    # case 1 - one match, make sure we keep going for 100 frames
    can = [CanData(address=address, dat=b'\x00' * length, src=src)
           for address, length in fingerprint.items() for src in (0, 1)]
    cases.append((FRAME_FINGERPRINT, car_model, can))

    # case 2 - no matches, make sure we keep going for 100 frames
    can = [CanData(address=1, dat=b'\x00' * 1, src=src) for src in (0, 1)]  # uncommon address
    cases.append((FRAME_FINGERPRINT, None, can))

    # case 3 - multiple matches, make sure we keep going for 200 frames to try to eliminate some
    can = [CanData(address=2016, dat=b'\x00' * 8, src=src) for src in (0, 1)]  # common address
    cases.append((FRAME_FINGERPRINT * 2, None, can))

    for expected_frames, car_model, can in cases:
      with subtests.test(expected_frames=expected_frames, car_model=car_model):
        frames = 0

        def can_recv(**kwargs):
          nonlocal frames
          frames += 1
          return [can]  # noqa: B023

        car_fingerprint, _ = can_fingerprint(can_recv)
        assert car_fingerprint == car_model
        assert frames == expected_frames + 2  # TODO: fix extra frames
