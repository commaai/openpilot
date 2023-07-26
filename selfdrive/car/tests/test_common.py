import unittest

from selfdrive.car import common_fault_avoidance

class TestCommon(unittest.TestCase):
  def _run_common_fault_avoidance(self, sequence, max_value, max_frames, cut_frames=1, request=None):
    if request is None:
      request = (1,) * len(sequence)

    history = []

    current_frames = 0
    for r, s in zip(request, sequence):
      current_frames, request = common_fault_avoidance(s, max_value, r, current_frames, max_frames, cut_frames)
      history.append(request)
    
    return history

  def _run_history_check(self, history, max_frames, cut_frames=1):
    self.assertTrue(all(history[0:15]))
    self.assertFalse(any(history[15::max_frames+cut_frames]))
    for i in range(max_frames):
      self.assertTrue(any(history[i+1::max_frames+cut_frames]))

  def test_common_fault_avoidance(self):
    # no fault for under 90
    sequence = (89,) * 50
    history = self._run_common_fault_avoidance(sequence, 90, 5)
    self.assertTrue(all(history))

    # fault after 5 frames of 90, and every 6th frame after
    sequence = (89,) * 10 + (91,) * 40
    history = self._run_common_fault_avoidance(sequence, 90, 5)
    self._run_history_check(history, 5, 1)

    # same for negative
    sequence = (-89,) * 10 + (-91,) * 40
    history = self._run_common_fault_avoidance(sequence, 90, 5)
    self._run_history_check(history, 5, 1)

    # no fault if we dip below 90 once every 5 frames
    sequence = (89,) * 10 + (91,91,91,91,89) * 8
    history = self._run_common_fault_avoidance(sequence, 90, 5)
    self.assertTrue(all(history))

    # history matches if we cut the request elsewhere once every 5 frames
    sequence = (89,) * 10 + (91,) * 40
    request = (1,1,1,1,0) * 8
    history = self._run_common_fault_avoidance(sequence, 90, 5, request=request)
    self.assertEqual(history, list(request))

    # in case you need two consecutive frames (hyundai max angle)
    sequence = (89,) * 10 + (91,) * 40
    history = self._run_common_fault_avoidance(sequence, 90, 5, 2)
    self._run_history_check(history, 5, 2)

    # zero tolerance policy (max_request_frames = 0) (subaru max angle)
    sequence = (89,) * 10 + (91,) * 40
    history = self._run_common_fault_avoidance(sequence, 90, 0)
    self.assertTrue(all(history[0:10]))
    self.assertFalse(any(history[10:]))


if __name__ == "__main__":
  unittest.main()