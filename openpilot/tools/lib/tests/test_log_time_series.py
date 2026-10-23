import unittest

import numpy as np

from openpilot.cereal import log
from openpilot.tools.lib.log_time_series import msgs_to_time_series


class TestLogTimeSeries(unittest.TestCase):
  def test_nested_fields_and_sorting(self):
    msgs = []
    for t in (3, 1, 2):
      msg = log.Event.new_message(logMonoTime=t * 10**9, valid=t != 2)
      msg.init('carState')
      msg.carState.vEgo = float(t)
      msg.carState.cruiseState.speed = float(t * 10)
      msgs.append(msg.as_reader())

    ts = msgs_to_time_series(iter(msgs))['carState']
    np.testing.assert_array_equal(ts['t'], [1., 2., 3.])
    np.testing.assert_array_equal(ts['vEgo'], [1., 2., 3.])
    np.testing.assert_array_equal(ts['cruiseState/speed'], [10., 20., 30.])
    np.testing.assert_array_equal(ts['_valid'], [True, False, True])
    self.assertIn('aEgo', ts)  # unset fields are included

  def test_list_fields(self):
    for lengths in ((2, 2), (0, 2, 1)):
      with self.subTest(lengths=lengths):
        msgs = []
        for i, length in enumerate(lengths):
          msg = log.Event.new_message(logMonoTime=i)
          msg.init('carState')
          msg.carState.init('buttonEvents', length)
          msgs.append(msg.as_reader())
        events = msgs_to_time_series(msgs)['carState']['buttonEvents']
        self.assertEqual(len(events), len(lengths))
        for row, length in zip(events, lengths, strict=True):
          self.assertIsInstance(row, np.ndarray)
          self.assertEqual(len(row), length)
          for event in row:
            self.assertEqual(event, {'pressed': False, 'type': 'unknown'})

  def test_enums_and_nested_lists(self):
    msg = log.Event.new_message()
    msg.init('lateralTorqueParameters')
    msg.lateralTorqueParameters.points = [[1., 2.], [3., 4.]]
    ts = msgs_to_time_series([msg.as_reader()])['lateralTorqueParameters']
    np.testing.assert_array_equal(ts['points'], [[[1., 2.], [3., 4.]]])

    msg = log.Event.new_message()
    msg.init('carState')
    msg.carState.gearShifter = 'drive'
    ts = msgs_to_time_series([msg.as_reader()])['carState']
    np.testing.assert_array_equal(ts['gearShifter'], ['drive'])

  def test_union_group(self):
    msg = log.Event.new_message()
    msg.init('controlsState')
    msg.controlsState.lateralControlState.init('torqueState')
    msg.controlsState.lateralControlState.torqueState.output = 0.5
    ts = msgs_to_time_series([msg.as_reader()])['controlsState']
    np.testing.assert_array_equal(ts['lateralControlState/torqueState/output'], [0.5])
    self.assertFalse(any('pidState' in key for key in ts))

  def test_skipped_services(self):
    msgs = []
    for service in ('qcomGnss', 'ubloxGnss', 'ubloxRaw'):
      msg = log.Event.new_message()
      if service == 'ubloxRaw':
        msg.ubloxRaw = b'raw'
      else:
        msg.init(service)
      msgs.append(msg.as_reader())
    self.assertEqual(msgs_to_time_series(msgs), {})
    self.assertEqual(msgs_to_time_series([]), {})


if __name__ == '__main__':
  unittest.main()
