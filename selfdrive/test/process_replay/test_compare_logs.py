#!/usr/bin/env python3
from hypothesis import given, settings, strategies as st
import unittest

from cereal import messaging
from openpilot.selfdrive.test.process_replay.compare_logs import compare_logs, format_diff, remove_ignored_fields

IGNORE_FIELDS = ["logMonoTime"]


class TestCompareLogs(unittest.TestCase):
  @staticmethod
  def _msg(which: str, data: None | dict = None):
    msg = messaging.new_message(which)
    if data is not None:
      getattr(msg, which).from_dict(data)
    return msg.as_reader()

  def test_remove_ignored_fields(self):
    msg = self._msg('carState', {'vEgo': 1.0, 'vEgoRaw': 2.0})
    stripped_msg = remove_ignored_fields(msg, IGNORE_FIELDS + ['carState.vEgo'])

    self.assertNotEqual(msg.logMonoTime, 0.0)
    self.assertEqual(msg.carState.vEgo, 1.0)

    self.assertEqual(stripped_msg.logMonoTime, 0.0)
    self.assertEqual(stripped_msg.carState.vEgo, 0.0)
    self.assertEqual(stripped_msg.carState.vEgoRaw, 2.0)

  @settings(max_examples=1000, deadline=None)
  @given(data=st.data())
  def test_fuzzy_diff(self, data):
    """compare_logs tries to format the diff nicely. check it fails for additions, removals, alignment"""

    services = ['controlsState', 'carState', 'carControl']

    ref_logs = data.draw(st.lists(st.sampled_from(services).map(lambda x: self._msg(x)), max_size=6))
    new_logs = data.draw(st.lists(st.sampled_from(services).map(lambda x: self._msg(x)), max_size=6))

    any_diff = [remove_ignored_fields(m, IGNORE_FIELDS).as_reader().to_dict(verbose=True) for m in ref_logs] != \
               [remove_ignored_fields(m, IGNORE_FIELDS).as_reader().to_dict(verbose=True) for m in new_logs]

    try:
      diff = compare_logs(ref_logs, new_logs, ignore_fields=IGNORE_FIELDS)
      _, _, failed = format_diff({"": {"": diff}}, {"": {"": {"ref": "", "new": ""}}}, {})
    except Exception:
      failed = True
    self.assertEqual(failed, any_diff, "compare_logs didn't catch diff")


if __name__ == "__main__":
  unittest.main()
