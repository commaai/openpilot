#!/usr/bin/env python3
import unittest
import random

from selfdrive.controls.lib.events import Alert, EVENTS
from selfdrive.controls.lib.alertmanager import AlertManager


class TestAlertManager(unittest.TestCase):

  def test_duration(self):
    """
      Enforce that an alert lasts for max(alert duration, duration the alert is added)
    """
    for duration in range(1, 100):
      alert = None
      while not isinstance(alert, Alert):
        event = random.choice([e for e in EVENTS.values() if len(e)])
        alert = random.choice(list(event.values()))

      alert.duration = duration

      # only added for <= alert duration
      AM = AlertManager()
      add_duration = random.randint(1, duration)
      for frame in range(duration+10):
        if frame < add_duration:
          AM.add_many(frame, [alert, ])
        AM.process_alerts(frame)
        self.assertEqual(AM.alert is not None, frame <= duration, msg=f"{frame=} {add_duration=} {duration=}")

      # added for > alert duration
      AM = AlertManager()
      add_duration = duration + random.randint(1, 10)
      for frame in range(add_duration+10):
        if frame < add_duration:
          AM.add_many(frame, [alert, ])
        AM.process_alerts(frame)
        self.assertEqual(AM.alert is not None, frame <= add_duration, msg=f"{frame=} {add_duration=} {duration=}")


if __name__ == "__main__":
  unittest.main()
