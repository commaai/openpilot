import random

from openpilot.selfdrive.selfdrived.events import Alert, EmptyAlert, EVENTS
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager


class TestAlertManager:

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

      # check two cases:
      # - alert is added to AM for <= the alert's duration
      # - alert is added to AM for > alert's duration
      for greater in (True, False):
        if greater:
          add_duration = duration + random.randint(1, 10)
        else:
          add_duration = random.randint(1, duration)
        show_duration = max(duration, add_duration)

        AM = AlertManager()
        for frame in range(duration+10):
          if frame < add_duration:
            AM.add_many(frame, [alert, ])
          AM.process_alerts(frame, set())

          shown = AM.current_alert != EmptyAlert
          should_show = frame <= show_duration
          assert shown == should_show, f"{frame=} {add_duration=} {duration=}"

      # check one case:
      # - if alert is re-added to AM before it ends the duration is extended
      if duration > 1:
        AM = AlertManager()
        show_duration = duration * 2
        for frame in range(duration * 2 + 10):
          if frame == 0:
            AM.add_many(frame, [alert, ])

          if frame == duration:
            # add alert one frame before it ends
            assert AM.current_alert == alert
            AM.add_many(frame, [alert, ])
          AM.process_alerts(frame, set())

          shown = AM.current_alert != EmptyAlert
          should_show = frame <= show_duration
          assert shown == should_show, f"{frame=} {duration=}"
