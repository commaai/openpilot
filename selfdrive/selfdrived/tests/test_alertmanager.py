import random

from openpilot.selfdrive.selfdrived.events import Alert, EmptyAlert, EVENTS
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager


class TestAlertManager:

  def test_duration(self):
    """
      Enforce that an alert lasts for max(alert duration, duration the alert is added)
    """
    for duration in range(1, 100):
      print('duration', duration)
      duration = 5
      alert = None
      while not isinstance(alert, Alert):
        event = random.choice([e for e in EVENTS.values() if len(e)])
        alert = random.choice(list(event.values()))

      alert.duration = duration

      # check one case:
      # - if alert is re-added to AM before it ends the duration is extended
      # if duration > 1:
      #   AM = AlertManager()
      #   show_duration = duration * 2
      #   for frame in range(duration * 2 + 10):
      #     print(frame)
      #     if frame == 0:
      #       AM.add_many(frame, [alert, ])
      #       print('added!')
      #
      #     if frame == duration:
      #       # add alert one frame before it ends
      #       assert AM.current_alert == alert
      #       AM.add_many(frame, [alert, ])
      #       print('added!')
      #     AM.process_alerts(frame, set())
      #     print('alert', AM.current_alert)
      #
      #     shown = AM.current_alert != EmptyAlert
      #     should_show = frame <= show_duration
      #     print('shown', shown, 'should_show', should_show)
      #     print()
      #     assert shown == should_show, f"{frame=} {duration=}"
      # break

      # check two cases:
      # - alert is added to AM for <= the alert's duration
      # - alert is added to AM for > alert's duration
      for greater in (True, False):
        print('greater', greater)
        if greater:
          add_duration = duration + random.randint(1, 10)
        else:
          add_duration = random.randint(1, duration)
        show_duration = max(duration, add_duration)
        print('add_duration', add_duration, 'show_duration', show_duration)

        AM = AlertManager()
        for frame in range(duration+10):
          print('frame', frame)
          if frame < add_duration:
            print('added!')
            AM.add_many(frame, [alert, ])
          AM.process_alerts(frame, set())
          print(AM.current_alert)

          shown = AM.current_alert != EmptyAlert
          should_show = frame <= show_duration
          print('shown', shown, 'should_show', should_show)
          assert shown == should_show, f"{frame=} {add_duration=} {duration=}"
          print()
