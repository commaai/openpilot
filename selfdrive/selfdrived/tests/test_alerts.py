import copy
import json
import os
import random
from PIL import Image, ImageDraw, ImageFont

from cereal import log, car
from cereal.messaging import SubMaster
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.events import Alert, EVENTS, ET
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.selfdrive.test.process_replay.process_replay import CONFIGS

AlertSize = log.SelfdriveState.AlertSize

OFFROAD_ALERTS_PATH = os.path.join(BASEDIR, "selfdrive/selfdrived/alerts_offroad.json")

# TODO: add callback alerts
ALERTS = []
for event_types in EVENTS.values():
  for alert in event_types.values():
    ALERTS.append(alert)


class TestAlerts:

  @classmethod
  def setup_class(cls):
    with open(OFFROAD_ALERTS_PATH) as f:
      cls.offroad_alerts = json.loads(f.read())

      # Create fake objects for callback
      cls.CS = car.CarState.new_message()
      cls.CP = car.CarParams.new_message()
      cfg = [c for c in CONFIGS if c.proc_name == 'controlsd'][0]
      cls.sm = SubMaster(cfg.pubs)

  def test_events_defined(self):
    # Ensure all events in capnp schema are defined in events.py
    events = car.OnroadEvent.EventName.schema.enumerants

    for name, e in events.items():
      if not name.endswith("DEPRECATED"):
        fail_msg = "%s @%d not in EVENTS" % (name, e)
        assert e in EVENTS.keys(), fail_msg

  # ensure alert text doesn't exceed allowed width
  def test_alert_text_length(self):
    font_path = os.path.join(BASEDIR, "selfdrive/assets/fonts")
    regular_font_path = os.path.join(font_path, "Inter-SemiBold.ttf")
    bold_font_path = os.path.join(font_path, "Inter-Bold.ttf")
    semibold_font_path = os.path.join(font_path, "Inter-SemiBold.ttf")

    max_text_width = 2160 - 300  # full screen width is usable, minus sidebar
    draw = ImageDraw.Draw(Image.new('RGB', (0, 0)))

    fonts = {
      AlertSize.small: [ImageFont.truetype(semibold_font_path, 74)],
      AlertSize.mid: [ImageFont.truetype(bold_font_path, 88),
                      ImageFont.truetype(regular_font_path, 66)],
    }

    for alert in ALERTS:
      if not isinstance(alert, Alert):
        alert = alert(self.CP, self.CS, self.sm, metric=False, soft_disable_time=100, personality=log.LongitudinalPersonality.standard)

      # for full size alerts, both text fields wrap the text,
      # so it's unlikely that they  would go past the max width
      if alert.alert_size in (AlertSize.none, AlertSize.full):
        continue

      for i, txt in enumerate([alert.alert_text_1, alert.alert_text_2]):
        if i >= len(fonts[alert.alert_size]):
          break

        font = fonts[alert.alert_size][i]
        left, _, right, _ = draw.textbbox((0, 0), txt, font)
        width = right - left
        msg = f"type: {alert.alert_type} msg: {txt}"
        assert width <= max_text_width, msg

  def test_alert_sanity_check(self):
    for event_types in EVENTS.values():
      for event_type, a in event_types.items():
        # TODO: add callback alerts
        if not isinstance(a, Alert):
          continue

        if a.alert_size == AlertSize.none:
          assert len(a.alert_text_1) == 0
          assert len(a.alert_text_2) == 0
        elif a.alert_size == AlertSize.small:
          assert len(a.alert_text_1) > 0
          assert len(a.alert_text_2) == 0
        elif a.alert_size == AlertSize.mid:
          assert len(a.alert_text_1) > 0
          assert len(a.alert_text_2) > 0
        else:
          assert len(a.alert_text_1) > 0

        assert a.duration >= 0.

        if event_type not in (ET.WARNING, ET.PERMANENT, ET.PRE_ENABLE):
          assert a.creation_delay == 0.

  def test_offroad_alerts(self):
    params = Params()
    for a in self.offroad_alerts:
      # set the alert
      alert = copy.copy(self.offroad_alerts[a])
      set_offroad_alert(a, True)
      alert['extra'] = ''
      assert json.dumps(alert) == params.get(a, encoding='utf8')

      # then delete it
      set_offroad_alert(a, False)
      assert params.get(a) is None

  def test_offroad_alerts_extra_text(self):
    params = Params()
    for i in range(50):
      # set the alert
      a = random.choice(list(self.offroad_alerts))
      alert = self.offroad_alerts[a]
      set_offroad_alert(a, True, extra_text="a"*i)

      written_alert = json.loads(params.get(a, encoding='utf8'))
      assert "a"*i == written_alert['extra']
      assert alert["text"] == written_alert['text']
