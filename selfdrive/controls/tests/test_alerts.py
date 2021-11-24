#!/usr/bin/env python3
import json
import os
import unittest
import random
from PIL import Image, ImageDraw, ImageFont

from cereal import log, car
from common.basedir import BASEDIR
from common.params import Params
from selfdrive.controls.lib.events import Alert, EVENTS
from selfdrive.controls.lib.alertmanager import set_offroad_alert

AlertSize = log.ControlsState.AlertSize

OFFROAD_ALERTS_PATH = os.path.join(BASEDIR, "selfdrive/controls/lib/alerts_offroad.json")

# TODO: add callback alerts
ALERTS = []
for event_types in EVENTS.values():
  for alert in event_types.values():
    if isinstance(alert, Alert):
      ALERTS.append(alert)


class TestAlerts(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    with open(OFFROAD_ALERTS_PATH) as f:
      cls.offroad_alerts = json.loads(f.read())

  def test_events_defined(self):
    # Ensure all events in capnp schema are defined in events.py
    events = car.CarEvent.EventName.schema.enumerants

    for name, e in events.items():
      if not name.endswith("DEPRECATED"):
        fail_msg = "%s @%d not in EVENTS" % (name, e)
        self.assertTrue(e in EVENTS.keys(), msg=fail_msg)

  # ensure alert text doesn't exceed allowed width
  def test_alert_text_length(self):
    font_path = os.path.join(BASEDIR, "selfdrive/assets/fonts")
    regular_font_path = os.path.join(font_path, "opensans_semibold.ttf")
    bold_font_path = os.path.join(font_path, "opensans_semibold.ttf")
    semibold_font_path = os.path.join(font_path, "opensans_semibold.ttf")

    max_text_width = 1920 - 300  # full screen width is useable, minus sidebar
    # TODO: get exact scale factor. found this empirically, works well enough
    font_scale_factor = 1.85  # factor to scale from nanovg units to PIL

    draw = ImageDraw.Draw(Image.new('RGB', (0, 0)))

    fonts = {
      AlertSize.small: [ImageFont.truetype(semibold_font_path, int(40 * font_scale_factor))],
      AlertSize.mid: [ImageFont.truetype(bold_font_path, int(48 * font_scale_factor)),
                      ImageFont.truetype(regular_font_path, int(36 * font_scale_factor))],
    }

    for alert in ALERTS:
      # for full size alerts, both text fields wrap the text,
      # so it's unlikely that they  would go past the max width
      if alert.alert_size in [AlertSize.none, AlertSize.full]:
        continue

      for i, txt in enumerate([alert.alert_text_1, alert.alert_text_2]):
        if i >= len(fonts[alert.alert_size]):
          break

        font = fonts[alert.alert_size][i]
        w, _ = draw.textsize(txt, font)
        msg = "type: %s msg: %s" % (alert.alert_type, txt)
        self.assertLessEqual(w, max_text_width, msg=msg)

  def test_alert_sanity_check(self):
    for a in ALERTS:
      if a.alert_size == AlertSize.none:
        self.assertEqual(len(a.alert_text_1), 0)
        self.assertEqual(len(a.alert_text_2), 0)
      elif a.alert_size == AlertSize.small:
        self.assertGreater(len(a.alert_text_1), 0)
        self.assertEqual(len(a.alert_text_2), 0)
      elif a.alert_size == AlertSize.mid:
        self.assertGreater(len(a.alert_text_1), 0)
        self.assertGreater(len(a.alert_text_2), 0)
      else:
        self.assertGreater(len(a.alert_text_1), 0)

      self.assertGreaterEqual(a.duration, 0.)

  def test_offroad_alerts(self):
    params = Params()
    for a in self.offroad_alerts:
      # set the alert
      alert = self.offroad_alerts[a]
      set_offroad_alert(a, True)
      self.assertTrue(json.dumps(alert) == params.get(a, encoding='utf8'))

      # then delete it
      set_offroad_alert(a, False)
      self.assertTrue(params.get(a) is None)

  def test_offroad_alerts_extra_text(self):
    params = Params()
    for i in range(50):
      # set the alert
      a = random.choice(list(self.offroad_alerts))
      alert = self.offroad_alerts[a]
      set_offroad_alert(a, True, extra_text="a"*i)

      expected_txt = alert['text'] + "a"*i
      written_txt = json.loads(params.get(a, encoding='utf8'))['text']
      self.assertTrue(expected_txt == written_txt)

if __name__ == "__main__":
  unittest.main()
