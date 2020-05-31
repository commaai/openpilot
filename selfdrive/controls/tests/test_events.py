#!/usr/bin/env python3
import os
import unittest
from PIL import Image, ImageDraw, ImageFont

from cereal import log, car
from common.basedir import BASEDIR
from selfdrive.controls.lib.events import Alert, EVENTS

AlertSize = log.ControlsState.AlertSize

class TestAlerts(unittest.TestCase):

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

    max_text_width = 1920 - 300 # full screen width is useable, minus sidebar
    # TODO: get exact scale factor. found this empirically, works well enough
    font_scale_factor = 1.85 # factor to scale from nanovg units to PIL

    draw = ImageDraw.Draw(Image.new('RGB', (0, 0)))

    fonts = {
      AlertSize.small: [ImageFont.truetype(semibold_font_path, int(40*font_scale_factor))],
      AlertSize.mid: [ImageFont.truetype(bold_font_path, int(48*font_scale_factor)),
                      ImageFont.truetype(regular_font_path, int(36*font_scale_factor))],
    }

    alerts = []
    for event_types in EVENTS.values():
      for alert in event_types.values():
        if isinstance(alert, Alert):
          alerts.append(alert)

    for alert in alerts:
      # for full size alerts, both text fields wrap the text,
      # so it's unlikely that they  would go past the max width
      if alert.alert_size in [AlertSize.none, AlertSize.full]:
        continue

      for i, txt in enumerate([alert.alert_text_1, alert.alert_text_2]):
        if i >= len(fonts[alert.alert_size]):
          break

        font = fonts[alert.alert_size][i]
        w, h = draw.textsize(txt, font)
        msg = "type: %s msg: %s" % (alert.alert_type, txt)
        self.assertLessEqual(w, max_text_width, msg=msg)

if __name__ == "__main__":
  unittest.main()
