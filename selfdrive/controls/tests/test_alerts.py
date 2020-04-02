#!/usr/bin/env python3
import os
import unittest
from PIL import Image, ImageDraw, ImageFont

from cereal import log
from common.basedir import BASEDIR
from selfdrive.controls.lib.alerts import ALERTS

AlertSize = log.ControlsState.AlertSize

MAX_TEXT_WIDTH = 1920 - 300 # full screen width is useable, minus sidebar
FONT_PATH = os.path.join(BASEDIR, "selfdrive/assets/fonts")

class TestAlerts(unittest.TestCase):

  # ensure alert text doesn't exceed allowed width
  def test_alert_length(self):
    img = Image.new('RGB', (1920, 1080), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # TODO: nanovg units are display units. PIL uses points
    fonts = {
            AlertSize.small: [ImageFont.truetype(FONT_PATH + "/opensans_semibold.ttf", 40)],
            AlertSize.mid: [ImageFont.truetype(FONT_PATH + "/opensans_bold.ttf", 48),
                              ImageFont.truetype(FONT_PATH + "/opensans_regular.ttf", 36)],
            AlertSize.full: [ImageFont.truetype(FONT_PATH + "/opensans_bold.ttf", 96),
                              ImageFont.truetype(FONT_PATH + "/opensans_regular.ttf", 48)],
            }

    for alert in ALERTS:
      if alert.alert_size == AlertSize.none:
        continue

      font = fonts[alert.alert_size][0]
      if alert.alert_size == AlertSize.full and len(alert.alert_text_1) > 15:
        font = ImageFont.truetype(FONT_PATH + "/opensans_bold.ttf", 72)

      w, h = draw.textsize(alert.alert_text_1, font)
      #print(alert.alert_type, w)
      self.assertLessEqual(w, MAX_TEXT_WIDTH, msg=alert.alert_text_1)

      if alert.alert_size != AlertSize.small:
        font = fonts[alert.alert_size][1]
        w, h = draw.textsize(alert.alert_text_2, font)
        self.assertLessEqual(w, MAX_TEXT_WIDTH, msg=alert.alert_text_2)

if __name__ == "__main__":
  unittest.main()
