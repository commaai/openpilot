#!/usr/bin/env python3
import os
import unittest
from PIL import Image, ImageDraw, ImageFont

from cereal import log
from common.basedir import BASEDIR
from selfdrive.controls.lib.alerts import ALERTS

AlertSize = log.ControlsState.AlertSize

FONT_PATH = os.path.join(BASEDIR, "selfdrive/assets/fonts")
REGULAR_FONT_PATH = os.path.join(FONT_PATH, "opensans_semibold.ttf")
BOLD_FONT_PATH = os.path.join(FONT_PATH, "opensans_semibold.ttf")
SEMIBOLD_FONT_PATH = os.path.join(FONT_PATH, "opensans_semibold.ttf")

MAX_TEXT_WIDTH = 1920 - 300 # full screen width is useable, minus sidebar
# TODO: get exact scale factor. found this empirically, works well enough
FONT_SIZE_SCALE = 1.85 # factor to scale from nanovg units to PIL

class TestAlerts(unittest.TestCase):

  # ensure alert text doesn't exceed allowed width
  def test_alert_length(self):
    img = Image.new('RGB', (1920, 1080), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    fonts = {
            AlertSize.small: [ImageFont.truetype(SEMIBOLD_FONT_PATH, int(40*FONT_SIZE_SCALE))],
            AlertSize.mid: [ImageFont.truetype(BOLD_FONT_PATH, int(48*FONT_SIZE_SCALE)),
                              ImageFont.truetype(REGULAR_FONT_PATH, int(36*FONT_SIZE_SCALE))],
            AlertSize.full: [ImageFont.truetype(BOLD_FONT_PATH, int(96*FONT_SIZE_SCALE)),
                              ImageFont.truetype(REGULAR_FONT_PATH, int(48*FONT_SIZE_SCALE))],
            }

    for alert in ALERTS:
      # TODO: test full size alerts
      if alert.alert_size in [AlertSize.none, AlertSize.full]:
        continue


      for i, txt in enumerate([alert.alert_text_1, alert.alert_text_2]):
        if i >= len(fonts[alert.alert_size]): break

        font = fonts[alert.alert_size][i]
        w, h = draw.textsize(txt, font)
        msg = "type: %s msg: %s" % (alert.alert_type, txt)
        self.assertLessEqual(w, MAX_TEXT_WIDTH, msg=msg)

if __name__ == "__main__":
  unittest.main()
