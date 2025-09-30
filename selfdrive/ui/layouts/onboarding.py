import os
import re
import time

import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.common.params_pyx import Params
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app


DEBUG = True

STEP_RECTS = [rl.Rectangle(104.0, 800.0, 633.0, 175.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0), rl.Rectangle(1835.0, 0.0, 2156.0, 1080.0),
              rl.Rectangle(1526.0, 473.0, 427.0, 472.0), rl.Rectangle(1643.0, 441.0, 217.0, 223.0), rl.Rectangle(1835.0, 0.0, 2155.0, 1080.0),
              rl.Rectangle(1786.0, 591.0, 267.0, 236.0), rl.Rectangle(1353.0, 0.0, 804.0, 1080.0), rl.Rectangle(1458.0, 485.0, 633.0, 211.0),
              rl.Rectangle(95.0, 794.0, 1158.0, 187.0), rl.Rectangle(1560.0, 170.0, 392.0, 397.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0),
              rl.Rectangle(1351.0, 0.0, 807.0, 1080.0), rl.Rectangle(1835.0, 0.0, 2158.0, 1080.0), rl.Rectangle(1531.0, 82.0, 441.0, 920.0),
              rl.Rectangle(1336.0, 438.0, 490.0, 393.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0),
              rl.Rectangle(612.0 - 525, 795.0, 662.0 + 525, 186.0)]


DM_RECORD_STEP = 9
DM_RECORD_YES_RECT = rl.Rectangle(695, 794, 558, 187)


class OnboardingDialog(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    self._step = 18
    self._load_images()

  def _load_images(self):
    self._images = []
    paths = [fn for fn in os.listdir(os.path.join(BASEDIR, "selfdrive/assets/training")) if re.match(r'^step\d*\.png$', fn)]
    paths = sorted(paths, key=lambda x: int(re.search(r'\d+', x).group()))
    for fn in paths:
      path = os.path.join(BASEDIR, "selfdrive/assets/training", fn)
      self._images.append(gui_app.texture(path, gui_app.width, gui_app.height))

  def _handle_mouse_release(self, mouse_pos):
    if rl.check_collision_point_rec(mouse_pos, STEP_RECTS[self._step]):
      if self._step >= len(self._images) - 1:
        self._completed_training()
        gui_app.set_modal_overlay(None)
        return

      if self._step == DM_RECORD_STEP:
        yes = rl.check_collision_point_rec(mouse_pos, DM_RECORD_YES_RECT)
        print(f"putting RecordFront to {yes}")
        self._params.put_bool("RecordFront", yes)

      self._step += 1

  def _completed_training(self):
    current_training_version = self._params.get("TrainingVersion")
    self._params.put("CompletedTrainingVersion", current_training_version)

  def _render(self, _):
    rl.draw_texture(self._images[self._step], 0, 0, rl.WHITE)

    if DEBUG:
      rl.draw_rectangle_lines_ex(STEP_RECTS[self._step], 3, rl.RED)

    return -1
