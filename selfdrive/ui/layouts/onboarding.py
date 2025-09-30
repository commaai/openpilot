import os
import re
import time

import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app


DEBUG = True

STEP_RECTS = [(rl.Rectangle(104.0, 800.0, 737.0, 975.0)), (rl.Rectangle(1835.0, 2.0, 2159.0, 1078.0)), (rl.Rectangle(1844.0, 4.0, 2156.0, 1078.0)),
              (rl.Rectangle(1526.0, 473.0, 1953.0, 945.0)), (rl.Rectangle(1643.0, 441.0, 1860.0, 664.0)), (rl.Rectangle(1844.0, 5.0, 2155.0, 1076.0)),
              (rl.Rectangle(1786.0, 591.0, 2053.0, 827.0)), (rl.Rectangle(1353.0, 2.0, 2157.0, 1078.0)), (rl.Rectangle(1458.0, 485.0, 2091.0, 696.0)),
              (rl.Rectangle(695.0, 794.0, 1253.0, 981.0)), (rl.Rectangle(1560.0, 170.0, 1952.0, 567.0)), (rl.Rectangle(1842.0, 1.0, 2159.0, 1078.0)),
              (rl.Rectangle(1351.0, 0.0, 2158.0, 1079.0)), (rl.Rectangle(1841.0, 1.0, 2158.0, 1078.0)), (rl.Rectangle(1531.0, 82.0, 1972.0, 1002.0)),
              (rl.Rectangle(1336.0, 438.0, 1826.0, 831.0)), (rl.Rectangle(1843.0, 1.0, 2159.0, 1077.0)), (rl.Rectangle(1842.0, 2.0, 1842.0, 3.0)),
              (rl.Rectangle(612.0, 795.0, 1274.0, 981.0))]


new_step_rects = []

for rect in STEP_RECTS:
  x = rect.x
  y = rect.y
  w = rect.width
  h = rect.height
  if abs(y - 0) < 10:
    y = 0
  else:
    h = h - y

  if abs(h - 1080) < 10:
    h = 1080

  if abs(x - 1835) < 10:
    x = 1835
    width = gui_app.width - 1835
  else:
    w = w - x

  new_step_rects.append(rl.Rectangle(x, y, w, h))

STEP_RECTS = new_step_rects

STEP_RECTS = [rl.Rectangle(104.0, 800.0, 633.0, 175.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0), rl.Rectangle(1835.0, 0.0, 2156.0, 1080.0),
              rl.Rectangle(1526.0, 473.0, 427.0, 472.0), rl.Rectangle(1643.0, 441.0, 217.0, 223.0), rl.Rectangle(1835.0, 0.0, 2155.0, 1080.0),
              rl.Rectangle(1786.0, 591.0, 267.0, 236.0), rl.Rectangle(1353.0, 0.0, 804.0, 1080.0), rl.Rectangle(1458.0, 485.0, 633.0, 211.0),
              rl.Rectangle(695.0, 794.0, 558.0, 187.0), rl.Rectangle(1560.0, 170.0, 392.0, 397.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0),
              rl.Rectangle(1351.0, 0.0, 807.0, 1080.0), rl.Rectangle(1835.0, 0.0, 2158.0, 1080.0), rl.Rectangle(1531.0, 82.0, 441.0, 920.0),
              rl.Rectangle(1336.0, 438.0, 490.0, 393.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0), rl.Rectangle(1835.0, 0.0, 2159.0, 1080.0),
              rl.Rectangle(612.0, 795.0, 662.0, 186.0)]


class OnboardingDialog(Widget):
  def __init__(self):
    super().__init__()

    self._step = 0
    self._top_left_corner: rl.Vector2 = None
    self._bottom_right_corner: rl.Vector2 = None

    self._step_coords = []

    self._load_images()

  def _load_images(self):
    self._images = []
    paths = [fn for fn in os.listdir(os.path.join(BASEDIR, "selfdrive/assets/training")) if re.match(r'^step\d*\.png$', fn)]
    paths = sorted(paths, key=lambda x: int(re.search(r'\d+', x).group()))
    print(paths)
    t = time.monotonic()
    for fn in paths:
      path = os.path.join(BASEDIR, "selfdrive/assets/training", fn)
      self._images.append(gui_app.texture(path, gui_app.width, gui_app.height))
    print(f"Loaded {len(self._images)} onboarding images in {time.monotonic() - t}s")

  def _handle_mouse_release(self, mouse_pos):
    if self._step >= len(self._images):
      print("Done with onboarding")
      gui_app.set_modal_overlay(None)
      return

    if rl.check_collision_point_rec(mouse_pos, STEP_RECTS[self._step]):
      print(f"Clicked in step {self._step} rect")
      self._step += 1

    return

    if self._top_left_corner is None:
      self._top_left_corner = mouse_pos
    elif self._bottom_right_corner is None:
      self._bottom_right_corner = mouse_pos
      self._step_coords.append((self._top_left_corner, self._bottom_right_corner))
      self._top_left_corner = None
      self._bottom_right_corner = None
      self._step += 1
      print('Got mouse coords')
      print(self._step_coords)

  def _render(self, _):
    rl.draw_texture(self._images[self._step], 0, 0, rl.WHITE)

    rl.draw_rectangle_lines_ex(STEP_RECTS[self._step], 3, rl.RED)

    # if self._top_left_corner is not None:
    #   mouse_pos = rl.get_mouse_position()
    #   rl.draw_rectangle_lines_ex(rl.Rectangle(self._top_left_corner.x, self._top_left_corner.y,
    #                                           mouse_pos.x - self._top_left_corner.x,
    #                                           mouse_pos.y - self._top_left_corner.y), 3, rl.RED)

    return -1
