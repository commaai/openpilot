import os
import re
import time

import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app


class OnboardingDialog(Widget):
  def __init__(self):
    super().__init__()

    self._step = 0

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

  def _render(self, _):
    rl.draw_texture(self._images[self._step], 0, 0, rl.WHITE)
