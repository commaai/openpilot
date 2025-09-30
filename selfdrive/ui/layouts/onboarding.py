import os
import re
import time  # noqa: F401

import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.common.params_pyx import Params as _Params
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

RESTART_TRAINING_RECT = rl.Rectangle(612.0 - 525, 795.0, 662.0 - 190, 186.0)


class OnboardingDialog(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    self._step = 0
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
      if self._step == DM_RECORD_STEP:
        yes = rl.check_collision_point_rec(mouse_pos, DM_RECORD_YES_RECT)
        print(f"putting RecordFront to {yes}")
        self._params.put_bool("RecordFront", yes)

      elif self._step == len(self._images) - 1:
        if rl.check_collision_point_rec(mouse_pos, RESTART_TRAINING_RECT):
          self._step = -1

      self._step += 1

      if self._step >= len(self._images):
        self._completed_training()
        gui_app.set_modal_overlay(None)
        return

  def _completed_training(self):
    self._step = 0
    current_training_version = self._params.get("TrainingVersion")
    self._params.put("CompletedTrainingVersion", current_training_version)

  def _render(self, _):
    rl.draw_texture(self._images[self._step], 0, 0, rl.WHITE)

    # progress bar
    if 0 < self._step < len(STEP_RECTS) - 1:
      h = 20
      w = int((self._step / (len(STEP_RECTS) - 1)) * self._rect.width)
      rl.draw_rectangle(int(self._rect.x), int(self._rect.y + self._rect.height - h),
                        w, h, rl.Color(70, 91, 234, 255))

    if DEBUG:
      rl.draw_rectangle_lines_ex(STEP_RECTS[self._step], 3, rl.RED)

    return -1


from openpilot.system.ui.lib.text_measure import measure_text_cached  # noqa: F401
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import Label, TextAlignment
from openpilot.common.params_pyx import Params

# --- Added: TermsPage (Python rewrite of C++ TermsPage) using Button/Label ---
class TermsPage(Widget):
  def __init__(self, on_accept=None, on_decline=None):
    super().__init__()
    self._on_accept = on_accept
    self._on_decline = on_decline

    # Layout constants roughly matching the C++ styling
    self._outer_margin = (45, 35, 45, 45)  # left, top, right, bottom
    self._content_margins = (165, 165, 165, 0)
    self._spacing = 90

    # Widgets
    self._title = Label("Welcome to openpilot", font_size=90, font_weight=FontWeight.MEDIUM,
                        text_alignment=TextAlignment.LEFT)
    desc_text = ("You must accept the Terms and Conditions to use openpilot. " +
                 "Read the latest terms at https://comma.ai/terms before continuing.")
    self._desc = Label(desc_text, font_size=80, font_weight=FontWeight.LIGHT,
                       text_alignment=TextAlignment.LEFT)

    self._decline_btn = Button("Decline", click_callback=self._on_decline_clicked)
    self._accept_btn = Button("Agree", button_style=ButtonStyle.PRIMARY,
                              click_callback=self._on_accept_clicked)

  def _on_accept_clicked(self):
    if self._on_accept:
      self._on_accept()

  def _on_decline_clicked(self):
    if self._on_decline:
      self._on_decline()

  def _update_layout_rects(self):
    # Compute layout rects based on current widget rect
    x, y, w, h = self._rect.x, self._rect.y, self._rect.width, self._rect.height

    lm, tm, rm, bm = self._outer_margin
    content_rect = rl.Rectangle(x + lm, y + tm, w - (lm + rm), h - (tm + bm))

    cl, ct, cr, cb = self._content_margins
    main_rect = rl.Rectangle(content_rect.x + cl, content_rect.y + ct,
                             content_rect.width - (cl + cr), content_rect.height - (ct + cb))

    # Title
    title_rect = rl.Rectangle(main_rect.x, main_rect.y, main_rect.width, 110)
    self._title.render(title_rect)

    # Description below title
    desc_y = title_rect.y + title_rect.height + self._spacing
    desc_rect = rl.Rectangle(main_rect.x, desc_y, main_rect.width, main_rect.height - (title_rect.height + self._spacing + 250))
    self._desc.render(desc_rect)

    # Buttons row at the bottom with spacing 45 and fixed height 160
    buttons_spacing = 45
    buttons_height = 160
    buttons_y = content_rect.y + content_rect.height - buttons_height

    total_spacing = buttons_spacing
    btn_width = (content_rect.width - total_spacing) / 2
    decline_rect = rl.Rectangle(content_rect.x, buttons_y, btn_width, buttons_height)
    accept_rect = rl.Rectangle(content_rect.x + btn_width + buttons_spacing, buttons_y, btn_width, buttons_height)

    self._decline_btn.render(decline_rect)
    self._accept_btn.render(accept_rect)

  def _render(self, _):
    # Rendering handled by sub-widgets via render() calls in _update_layout_rects
    return -1


# --- Onboarding helpers ---
def completed(params: _Params | None = None) -> bool:
  p = params or _Params()
  current_terms_version = p.get("TermsVersion")
  current_training_version = p.get("TrainingVersion")
  accepted_terms = p.get("HasAcceptedTerms") == current_terms_version
  training_done = p.get("CompletedTrainingVersion") == current_training_version
  return accepted_terms and training_done


def show_training_guide():
  gui_app.set_modal_overlay(OnboardingDialog())


def maybe_show_onboarding():
  p = _Params()
  current_terms_version = p.get("TermsVersion")
  current_training_version = p.get("TrainingVersion")
  accepted_terms = p.get("HasAcceptedTerms") == current_terms_version
  training_done = p.get("CompletedTrainingVersion") == current_training_version

  if not accepted_terms:
    def _on_accept():
      p.put("HasAcceptedTerms", current_terms_version)
      if p.get("CompletedTrainingVersion") != current_training_version:
        show_training_guide()

    def _on_decline():
      p.put_bool("DoUninstall", True)

    gui_app.set_modal_overlay(TermsPage(on_accept=_on_accept, on_decline=_on_decline))
  elif not training_done:
    show_training_guide()
