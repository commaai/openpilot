import os
import re
import time  # noqa: F401

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


# --- Added: TermsPage (Python rewrite of C++ TermsPage) ---
from openpilot.system.ui.lib.text_measure import measure_text_cached
class TermsPage(Widget):
  def __init__(self, on_accept=None, on_decline=None):
    super().__init__()
    self._on_accept = on_accept
    self._on_decline = on_decline

    # Layout constants roughly matching the C++ styling
    self._outer_margin = (45, 35, 45, 45)  # left, top, right, bottom
    self._content_margins = (165, 165, 165, 0)
    self._spacing = 90

    # Colors
    self._primary = rl.Color(70, 91, 234, 255)
    self._primary_pressed = rl.Color(48, 73, 244, 255)
    self._default_btn_bg = rl.Color(79, 79, 79, 255)
    self._text_color = rl.WHITE

    # Fonts
    self._title_font = gui_app.font()
    self._desc_font = gui_app.font()
    self._title_size = 90
    self._desc_size = 80

    # Buttons state
    self._pressed_accept = False
    self._pressed_decline = False

  def _update_layout_rects(self):
    # Compute layout rects based on current widget rect
    x, y, w, h = self._rect.x, self._rect.y, self._rect.width, self._rect.height

    lm, tm, rm, bm = self._outer_margin
    content_rect = rl.Rectangle(x + lm, y + tm, w - (lm + rm), h - (tm + bm))

    cl, ct, cr, cb = self._content_margins
    main_rect = rl.Rectangle(content_rect.x + cl, content_rect.y + ct,
                             content_rect.width - (cl + cr), content_rect.height - (ct + cb))

    # Title rect (align top-left, height derived from font size)
    self._title_rect = rl.Rectangle(main_rect.x, main_rect.y, main_rect.width, self._title_size + 10)

    # Description rect below title with spacing
    desc_y = self._title_rect.y + self._title_rect.height + self._spacing
    # Allow multi-line; give it generous height
    self._desc_rect = rl.Rectangle(main_rect.x, desc_y, main_rect.width, main_rect.height - (self._title_rect.height + self._spacing + 250))

    # Buttons row at the bottom with spacing 45 and fixed height 160
    buttons_spacing = 45
    buttons_height = 160
    buttons_y = content_rect.y + content_rect.height - buttons_height

    # Two equal width buttons split the content area
    total_spacing = buttons_spacing
    btn_width = (content_rect.width - total_spacing) / 2
    self._decline_rect = rl.Rectangle(content_rect.x, buttons_y, btn_width, buttons_height)
    self._accept_rect = rl.Rectangle(content_rect.x + btn_width + buttons_spacing, buttons_y, btn_width, buttons_height)

  def _handle_mouse_press(self, mouse_pos):
    if rl.check_collision_point_rec(mouse_pos, self._accept_rect):
      self._pressed_accept = True
    if rl.check_collision_point_rec(mouse_pos, self._decline_rect):
      self._pressed_decline = True
    return True

  def _handle_mouse_release(self, mouse_pos):
    accept_clicked = self._pressed_accept and rl.check_collision_point_rec(mouse_pos, self._accept_rect)
    decline_clicked = self._pressed_decline and rl.check_collision_point_rec(mouse_pos, self._decline_rect)
    self._pressed_accept = False
    self._pressed_decline = False

    if accept_clicked and self._on_accept:
      self._on_accept()
    elif decline_clicked and self._on_decline:
      self._on_decline()
    return True

  def _render(self, _):
    # Background handled by app; draw content
    # Title
    title = "Welcome to openpilot"
    rl.draw_text_ex(self._title_font, title,
                    rl.Vector2(self._title_rect.x, self._title_rect.y),
                    self._title_size, 0, self._text_color)

    # Description (no HTML span; we mimic color by including the link as plain text)
    desc = ("You must accept the Terms and Conditions to use openpilot. " +
            "Read the latest terms at https://comma.ai/terms before continuing.")

    # Render wrapped description
    self._render_wrapped_text(self._desc_rect, desc, self._desc_font, self._desc_size, self._text_color)

    # Buttons
    self._draw_button(self._decline_rect, "Decline", self._default_btn_bg)
    self._draw_button(self._accept_rect, "Agree", self._primary, pressed=self._pressed_accept, pressed_bg=self._primary_pressed)

    return -1

  def _draw_button(self, rect: rl.Rectangle, text: str, bg: rl.Color, pressed: bool = False, pressed_bg: rl.Color | None = None):
    roundness = 10 / (min(rect.width, rect.height) / 2)
    color = pressed_bg if (pressed and pressed_bg is not None) else bg
    rl.draw_rectangle_rounded(rect, roundness, 20, color)

    # Center text
    size = measure_text_cached(self._desc_font, text, 55)
    rl.draw_text_ex(self._desc_font,
                    text,
                    rl.Vector2(rect.x + (rect.width - size.x) / 2, rect.y + (rect.height - size.y) / 2),
                    55, 0, rl.WHITE)

  def _render_wrapped_text(self, rect: rl.Rectangle, text: str, font, font_size: int, color: rl.Color):
    # Simple word wrap rendering using raylib helpers
    words = text.split(" ")
    x = rect.x
    y = rect.y
    line = ""
    space_width = measure_text_cached(font, " ", font_size).x
    for word in words:
      w_size = measure_text_cached(font, word, font_size)
      l_size = measure_text_cached(font, line, font_size) if line else rl.Vector2(0, 0)
      if l_size.x + (space_width if line else 0) + w_size.x > rect.width:
        # draw current line
        rl.draw_text_ex(font, line, rl.Vector2(x, y), font_size, 0, color)
        y += w_size.y
        line = word
      else:
        line = word if not line else f"{line} {word}"

      # stop if exceeding height
      if y > rect.y + rect.height - w_size.y:
        break

    if line and y <= rect.y + rect.height:
      rl.draw_text_ex(font, line, rl.Vector2(x, y), font_size, 0, color)
