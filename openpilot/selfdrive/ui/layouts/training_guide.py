import os
import threading
from collections.abc import Callable
from typing import NamedTuple

import raylib_nb as rl
from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button
from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.onroad.alert_renderer import (ALERT_BORDER_RADIUS, ALERT_COLORS, ALERT_FONT_BIG, ALERT_FONT_SMALL, ALERT_HEIGHTS,
                                                          ALERT_LINE_SPACING, ALERT_MARGIN, ALERT_PADDING, AlertSize, AlertStatus)
from openpilot.selfdrive.ui.onroad.augmented_road_view import BORDER_COLORS
from openpilot.selfdrive.ui.onroad.hud_renderer import COLORS, FONT_SIZES, UI_CONFIG
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus

ASSETS = "training"
MARGIN = 112
PHOTO_X = 1350

TITLE_SIZE = 78
TITLE_LINE = 98
BODY_SIZE = 55
BODY_LINE = 73
BODY_GAP = 48
INTRO_SIZE = 62
INTRO_LINE = 90
ITEM_SIZE = 59

# step background, next button
NAVY = (rl.Color(11, 16, 55, 255), rl.Color(70, 91, 234, 255))
GREEN = (rl.Color(7, 104, 58, 255), rl.Color(35, 154, 97, 255))
TEAL = (rl.Color(20, 71, 65, 255), rl.Color(37, 163, 149, 255))
GRAY = (rl.Color(40, 40, 40, 255), rl.Color(101, 101, 101, 255))
BLUE = (rl.Color(9, 76, 90, 255), rl.Color(4, 140, 170, 255))
RED = (rl.Color(154, 35, 35, 255), rl.Color(220, 56, 56, 255))

# tap targets
NEXT_RECT = rl.Rectangle(1840, 0, 320, 1080)
BACK_RECT = rl.Rectangle(0, 0, 200, MARGIN)
BEGIN_RECT = rl.Rectangle(MARGIN, 804, 618, 163)
DM_NO_RECT = rl.Rectangle(MARGIN, 804, 530, 163)
DM_YES_RECT = rl.Rectangle(706, 804, 530, 163)
RESTART_RECT = rl.Rectangle(MARGIN, 804, 422, 163)
FINISH_RECT = rl.Rectangle(630, 804, 626, 163)
PATH_RECT = rl.Rectangle(1526, 473, 427, 472)
LEAD_RECT = rl.Rectangle(1643, 441, 217, 223)
SET_RECT = rl.Rectangle(1786, 591, 267, 236)
DRIVER_RECT = rl.Rectangle(1353, 0, 804, 1080)
ALERT_RECT = rl.Rectangle(1458, 485, 633, 211)
MAX_SPEED_RECT = rl.Rectangle(1560, 170, 392, 397)
WHEEL_RECT = rl.Rectangle(1351, 0, 807, 1080)
LIGHT_RECT = rl.Rectangle(1531, 82, 441, 920)
PEDAL_RECT = rl.Rectangle(1336, 438, 490, 393)

# what goes to the next step, for tests
STEP_RECTS = [BEGIN_RECT, NEXT_RECT, NEXT_RECT, PATH_RECT, LEAD_RECT, NEXT_RECT, SET_RECT, DRIVER_RECT, ALERT_RECT, DM_NO_RECT,
              MAX_SPEED_RECT, NEXT_RECT, WHEEL_RECT, NEXT_RECT, LIGHT_RECT, PEDAL_RECT, NEXT_RECT, NEXT_RECT, FINISH_RECT]


def _shade(color: rl.Color, factor: float) -> rl.Color:
  """factor > 0 lightens towards white, < 0 darkens towards black"""
  def ch(c):
    return int(c + (255 - c) * factor) if factor > 0 else int(c * (1 + factor))
  return rl.Color(ch(color.r), ch(color.g), ch(color.b), color.a)


class Span(NamedTuple):
  text: str
  bold: bool = False
  underline: bool = False


Paragraph = str | Span | tuple[str | Span, ...]


def _font(bold: bool) -> rl.Font:
  return gui_app.font(FontWeight.BOLD if bold else FontWeight.NORMAL)


def _wrap(paragraph: Paragraph, font_size: int, max_width: float) -> list[list[tuple[str, Span]]]:
  """Word wrap a paragraph of space separated styled spans into lines of (text, span) runs."""
  parts = paragraph if isinstance(paragraph, tuple) and not isinstance(paragraph, Span) else (paragraph,)
  spans = [Span(p) if isinstance(p, str) else p for p in parts]
  words = [(word, span) for span in spans for word in tr(span.text).split()]

  def width(text: str, span: Span) -> float:
    return measure_text_cached(_font(span.bold), text, font_size).x

  lines: list[list[tuple[str, Span]]] = []
  line: list[tuple[str, Span]] = []
  line_width = 0.
  for text, span in words:
    w = width(text, span)
    if line:
      # spaces between words of one span take its style, e.g. continuous underline
      space = (" ", span if line[-1][1] is span else Span(" "))
      if line_width + width(*space) + w <= max_width:
        line += [space, (text, span)]
        line_width += width(*space) + w
        continue
      lines.append(line)
    line = [(text, span)]
    line_width = w
  if line:
    lines.append(line)
  return lines


def draw_paragraphs(paragraphs: tuple[Paragraph, ...], x: float, y: float, max_width: float, font_size: int,
                    line_height: float, paragraph_gap: float = 0) -> float:
  """Draws left aligned paragraphs top down, returns the y below the last line."""
  for paragraph in paragraphs:
    for line in _wrap(paragraph, font_size, max_width):
      run_x = x
      for text, span in line:
        size = measure_text_cached(_font(span.bold), text, font_size)
        rl.draw_text_ex(_font(span.bold), text, rl.Vector2(run_x, y), font_size, 0, rl.WHITE)
        if span.underline:
          rl.draw_rectangle_rec(rl.Rectangle(run_x, y + font_size * 1.05, size.x, 4), rl.WHITE)
        run_x += size.x
      y += line_height
    y += paragraph_gap
  return y - paragraph_gap


def draw_distracted_screen(origin: rl.Vector2, photo: rl.Texture):
  """Redraw the device screen in the photo as the current onroad UI showing the distracted alert"""
  screen = rl.Rectangle(origin.x + 1542, origin.y + 388, 465, 235)
  w, h = gui_app.width, gui_app.height
  rl.rl_push_matrix()
  rl.rl_translatef(screen.x, screen.y, 0)
  rl.rl_scalef(screen.width / w, screen.height / h, 1)

  # engaged border, road camera (the road in front of the car in the photo)
  rl.draw_rectangle(0, 0, w, h, BORDER_COLORS[UIStatus.ENGAGED])
  content = rl.Rectangle(UI_BORDER_SIZE, UI_BORDER_SIZE, w - 2 * UI_BORDER_SIZE, h - 2 * UI_BORDER_SIZE)
  rl.draw_texture_pro(photo, rl.Rectangle(80, 720, 650, 325), content, rl.Vector2(0, 0), 0, rl.WHITE)

  # max and current speed
  bold, semi_bold, medium = gui_app.font(FontWeight.BOLD), gui_app.font(FontWeight.SEMI_BOLD), gui_app.font(FontWeight.MEDIUM)
  box = rl.Rectangle(content.x + 60, content.y + 45, UI_CONFIG.set_speed_width_imperial, UI_CONFIG.set_speed_height)
  rl.draw_rectangle_rounded(box, 0.35, 10, COLORS.BLACK_TRANSLUCENT)
  rl.draw_rectangle_rounded_lines_ex(box, 0.35, 10, 6, COLORS.BORDER_TRANSLUCENT)
  for text, font, size, y, color in ((tr("MAX"), semi_bold, FONT_SIZES.max_speed, box.y + 27, COLORS.ENGAGED),
                                     ("70", bold, FONT_SIZES.set_speed, box.y + 77, COLORS.WHITE)):
    tw = measure_text_cached(font, text, size).x
    rl.draw_text_ex(font, text, rl.Vector2(box.x + (box.width - tw) / 2, y), size, 0, color)
  for text, font, size, y, color in (("70", bold, FONT_SIZES.current_speed, 180, COLORS.WHITE),
                                     (tr("mph"), medium, FONT_SIZES.speed_unit, 290, COLORS.WHITE_TRANSLUCENT)):
    ts = measure_text_cached(font, text, size)
    rl.draw_text_ex(font, text, rl.Vector2(content.x + (content.width - ts.x) / 2, y - ts.y / 2), size, 0, color)

  # alert, same layout as AlertRenderer for a mid size alert
  alert_h = ALERT_HEIGHTS[AlertSize.mid]
  alert = rl.Rectangle(content.x + ALERT_MARGIN, content.y + content.height - alert_h + ALERT_MARGIN,
                       content.width - 2 * ALERT_MARGIN, alert_h - 2 * ALERT_MARGIN)
  rl.draw_rectangle_rounded(alert, ALERT_BORDER_RADIUS / (alert.height / 2), 10, ALERT_COLORS[AlertStatus.userPrompt])
  y = alert.y + ALERT_PADDING
  for text, font, size in ((tr("Pay Attention"), bold, ALERT_FONT_BIG), (tr("Driver Distracted"), _font(False), ALERT_FONT_SMALL)):
    tw = measure_text_cached(font, text, size).x
    rl.draw_text_ex(font, text, rl.Vector2(alert.x + (alert.width - tw) / 2, y), size, 0, rl.WHITE)
    y += ALERT_FONT_BIG + ALERT_LINE_SPACING
  rl.rl_pop_matrix()


class TextureLoader:
  """PNG loading is slow in raylib, so load images in a thread and make the textures on the main thread."""

  def __init__(self):
    self._images: dict[str, rl.Image] = {}
    self._textures: dict[str, rl.Texture] = {}
    self._lock = threading.Lock()

  def queue(self, paths: list[str]):
    threading.Thread(target=self._load, args=(paths,), daemon=True).start()

  def _load(self, paths: list[str]):
    for path in paths:
      image = gui_app._load_image_from_path(os.path.join(BASEDIR, "openpilot/selfdrive/assets", path))
      with self._lock:
        self._images[path] = image

  def get(self, path: str) -> rl.Texture:
    if path not in self._textures:
      with self._lock:
        image = self._images.pop(path, None)
      # tapped through faster than the thread, load it now
      self._textures[path] = gui_app._load_texture_from_image(image) if image is not None else gui_app.texture(path)
    return self._textures[path]


class TapArea(Widget):
  """Invisible tap target, e.g. part of a photo."""

  def __init__(self, click_callback: Callable[[], None]):
    super().__init__()
    self._click_callback = click_callback

  def _render(self, _):
    pass


class BackButton(Widget):
  """Small "< Back" in the top left margin."""

  def __init__(self, click_callback: Callable[[], None]):
    super().__init__()
    self._click_callback = click_callback

  def _render(self, rect: rl.Rectangle):
    color = rl.Color(255, 255, 255, 140 if self.is_pressed else 200)
    chevron = gui_app.texture(f"{ASSETS}/chevron.png", 22, 26, flip_x=True)
    rl.draw_texture_v(chevron, rl.Vector2(rect.x + 40, rect.y + 43), color)
    rl.draw_text_ex(gui_app.font(FontWeight.MEDIUM), tr("Back"), rl.Vector2(rect.x + 72, rect.y + 38), 30, 0, color)


class NextButton(Widget):
  """Colored column on the right with a chevron."""

  def __init__(self, color: rl.Color, click_callback: Callable[[], None]):
    super().__init__()
    self._color = color
    self._pressed_color = _shade(color, -0.2)
    self._click_callback = click_callback

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, self._pressed_color if self.is_pressed else self._color)
    chevron = gui_app.texture(f"{ASSETS}/chevron.png")
    rl.draw_texture_v(chevron, rl.Vector2(rect.x + (rect.width - chevron.width) / 2, rect.y + (rect.height - chevron.height) / 2), rl.WHITE)


class GuideButton(Button):
  def __init__(self, text: str, click_callback: Callable[[], None], color: rl.Color, text_color: rl.Color = rl.BLACK):
    super().__init__(lambda: tr(text), click_callback, font_size=55, font_weight=FontWeight.BOLD, border_radius=24)
    self._color = color
    self._pressed_color = _shade(color, -0.25 if color.r > 128 else 0.15)
    self._text_color = text_color

  def _update_state(self):
    self._background_color = self._pressed_color if self.is_pressed else self._color
    self._label.set_text_color(self._text_color)


class Step(Widget):
  def __init__(self, bg: rl.Color):
    super().__init__()
    self._bg = bg
    self._controls: list[tuple[Widget, rl.Rectangle]] = []

  def textures(self) -> list[str]:
    """Big textures to preload"""
    return []

  def add_control(self, widget: Widget, rect: rl.Rectangle):
    self._controls.append((self._child(widget), rect))

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, self._bg)
    self._draw(rect)
    for widget, r in self._controls:
      widget.render(rl.Rectangle(rect.x + r.x, rect.y + r.y, r.width, r.height))

  def _draw(self, rect: rl.Rectangle):
    pass


class SectionStep(Step):
  """Numbered section intro."""

  def __init__(self, number: int, title: str, body: tuple[Paragraph, ...], colors, on_next):
    super().__init__(colors[0])
    self._number = str(number)
    self._title = title
    self._body = body
    self.add_control(NextButton(colors[1], on_next), NEXT_RECT)

  def _draw(self, rect: rl.Rectangle):
    center = rl.Vector2(rect.x + MARGIN + 64, rect.y + 176)
    rl.draw_circle_v(center, 64, rl.Color(255, 255, 255, 64))
    size = measure_text_cached(_font(True), self._number, TITLE_SIZE)
    rl.draw_text_ex(_font(True), self._number, rl.Vector2(center.x - size.x / 2, center.y - size.y / 2), TITLE_SIZE, 0, rl.WHITE)
    draw_paragraphs((Span(self._title, bold=True),), rect.x + 288, rect.y + 127, NEXT_RECT.x - 288 - MARGIN, TITLE_SIZE, TITLE_LINE)
    draw_paragraphs(self._body, rect.x + MARGIN, rect.y + 294, 1640, INTRO_SIZE, INTRO_LINE)


class ChecklistStep(Step):
  def __init__(self, title: str, items: tuple[str, ...], mark: str, colors, on_next):
    super().__init__(colors[0])
    self._title = title
    self._items = items
    self._mark = f"{ASSETS}/{mark}.png"
    self.add_control(NextButton(colors[1], on_next), NEXT_RECT)

  def _draw(self, rect: rl.Rectangle):
    width = NEXT_RECT.x - 288 - MARGIN
    y = draw_paragraphs((Span(self._title, bold=True),), rect.x + MARGIN, rect.y + 112, width, TITLE_SIZE, TITLE_LINE) + 107
    mark = gui_app.texture(self._mark)
    for item in self._items:
      # items are centered on rows
      lines = len(_wrap(item, ITEM_SIZE, width))
      item_y = y - (lines - 1) * BODY_LINE / 4
      rl.draw_texture_v(mark, rl.Vector2(rect.x + 100, item_y + (lines * BODY_LINE - mark.height) / 2 - 6), rl.WHITE)
      draw_paragraphs((item,), rect.x + 288, item_y - 8, width, ITEM_SIZE, BODY_LINE)
      y += 166 + (lines - 1) * BODY_LINE


class PhotoStep(Step):
  """Text on the left and a photo on the right. Tap the photo (or part of it) to continue."""

  def __init__(self, icon: str, title: str, body: tuple[Paragraph, ...], prompt: str | None, photo: str, colors,
               loader: TextureLoader, tap_rect: rl.Rectangle | None = None, on_next: Callable[[], None] | None = None,
               overlay: Callable[[rl.Vector2, rl.Texture], None] | None = None):
    super().__init__(colors[0])
    self._overlay = overlay
    self._icon = f"{ASSETS}/icon_{icon}.png"
    self._title = title
    self._body = body
    self._prompt = prompt
    self._photo = f"{ASSETS}/photo_{photo}.png"
    self._loader = loader
    if tap_rect is not None and on_next is not None:
      self.add_control(TapArea(on_next), tap_rect)

  def textures(self) -> list[str]:
    return [self._photo]

  def _draw(self, rect: rl.Rectangle):
    photo = self._loader.get(self._photo)
    rl.draw_texture_v(photo, rl.Vector2(rect.x + PHOTO_X, rect.y), rl.WHITE)
    if self._overlay:
      self._overlay(rl.Vector2(rect.x, rect.y), photo)

    # icon is centered on the title
    title = Span(self._title, bold=True)
    title_x = MARGIN + 196
    title_width = PHOTO_X - title_x - MARGIN
    title_lines = len(_wrap(title, TITLE_SIZE, title_width))
    title_y = 136 - (title_lines - 1) * 23
    icon = gui_app.texture(self._icon)
    rl.draw_texture_v(icon, rl.Vector2(rect.x + MARGIN, rect.y + 112 + (title_lines - 1) * 25), rl.WHITE)
    draw_paragraphs((title,), rect.x + title_x, rect.y + title_y, title_width, TITLE_SIZE, TITLE_LINE)
    body_y = rect.y + 307 + (title_lines - 1) * 50
    draw_paragraphs(self._body, rect.x + MARGIN, body_y, PHOTO_X - 2 * MARGIN, BODY_SIZE, BODY_LINE, BODY_GAP)

    if self._prompt:
      draw_paragraphs((Span(self._prompt, bold=True),), rect.x + MARGIN, rect.y + 894, PHOTO_X - 2 * MARGIN, BODY_SIZE, BODY_LINE)


class HeroStep(Step):
  """Full screen photo with text and buttons."""

  def __init__(self, photo: str, title: str, body: Paragraph, loader: TextureLoader):
    super().__init__(rl.BLACK)
    self._photo = f"{ASSETS}/{photo}.png"
    self._title = title
    self._body = body
    self._loader = loader

  def textures(self) -> list[str]:
    return [self._photo]

  def _draw(self, rect: rl.Rectangle):
    rl.draw_texture_v(self._loader.get(self._photo), rl.Vector2(rect.x, rect.y), rl.WHITE)
    y = draw_paragraphs((Span(self._title, bold=True),), rect.x + MARGIN, rect.y + 112, 1180, TITLE_SIZE, TITLE_LINE)
    draw_paragraphs((self._body,), rect.x + MARGIN, y + 52, 1440, INTRO_SIZE, INTRO_LINE)


class TrainingGuide(Widget):
  def __init__(self, completed_callback=None):
    super().__init__()
    self._completed_callback = completed_callback
    self._step = 0

    nxt = self._next
    loader = TextureLoader()

    welcome = HeroStep("bg_welcome", tr_noop("Welcome to openpilot"),
                       tr_noop("Let's review the functionality and limitations of openpilot before getting started."),
                       loader)
    welcome.add_control(GuideButton(tr_noop("Begin Training"), nxt, rl.WHITE), BEGIN_RECT)

    dm_record = PhotoStep("driver", tr_noop("Improve Driver Monitoring"), (
      (
        tr_noop("Help improve driver monitoring by including your driving data in the training data set."),
        tr_noop("Your preference can be changed at any time in Settings. Would you like to share your data?"),
      ),
    ), None, "driver_cam", TEAL, loader)
    dm_record.add_control(GuideButton(tr_noop("No"), lambda: self._record_front(False), _shade(TEAL[0], -0.55), rl.WHITE), DM_NO_RECT)
    dm_record.add_control(GuideButton(tr_noop("Yes"), lambda: self._record_front(True), rl.WHITE), DM_YES_RECT)

    finish = HeroStep("bg_finish", tr_noop("Congratulations! You have completed openpilot training."),
                      (
                        tr_noop("This guide can be replayed at any time from the device settings."),
                        tr_noop("To read more about openpilot, visit github.com/commaai/openpilot and join the community at discord.comma.ai."),
                      ),
                      loader)
    finish.add_control(GuideButton(tr_noop("Restart"), self._restart, rl.Color(64, 64, 64, 255), rl.WHITE), RESTART_RECT)
    finish.add_control(GuideButton(tr_noop("Finish Training"), nxt, rl.WHITE), FINISH_RECT)

    self._steps: list[Step] = [
      welcome,
      SectionStep(1, tr_noop("What is openpilot?"), (
        tr_noop("openpilot is an advanced driver assistance system."),
        Span(tr_noop("A driver assistance system is not a self-driving car."), underline=True),
        (
          tr_noop("This means openpilot is designed to work with you, not without you."),
          tr_noop("Your attention is required at all times to drive with openpilot."),
        ),
      ), NAVY, nxt),
      ChecklistStep(tr_noop("Before we continue, please confirm the following:"), (
        tr_noop("I will keep my eyes on the road."),
        tr_noop("I will be ready to take over at any time."),
        tr_noop("I will be ready to take over at any time!"),
      ), "check", GREEN, nxt),
      PhotoStep("path", tr_noop("The Driving Path"), (
        tr_noop("openpilot uses the road-facing camera to plan the best path to drive."),
        tr_noop("Alongside this path, lane lines are displayed in white and road edges are displayed in red."),
      ), tr_noop("Tap the driving path to continue"), "path", NAVY, loader, PATH_RECT, nxt),
      PhotoStep("lead", tr_noop("The Lead Car Indicator"), ((
        tr_noop("The lead car indicator is displayed as a triangle under the lead car."),
        tr_noop("openpilot can detect 2 cars simultaneously. The second triangle will appear when a cut-in is detected."),
        tr_noop("If no triangle is present, your car may be using its stock ACC system, and some openpilot features may not be available."),
        "–",
        Span(tr_noop("Tap the lead car indicator"), bold=True),
      ),), None, "lead", NAVY, loader, LEAD_RECT, nxt),
      SectionStep(2, tr_noop("How to engage and control openpilot"), (
        (
          tr_noop("openpilot is controlled using your car's cruise control inputs."),
          tr_noop("These are usually located on the steering wheel or on a control lever near the steering column."),
        ),
      ), TEAL, nxt),
      PhotoStep("engage", tr_noop("Engage openpilot"), (
        (
          tr_noop("When you are ready to engage openpilot at a comfortable speed, engage the cruise control system and press “SET” to begin."),
          tr_noop("A green border will appear around the screen whenever openpilot is engaged."),
        ),
      ), tr_noop("Tap “SET” in the image to continue"), "set", TEAL, loader, SET_RECT, nxt),
      PhotoStep("driver", tr_noop("Driver Monitoring"), (
        (
          tr_noop("openpilot monitors your awareness through the device's driver-facing camera."),
          tr_noop("If openpilot senses that your eyes aren't on the road, the system will go through a series of alerts and actions."),
        ),
      ), tr_noop("Tap the driver to continue"), "driver", TEAL, loader, DRIVER_RECT, nxt),
      PhotoStep("distracted", tr_noop("Distracted Driving"), (
        (
          tr_noop("You must pay attention at all times."),
          tr_noop("If you are distracted, openpilot will show alerts of increasing severity, and you will be locked out from engaging."),
          tr_noop("If the problem persists, the system will begin decelerating the vehicle."),
        ),
      ), tr_noop("Tap the alert to continue"), "alert", TEAL, loader, ALERT_RECT, nxt, draw_distracted_screen),
      dm_record,
      PhotoStep("max_speed", tr_noop("Adjust the Max Speed"), (
        (
          tr_noop("You can adjust your maximum speed by pressing + or - on your vehicle's cruise control inputs."),
          tr_noop("The max speed is shown in the upper left corner of the display."),
        ),
      ), tr_noop("Tap the max speed to continue"), "max_speed", TEAL, loader, MAX_SPEED_RECT, nxt),
      SectionStep(3, tr_noop("How to change lanes with openpilot"), (
        (
          tr_noop("If you are traveling above 20 mph (32 km/h), openpilot can perform a lane change."),
          tr_noop("Keep in mind that it is not capable of checking if a lane change is safe. This is your job."),
          tr_noop("Once initiated, openpilot will change lanes regardless if another vehicle is present."),
        ),
      ), GRAY, nxt),
      PhotoStep("lane_change", tr_noop("Perform Lane Change"), (
        (
          tr_noop("With openpilot engaged, turn on your signal and check your surroundings."),
          tr_noop("When it's safe, gently nudge the steering wheel towards your desired lane."),
          tr_noop("The sequence of turn signal and wheel nudge will prompt openpilot to change lanes."),
        ),
      ), tr_noop("Tap the steering wheel to continue"), "wheel", GRAY, loader, WHEEL_RECT, nxt),
      SectionStep(4, tr_noop("How to disengage openpilot"), (
        tr_noop("When you encounter a potentially unsafe situation or need to exit a highway, disengage openpilot by pressing the brake pedal."),
      ), BLUE, nxt),
      PhotoStep("limited", tr_noop("Limited Features"), (
        tr_noop("Keep in mind that openpilot will not recognize certain scenarios including construction zones, close cut-ins, or pedestrians."),
        tr_noop("You must stay alert and always be ready to retake control of the vehicle."),
      ), tr_noop("Tap the light to continue"), "light", BLUE, loader, LIGHT_RECT, nxt),
      PhotoStep("disengage", tr_noop("Disengage openpilot"), (
        (
          tr_noop("While openpilot is engaged, you may keep your hands on the wheel to override steering controls."),
          tr_noop("Both steering and distance to the lead car will be managed by openpilot until the brake pedal is pressed to disengage."),
        ),
      ), tr_noop("Tap the brake pedal to continue"), "pedal", BLUE, loader, PEDAL_RECT, nxt),
      ChecklistStep(tr_noop("Let's review. openpilot can:"), (
        tr_noop("Determine a path to drive."),
        tr_noop("Maintain a maximum speed."),
        tr_noop("Maintain a safe distance from a lead car."),
        tr_noop("Change lanes with driver assistance."),
      ), "check", GREEN, nxt),
      ChecklistStep(tr_noop("openpilot cannot:"), (
        tr_noop("Stay engaged while the driver is distracted."),
        tr_noop("See other cars during a lane change."),
        tr_noop("Avoid construction barriers, cones or pedestrians."),
        tr_noop("React to unsafe situations like close vehicle cut-ins or road hazards."),
      ), "cross", RED, nxt),
      finish,
    ]
    assert len(self._steps) == len(STEP_RECTS)

    for step in self._steps:
      self._child(step)
    self._back_button = self._child(BackButton(self._back))
    self._back_button.set_visible(lambda: self._step > 0)
    loader.queue([t for step in self._steps for t in step.textures()])

  def _next(self):
    self._step += 1
    if self._step >= len(self._steps):
      self._step = 0
      if self._completed_callback:
        self._completed_callback()

      # NOTE: this pops OnboardingWindow during real onboarding
      gui_app.pop_widget()

  def _back(self):
    self._step = max(self._step - 1, 0)

  def _restart(self):
    self._step = 0

  def _record_front(self, record: bool):
    ui_state.params.put_bool("RecordFront", record, block=True)
    self._next()

  def _render(self, rect: rl.Rectangle):
    self._steps[self._step].render(rect)

    # progress bar
    if 0 < self._step < len(self._steps) - 1:
      h = 20
      w = int((self._step / (len(self._steps) - 1)) * rect.width)
      rl.draw_rectangle(int(rect.x), int(rect.y + rect.height - h), w, h, rl.Color(70, 91, 234, 255))

    self._back_button.render(rl.Rectangle(rect.x + BACK_RECT.x, rect.y + BACK_RECT.y, BACK_RECT.width, BACK_RECT.height))
    return -1
