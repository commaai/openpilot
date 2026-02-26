from enum import IntEnum

import weakref
import math
import numpy as np
import qrcode
import pyray as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import SmallCircleIconButton
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.mici_setup import GreyBigButton, TermsHeader, TermsPage as SetupTermsPage
from selfdrive.ui.mici.widgets.button import BigCircleButton
from selfdrive.ui.mici.widgets.dialog import BigConfirmationDialogV2
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog import DriverCameraDialog
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.version import terms_version, training_version


class OnboardingState(IntEnum):
  TERMS = 0
  ONBOARDING = 1


class DriverCameraSetupDialog(DriverCameraDialog):
  def __init__(self):
    super().__init__(no_escape=True)
    self.driver_state_renderer = DriverStateRenderer(inset=True)
    self.driver_state_renderer.set_rect(rl.Rectangle(0, 0, 120, 120))
    self.driver_state_renderer.load_icons()
    self.driver_state_renderer.set_force_active(True)

  def _render(self, rect):
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    self._camera_view._render(rect)

    if not self._camera_view.frame:
      gui_label(rect, tr("camera starting"), font_size=64, font_weight=FontWeight.BOLD,
                alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      rl.end_scissor_mode()
      return

    # Position dmoji on opposite side from driver
    is_rhd = self.driver_state_renderer.is_rhd
    self.driver_state_renderer.set_position(
      rect.x + 8 if is_rhd else rect.x + rect.width - self.driver_state_renderer.rect.width - 8,
      rect.y + 8,
    )
    self.driver_state_renderer.render()

    self._draw_face_detection(rect)

    rl.end_scissor_mode()


class TrainingGuidePreDMTutorial(SetupTermsPage):
  def __init__(self, continue_callback):
    super().__init__(continue_callback, continue_text="continue")
    self._title_header = TermsHeader("driver monitoring setup", gui_app.texture("icons_mici/setup/green_dm.png", 60, 60))

    self._dm_label = UnifiedLabel("Next, we'll ensure comma four is mounted properly.\n\nIf it does not have a clear view of the driver, " +
                                  "unplug and remount before continuing.", 42,
                                  FontWeight.ROMAN)

  def show_event(self):
    super().show_event()
    # Get driver monitoring model ready for next step
    ui_state.params.put_bool("IsDriverViewEnabled", True)

  @property
  def _content_height(self):
    return self._dm_label.rect.y + self._dm_label.rect.height - self._scroll_panel.get_offset()

  def _render_content(self, scroll_offset):
    self._title_header.render(rl.Rectangle(
      self._rect.x + 16,
      self._rect.y + 16 + scroll_offset,
      self._title_header.rect.width,
      self._title_header.rect.height,
    ))

    self._dm_label.render(rl.Rectangle(
      self._rect.x + 16,
      self._title_header.rect.y + self._title_header.rect.height + 16,
      self._rect.width - 32,
      self._dm_label.get_content_height(int(self._rect.width - 32)),
    ))


class DMBadFaceDetected(SetupTermsPage):
  def __init__(self, continue_callback, back_callback):
    super().__init__(continue_callback, back_callback, continue_text="power off")
    self._title_header = TermsHeader("make sure comma four can see your face", gui_app.texture("icons_mici/setup/orange_dm.png", 60, 60))
    self._dm_label = UnifiedLabel("Re-mount if your face is occluded or driver monitoring has difficulty tracking your face.", 42, FontWeight.ROMAN)

  @property
  def _content_height(self):
    return self._dm_label.rect.y + self._dm_label.rect.height - self._scroll_panel.get_offset()

  def _render_content(self, scroll_offset):
    self._title_header.render(rl.Rectangle(
      self._rect.x + 16,
      self._rect.y + 16 + scroll_offset,
      self._title_header.rect.width,
      self._title_header.rect.height,
    ))

    self._dm_label.render(rl.Rectangle(
      self._rect.x + 16,
      self._title_header.rect.y + self._title_header.rect.height + 16,
      self._rect.width - 32,
      self._dm_label.get_content_height(int(self._rect.width - 32)),
    ))


class TrainingGuideDMTutorial(Widget):
  PROGRESS_DURATION = 4
  LOOKING_THRESHOLD_DEG = 30.0

  def __init__(self, continue_callback):
    super().__init__()

    self_ref = weakref.ref(self)

    self._back_button = SmallCircleIconButton(gui_app.texture("icons_mici/setup/driver_monitoring/dm_question.png", 28, 48))
    self._back_button.set_click_callback(lambda: self_ref() and self_ref()._show_bad_face_page())
    self._good_button = SmallCircleIconButton(gui_app.texture("icons_mici/setup/driver_monitoring/dm_check.png", 42, 42))

    # Wrap the continue callback to restore settings
    def wrapped_continue_callback():
      device.set_offroad_brightness(None)
      continue_callback()

    self._good_button.set_click_callback(wrapped_continue_callback)
    self._good_button.set_enabled(False)

    self._progress = FirstOrderFilter(0.0, 0.5, 1 / gui_app.target_fps)
    self._dialog = DriverCameraSetupDialog()
    self._bad_face_page = DMBadFaceDetected(HARDWARE.shutdown, lambda: self_ref() and self_ref()._hide_bad_face_page())
    self._should_show_bad_face_page = False

    # Disable driver monitoring model when device times out for inactivity
    def inactivity_callback():
      ui_state.params.put_bool("IsDriverViewEnabled", False)

    device.add_interactive_timeout_callback(inactivity_callback)

  def _show_bad_face_page(self):
    self._bad_face_page.show_event()
    self.hide_event()
    self._should_show_bad_face_page = True

  def _hide_bad_face_page(self):
    self._bad_face_page.hide_event()
    self.show_event()
    self._should_show_bad_face_page = False

  def show_event(self):
    super().show_event()
    self._dialog.show_event()
    self._progress.x = 0.0

    device.set_offroad_brightness(100)

  def _update_state(self):
    super()._update_state()
    if device.awake and not ui_state.params.get_bool("IsDriverViewEnabled"):
      ui_state.params.put_bool_nonblocking("IsDriverViewEnabled", True)

    sm = ui_state.sm
    if sm.recv_frame.get("driverMonitoringState", 0) == 0:
      return

    dm_state = sm["driverMonitoringState"]
    driver_data = self._dialog.driver_state_renderer.get_driver_data()

    if len(driver_data.faceOrientation) == 3:
      pitch, yaw, _ = driver_data.faceOrientation
      looking_center = abs(math.degrees(pitch)) < self.LOOKING_THRESHOLD_DEG and abs(math.degrees(yaw)) < self.LOOKING_THRESHOLD_DEG
    else:
      looking_center = False

    # stay at 100% once reached
    if (dm_state.faceDetected and looking_center) or self._progress.x > 0.99:
      slow = self._progress.x < 0.25
      duration = self.PROGRESS_DURATION * 2 if slow else self.PROGRESS_DURATION
      self._progress.x += 1.0 / (duration * gui_app.target_fps)
      self._progress.x = min(1.0, self._progress.x)
    else:
      self._progress.update(0.0)

    self._good_button.set_enabled(self._progress.x >= 0.999)

  def _render(self, _):
    if self._should_show_bad_face_page:
      return self._bad_face_page.render(self._rect)

    self._dialog.render(self._rect)

    rl.draw_rectangle_gradient_v(int(self._rect.x), int(self._rect.y + self._rect.height - 80),
                                 int(self._rect.width), 80, rl.BLANK, rl.BLACK)

    # draw white ring around dm icon to indicate progress
    ring_thickness = 8

    # DM icon is 120x120, positioned on opposite side from driver
    dm_size = 120
    is_rhd = self._dialog.driver_state_renderer._is_rhd
    dm_center_x = (self._rect.x + dm_size / 2 + 8) if is_rhd else (self._rect.x + self._rect.width - dm_size / 2 - 8)
    dm_center_y = self._rect.y + dm_size / 2 + 8
    icon_edge_radius = dm_size / 2
    outer_radius = icon_edge_radius + 1  # 2px outward from icon edge
    inner_radius = outer_radius - ring_thickness  # Inset by ring_thickness
    start_angle = 90.0  # Start from bottom
    end_angle = start_angle + self._progress.x * 360.0  # Clockwise

    # Fade in alpha
    current_angle = end_angle - start_angle
    alpha = int(np.interp(current_angle, [0.0, 45.0], [0, 255]))

    # White to green
    color_t = np.clip(np.interp(current_angle, [45.0, 360.0], [0.0, 1.0]), 0.0, 1.0)
    r = int(np.interp(color_t, [0.0, 1.0], [255, 0]))
    g = int(np.interp(color_t, [0.0, 1.0], [255, 255]))
    b = int(np.interp(color_t, [0.0, 1.0], [255, 64]))
    ring_color = rl.Color(r, g, b, alpha)

    rl.draw_ring(
      rl.Vector2(dm_center_x, dm_center_y),
      inner_radius,
      outer_radius,
      start_angle,
      end_angle,
      36,
      ring_color,
    )

    if self._dialog._camera_view.frame:
      self._back_button.render(rl.Rectangle(
        self._rect.x + 8,
        self._rect.y + self._rect.height - self._back_button.rect.height,
        self._back_button.rect.width,
        self._back_button.rect.height,
      ))

      self._good_button.render(rl.Rectangle(
        self._rect.x + self._rect.width - self._good_button.rect.width - 8,
        self._rect.y + self._rect.height - self._good_button.rect.height,
        self._good_button.rect.width,
        self._good_button.rect.height,
      ))

    # rounded border
    rl.draw_rectangle_rounded_lines_ex(self._rect, 0.2 * 1.02, 10, 50, rl.BLACK)


class TrainingGuideRecordFront(SetupTermsPage):
  def __init__(self, continue_callback):
    def on_back():
      ui_state.params.put_bool("RecordFront", False)
      continue_callback()

    def on_continue():
      ui_state.params.put_bool("RecordFront", True)
      continue_callback()

    super().__init__(on_continue, back_callback=on_back, back_text="no", continue_text="yes")
    self._title_header = TermsHeader("improve driver monitoring", gui_app.texture("icons_mici/setup/green_dm.png", 60, 60))

    self._dm_label = UnifiedLabel("Do you want to upload driver camera data?", 42,
                                  FontWeight.ROMAN)

  def show_event(self):
    super().show_event()
    # Disable driver monitoring model after last step
    ui_state.params.put_bool("IsDriverViewEnabled", False)

  @property
  def _content_height(self):
    return self._dm_label.rect.y + self._dm_label.rect.height - self._scroll_panel.get_offset()

  def _render_content(self, scroll_offset):
    self._title_header.render(rl.Rectangle(
      self._rect.x + 16,
      self._rect.y + 16 + scroll_offset,
      self._title_header.rect.width,
      self._title_header.rect.height,
    ))

    self._dm_label.render(rl.Rectangle(
      self._rect.x + 16,
      self._title_header.rect.y + self._title_header.rect.height + 16,
      self._rect.width - 32,
      self._dm_label.get_content_height(int(self._rect.width - 32)),
    ))


class TrainingGuide(Widget):
  def __init__(self, completed_callback=None):
    super().__init__()
    self._completed_callback = completed_callback
    self._step = 0

    self_ref = weakref.ref(self)

    def on_continue():
      if obj := self_ref():
        obj._advance_step()

    self._steps = [
      TrainingGuidePreDMTutorial(continue_callback=on_continue),
      TrainingGuideDMTutorial(continue_callback=on_continue),
      TrainingGuideRecordFront(continue_callback=on_continue),
    ]

  def show_event(self):
    super().show_event()
    device.set_override_interactive_timeout(300)

  def hide_event(self):
    super().hide_event()
    device.set_override_interactive_timeout(None)

  def _advance_step(self):
    if self._step < len(self._steps) - 1:
      self._step += 1
      self._steps[self._step].show_event()
    else:
      self._step = 0
      if self._completed_callback:
        self._completed_callback()

  def _render(self, _):
    rl.draw_rectangle_rec(self._rect, rl.BLACK)
    if self._step < len(self._steps):
      self._steps[self._step].render(self._rect)


class SmallGreyBigButton(GreyBigButton):
  def __init__(self, text: str, icon: rl.Texture):
    super().__init__("", text, icon)
    self._rect.width = 198
    self._rect.height = 180

  def _draw_content(self, btn_y: float):
    if self._txt_icon:
      x = self._rect.x + 30
      y = btn_y + 30
      rl.draw_texture_ex(self._txt_icon, (x, y), 0, 1.0, rl.Color(255, 255, 255, int(255 * 0.9)))

    label_x = self._rect.x + self.LABEL_HORIZONTAL_PADDING
    label_y = btn_y + self.LABEL_VERTICAL_PADDING + (self._txt_icon.height + 10 if self._txt_icon else 0)
    sub_label_height = btn_y + self._rect.height - self.LABEL_VERTICAL_PADDING - label_y
    self._sub_label.render(rl.Rectangle(label_x, label_y, self._width_hint(), sub_label_height))

  def _render(self, _):
    rl.draw_rectangle_rounded(self._rect, 0.4, 10, rl.Color(34, 34, 34, 255))
    self._draw_content(self._rect.y)


class QRCodeWidget(Widget):
  def __init__(self, url: str, size: int = 180):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, size, size))
    self._size = size
    self._qr_texture: rl.Texture | None = None
    self._generate_qr(url)

  def _generate_qr(self, url: str):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=0)
    qr.add_data(url)
    qr.make(fit=True)

    pil_img = qr.make_image(fill_color="white", back_color="black").convert('RGBA')
    img_array = np.array(pil_img, dtype=np.uint8)

    rl_image = rl.Image()
    rl_image.data = rl.ffi.cast("void *", img_array.ctypes.data)
    rl_image.width = pil_img.width
    rl_image.height = pil_img.height
    rl_image.mipmaps = 1
    rl_image.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8

    self._qr_texture = rl.load_texture_from_image(rl_image)

  def _render(self, _):
    if self._qr_texture:
      scale = self._size / self._qr_texture.height
      rl.draw_texture_ex(self._qr_texture, rl.Vector2(self._rect.x, self._rect.y), 0.0, scale, rl.WHITE)

  def __del__(self):
    if self._qr_texture and self._qr_texture.id != 0:
      rl.unload_texture(self._qr_texture)


class TermsPage(Widget):
  def __init__(self, on_accept, on_decline):
    super().__init__()

    def show_accept_dialog():
      gui_app.push_widget(BigConfirmationDialogV2("accept\nterms", "icons_mici/setup/driver_monitoring/dm_check.png",
                                                  confirm_callback=on_accept))

    def show_decline_dialog():
      gui_app.push_widget(BigConfirmationDialogV2("decline &\nuninstall", "icons_mici/setup/cancel.png",
                                                  red=True, confirm_callback=on_decline))

    self._accept_button = BigCircleButton("icons_mici/setup/driver_monitoring/dm_check.png")
    self._accept_button.set_click_callback(show_accept_dialog)

    self._decline_button = BigCircleButton("icons_mici/setup/cancel.png", red=True)
    self._decline_button.set_click_callback(show_decline_dialog)

    self._scroller = Scroller([
      GreyBigButton("terms and\nconditions", "scroll to read and accept",
                    gui_app.texture("icons_mici/setup/green_info.png", 64, 64)),
      GreyBigButton("", "• openpilot is a driver assistance system.\n" +
                    "• You must pay attention\nat all times."),
      GreyBigButton("", "• You must be ready to\ntake over at any time.\n" +
                    "• You are fully responsible for driving the car."),
      SmallGreyBigButton("scan for\nfull terms", gui_app.texture("icons_mici/settings/device/pair.png", 64, 48)),
      QRCodeWidget("https://comma.ai/terms"),
      GreyBigButton("", "You must accept the Terms & Conditions to use openpilot. Read the latest at https://comma.ai/terms"),
      self._accept_button,
      self._decline_button,
    ])

    self._scroller.set_enabled(lambda: self.enabled)

  def hide_event(self):
    super().hide_event()
    self._scroller.hide_event()

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def _render(self, _):
    self._scroller.render(self._rect)


class OnboardingWindow(Widget):
  def __init__(self):
    super().__init__()

    self._accepted_terms: bool = ui_state.params.get("HasAcceptedTerms") == terms_version
    self._training_done: bool = ui_state.params.get("CompletedTrainingVersion") == training_version

    self._state = OnboardingState.TERMS if not self._accepted_terms else OnboardingState.ONBOARDING

    self.set_rect(rl.Rectangle(0, 0, 458, gui_app.height))

    # Windows
    self._terms = TermsPage(on_accept=self._on_terms_accepted, on_decline=self._on_uninstall)
    self._training_guide = TrainingGuide(completed_callback=self._on_completed_training)
    self._terms.set_enabled(lambda: self.enabled)  # for nav stack

  def _on_uninstall(self):
    print('uninstalling!')
    ui_state.params.put_bool("DoUninstall", True)
    gui_app.request_close()

  def show_event(self):
    super().show_event()
    device.set_override_interactive_timeout(300)

  def hide_event(self):
    super().hide_event()
    device.set_override_interactive_timeout(None)

  @property
  def completed(self) -> bool:
    return self._accepted_terms and self._training_done

  def close(self):
    ui_state.params.put_bool("IsDriverViewEnabled", False)
    gui_app.pop_widget()

  def _on_terms_accepted(self):
    ui_state.params.put("HasAcceptedTerms", terms_version)
    self._state = OnboardingState.ONBOARDING

  def _on_completed_training(self):
    ui_state.params.put("CompletedTrainingVersion", training_version)
    self.close()

  def _render(self, _):
    rl.draw_rectangle_rec(self._rect, rl.BLACK)
    if self._state == OnboardingState.TERMS:
      self._terms.render(self._rect)
    else:
      self._training_guide.render(self._rect)
