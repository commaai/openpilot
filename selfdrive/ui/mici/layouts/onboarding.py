import math
import numpy as np
import qrcode
import pyray as rl
from collections.abc import Callable
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import SmallCircleIconButton
from openpilot.system.ui.widgets.scroller import NavScroller, Scroller
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.mici_setup import GreyBigButton, BigPillButton
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.version import terms_version, training_version
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.selfdrive.ui.mici.widgets.button import BigCircleButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialogV2
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog import BaseDriverCameraDialog


class DriverCameraSetupDialog(BaseDriverCameraDialog):
  def __init__(self):
    super().__init__()
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


class TrainingGuidePreDMTutorial(NavScroller):
  def __init__(self, continue_callback: Callable[[], None]):
    super().__init__()

    continue_button = BigPillButton("next")
    continue_button.set_click_callback(continue_callback)

    self._scroller.add_widgets([
      GreyBigButton("driver monitoring\ncheck", "scroll to continue",
                    gui_app.texture("icons_mici/setup/green_dm.png", 64, 64)),
      GreyBigButton("", "Next, we'll check if comma four can detect the driver properly."),
      GreyBigButton("", "openpilot uses the cabin camera to check if the driver is distracted."),
      GreyBigButton("", "If it does not have a clear view of the driver, unplug and remount before continuing."),
      continue_button,
    ])

  def show_event(self):
    super().show_event()
    # Get driver monitoring model ready for next step
    ui_state.params.put_bool_nonblocking("IsDriverViewEnabled", True)


class DMBadFaceDetected(NavScroller):
  def __init__(self):
    super().__init__()

    back_button = BigPillButton("back")
    back_button.set_click_callback(self.dismiss)

    self._scroller.add_widgets([
      GreyBigButton("looking for driver", "make sure comma\nfour can see your face",
                    gui_app.texture("icons_mici/setup/orange_dm.png", 64, 64)),
      GreyBigButton("", "Remount if your face is blocked, or driver monitoring has difficulty tracking your face."),
      back_button,
    ])


class TrainingGuideDMTutorial(NavWidget):
  PROGRESS_DURATION = 4
  LOOKING_THRESHOLD_DEG = 30.0

  def __init__(self, continue_callback: Callable[[], None]):
    super().__init__()

    self._back_button = SmallCircleIconButton(gui_app.texture("icons_mici/setup/driver_monitoring/dm_question.png", 28, 48))
    self._back_button.set_click_callback(lambda: gui_app.push_widget(self._bad_face_page))
    self._back_button.set_touch_valid_callback(lambda: self.enabled and not self.is_dismissing)  # for nav stack
    self._good_button = SmallCircleIconButton(gui_app.texture("icons_mici/setup/driver_monitoring/dm_check.png", 42, 42))
    self._good_button.set_touch_valid_callback(lambda: self.enabled and not self.is_dismissing)  # for nav stack

    self._good_button.set_click_callback(continue_callback)
    self._good_button.set_enabled(False)

    self._progress = FirstOrderFilter(0.0, 0.5, 1 / gui_app.target_fps)
    self._dialog = DriverCameraSetupDialog()
    self._bad_face_page = DMBadFaceDetected()

    # Disable driver monitoring model when device times out for inactivity
    def inactivity_callback():
      ui_state.params.put_bool("IsDriverViewEnabled", False)

    device.add_interactive_timeout_callback(inactivity_callback)

  def show_event(self):
    super().show_event()
    self._dialog.show_event()
    self._progress.x = 0.0

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
    in_bad_face = gui_app.get_active_widget() == self._bad_face_page
    if ((dm_state.faceDetected and looking_center) or self._progress.x > 0.99) and not in_bad_face:
      slow = self._progress.x < 0.25
      duration = self.PROGRESS_DURATION * 2 if slow else self.PROGRESS_DURATION
      self._progress.x += 1.0 / (duration * gui_app.target_fps)
      self._progress.x = min(1.0, self._progress.x)
    else:
      self._progress.update(0.0)

    self._good_button.set_enabled(self._progress.x >= 0.999)

  def _render(self, _):
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
    rl.begin_scissor_mode(int(self._rect.x), int(self._rect.y), int(self._rect.width), int(self._rect.height))
    rl.draw_rectangle_rounded_lines_ex(self._rect, 0.2 * 1.02, 10, 50, rl.BLACK)
    rl.end_scissor_mode()


class TrainingGuideRecordFront(NavScroller):
  def __init__(self, continue_callback: Callable[[], None]):
    super().__init__()

    def show_accept_dialog():
      def on_accept():
        ui_state.params.put_bool_nonblocking("RecordFront", True)
        continue_callback()

      gui_app.push_widget(BigConfirmationDialogV2("allow data uploading", "icons_mici/setup/driver_monitoring/dm_check.png", exit_on_confirm=False,
                                                  confirm_callback=on_accept))

    def show_decline_dialog():
      def on_decline():
        ui_state.params.put_bool_nonblocking("RecordFront", False)
        continue_callback()

      gui_app.push_widget(BigConfirmationDialogV2("no, don't upload", "icons_mici/setup/cancel.png", exit_on_confirm=False, confirm_callback=on_decline))

    self._accept_button = BigCircleButton("icons_mici/setup/driver_monitoring/dm_check.png")
    self._accept_button.set_click_callback(show_accept_dialog)

    self._decline_button = BigCircleButton("icons_mici/setup/cancel.png")
    self._decline_button.set_click_callback(show_decline_dialog)

    self._scroller.add_widgets([
      GreyBigButton("driver camera data", "do you want to share video data for training?",
                    gui_app.texture("icons_mici/setup/green_dm.png", 64, 64)),
      GreyBigButton("", "Sharing your data with comma helps improve openpilot for everyone."),
      self._accept_button,
      self._decline_button,
    ])


class TrainingGuideAttentionNotice(Scroller):
  def __init__(self, continue_callback: Callable[[], None]):
    super().__init__()

    continue_button = BigPillButton("next")
    continue_button.set_click_callback(continue_callback)

    self._scroller.add_widgets([
      GreyBigButton("what is openpilot?", "scroll to continue",
                    gui_app.texture("icons_mici/setup/green_info.png", 64, 64)),
      GreyBigButton("", "1. openpilot is a driver assistance system."),
      GreyBigButton("", "2. You must pay attention at all times."),
      GreyBigButton("", "3. You must be ready to take over at any time."),
      GreyBigButton("", "4. You are fully responsible for driving the car."),
      continue_button,
    ])


class TrainingGuide(NavWidget):
  def __init__(self, completed_callback: Callable[[], None]):
    super().__init__()

    self._steps = [
      TrainingGuideAttentionNotice(continue_callback=lambda: gui_app.push_widget(self._steps[1])),
      TrainingGuidePreDMTutorial(continue_callback=lambda: gui_app.push_widget(self._steps[2])),
      TrainingGuideDMTutorial(continue_callback=lambda: gui_app.push_widget(self._steps[3])),
      TrainingGuideRecordFront(continue_callback=completed_callback),
    ]

    self._steps[0].set_enabled(lambda: self.enabled and not self.is_dismissing)  # for nav stack

  def show_event(self):
    super().show_event()
    self._steps[0].show_event()

  def _render(self, _):
    self._steps[0].render(self._rect)


class QRCodeWidget(Widget):
  def __init__(self, url: str, size: int = 170):
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


class TermsPage(Scroller):
  def __init__(self, on_accept, on_decline):
    super().__init__()

    def show_accept_dialog():
      gui_app.push_widget(BigConfirmationDialogV2("accept\nterms", "icons_mici/setup/driver_monitoring/dm_check.png",
                                                  confirm_callback=on_accept))

    def show_decline_dialog():
      gui_app.push_widget(BigConfirmationDialogV2("decline &\nuninstall", "icons_mici/setup/cancel.png",
                                                  red=True, exit_on_confirm=False, confirm_callback=on_decline))

    self._accept_button = BigCircleButton("icons_mici/setup/driver_monitoring/dm_check.png")
    self._accept_button.set_click_callback(show_accept_dialog)

    self._decline_button = BigCircleButton("icons_mici/setup/cancel.png", red=True)
    self._decline_button.set_click_callback(show_decline_dialog)

    self._terms_header = GreyBigButton("terms and\nconditions", "scroll to continue",
                                       gui_app.texture("icons_mici/setup/green_info.png", 64, 64))
    self._must_accept_card = GreyBigButton("", "You must accept the Terms & Conditions to use openpilot.")

    self._scroller.add_widgets([
      self._terms_header,
      GreyBigButton("swipe for QR code", "or go to https://comma.ai/terms",
                    gui_app.texture("icons_mici/setup/small_slider/slider_arrow.png", 64, 56, flip_x=True)),
      QRCodeWidget("https://comma.ai/terms"),
      self._must_accept_card,
      self._accept_button,
      self._decline_button,
    ])

  def _render(self, _):
    rl.draw_rectangle_rec(self._rect, rl.BLACK)
    super()._render(_)


class OnboardingWindow(Widget):
  def __init__(self, completed_callback: Callable[[], None]):
    super().__init__()
    self._completed_callback = completed_callback
    self._accepted_terms: bool = ui_state.params.get("HasAcceptedTerms") == terms_version
    self._training_done: bool = ui_state.params.get("CompletedTrainingVersion") == training_version

    self.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

    # Windows
    self._terms = TermsPage(on_accept=self._on_terms_accepted, on_decline=self._on_uninstall)
    self._terms.set_enabled(lambda: self.enabled)  # for nav stack
    self._training_guide = TrainingGuide(completed_callback=self._on_completed_training)
    self._training_guide.set_enabled(lambda: self.enabled)  # for nav stack

  def _on_uninstall(self):
    ui_state.params.put_bool("DoUninstall", True)

  def show_event(self):
    super().show_event()
    device.set_override_interactive_timeout(300)
    device.set_offroad_brightness(100)

  def hide_event(self):
    super().hide_event()
    # FIXME: when nav stack sends hide event to widget 2 below on push, this needs to be moved
    device.set_override_interactive_timeout(None)
    device.set_offroad_brightness(None)

  @property
  def completed(self) -> bool:
    return self._accepted_terms and self._training_done

  def close(self):
    ui_state.params.put_bool_nonblocking("IsDriverViewEnabled", False)
    self._completed_callback()

  def _on_terms_accepted(self):
    ui_state.params.put("HasAcceptedTerms", terms_version)
    gui_app.push_widget(self._training_guide)

  def _on_completed_training(self):
    ui_state.params.put("CompletedTrainingVersion", training_version)
    self.close()

  def _render(self, _):
    rl.draw_rectangle_rec(self._rect, rl.BLACK)
    self._terms.render(self._rect)
