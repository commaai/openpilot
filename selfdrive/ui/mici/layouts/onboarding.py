from enum import IntEnum
from collections.abc import Callable

import weakref
import pyray as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import SmallButton, SmallCircleIconButton
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.slider import SmallSlider
from openpilot.system.ui.mici_setup import TermsHeader, TermsPage as SetupTermsPage
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog import DriverCameraDialog
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.version import terms_version, training_version


class OnboardingState(IntEnum):
  TERMS = 0
  ONBOARDING = 1
  DECLINE = 2


class DriverCameraSetupDialog(DriverCameraDialog):
  def __init__(self, confirm_callback: Callable):
    super().__init__(no_escape=True)
    self.driver_state_renderer = DriverStateRenderer(confirm_callback=confirm_callback, inset=True)
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
      return -1

    # Position dmoji on opposite side from driver
    # TODO: we don't have design for RHD yet
    is_rhd = False
    self.driver_state_renderer.set_position(
      rect.x + 8 if is_rhd else rect.x + rect.width - self.driver_state_renderer.rect.width - 8,
      rect.y + 8,
    )
    self.driver_state_renderer.render()

    rl.end_scissor_mode()
    return -1


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
    super().__init__(continue_callback, back_callback, continue_text="reboot")
    # normal page like the rest with scroller and header that says "driver monitor couldn't detect a face" and then
    # say some things like "make sure the camera has a clear view of your face" and "ensure you're sitting upright"
    # and to remount. then show reboot btn at the bottom along with back button

    self._title_header = TermsHeader("driver monitor couldn't detect a face", gui_app.texture("icons_mici/setup/orange_dm.png", 60, 60))
    self._dm_label = UnifiedLabel("make sure the camera has a clear view of your face.\n\n" +
                                  "ensure you're sitting upright and remount if necessary.", 42,
                                  FontWeight.ROMAN)

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

  def __init__(self, continue_callback):
    super().__init__()
    self._title_header = TermsHeader("fill the circle to continue", gui_app.texture("icons_mici/setup/green_dm.png", 60, 60))

    self._back_button = SmallCircleIconButton(gui_app.texture("icons_mici/setup/driver_monitoring/dm_no_person.png", 48, 48))
    self._back_button.set_click_callback(self._show_bad_face_page)
    self._good_button = SmallCircleIconButton(gui_app.texture("icons_mici/setup/driver_monitoring/dm_check.png", 49, 36))
    self._good_button.set_click_callback(continue_callback)
    self._good_button.set_enabled(False)

    self._progress = FirstOrderFilter(0.0, 0.5, 1 / gui_app.target_fps)

    self._bad_face_page = DMBadFaceDetected(HARDWARE.reboot, self._hide_bad_face_page)
    self._show_time = 0.0

    self._show_bad_face_page = False

    # Wrap the continue callback to restore settings
    def wrapped_continue_callback():
      device.set_offroad_brightness(None)
      device.reset_interactive_timeout()
      continue_callback()

    self._dialog = DriverCameraSetupDialog(wrapped_continue_callback)

    # Disable driver monitoring model when device times out for inactivity
    def inactivity_callback():
      ui_state.params.put_bool("IsDriverViewEnabled", False)

    device.add_interactive_timeout_callback(inactivity_callback)

  def _show_bad_face_page(self):
    self._bad_face_page.show_event()
    self.hide_event()
    self._show_bad_face_page = True

  def _hide_bad_face_page(self):
    self._bad_face_page.hide_event()
    self.show_event()
    self._show_bad_face_page = False

  def show_event(self):
    super().show_event()
    self._progress.x = 0.0
    self._dialog.show_event()
    self._show_time = rl.get_time()

    device.set_offroad_brightness(100)
    device.reset_interactive_timeout(300)  # 5 minutes

  def _update_state(self):
    super()._update_state()
    if device.awake:
      ui_state.params.put_bool("IsDriverViewEnabled", True)

    # Update progress based on face detection
    sm = ui_state.sm
    if sm.recv_frame.get("driverMonitoringState", 0) > 0:
      dm_state = sm["driverMonitoringState"]

      if (dm_state.faceDetected and rl.get_time() - self._show_time > 2) or self._progress.x > 0.99:
        self._progress.x += 1.0 / (self.PROGRESS_DURATION * gui_app.target_fps)
        self._progress.x = min(1.0, self._progress.x)
      else:
        self._progress.update(0.0)

      # Enable continue button only when progress reaches 100%
      self._good_button.set_enabled(self._progress.x >= 0.999)

  def _render(self, _):
    if self._show_bad_face_page:
      return self._bad_face_page.render(self._rect)

    self._dialog.render(self._rect)

    rl.draw_rectangle_gradient_v(int(self._rect.x), int(self._rect.y + self._rect.height - self._title_header.rect.height * 1.5 - 32),
                                 int(self._rect.width), int(self._title_header.rect.height * 1.5 + 32),
                                 rl.BLANK, rl.Color(0, 0, 0, 150))

    # draw white ring around dm icon to indicate progress
    ring_thickness = 8

    # DM icon is 120x120, positioned at top-right with 8px padding
    dm_size = 120
    dm_center_x = self._rect.x + self._rect.width - dm_size / 2 - 8
    dm_center_y = self._rect.y + dm_size / 2 + 8
    icon_edge_radius = dm_size / 2
    outer_radius = icon_edge_radius + 1  # 2px outward from icon edge
    inner_radius = outer_radius - ring_thickness  # Inset by ring_thickness
    start_angle = 90.0  # Start from bottom
    end_angle = start_angle + self._progress.x * 360.0  # Clockwise

    # Fade in
    current_angle = end_angle - start_angle
    alpha = int((current_angle / 30.0) * 255) if current_angle <= 30.0 else 255

    # Turn green at 99% progress
    if self._progress.x >= 0.99:
      ring_color = rl.Color(0, 255, 64, alpha)
    else:
      ring_color = rl.Color(255, 255, 255, alpha)

    rl.draw_ring(
      rl.Vector2(dm_center_x, dm_center_y),
      inner_radius,
      outer_radius,
      start_angle,
      end_angle,
      36,  # segments for smooth arc
      ring_color,
    )

    # self._title_header.render(rl.Rectangle(
    #   self._rect.x + 16,
    #   self._rect.y + self._rect.height - self._title_header.rect.height - 16,
    #   self._title_header.rect.width,
    #   self._title_header.rect.height,
    # ))

    # back button bottom left, 8 px from left 0 px from bottom. continue btn is right same
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


class TrainingGuideAttentionNotice(SetupTermsPage):
  def __init__(self, continue_callback):
    super().__init__(continue_callback, continue_text="continue")
    self._title_header = TermsHeader("driver assistance", gui_app.texture("icons_mici/setup/warning.png", 60, 60))
    self._warning_label = UnifiedLabel("1. openpilot is a driver assistance system.\n\n" +
                                       "2. You must pay attention at all times.\n\n" +
                                       "3. You must be ready to take over at any time.\n\n" +
                                       "4. You are fully responsible for driving the car.", 42,
                                       FontWeight.ROMAN)

  @property
  def _content_height(self):
    return self._warning_label.rect.y + self._warning_label.rect.height - self._scroll_panel.get_offset()

  def _render_content(self, scroll_offset):
    self._title_header.render(rl.Rectangle(
      self._rect.x + 16,
      self._rect.y + 16 + scroll_offset,
      self._title_header.rect.width,
      self._title_header.rect.height,
    ))

    self._warning_label.render(rl.Rectangle(
      self._rect.x + 16,
      self._title_header.rect.y + self._title_header.rect.height + 16,
      self._rect.width - 32,
      self._warning_label.get_content_height(int(self._rect.width - 32)),
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
      # TrainingGuideAttentionNotice(continue_callback=on_continue),
      # TrainingGuidePreDMTutorial(continue_callback=on_continue),
      TrainingGuideDMTutorial(continue_callback=on_continue),
      TrainingGuideRecordFront(continue_callback=on_continue),
    ]

  def _advance_step(self):
    if self._step < len(self._steps) - 1:
      self._step += 1
      self._steps[self._step].show_event()
    else:
      self._step = 0
      if self._completed_callback:
        self._completed_callback()

  def _render(self, _):
    if self._step < len(self._steps):
      self._steps[self._step].render(self._rect)
    return -1


class DeclinePage(Widget):
  def __init__(self, back_callback=None):
    super().__init__()
    self._uninstall_slider = SmallSlider("uninstall openpilot", self._on_uninstall)

    self._back_button = SmallButton("back")
    self._back_button.set_click_callback(back_callback)

    self._warning_header = TermsHeader("you must accept the\nterms to use openpilot",
                                       gui_app.texture("icons_mici/setup/red_warning.png", 66, 60))

  def _on_uninstall(self):
    ui_state.params.put_bool("DoUninstall", True)
    gui_app.request_close()

  def _render(self, _):
    self._warning_header.render(rl.Rectangle(
      self._rect.x + 16,
      self._rect.y + 16,
      self._warning_header.rect.width,
      self._warning_header.rect.height,
    ))

    self._back_button.set_opacity(1 - self._uninstall_slider.slider_percentage)
    self._back_button.render(rl.Rectangle(
      self._rect.x + 8,
      self._rect.y + self._rect.height - self._back_button.rect.height,
      self._back_button.rect.width,
      self._back_button.rect.height,
    ))

    self._uninstall_slider.render(rl.Rectangle(
      self._rect.x + self._rect.width - self._uninstall_slider.rect.width,
      self._rect.y + self._rect.height - self._uninstall_slider.rect.height,
      self._uninstall_slider.rect.width,
      self._uninstall_slider.rect.height,
    ))


class TermsPage(SetupTermsPage):
  def __init__(self, on_accept=None, on_decline=None):
    super().__init__(on_accept, on_decline, "decline")

    info_txt = gui_app.texture("icons_mici/setup/green_info.png", 60, 60)
    self._title_header = TermsHeader("terms & conditions", info_txt)

    self._terms_label = UnifiedLabel("You must accept the Terms and Conditions to use openpilot. " +
                                     "Read the latest terms at https://comma.ai/terms before continuing.", 36,
                                     FontWeight.ROMAN)

  @property
  def _content_height(self):
    return self._terms_label.rect.y + self._terms_label.rect.height - self._scroll_panel.get_offset()

  def _render_content(self, scroll_offset):
    self._title_header.set_position(self._rect.x + 16, self._rect.y + 12 + scroll_offset)
    self._title_header.render()

    self._terms_label.render(rl.Rectangle(
      self._rect.x + 16,
      self._title_header.rect.y + self._title_header.rect.height + self.ITEM_SPACING,
      self._rect.width - 100,
      self._terms_label.get_content_height(int(self._rect.width - 100)),
    ))


class OnboardingWindow(Widget):
  def __init__(self):
    super().__init__()
    self._accepted_terms: bool = ui_state.params.get("HasAcceptedTerms") == terms_version
    self._training_done: bool = ui_state.params.get("CompletedTrainingVersion") == training_version

    self._state = OnboardingState.TERMS if not self._accepted_terms else OnboardingState.ONBOARDING

    self.set_rect(rl.Rectangle(0, 0, 458, gui_app.height))

    # Windows
    self._terms = TermsPage(on_accept=self._on_terms_accepted, on_decline=self._on_terms_declined)
    self._training_guide = TrainingGuide(completed_callback=self._on_completed_training)
    self._decline_page = DeclinePage(back_callback=self._on_decline_back)

  @property
  def completed(self) -> bool:
    return self._accepted_terms and self._training_done

  def _on_terms_declined(self):
    self._state = OnboardingState.DECLINE

  def _on_decline_back(self):
    self._state = OnboardingState.TERMS

  def close(self):
    ui_state.params.put_bool("IsDriverViewEnabled", False)
    gui_app.set_modal_overlay(None)

  def _on_terms_accepted(self):
    ui_state.params.put("HasAcceptedTerms", terms_version)
    self._state = OnboardingState.ONBOARDING

  def _on_completed_training(self):
    ui_state.params.put("CompletedTrainingVersion", training_version)
    self.close()

  def _render(self, _):
    if self._state == OnboardingState.TERMS:
      self._terms.render(self._rect)
    elif self._state == OnboardingState.ONBOARDING:
      self._training_guide.render(self._rect)
    elif self._state == OnboardingState.DECLINE:
      self._decline_page.render(self._rect)
    return -1
