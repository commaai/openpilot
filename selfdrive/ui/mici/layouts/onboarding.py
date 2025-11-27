from enum import IntEnum
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import SmallButton
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
    self.driver_state_renderer = DriverStateRenderer(confirm_mode=True, confirm_callback=confirm_callback)
    self.driver_state_renderer.set_rect(rl.Rectangle(0, 0, 200, 200))
    self.driver_state_renderer.load_icons()

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
    driver_state_rect = (
      rect.x if is_rhd else rect.x + rect.width - self.driver_state_renderer.rect.width,
      rect.y + (rect.height - self.driver_state_renderer.rect.height) / 2,
    )
    self.driver_state_renderer.set_position(*driver_state_rect)
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


class TrainingGuideDMTutorial(Widget):
  def __init__(self, continue_callback):
    super().__init__()
    self._title_header = TermsHeader("fill the circle to continue", gui_app.texture("icons_mici/setup/green_dm.png", 60, 60))

    self._original_continue_callback = continue_callback

    # Wrap the continue callback to restore settings
    def wrapped_continue_callback():
      self._restore_settings()
      continue_callback()

    self._dialog = DriverCameraSetupDialog(wrapped_continue_callback)

    # Disable driver monitoring model when device times out for inactivity
    def inactivity_callback():
      ui_state.params.put_bool("IsDriverViewEnabled", False)

    device.add_interactive_timeout_callback(inactivity_callback)

  def show_event(self):
    super().show_event()
    self._dialog.show_event()

    device.set_offroad_brightness(100)
    device.reset_interactive_timeout(300)  # 5 minutes

  def _restore_settings(self):
    device.set_offroad_brightness(None)
    device.reset_interactive_timeout()

  def _update_state(self):
    super()._update_state()
    if device.awake:
      ui_state.params.put_bool("IsDriverViewEnabled", True)

  def _render(self, _):
    self._dialog.render(self._rect)

    rl.draw_rectangle_gradient_v(int(self._rect.x), int(self._rect.y + self._rect.height - self._title_header.rect.height * 1.5 - 32),
                                 int(self._rect.width), int(self._title_header.rect.height * 1.5 + 32),
                                 rl.BLANK, rl.Color(0, 0, 0, 150))
    self._title_header.render(rl.Rectangle(
      self._rect.x + 16,
      self._rect.y + self._rect.height - self._title_header.rect.height - 16,
      self._title_header.rect.width,
      self._title_header.rect.height,
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

    self._steps = [
      TrainingGuideAttentionNotice(continue_callback=self._advance_step),
      TrainingGuidePreDMTutorial(continue_callback=self._advance_step),
      TrainingGuideDMTutorial(continue_callback=self._advance_step),
      TrainingGuideRecordFront(continue_callback=self._advance_step),
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
