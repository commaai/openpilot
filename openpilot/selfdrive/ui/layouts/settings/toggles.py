from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import multiple_button_item, param_toggle_item
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.selfdrive.ui.ui_state import ui_state

PERSONALITY_TO_INT = log.LongitudinalPersonality.schema.enumerants

PERSONALITY_DESCRIPTION = tr_noop(
  "Standard is recommended. In aggressive mode, openpilot will follow lead cars closer and be more aggressive with the gas and brake. " +
  "In relaxed mode openpilot will stay further away from lead cars. On supported cars, you can cycle through these personalities with " +
  "your steering wheel distance button."
)


class TogglesLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._is_release = self._params.get_bool("IsReleaseBranch")

    # param: (title, desc, icon, needs_restart)
    self._toggle_defs = {
      "OpenpilotEnabledToggle": (
        lambda: tr("Enable openpilot"),
        tr_noop(
          "Use the openpilot system for adaptive cruise control and lane keep driver assistance. " +
          "Your attention is required at all times to use this feature."
        ),
        "chffr_wheel.png",
        True,
      ),
      "ExperimentalMode": (
        lambda: tr("Experimental Mode"),
        "",
        "experimental_white.png",
        False,
      ),
      "DisengageOnAccelerator": (
        lambda: tr("Disengage on Accelerator Pedal"),
        tr_noop("When enabled, pressing the accelerator pedal will disengage openpilot."),
        "disengage_on_accelerator.png",
        False,
      ),
      "IsLdwEnabled": (
        lambda: tr("Enable Lane Departure Warnings"),
        tr_noop(
          "Receive alerts to steer back into the lane when your vehicle drifts over a detected lane line " +
          "without a turn signal activated while driving over 31 mph (50 km/h)."
        ),
        "warning.png",
        False,
      ),
      "AlwaysOnDM": (
        lambda: tr("Always-On Driver Monitoring"),
        tr_noop("Enable driver monitoring even when openpilot is not engaged."),
        "monitoring.png",
        False,
      ),
      "RecordFront": (
        lambda: tr("Record and Upload Cabin Camera"),
        tr_noop("Upload data from the cabin camera and help improve the driver monitoring algorithm."),
        "monitoring.png",
        True,
      ),
      "RecordAudio": (
        lambda: tr("Record and Upload Microphone Audio"),
        tr_noop("Record and store microphone audio while driving. The audio will be included in the dashcam video in comma connect."),
        "microphone.png",
        True,
      ),
      "IsMetric": (
        lambda: tr("Use Metric System"),
        tr_noop("Display speed in km/h instead of mph."),
        "metric.png",
        False,
      ),
    }

    self._long_personality_setting = multiple_button_item(
      lambda: tr("Driving Personality"),
      lambda: tr(PERSONALITY_DESCRIPTION),
      buttons=[lambda: tr("Aggressive"), lambda: tr("Standard"), lambda: tr("Relaxed")],
      button_width=255,
      callback=self._set_longitudinal_personality,
      selected_index=self._params.get("LongitudinalPersonality", return_default=True),
      icon="speed_limit.png"
    )

    self._toggles = {}
    for param, (title, desc, icon, needs_restart) in self._toggle_defs.items():
      toggle = param_toggle_item(
        param,
        title,
        desc,
        callback=lambda state, p=param: self._toggle_callback(state, p),
        icon=icon,
      )

      # Make description callable for live translation
      additional_desc = ""
      if needs_restart and not toggle.action_item.locked:
        additional_desc = tr("Changing this setting will restart openpilot if the car is powered on.")
      toggle.set_description(lambda og_desc=toggle.description, add_desc=additional_desc: tr(og_desc) + (" " + tr(add_desc) if add_desc else ""))

      self._toggles[param] = toggle

      # insert longitudinal personality after NDOG toggle
      if param == "DisengageOnAccelerator":
        self._toggles["LongitudinalPersonality"] = self._long_personality_setting

    self._exp_toggle = self._toggles["ExperimentalMode"]
    self._update_experimental_mode_icon()
    self._scroller = Scroller(list(self._toggles.values()), line_separator=True, spacing=0)

    ui_state.add_engaged_transition_callback(self._update_toggles)

  def _update_state(self):
    if ui_state.sm.updated["selfdriveState"]:
      personality = PERSONALITY_TO_INT[ui_state.sm["selfdriveState"].personality]
      if personality != ui_state.personality and ui_state.started:
        self._long_personality_setting.action_item.set_selected_button(personality)
      ui_state.personality = personality

  def show_event(self):
    super().show_event()
    self._scroller.show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()

    e2e_description = tr(
      "<h4>End-to-End Longitudinal Control</h4><br>" +
      "Let the driving model control the gas and brakes. openpilot will drive as it thinks a human would, including stopping for red lights and stop signs. " +
      "Since the driving model decides the speed to drive, the set speed will only act as an upper bound. Mistakes should be expected.<br>"
    )

    if ui_state.CP is not None:
      if ui_state.has_longitudinal_control:
        self._exp_toggle.action_item.set_enabled(True)
        self._exp_toggle.set_description(e2e_description)
        self._long_personality_setting.action_item.set_enabled(True)
      else:
        # no long for now
        self._exp_toggle.action_item.set_enabled(False)
        self._exp_toggle.action_item.set_state(False)
        self._long_personality_setting.action_item.set_enabled(False)
        self._params.remove(self._exp_toggle.action_item.param)

        unavailable = tr("Experimental mode is currently unavailable on this car since the car's stock ACC is used for longitudinal control.")

        long_desc = unavailable + " " + tr("openpilot longitudinal control may come in a future update.")
        if ui_state.CP.alphaLongitudinalAvailable:
          if self._is_release:
            long_desc = unavailable + " " + tr("An alpha version of openpilot longitudinal control can be tested, along with " +
                                               "Experimental mode, on non-release branches.")
          else:
            long_desc = tr("Enable the openpilot longitudinal control (alpha) toggle to allow Experimental mode.")

        self._exp_toggle.set_description("<b>" + long_desc + "</b><br><br>" + e2e_description)
    else:
      self._exp_toggle.set_description(e2e_description)

    self._update_experimental_mode_icon()

    for param, (_, _, _, needs_restart) in self._toggle_defs.items():
      action = self._toggles[param].action_item
      # refresh toggles from params to mirror external changes
      action.refresh()
      # these toggles need restart, block while engaged
      if needs_restart and not action.locked:
        action.set_enabled(not ui_state.engaged)

  def _render(self, rect):
    self._scroller.render(rect)

  def _update_experimental_mode_icon(self):
    icon = "experimental.png" if self._exp_toggle.action_item.get_state() else "experimental_white.png"
    self._exp_toggle.set_icon(icon)

  def _toggle_callback(self, state: bool, param: str):
    if param == self._exp_toggle.action_item.param:
      self._update_experimental_mode_icon()

    if self._toggle_defs[param][3]:
      self._params.put_bool("OnroadCycleRequested", True, block=True)

  def _set_longitudinal_personality(self, button_index: int):
    self._params.put("LongitudinalPersonality", button_index, block=True)
