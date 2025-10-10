from cereal import log
from openpilot.common.params import Params, UnknownKeyName
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import multiple_button_item, toggle_item
from openpilot.system.ui.widgets.scroller import Scroller

from openpilot.selfdrive.ui.ui_state import ui_state

PERSONALITY_TO_INT = log.LongitudinalPersonality.schema.enumerants

# Description constants
DESCRIPTIONS = {
  "OpenpilotEnabledToggle": (
    "Use the openpilot system for adaptive cruise control and lane keep driver assistance. " +
    "Your attention is required at all times to use this feature."
  ),
  "DisengageOnAccelerator": "When enabled, pressing the accelerator pedal will disengage openpilot.",
  "LongitudinalPersonality": (
    "Standard is recommended. In aggressive mode, openpilot will follow lead cars closer and be more aggressive with the gas and brake. " +
    "In relaxed mode openpilot will stay further away from lead cars. On supported cars, you can cycle through these personalities with " +
    "your steering wheel distance button."
  ),
  "IsLdwEnabled": (
    "Receive alerts to steer back into the lane when your vehicle drifts over a detected lane line " +
    "without a turn signal activated while driving over 31 mph (50 km/h)."
  ),
  "AlwaysOnDM": "Enable driver monitoring even when openpilot is not engaged.",
  'RecordFront': "Upload data from the driver facing camera and help improve the driver monitoring algorithm.",
  "IsMetric": "Display speed in km/h instead of mph.",
  "RecordAudio": "Record and store microphone audio while driving. The audio will be included in the dashcam video in comma connect.",
}


class TogglesLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()

    # param, title, desc, icon, needs_restart
    self._toggle_defs = {
      "OpenpilotEnabledToggle": (
        "Enable openpilot",
        DESCRIPTIONS["OpenpilotEnabledToggle"],
        "chffr_wheel.png",
        True,
      ),
      "ExperimentalMode": (
        "Experimental Mode",
        "",
        "experimental_white.png",
        False,
      ),
      "DisengageOnAccelerator": (
        "Disengage on Accelerator Pedal",
        DESCRIPTIONS["DisengageOnAccelerator"],
        "disengage_on_accelerator.png",
        False,
      ),
      "IsLdwEnabled": (
        "Enable Lane Departure Warnings",
        DESCRIPTIONS["IsLdwEnabled"],
        "warning.png",
        False,
      ),
      "AlwaysOnDM": (
        "Always-On Driver Monitoring",
        DESCRIPTIONS["AlwaysOnDM"],
        "monitoring.png",
        False,
      ),
      "RecordFront": (
        "Record and Upload Driver Camera",
        DESCRIPTIONS["RecordFront"],
        "monitoring.png",
        True,
      ),
      "RecordAudio": (
        "Record and Upload Microphone Audio",
        DESCRIPTIONS["RecordAudio"],
        "microphone.png",
        True,
      ),
      "IsMetric": (
        "Use Metric System",
        DESCRIPTIONS["IsMetric"],
        "metric.png",
        False,
      ),
    }

    self._long_personality_setting = multiple_button_item(
      "Driving Personality",
      DESCRIPTIONS["LongitudinalPersonality"],
      buttons=["Aggressive", "Standard", "Relaxed"],
      button_width=255,
      callback=self._set_longitudinal_personality,
      selected_index=self._params.get("LongitudinalPersonality", return_default=True),
      icon="speed_limit.png"
    )

    self._toggles = {}
    for param, (title, desc, icon, needs_restart) in self._toggle_defs.items():
      toggle = toggle_item(
        title,
        desc,
        self._params.get_bool(param),
        callback=lambda state, p=param: self._toggle_callback(state, p),
        icon=icon,
      )

      try:
        locked = self._params.get_bool(param + "Lock")
      except UnknownKeyName:
        locked = False
      toggle.action_item.set_enabled(not locked)

      if needs_restart and not locked:
        toggle.set_description(toggle.description + " Changing this setting will restart openpilot if the car is powered on.")

      self._toggles[param] = toggle

      # insert longitudinal personality after NDOG toggle
      if param == "DisengageOnAccelerator":
        self._toggles["LongitudinalPersonality"] = self._long_personality_setting

    self._update_experimental_mode_icon()
    self._scroller = Scroller(list(self._toggles.values()), line_separator=True, spacing=0)

  def _update_state(self):
    """
      // set up uiState update for personality setting
      QObject::connect(uiState(), &UIState::uiUpdate, this, &TogglesPanel::updateState);

      for (auto &[param, title, desc, icon, needs_restart] : toggle_defs) {
        auto toggle = new ParamControl(param, title, desc, icon, this);

        bool locked = params.getBool((param + "Lock").toStdString());
        toggle->setEnabled(!locked);

        if (needs_restart && !locked) {
          toggle->setDescription(toggle->getDescription() + tr(" Changing this setting will restart openpilot if the car is powered on."));


          QObject::connect(toggle, &ParamControl::toggleFlipped, [=](bool state) {
            params.putBool("OnroadCycleRequested", true);
          });
        }

        addItem(toggle);
        toggles[param.toStdString()] = toggle;

        // insert longitudinal personality after NDOG toggle
        if (param == "DisengageOnAccelerator") {
          addItem(long_personality_setting);
        }
      }

      // Toggles with confirmation dialogs
      toggles["ExperimentalMode"]->setActiveIcon("../assets/icons/experimental.svg");
      toggles["ExperimentalMode"]->setConfirmation(true, true);
    }
    """

    """
    void TogglesPanel::updateState(const UIState &s) {
      const SubMaster &sm = *(s.sm);

      if (sm.updated("selfdriveState")) {
        auto personality = sm["selfdriveState"].getSelfdriveState().getPersonality();
        if (personality != s.scene.personality && s.scene.started && isVisible()) {
          long_personality_setting->setCheckedButton(static_cast<int>(personality));
        }
        uiState()->scene.personality = personality;
      }
    }
    """

    if ui_state.sm.updated["selfdriveState"]:
      personality = PERSONALITY_TO_INT[ui_state.sm["selfdriveState"].personality]
      if personality != ui_state.personality and ui_state.started:
        self._long_personality_setting.action_item.set_selected_button(personality)
      ui_state.personality = personality

    # these toggles need restart, block while engaged
    for toggle_def in self._toggle_defs:
      if self._toggle_defs[toggle_def][3]:
        self._toggles[toggle_def].action_item.set_enabled(not ui_state.engaged)

  def show_event(self):
    self._update_toggles()

  def _update_toggles(self):
    """
    const QString e2e_description = QString("%1<br>"
                                            "<h4>%2</h4><br>"
                                            "%3<br>"
                                            "<h4>%4</h4><br>"
                                            "%5<br>")
                                    .arg(tr("openpilot defaults to driving in <b>chill mode</b>. Experimental mode enables <b>alpha-level features</b> that aren't ready for chill mode. Experimental features are listed below:"))
                                    .arg(tr("End-to-End Longitudinal Control"))
                                    .arg(tr("Let the driving model control the gas and brakes. openpilot will drive as it thinks a human would, including stopping for red lights and stop signs. "
                                            "Since the driving model decides the speed to drive, the set speed will only act as an upper bound. This is an alpha quality feature; "
                                            "mistakes should be expected."))
                                    .arg(tr("New Driving Visualization"))
                                    .arg(tr("The driving visualization will transition to the road-facing wide-angle camera at low speeds to better show some turns. The Experimental mode logo will also be shown in the top right corner."));

    const bool is_release = params.getBool("IsReleaseBranch");
    auto cp_bytes = params.get("CarParamsPersistent");
    if (!cp_bytes.empty()) {
      AlignedBuffer aligned_buf;
      capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
      cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

      if (hasLongitudinalControl(CP)) {
        // normal description and toggle
        experimental_mode_toggle->setEnabled(true);
        experimental_mode_toggle->setDescription(e2e_description);
        long_personality_setting->setEnabled(true);
      } else {
        // no long for now
        experimental_mode_toggle->setEnabled(false);
        long_personality_setting->setEnabled(false);
        params.remove("ExperimentalMode");

        const QString unavailable = tr("Experimental mode is currently unavailable on this car since the car's stock ACC is used for longitudinal control.");

        QString long_desc = unavailable + " " + \
                            tr("openpilot longitudinal control may come in a future update.");
        if (CP.getAlphaLongitudinalAvailable()) {
          if (is_release) {
            long_desc = unavailable + " " + tr("An alpha version of openpilot longitudinal control can be tested, along with Experimental mode, on non-release branches.");
          } else {
            long_desc = tr("Enable the openpilot longitudinal control (alpha) toggle to allow Experimental mode.");
          }
        }
        experimental_mode_toggle->setDescription("<b>" + long_desc + "</b><br><br>" + e2e_description);
      }

      experimental_mode_toggle->refresh();
    } else {
      experimental_mode_toggle->setDescription(e2e_description);
    }
    """

    e2e_description = (
      "openpilot defaults to driving in <b>chill mode</b>. Experimental mode enables <b>alpha-level features</b> that aren't ready for chill mode. " +
      "Experimental features are listed below:<br>" +
      "<h4>End-to-End Longitudinal Control</h4><br>" +
      "Let the driving model control the gas and brakes. openpilot will drive as it thinks a human would, including stopping for red lights and stop signs. " +
      "Since the driving model decides the speed to drive, the set speed will only act as an upper bound. This is an alpha quality feature; " +
      "mistakes should be expected.<br>" +
      "<h4>New Driving Visualization</h4><br>" +
      "The driving visualization will transition to the road-facing wide-angle camera at low speeds to better show some turns. " +
      "The Experimental mode logo will also be shown in the top right corner."
    )

    is_release = self._params.get_bool("IsReleaseBranch")

    if ui_state.CP is not None:
      if ui_state.has_longitudinal_control:
        self._toggles["ExperimentalMode"].action_item.set_enabled(True)
        self._toggles["ExperimentalMode"].set_description(e2e_description)
        self._long_personality_setting.action_item.set_enabled(True)
      else:
        # no long for now
        self._toggles["ExperimentalMode"].action_item.set_enabled(False)
        self._toggles["ExperimentalMode"].action_item.set_state(False)
        self._long_personality_setting.action_item.set_enabled(False)
        self._params.remove("ExperimentalMode")

        unavailable = "Experimental mode is currently unavailable on this car since the car's stock ACC is used for longitudinal control."

        long_desc = unavailable + " openpilot longitudinal control may come in a future update."
        if ui_state.CP.getAlphaLongitudinalAvailable():
          if is_release:
            long_desc = unavailable + " " + ("An alpha version of openpilot longitudinal control can be tested, along with " +
                                             "Experimental mode, on non-release branches.")
          else:
            long_desc = "Enable the openpilot longitudinal control (alpha) toggle to allow Experimental mode."

        self._toggles["ExperimentalMode"].set_description("<b>" + long_desc + "</b><br><br>" + e2e_description)

    self._update_experimental_mode_icon()

  def _render(self, rect):
    self._scroller.render(rect)

  def _update_experimental_mode_icon(self):
    icon = "experimental.png" if self._toggles["ExperimentalMode"].action_item.get_state() else "experimental_white.png"
    self._toggles["ExperimentalMode"].set_icon(icon)

  def _toggle_callback(self, state: bool, param: str):
    print(f"Toggled {param} to {state}")
    if param == "ExperimentalMode":
      self._update_experimental_mode_icon()

    self._params.put_bool(param, state)
    if self._toggle_defs[param][3]:
      self._params.put_bool("OnroadCycleRequested", True)

  def _set_longitudinal_personality(self, button_index: int):
    self._params.put("LongitudinalPersonality", button_index)
