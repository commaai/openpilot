#include "selfdrive/ui/qt/offroad/developer_panel.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"
#include "selfdrive/ui/qt/widgets/controls.h"

DeveloperPanel::DeveloperPanel(SettingsWindow *parent) : ListWidget(parent) {
  adbToggle = new ParamControl("AdbEnabled", tr("Enable ADB"),
            tr("ADB (Android Debug Bridge) allows connecting to your device over USB or over the network. See https://docs.comma.ai/how-to/connect-to-comma for more info."), "");
  addItem(adbToggle);

  // SSH keys
  addItem(new SshToggle());
  addItem(new SshControl());

  joystickToggle = new ParamControl("JoystickDebugMode", tr("Joystick Debug Mode"), "", "");
  QObject::connect(joystickToggle, &ParamControl::toggleFlipped, [=](bool state) {
    params.putBool("LongitudinalManeuverMode", false);
    longManeuverToggle->refresh();
    params.putBool("OnroadCycleRequested", true);
  });
  addItem(joystickToggle);

  longManeuverToggle = new ParamControl("LongitudinalManeuverMode", tr("Longitudinal Maneuver Mode"), "", "");
  QObject::connect(longManeuverToggle, &ParamControl::toggleFlipped, [=](bool state) {
    params.putBool("JoystickDebugMode", false);
    joystickToggle->refresh();
    params.putBool("OnroadCycleRequested", true);
  });
  addItem(longManeuverToggle);

  alphaLongitudinalToggle = new ParamControl(
    "AlphaLongitudinalEnabled",
    tr("openpilot Longitudinal Control (Alpha)"),
    QString("<b>%1</b><br><br>%2")
      .arg(tr("WARNING: openpilot longitudinal control is in alpha for this car and will disable Automatic Emergency Braking (AEB)."))
      .arg(tr("On this car, openpilot defaults to the car's built-in ACC instead of openpilot's longitudinal control. "
              "Enable this to switch to openpilot longitudinal control. Enabling Experimental mode is recommended when enabling openpilot longitudinal control alpha.")),
    ""
  );
  alphaLongitudinalToggle->setConfirmation(true, false);
  QObject::connect(alphaLongitudinalToggle, &ParamControl::toggleFlipped, [=]() {
    updateToggles(offroad);
    params.putBool("OnroadCycleRequested", true);
  });
  addItem(alphaLongitudinalToggle);

  // Joystick and longitudinal maneuvers should be hidden on release branches
  is_release = params.getBool("IsReleaseBranch");

  // Update toggles on offroad transition and engaged state change
  QObject::connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    updateToggles(offroad, engaged);
  });
  QObject::connect(uiState(), &UIState::engagedChanged, [=](bool engaged) {
    updateToggles(offroad, engaged);
  });
}

void DeveloperPanel::updateToggles(bool _offroad, bool _engaged) {
  for (auto btn : findChildren<ParamControl *>()) {
    btn->setVisible(!is_release);

    /*
     * alphaLongitudinalToggle should be toggelable when:
     * - visible, and
     * - during onroad & offroad states
     * - not engaged
     */
    if (btn != joystickToggle && btn != longManeuverToggle && btn != alphaLongitudinalToggle) {
      btn->setEnabled(_offroad);
    } else {
      btn->setEnabled(!_engaged);
    }
  }

  // longManeuverToggle and alphaLongitudinalToggle should not be toggleable if the car does not have longitudinal control
  auto cp_bytes = params.get("CarParamsPersistent");
  if (!cp_bytes.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

    if (!true || is_release) {
      params.remove("AlphaLongitudinalEnabled");
      alphaLongitudinalToggle->setEnabled(false);
    }

    /*
     * alphaLongitudinalToggle should be visible when:
     * - is not a release branch, and
     * - the car supports experimental longitudinal control (alpha)
     */
    alphaLongitudinalToggle->setVisible(true && !is_release);

    longManeuverToggle->setEnabled(hasLongitudinalControl(CP) && _offroad);
  } else {
    longManeuverToggle->setEnabled(false);
    alphaLongitudinalToggle->setVisible(false);
  }
  alphaLongitudinalToggle->refresh();

  offroad = _offroad;
  engaged = _engaged;
}

void DeveloperPanel::showEvent(QShowEvent *event) {
  updateToggles(offroad);
}
