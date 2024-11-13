#include <QDebug>

#include "selfdrive/ui/qt/offroad/developer_panel.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"
#include "selfdrive/ui/qt/widgets/controls.h"

DeveloperPanel::DeveloperPanel(SettingsWindow *parent) : ListWidget(parent) {
  // SSH keys
  addItem(new SshToggle());
  addItem(new SshControl());

  joystickToggle = new ParamControl("JoystickDebugMode", tr("Joystick Debug Mode"), "", "");
  QObject::connect(joystickToggle, &ParamControl::toggleFlipped, [=](bool state) {
    params.putBool("LongitudinalManeuverMode", false);
    longManeuverToggle->refresh();
  });
  addItem(joystickToggle);

  longManeuverToggle = new ParamControl("LongitudinalManeuverMode", tr("Longitudinal Maneuver Mode"), "", "");
  QObject::connect(longManeuverToggle, &ParamControl::toggleFlipped, [=](bool state) {
    params.putBool("JoystickDebugMode", false);
    joystickToggle->refresh();
  });
  addItem(longManeuverToggle);

  alphaLongToggle = new ParamControl("ExperimentalLongitudinalEnabled", tr("openpilot Longitudinal Control (Alpha)"), "", "../assets/offroad/icon_speed_limit.png");
  QObject::connect(alphaLongToggle, &ParamControl::toggleFlipped, [=](bool state) {
    updateToggles();
  });
  addItem(alphaLongToggle);
  alphaLongToggle->setConfirmation(true, false);

  // Joystick and longitudinal maneuvers should be hidden on release branches
  // also the toggles should be not available to change in onroad state
  const bool is_release = params.getBool("IsReleaseBranch");
  QObject::connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    for (auto btn : findChildren<ParamControl *>()) {
      if (btn != alphaLongToggle) {
        btn->setVisible(!is_release);
      }
      btn->setEnabled(offroad);
    }
  });
}

void DeveloperPanel::updateToggles() {
  const bool is_release = params.getBool("IsReleaseBranch");
  auto cp_bytes = params.get("CarParamsPersistent");
  if (!cp_bytes.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

    const QString alpha_long_description = QString("<b>%1</b><br><br>%2")
      .arg(tr("WARNING: openpilot longitudinal control is in alpha for this car and will disable Automatic Emergency Braking (AEB)."))
      .arg(tr("On this car, openpilot defaults to the car's built-in ACC instead of openpilot's longitudinal control. "
              "Enable this to switch to openpilot longitudinal control. Enabling Experimental mode is recommended when enabling openpilot longitudinal control alpha."));
    alphaLongToggle->setDescription("<b>" + alpha_long_description + "</b>");

    if (!CP.getExperimentalLongitudinalAvailable() && !CP.getOpenpilotLongitudinalControl()) {
      params.remove("ExperimentalLongitudinalEnabled");
      alphaLongToggle->setEnabled(false);
      alphaLongToggle->setDescription("<b>" + tr("openpilot longitudinal control may come in a future update.") + "</b>");
    } else {
      if (is_release) {
        params.remove("ExperimentalLongitudinalEnabled");
        alphaLongToggle->setEnabled(false);
        alphaLongToggle->setDescription("<b>" + tr("An alpha version of openpilot longitudinal control can be tested, along with Experimental mode, on non-release branches.") + "</b>");
      }
    }

    // The car already have openpilot longitudinal control and the toggle not is necessary
    if (CP.getOpenpilotLongitudinalControl()) {
      alphaLongToggle->setVisible(false);
    }

    alphaLongToggle->refresh();
  } else {
    alphaLongToggle->setDescription("<b>" + tr("openpilot longitudinal control may come in a future update.") + "</b>");
    alphaLongToggle->setEnabled(false);
  }
}

void DeveloperPanel::showEvent(QShowEvent *event) {
  updateToggles();
}
