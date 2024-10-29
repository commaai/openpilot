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

  alphaLongToggle = new ParamControl("ExperimentalLongitudinalEnabled", tr("openpilot Longitudinal Control (Alpha)"), "", "");
  // QObject::connect(alphaLongToggle, &ParamControl::toggleFlipped, [=](bool state) {
  //   params.putBool("JoystickDebugMode", false);
  //   joystickToggle->refresh();
  // });
  addItem(alphaLongToggle);

  // Joystick and longitudinal maneuvers should be hidden on release branches
  // also the toggles should be not available to change in onroad state
  // const bool is_release = params.getBool("IsReleaseBranch");
  const bool is_release = true;
  QObject::connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    for (auto btn : findChildren<ParamControl *>()) {
      if (btn != alphaLongToggle) {
        btn->setVisible(!is_release);
        btn->setEnabled(offroad);
      }
    }
  });

}
