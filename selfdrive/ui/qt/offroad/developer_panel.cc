#include <QDebug>

#include "selfdrive/ui/qt/offroad/settings.h"
#include "selfdrive/ui/qt/offroad/developer_panel.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"

DeveloperPanel::DeveloperPanel(SettingsWindow *parent) : ListWidget(parent) {
  // SSH keys
  addItem(new SshToggle());
  addItem(new SshControl());

  joystickDebugModeButton = new ButtonControl(tr("Joystick Debug Mode"), tr("JOYSTICK"));
  addItem(joystickDebugModeButton);
  connect(joystickDebugModeButton, &ButtonControl::clicked, [=]() {
    params.putBool("JoystickDebugMode", true);
  });

  LongitudinalManeuverModeButton = new ButtonControl(tr("Longitudinal Maneuver Mode"), tr("MANEUVER"));
  addItem(LongitudinalManeuverModeButton);
  connect(LongitudinalManeuverModeButton, &ButtonControl::clicked, [=]() {
    params.putBool("LongitudinalManeuverMode", true);
  });

  // Joystick and longitudinal maneuvers should be hidden on release branches
  const bool is_release = params.getBool("IsReleaseBranch");
  QObject::connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    for (auto btn : findChildren<ButtonControl *>()) {
      btn->setEnabled(offroad);
      btn->setVisible(!is_release);
    }
  });

}