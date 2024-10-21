#include <QDebug>

#include "selfdrive/ui/qt/offroad/settings.h"
#include "selfdrive/ui/qt/offroad/developer_panel.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"

DeveloperPanel::DeveloperPanel(SettingsWindow *parent) : ListWidget(parent) {
  // SSH keys
  addItem(new SshToggle());
  sshControlButton = new SshControl();
  addItem(sshControlButton);

  joystickDebugModeButton = new ButtonControl(tr("Joystick Debug Mode"), tr("JOYSTICK"));
  connect(joystickDebugModeButton, &ButtonControl::clicked, [=]() {
    if (ConfirmationDialog::confirm(tr("Are you sure you want to openpilot in JoystickDebugMode?"), tr("Joystick"), this)) {
      params.putBool("JoystickDebugMode", true);
    }
  });
  addItem(joystickDebugModeButton);

  LongitudinalManeuverModeButton = new ButtonControl(tr("Longitudinal Maneuver Mode"), tr("MANEUVER"));
  connect(LongitudinalManeuverModeButton, &ButtonControl::clicked, [=]() {
    if (ConfirmationDialog::confirm(tr("Are you sure you want to openpilot in LongitudinalManeuverMode?"), tr("Maneuver"), this)) {
      params.putBool("LongitudinalManeuverMode", true);
    }
  });
  addItem(LongitudinalManeuverModeButton);

  // Joystick and longitudinal maneuvers should be hidden on release branches
  // also the buttons should be not available to push in onroad state
  const bool is_release = params.getBool("IsReleaseBranch");
  QObject::connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    for (auto btn : findChildren<ButtonControl *>()) {
      if (btn != sshControlButton) {
        btn->setVisible(!is_release);
      }
      btn->setEnabled(offroad);
    }
  });

}