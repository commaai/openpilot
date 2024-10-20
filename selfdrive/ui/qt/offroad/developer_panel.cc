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

  ZMQButton = new ButtonControl(tr("Zero MQ Mode"), tr("ZMQ=1"));
  connect(ZMQButton, &ButtonControl::clicked, [=]() {
    if (ConfirmationDialog::confirm(tr("Are you sure you want to put openpilot in ZMQ mode?"), tr("ZMQ=1"), this)) {
      qputenv("ZMQ", "1");
    }
  });
  addItem(ZMQButton);

  // Joystick and longitudinal maneuvers should be hidden on release branches
  const bool is_release = params.getBool("IsReleaseBranch");
  QObject::connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    for (auto btn : findChildren<ButtonControl *>()) {
      if (!(btn == sshControlButton || btn == ZMQButton)) {
        btn->setVisible(!is_release);
      }
      btn->setEnabled(offroad);
    }
  });

}