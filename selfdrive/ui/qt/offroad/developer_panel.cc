#include <QDebug>
#include <QProcess>
#include "selfdrive/ui/qt/offroad/developer_panel.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "common/util.h"

DeveloperPanel::DeveloperPanel(SettingsWindow *parent) : ListWidget(parent) {
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

  adbToggle = new ParamControl("AdbOverTcp", tr("ADB over TCP"), tr("Enable ADB over TCP on port 5555"), "");
  QObject::connect(adbToggle, &ParamControl::toggleFlipped, [=](bool state) {
    QString command = state
                      ? "setprop service.adb.tcp.port 5555 && sudo systemctl start adbd"
                      : "sudo systemctl stop adbd";
    int exitCode = QProcess::execute("sh", {"-c", command});
    if (exitCode != 0) {
      qWarning() << "Failed to execute ADB command: " << command;
    }
  });
  addItem(adbToggle);

  is_release = params.getBool("IsReleaseBranch");

  QObject::connect(uiState(), &UIState::offroadTransition, this, &DeveloperPanel::updateToggles);
}

void DeveloperPanel::updateToggles(bool _offroad) {
  for (auto btn : findChildren<ParamControl *>()) {
    btn->setVisible(!is_release);
    btn->setEnabled(_offroad);
  }

  auto cp_bytes = params.get("CarParamsPersistent");
  if (!cp_bytes.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();
    longManeuverToggle->setEnabled(hasLongitudinalControl(CP) && _offroad);
  } else {
    longManeuverToggle->setEnabled(false);
  }

  adbToggle->setEnabled(_offroad);

  offroad = _offroad;
}

void DeveloperPanel::showEvent(QShowEvent *event) {
  updateToggles(offroad);
}