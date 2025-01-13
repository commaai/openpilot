#include <QDebug>

#include "selfdrive/ui/qt/offroad/developer_panel.h"
#include "selfdrive/ui/qt/widgets/ssh_keys.h"
#include "common/util.h"

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#else
#include "selfdrive/ui/qt/widgets/controls.h"
#endif

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

  // TODO-SP: Move to Vehicles panel when ported back
  hyundaiRadarTracksToggle = new ParamControl(
    "HyundaiRadarTracksToggle",
    tr("Hyundai: Enable Radar Tracks"),
    tr("Enable this to attempt to enable radar tracks for Hyundai, Kia, and Genesis models equipped with the supported Mando SCC radar. "
       "This allows sunnypilot to use radar data for improved lead tracking and overall longitudinal performance."), "");
  hyundaiRadarTracksToggle->setConfirmation(true, false);
  QObject::connect(hyundaiRadarTracksToggle, &ParamControl::toggleFlipped, [=](bool state) {
    updateToggles(offroad);
  });
  addItem(hyundaiRadarTracksToggle);

  auto enableGithubRunner = new ParamControl("EnableGithubRunner", tr("Enable GitHub runner service"), tr("Enables or disables the github runner service."), "");
  addItem(enableGithubRunner);

  // Joystick and longitudinal maneuvers should be hidden on release branches
  is_release = params.getBool("IsReleaseBranch");

  // Toggles should be not available to change in onroad state
  QObject::connect(uiState(), &UIState::offroadTransition, this, &DeveloperPanel::updateToggles);
}

void DeveloperPanel::updateToggles(bool _offroad) {
  for (auto btn : findChildren<ParamControl *>()) {
    btn->setVisible(!is_release);
    btn->setEnabled(_offroad);
  }

  // longManeuverToggle should not be toggleable if the car don't have longitudinal control
  auto cp_bytes = params.get("CarParamsPersistent");
  if (!cp_bytes.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

    auto hyundai = CP.getCarName() == "hyundai";
    auto hyundai_mando_radar = hyundai && (CP.getFlags() & 4096);

    longManeuverToggle->setEnabled(hasLongitudinalControl(CP) && _offroad);
    hyundaiRadarTracksToggle->setVisible(hyundai_mando_radar && hasLongitudinalControl(CP));
  } else {
    longManeuverToggle->setEnabled(false);
    hyundaiRadarTracksToggle->setVisible(false);
  }

  offroad = _offroad;
}

void DeveloperPanel::showEvent(QShowEvent *event) {
  updateToggles(offroad);
}
