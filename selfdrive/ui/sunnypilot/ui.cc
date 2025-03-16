/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/ui.h"

#include "common/watchdog.h"

void UIStateSP::updateStatus() {
  UIState::updateStatus();
}

UIStateSP::UIStateSP(QObject *parent) : UIState(parent) {
  sm = std::make_unique<SubMaster>(std::vector<const char*>{
    "modelV2", "controlsState", "liveCalibration", "radarState", "deviceState",
    "pandaStates", "carParams", "driverMonitoringState", "carState", "driverStateV2",
    "wideRoadCameraState", "managerState", "selfdriveState", "longitudinalPlan",
    "modelManagerSP", "selfdriveStateSP", "longitudinalPlanSP",
  });

  // update timer
  timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &UIStateSP::update);
  timer->start(1000 / UI_FREQ);
}

// This method overrides completely the update method from the parent class intentionally.
void UIStateSP::update() {
  update_sockets(this);
  update_state(this);
  updateStatus();

  if (sm->frame % UI_FREQ == 0) {
    watchdog_kick(nanos_since_boot());
  }
  emit uiUpdate(*this);
}

DeviceSP::DeviceSP(QObject *parent) : Device(parent) {
  QObject::connect(uiStateSP(), &UIStateSP::uiUpdate, this, &DeviceSP::update);
}

UIStateSP *uiStateSP() {
  static UIStateSP ui_state;
  return &ui_state;
}

void UIStateSP::setSunnylinkRoles(const std::vector<RoleModel>& roles) {
  sunnylinkRoles = roles;
  emit sunnylinkRolesChanged(roles);
}

void UIStateSP::setSunnylinkDeviceUsers(const std::vector<UserModel>& users) {
  sunnylinkUsers = users;
  emit sunnylinkDeviceUsersChanged(users);
}

DeviceSP *deviceSP() {
  static DeviceSP _device;
  return &_device;
}
