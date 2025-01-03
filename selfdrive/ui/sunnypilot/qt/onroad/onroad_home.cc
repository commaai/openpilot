/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/onroad_home.h"

#include "common/swaglog.h"
#include "selfdrive/ui/qt/util.h"

OnroadWindowSP::OnroadWindowSP(QWidget *parent) : OnroadWindow(parent) {
  QObject::connect(uiStateSP(), &UIStateSP::uiUpdate, this, &OnroadWindowSP::updateState);
  QObject::connect(uiStateSP(), &UIStateSP::offroadTransition, this, &OnroadWindowSP::offroadTransition);
}

void OnroadWindowSP::updateState(const UIStateSP &s) {
  if (!s.scene.started) {
    return;
  }

  OnroadWindow::updateState(s);
}

void OnroadWindowSP::mousePressEvent(QMouseEvent *e) {
  OnroadWindow::mousePressEvent(e);
}

void OnroadWindowSP::offroadTransition(bool offroad) {
  OnroadWindow::offroadTransition(offroad);
}
