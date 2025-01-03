/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/qt/onroad/onroad_home.h"

class OnroadWindowSP : public OnroadWindow {
  Q_OBJECT

public:
  OnroadWindowSP(QWidget *parent = 0);

private:
  void mousePressEvent(QMouseEvent *e) override;

protected slots:
  void offroadTransition(bool offroad) override;
  void updateState(const UIStateSP &s) override;
};
