/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/ui.h"

class UIStateSP : public UIState {
  Q_OBJECT

public:
  UIStateSP(QObject *parent = 0);
  void updateStatus() override;

signals:
  void uiUpdate(const UIStateSP &s);

private slots:
  void update() override;
};

UIStateSP *uiStateSP();
inline UIStateSP *uiState() { return uiStateSP(); };

// device management class
class DeviceSP : public Device {
  Q_OBJECT

public:
  DeviceSP(QObject *parent = 0);
};

DeviceSP *deviceSP();
inline DeviceSP *device() { return deviceSP(); }
