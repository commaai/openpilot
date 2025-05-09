/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/hyundai_settings.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/platform_selector.h"

class VehiclePanel : public QFrame {
  Q_OBJECT

public:
  explicit VehiclePanel(QWidget *parent = nullptr);
  void showEvent(QShowEvent *event) override;

public slots:
  void updatePanel(bool _offroad);

private:
  void resetBrandSettings();

  QStackedLayout* main_layout = nullptr;
  QWidget* vehicleScreen = nullptr;
  PlatformSelector* platformSelector = nullptr;

  // brand panels
  HyundaiSettings* hyundaiSettings = nullptr;

  bool offroad = false;

private slots:
  void updateBrandSettings();
};
