/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class BrandSettingsInterface : public QWidget {
  Q_OBJECT

public:
  explicit BrandSettingsInterface(QWidget *parent = nullptr);
  virtual ~BrandSettingsInterface() = default;

  void updatePanel(bool _offroad);
  virtual void updateSettings() = 0;

protected:
  ListWidget *list = nullptr;
  Params params;
  bool offroad = false;
};
