/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <map>
#include <string>

#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class AutoLaneChangeTimer : public OptionControlSP {
  Q_OBJECT

public:
  AutoLaneChangeTimer();

  void refresh();

signals:
  void toggleUpdated();

private:
  Params params;
};


class LaneChangeSettings : public QWidget {
  Q_OBJECT

public:
  explicit LaneChangeSettings(QWidget* parent = nullptr);
  void showEvent(QShowEvent *event) override;

signals:
  void backPress();

public slots:
  void updateToggles();

private:
  Params params;
  std::map<std::string, ParamControlSP*> toggles;

  AutoLaneChangeTimer *autoLaneChangeTimer;
};
