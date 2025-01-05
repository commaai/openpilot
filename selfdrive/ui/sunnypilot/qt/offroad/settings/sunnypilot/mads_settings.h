/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

class MadsSettings : public QWidget {
  Q_OBJECT

public:
  explicit MadsSettings(QWidget *parent = nullptr);

  void showEvent(QShowEvent *event) override;

signals:
  void backPress();

public slots:
  void updateToggles(bool _offroad);

private:
  Params params;
  bool offroad;

  ParamControl *madsMainCruiseToggle;
  ParamControl *madsUnifiedEngagementModeToggle;
  ButtonParamControl *madsPauseLateralOnBrake;
};
