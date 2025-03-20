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
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/lateral/mads_settings.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class LateralPanel : public QFrame {
  Q_OBJECT

public:
  explicit LateralPanel(SettingsWindowSP *parent = nullptr);
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent* event) override;

public slots:
  void updateToggles(bool _offroad);

private:
  QStackedLayout* main_layout = nullptr;
  QWidget* sunnypilotScreen = nullptr;
  ScrollViewSP *sunnypilotScroller = nullptr;
  std::vector<ParamControl *> toggleOffroadOnly;
  bool offroad;

  ParamControl *madsToggle;
  PushButtonSP *madsSettingsButton;
  MadsSettings *madsWidget = nullptr;
};
