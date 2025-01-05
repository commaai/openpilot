/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

class SunnylinkPanel : public QFrame {
  Q_OBJECT

public:
  explicit SunnylinkPanel(QWidget *parent = nullptr);
  void showEvent(QShowEvent *event) override;

public slots:
  void updatePanel(bool _offroad);

private:
  Params params;
  QStackedLayout *main_layout = nullptr;
  QWidget *sunnylinkScreen = nullptr;
  ScrollViewSP *sunnylinkScroller = nullptr;
  bool offroad;

  ParamControl *sunnylinkEnabledBtn;
};
