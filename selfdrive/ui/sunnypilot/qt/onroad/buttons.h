/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/qt/onroad/buttons.h"

class ExperimentalButtonSP : public ExperimentalButton {
  Q_OBJECT

public:
  explicit ExperimentalButtonSP(QWidget *parent = nullptr);
  void updateState(const UIState &s) override;

private:
  void drawButton(QPainter &p) override;

  bool dynamic_experimental_control;
  int dec_mpc_mode;
};
