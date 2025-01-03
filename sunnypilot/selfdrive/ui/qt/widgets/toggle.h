/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/qt/widgets/toggle.h"

class ToggleSP : public Toggle {
  Q_OBJECT

public:
  explicit ToggleSP(QWidget *parent = nullptr);

protected:
  void paintEvent(QPaintEvent *) override;
};
