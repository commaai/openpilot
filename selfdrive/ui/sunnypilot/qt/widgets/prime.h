/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/qt/widgets/prime.h"

#include "selfdrive/ui/sunnypilot/ui.h"

class SetupWidgetSP : public SetupWidget {
  Q_OBJECT

public:
  explicit SetupWidgetSP(QWidget *parent = nullptr);
};
