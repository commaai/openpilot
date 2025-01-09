/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/qt/offroad/offroad_home.h"

class OffroadHomeSP : public OffroadHome {
  Q_OBJECT

public:
  explicit OffroadHomeSP(QWidget *parent = 0);
};
