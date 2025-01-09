/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

#include "selfdrive/ui/sunnypilot/qt/widgets/drive_stats.h"

class TripsPanel : public QFrame {
  Q_OBJECT

public:
  explicit TripsPanel(QWidget* parent = 0);

private:
  Params params;

  QStackedLayout* center_layout;
  DriveStats *driveStatsWidget;
};
