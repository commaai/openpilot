/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <memory>

#include "selfdrive/ui/qt/sidebar.h"

#include "selfdrive/ui/sunnypilot/ui.h"

class SidebarSP : public Sidebar {
  Q_OBJECT

public slots:
  void updateState(const UIStateSP &s);

public:
  explicit SidebarSP(QWidget *parent = 0);

private:
  void paintSidebar(QPainter &p) override;
};
