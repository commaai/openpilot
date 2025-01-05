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
  Q_PROPERTY(ItemStatus sunnylinkStatus MEMBER sunnylink_status NOTIFY valueChanged);
  Q_PROPERTY(QString sidebarTemp MEMBER sidebar_temp_str NOTIFY valueChanged);

public slots:
  void updateState(const UIStateSP &s);

public:
  explicit SidebarSP(QWidget *parent = 0);

private:
  void drawSidebar(QPainter &p) override;

  Params params;
  QString sidebar_temp = "0";
  QString sidebar_temp_str = "0";

protected:
  const QColor progress_color = QColor(3, 132, 252);
  const QColor disabled_color = QColor(128, 128, 128);

  ItemStatus sunnylink_status;
};
