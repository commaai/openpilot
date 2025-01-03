/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "../../sunnypilot/selfdrive/ui/qt/sidebar.h"

#include <cmath>
#include <QMouseEvent>

#include "selfdrive/ui/qt/util.h"

SidebarSP::SidebarSP(QWidget *parent) : Sidebar(parent) {
  QObject::disconnect(uiState(), &UIState::uiUpdate, this, &Sidebar::updateState);
  QObject::connect(uiStateSP(), &UIStateSP::uiUpdate, this, &SidebarSP::updateState);
}

void SidebarSP::updateState(const UIStateSP &s) {
  if (!isVisible()) return;
  Sidebar::updateState(s);
}

void SidebarSP::paintSidebar(QPainter &p) {
  Sidebar::paintSidebar(p);
}
