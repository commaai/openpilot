/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "../../sunnypilot/selfdrive/ui/qt/home.h"

HomeWindowSP::HomeWindowSP(QWidget *parent) : HomeWindow(parent) {
}

void HomeWindowSP::updateState(const UIState &s) {
  HomeWindow::updateState(s);
}

void HomeWindowSP::mousePressEvent(QMouseEvent *e) {
  HomeWindow::mousePressEvent(e);
}
