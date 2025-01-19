/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/buttons.h"

#include <QPainter>

ExperimentalButtonSP::ExperimentalButtonSP(QWidget *parent) : ExperimentalButton(parent) {
  QObject::disconnect(uiState(), &UIState::uiUpdate, this, &ExperimentalButton::updateState);
  QObject::connect(uiState(), &UIState::uiUpdate, this, &ExperimentalButtonSP::updateState);
}

void ExperimentalButtonSP::updateState(const UIState &s) {
  ExperimentalButton::updateState(s);
  const auto long_plan_sp = (*s.sm)["longitudinalPlanSP"].getLongitudinalPlanSP();

  int mode = int(long_plan_sp.getDec().getState());
  if ((long_plan_sp.getDec().getActive() != dynamic_experimental_control) || (mode != dec_mpc_mode)) {
    dynamic_experimental_control = long_plan_sp.getDec().getActive();
    dec_mpc_mode = mode;
    update();
  }
}

void ExperimentalButtonSP::drawButton(QPainter &p) {
  if (dynamic_experimental_control) {
    QPixmap left_half = engage_img.copy(0, 0, engage_img.width() / 2, engage_img.height());
    QPixmap right_half = experimental_img.copy(experimental_img.width() / 2, 0, experimental_img.width() / 2, experimental_img.height());

    QPixmap combined_img(engage_img.width(), engage_img.height());
    combined_img.fill(Qt::transparent);

    QPainter combined_painter(&combined_img);

    combined_painter.setOpacity(dec_mpc_mode == 1 ? 0.1 : 1.0);
    combined_painter.drawPixmap(0, 0, left_half);

    combined_painter.setOpacity(dec_mpc_mode == 1 ? 1.0 : 0.1);
    combined_painter.drawPixmap(engage_img.width() / 2, 0, right_half);

    combined_painter.end();

    drawIcon(p, QPoint(btn_size / 2, btn_size / 2), combined_img, QColor(0, 0, 0, 166), (isDown() || !engageable) ? 0.6 : 1.0);
  } else {
    ExperimentalButton::drawButton(p);
  }
}
