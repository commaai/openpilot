/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/mazda_settings.h"

#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

MazdaSettings::MazdaSettings(QWidget *parent) : BrandSettingsInterface(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  ListWidget *list = new ListWidget(this, false);

  main_layout->addWidget(new ScrollViewSP(list, this));
}

void MazdaSettings::updatePanel(bool _offroad) {
  updateSettings();

  offroad = _offroad;
}

void MazdaSettings::updateSettings() {
  if (!isVisible()) {
    return;
  }
}
