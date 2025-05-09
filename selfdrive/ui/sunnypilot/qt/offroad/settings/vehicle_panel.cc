/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle_panel.h"

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/hyundai_settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

VehiclePanel::VehiclePanel(QWidget *parent) : QFrame(parent) {
  main_layout = new QStackedLayout(this);
  ListWidget *list = new ListWidget(this);

  vehicleScreen = new QWidget(this);
  QVBoxLayout *vlayout = new QVBoxLayout(vehicleScreen);
  vlayout->setContentsMargins(50, 20, 50, 20);

  platformSelector = new PlatformSelector();
  QObject::connect(platformSelector, &PlatformSelector::refreshPanel, this, &VehiclePanel::updateBrandSettings);
  list->addItem(platformSelector);

  ScrollViewSP *scroller = new ScrollViewSP(list, this);
  vlayout->addWidget(scroller);

  hyundaiSettings = new HyundaiSettings(this);
  vlayout->addWidget(hyundaiSettings);
  hyundaiSettings->setVisible(false);

  QObject::connect(uiState(), &UIState::offroadTransition, this, &VehiclePanel::updatePanel);

  main_layout->addWidget(vehicleScreen);
  main_layout->setCurrentWidget(vehicleScreen);
}

void VehiclePanel::showEvent(QShowEvent *event) {
  updatePanel(offroad);
}

void VehiclePanel::updatePanel(bool _offroad) {
  platformSelector->refresh(_offroad);

  updateBrandSettings();

  offroad = _offroad;
}

void VehiclePanel::updateBrandSettings() {
  if (!isVisible()) {
    return;
  }

  resetBrandSettings();

  QString brand = platformSelector->getPlatformBundle("brand").toString();
  if (brand == "hyundai") {
    hyundaiSettings->setVisible(true);
  }
}

void VehiclePanel::resetBrandSettings() {
  hyundaiSettings->setVisible(false);
}
