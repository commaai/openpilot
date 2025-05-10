/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle_panel.h"

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/brand_settings_factory.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/brands.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

VehiclePanel::VehiclePanel(QWidget *parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 20, 50, 20);

  ListWidget *list = new ListWidget(this);

  platformSelector = new PlatformSelector();
  QObject::connect(platformSelector, &PlatformSelector::refreshPanel, this, &VehiclePanel::updateBrandSettings);
  list->addItem(platformSelector);

  brandSettingsContainer = new QWidget(this);
  brandSettingsContainerLayout = new QVBoxLayout(brandSettingsContainer);
  brandSettingsContainerLayout->setContentsMargins(0, 0, 0, 0);
  brandSettingsContainerLayout->setSpacing(0);
  list->addItem(brandSettingsContainer);

  ScrollViewSP *scroller = new ScrollViewSP(list, this);
  main_layout->addWidget(scroller);

  currentBrandSettings = nullptr;

  QObject::connect(uiState(), &UIState::offroadTransition, this, &VehiclePanel::updatePanel);
}

void VehiclePanel::showEvent(QShowEvent *event) {
  updatePanel(offroad);
}

void VehiclePanel::updatePanel(bool _offroad) {
  offroad = _offroad;
  platformSelector->refresh(_offroad);
  updateBrandSettings();
}

void VehiclePanel::updateBrandSettings() {
  if (!isVisible()) {
    return;
  }

  if (currentBrandSettings) {
    brandSettingsContainerLayout->removeWidget(currentBrandSettings);
    delete currentBrandSettings;
    currentBrandSettings = nullptr;
  }

  if (BrandSettingsFactory::isBrandSupported(platformSelector->brand)) {
    currentBrandSettings = BrandSettingsFactory::createBrandSettings(platformSelector->brand, this);
    if (currentBrandSettings) {
      currentBrandSettings->setContentsMargins(0, 0, 0, 0);
      brandSettingsContainerLayout->addWidget(currentBrandSettings);
      currentBrandSettings->updatePanel(offroad);
    }
  }
}
