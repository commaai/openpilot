/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/lateral/mads_settings.h"

#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

MadsSettings::MadsSettings(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 20, 50, 20);
  main_layout->setSpacing(20);

  // Back button
  PanelBackButton *back = new PanelBackButton();
  connect(back, &QPushButton::clicked, [=]() { emit backPress(); });
  main_layout->addWidget(back, 0, Qt::AlignLeft);

  ListWidget *list = new ListWidget(this, false);
  // Main cruise
  madsMainCruiseToggle = new ParamControl(
    "MadsMainCruiseAllowed",
    tr("Toggle with Main Cruise"),
    tr("Note: For vehicles without LFA/LKAS button, disabling this will prevent lateral control engagement."),
    "");
  list->addItem(madsMainCruiseToggle);

  // Unified Engagement Mode
  madsUnifiedEngagementModeToggle = new ParamControl(
    "MadsUnifiedEngagementMode",
    tr("Unified Engagement Mode (UEM)"),
    QString("%1<br>"
            "<h4>%2</h4>")
    .arg(tr("Engage lateral and longitudinal control with cruise control engagement."))
    .arg(tr("Note: Once lateral control is engaged via UEM, it will remain engaged until it is manually disabled via the MADS button or car shut off.")),
    "");
  list->addItem(madsUnifiedEngagementModeToggle);

  // Steering Mode On Brake
  std::vector<QString> lateral_on_brake_texts{tr("Remain Active"), tr("Pause Steering"), tr("Disengage")};
  madsSteeringMode = new ButtonParamControl(
    "MadsSteeringMode",
    tr("Steering Mode on Brake Pedal"),
    tr("Choose how Automatic Lane Centering (ALC) behaves after the brake pedal is manually pressed in sunnypilot.\n\n"
       "Remain Active: ALC will remain active even after the brake pedal is pressed.\n"
       "Pause Steering: ALC will be paused when the brake pedal is manually pressed."),
       "Disengage: ALC will be disengaged after the brake pedal is pressed.\n"
    "",
    lateral_on_brake_texts,
    500);
  list->addItem(madsSteeringMode);

  QObject::connect(uiState(), &UIState::offroadTransition, this, &MadsSettings::updateToggles);

  main_layout->addWidget(new ScrollViewSP(list, this));
}

void MadsSettings::showEvent(QShowEvent *event) {
  updateToggles(offroad);
}

void MadsSettings::updateToggles(bool _offroad) {
  madsSteeringMode->setEnabled(_offroad);

  offroad = _offroad;
}
