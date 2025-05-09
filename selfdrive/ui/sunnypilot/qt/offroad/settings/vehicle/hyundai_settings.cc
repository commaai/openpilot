/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/vehicle/hyundai_settings.h"

#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"

HyundaiSettings::HyundaiSettings(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  ListWidget *list = new ListWidget(this, false);

  std::vector<QString> tuning_texts{ tr("Off"), tr("Dynamic"), tr("Predictive") };
  longitudinalTuningToggle = new ButtonParamControlSP(
    "HyundaiLongitudinalTuning",
    tr("Custom Longitudinal Tuning"),
    tr("Select a tuning mode.\n"
       "Off: no custom tuning applied.\n"
       "Dynamic: on-the-spot adjustments using dynamic calculations.\n"
       "Predictive: adjusts based on anticipated ACC variation."),
    "",
    tuning_texts,
    500
  );
  longitudinalTuningToggle->showDescription();
  longitudinalTuningToggle->setProperty("originalDesc", longitudinalTuningToggle->getDescription());
  list->addItem(longitudinalTuningToggle);

  QObject::connect(uiState(), &UIState::offroadTransition, this, &HyundaiSettings::updateSettings);

  main_layout->addWidget(new ScrollViewSP(list, this));
}

QString HyundaiSettings::toggleDisableMsg() const {
  if (!has_longitudinal_control) {
    return tr("This feature can only be used with openpilot longitudinal control enabled.");
  }

  if (!offroad) {
    return tr("Enable \"Always Offroad\" in Device panel, or turn vehicle off to select an option.");
  }

  return QString();
}

void HyundaiSettings::showEvent(QShowEvent *event) {
  updateSettings(offroad);
}

void HyundaiSettings::updateSettings(bool _offroad) {
  if (!isVisible()) {
    return;
  }

  auto cp_bytes = params.get("CarParamsPersistent");
  if (!cp_bytes.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

    has_longitudinal_control = hasLongitudinalControl(CP);

    QString longitudinal_tuning_disabled_msg = toggleDisableMsg();
    if (!longitudinal_tuning_disabled_msg.isEmpty()) {
      longitudinalTuningToggle->setEnabled(false);
      longitudinalTuningToggle->setDescription(longitudinal_tuning_disabled_msg);
    } else {
      longitudinalTuningToggle->setEnabled(true);
      longitudinalTuningToggle->setDescription(longitudinalTuningToggle->property("originalDesc").toString());
    }

    longitudinalTuningToggle->showDescription();
  } else {
    has_longitudinal_control = false;
    longitudinalTuningToggle->setEnabled(false);
  }

  offroad = _offroad;
}
