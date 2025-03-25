/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 *
 * Created by kumar on March 10, 2025
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/lateral/lane_change_settings.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/scrollview.h"
#include <map>
#include <string>
#include <tuple>
#include <vector>

LaneChangeSettings::LaneChangeSettings(QWidget* parent) : QWidget(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 20, 50, 20);
  main_layout->setSpacing(20);

  // Back button
  PanelBackButton* back = new PanelBackButton(tr("Back"));
  connect(back, &QPushButton::clicked, [=]() { emit backPress(); });
  main_layout->addWidget(back, 0, Qt::AlignLeft);

  ListWidgetSP *list = new ListWidgetSP(this, false);
  // param, title, desc, icon
  std::vector<std::tuple<QString, QString, QString, QString>> toggle_defs{
    {
      "AutoLaneChangeBsmDelay",
      tr("Auto Lane Change: Delay with Blind Spot"),
      tr("Toggle to enable a delay timer for seamless lane changes when blind spot monitoring (BSM) detects a obstructing vehicle, ensuring safe maneuvering."),
      "../assets/offroad/icon_blank.png",
    },
  };

  // Controls: Auto Lane Change Timer
  autoLaneChangeTimer = new AutoLaneChangeTimer();
  autoLaneChangeTimer->setUpdateOtherToggles(true);
  autoLaneChangeTimer->showDescription();
  connect(autoLaneChangeTimer, &OptionControlSP::updateLabels, autoLaneChangeTimer, &AutoLaneChangeTimer::refresh);
  connect(autoLaneChangeTimer, &AutoLaneChangeTimer::updateOtherToggles, this, &LaneChangeSettings::updateToggles);
  list->addItem(autoLaneChangeTimer);

  for (auto &[param, title, desc, icon] : toggle_defs) {
    auto toggle = new ParamControlSP(param, title, desc, icon, this);
    list->addItem(toggle);
    toggles[param.toStdString()] = toggle;
  }

  main_layout->addWidget(new ScrollViewSP(list, this));
}

void LaneChangeSettings::showEvent(QShowEvent *event) {
  updateToggles();
}

void LaneChangeSettings::updateToggles() {
  if (!isVisible()) {
    return;
  }

  auto auto_lane_change_bsm_delay_toggle = toggles["AutoLaneChangeBsmDelay"];
  auto autoLaneChangeTimer_param = std::atoi(params.get("AutoLaneChangeTimer").c_str());

  auto cp_bytes = params.get("CarParamsPersistent");
  if (!cp_bytes.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

    if (!CP.getEnableBsm()) {
      params.remove("AutoLaneChangeBsmDelay");
    }
    auto_lane_change_bsm_delay_toggle->setEnabled(CP.getEnableBsm() && (autoLaneChangeTimer_param > 0));
    auto_lane_change_bsm_delay_toggle->refresh();
  } else {
    auto_lane_change_bsm_delay_toggle->setEnabled(false);
  }
}

// Auto Lane Change Timer (ALCT)
AutoLaneChangeTimer::AutoLaneChangeTimer() : OptionControlSP(
  "AutoLaneChangeTimer",
  tr("Auto Lane Change by Blinker"),
  tr("Set a timer to delay the auto lane change operation when the blinker is used. "
     "No nudge on the steering wheel is required to auto lane change if a timer is set. Default is Nudge.\n"
     "Please use caution when using this feature. Only use the blinker when traffic and road conditions permit."),
  "../assets/offroad/icon_blank.png",
  {-1, 5}) {

  refresh();
}

void AutoLaneChangeTimer::refresh() {
  QString option = QString::fromStdString(params.get("AutoLaneChangeTimer"));
  const QString second = tr("s");

  static const QMap<QString, QString> options = {
    {"-1", tr("Off")},
    {"0",  tr("Nudge")},
    {"1",  tr("Nudgeless")},
    {"2",  "0.5 " + second},
    {"3",  "1 "   + second},
    {"4",  "2 "   + second},
    {"5",  "3 "   + second},
  };

  setLabel(options.value(option, tr("Nudge")));
}
