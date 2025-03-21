/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/device_panel.h"

#include "common/watchdog.h"
#include "selfdrive/ui/qt/qt_window.h"

DevicePanelSP::DevicePanelSP(SettingsWindowSP *parent) : DevicePanel(parent) {
  QGridLayout *device_grid_layout = new QGridLayout();
  device_grid_layout->setSpacing(30);
  device_grid_layout->setHorizontalSpacing(5);
  device_grid_layout->setVerticalSpacing(25);

  std::vector<std::pair<QString, QString>> device_btns = {
    {"dcamBtn", tr("Driver Camera Preview")},
    {"retrainingBtn", tr("Training Guide")},
    {"regulatoryBtn", tr("Regulatory")},
    {"translateBtn", tr("Language")},
    {"resetParams", tr("Reset Settings")},
  };

  int row = 0, col = 0;
  for (int i = 0; i < device_btns.size(); i++) {
    if (device_btns[i].first == "regulatoryBtn" && !Hardware::TICI()) {
      continue;
    }

    auto *btn = new PushButtonSP(device_btns[i].second, 720, this);
    device_grid_layout->addWidget(btn, row, col);
    buttons[device_btns[i].first] = btn;

    col++;
    if (col > 1) {
      col = 0;
      row++;
    }
  }

  connect(buttons["dcamBtn"], &PushButtonSP::clicked, [=]() { emit showDriverView(); });

  connect(buttons["retrainingBtn"], &PushButtonSP::clicked, [=]() {
    if (ConfirmationDialog::confirm(tr("Are you sure you want to review the training guide?"), tr("Review"), this)) {
      emit reviewTrainingGuide();
    }
  });

  if (Hardware::TICI()) {
    connect(buttons["regulatoryBtn"], &PushButtonSP::clicked, [=]() {
      const std::string txt = util::read_file("../assets/offroad/fcc.html");
      ConfirmationDialog::rich(QString::fromStdString(txt), this);
    });
  }

  connect(buttons["translateBtn"], &PushButtonSP::clicked, [=]() {
    QMap<QString, QString> langs = getSupportedLanguages();
    QString selection = MultiOptionDialog::getSelection(tr("Select a language"), langs.keys(), langs.key(uiState()->language), this);
    if (!selection.isEmpty()) {
      // put language setting, exit Qt UI, and trigger fast restart
      params.put("LanguageSetting", langs[selection].toStdString());
      qApp->exit(18);
      watchdog_kick(0);
    }
  });

  connect(buttons["resetParams"], &PushButtonSP::clicked, [=]() {
    if (ConfirmationDialog::confirm(tr("Are you sure you want to reset all sunnypilot settings to default ? This cannot be undone."), tr("Reset"), this)) {
      int rm = std::system("sudo rm -rf /data/params/d/*");
      if (rm == 0) {
        Hardware::reboot();
      }
    }
  });

  addItem(device_grid_layout);

  // offroad mode and power buttons

  QHBoxLayout *power_layout = new QHBoxLayout();
  power_layout->setSpacing(5);

  PushButtonSP *rebootBtn = new PushButtonSP(tr("Reboot"), 720, this);
  rebootBtn->setStyleSheet(rebootButtonStyle);
  power_layout->addWidget(rebootBtn);
  QObject::connect(rebootBtn, &PushButtonSP::clicked, this, &DevicePanelSP::reboot);

  PushButtonSP *poweroffBtn = new PushButtonSP(tr("Power Off"), 720, this);
  poweroffBtn->setStyleSheet(powerOffButtonStyle);
  power_layout->addWidget(poweroffBtn);
  QObject::connect(poweroffBtn, &PushButtonSP::clicked, this, &DevicePanelSP::poweroff);

  if (!Hardware::PC()) {
    connect(uiState(), &UIState::offroadTransition, poweroffBtn, &PushButtonSP::setVisible);
  }

  offroadBtn = new PushButtonSP(tr("Offroad Mode"));
  offroadBtn->setFixedWidth(power_layout->sizeHint().width());
  QObject::connect(offroadBtn, &PushButtonSP::clicked, this, &DevicePanelSP::setOffroadMode);

  QVBoxLayout *power_group_layout = new QVBoxLayout();
  power_group_layout->setSpacing(30);
  power_group_layout->addWidget(offroadBtn, 0, Qt::AlignHCenter);
  power_group_layout->addLayout(power_layout);

  addItem(power_group_layout);

  QObject::connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    for (auto btn : findChildren<PushButtonSP*>()) {
      if (btn != rebootBtn && btn != poweroffBtn && btn != offroadBtn) {
        btn->setEnabled(offroad);
      }
    }
  });
}

void DevicePanelSP::setOffroadMode() {
  if (!uiState()->engaged()) {
    if (params.getBool("OffroadMode")) {
      if (ConfirmationDialog::confirm(tr("Are you sure you want to exit Always Offroad mode?"), tr("Confirm"), this)) {
        // Check engaged again in case it changed while the dialog was open
        if (!uiState()->engaged()) {
          params.remove("OffroadMode");
        }
      }
    } else {
      if (ConfirmationDialog::confirm(tr("Are you sure you want to enter Always Offroad mode?"), tr("Confirm"), this)) {
        // Check engaged again in case it changed while the dialog was open
        if (!uiState()->engaged()) {
          params.putBool("OffroadMode", true);
        }
      }
    }
  } else {
    ConfirmationDialog::alert(tr("Disengage to Enter Always Offroad Mode"), this);
  }

  updateState();
}

void DevicePanelSP::showEvent(QShowEvent *event) {
  updateState();
}

void DevicePanelSP::updateState() {
  if (!isVisible()) {
    return;
  }

  bool offroad_mode_param = params.getBool("OffroadMode");
  offroadBtn->setText(offroad_mode_param ? tr("Exit Always Offroad") : tr("Always Offroad"));
  offroadBtn->setStyleSheet(offroad_mode_param ? alwaysOffroadStyle : autoOffroadStyle);
}
