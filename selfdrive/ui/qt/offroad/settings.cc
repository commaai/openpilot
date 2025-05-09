#include <cassert>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <QDebug>

#include "common/watchdog.h"
#include "common/util.h"
#include "common/params.h"
#include "selfdrive/ui/qt/network/networking.h"
#include "selfdrive/ui/qt/offroad/settings.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/prime.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"
#include "selfdrive/ui/qt/offroad/developer_panel.h"
#include "selfdrive/ui/qt/offroad/firehose.h"

static Params params;

TogglesPanel::TogglesPanel(SettingsWindow *parent) : ListWidget(parent) {
  // param, title, desc, icon
  std::vector<std::tuple<QString, QString, QString, QString>> toggle_defs{
    {"OpenpilotEnabledToggle", tr("Enable openpilot"),
      tr("Use the openpilot system for adaptive cruise control and lane keep driver assistance. Your attention is required at all times to use this feature. Changing this setting takes effect when the car is powered off."),
      "../assets/img_chffr_wheel.png"},
    {"ExperimentalMode", tr("Experimental Mode"), "", "../assets/img_experimental_white.svg"},
    {"DisengageOnAccelerator", tr("Disengage on Accelerator Pedal"),
      tr("When enabled, pressing the accelerator pedal will disengage openpilot."),
      "../assets/offroad/icon_disengage_on_accelerator.svg"},
    {"IsLdwEnabled", tr("Enable Lane Departure Warnings"),
      tr("Receive alerts to steer back into the lane when your vehicle drifts over a detected lane line without a turn signal activated while driving over 31 mph (50 km/h)."),
      "../assets/offroad/icon_warning.png"},
    {"AlwaysOnDM", tr("Always-On Driver Monitoring"),
      tr("Enable driver monitoring even when openpilot is not engaged."),
      "../assets/offroad/icon_monitoring.png"},
    {"RecordFront", tr("Record and Upload Driver Camera"),
      tr("Upload data from the driver facing camera and help improve the driver monitoring algorithm."),
      "../assets/offroad/icon_monitoring.png"},
    {"IsMetric", tr("Use Metric System"),
      tr("Display speed in km/h instead of mph."),
      "../assets/offroad/icon_metric.png"},
    {"LateralOnlyMode", tr("Lateral Only"),
      tr("Enable steering assistance only. Speed control is manual via the stock cruise SET button; lateral pauses on brake and resumes on release."),
      "../assets/offroad/icon_lateral_only.png"}
  };

  std::vector<QString> longi_button_texts{tr("Aggressive"), tr("Standard"), tr("Relaxed")};
  long_personality_setting = new ButtonParamControl(
    "LongitudinalPersonality", tr("Driving Personality"),
    tr("Standard is recommended. In aggressive mode, openpilot will follow lead cars closer and be more aggressive with the gas and brake. "
       "In relaxed mode openpilot will stay further away from lead cars. On supported cars, you can cycle through these personalities with "
       "your steering wheel distance button."),
    "../assets/offroad/icon_speed_limit.png",
    longi_button_texts
  );

  QObject::connect(uiState(), &UIState::uiUpdate, this, &TogglesPanel::updateState);

  for (auto &[param, title, desc, icon] : toggle_defs) {
    auto toggle = new ParamControl(param, title, desc, icon, this);
    bool locked = params.getBool((param + "Lock").toStdString());
    toggle->setEnabled(!locked);
    addItem(toggle);
    toggles[param.toStdString()] = toggle;
    if (param == "DisengageOnAccelerator") {
      addItem(long_personality_setting);
    }
  }

  toggles["ExperimentalMode"]->setActiveIcon("../assets/img_experimental.svg");
  toggles["ExperimentalMode"]->setConfirmation(true, true);
}

void TogglesPanel::updateState(const UIState &s) {
  const SubMaster &sm = *(s.sm);
  if (sm.updated("selfdriveState")) {
    auto personality = sm["selfdriveState"].getSelfdriveState().getPersonality();
    if (personality != s.scene.personality && s.scene.started && isVisible()) {
      long_personality_setting->setCheckedButton(static_cast<int>(personality));
    }
    uiState()->scene.personality = personality;
  }
}

void TogglesPanel::expandToggleDescription(const QString &param) {
  toggles[param.toStdString()]->showDescription();
}

void TogglesPanel::showEvent(QShowEvent *event) {
  updateToggles();
}

void TogglesPanel::updateToggles() {
  auto experimental_mode_toggle = toggles["ExperimentalMode"];
  const QString e2e_description = QString("%1<br>"
                                          "<h4>%2</h4><br>"
                                          "%3<br>"
                                          "<h4>%4</h4><br>"
                                          "%5<br>")
                                  .arg(tr("openpilot defaults to driving in <b>chill mode</b>. Experimental mode enables <b>alpha-level features</b> that aren't ready for chill mode. Experimental features are listed below:"))
                                  .arg(tr("End-to-End Longitudinal Control"))
                                  .arg(tr("Let the driving model control the gas and brakes. openpilot will drive as it thinks a human would, including stopping for red lights and stop signs. Mistakes should be expected."))
                                  .arg(tr("New Driving Visualization"))
                                  .arg(tr("The driving visualization will transition to the road-facing wide-angle camera at low speeds to better show some turns. The Experimental mode logo will also be shown in the top right corner."));

  const bool is_release = params.getBool("IsReleaseBranch");
  auto cp_bytes = params.get("CarParamsPersistent");
  if (!cp_bytes.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(cp_bytes.data(), cp_bytes.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();
    if (hasLongitudinalControl(CP)) {
      experimental_mode_toggle->setEnabled(true);
      experimental_mode_toggle->setDescription(e2e_description);
      long_personality_setting->setEnabled(true);
    } else {
      experimental_mode_toggle->setEnabled(false);
      long_personality_setting->setEnabled(false);
      params.remove("ExperimentalMode");
      const QString unavailable = tr("Experimental mode is currently unavailable on this car since the car's stock ACC is used for longitudinal control.");
      QString long_desc = unavailable + " " + tr("openpilot longitudinal control may come in a future update.");
      if (CP.getAlphaLongitudinalAvailable()) {
        if (is_release) {
          long_desc = unavailable + " " + tr("An alpha version of openpilot longitudinal control can be tested, along with Experimental mode, on non-release branches.");
        } else {
          long_desc = tr("Enable the openpilot longitudinal control (alpha) toggle to allow Experimental mode.");
        }
      }
      experimental_mode_toggle->setDescription("<b>" + long_desc + "</b><br><br>" + e2e_description);
    }
    experimental_mode_toggle->refresh();
  } else {
    experimental_mode_toggle->setDescription(e2e_description);
  }
}

DevicePanel::DevicePanel(SettingsWindow *parent) : ListWidget(parent) {
  setSpacing(50);
  addItem(new LabelControl(tr("Dongle ID"), getDongleId().value_or(tr("N/A"))));
  addItem(new LabelControl(tr("Serial"), params.get("HardwareSerial").c_str()));

  pair_device = new ButtonControl(tr("Pair Device"), tr("PAIR"),
                                  tr("Pair your device with comma connect (connect.comma.ai) and claim your comma prime offer."));
  connect(pair_device, &ButtonControl::clicked, [=]() { PairingPopup popup(this); popup.exec(); });
  addItem(pair_device);

  auto dcamBtn = new ButtonControl(tr("Driver Camera"), tr("PREVIEW"),
                                   tr("Preview the driver facing camera to ensure that driver monitoring has good visibility. (vehicle must be off)"));
  connect(dcamBtn, &ButtonControl::clicked, [=]() { emit showDriverView(); });
  addItem(dcamBtn);

  auto resetCalibBtn = new ButtonControl(tr("Reset Calibration"), tr("RESET"), "");
  connect(resetCalibBtn, &ButtonControl::showDescriptionEvent, this, &DevicePanel::updateCalibDescription);
  connect(resetCalibBtn, &ButtonControl::clicked, [&]() {
    if (ConfirmationDialog::confirm(tr("Are you sure you want to reset calibration?"), tr("Reset"), this)) {
      params.remove("CalibrationParams"); params.remove("LiveTorqueParameters");
    }
  });
  addItem(resetCalibBtn);

  auto retrainingBtn = new ButtonControl(tr("Review Training Guide"), tr("REVIEW"),
                                        tr("Review the rules, features, and limitations of openpilot"));
  connect(retrainingBtn, &ButtonControl::clicked, [=]() {
    if (ConfirmationDialog::confirm(tr("Are you sure you want to review the training guide?"), tr("Review"), this)) {
      emit reviewTrainingGuide();
    }
  });
  addItem(retrainingBtn);

  if (Hardware::TICI()) {
    auto regulatoryBtn = new ButtonControl(tr("Regulatory"), tr("VIEW"), "");
    connect(regulatoryBtn, &ButtonControl::clicked, [=]() {
      const std::string txt = util::read_file("../assets/offroad/fcc.html");
      ConfirmationDialog::rich(QString::fromStdString(txt), this);
    });
    addItem(regulatoryBtn);
  }

  auto translateBtn = new ButtonControl(tr("Change Language"), tr("CHANGE"), "");
  connect(translateBtn, &ButtonControl::clicked, [=]() {
    QMap<QString, QString> langs = getSupportedLanguages();
    QString selection = MultiOptionDialog::getSelection(tr("Select a language"), langs.keys(), langs.key(uiState()->language), this);
    if (!selection.isEmpty()) {
      params.put("LanguageSetting", langs[selection].toStdString());
      qApp->exit(18); watchdog_kick(0);
    }
  });
  addItem(translateBtn);

  QObject::connect(uiState()->prime_state, &PrimeState::changed, [this](PrimeState::Type type) {
    pair_device->setVisible(type == PrimeState::PRIME_TYPE_UNPAIRED);
  });
  QObject::connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    for (auto btn : findChildren<ButtonControl *>()) if (btn != pair_device) btn->setEnabled(offroad);
  });

  // power buttons
  QHBoxLayout *power_layout = new QHBoxLayout(); power_layout->setSpacing(30);

  QPushButton *reboot_btn = new QPushButton(tr("Reboot")); reboot_btn->setObjectName("reboot_btn");
  power_layout->addWidget(reboot_btn); QObject::connect(reboot_btn, &QPushButton::clicked, this, &DevicePanel::reboot);

  QPushButton *poweroff_btn = new QPushButton(tr("Power Off")); poweroff_btn->setObjectName("poweroff_btn");
  power_layout->addWidget(poweroff_btn); QObject::connect(poweroff_btn, &QPushButton::clicked, this, &DevicePanel::poweroff);

  if (!Hardware::PC()) connect(uiState(), &UIState::offroadTransition, poweroff_btn, &
