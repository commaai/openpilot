#include "selfdrive/ui/qt/widgets/offroad_alerts.h"

#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>

#include "common/util.h"
#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

AbstractAlert::AbstractAlert(bool hasRebootBtn, QWidget *parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setMargin(50);
  main_layout->setSpacing(30);

  QWidget *widget = new QWidget;
  scrollable_layout = new QVBoxLayout(widget);
  widget->setStyleSheet("background-color: transparent;");
  main_layout->addWidget(new ScrollView(widget));

  // bottom footer, dismiss + reboot buttons
  QHBoxLayout *footer_layout = new QHBoxLayout();
  main_layout->addLayout(footer_layout);

  QPushButton *dismiss_btn = new QPushButton(tr("Close"));
  dismiss_btn->setFixedSize(400, 125);
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignBottom | Qt::AlignLeft);
  QObject::connect(dismiss_btn, &QPushButton::clicked, this, &AbstractAlert::dismiss);

  snooze_btn = new QPushButton(tr("Snooze Update"));
  snooze_btn->setVisible(false);
  snooze_btn->setFixedSize(550, 125);
  footer_layout->addWidget(snooze_btn, 0, Qt::AlignBottom | Qt::AlignRight);
  QObject::connect(snooze_btn, &QPushButton::clicked, [=]() {
    params.putBool("SnoozeUpdate", true);
  });
  QObject::connect(snooze_btn, &QPushButton::clicked, this, &AbstractAlert::dismiss);
  snooze_btn->setStyleSheet(R"(color: white; background-color: #4F4F4F;)");

  if (hasRebootBtn) {
    QPushButton *rebootBtn = new QPushButton(tr("Reboot and Update"));
    rebootBtn->setFixedSize(600, 125);
    footer_layout->addWidget(rebootBtn, 0, Qt::AlignBottom | Qt::AlignRight);
    QObject::connect(rebootBtn, &QPushButton::clicked, [=]() { Hardware::reboot(); });
  }

  setStyleSheet(R"(
    * {
      font-size: 48px;
      color: white;
    }
    QFrame {
      border-radius: 30px;
      background-color: #393939;
    }
    QPushButton {
      color: black;
      font-weight: 500;
      border-radius: 30px;
      background-color: white;
    }
  )");
}

OffroadAlert::OffroadAlert(QWidget *parent) : AbstractAlert(parent) {
  for (auto &[key, severity, text] : allAlerts()) {
    QLabel *l = new QLabel(this);
    l->setMargin(60);
    l->setWordWrap(true);
    l->setStyleSheet(QString("background-color: %1").arg(severity == Severity::high ? "#E22C2C" : "#292929"));
    l->setVisible(false);
    scrollable_layout->addWidget(l);

    alerts[key] = {.text = text, .label = l};
  }
  scrollable_layout->addStretch(1);
}

int OffroadAlert::refresh() {
  Params params;
  int alertCount = 0;

  for (const auto &[key, a] : alerts) {
    if (params.exists(key)) {
      std::string extra = params.get(key);
      a.label->setText(extra.empty() ? a.text : a.text.arg(extra.c_str()));
      a.label->setVisible(true);
      ++alertCount;
    } else {
      a.label->setVisible(false);
    }
  }

  snooze_btn->setVisible(alerts["Offroad_ConnectivityNeeded"].label->isVisible());
  return alertCount;
}

const std::vector<std::tuple<std::string, Severity, QString>> OffroadAlert::allAlerts() {
  return {
    {"Offroad_TemperatureTooHigh", Severity::high, tr("Device temperature too high. System won't start.")},
    {"Offroad_ConnectivityNeeded", Severity::high, tr("Connect to internet to check for updates. openpilot won't automatically start until it connects to internet to check for updates.")},
    {"Offroad_UpdateFailed", Severity::high, tr("Unable to download updates\n%1")},
    {"Offroad_InvalidTime", Severity::high, tr("Invalid date and time settings, system won't start. Connect to internet to set time.")},
    {"Offroad_UnofficialHardware", Severity::high, tr("Device failed to register. It will not connect to or upload to comma.ai servers, and receives no support from comma.ai. If this is an official device, contact support@comma.ai.")},
    {"Offroad_StorageMissing", Severity::high, tr("NVMe drive not mounted.")},
    {"Offroad_BadNvme", Severity::high, tr("Unsupported NVMe drive detected. Device may draw significantly more power and overheat due to the unsupported NVMe.")},
    {"Offroad_CarUnrecognized", Severity::normal, tr("openpilot was unable to identify your car. Your car is either unsupported or its ECUs are not recognized. Please submit a pull request to add the firmware versions to the proper vehicle. Need help? Join discord.comma.ai.")},
    {"Offroad_ConnectivityNeededPrompt", Severity::normal, tr("Immediately connect to the internet to check for updates. If you do not connect to the internet, openpilot won't engage in %1")},
    {"Offroad_NoFirmware", Severity::normal, tr("openpilot was unable to identify your car. Check integrity of cables and ensure all connections are secure, particularly that the comma power is fully inserted in the OBD-II port of the vehicle. Need help? Join discord.comma.ai.")},
    {"Offroad_IsTakingSnapshot", Severity::normal, tr("Taking camera snapshots. System won't start until finished.")},
    {"Offroad_NeosUpdate", Severity::normal, tr("An update to your device's operating system is downloading in the background. You will be prompted to update when it's ready to install.")},
  };
}

UpdateAlert::UpdateAlert(QWidget *parent) : AbstractAlert(true, parent) {
  releaseNotes = new QLabel(this);
  releaseNotes->setWordWrap(true);
  releaseNotes->setAlignment(Qt::AlignTop);
  scrollable_layout->addWidget(releaseNotes);
}

bool UpdateAlert::refresh() {
  bool updateAvailable = params.getBool("UpdateAvailable");
  if (updateAvailable) {
    releaseNotes->setText(params.get("UpdaterNewReleaseNotes").c_str());
  }
  return updateAvailable;
}
