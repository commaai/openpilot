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

int OffroadAlert::refresh() {
  static std::map<std::string, QString> alerts_table = {
      {"Offroad_TemperatureTooHigh", tr("Device temperature too high. System won't start.")},
      {"Offroad_ConnectivityNeededPrompt", tr("Immediately connect to the internet to check for updates. If you do not connect to the internet, openpilot won't engage in %1")},
      {"Offroad_ConnectivityNeeded", tr("Connect to internet to check for updates. openpilot won't automatically start until it connects to internet to check for updates.")},
      {"Offroad_UpdateFailed", tr("Unable to download updates\n%1")},
      {"Offroad_InvalidTime", tr("Invalid date and time settings, system won't start. Connect to internet to set time.")},
      {"Offroad_IsTakingSnapshot", tr("Taking camera snapshots. System won't start until finished.")},
      {"Offroad_NeosUpdate", tr("An update to your device's operating system is downloading in the background. You will be prompted to update when it's ready to install.")},
      {"Offroad_UnofficialHardware", tr("Device failed to register. It will not connect to or upload to comma.ai servers, and receives no support from comma.ai. If this is an official device, contact support@comma.ai.")},
      {"Offroad_StorageMissing", tr("NVMe drive not mounted.")},
      {"Offroad_BadNvme", tr("Unsupported NVMe drive detected. Device may draw significantly more power and overheat due to the unsupported NVMe.")},
      {"Offroad_CarUnrecognized", tr("openpilot was unable to identify your car. Your car is either unsupported or its ECUs are not recognized. Please submit a pull request to add the firmware versions to the proper vehicle. Need help? Join discord.comma.ai.")},
      {"Offroad_NoFirmware", tr("openpilot was unable to identify your car. Check integrity of cables and ensure all connections are secure, particularly that the comma power is fully inserted in the OBD-II port of the vehicle. Need help? Join discord.comma.ai.")},
  };

  // build widgets for each offroad alert on first refresh
  if (alerts.empty()) {
    QString json = util::read_file("../controls/lib/alerts_offroad.json").c_str();
    QJsonObject obj = QJsonDocument::fromJson(json.toUtf8()).object();

    // descending sort labels by severity
    std::vector<std::pair<std::string, int>> sorted;
    for (auto it = obj.constBegin(); it != obj.constEnd(); ++it) {
      sorted.push_back({it.key().toStdString(), it.value()["severity"].toInt()});
    }
    std::sort(sorted.begin(), sorted.end(), [=](auto &l, auto &r) { return l.second > r.second; });

    for (auto &[key, severity] : sorted) {
      QLabel *l = new QLabel(this);
      alerts[key] = l;
      l->setMargin(60);
      l->setWordWrap(true);
      l->setStyleSheet(QString("background-color: %1").arg(severity ? "#E22C2C" : "#292929"));
      scrollable_layout->addWidget(l);
    }
    scrollable_layout->addStretch(1);
  }

  int alertCount = 0;
  for (const auto &[key, label] : alerts) {
    QString text;
    std::string bytes = params.get(key);
    if (bytes.size()) {
      auto doc_par = QJsonDocument::fromJson(bytes.c_str());
      QString extra_text = doc_par["text"].toString();
      text = extra_text.isEmpty() ? alerts_table[key] : alerts_table[key].arg(extra_text);
    }
    label->setText(text);
    label->setVisible(!text.isEmpty());
    alertCount += !text.isEmpty();
  }
  snooze_btn->setVisible(!alerts["Offroad_ConnectivityNeeded"]->text().isEmpty());
  return alertCount;
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
    releaseNotes->setText(params.get("ReleaseNotes").c_str());
  }
  return updateAvailable;
}
