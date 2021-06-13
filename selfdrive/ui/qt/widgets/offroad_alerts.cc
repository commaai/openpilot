#include "selfdrive/ui/qt/widgets/offroad_alerts.h"

#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVBoxLayout>

#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

OffroadAlert::OffroadAlert(QWidget* parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setMargin(50);
  main_layout->setSpacing(30);

  QWidget *alerts_widget = new QWidget(this);
  alerts_layout = new QVBoxLayout(alerts_widget);
  alerts_layout->setMargin(0);
  alerts_layout->setSpacing(30);
  alerts_widget->setStyleSheet("background-color: transparent;");

  // release notes
  releaseNotes.setWordWrap(true);
  releaseNotes.setVisible(false);
  releaseNotes.setStyleSheet("font-size: 48px;");
  releaseNotes.setAlignment(Qt::AlignTop);

  releaseNotesScroll = new ScrollView(&releaseNotes, this);
  main_layout->addWidget(releaseNotesScroll);

  alertsScroll = new ScrollView(alerts_widget, this);
  main_layout->addWidget(alertsScroll);

  // bottom footer, dismiss + reboot buttons
  QHBoxLayout *footer_layout = new QHBoxLayout();
  main_layout->addLayout(footer_layout);

  QPushButton *dismiss_btn = new QPushButton("Dismiss");
  dismiss_btn->setFixedSize(400, 125);
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignBottom | Qt::AlignLeft);
  QObject::connect(dismiss_btn, &QPushButton::released, this, &OffroadAlert::closeAlerts);

  rebootBtn.setText("Reboot and Update");
  rebootBtn.setFixedSize(600, 125);
  rebootBtn.setVisible(false);
  footer_layout->addWidget(&rebootBtn, 0, Qt::AlignBottom | Qt::AlignRight);
  QObject::connect(&rebootBtn, &QPushButton::released, [=]() { Hardware::reboot(); });

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

void OffroadAlert::refresh() {
  if (alerts.empty()) {
    // setup labels for each alert
    QString json = QString::fromStdString(util::read_file("../controls/lib/alerts_offroad.json"));
    QJsonObject obj = QJsonDocument::fromJson(json.toUtf8()).object();
    for (auto &k : obj.keys()) {
      QLabel *l = new QLabel(this);
      alerts[k.toStdString()] = l;
      int severity = obj[k].toObject()["severity"].toInt();

      l->setMargin(60);
      l->setWordWrap(true);
      l->setStyleSheet("background-color: " + QString(severity ? "#E22C2C" : "#292929"));
      l->setVisible(false);
      alerts_layout->addWidget(l);
    }
    alerts_layout->addStretch(1);
  }

  updateAlerts();

  rebootBtn.setVisible(updateAvailable);
  releaseNotesScroll->setVisible(updateAvailable);
  releaseNotes.setText(QString::fromStdString(params.get("ReleaseNotes")));

  alertsScroll->setVisible(!updateAvailable);
  for (const auto& [k, label] : alerts) {
    label->setVisible(!label->text().isEmpty());
  }
}

void OffroadAlert::updateAlerts() {
  alertCount = 0;
  updateAvailable = params.getBool("UpdateAvailable");
  for (const auto& [key, label] : alerts) {
    auto bytes = params.get(key.c_str());
    if (bytes.size()) {
      QJsonDocument doc_par = QJsonDocument::fromJson(QByteArray(bytes.data(), bytes.size()));
      QJsonObject obj = doc_par.object();
      label->setText(obj.value("text").toString());
      alertCount++;
    } else {
      label->setText("");
    }
  }
}
