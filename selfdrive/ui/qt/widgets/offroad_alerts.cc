#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonObject>
#include <QJsonDocument>

#include "offroad_alerts.hpp"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/common/util.h"

OffroadAlert::OffroadAlert(QWidget* parent) : QFrame(parent) {
  QVBoxLayout *layout = new QVBoxLayout();
  layout->setMargin(50);

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
    layout->addWidget(l);
  }

  // release notes
  releaseNotes.setVisible(false);
  releaseNotes.setStyleSheet("font-size: 48px;");
  layout->addWidget(&releaseNotes);

  // bottom footer, dismiss + reboot buttons
  QHBoxLayout *footer_layout = new QHBoxLayout();
  layout->addLayout(footer_layout);

  QPushButton *dismiss_btn = new QPushButton("Dismiss");
  dismiss_btn->setFixedSize(400, 125);
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignBottom | Qt::AlignLeft);
  QObject::connect(dismiss_btn, SIGNAL(released()), this, SIGNAL(closeAlerts()));

  rebootBtn.setText("Reboot and Update");
  rebootBtn.setFixedSize(600, 125);
  rebootBtn.setVisible(false);
  footer_layout->addWidget(&rebootBtn, 0, Qt::AlignBottom | Qt::AlignRight);
  QObject::connect(&rebootBtn, &QPushButton::released, [=]() { Hardware::reboot(); });

  setLayout(layout);
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
  updateAlerts();

  rebootBtn.setVisible(updateAvailable);
  releaseNotes.setVisible(updateAvailable);
  releaseNotes.setText(QString::fromStdString(params.get("ReleaseNotes")));

  for (const auto& [k, label] : alerts) {
    label->setVisible(!updateAvailable && !label->text().isEmpty());
  }
}

void OffroadAlert::updateAlerts() {
  alertCount = 0;
  updateAvailable = params.read_db_bool("UpdateAvailable");
  for (const auto& [key, label] : alerts) {
    auto bytes = params.read_db_bytes(key.c_str());
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
