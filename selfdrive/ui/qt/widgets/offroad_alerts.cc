#include <QFile>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonObject>
#include <QJsonDocument>

#include "offroad_alerts.hpp"
#include "selfdrive/hardware/hw.h"

OffroadAlert::OffroadAlert(QWidget* parent) : QFrame(parent) {
  QVBoxLayout *layout = new QVBoxLayout();
  layout->setMargin(50);

  // setup labels for each alert
  QFile inFile("../controls/lib/alerts_offroad.json");
  bool ret = inFile.open(QIODevice::ReadOnly | QIODevice::Text);
  assert(ret);
  QJsonObject obj = QJsonDocument::fromJson(inFile.readAll()).object();
  for (auto &k : obj.keys()) {
    QLabel *l = new QLabel(this);
    alerts[k.toStdString()] = l;
    //int severity = obj[k]["severity"].toInt();
    int severity = 0;

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
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignLeft);
  QObject::connect(dismiss_btn, SIGNAL(released()), this, SIGNAL(closeAlerts()));

  rebootBtn.setText("Reboot and Update");
  rebootBtn.setFixedSize(600, 125);
  rebootBtn.setVisible(false);
  footer_layout->addWidget(&rebootBtn, 0, Qt::AlignRight);
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
  updateAvailable = params.read_db_bool("UpdateAvailable");
  updateAlerts();

  releaseNotes.setVisible(updateAvailable);
  rebootBtn.setVisible(updateAvailable);
  if (updateAvailable) {
    releaseNotes.setText(QString::fromStdString(params.get("ReleaseNotes")));
  }

  for (const auto& [k, label] : alerts) {
    label->setVisible(!updateAvailable);
  }
}

void OffroadAlert::updateAlerts() {
  for (const auto& [key, label] : alerts) {
    auto bytes = params.read_db_bytes(key.c_str());
    label->setText("");
    if (bytes.size()) {
      QJsonDocument doc_par = QJsonDocument::fromJson(QByteArray(bytes.data(), bytes.size()));
      QJsonObject obj = doc_par.object();
      label->setText(obj.value("text").toString());
    }
  }
}
