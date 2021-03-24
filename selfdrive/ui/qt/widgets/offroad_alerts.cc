#include <QFile>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDebug>

#include "offroad_alerts.hpp"
#include "common/params.h"
#include "selfdrive/hardware/hw.h"

OffroadAlert::OffroadAlert(QWidget* parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout();

  // alert widget
  alert_widget = new QWidget();
  QVBoxLayout *alert_layout = new QVBoxLayout(alert_widget);
  alert_layout->setSpacing(20);
  main_layout->addWidget(alert_widget, 1);

  // bottom footer
  QHBoxLayout *footer_layout = new QHBoxLayout();
  main_layout->addLayout(footer_layout);

  QPushButton *dismiss_btn = new QPushButton("Dismiss");
  dismiss_btn->setFixedSize(400, 125);
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignLeft);

  reboot_btn = new QPushButton("Reboot and Update");
  reboot_btn->setFixedSize(600, 125);
  reboot_btn->setVisible(false);
  footer_layout->addWidget(reboot_btn, 0, Qt::AlignRight);

  QObject::connect(dismiss_btn, SIGNAL(released()), this, SIGNAL(closeAlerts()));
  QObject::connect(reboot_btn, &QPushButton::released, [=]() { Hardware::reboot(); });

  setLayout(main_layout);
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
  main_layout->setMargin(50);

  QFile inFile("../controls/lib/alerts_offroad.json");
  bool ret = inFile.open(QIODevice::ReadOnly | QIODevice::Text);
  assert(ret);
  QJsonDocument doc = QJsonDocument::fromJson(inFile.readAll());
  assert(!doc.isNull());
  alert_keys = doc.object().keys();
}

void OffroadAlert::refresh() {
  updateAvailable = Params().read_db_bool("UpdateAvailable");
  if (!updateAvailable) parse_alerts();

  labels.resize(updateAvailable ? 1 : alerts.size());
  for (auto &l : labels) { 
    if (!l) l = std::make_unique<QLabel>();
    alert_widget->layout()->addWidget(l.get());
  }

  reboot_btn->setVisible(updateAvailable);
  if (updateAvailable) {
    labels[0]->setStyleSheet(R"(font-size: 48px;)");
    labels[0]->setText(QString::fromStdString(Params().get("ReleaseNotes")));
  } else {
    for (int i = 0; i < alerts.size(); ++i) {
      labels[i]->setText(alerts[i].text);
      labels[i]->setStyleSheet("background-color: " + QString(alerts[i].severity ? "#E22C2C" : "#292929"));
      labels[i]->setMargin(60);
      labels[i]->setWordWrap(true);
    }
  }
}

void OffroadAlert::parse_alerts() {
  alerts.clear();
  Params params;
  for (const QString &key : alert_keys) {
    if (auto bytes = params.read_db_bytes(key.toStdString().c_str()); bytes.size()) {
      QJsonDocument doc_par = QJsonDocument::fromJson(QByteArray(bytes.data(), bytes.size()));
      QJsonObject obj = doc_par.object();
      alerts.push_back({obj.value("text").toString(), obj.value("severity").toInt()});
    }
  }
}
