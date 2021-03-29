#include <QFile>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDebug>

#include "offroad_alerts.hpp"
#include "common/params.h"
#include "selfdrive/hardware/hw.h"

void cleanStackedWidget(QStackedWidget* swidget) {
  while(swidget->count() > 0) {
    QWidget *w = swidget->widget(0);
    swidget->removeWidget(w);
    w->deleteLater();
  }
}

OffroadAlert::OffroadAlert(QWidget* parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(25);

  alerts_stack = new QStackedWidget();
  main_layout->addWidget(alerts_stack, 1);

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
  parse_alerts();
  cleanStackedWidget(alerts_stack);

  updateAvailable = Params().read_db_bool("UpdateAvailable");
  reboot_btn->setVisible(updateAvailable);

  QVBoxLayout *layout = new QVBoxLayout;
  layout->setSpacing(20);

  if (updateAvailable) {
    QLabel *body = new QLabel(QString::fromStdString(Params().get("ReleaseNotes")));
    body->setStyleSheet(R"(font-size: 48px;)");
    layout->addWidget(body, 0, Qt::AlignLeft | Qt::AlignTop);
  } else {
    for (const auto &alert : alerts) {
      QLabel *l = new QLabel(alert.text);
      l->setMargin(60);
      l->setWordWrap(true);
      l->setStyleSheet("background-color: " + QString(alert.severity ? "#E22C2C" : "#292929"));
      layout->addWidget(l, 0, Qt::AlignTop);
    }
  }

  QWidget *w = new QWidget();
  w->setLayout(layout);
  alerts_stack->addWidget(w);
}

void OffroadAlert::parse_alerts() {
  alerts.clear();
  for (const QString &key : alert_keys) {
    std::vector<char> bytes = Params().read_db_bytes(key.toStdString().c_str());
    if (bytes.size()) {
      QJsonDocument doc_par = QJsonDocument::fromJson(QByteArray(bytes.data(), bytes.size()));
      QJsonObject obj = doc_par.object();
      Alert alert = {obj.value("text").toString(), obj.value("severity").toInt()};
      alerts.push_back(alert);
    }
  }
}
