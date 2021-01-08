#include <QFile>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDebug>

#include "offroad_alerts.hpp"
#include "common/params.h"


void cleanStackedWidget(QStackedWidget* swidget) {
  while(swidget->count() > 0) {
    QWidget *w = swidget->widget(0);
    swidget->removeWidget(w);
    w->deleteLater();
  }
}

OffroadAlert::OffroadAlert(QWidget* parent) {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setMargin(25);

  alerts_stack = new QStackedWidget();
  main_layout->addWidget(alerts_stack, 1);

  // bottom footer
  QHBoxLayout *footer_layout = new QHBoxLayout();
  main_layout->addLayout(footer_layout);

  QPushButton *dismiss_btn = new QPushButton("Dismiss");
  dismiss_btn->setFixedSize(453, 125);
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignLeft);

  reboot_btn = new QPushButton("Reboot and Update");
  reboot_btn->setFixedSize(453, 125);
  reboot_btn->setVisible(false);
  footer_layout->addWidget(reboot_btn, 0, Qt::AlignRight);

  QObject::connect(dismiss_btn, SIGNAL(released()), this, SIGNAL(closeAlerts()));
#ifdef __aarch64__
  QObject::connect(reboot_btn, &QPushButton::released, [=]() {std::system("sudo reboot");});
#endif

  setLayout(main_layout);
  setStyleSheet(R"(
    * {
      color: white;
    }
    QFrame {
      border-radius: 30px;
      background-color: #393939;
    }
    QPushButton {
      color: black;
      font-size: 40px;
      font-weight: 600;
      border-radius: 20px;
      background-color: white;
    }
  )");
}

void OffroadAlert::refresh() {
  parse_alerts();
  cleanStackedWidget(alerts_stack);

  std::vector<char> bytes = Params().read_db_bytes("UpdateAvailable");
  updateAvailable = bytes.size() && bytes[0] == '1';

  reboot_btn->setVisible(updateAvailable);

  QVBoxLayout *layout = new QVBoxLayout;

  if (updateAvailable) {
    QLabel *title = new QLabel("Update Available");
    title->setStyleSheet(R"(
      font-size: 72px;
      font-weight: 700;
    )");
    layout->addWidget(title, 0, Qt::AlignLeft | Qt::AlignTop);

    QString release_notes = QString::fromStdString(Params().get("ReleaseNotes"));
    QLabel *body = new QLabel(release_notes);
    body->setStyleSheet(R"(
      font-size: 48px;
      font-weight: 600;
    )");
    layout->addWidget(body, 1, Qt::AlignLeft | Qt::AlignTop);
  } else {
    // TODO: paginate the alerts
    for (const auto &alert : alerts) {
      QLabel *l = new QLabel(alert.text);
      l->setWordWrap(true);
      l->setMargin(60);

      QString style = R"(
        font-size: 48px;
        font-weight: 600;
      )";
      style.append("background-color: " + QString(alert.severity ? "#E22C2C" : "#292929"));
      l->setStyleSheet(style);

      layout->addWidget(l, 0, Qt::AlignTop);
    }
    layout->setSpacing(20);
  }

  QWidget *w = new QWidget();
  w->setLayout(layout);
  alerts_stack->addWidget(w);
}

void OffroadAlert::parse_alerts() {
  alerts.clear();

  // TODO: only read this once
  QFile inFile("../controls/lib/alerts_offroad.json");
  inFile.open(QIODevice::ReadOnly | QIODevice::Text);
  QByteArray data = inFile.readAll();
  inFile.close();

  QJsonDocument doc = QJsonDocument::fromJson(data);
  if (doc.isNull()) {
    qDebug() << "Parse failed";
  }

  QJsonObject json = doc.object();
  for (const QString &key : json.keys()) {
    std::vector<char> bytes = Params().read_db_bytes(key.toStdString().c_str());

    if (bytes.size()) {
      QJsonDocument doc_par = QJsonDocument::fromJson(QByteArray(bytes.data(), bytes.size()));
      QJsonObject obj = doc_par.object();
      Alert alert = {obj.value("text").toString(), obj.value("severity").toInt()};
      alerts.push_back(alert);
    }
  }
}
