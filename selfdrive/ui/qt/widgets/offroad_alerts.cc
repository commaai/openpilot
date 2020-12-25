#include <QFile>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
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

  // build alert widgets
  alerts_stack = new QStackedWidget();
  main_layout->addWidget(alerts_stack, 1);

  // bottom footer
  // TODO: add page indicator
  QVBoxLayout *footer_layout = new QVBoxLayout();

  main_layout->addLayout(footer_layout);

  QPushButton *dismiss_btn = new QPushButton("Dismiss");
  dismiss_btn->setFixedSize(453, 125);
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignLeft);
  QObject::connect(dismiss_btn, SIGNAL(released()), this, SIGNAL(closeAlerts()));
  
  refresh();

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

  updateAvailable = false;
  std::vector<char> bytes = Params().read_db_bytes("UpdateAvailable");
  if (bytes.size() && bytes[0] == '1') {
    updateAvailable = true;
  }

  /*
#ifdef __aarch64__
    QObject::connect(update_button, &QPushButton::released, [=]() {std::system("sudo reboot");});
#endif
   
   else {
    vlayout->addSpacing(60);

    for (const auto &alert : alerts) {
      QLabel *l = new QLabel(alert.text);
      l->setWordWrap(true);
      l->setMargin(60);

      QString style = R"(
        font-size: 40px;
        font-weight: bold;
        border-radius: 30px;
        border: 2px solid;
        border-color: white;
      )";
      style.append("background-color: " + QString(alert.severity ? "#971b1c" : "#114267"));

      l->setStyleSheet(style);
      vlayout->addWidget(l);
      vlayout->addSpacing(20);
    }
  }
  */

  if (updateAvailable) {
    QVBoxLayout *update_layout = new QVBoxLayout();

    QLabel *title = new QLabel("Update Available");
    title->setStyleSheet(R"(
      font-size: 72px;
      font-weight: 700;
    )");
    update_layout->addWidget(title, 0, Qt::AlignLeft | Qt::AlignTop);

    QString release_notes = QString::fromStdString(Params().get("ReleaseNotes"));
    QLabel *body = new QLabel(release_notes);
    body->setStyleSheet(R"(
      font-size: 48px;
      font-weight: 600;
    )");
    update_layout->addWidget(body, 1, Qt::AlignLeft | Qt::AlignTop);

    QWidget *w = new QWidget();
    w->setLayout(update_layout);
    alerts_stack->addWidget(w);
  }
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
