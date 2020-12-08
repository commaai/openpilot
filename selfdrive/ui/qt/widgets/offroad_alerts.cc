#include <QLabel>
#include <QFile>
#include <QPushButton>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDebug>

#include "offroad_alerts.hpp"

#include "common/params.h"


void cleanLayout(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()) {
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      cleanLayout(childLayout);
    }
    delete item;
  }
}

QString vectorToQString(std::vector<char> v) {
  return QString::fromStdString(std::string(v.begin(), v.end()));
}

OffroadAlert::OffroadAlert(QWidget* parent) {
  vlayout = new QVBoxLayout;
  refresh();
  setLayout(vlayout);
}

void OffroadAlert::refresh() {
  cleanLayout(vlayout);
  parse_alerts();

  updateAvailable = false;
  std::vector<char> bytes = Params().read_db_bytes("UpdateAvailable");
  if (bytes.size() && bytes[0] == '1') {
    updateAvailable = true;
  }

  if (updateAvailable) {
    // If there is an update available, don't show alerts
    alerts.clear();

    QFrame *f = new QFrame();

    QVBoxLayout *update_layout = new QVBoxLayout;
    update_layout->setMargin(10);
    update_layout->setSpacing(20);

    QLabel *title = new QLabel("Update available");
    title->setStyleSheet(R"(
      font-size: 55px;
      font-weight: bold;
    )");
    update_layout->addWidget(title, 0, Qt::AlignTop);

    QString release_notes = QString::fromStdString(Params().get("ReleaseNotes"));
    QLabel *notes_label = new QLabel(release_notes);
    notes_label->setStyleSheet(R"(font-size: 40px;)");
    notes_label->setWordWrap(true);
    update_layout->addWidget(notes_label, 1, Qt::AlignTop);

    QPushButton *update_button = new QPushButton("Reboot and Update");
    update_layout->addWidget(update_button);
#ifdef __aarch64__
    QObject::connect(update_button, &QPushButton::released, [=]() {std::system("sudo reboot");});
#endif

    f->setLayout(update_layout);
    f->setStyleSheet(R"(
      .QFrame{
        border-radius: 20px;
        border: 2px solid white;
        background-color: #114267;
      }
      QPushButton {
        padding: 20px;
        font-size: 35px;
        color: white;
        background-color: blue;
      }
    )");

    vlayout->addWidget(f);
    vlayout->addSpacing(60);
  } else {
    vlayout->addSpacing(60);

    for (auto alert : alerts) {
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

  QPushButton *hide_btn = new QPushButton(updateAvailable ? "Later" : "Hide alerts");
  hide_btn->setStyleSheet(R"(
    padding: 20px;
    font-size: 35px;
    color: white;
    background-color: blue;
  )");
  vlayout->addWidget(hide_btn);
  QObject::connect(hide_btn, SIGNAL(released()), this, SIGNAL(closeAlerts()));
}

void OffroadAlert::parse_alerts() {
  alerts.clear();
  // We launch in selfdrive/ui
  QFile inFile("../controls/lib/alerts_offroad.json");
  inFile.open(QIODevice::ReadOnly | QIODevice::Text);
  QByteArray data = inFile.readAll();
  inFile.close();

  QJsonDocument doc = QJsonDocument::fromJson(data);
  if (doc.isNull()) {
    qDebug() << "Parse failed";
  }

  QJsonObject json = doc.object();
  for (const QString& key : json.keys()) {
    std::vector<char> bytes = Params().read_db_bytes(key.toStdString().c_str());

    if (bytes.size()) {
      QJsonDocument doc_par = QJsonDocument::fromJson(QByteArray(bytes.data(), bytes.size()));
      QJsonObject obj = doc_par.object();
      Alert alert = {obj.value("text").toString(), obj.value("severity").toInt()};
      alerts.push_back(alert);
    }
  }
}
