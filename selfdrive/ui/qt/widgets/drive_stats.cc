#include <cassert>
#include <iostream>

#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QNetworkRequest>
#include <QStackedLayout>
#include <QTimer>
#include <QVBoxLayout>

#include "api.hpp"
#include "common/params.h"
#include "common/util.h"
#include "drive_stats.hpp"
#include "home.hpp"

const double MILE_TO_KM = 1.60934;

#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif

// TODO: function is also in offroad/wifi.cc. Put in library file
static void clearLayouts(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()) {
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      clearLayouts(childLayout);
    }
    delete item;
  }
}

static QLayout* build_stat(QString name, int stat, QWidget * parent=nullptr) {
  QVBoxLayout* layout = new QVBoxLayout(parent);

  QLabel* metric = new QLabel(QString("%1").arg(stat), parent);
  metric->setStyleSheet(R"(
    font-size: 72px;
    font-weight: 700;
  )");
  layout->addWidget(metric, 0, Qt::AlignLeft);

  QLabel* label = new QLabel(name, parent);
  label->setStyleSheet(R"(
    font-size: 32px;
    font-weight: 600;
  )");
  layout->addWidget(label, 0, Qt::AlignLeft);

  return layout;
}

void DriveStats::parseError(QString response) {
  clearLayouts(vlayout);
  vlayout->addWidget(new QLabel("No internet connection", this));
}

void DriveStats::parseResponse(QString response) {
  response.chop(1);
  clearLayouts(vlayout);
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting past drives statistics";
    return;
  }

  QString IsMetric = QString::fromStdString(Params().get("IsMetric"));
  bool metric = (IsMetric == "1");

  QJsonObject json = doc.object();
  auto all = json["all"].toObject();
  auto week = json["week"].toObject();

  QGridLayout* gl = new QGridLayout(this);

  int all_distance = all["distance"].toDouble() * (metric ? MILE_TO_KM : 1);
  gl->addWidget(new QLabel("ALL TIME", this), 0, 0, 1, 3);
  gl->addLayout(build_stat("DRIVES", all["routes"].toDouble(), this), 1, 0, 3, 1);
  gl->addLayout(build_stat(metric ? "KM" : "MILES", all_distance, this), 1, 1, 3, 1);
  gl->addLayout(build_stat("HOURS", all["minutes"].toDouble() / 60, this), 1, 2, 3, 1);

  int week_distance = week["distance"].toDouble() * (metric ? MILE_TO_KM : 1);
  gl->addWidget(new QLabel("PAST WEEK", this), 6, 0, 1, 3);
  gl->addLayout(build_stat("DRIVES", week["routes"].toDouble(), this), 7, 0, 3, 1);
  gl->addLayout(build_stat(metric ? "KM" : "MILES", week_distance, this), 7, 1, 3, 1);
  gl->addLayout(build_stat("HOURS", week["minutes"].toDouble() / 60, this), 7, 2, 3, 1);

  QWidget* q = new QWidget(this);
  q->setLayout(gl);

  vlayout->addWidget(q);
}

DriveStats::DriveStats(QWidget* parent) : QWidget(parent) {
  vlayout = new QVBoxLayout(this);
  setLayout(vlayout);
  setStyleSheet(R"(
    QLabel {
      font-size: 48px;
      font-weight: 600;
    }
  )");

  QString dongleId = QString::fromStdString(Params().get("DongleId"));
  QString url = "https://api.commadotai.com/v1.1/devices/" + dongleId + "/stats";
  RequestRepeater* repeater = new RequestRepeater(this, url, 13);
  QObject::connect(repeater, SIGNAL(receivedResponse(QString)), this, SLOT(parseResponse(QString)));
  QObject::connect(repeater, SIGNAL(failedResponse(QString)), this, SLOT(parseError(QString)));

}
