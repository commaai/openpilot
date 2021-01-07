#include <cassert>
#include <iostream>

#include <QFile>
#include <QDebug>
#include <QVBoxLayout>
#include <QLabel>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>

#include "drive_stats.hpp"
#include "common/params.h"
#include "common/utilpp.h"
#include "api.hpp"

constexpr double MILE_TO_KM = 1.60934;


#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif


QLayout *build_stat(QString name, int stat) {
  QVBoxLayout *layout = new QVBoxLayout;

  QLabel *metric = new QLabel(QString("%1").arg(stat));
  metric->setStyleSheet(R"(
    font-size: 72px;
    font-weight: 700;
  )");
  layout->addWidget(metric, 0, Qt::AlignLeft);
  
  QLabel *label = new QLabel(name);
  label->setStyleSheet(R"(
    font-size: 32px;
    font-weight: 600;
  )");
  layout->addWidget(label, 0, Qt::AlignLeft);
  
  return layout;
}

void DriveStats::replyFinished(QNetworkReply *l) {
  QString answer = l->readAll();
  answer.chop(1);

  QJsonDocument doc = QJsonDocument::fromJson(answer.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed";
  }
  QString IsMetric = QString::fromStdString(Params().get("IsMetric"));
  bool metric = (IsMetric =="1");

  QJsonObject json = doc.object();
  auto all = json["all"].toObject();
  auto week = json["week"].toObject();

  QGridLayout *gl = new QGridLayout();

  int all_distance = all["distance"].toDouble()*(metric ? MILE_TO_KM : 1);
  gl->addWidget(new QLabel("ALL TIME"), 0, 0, 1, 3);
  gl->addLayout(build_stat("DRIVES", all["routes"].toDouble()), 1, 0, 3, 1);
  gl->addLayout(build_stat(metric ? "KM" : "MILES", all_distance), 1, 1, 3, 1);
  gl->addLayout(build_stat("HOURS", all["minutes"].toDouble() / 60), 1, 2, 3, 1);

  int week_distance = week["distance"].toDouble()*(metric ? MILE_TO_KM : 1);
  gl->addWidget(new QLabel("PAST WEEK"), 6, 0, 1, 3);
  gl->addLayout(build_stat("DRIVES", week["routes"].toDouble()), 7, 0, 3, 1);
  gl->addLayout(build_stat(metric ? "KM" : "MILES", week_distance), 7, 1, 3, 1);
  gl->addLayout(build_stat("HOURS", week["minutes"].toDouble() / 60), 7, 2, 3, 1);

  setLayout(gl);
  setStyleSheet(R"(
    QLabel {
      font-size: 48px;
      font-weight: 600;
    }
  )");
}

DriveStats::DriveStats(QWidget *parent) : QWidget(parent) {
  QString dongle_id = QString::fromStdString(Params().get("DongleId"));

  QVector<QPair<QString, QJsonValue>> payloads;
  auto t = QDateTime::currentSecsSinceEpoch();
  payloads.push_back(qMakePair(QString("identity"), dongle_id));
  payloads.push_back(qMakePair(QString("nbf"), t));
  payloads.push_back(qMakePair(QString("iat"), t));
  payloads.push_back(qMakePair(QString("exp"), t + 3600));
  QString token = CommaApi::create_jwt(payloads);

  QNetworkAccessManager *manager = new QNetworkAccessManager(this);
  connect(manager, &QNetworkAccessManager::finished, this, &DriveStats::replyFinished);

  QNetworkRequest request;
  request.setUrl(QUrl("https://api.commadotai.com/v1.1/devices/" + dongle_id + "/stats"));
  request.setRawHeader("Authorization", ("JWT "+token).toUtf8());

  manager->get(request);
}
