#include "selfdrive/ui/qt/widgets/drive_stats.h"

#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVBoxLayout>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/request_repeater.h"

const double MILE_TO_KM = 1.60934;

static QLayout* build_stat_layout(QLabel** metric, const QString& name) {
  QVBoxLayout* layout = new QVBoxLayout;
  layout->setMargin(0);
  *metric = new QLabel("0");
  (*metric)->setStyleSheet("font-size: 80px; font-weight: 600;");
  layout->addWidget(*metric, 0, Qt::AlignLeft);

  QLabel* label = new QLabel(name);
  label->setStyleSheet("font-size: 45px; font-weight: 500;");
  layout->addWidget(label, 0, Qt::AlignLeft);
  return layout;
}

void DriveStats::parseResponse(const QString& response) {
  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting past drives statistics";
    return;
  }

  auto update = [](const QJsonObject &obj, StatsLabels& labels, bool metric) {
    labels.routes->setText(QString::number((int)obj["routes"].toDouble()));
    labels.distance->setText(QString::number(int(obj["distance"].toDouble() * (metric ? MILE_TO_KM : 1))));
    labels.hours->setText(QString::number((int)(obj["minutes"].toDouble() / 60)));
  };

  QJsonObject json = doc.object();
  update(json["all"].toObject(), all_, metric);
  update(json["week"].toObject(), week_, metric);
}

DriveStats::DriveStats(QWidget* parent) : QWidget(parent) {
  metric = Params().getBool("IsMetric");
  QString distance_unit = metric ? "KM" : "MILES";

  auto add_stats_layouts = [&](QGridLayout* gl, StatsLabels& labels, int row, const QString &distance_unit) {
    gl->addLayout(build_stat_layout(&labels.routes, "DRIVES"), row, 0, 3, 1);
    gl->addLayout(build_stat_layout(&labels.distance, distance_unit), row, 1, 3, 1);
    gl->addLayout(build_stat_layout(&labels.hours, "HOURS"), row, 2, 3, 1);
  };

  QGridLayout* gl = new QGridLayout(this);
  gl->setMargin(0);

  gl->addWidget(new QLabel("ALL TIME"), 0, 0, 1, 3);
  add_stats_layouts(gl, all_, 1, distance_unit);

  gl->addWidget(new QLabel("PAST WEEK"), 6, 0, 1, 3);
  add_stats_layouts(gl, week_, 7, distance_unit);

  QString dongleId = QString::fromStdString(Params().get("DongleId"));
  QString url = "https://api.commadotai.com/v1.1/devices/" + dongleId + "/stats";
  RequestRepeater *repeater = new RequestRepeater(this, url, "ApiCache_DriveStats", 30);
  QObject::connect(repeater, &RequestRepeater::receivedResponse, this, &DriveStats::parseResponse);

  setLayout(gl);
  setStyleSheet(R"(QLabel {font-size: 48px; font-weight: 500;})");
}
