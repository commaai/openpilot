#include "selfdrive/ui/qt/widgets/drive_stats.h"

#include <QDebug>
#include <QGridLayout>
#include <QJsonObject>
#include <QVBoxLayout>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/request_repeater.h"

const double MILE_TO_KM = 1.60934;

namespace {

QLabel* numberLabel() {
  QLabel* label = new QLabel("0");
  label->setStyleSheet("font-size: 80px; font-weight: 600;");
  return label;
}

QLabel* unitLabel(const QString& name) {
  QLabel* label = new QLabel(name);
  label->setStyleSheet("font-size: 45px; font-weight: 500;");
  return label;
}

}  // namespace

DriveStats::DriveStats(QWidget* parent) : QWidget(parent) {
  metric_ = Params().getBool("IsMetric");

  QGridLayout* main_layout = new QGridLayout(this);
  main_layout->setMargin(0);

  auto add_stats_layouts = [=](const QString &title, StatsLabels& labels) {
    int row = main_layout->rowCount();
    main_layout->addWidget(new QLabel(title), row++, 0, 1, 3);

    main_layout->addWidget(labels.routes = numberLabel(), row, 0, Qt::AlignLeft);
    main_layout->addWidget(labels.distance = numberLabel(), row, 1, Qt::AlignLeft);
    main_layout->addWidget(labels.hours = numberLabel(), row, 2, Qt::AlignLeft);

    main_layout->addWidget(unitLabel("DRIVES"), row + 1, 0, Qt::AlignLeft);
    main_layout->addWidget(labels.distance_unit = unitLabel(getDistanceUnit()), row + 1, 1, Qt::AlignLeft);
    main_layout->addWidget(unitLabel("HOURS"), row + 1, 2, Qt::AlignLeft);
  };

  add_stats_layouts("ALL TIME", all_);
  add_stats_layouts("PAST WEEK", week_);

  std::string dongle_id = Params().get("DongleId");
  if (util::is_valid_dongle_id(dongle_id)) {
    std::string url = "https://api.commadotai.com/v1.1/devices/" + dongle_id + "/stats";
    RequestRepeater* repeater = new RequestRepeater(this, QString::fromStdString(url), "ApiCache_DriveStats", 30);
    QObject::connect(repeater, &RequestRepeater::receivedResponse, this, &DriveStats::parseResponse);
  }

  setStyleSheet(R"(QLabel {font-size: 48px; font-weight: 500;})");
}

void DriveStats::updateStats() {
  auto update = [=](const QJsonObject& obj, StatsLabels& labels) {
    labels.routes->setText(QString::number((int)obj["routes"].toDouble()));
    labels.distance->setText(QString::number(int(obj["distance"].toDouble() * (metric_ ? MILE_TO_KM : 1))));
    labels.distance_unit->setText(getDistanceUnit());
    labels.hours->setText(QString::number((int)(obj["minutes"].toDouble() / 60)));
  };

  QJsonObject json = stats_.object();
  update(json["all"].toObject(), all_);
  update(json["week"].toObject(), week_);
}

void DriveStats::parseResponse(const QString& response) {
  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting past drives statistics";
    return;
  }
  stats_ = doc;
  updateStats();
}

void DriveStats::showEvent(QShowEvent* event) {
  bool metric = Params().getBool("IsMetric");
  if (metric_ != metric) {
    metric_ = metric;
    updateStats();
  }
}
