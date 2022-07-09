#include "selfdrive/ui/qt/widgets/drive_stats.h"

#include <QDebug>
#include <QGridLayout>
#include <QJsonObject>
#include <QVBoxLayout>

#include "common/params.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/util.h"

static QLabel* newLabel(const QString& text, const QString &type) {
  QLabel* label = new QLabel(text);
  label->setProperty("type", type);
  return label;
}

DriveStats::DriveStats(QWidget* parent) : QFrame(parent) {
  metric_ = Params().getBool("IsMetric");

  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 50, 50, 60);

  auto add_stats_layouts = [=](StatsLabels& labels) {
    QGridLayout* grid_layout = new QGridLayout;
    grid_layout->setVerticalSpacing(10);
    grid_layout->setContentsMargins(0, 10, 0, 10);

    int row = 0;
    grid_layout->addWidget(labels.title = newLabel("", "title"), row++, 0, 1, 3);
    grid_layout->addItem(new QSpacerItem(0, 50), row++, 0, 1, 1);

    grid_layout->addWidget(labels.routes = newLabel("0", "number"), row, 0, Qt::AlignLeft);
    grid_layout->addWidget(labels.distance = newLabel("0", "number"), row, 1, Qt::AlignLeft);
    grid_layout->addWidget(labels.hours = newLabel("0", "number"), row, 2, Qt::AlignLeft);

    grid_layout->addWidget(labels.routes_title = newLabel("", "unit"), row + 1, 0, Qt::AlignLeft);
    grid_layout->addWidget(labels.distance_title = newLabel("", "unit"), row + 1, 1, Qt::AlignLeft);
    grid_layout->addWidget(labels.hours_title = newLabel("", "unit"), row + 1, 2, Qt::AlignLeft);

    main_layout->addLayout(grid_layout);
  };

  add_stats_layouts(all_);
  main_layout->addStretch();
  add_stats_layouts(week_);

  if (auto dongleId = getDongleId()) {
    QString url = CommaApi::BASE_URL + "/v1.1/devices/" + *dongleId + "/stats";
    RequestRepeater* repeater = new RequestRepeater(this, url, "ApiCache_DriveStats", 30);
    QObject::connect(repeater, &RequestRepeater::requestDone, this, &DriveStats::parseResponse);
  }

  setStyleSheet(R"(
    DriveStats {
      background-color: #333333;
      border-radius: 10px;
    }

    QLabel[type="title"] { font-size: 51px; font-weight: 500; }
    QLabel[type="number"] { font-size: 78px; font-weight: 500; }
    QLabel[type="unit"] { font-size: 51px; font-weight: 300; color: #A0A0A0; }
  )");

  translateUi();
}

void DriveStats::translateUi() {
  all_.title->setText(tr("ALL TIME"));
  week_.title->setText(tr("PAST WEEK"));
  for (auto &s : {all_, week_}) {
    s.routes_title->setText(tr("Drives"));
    s.distance_title->setText(getDistanceUnit());
    s.hours_title->setText(tr("Hours"));
  }
}

void DriveStats::updateStats() {
  auto update = [=](const QJsonObject& obj, StatsLabels& labels) {
    labels.routes->setText(QString::number((int)obj["routes"].toDouble()));
    labels.distance->setText(QString::number(int(obj["distance"].toDouble() * (metric_ ? MILE_TO_KM : 1))));
    labels.distance_title->setText(getDistanceUnit());
    labels.hours->setText(QString::number((int)(obj["minutes"].toDouble() / 60)));
  };

  QJsonObject json = stats_.object();
  update(json["all"].toObject(), all_);
  update(json["week"].toObject(), week_);
}

void DriveStats::parseResponse(const QString& response, bool success) {
  if (!success) return;

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
