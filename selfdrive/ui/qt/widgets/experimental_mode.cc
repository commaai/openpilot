#include "selfdrive/ui/qt/widgets/experimental_mode.h"

#include <QDebug>
#include <QVBoxLayout>

#include "common/params.h"
#include "selfdrive/ui/qt/util.h"

ExperimentalMode::ExperimentalMode(QWidget* parent) : QFrame(parent) {
  metric_ = Params().getBool("IsMetric");

  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 50, 50, 60);
  main_layout->addWidget(new QLabel("MODE: CHILL"));

//  auto add_stats_layouts = [=](const QString &title, StatsLabels& labels) {
//    QGridLayout* grid_layout = new QGridLayout;
//    grid_layout->setVerticalSpacing(10);
//    grid_layout->setContentsMargins(0, 10, 0, 10);
//
//    int row = 0;
//    grid_layout->addWidget(newLabel(title, "title"), row++, 0, 1, 3);
//    grid_layout->addItem(new QSpacerItem(0, 50), row++, 0, 1, 1);
//
//    grid_layout->addWidget(labels.routes = newLabel("0", "number"), row, 0, Qt::AlignLeft);
//    grid_layout->addWidget(labels.distance = newLabel("0", "number"), row, 1, Qt::AlignLeft);
//    grid_layout->addWidget(labels.hours = newLabel("0", "number"), row, 2, Qt::AlignLeft);
//
//    grid_layout->addWidget(newLabel((tr("Drives")), "unit"), row + 1, 0, Qt::AlignLeft);
//    grid_layout->addWidget(labels.distance_unit = newLabel(getDistanceUnit(), "unit"), row + 1, 1, Qt::AlignLeft);
//    grid_layout->addWidget(newLabel(tr("Hours"), "unit"), row + 1, 2, Qt::AlignLeft);
//
//    main_layout->addLayout(grid_layout);
//  };
//
//  add_stats_layouts(tr("ALL TIME"), all_);
//  main_layout->addStretch();
//  add_stats_layouts(tr("PAST WEEK"), week_);
//
//  if (auto dongleId = getDongleId()) {
//    QString url = CommaApi::BASE_URL + "/v1.1/devices/" + *dongleId + "/stats";
//    RequestRepeater* repeater = new RequestRepeater(this, url, "ApiCache_ExperimentalMode", 30);
//    QObject::connect(repeater, &RequestRepeater::requestDone, this, &ExperimentalMode::parseResponse);
//  }

  setStyleSheet(R"(
    ExperimentalMode {
      background-color: #333333;
      border-radius: 10px;
    }

    QLabel[type="title"] { font-size: 51px; font-weight: 500; }
    QLabel[type="number"] { font-size: 78px; font-weight: 500; }
    QLabel[type="unit"] { font-size: 51px; font-weight: 300; color: #A0A0A0; }
  )");
}

void ExperimentalMode::updateStats() {
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

void ExperimentalMode::parseResponse(const QString& response, bool success) {
  if (!success) return;

  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting past drives statistics";
    return;
  }
  stats_ = doc;
  updateStats();
}

void ExperimentalMode::showEvent(QShowEvent* event) {
  bool metric = Params().getBool("IsMetric");
  if (metric_ != metric) {
    metric_ = metric;
    updateStats();
  }
}
