#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QLineEdit>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>

#include "drive_stats.hpp"
#include "common/params.h"
double MILE_TO_KM = 1.60934;

std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

QWidget *widget(QLayout *l){
  QWidget *q = new QWidget();
  q->setLayout(l);
  return q;
}
void DriveStats::replyFinished(QNetworkReply *l){
  QString answer = l->readAll();
  answer.chop(1);
  // qDebug()<<answer;
  QJsonDocument doc = QJsonDocument::fromJson(answer.toUtf8());
  if (doc.isNull()) {
    qDebug() << "Parse failed";
  }
  QString IsMetric = QString::fromStdString(Params().get("IsMetric"));
  bool metric = (IsMetric =="1");

  QJsonObject json = doc.object();
  auto all = json["all"].toObject();
  auto week = json["week"].toObject();

  
  int all_distance = all["distance"].toDouble()*(metric ? MILE_TO_KM : 1);
  int all_minutes = all["minutes"].toDouble();
  int all_routes = all["routes"].toDouble();
  int week_distance = week["distance"].toDouble()*(metric ? MILE_TO_KM : 1);
  int week_minutes = week["minutes"].toDouble();
  int week_routes = week["routes"].toDouble();

  QGridLayout *gl = new QGridLayout();

  QLabel *past_week = new QLabel("PAST WEEK");
  gl->addWidget(past_week, 0, 0, 1, 3);

  QVBoxLayout *all_drives_layout = new QVBoxLayout;
  all_drives_layout->addWidget(new QLabel(QString("%1").arg(all_routes)));
  all_drives_layout->addWidget(new QLabel("DRIVES"));
  gl->addWidget(widget(all_drives_layout), 1, 0);

  QVBoxLayout *all_distance_layout = new QVBoxLayout;
  all_distance_layout->addWidget(new QLabel(QString("%1").arg(all_distance)));
  all_distance_layout->addWidget(new QLabel(metric ? "KILOMETERS" : "MILES"));
  gl->addWidget(widget(all_distance_layout), 1, 1);

  QVBoxLayout *all_hours_layout = new QVBoxLayout;
  all_hours_layout->addWidget(new QLabel(QString("%1").arg(all_minutes/60)));
  all_hours_layout->addWidget(new QLabel("HOURS"));
  gl->addWidget(widget(all_hours_layout), 1, 2);

  QFrame *lineA = new QFrame;
  lineA->setFrameShape(QFrame::HLine);
  lineA->setFrameShadow(QFrame::Sunken);
  lineA->setProperty("class", "line");
  gl->addWidget(lineA, 2, 0, 1, 3);


  QLabel *all_time = new QLabel("ALL TIME");
  gl->addWidget(all_time, 3, 0, 1, 3);


  QVBoxLayout *week_drives_layout = new QVBoxLayout;
  week_drives_layout->addWidget(new QLabel(QString("%1").arg(week_routes)));
  week_drives_layout->addWidget(new QLabel("DRIVES"));
  gl->addWidget(widget(week_drives_layout), 4, 0);

  QVBoxLayout *week_distance_layout = new QVBoxLayout;
  week_distance_layout->addWidget(new QLabel(QString("%1").arg(week_distance)));
  week_distance_layout->addWidget(new QLabel(metric ? "KILOMETERS" : "MILES"));
  gl->addWidget(widget(week_distance_layout), 4, 1);

  QVBoxLayout *week_hours_layout = new QVBoxLayout;
  week_hours_layout->addWidget(new QLabel(QString("%1").arg(week_minutes/60)));
  week_hours_layout->addWidget(new QLabel("HOURS"));
  gl->addWidget(widget(week_hours_layout), 4, 2);


  f->setLayout(gl);
  f->setStyleSheet(R"(
    [class="line"]{
      border: 2px solid white;
    }
    [class="outside"]{
      border-radius: 20px;
      border: 2px solid white;
      padding: 30px;
    }
    QLabel{
      background-color: #114365;
      font-size: 50px;
      margin: 10px;
    }
  )");
  
}
DriveStats::DriveStats(QWidget *parent) : QWidget(parent){
  f = new QFrame;
  f->setProperty("class", "outside");
  QVBoxLayout *v = new QVBoxLayout;
  v->addWidget(f);
  setLayout(v);
  

  std::string result = exec("python -c \"from common.api import Api; from common.params import Params; print(Api(Params().get('DongleId', encoding='utf8')).get_token());\" 2>/dev/null");

  QString auth_token = QString::fromStdString(result);
  // qDebug()<<auth_token;
  QNetworkAccessManager *manager = new QNetworkAccessManager(this);
  connect(manager, &QNetworkAccessManager::finished, this, &DriveStats::replyFinished);

  QNetworkRequest request;
  QString dongleId = QString::fromStdString(Params().get("DongleId"));
  request.setUrl(QUrl("https://api.commadotai.com/v1.1/devices/" + dongleId + "/stats"));
  request.setRawHeader("Authorization", ("JWT "+auth_token).toUtf8());

  manager->get(request);
}
