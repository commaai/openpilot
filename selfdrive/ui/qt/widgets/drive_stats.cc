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
  QJsonDocument doc = QJsonDocument::fromJson(answer.toUtf8());
  if (doc.isNull()) {
    qDebug() << "Parse failed";
  }

  QJsonObject json = doc.object();
  auto all = json["all"].toObject();
  auto week = json["week"].toObject();

  auto all_distance = all["distance"].toDouble();
  auto all_minutes = all["minutes"].toDouble();
  auto all_routes = all["routes"].toDouble();
  auto week_distance = week["distance"].toDouble();
  auto week_minutes = week["minutes"].toDouble();
  auto week_routes = week["routes"].toDouble();

  QGridLayout *gl = new QGridLayout();

  QVBoxLayout *all_drives_layout = new QVBoxLayout;
  all_drives_layout->addWidget(new QLabel(QString("%1").arg(all_routes)));
  all_drives_layout->addWidget(new QLabel("DRIVES"));
  gl->addWidget(widget(all_drives_layout), 0, 0);

  QVBoxLayout *all_distance_layout = new QVBoxLayout;
  all_distance_layout->addWidget(new QLabel(QString("%1").arg(all_distance)));
  all_distance_layout->addWidget(new QLabel("MILES"));
  gl->addWidget(widget(all_distance_layout), 0, 1);

  QVBoxLayout *all_hours_layout = new QVBoxLayout;
  all_hours_layout->addWidget(new QLabel(QString("%1").arg(all_minutes/60)));
  all_hours_layout->addWidget(new QLabel("HOURS"));
  gl->addWidget(widget(all_hours_layout), 0, 2);


  QVBoxLayout *week_drives_layout = new QVBoxLayout;
  week_drives_layout->addWidget(new QLabel(QString("%1").arg(week_routes)));
  week_drives_layout->addWidget(new QLabel("DRIVES"));
  gl->addWidget(widget(week_drives_layout), 1, 0);

  QVBoxLayout *week_distance_layout = new QVBoxLayout;
  week_distance_layout->addWidget(new QLabel(QString("%1").arg(week_distance)));
  week_distance_layout->addWidget(new QLabel("MILES"));
  gl->addWidget(widget(week_distance_layout), 1, 1);

  QVBoxLayout *week_hours_layout = new QVBoxLayout;
  week_hours_layout->addWidget(new QLabel(QString("%1").arg(week_minutes/60)));
  week_hours_layout->addWidget(new QLabel("HOURS"));
  gl->addWidget(widget(week_hours_layout), 1, 2);


  f->setLayout(gl);
  f->setStyleSheet(R"(
    QFrame {
      border-radius: 20px;
      border: 2px solid white;
    }
    QLabel{
      border-radius: 20px;
      border: 0px solid red;
    }
    background-color: #114365;
    font-size: 50px;
  )");
  
}
DriveStats::DriveStats(QWidget *parent) : QWidget(parent){
  f = new QFrame;
  QVBoxLayout *v = new QVBoxLayout;
  v->addWidget(f);
  setLayout(v);
  

  std::string result = exec("python -c \"from common.api import Api; from common.params import Params; print(Api(Params().get('DongleId', encoding='utf8')).get_token());\" 2>/dev/null");
  result = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJpZGVudGl0eSI6Ijk4YzA5NGI4ZTg1YmUxODkiLCJuYmYiOjE2MDc5NjkwMTksImlhdCI6MTYwNzk2OTAxOSwiZXhwIjoxNjA3OTcyNjE5fQ.kbF6rC6HXLWCHQX_nsceJXO-GTlqBPXCX64VZt5uQ3HZ-09UV9T7sZSwpRnPJLWuDdD2B22uxxYfoTx3DxwrteSjkpz9TMu0CAJ-v3oMFhwK8DBZHj2vVWlVIU9roazEg1LD16iKQxxkxdI_1wxd0jTVBYZwctQi5Bih41vwfGdg8p9tK5ZMQzxBLHRw9pBFUmJGXM_fWcCUKU77Fy8GakTYI6_wKPZd1lO65-s19e68gVGjLc3yNfw0C9TcsE67Jct75NdLWPjG5A_k19TxVcLHYcqaMlUbvT5btuJu35uzqZAOr_DnaUG2LPPcisY-_bFD-axHCdV7ZR6OGUlL6Q";

  QString auth_token = QString::fromStdString(result);

  // qDebug()<<auth_token;
  QNetworkAccessManager *manager = new QNetworkAccessManager(this);
  connect(manager, &QNetworkAccessManager::finished, this, &DriveStats::replyFinished);

  QNetworkRequest request;
  request.setUrl(QUrl("https://api.commadotai.com/v1.1/devices/98c094b8e85be189/stats"));
  request.setRawHeader("Authorization", ("JWT "+auth_token).toUtf8());

  manager->get(request);
}