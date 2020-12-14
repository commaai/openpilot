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
  // qDebug().noquote() << answer;
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

  QVBoxLayout *vlayout = new QVBoxLayout;
    QHBoxLayout *hlayout_all = new QHBoxLayout;
        QVBoxLayout *all_drives_layout = new QVBoxLayout;
        all_drives_layout->addWidget(new QLabel(QString("%1").arg(all_routes)));
        all_drives_layout->addWidget(new QLabel("DRIVES"));
    hlayout_all->addWidget(widget(all_drives_layout));
        QVBoxLayout *all_distance_layout = new QVBoxLayout;
        all_distance_layout->addWidget(new QLabel(QString("%1").arg(all_distance)));
        all_distance_layout->addWidget(new QLabel("MILES"));
    hlayout_all->addWidget(widget(all_distance_layout));
        QVBoxLayout *all_hours_layout = new QVBoxLayout;
        all_hours_layout->addWidget(new QLabel(QString("%1").arg(all_minutes/60)));
        all_hours_layout->addWidget(new QLabel("HOURS"));
    hlayout_all->addWidget(widget(all_hours_layout));
    vlayout->addWidget(widget(hlayout_all));

    QHBoxLayout *hlayout_week = new QHBoxLayout;
        QVBoxLayout *week_drives_layout = new QVBoxLayout;
        week_drives_layout->addWidget(new QLabel(QString("%1").arg(week_routes)));
        week_drives_layout->addWidget(new QLabel("DRIVES"));
    hlayout_week->addWidget(widget(week_drives_layout));
        QVBoxLayout *week_distance_layout = new QVBoxLayout;
        week_distance_layout->addWidget(new QLabel(QString("%1").arg(week_distance)));
        week_distance_layout->addWidget(new QLabel("MILES"));
    hlayout_week->addWidget(widget(week_distance_layout));
        QVBoxLayout *week_hours_layout = new QVBoxLayout;
        week_hours_layout->addWidget(new QLabel(QString("%1").arg(week_minutes/60)));
        week_hours_layout->addWidget(new QLabel("HOURS"));
    hlayout_week->addWidget(widget(week_hours_layout));
    vlayout->addWidget(widget(hlayout_week));

  f->setLayout(vlayout);
  f->setStyleSheet(R"(
    QFrame{
      border-radius: 20px;
      border: 2px solid white;
    };
    QVBoxLayout{
      background-color: #114365;
    };
    QLabel{
      color: white;
    };
    font-size: 50px;
  )");
  
}
DriveStats::DriveStats(QWidget *parent) : QWidget(parent){
  f = new QFrame;
  QVBoxLayout *v = new QVBoxLayout;
  v->addWidget(f);
  setLayout(v);
  

  std::string result = exec("python -c \"from common.api import Api; from common.params import Params; print(Api(Params().get('DongleId', encoding='utf8')).get_token());\" 2>/dev/null");
  result = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJpZGVudGl0eSI6Ijk4YzA5NGI4ZTg1YmUxODkiLCJuYmYiOjE2MDc5NjExMzgsImlhdCI6MTYwNzk2MTEzOCwiZXhwIjoxNjA3OTY0NzM4fQ.Z_AEBqSpSXYeYiVAHU5IJyN9tE3WsphYn1yP1o8MtCKjLYeLuo-wPdPEMRu8vZsVIkBRu-IbYfWaxp4SCaXMThazuJWPerbxZpGYyC1bYZg4uaEhANEFVwVNDb2YJRku4B5YBmeWoc6-o4JsCTuQn3OdqQkt9dBMonZCa-BVZgy-IqnQ4JAKPUO_TJK0HV-45d0bfhMCK-Oi5gP8gxJefRS3q-2TjyNVjIP4n3EJNLwsgytVqPIr42FTI6Qiq9S7kFMOnqaGEI1ZYaPrYI7hTobHa3dtwcawBeppFhdi79l62qbIzORuH9n8guJaQTOpRkFCyN4uQyNBApuURpzYjg";

  QString auth_token = QString::fromStdString(result);

  // qDebug()<<auth_token;
  QNetworkAccessManager *manager = new QNetworkAccessManager(this);
  connect(manager, &QNetworkAccessManager::finished, this, &DriveStats::replyFinished);

  QNetworkRequest request;
  request.setUrl(QUrl("https://api.commadotai.com/v1.1/devices/98c094b8e85be189/stats"));
  request.setRawHeader("Authorization", ("JWT "+auth_token).toUtf8());

  manager->get(request);
}