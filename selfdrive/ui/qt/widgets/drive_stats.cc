#include <iostream>

#include <QFile>
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
#include <QCryptographicHash>

#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/pem.h>

#include "drive_stats.hpp"
#include "common/params.h"
#include "common/utilpp.h"
double MILE_TO_KM = 1.60934;


#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif

QByteArray rsa_sign(QByteArray data){
  auto file = QFile(private_key_path.c_str());
  bool r = file.open(QIODevice::ReadOnly);
  assert(r);

  auto key = file.readAll();

  BIO *mem = BIO_new_mem_buf(key.data(), key.size());
  assert(mem);

  RSA *rsa_private = PEM_read_bio_RSAPrivateKey(mem, NULL, NULL, NULL);
  assert(rsa_private);

  auto sig = QByteArray();
  sig.resize(RSA_size(rsa_private));

  unsigned int sig_len;
  int ret = RSA_sign(NID_sha256, (unsigned char*)data.data(), data.size(), (unsigned char*)sig.data(), &sig_len, rsa_private);

  assert(ret == 1);
  assert(sig_len == sig.size());

  BIO_free(mem);
  RSA_free(rsa_private);

  return sig;
}

QString create_jwt(QString dongle_id, int expiry=3600){
  QJsonObject header;
  header.insert("alg", "RS256");
  header.insert("typ", "JWT");

  auto t = QDateTime::currentSecsSinceEpoch();
  QJsonObject payload;
  payload.insert("identity", dongle_id);
  payload.insert("nbf", t);
  payload.insert("iat", t);
  payload.insert("exp", t + expiry);

  QString jwt =
    QJsonDocument(header).toJson(QJsonDocument::Compact).toBase64() +
    '.' +
    QJsonDocument(payload).toJson(QJsonDocument::Compact).toBase64();

  auto hash = QCryptographicHash::hash(jwt.toUtf8(), QCryptographicHash::Sha256);
  auto sig = rsa_sign(hash);

  jwt += '.' + sig.toBase64();

  return jwt;
}

QString bold(QString s) {
  return "<b>" + s + "</b>";
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
    qDebug() << "JSON Parse failed";
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

  QLabel *all_time = new QLabel(bold("ALL TIME"));
  gl->addWidget(all_time, 0, 0, 1, 3);

  QVBoxLayout *all_drives_layout = new QVBoxLayout;
  all_drives_layout->addWidget(new QLabel(bold(QString("%1").arg(all_routes))), 1, Qt::AlignCenter);
  all_drives_layout->addWidget(new QLabel("DRIVES"),1, Qt::AlignCenter);
  gl->addWidget(widget(all_drives_layout), 1, 0, 3, 1);

  QVBoxLayout *all_distance_layout = new QVBoxLayout;
  all_distance_layout->addWidget(new QLabel(bold(QString("%1").arg(all_distance))), 1, Qt::AlignCenter);
  all_distance_layout->addWidget(new QLabel(metric ? "KM" : "MILES"),1, Qt::AlignCenter);
  gl->addWidget(widget(all_distance_layout), 1, 1, 3, 1);

  QVBoxLayout *all_hours_layout = new QVBoxLayout;
  all_hours_layout->addWidget(new QLabel(bold(QString("%1").arg(all_minutes/60))), 1, Qt::AlignCenter);
  all_hours_layout->addWidget(new QLabel("HOURS"),1, Qt::AlignCenter);
  gl->addWidget(widget(all_hours_layout), 1, 2, 3, 1);

  QFrame *lineA = new QFrame;

  lineA->setFrameShape(QFrame::HLine);
  lineA->setFrameShadow(QFrame::Sunken);
  lineA->setProperty("class", "line");
  gl->addWidget(lineA, 5, 0, 1, 3);

  QLabel *past_week = new QLabel(bold("PAST WEEK"));
  gl->addWidget(past_week, 6, 0, 1, 3);

  QVBoxLayout *week_drives_layout = new QVBoxLayout;
  week_drives_layout->addWidget(new QLabel(bold(QString("%1").arg(week_routes))), 1, Qt::AlignCenter);
  week_drives_layout->addWidget(new QLabel("DRIVES"),1, Qt::AlignCenter);
  gl->addWidget(widget(week_drives_layout), 7, 0, 3, 1);

  QVBoxLayout *week_distance_layout = new QVBoxLayout;
  week_distance_layout->addWidget(new QLabel(bold(QString("%1").arg(week_distance))), 1, Qt::AlignCenter);
  week_distance_layout->addWidget(new QLabel(metric ? "KM" : "MILES"),1, Qt::AlignCenter);
  gl->addWidget(widget(week_distance_layout), 7, 1, 3, 1);

  QVBoxLayout *week_hours_layout = new QVBoxLayout;
  week_hours_layout->addWidget(new QLabel(bold(QString("%1").arg(week_minutes/60))), 1, Qt::AlignCenter);
  week_hours_layout->addWidget(new QLabel("HOURS"),1, Qt::AlignCenter);
  gl->addWidget(widget(week_hours_layout), 7, 2, 3, 1);


  f->setLayout(gl);
  f->setStyleSheet(R"(
    [class="line"]{
      border: 2px solid white;
    }
    [class="outside"]{
      border-radius: 20px;
      border: 2px solid white;
      padding: 10px;
    }
    QLabel{
      font-size: 70px;
      font-weight: 200;
    }
  )");

}
DriveStats::DriveStats(QWidget *parent) : QWidget(parent){
  f = new QFrame;
  f->setProperty("class", "outside");
  QVBoxLayout *v = new QVBoxLayout;
  v->addWidget(f);
  setLayout(v);

  QString dongle_id = QString::fromStdString(Params().get("DongleId"));
  QString token = create_jwt(dongle_id);

  QNetworkAccessManager *manager = new QNetworkAccessManager(this);
  connect(manager, &QNetworkAccessManager::finished, this, &DriveStats::replyFinished);

  QNetworkRequest request;
  request.setUrl(QUrl("https://api.commadotai.com/v1.1/devices/" + dongle_id + "/stats"));
  request.setRawHeader("Authorization", ("JWT "+token).toUtf8());

  manager->get(request);
}
