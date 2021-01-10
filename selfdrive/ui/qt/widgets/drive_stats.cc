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
#include <QCryptographicHash>

#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/pem.h>

#include "drive_stats.hpp"
#include "common/params.h"
#include "common/util.h"


constexpr double MILE_TO_KM = 1.60934;


#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif


QByteArray rsa_sign(QByteArray data) {
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

QString create_jwt(QString dongle_id, int expiry=3600) {
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
  QString token = create_jwt(dongle_id);

  QNetworkAccessManager *manager = new QNetworkAccessManager(this);
  connect(manager, &QNetworkAccessManager::finished, this, &DriveStats::replyFinished);

  QNetworkRequest request;
  request.setUrl(QUrl("https://api.commadotai.com/v1.1/devices/" + dongle_id + "/stats"));
  request.setRawHeader("Authorization", ("JWT "+token).toUtf8());

  manager->get(request);
}
