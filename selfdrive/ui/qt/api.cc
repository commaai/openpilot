#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QString>
#include <QWidget>
#include <QTimer>
#include <QRandomGenerator>

#include "api.hpp"
#include "home.hpp"
#include "common/params.h"
#include "common/util.h"

#include <QSslSocket>

#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif

QByteArray CommaApi::rsa_sign(QByteArray data) {
  auto file = QFile(private_key_path.c_str());
  if (!file.open(QIODevice::ReadOnly)) {
    qDebug() << "No RSA private key found, please run manager.py or registration.py";
    return QByteArray();
  }
  auto key = file.readAll();
  file.close();
  file.deleteLater();
  BIO* mem = BIO_new_mem_buf(key.data(), key.size());
  assert(mem);
  RSA* rsa_private = PEM_read_bio_RSAPrivateKey(mem, NULL, NULL, NULL);
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

QString CommaApi::create_jwt(QVector<QPair<QString, QJsonValue>> payloads, int expiry) {
  QString dongle_id = QString::fromStdString(Params().get("DongleId"));

  QJsonObject header;
  header.insert("alg", "RS256");

  QJsonObject payload;
  payload.insert("identity", dongle_id);

  auto t = QDateTime::currentSecsSinceEpoch();
  payload.insert("nbf", t);
  payload.insert("iat", t);
  payload.insert("exp", t + expiry);
  for (auto load : payloads) {
    payload.insert(load.first, load.second);
  }

  auto b64_opts = QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals;
  QString jwt = QJsonDocument(header).toJson(QJsonDocument::Compact).toBase64(b64_opts) + '.' +
                QJsonDocument(payload).toJson(QJsonDocument::Compact).toBase64(b64_opts);

  auto hash = QCryptographicHash::hash(jwt.toUtf8(), QCryptographicHash::Sha256);
  auto sig = rsa_sign(hash);
  jwt += '.' + sig.toBase64(b64_opts);
  return jwt;
}

QString CommaApi::create_jwt() {
  return create_jwt(*(new QVector<QPair<QString, QJsonValue>>()));
}

RequestRepeater::RequestRepeater(QWidget* parent, QString requestURL, int period_seconds, QVector<QPair<QString, QJsonValue>> payloads, bool disableWithScreen)
  : disableWithScreen(disableWithScreen), QObject(parent)  {
  networkAccessManager = new QNetworkAccessManager(this);

  reply = NULL;

  QTimer* timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, [=](){sendRequest(requestURL, payloads);});
  timer->start(period_seconds * 1000);

  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(20000);
  connect(networkTimer, SIGNAL(timeout()), this, SLOT(requestTimeout()));
}

void RequestRepeater::sendRequest(QString requestURL, QVector<QPair<QString, QJsonValue>> payloads){
  // No network calls onroad
  if(GLWindow::ui_state.scene.started){
    return;
  }
  if (!active || (!GLWindow::ui_state.awake && disableWithScreen)) {
    return;
  }
  if(reply != NULL){
    return;
  }

  aborted = false;
  QString token = CommaApi::create_jwt(payloads);
  QNetworkRequest request;
  request.setUrl(QUrl(requestURL));
  request.setRawHeader(QByteArray("Authorization"), ("JWT " + token).toUtf8());

#ifdef QCOM
  QSslConfiguration ssl = QSslConfiguration::defaultConfiguration();
  ssl.setCaCertificates(QSslCertificate::fromPath("/usr/etc/tls/cert.pem",
                        QSsl::Pem, QRegExp::Wildcard));
  request.setSslConfiguration(ssl);
#endif

  reply = networkAccessManager->get(request);

  networkTimer->start();
  connect(reply, SIGNAL(finished()), this, SLOT(requestFinished()));
}

void RequestRepeater::requestTimeout(){
  aborted = true;
  reply->abort();
}

// This function should always emit something
void RequestRepeater::requestFinished(){
  if (!aborted) {
    networkTimer->stop();
    QString response = reply->readAll();
    if (reply->error() == QNetworkReply::NoError) {
      emit receivedResponse(response);
    } else {
      qDebug() << reply->errorString();
      emit failedResponse(reply->errorString());
    }
  } else {
    emit failedResponse("network timeout");
  }
  reply->deleteLater();
  reply = NULL;
}
