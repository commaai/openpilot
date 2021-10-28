#include "selfdrive/ui/qt/api.h"

#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>

#include <QCryptographicHash>
#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QNetworkRequest>

#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"

namespace CommaApi {

QByteArray rsa_sign(const QByteArray &data) {
  static std::string key = util::read_file(Path::rsa_file());
  if (key.empty()) {
    qDebug() << "No RSA private key found, please run manager.py or registration.py";
    return {};
  }

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

QString create_jwt(const QJsonObject &payloads, int expiry) {
  QJsonObject header = {{"alg", "RS256"}};

  auto t = QDateTime::currentSecsSinceEpoch();
  QJsonObject payload = {{"identity", getDongleId().value_or("")}, {"nbf", t}, {"iat", t}, {"exp", t + expiry}};
  for (auto it = payloads.begin(); it != payloads.end(); ++it) {
    payload.insert(it.key(), it.value());
  }

  auto b64_opts = QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals;
  QString jwt = QJsonDocument(header).toJson(QJsonDocument::Compact).toBase64(b64_opts) + '.' +
                QJsonDocument(payload).toJson(QJsonDocument::Compact).toBase64(b64_opts);

  auto hash = QCryptographicHash::hash(jwt.toUtf8(), QCryptographicHash::Sha256);
  auto sig = rsa_sign(hash);
  jwt += '.' + sig.toBase64(b64_opts);
  return jwt;
}

}  // namespace CommaApi

HttpRequest::HttpRequest(QObject *parent, bool create_jwt, int timeout) : create_jwt(create_jwt), QObject(parent) {
  networkAccessManager = new QNetworkAccessManager(this);

  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(timeout);
  connect(networkTimer, &QTimer::timeout, this, &HttpRequest::requestTimeout);
}

bool HttpRequest::active() {
  return reply != nullptr;
}

void HttpRequest::sendRequest(const QString &requestURL, const HttpRequest::Method method) {
  if (active()) {
    qDebug() << "HttpRequest is active";
    return;
  }
  QString token;
  if(create_jwt) {
    token = CommaApi::create_jwt();
  } else {
    QString token_json = QString::fromStdString(util::read_file(util::getenv("HOME") + "/.comma/auth.json"));
    QJsonDocument json_d = QJsonDocument::fromJson(token_json.toUtf8());
    token = json_d["access_token"].toString();
  }

  QNetworkRequest request;
  request.setUrl(QUrl(requestURL));

  if (!token.isEmpty()) {
    request.setRawHeader(QByteArray("Authorization"), ("JWT " + token).toUtf8());
  }

  if (method == HttpRequest::Method::GET) {
    reply = networkAccessManager->get(request);
  } else if (method == HttpRequest::Method::DELETE) {
    reply = networkAccessManager->deleteResource(request);
  }

  networkTimer->start();
  connect(reply, &QNetworkReply::finished, this, &HttpRequest::requestFinished);
}

void HttpRequest::requestTimeout() {
  reply->abort();
}

// This function should always emit something
void HttpRequest::requestFinished() {
  bool success = false;
  if (reply->error() != QNetworkReply::OperationCanceledError) {
    networkTimer->stop();
    QString response = reply->readAll();

    if (reply->error() == QNetworkReply::NoError) {
      success = true;
      emit receivedResponse(response);
    } else {
      emit failedResponse(reply->errorString());

      if (reply->error() == QNetworkReply::ContentAccessDenied || reply->error() == QNetworkReply::AuthenticationRequiredError) {
        qWarning() << ">>  Unauthorized. Authenticate with tools/lib/auth.py  <<";
      }
    }
  } else {
    networkAccessManager->clearAccessCache();
    networkAccessManager->clearConnectionCache();
    emit timeoutResponse("timeout");
  }
  emit requestDone(success);
  reply->deleteLater();
  reply = nullptr;
}
