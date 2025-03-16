#include "selfdrive/ui/qt/api.h"

#include <openssl/pem.h>
#include <openssl/rsa.h>

#include <QApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDebug>
#include <QJsonDocument>
#include <QNetworkRequest>

#include <memory>
#include <string>

#include "common/util.h"
#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"

namespace CommaApi {

RSA *get_rsa_private_key() {
  static std::unique_ptr<RSA, decltype(&RSA_free)> rsa_private(nullptr, RSA_free);
  if (!rsa_private) {
    FILE *fp = fopen(Path::rsa_file().c_str(), "rb");
    if (!fp) {
      qDebug() << "No RSA private key found, please run manager.py or registration.py";
      return nullptr;
    }
    rsa_private.reset(PEM_read_RSAPrivateKey(fp, NULL, NULL, NULL));
    fclose(fp);
  }
  return rsa_private.get();
}

QByteArray rsa_sign(const QByteArray &data) {
  RSA *rsa_private = get_rsa_private_key();
  if (!rsa_private) return {};

  QByteArray sig(RSA_size(rsa_private), Qt::Uninitialized);
  unsigned int sig_len;
  int ret = RSA_sign(NID_sha256, (unsigned char*)data.data(), data.size(), (unsigned char*)sig.data(), &sig_len, rsa_private);
  assert(ret == 1);
  assert(sig.size() == sig_len);
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
  return jwt + "." + rsa_sign(hash).toBase64(b64_opts);
}

}  // namespace CommaApi

HttpRequest::HttpRequest(QObject *parent, bool create_jwt, int timeout) : create_jwt(create_jwt), QObject(parent) {
  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(timeout);
  connect(networkTimer, &QTimer::timeout, this, &HttpRequest::requestTimeout);
}

bool HttpRequest::active() const {
  return reply != nullptr;
}

bool HttpRequest::timeout() const {
  return reply && reply->error() == QNetworkReply::OperationCanceledError;
}

QNetworkRequest HttpRequest::prepareRequest(const QString &requestURL) {
  QNetworkRequest request;
  QString token;
  if (create_jwt) {
    token = GetJwtToken();
  } else {
    QString token_json = QString::fromStdString(util::read_file(util::getenv("HOME") + "/.comma/auth.json"));
    QJsonDocument json_d = QJsonDocument::fromJson(token_json.toUtf8());
    token = json_d["access_token"].toString();
  }

  request.setUrl(QUrl(requestURL));
  request.setRawHeader("User-Agent", GetUserAgent().toUtf8());

  if (!token.isEmpty()) {
    request.setRawHeader(QByteArray("Authorization"), ("JWT " + token).toUtf8());
  }
  return request;
}

void HttpRequest::sendRequest(const QString &requestURL, const Method method) {
  if (active()) {
    qDebug() << "HttpRequest is active";
    return;
  }
  
  QNetworkRequest request = prepareRequest(requestURL);
  if (method == Method::GET) {
    reply = nam()->get(request);
  } else if (method == Method::DELETE) {
    reply = nam()->deleteResource(request);
  }

  networkTimer->start();
  connect(reply, &QNetworkReply::finished, this, &HttpRequest::requestFinished);
}

void HttpRequest::requestTimeout() {
  reply->abort();
}

void HttpRequest::requestFinished() {
  networkTimer->stop();

  if (reply->error() == QNetworkReply::NoError) {
    emit requestDone(reply->readAll(), true, reply->error());
  } else {
    QString error;
    if (reply->error() == QNetworkReply::OperationCanceledError) {
      nam()->clearAccessCache();
      nam()->clearConnectionCache();
      error = "Request timed out";
    } else {
      error = reply->errorString();
    }
    emit requestDone(error, false, reply->error());
  }

  reply->deleteLater();
  reply = nullptr;
}

QNetworkAccessManager *HttpRequest::nam() {
  static QNetworkAccessManager *networkAccessManager = new QNetworkAccessManager(qApp);
  return networkAccessManager;
}
