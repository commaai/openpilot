#include "tools/cabana/utils/api.h"

#include <openssl/evp.h>
#include <openssl/pem.h>

#include <QApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDebug>
#include <QJsonDocument>
#include <QNetworkRequest>

#include <memory>
#include <string>

#include "common/params.h"
#include "common/util.h"
#include "system/hardware/hw.h"
#include "tools/cabana/utils/util.h"

QString getVersion() {
  static QString version =  QString::fromStdString(Params().get("Version"));
  return version;
}

QString getUserAgent() {
  return "openpilot-" + getVersion();
}

std::optional<QString> getDongleId() {
  std::string id = Params().get("DongleId");

  if (!id.empty() && (id != "UnregisteredDevice")) {
    return QString::fromStdString(id);
  } else {
    return {};
  }
}

namespace CommaApi {

EVP_PKEY *get_private_key() {
  static std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> pkey(nullptr, EVP_PKEY_free);
  if (!pkey) {
    FILE *fp = fopen(Path::rsa_file().c_str(), "rb");
    if (!fp) {
      qDebug() << "No private key found, please run manager.py or registration.py";
      return nullptr;
    }
    pkey.reset(PEM_read_PrivateKey(fp, nullptr, nullptr, nullptr));
    fclose(fp);
  }
  return pkey.get();
}

QByteArray rsa_sign(const QByteArray &data) {
  EVP_PKEY *pkey = get_private_key();
  if (!pkey) return {};

  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  if (!mdctx) return {};

  QByteArray sig(EVP_PKEY_size(pkey), Qt::Uninitialized);
  size_t sig_len = sig.size();

  int ret = EVP_DigestSignInit(mdctx, nullptr, EVP_sha256(), nullptr, pkey);
  ret &= EVP_DigestSignUpdate(mdctx, data.data(), data.size());
  ret &= EVP_DigestSignFinal(mdctx, (unsigned char*)sig.data(), &sig_len);

  EVP_MD_CTX_free(mdctx);

  if (ret != 1) return {};
  sig.resize(sig_len);
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

void HttpRequest::sendRequest(const QString &requestURL, const HttpRequest::Method method) {
  if (active()) {
    qDebug() << "HttpRequest is active";
    return;
  }
  QString token;
  if (create_jwt) {
    token = CommaApi::create_jwt();
  } else {
    QString token_json = QString::fromStdString(util::read_file(util::getenv("HOME") + "/.comma/auth.json"));
    QJsonDocument json_d = QJsonDocument::fromJson(token_json.toUtf8());
    token = json_d["access_token"].toString();
  }

  QNetworkRequest request;
  request.setUrl(QUrl(requestURL));
  request.setRawHeader("User-Agent", getUserAgent().toUtf8());

  if (!token.isEmpty()) {
    request.setRawHeader(QByteArray("Authorization"), ("JWT " + token).toUtf8());
  }

  if (method == HttpRequest::Method::GET) {
    reply = nam()->get(request);
  } else if (method == HttpRequest::Method::DELETE) {
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
