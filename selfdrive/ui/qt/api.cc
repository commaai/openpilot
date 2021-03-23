#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkRequest>
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

QString CommaApi::create_jwt(const QMap<QString, QJsonValue> *payloads, int expiry) {
  QString dongle_id = QString::fromStdString(Params().get("DongleId"));

  QJsonObject header;
  header.insert("alg", "RS256");

  QJsonObject payload;
  payload.insert("identity", dongle_id);

  auto t = QDateTime::currentSecsSinceEpoch();
  payload.insert("nbf", t);
  payload.insert("iat", t);
  payload.insert("exp", t + expiry);
  if (payloads) {
    auto it = payloads->constBegin();
    while (it != payloads->constEnd()) {
      payload.insert(it.key(), it.value());
      ++it;
    }
  }

  auto b64_opts = QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals;
  QString jwt = QJsonDocument(header).toJson(QJsonDocument::Compact).toBase64(b64_opts) + '.' +
                QJsonDocument(payload).toJson(QJsonDocument::Compact).toBase64(b64_opts);

  auto hash = QCryptographicHash::hash(jwt.toUtf8(), QCryptographicHash::Sha256);
  auto sig = rsa_sign(hash);
  jwt += '.' + sig.toBase64(b64_opts);
  return jwt;
}

TimeoutRequest::TimeoutRequest(QObject *parent, int timeout_ms) : networkTimer(this), QObject(parent) {
  networkTimer.setSingleShot(true);
  networkTimer.setInterval(timeout_ms);
  connect(&networkTimer, &QTimer::timeout, [=] {
    reply->abort();
  });
#ifdef QCOM
  if (!ssl) {
    ssl = new QSslConfiguration(QSslConfiguration::defaultConfiguration());
    ssl->setCaCertificates(QSslCertificate::fromPath("/usr/etc/tls/cert.pem", QSsl::Pem, QRegExp::Wildcard));
  }
#endif
};

void TimeoutRequest::send(const QString &url, const QMap<QString, QString> *headers) {
  if (reply != nullptr) return;

  QUrl requestUrl(url);
  QNetworkRequest request(requestUrl);
  if (requestUrl.scheme() == "https" && ssl) {
    request.setSslConfiguration(*ssl);
  }
  if (headers) {
    auto it = headers->constBegin();
    while (it != headers->constEnd()) {
      request.setRawHeader(it.key().toUtf8(), it.value().toUtf8());
      ++it;
    }
  }

  reply = networkAccessManager.get(request);
  networkTimer.start();

  QObject::connect(reply, &QNetworkReply::finished, [=]() {
    if (networkTimer.isActive()) {
      networkTimer.stop();
      if (reply->error() == QNetworkReply::NoError) {
        emit finished(reply->readAll(), false);
      } else {
        qDebug() << reply->errorString();
        emit finished(reply->errorString(), true);
      }
    } else {
      emit finished("network timeout", true);
    }
    reply->deleteLater();
    reply = nullptr;
  });
}

RequestRepeater::RequestRepeater(QObject *parent, const QString &cache_key, const QString &url, int period_seconds, int timeout_ms,
                                 bool stop_on_success, const QMap<QString, QJsonValue> *payloads, bool disableWithScreen) : request(parent, timeout_ms) {
  if (!cache_key.isEmpty()) {
    if (std::string cached_resp = Params().get(cache_key.toStdString()); !cached_resp.empty()) {
      QTimer::singleShot(0, [=] {
        emit finished(QString::fromStdString(cached_resp), false);
      });
    }
  }

  QObject::connect(&request, &TimeoutRequest::finished, [=](const QString &response, bool err) {
    if (!err && !cache_key.isEmpty()) {
      Params().write_db_value(cache_key.toStdString(), response.toStdString());
    }
    emit finished(response, err);
  });

  QTimer *timer = new QTimer(this);
  timer->start(period_seconds * 1000);
  QObject::connect(timer, &QTimer::timeout, [=]() {
    // No network calls onroad
    if (GLWindow::ui_state.scene.started) {
      return;
    }
    if (!active || (!GLWindow::ui_state.awake && disableWithScreen)) {
      return;
    }
    QMap<QString, QString> headers{{"Authorization", "JWT " + CommaApi::create_jwt(payloads)}};
    request.send(url, &headers);
  });
}
