#pragma once

#include <QCryptographicHash>
#include <QJsonValue>
#include <QNetworkReply>
#include <QString>
#include <QWidget>
#include <QTimer>
#include <atomic>
#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>

class CommaApi : public QObject {
  Q_OBJECT

public:
  static QByteArray rsa_sign(QByteArray data);
  static QString create_jwt(const QMap<QString, QJsonValue> *payloads = nullptr, int expiry=3600);
};

class TimeoutRequest : public QObject {
  Q_OBJECT
public:
  explicit TimeoutRequest(QObject *parent, int timeout_ms);
  void send(const QString &url, const QMap<QString, QString> *headers = nullptr);

  template <typename Functor>
  static void get(QObject *parent, const QString &url, int timeout_ms, const QMap<QString, QString> *headers, Functor functor) {
    TimeoutRequest *req = new TimeoutRequest(parent, timeout_ms);
    QObject::connect(req, &TimeoutRequest::finished, [=](const QString &resp, bool err) {
      functor(resp, err);
      req->deleteLater();
    });
    req->send(url, headers);
  }
signals:
  void finished(const QString &response, bool err);

private:
  QNetworkReply *reply = nullptr;
  QTimer networkTimer;
  static inline QSslConfiguration *ssl = nullptr;
  static inline QNetworkAccessManager networkAccessManager;
};

class RequestRepeater : public QObject {
  Q_OBJECT
public:
  explicit RequestRepeater(QObject *parent, const QString &cache_key, const QString &url, int period_seconds,
                           int timeout_ms, bool stop_on_success = false, const QMap<QString, QJsonValue> *payloads = nullptr, bool disableWithScreen = true);
  bool active = true;
signals:
  void finished(const QString &response, bool err);

private:
  TimeoutRequest request;
};
