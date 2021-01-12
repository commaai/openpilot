#pragma once

#include <QCryptographicHash>
#include <QJsonValue>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPair>
#include <QString>
#include <QVector>
#include <QWidget>

#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>

class CommaApi : public QWidget {
  Q_OBJECT
public:
  explicit CommaApi(QWidget* parent);
  QByteArray rsa_sign(QByteArray data);
  QString create_jwt(QVector<QPair<QString, QJsonValue>> payloads, int expiry = 3600);
  QString create_jwt();
  QNetworkReply* get(QNetworkRequest request);

private:
  QNetworkAccessManager* networkAccessManager;
};
