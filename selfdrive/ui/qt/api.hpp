#pragma once

#include <QWidget>
#include <QVector>
#include <QPair>
#include <QString>
#include <QJsonValue>
#include <QCryptographicHash>
#include <QNetworkReply>
#include <QNetworkRequest>

#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/pem.h>


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
