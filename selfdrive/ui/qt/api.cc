#include <QWidget>
#include <QFile>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDateTime>
#include <QString>
#include <QDebug>
#include <QNetworkReply>
#include <QNetworkRequest>

#include "api.hpp"
#include "common/params.h"
#include "common/utilpp.h"

#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif
CommaApi::CommaApi(QWidget* parent){
  networkAccessManager = new QNetworkAccessManager(parent);
}

QByteArray CommaApi::rsa_sign(QByteArray data) {
  auto file = QFile(private_key_path.c_str());
  if (!file.open(QIODevice::ReadOnly)) {
    qDebug() << file.errorString();
    assert(false);
  }

  auto key = file.readAll();

  file.close();
  file.deleteLater();

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

QString CommaApi::create_jwt(QVector<QPair<QString, QJsonValue>> payloads, int expiry) {
  QJsonObject header;
  header.insert("alg", "RS256");

  QJsonObject payload;

  QString dongle_id = QString::fromStdString(Params().get("DongleId"));
  payload.insert("identity", dongle_id);
  
  auto t = QDateTime::currentSecsSinceEpoch();
  payload.insert("nbf", t);
  payload.insert("iat", t);
  payload.insert("exp", t + expiry);

  for(auto load : payloads){
    payload.insert(load.first, load.second);
  }

  QString jwt =
    QJsonDocument(header).toJson(QJsonDocument::Compact).toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals) +
    '.' +
    QJsonDocument(payload).toJson(QJsonDocument::Compact).toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals);
  auto hash = QCryptographicHash::hash(jwt.toUtf8(), QCryptographicHash::Sha256);
  auto sig = rsa_sign(hash);

  jwt += '.' + sig.toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals);

  return jwt;
}
QString CommaApi::create_jwt() {
  return create_jwt(*(new QVector<QPair<QString, QJsonValue>>()));
}

QNetworkReply* CommaApi::get(QNetworkRequest request){
  return networkAccessManager->get(request);
}
