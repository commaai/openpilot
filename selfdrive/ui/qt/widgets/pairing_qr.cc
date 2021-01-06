#include <cassert>
#include <iostream>

#include <QDebug>
#include <QLabel>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QVBoxLayout>
#include <QLabel>
#include <QCryptographicHash>
#include <QFile>

#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/pem.h>

#include "pairing_qr.hpp"
#include "common/params.h"
#include "common/utilpp.h"
#include "QrCode.hpp"

using qrcodegen::QrCode;

#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif

QByteArray rsa_sign2(QByteArray data) {
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

QString create_jwt(int expiry=3600) {
  QJsonObject header;
  header.insert("alg", "RS256");
  header.insert("typ", "JWT");

  auto t = QDateTime::currentSecsSinceEpoch();
  QJsonObject payload;
  payload.insert("nbf", t);
  payload.insert("iat", t);
  payload.insert("exp", t + expiry);
  payload.insert("pair", "true");

  QString jwt =
    QJsonDocument(header).toJson(QJsonDocument::Compact).toBase64() +
    '.' +
    QJsonDocument(payload).toJson(QJsonDocument::Compact).toBase64();

  auto hash = QCryptographicHash::hash(jwt.toUtf8(), QCryptographicHash::Sha256);
  auto sig = rsa_sign2(hash);

  jwt += '.' + sig.toBase64();

  return jwt;
}


PairingQRWidget::PairingQRWidget(QWidget *parent) : QWidget(parent) {
  qrCode = new QLabel;
  qrCode->setFixedSize(500, 500);
  qrCode->setScaledContents(true);
  QVBoxLayout *v = new QVBoxLayout;
  v->addWidget(qrCode);
  this->setLayout(v);
  QString IMEI = QString::fromStdString(Params().get("IMEI"));
  QString serial = QString::fromStdString(Params().get("HardwareSerial"));
  QString pairToken = create_jwt();
  QString qrString = IMEI + "--" + serial + "--" + pairToken;
  qDebug() << qrString;
  this->updateQrCode(qrString);
}

void PairingQRWidget::updateQrCode(QString text) {
  QrCode qr = QrCode::encodeText( text.toUtf8().data(), QrCode::Ecc::LOW);
  qint32 sz = qr.getSize();
  // We make the image larger so we can have a white border
  QImage im(sz+2,sz+2, QImage::Format_RGB32);
  QRgb black = qRgb(  0,  0,  0);
  QRgb white = qRgb(255,255,255);

  for (int y = 0; y < sz+2; y++) {
    for (int x = 0; x < sz+2; x++) {
      im.setPixel(x, y, white);
    }
  }
  for (int y = 0; y < sz; y++) {
    for (int x = 0; x < sz; x++) {
      im.setPixel(x+1,y+1,qr.getModule(x, y) ? black : white );
    }
  }
  qrCode->setPixmap( QPixmap::fromImage(im.scaled(256,256,Qt::KeepAspectRatio,Qt::FastTransformation),Qt::MonoOnly) );
}