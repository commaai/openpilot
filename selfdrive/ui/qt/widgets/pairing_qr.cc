#include <cassert>
#include <iostream>

#include <QDebug>
#include <QLabel>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>

#include "pairing_qr.hpp"
#include "common/params.h"
#include "QrCode.hpp"

using qrcodegen::QrCode;

PairingQRWidget::PairingQRWidget(QWidget *parent) : QWidget(parent) {
  QString dongle_id = QString::fromStdString(Params().get("DongleId"));
  qDebug() << dongle_id;
  QrCode qr = QrCode::encodeText("Hello, world!", QrCode::Ecc::MEDIUM);

  // Read the black & white pixels
  for (int y = 0; y < qr.getSize(); y++) {
      QDebug deb = qDebug();
      for (int x = 0; x < qr.getSize(); x++) {
          int color = qr.getModule(x, y);  // 0 for white, 1 for black

          if (color) {
            deb << "#";
          } else {
            deb << ".";
          }
      }
  }
  qDebug()<<"Done with QR code";
}