#pragma once

#include <QWidget>
#include <QNetworkReply>
#include <QLabel>


class PairingQRWidget : public QWidget {
  Q_OBJECT

public:
  explicit PairingQRWidget(QWidget *parent = 0);

private:
  QLabel *qrCode;
  void updateQrCode(QString text);
};
