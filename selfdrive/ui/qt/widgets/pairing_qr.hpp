#pragma once

#include <QWidget>
#include <QNetworkReply>


class PairingQRWidget : public QWidget {
  Q_OBJECT

public:
  explicit PairingQRWidget(QWidget *parent = 0);

// private:
//   void replyFinished(QNetworkReply *l);
};
