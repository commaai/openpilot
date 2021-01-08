#pragma once

#include <QWidget>
#include <QNetworkReply>
#include <QLabel>
#include <QStackedLayout>


class PairingQRWidget : public QWidget {
  Q_OBJECT

public:
  explicit PairingQRWidget(QWidget *parent = 0);

private:
  QLabel *qrCode;
  void updateQrCode(QString text);
};


class PrimeUserWidget : public QWidget {
  Q_OBJECT
public:
  explicit PrimeUserWidget(QWidget *parent = 0);
private:
  QVBoxLayout *mainLayout;
  QLabel *username;
  QLabel *points;
  void replyFinished(QNetworkReply *l);

private slots:
  void refresh();
};


class PrimeAdWidget : public QWidget {
  Q_OBJECT
public:
  explicit PrimeAdWidget(QWidget *parent = 0);
};


class SetupWidget : public QWidget {
  Q_OBJECT

public:
  explicit SetupWidget(QWidget *parent = 0);

private:
  QStackedLayout *mainLayout;
  void replyFinished(QNetworkReply *l);

private slots:
  void refresh();

};
