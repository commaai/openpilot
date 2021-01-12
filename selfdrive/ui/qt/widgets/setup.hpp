#pragma once

#include <QLabel>
#include <QNetworkReply>
#include <QStackedLayout>
#include <QWidget>

#include "api.hpp"

class PairingQRWidget : public QWidget {
  Q_OBJECT

public:
  explicit PairingQRWidget(QWidget* parent = 0);

private:
  QLabel* qrCode;
  void updateQrCode(QString text);
};

class PrimeUserWidget : public QWidget {
  Q_OBJECT
public:
  explicit PrimeUserWidget(QWidget* parent = 0);

private:
  QVBoxLayout* mainLayout;
  QLabel* username;
  QLabel* points;
  QNetworkReply* reply;
  CommaApi* api;
  void replyFinished();

private slots:
  void refresh();
};

class PrimeAdWidget : public QWidget {
  Q_OBJECT
public:
  explicit PrimeAdWidget(QWidget* parent = 0);
};

class SetupWidget : public QWidget {
  Q_OBJECT

public:
  explicit SetupWidget(QWidget* parent = 0);

private:
  QStackedLayout* mainLayout;
  QNetworkReply* reply;
  CommaApi* api;
  void replyFinished();

private slots:
  void refresh();
};
