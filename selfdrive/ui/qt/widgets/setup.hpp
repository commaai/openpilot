#pragma once

#include <QLabel>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QWidget>

#include "api.hpp"

class PairingQRWidget : public QWidget {
  Q_OBJECT

public:
  explicit PairingQRWidget(QWidget* parent = 0);

private:
  QLabel* qrCode;
  void updateQrCode(QString text);

private slots:
  void refresh();
};

class PrimeUserWidget : public QWidget {
  Q_OBJECT
public:
  explicit PrimeUserWidget(QWidget* parent = 0);

private:
  QVBoxLayout* mainLayout;
  QLabel* username;
  QLabel* points;
  CommaApi* api;

private slots:
  void replyFinished(QString response);
};

class PrimeAdWidget : public QWidget {
  Q_OBJECT
public:
  explicit PrimeAdWidget(QWidget* parent = 0);
};

class SetupWidget : public QFrame {
  Q_OBJECT

public:
  explicit SetupWidget(QWidget* parent = 0);

private:
  QStackedWidget* mainLayout;
  CommaApi* api;
  PrimeAdWidget *primeAd;
  PrimeUserWidget *primeUser;
  bool showQr = false;

private slots:
  void parseError(QString response);
  void replyFinished(QString response);
  void showQrCode();
};
