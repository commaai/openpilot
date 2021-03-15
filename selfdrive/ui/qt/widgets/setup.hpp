#pragma once

#include <QLabel>
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

class SetupWidget : public QWidget {
  Q_OBJECT

public:
  explicit SetupWidget(QWidget* parent = 0);

private:
  QStackedLayout* mainLayout;
  CommaApi* api;
  bool showQr = false;

private slots:
  void parseError(QString response);
  void replyFinished(QString response);
  void showQrCode();
};
