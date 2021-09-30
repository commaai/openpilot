#pragma once

#include <QLabel>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QWidget>

#include "selfdrive/ui/qt/widgets/input.h"

// pairing QR code
class PairingQRWidget : public QWidget {
  Q_OBJECT

public:
  explicit PairingQRWidget(QWidget* parent = 0);

private:
  QLabel* qrCode;
  void updateQrCode(const QString &text);
  void showEvent(QShowEvent *event) override;

private slots:
  void refresh();
};

// pairing popup widget
class PairingDialog : public QDialogBase {
  Q_OBJECT

public:
  explicit PairingDialog(QWidget* parent);
  static bool show(QWidget *parent);
};

// widget for paired users with prime
class PrimeUserWidget : public QWidget {
  Q_OBJECT
public:
  explicit PrimeUserWidget(QWidget* parent = 0);

private:
  QVBoxLayout* mainLayout;
  QLabel* points;

private slots:
  void replyFinished(const QString &response);
};


// widget for paired users without prime
class PrimeAdWidget : public QFrame {
  Q_OBJECT
public:
  explicit PrimeAdWidget(QWidget* parent = 0);
};

// container widget
class SetupWidget : public QFrame {
  Q_OBJECT

public:
  explicit SetupWidget(QWidget* parent = 0);

private:
  QStackedWidget* mainLayout;
  PrimeAdWidget *primeAd;
  PrimeUserWidget *primeUser;
  bool showQr = false;

private slots:
  void parseError(const QString &response);
  void replyFinished(const QString &response);
  void showQrCode();
};
