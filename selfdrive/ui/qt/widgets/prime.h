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
  void paintEvent(QPaintEvent*) override;

private:
  QPixmap img;
  QTimer *timer;
  void updateQrCode(const QString &text);
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;

private slots:
  void refresh();
};


// pairing popup widget
class PairingPopup : public DialogBase {
  Q_OBJECT

public:
  explicit PairingPopup(QWidget* parent);
  int exec() override;
};


// widget for paired users with prime
class PrimeUserWidget : public QFrame {
  Q_OBJECT

public:
  explicit PrimeUserWidget(QWidget* parent = 0);
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

signals:
  void openSettings(int index = 0, const QString &param = "");

private:
  PairingPopup *popup;
  QStackedWidget *mainLayout;
};
