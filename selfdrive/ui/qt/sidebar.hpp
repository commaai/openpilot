#pragma once

#include <QtWidgets>

#define QT_COLOR_WHITE QColor(255, 255, 255)
#define QT_COLOR_YELLOW QColor(218, 202, 37)
#define QT_COLOR_RED QColor(201, 34, 49)

class SettingsBtn : public QAbstractButton {
  Q_OBJECT

public:
  SettingsBtn(QWidget* parent = 0);
protected:
  void paintEvent(QPaintEvent*) override;
private:
  QImage image;
};

class SignalWidget : public QFrame {
  Q_OBJECT

public:
  SignalWidget(QString text, int strength, QWidget* parent = 0);
  QLabel* label;
  int _strength = 0;

protected:
  void paintEvent(QPaintEvent*) override;

private:
  float _dotspace = 37; //spacing between dots
  float _top = 10;
  float _dia = 28; //dot diameter
};


class StatusWidget : public QFrame {
  Q_OBJECT

public:
  StatusWidget(QString label, QString msg, QColor color, QWidget* parent = 0);

protected:
  void paintEvent(QPaintEvent*) override;
private:
  QColor _severity = QT_COLOR_YELLOW;
  QLabel* l_label;
  QLabel* l_msg;
};

class Sidebar : public QFrame {
  Q_OBJECT


public:
  explicit Sidebar(QWidget* parent = 0);
};
