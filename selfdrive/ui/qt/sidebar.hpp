#pragma once

#include <QtWidgets>


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
  float _firstdot = 85; //1st dot x pos
  float _dotspace = 30; //spacing between dots
  float _top = 30;
  float _dia = 20; //dot diameter
};


class StatusWidget : public QFrame {
  Q_OBJECT

public:
  StatusWidget(QString text, QColor color, QWidget* parent = 0);

protected:
  void paintEvent(QPaintEvent*) override;
private:
  QColor ind_color = QColor(234,192,11); //init to yellow?
  QLabel* label;
};

class Sidebar : public QFrame {
  Q_OBJECT


public:
  explicit Sidebar(QWidget* parent = 0);
};
