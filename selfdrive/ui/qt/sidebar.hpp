#pragma once

#include <QtWidgets>

#include <ui.hpp>

#define COLOR_GOOD QColor(255, 255, 255)
#define COLOR_WARNING QColor(218, 202, 37)
#define COLOR_DANGER QColor(201, 34, 49)

class SignalWidget : public QFrame {
  Q_OBJECT

public:
  SignalWidget(QString text, int strength, QWidget* parent = 0);
  void update(QString text, int strength);
  QLabel label;
  int _strength = 0;

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QVBoxLayout layout;

  const float _dotspace = 37; // spacing between dots
  const float _top = 10;
  const float _dia = 28; // dot diameter
};

class StatusWidget : public QFrame {
  Q_OBJECT

public:
  StatusWidget(QString label, QString msg, QColor c, QWidget* parent = 0);
  void update(QString label, QString msg, QColor c);

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QColor color = COLOR_WARNING;
  QLabel status;
  QLabel substatus;
  QVBoxLayout layout;
};

class Sidebar : public QFrame {
  Q_OBJECT

public:
  explicit Sidebar(QWidget* parent = 0);

signals:
  void openSettings();

public slots:
  void update(const UIState &s);

private:
  SignalWidget *signal;
  StatusWidget *temp;
  StatusWidget *panda;
  StatusWidget *connect;
};
