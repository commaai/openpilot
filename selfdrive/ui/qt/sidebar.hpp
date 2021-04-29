#pragma once

#include <QFrame>
#include <QLabel>

#include <ui.hpp>

class SignalWidget : public QFrame {
  Q_OBJECT

public:
  SignalWidget(QString text, int strength, QWidget* parent = 0);
  void update(QString text, int strength);
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
  void update(QString label, QString msg, int severity);

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QColor _severity = QColor(218, 202, 37);
  QLabel* l_label;
  QLabel* l_msg;
};

class Sidebar : public QFrame {
  Q_OBJECT

public:
  explicit Sidebar(QWidget* parent = 0);

public slots:
  void update(UIState *s);

private:
  SignalWidget *signal;
  StatusWidget *temp;
  StatusWidget *vehicle;
  StatusWidget *connect;
};
