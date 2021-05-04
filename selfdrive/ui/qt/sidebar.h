#pragma once

#include <QtWidgets>

#include <ui.h>

#define COLOR_GOOD QColor(255, 255, 255)
#define COLOR_WARNING QColor(218, 202, 37)
#define COLOR_DANGER QColor(201, 34, 49)

class SignalWidget : public QFrame {
  Q_OBJECT

public:
  SignalWidget(QWidget* parent = 0);
  void update(const QString &text, int strength);
  QLabel *label;
  int _strength = 0;

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QVBoxLayout *layout;

  const float _dotspace = 37; // spacing between dots
  const float _top = 10;
  const float _dia = 28; // dot diameter
};

class StatusWidget : public QFrame {
  Q_OBJECT

public:
  StatusWidget(bool has_substatus, QWidget* parent = 0);
  void update(const QString &label, const QColor &c, const QString &msg = "");

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QLabel *status;
  QLabel *substatus = nullptr;
  QColor color = COLOR_WARNING;
  QVBoxLayout *layout;
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
