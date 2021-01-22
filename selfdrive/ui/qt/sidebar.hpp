#pragma once

#include <QtWidgets>


class Indicator : public QWidget {
  Q_OBJECT

public:
  Indicator(QWidget* parent = 0);
  QGraphicsDropShadowEffect *dropshadow;

protected:
  void paintEvent(QPaintEvent*) override;
};



class StatusWidget : public QFrame {
  Q_OBJECT

public:
  StatusWidget(QString text, QWidget* parent = 0);
  Indicator* indicator;
  QLabel* label;

protected:
  void paintEvent(QPaintEvent*) override;
};

class Sidebar : public QFrame {
  Q_OBJECT


public:
  explicit Sidebar(QWidget* parent = 0);

// private:
//   QTimer* timer;

//   // offroad home screen widgets
//   QLabel* date;
//   QStackedLayout* center_layout;
//   OffroadAlert* alerts_widget;
//   QPushButton* alert_notification;

// public slots:
//   void closeAlerts();
//   void openAlerts();
//   void refresh();
};
