#pragma once

#include <QFrame>
#include <QStackedWidget>

struct Alert {
  QString text;
  int severity;
};

class OffroadAlert : public QFrame {
  Q_OBJECT

public:
  explicit OffroadAlert(QWidget *parent = 0);
  QVector<Alert> alerts;
  bool updateAvailable;

private:
  QStackedWidget *alerts_stack;
  void parse_alerts();

signals:
  void closeAlerts();

public slots:
  void refresh();
};
