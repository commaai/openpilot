#pragma once


#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QCheckBox>
#include <QStackedLayout>
#include <QPushButton>

struct Alert{
  QString text;
  int severity;
};

class OffroadAlert : public QWidget{
  Q_OBJECT

public:
  explicit OffroadAlert(QWidget *parent = 0);
  bool show_alert;
  QVector<Alert> alerts;

private:
  QVBoxLayout *vlayout;

  void parse_alerts();

signals:
  void closeAlerts();

public slots:
  void refresh();
};
