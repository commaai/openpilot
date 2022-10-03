#pragma once

#include <QVBoxLayout>
#include <QWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <map>

#include "tools/cabana/parser.h"

class ChartsWidget : public QWidget {
  Q_OBJECT

 public:
  ChartsWidget(QWidget *parent = nullptr);
  inline bool hasChart(const QString &id, const QString &sig_name) {
    return charts.find(id+sig_name) != charts.end();
  }
  void addChart(const QString &id, const QString &sig_name);
  void removeChart(const QString &id, const QString &sig_name);
  void updateState();

 protected:
  QVBoxLayout *main_layout;
  struct SignalChart {
    QString id;
    QString sig_name;
    int max_y = 0;
    int min_y = 0;
    QList<QPointF> data;
    QtCharts::QChartView *chart_view = nullptr;
  };
  std::map<QString, SignalChart> charts;
};
