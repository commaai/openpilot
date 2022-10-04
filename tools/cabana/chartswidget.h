#pragma once

#include <map>

#include <QLabel>
#include <QVBoxLayout>
#include <QWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>

#include "tools/cabana/parser.h"

using namespace QtCharts;

class LineMarker : public QWidget {
Q_OBJECT

public:
  LineMarker(const QString &id, const QString &sig_name, QChart *chart, QWidget *parent);
  void updateSigData();
  void paintEvent(QPaintEvent *event) override;
  QString id;
  QString sig_name;
  QChart *chart;
  std::pair<double, double> x_range;
};

class ChartWidget : public QWidget {
Q_OBJECT

public:
  ChartWidget(const QString &id, const QString &sig_name, QWidget *parent);
  inline QChart *chart() const { return chart_view->chart(); }

signals:
  void remove(const QString &id, const QString &sig_name);

protected:
  void updateState();
  void addData(const CanData &can_data, const Signal &sig);
  void updateSeries();

  QString id;
  QString sig_name;
  QLabel *zoom_label;
  QChartView *chart_view = nullptr;
  LineMarker *line_marker = nullptr;
  QList<QPointF> vals;
};

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
  std::map<QString, ChartWidget *> charts;
};
