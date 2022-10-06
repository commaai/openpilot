#pragma once

#include <map>

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>

#include "tools/cabana/parser.h"

using namespace QtCharts;

class LineMarker : public QWidget {
Q_OBJECT

public:
  LineMarker(QChart *chart, QWidget *parent);
  void paintEvent(QPaintEvent *event) override;

private:
  QChart *chart;
};

class ChartWidget : public QWidget {
Q_OBJECT

public:
  ChartWidget(const QString &id, const QString &sig_name, QWidget *parent);
  inline QChart *chart() const { return chart_view->chart(); }

protected:
  void updateState();
  void addData(const CanData &can_data, const Signal &sig);
  void updateSeries();
  void rangeChanged(qreal min, qreal max);

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
  void removeAll();
  void updateState();

signals:
  void floatingCharts(bool floating);

protected:
  void updateFloatButton();

  QWidget *title_bar;
  bool floating = false;
  QPushButton *floating_btn;
  QVBoxLayout *charts_layout;
  std::map<QString, ChartWidget *> charts;
};
