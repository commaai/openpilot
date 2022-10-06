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
  LineMarker(QWidget *parent) : QWidget(parent) {}
  void setX(double x);

private:
  void paintEvent(QPaintEvent *event) override;
  double x_pos = 0.0;
};

class ChartWidget : public QWidget {
Q_OBJECT

public:
  ChartWidget(const QString &id, const QString &sig_name, QWidget *parent);
  inline QChart *chart() const { return chart_view->chart(); }

private:
  void updateState();
  void addData(const CanData &can_data, const Signal &sig);
  void updateSeries();
  void rangeChanged(qreal min, qreal max);

  QString id;
  QString sig_name;
  QChartView *chart_view = nullptr;
  LineMarker *line_marker = nullptr;
  double line_marker_x = 0.0;
  QList<QPointF> vals;
};

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void addChart(const QString &id, const QString &sig_name);
  void removeChart(const QString &id, const QString &sig_name);
  void removeAll();
  inline bool hasChart(const QString &id, const QString &sig_name) {
    return charts.find(id + sig_name) != charts.end();
  }

signals:
  void dock(bool floating);

private:
  void updateState();
  void updateDockButton();

  QWidget *title_bar;
  QLabel *title_label;
  bool docking = true;
  QPushButton *dock_btn;
  QPushButton *reset_zoom_btn;
  QPushButton *remove_all_btn;
  QVBoxLayout *charts_layout;
  std::map<QString, ChartWidget *> charts;
};
