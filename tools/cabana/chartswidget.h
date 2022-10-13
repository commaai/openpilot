#pragma once

#include <map>

#include <QLabel>
#include <QGraphicsLineItem>
#include <QGraphicsSimpleTextItem>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

using namespace QtCharts;

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(const QString &id, const QString &sig_name, QWidget *parent = nullptr);

private:
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void enterEvent(QEvent *event) override;
  void leaveEvent(QEvent *event) override;

  void updateSeries();
  void rangeChanged(qreal min, qreal max);
  void updateAxisY();
  void updateState();

  QGraphicsLineItem *track_line;
  QGraphicsSimpleTextItem *value_text;
  QGraphicsLineItem *line_marker;
  QList<QPointF> vals;
  QString id;
  QString sig_name;
};

class ChartWidget : public QWidget {
Q_OBJECT

public:
  ChartWidget(const QString &id, const QString &sig_name, QWidget *parent);
  inline QChart *chart() const { return chart_view->chart(); }

signals:
  void remove();

protected:
  QString id;
  QString sig_name;
  ChartView *chart_view = nullptr;
};

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void addChart(const QString &id, const QString &sig_name);
  void removeChart(const QString &id, const QString &sig_name);
  inline bool hasChart(const QString &id, const QString &sig_name) {
    return charts.find(id + sig_name) != charts.end();
  }

signals:
  void dock(bool floating);

private:
  void updateState();
  void updateTitleBar();
  void removeAll();
  bool eventFilter(QObject *obj, QEvent *event);

  QWidget *title_bar;
  QLabel *title_label;
  QLabel *range_label;
  bool docking = true;
  QPushButton *dock_btn;
  QPushButton *reset_zoom_btn;
  QPushButton *remove_all_btn;
  QVBoxLayout *charts_layout;
  std::map<QString, ChartWidget *> charts;
};
