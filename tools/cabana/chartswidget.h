#pragma once

#include <map>

#include <QLabel>
#include <QGraphicsLineItem>
#include <QGraphicsSimpleTextItem>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QtCharts/QChartView>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

using namespace QtCharts;

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(const QString &id, const Signal *sig, QWidget *parent = nullptr);
  void updateSeries(const std::pair<double, double> &range);
  void setRange(double min, double max, bool force_update = false);
  void updateLineMarker(double current_sec);

signals:
  void zoomIn(double min, double max);
  void zoomReset();

private:
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void enterEvent(QEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void adjustChartMargins();

  void updateAxisY();

  QGraphicsLineItem *track_line;
  QGraphicsSimpleTextItem *value_text;
  QGraphicsLineItem *line_marker;
  QList<QPointF> vals;
  QString id;
  const Signal *signal;
};

class ChartWidget : public QWidget {
Q_OBJECT

public:
  ChartWidget(const QString &id, const Signal *sig, QWidget *parent);
  void updateTitle();

signals:
  void remove(const QString &msg_id, const Signal *sig);

public:
  QString id;
  const Signal *signal;
  QLabel *msg_name_label;
  QLabel *sig_name_label;
  ChartView *chart_view = nullptr;
};

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void addChart(const QString &id, const Signal *sig);
  void removeChart(ChartWidget *chart);

signals:
  void dock(bool floating);
  void rangeChanged(double min, double max, bool is_zommed);

private:
  void eventsMerged();
  void updateState();
  void zoomIn(double min, double max);
  void zoomReset();
  void signalUpdated(const Signal *sig);
  void updateTitleBar();
  void removeAll(const Signal *sig = nullptr);
  bool eventFilter(QObject *obj, QEvent *event) override;

  QWidget *title_bar;
  QLabel *title_label;
  QLabel *range_label;
  bool docking = true;
  QPushButton *dock_btn;
  QPushButton *reset_zoom_btn;
  QPushButton *remove_all_btn;
  QVBoxLayout *charts_layout;
  QList<ChartWidget *> charts;

  bool is_zoomed = false;
  std::pair<double, double> event_range;
  std::pair<double, double> display_range;
  std::pair<double, double> zoomed_range;
};
