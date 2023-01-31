#pragma once

#include <QLabel>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsProxyWidget>
#include <QGraphicsTextItem>
#include <QPushButton>
#include <QVBoxLayout>
#include <QtCharts/QChartView>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

using namespace QtCharts;

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(const QString &id, const Signal *sig, QWidget *parent = nullptr);
  void updateSeries(const std::pair<double, double> range);
  void setRange(double min, double max, bool force_update = false);
  void updateLineMarker(double current_sec);
  void updateFromSettings();
  void updateTitle();

  QString id;
  const Signal *signal;

signals:
  void zoomIn(double min, double max);
  void zoomReset();
  void remove(const QString &msg_id, const Signal *sig);

private:
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void adjustChartMargins();
  void updateAxisY();

  QGraphicsItemGroup *item_group;
  QGraphicsLineItem *line_marker, *track_line;
  QGraphicsEllipseItem *track_ellipse;
  QGraphicsTextItem *value_text, *msg_title;
  QGraphicsProxyWidget *close_btn_proxy;
  QVector<QPointF> vals;
 };

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void showChart(const QString &id, const Signal *sig, bool show);
  void removeChart(ChartView *chart);
  bool isChartOpened(const QString &id, const Signal *sig);

signals:
  void dock(bool floating);
  void rangeChanged(double min, double max, bool is_zommed);
  void chartOpened(const QString &id, const Signal *sig);
  void chartClosed(const QString &id, const Signal *sig);

private:
  void eventsMerged();
  void updateState();
  void zoomIn(double min, double max);
  void zoomReset();
  void signalUpdated(const Signal *sig);
  void updateToolBar();
  void removeAll(const Signal *sig = nullptr);
  bool eventFilter(QObject *obj, QEvent *event) override;

  QLabel *title_label;
  QLabel *range_label;
  bool docking = true;
  QAction *dock_btn;
  QAction *reset_zoom_btn;
  QAction *remove_all_btn;
  QVBoxLayout *charts_layout;
  QList<ChartView *> charts;
  bool is_zoomed = false;
  std::pair<double, double> event_range;
  std::pair<double, double> display_range;
  std::pair<double, double> zoomed_range;
};
