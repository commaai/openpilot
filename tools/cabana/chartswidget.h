#pragma once

#include <QLabel>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsProxyWidget>
#include <QGraphicsTextItem>
#include <QPushButton>
#include <QVBoxLayout>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

using namespace QtCharts;

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(QWidget *parent = nullptr);
  void addSignal(const QString &msg_id, const Signal *sig);
  void removeSignal(const QString &msg_id, const Signal *sig);
  void updateSeries(const Signal *sig = nullptr);
  void setEventsRange(const std::pair<double, double> &range);
  void setDisplayRange(double min, double max, bool force_update = false);
  void updateLineMarker(double current_sec);
  void updateFromSettings();
  void updateTitle();

  struct SigItem {
    QString msg_id;
    const Signal *signal = nullptr;
    QLineSeries *series = nullptr;
    QVector<QPointF> vals;
  };
  QList<SigItem> sigs;

signals:
  void zoomIn(double min, double max);
  void zoomReset();
  void remove();

private:
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void adjustChartMargins();
  void updateAxisY();

  QValueAxis *axis_x;
  QValueAxis *axis_y;
  QGraphicsItemGroup *item_group;
  QGraphicsLineItem *line_marker, *track_line;
  QGraphicsEllipseItem *track_ellipse;
  QGraphicsTextItem *value_text;
  QGraphicsProxyWidget *close_btn_proxy;
  std::pair<double, double> events_range = {0, 0};
 };

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void showChart(const QString &id, const Signal *sig, bool show, bool merge);
  void removeChart(ChartView *chart);
  void removeSignal(const Signal *sig);
  inline bool isChartOpened(const QString &id, const Signal *sig) { return findChart(id, sig) != nullptr; }

signals:
  void dock(bool floating);
  void rangeChanged(double min, double max, bool is_zommed);
  void chartOpened(const QString &id, const Signal *sig);
  void chartClosed(const QString &id, const Signal *sig);

private:
  void msgRemoved(uint32_t address);
  void msgUpdated(uint32_t address);
  void eventsMerged();
  void updateState();
  void zoomIn(double min, double max);
  void zoomReset();
  void signalUpdated(const Signal *sig);
  void updateToolBar();
  void removeAll();
  bool eventFilter(QObject *obj, QEvent *event) override;
  ChartView *findChart(const QString &id, const Signal *sig);

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
