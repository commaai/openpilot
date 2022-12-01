#pragma once

#include <QLabel>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsProxyWidget>
#include <QGraphicsTextItem>
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
  ~ChartView();
  void addSeries(const QString &msg_id, const Signal *sig);
  void removeSeries(const QString &msg_id, const Signal *sig);
  bool hasSeries(const QString &msg_id, const Signal *sig) const;
  void updateSeries(const Signal *sig = nullptr);
  void setEventsRange(const std::pair<double, double> &range);
  void setDisplayRange(double min, double max);

  struct SigItem {
    QString msg_id;
    uint8_t source = 0;
    uint32_t address = 0;
    const Signal *sig = nullptr;
    QLineSeries *series = nullptr;
    double min_y = 0;
    double max_y = 0;
    QVector<QPointF> vals;
  };

signals:
  void seriesRemoved(const QString &id, const Signal *sig);
  void zoomIn(double min, double max);
  void zoomReset();
  void remove();

private slots:
  void msgRemoved(uint32_t address);
  void msgUpdated(uint32_t address);
  void signalUpdated(const Signal *sig);
  void signalRemoved(const Signal *sig);

private:
  QList<ChartView::SigItem>::iterator removeSeries(const QList<ChartView::SigItem>::iterator &it);
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void adjustChartMargins();
  void updateAxisY();
  void updateTitle();
  void updateFromSettings();
  void drawForeground(QPainter *painter, const QRectF &rect) override;

  QValueAxis *axis_x;
  QValueAxis *axis_y;
  QGraphicsItemGroup *item_group;
  QGraphicsLineItem *track_line;
  QGraphicsEllipseItem *track_ellipse;
  QGraphicsTextItem *value_text;
  QGraphicsProxyWidget *close_btn_proxy;
  std::pair<double, double> events_range = {0, 0};
  QList<SigItem> sigs;
 };

class ChartsWidget : public QWidget {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void showChart(const QString &id, const Signal *sig, bool show, bool merge);
  void removeChart(ChartView *chart);
  inline bool isChartOpened(const QString &id, const Signal *sig) { return findChart(id, sig) != nullptr; }

signals:
  void dock(bool floating);
  void rangeChanged(double min, double max, bool is_zommed);
  void chartOpened(const QString &id, const Signal *sig);
  void chartClosed(const QString &id, const Signal *sig);

private:
  void eventsMerged();
  void updateState();
  void updateDisplayRange();
  void zoomIn(double min, double max);
  void zoomReset();
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
  bool use_dark_theme = false;
};
