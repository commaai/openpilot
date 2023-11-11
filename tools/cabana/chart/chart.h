#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include <QMenu>
#include <QGraphicsPixmapItem>
#include <QGraphicsProxyWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QLegendMarker>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>
using namespace QtCharts;

#include "tools/cabana/chart/tiplabel.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

enum class SeriesType {
  Line = 0,
  StepLine,
  Scatter
};

class ChartsWidget;
class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(const std::pair<double, double> &x_range, ChartsWidget *parent = nullptr);
  void addSignal(const MessageId &msg_id, const cabana::Signal *sig);
  bool hasSignal(const MessageId &msg_id, const cabana::Signal *sig) const;
  void updateSeries(const cabana::Signal *sig = nullptr, const MessageEventsMap *msg_new_events = nullptr);
  void updatePlot(double cur, double min, double max);
  void setSeriesType(SeriesType type);
  void updatePlotArea(int left, bool force = false);
  void showTip(double sec);
  void hideTip();
  void startAnimation();

  struct SigItem {
    MessageId msg_id;
    const cabana::Signal *sig = nullptr;
    QXYSeries *series = nullptr;
    std::vector<QPointF> vals;
    std::vector<QPointF> step_vals;
    QPointF track_pt{};
    SegmentTree segment_tree;
    double min = 0;
    double max = 0;
  };

signals:
  void axisYLabelWidthChanged(int w);

private slots:
  void signalUpdated(const cabana::Signal *sig);
  void manageSignals();
  void handleMarkerClicked();
  void msgUpdated(MessageId id);
  void msgRemoved(MessageId id) { removeIf([=](auto &s) { return s.msg_id.address == id.address && !dbc()->msg(id); }); }
  void signalRemoved(const cabana::Signal *sig) { removeIf([=](auto &s) { return s.sig == sig; }); }

private:
  void appendCanEvents(const cabana::Signal *sig, const std::vector<const CanEvent *> &events,
                       std::vector<QPointF> &vals, std::vector<QPointF> &step_vals);
  void createToolButtons();
  void addSeries(QXYSeries *series);
  void contextMenuEvent(QContextMenuEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void dragEnterEvent(QDragEnterEvent *event) override;
  void dragLeaveEvent(QDragLeaveEvent *event) override { drawDropIndicator(false); }
  void dragMoveEvent(QDragMoveEvent *event) override;
  void dropEvent(QDropEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  QSize sizeHint() const override;
  void updateAxisY();
  void updateTitle();
  void resetChartCache();
  void setTheme(QChart::ChartTheme theme);
  void paintEvent(QPaintEvent *event) override;
  void drawForeground(QPainter *painter, const QRectF &rect) override;
  void drawBackground(QPainter *painter, const QRectF &rect) override;
  void drawDropIndicator(bool draw) { if (std::exchange(can_drop, draw) != can_drop) viewport()->update(); }
  void drawSignalValue(QPainter *painter);
  void drawTimeline(QPainter *painter);
  void drawRubberBandTimeRange(QPainter *painter);
  std::tuple<double, double, int> getNiceAxisNumbers(qreal min, qreal max, int tick_count);
  qreal niceNumber(qreal x, bool ceiling);
  QXYSeries *createSeries(SeriesType type, QColor color);
  void setSeriesColor(QXYSeries *, QColor color);
  void updateSeriesPoints();
  void removeIf(std::function<bool(const SigItem &)> predicate);
  inline void clearTrackPoints() { for (auto &s : sigs) s.track_pt = {}; }

  int y_label_width = 0;
  int align_to = 0;
  QValueAxis *axis_x;
  QValueAxis *axis_y;
  QMenu *menu;
  QAction *split_chart_act;
  QAction *close_act;
  QGraphicsPixmapItem *move_icon;
  QGraphicsProxyWidget *close_btn_proxy;
  QGraphicsProxyWidget *manage_btn_proxy;
  TipLabel *tip_label;
  std::vector<SigItem> sigs;
  double cur_sec = 0;
  SeriesType series_type = SeriesType::Line;
  bool is_scrubbing = false;
  bool resume_after_scrub = false;
  QPixmap chart_pixmap;
  bool can_drop = false;
  double tooltip_x = -1;
  QFont signal_value_font;
  ChartsWidget *charts_widget;
  friend class ChartsWidget;
};
