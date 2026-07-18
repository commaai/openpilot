#pragma once

#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include <QMenu>

#include "tools/cabana/chart/tiplabel.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

enum class SeriesType {
  Line = 0,
  StepLine,
  Scatter
};

class ChartsWidget;
class ChartView : public QWidget {
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
  double secondsAtPoint(const QPointF &pt) const {
    return x_min + (pt.x() - plot_area.left()) * (x_max - x_min) / std::max(plot_area.width(), 1);
  }

  struct SigItem {
    MessageId msg_id;
    const cabana::Signal *sig = nullptr;
    QColor color;
    bool visible = true;
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
  void msgUpdated(MessageId id);
  void msgRemoved(MessageId id) { removeIf([=](auto &s) { return s.msg_id.address == id.address && !dbc()->msg(id); }); }
  void signalRemoved(const cabana::Signal *sig) { removeIf([=](auto &s) { return s.sig == sig; }); }

private:
  void appendCanEvents(const cabana::Signal *sig, const std::vector<const CanEvent *> &events,
                       std::vector<QPointF> &vals, std::vector<QPointF> &step_vals);
  void createToolButtons();
  void contextMenuEvent(QContextMenuEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  QSize sizeHint() const override;
  void updateAxisY();
  void updateTitle();
  void resetChartCache();
  void paintEvent(QPaintEvent *event) override;
  void drawStaticLayer(QPainter *painter);
  void drawAxes(QPainter *painter);
  void drawLegend(QPainter *painter);
  void drawSeries(QPainter *painter);
  void drawForeground(QPainter *painter);
  void drawSignalValue(QPainter *painter);
  void drawTimeline(QPainter *painter);
  void drawRubberBandTimeRange(QPainter *painter);
  int xAxisPrecision() const;
  std::tuple<double, double, int> getNiceAxisNumbers(qreal min, qreal max, int tick_count);
  qreal niceNumber(qreal x, bool ceiling);
  QColor uniqueColor(QColor color, const cabana::Signal *exclude = nullptr) const;
  void removeIf(std::function<bool(const SigItem &)> predicate);
  void takeSignalsFrom(ChartView *source);
  void setDropHighlight(bool highlight) { if (std::exchange(can_drop, highlight) != highlight) update(); }
  inline void clearTrackPoints() { for (auto &s : sigs) s.track_pt = {}; }
  inline qreal xPos(double sec) const { return plot_area.left() + (sec - x_min) / (x_max - x_min) * plot_area.width(); }
  inline qreal yPos(double val) const { return plot_area.bottom() - (val - y_min) / (y_max - y_min) * plot_area.height(); }

  // layout
  QRect plot_area;
  QRect move_icon_rect;
  std::vector<QRect> legend_rects;
  // axes
  double x_min;
  double x_max;
  double y_min = 0;
  double y_max = 1;
  int y_tick_count = 3;
  int y_precision = 0;
  QString y_unit;
  int y_label_width = 0;
  int align_to = 0;
  // interaction
  enum class MouseMode { None, Rubber, Scrub };
  MouseMode mouse_mode = MouseMode::None;
  QPoint press_pos;
  QRect rubber_rect;
  bool resume_after_scrub = false;

  QMenu *menu;
  QAction *split_chart_act;
  QAction *close_act;
  ToolButton *manage_btn;
  ToolButton *close_btn;
  TipLabel *tip_label;
  std::vector<SigItem> sigs;
  double cur_sec = 0;
  SeriesType series_type = SeriesType::Line;
  QPixmap chart_pixmap;
  bool can_drop = false;
  double tooltip_x = -1;
  QFont signal_value_font;
  ChartsWidget *charts_widget;
  friend class ChartsWidget;
};
