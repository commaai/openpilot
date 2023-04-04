#pragma once

#include <QGridLayout>
#include <QLabel>
#include <QListWidget>
#include <QGraphicsPixmapItem>
#include <QGraphicsProxyWidget>
#include <QTimer>
#include <QUndoCommand>
#include <QUndoStack>
#include <QtCharts/QChartView>
#include <QtCharts/QLegendMarker>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
using namespace QtCharts;

const int CHART_MIN_WIDTH = 300;

enum class SeriesType {
  Line = 0,
  StepLine,
  Scatter
};

class ValueTipLabel : public QLabel {
public:
  ValueTipLabel(QWidget *parent = nullptr);
  void showText(const QPoint &pt, const QString &sec, int right_edge);
  void paintEvent(QPaintEvent *ev) override;
};

class ChartView : public QChartView {
  Q_OBJECT

public:
  ChartView(const std::pair<double, double> &x_range, QWidget *parent = nullptr);
  void addSeries(const MessageId &msg_id, const cabana::Signal *sig);
  bool hasSeries(const MessageId &msg_id, const cabana::Signal *sig) const;
  void updateSeries(const cabana::Signal *sig = nullptr);
  void updatePlot(double cur, double min, double max);
  void setSeriesType(SeriesType type);
  void updatePlotArea(int left, bool force = false);
  void showTip(double sec);
  void hideTip();

  struct SigItem {
    MessageId msg_id;
    const cabana::Signal *sig = nullptr;
    QXYSeries *series = nullptr;
    QVector<QPointF> vals;
    QVector<QPointF> step_vals;
    uint64_t last_value_mono_time = 0;
    QPointF track_pt{};
    SegmentTree segment_tree;
    double min = 0;
    double max = 0;
  };

signals:
  void seriesRemoved(const MessageId &id, const cabana::Signal *sig);
  void seriesAdded(const MessageId &id, const cabana::Signal *sig);
  void zoomIn(double min, double max);
  void zoomUndo();
  void remove();
  void axisYLabelWidthChanged(int w);
  void hovered(double sec);

private slots:
  void signalUpdated(const cabana::Signal *sig);
  void manageSeries();
  void handleMarkerClicked();
  void msgUpdated(MessageId id);
  void msgRemoved(MessageId id) { removeIf([=](auto &s) { return s.msg_id == id; }); }
  void signalRemoved(const cabana::Signal *sig) { removeIf([=](auto &s) { return s.sig == sig; }); }

private:
  void createToolButtons();
  void mousePressEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void dragMoveEvent(QDragMoveEvent *event) override;
  void dropEvent(QDropEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  QSize sizeHint() const override { return {CHART_MIN_WIDTH, settings.chart_height}; }
  void updateAxisY();
  void updateTitle();
  void resetChartCache();
  void paintEvent(QPaintEvent *event) override;
  void drawForeground(QPainter *painter, const QRectF &rect) override;
  void drawBackground(QPainter *painter, const QRectF &rect) override;
  std::tuple<double, double, int> getNiceAxisNumbers(qreal min, qreal max, int tick_count);
  qreal niceNumber(qreal x, bool ceiling);
  QXYSeries *createSeries(SeriesType type, QColor color);
  void updateSeriesPoints();
  void removeIf(std::function<bool(const SigItem &)> predicate);
  inline void clearTrackPoints() { for (auto &s : sigs) s.track_pt = {}; }

  int y_label_width = 0;
  int align_to = 0;
  QValueAxis *axis_x;
  QValueAxis *axis_y;
  QGraphicsPixmapItem *move_icon;
  QGraphicsProxyWidget *close_btn_proxy;
  QGraphicsProxyWidget *manage_btn_proxy;
  ValueTipLabel tip_label;
  QList<SigItem> sigs;
  double cur_sec = 0;
  const QString mime_type = "application/x-cabanachartview";
  SeriesType series_type = SeriesType::Line;
  bool is_scrubbing = false;
  bool resume_after_scrub = false;
  QPixmap chart_pixmap;
  double tooltip_x = -1;
  friend class ChartsWidget;
 };

class ChartsWidget : public QFrame {
  Q_OBJECT

public:
  ChartsWidget(QWidget *parent = nullptr);
  void showChart(const MessageId &id, const cabana::Signal *sig, bool show, bool merge);
  inline bool hasSignal(const MessageId &id, const cabana::Signal *sig) { return findChart(id, sig) != nullptr; }

public slots:
  void setColumnCount(int n);
  void removeAll();
  void setZoom(double min, double max);

signals:
  void dock(bool floating);
  void rangeChanged(double min, double max, bool is_zommed);
  void seriesChanged();

private:
  void resizeEvent(QResizeEvent *event) override;
  bool event(QEvent *event) override;
  void alignCharts();
  void newChart();
  ChartView *createChart();
  void removeChart(ChartView *chart);
  void eventsMerged();
  void updateState();
  void zoomIn(double min, double max);
  void zoomReset();
  void updateToolBar();
  void setMaxChartRange(int value);
  void updateLayout();
  void settingChanged();
  void showValueTip(double sec);
  bool eventFilter(QObject *obj, QEvent *event) override;
  ChartView *findChart(const MessageId &id, const cabana::Signal *sig);

  QLabel *title_label;
  QLabel *range_lb;
  LogSlider *range_slider;
  QAction *range_lb_action;
  QAction *range_slider_action;
  bool docking = true;
  QAction *dock_btn;

  QAction *undo_zoom_action;
  QAction *redo_zoom_action;
  QAction *reset_zoom_action;
  QUndoStack *zoom_undo_stack;

  QAction *remove_all_btn;
  QGridLayout *charts_layout;
  QList<ChartView *> charts;
  QWidget *charts_container;
  QScrollArea *charts_scroll;
  uint32_t max_chart_range = 0;
  bool is_zoomed = false;
  std::pair<double, double> display_range;
  std::pair<double, double> zoomed_range;
  QAction *columns_action;
  int column_count = 1;
  int current_column_count = 0;
  QTimer align_timer;
  friend class ZoomCommand;
};

class ZoomCommand : public QUndoCommand {
public:
  ZoomCommand(ChartsWidget *charts, std::pair<double, double> range) : charts(charts), range(range), QUndoCommand() {
    prev_range = charts->is_zoomed ? charts->zoomed_range : charts->display_range;
    setText(QObject::tr("Zoom to %1-%2").arg(range.first, 0, 'f', 1).arg(range.second, 0, 'f', 1));
  }
  void undo() override { charts->setZoom(prev_range.first, prev_range.second); }
  void redo() override { charts->setZoom(range.first, range.second); }
  ChartsWidget *charts;
  std::pair<double, double> prev_range, range;
};

class SeriesSelector : public QDialog {
public:
  struct ListItem : public QListWidgetItem {
    ListItem(const MessageId &msg_id, const cabana::Signal *sig, QListWidget *parent) : msg_id(msg_id), sig(sig), QListWidgetItem(parent) {}
    MessageId msg_id;
    const cabana::Signal *sig;
  };

  SeriesSelector(QString title, QWidget *parent);
  QList<ListItem *> seletedItems();
  inline void addSelected(const MessageId &id, const cabana::Signal *sig) { addItemToList(selected_list, id, sig, true); }

private:
  void updateAvailableList(int index);
  void addItemToList(QListWidget *parent, const MessageId id, const cabana::Signal *sig, bool show_msg_name = false);
  void add(QListWidgetItem *item);
  void remove(QListWidgetItem *item);

  QComboBox *msgs_combo;
  QListWidget *available_list;
  QListWidget *selected_list;
};
