#pragma once

#include <unordered_map>
#include <utility>

#include <QGridLayout>
#include <QLabel>
#include <QScrollArea>
#include <QTimer>
#include <QToolBar>
#include <QUndoCommand>
#include <QUndoStack>

#include "tools/cabana/chart/signalselector.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

const int CHART_MIN_WIDTH = 300;
const QString CHART_MIME_TYPE = "application/x-cabanachartview";

class ChartView;
class ChartsWidget;

class ChartsContainer : public QWidget {
public:
  ChartsContainer(ChartsWidget *parent);
  void dragEnterEvent(QDragEnterEvent *event) override;
  void dropEvent(QDropEvent *event) override;
  void dragLeaveEvent(QDragLeaveEvent *event) override { drawDropIndicator({}); }
  void drawDropIndicator(const QPoint &pt) { drop_indictor_pos = pt; update(); }
  void paintEvent(QPaintEvent *ev) override;
  ChartView *getDropAfter(const QPoint &pos) const;

  QGridLayout *charts_layout;
  ChartsWidget *charts_widget;
  QPoint drop_indictor_pos;
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
  void timeRangeChanged(const std::optional<std::pair<double, double>> &time_range);
  void setIsDocked(bool dock);

signals:
  void toggleChartsDocking();
  void seriesChanged();
  void showTip(double seconds);

private:
  QSize minimumSizeHint() const override;
  bool event(QEvent *event) override;
  void alignCharts();
  void newChart();
  ChartView *createChart(int pos = 0);
  void removeChart(ChartView *chart);
  void splitChart(ChartView *chart);
  QRect chartVisibleRect(ChartView *chart);
  void eventsMerged(const MessageEventsMap &new_events);
  void updateState();
  void zoomReset();
  void startAutoScroll();
  void stopAutoScroll();
  void doAutoScroll();
  void updateToolBar();
  void updateTabBar();
  void setMaxChartRange(int value);
  void updateLayout(bool force = false);
  void settingChanged();
  void showValueTip(double sec);
  bool eventFilter(QObject *obj, QEvent *event) override;
  void newTab();
  void removeTab(int index);
  inline QList<ChartView *> &currentCharts() { return tab_charts[tabbar->tabData(tabbar->currentIndex()).toInt()]; }
  ChartView *findChart(const MessageId &id, const cabana::Signal *sig);

  QLabel *title_label;
  QLabel *range_lb;
  LogSlider *range_slider;
  QAction *range_lb_action;
  QAction *range_slider_action;
  bool is_docked = true;
  ToolButton *dock_btn;

  QToolBar *toolbar;
  QAction *undo_zoom_action;
  QAction *redo_zoom_action;
  QAction *reset_zoom_action;
  ToolButton *reset_zoom_btn;
  QUndoStack *zoom_undo_stack;

  ToolButton *remove_all_btn;
  QList<ChartView *> charts;
  std::unordered_map<int, QList<ChartView *>> tab_charts;
  TabBar *tabbar;
  ChartsContainer *charts_container;
  QScrollArea *charts_scroll;
  uint32_t max_chart_range = 0;
  std::pair<double, double> display_range;
  QAction *columns_action;
  int column_count = 1;
  int current_column_count = 0;
  int auto_scroll_count = 0;
  QTimer *auto_scroll_timer;
  QTimer *align_timer;
  int current_theme = 0;
  bool value_tip_visible_ = false;
  friend class ChartView;
  friend class ChartsContainer;
};

class ZoomCommand : public QUndoCommand {
public:
  ZoomCommand(std::pair<double, double> range) : range(range), QUndoCommand() {
    prev_range = can->timeRange();
    setText(QObject::tr("Zoom to %1-%2").arg(range.first, 0, 'f', 2).arg(range.second, 0, 'f', 2));
  }
  void undo() override { can->setTimeRange(prev_range); }
  void redo() override { can->setTimeRange(range); }
  std::optional<std::pair<double, double>> prev_range, range;
};
