#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/ui/chart/signalselector.h"
#include "tools/cabana/ui/widgets/tabbar.h"
#include "tools/cabana/commands.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/utils/util.h"

const int CHART_MIN_WIDTH = 300;

// a slider whose value is mapped onto a log10 scale
class LogSlider {
public:
  LogSlider(double factor) : scale(factor) {}

  void setRange(double min, double max) {
    scale.setRange(min, max);
    min_ = min;
    max_ = max;
    setValue(pos_);  // the raw position is re-mapped as a value
  }
  int value() const { return scale.value(pos_, minimum(), maximum()); }
  void setValue(int v) { pos_ = scale.position(v, minimum(), maximum()); }
  int minimum() const { return min_; }
  int maximum() const { return max_; }
  bool draw(const char *label, float width);

private:
  LogScale scale;
  int min_ = 0;
  int max_ = 1;
  int pos_ = 0;
};

class ChartView;
class ChartsWidget;

class ChartsContainer {
public:
  ChartsContainer(ChartsWidget *parent);
  void drawDropIndicator(const ImVec2 &pt) { drop_indictor_pos = pt; }
  void draw();  // grid layout of the current tab's charts
  void paintEvent();
  ChartView *getDropAfter(const ImVec2 &pos) const;
  ChartView *childAt(const ImVec2 &pos) const;
  int horizontalSpacing() const;

  ImRect geometry;  // screen coordinates
  ChartsWidget *charts_widget;
  ImVec2 drop_indictor_pos;
};

class ChartsWidget {
public:
  ChartsWidget();
  ~ChartsWidget();
  void draw();  // content only; MainWindow wraps it in a child region or the floating window
  void showChart(const MessageId &id, const cabana::Signal *sig, bool show, bool merge);
  inline bool hasSignal(const MessageId &id, const cabana::Signal *sig) { return findChart(id, sig) != nullptr; }
  std::vector<std::string> serializeChartIds() const;
  void restoreChartsFromIds(const std::vector<std::string> &chart_ids);
  std::string whatsThis() const;

  void setColumnCount(int n);
  void removeAll();
  void timeRangeChanged(const std::optional<std::pair<double, double>> &time_range);
  void setIsDocked(bool dock);

  Observable<> toggleChartsDocking;
  Observable<> seriesChanged;
  Observable<double> showTip;

private:
  ImVec2 minimumSizeHint() const;
  void event();
  void newChart();
  ChartView *createChart(int pos = 0);
  void removeChart(ChartView *chart);
  void splitChart(ChartView *chart);
  ImRect chartVisibleRect(ChartView *chart);
  void eventsMerged(const MessageEventsMap &new_events);
  void updateState();
  void zoomReset();
  void startChartDrag(ChartView *chart, const ImVec2 &global_pos);
  void dragChartMove(const ImVec2 &global_pos);
  void dragChartRelease(const ImVec2 &global_pos);
  void cancelChartDrag();
  bool chartDragActive() const { return drag.source != nullptr; }
  void startAutoScroll(const ImVec2 &global_pos);
  void stopAutoScroll();
  void doAutoScroll();
  void drawToolBar();
  void updateTabBar();
  void setMaxChartRange(int value);
  void updateLayout(bool force = false);
  void settingChanged();
  void showValueTip(double sec);
  void eventFilter();
  void newTab();
  void removeTab(int index);
  inline std::vector<ChartView *> &currentCharts() { return tab_charts[tabbar.tabData(tabbar.currentIndex())]; }
  ChartView *findChart(const MessageId &id, const cabana::Signal *sig);
  // draws the selector until closed, then runs `accepted` (unless `owner` was removed)
  void execSignalSelector(std::unique_ptr<SignalSelector> dlg, ChartView *owner, std::function<void(SignalSelector &)> accepted);
  void drawDragPreview();

  LogSlider range_slider{1000};
  bool is_docked = true;
  bool float_window_init_ = false;  // the floating window geometry is set once, right after undocking

  UndoStack zoom_undo_stack;

  std::vector<ChartView *> charts;
  std::unordered_map<int, std::vector<ChartView *>> tab_charts;
  TabBar tabbar;
  ChartsContainer *charts_container;
  ImGuiWindow *charts_scroll = nullptr;  // the scroll area child window
  ImRect charts_scroll_viewport;
  uint32_t max_chart_range = 0;
  std::pair<double, double> display_range;
  bool columns_action_visible = false;
  int column_count = 1;
  int current_column_count = 0;
  struct ChartDrag {
    ChartView *source = nullptr;
    ImVec2 press_pos;  // global
    bool active = false;
  } drag;
  // the drag preview is a 50% alpha copy of the whole chart tile, drawn in a window that takes no input
  ImVec2 drag_preview_pos;
  ImVec2 drag_preview_size;
  bool drag_preview_visible = false;
  ChartView *drop_target = nullptr;
  int auto_scroll_count = 0;
  ImVec2 auto_scroll_pos;
  bool auto_scroll_timer_active = false;
  double auto_scroll_timer_next = 0;
  int current_theme = 0;
  bool value_tip_visible_ = false;
  bool any_plot_hovered_ = false;
  std::vector<std::unique_ptr<ChartView>> deleted_charts_;  // freed at the start of the next draw()
  std::unique_ptr<SignalSelector> signal_selector_;
  ChartView *signal_selector_owner_ = nullptr;
  std::function<void(SignalSelector &)> signal_selector_accepted_;
  Connections connections_;
  friend class ChartView;
  friend class ChartsContainer;
};

class ZoomCommand : public UndoCommand {
public:
  ZoomCommand(std::pair<double, double> range) : range(range) {
    prev_range = can->timeRange();
  }
  void undo() override { can->setTimeRange(prev_range); }
  void redo() override { can->setTimeRange(range); }
  std::optional<std::pair<double, double>> prev_range, range;
};
