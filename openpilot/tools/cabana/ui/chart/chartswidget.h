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
#include "tools/cabana/commands.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/utils/util.h"

const int CHART_MIN_WIDTH = 300;

// qtutil.h LogSlider: a QSlider whose value is mapped onto a log10 scale
class LogSlider {
public:
  LogSlider(double factor) : scale(factor) {}

  void setRange(double min, double max) {
    scale.setRange(min, max);
    min_ = min;
    max_ = max;
    setValue(pos_);  // Qt: setValue(QSlider::value()), the raw position is re-mapped as a value
  }
  int value() const { return scale.value(pos_, minimum(), maximum()); }
  void setValue(int v) { pos_ = scale.position(v, minimum(), maximum()); }
  int minimum() const { return min_; }
  int maximum() const { return max_; }
  bool draw(const char *label, float width);  // valueChanged

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
  void updateToolBar();
  void drawToolBar();
  void updateTabBar();
  void drawTabBar();
  void setMaxChartRange(int value);
  void updateLayout(bool force = false);
  void settingChanged();
  void showValueTip(double sec);
  void eventFilter();
  void newTab();
  void removeTab(int index);
  inline std::vector<ChartView *> &currentCharts() { return tab_charts[tabs_[current_tab_index_].id]; }
  ChartView *findChart(const MessageId &id, const cabana::Signal *sig);
  // QDialog::exec() replacement: draws the selector until closed, then runs `accepted` (unless `owner` was removed)
  void execSignalSelector(std::unique_ptr<SignalSelector> dlg, ChartView *owner, std::function<void(SignalSelector &)> accepted);
  void drawDragPreview();

  std::string title_label;
  std::string range_lb;
  LogSlider range_slider{1000};
  bool range_lb_visible = true;
  bool range_slider_visible = true;
  bool is_docked = true;
  bool float_window_init_ = false;  // the floating window geometry is set once, right after undocking
  const char *dock_btn_icon = "";
  std::string dock_btn_tooltip;

  bool undo_zoom_enabled = false;
  bool redo_zoom_enabled = false;
  bool undo_zoom_visible = false;
  bool redo_zoom_visible = false;
  bool reset_zoom_visible = false;
  std::string reset_zoom_text;
  UndoStack zoom_undo_stack;

  bool remove_all_enabled = false;
  std::vector<ChartView *> charts;
  std::unordered_map<int, std::vector<ChartView *>> tab_charts;
  struct Tab {
    int id;
    std::string text;
    ImRect rect;  // last drawn tab item, for tabAt()
  };
  std::vector<Tab> tabs_;  // tabbar
  int current_tab_index_ = 0;
  int pending_tab_index_ = -1;
  ChartsContainer *charts_container;
  ImGuiWindow *charts_scroll = nullptr;  // the scroll area child window
  ImRect charts_scroll_viewport;
  uint32_t max_chart_range = 0;
  std::pair<double, double> display_range;
  std::string columns_action_text;
  bool columns_action_visible = false;
  int column_count = 1;
  int current_column_count = 0;
  struct ChartDrag {
    ChartView *source = nullptr;
    ImVec2 press_pos;  // global
    bool active = false;
  } drag;
  // Qt drags a 50% alpha snapshot of the whole chart tile: the tile is re-rendered into a window with no input
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
  std::vector<std::unique_ptr<ChartView>> deleted_charts_;  // deleteLater: freed at the start of the next draw()
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
