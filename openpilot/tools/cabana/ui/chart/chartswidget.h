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
#include "tools/cabana/lua/lua_evaluator.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/utils/util.h"

const int CHART_MIN_WIDTH = 300;

// a slider whose value is mapped onto a log10 scale
class LogSlider {
public:
  LogSlider(double factor) : scale_(factor) {}

  void setRange(double min, double max) {
    scale_.setRange(min, max);
    min_ = min;
    max_ = max;
    setValue(pos_);  // the raw position is re-mapped as a value
  }
  int value() const { return scale_.value(pos_, minimum(), maximum()); }
  void setValue(int v) { pos_ = scale_.position(v, minimum(), maximum()); }
  int minimum() const { return min_; }
  int maximum() const { return max_; }
  bool draw(const char *label, float width);

private:
  LogScale scale_;
  int min_ = 0;
  int max_ = 1;
  int pos_ = 0;
};

class ChartView;
class ChartsWidget;

class ChartsContainer {
public:
  ChartsContainer(ChartsWidget *parent) : charts_widget_(parent) {}
  void setDropIndicator(const ImVec2 &pt) { drop_indicator_pos_ = pt; }
  void draw();  // grid layout of the current tab's charts
  ChartView *getDropAfter(const ImVec2 &pos) const;
  ChartView *childAt(const ImVec2 &pos) const;
  const ImRect &geometry() const { return geometry_; }  // screen coordinates

private:
  void drawDropIndicator();

  ImRect geometry_;
  ChartsWidget *charts_widget_;
  ImVec2 drop_indicator_pos_;
};

class ChartsWidget {
public:
  ChartsWidget();
  ~ChartsWidget();  // out of line: the header users only see a forward declared ChartView
  void draw();  // content only; MainWindow owns the dockable panel
  size_t chartCount() const { return charts_.size(); }
  void showChart(const MessageId &id, const cabana::Signal *sig, bool show, bool merge);
  inline bool hasSignal(const MessageId &id, const cabana::Signal *sig) { return findChart(id, sig) != nullptr; }
  void showCerealSignal(const std::string &path, bool merge);
  std::vector<std::string> serializeChartIds() const;
  void restoreChartsFromIds(const std::vector<std::string> &chart_ids);
  bool loadLayout(const std::string &path);  // returns false + logs error on failure
  void saveLayout(const std::string &path);
  std::string whatsThis() const;

  void setColumnCount(int n);
  void removeAll();

  Observable<> seriesChanged;
  Observable<double> showTip;

private:
  void handleEvents();  // the back button, focus loss, the chart drag and the value tip leave
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
  bool chartDragActive() const { return drag_.source != nullptr; }
  void startAutoScroll(const ImVec2 &global_pos);
  void stopAutoScroll();
  void doAutoScroll();
  void drawToolBar();
  void updateTabBar();
  void setMaxChartRange(int value);
  void updateLayout();
  void settingChanged();
  void showValueTip(double sec);
  void newTab();
  void removeTab(int index);
  inline std::vector<ChartView *> &currentCharts() { return tab_charts_[tabbar_.tabData(tabbar_.currentIndex())]; }
  ChartView *findChart(const MessageId &id, const cabana::Signal *sig);
  // draws the selector until closed, then runs `accepted` (unless `owner` was removed)
  void execSignalSelector(std::unique_ptr<SignalSelector> dlg, ChartView *owner, std::function<void(SignalSelector &)> accepted);
  void drawDragPreview();

  LogSlider range_slider_{1000};

  UndoStack zoom_undo_stack_;

  std::vector<std::unique_ptr<ChartView>> charts_;
  std::unordered_map<int, std::vector<ChartView *>> tab_charts_;
  std::unordered_map<int, std::string> tab_names_;  // custom names set by loadLayout
  TabBar tabbar_;
  ChartsContainer charts_container_{this};
  ImGuiWindow *charts_scroll_ = nullptr;  // the scroll area child window
  ImRect charts_scroll_viewport_;
  int max_chart_range_ = 0;
  std::pair<double, double> display_range_;
  bool columns_action_visible_ = false;
  int column_count_ = 1;
  int current_column_count_ = 0;
  struct ChartDrag {
    ChartView *source = nullptr;
    ImVec2 press_pos;  // global
    bool active = false;
  } drag_;
  // the drag preview is a 50% alpha copy of the whole chart tile, drawn in a window that takes no input
  ImVec2 drag_preview_pos_;
  ImVec2 drag_preview_size_;
  bool drag_preview_visible_ = false;
  ChartView *drop_target_ = nullptr;
  int auto_scroll_count_ = 0;
  ImVec2 auto_scroll_pos_;
  bool auto_scroll_timer_active_ = false;
  double auto_scroll_timer_next_ = 0;
  bool value_tip_visible_ = false;
  bool any_plot_hovered_ = false;
  std::vector<std::unique_ptr<ChartView>> deleted_charts_;  // freed at the start of the next draw()
  std::unique_ptr<SignalSelector> signal_selector_;
  ChartView *signal_selector_owner_ = nullptr;
  std::function<void(SignalSelector &)> signal_selector_accepted_;
  // cereal signal browser state
  bool cereal_browser_open_ = false;
  char cereal_filter_buf_[256] = {};
  std::vector<std::string> cereal_filtered_paths_;
  void drawCerealBrowser();
  void cerealSeriesUpdated();
  // GPS map panel
  bool map_panel_open_ = false;
  void drawMapPanel();
  // Events log panel
  bool log_panel_open_ = false;
  char log_filter_buf_[256] = {};
  void drawLogPanel();
  // Lua derived signal panel state
  bool lua_panel_open_ = false;
  bool lua_evaluating_ = false;  // re-entrancy guard for cerealSeriesUpdated
  std::vector<LuaSnippet> lua_snippets_;
  int lua_selected_ = -1;
  char lua_name_buf_[256] = {};
  char lua_linked_buf_[256] = {};
  char lua_v1_buf_[256] = {};
  char lua_v2_buf_[256] = {};
  char lua_v3_buf_[256] = {};
  char lua_v4_buf_[256] = {};
  char lua_global_buf_[2048] = {};
  char lua_func_buf_[4096] = {};
  std::string lua_error_;
  std::string lua_success_;
  void drawLuaPanel();
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
