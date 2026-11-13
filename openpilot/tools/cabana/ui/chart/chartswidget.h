#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/core/analysis.h"
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
  void requestCerealUpdate(bool data_changed = false);
  void requestMapUpdate(bool data_changed);
  void requestDerivedUpdate(bool data_changed);
  std::shared_ptr<std::atomic<uint64_t>> derived_generation_ = std::make_shared<std::atomic<uint64_t>>(0);
  bool derived_running_ = false, derived_dirty_ = false;
  std::shared_ptr<std::atomic<uint64_t>> map_generation_ = std::make_shared<std::atomic<uint64_t>>(0);
  bool map_running_ = false, map_dirty_ = false;
  bool cereal_running_ = false;
  bool cereal_dirty_ = false;
  void loadAnalysisLayoutFile(const std::string &path);
  void loadAnalysisLayout(const cabana::AnalysisLayout &layout);
  cabana::AnalysisLayout analysisLayout() const;
  void editPythonSeries(ChartView *owner = nullptr, const std::string &name = "Derived signal",
                        const cabana::PythonSeries &spec = {});
  void draw();  // content only; MainWindow owns the dockable panel
  size_t chartCount() const { return charts_.size(); }
  void showCerealChart(const std::string &path, bool merge = false);
  void showChart(const MessageId &id, const cabana::Signal *sig, bool show, bool merge);
  inline bool hasSignal(const MessageId &id, const cabana::Signal *sig) { return findChart(id, sig) != nullptr; }
  std::vector<std::string> serializeChartIds() const;
  void restoreChartsFromIds(const std::vector<std::string> &chart_ids);
  std::string whatsThis() const;

  void setColumnCount(int n);
  void removeAll();

  Observable<> seriesChanged;
  Observable<double> showTip;

private:
  void drawPythonEditor();
  std::map<std::string, cabana::PythonSeries> derived_definitions_;
  bool changing_layout_ = false;
  std::map<int, std::string> tab_names_;
  std::map<int, cabana::LayoutNode> tab_roots_;
  uint64_t layout_generation_ = 0;
  void remapLayoutTree(int id, const std::vector<ChartView *> &before);
  bool python_editor_open_ = false;
  ChartView *python_owner_ = nullptr;
  std::string python_name_, python_inputs_, python_error_;
  cabana::PythonSeries python_spec_;
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
  std::shared_ptr<std::atomic<uint64_t>> cereal_generation_ = std::make_shared<std::atomic<uint64_t>>(0);
  std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
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
