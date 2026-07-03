#pragma once

// Shared state/decls for the charts panel, split across charts_panel.cc
// (toolbar, grid layout, charts_show_signal/is_showing API, event wiring),
// chart_view.cc (per-chart ImPlot rendering, data decimation, mouse
// interaction, zoom) and signal_selector.cc (the "add signals" modal).
//
// Parity spec (frozen Qt reference): tools/cabana/chart/chartswidget.{h,cc},
// chart.{h,cc}, signalselector.{h,cc}, tiplabel.{h,cc}.

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "imgui.h"

#include "tools/cabana/commands.h"
#include "tools/cabana/imgui/app.h"

constexpr int MAX_COLUMN_COUNT = 4;         // chartswidget.cc
constexpr float CHART_MIN_WIDTH = 300.0f;   // chartswidget.h
constexpr double MIN_ZOOM_SECONDS = 0.01;   // chart.cc

// mirrors chart.h SeriesType, kept as plain ints so it can be stored
// directly in settings.chart_series_type without a cast at every use.
enum SeriesType { SERIES_LINE = 0, SERIES_STEP_LINE = 1, SERIES_SCATTER = 2 };

// One plotted signal within a chart. Raw (xs, ys) hold every decoded sample
// currently known for the signal, scoped to what's been fetched via
// eventsInRange()/appended via eventsMerged() -- see the data-lifecycle
// comment block in chart_view.cc for exactly when each happens.
struct SigItem {
  MessageId msg_id;
  const cabana::Signal *sig = nullptr;
  ColorRGBA display_color{};  // sig->color with chart-local collision avoidance (mirrors ChartView::setSeriesColor)
  bool visible = true;        // legend click toggle (mirrors QXYSeries::isVisible())
  std::vector<double> xs;     // seconds, ascending
  std::vector<double> ys;
  double min = 0.0;  // over the currently-visible decimated window; refreshed every draw
  double max = 0.0;
};

struct ChartState {
  int id = 0;  // stable identity for ImGui IDs + selector "manage" target (vector index can shift)
  std::vector<SigItem> sigs;
  int series_type = SERIES_LINE;  // per-chart override; mirrors ChartView::series_type

  // Plot geometry from the most recent frame this chart was drawn, kept only
  // for the "cursor drag continues even if mouse leaves the plot rect"
  // bookkeeping performed inline in chart_view.cc while the chart's own
  // BeginPlot/EndPlot block is active (see draw_one_chart()).
};

// Modal "add/manage signals" dialog state (signalselector.cc, ported).
struct SelectorState {
  bool open_request = false;              // set by open_signal_selector(), consumed once via ImGui::OpenPopup
  bool visible = false;                   // true while the modal popup is open
  std::string title;
  int target_chart_id = -1;               // -1 => "New Chart": accept creates a chart; else "Manage": accept replaces that chart's signals
  std::vector<std::pair<MessageId, const cabana::Signal *>> selected;  // right pane, in add-order
  int chosen_msg_combo = -1;              // index into the combo's message list (left pane)
};

struct ChartsState {
  bool wired = false;
  std::vector<ChartState> charts;
  int next_chart_id = 1;

  int column_count = 1;
  int max_chart_range = 180;  // seconds; mirrors ChartsWidget::max_chart_range (settings.chart_range)
  std::pair<double, double> display_range{0.0, 60.0};

  UndoStack zoom_stack;  // OWN instance -- not UndoStack::instance() (that's the DBC-edit undo stack)

  // Mouse interaction, at most one active at a time across all charts.
  int drag_chart_id = -1;   // box-zoom rubber band in progress on this chart
  double drag_start_t = 0.0;
  int scrub_chart_id = -1;  // shift-drag scrub in progress on this chart
  bool scrub_resume_after = false;

  // Combined value tooltip (spec: one shared readout, not N synced floating
  // labels like Qt's per-chart TipLabel -- see report).
  std::optional<double> hover_time;

  // Deferred per-frame actions (avoid mutating `charts` mid-iteration).
  int pending_close_id = -1;
  int pending_split_id = -1;

  SelectorState selector;
};

extern ChartsState g_charts;

// charts_panel.cc
void ensure_charts_wired();
std::pair<double, double> chart_effective_range();
ChartState *find_chart_by_id(int id);
ChartState &create_chart();
void close_chart(int chart_id);
void remove_signals_if(ChartState &chart, const std::function<bool(const SigItem &)> &pred);
void add_signal_to_chart(ChartState &chart, const MessageId &id, const cabana::Signal *sig);
void split_chart(int chart_idx);
void request_manage_chart(int chart_id);
void request_new_chart();

// chart_view.cc
void rebuild_signal(SigItem &s, const std::pair<double, double> &range);
void draw_one_chart(AppState &app, ChartState &chart, float width, int chart_index);
void draw_hover_tooltip();  // combined value readout across all charts at g_charts.hover_time

// signal_selector.cc
void draw_signal_selector_modal();
