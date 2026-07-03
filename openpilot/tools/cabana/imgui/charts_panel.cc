// ImGui port of tools/cabana/chart/chartswidget.{h,cc} (ChartsWidget), the
// frozen Qt reference. This file owns the toolbar, the scrolling chart grid,
// event wiring (DBC/stream), and the charts_show_signal()/charts_is_showing()
// contract declared in app.h. Per-chart rendering lives in chart_view.cc, the
// "add/manage signals" modal in signal_selector.cc -- see charts_internal.h
// for the shared state they operate on.
//
// Known, deliberate scope cuts vs the Qt reference (see full list in the
// comment block at the bottom of this file):
//  - No multi-tab charts (ChartsWidget's TabBar) -- a single flat chart list.
//  - No drag-and-drop chart reordering between grid cells.
//  - No floating/undocked charts window (dock_btn) -- explicitly SKIPPED per
//    the task spec: the imgui panel is already dockable.
#include "tools/cabana/imgui/charts_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "implot.h"

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/settings.h"

ChartsState g_charts;

namespace {

constexpr const char *kSeriesTypeLabels[] = {"Line", "Step Line", "Scatter"};

std::string format_range_seconds(int sec) {
  char buf[32];
  if (sec >= 3600) {
    std::snprintf(buf, sizeof(buf), "%d:%02d:%02d", sec / 3600, (sec % 3600) / 60, sec % 60);
  } else {
    std::snprintf(buf, sizeof(buf), "%02d:%02d", sec / 60, sec % 60);
  }
  return buf;
}

// -- data lifecycle: full rebuild triggers (mirrors chart.cc's dbc()/can-> connections) --

void rebuild_all(const std::pair<double, double> &range) {
  for (ChartState &c : g_charts.charts) {
    for (SigItem &s : c.sigs) rebuild_signal(s, range);
  }
}

void remove_empty_charts() {
  for (auto it = g_charts.charts.begin(); it != g_charts.charts.end();) {
    if (it->sigs.empty()) {
      it = g_charts.charts.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace

// -- shared helpers (used by chart_view.cc / signal_selector.cc) ------------

std::pair<double, double> chart_effective_range() {
  return can->timeRange().value_or(g_charts.display_range);
}

ChartState *find_chart_by_id(int id) {
  for (ChartState &c : g_charts.charts) {
    if (c.id == id) return &c;
  }
  return nullptr;
}

ChartState &create_chart() {
  ChartState c;
  c.id = g_charts.next_chart_id++;
  c.series_type = settings.chart_series_type;
  g_charts.charts.push_back(std::move(c));
  return g_charts.charts.back();
}

void close_chart(int chart_id) {
  for (auto it = g_charts.charts.begin(); it != g_charts.charts.end(); ++it) {
    if (it->id == chart_id) {
      g_charts.charts.erase(it);
      return;
    }
  }
}

void remove_signals_if(ChartState &chart, const std::function<bool(const SigItem &)> &pred) {
  auto &sigs = chart.sigs;
  sigs.erase(std::remove_if(sigs.begin(), sigs.end(), pred), sigs.end());
}

// mirrors ChartView::setSeriesColor(): nudge the hue away from same-chart
// collisions so overlapping signal colors stay visually distinguishable.
namespace {
float rgb_hue(const ColorRGBA &c) {
  float r = c.r / 255.0f, g = c.g / 255.0f, b = c.b / 255.0f;
  float mx = std::max({r, g, b}), mn = std::min({r, g, b}), d = mx - mn;
  if (d < 1e-6f) return 0.0f;
  float h;
  if (mx == r) h = std::fmod((g - b) / d, 6.0f);
  else if (mx == g) h = (b - r) / d + 2.0f;
  else h = (r - g) / d + 4.0f;
  h *= 60.0f;
  if (h < 0.0f) h += 360.0f;
  return h / 360.0f;
}

ColorRGBA hsv_to_rgba(float h, float s, float v) {
  float r, g, b;
  ImGui::ColorConvertHSVtoRGB(h, s, v, r, g, b);
  return ColorRGBA{static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f)),
                    static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f)),
                    static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f)), 255};
}

ColorRGBA pick_display_color(const ChartState &chart, const ColorRGBA &base) {
  for (const SigItem &s : chart.sigs) {
    if (std::fabs(rgb_hue(base) - rgb_hue(s.display_color)) < 0.1f) {
      float last_h = rgb_hue(chart.sigs.back().display_color);
      float new_h = std::fmod(last_h + 60.0f / 360.0f, 1.0f);
      float sat = 0.35f + (std::rand() % 66) / 100.0f;
      float val = 0.85f + (std::rand() % 16) / 100.0f;
      return hsv_to_rgba(new_h, sat, val);
    }
  }
  return base;
}
}  // namespace

void add_signal_to_chart(ChartState &chart, const MessageId &id, const cabana::Signal *sig) {
  for (const SigItem &s : chart.sigs) {
    if (s.msg_id == id && s.sig == sig) return;  // hasSignal()
  }
  SigItem item;
  item.msg_id = id;
  item.sig = sig;
  item.display_color = pick_display_color(chart, sig->color);
  rebuild_signal(item, chart_effective_range());
  chart.sigs.push_back(std::move(item));
}

void split_chart(int chart_idx) {
  if (chart_idx < 0 || chart_idx >= static_cast<int>(g_charts.charts.size())) return;
  ChartState &src = g_charts.charts[chart_idx];
  if (src.sigs.size() <= 1) return;

  std::vector<SigItem> extracted(std::make_move_iterator(src.sigs.begin() + 1), std::make_move_iterator(src.sigs.end()));
  const int series_type = src.series_type;
  src.sigs.resize(1);

  int insert_pos = chart_idx + 1;
  for (SigItem &item : extracted) {
    ChartState c;
    c.id = g_charts.next_chart_id++;
    c.series_type = series_type;
    c.sigs.push_back(std::move(item));
    g_charts.charts.insert(g_charts.charts.begin() + insert_pos, std::move(c));
    ++insert_pos;
  }
}

void request_manage_chart(int chart_id) {
  SelectorState &sel = g_charts.selector;
  sel.title = "Manage Chart";
  sel.target_chart_id = chart_id;
  sel.selected.clear();
  if (ChartState *c = find_chart_by_id(chart_id)) {
    for (const SigItem &s : c->sigs) sel.selected.emplace_back(s.msg_id, s.sig);
  }
  sel.chosen_msg_combo = -1;
  sel.open_request = true;
}

void request_new_chart() {
  SelectorState &sel = g_charts.selector;
  sel.title = "New Chart";
  sel.target_chart_id = -1;
  sel.selected.clear();
  sel.chosen_msg_combo = -1;
  sel.open_request = true;
}

// -- event wiring (once) -----------------------------------------------------

void ensure_charts_wired() {
  if (g_charts.wired) return;
  g_charts.wired = true;

  g_charts.column_count = std::clamp(settings.chart_column_count, 1, MAX_COLUMN_COUNT);
  g_charts.max_chart_range = std::clamp(settings.chart_range, 1, std::max(1, settings.max_cached_minutes * 60));
  g_charts.display_range = {can->minSeconds(), can->minSeconds() + g_charts.max_chart_range};

  // mirrors QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &ChartsWidget::removeAll);
  dbc()->DBCFileChanged.connect([]() {
    g_charts.charts.clear();
    g_charts.zoom_stack.clear();
    can->setTimeRange(std::nullopt);
  });
  // mirrors ChartView::signalRemoved()
  dbc()->signalRemoved.connect([](const cabana::Signal *sig) {
    for (ChartState &c : g_charts.charts) {
      remove_signals_if(c, [&](const SigItem &s) { return s.sig == sig; });
    }
    remove_empty_charts();
  });
  // mirrors ChartView::msgRemoved()
  dbc()->msgRemoved.connect([](MessageId id) {
    for (ChartState &c : g_charts.charts) {
      remove_signals_if(c, [&](const SigItem &s) { return s.msg_id.address == id.address && dbc()->msg(id) == nullptr; });
    }
    remove_empty_charts();
  });
  // mirrors ChartView::signalUpdated(): recolor + full rebuild (factor/offset/etc may have changed)
  dbc()->signalUpdated.connect([](const cabana::Signal *sig) {
    const auto range = chart_effective_range();
    for (ChartState &c : g_charts.charts) {
      for (SigItem &s : c.sigs) {
        if (s.sig == sig) {
          s.display_color = pick_display_color(c, sig->color);
          rebuild_signal(s, range);
        }
      }
    }
  });
  // mirrors ChartsWidget::eventsMerged(): incremental append, no rebuild
  can->eventsMerged.connect([](const MessageEventsMap &new_events) {
    for (ChartState &c : g_charts.charts) {
      for (SigItem &s : c.sigs) {
        auto it = new_events.find(s.msg_id);
        if (it == new_events.end() || it->second.empty()) continue;
        for (const CanEvent *e : it->second) {
          double v = 0.0;
          if (!s.sig->getValue(e->dat, e->size, &v)) continue;
          const double t = can->toSeconds(e->mono_time);
          if (s.xs.empty() || t >= s.xs.back()) {
            s.xs.push_back(t);
            s.ys.push_back(v);
          } else {
            auto pos = std::lower_bound(s.xs.begin(), s.xs.end(), t);
            const size_t idx = static_cast<size_t>(pos - s.xs.begin());
            s.xs.insert(pos, t);
            s.ys.insert(s.ys.begin() + idx, v);
          }
        }
      }
    }
  });
  // mirrors ChartsWidget's seekedTo-driven refresh: full rebuild scoped to
  // the (freshly recomputed) display range at the new position.
  can->seekedTo.connect([](double) {
    if (!g_charts.charts.empty()) {
      const double cur_sec = can->currentSec();
      double pos = (cur_sec - g_charts.display_range.first) / std::max<double>(1.0, g_charts.max_chart_range);
      if (pos < 0 || pos > 0.8) {
        g_charts.display_range.first = std::max(can->minSeconds(), cur_sec - g_charts.max_chart_range * 0.1);
      }
      double max_sec = std::min(g_charts.display_range.first + g_charts.max_chart_range, can->maxSeconds());
      g_charts.display_range.first = std::max(can->minSeconds(), max_sec - g_charts.max_chart_range);
      g_charts.display_range.second = g_charts.display_range.first + g_charts.max_chart_range;
    }
    rebuild_all(chart_effective_range());
  });
}

// -- CABANA_TEST_CHARTS debug hook (see report) ------------------------------
// Format: "src:addr:signal[,src:addr:signal...]" e.g. "0:1D0:WHEEL_SPEED_FL".
// Applied once, at the first draw, so headless/interactive verification can
// populate the panel without needing the (separately owned) signal editor's
// plot buttons to exist yet.
void apply_test_hook_once() {
  static bool applied = false;
  if (applied) return;
  applied = true;
  const char *env = std::getenv("CABANA_TEST_CHARTS");
  if (env == nullptr || *env == '\0') return;

  std::string s(env);
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t comma = s.find(',', pos);
    const std::string token = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    const size_t c1 = token.find(':');
    const size_t c2 = (c1 == std::string::npos) ? std::string::npos : token.find(':', c1 + 1);
    if (c1 != std::string::npos && c2 != std::string::npos) {
      const MessageId id = MessageId::fromString(token.substr(0, c2));
      const std::string sig_name = token.substr(c2 + 1);
      if (cabana::Msg *m = dbc()->msg(id); m != nullptr) {
        if (cabana::Signal *sig = m->sig(sig_name); sig != nullptr) {
          charts_show_signal(id, sig, true);
        }
      }
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
}

// -- toolbar: mirrors ChartsWidget::createToolBar()/updateToolBar() ---------

namespace {

int max_feasible_columns(float avail_w) {
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  int n = MAX_COLUMN_COUNT;
  for (; n > 1; --n) {
    if (static_cast<float>(n) * CHART_MIN_WIDTH + static_cast<float>(n - 1) * spacing < avail_w) break;
  }
  return n;
}

void draw_toolbar(float avail_w) {
  ImGui::Text("Charts: %d", static_cast<int>(g_charts.charts.size()));
  ImGui::SameLine();
  if (ImGui::Button("New Chart")) request_new_chart();

  ImGui::SameLine();
  ImGui::SetNextItemWidth(120.0f);
  if (ImGui::BeginCombo("##chart_type", (std::string("Type: ") + kSeriesTypeLabels[settings.chart_series_type]).c_str())) {
    for (int i = 0; i < 3; ++i) {
      if (ImGui::Selectable(kSeriesTypeLabels[i], i == settings.chart_series_type)) {
        settings.chart_series_type = i;
        for (ChartState &c : g_charts.charts) c.series_type = i;
      }
    }
    ImGui::EndCombo();
  }

  const int max_cols = max_feasible_columns(avail_w);
  if (max_cols > 1) {
    ImGui::SameLine();
    g_charts.column_count = std::clamp(g_charts.column_count, 1, max_cols);
    ImGui::SetNextItemWidth(110.0f);
    if (ImGui::BeginCombo("##columns", ("Columns: " + std::to_string(g_charts.column_count)).c_str())) {
      for (int i = 1; i <= MAX_COLUMN_COUNT; ++i) {
        if (ImGui::Selectable(std::to_string(i).c_str(), i == g_charts.column_count)) {
          g_charts.column_count = i;
          settings.chart_column_count = i;
        }
      }
      ImGui::EndCombo();
    }
  }

  const bool is_zoomed = can->timeRange().has_value();
  ImGui::SameLine();
  if (!is_zoomed) {
    // range slider: settings.chart_range within [1, max_cached_minutes*60], log scale like Qt's LogSlider
    ImGui::TextUnformatted(format_range_seconds(g_charts.max_chart_range).c_str());
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150.0f);
    const int max_range = std::max(1, settings.max_cached_minutes * 60);
    if (ImGui::SliderInt("##chart_range", &g_charts.max_chart_range, 1, max_range, "", ImGuiSliderFlags_Logarithmic)) {
      g_charts.max_chart_range = std::clamp(g_charts.max_chart_range, 1, max_range);
      settings.chart_range = g_charts.max_chart_range;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Set the chart range");
  } else {
    const auto &range = *can->timeRange();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f-%.2f", range.first, range.second);
    ImGui::BeginDisabled(!g_charts.zoom_stack.canUndo());
    if (ImGui::Button("Undo Zoom")) g_charts.zoom_stack.undo();
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!g_charts.zoom_stack.canRedo());
    if (ImGui::Button("Redo Zoom")) g_charts.zoom_stack.redo();
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(buf)) {
      can->setTimeRange(std::nullopt);
      g_charts.zoom_stack.clear();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset Zoom");
  }

  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("Remove All").x -
                                                                    ImGui::GetStyle().FramePadding.x * 2.0f));
  ImGui::BeginDisabled(g_charts.charts.empty());
  if (ImGui::Button("Remove All")) {
    g_charts.charts.clear();
    can->setTimeRange(std::nullopt);
    g_charts.zoom_stack.clear();
  }
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove all charts");
}

void draw_grid(AppState &app) {
  if (g_charts.charts.empty()) {
    ImGui::Spacing();
    ImGui::TextDisabled("No charts. Click \"New Chart\" or plot a signal from the signal editor.");
    return;
  }

  const float avail_w = ImGui::GetContentRegionAvail().x;
  const int cols = std::clamp(g_charts.column_count, 1, max_feasible_columns(avail_w));
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  const float chart_w = (avail_w - spacing * static_cast<float>(cols - 1)) / static_cast<float>(cols);

  // ImPlot's built-in Pan/Select gestures would fight our own manual
  // box-zoom/scrub/seek mouse handling (see chart_view.cc): each chart's
  // axes carry ImPlotAxisFlags_Lock (below) so pan/box-select can never
  // mutate the range we set explicitly every frame, and ImPlotFlags_NoBoxSelect
  // keeps the selection rectangle from ever drawing. (An earlier version of
  // this tried to disable Pan/Select via ImPlot::GetInputMap() using an
  // out-of-range ImGuiMouseButton_COUNT sentinel -- that crashes ImPlot's
  // GetMouseDragDelta()/IsMouseDragging(), which index MouseDown[] with the
  // mapped button with no bounds check. Axis locking avoids the input map
  // entirely.)
  g_charts.hover_time.reset();
  for (int i = 0; i < static_cast<int>(g_charts.charts.size()); ++i) {
    draw_one_chart(app, g_charts.charts[static_cast<size_t>(i)], chart_w, i);
    if ((i + 1) % cols != 0 && i + 1 < static_cast<int>(g_charts.charts.size())) ImGui::SameLine();
  }

  draw_hover_tooltip();

  if (g_charts.pending_close_id >= 0) {
    close_chart(g_charts.pending_close_id);
    g_charts.pending_close_id = -1;
  }
  if (g_charts.pending_split_id >= 0) {
    for (int i = 0; i < static_cast<int>(g_charts.charts.size()); ++i) {
      if (g_charts.charts[static_cast<size_t>(i)].id == g_charts.pending_split_id) {
        split_chart(i);
        break;
      }
    }
    g_charts.pending_split_id = -1;
  }
}

}  // namespace

void draw_charts_panel(AppState &app) {
  ensure_charts_wired();
  apply_test_hook_once();

  if (!g_charts.charts.empty() && !can->timeRange().has_value()) {
    const double cur_sec = can->currentSec();
    double pos = (cur_sec - g_charts.display_range.first) / std::max<double>(1.0, g_charts.max_chart_range);
    if (pos < 0 || pos > 0.8) {
      g_charts.display_range.first = std::max(can->minSeconds(), cur_sec - g_charts.max_chart_range * 0.1);
    }
    double max_sec = std::min(g_charts.display_range.first + g_charts.max_chart_range, can->maxSeconds());
    g_charts.display_range.first = std::max(can->minSeconds(), max_sec - g_charts.max_chart_range);
    g_charts.display_range.second = g_charts.display_range.first + g_charts.max_chart_range;
  }

  if (ImGui::Begin(CHARTS_WINDOW_TITLE)) {
    draw_toolbar(ImGui::GetContentRegionAvail().x);
    ImGui::Separator();
    if (ImGui::BeginChild("##charts_scroll", ImGui::GetContentRegionAvail())) {
      draw_grid(app);
    }
    ImGui::EndChild();
  }
  ImGui::End();

  draw_signal_selector_modal();
}

void charts_show_signal(const MessageId &id, const cabana::Signal *sig, bool show) {
  ensure_charts_wired();
  ChartState *existing = nullptr;
  for (ChartState &c : g_charts.charts) {
    for (const SigItem &s : c.sigs) {
      if (s.msg_id == id && s.sig == sig) {
        existing = &c;
        break;
      }
    }
    if (existing != nullptr) break;
  }

  if (show) {
    if (existing != nullptr) return;
    // showChart(): a plotted signal always lands in a NEW chart (merge=false).
    ChartState &c = create_chart();
    add_signal_to_chart(c, id, sig);
  } else if (existing != nullptr) {
    remove_signals_if(*existing, [&](const SigItem &s) { return s.msg_id == id && s.sig == sig; });
    remove_empty_charts();
  }
}

bool charts_is_showing(const MessageId &id, const cabana::Signal *sig) {
  for (const ChartState &c : g_charts.charts) {
    for (const SigItem &s : c.sigs) {
      if (s.msg_id == id && s.sig == sig) return true;
    }
  }
  return false;
}

// Implemented vs tools/cabana/chart/chartswidget.cc (ChartsWidget), frozen
// Qt reference:
//  - Toolbar: chart count label, New Chart, global series-type combo
//    (applies to all charts + becomes the default for new ones), columns
//    combo (hidden below 2 feasible columns, exactly mirroring
//    updateLayout()'s show_column_cb), range slider (log scale via
//    ImGuiSliderFlags_Logarithmic, replacing Qt's custom LogSlider) shown
//    only when not zoomed, undo/redo/reset-zoom shown only when zoomed
//    (reset button's label becomes "min-max" like Qt's reset_zoom_btn),
//    Remove All (disabled when empty).
//  - "show all values" tooltip toggle: NOT present in the Qt reference
//    (chartswidget.{h,cc} has no such control) -- skipped as instructed.
//  - Dock/undock button: SKIPPED per the task spec (imgui panel is already
//    dockable).
//  - Grid layout: settings.chart_column_count columns, scrolling region,
//    CHART_MIN_WIDTH-driven column shrink.
//  - Data lifecycle: eventsMerged incrementally appends (no rebuild) during
//    normal playback; seekedTo/DBCFileChanged/signalUpdated/signalRemoved/
//    msgRemoved trigger the appropriate rebuild/removal, wired once here.
//  - charts_show_signal()/charts_is_showing(): showChart() semantics --
//    every new signal lands in a brand-new chart (Qt's merge=true path is
//    only reachable via Shift+click in signalview.cc, which app.h's
//    charts_show_signal(id, sig, show) signature has no way to convey, so
//    it's dropped; every call behaves like Qt's merge=false).
//
// Deliberate scope cuts (see also chart_view.cc / signal_selector.cc):
//  - No multi-tab charts (TabBar/newTab/removeTab) -- one flat chart list.
//    Nothing in the task's enumerated toolbar spec calls for tabs.
//  - No drag-and-drop chart reordering or cross-chart signal drag/drop
//    (ChartsContainer's DnD, ChartView::mousePressEvent's move_icon drag).
//    "Manage Signals" (per chart) and "Split Chart" cover the same reachable
//    end states (moving a signal to a different/new chart) without needing
//    pixel-level DnD plumbing in an immediate-mode grid.
//  - No auto-scroll-while-dragging (startAutoScroll/doAutoScroll): only
//    relevant to the DnD reordering this port doesn't implement.
//  - settings.changed does NOT reset every chart's height/series-type
//    (ChartsWidget::settingChanged() does this on *any* settings change,
//    including unrelated ones like multiple_lines_hex -- this reads like an
//    accidental Qt over-broad connection rather than intentional behavior,
//    and reproducing it would let e.g. toggling a Messages-panel checkbox
//    silently clobber a user's per-chart series-type override). The
//    equivalent *intentional* paths (global Type combo, max_cached_minutes
//    clamp) are still implemented directly.
