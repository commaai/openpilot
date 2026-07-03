// Per-chart rendering: ImPlot series, decimation, y-axis autofit, mouse
// interaction (box-zoom / shift-scrub / click-seek / right-click-undo),
// legend + live value readout, and the combined hover tooltip.
//
// Ported from tools/cabana/chart/chart.cc (ChartView), the frozen Qt
// reference, adapted to ImPlot's immediate-mode plotting instead of
// QtCharts' retained QXYSeries scene graph. Envelope decimation follows
// tools/jotpluggler/plot.cc's app_decimate_samples_impl() (min/max bucket
// decimation) rather than Qt's SegmentTree -- see the report for why.
#include "tools/cabana/imgui/charts_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "implot.h"

#include "tools/cabana/settings.h"

// -- data lifecycle -----------------------------------------------------
//
//  - rebuild_signal(): full re-decode via can->eventsInRange(msg_id, range),
//    used when a signal is added to a chart and on every full-rebuild
//    trigger wired in charts_panel.cc's ensure_charts_wired() (seekedTo,
//    DBCFileChanged [charts are wiped instead], signalUpdated).
//  - Incremental append during normal playback happens in charts_panel.cc's
//    eventsMerged handler directly (no per-signal function needed there --
//    it's a tight loop over already-decoded new_events).
//
// Raw (xs, ys) are intentionally NOT capped to the current display width --
// they hold every sample since the last rebuild, so zooming into an already
// fetched sub-range never needs a refetch. Decimation for plotting happens
// fresh every frame from whatever window is on screen (see
// decimate_visible()); this mirrors jotpluggler's draw_plot() exactly and
// avoids the cache-invalidation bookkeeping Qt's resetChartCache()/
// SegmentTree needed.
void rebuild_signal(SigItem &s, const std::pair<double, double> &range) {
  s.xs.clear();
  s.ys.clear();
  auto [first, last] = can->eventsInRange(s.msg_id, range);
  s.xs.reserve(static_cast<size_t>(std::distance(first, last)));
  s.ys.reserve(static_cast<size_t>(std::distance(first, last)));
  double v = 0.0;
  for (auto it = first; it != last; ++it) {
    const CanEvent *e = *it;
    if (s.sig->getValue(e->dat, e->size, &v)) {
      s.xs.push_back(can->toSeconds(e->mono_time));
      s.ys.push_back(v);
    }
  }
}

namespace {

// mirrors tools/cabana/commands.h UndoCommand + chartswidget.h's ZoomCommand,
// but pushed onto g_charts.zoom_stack (OUR OWN UndoStack instance), never
// UndoStack::instance() (that stack is for DBC edits).
class ChartZoomCommand : public UndoCommand {
public:
  explicit ChartZoomCommand(std::pair<double, double> range) : range_(range) {
    prev_range_ = can->timeRange();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Zoom to %.2f-%.2f", range.first, range.second);
    setText(buf);
  }
  void undo() override { can->setTimeRange(prev_range_); }
  void redo() override { can->setTimeRange(range_); }

private:
  std::optional<std::pair<double, double>> prev_range_, range_;
};

// -- envelope decimation: port of tools/jotpluggler/plot.cc
// app_decimate_samples_impl() (min/max bucket decimation), operating on an
// [begin,end) window of already-sorted (xs, ys) rather than a full vector.
void decimate_window(const std::vector<double> &xs, const std::vector<double> &ys, size_t begin, size_t end,
                      int max_points, std::vector<double> *xs_out, std::vector<double> *ys_out) {
  xs_out->clear();
  ys_out->clear();
  const size_t n = end - begin;
  if (n == 0) return;
  if (max_points <= 0 || static_cast<int>(n) <= max_points) {
    xs_out->assign(xs.begin() + static_cast<long>(begin), xs.begin() + static_cast<long>(end));
    ys_out->assign(ys.begin() + static_cast<long>(begin), ys.begin() + static_cast<long>(end));
    return;
  }

  const size_t bucket_count = std::max<size_t>(1, static_cast<size_t>(max_points / 4));
  const size_t bucket_size = std::max<size_t>(1, static_cast<size_t>(std::ceil(static_cast<double>(n) / static_cast<double>(bucket_count))));
  xs_out->reserve(bucket_count * 4 + 2);
  ys_out->reserve(bucket_count * 4 + 2);

  size_t last_index = std::numeric_limits<size_t>::max();
  auto append_index = [&](size_t index) {
    if (index >= end || index == last_index) return;
    xs_out->push_back(xs[index]);
    ys_out->push_back(ys[index]);
    last_index = index;
  };

  for (size_t start = begin; start < end; start += bucket_size) {
    const size_t bucket_end = std::min(end, start + bucket_size);
    size_t min_index = start, max_index = start;
    for (size_t i = start + 1; i < bucket_end; ++i) {
      if (ys[i] < ys[min_index]) min_index = i;
      if (ys[i] > ys[max_index]) max_index = i;
    }
    std::array<size_t, 4> idx = {start, min_index, max_index, bucket_end - 1};
    std::sort(idx.begin(), idx.end());
    for (size_t i : idx) append_index(i);
  }
}

// mirrors ChartView::niceNumber()/getNiceAxisNumbers(): round the y-axis
// bounds to a "nice" 1/2/5 * 10^n step so labels don't show arbitrary
// padded floats. Ported verbatim (same math), only the types changed.
double nice_number(double x, bool ceiling) {
  if (!(x > 0.0)) return x <= 0.0 ? 1e-6 : x;
  const double z = std::pow(10.0, std::floor(std::log10(x)));
  double q = x / z;
  if (ceiling) {
    if (q <= 1.0) q = 1;
    else if (q <= 2.0) q = 2;
    else if (q <= 5.0) q = 5;
    else q = 10;
  } else {
    if (q < 1.5) q = 1;
    else if (q < 3.0) q = 2;
    else if (q < 7.0) q = 5;
    else q = 10;
  }
  return q * z;
}

struct NiceAxis {
  double min, max;
  int tick_count;
};

NiceAxis get_nice_axis_numbers(double min, double max, int tick_count) {
  const double range = nice_number(max - min, true);
  const double step = nice_number(range / static_cast<double>(tick_count - 1), false);
  const double nmin = std::floor(min / step);
  const double nmax = std::ceil(max / step);
  return {nmin * step, nmax * step, static_cast<int>(nmax - nmin) + 1};
}

// index range in `xs` covering [lo, hi] (mirrors chart.cc's xLessThan lower_bound pattern)
std::pair<size_t, size_t> window_indices(const std::vector<double> &xs, double lo, double hi) {
  auto first = std::lower_bound(xs.begin(), xs.end(), lo);
  auto last = std::upper_bound(first, xs.end(), hi);
  return {static_cast<size_t>(first - xs.begin()), static_cast<size_t>(last - xs.begin())};
}

// value at-or-before `t` (mirrors chart.cc's reverse lower_bound pattern used
// by both showTip() and drawSignalValue())
std::optional<double> value_at_or_before(const std::vector<double> &xs, const std::vector<double> &ys, double t) {
  if (xs.empty() || t < xs.front()) return std::nullopt;
  auto it = std::upper_bound(xs.begin(), xs.end(), t);
  if (it == xs.begin()) return std::nullopt;
  const size_t idx = static_cast<size_t>((it - xs.begin()) - 1);
  return ys[idx];
}

std::string legend_label(const SigItem &s) {
  return s.sig->name + "  " + msgName(s.msg_id) + " " + s.msg_id.toString();
}

}  // namespace

// mirrors ChartView::drawSignalValue()/legend text + the "remove-series (x)"
// requirement from the task spec (Qt only offers per-series removal via the
// Manage Signals dialog; this adds a direct inline control).
static void draw_legend_row(ChartState &chart, SigItem &s, int idx, double cur_sec) {
  ImGui::PushID(idx);
  // Captured BEFORE any item on this row: ImGui::SameLine(x) takes an offset
  // from the line's start, not from the current cursor -- using
  // GetContentRegionAvail() *after* advancing (as an earlier version of this
  // did) measures the wrong thing and the value/remove-button silently
  // overlap the text before them. See report.
  const float row_w = ImGui::GetContentRegionAvail().x;
  const float indent = ImGui::GetTextLineHeight() * 0.7f + 6.0f;
  const float btn_w = ImGui::GetFrameHeight();

  ImVec2 p0 = ImGui::GetCursorScreenPos();
  const float sq = ImGui::GetTextLineHeight() * 0.7f;
  const float pad = (ImGui::GetTextLineHeight() - sq) * 0.5f;
  ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p0.x + 2.0f, p0.y + pad), ImVec2(p0.x + 2.0f + sq, p0.y + pad + sq),
                                             to_im_color(s.display_color));
  ImGui::Dummy(ImVec2(indent, ImGui::GetTextLineHeight()));
  ImGui::SameLine();

  if (!s.visible) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  const bool toggled =
      ImGui::Selectable(s.sig->name.c_str(), false, ImGuiSelectableFlags_None, ImVec2(ImGui::CalcTextSize(s.sig->name.c_str()).x, 0));
  if (!s.visible) ImGui::PopStyleColor();
  if (toggled) s.visible = !s.visible;
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click to %s this series", s.visible ? "hide" : "show");

  const auto val = value_at_or_before(s.xs, s.ys, cur_sec);
  const std::string val_str = val.has_value() ? s.sig->formatValue(*val) : "--";
  const float val_w = ImGui::CalcTextSize(val_str.c_str()).x;
  ImGui::SameLine(std::max(0.0f, row_w - val_w - btn_w - 12.0f));
  ImGui::TextUnformatted(val_str.c_str());

  ImGui::SameLine(std::max(0.0f, row_w - btn_w));
  if (ImGui::SmallButton("x")) {
    remove_signals_if(chart, [&](const SigItem &item) { return item.msg_id == s.msg_id && item.sig == s.sig; });
    // mirrors ChartView::removeIf(): a chart with no signals left closes
    // itself. Deferred like the header's own close/split buttons below --
    // draw_one_chart() is still using `chart` after this call returns, so
    // erasing it from g_charts.charts here would dangle the reference.
    if (chart.sigs.empty()) g_charts.pending_close_id = chart.id;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove series");

  // second row: dimmed "msg name id", indented under the signal name (mirrors
  // the faded message-detail span in ChartView::updateTitle()'s legend text)
  ImGui::Dummy(ImVec2(indent, 0.0f));
  ImGui::SameLine();
  ImGui::TextDisabled("%s %s", msgName(s.msg_id).c_str(), s.msg_id.toString().c_str());

  ImGui::PopID();
}

void draw_one_chart(AppState &app, ChartState &chart, float width, int chart_index) {
  const auto range = chart_effective_range();
  const double cur_sec = can->currentSec();
  const int max_points = std::max(256, static_cast<int>(width) * 2);  // ~2x pixel width, mirrors jotpluggler

  ImGui::PushID(chart.id);
  ImGui::BeginGroup();
  ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
  ImGui::BeginChild("##chart_cell", ImVec2(width, static_cast<float>(settings.chart_height)), ImGuiChildFlags_Borders);

  // -- header: series-type override, manage, split, close (mirrors
  // ChartView's context menu / corner overlay buttons, laid out inline
  // instead of floating -- see report) --------------------------------
  static const char *kTypeLabels[] = {"Line", "Step", "Scatter"};
  ImGui::SetNextItemWidth(70.0f);
  if (ImGui::BeginCombo("##series_type", kTypeLabels[chart.series_type])) {
    for (int i = 0; i < 3; ++i) {
      if (ImGui::Selectable(kTypeLabels[i], i == chart.series_type)) chart.series_type = i;
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("Manage")) request_manage_chart(chart.id);
  ImGui::SameLine();
  ImGui::BeginDisabled(chart.sigs.size() <= 1);
  if (ImGui::SmallButton("Split")) g_charts.pending_split_id = chart.id;
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeight()));
  if (ImGui::SmallButton("x##close_chart")) g_charts.pending_close_id = chart.id;
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove Chart");

  // -- legend rows (name, msg, live value @ cur_sec, remove) -----------
  for (int i = 0; i < static_cast<int>(chart.sigs.size()); ++i) {
    draw_legend_row(chart, chart.sigs[static_cast<size_t>(i)], i, cur_sec);
  }

  // -- decimate the visible window + compute y-axis fit -----------------
  struct Prepared {
    SigItem *item;
    std::vector<double> xs, ys;
  };
  std::vector<Prepared> prepared;
  prepared.reserve(chart.sigs.size());
  double y_min = std::numeric_limits<double>::max();
  double y_max = std::numeric_limits<double>::lowest();
  std::string unit;
  bool unit_set = false, unit_uniform = true;
  for (SigItem &s : chart.sigs) {
    if (!s.visible) continue;
    auto [b, e] = window_indices(s.xs, range.first, range.second);
    Prepared p{&s, {}, {}};
    decimate_window(s.xs, s.ys, b, e, max_points, &p.xs, &p.ys);
    s.min = std::numeric_limits<double>::max();
    s.max = std::numeric_limits<double>::lowest();
    for (double y : p.ys) {
      s.min = std::min(s.min, y);
      s.max = std::max(s.max, y);
    }
    if (!p.ys.empty()) {
      y_min = std::min(y_min, s.min);
      y_max = std::max(y_max, s.max);
    }
    if (!unit_set) {
      unit = s.sig->unit;
      unit_set = true;
    } else if (unit != s.sig->unit) {
      unit_uniform = false;
    }
    prepared.push_back(std::move(p));
  }
  if (y_min > y_max) {
    y_min = 0.0;
    y_max = 0.0;
  }
  const double delta = std::fabs(y_max - y_min) < 1e-3 ? 1.0 : (y_max - y_min) * 0.05;
  NiceAxis ny = get_nice_axis_numbers(y_min - delta, y_max + delta, 3);
  int decimals = 0;
  if (ny.max > ny.min && ny.tick_count > 1) {
    decimals = std::clamp(static_cast<int>(-std::floor(std::log10((ny.max - ny.min) / (ny.tick_count - 1)))), 0, 6);
  }
  char y_fmt[16];
  std::snprintf(y_fmt, sizeof(y_fmt), "%%.%df", decimals);

  // -- plot --------------------------------------------------------------
  // ImPlotAxisFlags_Lock: the axis range is fully driven by our own
  // SetupAxisLimits(..., ImPlotCond_Always) calls below (shared X range,
  // computed Y fit) -- Lock keeps ImPlot's own pan/box-zoom from ever
  // mutating it, so our manual mouse handling (further down) is the only
  // thing that can change what's on screen. See draw_grid() in
  // charts_panel.cc for why this is used instead of touching the global
  // ImPlot input map.
  const ImPlotAxisFlags x_flags =
      ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_Lock;
  const ImPlotAxisFlags y_flags = x_flags;
  const ImPlotFlags plot_flags = ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect;

  push_mono_font();
  ImVec2 plot_size = ImGui::GetContentRegionAvail();
  const std::string y_axis_label = unit_uniform ? unit : std::string();
  if (ImPlot::BeginPlot("##plot", plot_size, plot_flags)) {
    ImPlot::SetupAxis(ImAxis_X1, nullptr, x_flags);
    ImPlot::SetupAxisFormat(ImAxis_X1, "%.2f");
    ImPlot::SetupAxis(ImAxis_Y1, y_axis_label.c_str(), y_flags);
    ImPlot::SetupAxisFormat(ImAxis_Y1, y_fmt);
    ImPlot::SetupAxisLimits(ImAxis_X1, range.first, range.second, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, ny.min, ny.max, ImPlotCond_Always);

    for (size_t i = 0; i < prepared.size(); ++i) {
      const Prepared &p = prepared[i];
      if (p.xs.size() < 2) continue;
      const std::string id = "##s" + std::to_string(i);
      ImPlotSpec spec;
      spec.LineColor = ImGui::ColorConvertU32ToFloat4(to_im_color(p.item->display_color));
      spec.LineWeight = 2.0f;
      spec.Flags = ImPlotItemFlags_NoLegend | ImPlotLineFlags_SkipNaN;
      if (chart.series_type == SERIES_STEP_LINE) {
        spec.Flags |= ImPlotStairsFlags_PreStep;
        ImPlot::PlotStairs(id.c_str(), p.xs.data(), p.ys.data(), static_cast<int>(p.xs.size()), spec);
      } else if (chart.series_type == SERIES_SCATTER) {
        spec.Marker = ImPlotMarker_Circle;
        spec.MarkerSize = 3.0f;
        spec.MarkerFillColor = spec.LineColor;
        spec.LineWeight = 0.0f;
        ImPlot::PlotScatter(id.c_str(), p.xs.data(), p.ys.data(), static_cast<int>(p.xs.size()), spec);
      } else {
        ImPlot::PlotLine(id.c_str(), p.xs.data(), p.ys.data(), static_cast<int>(p.xs.size()), spec);
      }
    }

    // cursor line (mirrors ChartView::drawTimeline())
    const double clamped_cur = std::clamp(cur_sec, range.first, range.second);
    ImPlotSpec cursor_spec;
    cursor_spec.LineColor = ImVec4(0.5f, 0.5f, 0.5f, 0.9f);
    cursor_spec.LineWeight = 1.0f;
    cursor_spec.Flags = ImPlotItemFlags_NoLegend | ImPlotItemFlags_NoFit;
    ImPlot::PlotInfLines("##cursor", &clamped_cur, 1, cursor_spec);

    // track dot(s) at the shared hover time (mirrors ChartView::drawForeground())
    if (g_charts.hover_time.has_value()) {
      const double t = *g_charts.hover_time;
      ImDrawList *dl = ImPlot::GetPlotDrawList();
      for (Prepared &p : prepared) {
        if (!p.item->visible) continue;
        auto v = value_at_or_before(p.item->xs, p.item->ys, t);
        if (v.has_value()) {
          ImVec2 px = ImPlot::PlotToPixels(ImPlotPoint(t, *v));
          dl->AddCircleFilled(px, 4.0f, to_im_color(p.item->display_color));
        }
      }
    }

    // -- mouse interaction: box-zoom (left-drag), shift-drag scrub,
    // click-to-seek, right-click-to-undo -- mirrors ChartView::mouse*Event().
    const bool hovered = ImPlot::IsPlotHovered();
    const bool idle = g_charts.drag_chart_id < 0 && g_charts.scrub_chart_id < 0;
    if (hovered && idle) {
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (ImGui::GetIO().KeyShift) {
          g_charts.scrub_chart_id = chart.id;
          g_charts.scrub_resume_after = !can->isPaused();
          if (g_charts.scrub_resume_after) can->pause(true);
        } else {
          g_charts.drag_chart_id = chart.id;
          g_charts.drag_start_t = ImPlot::GetPlotMousePos().x;
        }
      } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        g_charts.zoom_stack.undo();
      }
    }

    if (g_charts.drag_chart_id == chart.id) {
      const double drag_t = std::clamp(ImPlot::GetPlotMousePos().x, can->minSeconds(), can->maxSeconds());
      const double lo = std::min(g_charts.drag_start_t, drag_t);
      const double hi = std::max(g_charts.drag_start_t, drag_t);
      ImVec2 pa = ImPlot::PlotToPixels(ImPlotPoint(lo, ny.min));
      ImVec2 pb = ImPlot::PlotToPixels(ImPlotPoint(hi, ny.max));
      const ImVec2 rmin(std::min(pa.x, pb.x), std::min(pa.y, pb.y));
      const ImVec2 rmax(std::max(pa.x, pb.x), std::max(pa.y, pb.y));
      // Theme-neutral accent (not white/black) so the selection reads
      // against both the light and dark ImPlot plot backgrounds -- an
      // earlier version used a near-white fill/border that was invisible on
      // light theme's near-white plot background. See report.
      ImDrawList *dl = ImPlot::GetPlotDrawList();
      dl->AddRectFilled(rmin, rmax, IM_COL32(66, 150, 250, 60));
      dl->AddRect(rmin, rmax, IM_COL32(66, 150, 250, 220));

      if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        ImVec2 start_px = ImPlot::PlotToPixels(ImPlotPoint(g_charts.drag_start_t, 0.0));
        ImVec2 end_px = ImPlot::PlotToPixels(ImPlotPoint(drag_t, 0.0));
        const float width_px = std::fabs(end_px.x - start_px.x);
        if (width_px <= 2.0f) {
          can->seekTo(lo);
        } else if (width_px > 10.0f && (hi - lo) > MIN_ZOOM_SECONDS) {
          g_charts.zoom_stack.pushCommand(new ChartZoomCommand({lo, hi}));
        }
        g_charts.drag_chart_id = -1;
      }
    }

    if (g_charts.scrub_chart_id == chart.id) {
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::GetIO().KeyShift) {
        can->seekTo(std::clamp(ImPlot::GetPlotMousePos().x, can->minSeconds(), can->maxSeconds()));
      } else {
        if (g_charts.scrub_resume_after) can->pause(false);
        g_charts.scrub_resume_after = false;
        g_charts.scrub_chart_id = -1;
      }
    }

    // shared hover time for the combined tooltip, suppressed while
    // dragging/scrubbing (mirrors ChartsWidget::eventFilter's !is_zooming guard)
    if (hovered && idle) {
      g_charts.hover_time = ImPlot::GetPlotMousePos().x;
    }

    ImPlot::EndPlot();
  }
  pop_mono_font();

  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::EndGroup();
  ImGui::PopID();
}

void draw_hover_tooltip() {
  if (!g_charts.hover_time.has_value() || g_charts.charts.empty()) return;
  const double t = *g_charts.hover_time;

  ImGui::BeginTooltip();
  push_mono_font();
  ImGui::Text("t = %.3f", t);
  pop_mono_font();
  for (const ChartState &c : g_charts.charts) {
    if (c.sigs.empty()) continue;
    ImGui::Separator();
    for (const SigItem &s : c.sigs) {
      if (!s.visible) continue;
      auto v = value_at_or_before(s.xs, s.ys, t);
      // s.min/s.max are left at their sentinel max()/lowest() init values
      // when the currently-visible decimated window has zero samples for
      // this signal (e.g. a sparsely-updated request/bool signal with no
      // events inside a narrow zoom) -- guard against printing those raw.
      char min_buf[32], max_buf[32];
      if (s.min <= s.max) {
        std::snprintf(min_buf, sizeof(min_buf), "%.3g", s.min);
        std::snprintf(max_buf, sizeof(max_buf), "%.3g", s.max);
      } else {
        std::snprintf(min_buf, sizeof(min_buf), "--");
        std::snprintf(max_buf, sizeof(max_buf), "--");
      }
      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(to_im_color(s.display_color)));
      ImGui::TextUnformatted("\xe2\x96\xa0");  // "■"
      ImGui::PopStyleColor();
      ImGui::SameLine();
      ImGui::Text("%s: %s (%s, %s)", legend_label(s).c_str(),
                  v.has_value() ? s.sig->formatValue(*v, false).c_str() : "--", min_buf, max_buf);
    }
  }
  ImGui::EndTooltip();
}
