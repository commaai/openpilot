#include "tools/loggy/panes/plot.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/panes/browser.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "implot.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace loggy {
namespace {

constexpr ImVec4 kPlotColors[] = {
  ImVec4(0.35f, 0.66f, 0.98f, 1.0f),
  ImVec4(0.33f, 0.78f, 0.55f, 1.0f),
  ImVec4(0.98f, 0.70f, 0.30f, 1.0f),
  ImVec4(0.85f, 0.48f, 0.78f, 1.0f),
  ImVec4(0.94f, 0.38f, 0.38f, 1.0f),
};

std::string legend_label(const PreparedPlotSeries &series, size_t label_width) {
  if (!series.has_tracker_value) return series.request.label;
  char value[64];
  std::snprintf(value, sizeof(value), "%.6g", series.tracker_value);
  const size_t width = std::max(label_width, series.request.label.size());
  return series.request.label + std::string(width - series.request.label.size() + 2, ' ') + value;
}

ImVec4 plot_color_for_request(const PlotSeriesRequest &request, size_t index) {
  unsigned int r = 0;
  unsigned int g = 0;
  unsigned int b = 0;
  if (request.color.size() == 7 && request.color[0] == '#' &&
      std::sscanf(request.color.c_str() + 1, "%02x%02x%02x", &r, &g, &b) == 3) {
    return ImVec4(static_cast<float>(r) / 255.0f,
                  static_cast<float>(g) / 255.0f,
                  static_cast<float>(b) / 255.0f,
                  1.0f);
  }
  return kPlotColors[index % std::size(kPlotColors)];
}

void draw_plot_coverage(const std::vector<PreparedPlotSeries> &series) {
  size_t visible = 0;
  size_t missing = 0;
  bool decimated = false;
  double covered = 0.0;
  for (const PreparedPlotSeries &item : series) {
    if (item.view.points.empty()) {
      ++missing;
    } else {
      ++visible;
      covered = std::max(covered, item.view.coverage.covered_seconds);
    }
    decimated = decimated || item.view.decimated;
  }

  ImGui::TextDisabled("%zu series", visible);
  ImGui::SameLine();
  ImGui::TextDisabled("| %.2fs covered", covered);
  if (missing > 0) {
    ImGui::SameLine();
    ImGui::TextDisabled("| %zu missing", missing);
  }
  if (decimated) {
    ImGui::SameLine();
    ImGui::TextDisabled("| decimated");
  }
}

bool accept_series_drop(PaneInstance *pane) {
  bool changed = false;
  if (!ImGui::BeginDragDropTarget()) return false;
  if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload(kLoggySeriesPathPayload, ImGuiDragDropFlags_AcceptBeforeDelivery)) {
    if (payload->Preview) {
      const ImVec2 min = ImGui::GetItemRectMin();
      const ImVec2 max = ImGui::GetItemRectMax();
      ImDrawList *draw_list = ImGui::GetWindowDrawList();
      draw_list->AddRectFilled(min, max, ImGui::GetColorU32(color_rgb(47, 101, 202, 0.16f)));
      draw_list->AddRect(min, max, ImGui::GetColorU32(color_rgb(75, 135, 230, 0.90f)), 0.0f, 0, 2.0f);
    }
    if (!payload->Delivery) {
      ImGui::EndDragDropTarget();
      return false;
    }
    if (payload->Data != nullptr && payload->DataSize > 0) {
      const char *data = static_cast<const char *>(payload->Data);
      const int size = data[payload->DataSize - 1] == '\0' ? payload->DataSize - 1 : payload->DataSize;
      if (size > 0) {
        pane->state_json = plot_state_with_added_series(pane->state_json, std::string_view(data, static_cast<size_t>(size)));
        changed = true;
      }
    }
  }
  ImGui::EndDragDropTarget();
  return changed;
}

int style_combo_index(PlotSeriesStyle style) {
  switch (style) {
    case PlotSeriesStyle::Line: return 1;
    case PlotSeriesStyle::Step: return 2;
    case PlotSeriesStyle::Scatter: return 3;
    case PlotSeriesStyle::Auto:
    default: return 0;
  }
}

PlotSeriesStyle style_from_combo_index(int index) {
  switch (index) {
    case 1: return PlotSeriesStyle::Line;
    case 2: return PlotSeriesStyle::Step;
    case 3: return PlotSeriesStyle::Scatter;
    case 0:
    default: return PlotSeriesStyle::Auto;
  }
}

bool draw_plot_display_controls(PlotSeriesStyle *style, PlotYLimits *limits) {
  if (style == nullptr || limits == nullptr) return false;

  bool changed = false;
  int style_index = style_combo_index(*style);
  ImGui::SetNextItemWidth(96.0f);
  if (ImGui::Combo("Style", &style_index, "Auto\0Line\0Step\0Scatter\0")) {
    *style = style_from_combo_index(style_index);
    changed = true;
  }

  ImGui::SameLine();
  if (ImGui::Button("Y")) ImGui::OpenPopup("##plot_y_limits");
  if (limits->active()) {
    ImGui::SameLine();
    if (limits->min_enabled && limits->max_enabled) {
      ImGui::TextDisabled("[%.6g, %.6g]", limits->min, limits->max);
    } else if (limits->min_enabled) {
      ImGui::TextDisabled("[%.6g, auto]", limits->min);
    } else {
      ImGui::TextDisabled("[auto, %.6g]", limits->max);
    }
  }

  if (ImGui::BeginPopup("##plot_y_limits")) {
    bool min_enabled = limits->min_enabled;
    if (ImGui::Checkbox("Min", &min_enabled)) {
      limits->min_enabled = min_enabled;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    double min_value = limits->min;
    if (ImGui::InputDouble("##plot_y_min", &min_value, 0.0, 0.0, "%.9g")) {
      limits->min = min_value;
      limits->min_enabled = true;
      changed = true;
    }

    bool max_enabled = limits->max_enabled;
    if (ImGui::Checkbox("Max", &max_enabled)) {
      limits->max_enabled = max_enabled;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    double max_value = limits->max;
    if (ImGui::InputDouble("##plot_y_max", &max_value, 0.0, 0.0, "%.9g")) {
      limits->max = max_value;
      limits->max_enabled = true;
      changed = true;
    }

    if (ImGui::Button("Apply")) {
      changed = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
      *limits = {};
      changed = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  return changed;
}

}  // namespace

void draw_plot_pane(Session &session, PaneInstance &pane) {
  const TimeRange range = session.view_range().range();
  PlotSeriesStyle style = parse_plot_series_style(pane.state_json);
  PlotYLimits y_limits = parse_plot_y_limits(pane.state_json);
  if (draw_plot_display_controls(&style, &y_limits)) {
    pane.state_json = plot_state_with_display_options(pane.state_json, style, y_limits);
  }

  const std::vector<PlotSeriesRequest> requests = parse_plot_series_requests(pane.state_json);
  const size_t pixel_cap = static_cast<size_t>(std::max(256.0f, ImGui::GetContentRegionAvail().x * 2.0f));
  const size_t max_points = parse_plot_max_points(pane.state_json, pixel_cap);
  std::vector<PreparedPlotSeries> series = prepare_plot_series(session.store(), requests, range,
                                                               session.playback().tracker_time(), max_points);
  const PlotYAxisBounds y_bounds = compute_plot_y_axis_bounds(series, y_limits);

  draw_plot_coverage(series);

  bool has_points = false;
  size_t label_width = 0;
  for (const PreparedPlotSeries &item : series) {
    has_points = has_points || !item.xs.empty();
    label_width = std::max(label_width, item.request.label.size());
  }

  double x_min = range.start;
  double x_max = range.end > range.start ? range.end : range.start + 1.0;
  ImPlotFlags plot_flags = ImPlotFlags_NoTitle | ImPlotFlags_NoMenus;
  if (!has_points) plot_flags |= ImPlotFlags_NoLegend;

  ImPlot::PushStyleColor(ImPlotCol_PlotBg, color_rgb(47, 49, 51));
  ImPlot::PushStyleColor(ImPlotCol_PlotBorder, color_rgb(92, 96, 98));
  ImPlot::PushStyleColor(ImPlotCol_LegendBg, color_rgb(53, 53, 53, 0.92f));
  ImPlot::PushStyleColor(ImPlotCol_LegendBorder, color_rgb(92, 96, 98));
  ImPlot::PushStyleColor(ImPlotCol_AxisGrid, color_rgb(92, 96, 98, 0.70f));
  ImPlot::PushStyleColor(ImPlotCol_Crosshairs, color_rgb(187, 187, 187, 0.70f));

  push_mono_font();
  bool dropped_series = false;
  ImGui::BeginChild("##loggy_plot_child", ImGui::GetContentRegionAvail(), false,
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  if (ImPlot::BeginPlot("##loggy_plot", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlot::SetupAxes(nullptr, nullptr,
                      ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight,
                      ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisFormat(ImAxis_X1, "%.1f");
    ImPlot::SetupAxisFormat(ImAxis_Y1, "%.6g");
    ImPlot::SetupAxisLinks(ImAxis_X1, &x_min, &x_max);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, session.playback().route_range().start,
                                       session.playback().route_range().end);
    if (y_bounds.active) ImPlot::SetupAxisLimits(ImAxis_Y1, y_bounds.min, y_bounds.max, ImPlotCond_Always);
    if (has_points) ImPlot::SetupLegend(ImPlotLocation_NorthEast);

    for (size_t i = 0; i < series.size(); ++i) {
      const PreparedPlotSeries &item = series[i];
      if (item.xs.size() < 2 || item.xs.size() != item.ys.size()) continue;

      const PlotSeriesStyle item_style = plot_effective_series_style(item, style);
      ImPlotSpec spec;
      spec.LineColor = plot_color_for_request(item.request, i);
      spec.LineWeight = item_style == PlotSeriesStyle::Step ? 1.8f : 2.0f;
      if (item_style == PlotSeriesStyle::Step) {
        spec.Flags = ImPlotStairsFlags_PreStep;
      } else if (item_style == PlotSeriesStyle::Scatter) {
        spec.Marker = ImPlotMarker_Circle;
        spec.MarkerSize = 3.0f;
        spec.MarkerFillColor = spec.LineColor;
        spec.LineWeight = 0.0f;
      } else {
        spec.Flags = ImPlotLineFlags_SkipNaN;
      }
      const std::string label = legend_label(item, label_width) + "###series_" + std::to_string(i) + "_" + item.request.path;
      if (item_style == PlotSeriesStyle::Step) {
        ImPlot::PlotStairs(label.c_str(), item.xs.data(), item.ys.data(), static_cast<int>(item.xs.size()), spec);
      } else if (item_style == PlotSeriesStyle::Scatter) {
        ImPlot::PlotScatter(label.c_str(), item.xs.data(), item.ys.data(), static_cast<int>(item.xs.size()), spec);
      } else {
        ImPlot::PlotLine(label.c_str(), item.xs.data(), item.ys.data(), static_cast<int>(item.xs.size()), spec);
      }
    }

    const double tracker = std::clamp(session.playback().tracker_time(),
                                      session.playback().route_range().start,
                                      session.playback().route_range().end);
    ImPlotSpec tracker_spec;
    tracker_spec.LineColor = color_rgb(220, 220, 220, 0.72f);
    tracker_spec.LineWeight = 1.0f;
    tracker_spec.Flags = ImPlotItemFlags_NoLegend;
    ImPlot::PlotInfLines("##tracker", &tracker, 1, tracker_spec);

    if (ImPlot::IsPlotHovered()) {
      const double hover_t = ImPlot::GetPlotMousePos().x;
      ImGui::BeginTooltip();
      ImGui::Text("t %.3f", hover_t);
      for (const PreparedPlotSeries &item : series) {
        if (item.xs.empty() || item.xs.size() != item.ys.size()) continue;
        const bool stairs = plot_effective_series_style(item, style) == PlotSeriesStyle::Step;
        ImGui::Text("%s %.6g", item.request.label.c_str(), plot_sample_at_time(item.xs, item.ys, stairs, hover_t));
      }
      ImGui::EndTooltip();
    }

    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      session.playback().seek(ImPlot::GetPlotMousePos().x);
    }
    ImPlot::EndPlot();
  }
  ImGui::EndChild();
  dropped_series = accept_series_drop(&pane);
  pop_mono_font();
  ImPlot::PopStyleColor(6);

  if (dropped_series) return;

  if (std::abs(x_min - range.start) > 1.0e-6 || std::abs(x_max - range.end) > 1.0e-6) {
    session.view_range().set_range({x_min, x_max});
  }
}

}  // namespace loggy
