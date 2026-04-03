#include "tools/jotpluggler/internal.h"

#include "implot.h"
#include "imgui_internal.h"

#include <cmath>
#include <cstdio>
#include <limits>

constexpr double PLOT_Y_PAD_FRACTION = 0.4;

struct PlotBounds {
  double x_min = 0.0;
  double x_max = 1.0;
  double y_min = 0.0;
  double y_max = 1.0;
};

bool curve_has_samples(const AppSession &session, const Curve &curve) {
  if (curve_has_local_samples(curve)) return true;
  if (curve.name.empty() || curve.name.front() != '/') {
    return false;
  }
  const RouteSeries *series = app_find_route_series(session, curve.name);
  return series != nullptr && series->times.size() > 1 && series->times.size() == series->values.size();
}

void extend_range(const std::vector<double> &values, bool *found, double *min_value, double *max_value) {
  if (values.empty()) {
    return;
  }
  const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
  if (!*found) {
    *min_value = *min_it;
    *max_value = *max_it;
    *found = true;
    return;
  }
  *min_value = std::min(*min_value, *min_it);
  *max_value = std::max(*max_value, *max_it);
}

void ensure_non_degenerate_range(double *min_value, double *max_value, double pad_fraction, double fallback_pad) {
  if (*max_value <= *min_value) {
    const double pad = std::max(std::abs(*min_value) * 0.1, fallback_pad);
    *min_value -= pad;
    *max_value += pad;
    return;
  }
  const double span = *max_value - *min_value;
  const double pad = std::max(span * pad_fraction, fallback_pad);
  *min_value -= pad;
  *max_value += pad;
}

struct PreparedCurve {
  int pane_curve_index = -1;
  std::string label;
  std::array<uint8_t, 3> color = {160, 170, 180};
  float line_weight = 2.0f;
  bool stairs = false;
  const EnumInfo *enum_info = nullptr;
  SeriesFormat display_info;
  std::optional<double> legend_value;
  std::vector<double> xs;
  std::vector<double> ys;
};

struct StateBlock {
  double t0 = 0.0;
  double t1 = 0.0;
  int value = 0;
  std::string label;
};

struct PaneValueFormatContext {
  SeriesFormat format;
  bool valid = false;
};

bool curves_are_bool_like(const std::vector<PreparedCurve> &prepared_curves) {
  if (prepared_curves.empty()) {
    return false;
  }
  for (const PreparedCurve &curve : prepared_curves) {
    if (!curve.display_info.integer_like || curve.ys.empty()) {
      return false;
    }
    bool found_finite = false;
    for (double value : curve.ys) {
      if (!std::isfinite(value)) continue;
      found_finite = true;
      if (std::abs(value) > 0.01 && std::abs(value - 1.0) > 0.01) {
        return false;
      }
    }
    if (!found_finite) {
      return false;
    }
  }
  return true;
}

ImU32 state_block_color(int value, float alpha = 1.0f) {
  static constexpr std::array<std::array<uint8_t, 3>, 8> kPalette = {{
    {{111, 143, 175}},
    {{0, 163, 108}},
    {{255, 195, 0}},
    {{199, 0, 57}},
    {{123, 97, 255}},
    {{0, 150, 136}},
    {{214, 48, 49}},
    {{52, 73, 94}},
  }};
  const size_t index = static_cast<size_t>(std::abs(value)) % kPalette.size();
  return ImGui::GetColorU32(color_rgb(kPalette[index], alpha));
}

std::string state_block_label(const PreparedCurve &curve, int value) {
  if (curve.enum_info != nullptr && value >= 0 && static_cast<size_t>(value) < curve.enum_info->names.size()) {
    const std::string &name = curve.enum_info->names[static_cast<size_t>(value)];
    if (!name.empty()) {
      return name;
    }
  }
  return std::to_string(value);
}

std::vector<StateBlock> build_state_blocks(const PreparedCurve &curve) {
  std::vector<StateBlock> blocks;
  if (curve.xs.size() < 2 || curve.xs.size() != curve.ys.size()) {
    return blocks;
  }

  int current_value = static_cast<int>(std::llround(curve.ys.front()));
  double start_time = curve.xs.front();
  for (size_t i = 1; i < curve.xs.size(); ++i) {
    const int value = static_cast<int>(std::llround(curve.ys[i]));
    if (value == current_value) {
      continue;
    }
    const double end_time = curve.xs[i];
    if (end_time > start_time) {
      blocks.push_back(StateBlock{
        .t0 = start_time,
        .t1 = end_time,
        .value = current_value,
        .label = state_block_label(curve, current_value),
      });
    }
    current_value = value;
    start_time = end_time;
  }

  const double final_time = curve.xs.back();
  if (final_time >= start_time) {
    blocks.push_back(StateBlock{
      .t0 = start_time,
      .t1 = final_time,
      .value = current_value,
      .label = state_block_label(curve, current_value),
    });
  }
  return blocks;
}

void app_decimate_samples_impl(const std::vector<double> &xs_in,
                           const std::vector<double> &ys_in,
                           int max_points,
                           std::vector<double> *xs_out,
                           std::vector<double> *ys_out) {

  const size_t bucket_count = std::max<size_t>(1, static_cast<size_t>(max_points / 4));
  const size_t bucket_size = std::max<size_t>(
    1,
    static_cast<size_t>(std::ceil(static_cast<double>(xs_in.size()) / static_cast<double>(bucket_count))));
  xs_out->reserve(bucket_count * 4 + 2);
  ys_out->reserve(bucket_count * 4 + 2);

  size_t last_index = std::numeric_limits<size_t>::max();
  auto append_index = [&](size_t index) {
    if (index >= xs_in.size() || index == last_index) {
      return;
    }
    xs_out->push_back(xs_in[index]);
    ys_out->push_back(ys_in[index]);
    last_index = index;
  };

  for (size_t start = 0; start < xs_in.size(); start += bucket_size) {
    const size_t end = std::min(xs_in.size(), start + bucket_size);
    size_t min_index = start;
    size_t max_index = start;
    for (size_t index = start + 1; index < end; ++index) {
      if (ys_in[index] < ys_in[min_index]) {
        min_index = index;
      }
      if (ys_in[index] > ys_in[max_index]) {
        max_index = index;
      }
    }

    std::array<size_t, 4> indices = {start, min_index, max_index, end - 1};
    std::sort(indices.begin(), indices.end());
    for (size_t index : indices) {
      append_index(index);
    }
  }
}

void app_decimate_samples(const std::vector<double> &xs_in,
                      const std::vector<double> &ys_in,
                      int max_points,
                      std::vector<double> *xs_out,
                      std::vector<double> *ys_out) {
  xs_out->clear();
  ys_out->clear();
  if (xs_in.empty() || xs_in.size() != ys_in.size()) {
    return;
  }
  if (max_points <= 0 || static_cast<int>(xs_in.size()) <= max_points) {
    *xs_out = xs_in;
    *ys_out = ys_in;
    return;
  }
  app_decimate_samples_impl(xs_in, ys_in, max_points, xs_out, ys_out);
}

void app_decimate_samples(std::vector<double> &&xs_in,
                      std::vector<double> &&ys_in,
                      int max_points,
                      std::vector<double> *xs_out,
                      std::vector<double> *ys_out) {
  xs_out->clear();
  ys_out->clear();
  if (xs_in.empty() || xs_in.size() != ys_in.size()) {
    return;
  }
  if (max_points <= 0 || static_cast<int>(xs_in.size()) <= max_points) {
    *xs_out = std::move(xs_in);
    *ys_out = std::move(ys_in);
    return;
  }
  app_decimate_samples_impl(xs_in, ys_in, max_points, xs_out, ys_out);
}

std::optional<double> app_sample_xy_value_at_time(const std::vector<double> &xs,
                                              const std::vector<double> &ys,
                                              bool stairs,
                                              double tm) {
  if (xs.size() < 2 || xs.size() != ys.size()) {
    return std::nullopt;
  }
  if (tm <= xs.front()) return ys.front();
  if (tm >= xs.back()) return ys.back();

  const auto upper = std::lower_bound(xs.begin(), xs.end(), tm);
  if (upper == xs.begin()) return ys.front();
  if (upper == xs.end()) return ys.back();

  const size_t upper_index = static_cast<size_t>(std::distance(xs.begin(), upper));
  const size_t lower_index = upper_index - 1;
  const double x0 = xs[lower_index];
  const double x1 = xs[upper_index];
  const double y0 = ys[lower_index];
  const double y1 = ys[upper_index];
  if (std::abs(tm - x1) < 1.0e-9) return y1;
  if (stairs || x1 <= x0) return y0;
  const double alpha = (tm - x0) / (x1 - x0);
  return y0 + (y1 - y0) * alpha;
}

int format_numeric_axis_tick(double value, char *buf, int size, void *user_data) {
  const auto *ctx = static_cast<const PaneValueFormatContext *>(user_data);
  if (ctx == nullptr || !ctx->valid) {
    return std::snprintf(buf, size, "%.6g", value);
  }
  if (ctx->format.integer_like) {
    const double nearest_int = std::round(value);
    if (std::abs(value - nearest_int) > 1.0e-6) {
      int decimals = 1;
      while (decimals < 4) {
        const double scale = std::pow(10.0, decimals);
        const double rounded = std::round(value * scale) / scale;
        if (std::abs(value - rounded) <= 1.0e-6) {
          break;
        }
        ++decimals;
      }
      return std::snprintf(buf, size, "%.*f", decimals, value);
    }
  }
  return std::snprintf(buf, size, ctx->format.fmt, value);
}

void merge_pane_value_format(PaneValueFormatContext *ctx, const SeriesFormat &format) {
  if (!ctx->valid) {
    ctx->format = format;
    ctx->valid = true;
    return;
  }
  ctx->format.has_negative = ctx->format.has_negative || format.has_negative;
  ctx->format.digits_before = std::max(ctx->format.digits_before, format.digits_before);
  ctx->format.decimals = std::max(ctx->format.decimals, format.decimals);
  ctx->format.integer_like = ctx->format.decimals == 0;
  const int sign_width = ctx->format.has_negative ? 1 : 0;
  const int dot_width = ctx->format.decimals > 0 ? 1 : 0;
  ctx->format.total_width = sign_width + ctx->format.digits_before + dot_width + ctx->format.decimals;
  std::snprintf(ctx->format.fmt, sizeof(ctx->format.fmt), "%%%d.%df",
                ctx->format.total_width, ctx->format.decimals);
}

std::string curve_legend_label(const PreparedCurve &curve, bool has_cursor_time, size_t label_width) {
  if (!has_cursor_time) return curve.label;
  if (!curve.legend_value.has_value()) return curve.label;
  const std::string value_text = format_display_value(*curve.legend_value, curve.display_info, curve.enum_info);
  if (value_text.empty()) return curve.label;
  const size_t padded_width = std::max(label_width, curve.label.size());
  return curve.label + std::string(padded_width - curve.label.size() + 2, ' ') + value_text;
}

bool build_curve_series(const AppSession &session,
                        const Curve &curve,
                        const UiState &state,
                        int max_points,
                        PreparedCurve *prepared) {
  std::vector<double> xs;
  std::vector<double> ys;
  if (curve_has_local_samples(curve)) {
    xs = curve.xs;
    ys = curve.ys;
  } else {
    const RouteSeries *series = app_find_route_series(session, curve.name);
    if (series == nullptr || series->times.size() < 2 || series->times.size() != series->values.size()) {
      return false;
    }

    size_t begin_index = 0;
    size_t end_index = series->times.size();
    if (state.has_shared_range && state.x_view_max > state.x_view_min) {
      auto begin_it = std::lower_bound(series->times.begin(), series->times.end(), state.x_view_min);
      auto end_it = std::upper_bound(series->times.begin(), series->times.end(), state.x_view_max);
      begin_index = begin_it == series->times.begin() ? 0 : static_cast<size_t>(std::distance(series->times.begin(), begin_it - 1));
      end_index = end_it == series->times.end() ? series->times.size() : static_cast<size_t>(std::distance(series->times.begin(), end_it + 1));
      end_index = std::min(end_index, series->times.size());
    }
    if (end_index <= begin_index + 1) return false;
    xs.assign(series->times.begin() + begin_index, series->times.begin() + end_index);
    ys.assign(series->values.begin() + begin_index, series->values.begin() + end_index);
  }

  std::vector<double> transformed_xs;
  std::vector<double> transformed_ys;
  if (curve.derivative) {
    if (xs.size() < 2) return false;
    transformed_xs.reserve(xs.size() - 1);
    transformed_ys.reserve(ys.size() - 1);
    for (size_t i = 1; i < xs.size(); ++i) {
      const double dt = curve.derivative_dt > 0.0 ? curve.derivative_dt : (xs[i] - xs[i - 1]);
      if (dt <= 0.0) continue;
      transformed_xs.push_back(xs[i]);
      transformed_ys.push_back((ys[i] - ys[i - 1]) / dt);
    }
  } else {
    transformed_xs = std::move(xs);
    transformed_ys = std::move(ys);
  }

  if (transformed_xs.size() < 2 || transformed_xs.size() != transformed_ys.size()) {
    return false;
  }

  for (double &value : transformed_ys) {
    value = value * curve.value_scale + curve.value_offset;
  }

  prepared->label = app_curve_display_name(curve);
  prepared->color = curve.color;
  prepared->line_weight = curve.derivative ? 1.8f : 2.25f;
  if (!curve.derivative
      && curve.value_scale == 1.0
      && curve.value_offset == 0.0
      && !curve_has_local_samples(curve)
      && !curve.name.empty()
      && curve.name.front() == '/') {
    auto it = session.route_data.enum_info.find(curve.name);
    if (it != session.route_data.enum_info.end()) {
      prepared->enum_info = &it->second;
    }
  }
  if (prepared->enum_info != nullptr) {
    prepared->display_info = compute_series_format(transformed_ys, true);
  } else if (!curve_has_local_samples(curve)
             && !curve.derivative
             && curve.value_scale == 1.0
             && curve.value_offset == 0.0
             && !curve.name.empty()
             && curve.name.front() == '/') {
    auto display_it = session.route_data.series_formats.find(curve.name);
    if (display_it != session.route_data.series_formats.end()) {
      prepared->display_info = display_it->second;
    } else {
      prepared->display_info = compute_series_format(transformed_ys, false);
    }
  } else {
    prepared->display_info = compute_series_format(transformed_ys, false);
  }
  const bool stairs = !curve.derivative && prepared->display_info.integer_like;
  if (state.has_tracker_time) {
    prepared->legend_value = app_sample_xy_value_at_time(transformed_xs, transformed_ys, stairs, state.tracker_time);
  }
  if (stairs) {
    prepared->xs = std::move(transformed_xs);
    prepared->ys = std::move(transformed_ys);
  } else {
    app_decimate_samples(std::move(transformed_xs), std::move(transformed_ys), max_points, &prepared->xs, &prepared->ys);
  }
  prepared->stairs = stairs;
  return prepared->xs.size() > 1 && prepared->xs.size() == prepared->ys.size();
}

bool draw_pane_close_button_overlay() {
  const ImVec2 window_pos = ImGui::GetWindowPos();
  const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
  const ImVec2 content_max = ImGui::GetWindowContentRegionMax();
  const ImRect rect(ImVec2(window_pos.x + content_max.x - 42.0f, window_pos.y + content_min.y + 4.0f),
                    ImVec2(window_pos.x + content_max.x - 4.0f, window_pos.y + content_min.y + 42.0f));
  const bool hovered = ImGui::IsMouseHoveringRect(rect.Min, rect.Max, false);
  const bool held = hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left);
  if (hovered) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
  }
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const float pad = 11.0f;
  const ImU32 color = hovered || held
    ? ImGui::GetColorU32(color_rgb(72, 79, 88))
    : ImGui::GetColorU32(color_rgb(138, 146, 156));
  draw_list->AddLine(ImVec2(rect.Min.x + pad, rect.Min.y + pad),
                     ImVec2(rect.Max.x - pad, rect.Max.y - pad),
                     color,
                     2.4f);
  draw_list->AddLine(ImVec2(rect.Min.x + pad, rect.Max.y - pad),
                     ImVec2(rect.Max.x - pad, rect.Min.y + pad),
                     color,
                     2.4f);
  return hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
}

void draw_pane_frame_overlay() {
  const ImVec2 window_pos = ImGui::GetWindowPos();
  const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
  const ImVec2 content_max = ImGui::GetWindowContentRegionMax();
  const ImRect frame_rect(ImVec2(window_pos.x + content_min.x, window_pos.y + content_min.y),
                          ImVec2(window_pos.x + content_max.x, window_pos.y + content_max.y));
  ImGui::GetWindowDrawList()->AddRect(frame_rect.Min,
                                      frame_rect.Max,
                                      ImGui::GetColorU32(color_rgb(186, 190, 196)),
                                      0.0f,
                                      0,
                                      1.0f);
}

PlotBounds compute_plot_bounds(const Pane &pane,
                               const std::vector<PreparedCurve> &prepared_curves,
                               const UiState &state) {
  PlotBounds bounds;
  bounds.x_min = state.has_shared_range ? state.x_view_min : 0.0;
  bounds.x_max = state.has_shared_range ? state.x_view_max : 1.0;
  if (bounds.x_max <= bounds.x_min) {
    bounds.x_max = bounds.x_min + 1.0;
  }

  bool found = false;
  double min_value = 0.0;
  double max_value = 1.0;
  for (const PreparedCurve &curve : prepared_curves) {
    extend_range(curve.ys, &found, &min_value, &max_value);
  }
  if (!found) {
    min_value = 0.0;
    max_value = 1.0;
  }
  if (curves_are_bool_like(prepared_curves)) {
    min_value = std::min(min_value, 0.0);
    max_value = std::max(max_value, 1.0);
  }
  ensure_non_degenerate_range(&min_value, &max_value, PLOT_Y_PAD_FRACTION, 0.1);
  if (pane.range.has_y_limit_min) {
    min_value = pane.range.y_limit_min;
  }
  if (pane.range.has_y_limit_max) {
    max_value = pane.range.y_limit_max;
  }
  ensure_non_degenerate_range(&min_value, &max_value, 0.0, 0.1);
  bounds.y_min = min_value;
  bounds.y_max = max_value;
  return bounds;
}

void draw_state_blocks_pane(const std::vector<PreparedCurve> &prepared_curves, UiState *state) {
  if (prepared_curves.empty() || !state->has_shared_range || state->x_view_max <= state->x_view_min) {
    return;
  }

  ImDrawList *draw_list = ImPlot::GetPlotDrawList();
  const ImVec2 plot_min = ImPlot::GetPlotPos();
  const ImVec2 plot_size = ImPlot::GetPlotSize();
  const int curve_count = static_cast<int>(prepared_curves.size());
  if (plot_size.x <= 2.0f || plot_size.y <= 2.0f || curve_count <= 0) {
    return;
  }

  float label_width = 0.0f;
  if (curve_count > 1) {
    for (const PreparedCurve &curve : prepared_curves) {
      label_width = std::max(label_width, ImGui::CalcTextSize(curve.label.c_str()).x);
    }
    label_width = std::clamp(label_width + 14.0f, 72.0f, std::min(160.0f, plot_size.x * 0.35f));
  }

  const float row_height = plot_size.y / static_cast<float>(curve_count);
  const float blocks_min_x = plot_min.x + label_width;
  const float blocks_max_x = plot_min.x + plot_size.x;
  const float blocks_width = std::max(1.0f, blocks_max_x - blocks_min_x);
  const double x_span = std::max(1.0e-9, state->x_view_max - state->x_view_min);

  struct HoveredBlock {
    int curve_index = -1;
    StateBlock block;
  };
  std::optional<HoveredBlock> hovered;

  const ImVec2 mouse_pos = ImGui::GetMousePos();
  const bool plot_hovered = ImPlot::IsPlotHovered();

  for (int curve_index = 0; curve_index < curve_count; ++curve_index) {
    const PreparedCurve &curve = prepared_curves[static_cast<size_t>(curve_index)];
    const float y0 = plot_min.y + row_height * static_cast<float>(curve_index);
    const float y1 = y0 + row_height;
    const std::vector<StateBlock> blocks = build_state_blocks(curve);

    if (curve_index > 0) {
      draw_list->AddLine(ImVec2(plot_min.x, y0), ImVec2(plot_min.x + plot_size.x, y0),
                         IM_COL32(210, 214, 220, 255), 1.0f);
    }
    if (curve_count > 1) {
      draw_list->AddLine(ImVec2(blocks_min_x, y0), ImVec2(blocks_min_x, y1),
                         IM_COL32(210, 214, 220, 255), 1.0f);
      const float label_left = plot_min.x + 6.0f;
      const float label_right = std::max(label_left + 12.0f, blocks_min_x - 6.0f);
      ImGui::PushStyleColor(ImGuiCol_Text, color_rgb(120, 128, 138));
      ImGui::RenderTextEllipsis(draw_list,
                                ImVec2(label_left, y0 + 4.0f),
                                ImVec2(label_right, y1 - 4.0f),
                                label_right,
                                curve.label.c_str(),
                                nullptr,
                                nullptr);
      ImGui::PopStyleColor();
    }

    for (const StateBlock &block : blocks) {
      const double visible_t0 = std::max(block.t0, state->x_view_min);
      const double visible_t1 = std::min(block.t1, state->x_view_max);
      if (visible_t1 <= visible_t0) {
        continue;
      }
      const float x0 = blocks_min_x + static_cast<float>((visible_t0 - state->x_view_min) / x_span) * blocks_width;
      const float x1 = blocks_min_x + static_cast<float>((visible_t1 - state->x_view_min) / x_span) * blocks_width;
      const ImU32 fill_color = state_block_color(block.value, 0.15f);
      const ImU32 line_color = state_block_color(block.value, 0.90f);
      draw_list->AddRectFilled(ImVec2(x0, y0), ImVec2(std::max(x1, x0 + 1.0f), y1), fill_color);
      draw_list->AddLine(ImVec2(x0, y0), ImVec2(x0, y1), line_color, 2.0f);

      const float block_width = x1 - x0;
      if (block_width > 14.0f) {
        const float text_left = x0 + 6.0f;
        const float text_right = x1 - 6.0f;
        if (text_right > text_left) {
          ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(state_block_color(block.value, 0.80f)));
          ImGui::RenderTextEllipsis(draw_list,
                                    ImVec2(text_left, y0 + 4.0f),
                                    ImVec2(text_right, y1 - 4.0f),
                                    text_right,
                                    block.label.c_str(),
                                    nullptr,
                                    nullptr);
          ImGui::PopStyleColor();
        }
      }

      if (plot_hovered && mouse_pos.x >= blocks_min_x && mouse_pos.x <= blocks_max_x && mouse_pos.y >= y0 && mouse_pos.y <= y1) {
        const double hover_time = state->x_view_min + static_cast<double>((mouse_pos.x - blocks_min_x) / blocks_width) * x_span;
        if (hover_time >= block.t0 && hover_time <= block.t1) {
          hovered = HoveredBlock{
            .curve_index = curve_index,
            .block = block,
          };
        }
      }
    }
  }

  if (hovered.has_value()) {
    const HoveredBlock &info = *hovered;
    ImGui::BeginTooltip();
    if (curve_count > 1) {
      ImGui::Text("%s: %s (%d)", prepared_curves[static_cast<size_t>(info.curve_index)].label.c_str(),
                  info.block.label.c_str(), info.block.value);
    } else {
      ImGui::Text("%s (%d)", info.block.label.c_str(), info.block.value);
    }
    ImGui::Separator();
    ImGui::Text("%.3fs -> %.3fs", info.block.t0, info.block.t1);
    ImGui::Text("duration: %.3fs", info.block.t1 - info.block.t0);
    ImGui::EndTooltip();
  }
}

void persist_shared_range_to_tab(WorkspaceTab *tab, const UiState &state) {
  if (tab == nullptr || !state.has_shared_range) {
    return;
  }
  const double x_min = state.x_view_min;
  const double x_max = state.x_view_max > state.x_view_min ? state.x_view_max : state.x_view_min + 1.0;
  for (Pane &pane : tab->panes) {
    pane.range.valid = true;
    pane.range.left = x_min;
    pane.range.right = x_max;
  }
}

void clear_pane_vertical_limits(Pane *pane) {
  if (pane == nullptr) {
    return;
  }
  pane->range.has_y_limit_min = false;
  pane->range.has_y_limit_max = false;
}

PlotBounds current_plot_bounds_for_pane(const AppSession &session, const Pane &pane, const UiState &state) {
  std::vector<PreparedCurve> prepared_curves;
  prepared_curves.reserve(pane.curves.size());
  constexpr int kAxisEditorMaxPoints = 2048;
  for (size_t curve_index = 0; curve_index < pane.curves.size(); ++curve_index) {
    const Curve &curve = pane.curves[curve_index];
    if (!curve.visible || !curve_has_samples(session, curve)) continue;
    PreparedCurve prepared;
    if (build_curve_series(session, curve, state, kAxisEditorMaxPoints, &prepared)) {
      prepared.pane_curve_index = static_cast<int>(curve_index);
      prepared_curves.push_back(std::move(prepared));
    }
  }
  return compute_plot_bounds(pane, prepared_curves, state);
}

void open_axis_limits_editor(const AppSession &session, UiState *state, int pane_index) {
  ensure_shared_range(state, session);
  clamp_shared_range(state, session);
  const WorkspaceTab *tab = app_active_tab(session.layout, *state);
  if (tab == nullptr || pane_index < 0 || pane_index >= static_cast<int>(tab->panes.size())) {
    return;
  }

  const Pane &pane = tab->panes[static_cast<size_t>(pane_index)];
  const PlotBounds bounds = current_plot_bounds_for_pane(session, pane, *state);
  AxisLimitsEditorState &editor = state->axis_limits;
  editor.open = true;
  editor.pane_index = pane_index;
  editor.x_min = state->x_view_min;
  editor.x_max = state->x_view_max;
  editor.y_min_enabled = pane.range.has_y_limit_min;
  editor.y_max_enabled = pane.range.has_y_limit_max;
  editor.y_min = pane.range.has_y_limit_min ? pane.range.y_limit_min : bounds.y_min;
  editor.y_max = pane.range.has_y_limit_max ? pane.range.y_limit_max : bounds.y_max;
}

bool apply_axis_limits_editor(AppSession *session, UiState *state) {
  WorkspaceTab *tab = app_active_tab(&session->layout, *state);
  if (tab == nullptr) return false;

  AxisLimitsEditorState &editor = state->axis_limits;
  if (editor.pane_index < 0 || editor.pane_index >= static_cast<int>(tab->panes.size())) {
    state->error_text = "The selected pane is no longer available.";
    state->open_error_popup = true;
    return false;
  }
  if (!std::isfinite(editor.x_min) || !std::isfinite(editor.x_max)) {
    state->error_text = "Axis limits must be finite numbers.";
    state->open_error_popup = true;
    return false;
  }
  if (editor.x_max <= editor.x_min) {
    state->error_text = "X max must be greater than X min.";
    state->open_error_popup = true;
    return false;
  }
  if (editor.y_min_enabled && !std::isfinite(editor.y_min)) {
    state->error_text = "Y min must be a finite number.";
    state->open_error_popup = true;
    return false;
  }
  if (editor.y_max_enabled && !std::isfinite(editor.y_max)) {
    state->error_text = "Y max must be a finite number.";
    state->open_error_popup = true;
    return false;
  }
  if (editor.y_min_enabled && editor.y_max_enabled && editor.y_max <= editor.y_min) {
    state->error_text = "Y max must be greater than Y min.";
    state->open_error_popup = true;
    return false;
  }

  const SketchLayout before_layout = session->layout;
  state->has_shared_range = true;
  state->x_view_min = editor.x_min;
  state->x_view_max = editor.x_max;
  if (session->data_mode == SessionDataMode::Stream) {
    state->follow_latest = infer_stream_follow_state(*state, *session);
  } else {
    state->follow_latest = false;
  }
  state->suppress_range_side_effects = true;
  clamp_shared_range(state, *session);
  persist_shared_range_to_tab(tab, *state);

  Pane &pane = tab->panes[static_cast<size_t>(editor.pane_index)];
  pane.range.has_y_limit_min = editor.y_min_enabled;
  pane.range.has_y_limit_max = editor.y_max_enabled;
  if (editor.y_min_enabled) {
    pane.range.y_limit_min = editor.y_min;
  }
  if (editor.y_max_enabled) {
    pane.range.y_limit_max = editor.y_max;
  }

  const PlotBounds bounds = current_plot_bounds_for_pane(*session, pane, *state);
  pane.range.valid = true;
  pane.range.left = state->x_view_min;
  pane.range.right = state->x_view_max;
  pane.range.bottom = bounds.y_min;
  pane.range.top = bounds.y_max;

  state->undo.push(before_layout);
  const bool ok = mark_layout_dirty(session, state);
  if (ok) {
    state->status_text = "Axis limits updated";
  }
  return ok;
}

void draw_plot(const AppSession &session, Pane *pane, UiState *state) {
  std::vector<PreparedCurve> prepared_curves;
  prepared_curves.reserve(pane->curves.size());
  const int max_points = std::max(256, static_cast<int>(ImGui::GetContentRegionAvail().x) * 2);
  for (size_t curve_index = 0; curve_index < pane->curves.size(); ++curve_index) {
    const Curve &curve = pane->curves[curve_index];
    if (!curve.visible || !curve_has_samples(session, curve)) continue;
    PreparedCurve prepared;
    if (build_curve_series(session, curve, *state, max_points, &prepared)) {
      prepared.pane_curve_index = static_cast<int>(curve_index);
      prepared_curves.push_back(std::move(prepared));
    }
  }

  const PlotBounds bounds = compute_plot_bounds(*pane, prepared_curves, *state);
  PaneValueFormatContext pane_value_format;
  bool state_block_mode = !prepared_curves.empty();
  size_t max_legend_label_width = 0;
  for (const PreparedCurve &curve : prepared_curves) {
    max_legend_label_width = std::max(max_legend_label_width, curve.label.size());
    if (curve.enum_info == nullptr) {
      state_block_mode = false;
      merge_pane_value_format(&pane_value_format, curve.display_info);
    }
  }
  const int supported_count = static_cast<int>(prepared_curves.size());
  const ImVec2 plot_size = ImGui::GetContentRegionAvail();
  const bool has_cursor_time = state->has_tracker_time;
  const double cursor_time = state->tracker_time;

  ImPlot::PushStyleColor(ImPlotCol_PlotBg, color_rgb(255, 255, 255));
  ImPlot::PushStyleColor(ImPlotCol_PlotBorder, color_rgb(186, 190, 196));
  ImPlot::PushStyleColor(ImPlotCol_LegendBg, color_rgb(248, 249, 251, 0.92f));
  ImPlot::PushStyleColor(ImPlotCol_LegendBorder, color_rgb(168, 175, 184));
  ImPlot::PushStyleColor(ImPlotCol_LegendText, color_rgb(57, 62, 69));
  ImPlot::PushStyleColor(ImPlotCol_TitleText, color_rgb(57, 62, 69));
  ImPlot::PushStyleColor(ImPlotCol_InlayText, color_rgb(95, 103, 112));
  ImPlot::PushStyleColor(ImPlotCol_AxisGrid, color_rgb(188, 196, 206));
  ImPlot::PushStyleColor(ImPlotCol_AxisText, color_rgb(95, 103, 112));
  ImPlot::PushStyleColor(ImPlotCol_AxisBg, color_rgb(255, 255, 255, 0.0f));
  ImPlot::PushStyleColor(ImPlotCol_AxisBgHovered, color_rgb(214, 220, 228, 0.45f));
  ImPlot::PushStyleColor(ImPlotCol_AxisBgActive, color_rgb(199, 209, 222, 0.55f));
  ImPlot::PushStyleColor(ImPlotCol_Selection, color_rgb(252, 211, 77, 0.28f));
  ImPlot::PushStyleColor(ImPlotCol_Crosshairs, color_rgb(120, 128, 138, 0.70f));
  ImPlot::PushStyleVar(ImPlotStyleVar_LegendPadding, ImVec2(56.0f, 10.0f));

  ImPlotFlags plot_flags = ImPlotFlags_NoTitle | ImPlotFlags_NoMenus;
  if (state_block_mode) {
    plot_flags |= ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText;
  }
  if (supported_count == 0) {
    plot_flags |= ImPlotFlags_NoLegend;
  }

  const ImPlotAxisFlags x_axis_flags = ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight;
  ImPlotAxisFlags y_axis_flags = ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight;
  if (state_block_mode) {
    y_axis_flags |= ImPlotAxisFlags_NoDecorations;
  }
  const bool explicit_y = pane->range.has_y_limit_min || pane->range.has_y_limit_max;
  if (!state_block_mode && !explicit_y && supported_count > 0) {
    y_axis_flags |= ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit;
  }

  const double previous_x_min = state->x_view_min;
  const double previous_x_max = state->x_view_max;
  app_push_mono_font();
  if (ImPlot::BeginPlot("##plot", plot_size, plot_flags)) {
    ImPlot::SetupAxes(nullptr, nullptr, x_axis_flags, y_axis_flags);
    ImPlot::SetupAxisFormat(ImAxis_X1, "%.1f");
    if (state_block_mode) {
      ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);
    } else if (pane_value_format.valid) {
      ImPlot::SetupAxisFormat(ImAxis_Y1, format_numeric_axis_tick, &pane_value_format);
    } else {
      ImPlot::SetupAxisFormat(ImAxis_Y1, "%.6g");
    }
    ImPlot::SetupAxisLinks(ImAxis_X1, &state->x_view_min, &state->x_view_max);
    if (state->route_x_max > state->route_x_min) {
      const double x_constraint_min = session.data_mode == SessionDataMode::Stream
        ? state->route_x_min - std::max(MIN_HORIZONTAL_ZOOM_SECONDS, session.stream_buffer_seconds)
        : state->route_x_min;
      ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, x_constraint_min, state->route_x_max);
    }
    if (!state_block_mode) {
      ImPlot::SetupMouseText(ImPlotLocation_SouthEast, ImPlotMouseTextFlags_NoAuxAxes);
    }
    if (!state_block_mode && (explicit_y || supported_count == 0)) {
      ImPlot::SetupAxisLimits(ImAxis_Y1, bounds.y_min, bounds.y_max, ImPlotCond_Always);
    }
    if (!state_block_mode && supported_count > 0) {
      ImPlot::SetupLegend(ImPlotLocation_NorthEast);
    }

    if (state_block_mode) {
      draw_state_blocks_pane(prepared_curves, state);
    } else {
      for (size_t i = 0; i < prepared_curves.size(); ++i) {
        const PreparedCurve &curve = prepared_curves[i];
        std::string series_id = curve_legend_label(curve, has_cursor_time, max_legend_label_width) + "##curve" + std::to_string(i);
        ImPlotSpec spec;
        spec.LineColor = color_rgb(curve.color);
        spec.LineWeight = curve.line_weight;
        spec.Flags = ImPlotLineFlags_SkipNaN;
        if (!curve.xs.empty() && curve.xs.size() == curve.ys.size()) {
          if (curve.stairs) {
            spec.Flags = ImPlotStairsFlags_PreStep;
            ImPlot::PlotStairs(series_id.c_str(), curve.xs.data(), curve.ys.data(), static_cast<int>(curve.xs.size()), spec);
          } else {
            ImPlot::PlotLine(series_id.c_str(), curve.xs.data(), curve.ys.data(), static_cast<int>(curve.xs.size()), spec);
          }
        }
      }
    }
    if (has_cursor_time) {
      const double clamped_cursor_time = std::clamp(cursor_time, state->route_x_min, state->route_x_max);
      ImPlotSpec cursor_spec;
      cursor_spec.LineColor = color_rgb(108, 118, 128, 0.7f);
      cursor_spec.LineWeight = 1.0f;
      cursor_spec.Flags = ImPlotItemFlags_NoLegend;
      ImPlot::PlotInfLines("##tracker_cursor", &clamped_cursor_time, 1, cursor_spec);
    }
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      state->tracker_time = std::clamp(ImPlot::GetPlotMousePos().x, state->route_x_min, state->route_x_max);
      state->has_tracker_time = true;
    }
    ImPlot::EndPlot();
  }
  app_pop_mono_font();
  clamp_shared_range(state, session);
  if (std::abs(state->x_view_min - previous_x_min) > 1.0e-6
      || std::abs(state->x_view_max - previous_x_max) > 1.0e-6) {
    if (!state->suppress_range_side_effects) {
      if (session.data_mode == SessionDataMode::Stream) {
        state->follow_latest = infer_stream_follow_state(*state, session);
      } else {
        state->follow_latest = false;
      }
    }
  }
  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor(12);
}

std::optional<PaneMenuAction> draw_pane_context_menu(const WorkspaceTab &tab, int pane_index) {
  if (!ImGui::BeginPopupContextWindow("##pane_context")) return std::nullopt;

  PaneMenuAction action;
  action.pane_index = pane_index;
  const Pane *pane = pane_index >= 0 && pane_index < static_cast<int>(tab.panes.size())
    ? &tab.panes[static_cast<size_t>(pane_index)]
    : nullptr;
  const bool has_curves = pane_index >= 0
    && pane_index < static_cast<int>(tab.panes.size())
    && !tab.panes[static_cast<size_t>(pane_index)].curves.empty();
  const bool is_plot = pane != nullptr && pane->kind == PaneKind::Plot;
  if (icon_menu_item(icon::SLIDERS, "Edit Axis Limits...", nullptr, false, is_plot)) {
    action.kind = PaneMenuActionKind::OpenAxisLimits;
  }
  icon_menu_item(icon::PALETTE, "Edit Curve Style...", nullptr, false, false && is_plot);
  if (action.kind == PaneMenuActionKind::None
      && icon_menu_item(icon::PLUS_SLASH_MINUS, "Apply filter to data...", nullptr, false, has_curves && is_plot)) {
    action.kind = PaneMenuActionKind::OpenCustomSeries;
  }
  ImGui::Separator();
  if (action.kind == PaneMenuActionKind::None && icon_menu_item(icon::DISTRIBUTE_HORIZONTAL, "Split Left / Right")) {
    action.kind = PaneMenuActionKind::SplitRight;
  } else if (action.kind == PaneMenuActionKind::None
             && icon_menu_item(icon::DISTRIBUTE_VERTICAL, "Split Top / Bottom")) {
    action.kind = PaneMenuActionKind::SplitBottom;
  }
  ImGui::Separator();
  if (icon_menu_item(icon::ZOOM_OUT, "Zoom Out", nullptr, false, is_plot)) {
    action.kind = PaneMenuActionKind::ResetView;
  } else if (icon_menu_item(icon::ARROW_LEFT_RIGHT, "Zoom Out Horizontally", nullptr, false, is_plot)) {
    action.kind = PaneMenuActionKind::ResetHorizontal;
  } else if (icon_menu_item(icon::ARROW_DOWN_UP, "Zoom Out Vertically", nullptr, false, is_plot)) {
    action.kind = PaneMenuActionKind::ResetVertical;
  }
  ImGui::Separator();
  if (icon_menu_item(icon::TRASH, "Remove ALL curves", nullptr, false, is_plot)) {
    action.kind = PaneMenuActionKind::Clear;
  }
  ImGui::Separator();
  icon_menu_item(icon::ARROW_LEFT_RIGHT, "Flip Horizontal Axis", nullptr, false, false);
  icon_menu_item(icon::ARROW_DOWN_UP, "Flip Vertical Axis", nullptr, false, false);
  ImGui::Separator();
  icon_menu_item(icon::FILES, "Copy", nullptr, false, false);
  icon_menu_item(icon::CLIPBOARD2, "Paste", nullptr, false, false);
  icon_menu_item(icon::FILE_EARMARK_IMAGE, "Copy image to clipboard", nullptr, false, false);
  icon_menu_item(icon::SAVE, "Save plot to file", nullptr, false, false);
  icon_menu_item(icon::BAR_CHART, "Show data statistics", nullptr, false, false);
  ImGui::Separator();
  if (icon_menu_item(icon::X_SQUARE, "Close Pane")) {
    action.kind = PaneMenuActionKind::Close;
  }
  ImGui::EndPopup();
  if (action.kind == PaneMenuActionKind::None) return std::nullopt;
  return action;
}
