#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/chart.h"

#include "tools/cabana/ui/threadpool.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <limits>

#include "tools/cabana/core/settings.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/chart/chartswidget.h"
#include "tools/cabana/ui/chart/downsample.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

const int AXIS_X_TOP_MARGIN = 4;
const int X_TICK_COUNT = 5;
const double MIN_ZOOM_SECONDS = 0.01;  // 10ms
const double EPSILON = 1e-6;
constexpr ImVec4 LAYOUT_MARGINS{0, 6, 0, 6};  // left, top, right, bottom
constexpr int LEGEND_SPACING = 5;
static inline bool xLessThan(const ImPlotPoint &p, double x) { return p.x < (x - EPSILON); }
static inline bool isNull(const ImPlotPoint &p) { return p.x == 0 && p.y == 0; }

static std::string formatNumber(double value, int precision) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.*f", precision, value);
  return buf;
}

// the decimals needed to tell tick_count ticks over range apart
static int axisPrecision(double range, int tick_count, int min_precision) {
  return std::max(int(-std::floor(std::log10(range / (tick_count - 1)))), min_precision);
}

static void addTextEllipsis(ImDrawList *dl, ImFont *font, ImU32 col, const ImVec2 &pos, float max_x, const std::string &text) {
  const float size = ImGui::GetFontSize();
  ImGui::PushFont(font, 0.0f);
  ImGui::RenderTextEllipsis(dl, pos, ImVec2(max_x, pos.y + size), max_x, text.c_str(), nullptr, nullptr);
  ImGui::PopFont();
}

ChartView::ChartView(const std::pair<double, double> &x_range, ChartsWidget *parent)
    : x_min_(x_range.first), x_max_(x_range.second), charts_widget_(parent) {
  series_type_ = (SeriesType)settings.chart_series_type;

  connections_.push_back(dbc()->signalRemoved.connect([this](const cabana::Signal *sig) { signalRemoved(sig); }));
  connections_.push_back(dbc()->signalUpdated.connect([this](const cabana::Signal *sig) { signalUpdated(sig); }));
  connections_.push_back(dbc()->msgRemoved.connect([this](MessageId id) { msgRemoved(id); }));
}

void ChartView::drawMenuActions() {
  // the current series type is marked with a radio bullet on the left
  const float indent = ImGui::GetFontSize();
  float label_width = ImGui::CalcTextSize("Manage Signals").x;
  for (const char *type : SERIES_TYPE_NAMES) label_width = std::max(label_width, ImGui::CalcTextSize(type).x);
  for (int i = 0; i < (int)std::size(SERIES_TYPE_NAMES); ++i) {
    if (radioMenuItem(SERIES_TYPE_NAMES[i], i == (int)series_type_, indent + label_width + indent)) {
      setSeriesType((SeriesType)i);
    }
  }
  ImGui::Separator();
  ImGui::Indent(indent);
  if (ImGui::MenuItem("Manage Signals")) manageSignals();
  if (ImGui::BeginMenu("Transforms and Statistics")) {
    for (size_t i = 0; i < sigs_.size(); ++i) {
      ImGui::PushID((int)i);
      if (ImGui::BeginMenu((sigs_[i].name() + sigs_[i].description()).c_str())) {
        drawSignalAnalysis(sigs_[i]);
        ImGui::EndMenu();
      }
      ImGui::PopID();
    }
    ImGui::EndMenu();
  }
  if (ImGui::MenuItem("Split Chart", nullptr, false, sigs_.size() > 1)) charts_widget_->splitChart(this);
  ImGui::Unindent(indent);
}

// the buttons and their menus are drawn every frame, at the rects updateLayout() placed them at
void ChartView::createToolButtons() {
  ImGui::SetCursorScreenPos(layout_.close_btn_rect.Min);
  bool close_clicked = iconButton("close_btn", icon::X_LG, "Remove Chart");

  ImGui::SetCursorScreenPos(layout_.manage_btn_rect.Min);
  if (iconButton("manage_btn", icon::THREE_DOTS_VERTICAL, "")) ImGui::OpenPopup("manage_menu");
  if (ImGui::BeginPopup("manage_menu")) {
    drawMenuActions();
    ImGui::EndPopup();
  }

  if (close_clicked) charts_widget_->removeChart(this);
}

void ChartView::addSignal(const MessageId &msg_id, const cabana::Signal *sig) {
  if (hasSignal(msg_id, sig)) return;

  sigs_.push_back({.msg_id = msg_id, .sig = sig, .color = uniqueColor(sig->color)});
  updateSeries(sig);
  charts_widget_->seriesChanged();
}

void ChartView::addTelemetry(const std::string &path, CabanaColor color) {
  if (std::any_of(sigs_.begin(), sigs_.end(), [&](const auto &s) { return s.path == path; })) return;
  sigs_.push_back({.path = path, .color = color});
  updateTelemetry();
  charts_widget_->seriesChanged();
  charts_widget_->analysisRequested();
}

void ChartView::updateTelemetry() {
  telemetry_dirty_ = true;
  pollTelemetry();
}

void ChartView::pollTelemetry() {
  if (telemetry_task_.valid()) {
    if (telemetry_task_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    telemetry_task_.get();
    for (size_t i = 0; i < std::min(sigs_.size(), telemetry_result_->size()); ++i) {
      auto &ready = (*telemetry_result_)[i];
      auto &s = sigs_[i];
      if (ready.path.empty() || s.path != ready.path) continue;
      if (s.transform.type != ready.transform.type || s.transform.scale != ready.transform.scale ||
          s.transform.offset != ready.transform.offset || s.transform.window != ready.transform.window) continue;
      s.raw_vals.swap(ready.raw_vals);
      s.vals.swap(ready.vals);
      s.step_vals.swap(ready.step_vals);
      std::swap(s.segment_tree, ready.segment_tree);
      s.transform_state = std::move(ready.transform_state);
      s.track_pt = {};
    }
    updateAxisY();
    ThreadPool::instance().run([retired = std::move(telemetry_result_)]() mutable { retired.reset(); });
  }
  if (!telemetry_dirty_) return;
  telemetry_dirty_ = false;
  if (std::none_of(sigs_.begin(), sigs_.end(), [](const auto &s) { return !s.path.empty(); })) return;
  telemetry_result_ = std::make_shared<std::vector<SigItem>>();
  const double origin = can->beginMonoTime() * 1e-9;
  std::vector<std::shared_ptr<const cabana::Samples>> samples;
  for (const auto &s : sigs_) {
    auto &item = telemetry_result_->emplace_back();
    if (s.path.empty()) continue;
    item.path = s.path;
    item.transform = s.transform;
    samples.push_back(charts_widget_->telemetrySnapshot(s.path));
  }
  telemetry_task_ = ThreadPool::instance().run([result = telemetry_result_, samples = std::move(samples), origin, build_tree = !can->liveStreaming()]() {
    size_t index = 0;
    for (auto &s : *result) {
      if (s.path.empty()) continue;
      const auto &points = samples[index++];
      if (points) {
        s.raw_vals.reserve(points->size());
        for (const auto &p : *points) s.raw_vals.emplace_back(p.x - origin, p.y);
      }
      buildSeries(s, 0, build_tree);
    }
  });
}

bool ChartView::hasSignal(const MessageId &msg_id, const cabana::Signal *sig) const {
  return std::any_of(sigs_.cbegin(), sigs_.cend(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
}

void ChartView::removeIf(std::function<bool(const SigItem &s)> predicate) {
  int prev_size = sigs_.size();
  sigs_.erase(std::remove_if(sigs_.begin(), sigs_.end(), predicate), sigs_.end());
  if (sigs_.empty() && prev_size > 0) {
    charts_widget_->removeChart(this);
  } else if (sigs_.size() != prev_size) {
    updateTelemetry();
    charts_widget_->seriesChanged();
    updateAxisY();
  }
}

void ChartView::signalUpdated(const cabana::Signal *sig) {
  auto it = std::find_if(sigs_.begin(), sigs_.end(), [sig](auto &s) { return s.sig == sig; });
  if (it != sigs_.end()) {
    if (!(it->color == sig->color)) {
      it->color = uniqueColor(sig->color, sig);
    }
    updateSeries(sig);
  }
}

void ChartView::manageSignals() {
  auto dlg = std::make_unique<SignalSelector>("Manage Chart");
  for (auto &s : sigs_) {
    if (s.path.empty()) dlg->addSelected(s.msg_id, s.sig);
    else dlg->addTelemetry(s.path);
  }
  // runs once the dialog is accepted, dropped if the chart is removed first
  charts_widget_->execSignalSelector(std::move(dlg), this, [this](SignalSelector &selector) {
    const auto &items = selector.selectedItems();
    for (const auto &s : items) {
      if (s.path.empty()) addSignal(s.msg_id, s.sig);
      else addTelemetry(s.path);
    }
    removeIf([&](auto &s) {
      return std::none_of(items.cbegin(), items.cend(), [&](auto &it) { return s.path == it.path && s.msg_id == it.msg_id && s.sig == it.sig; });
    });
  });
}

void ChartView::updateLayout() {
  const ImVec2 grip = ImGui::CalcTextSize(icon::GRIP_HORIZONTAL);
  const ImVec2 top_left = layout_.rect.Min + ImVec2(LAYOUT_MARGINS.x, LAYOUT_MARGINS.y + (title.empty() ? 0 : ImGui::GetTextLineHeightWithSpacing()));
  layout_.move_icon_rect = ImRect(top_left, top_left + grip);
  const ImVec2 btn_size(iconButtonWidth(), iconButtonWidth());
  const ImVec2 close_min(layout_.rect.Max.x - ImGui::GetStyle().WindowPadding.x - btn_size.x, top_left.y);
  layout_.close_btn_rect = ImRect(close_min, close_min + btn_size);
  const ImVec2 manage_min(close_min.x - btn_size.x - ImGui::GetStyle().ItemInnerSpacing.x, top_left.y);
  layout_.manage_btn_rect = ImRect(manage_min, manage_min + btn_size);

  ImFont *bold = boldFont();
  const float font_size = ImGui::GetFontSize();
  const float fm_height = ImGui::GetTextLineHeight();
  const int marker_size = markerSize();
  const int row_height = std::max<int>(marker_size, fm_height) + fm_height + 3;  // + the signal value line
  const int legend_left = layout_.move_icon_rect.Max.x + LEGEND_SPACING;
  const int legend_right = std::max<int>(layout_.manage_btn_rect.Min.x - ImGui::GetStyle().ItemInnerSpacing.x, legend_left + 10);

  // layout legend entries left-to-right, wrapping between the move icon and the buttons
  layout_.legend_rects.clear();
  int x = legend_left, y = top_left.y;
  for (auto &s : sigs_) {
    const std::string name = legendName(s);
    int w = marker_size + LEGEND_SPACING + bold->CalcTextSizeA(font_size, FLT_MAX, 0.0f, name.c_str()).x +
            ImGui::CalcTextSize(s.description().c_str()).x + 3;
    pushMonoFont(font_size);
    w = std::max(w, (int)std::ceil(ImGui::CalcTextSize("-0.00000e+000").x));
    popMonoFont();
    w = std::min(w, legend_right - legend_left);  // keep oversized entries clear of the header buttons
    if (x + w > legend_right && x > legend_left) {
      x = legend_left;
      y += row_height;
    }
    layout_.legend_rects.emplace_back(ImVec2(x, y), ImVec2(x + w, y + std::max<int>(marker_size, fm_height)));
    x += w + 12;
  }

  // add top space for the legend and signal values
  int adjust_top = (y + row_height) - top_left.y;
  adjust_top = std::max<int>(adjust_top, layout_.manage_btn_rect.Max.y - layout_.rect.Min.y + LAYOUT_MARGINS.y);
  layout_.header_bottom = top_left.y + adjust_top;
}

void ChartView::updatePlot(double cur, double min, double max) {
  cur_sec_ = cur;
  if (min != x_min_ || max != x_max_) {
    x_min_ = min;
    x_max_ = max;
    updateAxisY();
    if (tooltip_x_ >= 0) {
      showTip(secondsAtPoint({(float)tooltip_x_, 0}));
    }
  }
}

void ChartView::appendCanEvents(const cabana::Signal *sig, const std::vector<const CanEvent *> &events,
                                std::vector<ImPlotPoint> &vals) {
  vals.reserve(vals.size() + events.size());

  double value = 0;
  for (const CanEvent *e : events) {
    if (sig->getValue(e->dat, e->size, &value)) {
      const double ts = can->toSeconds(e->mono_time);
      vals.emplace_back(ts, value);
    }
  }
}

void ChartView::updateSeries(const cabana::Signal *sig, const MessageEventsMap *msg_new_events) {
  for (auto &s : sigs_) {
    if (s.path.empty() && (!sig || s.sig == sig)) {
      size_t begin = 0;
      if (!msg_new_events) s.raw_vals.clear();
      const auto &events = msg_new_events ? *msg_new_events : can->eventsMap();
      auto it = events.find(s.msg_id);
      if (msg_new_events && it == events.end()) continue;
      if (it != events.end()) {
        std::vector<ImPlotPoint> incoming;
        appendCanEvents(s.sig, it->second, incoming);
        const size_t old_size = s.raw_vals.size();
        if (old_size && (incoming.empty() || incoming.front().x >= s.raw_vals.back().x)) begin = old_size;
        s.raw_vals.insert(s.raw_vals.end(), incoming.begin(), incoming.end());
        if (begin == 0) std::inplace_merge(s.raw_vals.begin(), s.raw_vals.begin() + old_size, s.raw_vals.end(),
                          [](const auto &a, const auto &b) { return a.x < b.x; });
      }
      rebuildSeries(s, begin);
    }
  }
  updateAxisY();
}

void ChartView::rebuildSeries(SigItem &s, size_t begin) {
  buildSeries(s, begin, !can->liveStreaming());
}

void ChartView::buildSeries(SigItem &s, size_t begin, bool build_tree) {
  if (begin == 0) {
    s.vals.clear();
    s.step_vals.clear();
    s.transform_state = {};
  }
  for (size_t i = begin; i < s.raw_vals.size(); ++i) {
    const auto &pt = s.raw_vals[i];
    if (auto value = s.transform_state.append(pt.x, pt.y, s.transform)) {
      s.vals.emplace_back(pt.x, *value);
      if (!s.step_vals.empty()) s.step_vals.emplace_back(pt.x, s.step_vals.back().y);
      s.step_vals.emplace_back(pt.x, *value);
    }
  }
  if (build_tree) s.segment_tree.build(s.vals.size(), [&s](int i) { return s.vals[i].y; });
  s.track_pt = {};
}

void ChartView::configureSignal(size_t index, const chart::TransformSettings &transform, bool visible, std::optional<CabanaColor> color) {
  if (index >= sigs_.size()) return;
  auto &s = sigs_[index];
  s.transform = transform;
  if (!s.path.empty()) telemetry_dirty_ = true;
  s.visible = visible;
  if (color) s.color = *color;
  rebuildSeries(s);
  updateAxisY();
  hideTip();
}

std::string ChartView::signalUnit(const SigItem &s) {
  const std::string unit = s.path.empty() ? s.sig->unit : "";
  if (s.transform.type == chart::Transform::Derivative) return unit.empty() ? "1/s" : unit + "/s";
  if (s.transform.type == chart::Transform::Integral) return unit.empty() ? "s" : unit + "·s";
  return unit;
}

std::string ChartView::signalValue(const SigItem &s, double value) {
  return s.path.empty() && s.transform.original() ? s.sig->formatValue(value) : utils::toString(value) + " " + signalUnit(s);
}

void ChartView::drawSignalAnalysis(SigItem &s) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(s.name().c_str());
  alignRight(iconButtonWidth());
  if (iconButton("remove_signal", icon::X_LG, "Remove signal from chart")) {
    pending_signal_removal_ = &s - sigs_.data();
    ImGui::CloseCurrentPopup();
  }
  const std::string description = s.description();
  if (!description.empty()) ImGui::TextDisabled("%s", description.c_str());
  ImGui::Separator();
  auto transform = s.transform;
  int type = (int)transform.type;
  bool changed = ImGui::Combo("Transform", &type, chart::TRANSFORM_NAMES, (int)std::size(chart::TRANSFORM_NAMES));
  transform.type = (chart::Transform)type;
  changed |= ImGui::InputDouble("Scale", &transform.scale, 0, 0, "%.6g");
  changed |= ImGui::InputDouble("Offset", &transform.offset, 0, 0, "%.6g");
  if (transform.type == chart::Transform::MovingAverage) {
    changed |= ImGui::InputInt("Samples", &transform.window);
    transform.window = std::clamp(transform.window, 1, 100000);
  }
  ImGui::TextDisabled("Scale and offset apply before the transform.");
  if (ImGui::Button("Reset to original")) { transform = {}; changed = true; }
  if (changed && std::isfinite(transform.scale) && std::isfinite(transform.offset)) {
    configureSignal(&s - sigs_.data(), transform, s.visible);
  }
  ImGui::Separator();
  auto [first, last] = visibleRange(s.vals);
  ImGui::Text("Visible range: %.3f–%.3f s", x_min_, x_max_);
  ImGui::Text("Samples: %zu", (size_t)(last - first));
  if (first != last) {
    double min = first->y, max = first->y;
    long double sum = 0;
    for (auto it = first; it != last; ++it) {
      min = std::min(min, it->y);
      max = std::max(max, it->y);
      sum += it->y;
    }
    ImGui::Text("Min: %.6g   Max: %.6g", min, max);
    ImGui::Text("Mean: %.6g", (double)(sum / (last - first)));
  }
}

std::pair<ChartView::PointIter, ChartView::PointIter> ChartView::visibleRange(const std::vector<ImPlotPoint> &points) const {
  auto first = std::lower_bound(points.cbegin(), points.cend(), x_min_, xLessThan);
  auto last = std::lower_bound(first, points.cend(), x_max_, xLessThan);
  return {first, last};
}

const ImPlotPoint *ChartView::lastPointBefore(const SigItem &s, double sec) const {
  auto it = std::lower_bound(s.vals.crbegin(), s.vals.crend(), sec, [](auto &p, double x) { return p.x > x + EPSILON; });
  return it != s.vals.crend() && it->x >= x_min_ ? &*it : nullptr;
}

void ChartView::updateAxisY() {
  if (sigs_.empty()) return;

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();
  std::string unit = signalUnit(sigs_[0]);

  for (auto &s : sigs_) {
    if (!s.visible) continue;

    // Only show unit when all signals have the same unit
    if (unit != signalUnit(s)) {
      unit.clear();
    }

    auto [first, last] = visibleRange(s.vals);
    s.min = std::numeric_limits<double>::max();
    s.max = std::numeric_limits<double>::lowest();
    if (first == last) continue;
    if (can->liveStreaming()) {
      for (auto it = first; it != last; ++it) {
        if (it->y < s.min) s.min = it->y;
        if (it->y > s.max) s.max = it->y;
      }
    } else {
      std::tie(s.min, s.max) = s.segment_tree.minmax(std::distance(s.vals.cbegin(), first), std::distance(s.vals.cbegin(), last) - 1);
    }
    min = std::min(min, s.min);
    max = std::max(max, s.max);
  }
  if (min == std::numeric_limits<double>::max()) min = 0;
  if (max == std::numeric_limits<double>::lowest()) max = 0;

  y_unit_ = unit;

  if (limit_min) min = *limit_min;
  if (limit_max) max = *limit_max;
  if (max < min) max = min + 1;
  const double magnitude = std::max(std::abs(min), std::abs(max));
  double delta = max - min <= magnitude * 1e-9 ? (magnitude > 0 ? magnitude * 0.05 : 1) : (max - min) * 0.05;
  auto [min_y, max_y, tick_count] = getNiceAxisNumbers(min - delta, max + delta, 3);
  if (limit_min) min_y = *limit_min;
  if (limit_max) max_y = *limit_max;
  if (max_y <= min_y) max_y = min_y + 1;
  if (min_y != y_min_ || max_y != y_max_) {
    y_min_ = min_y;
    y_max_ = max_y;
    y_tick_count_ = tick_count;
    y_precision_ = axisPrecision(max_y - min_y, tick_count, 0);
  }
}

std::tuple<double, double, int> ChartView::getNiceAxisNumbers(double min, double max, int tick_count) {
  double range = niceNumber((max - min), true);  // range with ceiling
  double step = niceNumber(range / (tick_count - 1), false);
  min = std::floor(min / step);
  max = std::ceil(max / step);
  tick_count = int(max - min) + 1;
  return {min * step, max * step, tick_count};
}

int ChartView::xAxisPrecision() const {
  return axisPrecision(x_max_ - x_min_, X_TICK_COUNT, 2);
}

// nice numbers can be expressed as form of 1*10^n, 2* 10^n or 5*10^n
double ChartView::niceNumber(double x, bool ceiling) {
  double z = std::pow(10, std::floor(std::log10(x)));  // the largest 10^n smaller than x
  double q = x / z;  // 1 <= q < 10
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

void ChartView::drawContextMenu() {
  if (drawing_ghost_) return;
  // the menu opens on right press; a right release with no menu open reaches handleMouseRelease
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && layout_.plot_area.Contains(ImGui::GetMousePos()) &&
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !ImGui::IsAnyItemActive() && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
    ImGui::OpenPopup("context_menu");
  }
  if (ImGui::BeginPopup("context_menu")) {
    drawMenuActions();
    // the menu holds checkable entries, so every entry keeps the same left margin
    const float indent = ImGui::GetFontSize();
    ImGui::Indent(indent);
    ImGui::Separator();
    // the zoom entries come from the toolbar, where they are only visible while zoomed
    if (can->timeRange().has_value()) {
      const std::string undo_text = std::string(icon::ARROW_COUNTERCLOCKWISE) + " Undo Zoom";
      const std::string redo_text = std::string(icon::ARROW_CLOCKWISE) + " Redo Zoom";
      if (ImGui::MenuItem(undo_text.c_str(), nullptr, false, charts_widget_->zoom_undo_stack_.canUndo())) charts_widget_->zoom_undo_stack_.undo();
      if (ImGui::MenuItem(redo_text.c_str(), nullptr, false, charts_widget_->zoom_undo_stack_.canRedo())) charts_widget_->zoom_undo_stack_.redo();
      ImGui::Separator();
    }
    if (ImGui::MenuItem("Close")) charts_widget_->removeChart(this);
    ImGui::Unindent(indent);
    ImGui::EndPopup();
  }
}

void ChartView::handleMousePress() {
  if (drawing_ghost_) return;
  const ImVec2 pos = ImGui::GetMousePos();
  // a press on the close/manage buttons does not reach the widget
  const bool widget_pressed = !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) &&
                              ImGui::IsMouseClicked(ImGuiMouseButton_Left) && layout_.rect.Contains(pos) &&
                              ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
                              !layout_.close_btn_rect.Contains(pos) && !layout_.manage_btn_rect.Contains(pos);
  if (!widget_pressed) return;
  press_pos_ = pos;
  if (layout_.move_icon_rect.Contains(pos)) return;  // the move icon press is handled by the grip item (startChartDrag)

  if (ImGui::GetIO().KeyCtrl && layout_.plot_area.Contains(pos)) {
    mouse_mode_ = MouseMode::Pan;
    pan_range_ = {x_min_, x_max_};
    pan_previous_ = can->timeRange();
  } else if (ImGui::GetIO().KeyShift) {
    // Save current playback state when scrubbing
    resume_after_scrub_ = !can->isPaused();
    if (resume_after_scrub_) {
      can->pause(true);
    }
    mouse_mode_ = MouseMode::Scrub;
  } else if (layout_.plot_area.Contains(pos)) {
    mouse_mode_ = MouseMode::Rubber;
    rubber_rect_ = ImRect();
  }
}

void ChartView::handleMouseMove() {
  if (drawing_ghost_) return;
  const ImVec2 pos = ImGui::GetMousePos();
  const ImVec2 delta = ImGui::GetIO().MouseDelta;
  // a click alone must not hide the tip
  if (delta.x == 0 && delta.y == 0) return;
  // only the widget under the mouse, or the one dragging, reacts to a move
  if (mouse_mode_ == MouseMode::None && !layout_.rect.Contains(pos)) return;

  if (mouse_mode_ == MouseMode::Scrub && ImGui::GetIO().KeyShift) {
    if (layout_.plot_area.Contains(pos)) {
      can->seekTo(std::clamp(secondsAtPoint(pos), can->minSeconds(), can->maxSeconds()));
    }
  }

  if (mouse_mode_ == MouseMode::Pan) {
    const double width = pan_range_.second - pan_range_.first;
    const double shift = (press_pos_.x - pos.x) * width / std::max(layout_.plot_area.GetWidth(), 1.0f);
    const double left = std::clamp(pan_range_.first + shift, can->minSeconds(), std::max(can->minSeconds(), can->maxSeconds() - width));
    can->setTimeRange(std::make_pair(left, left + width), false);
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
  }

  if (mouse_mode_ == MouseMode::Rubber) {
    // horizontal selection, clamped to the plot area
    float left = std::clamp(std::min(press_pos_.x, pos.x), layout_.plot_area.Min.x, layout_.plot_area.Max.x);
    float right = std::clamp(std::max(press_pos_.x, pos.x), layout_.plot_area.Min.x, layout_.plot_area.Max.x);
    rubber_rect_ = ImRect(ImVec2(left, layout_.plot_area.Min.y), ImVec2(right, layout_.plot_area.Max.y));
  }

  clearTrackPoints();
  if (mouse_mode_ != MouseMode::Rubber && layout_.plot_area.Contains(pos) && (layout_.plot_hovered || mouse_mode_ != MouseMode::None) &&
      ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow)) {
    charts_widget_->showValueTip(secondsAtPoint(pos));
  } else if (tip_label_.isVisible()) {
    charts_widget_->showValueTip(-1);
  }
}

void ChartView::handleMouseRelease() {
  if (drawing_ghost_) return;
  const bool left_released = ImGui::IsMouseReleased(ImGuiMouseButton_Left);
  const bool right_released = ImGui::IsMouseReleased(ImGuiMouseButton_Right) && layout_.plot_area.Contains(ImGui::GetMousePos());
  if (!left_released && !right_released) return;
  if (left_released && mouse_mode_ == MouseMode::Pan) {
    mouse_mode_ = MouseMode::None;
    const auto range = can->timeRange();
    if (range && range != pan_previous_) charts_widget_->zoom_undo_stack_.push(new ZoomCommand(*range, pan_previous_));
  } else if (left_released && mouse_mode_ == MouseMode::Rubber) {
    mouse_mode_ = MouseMode::None;
    double min = std::clamp(secondsAtPoint(rubber_rect_.Min), can->minSeconds(), can->maxSeconds());
    double max = std::clamp(secondsAtPoint(rubber_rect_.Max), can->minSeconds(), can->maxSeconds());
    if (rubber_rect_.GetWidth() <= 10) {
      // Small movements are still clicks; use the same threshold as drag-to-zoom.
      can->seekTo(std::clamp(secondsAtPoint(press_pos_), can->minSeconds(), can->maxSeconds()));
    } else if ((max - min) > MIN_ZOOM_SECONDS) {
      charts_widget_->zoom_undo_stack_.push(new ZoomCommand({min, max}));
    }
    rubber_rect_ = ImRect();
  } else if (right_released && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
    charts_widget_->zoom_undo_stack_.undo();
  }

  if (mouse_mode_ == MouseMode::Scrub) {
    mouse_mode_ = MouseMode::None;
    if (resume_after_scrub_) {
      can->pause(false);
      resume_after_scrub_ = false;
    }
  }
}

void ChartView::takeSignalsFrom(ChartView *source) {
  for (auto &s : source->sigs_) {
    s.color = uniqueColor(s.color);
    sigs_.push_back(std::move(s));
  }
  source->sigs_.clear();
  updateTelemetry();
  updateAxisY();
  charts_widget_->removeChart(source);
}

std::vector<ChartView::SigItem> ChartView::takeExtraSignals() {
  std::vector<SigItem> extra;
  for (auto it = sigs_.begin() + 1; it != sigs_.end(); ++it) {
    if (it->sig) it->color = it->sig->color;
    extra.push_back(std::move(*it));
  }
  sigs_.resize(1);
  updateAxisY();
  return extra;
}

void ChartView::adoptSignal(SigItem s) {
  sigs_.push_back(std::move(s));
  updateTelemetry();
  updateAxisY();
}

void ChartView::showTip(double sec) {
  ImRect tip_area(ImVec2(layout_.rect.Min.x, layout_.plot_area.Min.y), ImVec2(layout_.rect.Max.x, layout_.plot_area.Max.y));
  ImRect visible_rect = charts_widget_->chartVisibleRect(this);
  visible_rect.ClipWith(tip_area);
  if (visible_rect.GetWidth() <= 0 || visible_rect.GetHeight() <= 0) {
    tip_label_.hide();
    return;
  }

  tooltip_x_ = xPos(sec);
  float x = -1;
  std::vector<TipLine> text_list;
  for (auto &s : sigs_) {
    if (s.visible) {
      std::string value = "--";
      if (const ImPlotPoint *pt = lastPointBefore(s, sec)) {
        value = utils::trimmed(signalValue(s, pt->y));
        s.track_pt = *pt;
        x = std::max(x, xPos(pt->x));
      }
      std::string name = legendName(s);
      std::string min = s.min == std::numeric_limits<double>::max() ? "--" : utils::toString(s.min);
      std::string max = s.max == std::numeric_limits<double>::lowest() ? "--" : utils::toString(s.max);
      text_list.push_back({.has_marker = true, .marker = toImU32(s.color), .name = name, .value = value, .min = min, .max = max});
    }
  }
  if (x < 0) {
    x = tooltip_x_;
  }
  ImVec2 pt(x, layout_.plot_area.Min.y);
  text_list.insert(text_list.begin(), TipLine{.name = formatNumber(sec, 2) + " s"});
  tip_label_.showText(pt, text_list, visible_rect);
}

void ChartView::hideTip() {
  clearTrackPoints();
  tooltip_x_ = -1;
  tip_label_.hide();
}

void ChartView::draw(float width) {
  ImGui::PushID(this);
  width = std::max(width, (float)CHART_MIN_WIDTH);
  layout_.plot_hovered = false;
  // the tile geometry is known before the child is entered, so it stays valid when imgui culls a scrolled out chart
  const ImVec2 tile_pos = ImGui::GetCursorScreenPos();
  ImVec2 tile_size(width, (float)settings.chart_height);
  layout_.rect = ImRect(tile_pos, tile_pos + tile_size);
  updateLayout();
  tile_size.y = std::max(tile_size.y, layout_.header_bottom - tile_pos.y + 140.0f);
  layout_.rect.Max.y = tile_pos.y + tile_size.y;
  if (ImGui::BeginChild("chart", tile_size, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
    updateLayout();
    paint();
    drawContextMenu();
  }
  ImGui::EndChild();
  // a chart scrolled out of the viewport draws no tip
  const ImRect visible_rect = charts_widget_->chartVisibleRect(this);
  if (!drawing_ghost_ && visible_rect.GetWidth() > 0 && visible_rect.GetHeight() > 0) tip_label_.draw();
  ImGui::PopID();
  if (pending_signal_removal_ >= 0) {
    const int index = std::exchange(pending_signal_removal_, -1);
    int i = 0;
    removeIf([index, &i](const SigItem &) { return i++ == index; });
  }
}

void ChartView::drawGhost(float width) {
  // the ghost is drawn in its own window: keep the geometry of the live tile so hit testing stays correct
  drawing_ghost_ = true;
  const Layout saved = layout_;
  draw(width);
  layout_ = saved;
  drawing_ghost_ = false;
}

void ChartView::paint() {
  drawStaticLayer();

  if (can_drop_) {
    ImGui::GetWindowDrawList()->AddRect(layout_.rect.Min, layout_.rect.Max, ImGui::GetColorU32(ImGuiCol_Header), ImGui::GetStyle().ChildRounding, 0, 4.0f);
  }
}

void ChartView::drawStaticLayer() {
  ImDrawList *painter = ImGui::GetWindowDrawList();
  painter->AddRectFilled(layout_.rect.Min, layout_.rect.Max, ImGui::GetColorU32(ImGuiCol_ChildBg), ImGui::GetStyle().ChildRounding);
  ImGui::SetCursorScreenPos(layout_.move_icon_rect.Min);
  ImGui::InvisibleButton("grip", layout_.move_icon_rect.GetSize());
  if (ImGui::IsItemActivated()) charts_widget_->startChartDrag(this, ImGui::GetMousePos());
  if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
  painter->AddText(layout_.move_icon_rect.Min, ImGui::GetColorU32(ImGuiCol_Text), icon::GRIP_HORIZONTAL);
  if (!title.empty()) {
    addTextEllipsis(painter, boldFont(), ImGui::GetColorU32(ImGuiCol_Text),
      layout_.rect.Min + ImVec2(6, 4), layout_.rect.Max.x - 6, title);
  }
  createToolButtons();
  drawLegend();
  drawSignalValue();  // drawn here because implot clips the plot frame
  drawAxes();
}

void ChartView::drawAxes() {
  ImGui::SetCursorScreenPos(ImVec2(layout_.rect.Min.x, layout_.header_bottom));
  const float plot_h = std::max(layout_.rect.Max.y - layout_.header_bottom - LAYOUT_MARGINS.w, 10.0f);
  ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(LAYOUT_MARGINS.x, AXIS_X_TOP_MARGIN));
  ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0, 0, 0, 0));
  ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0, 0, 0, 0));
  ImPlot::PushStyleColor(ImPlotCol_PlotBorder, palette().grid);
  ImPlot::PushStyleColor(ImPlotCol_AxisTick, ImVec4(0, 0, 0, 0));
  ImPlot::PushStyleColor(ImPlotCol_AxisText, ImGui::GetStyleColorVec4(ImGuiCol_Text));
  ImPlot::PushStyleVar(ImPlotStyleVar_MajorTickLen, ImVec2(0, 0));
  ImPlot::PushStyleVar(ImPlotStyleVar_MajorGridSize, ImVec2(1.0f, 1.0f));
  const ImPlotFlags flags = ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoMouseText |
                            ImPlotFlags_NoBoxSelect | ImPlotFlags_NoInputs | ImPlotFlags_NoFrame;
  const ImPlotAxisFlags axis_flags = ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_Lock;
  // reserve room for the right half of the last x tick label
  const float x_label_width = ImGui::CalcTextSize(formatNumber(x_max_, xAxisPrecision()).c_str()).x + 5;
  if (ImPlot::BeginPlot("##plot", ImVec2(layout_.rect.GetWidth() - x_label_width / 2, plot_h), flags)) {
    ImPlot::SetupAxis(ImAxis_X1, nullptr, axis_flags);
    ImPlot::SetupAxis(ImAxis_Y1, y_unit_.empty() ? nullptr : y_unit_.c_str(), axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_X1, x_min_, x_max_, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, y_min_, y_max_, ImPlotCond_Always);
    // the format must be set before the ticks are generated
    ImPlot::SetupAxisFormat(ImAxis_Y1, ("%." + std::to_string(y_precision_) + "f").c_str());
    ImPlot::SetupAxisTicks(ImAxis_Y1, y_min_, y_max_, y_tick_count_);
    ImPlot::SetupAxisFormat(ImAxis_X1, ("%." + std::to_string(xAxisPrecision()) + "f").c_str());
    ImPlot::SetupAxisTicks(ImAxis_X1, x_min_, x_max_, X_TICK_COUNT);
    ImPlot::SetupFinish();

    layout_.plot_area = ImRect(ImPlot::GetPlotPos(), ImPlot::GetPlotPos() + ImPlot::GetPlotSize());
    // ImPlotFlags_NoInputs disables implot's own hover tracking
    layout_.plot_hovered = layout_.plot_area.Contains(ImGui::GetMousePos()) && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
    drawSeries();
    if (sigs_.empty()) {
      drawText(ImGui::GetWindowDrawList(), layout_.plot_area, "Drag signals here to plot", ImGui::GetColorU32(ImGuiCol_TextDisabled));
    }
    if (!drawing_ghost_ && layout_.plot_hovered && ImGui::GetIO().KeyCtrl) {
      ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, ImGui::GetID("##plot"));
    }
    if (!drawing_ghost_ && layout_.plot_hovered && ImGui::GetIO().KeyCtrl && ImGui::GetIO().MouseWheel != 0 &&
        mouse_mode_ == MouseMode::None) {
      const double duration = can->maxSeconds() - can->minSeconds();
      if (duration > MIN_ZOOM_SECONDS) {
        const double anchor = std::clamp(secondsAtPoint(ImGui::GetMousePos()), x_min_, x_max_);
        const double fraction = (anchor - x_min_) / (x_max_ - x_min_);
        const double width = std::clamp((x_max_ - x_min_) * std::pow(0.8, ImGui::GetIO().MouseWheel), MIN_ZOOM_SECONDS, duration);
        const double left = std::clamp(anchor - fraction * width, can->minSeconds(), can->maxSeconds() - width);
        charts_widget_->zoom_undo_stack_.push(new ZoomCommand({left, left + width}));
      }
    }
    handleMousePress();
    handleMouseMove();
    handleMouseRelease();
    drawForeground();
    ImPlot::EndPlot();
  }
  if (!drawing_ghost_ && ImGui::BeginDragDropTargetCustom(layout_.rect, ImGui::GetID("signal_drop"))) {
    if (const auto *payload = ImGui::AcceptDragDropPayload("CABANA_TELEMETRY")) {
      addTelemetry(static_cast<const char *>(payload->Data));
      charts_widget_->updateState();
    }
    ImGui::EndDragDropTarget();
  }
  ImPlot::PopStyleColor(5);
  ImPlot::PopStyleVar(3);
}

void ChartView::drawLegend() {
  ImDrawList *painter = ImGui::GetWindowDrawList();
  const ImU32 title_color = ImGui::GetColorU32(ImGuiCol_Text);
  // Draw message details in similar color, but slightly fade it to the background
  const ImU32 msg_color = withAlpha(title_color, 180);
  ImFont *bold = boldFont();
  ImFont *normal = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const float marker_size = markerSize();

  for (int i = 0; i < sigs_.size() && i < layout_.legend_rects.size(); ++i) {
    const auto &s = sigs_[i];
    const ImRect &r = layout_.legend_rects[i];
    // toggle series visibility by clicking its legend entry
    ImGui::PushID(i);
    ImGui::SetCursorScreenPos(r.Min);
    if (ImGui::InvisibleButton("legend", ImVec2(std::max(r.GetWidth(), 1.0f), std::max(r.GetHeight(), 1.0f))) &&
        mouse_mode_ == MouseMode::None && sigs_.size() > 1) {
      sigs_[i].visible = !sigs_[i].visible;
      updateAxisY();
    }
    ImGui::SetItemTooltip("%s%s\nClick to show/hide · Right-click for transforms and statistics",
                          s.name().c_str(), s.transform.original() ? "" : " (transformed)");
    if (!drawing_ghost_ && ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
      ImGui::OpenPopup("signal_analysis");
    }
    if (ImGui::BeginPopup("signal_analysis")) {
      drawSignalAnalysis(sigs_[i]);
      ImGui::EndPopup();
    }
    ImGui::PopID();

    if (series_type_ == SeriesType::Scatter) {
      painter->AddCircleFilled(r.Min + ImVec2(marker_size / 2.0f, 2.0f + marker_size / 2.0f), marker_size / 2.0f, toImU32(s.color));
    } else {
      drawColorMarker(painter, r.Min, toImU32(s.color));
    }

    float x = r.Min.x + marker_size + LEGEND_SPACING;
    const float text_y = r.GetCenter().y - font_size / 2.0f;
    const std::string name = legendName(s);
    addTextEllipsis(painter, bold, title_color, ImVec2(x, text_y), r.Max.x, name);
    float name_w = std::min(bold->CalcTextSizeA(font_size, FLT_MAX, 0.0f, name.c_str()).x, r.Max.x - x);
    x += name_w;
    const std::string msg = s.description();
    addTextEllipsis(painter, normal, msg_color, ImVec2(x, text_y), r.Max.x, msg);
    if (!s.visible) {  // strike out
      const float y = r.GetCenter().y;
      painter->AddLine(ImVec2(r.Min.x + marker_size + LEGEND_SPACING, y), ImVec2(std::min(x + ImGui::CalcTextSize(msg.c_str()).x, r.Max.x), y), title_color);
    }
  }
}

std::string ChartView::legendName(const SigItem &s) const {
  std::string label = s.label();
  if (!s.path.empty() && std::any_of(sigs_.begin(), sigs_.end(), [&](const auto &other) {
        return &other != &s && other.label() == label;
      })) label = s.name();
  return label + (s.transform.original() ? "" : " *");
}

void ChartView::drawSeries() {
  for (int i = 0; i < sigs_.size(); ++i) {
    auto &s = sigs_[i];
    if (!s.visible) continue;

    // visible points in vals to compute point density
    auto [first, last] = visibleRange(s.vals);
    int num_points = std::max<int>(last - first, 1);
    double pixels_per_point = 0;
    if (first != last) {
      const ImPlotPoint &right_pt = last == s.vals.cend() ? s.vals.back() : *last;
      pixels_per_point = (xPos(right_pt.x) - xPos(first->x)) / num_points;
    }

    const std::string label = "##sig" + std::to_string(i);
    ImPlotSpec spec;
    spec.LineColor = toImVec4(s.color);
    spec.Stride = sizeof(ImPlotPoint);
    if (series_type_ == SeriesType::Scatter) {
      float radius = std::clamp(pixels_per_point / 2.0, 2.0, 8.0) / 2.0;
      spec.Marker = ImPlotMarker_Circle;
      spec.MarkerSize = radius;
      if (first != last) ImPlot::PlotScatter(label.c_str(), &first->x, &first->y, last - first, spec);
    } else {
      const auto &points = series_type_ == SeriesType::StepLine ? s.step_vals : s.vals;
      // one sample beyond each edge so the line runs out of the plot
      auto [begin, end] = visibleRange(points);
      if (begin != points.cbegin()) --begin;
      if (end != points.cend()) ++end;
      if (begin == end) continue;

      spec.LineWeight = 2;
      const int pixels = std::max(1, (int)layout_.plot_area.GetWidth());
      if (end - begin > pixels * 4) {
        const auto envelope = chart::pixelEnvelope(begin, end, x_min_, x_max_, pixels);
        ImPlot::PlotLine(label.c_str(), &envelope.front().x, &envelope.front().y, envelope.size(), spec);
      } else {
        ImPlot::PlotLine(label.c_str(), &begin->x, &begin->y, end - begin, spec);
      }

      // show points when zoomed in enough
      if ((num_points == 1 || pixels_per_point > 20) && first != last) {
        ImPlotSpec dots;
        dots.LineColor = toImVec4(s.color);
        dots.Stride = sizeof(ImPlotPoint);
        dots.Marker = ImPlotMarker_Circle;
        dots.MarkerSize = 4;
        ImPlot::PlotScatter((label + "_pts").c_str(), &first->x, &first->y, last - first, dots);
      }
    }
  }
}

void ChartView::drawForeground() {
  drawTimeline();
  ImDrawList *painter = ImPlot::GetPlotDrawList();
  ImPlot::PushPlotClipRect();
  float track_line_x = -1;
  for (auto &s : sigs_) {
    if (!isNull(s.track_pt) && s.visible) {
      ImVec2 pos(xPos(s.track_pt.x), yPos(s.track_pt.y));
      painter->AddCircleFilled(pos, 5.5f, toImU32(s.color.darker(125)));
      track_line_x = std::max(track_line_x, pos.x);
    }
  }
  if (track_line_x > 0) {
    const ImU32 dark_gray = IM_COL32(0x80, 0x80, 0x80, 0xff);
    for (float y = layout_.plot_area.Min.y; y < layout_.plot_area.Max.y; y += 8) {
      painter->AddLine(ImVec2(track_line_x, y), ImVec2(track_line_x, std::min(y + 4, layout_.plot_area.Max.y)), dark_gray, 1.0f);
    }
  }
  ImPlot::PopPlotClipRect();

  drawRubberBandTimeRange();
}

void ChartView::drawRubberBandTimeRange() {
  if (rubber_rect_.GetWidth() <= 1) return;

  ImDrawList *painter = ImPlot::GetPlotDrawList();
  // ImGuiCol_Header is translucent, so the 1px selection outline is drawn at full alpha
  const ImU32 highlight = withAlpha(ImGui::GetColorU32(ImGuiCol_Header), 255);
  painter->AddRectFilled(rubber_rect_.Min, rubber_rect_.Max, withAlpha(highlight, 50), ImGui::GetStyle().FrameRounding);
  painter->AddRect(rubber_rect_.Min, rubber_rect_.Max, highlight, ImGui::GetStyle().FrameRounding);

  // time labels at the bottom corners (below the plot, so clip to the widget instead of the plot)
  const ImU32 white = IM_COL32_WHITE;
  const ImU32 badge = ImGui::GetColorU32(palette().badge);
  painter = ImGui::GetWindowDrawList();
  painter->PushClipRect(layout_.rect.Min, layout_.rect.Max);
  for (const auto &pt : {rubber_rect_.GetBL(), rubber_rect_.GetBR()}) {
    std::string sec = formatNumber(secondsAtPoint(pt), 2);
    ImVec2 size = ImGui::CalcTextSize(sec.c_str()) + ImVec2(12, AXIS_X_TOP_MARGIN * 2);
    ImVec2 top_left = pt.x == rubber_rect_.Min.x ? ImVec2(pt.x - size.x, pt.y + 2) : ImVec2(pt.x, pt.y + 2);
    painter->AddRectFilled(top_left, top_left + size, badge, ImGui::GetStyle().FrameRounding);
    painter->AddText(top_left + ImVec2(6, AXIS_X_TOP_MARGIN), white, sec.c_str());
  }
  painter->PopClipRect();
}

void ChartView::drawTimeline() {
  ImDrawList *painter = ImPlot::GetPlotDrawList();
  float x = std::clamp(xPos(cur_sec_), layout_.plot_area.Min.x, layout_.plot_area.Max.x);
  painter->AddLine(ImVec2(x, layout_.plot_area.Min.y - 1.0f), ImVec2(x, layout_.plot_area.Max.y + 1.0f), ImGui::GetColorU32(ImGuiCol_Text), 1.0f);

  std::string time_str = formatNumber(cur_sec_, 2);
  ImVec2 time_str_size = ImGui::CalcTextSize(time_str.c_str()) + ImVec2(8, 2);
  ImVec2 time_str_pos(x - time_str_size.x / 2.0f, layout_.plot_area.Max.y + AXIS_X_TOP_MARGIN);
  painter->AddRectFilled(time_str_pos, time_str_pos + time_str_size, ImGui::GetColorU32(palette().badge), ImGui::GetStyle().FrameRounding);
  painter->AddText(time_str_pos + ImVec2(4, 1), IM_COL32_WHITE, time_str.c_str());
}

void ChartView::drawSignalValue() {
  pushMonoFont(ImGui::GetFontSize());
  ImDrawList *painter = ImGui::GetWindowDrawList();
  const ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
  for (int i = 0; i < sigs_.size() && i < layout_.legend_rects.size(); ++i) {
    const auto &s = sigs_[i];
    const ImPlotPoint *pt = lastPointBefore(s, cur_sec_);
    std::string value = pt ? signalValue(s, pt->y) : (s.path.empty() ? "--" : "No data");
    const ImVec2 value_min = layout_.legend_rects[i].GetBL() - ImVec2(0, 1);
    ImRect value_rect(value_min, value_min + layout_.legend_rects[i].GetSize());
    float w = ImGui::CalcTextSize(value.c_str()).x;
    if (w <= value_rect.GetWidth()) {
      painter->AddText(value_rect.Min, color, value.c_str());
    } else {
      addTextEllipsis(painter, ImGui::GetFont(), color, value_rect.Min, value_rect.Max.x, value);
    }
  }
  popMonoFont();
}

CabanaColor ChartView::uniqueColor(CabanaColor color, const cabana::Signal *exclude) const {
  auto separation = [&](float hue) {
    float distance = 1.0f;
    for (const auto &s : sigs_) {
      if (exclude && s.sig == exclude) continue;
      const float delta = std::abs(hue - s.color.hsv().hue);
      distance = std::min(distance, std::min(delta, 1.0f - delta));
    }
    return distance;
  };
  const float original_hue = color.hsv().hue;
  if (separation(original_hue) >= 0.1f) return color;

  float best_hue = original_hue, best_distance = -1;
  for (int i = 0; i < 36; ++i) {
    const float hue = std::fmod(original_hue + i / 36.0f, 1.0f);
    const float distance = separation(hue);
    if (distance > best_distance) {
      best_hue = hue;
      best_distance = distance;
    }
  }
  return CabanaColor::fromHsv(best_hue, 0.8f, 0.9f, color.a / 255.0f);
}
