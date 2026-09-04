#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/chart.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

#include "tools/cabana/core/settings.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/chart/chartswidget.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

const int AXIS_X_TOP_MARGIN = 4;
const int X_TICK_COUNT = 5;
const double MIN_ZOOM_SECONDS = 0.01;  // 10ms
const double EPSILON = 1e-6;
constexpr ImVec4 LAYOUT_MARGINS{8, 6, 8, 6};  // left, top, right, bottom
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
  if (ImGui::MenuItem("Split Chart", nullptr, false, sigs_.size() > 1)) charts_widget_->splitChart(this);
  ImGui::Unindent(indent);
}

// the buttons and their menus are drawn every frame, at the rects updateLayout() placed them at
void ChartView::createToolButtons() {
  ImGui::SetCursorScreenPos(layout_.close_btn_rect.Min);
  bool close_clicked = toolButton("close_btn", icon::X, "Remove Chart");

  ImGui::SetCursorScreenPos(layout_.manage_btn_rect.Min);
  if (toolButton("manage_btn", icon::LIST, "")) ImGui::OpenPopup("manage_menu");
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

bool ChartView::hasSignal(const MessageId &msg_id, const cabana::Signal *sig) const {
  return std::any_of(sigs_.cbegin(), sigs_.cend(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
}

void ChartView::removeIf(std::function<bool(const SigItem &s)> predicate) {
  int prev_size = sigs_.size();
  sigs_.erase(std::remove_if(sigs_.begin(), sigs_.end(), predicate), sigs_.end());
  if (sigs_.empty()) {
    charts_widget_->removeChart(this);
  } else if (sigs_.size() != prev_size) {
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
    dlg->addSelected(s.msg_id, s.sig);
  }
  // runs once the dialog is accepted, dropped if the chart is removed first
  charts_widget_->execSignalSelector(std::move(dlg), this, [this](SignalSelector &selector) {
    const auto &items = selector.selectedItems();
    for (const auto &s : items) {
      addSignal(s.msg_id, s.sig);
    }
    removeIf([&](auto &s) {
      return std::none_of(items.cbegin(), items.cend(), [&](auto &it) { return s.msg_id == it.msg_id && s.sig == it.sig; });
    });
  });
}

void ChartView::updateLayout() {
  const ImVec2 grip = ImGui::CalcTextSize(icon::GRIP_HORIZONTAL);
  const ImVec2 top_left = layout_.rect.Min + ImVec2(LAYOUT_MARGINS.x, LAYOUT_MARGINS.y);
  layout_.move_icon_rect = ImRect(top_left, top_left + grip);
  const ImVec2 pad = ImGui::GetStyle().FramePadding * 2;
  const ImVec2 close_size = ImGui::CalcTextSize(icon::X) + pad;
  const ImVec2 manage_size = ImGui::CalcTextSize(icon::LIST) + pad;
  const ImVec2 close_min(layout_.rect.Max.x - LAYOUT_MARGINS.z - close_size.x, top_left.y);
  layout_.close_btn_rect = ImRect(close_min, close_min + close_size);
  const ImVec2 manage_min(close_min.x - manage_size.x - ImGui::GetStyle().ItemSpacing.x, top_left.y);
  layout_.manage_btn_rect = ImRect(manage_min, manage_min + manage_size);

  ImFont *bold = boldFont();
  const float font_size = ImGui::GetFontSize();
  const float fm_height = ImGui::GetTextLineHeight();
  const int marker_size = markerSize();
  const int row_height = std::max<int>(marker_size, fm_height) + fm_height + 3;  // + the signal value line
  const int legend_left = layout_.move_icon_rect.Max.x + LAYOUT_MARGINS.x;
  const int legend_right = std::max<int>(layout_.manage_btn_rect.Min.x - LAYOUT_MARGINS.z, legend_left + 10);

  // layout legend entries left-to-right, wrapping between the move icon and the buttons
  layout_.legend_rects.clear();
  int x = legend_left, y = top_left.y;
  for (auto &s : sigs_) {
    int w = marker_size + 5 + bold->CalcTextSizeA(font_size, FLT_MAX, 0.0f, s.sig->name.c_str()).x +
            ImGui::CalcTextSize(msgLabel(s.msg_id).c_str()).x;
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
  layout_.header_bottom = layout_.rect.Min.y + adjust_top + LAYOUT_MARGINS.y;
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
                                std::vector<ImPlotPoint> &vals, std::vector<ImPlotPoint> &step_vals) {
  vals.reserve(vals.size() + events.size());
  step_vals.reserve(step_vals.size() + events.size() * 2);

  double value = 0;
  for (const CanEvent *e : events) {
    if (sig->getValue(e->dat, e->size, &value)) {
      const double ts = can->toSeconds(e->mono_time);
      vals.emplace_back(ts, value);
      if (!step_vals.empty())
        step_vals.emplace_back(ts, step_vals.back().y);
      step_vals.emplace_back(ts, value);
    }
  }
}

void ChartView::updateSeries(const cabana::Signal *sig, const MessageEventsMap *msg_new_events) {
  for (auto &s : sigs_) {
    if (!sig || s.sig == sig) {
      if (!msg_new_events) {
        s.vals.clear();
        s.step_vals.clear();
      }
      auto events = msg_new_events ? msg_new_events : &can->eventsMap();
      auto it = events->find(s.msg_id);
      if (it == events->end() || it->second.empty()) continue;

      if (s.vals.empty() || can->toSeconds(it->second.back()->mono_time) > s.vals.back().x) {
        appendCanEvents(s.sig, it->second, s.vals, s.step_vals);
      } else {
        std::vector<ImPlotPoint> vals, step_vals;
        appendCanEvents(s.sig, it->second, vals, step_vals);
        if (vals.empty()) continue;
        s.vals.insert(std::lower_bound(s.vals.begin(), s.vals.end(), vals.front().x, xLessThan),
                      vals.begin(), vals.end());
        s.step_vals.insert(std::lower_bound(s.step_vals.begin(), s.step_vals.end(), step_vals.front().x, xLessThan),
                           step_vals.begin(), step_vals.end());
      }

      if (!can->liveStreaming()) {
        s.segment_tree.build(s.vals.size(), [&vals = s.vals](int i) { return vals[i].y; });
      }
    }
  }
  updateAxisY();
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
  std::string unit = sigs_[0].sig->unit;

  for (auto &s : sigs_) {
    if (!s.visible) continue;

    // Only show unit when all signals have the same unit
    if (unit != s.sig->unit) {
      unit.clear();
    }

    auto [first, last] = visibleRange(s.vals);
    s.min = std::numeric_limits<double>::max();
    s.max = std::numeric_limits<double>::lowest();
    if (can->liveStreaming()) {
      for (auto it = first; it != last; ++it) {
        if (it->y < s.min) s.min = it->y;
        if (it->y > s.max) s.max = it->y;
      }
    } else {
      std::tie(s.min, s.max) = s.segment_tree.minmax(std::distance(s.vals.cbegin(), first), std::distance(s.vals.cbegin(), last));
    }
    min = std::min(min, s.min);
    max = std::max(max, s.max);
  }
  if (min == std::numeric_limits<double>::max()) min = 0;
  if (max == std::numeric_limits<double>::lowest()) max = 0;

  y_unit_ = unit;

  double delta = std::abs(max - min) < 1e-3 ? 1 : (max - min) * 0.05;
  auto [min_y, max_y, tick_count] = getNiceAxisNumbers(min - delta, max + delta, 3);
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
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !ImGui::IsAnyItemActive()) {
    ImGui::OpenPopup("context_menu");
  }
  context_menu_id_ = ImGui::GetID("context_menu");
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
  const bool widget_pressed = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && layout_.rect.Contains(pos) &&
                              ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
                              !layout_.close_btn_rect.Contains(pos) && !layout_.manage_btn_rect.Contains(pos);
  if (!widget_pressed) return;
  press_pos_ = pos;
  if (layout_.move_icon_rect.Contains(pos)) return;  // the move icon press is handled by the grip item (startChartDrag)

  if (ImGui::GetIO().KeyShift) {
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
  const bool right_released = ImGui::IsMouseReleased(ImGuiMouseButton_Right) && layout_.rect.Contains(ImGui::GetMousePos());
  if (!left_released && !right_released) return;
  if (left_released && mouse_mode_ == MouseMode::Rubber) {
    mouse_mode_ = MouseMode::None;
    // Prevent zooming/seeking past the end of the route
    double min = std::clamp(secondsAtPoint(rubber_rect_.Min), can->minSeconds(), can->maxSeconds());
    double max = std::clamp(secondsAtPoint(rubber_rect_.Max), can->minSeconds(), can->maxSeconds());
    if (rubber_rect_.GetWidth() <= 0) {
      // no rubber dragged, seek to mouse position
      can->seekTo(std::clamp(secondsAtPoint(press_pos_), can->minSeconds(), can->maxSeconds()));
    } else if (rubber_rect_.GetWidth() > 10 && (max - min) > MIN_ZOOM_SECONDS) {
      charts_widget_->zoom_undo_stack_.push(new ZoomCommand({min, max}));
    }
    rubber_rect_ = ImRect();
  } else if (right_released && !ImGui::IsPopupOpen(context_menu_id_, ImGuiPopupFlags_None)) {
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
    sigs_.push_back(std::move(s));
    sigs_.back().color = uniqueColor(sigs_.back().color, sigs_.back().sig);
  }
  source->sigs_.clear();
  updateAxisY();
  charts_widget_->removeChart(source);
}

std::vector<ChartView::SigItem> ChartView::takeExtraSignals() {
  std::vector<SigItem> extra;
  for (auto it = sigs_.begin() + 1; it != sigs_.end(); ++it) {
    it->color = it->sig->color;
    extra.push_back(std::move(*it));
  }
  sigs_.resize(1);
  updateAxisY();
  return extra;
}

void ChartView::adoptSignal(SigItem s) {
  sigs_.push_back(std::move(s));
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
        value = s.sig->formatValue(pt->y, false);
        s.track_pt = *pt;
        x = std::max(x, xPos(pt->x));
      }
      std::string name = sigs_.size() > 1 ? s.sig->name + ": " : "";
      std::string min = s.min == std::numeric_limits<double>::max() ? "--" : utils::toString(s.min);
      std::string max = s.max == std::numeric_limits<double>::lowest() ? "--" : utils::toString(s.max);
      text_list.push_back({.has_marker = true, .marker = toImU32(s.color), .name = name, .bold = value, .rest = " (" + min + ", " + max + ")"});
    }
  }
  if (x < 0) {
    x = tooltip_x_;
  }
  ImVec2 pt(x, layout_.plot_area.Min.y);
  text_list.insert(text_list.begin(), TipLine{.name = formatNumber(secondsAtPoint({x, 0}), 3)});
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
  const ImVec2 tile_size(width, (float)settings.chart_height);
  layout_.rect = ImRect(tile_pos, tile_pos + tile_size);
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
    ImGui::GetWindowDrawList()->AddRect(layout_.rect.Min, layout_.rect.Max, ImGui::GetColorU32(ImGuiCol_Header), 0.0f, 0, 4.0f);
  }
}

void ChartView::drawStaticLayer() {
  ImDrawList *painter = ImGui::GetWindowDrawList();
  painter->AddRectFilled(layout_.rect.Min, layout_.rect.Max, ImGui::GetColorU32(ImGuiCol_ChildBg));
  ImGui::SetCursorScreenPos(layout_.move_icon_rect.Min);
  ImGui::InvisibleButton("grip", layout_.move_icon_rect.GetSize());
  if (ImGui::IsItemActivated()) charts_widget_->startChartDrag(this, ImGui::GetMousePos());
  if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
  painter->AddText(layout_.move_icon_rect.Min, ImGui::GetColorU32(ImGuiCol_Text), icon::GRIP_HORIZONTAL);
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
  // every tick is a 1 px line in the text color at alpha 50, the edge ticks close the box, no tick marks.
  // that alpha washes out on the dark base, so the dark theme draws opaque guides in a mid gray instead.
  const bool dark = isDarkTheme();
  ImVec4 grid_color;
  if (dark) {
    grid_color = colorRgb(DarkTheme::light.r, DarkTheme::light.g, DarkTheme::light.b);
  } else {
    grid_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    grid_color.w = 50.0f / 255.0f;
  }
  ImPlot::PushStyleColor(ImPlotCol_AxisGrid, grid_color);
  ImPlot::PushStyleColor(ImPlotCol_PlotBorder, grid_color);
  ImPlot::PushStyleColor(ImPlotCol_AxisTick, ImVec4(0, 0, 0, 0));
  ImPlot::PushStyleColor(ImPlotCol_AxisText, ImGui::GetStyleColorVec4(ImGuiCol_Text));
  ImPlot::PushStyleVar(ImPlotStyleVar_MajorTickLen, ImVec2(0, 0));
  // MajorGridSize is the per-axis line thickness; thicker guides read better on the dark base
  ImPlot::PushStyleVar(ImPlotStyleVar_MajorGridSize, dark ? ImVec2(2.0f, 2.0f) : ImVec2(1.0f, 1.0f));
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
    handleMousePress();
    handleMouseMove();
    handleMouseRelease();
    drawForeground();
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(6);
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
    ImGui::PopID();

    if (series_type_ == SeriesType::Scatter) {
      painter->AddCircleFilled(r.Min + ImVec2(marker_size / 2.0f, 2.0f + marker_size / 2.0f), marker_size / 2.0f, toImU32(s.color));
    } else {
      drawColorMarker(painter, r.Min, toImU32(s.color));
    }

    float x = r.Min.x + marker_size + 5;
    const float text_y = r.GetCenter().y - font_size / 2.0f;
    addTextEllipsis(painter, bold, title_color, ImVec2(x, text_y), r.Max.x, s.sig->name);
    float name_w = std::min(bold->CalcTextSizeA(font_size, FLT_MAX, 0.0f, s.sig->name.c_str()).x, r.Max.x - x);
    x += name_w;
    const std::string msg = msgLabel(s.msg_id);
    addTextEllipsis(painter, normal, msg_color, ImVec2(x, text_y), r.Max.x, msg);
    if (!s.visible) {  // strike out
      const float y = r.GetCenter().y;
      painter->AddLine(ImVec2(r.Min.x + marker_size + 5, y), ImVec2(std::min(x + ImGui::CalcTextSize(msg.c_str()).x, r.Max.x), y), title_color);
    }
  }
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
      ImPlot::PlotLine(label.c_str(), &begin->x, &begin->y, end - begin, spec);

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
  painter->AddRectFilled(rubber_rect_.Min, rubber_rect_.Max, withAlpha(highlight, 50));
  painter->AddRect(rubber_rect_.Min, rubber_rect_.Max, highlight);

  // time labels at the bottom corners (below the plot, so clip to the widget instead of the plot)
  const ImU32 white = IM_COL32_WHITE;
  const ImU32 gray = IM_COL32(0xa0, 0xa0, 0xa4, 0xff);
  painter = ImGui::GetWindowDrawList();
  painter->PushClipRect(layout_.rect.Min, layout_.rect.Max);
  for (const auto &pt : {rubber_rect_.GetBL(), rubber_rect_.GetBR()}) {
    std::string sec = formatNumber(secondsAtPoint(pt), 2);
    ImVec2 size = ImGui::CalcTextSize(sec.c_str()) + ImVec2(12, AXIS_X_TOP_MARGIN * 2);
    ImVec2 top_left = pt.x == rubber_rect_.Min.x ? ImVec2(pt.x - size.x, pt.y + 2) : ImVec2(pt.x, pt.y + 2);
    painter->AddRectFilled(top_left, top_left + size, gray);
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
  const bool dark = isDarkTheme();
  painter->AddRectFilled(time_str_pos, time_str_pos + time_str_size, dark ? IM_COL32(0x80, 0x80, 0x80, 0xff) : IM_COL32(0xa0, 0xa0, 0xa4, 0xff), 3.0f);
  painter->AddText(time_str_pos + ImVec2(4, 1), IM_COL32_WHITE, time_str.c_str());
}

void ChartView::drawSignalValue() {
  ImDrawList *painter = ImGui::GetWindowDrawList();
  const ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
  for (int i = 0; i < sigs_.size() && i < layout_.legend_rects.size(); ++i) {
    const auto &s = sigs_[i];
    const ImPlotPoint *pt = lastPointBefore(s, cur_sec_);
    std::string value = pt ? s.sig->formatValue(pt->y) : "--";
    const ImVec2 value_min = layout_.legend_rects[i].GetBL() - ImVec2(0, 1);
    ImRect value_rect(value_min, value_min + layout_.legend_rects[i].GetSize());
    float w = ImGui::CalcTextSize(value.c_str()).x;
    if (w <= value_rect.GetWidth()) {
      painter->AddText(ImVec2(value_rect.GetCenter().x - w / 2, value_rect.Min.y), color, value.c_str());
    } else {
      addTextEllipsis(painter, ImGui::GetFont(), color, value_rect.Min, value_rect.Max.x, value);
    }
  }
}

CabanaColor ChartView::uniqueColor(CabanaColor color, const cabana::Signal *exclude) const {
  for (auto &s : sigs_) {
    if (s.sig != exclude && std::abs(color.hsv().hue - s.color.hsv().hue) < 0.1) {
      // use different color to distinguish it from others.
      auto last_color = sigs_.back().color;
      static thread_local std::mt19937 rng{std::random_device{}()};
      std::uniform_int_distribution<int> sat(35, 99);
      std::uniform_int_distribution<int> val(85, 99);
      color = CabanaColor::fromHsv(std::fmod(last_color.hsv().hue + 60 / 360.0, 1.0),
                                   sat(rng) / 100.0,
                                   val(rng) / 100.0,
                                   color.a / 255.0f);
      break;
    }
  }
  return color;
}
