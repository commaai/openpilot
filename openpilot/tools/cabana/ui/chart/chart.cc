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
#include "tools/cabana/ui/imgui_util.h"

const int AXIS_X_TOP_MARGIN = 4;
const int X_TICK_COUNT = 5;
const double MIN_ZOOM_SECONDS = 0.01;  // 10ms
// Define a small value of epsilon to compare double values
const float EPSILON = 0.000001;
static inline bool xLessThan(const ImPlotPoint &p, float x) { return p.x < (x - EPSILON); }
static inline bool isNull(const ImPlotPoint &p) { return p.x == 0 && p.y == 0; }

// QStyle layout margins (PM_LayoutLeftMargin etc.)
static ImVec4 layoutMargins() {
  return {8, 6, 8, 6};  // left, top, right, bottom
}

static std::string formatNumber(double value, int precision) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.*f", precision, value);
  return buf;
}

// QString::number(double) default format ('g', 6)
static std::string formatNumberG(double value) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%g", value);
  return buf;
}

static float hueF(const CabanaColor &c) {
  float h, s, v;
  ImGui::ColorConvertRGBtoHSV(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, h, s, v);
  return h;
}

struct FontInfo {
  ImFont *font;
  float size;
};
static FontInfo boldFont() {
  pushBoldFont();
  FontInfo f{ImGui::GetFont(), ImGui::GetFontSize()};
  popBoldFont();
  return f;
}

static void addTextEllipsis(ImDrawList *dl, const FontInfo &f, ImU32 col, const ImVec2 &pos, float max_x, const std::string &text) {
  ImGui::PushFont(f.font, 0.0f);
  ImGui::RenderTextEllipsis(dl, pos, ImVec2(max_x, pos.y + f.size), max_x, text.c_str(), nullptr, nullptr);
  ImGui::PopFont();
}

ChartView::ChartView(const std::pair<double, double> &x_range, ChartsWidget *parent)
    : x_min(x_range.first), x_max(x_range.second), charts_widget(parent) {
  series_type = (SeriesType)settings.chart_series_type;
  align_to = 50;
  tip_label = new TipLabel();
  // createToolButtons: the buttons are immediate mode, drawn from paintEvent

  connections_.push_back(dbc()->signalRemoved.connect([this](const cabana::Signal *sig) { signalRemoved(sig); }));
  connections_.push_back(dbc()->signalUpdated.connect([this](const cabana::Signal *sig) { signalUpdated(sig); }));
  connections_.push_back(dbc()->msgRemoved.connect([this](MessageId id) { msgRemoved(id); }));
  connections_.push_back(dbc()->msgUpdated.connect([this](MessageId id) { msgUpdated(id); }));
}

ChartView::~ChartView() {
  delete tip_label;
}

void ChartView::drawMenuActions() {
  // series types
  static const char *types[] = {"Line", "Step Line", "Scatter"};
  for (int i = 0; i < 3; ++i) {
    if (ImGui::MenuItem(types[i], nullptr, i == (int)series_type)) {
      setSeriesType((SeriesType)i);
    }
  }
  ImGui::Separator();
  if (ImGui::MenuItem("Manage Signals")) manageSignals();
  if (ImGui::MenuItem("Split Chart", nullptr, false, split_chart_enabled)) charts_widget->splitChart(this);
}

// immediate mode: the buttons (and their menus) are drawn every frame from paintEvent, after resizeEvent placed them
void ChartView::createToolButtons() {
  ImGui::SetCursorScreenPos(ImVec2(close_btn_rect.Min.x, close_btn_rect.Min.y));
  bool close_clicked = toolButton("close_btn", icon::X, "Remove Chart");

  ImGui::SetCursorScreenPos(ImVec2(manage_btn_rect.Min.x, manage_btn_rect.Min.y));
  if (toolButton("manage_btn", icon::LIST, "")) ImGui::OpenPopup("manage_menu");
  if (ImGui::BeginPopup("manage_menu")) {
    drawMenuActions();
    ImGui::EndPopup();
  }

  if (close_clicked) charts_widget->removeChart(this);  // close_act
}

ImVec2 ChartView::sizeHint() const {
  return {(float)CHART_MIN_WIDTH, (float)settings.chart_height};
}

void ChartView::addSignal(const MessageId &msg_id, const cabana::Signal *sig) {
  if (hasSignal(msg_id, sig)) return;

  sigs.push_back({.msg_id = msg_id, .sig = sig, .color = uniqueColor(sig->color)});
  updateSeries(sig);
  updateTitle();
  charts_widget->seriesChanged();
}

bool ChartView::hasSignal(const MessageId &msg_id, const cabana::Signal *sig) const {
  return std::any_of(sigs.cbegin(), sigs.cend(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
}

void ChartView::removeIf(std::function<bool(const SigItem &s)> predicate) {
  int prev_size = sigs.size();
  sigs.erase(std::remove_if(sigs.begin(), sigs.end(), predicate), sigs.end());
  if (sigs.empty()) {
    charts_widget->removeChart(this);
  } else if (sigs.size() != prev_size) {
    charts_widget->seriesChanged();
    updateAxisY();
    updateTitle();
  }
}

void ChartView::signalUpdated(const cabana::Signal *sig) {
  auto it = std::find_if(sigs.begin(), sigs.end(), [sig](auto &s) { return s.sig == sig; });
  if (it != sigs.end()) {
    if (!(it->color == sig->color)) {
      it->color = uniqueColor(sig->color, sig);
    }
    updateTitle();
    updateSeries(sig);
  }
}

void ChartView::msgUpdated(MessageId id) {
  if (std::any_of(sigs.cbegin(), sigs.cend(), [=](auto &s) { return s.msg_id.address == id.address; })) {
    updateTitle();
  }
}

void ChartView::manageSignals() {
  auto dlg = std::make_unique<SignalSelector>("Manage Chart");
  for (auto &s : sigs) {
    dlg->addSelected(s.msg_id, s.sig);
  }
  // exec() is non-blocking: the widget runs this once the dialog is accepted (dropped if the chart is removed first)
  charts_widget->execSignalSelector(std::move(dlg), this, [this](SignalSelector &selector) {
    auto items = selector.seletedItems();
    for (auto s : items) {
      addSignal(s->msg_id, s->sig);
    }
    removeIf([&](auto &s) {
      return std::none_of(items.cbegin(), items.cend(), [&](auto &it) { return s.msg_id == it->msg_id && s.sig == it->sig; });
    });
  });
}

void ChartView::resizeEvent() {
  const auto margins = layoutMargins();
  const ImVec2 grip = ImGui::CalcTextSize(icon::GRIP_HORIZONTAL);
  move_icon_rect = ImRect(rect.Min + ImVec2(margins.x, margins.y), rect.Min + ImVec2(margins.x, margins.y) + grip);
  const ImVec2 pad = ImGui::GetStyle().FramePadding * 2;
  const ImVec2 close_size = ImGui::CalcTextSize(icon::X) + pad;
  const ImVec2 manage_size = ImGui::CalcTextSize(icon::LIST) + pad;
  close_btn_rect = ImRect(ImVec2(rect.Max.x - margins.z - close_size.x, rect.Min.y + margins.y), ImVec2(0, 0));
  close_btn_rect.Max = close_btn_rect.Min + close_size;
  manage_btn_rect = ImRect(ImVec2(close_btn_rect.Min.x - manage_size.x - ImGui::GetStyle().ItemSpacing.x, rect.Min.y + margins.y), ImVec2(0, 0));
  manage_btn_rect.Max = manage_btn_rect.Min + manage_size;
  updatePlotArea(align_to, true);
}

void ChartView::updatePlotArea(int left_pos, bool force) {
  if (align_to != left_pos || force) {
    align_to = left_pos;
    if (!ImGui::GetCurrentContext() || !ImGui::GetCurrentWindowRead()) return;  // layout is recomputed from draw()

    const auto margins = layoutMargins();
    const FontInfo bfm = boldFont();
    const float fm_height = ImGui::GetTextLineHeight();
    const int marker_size = fm_height - 4;
    const int row_height = std::max<int>(marker_size, fm_height) + fm_height + 3;  // + signal_value_font height
    const int legend_left = move_icon_rect.Max.x + margins.x;
    const int legend_right = std::max<int>(manage_btn_rect.Min.x - margins.z, legend_left + 10);

    // layout legend entries left-to-right, wrapping between the move icon and the buttons
    legend_rects.clear();
    int x = legend_left, y = rect.Min.y + margins.y;
    for (auto &s : sigs) {
      int w = marker_size + 5 + bfm.font->CalcTextSizeA(bfm.size, FLT_MAX, 0.0f, s.sig->name.c_str()).x +
              ImGui::CalcTextSize((" " + msgName(s.msg_id) + " " + s.msg_id.toString()).c_str()).x;
      w = std::min(w, legend_right - legend_left);  // keep oversized entries clear of the header buttons
      if (x + w > legend_right && x > legend_left) {
        x = legend_left;
        y += row_height;
      }
      legend_rects.emplace_back(ImVec2(x, y), ImVec2(x + w, y + std::max<int>(marker_size, fm_height)));
      x += w + 12;
    }

    // add top space for the legend and signal values
    int adjust_top = (y + row_height) - rect.Min.y;
    adjust_top = std::max<int>(adjust_top, manage_btn_rect.Max.y - rect.Min.y + margins.y);
    header_bottom = rect.Min.y + adjust_top;
    // the x-axis label space and the left alignment (align_to) are handled by implot (BeginAlignedPlots)
    resetChartCache();
  }
}

void ChartView::updateTitle() {
  split_chart_enabled = sigs.size() > 1;
  updatePlotArea(align_to, true);
}

void ChartView::updatePlot(double cur, double min, double max) {
  cur_sec = cur;
  if (min != x_min || max != x_max) {
    x_min = min;
    x_max = max;
    updateAxisY();
    // update tooltip
    if (tooltip_x >= 0) {
      showTip(secondsAtPoint({(float)tooltip_x, 0}));
    }
    resetChartCache();
  }
}

void ChartView::appendCanEvents(const cabana::Signal *sig, const std::vector<const CanEvent *> &events,
                                std::vector<ImPlotPoint> &vals, std::vector<ImPlotPoint> &step_vals) {
  vals.reserve(vals.size() + events.capacity());
  step_vals.reserve(step_vals.size() + events.capacity() * 2);

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
  for (auto &s : sigs) {
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
  // no static layer cache in imgui; nothing to invalidate on the ui thread
}

// auto zoom on yaxis
void ChartView::updateAxisY() {
  if (sigs.empty()) return;

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();
  std::string unit = sigs[0].sig->unit;

  for (auto &s : sigs) {
    if (!s.visible) continue;

    // Only show unit when all signals have the same unit
    if (unit != s.sig->unit) {
      unit.clear();
    }

    auto first = std::lower_bound(s.vals.cbegin(), s.vals.cend(), x_min, xLessThan);
    auto last = std::lower_bound(first, s.vals.cend(), x_max, xLessThan);
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

  if (y_unit != unit) {
    y_unit = unit;
    y_label_width = 0;  // recalc width
  }

  double delta = std::abs(max - min) < 1e-3 ? 1 : (max - min) * 0.05;
  auto [min_y, max_y, tick_count] = getNiceAxisNumbers(min - delta, max + delta, 3);
  if (min_y != y_min || max_y != y_max || y_label_width == 0) {
    y_min = min_y;
    y_max = max_y;
    y_tick_count = tick_count;
    y_precision = std::max(int(-std::floor(std::log10((max_y - min_y) / (tick_count - 1)))), 0);
    // the label width needs the font: measured in draw() on the ui thread (updateSeries runs in worker threads)
    y_label_width = 0;
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
  return std::max(int(-std::floor(std::log10((x_max - x_min) / (X_TICK_COUNT - 1)))), 2);
}

// nice numbers can be expressed as form of 1*10^n, 2* 10^n or 5*10^n
double ChartView::niceNumber(double x, bool ceiling) {
  double z = std::pow(10, std::floor(std::log10(x))); //find corresponding number of the form of 10^n than is smaller than x
  double q = x / z; //q<10 && q>=1;
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

void ChartView::contextMenuEvent() {
  // Qt opens the menu on right press; the release never reaches mouseReleaseEvent's "undo zoom" branch
  if (ImGui::IsMouseReleased(ImGuiMouseButton_Right) && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !ImGui::IsAnyItemActive()) {
    ImGui::OpenPopup("context_menu");
  }
  if (ImGui::BeginPopup("context_menu")) {
    drawMenuActions();
    ImGui::Separator();
    if (ImGui::MenuItem("Undo Zoom", nullptr, false, charts_widget->zoom_undo_stack.canUndo())) charts_widget->zoom_undo_stack.undo();
    if (ImGui::MenuItem("Redo Zoom", nullptr, false, charts_widget->zoom_undo_stack.canRedo())) charts_widget->zoom_undo_stack.redo();
    ImGui::Separator();
    if (ImGui::MenuItem("Close")) charts_widget->removeChart(this);
    ImGui::EndPopup();
  }
}

void ChartView::mousePressEvent() {
  if (!ImGui::IsMouseClicked(ImGuiMouseButton_Left) || !plot_hovered) return;
  const ImVec2 pos = ImGui::GetMousePos();
  press_pos = pos;
  // the move icon press is handled by the grip item (startChartDrag)
  if (ImGui::GetIO().KeyShift) {
    // Save current playback state when scrubbing
    resume_after_scrub = !can->isPaused();
    if (resume_after_scrub) {
      can->pause(true);
    }
    mouse_mode = MouseMode::Scrub;
  } else if (plot_area.Contains(pos)) {
    mouse_mode = MouseMode::Rubber;
    rubber_rect = ImRect();
  }
}

void ChartView::mouseMoveEvent() {
  const ImVec2 pos = ImGui::GetMousePos();
  const ImVec2 delta = ImGui::GetIO().MouseDelta;
  // Qt only delivers move events to the widget under the mouse (or the one holding the implicit grab)
  if (mouse_mode == MouseMode::None && (!rect.Contains(pos) || (delta.x == 0 && delta.y == 0))) return;

  // Scrubbing
  if (mouse_mode == MouseMode::Scrub && ImGui::GetIO().KeyShift) {
    if (plot_area.Contains(pos)) {
      can->seekTo(std::clamp(secondsAtPoint(pos), can->minSeconds(), can->maxSeconds()));
    }
  }

  if (mouse_mode == MouseMode::Rubber) {
    // horizontal selection, clamped to the plot area
    float left = std::clamp(std::min(press_pos.x, pos.x), plot_area.Min.x, plot_area.Max.x);
    float right = std::clamp(std::max(press_pos.x, pos.x), plot_area.Min.x, plot_area.Max.x);
    rubber_rect = ImRect(ImVec2(left, plot_area.Min.y), ImVec2(right, plot_area.Max.y));
  }

  clearTrackPoints();
  if (mouse_mode != MouseMode::Rubber && plot_area.Contains(pos) && (plot_hovered || mouse_mode != MouseMode::None) &&
      ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow)) {  // isActiveWindow
    charts_widget->showValueTip(secondsAtPoint(pos));
  } else if (tip_label->isVisible()) {
    charts_widget->showValueTip(-1);
  }
}

void ChartView::mouseReleaseEvent() {
  if (!ImGui::IsMouseReleased(ImGuiMouseButton_Left)) return;
  if (mouse_mode == MouseMode::Rubber) {
    mouse_mode = MouseMode::None;
    // Prevent zooming/seeking past the end of the route
    double min = std::clamp(secondsAtPoint(rubber_rect.Min), can->minSeconds(), can->maxSeconds());
    double max = std::clamp(secondsAtPoint(rubber_rect.Max), can->minSeconds(), can->maxSeconds());
    if (rubber_rect.GetWidth() <= 0) {
      // no rubber dragged, seek to mouse position
      can->seekTo(std::clamp(secondsAtPoint(press_pos), can->minSeconds(), can->maxSeconds()));
    } else if (rubber_rect.GetWidth() > 10 && (max - min) > MIN_ZOOM_SECONDS) {
      charts_widget->zoom_undo_stack.push(new ZoomCommand({min, max}));
    }
    rubber_rect = ImRect();
  }
  // toggling series visibility by clicking a legend entry is handled by the legend items in drawLegend

  // Resume playback if we were scrubbing
  if (mouse_mode == MouseMode::Scrub) {
    mouse_mode = MouseMode::None;
    if (resume_after_scrub) {
      can->pause(false);
      resume_after_scrub = false;
    }
  }
}

void ChartView::takeSignalsFrom(ChartView *source) {
  for (auto &s : source->sigs) {
    sigs.push_back(std::move(s));
    sigs.back().color = uniqueColor(sigs.back().color, sigs.back().sig);
  }
  source->sigs.clear();
  updateAxisY();
  updateTitle();
  charts_widget->removeChart(source);
}

void ChartView::showTip(double sec) {
  ImRect tip_area(ImVec2(rect.Min.x, plot_area.Min.y), ImVec2(rect.Max.x, plot_area.Max.y));
  ImRect visible_rect = charts_widget->chartVisibleRect(this);
  visible_rect.ClipWith(tip_area);
  if (visible_rect.GetWidth() <= 0 || visible_rect.GetHeight() <= 0) {
    tip_label->hide();
    return;
  }

  tooltip_x = xPos(sec);
  float x = -1;
  std::vector<TipLine> text_list;
  for (auto &s : sigs) {
    if (s.visible) {
      std::string value = "--";
      // use reverse iterator to find last item <= sec.
      auto it = std::lower_bound(s.vals.crbegin(), s.vals.crend(), sec, [](auto &p, double v) { return p.x > v; });
      if (it != s.vals.crend() && it->x >= x_min) {
        value = s.sig->formatValue(it->y, false);
        s.track_pt = *it;
        x = std::max(x, xPos(it->x));
      }
      std::string name = sigs.size() > 1 ? s.sig->name + ": " : "";
      std::string min = s.min == std::numeric_limits<double>::max() ? "--" : formatNumberG(s.min);
      std::string max = s.max == std::numeric_limits<double>::lowest() ? "--" : formatNumberG(s.max);
      text_list.push_back({.has_marker = true, .marker = toImU32(s.color), .name = name, .bold = value, .rest = " (" + min + ", " + max + ")"});
    }
  }
  if (x < 0) {
    x = tooltip_x;
  }
  ImVec2 pt(x, plot_area.Min.y);
  text_list.insert(text_list.begin(), TipLine{.name = formatNumber(secondsAtPoint({x, 0}), 3)});
  tip_label->showText(pt, text_list, visible_rect);
}

void ChartView::hideTip() {
  clearTrackPoints();
  tooltip_x = -1;
  tip_label->hide();
}

void ChartView::resetChartCache() {
  // no static layer cache in imgui; the chart is redrawn every frame
}

void ChartView::draw(float width) {
  ImGui::PushID(this);
  width = std::max(width, (float)CHART_MIN_WIDTH);
  if (ImGui::BeginChild("chart", ImVec2(width, sizeHint().y), ImGuiChildFlags_None,
                        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
    rect = ImGui::GetCurrentWindow()->Rect();
    // the y label width needs the font, so it is measured here instead of in updateAxisY
    if (!sigs.empty() && y_label_width == 0) {
      int max_label_width = 0;
      for (int i = 0; i < y_tick_count; i++) {
        double value = y_min + (i * (y_max - y_min) / (y_tick_count - 1));
        max_label_width = std::max<int>(max_label_width, ImGui::CalcTextSize(formatNumber(value, y_precision).c_str()).x);
      }
      int title_spacing = y_unit.empty() ? 0 : ImGui::GetTextLineHeight();
      y_label_width = title_spacing + max_label_width + 15;
      axisYLabelWidthChanged(y_label_width);
    }
    resizeEvent();
    paintEvent();
    contextMenuEvent();
  }
  ImGui::EndChild();
  tip_label->paintEvent();
  ImGui::PopID();
}

void ChartView::paintEvent() {
  drawStaticLayer();

  if (can_drop) {
    ImGui::GetWindowDrawList()->AddRect(rect.Min, rect.Max, ImGui::GetColorU32(ImGuiCol_Header), 0.0f, 0, 4.0f);
  }
  // drawForeground is called from drawSeries while the plot is open
}

void ChartView::drawStaticLayer() {
  ImDrawList *painter = ImGui::GetWindowDrawList();
  painter->AddRectFilled(rect.Min, rect.Max, ImGui::GetColorU32(ImGuiCol_ChildBg));
  // move icon: the drag handle
  ImGui::SetCursorScreenPos(move_icon_rect.Min);
  ImGui::InvisibleButton("grip", move_icon_rect.GetSize());
  if (ImGui::IsItemActivated()) charts_widget->startChartDrag(this, ImGui::GetMousePos());
  if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
  painter->AddText(move_icon_rect.Min, ImGui::GetColorU32(ImGuiCol_Text), icon::GRIP_HORIZONTAL);
  createToolButtons();
  drawLegend();
  drawSignalValue();  // foreground in Qt; drawn here because implot clips the plot frame
  drawAxes();
}

void ChartView::drawAxes() {
  // the plot: axes, grid, tick labels and the y axis title (unit) are drawn by implot
  const auto margins = layoutMargins();
  ImGui::SetCursorScreenPos(ImVec2(rect.Min.x, header_bottom));
  const float plot_h = std::max(rect.Max.y - header_bottom - margins.w, 10.0f);
  ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(margins.x, AXIS_X_TOP_MARGIN));
  ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0, 0, 0, 0));
  ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0, 0, 0, 0));
  const ImPlotFlags flags = ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoMouseText |
                            ImPlotFlags_NoBoxSelect | ImPlotFlags_NoInputs | ImPlotFlags_NoFrame;
  const ImPlotAxisFlags axis_flags = ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_Lock;
  if (ImPlot::BeginPlot("##plot", ImVec2(rect.GetWidth(), plot_h), flags)) {
    ImPlot::SetupAxis(ImAxis_X1, nullptr, axis_flags);
    ImPlot::SetupAxis(ImAxis_Y1, y_unit.empty() ? nullptr : y_unit.c_str(), axis_flags);
    ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always);
    // y grid lines and tick labels
    ImPlot::SetupAxisTicks(ImAxis_Y1, y_min, y_max, y_tick_count);
    ImPlot::SetupAxisFormat(ImAxis_Y1, ("%." + std::to_string(y_precision) + "f").c_str());
    // x grid lines and tick labels
    ImPlot::SetupAxisTicks(ImAxis_X1, x_min, x_max, X_TICK_COUNT);
    ImPlot::SetupAxisFormat(ImAxis_X1, ("%." + std::to_string(xAxisPrecision()) + "f").c_str());
    ImPlot::SetupFinish();

    plot_area = ImRect(ImPlot::GetPlotPos(), ImPlot::GetPlotPos() + ImPlot::GetPlotSize());
    plot_hovered = ImPlot::IsPlotHovered();
    drawSeries();
    mousePressEvent();
    mouseMoveEvent();
    mouseReleaseEvent();
    drawForeground();
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor(2);
  ImPlot::PopStyleVar();
}

void ChartView::drawLegend() {
  ImDrawList *painter = ImGui::GetWindowDrawList();
  const ImU32 title_color = ImGui::GetColorU32(ImGuiCol_Text);
  // Draw message details in similar color, but slightly fade it to the background
  ImU32 msg_color = (title_color & ~IM_COL32_A_MASK) | (180 << IM_COL32_A_SHIFT);
  const FontInfo bold_font = boldFont();
  const FontInfo normal_font{ImGui::GetFont(), ImGui::GetFontSize()};
  const int marker_size = ImGui::GetTextLineHeight() - 4;

  for (int i = 0; i < sigs.size() && i < legend_rects.size(); ++i) {
    const auto &s = sigs[i];
    const ImRect &r = legend_rects[i];
    // toggle series visibility by clicking its legend entry (mouseReleaseEvent)
    ImGui::PushID(i);
    ImGui::SetCursorScreenPos(r.Min);
    if (ImGui::InvisibleButton("legend", ImVec2(std::max(r.GetWidth(), 1.0f), std::max(r.GetHeight(), 1.0f))) && sigs.size() > 1) {
      sigs[i].visible = !sigs[i].visible;
      updateAxisY();
      updateTitle();
    }
    ImGui::PopID();

    ImVec2 marker_rect(r.Min.x, r.GetCenter().y - marker_size / 2.0f);
    series_type == SeriesType::Scatter
        ? painter->AddCircleFilled(marker_rect + ImVec2(marker_size / 2.0f, marker_size / 2.0f), marker_size / 2.0f, toImU32(s.color))
        : painter->AddRectFilled(marker_rect, marker_rect + ImVec2(marker_size, marker_size), toImU32(s.color));

    float x = r.Min.x + marker_size + 5;
    const float text_y = r.GetCenter().y - normal_font.size / 2.0f;
    addTextEllipsis(painter, bold_font, title_color, ImVec2(x, text_y), r.Max.x, s.sig->name);
    float name_w = std::min(bold_font.font->CalcTextSizeA(bold_font.size, FLT_MAX, 0.0f, s.sig->name.c_str()).x, r.Max.x - x);
    x += name_w;
    std::string msg = " " + msgName(s.msg_id) + " " + s.msg_id.toString();
    addTextEllipsis(painter, normal_font, msg_color, ImVec2(x, text_y), r.Max.x, msg);
    if (!s.visible) {  // strike out
      const float y = r.GetCenter().y;
      painter->AddLine(ImVec2(r.Min.x + marker_size + 5, y), ImVec2(std::min(x + ImGui::CalcTextSize(msg.c_str()).x, r.Max.x), y), title_color);
    }
  }
}

void ChartView::drawSeries() {
  for (int i = 0; i < sigs.size(); ++i) {
    auto &s = sigs[i];
    if (!s.visible) continue;

    // visible points in vals to compute point density
    auto first = std::lower_bound(s.vals.cbegin(), s.vals.cend(), x_min, xLessThan);
    auto last = std::lower_bound(first, s.vals.cend(), x_max, xLessThan);
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
    if (series_type == SeriesType::Scatter) {
      float radius = std::clamp(pixels_per_point / 2.0, 2.0, 8.0) / 2.0;
      spec.Marker = ImPlotMarker_Circle;
      spec.MarkerSize = radius;
      if (first != last) ImPlot::PlotScatter(label.c_str(), &first->x, &first->y, last - first, spec);
    } else {
      const auto &points = series_type == SeriesType::StepLine ? s.step_vals : s.vals;
      auto begin = std::lower_bound(points.cbegin(), points.cend(), x_min, xLessThan);
      if (begin != points.cbegin()) --begin;
      auto end = std::lower_bound(begin, points.cend(), x_max, xLessThan);
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
  // drawSignalValue: see drawStaticLayer
  // draw track points
  ImDrawList *painter = ImPlot::GetPlotDrawList();
  ImPlot::PushPlotClipRect();
  float track_line_x = -1;
  for (auto &s : sigs) {
    if (!isNull(s.track_pt) && s.visible) {
      ImVec2 pos(xPos(s.track_pt.x), yPos(s.track_pt.y));
      painter->AddCircleFilled(pos, 5.5f, toImU32(s.color.darker(125)));
      track_line_x = std::max(track_line_x, pos.x);
    }
  }
  if (track_line_x > 0) {
    // dashed line
    const ImU32 dark_gray = IM_COL32(0x80, 0x80, 0x80, 0xff);
    for (float y = plot_area.Min.y; y < plot_area.Max.y; y += 8) {
      painter->AddLine(ImVec2(track_line_x, y), ImVec2(track_line_x, std::min(y + 4, plot_area.Max.y)), dark_gray, 1.0f);
    }
  }
  ImPlot::PopPlotClipRect();

  drawRubberBandTimeRange();
}

void ChartView::drawRubberBandTimeRange() {
  if (rubber_rect.GetWidth() <= 1) return;

  ImDrawList *painter = ImPlot::GetPlotDrawList();
  // selection rect
  ImU32 highlight = ImGui::GetColorU32(ImGuiCol_Header);
  ImU32 fill = (highlight & ~IM_COL32_A_MASK) | (50 << IM_COL32_A_SHIFT);
  painter->AddRectFilled(rubber_rect.Min, rubber_rect.Max, fill);
  painter->AddRect(rubber_rect.Min, rubber_rect.Max, highlight);

  // time labels at the bottom corners
  const ImU32 white = IM_COL32_WHITE;
  const ImU32 gray = IM_COL32(0xa0, 0xa0, 0xa4, 0xff);
  for (const auto &pt : {rubber_rect.GetBL(), rubber_rect.GetBR()}) {
    std::string sec = formatNumber(secondsAtPoint(pt), 2);
    ImVec2 size = ImGui::CalcTextSize(sec.c_str()) + ImVec2(12, AXIS_X_TOP_MARGIN * 2);
    ImVec2 top_left = pt.x == rubber_rect.Min.x ? ImVec2(pt.x - size.x, pt.y + 2) : ImVec2(pt.x, pt.y + 2);
    painter->AddRectFilled(top_left, top_left + size, gray);
    painter->AddText(top_left + ImVec2(6, AXIS_X_TOP_MARGIN), white, sec.c_str());
  }
}

void ChartView::drawTimeline() {
  ImDrawList *painter = ImPlot::GetPlotDrawList();
  // draw vertical time line
  float x = std::clamp(xPos(cur_sec), plot_area.Min.x, plot_area.Max.x);
  painter->AddLine(ImVec2(x, plot_area.Min.y - 1.0f), ImVec2(x, plot_area.Max.y + 1.0f), ImGui::GetColorU32(ImGuiCol_Text), 1.0f);

  // draw current time under the axis-x
  std::string time_str = formatNumber(cur_sec, 2);
  ImVec2 time_str_size = ImGui::CalcTextSize(time_str.c_str()) + ImVec2(8, 2);
  ImVec2 time_str_pos(x - time_str_size.x / 2.0f, plot_area.Max.y + AXIS_X_TOP_MARGIN);
  const bool dark = settings.theme == DARK_THEME;
  painter->AddRectFilled(time_str_pos, time_str_pos + time_str_size, dark ? IM_COL32(0x80, 0x80, 0x80, 0xff) : IM_COL32(0xa0, 0xa0, 0xa4, 0xff), 3.0f);
  painter->AddText(time_str_pos + ImVec2(4, 1), IM_COL32_WHITE, time_str.c_str());  // BrightText
}

void ChartView::drawSignalValue() {
  ImDrawList *painter = ImGui::GetWindowDrawList();
  const FontInfo font{ImGui::GetFont(), ImGui::GetFontSize()};  // signal_value_font (9pt in Qt; no small font here)
  const ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
  for (int i = 0; i < sigs.size() && i < legend_rects.size(); ++i) {
    const auto &s = sigs[i];
    auto it = std::lower_bound(s.vals.crbegin(), s.vals.crend(), cur_sec,
                               [](auto &p, double x) { return p.x > x + EPSILON; });
    std::string value = (it != s.vals.crend() && it->x >= x_min) ? s.sig->formatValue(it->y) : "--";
    ImRect value_rect(legend_rects[i].GetBL() - ImVec2(0, 1), legend_rects[i].GetBL() - ImVec2(0, 1) + legend_rects[i].GetSize());
    float w = ImGui::CalcTextSize(value.c_str()).x;
    if (w <= value_rect.GetWidth()) {
      painter->AddText(ImVec2(value_rect.GetCenter().x - w / 2, value_rect.Min.y), color, value.c_str());
    } else {
      addTextEllipsis(painter, font, color, value_rect.Min, value_rect.Max.x, value);
    }
  }
}

CabanaColor ChartView::uniqueColor(CabanaColor color, const cabana::Signal *exclude) const {
  for (auto &s : sigs) {
    if (s.sig != exclude && std::abs(hueF(color) - hueF(s.color)) < 0.1) {
      // use different color to distinguish it from others.
      auto last_color = sigs.back().color;
      static thread_local std::mt19937 rng{std::random_device{}()};
      std::uniform_int_distribution<int> sat(35, 99);
      std::uniform_int_distribution<int> val(85, 99);
      color = CabanaColor::fromHsv(std::fmod(hueF(last_color) + 60 / 360.0, 1.0),
                                   sat(rng) / 100.0,
                                   val(rng) / 100.0,
                                   color.a / 255.0f);
      break;
    }
  }
  return color;
}

void ChartView::setSeriesType(SeriesType type) {
  if (type != series_type) {
    series_type = type;
    updateTitle();
  }
}
