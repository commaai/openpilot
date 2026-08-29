#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/chartswidget.h"

#include "tools/cabana/ui/app.h"
#include "tools/cabana/ui/threadpool.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <future>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/strings.h"

const int MAX_COLUMN_COUNT = 4;
const int CHART_SPACING = 4;
const int START_DRAG_DISTANCE = 10;  // QApplication::startDragDistance()
const float TOOLBAR_ITEM_SPACING = 1.0f;    // QStyle::PM_ToolBarItemSpacing
const float TOOLBAR_BUTTON_PADDING = 4.0f;  // QToolButton (auto raise) horizontal margin
const float MENU_ARROW_SIZE = 6.0f;         // QStyle::PE_IndicatorArrowDown on a toolbutton menu
const float LAYOUT_HORIZONTAL_SPACING = 6.0f;  // QStyle::PM_LayoutHorizontalSpacing
const float SLIDER_LENGTH = 13.0f;          // QStyle::PM_SliderLength (Fusion)
const float SLIDER_THICKNESS = 13.0f;       // QStyle::PM_SliderThickness (Fusion)

static float buttonWidth(const std::string &label) {
  return ImGui::CalcTextSize(label.c_str(), nullptr, true).x + ImGui::GetStyle().FramePadding.x * 2;
}

// an auto-raise QToolButton with an InstantPopup menu: flat until hovered, with a small dropdown arrow
// drawn hard against the text (ItemInnerSpacing.x / 2) at the text baseline
static float menuButtonWidth(const std::string &text) {
  const ImGuiStyle &style = ImGui::GetStyle();
  return ImGui::CalcTextSize(text.c_str(), nullptr, true).x + style.ItemInnerSpacing.x * 0.5f + MENU_ARROW_SIZE +
         style.FramePadding.x * 2;
}

static bool menuButton(const char *id, const std::string &text, const char *popup_id) {
  const ImGuiStyle &style = ImGui::GetStyle();
  const bool popup_open = ImGui::IsPopupOpen(popup_id);
  // setAutoRaise(true): no frame, transparent until hovered; the button is drawn pressed while the menu is open
  ImGui::PushStyleColor(ImGuiCol_Button, popup_open ? style.Colors[ImGuiCol_ButtonActive] : ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
  bool clicked = ImGui::Button((text + "###" + id).c_str(), ImVec2(menuButtonWidth(text), 0.0f));
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor();
  // a 6 px arrow in the disabled text color, right after the text, sitting on the text baseline
  const ImVec2 min = ImGui::GetItemRectMin();
  const float x = min.x + style.FramePadding.x + ImGui::CalcTextSize(text.c_str(), nullptr, true).x +
                  style.ItemInnerSpacing.x * 0.5f;
  const float baseline = min.y + style.FramePadding.y + ImGui::GetFontBaked()->Ascent;
  ImGui::GetWindowDrawList()->AddTriangleFilled(ImVec2(x, baseline - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(x + MENU_ARROW_SIZE, baseline - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(x + MENU_ARROW_SIZE * 0.5f, baseline),
                                                ImGui::GetColorU32(ImGuiCol_TextDisabled));
  // QToolButton::InstantPopup: the menu drops down from below the button, not at the mouse cursor
  ImGui::SetNextWindowPos(ImVec2(min.x, ImGui::GetItemRectMax().y), ImGuiCond_Always);
  return clicked;
}

// QStyle::SC_SliderHandle (Fusion): a 13x13 handle filled with a subtle vertical gradient and a mid grey outline
static void drawSliderHandle(ImDrawList *p, const ImRect &r) {
  const bool dark = isDarkTheme();
  // buttonColor.lighter(104) / buttonColor.darker(104)
  const ImU32 top = dark ? IM_COL32(0x3e, 0x41, 0x43, 255) : IM_COL32(255, 255, 255, 255);
  const ImU32 bottom = dark ? IM_COL32(0x39, 0x3c, 0x3e, 255) : IM_COL32(0xf0, 0xf0, 0xf0, 255);
  // QFusionStylePrivate::outline: the top/left edge is one step lighter than the bottom/right edge
  const ImU32 outline_top = dark ? IM_COL32(0xa3, 0xa3, 0xa3, 255) : IM_COL32(0xab, 0xab, 0xab, 255);
  const ImU32 outline_bottom = dark ? IM_COL32(0x9c, 0x9c, 0x9c, 255) : IM_COL32(0xa4, 0xa4, 0xa4, 255);
  p->AddRectFilled(r.Min, r.Max, top, 2.0f);
  p->AddRectFilled(ImVec2(r.Min.x, r.GetCenter().y), r.Max, bottom, 2.0f, ImDrawFlags_RoundCornersBottom);
  p->AddRect(r.Min, r.Max, outline_bottom, 2.0f, 0, 1.0f);
  // the straight edges are drawn as crisp 1 px rects: an antialiased outline washes out to a much lighter grey
  const float c = 2.0f;  // corner radius
  p->AddRectFilled(ImVec2(r.Min.x + c, r.Min.y), ImVec2(r.Max.x - c, r.Min.y + 1.0f), outline_top);
  p->AddRectFilled(ImVec2(r.Min.x, r.Min.y + c), ImVec2(r.Min.x + 1.0f, r.Max.y - c), outline_top);
  p->AddRectFilled(ImVec2(r.Min.x + c, r.Max.y - 1.0f), ImVec2(r.Max.x - c, r.Max.y), outline_bottom);
  p->AddRectFilled(ImVec2(r.Max.x - 1.0f, r.Min.y + c), ImVec2(r.Max.x, r.Max.y - c), outline_bottom);
}

bool LogSlider::draw(const char *label, float width) {
  // Fusion QSlider: a full width groove with the part left of the handle filled, and a 13x13 handle on top
  // the groove is a grey track over the full width (QFusionStyle draws it with the outline color)
  const ImU32 groove_col = isDarkTheme() ? IM_COL32(0x2a, 0x2c, 0x2e, 255) : IM_COL32(0xc4, 0xc4, 0xc4, 255);
  const ImU32 fill_col = ImGui::GetColorU32(ImGuiCol_SliderGrab);
  ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_FrameBgActive, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_SliderGrab, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);  // a QSlider has no frame
  ImGui::SetNextItemWidth(width);
  bool changed = ImGui::SliderInt(label, &pos_, min_, max_, "", ImGuiSliderFlags_NoInput);
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(5);

  const ImVec2 bb_min = ImGui::GetItemRectMin(), bb_max = ImGui::GetItemRectMax();
  const float cy = (bb_min.y + bb_max.y) * 0.5f;
  const float groove_h = SLIDER_THICKNESS * 0.5f;
  const float handle_h = std::min(SLIDER_THICKNESS, bb_max.y - bb_min.y);
  const float x0 = bb_min.x + SLIDER_LENGTH * 0.5f, x1 = bb_max.x - SLIDER_LENGTH * 0.5f;
  const float t = max_ > min_ ? (float)(pos_ - min_) / (float)(max_ - min_) : 0.0f;
  const float hx = x0 + (x1 - x0) * t;
  ImDrawList *dl = ImGui::GetWindowDrawList();
  const float groove_y0 = cy - groove_h * 0.5f, groove_y1 = cy + groove_h * 0.5f;
  dl->AddRectFilled(ImVec2(bb_min.x, groove_y0), ImVec2(bb_max.x, groove_y1), groove_col, groove_h * 0.5f);
  dl->AddRectFilled(ImVec2(bb_min.x, groove_y0), ImVec2(hx, groove_y1), fill_col, groove_h * 0.5f);
  drawSliderHandle(dl, ImRect(ImVec2(hx - SLIDER_LENGTH * 0.5f, cy - handle_h * 0.5f),
                              ImVec2(hx + SLIDER_LENGTH * 0.5f, cy + handle_h * 0.5f)));
  return changed;
}

ChartsWidget::ChartsWidget() {
  // toolbar: see drawToolBar

  // range slider
  range_slider.setRange(1, settings.max_cached_minutes * 60);

  // zoom controls
  undo_zoom_enabled = false;
  redo_zoom_enabled = false;
  connections_.push_back(zoom_undo_stack.indexChanged.connect([this]() {
    undo_zoom_enabled = zoom_undo_stack.canUndo();
    redo_zoom_enabled = zoom_undo_stack.canRedo();
  }));

  // tabbar: see drawTabBar

  // charts
  charts_container = new ChartsContainer(this);

  // chart drag preview: see drawDragPreview

  // init settings
  current_theme = settings.theme;
  column_count = std::clamp(settings.chart_column_count, 1, MAX_COLUMN_COUNT);
  max_chart_range = std::clamp(settings.chart_range, 1, settings.max_cached_minutes * 60);
  display_range = std::make_pair(can->minSeconds(), can->minSeconds() + max_chart_range);
  range_slider.setValue(max_chart_range);
  updateToolBar();

  connections_.push_back(dbc()->fileChanged.connect([this]() { removeAll(); }));
  connections_.push_back(can->eventsMerged.connect([this](const MessageEventsMap &events) { eventsMerged(events); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *, bool) { updateState(); }));
  connections_.push_back(can->seeking.connect([this](double) { updateState(); }));
  connections_.push_back(can->timeRangeChanged.connect([this](const auto &range) { timeRangeChanged(range); }));
  connections_.push_back(settings.changed.connect([this]() { settingChanged(); }));
  connections_.push_back(seriesChanged.connect([this]() { updateTabBar(); }));

  setIsDocked(true);
  newTab();
}

ChartsWidget::~ChartsWidget() {
  for (auto c : charts) delete c;
  delete charts_container;
}

std::string ChartsWidget::whatsThis() const {
  return R"(
    <b>Chart View</b><br />
    <b>Click</b>: Click to seek to a corresponding time.<br />
    <b>Drag</b>: Zoom into the chart.<br />
    <b>Shift + Drag</b>: Scrub through the chart to view values.<br />
    <b>Right Mouse</b>: Open the context menu.<br />
  )";
}

void ChartsWidget::newTab() {
  static int tab_unique_id = 0;
  tabs_.push_back({tab_unique_id++, ""});
  int idx = tabs_.size() - 1;
  current_tab_index_ = idx;
  pending_tab_index_ = idx;
  updateTabBar();
}

void ChartsWidget::removeTab(int index) {
  int id = tabs_[index].id;
  for (auto &c : std::vector<ChartView *>(tab_charts[id])) {
    removeChart(c);
  }
  tab_charts.erase(id);
  tabs_.erase(tabs_.begin() + index);
  if (current_tab_index_ >= (int)tabs_.size()) current_tab_index_ = std::max<int>(tabs_.size() - 1, 0);
  updateTabBar();
}

void ChartsWidget::updateTabBar() {
  for (int i = 0; i < tabs_.size(); ++i) {
    const auto &charts_in_tab = tab_charts[tabs_[i].id];
    tabs_[i].text = "Tab " + std::to_string(i + 1) + " (" + std::to_string((int)charts_in_tab.size()) + ")";
  }
}

void ChartsWidget::drawTabBar() {
  // autoHide: the bar is only shown with more than one tab
  if (tabs_.size() <= 1) {
    current_tab_index_ = 0;
    pending_tab_index_ = -1;
    return;
  }
  int close_index = -1;
  if (ImGui::BeginTabBar("tabbar", ImGuiTabBarFlags_FittingPolicyScroll)) {
    for (int i = 0; i < tabs_.size(); ++i) {
      bool open = true;
      const std::string label = tabs_[i].text + "###tab" + std::to_string(tabs_[i].id);
      ImGuiTabItemFlags flags = i == pending_tab_index_ ? ImGuiTabItemFlags_SetSelected : 0;
      bool selected = ImGui::BeginTabItem(label.c_str(), &open, flags);
      tabs_[i].rect = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
      if (selected) {
        if (current_tab_index_ != i) {
          current_tab_index_ = i;
          updateLayout(true);  // currentChanged
        }
        ImGui::EndTabItem();
      }
      if (!open) close_index = i;
    }
    ImGui::EndTabBar();
  }
  pending_tab_index_ = -1;
  if (close_index >= 0) removeTab(close_index);
}

void ChartsWidget::eventsMerged(const MessageEventsMap &new_events) {
  std::vector<std::future<void>> futures;
  for (auto c : charts) {
    futures.push_back(ThreadPool::instance().run([c, &new_events]() { c->updateSeries(nullptr, &new_events); }));
  }
  for (auto &f : futures) f.get();
}

void ChartsWidget::timeRangeChanged(const std::optional<std::pair<double, double>> &time_range) {
  updateToolBar();
  updateState();
}

void ChartsWidget::zoomReset() {
  can->setTimeRange(std::nullopt);
  zoom_undo_stack.clear();
}

ImRect ChartsWidget::chartVisibleRect(ChartView *chart) {
  ImRect r = chart->rect;
  r.ClipWith(charts_scroll_viewport);
  return r;
}

void ChartsWidget::showValueTip(double sec) {
  if (chartDragActive()) sec = -1;  // Qt shows no value tip while a drag is in progress
  showTip(sec);
  if (sec < 0 && !value_tip_visible_) return;

  value_tip_visible_ = sec >= 0;
  for (auto c : currentCharts()) {
    value_tip_visible_ ? c->showTip(sec) : c->hideTip();
  }
}

void ChartsWidget::updateState() {
  if (charts.empty()) return;

  const auto &time_range = can->timeRange();
  const double cur_sec = can->currentSec();
  if (!time_range.has_value()) {
    double pos = (cur_sec - display_range.first) / std::max<float>(1.0, max_chart_range);
    if (pos < 0 || pos > 0.8) {
      display_range.first = std::max(can->minSeconds(), cur_sec - max_chart_range * 0.1);
    }
    double max_sec = std::min(display_range.first + max_chart_range, can->maxSeconds());
    display_range.first = std::max(can->minSeconds(), max_sec - max_chart_range);
    display_range.second = display_range.first + max_chart_range;
  }

  const auto &range = time_range ? *time_range : display_range;
  for (auto c : charts) {
    c->updatePlot(cur_sec, range.first, range.second);
  }
}

void ChartsWidget::setMaxChartRange(int value) {
  max_chart_range = settings.chart_range = range_slider.value();
  updateToolBar();
  updateState();
}

void ChartsWidget::setIsDocked(bool docked) {
  is_docked = docked;
  if (!docked) float_window_init_ = true;
  dock_btn_icon = is_docked ? icon::ARROW_UP_RIGHT_SQUARE : icon::ARROW_DOWN_LEFT_SQUARE;
  dock_btn_tooltip = is_docked ? "Float the charts window" : "Dock the charts window";
}

void ChartsWidget::updateToolBar() {
  title_label = "Charts: " + std::to_string(charts.size());
  columns_action_text = "Columns: " + std::to_string(column_count);
  range_lb = utils::formatSeconds(max_chart_range);

  bool is_zoomed = can->timeRange().has_value();
  range_lb_visible = !is_zoomed;
  range_slider_visible = !is_zoomed;
  undo_zoom_visible = is_zoomed;
  redo_zoom_visible = is_zoomed;
  reset_zoom_visible = is_zoomed;
  if (is_zoomed) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f-%.2f", can->timeRange()->first, can->timeRange()->second);
    reset_zoom_text = buf;
  } else {
    reset_zoom_text = "";
  }
  remove_all_enabled = !charts.empty();
}

void ChartsWidget::drawToolBar() {
  static const std::array<const char *, 3> types{"Line", "Step", "Scatter"};
  // QToolBar metrics: the items sit next to each other, the buttons only carry the auto raise margin
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(TOOLBAR_ITEM_SPACING, ImGui::GetStyle().ItemSpacing.y));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(TOOLBAR_BUTTON_PADDING, ImGui::GetStyle().FramePadding.y));
  const ImGuiStyle &style = ImGui::GetStyle();
  float slider_width = 150.0f;

  // QToolBar lays the actions out in order; what does not fit goes into the ">>" extension menu
  struct Item {
    float width;
    std::function<void()> draw;
  };
  std::vector<Item> left, right;

  left.push_back({buttonWidth(icon::FILE_PLUS), [this]() {
    if (toolButton("new_plot_btn", icon::FILE_PLUS, "New Chart")) newChart();
  }});
  left.push_back({buttonWidth(icon::WINDOW_STACK), [this]() {
    if (toolButton("new_tab_btn", icon::WINDOW_STACK, "New Tab")) newTab();
  }});
  // title_label carries a trailing PM_LayoutHorizontalSpacing content margin
  left.push_back({ImGui::CalcTextSize(title_label.c_str()).x + LAYOUT_HORIZONTAL_SPACING, [this]() {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(title_label.c_str());
    ImGui::SameLine(0.0f, LAYOUT_HORIZONTAL_SPACING);
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
  }});

  // chart type menu
  const std::string chart_type_text = std::string("Type: ") + types[std::clamp(settings.chart_series_type, 0, 2)];
  left.push_back({menuButtonWidth(chart_type_text), [this, chart_type_text]() {
    if (menuButton("chart_type", chart_type_text, "chart_type_menu")) ImGui::OpenPopup("chart_type_menu");
    if (ImGui::BeginPopup("chart_type_menu")) {
      for (int i = 0; i < types.size(); ++i) {
        if (ImGui::MenuItem(types[i])) {
          settings.chart_series_type = i;
          settingChanged();
        }
      }
      ImGui::EndPopup();
    }
  }});

  // columns menu
  if (columns_action_visible) {
    left.push_back({menuButtonWidth(columns_action_text), [this]() {
      if (menuButton("columns", columns_action_text, "columns_menu")) ImGui::OpenPopup("columns_menu");
      if (ImGui::BeginPopup("columns_menu")) {
        for (int i = 0; i < MAX_COLUMN_COUNT; ++i) {
          if (ImGui::MenuItem(std::to_string(i + 1).c_str())) setColumnCount(i + 1);
        }
        ImGui::EndPopup();
      }
    }});
  }

  // the spacer right aligns the rest
  if (range_lb_visible) {
    right.push_back({ImGui::CalcTextSize(range_lb.c_str()).x, [this]() {
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(range_lb.c_str());
    }});
  }
  if (range_slider_visible) {
    right.push_back({slider_width, [this, &slider_width]() {
      if (range_slider.draw("##range_slider", slider_width)) setMaxChartRange(range_slider.value());
      ImGui::SetItemTooltip("Set the chart range");
    }});
  }
  if (undo_zoom_visible) {
    right.push_back({buttonWidth(icon::ARROW_COUNTERCLOCKWISE), [this]() {
      ImGui::BeginDisabled(!undo_zoom_enabled);
      if (toolButton("undo_zoom", icon::ARROW_COUNTERCLOCKWISE, "Undo Zoom")) zoom_undo_stack.undo();
      ImGui::EndDisabled();
    }});
  }
  if (redo_zoom_visible) {
    right.push_back({buttonWidth(icon::ARROW_CLOCKWISE), [this]() {
      ImGui::BeginDisabled(!redo_zoom_enabled);
      if (toolButton("redo_zoom", icon::ARROW_CLOCKWISE, "Redo Zoom")) zoom_undo_stack.redo();
      ImGui::EndDisabled();
    }});
  }
  if (reset_zoom_visible) {
    right.push_back({buttonWidth(std::string(icon::ZOOM_OUT) + " " + reset_zoom_text), [this]() {
      if (toolButton("reset_zoom_btn", icon::ZOOM_OUT, "Reset Zoom", reset_zoom_text.c_str())) zoomReset();
    }});
  }
  right.push_back({buttonWidth(icon::X_SQUARE), [this]() {
    ImGui::BeginDisabled(!remove_all_enabled);
    if (toolButton("remove_all_btn", icon::X_SQUARE, "Remove all charts")) removeAll();
    ImGui::EndDisabled();
  }});
  right.push_back({buttonWidth(dock_btn_icon), [this]() {
    if (toolButton("dock_btn", dock_btn_icon, dock_btn_tooltip.c_str())) toggleChartsDocking();
  }});

  size_t n_left = left.size(), n_right = right.size();
  // QToolBarLayout: the item widths plus one spacing between neighbors
  auto group_width = [&](const std::vector<Item> &items, size_t count) {
    float w = 0;
    for (size_t i = 0; i < count; ++i) w += items[i].width + (i ? style.ItemSpacing.x : 0);
    return w;
  };
  auto total_width = [&]() {
    float w = group_width(left, n_left) + group_width(right, n_right);
    if (n_left > 0 && n_right > 0) w += style.ItemSpacing.x;
    return w;
  };

  float avail = ImGui::GetContentRegionAvail().x;
  if (range_slider_visible && total_width() > avail) {
    // QSlider shrinks first (never below 40px), the buttons stay pinned to the right edge
    const float shrink = std::min(slider_width - 40.0f, total_width() - avail);
    if (shrink > 0.0f) {
      slider_width -= shrink;
      right[range_lb_visible ? 1 : 0].width = slider_width;
    }
  }

  const float chevron_w = buttonWidth(icon::RAQUO);
  bool overflow = total_width() > avail;
  if (overflow) avail -= chevron_w + style.ItemSpacing.x;
  while (total_width() > avail && (n_right > 0 || n_left > 0)) {
    if (n_right > 0) --n_right; else --n_left;
  }

  float right_width = group_width(right, n_right);
  if (overflow) right_width += (n_right > 0 ? style.ItemSpacing.x : 0) + chevron_w;

  for (size_t i = 0; i < n_left; ++i) {
    if (i > 0) ImGui::SameLine();
    left[i].draw();
  }
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - right_width));
  for (size_t i = 0; i < n_right; ++i) {
    if (i > 0) ImGui::SameLine();
    right[i].draw();
  }
  if (overflow) {
    if (n_right > 0) ImGui::SameLine();
    if (ImGui::Button((std::string(icon::RAQUO) + "###toolbar_ext").c_str())) ImGui::OpenPopup("toolbar_ext_menu");
    ImGui::SetItemTooltip("More");
    // the popup opens inward: its right edge is aligned with the button so it stays inside the window
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y), ImGuiCond_Always, ImVec2(1, 0));
    if (ImGui::BeginPopup("toolbar_ext_menu")) {
      for (size_t i = n_left; i < left.size(); ++i) left[i].draw();
      for (size_t i = n_right; i < right.size(); ++i) right[i].draw();
      ImGui::EndPopup();
    }
  }
  ImGui::PopStyleVar(2);
}

void ChartsWidget::settingChanged() {
  if (std::exchange(current_theme, settings.theme) != current_theme) {
    // the toolbar icons are font glyphs; nothing to reload
  }
  if (range_slider.maximum() != settings.max_cached_minutes * 60) {
    range_slider.setRange(1, settings.max_cached_minutes * 60);
  }
  for (auto c : charts) {
    // the fixed height (settings.chart_height) is read in ChartView::draw
    c->setSeriesType((SeriesType)settings.chart_series_type);
  }
}

ChartView *ChartsWidget::findChart(const MessageId &id, const cabana::Signal *sig) {
  for (auto c : charts)
    if (c->hasSignal(id, sig)) return c;
  return nullptr;
}

ChartView *ChartsWidget::createChart(int pos) {
  auto chart = new ChartView(can->timeRange().value_or(display_range), this);
  // fixed height / min width / size policy: see ChartView::draw
  pos = std::clamp(pos, 0, (int)charts.size());
  charts.insert(charts.begin() + pos, chart);
  currentCharts().insert(currentCharts().begin() + pos, chart);
  updateLayout(true);
  updateToolBar();
  return chart;
}

void ChartsWidget::showChart(const MessageId &id, const cabana::Signal *sig, bool show, bool merge) {
  ChartView *chart = findChart(id, sig);
  if (show && !chart) {
    chart = merge && currentCharts().size() > 0 ? currentCharts().front() : createChart();
    chart->addSignal(id, sig);
    updateState();
  } else if (!show && chart) {
    chart->removeIf([&](auto &s) { return s.msg_id == id && s.sig == sig; });
  }
}

void ChartsWidget::splitChart(ChartView *src_chart) {
  if (src_chart->sigs.size() > 1) {
    int pos = std::find(charts.begin(), charts.end(), src_chart) - charts.begin() + 1;
    for (auto it = src_chart->sigs.begin() + 1; it != src_chart->sigs.end(); /**/) {
      auto c = createChart(pos);
      // Restore to the original color
      it->color = it->sig->color;
      c->sigs.emplace_back(std::move(*it));
      c->updateAxisY();
      c->updateTitle();
      it = src_chart->sigs.erase(it);
    }
    src_chart->updateAxisY();
    src_chart->updateTitle();
    updateState();
  }
}

std::vector<std::string> ChartsWidget::serializeChartIds() const {
  std::vector<std::string> chart_ids;
  for (auto c : charts) {
    std::string ids;
    for (const auto& s : c->sigs) {
      if (!ids.empty()) ids += ',';
      ids += s.msg_id.toString() + "|" + s.sig->name;
    }
    chart_ids.push_back(ids);
  }
  std::reverse(chart_ids.begin(), chart_ids.end());
  return chart_ids;
}

void ChartsWidget::restoreChartsFromIds(const std::vector<std::string>& chart_ids) {
  for (const auto& chart_id : chart_ids) {
    int index = 0;
    size_t start = 0;
    while (start <= chart_id.size()) {
      size_t end = chart_id.find(',', start);
      if (end == std::string::npos) end = chart_id.size();
      const std::string part = chart_id.substr(start, end - start);
      start = end + 1;
      const size_t sep = part.find('|');
      if (sep == std::string::npos || part.find('|', sep + 1) != std::string::npos) continue;
      MessageId msg_id = MessageId::fromString(part.substr(0, sep));
      if (auto* msg = dbc()->msg(msg_id))
        if (auto* sig = msg->sig(part.substr(sep + 1)))
          showChart(msg_id, sig, true, index++ > 0);
    }
  }
}

void ChartsWidget::setColumnCount(int n) {
  n = std::clamp(n, 1, MAX_COLUMN_COUNT);
  if (column_count != n) {
    column_count = settings.chart_column_count = n;
    updateToolBar();
    updateLayout();
  }
}

void ChartsWidget::updateLayout(bool force) {
  // the container has not been drawn yet (docked/floated this frame): keep the last known layout
  const float container_width = charts_container->geometry.GetWidth();
  if (container_width <= 0) return;

  int n = MAX_COLUMN_COUNT;
  for (; n > 1; --n) {
    if ((n * CHART_MIN_WIDTH + (n - 1) * charts_container->horizontalSpacing()) < container_width) break;
  }

  bool show_column_cb = n > 1;
  columns_action_visible = show_column_cb;

  n = std::min(column_count, n);
  if (n != current_column_count || force) {
    current_column_count = n;
    // the grid is laid out every frame in ChartsContainer::draw from currentCharts()
  }
}

void ChartsWidget::startChartDrag(ChartView *chart, const ImVec2 &global_pos) {
  stopAutoScroll();
  drag = {.source = chart, .press_pos = global_pos};
  showValueTip(-1);  // Qt shows no value tip while a drag is in progress
  // Qt grabs the chart into a pixmap scaled to CHART_MIN_WIDTH; the ghost re-renders the tile at that width
  drag_preview_size = ImVec2(CHART_MIN_WIDTH, (float)settings.chart_height);
}

void ChartsWidget::dragChartMove(const ImVec2 &global_pos) {
  if (!drag.active) {
    ImVec2 d = global_pos - drag.press_pos;
    if (std::abs(d.x) + std::abs(d.y) < START_DRAG_DISTANCE) return;
    drag.active = true;
    drag_preview_visible = true;
  }
  drag_preview_pos = global_pos + ImVec2(5, 5);

  // hovering a tab switches to it so the chart can be dropped into another tab
  int tab = -1;
  for (int i = 0; i < tabs_.size(); ++i) {
    if (tabs_.size() > 1 && tabs_[i].rect.Contains(global_pos)) tab = i;
  }
  if (tab >= 0 && tab != current_tab_index_) {
    pending_tab_index_ = tab;
  }

  const ImVec2 container_pos = global_pos;
  ChartView *target = nullptr;
  for (auto c : currentCharts()) {
    if (c != drag.source && c->rect.Contains(container_pos)) {
      target = c;
      break;
    }
  }
  if (std::exchange(drop_target, target) != target) {
    for (auto c : charts) c->setDropHighlight(c == target);
  }
  bool in_viewport = charts_scroll_viewport.Contains(global_pos);
  bool on_background = !target && in_viewport && !charts_container->childAt(container_pos);
  charts_container->drawDropIndicator(on_background ? container_pos : ImVec2());

  if (in_viewport) {
    startAutoScroll(global_pos);
  }
}

void ChartsWidget::cancelChartDrag() {
  drag = {};
  stopAutoScroll();
  drag_preview_visible = false;
  charts_container->drawDropIndicator({});
  if (auto target = std::exchange(drop_target, nullptr)) target->setDropHighlight(false);
}

void ChartsWidget::dragChartRelease(const ImVec2 &global_pos) {
  ChartView *source = drag.source;
  bool active = drag.active;
  ChartView *target = drop_target;
  cancelChartDrag();
  if (!active) return;

  const ImVec2 container_pos = global_pos;
  bool in_viewport = charts_scroll_viewport.Contains(global_pos);
  if (target) {
    // merge source into target
    target->takeSignalsFrom(source);
  } else if (in_viewport && !charts_container->childAt(container_pos)) {
    // reorder within the current tab
    auto w = charts_container->getDropAfter(container_pos);
    if (w != source) {
      for (auto &[_, list] : tab_charts) {
        list.erase(std::remove(list.begin(), list.end(), source), list.end());
      }
      auto &cur = currentCharts();
      int to = w ? std::find(cur.begin(), cur.end(), w) - cur.begin() + 1 : 0;
      cur.insert(cur.begin() + to, source);
      updateLayout(true);
      updateTabBar();
    }
  }
}

void ChartsWidget::drawDragPreview() {
  if (!drag_preview_visible || !drag.source) return;
  // Qt drags a 50% alpha pixmap of the whole tile (header + axes + plot): the tile is re-rendered into a
  // window that takes no input, so the live chart keeps handling the mouse.
  ImGui::SetNextWindowPos(drag_preview_pos);
  ImGui::SetNextWindowSize(drag_preview_size);
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                 ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking;
  if (ImGui::Begin("##chart_drag_ghost", nullptr, flags)) {
    drag.source->drawGhost(drag_preview_size.x);
  }
  ImGui::End();
  ImGui::PopStyleVar(3);
}

void ChartsWidget::startAutoScroll(const ImVec2 &global_pos) {
  auto_scroll_pos = global_pos;
  if (!auto_scroll_timer_active) auto_scroll_timer_next = ImGui::GetTime() + 0.05;
  auto_scroll_timer_active = true;
}

void ChartsWidget::stopAutoScroll() {
  auto_scroll_timer_active = false;
  auto_scroll_count = 0;
}

void ChartsWidget::doAutoScroll() {
  if (!charts_scroll) return;
  const int page_step = charts_scroll_viewport.GetHeight();
  if (auto_scroll_count < page_step) {
    ++auto_scroll_count;
  }

  int value = charts_scroll->Scroll.y;
  ImVec2 pos = auto_scroll_pos;
  ImRect area = charts_scroll_viewport;

  int new_value = value;
  if (pos.y - area.Min.y < settings.chart_height / 2) {
    new_value = value - auto_scroll_count;
  } else if (area.Max.y - pos.y < settings.chart_height / 2) {
    new_value = value + auto_scroll_count;
  }
  new_value = std::clamp<int>(new_value, 0, charts_scroll->ScrollMax.y);
  if (new_value != value) ImGui::SetScrollY(charts_scroll, new_value);
  if (value == new_value) {
    stopAutoScroll();
  } else if (chartDragActive()) {
    // refresh the drop indicator/target at the new scroll position
    dragChartMove(auto_scroll_pos);
  }
}

ImVec2 ChartsWidget::minimumSizeHint() const {
  return ImVec2(CHART_MIN_WIDTH * 1.5, 0);
}

void ChartsWidget::newChart() {
  execSignalSelector(std::make_unique<SignalSelector>("New Chart"), nullptr, [this](SignalSelector &dlg) {
    auto items = dlg.seletedItems();
    if (!items.empty()) {
      auto c = createChart();
      for (auto it : items) {
        c->addSignal(it->msg_id, it->sig);
      }
      updateState();
    }
  });
}

void ChartsWidget::execSignalSelector(std::unique_ptr<SignalSelector> dlg, ChartView *owner, std::function<void(SignalSelector &)> accepted) {
  signal_selector_ = std::move(dlg);
  signal_selector_owner_ = owner;
  signal_selector_accepted_ = std::move(accepted);
  signal_selector_->open();
}

void ChartsWidget::removeChart(ChartView *chart) {
  if (drag.source == chart) cancelChartDrag();
  if (drop_target == chart) drop_target = nullptr;
  if (signal_selector_owner_ == chart) {
    signal_selector_owner_ = nullptr;
    signal_selector_accepted_ = nullptr;
  }
  charts.erase(std::remove(charts.begin(), charts.end(), chart), charts.end());
  deleted_charts_.emplace_back(chart);  // deleteLater
  for (auto &[_, list] : tab_charts) {
    list.erase(std::remove(list.begin(), list.end(), chart), list.end());
  }
  updateToolBar();
  updateLayout(true);
  seriesChanged();
}

void ChartsWidget::removeAll() {
  while (tabs_.size() > 1) {
    tabs_.erase(tabs_.begin() + 1);
  }
  current_tab_index_ = 0;
  tab_charts.clear();

  if (!charts.empty()) {
    for (auto c : charts) {
      if (drag.source == c) cancelChartDrag();
      if (signal_selector_owner_ == c) {
        signal_selector_owner_ = nullptr;
        signal_selector_accepted_ = nullptr;
      }
      deleted_charts_.emplace_back(c);  // may be called from a chart's draw; freed next frame
    }
    charts.clear();
    drop_target = nullptr;
    updateToolBar();
    seriesChanged();
  }
  zoomReset();
}

void ChartsWidget::eventFilter() {
  // route all mouse events to the chart drag, even when the source chart is hidden by a tab switch
  if (chartDragActive()) {
    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      dragChartMove(ImGui::GetMousePos());
    } else {
      dragChartRelease(ImGui::GetMousePos());
    }
  }

  if (!value_tip_visible_) return;

  // the tip is drawn on the foreground draw list, so the mouse is never "on the tip"
  const ImVec2 delta = ImGui::GetIO().MouseDelta;
  if ((delta.x != 0 || delta.y != 0) && !any_plot_hovered_) {
    showValueTip(-1);
  } else if (!any_plot_hovered_ && !ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
    showValueTip(-1);  // QEvent::Leave: the mouse is not over the (floating) charts window anymore
  }
}

void ChartsWidget::event() {
  bool back_button = false;
  // Resize
  updateLayout();
  // MouseButtonPress: Qt::BackButton
  if (ImGui::IsMouseClicked(3) && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
    back_button = true;
  }
  // NativeGesture (swipe back) has no imgui equivalent
  // WindowDeactivate / FocusOut
  if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow)) {
    if (chartDragActive()) cancelChartDrag();
    showValueTip(-1);
  }

  if (back_button) {
    zoom_undo_stack.undo();
  }
}

void ChartsWidget::draw() {
  deleted_charts_.clear();
  // the floating window is a top level window sized to its contents: keep it inside the main viewport so its
  // toolbar stays reachable, then let the user resize it
  if (float_window_init_ && !is_docked) {
    float_window_init_ = false;
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    const ImVec2 size(viewport->WorkSize.x * 0.6f, viewport->WorkSize.y * 0.6f);
    ImGui::SetWindowSize(size);
    ImGui::SetWindowPos(viewport->WorkPos + (viewport->WorkSize - size) * 0.5f);
  }
  ImGui::PushID(this);
  // timers
  if (auto_scroll_timer_active && ImGui::GetTime() >= auto_scroll_timer_next) {
    auto_scroll_timer_next = ImGui::GetTime() + 0.05;
    doAutoScroll();
  }
  event();
  // the drop target and indicator must be resolved before the charts are painted, otherwise the highlight
  // lags a frame behind the target used on release and the drop lands on the wrong chart
  eventFilter();

  drawToolBar();
  drawTabBar();

  // charts scroll area
  any_plot_hovered_ = false;
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
  if (ImGui::BeginChild("charts_scroll", ImVec2(0, 0), ImGuiChildFlags_None, 0)) {
    charts_scroll = ImGui::GetCurrentWindow();
    charts_scroll_viewport = charts_scroll->InnerRect;
    charts_container->draw();
  }
  ImGui::EndChild();
  ImGui::PopStyleColor();

  drawDragPreview();

  if (signal_selector_ && !signal_selector_->draw()) {
    auto dlg = std::move(signal_selector_);
    auto accepted = std::move(signal_selector_accepted_);
    signal_selector_owner_ = nullptr;
    if (dlg->accepted() && accepted) accepted(*dlg);
  }
  ImGui::PopID();
}

// ChartsContainer

ChartsContainer::ChartsContainer(ChartsWidget *parent) : charts_widget(parent) {}

int ChartsContainer::horizontalSpacing() const {
  return CHART_SPACING;
}

void ChartsContainer::draw() {
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  const ImVec2 start = ImGui::GetCursorScreenPos();
  geometry = ImRect(start, start + ImVec2(window->InnerRect.GetWidth(), 0));
  charts_widget->updateLayout();

  const int n = std::max(charts_widget->current_column_count, 1);
  const float spacing = horizontalSpacing();
  const float width = (geometry.GetWidth() - (n - 1) * spacing) / n;
  const ImVec2 origin = ImGui::GetCursorScreenPos() + ImVec2(0, CHART_SPACING);
  auto current_charts = charts_widget->currentCharts();  // copy: drawing may remove charts
  float bottom = origin.y;
  const bool aligned = ImPlot::BeginAlignedPlots("charts_align", true);
  for (int i = 0; i < current_charts.size(); ++i) {
    ImVec2 pos = origin + ImVec2((i % n) * (width + spacing), (i / n) * (settings.chart_height + spacing));
    ImGui::SetCursorScreenPos(pos);
    current_charts[i]->draw(width);
    bottom = std::max(bottom, pos.y + settings.chart_height);
    if (current_charts[i]->plot_hovered) charts_widget->any_plot_hovered_ = true;  // the window must be hovered too
  }
  if (aligned) ImPlot::EndAlignedPlots();
  ImGui::SetCursorScreenPos(ImVec2(origin.x, bottom));
  ImGui::Dummy(ImVec2(geometry.GetWidth(), CHART_SPACING));
  geometry.Max.y = bottom + CHART_SPACING;
  paintEvent();
}

void ChartsContainer::paintEvent() {
  if (!(drop_indictor_pos.x == 0 && drop_indictor_pos.y == 0) && !childAt(drop_indictor_pos)) {
    ImRect r = geometry;
    r.Max.y = r.Min.y + CHART_SPACING;
    if (auto insert_after = getDropAfter(drop_indictor_pos)) {
      float h = r.GetHeight();
      r.Min.y = insert_after->rect.Max.y;
      r.Max.y = r.Min.y + h;
    }

    ImGui::GetWindowDrawList()->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_Header));
  }
}

ChartView *ChartsContainer::getDropAfter(const ImVec2 &pos) const {
  auto it = std::find_if(charts_widget->currentCharts().crbegin(), charts_widget->currentCharts().crend(), [&pos](auto c) {
    auto area = c->rect;
    return pos.x >= area.Min.x && pos.x <= area.Max.x && pos.y >= area.Max.y;
  });
  return it == charts_widget->currentCharts().crend() ? nullptr : *it;
}

ChartView *ChartsContainer::childAt(const ImVec2 &pos) const {
  for (auto c : charts_widget->currentCharts()) {
    if (c->rect.Contains(pos)) return c;
  }
  return nullptr;
}
