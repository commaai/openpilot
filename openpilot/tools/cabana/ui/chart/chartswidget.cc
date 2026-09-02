#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/chartswidget.h"

#include "tools/cabana/ui/threadpool.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <future>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

const int MAX_COLUMN_COUNT = 4;
const int CHART_SPACING = 4;
const int START_DRAG_DISTANCE = 10;
const float TOOLBAR_BUTTON_PADDING = 4.0f;  // auto raise button horizontal margin
const float MENU_ARROW_SIZE = 6.0f;         // dropdown arrow on a menu button
const float MENU_ARROW_SPACING = 5.0f;      // gap between the label and the dropdown arrow
const float LAYOUT_HORIZONTAL_SPACING = 6.0f;

static float buttonWidth(const std::string &label) {
  return ImGui::CalcTextSize(label.c_str(), nullptr, true).x + ImGui::GetStyle().FramePadding.x * 2;
}

// an auto-raise button with a menu: flat until hovered, with a small dropdown arrow drawn after the text
// at the text baseline
static float menuButtonWidth(const std::string &text) {
  const ImGuiStyle &style = ImGui::GetStyle();
  return ImGui::CalcTextSize(text.c_str(), nullptr, true).x + MENU_ARROW_SPACING + MENU_ARROW_SIZE +
         style.FramePadding.x * 2;
}

static bool menuButton(const char *id, const std::string &text, const char *popup_id) {
  const ImGuiStyle &style = ImGui::GetStyle();
  const bool popup_open = ImGui::IsPopupOpen(popup_id);
  // no frame, transparent until hovered; the button is drawn pressed while the menu is open
  ImGui::PushStyleColor(ImGuiCol_Button, popup_open ? style.Colors[ImGuiCol_ButtonActive] : ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
  bool clicked = ImGui::Button((text + "###" + id).c_str(), ImVec2(menuButtonWidth(text), 0.0f));
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor();
  // a 6 px arrow in the disabled text color, right after the text, sitting on the text baseline
  const ImVec2 min = ImGui::GetItemRectMin();
  const float x = min.x + style.FramePadding.x + ImGui::CalcTextSize(text.c_str(), nullptr, true).x +
                  MENU_ARROW_SPACING;
  const float baseline = min.y + style.FramePadding.y + ImGui::GetFontBaked()->Ascent;
  ImGui::GetWindowDrawList()->AddTriangleFilled(ImVec2(x, baseline - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(x + MENU_ARROW_SIZE, baseline - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(x + MENU_ARROW_SIZE * 0.5f, baseline),
                                                ImGui::GetColorU32(ImGuiCol_TextDisabled));
  // the menu drops down from below the button, not at the mouse cursor
  ImGui::SetNextWindowPos(ImVec2(min.x, ImGui::GetItemRectMax().y), ImGuiCond_Always);
  return clicked;
}

bool LogSlider::draw(const char *label, float width) {
  return fusionSliderInt(label, &pos_, min_, max_, width);
}

ChartsWidget::ChartsWidget() {
  range_slider.setRange(1, settings.max_cached_minutes * 60);

  tabbar.setAutoHide(true);
  tabbar.setExpanding(false);
  tabbar.setUsesScrollButtons(true);
  tabbar.setTabsClosable(true);

  column_count = std::clamp(settings.chart_column_count, 1, MAX_COLUMN_COUNT);
  max_chart_range = std::clamp(settings.chart_range, 1, settings.max_cached_minutes * 60);
  display_range = std::make_pair(can->minSeconds(), can->minSeconds() + max_chart_range);
  range_slider.setValue(max_chart_range);

  connections_.push_back(dbc()->fileChanged.connect([this]() { removeAll(); }));
  connections_.push_back(can->eventsMerged.connect([this](const MessageEventsMap &events) { eventsMerged(events); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *, bool) { updateState(); }));
  connections_.push_back(can->seeking.connect([this](double) { updateState(); }));
  connections_.push_back(can->timeRangeChanged.connect([this](const auto &range) { timeRangeChanged(range); }));
  connections_.push_back(settings.changed.connect([this]() { settingChanged(); }));
  connections_.push_back(seriesChanged.connect([this]() { updateTabBar(); }));
  connections_.push_back(tabbar.tabCloseRequested.connect([this](int index) { removeTab(index); }));
  connections_.push_back(tabbar.currentChanged.connect([this](int index) {
    if (index != -1) updateLayout();
  }));

  setIsDocked(true);
  newTab();
}

ChartsWidget::~ChartsWidget() {
  for (auto c : charts) delete c;
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
  int idx = tabbar.addTab("");
  tabbar.setTabData(idx, tab_unique_id++);
  tabbar.setCurrentIndex(idx);
  updateTabBar();
}

void ChartsWidget::removeTab(int index) {
  int id = tabbar.tabData(index);
  for (auto &c : std::vector<ChartView *>(tab_charts[id])) {
    removeChart(c);
  }
  tab_charts.erase(id);
  tabbar.removeTab(index);
  updateTabBar();
}

void ChartsWidget::updateTabBar() {
  for (int i = 0; i < tabbar.count(); ++i) {
    const auto &charts_in_tab = tab_charts[tabbar.tabData(i)];
    tabbar.setTabText(i, "Tab " + std::to_string(i + 1) + " (" + std::to_string((int)charts_in_tab.size()) + ")");
  }
}

void ChartsWidget::eventsMerged(const MessageEventsMap &new_events) {
  std::vector<std::future<void>> futures;
  for (auto c : charts) {
    futures.push_back(ThreadPool::instance().run([c, &new_events]() { c->updateSeries(nullptr, &new_events); }));
  }
  for (auto &f : futures) f.get();
}

void ChartsWidget::timeRangeChanged(const std::optional<std::pair<double, double>> &time_range) {
  updateState();
}

void ChartsWidget::zoomReset() {
  can->setTimeRange(std::nullopt);
  zoom_undo_stack.clear();
}

ImRect ChartsWidget::chartVisibleRect(ChartView *chart) {
  ImRect r = chart->layout_.rect;
  r.ClipWith(charts_scroll_viewport);
  return r;
}

void ChartsWidget::showValueTip(double sec) {
  if (chartDragActive()) sec = -1;  // no value tip while a drag is in progress
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
  updateState();
}

void ChartsWidget::setIsDocked(bool docked) {
  is_docked = docked;
  if (!docked) float_window_init_ = true;
}

void ChartsWidget::drawToolBar() {
  static const std::array<const char *, 3> types{"Line", "Step", "Scatter"};
  // the toolbar items sit next to each other, the buttons only carry the auto raise margin
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(TOOLBAR_ITEM_SPACING, ImGui::GetStyle().ItemSpacing.y));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(TOOLBAR_BUTTON_PADDING, ImGui::GetStyle().FramePadding.y));
  const ImGuiStyle &style = ImGui::GetStyle();
  float slider_width = 150.0f;
  const bool is_zoomed = can->timeRange().has_value();

  // the items are laid out in order, the left group then the right one; what does not fit goes into the
  // ">>" extension menu. the labels are captured by reference, they outlive the draw calls below
  struct Item {
    float width;
    std::function<void()> draw;
  };
  std::vector<Item> items;

  items.push_back({buttonWidth(icon::PLUS_SQUARE), [this]() {
    if (toolButton("new_plot_btn", icon::PLUS_SQUARE, "New Chart")) newChart();
  }});
  items.push_back({buttonWidth(icon::WINDOW_STACK), [this]() {
    if (toolButton("new_tab_btn", icon::WINDOW_STACK, "New Tab")) newTab();
  }});
  const std::string title_label = "Charts: " + std::to_string(charts.size());
  items.push_back({ImGui::CalcTextSize(title_label.c_str()).x + LAYOUT_HORIZONTAL_SPACING, [&title_label]() {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(title_label.c_str());
    ImGui::SameLine(0.0f, LAYOUT_HORIZONTAL_SPACING);
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
  }});

  const std::string chart_type_text = std::string("Type:  ") + types[std::clamp(settings.chart_series_type, 0, 2)];
  items.push_back({menuButtonWidth(chart_type_text), [this, &chart_type_text]() {
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

  const std::string columns_action_text = "Columns:  " + std::to_string(column_count);
  if (columns_action_visible) {
    items.push_back({menuButtonWidth(columns_action_text), [this, &columns_action_text]() {
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
  const size_t left_count = items.size();
  size_t slider_index = (size_t)-1;
  const std::string range_lb = is_zoomed ? std::string() : utils::formatSeconds(max_chart_range);
  std::string reset_zoom_text;
  if (!is_zoomed) {
    items.push_back({ImGui::CalcTextSize(range_lb.c_str()).x, [&range_lb]() {
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(range_lb.c_str());
    }});
    slider_index = items.size();
    items.push_back({slider_width, [this, &slider_width]() {
      if (range_slider.draw("##range_slider", slider_width)) setMaxChartRange(range_slider.value());
      ImGui::SetItemTooltip("Set the chart range");
    }});
  } else {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f-%.2f", can->timeRange()->first, can->timeRange()->second);
    reset_zoom_text = buf;
    items.push_back({buttonWidth(icon::ARROW_COUNTERCLOCKWISE), [this]() {
      ImGui::BeginDisabled(!zoom_undo_stack.canUndo());
      if (toolButton("undo_zoom", icon::ARROW_COUNTERCLOCKWISE, "Undo Zoom")) zoom_undo_stack.undo();
      ImGui::EndDisabled();
    }});
    items.push_back({buttonWidth(icon::ARROW_CLOCKWISE), [this]() {
      ImGui::BeginDisabled(!zoom_undo_stack.canRedo());
      if (toolButton("redo_zoom", icon::ARROW_CLOCKWISE, "Redo Zoom")) zoom_undo_stack.redo();
      ImGui::EndDisabled();
    }});
    items.push_back({buttonWidth(std::string(icon::ZOOM_OUT) + " " + reset_zoom_text), [this, &reset_zoom_text]() {
      if (toolButton("reset_zoom_btn", icon::ZOOM_OUT, "Reset Zoom", reset_zoom_text.c_str())) zoomReset();
    }});
  }
  items.push_back({buttonWidth(icon::X_SQUARE), [this]() {
    ImGui::BeginDisabled(charts.empty());
    if (toolButton("remove_all_btn", icon::X_SQUARE, "Remove all charts")) removeAll();
    ImGui::EndDisabled();
  }});
  const char *dock_btn_icon = is_docked ? icon::ARROW_UP_RIGHT_SQUARE : icon::ARROW_DOWN_LEFT_SQUARE;
  items.push_back({buttonWidth(dock_btn_icon), [this, dock_btn_icon]() {
    if (toolButton("dock_btn", dock_btn_icon, is_docked ? "Float the charts window" : "Dock the charts window")) toggleChartsDocking();
  }});

  size_t n_left = left_count, n_right = items.size() - left_count;
  // the item widths plus one spacing between neighbors
  auto group_width = [&](size_t first, size_t count) {
    float w = 0;
    for (size_t i = 0; i < count; ++i) w += items[first + i].width + (i ? style.ItemSpacing.x : 0);
    return w;
  };
  auto total_width = [&]() {
    float w = group_width(0, n_left) + group_width(left_count, n_right);
    if (n_left > 0 && n_right > 0) w += style.ItemSpacing.x;
    return w;
  };

  float avail = ImGui::GetContentRegionAvail().x;
  if (!is_zoomed && total_width() > avail) {
    // the slider shrinks first (never below 40px), the buttons stay pinned to the right edge
    const float shrink = std::min(slider_width - 40.0f, total_width() - avail);
    if (shrink > 0.0f) {
      slider_width -= shrink;
      items[slider_index].width = slider_width;
    }
  }

  const float chevron_w = buttonWidth(icon::RAQUO);
  bool overflow = total_width() > avail;
  if (overflow) avail -= chevron_w + style.ItemSpacing.x;
  while (total_width() > avail && (n_right > 0 || n_left > 0)) {
    if (n_right > 0) --n_right; else --n_left;
  }

  float right_width = group_width(left_count, n_right);
  if (overflow) right_width += (n_right > 0 ? style.ItemSpacing.x : 0) + chevron_w;

  for (size_t i = 0; i < n_left; ++i) {
    if (i > 0) ImGui::SameLine();
    items[i].draw();
  }
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - right_width));
  for (size_t i = 0; i < n_right; ++i) {
    if (i > 0) ImGui::SameLine();
    items[left_count + i].draw();
  }
  if (overflow) {
    if (n_right > 0) ImGui::SameLine();
    if (ImGui::Button((std::string(icon::RAQUO) + "###toolbar_ext").c_str())) ImGui::OpenPopup("toolbar_ext_menu");
    ImGui::SetItemTooltip("More");
    // the popup opens inward: its right edge is aligned with the button so it stays inside the window
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y), ImGuiCond_Always, ImVec2(1, 0));
    if (ImGui::BeginPopup("toolbar_ext_menu")) {
      for (size_t i = n_left; i < left_count; ++i) items[i].draw();
      for (size_t i = left_count + n_right; i < items.size(); ++i) items[i].draw();
      ImGui::EndPopup();
    }
  }
  ImGui::PopStyleVar(2);
}

void ChartsWidget::settingChanged() {
  if (range_slider.maximum() != settings.max_cached_minutes * 60) {
    range_slider.setRange(1, settings.max_cached_minutes * 60);
  }
  for (auto c : charts) {
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
  pos = std::clamp(pos, 0, (int)charts.size());
  charts.insert(charts.begin() + pos, chart);
  currentCharts().insert(currentCharts().begin() + pos, chart);
  updateLayout();
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
      it = src_chart->sigs.erase(it);
    }
    src_chart->updateAxisY();
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
    updateLayout();
  }
}

void ChartsWidget::updateLayout() {
  // the container has not been drawn yet (docked/floated this frame): keep the last known layout
  const float container_width = charts_container.geometry.GetWidth();
  if (container_width <= 0) return;

  int n = MAX_COLUMN_COUNT;
  for (; n > 1; --n) {
    if ((n * CHART_MIN_WIDTH + (n - 1) * CHART_SPACING) < container_width) break;
  }

  columns_action_visible = n > 1;
  current_column_count = std::min(column_count, n);
}

void ChartsWidget::startChartDrag(ChartView *chart, const ImVec2 &global_pos) {
  stopAutoScroll();
  drag = {.source = chart, .press_pos = global_pos};
  showValueTip(-1);  // no value tip while a drag is in progress
  // the drag preview re-renders the tile at CHART_MIN_WIDTH
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
  int tab = tabbar.tabAt(global_pos);
  if (tab >= 0 && tab != tabbar.currentIndex()) {
    tabbar.setCurrentIndex(tab);
  }

  const ImVec2 container_pos = global_pos;
  ChartView *target = nullptr;
  for (auto c : currentCharts()) {
    if (c != drag.source && c->layout_.rect.Contains(container_pos)) {
      target = c;
      break;
    }
  }
  if (std::exchange(drop_target, target) != target) {
    for (auto c : charts) c->setDropHighlight(c == target);
  }
  bool in_viewport = charts_scroll_viewport.Contains(global_pos);
  bool on_background = !target && in_viewport && !charts_container.childAt(container_pos);
  charts_container.drawDropIndicator(on_background ? container_pos : ImVec2());

  if (in_viewport) {
    startAutoScroll(global_pos);
  }
}

void ChartsWidget::cancelChartDrag() {
  drag = {};
  stopAutoScroll();
  drag_preview_visible = false;
  charts_container.drawDropIndicator({});
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
  } else if (in_viewport && !charts_container.childAt(container_pos)) {
    // reorder within the current tab
    auto w = charts_container.getDropAfter(container_pos);
    if (w != source) {
      for (auto &[_, list] : tab_charts) {
        list.erase(std::remove(list.begin(), list.end(), source), list.end());
      }
      auto &cur = currentCharts();
      int to = w ? std::find(cur.begin(), cur.end(), w) - cur.begin() + 1 : 0;
      cur.insert(cur.begin() + to, source);
      updateLayout();
      updateTabBar();
    }
  }
}

void ChartsWidget::drawDragPreview() {
  if (!drag_preview_visible || !drag.source) return;
  // the drag preview is the whole tile (header + axes + plot) at 50% alpha, re-rendered into a window that
  // takes no input, so the live chart keeps handling the mouse.
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
  deleted_charts_.emplace_back(chart);  // freed at the start of the next draw()
  for (auto &[_, list] : tab_charts) {
    list.erase(std::remove(list.begin(), list.end(), chart), list.end());
  }
  updateLayout();
  seriesChanged();
}

void ChartsWidget::removeAll() {
  while (tabbar.count() > 1) {
    tabbar.removeTab(1);
  }
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
  if (!any_plot_hovered_ &&
      (delta.x != 0 || delta.y != 0 || !ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows))) {
    showValueTip(-1);  // QEvent::Leave: the mouse moved off the plot or out of the charts window
  }
}

void ChartsWidget::event() {
  bool back_button = false;
  updateLayout();
  // the mouse back button undoes a zoom; there is no swipe-back gesture
  if (ImGui::IsMouseClicked(3) && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
    back_button = true;
  }
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
  if (auto_scroll_timer_active && ImGui::GetTime() >= auto_scroll_timer_next) {
    auto_scroll_timer_next = ImGui::GetTime() + 0.05;
    doAutoScroll();
  }
  event();
  // the drop target and indicator must be resolved before the charts are painted, otherwise the highlight
  // lags a frame behind the target used on release and the drop lands on the wrong chart
  eventFilter();

  drawToolBar();
  tabbar.draw();

  any_plot_hovered_ = false;
  if (ImGui::BeginChild("charts_scroll", ImVec2(0, 0), ImGuiChildFlags_None, 0)) {
    charts_scroll = ImGui::GetCurrentWindow();
    charts_scroll_viewport = charts_scroll->InnerRect;
    charts_container.draw();
  }
  ImGui::EndChild();

  drawDragPreview();

  if (signal_selector_ && !signal_selector_->draw()) {
    auto dlg = std::move(signal_selector_);
    auto accepted = std::move(signal_selector_accepted_);
    signal_selector_owner_ = nullptr;
    if (dlg->accepted() && accepted) accepted(*dlg);
  }
  ImGui::PopID();
}

ChartsContainer::ChartsContainer(ChartsWidget *parent) : charts_widget(parent) {}

void ChartsContainer::draw() {
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  const ImVec2 start = ImGui::GetCursorScreenPos();
  geometry = ImRect(start, start + ImVec2(window->InnerRect.GetWidth(), 0));
  charts_widget->updateLayout();

  const int n = std::max(charts_widget->current_column_count, 1);
  const float spacing = CHART_SPACING;
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
    if (current_charts[i]->layout_.plot_hovered) charts_widget->any_plot_hovered_ = true;  // the window must be hovered too
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
      r.Min.y = insert_after->layout_.rect.Max.y;
      r.Max.y = r.Min.y + h;
    }

    ImGui::GetWindowDrawList()->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_Header));
  }
}

ChartView *ChartsContainer::getDropAfter(const ImVec2 &pos) const {
  auto it = std::find_if(charts_widget->currentCharts().crbegin(), charts_widget->currentCharts().crend(), [&pos](auto c) {
    auto area = c->layout_.rect;
    return pos.x >= area.Min.x && pos.x <= area.Max.x && pos.y >= area.Max.y;
  });
  return it == charts_widget->currentCharts().crend() ? nullptr : *it;
}

ChartView *ChartsContainer::childAt(const ImVec2 &pos) const {
  for (auto c : charts_widget->currentCharts()) {
    if (c->layout_.rect.Contains(pos)) return c;
  }
  return nullptr;
}
