#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/chartswidget.h"

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

static float buttonWidth(const std::string &label) {
  return ImGui::CalcTextSize(label.c_str(), nullptr, true).x + ImGui::GetStyle().FramePadding.x * 2;
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
  return "Chart View\n"
         "Click: Click to seek to a corresponding time.\n"
         "Drag: Zoom into the chart.\n"
         "Shift + Drag: Scrub through the chart to view values.\n"
         "Right Mouse: Open the context menu.\n";
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
    futures.push_back(std::async(std::launch::async, &ChartView::updateSeries, c, nullptr, &new_events));
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
  const ImGuiStyle &style = ImGui::GetStyle();

  if (toolButton("new_plot_btn", icon::FILE_PLUS, "New Chart")) newChart();
  ImGui::SameLine();
  if (toolButton("new_tab_btn", icon::WINDOW_STACK, "New Tab")) newTab();
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(title_label.c_str());
  ImGui::SameLine(0, style.ItemSpacing.x * 2);

  // chart type menu
  std::string chart_type_text = std::string("Type: ") + types[std::clamp(settings.chart_series_type, 0, 2)];
  if (ImGui::Button((chart_type_text + "###chart_type").c_str())) ImGui::OpenPopup("chart_type_menu");
  if (ImGui::BeginPopup("chart_type_menu")) {
    for (int i = 0; i < types.size(); ++i) {
      if (ImGui::MenuItem(types[i])) {
        settings.chart_series_type = i;
        settingChanged();
      }
    }
    ImGui::EndPopup();
  }

  // columns menu
  if (columns_action_visible) {
    ImGui::SameLine();
    if (ImGui::Button((columns_action_text + "###columns").c_str())) ImGui::OpenPopup("columns_menu");
    if (ImGui::BeginPopup("columns_menu")) {
      for (int i = 0; i < MAX_COLUMN_COUNT; ++i) {
        if (ImGui::MenuItem(std::to_string(i + 1).c_str())) setColumnCount(i + 1);
      }
      ImGui::EndPopup();
    }
  }

  // spacer: right align the rest
  float slider_width = 150.0f;
  float right_width = 0;
  bool show_range_lb = range_lb_visible, show_range_slider = range_slider_visible;
  if (range_lb_visible) right_width += ImGui::CalcTextSize(range_lb.c_str()).x + style.ItemSpacing.x;
  if (range_slider_visible) right_width += slider_width + style.ItemSpacing.x;
  if (undo_zoom_visible) right_width += buttonWidth(icon::ARROW_COUNTERCLOCKWISE) + style.ItemSpacing.x;
  if (redo_zoom_visible) right_width += buttonWidth(icon::ARROW_CLOCKWISE) + style.ItemSpacing.x;
  if (reset_zoom_visible) right_width += buttonWidth(std::string(icon::ZOOM_OUT) + " " + reset_zoom_text) + style.ItemSpacing.x;
  right_width += buttonWidth(icon::X_SQUARE) + style.ItemSpacing.x;
  right_width += buttonWidth(dock_btn_icon);
  ImGui::SameLine();
  if (range_slider_visible && ImGui::GetContentRegionAvail().x < right_width) {
    // QSlider shrinks first (never below 40px), the buttons stay pinned to the right edge
    const float shrink = std::min(slider_width - 40.0f, right_width - ImGui::GetContentRegionAvail().x);
    slider_width -= std::max(shrink, 0.0f);
    right_width -= std::max(shrink, 0.0f);
  }
  // QToolBar moves what still does not fit into its extension menu; here the range label, then the slider, are hidden
  if (show_range_lb && ImGui::GetContentRegionAvail().x < right_width) {
    show_range_lb = false;
    right_width -= ImGui::CalcTextSize(range_lb.c_str()).x + style.ItemSpacing.x;
  }
  if (show_range_slider && ImGui::GetContentRegionAvail().x < right_width) {
    show_range_slider = false;
    right_width -= slider_width + style.ItemSpacing.x;
  }
  const float x = ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - right_width);
  ImGui::SetCursorPosX(x);

  if (show_range_lb) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(range_lb.c_str());
    ImGui::SameLine();
  }
  if (show_range_slider) {
    if (range_slider.draw("##range_slider", slider_width)) setMaxChartRange(range_slider.value());
    ImGui::SetItemTooltip("Set the chart range");
    ImGui::SameLine();
  }

  // zoom controls
  if (undo_zoom_visible) {
    ImGui::BeginDisabled(!undo_zoom_enabled);
    if (toolButton("undo_zoom", icon::ARROW_COUNTERCLOCKWISE, "Undo Zoom")) zoom_undo_stack.undo();
    ImGui::EndDisabled();
    ImGui::SameLine();
  }
  if (redo_zoom_visible) {
    ImGui::BeginDisabled(!redo_zoom_enabled);
    if (toolButton("redo_zoom", icon::ARROW_CLOCKWISE, "Redo Zoom")) zoom_undo_stack.redo();
    ImGui::EndDisabled();
    ImGui::SameLine();
  }
  if (reset_zoom_visible) {
    if (toolButton("reset_zoom_btn", icon::ZOOM_OUT, "Reset Zoom", reset_zoom_text.c_str())) zoomReset();
    ImGui::SameLine();
  }

  ImGui::BeginDisabled(!remove_all_enabled);
  if (toolButton("remove_all_btn", icon::X_SQUARE, "Remove all charts")) removeAll();
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (toolButton("dock_btn", dock_btn_icon, dock_btn_tooltip.c_str())) toggleChartsDocking();
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
    c->resetChartCache();
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
  chart->connections_.push_back(chart->axisYLabelWidthChanged.connect([this](int) { align_timer = true; }));  // dropped with the chart
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
    src_chart->resetChartCache();
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
  int n = MAX_COLUMN_COUNT;
  for (; n > 1; --n) {
    if ((n * CHART_MIN_WIDTH + (n - 1) * charts_container->horizontalSpacing()) < charts_container->geometry.GetWidth()) break;
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
  // the preview is a list of the chart's signal names instead of a scaled grab of the chart
  drag_preview.clear();
  for (auto &s : chart->sigs) drag_preview.push_back(s.sig->name);
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
  if (!drag_preview_visible) return;
  ImDrawList *dl = ImGui::GetForegroundDrawList();
  const float line_h = ImGui::GetTextLineHeightWithSpacing();
  ImVec2 size(CHART_MIN_WIDTH, line_h * std::max<int>(drag_preview.size(), 1) + 8);
  dl->AddRectFilled(drag_preview_pos, drag_preview_pos + size, ImGui::GetColorU32(ImGuiCol_PopupBg, 0.9f));
  dl->AddRect(drag_preview_pos, drag_preview_pos + size, ImGui::GetColorU32(ImGuiCol_Border));
  float y = drag_preview_pos.y + 4;
  for (const auto &name : drag_preview) {
    dl->AddText(ImVec2(drag_preview_pos.x + 6, y), ImGui::GetColorU32(ImGuiCol_Text), name.c_str());
    y += line_h;
  }
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
  alignCharts();
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

void ChartsWidget::alignCharts() {
  int plot_left = 0;
  for (auto c : charts) {
    plot_left = std::max(plot_left, c->y_label_width);
  }
  plot_left = std::max((plot_left / 10) * 10 + 10, 50);
  // the actual y axis alignment is done by ImPlot::BeginAlignedPlots in ChartsContainer::draw
  for (auto c : charts) {
    c->updatePlotArea(plot_left);
  }
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
  ImGui::PushID(this);
  // timers
  if (align_timer.exchange(false)) alignCharts();
  if (auto_scroll_timer_active && ImGui::GetTime() >= auto_scroll_timer_next) {
    auto_scroll_timer_next = ImGui::GetTime() + 0.05;
    doAutoScroll();
  }
  event();

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

  eventFilter();
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
    if (current_charts[i]->plot_area.Contains(ImGui::GetMousePos())) charts_widget->any_plot_hovered_ = true;
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
