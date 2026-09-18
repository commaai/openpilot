#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/chartswidget.h"

#include "tools/cabana/ui/threadpool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/layout_manager.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

const int MAX_COLUMN_COUNT = 4;
const int CHART_SPACING = 4;
const int START_DRAG_DISTANCE = 10;
const float LAYOUT_HORIZONTAL_SPACING = 6.0f;
const float MIN_RANGE_SLIDER_WIDTH = 40.0f;

bool LogSlider::draw(const char *label, float width) {
  return fusionSliderInt(label, &pos_, min_, max_, width);
}

ChartsWidget::ChartsWidget() {
  range_slider_.setRange(1, settings.max_cached_minutes * 60);

  tabbar_.setAutoHide(true);
  tabbar_.setUsesScrollButtons(true);
  tabbar_.setTabsClosable(true);

  column_count_ = std::clamp(settings.chart_column_count, 1, MAX_COLUMN_COUNT);
  max_chart_range_ = std::clamp(settings.chart_range, 1, settings.max_cached_minutes * 60);
  display_range_ = std::make_pair(can->minSeconds(), can->minSeconds() + max_chart_range_);
  range_slider_.setValue(max_chart_range_);

  connections_.push_back(dbc()->fileChanged.connect([this]() {
    std::vector<ChartView *> charts;
    for (const auto &chart : charts_) charts.push_back(chart.get());
    for (auto *chart : charts) chart->removeIf([](const auto &s) { return s.sig != nullptr; });
  }));
  connections_.push_back(can->cerealEventsMerged.connect([this]() {
    std::vector<std::future<void>> futures;
    for (const auto &chart : charts_) {
      futures.push_back(ThreadPool::instance().run([ptr = chart.get()]() { ptr->updateCerealSeries(); }));
    }
    for (auto &future : futures) future.get();
    updateState();
  }));
  connections_.push_back(can->eventsMerged.connect([this](const MessageEventsMap &events) { eventsMerged(events); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *, bool) { updateState(); }));
  connections_.push_back(can->seeking.connect([this](double) { updateState(); }));
  connections_.push_back(can->timeRangeChanged.connect([this](const auto &) { updateState(); }));
  connections_.push_back(settings.changed.connect([this]() { settingChanged(); }));
  connections_.push_back(seriesChanged.connect([this]() { updateTabBar(); }));
  connections_.push_back(tabbar_.tabCloseRequested.connect([this](int index) { removeTab(index); }));
  connections_.push_back(tabbar_.tabContextMenu.connect([this](int index) {
    if (ImGui::BeginPopupContextItem()) {
      if (ImGui::MenuItem("Close Other Tabs")) {
        tabbar_.moveTab(index, 0);
        tabbar_.setCurrentIndex(0);
        while (tabbar_.count() > 1) removeTab(1);
      }
      ImGui::EndPopup();
    }
  }));
  connections_.push_back(tabbar_.currentChanged.connect([this](int index) {
    if (index != -1) updateLayout();
  }));

  setIsDocked(true);
  newTab();
}

ChartsWidget::~ChartsWidget() = default;

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
  int idx = tabbar_.addTab("");
  tab_names_[tab_unique_id] = "Tab " + std::to_string(idx + 1);
  tabbar_.setTabData(idx, tab_unique_id++);
  tabbar_.setCurrentIndex(idx);
  updateTabBar();
}

void ChartsWidget::removeTab(int index) {
  int id = tabbar_.tabData(index);
  for (auto &c : std::vector<ChartView *>(tab_charts_[id])) {
    removeChart(c);
  }
  tab_charts_.erase(id);
  tab_names_.erase(id);
  tabbar_.removeTab(index);
  updateTabBar();
}

void ChartsWidget::updateTabBar() {
  for (int i = 0; i < tabbar_.count(); ++i) {
    const auto &charts_in_tab = tab_charts_[tabbar_.tabData(i)];
    tabbar_.setTabText(i, tab_names_[tabbar_.tabData(i)] + " (" + std::to_string((int)charts_in_tab.size()) + ")");
  }
}

void ChartsWidget::eventsMerged(const MessageEventsMap &new_events) {
  std::vector<std::future<void>> futures;
  for (auto &c : charts_) {
    futures.push_back(ThreadPool::instance().run([c = c.get(), &new_events]() { c->updateSeries(nullptr, &new_events); }));
  }
  for (auto &f : futures) f.get();
}

void ChartsWidget::zoomReset() {
  can->setTimeRange(std::nullopt);
  zoom_undo_stack_.clear();
}

ImRect ChartsWidget::chartVisibleRect(ChartView *chart) {
  ImRect r = chart->rect();
  r.ClipWith(charts_scroll_viewport_);
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
  if (charts_.empty()) return;

  const auto &time_range = can->timeRange();
  const double cur_sec = can->currentSec();
  if (!time_range.has_value()) {
    double pos = (cur_sec - display_range_.first) / std::max<float>(1.0, max_chart_range_);
    if (pos < 0 || pos > 0.8) {
      display_range_.first = std::max(can->minSeconds(), cur_sec - max_chart_range_ * 0.1);
    }
    double max_sec = std::min(display_range_.first + max_chart_range_, can->maxSeconds());
    display_range_.first = std::max(can->minSeconds(), max_sec - max_chart_range_);
    display_range_.second = display_range_.first + max_chart_range_;
  } else if (!can->isPaused()) {
    const double window_len = time_range->second - time_range->first;
    if (window_len > 0.05) {
      const double pos = (cur_sec - time_range->first) / window_len;
      if (pos > 0.85 || pos < 0.0) {
        double new_min = std::max(can->minSeconds(), cur_sec - window_len * 0.7);
        double new_max = std::min(can->maxSeconds(), new_min + window_len);
        if (new_max - new_min >= window_len * 0.95) {
          can->setTimeRange(std::make_pair(new_min, new_max));
        }
      }
    }
  }

  const auto &range = time_range ? *time_range : display_range_;
  for (auto &c : charts_) {
    c->updatePlot(cur_sec, range.first, range.second);
  }
}

void ChartsWidget::setMaxChartRange(int value) {
  max_chart_range_ = settings.chart_range = value;
  updateState();
}

void ChartsWidget::setIsDocked(bool docked) {
  is_docked_ = docked;
  if (!docked) float_window_init_ = true;
}

void ChartsWidget::drawToolBar() {
  beginToolbar();
  float slider_width = 150.0f;
  const bool is_zoomed = can->timeRange().has_value();

  // the labels are captured by reference, they outlive the draw calls below
  std::vector<ToolbarItem> items;
  items.push_back({toolbarButtonWidth(icon::PLUS_SQUARE), [this]() {
    if (toolButton("new_plot_btn", icon::PLUS_SQUARE, "New Chart")) newChart();
  }});
  items.push_back({toolbarButtonWidth(icon::WINDOW_STACK), [this]() {
    if (toolButton("new_tab_btn", icon::WINDOW_STACK, "New Tab")) newTab();
  }});
  const std::string title_label = "Charts: " + std::to_string(charts_.size());
  items.push_back({ImGui::CalcTextSize(title_label.c_str()).x + LAYOUT_HORIZONTAL_SPACING, [&title_label]() {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(title_label.c_str());
    ImGui::SameLine(0.0f, LAYOUT_HORIZONTAL_SPACING);
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
  }});

  const int type_count = (int)std::size(SERIES_TYPE_NAMES);
  const std::string chart_type_text = std::string("Type:  ") + SERIES_TYPE_NAMES[std::clamp(settings.chart_series_type, 0, type_count - 1)];
  items.push_back({menuButtonWidth(chart_type_text), [this, &chart_type_text]() {
    menuButton("chart_type", chart_type_text, "chart_type_menu");
    if (ImGui::BeginPopup("chart_type_menu")) {
      for (int i = 0; i < type_count; ++i) {
        if (ImGui::MenuItem(SERIES_TYPE_NAMES[i])) {
          settings.chart_series_type = i;
          settingChanged();
        }
      }
      ImGui::EndPopup();
    }
  }});

  const std::string columns_action_text = "Columns:  " + std::to_string(column_count_);
  if (columns_action_visible_) {
    items.push_back({menuButtonWidth(columns_action_text), [this, &columns_action_text]() {
      menuButton("columns", columns_action_text, "columns_menu");
      if (ImGui::BeginPopup("columns_menu")) {
        for (int i = 0; i < MAX_COLUMN_COUNT; ++i) {
          if (ImGui::MenuItem(std::to_string(i + 1).c_str())) setColumnCount(i + 1);
        }
        ImGui::EndPopup();
      }
    }});
  }

  const std::string layout_btn_text = std::string("Layout:  ") + (current_layout_name_.empty() ? "None" : current_layout_name_);
  items.push_back({menuButtonWidth(layout_btn_text), [this, &layout_btn_text]() {
    menuButton("layout_preset", layout_btn_text, "layout_preset_menu");
    if (ImGui::BeginPopup("layout_preset_menu")) {
      for (const auto &preset : cabana::LayoutManager::availablePresets()) {
        if (ImGui::MenuItem(preset.c_str(), nullptr, current_layout_name_ == preset)) {
          loadLayoutFile(cabana::LayoutManager::presetPath(preset));
        }
      }
      ImGui::EndPopup();
    }
  }});

  items.push_back({toolbarButtonWidth(icon::LIST), [this]() {
    if (toolButton("cereal_browser_btn", icon::LIST, "Browse Cereal Signals")) {
      cereal_browser_visible = !cereal_browser_visible;
    }
  }});

  // the spacer right aligns the rest
  const size_t spacer_index = items.size();
  size_t slider_index = (size_t)-1;
  const std::string range_lb = is_zoomed ? std::string() : utils::formatSeconds(max_chart_range_);
  std::string reset_zoom_text;
  if (!is_zoomed) {
    items.push_back({ImGui::CalcTextSize(range_lb.c_str()).x, [&range_lb]() {
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(range_lb.c_str());
    }});
    slider_index = items.size();
    items.push_back({slider_width, [this, &slider_width]() {
      if (range_slider_.draw("##range_slider", slider_width)) setMaxChartRange(range_slider_.value());
      ImGui::SetItemTooltip("Set the chart range");
    }});
  } else {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f-%.2f", can->timeRange()->first, can->timeRange()->second);
    reset_zoom_text = buf;
    items.push_back({toolbarButtonWidth(icon::ARROW_COUNTERCLOCKWISE), [this]() {
      ImGui::BeginDisabled(!zoom_undo_stack_.canUndo());
      if (toolButton("undo_zoom", icon::ARROW_COUNTERCLOCKWISE, "Undo Zoom")) zoom_undo_stack_.undo();
      ImGui::EndDisabled();
    }});
    items.push_back({toolbarButtonWidth(icon::ARROW_CLOCKWISE), [this]() {
      ImGui::BeginDisabled(!zoom_undo_stack_.canRedo());
      if (toolButton("redo_zoom", icon::ARROW_CLOCKWISE, "Redo Zoom")) zoom_undo_stack_.redo();
      ImGui::EndDisabled();
    }});
    items.push_back({toolbarButtonWidth(std::string(icon::ZOOM_OUT) + " " + reset_zoom_text), [this, &reset_zoom_text]() {
      if (toolButton("reset_zoom_btn", icon::ZOOM_OUT, "Reset Zoom", reset_zoom_text.c_str())) zoomReset();
    }});
  }
  items.push_back({toolbarButtonWidth(icon::X_SQUARE), [this]() {
    ImGui::BeginDisabled(charts_.empty());
    if (toolButton("remove_all_btn", icon::X_SQUARE, "Remove all charts")) removeAll();
    ImGui::EndDisabled();
  }});
  const char *dock_btn_icon = is_docked_ ? icon::ARROW_UP_RIGHT_SQUARE : icon::ARROW_DOWN_LEFT_SQUARE;
  items.push_back({toolbarButtonWidth(dock_btn_icon), [this, dock_btn_icon]() {
    if (toolButton("dock_btn", dock_btn_icon, is_docked_ ? "Float the charts window" : "Dock the charts window")) toggleChartsDocking();
  }});

  // the slider shrinks first, the buttons stay pinned to the right edge
  if (slider_index != (size_t)-1) {
    const float shrink = std::min(slider_width - MIN_RANGE_SLIDER_WIDTH, toolbarWidth(items, spacer_index) - ImGui::GetContentRegionAvail().x);
    if (shrink > 0.0f) {
      slider_width -= shrink;
      items[slider_index].width = slider_width;
    }
  }
  drawToolbar(items, spacer_index);
  endToolbar();
}

void ChartsWidget::settingChanged() {
  if (range_slider_.maximum() != settings.max_cached_minutes * 60) {
    range_slider_.setRange(1, settings.max_cached_minutes * 60);
  }
  for (auto &c : charts_) {
    c->setSeriesType((SeriesType)settings.chart_series_type);
  }
}

ChartView *ChartsWidget::findChart(const MessageId &id, const cabana::Signal *sig) {
  for (auto &c : charts_)
    if (c->hasSignal(id, sig)) return c.get();
  return nullptr;
}

ChartView *ChartsWidget::createChart(int pos) {
  auto chart = std::make_unique<ChartView>(can->timeRange().value_or(display_range_), this);
  ChartView *ptr = chart.get();
  pos = std::clamp(pos, 0, (int)charts_.size());
  charts_.insert(charts_.begin() + pos, std::move(chart));
  currentCharts().insert(currentCharts().begin() + pos, ptr);
  updateLayout();
  return ptr;
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

void ChartsWidget::showCerealChart(const std::string &path, bool merge) {
  for (const auto &chart : charts_) if (chart->hasCerealSignal(path)) return;
  auto *chart = merge && !currentCharts().empty() ? currentCharts().front() : createChart();
  chart->addCerealSignal(path);
  updateState();
}

void ChartsWidget::drawCerealBrowser() {
  if (!cereal_browser_visible) return;
  ImGui::SetNextWindowSize(ImVec2(450, 550), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("Cereal Signals", &cereal_browser_visible)) {
    cereal_filter_.Draw("Search");
    ImGui::TextWrapped("Double-click to plot. Shift + double-click overlays the first chart. Drag a signal onto a chart to overlay it.");
    auto snapshot = can->cereal_series.snapshot();
    if (snapshot.revision != cereal_revision_) {
      std::set<std::string> paths;
      for (const auto &[segment, series] : snapshot.segments) {
        for (const auto &[path, values] : *series) paths.insert(path);
      }
      cereal_paths_.assign(paths.begin(), paths.end());
      cereal_revision_ = snapshot.revision;
    }
    if (cereal_paths_.empty()) ImGui::TextUnformatted("No cereal samples loaded.");
    std::vector<const std::string *> filtered;
    for (const auto &path : cereal_paths_) if (cereal_filter_.PassFilter(path.c_str())) filtered.push_back(&path);
    ImGui::BeginChild("Series");
    ImGuiListClipper clipper;
    clipper.Begin(filtered.size());
    while (clipper.Step()) {
      for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
        const auto &path = *filtered[i];
        if (ImGui::Selectable(path.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick) && ImGui::IsMouseDoubleClicked(0)) {
          showCerealChart(path, ImGui::GetIO().KeyShift);
        }
        if (ImGui::BeginDragDropSource()) {
          ImGui::SetDragDropPayload("CABANA_CEREAL", path.c_str(), path.size() + 1);
          ImGui::TextUnformatted(path.c_str());
          ImGui::EndDragDropSource();
        }
      }
    }
    ImGui::EndChild();
  }
  ImGui::End();
}

void ChartsWidget::splitChart(ChartView *src_chart) {
  if (src_chart->signals().size() > 1) {
    auto it = std::find_if(charts_.begin(), charts_.end(), [src_chart](auto &c) { return c.get() == src_chart; });
    const int pos = it - charts_.begin() + 1;
    for (auto &s : src_chart->takeExtraSignals()) {
      createChart(pos)->adoptSignal(std::move(s));
    }
    updateState();
  }
}

std::vector<std::string> ChartsWidget::serializeChartIds() const {
  std::vector<std::string> chart_ids;
  for (auto &c : charts_) {
    std::string ids;
    for (const auto &s : c->signals()) {
      if (!ids.empty()) ids += ',';
      ids += s.sig ? s.msg_id.toString() + "|" + s.sig->name : "cereal|" + s.cereal_path;
    }
    chart_ids.push_back(ids);
  }
  std::reverse(chart_ids.begin(), chart_ids.end());
  return chart_ids;
}

void ChartsWidget::restoreChartsFromIds(const std::vector<std::string> &chart_ids, bool restore_can) {
  for (const auto &chart_id : chart_ids) {
    int index = 0;
    for (const auto &part : utils::split(chart_id, ',')) {
      if (part.rfind("cereal|", 0) == 0) {
        showCerealChart(part.substr(7), index++ > 0);
        continue;
      }
      if (!restore_can) continue;
      const size_t sep = part.find('|');
      if (sep == std::string::npos) continue;
      MessageId msg_id = MessageId::fromString(part.substr(0, sep));
      if (auto *msg = dbc()->msg(msg_id))
        if (auto *sig = msg->sig(part.substr(sep + 1)))
          showChart(msg_id, sig, true, index++ > 0);
    }
  }
}

cabana::Layout ChartsWidget::captureLayout() const {
  cabana::Layout layout;
  layout.name = current_layout_name_;
  layout.current_tab_index = tabbar_.currentIndex();
  for (int i = 0; i < tabbar_.count(); ++i) {
    cabana::LayoutTab tab;
    const int id = tabbar_.tabData(i);
    tab.name = tab_names_.at(id);
    auto found = tab_charts_.find(id);
    if (found != tab_charts_.end()) for (const auto *chart : found->second) {
      cabana::LayoutPane pane;
      pane.title = chart->title();
      pane.y_limits = chart->yLimits();
      pane.series_type = static_cast<int>(chart->seriesType());
      for (const auto &signal : chart->signals()) {
        cabana::LayoutCurve curve;
        curve.name = signal.name();
        if (signal.sig) curve.can_id = signal.msg_id.toString();
        char color[8];
        snprintf(color, sizeof(color), "#%02x%02x%02x", signal.color.r, signal.color.g, signal.color.b);
        curve.color_hex = color;
        curve.visible = signal.visible;
        curve.custom_python = signal.custom_python;
        curve.derivative = signal.derivative;
        curve.derivative_dt = signal.derivative_dt;
        curve.scale = signal.scale;
        curve.offset = signal.offset;
        pane.curves.push_back(std::move(curve));
      }
      tab.panes.push_back(std::move(pane));
    }
    layout.tabs.push_back(std::move(tab));
  }
  return layout;
}

void ChartsWidget::loadLayoutFile(const std::filesystem::path &path) {
  try {
    const cabana::Layout layout = cabana::LayoutManager::loadLayout(path);
    // Validate CAN references before discarding the current workspace.
    for (const auto &tab : layout.tabs) for (const auto &pane : tab.panes) for (const auto &curve : pane.curves) {
      if (curve.can_id.empty()) continue;
      auto *message = dbc()->msg(MessageId::fromString(curve.can_id));
      if (!message || !message->sig(curve.name)) throw std::runtime_error("Missing CAN signal " + curve.can_id + "|" + curve.name);
    }
    removeAll();
    tab_names_.clear();
    while (tabbar_.count() > 0) {
      tabbar_.removeTab(0);
    }
    tab_charts_.clear();

    for (size_t t = 0; t < layout.tabs.size(); ++t) {
      const auto &tab = layout.tabs[t];
      newTab();
      tab_names_[tabbar_.tabData(tabbar_.currentIndex())] = tab.name;

      for (const auto &pane : tab.panes) {
        if (!pane.kind.empty()) continue;
        auto c = createChart(currentCharts().size());
        c->setSeriesType(static_cast<SeriesType>(pane.series_type));
        if (!pane.title.empty() && pane.title != "...") {
          c->setTitle(pane.title);
        }
        if (pane.y_limits.has_value()) {
          c->setYLimits(pane.y_limits->first, pane.y_limits->second);
        }
        for (const auto &curve : pane.curves) {
          c->addLayoutCurve(curve);
        }
      }
    }

    if (tabbar_.count() > 0) {
      tabbar_.setCurrentIndex(std::clamp(layout.current_tab_index, 0, tabbar_.count() - 1));
    }
    current_layout_name_ = layout.name;
    updateLayout();
    updateTabBar();
  } catch (const std::exception &err) {
    fprintf(stderr, "Failed to load layout %s: %s\n", path.c_str(), err.what());
  }
}

void ChartsWidget::setColumnCount(int n) {
  n = std::clamp(n, 1, MAX_COLUMN_COUNT);
  if (column_count_ != n) {
    column_count_ = settings.chart_column_count = n;
    updateLayout();
  }
}

void ChartsWidget::updateLayout() {
  // the container has not been drawn yet (docked/floated this frame): keep the last known layout
  const float container_width = charts_container_.geometry().GetWidth();
  if (container_width <= 0) return;

  int n = MAX_COLUMN_COUNT;
  for (; n > 1; --n) {
    if ((n * CHART_MIN_WIDTH + (n - 1) * CHART_SPACING) < container_width) break;
  }

  columns_action_visible_ = n > 1;
  current_column_count_ = std::min(column_count_, n);
}

void ChartsWidget::startChartDrag(ChartView *chart, const ImVec2 &global_pos) {
  stopAutoScroll();
  drag_ = {.source = chart, .press_pos = global_pos};
  showValueTip(-1);  // no value tip while a drag is in progress
  // the drag preview re-renders the tile at CHART_MIN_WIDTH
  drag_preview_size_ = ImVec2(CHART_MIN_WIDTH, (float)settings.chart_height);
}

void ChartsWidget::dragChartMove(const ImVec2 &global_pos) {
  if (!drag_.active) {
    ImVec2 d = global_pos - drag_.press_pos;
    if (std::abs(d.x) + std::abs(d.y) < START_DRAG_DISTANCE) return;
    drag_.active = true;
    drag_preview_visible_ = true;
  }
  drag_preview_pos_ = global_pos + ImVec2(5, 5);

  // hovering a tab switches to it so the chart can be dropped into another tab
  int tab = tabbar_.tabAt(global_pos);
  if (tab >= 0 && tab != tabbar_.currentIndex()) {
    tabbar_.setCurrentIndex(tab);
  }

  ChartView *target = nullptr;
  for (auto c : currentCharts()) {
    if (c != drag_.source && c->rect().Contains(global_pos)) {
      target = c;
      break;
    }
  }
  if (std::exchange(drop_target_, target) != target) {
    for (auto &c : charts_) c->setDropHighlight(c.get() == target);
  }
  bool in_viewport = charts_scroll_viewport_.Contains(global_pos);
  bool on_background = !target && in_viewport && !charts_container_.childAt(global_pos);
  charts_container_.setDropIndicator(on_background ? global_pos : ImVec2());

  if (in_viewport) {
    startAutoScroll(global_pos);
  }
}

void ChartsWidget::cancelChartDrag() {
  drag_ = {};
  stopAutoScroll();
  drag_preview_visible_ = false;
  charts_container_.setDropIndicator({});
  if (auto target = std::exchange(drop_target_, nullptr)) target->setDropHighlight(false);
}

void ChartsWidget::dragChartRelease(const ImVec2 &global_pos) {
  ChartView *source = drag_.source;
  bool active = drag_.active;
  ChartView *target = drop_target_;
  cancelChartDrag();
  if (!active) return;

  bool in_viewport = charts_scroll_viewport_.Contains(global_pos);
  if (target) {
    // merge source into target
    target->takeSignalsFrom(source);
  } else if (in_viewport && !charts_container_.childAt(global_pos)) {
    // reorder within the current tab
    auto w = charts_container_.getDropAfter(global_pos);
    if (w != source) {
      for (auto &[_, list] : tab_charts_) {
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
  if (!drag_preview_visible_ || !drag_.source) return;
  // the drag preview is the whole tile (header + axes + plot) at 50% alpha, re-rendered into a window that
  // takes no input, so the live chart keeps handling the mouse.
  ImGui::SetNextWindowPos(drag_preview_pos_);
  ImGui::SetNextWindowSize(drag_preview_size_);
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                 ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking;
  if (ImGui::Begin("##chart_drag_ghost", nullptr, flags)) {
    drag_.source->drawGhost(drag_preview_size_.x);
  }
  ImGui::End();
  ImGui::PopStyleVar(3);
}

void ChartsWidget::startAutoScroll(const ImVec2 &global_pos) {
  auto_scroll_pos_ = global_pos;
  if (!auto_scroll_timer_active_) auto_scroll_timer_next_ = ImGui::GetTime() + 0.05;
  auto_scroll_timer_active_ = true;
}

void ChartsWidget::stopAutoScroll() {
  auto_scroll_timer_active_ = false;
  auto_scroll_count_ = 0;
}

void ChartsWidget::doAutoScroll() {
  if (!charts_scroll_) return;
  const int page_step = charts_scroll_viewport_.GetHeight();
  if (auto_scroll_count_ < page_step) {
    ++auto_scroll_count_;
  }

  int value = charts_scroll_->Scroll.y;
  ImVec2 pos = auto_scroll_pos_;
  ImRect area = charts_scroll_viewport_;

  int new_value = value;
  if (pos.y - area.Min.y < settings.chart_height / 2) {
    new_value = value - auto_scroll_count_;
  } else if (area.Max.y - pos.y < settings.chart_height / 2) {
    new_value = value + auto_scroll_count_;
  }
  new_value = std::clamp<int>(new_value, 0, charts_scroll_->ScrollMax.y);
  if (new_value != value) ImGui::SetScrollY(charts_scroll_, new_value);
  if (value == new_value) {
    stopAutoScroll();
  } else if (chartDragActive()) {
    // refresh the drop indicator/target at the new scroll position
    dragChartMove(auto_scroll_pos_);
  }
}

void ChartsWidget::newChart() {
  execSignalSelector(std::make_unique<SignalSelector>("New Chart"), nullptr, [this](SignalSelector &dlg) {
    const auto &items = dlg.selectedItems();
    if (!items.empty()) {
      auto c = createChart();
      for (const auto &it : items) {
        c->addSignal(it.msg_id, it.sig);
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
  if (drag_.source == chart) cancelChartDrag();
  if (drop_target_ == chart) drop_target_ = nullptr;
  if (signal_selector_owner_ == chart) {
    signal_selector_owner_ = nullptr;
    signal_selector_accepted_ = nullptr;
  }
  auto it = std::find_if(charts_.begin(), charts_.end(), [chart](auto &c) { return c.get() == chart; });
  if (it != charts_.end()) {
    deleted_charts_.push_back(std::move(*it));  // may be called from the chart's draw; freed next frame
    charts_.erase(it);
  }
  for (auto &[_, list] : tab_charts_) {
    list.erase(std::remove(list.begin(), list.end(), chart), list.end());
  }
  updateLayout();
  seriesChanged();
}

void ChartsWidget::removeAll() {
  while (tabbar_.count() > 1) {
    tabbar_.removeTab(1);
  }
  std::vector<ChartView *> all;
  for (auto &c : charts_) all.push_back(c.get());
  for (auto c : all) removeChart(c);
  tab_charts_.clear();
  zoomReset();
}

void ChartsWidget::handleEvents() {
  // the mouse back button undoes a zoom; there is no swipe-back gesture
  if (ImGui::IsMouseClicked(3) && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
    zoom_undo_stack_.undo();
  }
  if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow)) {
    if (chartDragActive()) cancelChartDrag();
    showValueTip(-1);
  }

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
    showValueTip(-1);  // the mouse moved off the plot or out of the charts window
  }
}

void ChartsWidget::draw() {
  deleted_charts_.clear();
  // the floating window is a top level window sized to its contents: keep it inside the main viewport so its
  // toolbar stays reachable, then let the user resize it
  if (float_window_init_ && !is_docked_) {
    float_window_init_ = false;
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    const ImVec2 size(viewport->WorkSize.x * 0.6f, viewport->WorkSize.y * 0.6f);
    ImGui::SetWindowSize(size);
    ImGui::SetWindowPos(viewport->WorkPos + (viewport->WorkSize - size) * 0.5f);
  }
  ImGui::PushID(this);
  if (auto_scroll_timer_active_ && ImGui::GetTime() >= auto_scroll_timer_next_) {
    auto_scroll_timer_next_ = ImGui::GetTime() + 0.05;
    doAutoScroll();
  }
  // the drop target and indicator must be resolved before the charts are painted, otherwise the highlight
  // lags a frame behind the target used on release and the drop lands on the wrong chart
  handleEvents();

  drawToolBar();
  tabbar_.draw();

  any_plot_hovered_ = false;
  if (ImGui::BeginChild("charts_scroll", ImVec2(0, 0), ImGuiChildFlags_None, 0)) {
    charts_scroll_ = ImGui::GetCurrentWindow();
    charts_scroll_viewport_ = charts_scroll_->InnerRect;
    charts_container_.draw();
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

void ChartsContainer::draw() {
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  const ImVec2 start = ImGui::GetCursorScreenPos();
  geometry_ = ImRect(start, start + ImVec2(window->InnerRect.GetWidth(), 0));
  charts_widget_->updateLayout();

  const int n = std::max(charts_widget_->current_column_count_, 1);
  const float spacing = CHART_SPACING;
  const float width = (geometry_.GetWidth() - (n - 1) * spacing) / n;
  const ImVec2 origin = ImGui::GetCursorScreenPos() + ImVec2(0, CHART_SPACING);
  auto current_charts = charts_widget_->currentCharts();  // copy: drawing may remove charts
  float bottom = origin.y;
  const bool aligned = ImPlot::BeginAlignedPlots("charts_align", true);
  for (int i = 0; i < current_charts.size(); ++i) {
    ImVec2 pos = origin + ImVec2((i % n) * (width + spacing), (i / n) * (settings.chart_height + spacing));
    ImGui::SetCursorScreenPos(pos);
    current_charts[i]->draw(width);
    bottom = std::max(bottom, pos.y + settings.chart_height);
    if (current_charts[i]->plotHovered()) charts_widget_->any_plot_hovered_ = true;  // the window must be hovered too
  }
  if (aligned) ImPlot::EndAlignedPlots();
  ImGui::SetCursorScreenPos(ImVec2(origin.x, bottom));
  ImGui::Dummy(ImVec2(geometry_.GetWidth(), CHART_SPACING));
  geometry_.Max.y = bottom + CHART_SPACING;
  drawDropIndicator();
}

void ChartsContainer::drawDropIndicator() {
  if (!(drop_indicator_pos_.x == 0 && drop_indicator_pos_.y == 0) && !childAt(drop_indicator_pos_)) {
    ImRect r = geometry_;
    r.Max.y = r.Min.y + CHART_SPACING;
    if (auto insert_after = getDropAfter(drop_indicator_pos_)) {
      float h = r.GetHeight();
      r.Min.y = insert_after->rect().Max.y;
      r.Max.y = r.Min.y + h;
    }

    ImGui::GetWindowDrawList()->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_Header));
  }
}

ChartView *ChartsContainer::getDropAfter(const ImVec2 &pos) const {
  const auto &charts = charts_widget_->currentCharts();
  auto it = std::find_if(charts.crbegin(), charts.crend(), [&pos](auto c) {
    const ImRect &area = c->rect();
    return pos.x >= area.Min.x && pos.x <= area.Max.x && pos.y >= area.Max.y;
  });
  return it == charts.crend() ? nullptr : *it;
}

ChartView *ChartsContainer::childAt(const ImVec2 &pos) const {
  for (auto c : charts_widget_->currentCharts()) {
    if (c->rect().Contains(pos)) return c;
  }
  return nullptr;
}
