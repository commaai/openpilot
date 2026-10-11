#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/chartswidget.h"

#include "tools/cabana/ui/threadpool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "json11/json11.hpp"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

const int MAX_COLUMN_COUNT = 4;
const int START_DRAG_DISTANCE = 10;
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

  connections_.push_back(dbc()->fileChanged.connect([this]() { removeAll(); }));
  connections_.push_back(can->eventsMerged.connect([this](const MessageEventsMap &events) { eventsMerged(events); }));
  connections_.push_back(can->cerealSeriesUpdated.connect([this]() { cerealSeriesUpdated(); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *, bool) { updateState(); }));
  connections_.push_back(can->seeking.connect([this](double) { updateState(); }));
  connections_.push_back(can->timeRangeChanged.connect([this](const auto &) { updateState(); }));
  connections_.push_back(settings.changed.connect([this]() { settingChanged(); }));
  connections_.push_back(seriesChanged.connect([this]() { updateTabBar(); }));
  connections_.push_back(tabbar_.tabCloseRequested.connect([this](int index) { removeTab(index); }));
  connections_.push_back(tabbar_.tabContextMenu.connect([this](int index) {
    if (dropdown::BeginPopupContextItem()) {
      if (dropdown::Item("Close Other Tabs")) {
        tabbar_.moveTab(index, 0);
        tabbar_.setCurrentIndex(0);
        while (tabbar_.count() > 1) removeTab(1);
      }
      dropdown::EndPopup();
    }
  }));
  connections_.push_back(tabbar_.currentChanged.connect([this](int index) {
    if (index != -1) updateLayout();
  }));

  newTab();
}

ChartsWidget::~ChartsWidget() = default;

std::string ChartsWidget::whatsThis() const {
  return R"(
    <b>Chart View</b><br />
    <b>Click</b>: Click to seek to a corresponding time.<br />
    <b>Drag</b>: Zoom into the chart.<br />
    <b>Shift + Drag</b>: Scrub through the chart to view values.<br />
    <b>Right-click</b>: Open the context menu.<br />
  )";
}

void ChartsWidget::newTab() {
  static int tab_unique_id = 0;
  int idx = tabbar_.addTab("");
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
    int id = tabbar_.tabData(i);
    const auto &charts_in_tab = tab_charts_[id];
    const auto it = tab_names_.find(id);
    const std::string prefix = (it != tab_names_.end()) ? it->second : "Tab " + std::to_string(i + 1);
    tabbar_.setTabText(i, prefix + " (" + std::to_string((int)charts_in_tab.size()) + ")");
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

void ChartsWidget::drawToolBar() {
  float slider_width = 150.0f;
  const bool is_zoomed = can->timeRange().has_value();

  // the labels are captured by reference, they outlive the draw calls below
  std::vector<ToolbarItem> items;
  items.push_back({iconButtonWidth(), [this]() {
    if (stepButton("new_plot_btn", true, "New Chart")) newChart();
  }});
  items.push_back({iconButtonWidth(), [this]() {
    if (iconButton("new_tab_btn", icon::WINDOW_PLUS, "New Tab")) newTab();
  }});
  items.push_back({iconButtonWidth(), [this]() {
    if (iconButton("cereal_browser_btn", icon::GRAPH_UP, "Browse Cereal Signals"))
      cereal_browser_open_ = !cereal_browser_open_;
  }});
  items.push_back({iconButtonWidth(), [this]() {
    if (iconButton("python_panel_btn", icon::PENCIL, "Derived Signals (Python)"))
      python_panel_open_ = !python_panel_open_;
  }});
  items.push_back({iconButtonWidth(), [this]() {
    if (iconButton("map_panel_btn", icon::GEO_ALT, "GPS Map"))
      map_panel_open_ = !map_panel_open_;
  }});
  items.push_back({iconButtonWidth(), [this]() {
    if (iconButton("log_panel_btn", icon::JOURNAL_TEXT, "Events Log"))
      log_panel_open_ = !log_panel_open_;
  }});

  const int type_count = (int)std::size(SERIES_TYPE_NAMES);
  const std::string chart_type_text = std::string("Type:  ") + SERIES_TYPE_NAMES[std::clamp(settings.chart_series_type, 0, type_count - 1)];
  auto chart_type_items = [this]() {
    for (int i = 0; i < type_count; ++i) {
      if (dropdown::Item(SERIES_TYPE_NAMES[i], nullptr, settings.chart_series_type == i)) {
        settings.chart_series_type = i;
        settingChanged();
      }
    }
  };
  items.push_back(toolbarMenu("chart_type", chart_type_text, "Type", chart_type_items));

  const std::string columns_action_text = "Columns:  " + std::to_string(column_count_);
  if (columns_action_visible_) {
    auto column_items = [this]() {
      for (int i = 0; i < MAX_COLUMN_COUNT; ++i) {
        if (dropdown::Item(std::to_string(i + 1).c_str(), nullptr, column_count_ == i + 1)) setColumnCount(i + 1);
      }
    };
    items.push_back(toolbarMenu("columns", columns_action_text, "Columns", column_items));
  }

  // the spacer right aligns the rest
  const size_t spacer_index = items.size();
  size_t slider_index = (size_t)-1;
  const std::string range_lb = is_zoomed ? std::string() : utils::formatSeconds(max_chart_range_);
  std::string reset_zoom_text;
  if (!is_zoomed) {
    // the range label and the slider are one unit: drawn inline and moved to the overflow menu together
    slider_index = items.size();
    const float label_width = ImGui::CalcTextSize(range_lb.c_str()).x + ImGui::GetStyle().ItemInnerSpacing.x;
    items.push_back({label_width + slider_width, [this, &range_lb, &slider_width]() {
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted(range_lb.c_str());
      ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
      // Restore the slider width in overflow; the toolbar may have shrunk it.
      const bool in_menu = ImGui::GetCurrentWindow()->Flags & ImGuiWindowFlags_Popup;
      const float width = in_menu ? std::max(ImGui::GetContentRegionAvail().x, 150.0f) : slider_width;
      if (range_slider_.draw("##range_slider", width)) setMaxChartRange(range_slider_.value());
      ImGui::SetItemTooltip("Set the chart range");
    }});
  } else {
    const auto &range = *can->timeRange();
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f-%.2f", range.first, range.second);
    reset_zoom_text = buf;
    // The undo/redo/reset buttons form one group. The reset button has a fixed width in the mono font,
    // sized for the longest range the stream can show, so its neighbors do not shift as the range changes.
    const int digits = std::max({1, (int)std::to_string((long long)can->maxSeconds()).size(), (int)std::to_string((long long)range.second).size()});
    const std::string widest = std::string(digits, '0') + ".00";
    pushMonoFont(ImGui::GetFontSize());
    const float reset_zoom_width = iconTextButtonWidth(icon::ZOOM_OUT, widest + "-" + widest);
    popMonoFont();
    items.push_back({iconButtonWidth() * 2 + ImGui::GetStyle().ItemSpacing.x, [this]() {
      ImGui::BeginDisabled(!zoom_undo_stack_.canUndo());
      if (iconButton("undo_zoom", icon::ARROW_COUNTERCLOCKWISE, "Undo Zoom")) zoom_undo_stack_.undo();
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(!zoom_undo_stack_.canRedo());
      if (iconButton("redo_zoom", icon::ARROW_CLOCKWISE, "Redo Zoom")) zoom_undo_stack_.redo();
      ImGui::EndDisabled();
    }});
    items.push_back({reset_zoom_width, [this, &reset_zoom_text, reset_zoom_width]() {
      pushMonoFont(ImGui::GetFontSize());
      const bool clicked = iconTextButton("reset_zoom_btn", icon::ZOOM_OUT, reset_zoom_text, reset_zoom_width);
      popMonoFont();
      if (clicked) zoomReset();
      ImGui::SetItemTooltip("Reset Zoom");
    }});
  }
  items.push_back(toolbarAction("remove_all_btn", icon::TRASH, "Remove all charts", [this]() { removeAll(); }, !charts_.empty()));

  // the slider shrinks first, the buttons stay pinned to the right edge
  if (slider_index != (size_t)-1) {
    const float shrink = std::min(slider_width - MIN_RANGE_SLIDER_WIDTH, toolbarWidth(items, spacer_index) - ImGui::GetContentRegionAvail().x);
    if (shrink > 0.0f) {
      slider_width -= shrink;
      items[slider_index].width -= shrink;
    }
  }
  drawToolbar(items, spacer_index);
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
  auto &current = currentCharts();
  current.insert(current.begin() + std::min(pos, (int)current.size()), ptr);
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

void ChartsWidget::splitChart(ChartView *src_chart) {
  if (src_chart->signals().size() > 1) {
    auto &current = currentCharts();
    const int pos = std::find(current.begin(), current.end(), src_chart) - current.begin() + 1;
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
      if (s.isCereal()) {
        ids += "cereal:" + s.cereal_path;
      } else {
        ids += s.msg_id.toString() + "|" + s.sig->name;
      }
    }
    chart_ids.push_back(ids);
  }
  std::reverse(chart_ids.begin(), chart_ids.end());
  return chart_ids;
}

void ChartsWidget::restoreChartsFromIds(const std::vector<std::string> &chart_ids) {
  for (const auto &chart_id : chart_ids) {
    int index = 0;
    for (const auto &part : utils::split(chart_id, ',')) {
      if (part.rfind("cereal:", 0) == 0) {
        showCerealSignal(part.substr(7), index++ > 0);
        continue;
      }
      const size_t sep = part.find('|');
      if (sep == std::string::npos) continue;
      MessageId msg_id = MessageId::fromString(part.substr(0, sep));
      if (auto *msg = dbc()->msg(msg_id))
        if (auto *sig = msg->sig(part.substr(sep + 1)))
          showChart(msg_id, sig, true, index++ > 0);
    }
  }
}

bool ChartsWidget::loadLayout(const std::string &path) {
  std::ifstream f(path);
  if (!f) { can->error("Layout file not found: " + path); return false; }
  const std::string src{std::istreambuf_iterator<char>(f), {}};
  std::string err;
  auto doc = json11::Json::parse(src, err);
  if (!err.empty() || !doc.is_object()) { can->error("Layout parse error: " + err); return false; }

  // create one tab per JSON tab
  for (const auto &tab_j : doc["tabs"].array_items()) {
    newTab();
    const std::string tab_name = tab_j["name"].string_value();
    if (!tab_name.empty())
      tab_names_[tabbar_.tabData(tabbar_.currentIndex())] = tab_name;
    for (const auto &chart_j : tab_j["charts"].array_items()) {
      int idx = 0;
      for (const auto &sig_j : chart_j["signals"].array_items()) {
        const std::string &path_s = sig_j["path"].string_value();
        if (!path_s.empty()) { showCerealSignal(path_s, idx++ > 0); continue; }
        const std::string &can_s = sig_j["can"].string_value();
        if (!can_s.empty()) {
          const size_t sep = can_s.find('|');
          if (sep != std::string::npos) {
            MessageId msg_id = MessageId::fromString(can_s.substr(0, sep));
            if (auto *msg = dbc()->msg(msg_id))
              if (auto *sig = msg->sig(can_s.substr(sep + 1)))
                showChart(msg_id, sig, true, idx++ > 0);
          }
        }
      }
      // derived signals (Lua snippets) — placeholder added now, filled when snippets run
      for (const auto &name_j : chart_j["_skipped"].array_items()) {
        const std::string &name = name_j.string_value();
        if (!name.empty()) showCerealSignal(name, idx++ > 0);
      }
    }
  }

  // populate python snippets panel
  for (const auto &s_j : doc["snippets"].array_items()) {
    PythonSnippet s;
    s.name          = s_j["name"].string_value();
    s.linked_source = s_j["linked_source"].string_value();
    s.code          = s_j["code"].string_value();
    for (const auto &v : s_j["additional_sources"].array_items())
      s.additional_sources.push_back(v.string_value());
    python_snippets_.push_back(std::move(s));
  }
  if (!python_snippets_.empty()) {
    python_panel_open_ = true;
    for (auto &snip : python_snippets_) {
      CerealSeriesMap result;
      std::string py_err = pythonEvaluate(snip, can->cerealSeries(), result);
      if (py_err.empty() && !result.empty())
        can->insertDerivedSeries(snip.name, std::move(result.begin()->second));
    }
  }
  return true;
}

void ChartsWidget::saveLayout(const std::string &path) {
  json11::Json::array tabs_j;
  for (int i = 0; i < tabbar_.count(); ++i) {
    int id = tabbar_.tabData(i);
    json11::Json::array charts_j;
    for (auto *c : tab_charts_[id]) {
      json11::Json::array sigs_j;
      for (const auto &s : c->signals()) {
        if (s.isCereal())
          sigs_j.push_back(json11::Json::object{{"path", s.cereal_path}});
        else
          sigs_j.push_back(json11::Json::object{{"can", s.msg_id.toString() + "|" + s.sig->name}});
      }
      if (!sigs_j.empty())
        charts_j.push_back(json11::Json::object{{"signals", sigs_j}});
    }
    const auto nit = tab_names_.find(id);
    const std::string tab_name = (nit != tab_names_.end()) ? nit->second : "Tab " + std::to_string(i + 1);
    tabs_j.push_back(json11::Json::object{{"name", tab_name}, {"charts", charts_j}});
  }

  json11::Json::array snippets_j;
  for (const auto &s : python_snippets_) {
    json11::Json::array addl;
    for (const auto &a : s.additional_sources) addl.push_back(json11::Json(a));
    snippets_j.push_back(json11::Json::object{
      {"name", s.name}, {"linked_source", s.linked_source},
      {"code", s.code}, {"additional_sources", addl}
    });
  }

  std::filesystem::path fp(path);
  json11::Json doc = json11::Json::object{
    {"name", fp.stem().string()}, {"tabs", tabs_j}, {"snippets", snippets_j}
  };

  std::ofstream f(path);
  if (!f) { can->error("Could not write layout: " + path); return; }
  f << doc.dump();
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
    if ((n * CHART_MIN_WIDTH + (n - 1) * ImGui::GetStyle().ItemSpacing.x) < container_width) break;
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

  // The tip is drawn without an input item, so the mouse is never "on the tip".
  const ImVec2 delta = ImGui::GetIO().MouseDelta;
  if (!any_plot_hovered_ &&
      (delta.x != 0 || delta.y != 0 || !ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows))) {
    showValueTip(-1);  // the mouse moved off the plot or out of the charts window
  }
}

void ChartsWidget::draw() {
  deleted_charts_.clear();
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
  if (cereal_browser_open_) drawCerealBrowser();
  if (python_panel_open_) drawPythonPanel();
  if (map_panel_open_) drawMapPanel();
  if (log_panel_open_) drawLogPanel();
  ImGui::PopID();
}

void ChartsContainer::draw() {
  const ImVec2 start = ImGui::GetCursorScreenPos();
  const float width_avail = ImGui::GetContentRegionAvail().x;
  geometry_ = ImRect(start, start + ImVec2(width_avail, 0));
  charts_widget_->updateLayout();

  const int n = std::max(charts_widget_->current_column_count_, 1);
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  const float width = (geometry_.GetWidth() - (n - 1) * spacing) / n;
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  auto current_charts = charts_widget_->currentCharts();  // copy: drawing may remove charts
  float bottom = origin.y;
  const bool aligned = ImPlot::BeginAlignedPlots("charts_align", true);
  for (int i = 0; i < current_charts.size(); ++i) {
    ImVec2 pos = origin + ImVec2((i % n) * (width + spacing), (i / n) * (settings.chart_height + ImGui::GetStyle().ItemSpacing.y));
    ImGui::SetCursorScreenPos(pos);
    current_charts[i]->draw(width);
    bottom = std::max(bottom, pos.y + settings.chart_height);
    if (current_charts[i]->plotHovered()) charts_widget_->any_plot_hovered_ = true;  // the window must be hovered too
  }
  if (aligned) ImPlot::EndAlignedPlots();
  ImGui::SetCursorScreenPos(ImVec2(origin.x, bottom));
  ImGui::Dummy(ImVec2(geometry_.GetWidth(), ImGui::GetStyle().ItemSpacing.y));
  geometry_.Max.y = bottom + ImGui::GetStyle().ItemSpacing.y;
  drawDropIndicator();
}

void ChartsContainer::drawDropIndicator() {
  if (!(drop_indicator_pos_.x == 0 && drop_indicator_pos_.y == 0) && !childAt(drop_indicator_pos_)) {
    ImRect r = geometry_;
    r.Max.y = r.Min.y + ImGui::GetStyle().ItemSpacing.y;
    if (auto insert_after = getDropAfter(drop_indicator_pos_)) {
      float h = r.GetHeight();
      r.Min.y = insert_after->rect().Max.y;
      r.Max.y = r.Min.y + h;
    }

    ImGui::GetWindowDrawList()->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_Header), ImGui::GetStyle().FrameRounding);
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

void ChartsWidget::showCerealSignal(const std::string &path, bool merge) {
  // find any existing chart containing this cereal path
  ChartView *existing = nullptr;
  for (auto &c : charts_)
    if (c->hasCerealSignal(path)) { existing = c.get(); break; }

  if (existing) {
    // switch to the tab that owns this chart
    for (int i = 0; i < tabbar_.count(); ++i) {
      const auto &tab_c = tab_charts_[tabbar_.tabData(i)];
      if (std::find(tab_c.begin(), tab_c.end(), existing) != tab_c.end()) {
        tabbar_.setCurrentIndex(i);
        break;
      }
    }
  } else {
    ChartView *chart = merge && !currentCharts().empty() ? currentCharts().front() : createChart();
    chart->addCerealSignal(path);
    updateState();
  }
}

void ChartsWidget::cerealSeriesUpdated() {
  for (auto &c : charts_) c->updateCerealSeries();

  // re-run any snippet whose output series doesn't exist yet (e.g. after layout load at startup)
  if (!python_evaluating_ && !python_snippets_.empty() && !can->cerealSeries().empty()) {
    python_evaluating_ = true;
    for (auto &s : python_snippets_) {
      CerealSeriesMap result;
      std::string py_err = pythonEvaluate(s, can->cerealSeries(), result);
      if (py_err.empty() && !result.empty())
        can->insertDerivedSeries(s.name, std::move(result.begin()->second));
    }
    python_evaluating_ = false;
  }
}

void ChartsWidget::drawCerealBrowser() {
  ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Cereal Signals", &cereal_browser_open_)) {
    ImGui::End();
    return;
  }

  // search filter
  bool filter_changed = ImGui::InputText("##cereal_filter", cereal_filter_buf_, sizeof(cereal_filter_buf_));
  ImGui::SameLine();
  if (ImGui::Button("Clear")) {
    cereal_filter_buf_[0] = '\0';
    filter_changed = true;
  }

  const auto &series = can->cerealSeries();
  if (filter_changed || cereal_filtered_paths_.empty()) {
    cereal_filtered_paths_.clear();
    const std::string filter(cereal_filter_buf_);
    for (const auto &[path, _] : series) {
      if (filter.empty() || path.find(filter) != std::string::npos)
        cereal_filtered_paths_.push_back(path);
    }
    std::sort(cereal_filtered_paths_.begin(), cereal_filtered_paths_.end());
  }

  ImGui::Text("%zu signals", cereal_filtered_paths_.size());
  ImGui::Separator();

  if (ImGui::BeginChild("cereal_list")) {
    for (const auto &path : cereal_filtered_paths_) {
      bool selected = false;
      for (auto &c : charts_) selected |= c->hasCerealSignal(path);
      if (ImGui::Selectable(path.c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick)) {
        if (ImGui::IsMouseDoubleClicked(0)) {
          bool merge = ImGui::GetIO().KeyShift;
          showCerealSignal(path, merge);
        }
      }
      if (ImGui::IsItemHovered()) {
        auto it = series.find(path);
        if (it != series.end() && !it->second.empty()) {
          ImGui::SetTooltip("%.4g (latest)  %zu samples",
                            it->second.back().v, it->second.size());
        }
      }
    }
  }
  ImGui::EndChild();
  ImGui::End();
}

// ── Python derived signal panel ───────────────────────────────────────────────

static void copyPySnippetToBuffers(const PythonSnippet &s, char *name, char *linked,
                                   char *v1, char *v2, char *v3, char *v4, char *code) {
  snprintf(name,   256,  "%s", s.name.c_str());
  snprintf(linked, 256,  "%s", s.linked_source.c_str());
  auto src = [&](int i) -> const char * { return i < (int)s.additional_sources.size() ? s.additional_sources[i].c_str() : ""; };
  snprintf(v1, 256, "%s", src(0));
  snprintf(v2, 256, "%s", src(1));
  snprintf(v3, 256, "%s", src(2));
  snprintf(v4, 256, "%s", src(3));
  snprintf(code, 4096, "%s", s.code.c_str());
}

static PythonSnippet buffersToPySnippet(const char *name, const char *linked,
                                        const char *v1, const char *v2, const char *v3, const char *v4,
                                        const char *code) {
  PythonSnippet s;
  s.name          = name;
  s.linked_source = linked;
  s.code          = code;
  for (const char *src : {v1, v2, v3, v4})
    if (src[0]) s.additional_sources.push_back(src);
  return s;
}

static const char *DEFAULT_PYTHON_CODE =
  "def derive(value, v1, v2, v3, v4, time):\n"
  "    return value\n";

void ChartsWidget::drawPythonPanel() {
  ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Derived Signals (Python)", &python_panel_open_)) { ImGui::End(); return; }

  // left: snippet list
  ImGui::BeginChild("py_list", ImVec2(160, 0), true);
  for (int i = 0; i < (int)python_snippets_.size(); i++) {
    if (ImGui::Selectable(python_snippets_[i].name.c_str(), python_selected_ == i)) {
      python_selected_ = i;
      copyPySnippetToBuffers(python_snippets_[i], python_name_buf_, python_linked_buf_,
                             python_v1_buf_, python_v2_buf_, python_v3_buf_, python_v4_buf_,
                             python_code_buf_);
      python_error_.clear();
      python_success_.clear();
    }
  }
  if (ImGui::Button("+ New")) {
    PythonSnippet blank;
    blank.name = "new_signal";
    blank.code = DEFAULT_PYTHON_CODE;
    python_snippets_.push_back(std::move(blank));
    python_selected_ = (int)python_snippets_.size() - 1;
    copyPySnippetToBuffers(python_snippets_[python_selected_], python_name_buf_, python_linked_buf_,
                           python_v1_buf_, python_v2_buf_, python_v3_buf_, python_v4_buf_,
                           python_code_buf_);
    python_error_.clear();
    python_success_.clear();
  }
  ImGui::EndChild();

  ImGui::SameLine();

  // right: editor
  ImGui::BeginChild("py_editor", ImVec2(0, 0));
  if (python_selected_ < 0 || python_selected_ >= (int)python_snippets_.size()) {
    ImGui::TextUnformatted("Select or create a snippet on the left.");
    ImGui::EndChild();
    ImGui::End();
    return;
  }

  ImGui::InputText("Name", python_name_buf_, sizeof(python_name_buf_));
  ImGui::InputText("Linked source (value)", python_linked_buf_, sizeof(python_linked_buf_));
  ImGui::InputText("v1", python_v1_buf_, sizeof(python_v1_buf_)); ImGui::SameLine();
  ImGui::InputText("v2", python_v2_buf_, sizeof(python_v2_buf_)); ImGui::SameLine();
  ImGui::InputText("v3", python_v3_buf_, sizeof(python_v3_buf_)); ImGui::SameLine();
  ImGui::InputText("v4", python_v4_buf_, sizeof(python_v4_buf_));

  ImGui::TextUnformatted("Python code (define derive(value, v1, v2, v3, v4, time) → number):");
  ImGui::InputTextMultiline("##pycode", python_code_buf_, sizeof(python_code_buf_), ImVec2(-1, 180));

  if (!python_error_.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 80, 80, 255));
    ImGui::TextWrapped("%s", python_error_.c_str());
    ImGui::PopStyleColor();
  } else if (!python_success_.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(80, 200, 80, 255));
    ImGui::TextUnformatted(python_success_.c_str());
    ImGui::PopStyleColor();
  }

  if (ImGui::Button("Run")) {
    PythonSnippet s = buffersToPySnippet(python_name_buf_, python_linked_buf_,
                                         python_v1_buf_, python_v2_buf_, python_v3_buf_, python_v4_buf_,
                                         python_code_buf_);
    python_snippets_[python_selected_] = s;
    python_success_.clear();
    CerealSeriesMap result;
    python_error_ = pythonEvaluate(s, can->cerealSeries(), result);
    if (python_error_.empty() && !result.empty()) {
      auto &pts = result.begin()->second;
      python_success_ = "OK — " + std::to_string(pts.size()) + " samples computed";
      can->insertDerivedSeries(s.name, std::move(pts));
    } else if (python_error_.empty()) {
      python_error_ = "linked source '" + s.linked_source + "' not found in loaded data";
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Plot")) {
    if (python_selected_ >= 0) {
      PythonSnippet s = buffersToPySnippet(python_name_buf_, python_linked_buf_,
                                           python_v1_buf_, python_v2_buf_, python_v3_buf_, python_v4_buf_,
                                           python_code_buf_);
      if (!can->cerealSeries().count(s.name)) {
        CerealSeriesMap result;
        if (pythonEvaluate(s, can->cerealSeries(), result).empty() && !result.empty())
          can->insertDerivedSeries(s.name, std::move(result.begin()->second));
      }
      if (can->cerealSeries().count(s.name))
        showCerealSignal(s.name, ImGui::GetIO().KeyShift);
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Delete") && python_selected_ >= 0) {
    python_snippets_.erase(python_snippets_.begin() + python_selected_);
    python_selected_ = std::min(python_selected_, (int)python_snippets_.size() - 1);
    python_error_.clear();
  }

  ImGui::EndChild();
  ImGui::End();
}

void ChartsWidget::drawMapPanel() {
  ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("GPS Map", &map_panel_open_)) { ImGui::End(); return; }

  const auto &series = can->cerealSeries();
  const char *lat_paths[] = {"/gpsLocationExternal/latitude", "/gpsLocation/latitude"};
  const char *lon_paths[] = {"/gpsLocationExternal/longitude", "/gpsLocation/longitude"};
  const std::vector<CerealPoint> *lats = nullptr, *lons = nullptr;
  for (int i = 0; i < 2; ++i) {
    auto la = series.find(lat_paths[i]), lo = series.find(lon_paths[i]);
    if (la != series.end() && lo != series.end() && !la->second.empty()) {
      lats = &la->second; lons = &lo->second; break;
    }
  }

  if (!lats) {
    ImGui::TextDisabled("No GPS data (gpsLocation / gpsLocationExternal).");
    ImGui::End(); return;
  }

  std::vector<double> xs, ys;
  xs.reserve(lats->size());
  ys.reserve(lats->size());
  for (const auto &lp : *lats) {
    auto it = std::lower_bound(lons->begin(), lons->end(), lp.t,
                               [](const CerealPoint &p, double t) { return p.t < t; });
    if (it == lons->end()) continue;
    xs.push_back(it->v);
    ys.push_back(lp.v);
  }
  if (xs.empty()) { ImGui::TextDisabled("No GPS samples."); ImGui::End(); return; }

  double cur = can->currentSec();
  auto cur_la = std::lower_bound(lats->begin(), lats->end(), cur,
                                 [](const CerealPoint &p, double t) { return p.t < t; });
  auto cur_lo = std::lower_bound(lons->begin(), lons->end(), cur,
                                 [](const CerealPoint &p, double t) { return p.t < t; });
  if (cur_la != lats->begin()) --cur_la;
  if (cur_lo != lons->begin()) --cur_lo;
  double cur_lon = cur_lo->v, cur_lat = cur_la->v;

  if (ImPlot::BeginPlot("##gpsmap", ImVec2(-1, -1), ImPlotFlags_Equal | ImPlotFlags_NoTitle)) {
    ImPlot::SetupAxis(ImAxis_X1, "Longitude");
    ImPlot::SetupAxis(ImAxis_Y1, "Latitude");
    ImPlotSpec route_spec;
    route_spec.LineColor = ImVec4(0.3f, 0.6f, 1.0f, 0.9f);
    route_spec.LineWeight = 2.0f;
    ImPlot::PlotLine("Route", xs.data(), ys.data(), (int)xs.size(), route_spec);
    ImPlotSpec pos;
    pos.Marker = ImPlotMarker_Circle;
    pos.MarkerSize = 8.0f;
    pos.MarkerFillColor = ImVec4(1, 0.4f, 0.1f, 1);
    pos.MarkerLineColor = ImVec4(1, 1, 1, 1);
    ImPlot::PlotScatter("Position", &cur_lon, &cur_lat, 1, pos);
    ImPlot::EndPlot();
  }
  ImGui::End();
}

void ChartsWidget::drawLogPanel() {
  ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Events Log", &log_panel_open_)) { ImGui::End(); return; }

  ImGui::SetNextItemWidth(-1);
  ImGui::InputTextWithHint("##logfilter", "Filter signals...", log_filter_buf_, sizeof(log_filter_buf_));
  ImGui::Separator();

  const auto &series = can->cerealSeries();
  double cur = can->currentSec();
  std::string_view filter(log_filter_buf_);

  if (ImGui::BeginTable("##logtable", 3,
      ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable,
      ImVec2(-1, -1))) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Signal",   ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Value",    ImGuiTableColumnFlags_WidthFixed, 100);
    ImGui::TableSetupColumn("Age (s)",  ImGuiTableColumnFlags_WidthFixed, 70);
    ImGui::TableHeadersRow();

    for (const auto &[path, pts] : series) {
      if (!filter.empty() && path.find(filter) == std::string::npos) continue;
      if (pts.empty()) continue;
      auto it = std::lower_bound(pts.begin(), pts.end(), cur,
                                 [](const CerealPoint &p, double t) { return p.t < t; });
      if (it != pts.begin()) --it;
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(path.c_str());
      ImGui::TableSetColumnIndex(1); ImGui::Text("%.4g", it->v);
      ImGui::TableSetColumnIndex(2);
      double age = cur - it->t;
      ImGui::TextColored(age < 1.0 ? ImVec4(0.4f,1,0.4f,1) : ImVec4(0.7f,0.7f,0.7f,1), "%.2f", age);
    }
    ImGui::EndTable();
  }
  ImGui::End();
}
