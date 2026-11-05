#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/chartswidget.h"

#include "tools/cabana/ui/threadpool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#ifndef CABANA_REPO_ROOT
#define CABANA_REPO_ROOT "."
#endif
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/core/csv.h"
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

  connections_.push_back(dbc()->fileChanged.connect([this]() {
    std::vector<ChartView *> charts;
    for (auto &chart : charts_) charts.push_back(chart.get());
    for (auto *chart : charts) chart->removeIf([](const auto &s) { return !s.isCereal(); });
  }));
  connections_.push_back(can->eventsMerged.connect([this](const MessageEventsMap &events) { eventsMerged(events); }));
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *, bool) { updateState(); }));
  connections_.push_back(can->seeking.connect([this](double) { updateState(); requestDerivedUpdate(false); }));
  connections_.push_back(can->timeRangeChanged.connect([this](const auto &) { updateState(); }));
  connections_.push_back(settings.changed.connect([this]() { settingChanged(); }));
  connections_.push_back(seriesChanged.connect([this]() { updateTabBar(); requestCerealUpdate(); }));
  connections_.push_back(can->cerealChanged.connect([this]() { requestCerealUpdate(true); }));
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

ChartsWidget::~ChartsWidget() {
  ++*cereal_generation_;
  ++*map_generation_;
  ++*derived_generation_;
  alive_.reset();
}

void ChartsWidget::requestCerealUpdate(bool data_changed) {
  if (changing_layout_) return;
  requestMapUpdate(data_changed);
  requestDerivedUpdate(data_changed);
  if (data_changed && cereal_running_) { cereal_dirty_ = true; return; }
  cereal_running_ = cereal_dirty_ = false;
  const auto generation = ++*cereal_generation_;
  auto reader = can->cerealReader();
  std::set<std::string> paths;
  for (const auto &chart : charts_) for (const auto &series : chart->signals()) {
    if (series.isCereal() && !series.python) paths.insert(series.cereal_path);
  }
  if (!reader || paths.empty()) return;
  cereal_running_ = true;
  auto state = cereal_generation_;
  auto publish = utils::guarded(alive_, [this, state, generation](const std::map<std::string, cabana::CerealPoints> &result,
                                                               const std::map<std::string, std::string> &errors) {
    if (state->load() != generation) return;
    for (auto &chart : charts_) {
      for (const auto &[path, points] : result) chart->setCerealSeries(path, points);
      for (const auto &[path, error] : errors) chart->setCerealError(path, error);
    }
    cereal_running_ = false;
    updateState();
    if (cereal_dirty_) requestCerealUpdate(true);
  });
  ThreadPool::analysis().run([reader = std::move(reader), paths = std::move(paths), state, generation, publish]() {
    auto cancelled = [state, generation]() { return state->load() != generation; };
    std::map<std::string, cabana::CerealPoints> result;
    std::map<std::string, std::string> errors;
    for (const auto &path : paths) {
      if (cancelled()) return;
      try { result[path] = reader(path, cancelled); }
      catch (const std::exception &error) { errors[path] = error.what(); }
    }
    if (!cancelled()) utils::runOnMainThread([publish, result = std::move(result), errors]() { publish(result, errors); });
  });
}

void ChartsWidget::requestDerivedUpdate(bool data_changed) {
  if (data_changed && derived_running_) { derived_dirty_ = true; return; }
  const auto generation = ++*derived_generation_;
  derived_running_ = derived_dirty_ = false;
  std::map<std::string, cabana::PythonSeries> expressions;
  std::map<std::string, std::string> aliases;
  const auto raw_names = can->cerealSignalNames();
  for (const auto &[name, spec] : derived_definitions_) {
    const auto key = cabana::pythonSeriesKey(spec);
    expressions[key] = spec;
    if (std::find(raw_names.begin(), raw_names.end(), name) == raw_names.end()) aliases[name] = key;
  }
  std::set<std::string> requested;

  for (const auto &chart : charts_) for (const auto &series : chart->signals()) {
    if (!series.python) continue;
    const auto key = series.key();
    expressions[key] = *series.python;
    requested.insert(key);
    auto name = series.cereal_path;
    if (!name.empty() && name.front() == '/') name.erase(0, 1);
    if (std::find(raw_names.begin(), raw_names.end(), name) != raw_names.end()) continue;
    auto found = aliases.find(name);
    if (found == aliases.end()) aliases[name] = key;
    else if (found->second != key) found->second.clear();
  }
  if (requested.empty()) return;
  auto reader = can->cerealHistoryReader();
  if (!reader) return;
  derived_running_ = true;
  const auto origin = can->beginMonoTime();
  auto state = derived_generation_;
  auto publish = utils::guarded(alive_, [this, state, generation](const std::map<std::string, cabana::CerealPoints> &result,
                                                               const std::map<std::string, std::string> &errors) {
    if (state->load() != generation) return;
    for (auto &chart : charts_) {
      for (const auto &[key, points] : result) chart->setCerealSeries(key, points);
      for (const auto &[key, error] : errors) chart->setCerealError(key, error);
    }
    derived_running_ = false;
    updateState();
    if (derived_dirty_) requestDerivedUpdate(true);
  });
  ThreadPool::history().run([reader, expressions, aliases, requested, origin, state, generation, publish]() {
    auto cancelled = [state, generation]() { return state->load() != generation; };
    std::set<std::string> paths;
    std::set<std::string> needed;
    const auto collect = [&](auto &&self, const std::string &key) -> void {
      if (!needed.insert(key).second) return;
      const auto &spec = expressions.at(key);
      auto sources = spec.additional_sources; sources.push_back(spec.linked_source);
      for (const auto &source : sources) {
        auto found = aliases.find(source);
        if (found != aliases.end() && !found->second.empty()) self(self, found->second);
        else paths.insert(source);
      }
    };
    for (const auto &key : requested) collect(collect, key);
    for (const auto &[name, _] : aliases) paths.erase(name);
    std::map<std::string, cabana::CerealPoints> inputs, result;
    std::map<std::string, std::string> errors;
    std::map<std::string, bool> visiting;
    std::string history_error;
    try { inputs = reader(paths, cancelled); }
    catch (const std::exception &failure) { history_error = failure.what(); }
    const auto evaluate = [&](auto &&self, const std::string &key) -> void {
      if (cancelled()) throw std::runtime_error("Calculation cancelled");
      if (result.count(key)) return;
      if (errors.count(key)) throw std::runtime_error(errors.at(key));
      if (visiting[key]) throw std::invalid_argument("Derived curves have a cyclic dependency");
      visiting[key] = true;
      try {
        const auto &spec = expressions.at(key);
        auto sources = spec.additional_sources; sources.push_back(spec.linked_source);
        for (const auto &source : sources) {
          const auto dependency = aliases.find(source);
          if (dependency == aliases.end()) continue;
          if (dependency->second.empty()) throw std::invalid_argument("Ambiguous derived curve name: " + source);
          self(self, dependency->second);
          inputs[source] = result.at(dependency->second);
        }
        if (!history_error.empty()) throw std::runtime_error(history_error);
        result[key] = cabana::evaluatePythonSeries(spec, inputs, cancelled, 15.0, origin);
        visiting[key] = false;
      } catch (...) { visiting[key] = false; throw; }
    };
    for (const auto &key : requested) {
      if (cancelled()) return;
      try { evaluate(evaluate, key); }
      catch (const std::exception &failure) { errors[key] = failure.what(); }
    }
    if (!cancelled()) utils::runOnMainThread([publish, result = std::move(result), errors]() { publish(result, errors); });
  });
}

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
  tab_names_.erase(id); tab_roots_.erase(id);
  tabbar_.removeTab(index);
  updateTabBar();
}

void ChartsWidget::updateTabBar() {
  for (int i = 0; i < tabbar_.count(); ++i) {
    const auto &charts_in_tab = tab_charts_[tabbar_.tabData(i)];
    const auto name = tab_names_.find(tabbar_.tabData(i));
    const std::string title = name == tab_names_.end() || name->second.empty() ? "Tab " + std::to_string(i + 1) : name->second;
    tabbar_.setTabText(i, title + " (" + std::to_string(charts_in_tab.size()) + ")");
  }
}

void ChartsWidget::eventsMerged(const MessageEventsMap &new_events) {
  std::vector<std::future<void>> futures;
  for (auto &c : charts_) {
    futures.push_back(ThreadPool::instance().run([c = c.get(), &new_events]() { c->updateSeries(nullptr, &new_events); }));
  }
  for (auto &f : futures) f.get();
  requestCerealUpdate(true);
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

  items.push_back(toolbarMenu("analysis_layouts", "Layouts", "Layouts", [this]() {
    if (ImGui::BeginMenu("Bundled layouts")) {
      std::error_code error;
      const auto directory = std::filesystem::path(CABANA_REPO_ROOT) / "openpilot/tools/cabana/layouts";
      for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
        if (entry.path().extension() == ".json" && ImGui::MenuItem(entry.path().stem().c_str())) loadAnalysisLayoutFile(entry.path().string());
      }
      ImGui::EndMenu();
    }
    if (dropdown::Item("Use grid layout")) {
      tab_roots_.erase(tabbar_.tabData(tabbar_.currentIndex())); ++layout_generation_; updateLayout();
    }
    if (dropdown::Item("Open layout...")) {
      FileDialog::getOpenFileName("Open analysis layout", settings.last_dir, ".json,.xml",
                                 utils::guarded(alive_, [this](const std::string &path) { loadAnalysisLayoutFile(path); }));
    }
    if (dropdown::Item("Export plotted CSV...")) {
      FileDialog::getSaveFileName("Export plotted series", settings.last_dir + "/plots.csv", ".csv",
        utils::guarded(alive_, [this](const std::string &path) {
          std::vector<cabana::CsvSeries> series;
          for (const auto &chart : charts_) for (const auto &signal : chart->signals()) {
            if (!signal.visible) continue;
            cabana::CsvSeries item; item.name = signal.name();
            if (!signal.text_codes.empty()) item.labels = signal.enum_labels;
            for (const auto &point : signal.vals) item.points.push_back({point.x, point.y});
            series.push_back(std::move(item));
          }
          auto failure = utils::guarded(alive_, [](const std::string &error) { can->error(error); });
          ThreadPool::analysis().run([path, series = std::move(series), failure]() {
            try {
              std::ofstream output(path);
              if (!output) throw std::runtime_error("Cannot open CSV file: " + path);
              cabana::writeSeriesCsv(output, series);
            } catch (const std::exception &error) {
              const std::string message = error.what();
              utils::runOnMainThread([failure, message]() { failure(message); });
            }
          });
        }));
    }
    if (dropdown::Item("Save layout...")) {
      FileDialog::getSaveFileName("Save analysis layout", settings.last_dir + "/analysis.json", ".json",
        utils::guarded(alive_, [this](const std::string &path) {
          try {
            std::ofstream file(path);
            file << cabana::writeAnalysisLayout(analysisLayout());
            if (!file) throw std::runtime_error("Cannot save analysis layout: " + path);
          } catch (const std::exception &error) { can->error(error.what()); }
        }));
    }
    if (dropdown::Item("New Python curve...")) editPythonSeries();
    if (dropdown::Item("New map pane")) {
      cabana::LayoutChart pane; pane.kind = "map"; pane.title = "Map";
      createChart()->applyLayout(pane); seriesChanged();
    }
    if (ImGui::BeginMenu("New camera pane")) {
      for (const std::string view : {"road", "cabin", "wide_road", "qroad", "thumbnail"}) {
        if (ImGui::MenuItem(view.c_str())) {
          cabana::LayoutChart pane; pane.kind = view == "thumbnail" ? "thumbnail" : "camera";
          pane.title = view; pane.camera_view = view == "thumbnail" ? "road" : view;
          createChart()->applyLayout(pane); seriesChanged();
        }
      }
      ImGui::EndMenu();
    }
  }));
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
  const int tab_id = tabbar_.tabData(tabbar_.currentIndex());
  const auto before = currentCharts();
  auto chart = std::make_unique<ChartView>(can->timeRange().value_or(display_range_), this);
  ChartView *ptr = chart.get();
  pos = std::clamp(pos, 0, (int)charts_.size());
  charts_.insert(charts_.begin() + pos, std::move(chart));
  auto &current = currentCharts();
  current.insert(current.begin() + std::min(pos, (int)current.size()), ptr);
  remapLayoutTree(tab_id, before);
  ++layout_generation_;
  updateLayout();
  return ptr;
}

void ChartsWidget::showCerealChart(const std::string &path, bool merge) {
  auto chart = merge && !currentCharts().empty() ? currentCharts().front() : createChart();
  chart->addCerealSignal(path);
  updateState();
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
  return {cabana::writeAnalysisLayout(analysisLayout())};
}

void ChartsWidget::restoreChartsFromIds(const std::vector<std::string> &chart_ids) {
  for (const auto &chart_id : chart_ids) {
    if (!chart_id.empty() && chart_id.front() == '{') {
      try { loadAnalysisLayout(cabana::parseAnalysisLayout(chart_id)); }
      catch (const std::exception &error) { can->error(error.what()); }
      continue;
    }
    int index = 0;
    for (const auto &part : utils::split(chart_id, ',')) {
      const size_t sep = part.find('|');
      if (sep == std::string::npos) continue;
      if (part.substr(0, sep) == "cereal") {
        showCerealChart(part.substr(sep + 1), index++ > 0);
        continue;
      }
      MessageId msg_id = MessageId::fromString(part.substr(0, sep));
      if (auto *msg = dbc()->msg(msg_id))
        if (auto *sig = msg->sig(part.substr(sep + 1)))
          showChart(msg_id, sig, true, index++ > 0);
    }
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
    if ((n * CHART_MIN_WIDTH + (n - 1) * ImGui::GetStyle().ItemSpacing.x) < container_width) break;
  }

  columns_action_visible_ = n > 1 && !tab_roots_.count(tabbar_.tabData(tabbar_.currentIndex()));
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
    if (c != drag_.source && c->isPlot() && drag_.source->isPlot() && c->rect().Contains(global_pos)) {
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
    const auto previous_tabs = tab_charts_;
    auto w = charts_container_.getDropAfter(global_pos);
    if (w != source) {
      for (auto &[_, list] : tab_charts_) {
        list.erase(std::remove(list.begin(), list.end(), source), list.end());
      }
      auto &cur = currentCharts();
      int to = w ? std::find(cur.begin(), cur.end(), w) - cur.begin() + 1 : 0;
      cur.insert(cur.begin() + to, source);
      for (const auto &[id, before] : previous_tabs) remapLayoutTree(id, before);
      ++layout_generation_;
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
        if (!it.cereal_path.empty()) c->addCerealSignal(it.cereal_path);
        else c->addSignal(it.msg_id, it.sig);
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
  const auto previous_tabs = tab_charts_;
  ++layout_generation_;
  if (python_owner_ == chart) { python_owner_ = nullptr; python_editor_open_ = false; }
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
  for (const auto &[id, before] : previous_tabs) remapLayoutTree(id, before);
  updateLayout();
  seriesChanged();
}

void ChartsWidget::removeAll() {
  tab_roots_.clear();
  tab_names_.clear();
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
  const auto &visible_charts = currentCharts();
  for (const auto &chart : charts_) {
    if (std::find(visible_charts.begin(), visible_charts.end(), chart.get()) == visible_charts.end()) chart->hidePane();
  }
  if (ImGui::BeginChild("charts_scroll", ImVec2(0, 0), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar)) {
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
  drawPythonEditor();
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
  auto tree = charts_widget_->tab_roots_.find(charts_widget_->tabbar_.tabData(charts_widget_->tabbar_.currentIndex()));
  if (tree != charts_widget_->tab_roots_.end()) {
    const auto minimum = [&](auto &&self, const cabana::LayoutNode &node) -> ImVec2 {
      if (node.chart >= 0) return ImVec2(CHART_MIN_WIDTH, 180);
      ImVec2 result;
      for (size_t i = 0; i < node.children.size(); ++i) {
        const auto child = self(self, node.children[i]);
        const double fraction = node.sizes[i] / std::accumulate(node.sizes.begin(), node.sizes.end(), 0.0);
        if (node.horizontal) { result.x = std::max(result.x, static_cast<float>(child.x / fraction)); result.y = std::max(result.y, child.y); }
        else { result.y = std::max(result.y, static_cast<float>(child.y / fraction)); result.x = std::max(result.x, child.x); }
      }
      if (node.horizontal) result.x += 6 * (node.children.size() - 1);
      else result.y += 6 * (node.children.size() - 1);
      return result;
    };
    ImVec2 size = minimum(minimum, tree->second);
    size.x = std::max(size.x, width_avail);
    size.y = std::max(size.y, ImGui::GetContentRegionAvail().y);
    const auto render = [&](auto &&self, cabana::LayoutNode &node, ImVec2 pos, ImVec2 extent) -> void {
      if (node.chart >= 0) {
        if (node.chart < static_cast<int>(current_charts.size())) {
          ImGui::SetCursorScreenPos(pos);
          auto *chart = current_charts[node.chart];
          chart->draw(extent.x, extent.y);
          if (chart->plotHovered()) charts_widget_->any_plot_hovered_ = true;
        }
        return;
      }
      ImGui::PushID("node");
      const float length = node.horizontal ? extent.x : extent.y;
      const float gap = 6;
      const float available = std::max(1.0f, length - gap * (node.children.size() - 1));
      double total = std::accumulate(node.sizes.begin(), node.sizes.end(), 0.0);
      float offset = 0;
      for (size_t i = 0; i < node.children.size(); ++i) {
        const float part = available * node.sizes[i] / total;
        const ImVec2 child_pos = pos + (node.horizontal ? ImVec2(offset, 0) : ImVec2(0, offset));
        ImGui::PushID(static_cast<int>(i));
        self(self, node.children[i], child_pos, node.horizontal ? ImVec2(part, extent.y) : ImVec2(extent.x, part));
        ImGui::PopID();
        offset += part;
        if (i + 1 < node.children.size()) {
          ImGui::PushID(static_cast<int>(i));
          ImGui::SetCursorScreenPos(pos + (node.horizontal ? ImVec2(offset, 0) : ImVec2(0, offset)));
          ImGui::InvisibleButton("splitter", node.horizontal ? ImVec2(gap, extent.y) : ImVec2(extent.x, gap));
          if (ImGui::IsItemHovered() || ImGui::IsItemActive()) ImGui::SetMouseCursor(node.horizontal ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
          if (ImGui::IsItemActive()) {
            const float movement = node.horizontal ? ImGui::GetIO().MouseDelta.x : ImGui::GetIO().MouseDelta.y;
            const double combined = node.sizes[i] + node.sizes[i + 1];
            const double first = std::clamp(node.sizes[i] + movement / available * total, combined * 0.05, combined * 0.95);
            node.sizes[i] = first; node.sizes[i + 1] = combined - first;
          }
          ImGui::PopID();
          offset += gap;
        }
      }
      ImGui::PopID();
    };
    // Keep a local tree: a pane may delete itself while its draw callback runs.
    const auto generation = charts_widget_->layout_generation_;
    auto working_tree = tree->second;
    render(render, working_tree, origin, size);
    if (generation == charts_widget_->layout_generation_) tree->second = std::move(working_tree);
    bottom = origin.y + size.y;
    geometry_.Max.x = origin.x + size.x;
  } else {
  const bool aligned = ImPlot::BeginAlignedPlots("charts_align", true);
  for (int i = 0; i < current_charts.size(); ++i) {
    ImVec2 pos = origin + ImVec2((i % n) * (width + spacing), (i / n) * (settings.chart_height + ImGui::GetStyle().ItemSpacing.y));
    ImGui::SetCursorScreenPos(pos);
    current_charts[i]->draw(width);
    bottom = std::max(bottom, pos.y + settings.chart_height);
    if (current_charts[i]->plotHovered()) charts_widget_->any_plot_hovered_ = true;  // the window must be hovered too
  }
  if (aligned) ImPlot::EndAlignedPlots();
  }
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

cabana::AnalysisLayout ChartsWidget::analysisLayout() const {
  cabana::AnalysisLayout layout;
  layout.calculations = derived_definitions_;
  layout.current_tab = std::max(0, tabbar_.currentIndex());
  for (int i = 0; i < tabbar_.count(); ++i) {
    const int id = tabbar_.tabData(i);
    cabana::LayoutTab tab;
    auto name = tab_names_.find(id);
    tab.name = name == tab_names_.end() ? "Tab " + std::to_string(i + 1) : name->second;
    auto charts = tab_charts_.find(id);
    if (charts != tab_charts_.end()) {
      for (auto *chart : charts->second) tab.charts.push_back(chart->analysisLayout());
    }
    auto tree = tab_roots_.find(id);
    if (tree != tab_roots_.end()) tab.root = tree->second;
    layout.tabs.push_back(std::move(tab));
  }
  return layout;
}

void ChartsWidget::loadAnalysisLayoutFile(const std::string &path) {
  try {
    std::filesystem::path file(path);
    if (!std::filesystem::exists(file)) {
      const auto requested = std::filesystem::path(path).filename();
      file = std::filesystem::path(CABANA_REPO_ROOT) / "openpilot/tools/cabana/layouts" /
             (requested.has_extension() ? requested.string() : requested.string() + ".json");
    }
    std::ifstream input(file);
    if (!input) throw std::runtime_error("Layout not found: " + path);
    const std::string contents = file.extension() == ".xml" ? cabana::convertLegacyLayout(file.string()) :
      std::string(std::istreambuf_iterator<char>(input), {});
    loadAnalysisLayout(cabana::parseAnalysisLayout(contents));
  } catch (const std::exception &error) { can->error(error.what()); }
}

void ChartsWidget::loadAnalysisLayout(const cabana::AnalysisLayout &layout) {
  // Validate CAN identities before altering the current workspace.
  for (const auto &tab : layout.tabs) for (const auto &chart : tab.charts) for (const auto &curve : chart.curves) {
    if (curve.can_id.empty()) continue;
    auto id = MessageId::parse(curve.can_id);
    if (!id || !dbc()->msg(*id) || !dbc()->msg(*id)->sig(curve.name)) {
      throw std::invalid_argument("Load the DBC containing CAN signal: " + curve.can_id + "|" + curve.name);
    }
  }
  changing_layout_ = true;
  ++*cereal_generation_;
  removeAll();
  derived_definitions_ = layout.calculations;
  for (const auto &tab : layout.tabs) for (const auto &chart : tab.charts) for (const auto &curve : chart.curves) {
    if (curve.python) {
      auto name = curve.name;
      if (!name.empty() && name.front() == '/') name.erase(0, 1);
      derived_definitions_.emplace(name, *curve.python);
    }
  }
  for (size_t i = 0; i < layout.tabs.size(); ++i) {
    if (i) newTab();
    const auto &tab = layout.tabs[i];
    tab_names_[tabbar_.tabData(tabbar_.currentIndex())] = tab.name;
    for (auto it = tab.charts.rbegin(); it != tab.charts.rend(); ++it) createChart()->applyLayout(*it);
    if (tab.root) tab_roots_[tabbar_.tabData(tabbar_.currentIndex())] = *tab.root;
  }
  tabbar_.setCurrentIndex(layout.current_tab);
  changing_layout_ = false;
  updateTabBar();
  updateState();
  requestCerealUpdate();
}

void ChartsWidget::editPythonSeries(ChartView *owner, const std::string &name, const cabana::PythonSeries &spec) {
  python_owner_ = owner;
  python_name_ = name;
  python_spec_ = spec;
  if (python_spec_.linked_source.empty()) python_spec_.linked_source = "carState/vEgo";
  if (python_spec_.function_code.empty()) python_spec_.function_code = "return value";
  python_inputs_.clear();
  for (const auto &path : spec.additional_sources) python_inputs_ += path + "\n";
  python_error_.clear();
  python_editor_open_ = true;
}

void ChartsWidget::drawPythonEditor() {
  if (!python_editor_open_) return;
  ImGui::SetNextWindowSize(ImVec2(700, 520), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("Derived signal###CabanaPythonEditor", &python_editor_open_)) {
    inputText("Name", &python_name_);
    inputText("Source", &python_spec_.linked_source, "carState/vEgo");
    ImGui::TextUnformatted("Additional sources (one path per line, available as v1, v2, ...)");
    inputTextMultiline("##additional", &python_inputs_, ImVec2(-1, 70));
    ImGui::TextUnformatted("Initialization (runs once for each evaluation)");
    inputTextMultiline("##globals", &python_spec_.globals_code, ImVec2(-1, 70));
    const char *languages[] = {"Python / NumPy", "Lua (legacy XML)"};
    int language = python_spec_.language == "lua" ? 1 : 0;
    if (ImGui::Combo("Language", &language, languages, 2)) python_spec_.language = language ? "lua" : "python";
    ImGui::TextUnformatted("Expression or function body, e.g. return value - v1");
    inputTextMultiline("##code", &python_spec_.function_code, ImVec2(-1, 160));
    if (ImGui::Button("Plot")) {
      if (python_name_.empty() || python_spec_.linked_source.empty() || python_spec_.function_code.empty()) {
        python_error_ = "Name, source, and function are required.";
      } else {
        if (python_spec_.linked_source.front() == '/') python_spec_.linked_source.erase(0, 1);
        python_spec_.additional_sources.clear();
        std::istringstream sources(python_inputs_);
        std::string path;
        while (std::getline(sources, path)) {
          const auto first = path.find_first_not_of(" \t\r");
          if (first == std::string::npos) continue;
          path = path.substr(first, path.find_last_not_of(" \t\r") - first + 1);
          if (path.front() == '/') path.erase(0, 1);
          python_spec_.additional_sources.push_back(path);
        }
        derived_definitions_[python_name_] = python_spec_;
        auto *chart = python_owner_ ? python_owner_ : createChart();
        chart->addPythonSignal(python_name_, python_spec_);
        python_editor_open_ = false;
      }
    }
    if (!python_error_.empty()) ImGui::TextWrapped("%s", python_error_.c_str());
  }
  ImGui::End();
}

void ChartsWidget::remapLayoutTree(int id, const std::vector<ChartView *> &before) {
  auto root = tab_roots_.find(id);
  if (root == tab_roots_.end()) return;
  const auto &after = tab_charts_[id];
  std::set<int> used;
  const auto remap = [&](auto &&self, cabana::LayoutNode node) -> std::optional<cabana::LayoutNode> {
    if (node.chart >= 0) {
      if (node.chart >= static_cast<int>(before.size())) return std::nullopt;
      const auto found = std::find(after.begin(), after.end(), before[node.chart]);
      if (found == after.end()) return std::nullopt;
      node.chart = found - after.begin(); used.insert(node.chart); return node;
    }
    std::vector<cabana::LayoutNode> children;
    std::vector<double> sizes;
    for (size_t i = 0; i < node.children.size(); ++i) {
      if (auto child = self(self, node.children[i])) { children.push_back(std::move(*child)); sizes.push_back(node.sizes[i]); }
    }
    if (children.empty()) return std::nullopt;
    if (children.size() == 1) return children.front();
    node.children = std::move(children); node.sizes = std::move(sizes); return node;
  };
  auto updated = remap(remap, root->second);
  for (size_t i = 0; i < after.size(); ++i) {
    if (used.count(i)) continue;
    cabana::LayoutNode leaf; leaf.chart = i;
    if (!updated) updated = leaf;
    else {
      cabana::LayoutNode split;
      split.children = {leaf, *updated}; split.sizes = {1, static_cast<double>(std::max<size_t>(used.size(), 1))};
      updated = std::move(split);
    }
    used.insert(i);
  }
  if (updated) root->second = std::move(*updated);
  else tab_roots_.erase(root);
}

void ChartsWidget::requestMapUpdate(bool data_changed) {
  if (data_changed && map_running_) { map_dirty_ = true; return; }
  const auto generation = ++*map_generation_;
  map_running_ = map_dirty_ = false;
  if (std::none_of(charts_.begin(), charts_.end(), [](const auto &chart) { return chart->isMap(); })) return;
  auto reader = can->cerealHistoryReader(true);
  if (!reader) return;
  map_running_ = true;
  auto state = map_generation_;
  auto publish = utils::guarded(alive_, [this, state, generation](const std::map<std::string, cabana::CerealPoints> &result,
                                                               const std::string &error) {
    if (state->load() != generation) return;
    for (auto &chart : charts_) if (chart->isMap()) {
      chart->setMapError(error);
      if (error.empty()) chart->setMapSeries(result);
    }
    map_running_ = false;
    if (map_dirty_) requestMapUpdate(true);
  });
  ThreadPool::history().run([reader, publish, state, generation]() {
    auto cancelled = [state, generation]() { return state->load() != generation; };
    std::set<std::string> paths;
    for (const std::string service : {"gpsLocation", "gpsLocationExternal"}) {
      for (const std::string field : {"latitude", "longitude", "bearingDeg", "hasFix"}) paths.insert(service + "/" + field);
    }
    paths.insert("liveLocationKalman/positionGeodetic/value/0");
    paths.insert("liveLocationKalman/positionGeodetic/value/1");
    paths.insert("liveLocationKalman/positionGeodetic/valid");
    paths.insert("selfdriveState/enabled");
    std::map<std::string, cabana::CerealPoints> result;
    std::string error;
    try { result = reader(paths, cancelled); }
    catch (const std::exception &failure) { error = failure.what(); }
    if (!cancelled()) utils::runOnMainThread([publish, result = std::move(result), error]() { publish(result, error); });
  });
}
