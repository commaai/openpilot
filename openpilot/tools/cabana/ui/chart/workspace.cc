#include "tools/cabana/ui/chart/chartswidget.h"

#include <cmath>
#include "tools/cabana/ui/threadpool.h"
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

#include "json11/json11.hpp"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/chart/layout.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/utils/strings.h"

using json11::Json;

void ChartsWidget::fitTimeRange() {
  double min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::lowest();
  for (auto *c : currentCharts()) for (const auto &s : c->signals()) {
    if (!s.visible || s.vals.empty()) continue;
    min = std::min(min, s.vals.front().x);
    max = std::max(max, s.vals.back().x);
  }
  if (max > min) zoom_undo_stack_.push(new ZoomCommand({min, max}));
}

std::string ChartsWidget::serializeLayout() const {
  Json::array tabs, names, equations;
  for (int i = 0; i < tabbar_.count(); ++i) {
    const int id = tabbar_.tabData(i);
    names.push_back(tab_names_.count(id) ? tab_names_.at(id) : "Tab " + std::to_string(i + 1));
    Json::array charts;
    auto tab = tab_charts_.find(id);
    if (tab != tab_charts_.end()) for (auto *c : tab->second) {
      Json::array signals;
      for (const auto &s : c->signals()) {
        char color[8];
        snprintf(color, sizeof(color), "#%02x%02x%02x", s.color.r, s.color.g, s.color.b);
        Json::object signal{{"signal", s.name()}, {"color", color},
          {"visible", s.visible}, {"transform", (int)s.transform.type}, {"scale", s.transform.scale},
          {"offset", s.transform.offset}, {"window", s.transform.window}};
        if (s.path.empty()) signal["message"] = s.msg_id.toString();
        else signal["path"] = s.path;
        signals.push_back(signal);
      }
      Json::object chart{{"type", (int)c->seriesType()}, {"title", c->title}, {"signals", signals}};
      if (c->limit_min) chart["y_min"] = *c->limit_min;
      if (c->limit_max) chart["y_max"] = *c->limit_max;
      charts.push_back(chart);
    }
    tabs.push_back(charts);
  }
  for (const auto &e : equations_) {
    Json::array additional;
    for (const auto &s : e.additional) additional.push_back(s);
    equations.push_back(Json::object{{"name", e.name}, {"language", "python"}, {"source", e.source}, {"globals", e.globals},
                                   {"function", e.function}, {"additional", additional}});
  }
  return Json(Json::object{{"cabana_layout", 3}, {"columns", column_count_},
    {"range", max_chart_range_}, {"tabs", tabs}, {"tab_names", names}, {"equations", equations}}).dump();
}

void ChartsWidget::saveLayout() {
  FileDialog::getSaveFileName("Save Chart Layout", settings.last_dir + "/charts.json", ".json",
    [contents = serializeLayout()](const std::string &path) {
      if (path.empty()) return;
      std::ofstream out(path);
      out << contents << '\n';
      out.close();
      if (!out) MessageBox::warning("Save Layout", "Could not write the chart layout.");
    });
}

void ChartsWidget::loadLayout() {
  FileDialog::getOpenFileName("Open Chart Layout", settings.last_dir, ".json",
    [this](const std::string &path) { if (!path.empty()) openLayout(path); });
}

bool ChartsWidget::openLayout(const std::string &name, bool defer_missing_can) {
  auto path = std::filesystem::path(name);
  if (!std::filesystem::exists(path) && (path.parent_path().empty() || path.parent_path() == "layouts")) {
    path = executableDir() / "layouts" / (path.has_extension() ? path.filename().string() : path.filename().string() + ".json");
  }
  try {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not read the chart layout");
    const std::string contents{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    return restoreLayout(contents, defer_missing_can);
  } catch (const std::exception &e) {
    MessageBox::warning("Open Layout", e.what());
    return false;
  }
}

bool ChartsWidget::restoreLayout(const std::string &contents, bool defer_missing_can) {
  auto layout = chart::parseLayout(contents);
  if (!layout) { MessageBox::warning("Open Layout", "This is not a supported Cabana chart layout."); return false; }
  // Resolve CAN definitions before replacing charts. Cereal paths may arrive in later segments.
  for (const auto &tab : layout->tabs) for (const auto &chart : tab) for (const auto &s : chart.signals) {
    if (!s.path.empty()) continue;
    auto *msg = dbc()->msg(s.id);
    if (!msg || !msg->sig(s.name)) {
      if (!defer_missing_can) MessageBox::warning("Open Layout", "Load the matching DBC first. Missing " + s.id.toString() + " / " + s.name);
      return false;
    }
  }
  removeAll();
  equations_ = layout->equations;
  rebuildSignalBrowser();
  if (!equations_.empty() || std::any_of(layout->tabs.begin(), layout->tabs.end(), [](const auto &tab) {
    return std::any_of(tab.begin(), tab.end(), [](const auto &chart) {
      return std::any_of(chart.signals.begin(), chart.signals.end(), [](const auto &s) { return !s.path.empty(); });
    });
  })) analysisRequested();
  for (size_t i = 0; i < layout->tabs.size(); ++i) {
    if (i) newTab();
    if (i < layout->tab_names.size()) tab_names_[tabbar_.tabData(tabbar_.currentIndex())] = layout->tab_names[i];
    for (const auto &saved : layout->tabs[i]) {
      auto *c = createChart(currentCharts().size());
      c->title = saved.title == "..." ? "" : saved.title;
      c->limit_min = saved.y_min;
      c->limit_max = saved.y_max;
      c->setSeriesType((SeriesType)saved.type);
      for (const auto &s : saved.signals) {
        const size_t count = c->signals().size();
        if (s.path.empty()) c->addSignal(s.id, dbc()->msg(s.id)->sig(s.name));
        else c->addTelemetry(s.path, s.color);
        if (c->signals().size() != count) c->configureSignal(count, s.transform, s.visible, s.color);
      }
    }
  }
  tabbar_.setCurrentIndex(0);
  setColumnCount(layout->columns);
  setMaxChartRange(std::min(layout->range, range_slider_.maximum()));
  range_slider_.setValue(max_chart_range_);
  telemetryChanged();
  updateTabBar();
  updateState();
  return true;
}

const std::vector<cabana::Sample> *ChartsWidget::telemetrySeries(const std::string &path) const {
  return telemetrySnapshot(path).get();
}

std::shared_ptr<const cabana::Samples> ChartsWidget::telemetrySnapshot(const std::string &path) const {
  auto derived = calculated_.find(path);
  if (derived != calculated_.end()) return derived->second;
  auto raw = can->telemetry.find(path);
  return raw == can->telemetry.end() ? nullptr : raw->second;
}

void ChartsWidget::telemetryChanged() {
  telemetry_dirty_ = true;
  if (browser_paths_.empty() || browser_telemetry_count_ != can->telemetry.size()) {
    rebuildSignalBrowser();
  }
  pollTelemetry();
}

void ChartsWidget::rebuildSignalBrowser() {
  browser_telemetry_count_ = can->telemetry.size();
  browser_paths_.clear();
  for (const auto &[path, _] : can->telemetry) browser_paths_.push_back(path);
  for (const auto &e : equations_) browser_paths_.push_back(e.name);
  std::sort(browser_paths_.begin(), browser_paths_.end());
  browser_paths_.erase(std::unique(browser_paths_.begin(), browser_paths_.end()), browser_paths_.end());
  browser_tree_.rebuild(browser_paths_);
  browser_tree_dirty_ = true;
}

void ChartsWidget::pollTelemetry() {
  if (equation_task_.valid()) {
    if (equation_task_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    equation_task_.get();
    if (equation_result_->revision == equation_revision_) {
      calculated_.swap(equation_result_->values);
      equation_errors_ = std::move(equation_result_->errors);
      for (auto &c : charts_) c->updateTelemetry();
      updateState();
    }
    ThreadPool::instance().run([retired = std::move(equation_result_)]() mutable { retired.reset(); });
  }
  if (!telemetry_dirty_) return;
  telemetry_dirty_ = false;
  // Retain immutable inputs without copying samples on the UI thread.
  cabana::TelemetrySnapshot snapshot;
  for (const auto &e : equations_) {
    auto add = [&](const std::string &path) {
      auto it = can->telemetry.find(path);
      if (it != can->telemetry.end() && !snapshot.count(path)) snapshot.emplace(path, it->second);
    };
    add(e.source);
    for (const auto &path : e.additional) add(path);
  }
  equation_result_ = std::make_shared<EquationResult>();
  equation_result_->revision = equation_revision_;
  equation_task_ = ThreadPool::instance().run([equations = equations_, snapshot = std::move(snapshot), result = equation_result_]() mutable {
    cabana::Telemetry inputs;
    for (const auto &[path, samples] : snapshot) inputs.emplace(path, *samples);
    std::vector<const cabana::Equation *> pending;
    for (const auto &e : equations) pending.push_back(&e);
    for (size_t pass = 0; pass < equations.size() && !pending.empty(); ++pass) {
      for (auto it = pending.begin(); it != pending.end();) {
        const auto &e = **it;
        auto available = [&](const std::string &path) { auto p = inputs.find(path); return p != inputs.end() && !p->second.empty(); };
        if (!available(e.source) || !std::all_of(e.additional.begin(), e.additional.end(), available)) { ++it; continue; }
        try { inputs[e.name] = cabana::evaluateEquation(e, inputs); }
        catch (const std::exception &error) { result->errors += e.name + ": " + error.what() + "\n"; }
        it = pending.erase(it);
      }
    }
    for (auto *e : pending) result->errors += e->name + ": waiting for input signals (or cyclic dependency)\n";
    for (const auto &e : equations) {
      auto it = inputs.find(e.name);
      if (it != inputs.end()) result->values.emplace(e.name, std::make_shared<const cabana::Samples>(std::move(it->second)));
    }
  });
}

void ChartsWidget::exportCsv() {
  // Snapshot the visible tab/range now, so playback or later edits cannot change the export.
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "chart,message,signal,transform,scale,offset,window,time,value\n" << std::setprecision(17);
  const auto range = can->timeRange().value_or(display_range_);
  size_t rows = 0;
  int index = 0;
  for (auto *c : currentCharts()) {
    ++index;
    for (const auto &s : c->signals()) {
      if (!s.visible) continue;
      const auto prefix = std::to_string(index) + ',' + chart::csvField(s.path.empty() ? s.msg_id.toString() : "cereal") + ',' +
        chart::csvField(s.name()) + ',' + chart::csvField(chart::TRANSFORM_NAMES[(int)s.transform.type]) + ',';
      auto first = std::lower_bound(s.vals.begin(), s.vals.end(), range.first, [](const auto &p, double t) { return p.x < t; });
      for (auto it = first; it != s.vals.end() && it->x < range.second; ++it) {
        out << prefix << s.transform.scale << ',' << s.transform.offset << ',' << s.transform.window << ',' << it->x << ',' << it->y << '\n';
        ++rows;
      }
    }
  }
  if (!rows) { MessageBox::information("Export CSV", "There are no visible samples in this time range."); return; }
  FileDialog::getSaveFileName("Export Visible Chart Data", settings.last_dir + "/charts.csv", ".csv",
    [contents = out.str()](const std::string &path) {
      if (path.empty()) return;
      std::ofstream file(path);
      file << contents;
      file.close();
      if (!file) MessageBox::warning("Export CSV", "Could not write the chart data.");
    });
}

void ChartsWidget::drawSignalBrowser() {
  const bool filter_changed = inputText("##search_telemetry", &browser_filter_, "Search route signals...");
  if (filter_changed || browser_tree_dirty_) {
    browser_tree_.filter(browser_filter_);
    browser_search_expanded_.clear();
    if (!browser_filter_.empty()) {
      for (const auto &node : browser_tree_.nodes) {
        if (node.matches && !node.children.empty()) browser_search_expanded_.insert(node.key);
      }
    }
    browser_tree_dirty_ = false;
  }
  auto &expanded = browser_filter_.empty() ? browser_expanded_ : browser_search_expanded_;
  ImGui::TextDisabled("Double-click to plot · Drag onto a chart to compare");
  ImGui::AlignTextToFramePadding();
  ImGui::Text("%zu signals", browser_tree_.nodes[0].matches);
  alignRight(iconButtonWidth() * 2 + ImGui::GetStyle().ItemInnerSpacing.x);
  if (iconButton("expand_signals", icon::PLUS_LG, "Expand all")) {
    for (const auto &node : browser_tree_.nodes) {
      if (node.matches && !node.children.empty()) expanded.insert(node.key);
    }
  }
  ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
  if (iconButton("collapse_signals", icon::ARROWS_COLLAPSE, "Collapse all")) expanded.clear();
  if (browser_paths_.empty()) ImGui::TextWrapped("Open a route or start a cereal stream to browse its numeric signals.");
  else if (!browser_tree_.nodes[0].matches) ImGui::TextDisabled("No signals match your search.");
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, ImGui::GetStyle().WindowPadding.y));
  const bool browser_visible = ImGui::BeginChild("signal_browser_list", ImVec2(0, 0), ImGuiChildFlags_AlwaysUseWindowPadding,
                                                ImGuiWindowFlags_HorizontalScrollbar);
  ImGui::PopStyleVar();
  if (browser_visible) {
    if (filter_changed) ImGui::SetScrollY(0);
    const auto rows = browser_tree_.visible(expanded);
    ImGuiListClipper clipper;
    clipper.Begin(rows.size(), ImGui::GetTextLineHeightWithSpacing());
    while (clipper.Step()) for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      const auto &node = browser_tree_.nodes[rows[i]];
      const bool branch = !node.children.empty();
      const std::string label = chart::SignalTree::isIndex(node.name) ? browser_tree_.nodes[node.parent].name + "/" + node.name : node.name;
      ImGui::PushID(node.key.c_str());
      const float indent = node.depth * ImGui::GetStyle().IndentSpacing * 0.5f;
      if (indent > 0) ImGui::Indent(indent);
      ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
      if (!branch) flags |= ImGuiTreeNodeFlags_Leaf;
      ImGui::SetNextItemOpen(branch && expanded.count(node.key), ImGuiCond_Always);
      const bool open = ImGui::TreeNodeEx("node", flags, "%s", label.c_str());
      if (branch && ImGui::IsItemToggledOpen()) {
        if (open) expanded.insert(node.key);
        else expanded.erase(node.key);
      }
      if (node.signal_matches) {
        const auto &path = node.path;
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0) && !ImGui::IsItemToggledOpen()) {
          auto *c = createChart();
          c->addTelemetry(path);
          updateState();
        }
        if (ImGui::IsItemHovered()) {
          const auto *points = telemetrySeries(path);
          const double time = can->beginMonoTime() * 1e-9 + can->currentSec();
          if (points && !points->empty()) ImGui::SetTooltip("%s\nValue: %.8g", path.c_str(), cabana::nearestValue(*points, time));
          else ImGui::SetTooltip("%s", path.c_str());
        }
        if (ImGui::BeginDragDropSource()) {
          ImGui::SetDragDropPayload("CABANA_TELEMETRY", path.c_str(), path.size() + 1);
          ImGui::TextUnformatted(path.c_str());
          ImGui::EndDragDropSource();
        }
      } else if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s\n%zu signals", node.key.c_str(), node.matches);
      }
      if (indent > 0) ImGui::Unindent(indent);
      ImGui::PopID();
    }
  }
  ImGui::EndChild();
}
