#include "tools/cabana/ui/chart/chartswidget.h"

#include <cmath>
#include "tools/cabana/ui/threadpool.h"
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>

#include "common/util.h"
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
        Json::object signal{{"color", s.color.toHex()},
          {"visible", s.visible}, {"transform", (int)s.transform.type}, {"scale", s.transform.scale},
          {"offset", s.transform.offset}, {"window", s.transform.window}};
        for (const auto &[key, value] : chart::SIGNAL_DEFAULTS) if (signal.at(key) == value) signal.erase(key);
        if (s.path.empty()) { signal["message"] = s.msg_id.toString(); signal["signal"] = s.name(); }
        else signal["path"] = s.path;
        signals.push_back(signal);
      }
      Json::object chart{{"type", (int)c->seriesType()}, {"signals", signals}};
      if (!c->title.empty()) chart["title"] = c->title;
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

static bool writeFile(const std::string &path, const std::string &contents) {
  std::ofstream out(path);
  out << contents;
  out.close();
  return bool(out);
}

void ChartsWidget::saveLayout() {
  FileDialog::getSaveFileName("Save Chart Layout", settings.last_dir + "/charts.json", ".json",
    [contents = serializeLayout() + '\n'](const std::string &path) {
      if (!path.empty() && !writeFile(path, contents)) MessageBox::warning("Save Layout", "Could not write the chart layout.");
    });
}

void ChartsWidget::loadLayout() {
  FileDialog::getOpenFileName("Open Chart Layout", settings.last_dir, ".json",
    [this](const std::string &path) { if (!path.empty()) openLayout(path); });
}

ChartsWidget::LayoutStatus ChartsWidget::openLayout(const std::string &path, bool defer_missing_can) {
  const std::string contents = util::read_file(path);
  if (contents.empty()) {
    MessageBox::warning("Open Layout", "Could not read the chart layout");
    return LayoutStatus::Failed;
  }
  return restoreLayout(contents, defer_missing_can);
}

ChartsWidget::LayoutStatus ChartsWidget::restoreLayout(const std::string &contents, bool defer_missing_can) {
  auto layout = chart::parseLayout(contents);
  if (!layout) { MessageBox::warning("Open Layout", "This is not a supported Cabana chart layout."); return LayoutStatus::Failed; }
  // Resolve CAN definitions before replacing charts. Cereal paths may arrive in later segments.
  for (const auto &tab : layout->tabs) for (const auto &chart : tab) for (const auto &s : chart.signals) {
    if (!s.path.empty()) continue;
    auto *msg = dbc()->msg(s.id);
    if (!msg || !msg->sig(s.name)) {
      if (!defer_missing_can) MessageBox::warning("Open Layout", "Load the matching DBC first. Missing " + s.id.toString() + " / " + s.name);
      return defer_missing_can ? LayoutStatus::MissingCan : LayoutStatus::Failed;
    }
  }
  removeAll();
  equations_ = layout->equations;
  rebuildSignalBrowser();
  if (!equations_.empty()) analysisRequested();
  for (size_t i = 0; i < layout->tabs.size(); ++i) {
    if (i) newTab();
    if (i < layout->tab_names.size()) tab_names_[tabbar_.tabData(tabbar_.currentIndex())] = layout->tab_names[i];
    for (const auto &saved : layout->tabs[i]) {
      auto *c = createChart(currentCharts().size());
      c->title = saved.title;
      c->limit_min = saved.y_min;
      c->limit_max = saved.y_max;
      c->setSeriesType((SeriesType)saved.type);
      for (const auto &s : saved.signals) {
        const size_t count = c->signals().size();
        if (s.path.empty()) c->addSignal(s.id, dbc()->msg(s.id)->sig(s.name));
        else c->addFields(s.path, s.color);
        if (c->signals().size() != count) c->configureSignal(count, s.transform, s.visible, s.color);
      }
    }
  }
  tabbar_.setCurrentIndex(0);
  setColumnCount(layout->columns);
  setMaxChartRange(std::min(layout->range, range_slider_.maximum()));
  range_slider_.setValue(max_chart_range_);
  fieldsChanged();
  updateTabBar();
  updateState();
  return LayoutStatus::Restored;
}

std::shared_ptr<const cabana::Samples> ChartsWidget::fieldsSnapshot(const std::string &path) const {
  auto derived = calculated_.find(path);
  if (derived != calculated_.end()) return derived->second;
  auto raw = can->fields.find(path);
  return raw == can->fields.end() ? nullptr : raw->second;
}

void ChartsWidget::fieldsChanged() {
  fields_dirty_ = true;
  if (browser_field_count_ != can->fields.size()) {
    rebuildSignalBrowser();
  }
  pollFields();
}

void ChartsWidget::rebuildSignalBrowser() {
  browser_field_count_ = can->fields.size();
  std::vector<std::string> paths;
  for (const auto &[path, _] : can->fields) paths.push_back(path);
  std::unordered_set<std::string> custom_paths;
  for (const auto &e : equations_) {
    paths.push_back(e.name);
    custom_paths.insert(e.name);
  }
  browser_tree_.rebuild(paths, custom_paths);
  browser_tree_dirty_ = true;
}

void ChartsWidget::pollFields() {
  if (equation_task_.valid()) {
    if (equation_task_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    equation_task_.get();
    if (equation_result_->revision == equation_revision_) {
      calculated_.swap(equation_result_->values);
      equation_errors_ = std::move(equation_result_->errors);
      for (auto &c : charts_) c->updateFields();
      updateState();
    }
    ThreadPool::instance().run([retired = std::move(equation_result_)]() mutable { retired.reset(); });
  }
  if (!fields_dirty_) return;
  fields_dirty_ = false;
  // Retain immutable inputs without copying samples on the UI thread.
  cabana::FieldsSnapshot snapshot;
  for (const auto &e : equations_) {
    auto add = [&](const std::string &path) {
      auto it = can->fields.find(path);
      if (it != can->fields.end() && !snapshot.count(path)) snapshot.emplace(path, it->second);
    };
    add(e.source);
    for (const auto &path : e.additional) add(path);
  }
  equation_result_ = std::make_shared<EquationResult>();
  equation_result_->revision = equation_revision_;
  equation_task_ = ThreadPool::instance().run([equations = equations_, snapshot = std::move(snapshot), result = equation_result_]() mutable {
    std::vector<const cabana::Equation *> pending;
    for (const auto &e : equations) pending.push_back(&e);
    for (size_t pass = 0; pass < equations.size() && !pending.empty(); ++pass) {
      for (auto it = pending.begin(); it != pending.end();) {
        const auto &e = **it;
        auto available = [&](const std::string &path) { auto p = snapshot.find(path); return p != snapshot.end() && !p->second->empty(); };
        if (!available(e.source) || !std::all_of(e.additional.begin(), e.additional.end(), available)) { ++it; continue; }
        try { snapshot[e.name] = std::make_shared<const cabana::Samples>(cabana::evaluateEquation(e, snapshot)); }
        catch (const std::exception &error) { result->errors += e.name + ": " + error.what() + "\n"; }
        it = pending.erase(it);
      }
    }
    for (auto *e : pending) result->errors += e->name + ": waiting for input signals (or cyclic dependency)\n";
    for (const auto &e : equations) {
      auto it = snapshot.find(e.name);
      if (it != snapshot.end()) result->values.emplace(e.name, it->second);
    }
  });
}

void ChartsWidget::exportCsv() {
  // Snapshot the visible tab/range now, so playback or later edits cannot change the export.
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "chart,source,name,transform,scale,offset,window,time,value\n" << std::setprecision(17);
  const auto range = can->timeRange().value_or(display_range_);
  size_t rows = 0;
  int index = 0;
  for (auto *c : currentCharts()) {
    ++index;
    for (const auto &s : c->signals()) {
      if (!s.visible) continue;
      const auto prefix = std::to_string(index) + ',' + chart::csvField(s.path.empty() ? s.msg_id.toString() : "openpilot") + ',' +
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
      if (!path.empty() && !writeFile(path, contents)) MessageBox::warning("Export CSV", "Could not write the chart data.");
    });
}

void ChartsWidget::drawSignalBrowser() {
  ImGui::SetNextItemWidth(-1.0f);
  const bool filter_changed = inputText("##search_fields", &browser_filter_, "Search openpilot messages...");
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
  ImGui::PushTextWrapPos(0.0f);
  ImGui::TextDisabled("Double-click to plot · Drag onto a chart to compare");
  ImGui::PopTextWrapPos();
  ImGui::AlignTextToFramePadding();
  ImGui::Text("%zu fields", browser_tree_.nodes[0].matches);
  alignRight(iconButtonWidth() * 2 + ImGui::GetStyle().ItemInnerSpacing.x);
  if (iconButton("expand_signals", icon::PLUS_LG, "Expand all")) {
    for (const auto &node : browser_tree_.nodes) {
      if (node.matches && !node.children.empty()) expanded.insert(node.key);
    }
  }
  ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
  if (iconButton("collapse_signals", icon::ARROWS_COLLAPSE, "Collapse all")) expanded.clear();
  if (browser_tree_.nodes[0].children.empty()) ImGui::TextWrapped("Open a route or start a stream to browse openpilot messages.");
  else if (!browser_tree_.nodes[0].matches) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("No fields match your search.");
    ImGui::PopTextWrapPos();
  }
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
          c->addFields(path);
          updateState();
        }
        if (ImGui::IsItemHovered()) {
          const auto points = fieldsSnapshot(path);
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
        ImGui::SetTooltip("%s\n%zu fields", node.key.c_str(), node.matches);
      }
      if (indent > 0) ImGui::Unindent(indent);
      ImGui::PopID();
    }
  }
  ImGui::EndChild();
}
