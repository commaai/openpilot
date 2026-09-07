#include "tools/cabana/ui/chart/chartswidget.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <cerrno>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

#include "json11/json11.hpp"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/chart/layout.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/util.h"
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
    equations.push_back(Json::object{{"name", e.name}, {"source", e.source}, {"globals", e.globals},
                                   {"function", e.function}, {"additional", additional}});
  }
  return Json(Json::object{{"cabana_layout", 2}, {"columns", column_count_},
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
  FileDialog::getOpenFileName("Open Cabana or PlotJuggler Layout", settings.last_dir, "",
    [this](const std::string &path) { if (!path.empty()) openLayout(path); });
}

bool ChartsWidget::openLayout(const std::string &name) {
  auto path = std::filesystem::path(name);
  if (!std::filesystem::exists(path) && path.parent_path() == "layouts") path = executableDir() / "../plotjuggler" / path;
  if (!std::filesystem::exists(path) && path.parent_path().empty()) {
    path = executableDir() / "../plotjuggler/layouts" / (path.extension() == ".xml" ? name : name + ".xml");
  }
  try {
    std::string contents;
    if (path.extension() == ".xml") {
      // No shell: layout paths are data, including spaces and metacharacters.
      const std::string script = (executableDir() / "analysis/import_layout.py").string();
      int pipes[2];
      if (pipe(pipes) != 0) throw std::runtime_error("Could not start the layout importer");
      pid_t pid = fork();
      if (pid == 0) {
        dup2(pipes[1], STDOUT_FILENO);
        dup2(pipes[1], STDERR_FILENO);
        close(pipes[0]); close(pipes[1]);
        execlp("python3", "python3", script.c_str(), path.c_str(), static_cast<char *>(nullptr));
        _exit(127);
      }
      close(pipes[1]);
      if (pid < 0) { close(pipes[0]); throw std::runtime_error("Could not start the layout importer"); }
      char buffer[4096];
      ssize_t count;
      while ((count = read(pipes[0], buffer, sizeof(buffer))) > 0) contents.append(buffer, count);
      close(pipes[0]);
      int status = 0;
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
      if (!WIFEXITED(status) || WEXITSTATUS(status)) throw std::runtime_error(contents.empty() ? "Layout import failed" : contents);
    } else {
      std::ifstream in(path);
      if (!in) throw std::runtime_error("Could not read the chart layout");
      contents.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    return restoreLayout(contents);
  } catch (const std::exception &e) {
    MessageBox::warning("Open Layout", e.what());
    return false;
  }
}

bool ChartsWidget::restoreLayout(const std::string &contents) {
  auto layout = chart::parseLayout(contents);
  if (!layout) { MessageBox::warning("Open Layout", "This is not a supported Cabana chart layout."); return false; }
  // Resolve CAN definitions before replacing charts. Cereal paths may arrive in later segments.
  for (const auto &tab : layout->tabs) for (const auto &chart : tab) for (const auto &s : chart.signals) {
    if (!s.path.empty()) continue;
    auto *msg = dbc()->msg(s.id);
    if (!msg || !msg->sig(s.name)) {
      MessageBox::warning("Open Layout", "Load the matching DBC first. Missing " + s.id.toString() + " / " + s.name);
      return false;
    }
  }
  removeAll();
  equations_ = layout->equations;
  if (!equations_.empty() || std::any_of(layout->tabs.begin(), layout->tabs.end(), [](const auto &tab) {
    return std::any_of(tab.begin(), tab.end(), [](const auto &chart) {
      return std::any_of(chart.signals.begin(), chart.signals.end(), [](const auto &s) { return !s.path.empty(); });
    });
  })) analysisRequested();
  for (size_t i = 0; i < layout->tabs.size(); ++i) {
    if (i) newTab();
    if (i < layout->tab_names.size()) tab_names_[tabbar_.tabData(tabbar_.currentIndex())] = layout->tab_names[i];
    for (const auto &saved : layout->tabs[i]) {
      if (saved.signals.empty()) continue;
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
  auto derived = calculated_.find(path);
  if (derived != calculated_.end()) return &derived->second;
  auto raw = can->telemetry.find(path);
  return raw == can->telemetry.end() ? nullptr : &raw->second;
}

void ChartsWidget::telemetryChanged() {
  calculated_.clear();
  equation_errors_.clear();
  std::vector<const cabana::Equation *> pending;
  for (const auto &e : equations_) pending.push_back(&e);
  // Resolve named-equation dependencies independently of their order in the XML file.
  for (size_t pass = 0; pass < equations_.size() && !pending.empty(); ++pass) {
    for (auto it = pending.begin(); it != pending.end();) {
      const auto &e = **it;
      cabana::Telemetry inputs;
      bool ready = true;
      auto add = [&](const std::string &path) {
        if (auto *samples = telemetrySeries(path); samples && !samples->empty()) inputs[path] = *samples;
        else ready = false;
      };
      add(e.source);
      for (const auto &path : e.additional) add(path);
      if (!ready) { ++it; continue; }
      try { calculated_[e.name] = cabana::evaluateEquation(e, inputs); }
      catch (const std::exception &error) { equation_errors_ += e.name + ": " + error.what() + "\n"; }
      it = pending.erase(it);
    }
  }
  for (auto *e : pending) equation_errors_ += e->name + ": waiting for input signals (or cyclic dependency)\n";
  browser_paths_.clear();
  for (const auto &[path, _] : can->telemetry) browser_paths_.push_back(path);
  for (const auto &[path, _] : calculated_) browser_paths_.push_back(path);
  std::sort(browser_paths_.begin(), browser_paths_.end());
  for (auto &c : charts_) c->updateTelemetry();
  updateState();
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
  inputText("##search_telemetry", &browser_filter_, "Search route signals...");
  ImGui::TextDisabled("Double-click to plot · Drag onto a chart to compare");
  std::vector<const std::string *> matches;
  for (const auto &path : browser_paths_) if (utils::containsCI(path, browser_filter_)) matches.push_back(&path);
  ImGui::Text("%zu signals", matches.size());
  if (matches.empty()) ImGui::TextWrapped("Open a route or start a cereal stream to browse its numeric signals.");
  if (ImGui::BeginChild("signal_browser_list")) {
    ImGuiListClipper clipper;
    clipper.Begin(matches.size());
    while (clipper.Step()) for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      const auto &path = *matches[i];
      ImGui::PushID(path.c_str());
      if (ImGui::Selectable(path.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick) && ImGui::IsMouseDoubleClicked(0)) {
        auto *c = createChart();
        c->addTelemetry(path);
        updateState();
      }
      if (ImGui::IsItemHovered()) {
        const auto *points = telemetrySeries(path);
        const double time = can->beginMonoTime() * 1e-9 + can->currentSec();
        if (points && !points->empty()) ImGui::SetTooltip("%s\nValue: %.8g", path.c_str(), cabana::nearestValue(*points, time));
      }
      if (ImGui::BeginDragDropSource()) {
        ImGui::SetDragDropPayload("CABANA_TELEMETRY", path.c_str(), path.size() + 1);
        ImGui::TextUnformatted(path.c_str());
        ImGui::EndDragDropSource();
      }
      ImGui::PopID();
    }
  }
  ImGui::EndChild();
}
