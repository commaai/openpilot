#include "tools/cabana/ui/chart/logpanel.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <tuple>

#include "imgui.h"
#include "implot.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

void LogPanel::loadLayout(const std::string &path) {
  try {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Unable to read " + path);
    const std::string text((std::istreambuf_iterator<char>(in)), {});
    auto plots = cabana::parseLogLayout(text);
    plots_ = std::move(plots);
    active_plot_ = plots_.empty() ? -1 : 0;
    dirty_ = true;
  } catch (const std::exception &e) {
    MessageBox::warning("Log Layout", e.what());
  }
}

void LogPanel::writeFile(const std::string &path, const std::string &text) {
  std::ofstream out(path);
  out << text;
  out.close();
  if (!out) MessageBox::warning("Log Signals", "Unable to write " + path);
}

void LogPanel::fileActions() {
  if (ImGui::Button("Load layout")) {
    FileDialog::getOpenFileName("Load Log Layout", "", ".json", [this, alive = std::weak_ptr<bool>(alive_)](const auto &path) {
      if (!alive.expired() && !path.empty()) loadLayout(path);
    });
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(plots_.empty());
  if (ImGui::Button("Save layout")) {
    FileDialog::getSaveFileName("Save Log Layout", "log-layout.json", ".json", [this, alive = std::weak_ptr<bool>(alive_)](const auto &path) {
      if (!alive.expired() && !path.empty()) writeFile(path, cabana::serializeLogLayout(plots_));
    });
  }
  ImGui::SameLine();
  if (ImGui::Button("Export visible CSV")) {
    FileDialog::getSaveFileName("Export Log Signals", "log-signals.csv", ".csv", [this, alive = std::weak_ptr<bool>(alive_)](const auto &path) {
      if (!alive.expired() && !path.empty()) {
        writeFile(path, cabana::logCsv(can->logSegments(), plots_, can->beginMonoTime(), range_min_, range_max_));
      }
    });
  }
  ImGui::EndDisabled();
}

void LogPanel::refresh() {
  if (revision_ != can->logRevision()) {
    revision_ = can->logRevision();
    std::set<std::string> names;
    for (const auto &[_, signals] : can->logSegments()) {
      for (const auto &[name, samples] : *signals) names.insert(name);
    }
    names_.assign(names.begin(), names.end());
    dirty_ = true;
  }
  if (!dirty_) return;
  points_.clear();
  line_indices_.clear();
  for (const auto &plot : plots_) {
    auto &curves = points_.emplace_back();
    auto &lines = line_indices_.emplace_back();
    for (const auto &curve : plot) {
      auto &points = curves.emplace_back(cabana::logPoints(can->logSegments(), curve, can->beginMonoTime()));
      auto &indices = lines.emplace_back();
      for (size_t i = 1; i < points.size(); ++i) {
        if (std::isfinite(points[i - 1].y) && std::isfinite(points[i].y)) {
          indices.push_back(i - 1);
          indices.push_back(i);
        }
      }
    }
  }
  dirty_ = false;
}

void LogPanel::draw() {
  if (!can->logSignalsEnabled()) {
    ImGui::TextWrapped("Open a route with --log to plot openpilot log fields alongside CAN data and video.");
    return;
  }
  refresh();
  fileActions();
  ImGui::Checkbox("Follow playback", &follow_);
  ImGui::SameLine();
  if (ImGui::Button(can->isPaused() ? "Play" : "Pause")) can->pause(!can->isPaused());
  ImGui::SameLine();
  ImGui::Text("%.2f s | %zu cached segments", can->currentSec(), can->logSegments().size());
  ImGui::TextDisabled("Drag to pan; wheel to zoom; Shift+click to seek the video. Export uses cached samples only.");

  if (follow_) {
    if (can->timeRange()) {
      std::tie(range_min_, range_max_) = *can->timeRange();
    } else {
      range_min_ = std::max(can->minSeconds(), can->currentSec() - 15.0);
      range_max_ = range_min_ + 30.0;
    }
  }
  ImGui::SetNextItemWidth(-1);
  inputText("##log_filter", &filter_, "Search log fields (e.g. carState/vEgo)");
  if (ImGui::BeginCombo("Add signal", "Select a log field")) {
    for (const auto &name : names_) {
      if (!filter_.empty() && !utils::containsCI(name, filter_)) continue;
      if (ImGui::Selectable(name.c_str())) {
        if (active_plot_ < 0 || active_plot_ >= (int)plots_.size()) {
          plots_.push_back({});
          active_plot_ = plots_.size() - 1;
        }
        auto &plot = plots_[active_plot_];
        if (std::none_of(plot.begin(), plot.end(), [&](const auto &curve) { return curve.signal == name; })) {
          plot.push_back({name});
          dirty_ = true;
        }
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::Button("New plot")) active_plot_ = -1;
  ImGui::SameLine();
  if (active_plot_ < 0) ImGui::TextDisabled("Next signal starts a new plot");
  else ImGui::TextDisabled("Adding to plot %d", active_plot_ + 1);

  refresh();
  if (names_.empty()) ImGui::TextWrapped("Waiting for numeric log fields in the loaded segments.");
  ImGui::BeginChild("log_plots");
  int remove_plot = -1;
  for (size_t i = 0; i < plots_.size(); ++i) {
    ImGui::PushID((int)i);
    if (ImGui::RadioButton(("Plot " + std::to_string(i + 1)).c_str(), active_plot_ == (int)i)) active_plot_ = i;
    ImGui::SameLine();
    if (ImGui::Button("Curves")) ImGui::OpenPopup("curves");
    ImGui::SameLine();
    if (ImGui::Button("Remove plot")) remove_plot = i;
    if (ImGui::BeginPopup("curves")) {
      int remove_curve = -1;
      for (size_t j = 0; j < plots_[i].size(); ++j) {
        auto &curve = plots_[i][j];
        ImGui::PushID((int)j);
        ImGui::TextUnformatted(curve.signal.c_str());
        double scale = curve.scale, offset = curve.offset;
        if (ImGui::InputDouble("Scale", &scale) && std::isfinite(scale)) { curve.scale = scale; dirty_ = true; }
        if (ImGui::InputDouble("Offset", &offset) && std::isfinite(offset)) { curve.offset = offset; dirty_ = true; }
        if (ImGui::Checkbox("Derivative (per second)", &curve.derivative)) dirty_ = true;
        if (ImGui::Button("Remove curve")) remove_curve = j;
        ImGui::Separator();
        ImGui::PopID();
      }
      if (remove_curve >= 0) {
        plots_[i].erase(plots_[i].begin() + remove_curve);
        dirty_ = true;
        if (plots_[i].empty()) remove_plot = i;
      }
      ImGui::EndPopup();
    }
    // A curve edit changes the number/order of cached series before this frame's plot is drawn.
    refresh();
    ImPlot::SetNextAxisLimits(ImAxis_X1, range_min_, range_max_, ImPlotCond_Always);
    if (ImPlot::BeginPlot("##log_plot", ImVec2(-1, 240))) {
      ImPlot::SetupAxes("Time (s)", nullptr, ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
      for (size_t j = 0; j < plots_[i].size(); ++j) {
        auto &points = points_[i][j];
        const auto &curve = plots_[i][j];
        const std::string label = curve.signal + (curve.derivative ? " (d/dt)" : "") + "##" + std::to_string(j);
        // Explicit finite pairs preserve gaps without passing NaNs to the renderer.
        using LineData = std::pair<const std::vector<cabana::LogPoint> *, const std::vector<int> *>;
        auto &indices = line_indices_[i][j];
        LineData data{&points, &indices};
        ImPlot::PlotLineG(label.c_str(), [](int idx, void *user_data) {
          const auto &[line_points, line_indices] = *static_cast<const LineData *>(user_data);
          const auto &p = (*line_points)[(*line_indices)[idx]];
          return ImPlotPoint(p.x, p.y);
        }, &data, indices.size(), ImPlotSpec(ImPlotProp_Flags, ImPlotLineFlags_Segments));
      }
      const double cursor = can->currentSec();
      ImPlot::PlotInfLines("Playback", &cursor, 1);
      if (ImPlot::IsPlotHovered()) {
        if (ImGui::GetIO().KeyShift && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
          can->seekTo(std::clamp(ImPlot::GetPlotMousePos().x, can->minSeconds(), can->maxSeconds()));
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) || ImGui::GetIO().MouseWheel != 0) follow_ = false;
      }
      const auto limits = ImPlot::GetPlotLimits();
      if (!follow_) { range_min_ = limits.X.Min; range_max_ = limits.X.Max; }
      ImPlot::EndPlot();
    }
    for (size_t j = 0; j < plots_[i].size(); ++j) {
      if (points_[i][j].empty()) ImGui::TextDisabled("No cached samples: %s", plots_[i][j].signal.c_str());
    }
    ImGui::PopID();
  }
  if (remove_plot >= 0) {
    plots_.erase(plots_.begin() + remove_plot);
    active_plot_ = plots_.empty() ? -1 : std::min(active_plot_, (int)plots_.size() - 1);
    dirty_ = true;
  }
  ImGui::EndChild();
}
