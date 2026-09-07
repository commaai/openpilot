#include "tools/cabana/ui/chart/chartswidget.h"

#include <algorithm>

#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

void ChartsWidget::openFunctionEditor(const cabana::Equation *equation) {
  function_draft_ = equation ? *equation : cabana::Equation{"", "", "", "return value", {}};
  function_original_name_ = equation ? equation->name : "";
  function_plot_ = !equation;
  function_sources_.clear();
  for (const auto &[path, _] : can->fields) function_sources_.push_back(path);
  for (const auto &e : equations_) {
    if (e.name != function_original_name_) function_sources_.push_back(e.name);
  }
  std::sort(function_sources_.begin(), function_sources_.end());
  function_editor_open_ = true;
  function_editor_show_ = false;
}

void ChartsWidget::drawFunctionEditor() {
  if (!function_editor_open_) return;
  if (!function_editor_show_) {
    ImGui::OpenPopup("Custom Function");
    function_editor_show_ = true;
  }
  setNextDialogWindow(ImVec2(720, 650));
  if (!ImGui::BeginPopupModal("Custom Function", &function_editor_open_, ImGuiWindowFlags_NoSavedSettings)) return;

  auto &e = function_draft_;
  const bool editing = !function_original_name_.empty();
  // Keep the identity stable: charts and other functions refer to this name.
  ImGui::TextUnformatted("Name");
  ImGui::SetNextItemWidth(-1);
  ImGui::BeginDisabled(editing);
  inputText("##name", &e.name, "e.g. speed_mph");
  ImGui::EndDisabled();
  if (editing) ImGui::SetItemTooltip("The name is used by charts and other functions.");

  const float footer_height = ImGui::GetFrameHeightWithSpacing() * 4;
  if (ImGui::BeginChild("function_body", ImVec2(0, -footer_height))) {
    auto signalInput = [&](const char *label, std::string &path) {
      ImGui::PushID(label);
      ImGui::TextUnformatted(label);
      ImGui::SetNextItemWidth(-ImGui::GetFrameHeight() * 3);
      inputText("##path", &path, "Signal path or function name");
      if (ImGui::BeginDragDropTarget()) {
        if (auto *payload = ImGui::AcceptDragDropPayload("CABANA_TELEMETRY")) path = (const char *)payload->Data;
        ImGui::EndDragDropTarget();
      }
      ImGui::SameLine();
      if (ImGui::Button("Browse")) ImGui::OpenPopup("signals");
      if (ImGui::BeginPopup("signals")) {
        if (ImGui::IsWindowAppearing()) {
          function_filter_.clear();
          ImGui::SetKeyboardFocusHere();
        }
        inputText("##filter", &function_filter_, "Search signals...");
        if (ImGui::BeginChild("matches", ImVec2(450, 180))) {
          for (const auto &candidate : function_sources_) {
            if (candidate != e.name && utils::containsCI(candidate, function_filter_) &&
                ImGui::Selectable(candidate.c_str(), candidate == path)) {
              path = candidate;
              ImGui::CloseCurrentPopup();
            }
          }
        }
        ImGui::EndChild();
        ImGui::EndPopup();
      }
      ImGui::PopID();
    };
    signalInput("Primary signal (value)", e.source);
    ImGui::TextWrapped("Runs once per primary sample. time is its monotonic timestamp in seconds; value is its value.");
    for (size_t i = 0; i < e.additional.size(); ++i) {
      ImGui::PushID((int)i);
      signalInput(("Additional input (v" + std::to_string(i + 1) + ")").c_str(), e.additional[i]);
      if (ImGui::SmallButton("Remove input")) {
        e.additional.erase(e.additional.begin() + i);
        ImGui::PopID();
        break;
      }
      ImGui::PopID();
    }
    if (ImGui::Button("Add input")) e.additional.emplace_back();
    ImGui::TextWrapped("Additional inputs use the nearest sample in time, in order: v1, v2, ...");
    ImGui::Spacing();
    ImGui::TextUnformatted("Python function body");
    inputTextMultiline("##function", &e.function, ImVec2(-1, 150), ImGuiInputTextFlags_AllowTabInput);
    ImGui::TextWrapped("Return a number or (time, value). Example: return value * 2.23694 converts m/s to mph. math is available.");
    if (ImGui::CollapsingHeader("Global code (optional)")) {
      ImGui::TextWrapped("Runs before the first sample on each recalculation. Use for imports, constants, and initial state.");
      inputTextMultiline("##globals", &e.globals, ImVec2(-1, 100), ImGuiInputTextFlags_AllowTabInput);
    }
  }
  ImGui::EndChild();

  const auto name = editing ? e.name : utils::trimmed(e.name);
  std::string error;
  if (name.empty() || utils::trimmed(e.source).empty() || utils::trimmed(e.function).empty()) {
    error = "Enter a name, primary signal, and function body.";
  } else if ((!editing && std::any_of(equations_.begin(), equations_.end(), [&](const auto &other) { return other.name == name; })) ||
             can->fields.count(name)) {
    error = "This name is already used by a signal or function.";
  } else if (utils::trimmed(e.source) == name || std::any_of(e.additional.begin(), e.additional.end(), [&](const auto &p) { return utils::trimmed(p) == name; })) {
    error = "A function cannot use itself as an input.";
  } else if (std::any_of(e.additional.begin(), e.additional.end(), [](const auto &p) { return utils::trimmed(p).empty(); })) {
    error = "Choose a signal for each additional input or remove it.";
  }
  ImGui::Checkbox("Plot in a new chart", &function_plot_);
  ImGui::TextWrapped("%s", error.empty() ? "Saved with the layout. Calculation errors appear above the charts." : error.c_str());
  ImGui::BeginDisabled(!error.empty());
  const bool save = ImGui::Button(editing ? "Save changes" : "Create function");
  ImGui::EndDisabled();
  ImGui::SameLine();
  const bool cancel = ImGui::Button("Cancel");
  bool remove = false;
  if (editing) {
    std::string dependents;
    for (const auto &other : equations_) {
      if (other.name != function_original_name_ && (other.source == function_original_name_ ||
          std::find(other.additional.begin(), other.additional.end(), function_original_name_) != other.additional.end())) {
        if (!dependents.empty()) dependents += ", ";
        dependents += other.name;
      }
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!dependents.empty());
    remove = ImGui::Button("Delete function");
    ImGui::EndDisabled();
    disabledItemTooltip(dependents.empty() ? "Remove this function from the layout and all charts." :
                        ("Update or delete these functions first: " + dependents).c_str());
  }
  if (remove) {
    equations_.erase(std::remove_if(equations_.begin(), equations_.end(),
      [&](const auto &other) { return other.name == function_original_name_; }), equations_.end());
    // Removing a series may remove its chart, so retain a separate list while walking all tabs.
    std::vector<ChartView *> charts;
    for (const auto &chart : charts_) charts.push_back(chart.get());
    for (auto *chart : charts) {
      chart->removeIf([&](const auto &signal) { return signal.path == function_original_name_; });
    }
  }
  if (save) {
    e.name = name;
    e.source = utils::trimmed(e.source);
    for (auto &path : e.additional) path = utils::trimmed(path);
    auto existing = std::find_if(equations_.begin(), equations_.end(), [&](const auto &other) { return other.name == function_original_name_; });
    if (existing == equations_.end()) equations_.push_back(e);
    else *existing = e;
  }
  if (save || remove) {
    // Discard old results, including any in-flight evaluation of the previous definition.
    ++equation_revision_;
    calculated_.clear();
    equation_errors_.clear();
    for (auto &chart : charts_) chart->updateFields();
    rebuildSignalBrowser();
    if (save) analysisRequested();
    fieldsChanged();
    if (save && function_plot_) createChart()->addFields(e.name);
    updateState();
  }
  if (save || remove || cancel) {
    function_editor_open_ = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}
