#include "tools/cabana/ui/chart/chartswidget.h"

#include <algorithm>

#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/ui/icons.h"
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
  ImGui::SetNextWindowSizeConstraints(ImVec2(560, 440), ImVec2(FLT_MAX, FLT_MAX));
  if (!ImGui::BeginPopupModal("Custom Function", &function_editor_open_, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) return;

  auto &e = function_draft_;
  const bool editing = !function_original_name_.empty();
  // Keep the identity stable: charts and other functions refer to this name.
  ImGui::TextUnformatted("Name");
  ImGui::SetNextItemWidth(-1);
  ImGui::BeginDisabled(editing);
  inputText("##name", &e.name, "e.g. speed_mph");
  ImGui::EndDisabled();
  if (editing) ImGui::SetItemTooltip("The name is used by charts and other functions.");

  const auto &style = ImGui::GetStyle();
  const float content_bottom = ImGui::GetCursorPosY() + ImGui::GetContentRegionAvail().y;
  const float footer_height = 2 * ImGui::GetFrameHeight() + 2 * ImGui::GetTextLineHeight() + 4 * style.ItemSpacing.y + 1;
  const float footer_top = content_bottom - footer_height;
  if (ImGui::BeginChild("function_body", ImVec2(0, std::max(1.0f, footer_top - ImGui::GetCursorPosY() - style.ItemSpacing.y)),
                        ImGuiChildFlags_AlwaysUseWindowPadding)) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Inputs");
    alignRight(iconButtonWidth());
    if (iconButton("add_input", icon::PLUS_LG, "Add input")) e.additional.emplace_back();
    int remove_input = -1;
    if (ImGui::BeginTable("inputs", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoPadOuterX)) {
      ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("value").x);
      ImGui::TableSetupColumn("Signal", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, iconButtonWidth() * 2 + style.ItemInnerSpacing.x);
      auto signalInput = [&](const char *label, std::string &path, int index) {
        ImGui::PushID(index);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::SetItemTooltip(index < 0 ? "Primary signal: runs once per sample." : "Nearest sample to the primary signal's timestamp.");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        inputText("##path", &path, "Signal path or function name");
        if (ImGui::BeginDragDropTarget()) {
          if (auto *payload = ImGui::AcceptDragDropPayload("CABANA_TELEMETRY")) path = (const char *)payload->Data;
          ImGui::EndDragDropTarget();
        }
        ImGui::TableNextColumn();
        if (iconButton("browse", icon::FOLDER, "Browse signals")) ImGui::OpenPopup("signals");
        if (ImGui::BeginPopup("signals")) {
          if (ImGui::IsWindowAppearing()) {
            function_filter_.clear();
            ImGui::SetKeyboardFocusHere();
          }
          ImGui::SetNextItemWidth(-1);
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
        ImGui::SameLine(0, style.ItemInnerSpacing.x);
        ImGui::BeginDisabled(index < 0);
        if (iconButton("remove", icon::X_LG)) remove_input = index;
        ImGui::EndDisabled();
        disabledItemTooltip(index < 0 ? "The primary input is required." : "Remove input");
        ImGui::PopID();
      };
      signalInput("value", e.source, -1);
      for (size_t i = 0; i < e.additional.size(); ++i) {
        signalInput(("v" + std::to_string(i + 1)).c_str(), e.additional[i], i);
      }
      ImGui::EndTable();
    }
    if (remove_input >= 0) e.additional.erase(e.additional.begin() + remove_input);
    ImGui::TextDisabled("time: primary timestamp in seconds · v1, v2, ...: nearest sample");
    ImGui::Spacing();
    ImGui::SeparatorText("Python function body");
    inputTextMultiline("##function", &e.function, ImVec2(-1, ImGui::GetTextLineHeightWithSpacing() * 8), ImGuiInputTextFlags_AllowTabInput);
    ImGui::TextDisabled("Return a number or (time, value). math is available.");
    ImGui::SetItemTooltip("Example: return value * 2.23694 converts m/s to mph.");
    if (ImGui::CollapsingHeader("Global code (optional)")) {
      ImGui::TextWrapped("Runs before the first sample on each recalculation. Use for imports, constants, and initial state.");
      inputTextMultiline("##globals", &e.globals, ImVec2(-1, ImGui::GetTextLineHeightWithSpacing() * 5), ImGuiInputTextFlags_AllowTabInput);
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
  ImGui::SetCursorPosY(footer_top);
  ImGui::Separator();
  checkBox("Plot in a new chart", &function_plot_);
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  ImGui::TextWrapped("%s", error.empty() ? "Saved with the layout. Calculation errors appear above the charts." : error.c_str());
  ImGui::PopStyleColor();
  ImGui::SetCursorPosY(content_bottom - ImGui::GetFrameHeight());
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
    ImGui::BeginDisabled(!dependents.empty());
    remove = ImGui::Button("Delete function");
    ImGui::EndDisabled();
    disabledItemTooltip(dependents.empty() ? "Remove this function from the layout and all charts." :
                        ("Update or delete these functions first: " + dependents).c_str());
  }
  if (editing) ImGui::SameLine();
  bool save = false, cancel = false;
  dialogButtons(editing ? "Save" : "Create", &save, &cancel, error.empty());
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
