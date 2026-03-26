#include "tools/jotpluggler/app.h"
#include "tools/jotpluggler/common.h"

#include "implot.h"

#include <cfloat>
#include <chrono>
#include <cstring>
#include <regex>
#include <set>
#include <stdexcept>
#include <unistd.h>

#include "third_party/json11/json11.hpp"

namespace fs = std::filesystem;

namespace {

struct PythonEvalResult {
  std::vector<double> xs;
  std::vector<double> ys;
};

struct CustomSeriesTemplate {
  const char *name;
  const char *globals_code;
  const char *function_code;
  const char *preview_text;
  int required_additional_sources;
  const char *requirement_text;
};

void write_binary_vector(const fs::path &path, const std::vector<double> &values) {
  write_file_or_throw(path, values.data(), values.size() * sizeof(double));
}

std::vector<double> read_binary_vector(const fs::path &path) {
  const std::string raw = read_file_or_throw(path);
  if (raw.size() % sizeof(double) != 0) {
    throw std::runtime_error("Invalid binary series file: " + path.string());
  }
  std::vector<double> values(raw.size() / sizeof(double));
  if (!values.empty()) {
    std::memcpy(values.data(), raw.data(), raw.size());
  }
  return values;
}

void write_text_file(const fs::path &path, std::string_view text) {
  write_file_or_throw(path, text);
}

fs::path create_custom_series_temp_dir() {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path dir = fs::temp_directory_path() / ("jotpluggler_math_" + std::to_string(::getpid()) + "_" + std::to_string(stamp));
  fs::create_directories(dir);
  return dir;
}

void reset_custom_series_editor(CustomSeriesEditorState *editor) {
  *editor = CustomSeriesEditorState{};
}

bool add_additional_source(CustomSeriesEditorState *editor, const std::string &path) {
  if (path.empty() || path == editor->linked_source) return false;
  if (std::find(editor->additional_sources.begin(), editor->additional_sources.end(), path) != editor->additional_sources.end()) {
    return false;
  }
  editor->additional_sources.push_back(path);
  return true;
}

std::string next_custom_curve_name(const Pane &pane) {
  std::set<std::string> used;
  for (const Curve &curve : pane.curves) {
    if (!curve.label.empty()) {
      used.insert(curve.label);
    }
    if (!curve.name.empty()) {
      used.insert(curve.name);
    }
  }
  for (int i = 1; i < 1000; ++i) {
    const std::string candidate = "series" + std::to_string(i);
    if (used.find(candidate) == used.end()) {
      return candidate;
    }
  }
  return "series";
}

Curve make_custom_curve(const Pane &pane,
                        const std::string &name,
                        const CustomPythonSeries &spec,
                        PythonEvalResult result) {
  Curve curve;
  curve.name = name;
  curve.label = name;
  curve.color = app_next_curve_color(pane);
  curve.runtime_only = true;
  curve.custom_python = spec;
  curve.xs = std::move(result.xs);
  curve.ys = std::move(result.ys);
  return curve;
}

bool upsert_custom_curve_in_pane(WorkspaceTab *tab, int pane_index, Curve curve) {
  if (pane_index < 0 || pane_index >= static_cast<int>(tab->panes.size())) {
    return false;
  }
  Pane &pane = tab->panes[static_cast<size_t>(pane_index)];
  for (Curve &existing : pane.curves) {
    if (existing.runtime_only && existing.name == curve.name) {
      existing.visible = true;
      existing.label = curve.label;
      existing.custom_python = curve.custom_python;
      existing.xs = std::move(curve.xs);
      existing.ys = std::move(curve.ys);
      return false;
    }
  }
  pane.curves.push_back(std::move(curve));
  return true;
}

std::set<std::string> collect_custom_series_paths(const CustomPythonSeries &spec,
                                                  std::string_view globals_code,
                                                  std::string_view function_code) {
  std::set<std::string> paths;
  if (!spec.linked_source.empty()) {
    paths.insert(spec.linked_source);
  }
  paths.insert(spec.additional_sources.begin(), spec.additional_sources.end());

  static const std::regex kPathRegex(R"([tv]\(\s*["']([^"']+)["']\s*\))");
  const auto collect_from = [&](std::string_view code) {
    std::string owned(code);
    for (std::sregex_iterator it(owned.begin(), owned.end(), kPathRegex), end; it != end; ++it) {
      paths.insert((*it)[1].str());
    }
  };
  collect_from(globals_code);
  collect_from(function_code);
  return paths;
}

PythonEvalResult evaluate_custom_python_series(const AppSession &session,
                                               const CustomPythonSeries &spec) {
  const std::set<std::string> referenced_paths =
    collect_custom_series_paths(spec, spec.globals_code, spec.function_code);
  if (referenced_paths.empty()) throw std::runtime_error("No input series referenced. Set an input timeseries or reference route paths in code.");

  const fs::path temp_dir = create_custom_series_temp_dir();
  try {
    const fs::path globals_path = temp_dir / "globals.py";
    const fs::path code_path = temp_dir / "code.py";
    const fs::path manifest_path = temp_dir / "manifest.json";
    const fs::path out_t_path = temp_dir / "result.t.bin";
    const fs::path out_v_path = temp_dir / "result.v.bin";

    write_text_file(globals_path, spec.globals_code);
    write_text_file(code_path, spec.function_code);

    json11::Json::array paths_json(session.route_data.paths.begin(), session.route_data.paths.end());
    json11::Json::array additional_json(spec.additional_sources.begin(), spec.additional_sources.end());
    json11::Json::array series_json;
    size_t series_index = 0;
    for (const std::string &path : referenced_paths) {
      const RouteSeries *series = app_find_route_series(session, path);
      if (series == nullptr || series->times.size() < 2 || series->times.size() != series->values.size()) {
        throw std::runtime_error("Missing route series " + path);
      }
      const std::string prefix = "series_" + std::to_string(series_index++);
      const fs::path time_path = temp_dir / (prefix + ".t.bin");
      const fs::path value_path = temp_dir / (prefix + ".v.bin");
      write_binary_vector(time_path, series->times);
      write_binary_vector(value_path, series->values);
      series_json.push_back(json11::Json::object{
        {"path", path}, {"t", time_path.string()}, {"v", value_path.string()}});
    }
    const json11::Json manifest_json = json11::Json::object{
      {"paths", std::move(paths_json)},
      {"linked_source", spec.linked_source},
      {"additional_sources", std::move(additional_json)},
      {"series", std::move(series_json)},
    };
    write_text_file(manifest_path, manifest_json.dump());

    const CommandResult process = run_process_capture_output({
      "python3",
      (repo_root() / "tools" / "jotpluggler" / "math_eval.py").string(),
      manifest_path.string(),
      globals_path.string(),
      code_path.string(),
      out_t_path.string(),
      out_v_path.string(),
    });
    if (process.exit_code != 0) {
      const std::string error_text = util::strip(process.output);
      throw std::runtime_error(error_text.empty() ? "Python evaluation failed" : error_text);
    }

    PythonEvalResult result;
    result.xs = read_binary_vector(out_t_path);
    result.ys = read_binary_vector(out_v_path);
    if (result.xs.size() < 2 || result.xs.size() != result.ys.size()) {
      throw std::runtime_error("Custom series returned invalid output");
    }
    fs::remove_all(temp_dir);
    return result;
  } catch (...) {
    std::error_code ignore_error;
    fs::remove_all(temp_dir, ignore_error);
    throw;
  }
}

void refresh_custom_curve_samples(AppSession *session, UiState *state, Curve *curve) {
  if (!curve->custom_python.has_value()) {
    return;
  }
  if (!session->route_data.has_time_range || session->route_data.series.empty()) {
    curve->runtime_error_message.clear();
    curve->xs.clear();
    curve->ys.clear();
    return;
  }
  try {
    PythonEvalResult result = evaluate_custom_python_series(*session, *curve->custom_python);
    curve->runtime_error_message.clear();
    curve->xs = std::move(result.xs);
    curve->ys = std::move(result.ys);
  } catch (const std::exception &err) {
    curve->xs.clear();
    curve->ys.clear();
    const std::string err_text = err.what();
    if (session->data_mode == SessionDataMode::Stream && util::starts_with(err_text, "Missing route series ")) {
      curve->runtime_error_message = err_text;
      return;
    }
    const std::string error_message = std::string("Failed to evaluate custom series \"")
      + app_curve_display_name(*curve) + "\":\n\n" + err_text;
    if (curve->runtime_error_message != error_message) {
      curve->runtime_error_message = error_message;
      state->error_text = error_message;
      state->open_error_popup = true;
    }
  }
}

const std::array<CustomSeriesTemplate, 4> &custom_series_templates() {
  static constexpr std::array<CustomSeriesTemplate, 4> kTemplates = {{
    {
      .name = "Derivative",
      .globals_code = "",
      .function_code = "return np.gradient(value, time)",
      .preview_text = "return np.gradient(value, time)",
      .required_additional_sources = 0,
      .requirement_text = "",
    },
    {
      .name = "Difference",
      .globals_code = "",
      .function_code = "return value - v1",
      .preview_text = "Requires one additional source timeseries.\n\nreturn value - v1",
      .required_additional_sources = 1,
      .requirement_text = "Difference requires one additional source timeseries for v1.",
    },
    {
      .name = "Smoothing",
      .globals_code = "window = 20\nweights = np.ones(window) / window",
      .function_code = "return np.convolve(value, weights, mode='same')",
      .preview_text = "window = 20\nweights = np.ones(window) / window\n\nreturn np.convolve(value, weights, mode='same')",
      .required_additional_sources = 0,
      .requirement_text = "",
    },
    {
      .name = "Integral",
      .globals_code = "",
      .function_code = "dt = np.mean(np.diff(time))\nreturn np.cumsum(value) * dt",
      .preview_text = "dt = np.mean(np.diff(time))\nreturn np.cumsum(value) * dt",
      .required_additional_sources = 0,
      .requirement_text = "",
    },
  }};
  return kTemplates;
}

void draw_custom_series_help_popup(CustomSeriesEditorState *editor) {
  if (editor->open_help) {
    ImGui::OpenPopup("Custom Series Help");
    editor->open_help = false;
  }
  if (!ImGui::BeginPopupModal("Custom Series Help", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextUnformatted("Available variables");
  ImGui::Separator();
  ImGui::BulletText("np: numpy");
  ImGui::BulletText("t(path), v(path): timestamps and values for a route series");
  ImGui::BulletText("paths: all available route series paths");
  ImGui::BulletText("time, value: linked input timeseries");
  ImGui::BulletText("t1, v1, t2, v2, ...: additional source timeseries");
  ImGui::Spacing();
  ImGui::TextWrapped("Write either a single expression like \"return np.gradient(value, time)\" "
                     "or a multi-line Python body that returns an array or a (times, values) tuple.");
  ImGui::Spacing();
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void draw_custom_series_preview(const AppSession &session, CustomSeriesEditorState *editor) {
  std::vector<double> preview_xs;
  std::vector<double> preview_ys;
  std::string preview_label = editor->preview_label;
  if (editor->preview_is_result && editor->preview_xs.size() > 1 && editor->preview_xs.size() == editor->preview_ys.size()) {
    preview_xs = editor->preview_xs;
    preview_ys = editor->preview_ys;
    if (preview_label.empty()) {
      preview_label = "Result preview";
    }
  } else if (!editor->linked_source.empty()) {
    if (const RouteSeries *series = app_find_route_series(session, editor->linked_source); series != nullptr
        && series->times.size() > 1 && series->times.size() == series->values.size()) {
      preview_xs = series->times;
      preview_ys = series->values;
      preview_label = "Input preview (not result)";
    }
  }

  if (!preview_xs.empty() && preview_xs.size() == preview_ys.size()) {
    std::vector<double> plot_xs;
    std::vector<double> plot_ys;
    app_decimate_samples(preview_xs, preview_ys, 1200, &plot_xs, &plot_ys);
    const double preview_x_min = preview_xs.front();
    const double preview_x_max = preview_xs.back() > preview_xs.front()
      ? preview_xs.back()
      : preview_xs.front() + 1e-6;
    std::string plot_id = "##custom_series_preview";
    if (editor->preview_is_result) {
      plot_id += "_result_";
      plot_id += editor->name.empty() ? preview_label : editor->name;
    } else if (!editor->linked_source.empty()) {
      plot_id += "_input_";
      plot_id += editor->linked_source;
    }
    ImGui::TextUnformatted(preview_label.c_str());
    if (!editor->linked_source.empty() && !editor->preview_is_result) {
      ImGui::SameLine();
      ImGui::TextDisabled("%s", editor->linked_source.c_str());
    }
    if (ImPlot::BeginPlot(plot_id.c_str(),
                          ImVec2(-1.0f, std::max(180.0f, ImGui::GetContentRegionAvail().y - 6.0f)),
                          ImPlotFlags_NoTitle | ImPlotFlags_NoMenus | ImPlotFlags_NoLegend)) {
      ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight,
                        ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
      ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, preview_x_min, preview_x_max);
      ImPlot::SetupAxisLimits(ImAxis_X1, preview_x_min, preview_x_max, ImPlotCond_Once);
      ImPlot::SetupAxisFormat(ImAxis_X1, "%.1f");
      ImPlot::SetupAxisFormat(ImAxis_Y1, "%.6g");
      ImPlotSpec spec;
      spec.LineColor = color_rgb(35, 107, 180);
      spec.LineWeight = 2.0f;
      ImPlot::PlotLine("##custom_preview_line", plot_xs.data(), plot_ys.data(), static_cast<int>(plot_xs.size()), spec);
      ImPlot::EndPlot();
    }
  } else {
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 72.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, color_rgb(116, 124, 133));
    ImGui::TextWrapped("Choose an input timeseries or click Preview to evaluate the custom result.");
    ImGui::PopStyleColor();
  }
}

std::string custom_series_name_status(const Pane &pane, std::string_view name) {
  const std::string trimmed = util::strip(std::string(name));
  if (trimmed.empty()) return "name required";
  if (!trimmed.empty() && trimmed.front() == '/') {
    return "cannot start with /";
  }
  for (const Curve &curve : pane.curves) {
    if (curve.runtime_only && curve.name == trimmed) return "updates existing curve";
  }
  return "new curve";
}

const CustomSeriesTemplate &selected_custom_series_template(const CustomSeriesEditorState &editor) {
  const auto &templates = custom_series_templates();
  return templates[static_cast<size_t>(std::clamp(editor.selected_template, 0, static_cast<int>(templates.size()) - 1))];
}

bool custom_series_template_ready(const CustomSeriesEditorState &editor) {
  const CustomSeriesTemplate &templ = selected_custom_series_template(editor);
  return !editor.linked_source.empty()
      && static_cast<int>(editor.additional_sources.size()) >= templ.required_additional_sources;
}

bool prepare_custom_series_spec(CustomSeriesEditorState *editor,
                                UiState *state,
                                bool require_name,
                                CustomPythonSeries *out_spec) {
  editor->name = util::strip(editor->name);
  editor->linked_source = util::strip(editor->linked_source);
  for (std::string &path : editor->additional_sources) {
    path = util::strip(path);
  }
  editor->additional_sources.erase(
    std::remove_if(editor->additional_sources.begin(), editor->additional_sources.end(),
                   [&](const std::string &path) { return path.empty() || path == editor->linked_source; }),
    editor->additional_sources.end());

  if (require_name && editor->name.empty()) {
    state->error_text = "Custom series name is required.";
    state->open_error_popup = true;
    return false;
  }
  if (require_name && !editor->name.empty() && editor->name.front() == '/') {
    state->error_text = "Custom series names may not start with '/'.";
    state->open_error_popup = true;
    return false;
  }

  *out_spec = CustomPythonSeries{
    .linked_source = editor->linked_source,
    .additional_sources = editor->additional_sources,
    .globals_code = editor->globals_code,
    .function_code = editor->function_code,
  };
  return true;
}

bool preview_custom_series_editor(AppSession *session, UiState *state) {
  CustomSeriesEditorState &editor = state->custom_series;
  const CustomSeriesTemplate &templ = selected_custom_series_template(editor);
  if (editor.linked_source.empty()) {
    state->error_text = "Choose an input timeseries before previewing.";
    state->open_error_popup = true;
    state->status_text = "Custom series preview failed";
    return false;
  }
  if (static_cast<int>(editor.additional_sources.size()) < templ.required_additional_sources) {
    state->error_text = templ.requirement_text;
    state->open_error_popup = true;
    state->status_text = "Custom series preview failed";
    return false;
  }
  CustomPythonSeries spec;
  if (!prepare_custom_series_spec(&editor, state, false, &spec)) return false;

  try {
    PythonEvalResult result = evaluate_custom_python_series(*session, spec);
    editor.preview_label = editor.name.empty() ? "Result preview" : editor.name;
    editor.preview_xs = std::move(result.xs);
    editor.preview_ys = std::move(result.ys);
    editor.preview_is_result = true;
    state->status_text = "Previewed custom series";
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Custom series preview failed";
    return false;
  }
}

bool apply_custom_series_editor(AppSession *session, UiState *state) {
  WorkspaceTab *tab = app_active_tab(&session->layout, *state);
  TabUiState *tab_state = app_active_tab_state(state);
  if (tab == nullptr || tab_state == nullptr) {
    state->status_text = "No active pane";
    return false;
  }
  if (tab_state->active_pane_index < 0 || tab_state->active_pane_index >= static_cast<int>(tab->panes.size())) {
    state->status_text = "No active pane";
    return false;
  }

  CustomSeriesEditorState &editor = state->custom_series;
  CustomPythonSeries spec;
  if (!prepare_custom_series_spec(&editor, state, true, &spec)) return false;

  try {
    PythonEvalResult result = evaluate_custom_python_series(*session, spec);
    const SketchLayout before_layout = session->layout;
    Pane &pane = tab->panes[static_cast<size_t>(tab_state->active_pane_index)];
    editor.preview_label = editor.name;
    editor.preview_xs = result.xs;
    editor.preview_ys = result.ys;
    editor.preview_is_result = true;
    const bool inserted = upsert_custom_curve_in_pane(tab,
                                                      tab_state->active_pane_index,
                                                      make_custom_curve(pane, editor.name, spec, std::move(result)));
    state->undo.push(before_layout);
    state->status_text = inserted ? "Created custom series " + editor.name
                                  : "Updated custom series " + editor.name;
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Custom series failed";
    return false;
  }
}

}  // namespace

void open_custom_series_editor(UiState *state, const std::string &preferred_source) {
  CustomSeriesEditorState &editor = state->custom_series;
  if (!editor.open && editor.name.empty() && editor.linked_source.empty() && editor.function_code == "return value") {
    editor.focus_name = true;
  }
  if (editor.linked_source.empty() && !preferred_source.empty()) {
    editor.linked_source = preferred_source;
  }
  editor.open = true;
  editor.request_select = true;
}

std::string preferred_custom_series_source(const Pane &pane) {
  for (const Curve &curve : pane.curves) {
    if (!curve.name.empty() && curve.name.front() == '/') {
      return curve.name;
    }
    if (curve.custom_python.has_value() && !curve.custom_python->linked_source.empty()) {
      return curve.custom_python->linked_source;
    }
  }
  return {};
}

void refresh_all_custom_curves(AppSession *session, UiState *state) {
  for (WorkspaceTab &tab : session->layout.tabs) {
    for (Pane &pane : tab.panes) {
      for (Curve &curve : pane.curves) {
        refresh_custom_curve_samples(session, state, &curve);
      }
    }
  }
}

void draw_editor_source_panel(UiState *state, CustomSeriesEditorState &editor) {
  ImGui::TextWrapped("Input timeseries. Provides arguments time and value:");
  ImGui::SetNextItemWidth(-FLT_MIN);
  input_text_string("##custom_linked_source", &editor.linked_source, ImGuiInputTextFlags_ReadOnly);
  if (ImGui::BeginDragDropTarget()) {
    if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("JOTP_BROWSER_PATH")) {
      editor.linked_source = static_cast<const char *>(payload->Data);
      editor.additional_sources.erase(
        std::remove(editor.additional_sources.begin(), editor.additional_sources.end(), editor.linked_source),
        editor.additional_sources.end());
      editor.preview_is_result = false;
    }
    ImGui::EndDragDropTarget();
  }
  if (ImGui::Button("Use Selected", ImVec2(120.0f, 0.0f)) && !state->selected_browser_path.empty()) {
    editor.linked_source = state->selected_browser_path;
    editor.additional_sources.erase(
      std::remove(editor.additional_sources.begin(), editor.additional_sources.end(), editor.linked_source),
      editor.additional_sources.end());
    editor.preview_is_result = false;
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear", ImVec2(120.0f, 0.0f))) {
    editor.linked_source.clear();
    editor.preview_is_result = false;
  }

  ImGui::Spacing();
  ImGui::TextUnformatted("Additional source timeseries:");
  ImGui::SameLine();
  const CustomSeriesTemplate &tmpl = selected_custom_series_template(editor);
  if (tmpl.required_additional_sources > 0) {
    const bool ready = static_cast<int>(editor.additional_sources.size()) >= tmpl.required_additional_sources;
    ImGui::TextColored(ready ? color_rgb(58, 126, 73) : color_rgb(180, 122, 44), "%s", tmpl.requirement_text);
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(editor.selected_additional_source < 0
                       || editor.selected_additional_source >= static_cast<int>(editor.additional_sources.size()));
  if (ImGui::Button("Remove Selected", ImVec2(140.0f, 0.0f))
      && editor.selected_additional_source >= 0
      && editor.selected_additional_source < static_cast<int>(editor.additional_sources.size())) {
    editor.additional_sources.erase(editor.additional_sources.begin()
      + static_cast<std::ptrdiff_t>(editor.selected_additional_source));
    editor.selected_additional_source = editor.additional_sources.empty()
      ? -1 : std::clamp(editor.selected_additional_source, 0, static_cast<int>(editor.additional_sources.size()) - 1);
    editor.preview_is_result = false;
  }
  ImGui::EndDisabled();

  if (ImGui::BeginChild("##custom_additional_sources", ImVec2(0.0f, 156.0f), true)) {
    if (ImGui::BeginTable("##custom_additional_table", 2,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableSetupColumn("id", ImGuiTableColumnFlags_WidthFixed, 42.0f);
      ImGui::TableSetupColumn("path", ImGuiTableColumnFlags_WidthStretch);
      for (size_t i = 0; i < editor.additional_sources.size(); ++i) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("v%zu", i + 1);
        ImGui::TableNextColumn();
        if (ImGui::Selectable(editor.additional_sources[i].c_str(),
                              editor.selected_additional_source == static_cast<int>(i),
                              ImGuiSelectableFlags_SpanAllColumns)) {
          editor.selected_additional_source = static_cast<int>(i);
        }
      }
      ImGui::EndTable();
    }
    if (ImGui::BeginDragDropTarget()) {
      if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("JOTP_BROWSER_PATH")) {
        if (add_additional_source(&editor, static_cast<const char *>(payload->Data)))
          editor.preview_is_result = false;
      }
      ImGui::EndDragDropTarget();
    }
  }
  ImGui::EndChild();
  if (ImGui::Button("Add Selected", ImVec2(120.0f, 0.0f))) {
    for (const std::string &path : state->selected_browser_paths) {
      if (add_additional_source(&editor, path)) editor.preview_is_result = false;
    }
  }

  ImGui::Spacing();
  ImGui::SeparatorText("Function library");
  const auto &templates = custom_series_templates();
  if (ImGui::BeginChild("##custom_series_template_list", ImVec2(0.0f, 132.0f), true)) {
    for (size_t i = 0; i < templates.size(); ++i) {
      if (ImGui::Selectable(templates[i].name, editor.selected_template == static_cast<int>(i),
                            ImGuiSelectableFlags_AllowDoubleClick)) {
        editor.selected_template = static_cast<int>(i);
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          editor.globals_code = templates[i].globals_code;
          editor.function_code = templates[i].function_code;
          editor.preview_is_result = false;
        }
      }
    }
  }
  ImGui::EndChild();
  if (ImGui::Button("Use Selected Example")) {
    const auto &sel = selected_custom_series_template(editor);
    editor.globals_code = sel.globals_code;
    editor.function_code = sel.function_code;
    editor.preview_is_result = false;
  }
  ImGui::Spacing();
  ImGui::TextDisabled("Preview");
  ImGui::BeginChild("##custom_series_template_preview", ImVec2(0.0f, 0.0f), true);
  ImGui::TextUnformatted(selected_custom_series_template(editor).preview_text);
  ImGui::EndChild();
}

void draw_editor_code_panel(CustomSeriesEditorState &editor, const Pane *active_pane) {
  const std::string name_status = active_pane != nullptr ? custom_series_name_status(*active_pane, editor.name) : "no active pane";
  ImGui::TextUnformatted("New name:");
  ImGui::SameLine();
  const bool name_error = name_status == "name required" || name_status == "cannot start with /";
  ImGui::TextColored(name_error ? color_rgb(200, 72, 64) : color_rgb(58, 126, 73), "%s", name_status.c_str());
  if (editor.focus_name) { ImGui::SetKeyboardFocusHere(); editor.focus_name = false; }
  ImGui::SetNextItemWidth(-FLT_MIN);
  input_text_string("##custom_series_name", &editor.name, ImGuiInputTextFlags_AutoSelectAll);

  ImGui::Spacing();
  ImGui::SeparatorText("Global variables");
  ImGui::SameLine();
  if (ImGui::SmallButton("Help")) editor.open_help = true;
  const float globals_h = std::max(96.0f, ImGui::GetContentRegionAvail().y * 0.28f);
  if (input_text_multiline_string("##custom_series_globals", &editor.globals_code,
                                  ImVec2(-FLT_MIN, globals_h), ImGuiInputTextFlags_AllowTabInput))
    editor.preview_is_result = false;

  ImGui::Spacing();
  ImGui::TextUnformatted("def calc(time, value):");
  const float func_h = std::max(180.0f, ImGui::GetContentRegionAvail().y - 16.0f);
  if (input_text_multiline_string("##custom_series_function", &editor.function_code,
                                  ImVec2(-FLT_MIN, func_h), ImGuiInputTextFlags_AllowTabInput))
    editor.preview_is_result = false;
}

void draw_custom_series_editor(AppSession *session, UiState *state) {
  CustomSeriesEditorState &editor = state->custom_series;
  if (!editor.open) return;

  WorkspaceTab *tab = app_active_tab(&session->layout, *state);
  TabUiState *tab_state = app_active_tab_state(state);
  Pane *active_pane = (tab && tab_state && tab_state->active_pane_index >= 0
                       && tab_state->active_pane_index < static_cast<int>(tab->panes.size()))
    ? &tab->panes[static_cast<size_t>(tab_state->active_pane_index)] : nullptr;
  if (editor.focus_name && active_pane && editor.name.empty())
    editor.name = next_custom_curve_name(*active_pane);

  draw_custom_series_help_popup(&editor);

  if (ImGui::BeginTabBar("##custom_series_tabs")) {
    if (ImGui::BeginTabItem("Single Function")) {
      const float footer_height = ImGui::GetFrameHeightWithSpacing() * 2.0f + 10.0f;
      if (ImGui::BeginChild("##custom_series_body",
                            ImVec2(0.0f, std::max(1.0f, ImGui::GetContentRegionAvail().y - footer_height)), false)) {
        if (ImGui::BeginChild("##custom_series_preview_child",
                              ImVec2(0.0f, std::max(200.0f, ImGui::GetContentRegionAvail().y * 0.28f)), true))
          draw_custom_series_preview(*session, &editor);
        ImGui::EndChild();
        ImGui::Spacing();

        if (ImGui::BeginTable("##custom_series_editor_table", 2,
                              ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp,
                              ImVec2(0.0f, std::max(1.0f, ImGui::GetContentRegionAvail().y)))) {
          ImGui::TableSetupColumn("left", ImGuiTableColumnFlags_WidthFixed, 320.0f);
          ImGui::TableSetupColumn("right", ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableNextColumn();
          if (ImGui::BeginChild("##custom_series_left", ImVec2(0.0f, 0.0f), false))
            draw_editor_source_panel(state, editor);
          ImGui::EndChild();
          ImGui::TableNextColumn();
          if (ImGui::BeginChild("##custom_series_right", ImVec2(0.0f, 0.0f), false))
            draw_editor_code_panel(editor, active_pane);
          ImGui::EndChild();
          ImGui::EndTable();
        }
      }
      ImGui::EndChild();

      ImGui::Spacing();
      if (ImGui::Button("New", ImVec2(120.0f, 0.0f))) {
        reset_custom_series_editor(&editor);
        if (!state->selected_browser_path.empty()) editor.linked_source = state->selected_browser_path;
        editor.open = true;
        editor.focus_name = true;
      }
      ImGui::SameLine();
      ImGui::BeginDisabled(!custom_series_template_ready(editor));
      if (ImGui::Button("Preview Result", ImVec2(120.0f, 0.0f)))
        preview_custom_series_editor(session, state);
      ImGui::EndDisabled();
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && !custom_series_template_ready(editor)) {
        if (editor.linked_source.empty()) ImGui::SetTooltip("Choose an input timeseries first.");
        else ImGui::SetTooltip("%s", selected_custom_series_template(editor).requirement_text);
      }
      ImGui::SameLine();
      if (ImGui::Button("Apply", ImVec2(120.0f, 0.0f))) apply_custom_series_editor(session, state);
      ImGui::SameLine();
      if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) { editor.open = false; editor.request_select = false; }
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}
