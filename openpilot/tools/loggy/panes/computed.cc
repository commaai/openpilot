#include "tools/loggy/panes/computed.h"

#include "tools/loggy/backend/csv.h"
#include "tools/loggy/panes/browser.h"
#include "tools/loggy/shell/native_dialog.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "json11/json11.hpp"

#include <algorithm>
#include <any>
#include <array>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <utility>

namespace loggy {
namespace {

struct ComputedTemplate;
struct ComputedEditorState;
struct ComputedPreviewRow;

std::vector<std::string> computed_sources_from_text(std::string_view text);
std::string computed_sources_text(const std::vector<std::string> &sources);
ComputedEditorState parse_computed_editor_state(std::string_view state_json);
std::string computed_editor_state_json(const ComputedEditorState &state);
bool apply_computed_template(ComputedEditorState *state, int template_index);
ComputedSeriesSpec computed_spec_from_editor_state(const ComputedEditorState &state);
std::string computed_default_export_path(std::string_view output_path);
std::vector<ComputedPreviewRow> computed_preview_rows(const Store &store,
                                                      std::string_view path,
                                                      TimeRange range,
                                                      size_t max_rows);

struct ComputedTemplate {
  const char *name = "";
  const char *globals_code = "";
  const char *function_code = "";
  int required_additional_sources = 0;
};

struct ComputedEditorState {
  std::string name = "custom";
  std::string linked_source = "/carState/vEgo";
  std::vector<std::string> additional_sources;
  std::string globals_code;
  std::string function_code = "return value";
  int template_index = 0;
  std::string output_path;
  std::string status;
  std::string export_path = "/tmp/loggy_computed.csv";
  std::string export_status;
};

struct ComputedPreviewRow {
  double time = 0.0;
  double value = 0.0;
};

constexpr size_t kComputedTemplateCount = 5;
using ComputedTemplateArray = std::array<ComputedTemplate, kComputedTemplateCount>;

constexpr ComputedTemplateArray kComputedTemplates = {{
  {
    .name = "Custom",
    .globals_code = "",
    .function_code = "return value",
    .required_additional_sources = 0,
  },
  {
    .name = "Derivative",
    .globals_code = "",
    .function_code = "return np.gradient(value, time)",
    .required_additional_sources = 0,
  },
  {
    .name = "Difference",
    .globals_code = "",
    .function_code = "return value - v1",
    .required_additional_sources = 1,
  },
  {
    .name = "Smoothing",
    .globals_code = "window = 20\nweights = np.ones(window) / window",
    .function_code = "return np.convolve(value, weights, mode='same')",
    .required_additional_sources = 0,
  },
  {
    .name = "Integral",
    .globals_code = "",
    .function_code = "dt = np.mean(np.diff(time))\nreturn np.cumsum(value) * dt",
    .required_additional_sources = 0,
  },
}};

const ComputedTemplateArray &computed_templates();
std::vector<std::string> computed_sources_from_text(std::string_view text);
std::string computed_sources_text(const std::vector<std::string> &sources);
ComputedEditorState parse_computed_editor_state(std::string_view state_json);
std::string computed_editor_state_json(const ComputedEditorState &state);
bool apply_computed_template(ComputedEditorState *state, int template_index);
ComputedSeriesSpec computed_spec_from_editor_state(const ComputedEditorState &state);
std::string computed_default_export_path(std::string_view output_path);
std::vector<ComputedPreviewRow> computed_preview_rows(const Store &store,
                                                      std::string_view path,
                                                      TimeRange range,
                                                      size_t max_rows);

template <size_t N>
void copy_to_buffer(std::array<char, N> *buffer, const std::string &text) {
  static_assert(N > 0);
  std::snprintf(buffer->data(), buffer->size(), "%s", text.c_str());
}

std::string trim(std::string_view text) {
  size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) ++begin;
  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) --end;
  return std::string(text.substr(begin, end - begin));
}

bool input_string(const char *label, std::string *value, float width) {
  std::array<char, 512> buffer{};
  copy_to_buffer(&buffer, *value);
  ImGui::SetNextItemWidth(width);
  if (!ImGui::InputText(label, buffer.data(), buffer.size())) return false;
  *value = trim(buffer.data());
  return true;
}

bool input_multiline(const char *label, std::string *value, ImVec2 size) {
  std::array<char, 8192> buffer{};
  copy_to_buffer(&buffer, *value);
  if (!ImGui::InputTextMultiline(label, buffer.data(), buffer.size(), size)) return false;
  *value = buffer.data();
  return true;
}

struct ComputedPaneTransientState {
  ComputedEditorState state;
  std::string loaded_json;
};

ComputedEditorState &computed_pane_state(PaneInstance &pane) {
  auto *transient = std::any_cast<ComputedPaneTransientState>(&pane.transient_state);
  if (transient == nullptr || transient->loaded_json != pane.state_json) {
    pane.transient_state = ComputedPaneTransientState{
      .state = parse_computed_editor_state(pane.state_json),
      .loaded_json = pane.state_json,
    };
    transient = std::any_cast<ComputedPaneTransientState>(&pane.transient_state);
  }
  return transient->state;
}

void save_computed_pane_state(PaneInstance &pane, const ComputedEditorState &state) {
  pane.state_json = computed_editor_state_json(state);
  auto *transient = std::any_cast<ComputedPaneTransientState>(&pane.transient_state);
  if (transient != nullptr) {
    transient->state = state;
    transient->loaded_json = pane.state_json;
  }
}

std::string computed_operation_token(const ComputedEditorState &state) {
  std::string out = "python";
  out += "|linked=" + trim(state.linked_source);
  out += "|globals=" + state.globals_code;
  out += "|function=" + state.function_code;
  for (const std::string &source : state.additional_sources) {
    out += "|source=" + trim(source);
  }
  return out;
}

TimeRange computed_editor_range(const Session &session) {
  TimeRange range = session.playback.route_range();
  if (!range.valid() || range.span() <= 0.0) range = session.view_range.range();
  return range;
}

void draw_output_path(ComputedEditorState *state) {
  if (state == nullptr || state->output_path.empty()) return;
  ImGui::TextUnformatted("Output");
  ImGui::SameLine();
  if (ImGui::SmallButton("Copy")) ImGui::SetClipboardText(state->output_path.c_str());

  push_mono_font();
  ImGui::Selectable(state->output_path.c_str(), false);
  pop_mono_font();
  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
    ImGui::SetDragDropPayload(kLoggySeriesPathPayload, state->output_path.c_str(), state->output_path.size() + 1);
    ImGui::TextUnformatted(state->output_path.c_str());
    ImGui::EndDragDropSource();
  }
}

void draw_export_controls(const Store &store, ComputedEditorState *state, TimeRange range, bool *changed) {
  if (state == nullptr || changed == nullptr) return;
  if (state->output_path.empty()) return;
  if (state->export_path.empty()) state->export_path = computed_default_export_path(state->output_path);

  std::array<char, 256> export_buf{};
  copy_to_buffer(&export_buf, state->export_path);
  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.58f, 220.0f, 520.0f));
  if (ImGui::InputTextWithHint("Export CSV", "/tmp/loggy_computed.csv", export_buf.data(), export_buf.size())) {
    state->export_path = trim(export_buf.data());
    state->export_status.clear();
    *changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Browse##computed_export")) {
    std::string error;
    const std::optional<std::string> path = native_dialog_choose_path(
      NativeDialogType::SaveFile,
      {.title = "Save Computed CSV", .path = state->export_path, .confirm_overwrite = true},
      error);
    if (path.has_value()) {
      state->export_path = trim(*path);
      state->export_status.clear();
    } else {
      state->export_status = error.empty() ? "Export path unchanged" : "Dialog failed: " + error;
    }
    *changed = true;
  }

  const std::string csv = series_csv(store, state->output_path, range);
  if (ImGui::GetContentRegionAvail().x > 108.0f) ImGui::SameLine();
  if (ImGui::Button("Copy CSV")) {
    ImGui::SetClipboardText(csv.c_str());
    state->export_status = "Copied CSV";
    *changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  if (ImGui::Button("Save CSV")) {
    std::string error;
    if (write_csv_file(state->export_path, csv, error)) {
      state->export_status = "Saved " + state->export_path;
    } else {
      state->export_status = "Export failed: " + error;
    }
    *changed = true;
  }
  if (!state->export_status.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", state->export_status.c_str());
  }
}

void draw_preview_table(const Store &store, const ComputedEditorState &state, TimeRange range) {
  if (state.output_path.empty()) return;
  const std::vector<ComputedPreviewRow> rows = computed_preview_rows(store, state.output_path, range, 80);
  ImGui::TextDisabled("%zu preview points", rows.size());
  if (rows.empty()) return;

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingStretchProp |
                                    ImGuiTableFlags_ScrollY;
  const float table_height = std::max(120.0f, ImGui::GetContentRegionAvail().y);
  if (!ImGui::BeginTable("##computed_preview", 2, flags, ImVec2(0.0f, table_height))) return;
  ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 96.0f);
  ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableHeadersRow();
  push_mono_font();
  for (const ComputedPreviewRow &row : rows) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%.6f", row.time);
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%.9g", row.value);
  }
  pop_mono_font();
  ImGui::EndTable();
}

const ComputedTemplateArray &computed_templates() {
  return kComputedTemplates;
}

std::vector<std::string> computed_sources_from_text(std::string_view text) {
  std::vector<std::string> sources;
  std::string token;
  auto flush = [&]() {
    std::string path = trim(token);
    token.clear();
    if (path.empty()) return;
    if (std::find(sources.begin(), sources.end(), path) == sources.end()) sources.push_back(std::move(path));
  };

  for (char c : text) {
    if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c)) != 0) {
      flush();
    } else {
      token.push_back(c);
    }
  }
  flush();
  return sources;
}

std::string computed_sources_text(const std::vector<std::string> &sources) {
  std::string out;
  for (const std::string &source : sources) {
    const std::string path = trim(source);
    if (path.empty()) continue;
    if (!out.empty()) out.push_back('\n');
    out += path;
  }
  return out;
}

ComputedEditorState parse_computed_editor_state(std::string_view state_json) {
  ComputedEditorState state;
  if (state_json.empty()) return state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;

  if (json["name"].is_string()) state.name = json["name"].string_value();
  if (json["linked_source"].is_string()) state.linked_source = json["linked_source"].string_value();
  if (json["additional_sources"].is_array()) {
    state.additional_sources.clear();
    for (const json11::Json &source : json["additional_sources"].array_items()) {
      if (source.is_string()) state.additional_sources.push_back(source.string_value());
    }
    state.additional_sources = computed_sources_from_text(computed_sources_text(state.additional_sources));
  } else if (json["additional_sources"].is_string()) {
    state.additional_sources = computed_sources_from_text(json["additional_sources"].string_value());
  }
  if (json["globals_code"].is_string()) state.globals_code = json["globals_code"].string_value();
  if (json["function_code"].is_string()) state.function_code = json["function_code"].string_value();
  if (json["template_index"].is_number()) {
    state.template_index = std::clamp(json["template_index"].int_value(), 0,
                                      static_cast<int>(computed_templates().size()) - 1);
  }
  if (json["output_path"].is_string()) state.output_path = json["output_path"].string_value();
  if (json["status"].is_string()) state.status = json["status"].string_value();
  if (json["export_path"].is_string()) state.export_path = json["export_path"].string_value();
  if (json["export_status"].is_string()) state.export_status = json["export_status"].string_value();
  return state;
}

std::string computed_editor_state_json(const ComputedEditorState &state) {
  json11::Json::array additional_sources;
  for (const std::string &source : state.additional_sources) {
    const std::string path = trim(source);
    if (!path.empty()) additional_sources.push_back(path);
  }
  return json11::Json(json11::Json::object{
    {"name", trim(state.name).empty() ? "custom" : trim(state.name)},
    {"linked_source", trim(state.linked_source)},
    {"additional_sources", std::move(additional_sources)},
    {"globals_code", state.globals_code},
    {"function_code", state.function_code.empty() ? "return value" : state.function_code},
    {"template_index", std::clamp(state.template_index, 0, static_cast<int>(computed_templates().size()) - 1)},
    {"output_path", state.output_path},
    {"status", state.status},
    {"export_path", state.export_path},
    {"export_status", state.export_status},
  }).dump();
}

bool apply_computed_template(ComputedEditorState *state, int template_index) {
  if (state == nullptr || template_index < 0 ||
      template_index >= static_cast<int>(computed_templates().size())) {
    return false;
  }
  const ComputedTemplate &templ = computed_templates()[static_cast<size_t>(template_index)];
  state->template_index = template_index;
  state->globals_code = templ.globals_code;
  state->function_code = templ.function_code;
  state->output_path.clear();
  state->status.clear();
  return true;
}

ComputedSeriesSpec computed_spec_from_editor_state(const ComputedEditorState &state) {
  ComputedSeriesSpec spec;
  spec.kind = ComputedSeriesKind::CustomPython;
  spec.label = trim(state.name).empty() ? "custom" : trim(state.name);
  spec.python_linked_source = trim(state.linked_source);
  spec.python_additional_sources = computed_sources_from_text(computed_sources_text(state.additional_sources));
  spec.python_globals_code = state.globals_code;
  spec.python_function_code = state.function_code.empty() ? "return value" : state.function_code;
  const std::string operation = computed_operation_token(state);
  spec.output_path = computed_output_path(spec.python_linked_source, spec.label, operation);
  return spec;
}

std::string computed_default_export_path(std::string_view output_path) {
  std::string name(output_path);
  const size_t slash = name.find_last_of('/');
  if (slash != std::string::npos) name = name.substr(slash + 1);
  if (name.empty()) name = "computed";
  std::replace_if(name.begin(), name.end(), [](unsigned char c) {
    return std::isalnum(c) == 0 && c != '-' && c != '_';
  }, '_');
  return (std::filesystem::temp_directory_path() / ("loggy_" + name + ".csv")).string();
}

std::vector<ComputedPreviewRow> computed_preview_rows(const Store &store,
                                                      std::string_view path,
                                                      TimeRange range,
                                                      size_t max_rows) {
  std::vector<ComputedPreviewRow> rows;
  if (path.empty() || max_rows == 0) return rows;
  const SeriesView view = store.series(path, range.start_, range.end, max_rows);
  rows.reserve(view.points.size());
  for (const SeriesPoint &point : view.points) rows.push_back({point.t, point.value});
  return rows;
}

}  // namespace

void draw_computed_pane(Session &session, PaneInstance &pane) {
  ComputedEditorState &state = computed_pane_state(pane);
  bool changed = false;

  const float full_width = ImGui::GetContentRegionAvail().x;
  const float item_width = std::clamp(full_width * 0.58f, 180.0f, 520.0f);
  changed = input_string("Name", &state.name, item_width) || changed;
  changed = input_string("Source", &state.linked_source, item_width) || changed;

  std::string additional_sources = computed_sources_text(state.additional_sources);
  const float multiline_width = std::max(220.0f, full_width - 8.0f);
  if (input_multiline("Additional", &additional_sources, ImVec2(multiline_width, 58.0f))) {
    state.additional_sources = computed_sources_from_text(additional_sources);
    changed = true;
  }

  const ComputedTemplateArray &templates = computed_templates();
  state.template_index = std::clamp(state.template_index, 0, static_cast<int>(templates.size()) - 1);
  ImGui::SetNextItemWidth(std::min(180.0f, item_width));
  if (ImGui::BeginCombo("Template", templates[static_cast<size_t>(state.template_index)].name)) {
    for (int i = 0; i < static_cast<int>(templates.size()); ++i) {
      const bool selected = i == state.template_index;
      if (ImGui::Selectable(templates[static_cast<size_t>(i)].name, selected)) {
        state.template_index = i;
        changed = true;
      }
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::Button("Apply")) changed = apply_computed_template(&state, state.template_index) || changed;

  changed = input_multiline("Globals", &state.globals_code, ImVec2(multiline_width, 82.0f)) || changed;
  changed = input_multiline("Function", &state.function_code, ImVec2(multiline_width, 118.0f)) || changed;

  if (ImGui::Button("Run")) {
    state.additional_sources = computed_sources_from_text(computed_sources_text(state.additional_sources));
    const ComputedSeriesSpec spec = computed_spec_from_editor_state(state);
    state.output_path = spec.output_path;
    std::vector<ComputedSeriesStatus> statuses;
    StoreBatch batch = materialize_computed_series_batch(session.store, {spec}, computed_editor_range(session), &statuses);
    const ComputedSeriesStatus status = statuses.empty() ? ComputedSeriesStatus{.output_path = spec.output_path,
                                                                                 .ok = false,
                                                                                 .message = "no status"}
                                                         : statuses.front();
    if (status.ok) {
      session.store.stage(std::move(batch));
      state.status = "ok: " + std::to_string(status.output_points) + " points";
      if (state.export_path.empty()) state.export_path = computed_default_export_path(state.output_path);
    } else {
      state.status = status.message.empty() ? "failed" : status.message;
    }
    changed = true;
  }

  if (!state.status.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", state.status.c_str());
  }

  draw_output_path(&state);
  draw_export_controls(session.store, &state, computed_editor_range(session), &changed);
  draw_preview_table(session.store, state, computed_editor_range(session));

  if (changed) save_computed_pane_state(pane, state);
}

}  // namespace loggy
