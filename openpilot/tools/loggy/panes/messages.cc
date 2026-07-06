#include "tools/loggy/panes/messages.h"

#include "tools/loggy/backend/csv.h"
#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace loggy {
namespace {

// Byte-change tint fades over this much ROUTE time, so it reads the same whether playing or
// paused-and-scrubbing (see REVIEW.md defect B).
constexpr double kByteChangeFadeSeconds = 1.0;

struct MessageTableState {
  std::string filter;
  std::string name_filter;
  std::string node_filter;
  int bus_filter = -1;
  bool show_inactive = false;
  bool suppress_highlighted = false;
  bool suppress_signals = false;
  size_t max_rows = 500;
};

struct MessageTableRow {
  MessageId id;
  MessageSummary summary;
  std::string name;
  std::string node;
  Msg *msg = nullptr;
};

MessageTableState parse_message_table_state(std::string_view state_json) {
  MessageTableState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["name_filter"].is_string()) state.name_filter = json["name_filter"].string_value();
  if (json["node_filter"].is_string()) state.node_filter = json["node_filter"].string_value();
  if (json["bus_filter"].is_number()) state.bus_filter = std::clamp(json["bus_filter"].int_value(), -1, 255);
  if (json["show_inactive"].is_bool()) state.show_inactive = json["show_inactive"].bool_value();
  if (json["suppress_highlighted"].is_bool()) state.suppress_highlighted = json["suppress_highlighted"].bool_value();
  if (json["suppress_signals"].is_bool()) state.suppress_signals = json["suppress_signals"].bool_value();
  if (json["max_rows"].is_number()) state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 1, 5000));
  return state;
}

std::string message_table_state_json(const MessageId &id, const MessageTableState &state) {
  return json11::Json(json11::Json::object{
    {"id", id.to_string()},
    {"source", static_cast<int>(id.source)},
    {"address", static_cast<int>(id.address)},
    {"filter", state.filter},
    {"name_filter", state.name_filter},
    {"node_filter", state.node_filter},
    {"bus_filter", state.bus_filter},
    {"show_inactive", state.show_inactive},
    {"suppress_highlighted", state.suppress_highlighted},
    {"suppress_signals", state.suppress_signals},
    {"max_rows", static_cast<int>(state.max_rows)},
  }).dump();
}

std::string lower_message_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

bool message_id_matches_filter(const MessageId &id, std::string_view filter) {
  if (filter.empty()) return true;
  const std::string needle = lower_message_text(filter);
  const std::string id_text = lower_message_text(id.to_string());
  char hex_buf[32];
  std::snprintf(hex_buf, sizeof(hex_buf), "%X", id.address);
  const std::string hex_text = lower_message_text(hex_buf);
  char prefixed_hex_buf[36];
  std::snprintf(prefixed_hex_buf, sizeof(prefixed_hex_buf), "0x%X", id.address);
  const std::string prefixed_hex_text = lower_message_text(prefixed_hex_buf);
  return id_text.find(needle) != std::string::npos
      || hex_text.find(needle) != std::string::npos
      || prefixed_hex_text.find(needle) != std::string::npos;
}

bool message_metadata_matches_filter(std::string_view text, std::string_view filter) {
  if (filter.empty()) return true;
  if (text.empty()) return false;
  return lower_message_text(text).find(lower_message_text(filter)) != std::string::npos;
}

std::vector<MessageTableRow> prepare_message_table_skeletons(DBCManager &manager,
                                                            const Store &store,
                                                            const MessageTableState &state) {
  const std::vector<MessageId> ids = store.can_message_ids();
  std::vector<MessageTableRow> rows;
  rows.reserve(ids.size());
  for (const MessageId &id : ids) {
    if (state.bus_filter >= 0 && id.source != static_cast<uint8_t>(state.bus_filter)) continue;
    if (!message_id_matches_filter(id, state.filter)) continue;
    Msg *msg = manager.msg(id);
    const std::string name = msg ? msg->name : std::string{};
    const std::string node = msg ? (msg->transmitter.empty() ? DEFAULT_NODE_NAME : msg->transmitter) : std::string{};
    if (!message_metadata_matches_filter(name, state.name_filter)) continue;
    if (!message_metadata_matches_filter(node, state.node_filter)) continue;
    MessageTableRow row;
    row.id = id;
    row.name = name;
    row.node = node;
    row.msg = msg;
    rows.push_back(row);
  }
  return rows;
}

struct MessagesPaneTransientState {
  std::string state_json;
  MessageTableState state;
  MessageId active_id = kDefaultLoggyMessageId;
  // Row skeletons (id/name/node/msg) survive across frames; rebuilt only when the store gains
  // ids, the DBC mutates, or the filters change. Summaries refresh per frame (binary search).
  std::vector<MessageTableRow> skeletons;
  uint64_t store_gen = UINT64_MAX;
  uint64_t dbc_gen = UINT64_MAX;
  std::string skel_key;
};


float byte_change_alpha(double last_change, double tracker) {
  if (!std::isfinite(last_change)) return 0.0f;
  const double age = std::max(0.0, tracker - last_change);
  return static_cast<float>(std::clamp(1.0 - age / kByteChangeFadeSeconds, 0.0, 1.0));
}

bool byte_covered_by_signal(const Msg *msg, size_t byte_index) {
  return msg != nullptr && byte_index < msg->mask.size() && msg->mask[byte_index] != 0;
}

void draw_bytes(const std::vector<uint8_t> &bytes, const std::vector<float> *change_alpha) {
  push_mono_font();
  const ImVec2 cell(ImGui::CalcTextSize("00").x + 8.0f, ImGui::GetTextLineHeight() + 4.0f);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImU32 text = ImGui::GetColorU32(ImGuiCol_Text);
  const ImU32 fill = ImGui::GetColorU32(color_rgb(76, 80, 82));
  for (size_t i = 0; i < bytes.size(); ++i) {
    const ImVec2 p0(origin.x + static_cast<float>(i) * cell.x, origin.y);
    const ImVec2 p1(p0.x + cell.x, p0.y + cell.y);
    draw_list->AddRectFilled(p0, p1, fill);
    const float alpha = (change_alpha != nullptr && i < change_alpha->size()) ? (*change_alpha)[i] : 0.0f;
    if (alpha > 0.0f) {
      // Same overlay-tint style as binary.cc's heat_color: translucent accent over the base fill.
      draw_list->AddRectFilled(p0, p1, ImGui::GetColorU32(color_rgb(47, 101, 202, 0.18f + 0.58f * alpha)));
    }
    char buf[3];
    std::snprintf(buf, sizeof(buf), "%02X", bytes[i]);
    const ImVec2 text_size = ImGui::CalcTextSize(buf);
    draw_list->AddText(ImVec2(p0.x + (cell.x - text_size.x) * 0.5f, p0.y + (cell.y - text_size.y) * 0.5f), text, buf);
  }
  ImGui::Dummy(ImVec2(std::max(1.0f, static_cast<float>(bytes.size()) * cell.x), cell.y));
  pop_mono_font();
}

std::vector<int> collect_message_buses(const std::vector<MessageId> &ids) {
  std::vector<int> buses;
  buses.reserve(ids.size());
  for (const MessageId &id : ids) {
    buses.push_back(static_cast<int>(id.source));
  }
  std::sort(buses.begin(), buses.end());
  buses.erase(std::unique(buses.begin(), buses.end()), buses.end());
  return buses;
}

bool draw_bus_filter_combo(MessageTableState *state, const std::vector<int> &buses) {
  const std::string preview = state->bus_filter < 0 ? "All" : std::to_string(state->bus_filter);
  ImGui::SetNextItemWidth(86.0f);
  bool changed = false;
  if (!ImGui::BeginCombo("Bus", preview.c_str())) return false;

  if (ImGui::Selectable("All", state->bus_filter < 0)) {
    state->bus_filter = -1;
    changed = true;
  }
  if (state->bus_filter < 0) ImGui::SetItemDefaultFocus();

  for (const int bus : buses) {
    char label[16];
    std::snprintf(label, sizeof(label), "%d", bus);
    const bool selected = state->bus_filter == bus;
    if (ImGui::Selectable(label, selected)) {
      state->bus_filter = bus;
      changed = true;
    }
    if (selected) ImGui::SetItemDefaultFocus();
  }

  ImGui::EndCombo();
  return changed;
}

MessagesPaneTransientState &messages_pane_transient_state(PaneInstance &pane) {
  if (MessagesPaneTransientState *state = std::any_cast<MessagesPaneTransientState>(&pane.transient_state)) {
    return *state;
  }
  pane.transient_state = MessagesPaneTransientState{};
  return std::any_cast<MessagesPaneTransientState &>(pane.transient_state);
}

void activate_message_row(const MessageTableRow &row, const MessageTableState &state,
                          SelectionContext *selection, PaneInstance *pane) {
  selection->selected_msg_id = row.id;
  selection->has_selected_msg = true;
  pane->state_json = message_table_state_json(row.id, state);
}

MessageTableState &messages_pane_state(const Store &store, PaneInstance &pane) {
  MessagesPaneTransientState &transient = messages_pane_transient_state(pane);
  if (transient.state_json != pane.state_json) {
    transient.state = parse_message_table_state(pane.state_json);
    transient.active_id = initial_message_id_for_store(store, pane.state_json, std::nullopt);
    transient.state_json = pane.state_json;
  }
  return transient.state;
}

// Column order must match the TableSetupColumn calls below (0=Name .. 5=Count; Bytes is NoSort).
int message_row_compare(const MessageTableRow &a, const MessageTableRow &b, int column) {
  switch (column) {
    case 0: {
      const std::string an = a.name.empty() ? ("CAN " + a.id.to_string()) : a.name;
      const std::string bn = b.name.empty() ? ("CAN " + b.id.to_string()) : b.name;
      return an.compare(bn);
    }
    case 1:
      return (a.id.source > b.id.source) - (a.id.source < b.id.source);
    case 2:
      return (a.id.address > b.id.address) - (a.id.address < b.id.address);
    case 3:
      return a.node.compare(b.node);
    case 4:
      return (a.summary.frequency_hz > b.summary.frequency_hz) - (a.summary.frequency_hz < b.summary.frequency_hz);
    case 5:
      return (a.summary.count > b.summary.count) - (a.summary.count < b.summary.count);
    default:
      return 0;
  }
}

void sort_message_table_rows(std::vector<MessageTableRow *> *rows, const ImGuiTableSortSpecs &specs) {
  if (specs.SpecsCount == 0) return;
  const ImGuiTableColumnSortSpecs &spec = specs.Specs[0];
  std::stable_sort(rows->begin(), rows->end(), [&](const MessageTableRow *pa, const MessageTableRow *pb) {
    const MessageTableRow &a = *pa;
    const MessageTableRow &b = *pb;
    const int cmp = message_row_compare(a, b, spec.ColumnIndex);
    return spec.SortDirection == ImGuiSortDirection_Descending ? cmp > 0 : cmp < 0;
  });
}

void draw_message_row(const Store &store, MessageTableRow &row, const MessageTableState &state,
                      SelectionContext *selection, PaneInstance *pane, TimeRange summary_range) {
  const double tracker_time = summary_range.end;
  // The count-only pass skipped payload bytes; fetch them here, for visible rows only.
  if (row.summary.count > 0 && row.summary.latest_data.empty()) {
    row.summary.latest_data = store.can_event_summary(row.id, summary_range).latest_data;
  }
  const MessageSummary &summary = row.summary;
  const bool selected_row = selection->has_selected_msg && selection->selected_msg_id == row.id;
  const std::string name = row.name.empty() ? ("CAN " + row.id.to_string()) : row.name;

  ImGui::TableNextRow(ImGuiTableRowFlags_None, std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  ImGui::PushID(static_cast<int>(row.id.source));
  ImGui::PushID(static_cast<int>(row.id.address));

  ImGui::TableSetColumnIndex(0);
  // No AllowOverlap: with it, a wheel-scroll leaves a stale hover fill on whatever row lands
  // under the cursor, which reads as a false selection.
  const bool selected = ImGui::Selectable(name.c_str(), selected_row, ImGuiSelectableFlags_SpanAllColumns);
  if (selected || ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
    activate_message_row(row, state, selection, pane);
  }

  ImGui::TableSetColumnIndex(1);
  ImGui::Text("%u", static_cast<unsigned>(row.id.source));

  ImGui::TableSetColumnIndex(2);
  push_mono_font();
  ImGui::Text("0x%X", row.id.address);
  pop_mono_font();

  ImGui::TableSetColumnIndex(3);
  if (row.node.empty()) {
    ImGui::TextDisabled("--");
  } else {
    ImGui::TextUnformatted(row.node.c_str());
  }

  ImGui::TableSetColumnIndex(4);
  if (summary.count > 1) ImGui::Text("%.2f", summary.frequency_hz);
  else ImGui::TextUnformatted("--");

  ImGui::TableSetColumnIndex(5);
  ImGui::Text("%zu", summary.count);

  ImGui::TableSetColumnIndex(6);
  if (!summary.latest_data.empty()) {
    std::vector<float> alphas;
    if (!state.suppress_highlighted) {
      const std::vector<double> last_change =
          store.byte_change_times(row.id, TimeRange{tracker_time - kByteChangeFadeSeconds, tracker_time}, summary.latest_data.size());
      alphas.resize(last_change.size());
      for (size_t b = 0; b < last_change.size(); ++b) {
        alphas[b] = (state.suppress_signals && byte_covered_by_signal(row.msg, b))
                        ? 0.0f
                        : byte_change_alpha(last_change[b], tracker_time);
      }
    }
    draw_bytes(summary.latest_data, alphas.empty() ? nullptr : &alphas);
  } else {
    ImGui::TextDisabled("No events in view");
  }

  ImGui::PopID();
  ImGui::PopID();
}

}  // namespace

void draw_messages_pane(Session &session, PaneInstance &pane) {
  SelectionContext &selection = session.selection(pane.selection_group);
  MessagesPaneTransientState &transient = messages_pane_transient_state(pane);
  MessageTableState &state = messages_pane_state(session.store, pane);
  const std::optional<MessageId> selected = selection.has_selected_msg ? std::optional<MessageId>(selection.selected_msg_id) : std::nullopt;
  MessageId active_id = selected.has_value() ? *selected : transient.active_id;
  transient.active_id = active_id;
  const std::vector<MessageId> all_ids = session.store.can_message_ids();
  const std::vector<int> buses = collect_message_buses(all_ids);

  bool controls_changed = false;
  const float filter_width = std::clamp(ImGui::GetContentRegionAvail().x * 0.45f, 132.0f, 260.0f);
  ImGui::SetNextItemWidth(filter_width);
  if (input_text_with_hint("Filter", "ID", &state.filter)) {
    controls_changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 108.0f) ImGui::SameLine();
  const float name_filter_width = std::clamp(ImGui::GetContentRegionAvail().x * 0.40f, 84.0f, 170.0f);
  ImGui::SetNextItemWidth(name_filter_width);
  if (input_text_with_hint("Name", "Name", &state.name_filter)) {
    controls_changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 106.0f) ImGui::SameLine();
  const float node_filter_width = std::clamp(ImGui::GetContentRegionAvail().x * 0.40f, 84.0f, 170.0f);
  ImGui::SetNextItemWidth(node_filter_width);
  if (input_text_with_hint("Node", "Node", &state.node_filter)) {
    controls_changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 116.0f) ImGui::SameLine();
  controls_changed = draw_bus_filter_combo(&state, buses) || controls_changed;
  if (ImGui::GetContentRegionAvail().x > 48.0f) ImGui::SameLine();
  if (ImGui::Checkbox("Show all", &state.show_inactive)) {
    controls_changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 150.0f) ImGui::SameLine();
  if (ImGui::Checkbox("Suppress Highlighted", &state.suppress_highlighted)) {
    controls_changed = true;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Mute the byte-change tint until cleared");
  if (ImGui::GetContentRegionAvail().x > 130.0f) ImGui::SameLine();
  if (ImGui::Checkbox("Suppress Signals", &state.suppress_signals)) {
    controls_changed = true;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Mute the byte-change tint on bytes covered by loaded DBC signals");

  const std::string skel_key = state.filter + '\x1f' + state.name_filter + '\x1f'
                             + state.node_filter + '\x1f' + std::to_string(state.bus_filter);
  if (transient.store_gen != session.store.generation() || transient.dbc_gen != session.dbc.generation() ||
      transient.skel_key != skel_key) {
    transient.skeletons = prepare_message_table_skeletons(session.dbc, session.store, state);
    transient.store_gen = session.store.generation();
    transient.dbc_gen = session.dbc.generation();
    transient.skel_key = skel_key;
  }

  // Playhead-bound: Bytes/Count reflect events up to the tracker, like cabana, not the full route.
  // Payload bytes are fetched only for visible rows (draw_message_row); this pass is count-only.
  // It runs over ALL matching ids, not just visible rows, because sorting/visibility need every
  // row's count — each entry is one binary search, so this stays cheap at CAN-id scale.
  const TimeRange summary_range{session.playback.route_range().start_, session.playback.tracker_time()};
  for (MessageTableRow &row : transient.skeletons) {
    row.summary = summarize_message_events(session.store, row.id, summary_range, false);
  }
  std::vector<MessageTableRow *> rows;
  rows.reserve(transient.skeletons.size());
  for (MessageTableRow &row : transient.skeletons) {
    if (!state.show_inactive && row.summary.count == 0) continue;
    rows.push_back(&row);
    if (rows.size() >= state.max_rows) break;
  }

  // selected_msg_id is shared across every pane in this selection_group (History, Find Bits,
  // Find Signal, Signal, ...), not owned by this table. Only replace it when there is truly no
  // selection yet or the id no longer exists in the store at all — never merely because this
  // pane's OWN text/bus/show-inactive filter currently hides it, or the row hasn't produced a
  // count yet in the playhead-bound summary window. Otherwise a Find-tool selection that this
  // table's filter doesn't happen to show gets silently stomped back to this table's front row
  // on the very next frame, and every other pane (History's Save Msg included) follows it.
  const bool selection_exists_in_store = selection.has_selected_msg &&
      std::find(all_ids.begin(), all_ids.end(), selection.selected_msg_id) != all_ids.end();
  if (!rows.empty() && (!selection.has_selected_msg || !selection_exists_in_store)) {
    active_id = rows.front()->id;
    selection.selected_msg_id = active_id;
    selection.has_selected_msg = true;
    pane.state_json = message_table_state_json(active_id, state);
    transient.state_json = pane.state_json;
  } else if (selection.has_selected_msg) {
    active_id = selection.selected_msg_id;
  }
  if (controls_changed) {
    pane.state_json = message_table_state_json(active_id, state);
    transient.state_json = pane.state_json;
  }
  transient.active_id = active_id;

  if (ImGui::GetContentRegionAvail().x > 142.0f) ImGui::SameLine();
  ImGui::TextDisabled("%zu/%zu CAN ids", rows.size(), all_ids.size());

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                    ImGuiTableFlags_Sortable;
  if (rows.empty()) {
    ImGui::TextDisabled("No CAN messages in view or filter");
    return;
  }

  size_t max_bytes = 8;
  for (const MessageTableRow *row : rows) {
    if (row->msg != nullptr) max_bytes = std::max(max_bytes, static_cast<size_t>(row->msg->size));
  }
  if (!ImGui::BeginTable("##loggy_messages", 7, flags, ImGui::GetContentRegionAvail())) return;
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort, 126.0f);
  ImGui::TableSetupColumn("Bus", ImGuiTableColumnFlags_WidthFixed, 42.0f);
  ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 72.0f);
  ImGui::TableSetupColumn("Node", ImGuiTableColumnFlags_WidthFixed, 72.0f);
  ImGui::TableSetupColumn("Freq", ImGuiTableColumnFlags_WidthFixed, 64.0f);
  ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 70.0f);
  ImGui::TableSetupColumn("Bytes", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort,
                          std::max(160.0f, std::min(420.0f, static_cast<float>(max_bytes) * 30.0f)));
  ImGui::TableHeadersRow();

  // Rows are rebuilt fresh every frame, so the current sort spec must be reapplied every frame
  // rather than only when ImGui flags it dirty.
  if (ImGuiTableSortSpecs *sort_specs = ImGui::TableGetSortSpecs()) {
    sort_message_table_rows(&rows, *sort_specs);
    sort_specs->SpecsDirty = false;
  }

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  while (clipper.Step()) {
    for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
      draw_message_row(session.store, *rows[static_cast<size_t>(row_idx)], state, &selection, &pane, summary_range);
    }
  }
  ImGui::EndTable();
}

}  // namespace loggy
