#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"

// Messages table panel -- parity spec: tools/cabana/messageswidget.{h,cc}
// (MessageListModel + MessageView + MessageViewHeader + MessagesWidget),
// frozen Qt reference. Byte-cell rendering mirrors MessageBytesDelegate::paint
// in tools/cabana/utils/util.cc.

namespace {

enum Column {
  COL_NAME = 0,
  COL_SOURCE,
  COL_ADDRESS,
  COL_NODE,
  COL_FREQ,
  COL_COUNT,
  COL_DATA,
  NUM_COLUMNS,
};

constexpr const char *COLUMN_LABELS[NUM_COLUMNS] = {"Name", "Bus", "ID", "Node", "Freq", "Count", "Bytes"};
constexpr const char *COLUMN_FILTER_HINTS[NUM_COLUMNS] = {
    "Filter Name", "Filter Bus", "Filter ID", "Filter Node", "Filter Freq", "Filter Count", "Filter Bytes"};

struct Row {
  MessageId id;
  std::string name;
  std::string node;
};

// MessagesWidget's toolbar + MessageListModel + MessageViewHeader state, all
// flattened into one file-scope struct (mirrors the file-scope statics used
// by transport.cc for equivalent drag/seek state).
struct MessagesPanelState {
  std::vector<Row> rows;                 // filtered + sorted, refreshed on dirty
  std::set<MessageId> dbc_messages;      // DBC-defined messages never seen live (INVALID_SOURCE)
  std::map<int, std::string> filters;    // column -> filter text (only non-empty columns present)
  char filter_buf[NUM_COLUMNS][128] = {};
  // Node hidden by default so Bytes fits the default dock width (View... re-enables it)
  bool column_visible[NUM_COLUMNS] = {true, true, true, false, true, true, true};

  int sort_column = COL_NAME;
  bool sort_ascending = true;
  bool show_inactive_messages = true;
  bool dirty = true;

  int sort_threshold = 0;    // mirrors MessageListModel::sort_threshold_
  int suppressed_count = 0;  // mirrors the "Clear (n)" button label
  int pending_scroll_index = -1;
  std::optional<MessageId> last_synced_selection;  // detects external selection changes (e.g. --demo auto-select)
};

MessagesPanelState g_state;

// -- small string helpers (no Qt available in this Qt-free core) ----------

bool contains_ci(const std::string &haystack, const std::string &needle) {
  if (needle.empty()) return true;
  auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(), [](char a, char b) {
    return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
  });
  return it != haystack.end();
}

std::string to_hex_string(uint32_t value) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%02X", value);
  return buf;
}

std::string to_hex(const std::vector<uint8_t> &dat) {
  static const char digits[] = "0123456789ABCDEF";
  std::string out;
  out.reserve(dat.size() * 2);
  for (uint8_t b : dat) {
    out.push_back(digits[b >> 4]);
    out.push_back(digits[b & 0xF]);
  }
  return out;
}

std::string format_freq(double freq) {
  char buf[32];
  if (freq > 0) {
    if (freq >= 0.95) {
      std::snprintf(buf, sizeof(buf), "%g", std::nearbyint(freq));
    } else {
      std::snprintf(buf, sizeof(buf), "%.2f", freq);
    }
    return buf;
  }
  return "--";
}

// mirrors messageswidget.cc's static parseRange(): "1" -> {1,1}, "1-3" -> {1,3}, "1-" -> {1,inf}
std::vector<std::string> split_dash(const std::string &s) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (true) {
    size_t pos = s.find('-', start);
    if (pos == std::string::npos) {
      parts.push_back(s.substr(start));
      break;
    }
    parts.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  return parts;
}

bool parse_uint(const std::string &s, int base, unsigned int *out) {
  if (s.empty() || s[0] == '-') return false;  // QString::toUInt rejects negatives
  char *end = nullptr;
  errno = 0;
  unsigned long v = std::strtoul(s.c_str(), &end, base);
  if (end != s.c_str() + s.size() || errno == ERANGE) return false;
  *out = static_cast<unsigned int>(v);
  return true;
}

bool parse_range(const std::string &filter, uint32_t value, int base = 10) {
  unsigned int min_v = 0;
  unsigned int max_v = std::numeric_limits<unsigned int>::max();
  std::vector<std::string> parts = split_dash(filter);
  bool ok = parts.size() >= 1 && parts.size() <= 2;
  if (ok && !parts[0].empty()) ok = parse_uint(parts[0], base, &min_v);
  if (ok && parts.size() == 1) {
    max_v = min_v;
  } else if (ok && parts.size() == 2 && !parts[1].empty()) {
    ok = parse_uint(parts[1], base, &max_v);
  }
  return ok && value >= min_v && value <= max_v;
}

// -- data refresh: mirrors MessageListModel::filterAndSort() / sort() / match() --

bool row_matches(const Row &row) {
  if (g_state.filters.empty()) return true;
  const CanData &data = can->lastMessage(row.id);
  for (const auto &[col, txt] : g_state.filters) {
    bool match = true;
    switch (col) {
      case COL_NAME: {
        match = contains_ci(row.name, txt);
        if (!match) {
          if (cabana::Msg *m = dbc()->msg(row.id); m != nullptr) {
            for (const cabana::Signal *s : m->getSignals()) {
              if (contains_ci(s->name, txt)) {
                match = true;
                break;
              }
            }
          }
        }
        break;
      }
      case COL_SOURCE:
        match = parse_range(txt, row.id.source);
        break;
      case COL_ADDRESS:
        match = contains_ci(to_hex_string(row.id.address), txt) || parse_range(txt, row.id.address, 16);
        break;
      case COL_NODE:
        match = contains_ci(row.node, txt);
        break;
      case COL_FREQ:
        match = parse_range(txt, static_cast<uint32_t>(std::max(0.0, data.freq)));
        break;
      case COL_COUNT:
        match = parse_range(txt, data.count);
        break;
      case COL_DATA:
        match = contains_ci(to_hex(data.dat), txt);
        break;
      default:
        break;
    }
    if (!match) return false;
  }
  return true;
}

void sort_rows(std::vector<Row> &rows) {
  auto compare = [](const Row &l, const Row &r) {
    switch (g_state.sort_column) {
      case COL_NAME: return std::tie(l.name, l.id) < std::tie(r.name, r.id);
      case COL_SOURCE: return std::tie(l.id.source, l.id.address) < std::tie(r.id.source, r.id.address);
      case COL_ADDRESS: return std::tie(l.id.address, l.id.source) < std::tie(r.id.address, r.id.source);
      case COL_NODE: return std::tie(l.node, l.id) < std::tie(r.node, r.id);
      case COL_FREQ: return std::tie(can->lastMessage(l.id).freq, l.id) < std::tie(can->lastMessage(r.id).freq, r.id);
      case COL_COUNT: return std::tie(can->lastMessage(l.id).count, l.id) < std::tie(can->lastMessage(r.id).count, r.id);
      default: return false;  // DATA column: not sortable
    }
  };
  if (g_state.sort_ascending) {
    std::stable_sort(rows.begin(), rows.end(), compare);
  } else {
    std::stable_sort(rows.rbegin(), rows.rend(), compare);
  }
}

void refilter_and_sort() {
  // merge CAN and DBC messages (messageswidget.cc filterAndSort())
  std::vector<MessageId> all_messages;
  all_messages.reserve(can->lastMessages().size() + g_state.dbc_messages.size());
  std::set<MessageId> remaining_dbc = g_state.dbc_messages;
  for (const auto &[id, m] : can->lastMessages()) {
    all_messages.push_back(id);
    remaining_dbc.erase(MessageId{.source = INVALID_SOURCE, .address = id.address});
  }
  all_messages.insert(all_messages.end(), remaining_dbc.begin(), remaining_dbc.end());

  std::vector<Row> rows;
  rows.reserve(all_messages.size());
  for (const MessageId &id : all_messages) {
    if (!g_state.show_inactive_messages && !can->isMessageActive(id)) continue;
    cabana::Msg *msg = dbc()->msg(id);
    Row row{.id = id, .name = msg ? msg->name : UNTITLED, .node = msg ? msg->transmitter : std::string()};
    if (row_matches(row)) rows.push_back(std::move(row));
  }
  sort_rows(rows);
  g_state.rows = std::move(rows);
}

void refresh_dbc_messages() {
  g_state.dbc_messages.clear();
  for (const auto &[addr, m] : dbc()->getMessages(INVALID_SOURCE)) {
    g_state.dbc_messages.insert(MessageId{.source = INVALID_SOURCE, .address = addr});
  }
}

void ensure_connected() {
  static bool connected = false;
  if (connected) return;
  connected = true;

  // mirrors MessageListModel::msgsReceived(): full refilter on new ids, or
  // throttled to settings.fps while a FREQ/COUNT/DATA filter is live (those
  // columns' matches depend on live values, not just id/name/node).
  can->msgsReceived.connect([](const std::set<MessageId> * /*new_msgs*/, bool has_new_ids) {
    const bool slow_filter_active =
        g_state.filters.count(COL_FREQ) || g_state.filters.count(COL_COUNT) || g_state.filters.count(COL_DATA);
    if (has_new_ids || (slow_filter_active && ++g_state.sort_threshold >= std::max(1, settings.fps))) {
      g_state.sort_threshold = 0;
      g_state.dirty = true;
    }
  });
  dbc()->DBCFileChanged.connect([]() {
    refresh_dbc_messages();
    g_state.dirty = true;
  });
  UndoStack::instance()->indexChanged.connect([](int) {
    refresh_dbc_messages();
    g_state.dirty = true;
  });

  refresh_dbc_messages();
  refilter_and_sort();
}

// -- toolbar: mirrors MessagesWidget::createToolBar() (placed above the
// table in the Qt source, not below -- see report) --------------------------

void draw_toolbar(AppState & /*app*/) {
  if (ImGui::Button("Suppress Highlighted")) {
    g_state.suppressed_count = static_cast<int>(can->suppressHighlighted());
  }
  ImGui::SameLine();
  const std::string clear_label =
      g_state.suppressed_count > 0 ? ("Clear (" + std::to_string(g_state.suppressed_count) + ")") : std::string("Clear");
  ImGui::BeginDisabled(g_state.suppressed_count <= 0);
  if (ImGui::Button(clear_label.c_str())) {
    can->clearSuppressed();
    g_state.suppressed_count = 0;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clear suppressed");
  ImGui::EndDisabled();

  // right-aligned group: "Suppress Signals" checkbox + "View..." menu button
  const float view_btn_w = ImGui::CalcTextSize("View...").x + ImGui::GetStyle().FramePadding.x * 2.0f;
  const float checkbox_w =
      ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize("Suppress Signals").x;
  const float right_group_w = checkbox_w + ImGui::GetStyle().ItemSpacing.x + view_btn_w;
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - right_group_w));

  bool suppress_defined = settings.suppress_defined_signals;
  if (ImGui::Checkbox("Suppress Signals", &suppress_defined)) {
    settings.suppress_defined_signals = suppress_defined;
    can->suppressDefinedSignals(suppress_defined);
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Suppress defined signals");

  ImGui::SameLine();
  if (ImGui::Button("View...")) ImGui::OpenPopup("##messages_view_menu");
  if (ImGui::BeginPopup("##messages_view_menu")) {
    // mirrors MessagesWidget::menuAboutToShow(): per-column show/hide (Name can't be hidden)
    ImGui::MenuItem(COLUMN_LABELS[COL_NAME], nullptr, &g_state.column_visible[COL_NAME], false);
    for (int col = 1; col < NUM_COLUMNS; ++col) {
      ImGui::MenuItem(COLUMN_LABELS[col], nullptr, &g_state.column_visible[col]);
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Multi-Line bytes", nullptr, &settings.multiple_lines_hex)) {
      // row height / DATA column width recompute live from settings.multiple_lines_hex below
    }
    if (ImGui::MenuItem("Show inactive messages", nullptr, &g_state.show_inactive_messages)) {
      g_state.dirty = true;
    }
    ImGui::EndPopup();
  }
}

void draw_filter_row() {
  ImGui::TableNextRow();
  for (int col = 0; col < NUM_COLUMNS; ++col) {
    if (!ImGui::TableSetColumnIndex(col)) continue;
    ImGui::PushID(col);
    ImGui::SetNextItemWidth(-FLT_MIN);
    const bool changed =
        ImGui::InputTextWithHint("##filter", COLUMN_FILTER_HINTS[col], g_state.filter_buf[col], sizeof(g_state.filter_buf[col]));
    if (changed) {
      if (g_state.filter_buf[col][0] != '\0') {
        g_state.filters[col] = g_state.filter_buf[col];
      } else {
        g_state.filters.erase(col);
      }
      refilter_and_sort();
    }
    ImGui::PopID();
  }
}

// keeps up/down-arrow row navigation out of text-entry focus (filter boxes)
void handle_keyboard_nav(AppState &app) {
  const ImGuiIO &io = ImGui::GetIO();
  if (io.WantTextInput || g_state.rows.empty()) return;
  if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) return;

  int dir = 0;
  if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) dir = 1;
  else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) dir = -1;
  if (dir == 0) return;

  int idx = 0;
  if (app.selected_msg_id) {
    auto it = std::find_if(g_state.rows.begin(), g_state.rows.end(),
                           [&](const Row &r) { return r.id == *app.selected_msg_id; });
    idx = (it != g_state.rows.end()) ? static_cast<int>(std::distance(g_state.rows.begin(), it)) + dir : 0;
  }
  idx = std::clamp(idx, 0, static_cast<int>(g_state.rows.size()) - 1);
  app.selected_msg_id = g_state.rows[static_cast<size_t>(idx)].id;
  g_state.pending_scroll_index = idx;
}

void draw_row(AppState &app, const Row &row, int row_index, float row_height) {
  const CanData &data = can->lastMessage(row.id);
  const bool has_data = row.id.source != INVALID_SOURCE;
  const bool active = can->isMessageActive(row.id);
  const bool selected = app.selected_msg_id.has_value() && *app.selected_msg_id == row.id;

  ImGui::TableNextRow(ImGuiTableRowFlags_None, row_height);
  ImGui::PushID(row_index);
  if (!active) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));

  // theme.cc's light-theme Header color is a near-background hairline (by
  // design, for the docking tab bar); boost it here so the selected row
  // actually reads as selected against the light background too.
  if (selected) {
    ImVec4 accent = ImGui::GetStyleColorVec4(ImGuiCol_TabSelectedOverline);
    accent.w = 0.65f;
    ImGui::PushStyleColor(ImGuiCol_Header, accent);
    accent.w = 0.8f;
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, accent);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, accent);
  }

  ImGui::TableSetColumnIndex(COL_NAME);
  if (ImGui::Selectable("##row_select", selected,
                        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap,
                        ImVec2(0.0f, row_height))) {
    app.selected_msg_id = row.id;
  }
  if (selected) ImGui::PopStyleColor(3);
  if (row_index == g_state.pending_scroll_index) {
    ImGui::SetScrollHereY(0.35f);
    g_state.pending_scroll_index = -1;
  }
  ImGui::SameLine();
  ImGui::TextUnformatted(row.name.c_str());
  if (ImGui::IsItemHovered()) {
    if (cabana::Msg *m = dbc()->msg(row.id); m != nullptr && !m->comment.empty()) {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(row.name.c_str());
      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
      ImGui::TextWrapped("%s", m->comment.c_str());
      ImGui::PopStyleColor();
      ImGui::EndTooltip();
    }
  }

  if (ImGui::TableSetColumnIndex(COL_SOURCE)) {
    if (has_data) ImGui::Text("%u", static_cast<unsigned>(row.id.source));
    else ImGui::TextUnformatted("N/A");
  }

  if (ImGui::TableSetColumnIndex(COL_ADDRESS)) {
    push_mono_font();
    ImGui::Text("0x%02X", row.id.address);
    pop_mono_font();
  }

  if (ImGui::TableSetColumnIndex(COL_NODE)) {
    ImGui::TextUnformatted(row.node.c_str());
  }

  if (ImGui::TableSetColumnIndex(COL_FREQ)) {
    if (has_data) ImGui::TextUnformatted(format_freq(data.freq).c_str());
    else ImGui::TextUnformatted("N/A");
  }

  if (ImGui::TableSetColumnIndex(COL_COUNT)) {
    if (has_data) ImGui::Text("%u", data.count);
    else ImGui::TextUnformatted("N/A");
  }

  if (ImGui::TableSetColumnIndex(COL_DATA)) {
    if (has_data) {
      const ColorRGBA *colors = (active && !data.colors.empty()) ? data.colors.data() : nullptr;
      draw_message_bytes(data.dat.data(), data.dat.size(), colors, settings.multiple_lines_hex, ImGui::GetColumnWidth());
    } else {
      ImGui::TextUnformatted("N/A");
    }
  }

  if (!active) ImGui::PopStyleColor();
  ImGui::PopID();
}

void draw_table(AppState &app) {
  // MessageView::updateBytesSectionSize(): DATA column width (single-line) /
  // row height (multi-line) sized off the widest live message.
  size_t max_bytes = 8;
  for (const auto &[id, m] : can->lastMessages()) max_bytes = std::max(max_bytes, m.dat.size());

  push_mono_font();
  const float byte_w = ImGui::CalcTextSize("00").x + 8.0f;
  const float byte_h = ImGui::GetTextLineHeight() + 4.0f;
  pop_mono_font();

  const bool multi_line = settings.multiple_lines_hex;
  const int bytes_per_row = multi_line ? static_cast<int>(std::min<size_t>(8, max_bytes)) : static_cast<int>(max_bytes);
  const int line_rows = multi_line ? static_cast<int>((max_bytes + 7) / 8) : 1;
  const float row_height = std::max(ImGui::GetFrameHeight(), byte_h * static_cast<float>(std::max(1, line_rows)));
  const float data_col_w = static_cast<float>(std::max(1, bytes_per_row)) * byte_w + 8.0f;

  handle_keyboard_nav(app);

  // Qt's QTreeView scrolls the current index into view on selection change
  // (both from our own row clicks and from external callers, e.g. --demo's
  // auto-select-busiest-message capture path).
  if (app.selected_msg_id != g_state.last_synced_selection) {
    g_state.last_synced_selection = app.selected_msg_id;
    if (app.selected_msg_id) {
      auto it = std::find_if(g_state.rows.begin(), g_state.rows.end(),
                             [&](const Row &r) { return r.id == *app.selected_msg_id; });
      if (it != g_state.rows.end()) g_state.pending_scroll_index = static_cast<int>(std::distance(g_state.rows.begin(), it));
    }
  }

  // NOTE: WidthStretch columns (Name/Node) don't combine well with ScrollX in
  // Dear ImGui tables (stretch has no fixed budget to divide when the table
  // can grow unbounded), so we don't set ScrollX here -- the Bytes column
  // just clips at the right edge for very wide (CAN FD) messages instead of
  // Qt's shift+wheel horizontal scroll. See report for detail.
  constexpr ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_Sortable |
                                    ImGuiTableFlags_Resizable | ImGuiTableFlags_Hideable |
                                    ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter;
  if (!ImGui::BeginTable("##messages_table", NUM_COLUMNS, flags, ImGui::GetContentRegionAvail())) return;

  ImGui::TableSetupScrollFreeze(0, 2);  // header row + filter row stay pinned while scrolling
  // Fixed default widths: the dock is narrow enough that stretch weights
  // starve the Name column below usability; all are user-resizable and
  // persisted via imgui.ini.
  ImGui::TableSetupColumn(COLUMN_LABELS[COL_NAME],
                          ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort |
                              ImGuiTableColumnFlags_PreferSortAscending | ImGuiTableColumnFlags_NoHide,
                          130.0f);
  ImGui::TableSetupColumn(COLUMN_LABELS[COL_SOURCE], ImGuiTableColumnFlags_WidthFixed, 40.0f);
  ImGui::TableSetupColumn(COLUMN_LABELS[COL_ADDRESS], ImGuiTableColumnFlags_WidthFixed, 60.0f);
  ImGui::TableSetupColumn(COLUMN_LABELS[COL_NODE], ImGuiTableColumnFlags_WidthFixed, 56.0f);
  ImGui::TableSetupColumn(COLUMN_LABELS[COL_FREQ], ImGuiTableColumnFlags_WidthFixed, 52.0f);
  ImGui::TableSetupColumn(COLUMN_LABELS[COL_COUNT], ImGuiTableColumnFlags_WidthFixed, 60.0f);
  ImGui::TableSetupColumn(COLUMN_LABELS[COL_DATA], ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort,
                          data_col_w);

  for (int col = 0; col < NUM_COLUMNS; ++col) {
    ImGui::TableSetColumnEnabled(col, g_state.column_visible[col]);
  }

  ImGui::TableHeadersRow();

  if (ImGuiTableSortSpecs *specs = ImGui::TableGetSortSpecs()) {
    if (specs->SpecsDirty && specs->SpecsCount > 0) {
      g_state.sort_column = specs->Specs[0].ColumnIndex;
      g_state.sort_ascending = specs->Specs[0].SortDirection != ImGuiSortDirection_Descending;
      refilter_and_sort();
      specs->SpecsDirty = false;
    }
  }

  draw_filter_row();

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(g_state.rows.size()), row_height);
  if (g_state.pending_scroll_index >= 0 && g_state.pending_scroll_index < static_cast<int>(g_state.rows.size())) {
    clipper.IncludeItemByIndex(g_state.pending_scroll_index);
  }
  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      draw_row(app, g_state.rows[static_cast<size_t>(i)], i, row_height);
    }
  }
  ImGui::EndTable();
}

}  // namespace

// -- shared widget contracts (app.h) ---------------------------------------

ImU32 to_im_color(const ColorRGBA &c) {
  return IM_COL32(c.r, c.g, c.b, c.a);
}

// Mirrors MessageBytesDelegate::paint(): hex byte pairs in mono font, each
// byte's background filled with its (already-faded) highlight color when
// present. `colors` may be null (e.g. inactive rows suppress highlighting,
// or the caller has no highlight data at all).
void draw_message_bytes(const uint8_t *dat, size_t size, const ColorRGBA *colors, bool multiple_lines, float avail_width) {
  push_mono_font();
  const ImVec2 cell(ImGui::CalcTextSize("00").x + 8.0f, ImGui::GetTextLineHeight() + 4.0f);
  const int max_cols = multiple_lines
                          ? std::max(1, std::min<int>(8, static_cast<int>(avail_width / std::max(1.0f, cell.x))))
                          : static_cast<int>(size);

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImU32 text_color = ImGui::GetColorU32(ImGuiCol_Text);

  for (size_t i = 0; i < size; ++i) {
    const int col = multiple_lines ? static_cast<int>(i) % max_cols : static_cast<int>(i);
    const int row = multiple_lines ? static_cast<int>(i) / max_cols : 0;
    const ImVec2 p0(origin.x + static_cast<float>(col) * cell.x, origin.y + static_cast<float>(row) * cell.y);
    const ImVec2 p1(p0.x + cell.x, p0.y + cell.y);

    if (colors != nullptr && colors[i].a > 0) {
      draw_list->AddRectFilled(p0, p1, to_im_color(colors[i]));
    }

    char buf[3];
    std::snprintf(buf, sizeof(buf), "%02X", dat[i]);
    const ImVec2 text_size = ImGui::CalcTextSize(buf);
    const ImVec2 text_pos(p0.x + (cell.x - text_size.x) * 0.5f, p0.y + (cell.y - text_size.y) * 0.5f);
    draw_list->AddText(text_pos, text_color, buf);
  }

  const int rows = multiple_lines ? static_cast<int>((size + static_cast<size_t>(max_cols) - 1) / static_cast<size_t>(max_cols)) : 1;
  const float used_cols = static_cast<float>(std::min<size_t>(static_cast<size_t>(max_cols), size));
  ImGui::Dummy(ImVec2(std::max(1.0f, used_cols * cell.x), std::max(1.0f, static_cast<float>(rows) * cell.y)));
  pop_mono_font();
}

void draw_messages_panel(AppState &app) {
  ensure_connected();
  if (g_state.dirty) {
    refilter_and_sort();
    g_state.dirty = false;
  }

  if (ImGui::Begin(MESSAGES_WINDOW_TITLE)) {
    draw_toolbar(app);
    ImGui::Separator();
    draw_table(app);
  }
  ImGui::End();
}
