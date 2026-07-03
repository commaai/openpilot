// Signal editor panel -- ImGui port of tools/cabana/signalview.{h,cc}
// (SignalModel/SignalView/SignalItemDelegate/ValueDescriptionDlg), the
// frozen Qt reference this file mirrors for parity. Sparkline rendering
// ports tools/cabana/chart/sparkline.{h,cc}. All edits are pushed through
// UndoStack::push(new EditSignalCommand(...)) -- see commit_signal_edit()
// below, which mirrors SignalModel::saveSignal() exactly (including the
// duplicate-name check and the start_bit flip on endianness change).
//
// No persistent model like Qt's SignalModel: the row list is rebuilt from
// dbc()->msg(id)->getSignals() + a substring filter every frame (cheap --
// signal counts per message rarely exceed a few dozen). Per-signal UI state
// (expanded/extra-info flags, in-progress edit buffers) is kept in a
// pointer-keyed side table (signals are heap objects that survive in-place
// edits/renames -- see cabana::Msg::updateSignal(); only add/remove change
// the pointer set) that's pruned on the DBC events that can invalidate it.

#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "tools/cabana/commands.h"
#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/imgui/signal_state.h"
#include "tools/cabana/settings.h"

namespace {

// -- small string helpers (no Qt available in this Qt-free core) ----------

bool contains_ci(const std::string &haystack, const std::string &needle) {
  if (needle.empty()) return true;
  auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(), [](char a, char b) {
    return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
  });
  return it != haystack.end();
}

std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\n\r");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\n\r");
  return s.substr(a, b - a + 1);
}

std::string elide_text(const std::string &text, float max_width) {
  if (ImGui::CalcTextSize(text.c_str()).x <= max_width) return text;
  std::string s = text;
  while (!s.empty() && ImGui::CalcTextSize((s + "...").c_str()).x > max_width) s.pop_back();
  return s + "...";
}

// mirrors signalview.cc's static signalTypeToString()
std::string signal_type_to_string(cabana::Signal::Type type) {
  if (type == cabana::Signal::Type::Multiplexor) return "Multiplexor Signal";
  if (type == cabana::Signal::Type::Multiplexed) return "Multiplexed Signal";
  return "Normal Signal";
}

// mirrors utils::formatSeconds() as used for the sparkline range label
// (settings.sparkline_range is clamped to [1,30], so this is always mm:ss)
std::string format_seconds(int sec) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%02d:%02d", sec / 60, sec % 60);
  return buf;
}

// mirrors SignalModel::data()'s Item::Desc join: `val "desc" val "desc" ...`
std::string format_val_desc(const ValueDescription &vd) {
  std::string out;
  for (const auto &[val, desc] : vd) {
    if (!out.empty()) out += ' ';
    out += doubleToString(val) + " \"" + desc + "\"";
  }
  return out;
}

// NameValidator port: word chars only, space -> underscore (matches
// util.cc's `input.replace(' ', '_')` before the ^(\w+) regex validate)
int name_char_filter(ImGuiInputTextCallbackData *data) {
  if (data->EventChar == ' ') {
    data->EventChar = '_';
    return 0;
  }
  if (data->EventChar < 256 && (std::isalnum(static_cast<unsigned char>(data->EventChar)) || data->EventChar == '_')) return 0;
  return 1;  // discard
}

// -- per-signal UI state (expand flags + in-progress edit buffers) --------

struct SigUiState {
  bool expanded = false;
  bool extra_info_open = false;

  char name_buf[128] = {};
  bool name_active = false;
  char node_buf[128] = {};
  bool node_active = false;
  char offset_buf[64] = {};
  bool offset_active = false;
  char factor_buf[64] = {};
  bool factor_active = false;
  char min_buf[64] = {};
  bool min_active = false;
  char max_buf[64] = {};
  bool max_active = false;
  char unit_buf[64] = {};
  bool unit_active = false;
  char comment_buf[512] = {};
  bool comment_active = false;
  int size_val = 0;
  bool size_active = false;
  int mux_val = 0;
  bool mux_active = false;
};

// -- sparkline cache: ports Sparkline::update()/render() (chart/sparkline.cc) --

struct SparklineCache {
  std::vector<ImVec2> points;         // raw (t_sec_from_range_start, value)
  std::vector<ImVec2> render_points;  // normalized to the [0,size] box, cached at last compute
  bool draw_points = false;
  bool empty = true;
  double min_val = 0, max_val = 0;    // may be padded +-1 for a flat line -- mirrors Qt's
                                       // Sparkline::render() mutating these fields, which
                                       // SignalItemDelegate::paint() then uses for the label text
  double freq = 0;
  ImVec2 last_size{0.0f, 0.0f};
};

void render_sparkline_points(SparklineCache &c, int range, ImVec2 size) {
  const bool is_flat = (c.min_val == c.max_val);
  if (is_flat) {
    c.min_val -= 1.0;
    c.max_val += 1.0;
  }
  const double xscale = (size.x - 1.0) / std::max(1, range);
  const double yscale = (size.y - 3.0) / (c.max_val - c.min_val);
  c.draw_points = (c.points.back().x * xscale / static_cast<double>(c.points.size())) > 8.0;

  c.render_points.clear();
  c.render_points.reserve(c.points.size());
  const auto to_render = [&](const ImVec2 &p) {
    return ImVec2(static_cast<float>(p.x * xscale), static_cast<float>(1.0 + (c.max_val - p.y) * yscale));
  };
  if (c.draw_points) {
    for (const ImVec2 &p : c.points) c.render_points.push_back(to_render(p));
  } else if (is_flat) {
    const float y = static_cast<float>(size.y / 2.0);
    c.render_points.emplace_back(0.0f, y);
    c.render_points.emplace_back(static_cast<float>(c.points.back().x * xscale), y);
  } else {
    double prev_y = c.points.front().y;
    c.render_points.push_back(to_render(c.points.front()));
    bool in_flat = false;
    for (size_t i = 1; i < c.points.size(); ++i) {
      const double y = c.points[i].y;
      if (std::abs(y - prev_y) < 1e-6) {
        in_flat = true;
      } else {
        if (in_flat) c.render_points.push_back(to_render(ImVec2(c.points[i - 1].x, static_cast<float>(prev_y))));
        c.render_points.push_back(to_render(c.points[i]));
        in_flat = false;
      }
      prev_y = y;
    }
    if (in_flat) c.render_points.push_back(to_render(ImVec2(c.points.back().x, static_cast<float>(prev_y))));
  }
}

void update_sparkline_cache(const cabana::Signal *sig, const MessageId &msg_id, int range, ImVec2 size, SparklineCache &c) {
  c.last_size = size;
  const CanData &last = can->lastMessage(msg_id);
  auto [first, last_it] = can->eventsInRange(msg_id, std::make_pair(last.ts - range, last.ts));
  if (first == last_it || size.x <= 0.0f || size.y <= 0.0f) {
    c.empty = true;
    c.points.clear();
    return;
  }

  c.points.clear();
  c.points.reserve(std::distance(first, last_it));
  double min_v = std::numeric_limits<double>::max();
  double max_v = std::numeric_limits<double>::lowest();
  const uint64_t start_time = (*first)->mono_time;
  double value = 0.0;
  for (auto it = first; it != last_it; ++it) {
    if (sig->getValue((*it)->dat, (*it)->size, &value)) {
      min_v = std::min(min_v, value);
      max_v = std::max(max_v, value);
      c.points.emplace_back(static_cast<float>(((*it)->mono_time - start_time) / 1e9), static_cast<float>(value));
    }
  }
  if (c.points.empty()) {
    c.empty = true;
    return;
  }
  c.min_val = min_v;
  c.max_val = max_v;
  c.freq = c.points.size() / std::max(static_cast<double>(c.points.back().x - c.points.front().x), 1.0);
  render_sparkline_points(c, range, size);
  c.empty = false;
}

void draw_sparkline_widget(const SparklineCache &c, const ColorRGBA &color, ImVec2 pos) {
  if (c.empty || c.render_points.empty()) return;
  static thread_local std::vector<ImVec2> screen_pts;
  screen_pts.clear();
  screen_pts.reserve(c.render_points.size());
  for (const ImVec2 &p : c.render_points) screen_pts.emplace_back(pos.x + p.x, pos.y + p.y);

  ImDrawList *dl = ImGui::GetWindowDrawList();
  const ImU32 col = IM_COL32(color.r, color.g, color.b, color.a);
  dl->AddPolyline(screen_pts.data(), static_cast<int>(screen_pts.size()), col, 0, 1.0f);
  if (c.draw_points) {
    for (const ImVec2 &p : screen_pts) dl->AddCircleFilled(p, 1.5f, col);
  } else {
    dl->AddCircleFilled(screen_pts.back(), 1.5f, col);
  }
}

// -- value-description modal: ports ValueDescriptionDlg (add/remove/OK/Cancel) --
// Deviation from Qt: each row gets its own remove button instead of a
// toolbar button acting on a table selection -- functionally equivalent
// add/remove, simpler to build in immediate mode. See report.

struct ValDescRow {
  char val[32] = {};
  char desc[256] = {};
};

struct ValueDescEditorState {
  bool open_request = false;
  MessageId msg_id{};
  const cabana::Signal *sig = nullptr;
  std::string title;
  std::vector<ValDescRow> rows;
};

// -- error modal: ports the QMessageBox::warning in SignalModel::saveSignal() --

struct ErrorModalState {
  bool open_request = false;
  std::string message;
};

// -- panel state --------------------------------------------------------

struct SignalViewState {
  bool has_msg_id = false;
  MessageId msg_id{};
  char filter_buf[128] = {};
  std::string filter;

  bool sparkline_dirty = true;
  std::unordered_map<const cabana::Signal *, SigUiState> ui;
  std::unordered_map<const cabana::Signal *, SparklineCache> sparklines;
  const cabana::Signal *pending_scroll_sig = nullptr;

  ValueDescEditorState val_desc_editor;
  ErrorModalState error_modal;
};

SignalViewState g_state;

SigUiState &ui_state_for(const cabana::Signal *sig) { return g_state.ui[sig]; }

void reset_for_new_message(const MessageId &id) {
  g_state.has_msg_id = true;
  g_state.msg_id = id;
  g_state.filter_buf[0] = '\0';
  g_state.filter.clear();
  g_state.ui.clear();
  g_state.sparklines.clear();
  g_state.sparkline_dirty = true;
  g_state.pending_scroll_sig = nullptr;
}

void forget_signal(const cabana::Signal *sig) {
  g_state.ui.erase(sig);
  g_state.sparklines.erase(sig);
  if (selected_signal() == sig) set_selected_signal(nullptr);
  if (hovered_signal() == sig) set_hovered_signal(nullptr);
}

void ensure_connected() {
  // Global objects (dbc(), UndoStack::instance()): connect once, keep the
  // once-guard.
  static bool connected = false;
  if (!connected) {
    connected = true;
    dbc()->DBCFileChanged.connect([]() {
      g_state.ui.clear();
      g_state.sparklines.clear();
      g_state.sparkline_dirty = true;
      set_selected_signal(nullptr);
      set_hovered_signal(nullptr);
    });
    // mirrors SignalModel::handleMsgChanged(): full refresh when the current message itself changes
    dbc()->msgUpdated.connect([](MessageId id) {
      if (g_state.has_msg_id && id.address == g_state.msg_id.address) {
        g_state.ui.clear();
        g_state.sparkline_dirty = true;
      }
    });
    dbc()->msgRemoved.connect([](MessageId id) {
      if (g_state.has_msg_id && id.address == g_state.msg_id.address) {
        g_state.ui.clear();
        g_state.sparklines.clear();
      }
    });
    // mirrors SignalView::handleSignalAdded(): select (don't force-expand) the new row
    dbc()->signalAdded.connect([](MessageId id, const cabana::Signal *sig) {
      if (!g_state.has_msg_id || id.address != g_state.msg_id.address) return;
      g_state.sparkline_dirty = true;
      set_selected_signal(sig, /*from_binary_view=*/false);
      g_state.pending_scroll_sig = sig;
    });
    dbc()->signalUpdated.connect([](const cabana::Signal * /*sig*/) { g_state.sparkline_dirty = true; });
    dbc()->signalRemoved.connect([](const cabana::Signal *sig) {
      forget_signal(sig);
      g_state.sparkline_dirty = true;
    });
    UndoStack::instance()->indexChanged.connect([](int) { g_state.sparkline_dirty = true; });
  }

  // The stream: File > Open Stream can swap `can` to a brand-new
  // AbstractStream at runtime (see stream_selector.cc's swap_stream()), so
  // rebind to whichever instance is current rather than connecting once.
  static AbstractStream *wired_stream = nullptr;
  if (wired_stream != can) {
    wired_stream = can;
    // mirrors QObject::connect(can, &AbstractStream::msgsReceived, this, &SignalView::updateState)
    can->msgsReceived.connect([](const std::set<MessageId> *msgs, bool /*has_new_ids*/) {
      if (!g_state.has_msg_id) return;
      if (msgs != nullptr && !msgs->count(g_state.msg_id)) return;
      g_state.sparkline_dirty = true;
    });
    // g_state.sparklines holds copied doubles (not stream pointers), so it's
    // stale-but-safe across a swap -- just flag it dirty so the next draw
    // recomputes every entry from the new stream's events instead of showing
    // frozen values from the old one.
    g_state.sparkline_dirty = true;
  }
}

// mirrors SignalModel::saveSignal(): duplicate-name check + start_bit flip
// on endianness change, then UndoStack::push(new EditSignalCommand(...)).
bool commit_signal_edit(const MessageId &msg_id, const cabana::Signal *origin, cabana::Signal s) {
  cabana::Msg *msg = dbc()->msg(msg_id);
  if (msg == nullptr) return false;
  if (s.name != origin->name && msg->sig(s.name) != nullptr) {
    g_state.error_modal.message = "There is already a signal with the same name '" + s.name + "'";
    g_state.error_modal.open_request = true;
    return false;
  }
  if (s.is_little_endian != origin->is_little_endian) {
    s.start_bit = flipBitPos(s.start_bit);
  }
  UndoStack::push(new EditSignalCommand(msg_id, origin, s));
  return true;
}

void open_value_desc_editor(const MessageId &msg_id, const cabana::Signal *sig) {
  ValueDescEditorState &ed = g_state.val_desc_editor;
  ed.msg_id = msg_id;
  ed.sig = sig;
  ed.title = sig->name;
  ed.rows.clear();
  for (const auto &[val, desc] : sig->val_desc) {
    ValDescRow row;
    std::snprintf(row.val, sizeof(row.val), "%s", doubleToString(val).c_str());
    std::snprintf(row.desc, sizeof(row.desc), "%s", desc.c_str());
    ed.rows.push_back(row);
  }
  ed.open_request = true;
}

void draw_error_modal() {
  ErrorModalState &err = g_state.error_modal;
  if (err.open_request) {
    ImGui::OpenPopup("Failed to save signal");
    err.open_request = false;
  }
  ImGui::SetNextWindowSize(ImVec2(360.0f, 0.0f), ImGuiCond_Appearing);
  if (ImGui::BeginPopupModal("Failed to save signal", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped("%s", err.message.c_str());
    ImGui::Spacing();
    if (ImGui::Button("OK") || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

// mirrors ValueDescriptionDlg
void draw_value_desc_modal() {
  ValueDescEditorState &ed = g_state.val_desc_editor;
  if (ed.open_request) {
    ImGui::OpenPopup("Value Descriptions");
    ed.open_request = false;
  }
  ImGui::SetNextWindowSize(ImVec2(500.0f, 400.0f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Value Descriptions", nullptr, ImGuiWindowFlags_None)) return;

  ImGui::TextDisabled("%s", ed.title.c_str());
  if (ImGui::Button("+")) ed.rows.push_back({});
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Add value description");

  const float table_h = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() - ImGui::GetStyle().ItemSpacing.y;
  if (ImGui::BeginTable("##val_desc_table", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                        ImVec2(0.0f, std::max(60.0f, table_h)))) {
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("##remove", ImGuiTableColumnFlags_WidthFixed, 28.0f);
    ImGui::TableHeadersRow();

    int remove_index = -1;
    for (int i = 0; i < static_cast<int>(ed.rows.size()); ++i) {
      ImGui::PushID(i);
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::InputText("##val", ed.rows[i].val, sizeof(ed.rows[i].val), ImGuiInputTextFlags_CharsScientific);
      ImGui::TableSetColumnIndex(1);
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::InputText("##desc", ed.rows[i].desc, sizeof(ed.rows[i].desc));
      ImGui::TableSetColumnIndex(2);
      if (ImGui::SmallButton("x")) remove_index = i;
      ImGui::PopID();
    }
    if (remove_index >= 0) ed.rows.erase(ed.rows.begin() + remove_index);
    ImGui::EndTable();
  }

  if (ImGui::Button("OK")) {
    cabana::Signal s = *ed.sig;
    s.val_desc.clear();
    for (const ValDescRow &row : ed.rows) {
      std::string val_s = trim(row.val);
      std::string desc_s = trim(row.desc);
      if (!val_s.empty() && !desc_s.empty()) {
        s.val_desc.push_back({std::strtod(val_s.c_str(), nullptr), desc_s});
      }
    }
    commit_signal_edit(ed.msg_id, ed.sig, s);
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
}

// -- property form (mirrors SignalModel::insertItem()'s per-signal rows) --

void begin_field_row(const char *label, float indent, float col2_x) {
  ImGui::SetCursorPosX(indent);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(label);
  ImGui::SameLine(col2_x);
}

void draw_double_field(const char *str_id, const char *label, float indent, float col2, const MessageId &msg_id, const cabana::Signal *sig,
                       char *buf, size_t buf_size, bool *active, double current, void (*set_fn)(cabana::Signal &, double)) {
  begin_field_row(label, indent, col2);
  if (!*active) std::snprintf(buf, buf_size, "%s", doubleToString(current).c_str());
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::PushID(str_id);
  ImGui::InputText("##v", buf, buf_size, ImGuiInputTextFlags_CharsScientific);
  *active = ImGui::IsItemActive();
  const bool commit = ImGui::IsItemDeactivatedAfterEdit();
  ImGui::PopID();
  if (commit) {
    cabana::Signal s = *sig;
    set_fn(s, std::strtod(buf, nullptr));
    commit_signal_edit(msg_id, sig, s);
  }
}

void draw_text_field(const char *str_id, const char *label, float indent, float col2, const MessageId &msg_id, const cabana::Signal *sig,
                     char *buf, size_t buf_size, bool *active, const std::string &current, void (*set_fn)(cabana::Signal &, std::string)) {
  begin_field_row(label, indent, col2);
  if (!*active) std::snprintf(buf, buf_size, "%s", current.c_str());
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::PushID(str_id);
  ImGui::InputText("##v", buf, buf_size);
  *active = ImGui::IsItemActive();
  const bool commit = ImGui::IsItemDeactivatedAfterEdit();
  ImGui::PopID();
  if (commit) {
    cabana::Signal s = *sig;
    set_fn(s, trim(buf));
    commit_signal_edit(msg_id, sig, s);
  }
}

void draw_signal_property_form(const MessageId &msg_id, const cabana::Signal *sig, SigUiState &ui) {
  const float indent = 30.0f;
  const float extra_indent = 50.0f;
  const float col2 = std::max(140.0f, ImGui::GetContentRegionAvail().x * 0.34f);

  // Name
  begin_field_row("Name", indent, col2);
  if (!ui.name_active) std::snprintf(ui.name_buf, sizeof(ui.name_buf), "%s", sig->name.c_str());
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputText("##name", ui.name_buf, sizeof(ui.name_buf), ImGuiInputTextFlags_CallbackCharFilter, name_char_filter);
  ui.name_active = ImGui::IsItemActive();
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    cabana::Signal s = *sig;
    s.name = ui.name_buf;
    commit_signal_edit(msg_id, sig, s);
  }

  // Size
  begin_field_row("Size", indent, col2);
  if (!ui.size_active) ui.size_val = sig->size;
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputInt("##size", &ui.size_val, 1, 8);
  ui.size_active = ImGui::IsItemActive();
  ui.size_val = std::clamp(ui.size_val, 1, CAN_MAX_DATA_BYTES);
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    cabana::Signal s = *sig;
    s.size = ui.size_val;
    commit_signal_edit(msg_id, sig, s);
  }

  // Receiver Nodes
  draw_text_field("node", "Receiver Nodes", indent, col2, msg_id, sig, ui.node_buf, sizeof(ui.node_buf), &ui.node_active, sig->receiver_name,
                  [](cabana::Signal &s, std::string v) { s.receiver_name = std::move(v); });

  // Little Endian
  begin_field_row("Little Endian", indent, col2);
  bool le = sig->is_little_endian;
  if (ImGui::Checkbox("##le", &le)) {
    cabana::Signal s = *sig;
    s.is_little_endian = le;
    commit_signal_edit(msg_id, sig, s);
  }

  // Signed
  begin_field_row("Signed", indent, col2);
  bool is_signed = sig->is_signed;
  if (ImGui::Checkbox("##signed", &is_signed)) {
    cabana::Signal s = *sig;
    s.is_signed = is_signed;
    commit_signal_edit(msg_id, sig, s);
  }

  // Offset
  draw_double_field("offset", "Offset", indent, col2, msg_id, sig, ui.offset_buf, sizeof(ui.offset_buf), &ui.offset_active, sig->offset,
                    [](cabana::Signal &s, double v) { s.offset = v; });

  // Factor
  draw_double_field("factor", "Factor", indent, col2, msg_id, sig, ui.factor_buf, sizeof(ui.factor_buf), &ui.factor_active, sig->factor,
                    [](cabana::Signal &s, double v) { s.factor = v; });

  // Type (Normal / Multiplexor / Multiplexed) -- mirrors SignalItemDelegate::createEditor()'s combo population rules
  begin_field_row("Type", indent, col2);
  {
    struct TypeOption {
      cabana::Signal::Type type;
      std::string label;
    };
    std::vector<TypeOption> options{{cabana::Signal::Type::Normal, signal_type_to_string(cabana::Signal::Type::Normal)}};
    cabana::Msg *msg = dbc()->msg(msg_id);
    if (msg != nullptr && msg->multiplexor == nullptr) {
      options.push_back({cabana::Signal::Type::Multiplexor, signal_type_to_string(cabana::Signal::Type::Multiplexor)});
    } else if (sig->type != cabana::Signal::Type::Multiplexor) {
      options.push_back({cabana::Signal::Type::Multiplexed, signal_type_to_string(cabana::Signal::Type::Multiplexed)});
    }
    if (std::none_of(options.begin(), options.end(), [&](const TypeOption &o) { return o.type == sig->type; })) {
      options.push_back({sig->type, signal_type_to_string(sig->type)});
    }
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##type", signal_type_to_string(sig->type).c_str())) {
      for (const TypeOption &opt : options) {
        const bool is_current = (opt.type == sig->type);
        if (ImGui::Selectable(opt.label.c_str(), is_current) && !is_current) {
          cabana::Signal s = *sig;
          s.type = opt.type;
          commit_signal_edit(msg_id, sig, s);
        }
      }
      ImGui::EndCombo();
    }
  }

  // Multiplex Value -- editable only for Multiplexed signals
  begin_field_row("Multiplex Value", indent, col2);
  ImGui::BeginDisabled(sig->type != cabana::Signal::Type::Multiplexed);
  if (!ui.mux_active) ui.mux_val = sig->multiplex_value;
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputInt("##mux", &ui.mux_val, 1, 1);
  ui.mux_active = ImGui::IsItemActive();
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    cabana::Signal s = *sig;
    s.multiplex_value = ui.mux_val;
    commit_signal_edit(msg_id, sig, s);
  }
  ImGui::EndDisabled();

  // Extra Info -- nested expandable node (Unit/Comment/Min/Max/Value Table)
  ImGui::SetCursorPosX(indent);
  {
    ImGui::PushID("extra_info_toggle");
    const bool clicked = ImGui::Selectable(ui.extra_info_open ? "v Extra Info" : "> Extra Info", false);
    if (clicked) ui.extra_info_open = !ui.extra_info_open;
    ImGui::PopID();
  }

  if (ui.extra_info_open) {
    draw_text_field("unit", "Unit", extra_indent, col2, msg_id, sig, ui.unit_buf, sizeof(ui.unit_buf), &ui.unit_active, sig->unit,
                    [](cabana::Signal &s, std::string v) { s.unit = std::move(v); });
    draw_text_field("comment", "Comment", extra_indent, col2, msg_id, sig, ui.comment_buf, sizeof(ui.comment_buf), &ui.comment_active,
                    sig->comment, [](cabana::Signal &s, std::string v) { s.comment = std::move(v); });
    draw_double_field("min", "Minimum Value", extra_indent, col2, msg_id, sig, ui.min_buf, sizeof(ui.min_buf), &ui.min_active, sig->min,
                      [](cabana::Signal &s, double v) { s.min = v; });
    draw_double_field("max", "Maximum Value", extra_indent, col2, msg_id, sig, ui.max_buf, sizeof(ui.max_buf), &ui.max_active, sig->max,
                      [](cabana::Signal &s, double v) { s.max = v; });

    begin_field_row("Value Table", extra_indent, col2);
    std::string summary = format_val_desc(sig->val_desc);
    if (summary.empty()) summary = "(none)";
    ImGui::PushID("val_desc_btn");
    if (ImGui::Selectable(elide_text(summary, ImGui::GetContentRegionAvail().x).c_str(), false)) {
      open_value_desc_editor(msg_id, sig);
    }
    ImGui::PopID();
    if (ImGui::IsItemHovered() && !summary.empty()) ImGui::SetTooltip("%s", summary.c_str());
  }
}

// -- collapsed row: color chip + name + sparkline + value + plot/remove --

constexpr float ROW_HEIGHT = 26.0f;
constexpr float CHIP_SIZE = 18.0f;
constexpr float BTN_SIZE = 20.0f;
constexpr float SPARKLINE_W = 90.0f;
constexpr float SPARKLINE_H = 20.0f;

ImU32 chip_color(const ColorRGBA &c, bool highlight) {
  if (!highlight) return IM_COL32(c.r, c.g, c.b, c.a);
  const auto scale = [](uint8_t v) { return static_cast<uint8_t>(std::clamp(v * 0.8f, 0.0f, 255.0f)); };
  return IM_COL32(scale(c.r), scale(c.g), scale(c.b), c.a);
}

void toggle_expand(const cabana::Signal *sig, SigUiState &ui) {
  ui.expanded = !ui.expanded;
  set_selected_signal(sig, /*from_binary_view=*/false);
}

// mirrors SignalItemDelegate::paint()'s column-0 color label + multiplex indicator.
// Returns the x where the signal name should start.
float draw_row_left(ImVec2 origin, int row_index_1based, const cabana::Signal *sig, bool highlight) {
  ImDrawList *dl = ImGui::GetWindowDrawList();
  float x = origin.x + 4.0f;

  // expand/collapse chevron (decorative -- the whole row is one click target, see draw_signal_row())
  const char *chevron = ui_state_for(sig).expanded ? "v" : ">";
  const ImVec2 chevron_size = ImGui::CalcTextSize(chevron);
  dl->AddText(ImVec2(x, origin.y + (ROW_HEIGHT - chevron_size.y) * 0.5f), ImGui::GetColorU32(ImGuiCol_TextDisabled), chevron);
  x += chevron_size.x + 6.0f;

  // color chip with 1-based row index -- mirrors the "N" label in SignalItemDelegate
  const ImVec2 chip_min(x, origin.y + (ROW_HEIGHT - CHIP_SIZE) * 0.5f);
  const ImVec2 chip_max(chip_min.x + CHIP_SIZE, chip_min.y + CHIP_SIZE);
  dl->AddRectFilled(chip_min, chip_max, chip_color(sig->color, highlight), 3.0f);
  char idx_buf[16];
  std::snprintf(idx_buf, sizeof(idx_buf), "%d", row_index_1based);
  const ImVec2 idx_size = ImGui::CalcTextSize(idx_buf);
  dl->AddText(ImVec2(chip_min.x + (CHIP_SIZE - idx_size.x) * 0.5f, chip_min.y + (CHIP_SIZE - idx_size.y) * 0.5f),
             highlight ? IM_COL32_WHITE : IM_COL32_BLACK, idx_buf);
  x = chip_max.x + 6.0f;

  // multiplex indicator ("M" / "mN") -- mirrors the gray rounded badge in SignalItemDelegate
  if (sig->type != cabana::Signal::Type::Normal) {
    char badge[16];
    if (sig->type == cabana::Signal::Type::Multiplexor) std::snprintf(badge, sizeof(badge), " M ");
    else std::snprintf(badge, sizeof(badge), " m%d ", sig->multiplex_value);
    const ImVec2 badge_size = ImGui::CalcTextSize(badge);
    const ImVec2 badge_min(x, origin.y + (ROW_HEIGHT - badge_size.y) * 0.5f - 2.0f);
    const ImVec2 badge_max(badge_min.x + badge_size.x, badge_min.y + badge_size.y + 4.0f);
    dl->AddRectFilled(badge_min, badge_max, IM_COL32(128, 128, 128, 255), 3.0f);
    dl->AddText(ImVec2(badge_min.x, badge_min.y + 2.0f), IM_COL32_WHITE, badge);
    x = badge_max.x + 6.0f;
  }

  return x;
}

// Returns true if this row is hovered this frame (used by the caller to
// decide whether to clear cross-panel hover once no row claims it).
bool draw_row(const MessageId &msg_id, const cabana::Signal *sig, int row_index_1based, float name_col_right) {
  SigUiState &ui = ui_state_for(sig);
  ImGui::PushID(sig);

  if (sig == g_state.pending_scroll_sig) {
    ImGui::SetScrollHereY(0.0f);
    g_state.pending_scroll_sig = nullptr;
  }

  const float avail_w = ImGui::GetContentRegionAvail().x;
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const bool selected = (sig == selected_signal());
  const bool highlight = (sig == hovered_signal());

  // Row-wide click target: toggles expand + selects (mirrors SignalView::rowClicked()
  // + native QTreeView row selection). Hover drives cross-panel highlighting the
  // way Qt's mouse-tracking `tree->entered` signal does -- see signal_state.h.
  const bool clicked = ImGui::Selectable("##row", selected, ImGuiSelectableFlags_AllowOverlap, ImVec2(avail_w, ROW_HEIGHT));
  const bool row_hovered = ImGui::IsItemHovered();
  if (row_hovered) set_hovered_signal(sig);
  if (clicked) toggle_expand(sig, ui);
  if (row_hovered && !clicked) {
    ImGui::SetTooltip("%s\nStart Bit: %d  Size: %d\nMSB: %d  LSB: %d\nLittle Endian: %s  Signed: %s", sig->name.c_str(), sig->start_bit,
                      sig->size, sig->msb, sig->lsb, sig->is_little_endian ? "Y" : "N", sig->is_signed ? "Y" : "N");
  }

  const float name_x = draw_row_left(origin, row_index_1based, sig, highlight);

  // name (elided to the reserved name column width)
  {
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const std::string name = elide_text(sig->name, std::max(10.0f, name_col_right - name_x));
    const ImVec2 text_size = ImGui::CalcTextSize(name.c_str());
    dl->AddText(ImVec2(name_x, origin.y + (ROW_HEIGHT - text_size.y) * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), name.c_str());
  }

  // right-anchored: remove button, plot toggle, value text, sparkline
  float right_x = origin.x + avail_w - 4.0f;

  ImGui::SetCursorScreenPos(ImVec2(right_x - BTN_SIZE, origin.y + (ROW_HEIGHT - BTN_SIZE) * 0.5f));
  if (ImGui::SmallButton("x")) UndoStack::push(new RemoveSigCommand(msg_id, sig));
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove signal");
  right_x -= BTN_SIZE + 4.0f;

  const bool plotted = charts_is_showing(msg_id, sig);
  ImGui::SetCursorScreenPos(ImVec2(right_x - BTN_SIZE, origin.y + (ROW_HEIGHT - BTN_SIZE) * 0.5f));
  if (plotted) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  if (ImGui::SmallButton("~")) charts_show_signal(msg_id, sig, !plotted);
  if (plotted) ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip(plotted ? "Close Plot" : "Show Plot");
  right_x -= BTN_SIZE + 6.0f;

  const CanData &last = can->lastMessage(msg_id);
  double value = 0.0;
  const std::string value_text = sig->getValue(last.dat.data(), last.dat.size(), &value) ? sig->formatValue(value) : "-";

  SparklineCache &spark = g_state.sparklines[sig];
  const ImVec2 spark_size(SPARKLINE_W, SPARKLINE_H);
  const bool need_recompute = g_state.sparkline_dirty || spark.last_size.x != spark_size.x || spark.last_size.y != spark_size.y;
  if (need_recompute) {
    update_sparkline_cache(sig, msg_id, settings.sparkline_range, spark_size, spark);
  }

  ImDrawList *dl = ImGui::GetWindowDrawList();
  const ImU32 value_color = ImGui::GetColorU32(ImGuiCol_Text);
  const ImVec2 value_size = ImGui::CalcTextSize(value_text.c_str());
  right_x -= value_size.x;
  dl->AddText(ImVec2(right_x, origin.y + (ROW_HEIGHT - value_size.y) * 0.5f), value_color, value_text.c_str());
  right_x -= 8.0f;

  // min/max labels (highlighted/selected) or multiplexed freq -- mirrors the delegate's
  // conditional block between the sparkline and the value text
  if (!spark.empty && (highlight || selected)) {
    char min_buf[16], max_buf[16];
    std::snprintf(min_buf, sizeof(min_buf), "%s", doubleToString(spark.min_val).c_str());
    std::snprintf(max_buf, sizeof(max_buf), "%s", doubleToString(spark.max_val).c_str());
    const float w = std::max(ImGui::CalcTextSize(min_buf).x, ImGui::CalcTextSize(max_buf).x);
    right_x -= w + 6.0f;
    dl->AddText(ImVec2(right_x, origin.y + 1.0f), ImGui::GetColorU32(ImGuiCol_TextDisabled), max_buf);
    dl->AddText(ImVec2(right_x, origin.y + ROW_HEIGHT - ImGui::GetTextLineHeight() - 1.0f), ImGui::GetColorU32(ImGuiCol_TextDisabled), min_buf);
  } else if (!spark.empty && sig->type == cabana::Signal::Type::Multiplexed) {
    char freq_buf[24];
    std::snprintf(freq_buf, sizeof(freq_buf), "%.2g hz", spark.freq);
    const float w = ImGui::CalcTextSize(freq_buf).x;
    right_x -= w + 6.0f;
    dl->AddText(ImVec2(right_x, origin.y + (ROW_HEIGHT - ImGui::GetTextLineHeight()) * 0.5f), ImGui::GetColorU32(ImGuiCol_TextDisabled), freq_buf);
  }

  right_x -= SPARKLINE_W;
  draw_sparkline_widget(spark, sig->color, ImVec2(right_x, origin.y + (ROW_HEIGHT - SPARKLINE_H) * 0.5f));

  ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + ROW_HEIGHT));

  // NOTE: the property form is drawn *inside* this row's PushID(sig) scope
  // (popped only after) so its widget IDs ("##name", "extra_info_toggle",
  // ...) are unique per signal -- otherwise two simultaneously-expanded rows
  // would collide (ImGui's "2 visible items with conflicting ID" assert).
  if (ui.expanded) {
    ImGui::Indent();
    draw_signal_property_form(msg_id, sig, ui);
    ImGui::Unindent();
    ImGui::Separator();
  }

  ImGui::PopID();
  return row_hovered;
}

// -- title bar: signal count + filter + sparkline range slider + collapse-all --

void draw_title_bar(int signal_count) {
  char count_buf[32];
  std::snprintf(count_buf, sizeof(count_buf), "Signals: %d", signal_count);
  ImGui::TextUnformatted(count_buf);
  ImGui::SameLine();

  ImGui::SetNextItemWidth(160.0f);
  if (ImGui::InputTextWithHint("##signal_filter", "Filter Signal", g_state.filter_buf, sizeof(g_state.filter_buf),
                               ImGuiInputTextFlags_CharsNoBlank)) {
    g_state.filter = g_state.filter_buf;
  }
  ImGui::SameLine();

  const std::string range_label = format_seconds(settings.sparkline_range);
  ImGui::TextUnformatted(range_label.c_str());
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100.0f);
  int range = settings.sparkline_range;
  if (ImGui::SliderInt("##sparkline_range", &range, 1, 30, "")) {
    settings.sparkline_range = std::clamp(range, 1, 30);
    g_state.sparkline_dirty = true;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sparkline time range");
  ImGui::SameLine();

  if (ImGui::SmallButton("Collapse All")) {
    for (auto &[sig, ui] : g_state.ui) {
      ui.expanded = false;
      ui.extra_info_open = false;
    }
  }
}

}  // namespace

void draw_signal_view(AppState &app) {
  if (!app.selected_msg_id) return;
  ensure_connected();

  const MessageId msg_id = *app.selected_msg_id;
  if (!g_state.has_msg_id || msg_id != g_state.msg_id) {
    reset_for_new_message(msg_id);
  }

  if (consume_selection_from_binary_view()) {
    if (const cabana::Signal *sig = selected_signal(); sig != nullptr) {
      toggle_expand(sig, ui_state_for(sig));
      g_state.pending_scroll_sig = sig;
    }
  }

  cabana::Msg *msg = dbc()->msg(msg_id);
  std::vector<const cabana::Signal *> filtered;
  if (msg != nullptr) {
    for (const cabana::Signal *sig : msg->getSignals()) {
      if (g_state.filter.empty() || contains_ci(sig->name, g_state.filter)) filtered.push_back(sig);
    }
  }

  draw_title_bar(static_cast<int>(filtered.size()));
  ImGui::Separator();

  const float name_col_right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x * 0.42f;
  const bool consume_dirty = g_state.sparkline_dirty;
  bool child_hovered = false;
  bool any_row_hovered = false;
  if (ImGui::BeginChild("##signal_rows", ImGui::GetContentRegionAvail(), ImGuiChildFlags_None)) {
    child_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
    if (filtered.empty()) {
      ImGui::TextDisabled("%s", msg == nullptr ? "No signals -- message not defined in a DBC" : "No signals match the filter");
    } else {
      for (size_t i = 0; i < filtered.size(); ++i) {
        any_row_hovered |= draw_row(msg_id, filtered[i], static_cast<int>(i) + 1, name_col_right);
      }
    }
  }
  ImGui::EndChild();
  if (consume_dirty) g_state.sparkline_dirty = false;

  // mirrors TreeView::leaveEvent()/viewportEntered -> highlight(nullptr): clear
  // hover only when the mouse is within this panel but over no row (empty
  // space) or has left the panel entirely -- don't clobber a hover the binary
  // view set this same frame just because the mouse is over *it*, not us.
  if (child_hovered && !any_row_hovered) {
    set_hovered_signal(nullptr);
  }

  draw_value_desc_modal();
  draw_error_modal();
}
