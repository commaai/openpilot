// ImGui port of tools/cabana/tools/findsimilarbits.{h,cc} (FindSimilarBitsDlg),
// the frozen Qt reference this file mirrors for parity: pick one bit of one
// message on one bus, and rank every bit of every message on another bus by
// how often it agrees (or disagrees) with that reference bit. calcBits() is
// ported verbatim; only the presentation layer changes (QTableWidget ->
// ImGui table, QDialog -> a floating, non-modal ImGui window).
//
// Non-blocking scan: calcBits() is a single forward pass over
// can->allEvents() (every event across every bus, chronologically merged --
// the demo route's full event set, ~1-2M events for a typical route). The Qt
// reference runs this synchronously on the GUI thread, freezing it for the
// duration. Since AbstractStream::update() only ever mutates events_/
// all_events_ on the UI thread once per frame (see find_signal.cc's header
// comment for the full argument), this file time-slices the same pass
// cooperatively across frames instead of spawning a reader thread: process a
// bounded slice of allEvents() per frame (checking the clock every 4096
// events to keep the check overhead itself negligible), carrying the
// running `bit_to_find` / mismatch accumulators across frames exactly as
// calcBits() carries them across its single-threaded loop iterations -- the
// algorithm is inherently sequential (bit_to_find is a moving "last known
// value" as of wherever the scan currently is in time), so this is a
// faithful mechanical translation, not a behavioural change. See the report
// for measured timings.

#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\n\r");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\n\r");
  return s.substr(a, b - a + 1);
}

int parse_int_strict(const char *buf) {
  std::string s = trim(buf);
  if (s.empty()) return 0;
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  return (end == s.c_str() + s.size()) ? static_cast<int>(v) : 0;
}

struct MismatchedRow {
  uint32_t address = 0, byte_idx = 0, bit_idx = 0, mismatches = 0, total = 0;
  float perc = 0.0f;
};

struct MsgEntry {
  uint32_t address = 0;
  std::string label;
};

std::vector<MsgEntry> build_msg_list() {
  std::vector<MsgEntry> out;
  for (const auto &[address, m] : dbc()->getMessages(INVALID_SOURCE)) {
    out.push_back({address, m.name});
  }
  std::sort(out.begin(), out.end(), [](const MsgEntry &a, const MsgEntry &b) { return a.label < b.label; });
  return out;
}

std::vector<int> bus_list() {
  return std::vector<int>(can->sources.begin(), can->sources.end());  // SourceSet is a std::set<int>, already sorted
}

struct FindSimilarBitsState {
  bool open_request = false;
  bool visible = false;

  int src_bus_idx = 0;
  int find_bus_idx = 0;
  uint32_t selected_address = 0;
  bool msg_selected_init = false;
  int byte_idx = 0;
  int bit_idx = 0;
  int equal_index = 0;  // 0 = Yes, 1 = No
  char min_msgs_buf[16] = "100";

  // cooperative scan job -- mirrors calcBits()'s locals, made persistent
  // across frames instead of a single call-stack pass.
  bool scanning = false;
  size_t scan_cursor = 0;
  int bit_to_find = -1;
  std::unordered_map<uint32_t, std::vector<uint32_t>> mismatches;
  std::unordered_map<uint32_t, uint32_t> msg_count;
  uint8_t scan_bus = 0, scan_find_bus = 0;
  uint32_t scan_selected_address = 0;
  int scan_byte_idx = 0, scan_bit_idx = 0;
  bool scan_equal = true;
  int scan_min_msgs = 0;
  std::chrono::steady_clock::time_point scan_start;
  double last_scan_ms = 0.0;

  bool has_results = false;
  std::vector<MismatchedRow> results;
};

FindSimilarBitsState g_state;

// mirrors FindSimilarBitsDlg::calcBits(), split across frames
void step_find_scan(FindSimilarBitsState &st) {
  const auto &events = can->allEvents();
  const auto frame_start = std::chrono::steady_clock::now();
  constexpr auto budget = std::chrono::milliseconds(8);
  constexpr size_t check_every = 4096;
  size_t since_check = 0;

  while (st.scan_cursor < events.size()) {
    const CanEvent *e = events[st.scan_cursor++];
    if (e->src == st.scan_bus) {
      if (e->address == st.scan_selected_address && e->size > st.scan_byte_idx) {
        st.bit_to_find = ((e->dat[st.scan_byte_idx] >> (7 - st.scan_bit_idx)) & 1) != 0;
      }
    }
    if (e->src == st.scan_find_bus) {
      ++st.msg_count[e->address];
      if (st.bit_to_find != -1) {
        auto &mismatched = st.mismatches[e->address];
        if (mismatched.size() < static_cast<size_t>(e->size) * 8) {
          mismatched.resize(static_cast<size_t>(e->size) * 8);
        }
        for (int i = 0; i < e->size; ++i) {
          for (int j = 0; j < 8; ++j) {
            int bit = ((e->dat[i] >> (7 - j)) & 1) != 0;
            mismatched[i * 8 + j] += st.scan_equal ? (bit != st.bit_to_find) : (bit == st.bit_to_find);
          }
        }
      }
    }

    if (++since_check >= check_every) {
      since_check = 0;
      if (std::chrono::steady_clock::now() - frame_start > budget) return;  // yield to next frame
    }
  }

  // finished -- reduce + sort, mirrors the tail of calcBits()
  std::vector<MismatchedRow> result;
  result.reserve(st.mismatches.size());
  for (auto &[address, mismatched] : st.mismatches) {
    auto it = st.msg_count.find(address);
    const uint32_t cnt = (it != st.msg_count.end()) ? it->second : 0;
    if (cnt > static_cast<uint32_t>(st.scan_min_msgs)) {
      for (int i = 0; i < static_cast<int>(mismatched.size()); ++i) {
        const float perc = static_cast<float>((mismatched[i] / static_cast<double>(cnt)) * 100.0);
        if (perc < 50.0f) {
          result.push_back({address, static_cast<uint32_t>(i / 8), static_cast<uint32_t>(i % 8), mismatched[i], cnt, perc});
        }
      }
    }
  }
  std::sort(result.begin(), result.end(), [](const MismatchedRow &l, const MismatchedRow &r) { return l.perc < r.perc; });

  st.results = std::move(result);
  st.has_results = true;
  st.scanning = false;
  st.last_scan_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st.scan_start).count();
}

// mirrors FindSimilarBitsDlg::find()
void start_find(FindSimilarBitsState &st) {
  const std::vector<int> buses = bus_list();
  if (buses.empty()) return;

  st.scan_bus = static_cast<uint8_t>(buses[static_cast<size_t>(std::clamp(st.src_bus_idx, 0, static_cast<int>(buses.size()) - 1))]);
  st.scan_find_bus = static_cast<uint8_t>(buses[static_cast<size_t>(std::clamp(st.find_bus_idx, 0, static_cast<int>(buses.size()) - 1))]);
  st.scan_selected_address = st.selected_address;
  st.scan_byte_idx = st.byte_idx;
  st.scan_bit_idx = st.bit_idx;
  st.scan_equal = st.equal_index == 0;
  st.scan_min_msgs = parse_int_strict(st.min_msgs_buf);

  st.mismatches.clear();
  st.msg_count.clear();
  st.bit_to_find = -1;
  st.scan_cursor = 0;
  st.results.clear();  // mirrors table->clear() at the top of find()
  st.has_results = false;
  st.scanning = true;
  st.scan_start = std::chrono::steady_clock::now();
}

// -- UI -----------------------------------------------------------------

void draw_bus_combo(const char *str_id, const std::vector<int> &buses, int &idx) {
  if (!buses.empty()) idx = std::clamp(idx, 0, static_cast<int>(buses.size()) - 1);
  char preview[16] = "-";
  if (!buses.empty()) std::snprintf(preview, sizeof(preview), "%d", buses[static_cast<size_t>(idx)]);
  ImGui::SetNextItemWidth(60.0f);
  if (ImGui::BeginCombo(str_id, preview)) {
    for (size_t i = 0; i < buses.size(); ++i) {
      char label[16];
      std::snprintf(label, sizeof(label), "%d", buses[i]);
      if (ImGui::Selectable(label, static_cast<int>(i) == idx)) idx = static_cast<int>(i);
    }
    ImGui::EndCombo();
  }
}

void draw_msg_combo(FindSimilarBitsState &st, const std::vector<MsgEntry> &msgs) {
  if (!msgs.empty() && (!st.msg_selected_init ||
                        std::none_of(msgs.begin(), msgs.end(), [&](const MsgEntry &m) { return m.address == st.selected_address; }))) {
    st.selected_address = msgs.front().address;
    st.msg_selected_init = true;
  }
  const char *preview = "No messages";
  for (const MsgEntry &m : msgs) {
    if (m.address == st.selected_address) {
      preview = m.label.c_str();
      break;
    }
  }
  ImGui::SetNextItemWidth(180.0f);
  if (ImGui::BeginCombo("##msg_cb", preview)) {
    for (const MsgEntry &m : msgs) {
      if (ImGui::Selectable(m.label.c_str(), m.address == st.selected_address)) st.selected_address = m.address;
    }
    ImGui::EndCombo();
  }
}

void draw_controls(FindSimilarBitsState &st) {
  const std::vector<int> buses = bus_list();
  const std::vector<MsgEntry> msgs = build_msg_list();
  constexpr float label_col = 90.0f;

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Find From:");
  ImGui::SameLine(label_col);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Bus");
  ImGui::SameLine();
  draw_bus_combo("##src_bus", buses, st.src_bus_idx);
  ImGui::SameLine();
  draw_msg_combo(st, msgs);
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Byte Index");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(50.0f);
  ImGui::InputInt("##byte_idx", &st.byte_idx, 0, 0);
  st.byte_idx = std::clamp(st.byte_idx, 0, 63);
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Bit Index");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(50.0f);
  ImGui::InputInt("##bit_idx", &st.bit_idx, 0, 0);
  st.bit_idx = std::clamp(st.bit_idx, 0, 7);

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Find In:");
  ImGui::SameLine(label_col);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Bus");
  ImGui::SameLine();
  draw_bus_combo("##find_bus", buses, st.find_bus_idx);
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Equal");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(60.0f);
  if (ImGui::BeginCombo("##equal", st.equal_index == 0 ? "Yes" : "No")) {
    if (ImGui::Selectable("Yes", st.equal_index == 0)) st.equal_index = 0;
    if (ImGui::Selectable("No", st.equal_index == 1)) st.equal_index = 1;
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Min msg count");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(70.0f);
  ImGui::InputText("##min_msgs", st.min_msgs_buf, sizeof(st.min_msgs_buf), ImGuiInputTextFlags_CharsDecimal);
  ImGui::SameLine();
  ImGui::BeginDisabled(st.scanning);
  if (ImGui::Button(st.scanning ? "Finding ...." : "Find")) start_find(st);
  ImGui::EndDisabled();
}

void draw_results_table(AppState &app, FindSimilarBitsState &st, float height) {
  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
  if (!ImGui::BeginTable("##find_similar_bits_results", 6, flags, ImVec2(0.0f, height))) return;

  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("address", ImGuiTableColumnFlags_WidthFixed, 70.0f);
  ImGui::TableSetupColumn("byte idx", ImGuiTableColumnFlags_WidthFixed, 70.0f);
  ImGui::TableSetupColumn("bit idx", ImGuiTableColumnFlags_WidthFixed, 60.0f);
  ImGui::TableSetupColumn("mismatches", ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn("total msgs", ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn("% mismatched", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableHeadersRow();

  const std::vector<int> buses = bus_list();
  const uint8_t live_find_bus =
      buses.empty() ? 0 : static_cast<uint8_t>(buses[static_cast<size_t>(std::clamp(st.find_bus_idx, 0, static_cast<int>(buses.size()) - 1))]);

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(st.results.size()));
  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      const MismatchedRow &r = st.results[static_cast<size_t>(i)];
      ImGui::TableNextRow();
      ImGui::PushID(i);

      ImGui::TableSetColumnIndex(0);
      char addr_buf[16];
      std::snprintf(addr_buf, sizeof(addr_buf), "%x", r.address);
      const bool clicked =
          ImGui::Selectable(addr_buf, false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick);
      if (clicked && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        // mirrors FindSimilarBitsDlg's double-click handler: uses the *current*
        // find_bus_combo selection, not a snapshot of the bus at scan time.
        app.selected_msg_id = MessageId{.source = live_find_bus, .address = r.address};
      }

      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%u", r.byte_idx);
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%u", r.bit_idx);
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%u", r.mismatches);
      ImGui::TableSetColumnIndex(4);
      ImGui::Text("%u", r.total);
      ImGui::TableSetColumnIndex(5);
      ImGui::Text("%.2f", r.perc);

      ImGui::PopID();
    }
  }
  ImGui::EndTable();
}

}  // namespace

void draw_find_similar_bits(AppState &app) {
  if (g_state.scanning) step_find_scan(g_state);

  if (g_state.open_request) {
    g_state.open_request = false;
    g_state.visible = true;
    ImGui::SetNextWindowFocus();
  }
  if (!g_state.visible) return;

  ImGui::SetNextWindowSize(ImVec2(700.0f, 500.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSizeConstraints(ImVec2(700.0f, 500.0f), ImVec2(FLT_MAX, FLT_MAX));
  if (ImGui::Begin("Find similar bits", &g_state.visible)) {
    draw_controls(g_state);
    ImGui::Spacing();
    draw_results_table(app, g_state, ImGui::GetContentRegionAvail().y);
  }
  ImGui::End();
}

void open_find_similar_bits() {
  g_state.open_request = true;
}
