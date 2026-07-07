// ImGui port of tools/cabana/tools/findsignal.{h,cc} (FindSignalModel +
// FindSignalDlg), the frozen Qt reference this file mirrors for parity: a
// modeless "find a signal by its value behaviour over time" tool. The engine
// (candidate generation, successive-constraint search, undo history) is
// ported field-for-field; only the presentation layer changes (QAbstractTableModel
// -> a plain vector redrawn each frame, QDialog -> a floating, non-modal
// ImGui window).
//
// Non-blocking scan: the Qt reference spawns std::thread workers inside
// FindSignalModel::search() but *joins* them before returning -- the call is
// synchronous from the GUI's point of view and freezes Qt's event loop for
// its duration (the "Finding ...." button text is the only feedback). The
// spec here asks for the opposite: never block the render loop noticeably.
// Since AbstractStream::update() (which mutates the shared events_ map via
// mergeEvents()) only ever runs on the UI thread, once per frame, spawning
// worker threads that read can->events() concurrently with the render loop
// would race with it. Instead this file time-slices the scan cooperatively
// on the UI thread: step_search()/step_find_scan() process events in ~8ms
// bursts once per frame, yielding back to the renderer in between, so a scan
// that would have frozen Qt for hundreds of ms here just shows "Finding
// ...." for a few frames while the rest of the UI stays responsive. See the
// report for measured timings. This trades away the Qt version's multi-core
// parallelism for safety + responsiveness -- single-threaded is fast enough
// in practice (see report).

#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"

namespace {

// -- small string/number helpers (no Qt available in this Qt-free core;
// these mirror QString::toDouble()/toULong()'s "whole trimmed string must be
// a valid number, else silently 0" behaviour, which the Qt reference relies
// on without ever checking the `ok` out-param) --------------------------

std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\n\r");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\n\r");
  return s.substr(a, b - a + 1);
}

std::vector<std::string> split_commas(const std::string &s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    size_t pos = s.find(',', start);
    if (pos == std::string::npos) {
      out.push_back(s.substr(start));
      break;
    }
    out.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  return out;
}

double parse_qdouble(const char *buf) {
  std::string s = trim(buf);
  if (s.empty()) return 0.0;
  char *end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  return (end == s.c_str() + s.size()) ? v : 0.0;
}

unsigned long parse_qulong(const std::string &s, int base) {
  if (s.empty()) return 0;
  char *end = nullptr;
  unsigned long v = std::strtoul(s.c_str(), &end, base);
  return (end == s.c_str() + s.size()) ? v : 0;
}

std::string join_values(const std::vector<std::string> &values) {
  std::string out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) out += ' ';
    out += values[i];
  }
  return out;
}

// -- engine: ports FindSignalModel::SearchSignal / search() / undo() / reset()
// and FindSignalDlg::setInitialSignals() -----------------------------------

struct SearchSignal {
  MessageId id{};
  uint64_t mono_time = 0;
  cabana::Signal sig{};
  double value = 0.;
  std::vector<std::string> values;  // formatted "(time, value)" -- one appended per successful search step
};

// Cooperative scan job: FindSignalModel::search()'s parallel-for over
// prev_sigs, but time-sliced across frames instead of joined std::threads.
struct ScanJob {
  bool active = false;
  std::vector<SearchSignal> pending;
  size_t cursor = 0;
  std::vector<SearchSignal> result;
  std::function<bool(double)> cmp;
  std::chrono::steady_clock::time_point t_start;
};

struct FindSignalState {
  bool open_request = false;
  bool visible = false;
  bool just_opened = false;

  // "Messages" group
  char bus_buf[128] = "";
  char address_buf[128] = "";
  char first_time_buf[32] = "0";
  char last_time_buf[32] = "MAX";

  // "Signal" group
  int min_size = 8;
  int max_size = 8;
  bool little_endian = true;
  bool is_signed = false;
  char factor_buf[32] = "1.0";
  char offset_buf[32] = "0.0";

  // "Find signal" row
  int compare_index = 0;  // "=", ">", ">=", "!=", "<", "<=", "between"
  char value1_buf[32] = "";
  char value2_buf[32] = "";

  // FindSignalModel state
  std::vector<SearchSignal> filtered_signals;
  std::vector<SearchSignal> initial_signals;
  std::vector<std::vector<SearchSignal>> histories;
  uint64_t last_time = std::numeric_limits<uint64_t>::max();

  bool scanning = false;
  bool stats_ever_shown = false;  // stats_label only gets text after the first search/undo/reset
  ScanJob job;
  double last_scan_ms = 0.0;
};

FindSignalState g_state;

constexpr const char *COMPARE_LABELS[] = {"=", ">", ">=", "!=", "<", "<=", "between"};
constexpr int COMPARE_COUNT = 7;

// mirrors FindSignalDlg::setInitialSignals()
void set_initial_signals(FindSignalState &st) {
  std::set<unsigned long> buses;
  for (const std::string &tok : split_commas(st.bus_buf)) {
    std::string t = trim(tok);
    if (!t.empty()) buses.insert(parse_qulong(t, 10));
  }
  std::set<unsigned long> addresses;
  for (const std::string &tok : split_commas(st.address_buf)) {
    std::string t = trim(tok);
    if (!t.empty()) addresses.insert(parse_qulong(t, 16));
  }

  cabana::Signal sig{};
  sig.is_little_endian = st.little_endian;
  sig.is_signed = st.is_signed;
  sig.factor = parse_qdouble(st.factor_buf);
  sig.offset = parse_qdouble(st.offset_buf);

  const double first_time_val = parse_qdouble(st.first_time_buf);
  const double last_time_val = parse_qdouble(st.last_time_buf);  // "MAX" -> 0.0 (unparsable), matches Qt's toDouble()
  const double first_sec = std::min(first_time_val, last_time_val);
  const double last_sec = std::max(first_time_val, last_time_val);
  const uint64_t first_time = can->toMonoTime(first_sec);
  st.last_time = std::numeric_limits<uint64_t>::max();
  if (last_sec > 0) {
    st.last_time = can->toMonoTime(last_sec);
  }

  st.initial_signals.clear();
  for (const auto &[id, m] : can->lastMessages()) {
    const bool bus_ok = buses.empty() || buses.count(id.source);
    const bool addr_ok = addresses.empty() || addresses.count(id.address);
    if (!bus_ok || !addr_ok) continue;

    const auto &events = can->events(id);
    auto e = std::lower_bound(events.cbegin(), events.cend(), first_time, CompareCanEvent());
    if (e == events.cend()) continue;

    const int total_size = static_cast<int>(m.dat.size()) * 8;
    for (int size = st.min_size; size <= st.max_size; ++size) {
      for (int start = 0; start <= total_size - size; ++start) {
        SearchSignal s;
        s.id = id;
        s.mono_time = first_time;
        s.sig = sig;
        s.sig.start_bit = start;
        s.sig.size = size;
        updateMsbLsb(s.sig);
        s.value = get_raw_value((*e)->dat, (*e)->size, s.sig);
        st.initial_signals.push_back(std::move(s));
      }
    }
  }
}

// mirrors FindSignalModel::search()'s per-candidate worker body, minus the
// threading -- see file header for why.
void step_search(FindSignalState &st) {
  ScanJob &job = st.job;
  const auto frame_start = std::chrono::steady_clock::now();
  constexpr auto budget = std::chrono::milliseconds(8);

  while (job.cursor < job.pending.size()) {
    const SearchSignal &s = job.pending[job.cursor++];
    const auto &events = can->events(s.id);
    auto first = std::upper_bound(events.cbegin(), events.cend(), s.mono_time, CompareCanEvent());
    auto last = events.cend();
    if (st.last_time < std::numeric_limits<uint64_t>::max()) {
      last = std::upper_bound(events.cbegin(), events.cend(), st.last_time, CompareCanEvent());
    }

    auto it = std::find_if(first, last, [&](const CanEvent *e) { return job.cmp(get_raw_value(e->dat, e->size, s.sig)); });
    if (it != last) {
      SearchSignal ns;
      ns.id = s.id;
      ns.mono_time = (*it)->mono_time;
      ns.sig = s.sig;
      ns.values = s.values;
      char buf[64];
      std::snprintf(buf, sizeof(buf), "(%.3f, %g)", can->toSeconds((*it)->mono_time), get_raw_value((*it)->dat, (*it)->size, s.sig));
      ns.values.emplace_back(buf);
      job.result.push_back(std::move(ns));
    }

    if (std::chrono::steady_clock::now() - frame_start > budget) return;  // yield to next frame
  }

  // finished -- mirrors endResetModel() + the modelReset slot
  st.last_scan_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - job.t_start).count();
  st.histories.push_back(job.result);
  st.filtered_signals = std::move(job.result);
  st.scanning = false;
  st.stats_ever_shown = true;
  job.active = false;
}

// mirrors FindSignalDlg::search()
void start_search(FindSignalState &st) {
  if (st.histories.empty()) {
    set_initial_signals(st);
  }
  const double v1 = parse_qdouble(st.value1_buf);
  const double v2 = parse_qdouble(st.value2_buf);
  std::function<bool(double)> cmp;
  switch (st.compare_index) {
    case 0: cmp = [v1](double v) { return v == v1; }; break;
    case 1: cmp = [v1](double v) { return v > v1; }; break;
    case 2: cmp = [v1](double v) { return v >= v1; }; break;
    case 3: cmp = [v1](double v) { return v != v1; }; break;
    case 4: cmp = [v1](double v) { return v < v1; }; break;
    case 5: cmp = [v1](double v) { return v <= v1; }; break;
    default: cmp = [v1, v2](double v) { return v >= v1 && v <= v2; }; break;  // "between"
  }

  st.job = ScanJob{};
  st.job.pending = st.histories.empty() ? st.initial_signals : st.histories.back();
  st.job.cmp = std::move(cmp);
  st.job.t_start = std::chrono::steady_clock::now();
  st.job.active = true;
  st.scanning = true;
}

// mirrors FindSignalModel::undo()
void do_undo(FindSignalState &st) {
  if (!st.histories.empty()) {
    st.histories.pop_back();
    st.filtered_signals = st.histories.empty() ? std::vector<SearchSignal>{} : st.histories.back();
    st.stats_ever_shown = true;
  }
}

// mirrors FindSignalModel::reset()
void do_reset(FindSignalState &st) {
  st.histories.clear();
  st.filtered_signals.clear();
  st.initial_signals.clear();
  st.stats_ever_shown = true;
}

// -- UI ---------------------------------------------------------------------

void draw_messages_group(FindSignalState &st) {
  ImGui::SeparatorText("Messages");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Bus");
  ImGui::SameLine(90.0f);
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputTextWithHint("##bus", "comma-separated values. Leave blank for all", st.bus_buf, sizeof(st.bus_buf));

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Address");
  ImGui::SameLine(90.0f);
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputTextWithHint("##address", "comma-separated hex values. Leave blank for all", st.address_buf, sizeof(st.address_buf));

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Time");
  ImGui::SameLine(90.0f);
  ImGui::SetNextItemWidth(70.0f);
  ImGui::InputText("##first_time", st.first_time_buf, sizeof(st.first_time_buf), ImGuiInputTextFlags_CharsScientific);
  ImGui::SameLine();
  ImGui::TextUnformatted("-");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(70.0f);
  ImGui::InputText("##last_time", st.last_time_buf, sizeof(st.last_time_buf), ImGuiInputTextFlags_CharsScientific);
  ImGui::SameLine();
  ImGui::TextUnformatted("seconds");
}

void draw_signal_group(FindSignalState &st) {
  ImGui::SeparatorText("Signal");
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Size");
  ImGui::SameLine(90.0f);
  ImGui::SetNextItemWidth(60.0f);
  ImGui::InputInt("##min_size", &st.min_size, 0, 0);
  st.min_size = std::clamp(st.min_size, 1, 64);
  ImGui::SameLine();
  ImGui::TextUnformatted("-");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(60.0f);
  ImGui::InputInt("##max_size", &st.max_size, 0, 0);
  st.max_size = std::clamp(st.max_size, 1, 64);
  ImGui::SameLine();
  ImGui::Checkbox("Little endian", &st.little_endian);

  // "Signed" wraps to its own line (indented under the Size fields): the two
  // checkboxes plus Size's two spinboxes don't fit on one line at the
  // dialog's default width -- an accepted layout deviation from Qt's tighter
  // QHBoxLayout, see report.
  ImGui::Indent(90.0f);
  ImGui::Checkbox("Signed", &st.is_signed);
  ImGui::Unindent(90.0f);

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Factor");
  ImGui::SameLine(90.0f);
  ImGui::SetNextItemWidth(120.0f);
  ImGui::InputText("##factor", st.factor_buf, sizeof(st.factor_buf), ImGuiInputTextFlags_CharsScientific);

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Offset");
  ImGui::SameLine(90.0f);
  ImGui::SetNextItemWidth(120.0f);
  ImGui::InputText("##offset", st.offset_buf, sizeof(st.offset_buf), ImGuiInputTextFlags_CharsScientific);
}

// Guards histories/initial_signals/filtered_signals from being mutated by
// Undo/Reset while a scan is writing into them via step_search()'s finalize
// -- the Qt reference doesn't need this because its search() call is
// synchronous/atomic from the event loop's point of view; ours spans frames.
void draw_find_row(AppState &app, FindSignalState &st) {
  const bool groups_disabled = st.scanning || !st.histories.empty();

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Value");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(90.0f);
  if (ImGui::BeginCombo("##compare", COMPARE_LABELS[st.compare_index])) {
    for (int i = 0; i < COMPARE_COUNT; ++i) {
      if (ImGui::Selectable(COMPARE_LABELS[i], i == st.compare_index)) st.compare_index = i;
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100.0f);
  if (st.just_opened) {
    ImGui::SetKeyboardFocusHere();
    st.just_opened = false;
  }
  ImGui::InputText("##value1", st.value1_buf, sizeof(st.value1_buf), ImGuiInputTextFlags_CharsScientific);

  const bool between = st.compare_index == COMPARE_COUNT - 1;
  if (between) {
    ImGui::SameLine();
    ImGui::TextUnformatted("-");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100.0f);
    ImGui::InputText("##value2", st.value2_buf, sizeof(st.value2_buf), ImGuiInputTextFlags_CharsScientific);
  }

  ImGui::SameLine();
  const bool undo_enabled = !groups_disabled && st.histories.size() > 1;
  ImGui::BeginDisabled(!undo_enabled);
  if (ImGui::Button("Undo prev find")) do_undo(st);
  ImGui::EndDisabled();

  ImGui::SameLine();
  const bool search_enabled = !st.scanning && (st.histories.empty() || !st.filtered_signals.empty());
  ImGui::BeginDisabled(!search_enabled);
  const char *search_label = st.scanning ? "Finding ...." : (st.histories.empty() ? "Find" : "Find Next");
  if (ImGui::Button(search_label)) start_search(st);
  ImGui::EndDisabled();

  ImGui::SameLine();
  const bool reset_enabled = !groups_disabled;
  ImGui::BeginDisabled(!reset_enabled);
  if (ImGui::Button("Reset")) do_reset(st);
  ImGui::EndDisabled();
}

void draw_results_table(AppState &app, FindSignalState &st, float height) {
  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
  if (!ImGui::BeginTable("##find_signal_results", 3, flags, ImVec2(0.0f, height))) return;

  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn("Start Bit, size", ImGuiTableColumnFlags_WidthFixed, 130.0f);
  ImGui::TableSetupColumn("(time, value)", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableHeadersRow();

  const size_t n = std::min<size_t>(st.filtered_signals.size(), 300);  // mirrors FindSignalModel::rowCount()'s 300 cap
  for (size_t i = 0; i < n; ++i) {
    const SearchSignal &s = st.filtered_signals[i];
    ImGui::TableNextRow();
    ImGui::PushID(static_cast<int>(i));

    ImGui::TableSetColumnIndex(0);
    const std::string id_str = s.id.toString();
    const bool clicked =
        ImGui::Selectable(id_str.c_str(), false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick);
    if (clicked && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      app.selected_msg_id = s.id;  // mirrors FindSignalDlg::openMessage -> MessagesWidget::selectMessage
    }
    if (ImGui::BeginPopupContextItem()) {
      if (ImGui::MenuItem("Create Signal")) {
        UndoStack::push(new AddSigCommand(s.id, s.sig, static_cast<int>(can->lastMessage(s.id).dat.size())));
        app.selected_msg_id = s.id;
      }
      ImGui::EndPopup();
    }

    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%d, %d", s.sig.start_bit, s.sig.size);

    ImGui::TableSetColumnIndex(2);
    const std::string joined = join_values(s.values);
    ImGui::TextUnformatted(joined.c_str());

    ImGui::PopID();
  }
  ImGui::EndTable();
}

}  // namespace

void draw_find_signal(AppState &app) {
  if (g_state.scanning) step_search(g_state);

  if (g_state.open_request) {
    g_state.open_request = false;
    g_state.visible = true;
    g_state.just_opened = true;
    ImGui::SetNextWindowFocus();
  }
  if (!g_state.visible) return;

  ImGui::SetNextWindowSize(ImVec2(700.0f, 650.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSizeConstraints(ImVec2(700.0f, 650.0f), ImVec2(FLT_MAX, FLT_MAX));
  if (ImGui::Begin("Find Signal", &g_state.visible)) {
    const bool groups_disabled = g_state.scanning || !g_state.histories.empty();
    ImGui::BeginDisabled(groups_disabled);
    if (ImGui::BeginTable("##find_signal_groups", 2)) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      draw_messages_group(g_state);
      ImGui::TableSetColumnIndex(1);
      draw_signal_group(g_state);
      ImGui::EndTable();
    }
    ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::SeparatorText("Find signal");
    draw_find_row(app, g_state);

    ImGui::Spacing();
    const float stats_h = ImGui::GetFrameHeightWithSpacing();
    const float table_h = std::max(80.0f, ImGui::GetContentRegionAvail().y - stats_h);
    draw_results_table(app, g_state, table_h);

    if (g_state.stats_ever_shown && !g_state.scanning) {
      char buf[160];
      std::snprintf(buf, sizeof(buf), "%zu matches. right click on an item to create signal. double click to open message",
                    g_state.filtered_signals.size());
      ImGui::TextUnformatted(buf);
    } else {
      ImGui::Spacing();
    }
  }
  ImGui::End();
}

void open_find_signal() {
  g_state.open_request = true;
}
