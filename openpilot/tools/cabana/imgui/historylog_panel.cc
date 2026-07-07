// ImGui port of tools/cabana/historylog.cc (HistoryLogModel / HeaderView /
// LogsWidget). Behaviors are ported 1:1 from that frozen Qt source unless
// noted otherwise below -- see the comment above draw_history_log().
#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tools/cabana/commands.h"
#include "tools/cabana/dbc/dbcmanager.h"

namespace {

// mirrors LogsWidget's comp_box items {">", "=", "!=", "<"}
enum class CmpOp { Greater, Equal, NotEqual, Less };
constexpr const char *kCmpLabels[] = {">", "=", "!=", "<"};

bool apply_cmp(CmpOp op, double lhs, double rhs) {
  switch (op) {
    case CmpOp::Greater: return lhs > rhs;
    case CmpOp::Equal: return lhs == rhs;
    case CmpOp::NotEqual: return lhs != rhs;
    case CmpOp::Less: return lhs < rhs;
  }
  return true;
}

// mirrors HistoryLogModel::data() col==0: QString::number(can->toSeconds(mono_time), 'f', 3).
// Note: unlike the transport bar / video widget, the Qt reference does NOT run this
// through utils::formatSeconds() or honor settings.absolute_time -- it's always plain
// elapsed seconds with 3 decimals. Verified directly in historylog.cc.
std::string format_row_time(double sec) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.3f", sec);
  return buf;
}

// mirrors HeaderView::paintSection()'s `text.replace(QChar('_'), ' ')`
std::string humanize(std::string s) {
  std::replace(s.begin(), s.end(), '_', ' ');
  return s;
}

void right_aligned_text(const std::string &s) {
  const float avail = ImGui::GetContentRegionAvail().x;
  const float w = ImGui::CalcTextSize(s.c_str()).x;
  if (w < avail) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - w));
  ImGui::TextUnformatted(s.c_str());
}

// Per-message-tab UI/row state. Qt keeps exactly one HistoryLogModel shared by
// every open message tab (DetailWidget owns a single LogsWidget), reset via
// setMessage()/reset() whenever the displayed message changes -- so a single
// instance here matches, rather than one per MessageId.
struct LogState {
  // Persistent across message switches: mirrors HistoryLogModel::hex_mode,
  // which setMessage()/reset() never touches (only the display-type combo does).
  bool hex_mode = false;

  // Rebuilt on every reset() (message switch, seek, DBC change, undo/redo).
  MessageId msg_id{};
  bool has_msg_id = false;
  std::vector<cabana::Signal *> sigs;

  // Filter state, cleared by reset() (mirrors setFilter(0, "", nullptr)).
  int filter_sig_idx = 0;
  int filter_cmp_idx = 0;
  char filter_buf[64] = "";
  bool filter_touched = false;  // mirrors QLineEdit::isModified()
  bool filter_active = false;   // mirrors filter_cmp != nullptr
  double filter_value = 0.0;

  // Row cache: matching events, newest-first (mirrors HistoryLogModel::messages).
  // Holds raw CanEvent* (stable for the stream's lifetime -- events_ is never
  // pruned and existing pointers are never invalidated by later merges) rather
  // than copies, so per-row values are decoded lazily at draw time.
  std::vector<const CanEvent *> rows;
  bool reached_end = false;  // mirrors !canFetchMore(): the oldest event has been reached

  // set true once during draw_history_log() whenever a fresh full/soft reset
  // still needs its first (synchronous) row batch.
};

LogState g_state;

void reset_filter(LogState &st) {
  st.filter_sig_idx = 0;
  st.filter_cmp_idx = 0;
  st.filter_buf[0] = '\0';
  st.filter_touched = false;
  st.filter_active = false;
  st.filter_value = 0.0;
}

// mirrors LogsWidget::showEvent() -> model->updateState(true): clears the row
// cache only, keeps sigs/filter as-is.
void soft_reset(LogState &st) {
  st.rows.clear();
  st.reached_end = false;
}

// mirrors HistoryLogModel::reset(): re-derives sigs from the DBC and clears
// the filter, in addition to the row cache.
void full_reset(LogState &st) {
  cabana::Msg *dbc_msg = dbc()->msg(st.msg_id);
  st.sigs = dbc_msg ? dbc_msg->getSignals() : std::vector<cabana::Signal *>{};
  soft_reset(st);
  reset_filter(st);
}

double decode_signal(const cabana::Signal *sig, const CanEvent *e) {
  double v = 0.0;
  sig->getValue(e->dat, e->size, &v);
  return v;
}

// mirrors the filter check inside HistoryLogModel::fetchData():
//   if (!filter_cmp || filter_cmp(values[filter_sig_idx], filter_value))
bool matches_filter(const LogState &st, const CanEvent *e) {
  if (!st.filter_active) return true;
  if (st.filter_sig_idx < 0 || st.filter_sig_idx >= (int)st.sigs.size()) return true;
  const double v = decode_signal(st.sigs[st.filter_sig_idx], e);
  return apply_cmp(static_cast<CmpOp>(st.filter_cmp_idx), v, st.filter_value);
}

// Perf guard with no Qt equivalent: HistoryLogModel::fetchData() has no cap on
// how many *non-matching* events it walks past while filling one batch, so a
// restrictive filter over a busy, long route can make a single fetch scan
// arbitrarily far. Since events(id) can be hundreds of thousands of entries
// (spec requirement: bounded per-frame work), cap the raw scan distance per
// call; the scan resumes on a later frame/scroll instead of blocking one frame.
constexpr size_t kMaxScanPerCall = 20000;
constexpr size_t kFetchChunk = 64;  // mirrors HistoryLogModel::batch_size (50)

// Scans `events` backward in time starting just before `from_time_excl` down
// to (but not including) `stop_time_excl`, appending matching events
// (newest-first) to `out`. Mirrors HistoryLogModel::fetchData()'s reverse
// upper_bound walk. `max_matches` caps the number appended (0 = unlimited,
// used for the "new rows since last frame" top extension, matching Qt's
// uncapped min_time>0 path). Returns true if the walk reached the oldest
// recorded event for this id (i.e. there is nothing older left to fetch).
bool scan_backward(const std::vector<const CanEvent *> &events, const LogState &st, uint64_t from_time_excl,
                    uint64_t stop_time_excl, size_t max_matches, std::vector<const CanEvent *> &out) {
  auto first = std::upper_bound(events.rbegin(), events.rend(), from_time_excl,
                                 [](uint64_t ts, const CanEvent *e) { return ts > e->mono_time; });
  size_t examined = 0;
  for (; first != events.rend() && (*first)->mono_time > stop_time_excl; ++first) {
    if (++examined > kMaxScanPerCall) return false;
    const CanEvent *e = *first;
    if (matches_filter(st, e)) {
      out.push_back(e);
      if (max_matches > 0 && out.size() >= max_matches) return false;
    }
  }
  return first == events.rend();
}

// mirrors the initial-load / fetchMore() path (min_time == 0 in Qt): appends
// one more batch of older matching rows to the bottom of the cache.
void extend_older(LogState &st, AppState &app) {
  const auto &events = app.stream->events(st.msg_id);
  const uint64_t from_time_excl = st.rows.empty()
                                       ? app.stream->toMonoTime(app.stream->lastMessage(st.msg_id).ts) + 1
                                       : st.rows.back()->mono_time;
  std::vector<const CanEvent *> found;
  const bool exhausted = scan_backward(events, st, from_time_excl, 0, kFetchChunk, found);
  st.rows.insert(st.rows.end(), found.begin(), found.end());
  if (exhausted) st.reached_end = true;
}

// mirrors HistoryLogModel::updateState(false): pulls in events newer than the
// current top-of-cache row, uncapped (matches Qt's min_time>0 fetch path).
void extend_newer(LogState &st, AppState &app) {
  const uint64_t current_time = app.stream->toMonoTime(app.stream->lastMessage(st.msg_id).ts) + 1;
  const uint64_t stop_time_excl = st.rows.front()->mono_time;
  const auto &events = app.stream->events(st.msg_id);
  std::vector<const CanEvent *> found;
  scan_backward(events, st, current_time, stop_time_excl, /*max_matches=*/0, found);
  if (!found.empty()) {
    st.rows.insert(st.rows.begin(), found.begin(), found.end());
  }
}

// mirrors LogsWidget's filter toolbar (display type / signal / comparator /
// value) plus the filterChanged() guard. Draws nothing when the message has
// no signals (Qt: filters_widget->setVisible(!model->sigs.empty())).
void draw_filter_toolbar(LogState &st, AppState &app) {
  if (st.sigs.empty()) return;

  bool mode_changed = false;
  ImGui::SetNextItemWidth(90.0f);
  if (ImGui::BeginCombo("##log_display_mode", st.hex_mode ? "Hex" : "Signal")) {
    if (ImGui::Selectable("Signal", !st.hex_mode) && st.hex_mode) {
      st.hex_mode = false;
      mode_changed = true;
    }
    if (ImGui::Selectable("Hex", st.hex_mode) && !st.hex_mode) {
      st.hex_mode = true;
      mode_changed = true;
    }
    ImGui::EndCombo();
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Display signal value or raw hex value");

  ImGui::SameLine();
  bool sig_changed = false;
  ImGui::SetNextItemWidth(160.0f);
  const char *sig_preview =
      (st.filter_sig_idx >= 0 && st.filter_sig_idx < (int)st.sigs.size()) ? st.sigs[st.filter_sig_idx]->name.c_str() : "";
  if (ImGui::BeginCombo("##log_filter_signal", sig_preview)) {
    for (int i = 0; i < (int)st.sigs.size(); ++i) {
      const bool selected = (i == st.filter_sig_idx);
      if (ImGui::Selectable(st.sigs[i]->name.c_str(), selected) && !selected) {
        st.filter_sig_idx = i;
        sig_changed = true;
      }
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine();
  bool cmp_changed = false;
  ImGui::SetNextItemWidth(50.0f);
  if (ImGui::BeginCombo("##log_filter_cmp", kCmpLabels[st.filter_cmp_idx])) {
    for (int i = 0; i < 4; ++i) {
      const bool selected = (i == st.filter_cmp_idx);
      if (ImGui::Selectable(kCmpLabels[i], selected) && !selected) {
        st.filter_cmp_idx = i;
        cmp_changed = true;
      }
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine();
  ImGui::SetNextItemWidth(120.0f);
  const bool text_changed =
      ImGui::InputText("##log_filter_value", st.filter_buf, sizeof(st.filter_buf), ImGuiInputTextFlags_CharsScientific);
  if (text_changed) st.filter_touched = true;

  if (mode_changed) {
    // mirrors HistoryLogModel::setHexMode(): full reset (re-derive sigs, clear filter)
    full_reset(st);
    return;
  }

  if (sig_changed || cmp_changed || text_changed) {
    // mirrors LogsWidget::filterChanged():
    //   if (value_edit->text().isEmpty() && !value_edit->isModified()) return;
    const bool blocked_by_guard = (st.filter_buf[0] == '\0') && !st.filter_touched;
    if (!blocked_by_guard) {
      st.filter_value = st.filter_buf[0] ? std::atof(st.filter_buf) : 0.0;
      st.filter_active = st.filter_buf[0] != '\0';  // filter_cmp = value.isEmpty() ? nullptr : cmp
      soft_reset(st);                               // mirrors setFilter() -> updateState(true)
    }
  }
}

void draw_table(LogState &st, AppState &app) {
  const bool hex_mode = st.sigs.empty() || st.hex_mode;
  const int col_count = hex_mode ? 2 : (int)st.sigs.size() + 1;

  constexpr ImGuiTableFlags table_flags =
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable;
  if (!ImGui::BeginTable("history_log_table", col_count, table_flags, ImGui::GetContentRegionAvail())) return;

  // mirrors HeaderView::sectionSizeFromContents()'s fixed "000000.000"-sized time column
  const float time_col_w = ImGui::CalcTextSize("000000.000").x + 10.0f;
  ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, time_col_w);
  if (hex_mode) {
    ImGui::TableSetupColumn("Data", ImGuiTableColumnFlags_WidthStretch);
  } else {
    for (const cabana::Signal *sig : st.sigs) ImGui::TableSetupColumn(sig->name.c_str(), ImGuiTableColumnFlags_WidthStretch);
  }
  ImGui::TableSetupScrollFreeze(0, 1);

  ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
  ImGui::TableSetColumnIndex(0);
  ImGui::TableHeader("Time");
  if (hex_mode) {
    ImGui::TableSetColumnIndex(1);
    ImGui::TableHeader("Data");
  } else {
    for (int c = 0; c < (int)st.sigs.size(); ++c) {
      ImGui::TableSetColumnIndex(c + 1);
      const cabana::Signal *sig = st.sigs[c];
      // mirrors HeaderView paint's alpha-blended signal-color background
      ColorRGBA bg = sig->color;
      bg.a = 128;
      ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, to_im_color(bg));
      std::string label = humanize(sig->name);
      if (!sig->unit.empty()) {
        label += " (";
        label += humanize(sig->unit);
        label += ")";
      }
      ImGui::TableHeader(label.c_str());
    }
  }

  ImGuiListClipper clipper;
  clipper.Begin((int)st.rows.size());
  while (clipper.Step()) {
    for (int row_n = clipper.DisplayStart; row_n < clipper.DisplayEnd; ++row_n) {
      const CanEvent *e = st.rows[row_n];
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      right_aligned_text(format_row_time(app.stream->toSeconds(e->mono_time)));
      if (hex_mode) {
        ImGui::TableSetColumnIndex(1);
        // spec: raw-bytes rows render in mono font with no per-byte highlight colors
        push_mono_font();
        draw_message_bytes(e->dat, e->size, /*colors=*/nullptr, /*multiple_lines=*/false, ImGui::GetContentRegionAvail().x);
        pop_mono_font();
      } else {
        for (int c = 0; c < (int)st.sigs.size(); ++c) {
          ImGui::TableSetColumnIndex(c + 1);
          const double v = decode_signal(st.sigs[c], e);
          right_aligned_text(st.sigs[c]->formatValue(v, false));
        }
      }
    }
  }
  const int display_end = clipper.DisplayEnd;
  clipper.End();

  // mirrors canFetchMore()/fetchMore(): once the view has scrolled to the
  // bottom of what's cached, pull in one more batch of older rows.
  if (!st.rows.empty() && !st.reached_end && display_end >= (int)st.rows.size()) {
    extend_older(st, app);
  }

  ImGui::EndTable();
}

}  // namespace

// Ported from tools/cabana/historylog.cc (HistoryLogModel / HeaderView /
// LogsWidget), the frozen Qt reference. Behaviors implemented:
//  - Columns: Time + per-signal columns (name + "(unit)", '_' -> ' '), or
//    Time + Data when the message has no signals or the Hex toggle is set.
//  - Rows: past values of the selected message from stream->events(id),
//    newest-first, decoded via cabana::Signal::getValue()/formatValue(false).
//  - Filters: signal + comparator (>, =, !=, <) + value, with the exact
//    "empty and never edited" no-op guard from LogsWidget::filterChanged().
//  - Live updates: new rows prepended as they arrive; older rows paged in via
//    ImGuiListClipper + a canFetchMore()-style bottom extension.
//  - Reset triggers: message switch, seekedTo, DBCFileChanged, UndoStack
//    index change (all four go through the same full_reset(), matching Qt).
//
// Known deviations from the Qt reference (see comments at each site above):
//  - draw_message_bytes() is always called with colors=nullptr / single-line,
//    per this port's spec -- Qt's hex-mode bit-flip fade coloring (CanData::
//    compute()) is a stateful, sequential-playback-order computation that
//    doesn't have a sound lazy/clipper-friendly equivalent, and the Qt
//    reference itself only colors newly-arrived top rows, not scrolled-in
//    history, so this is a deliberate simplification rather than a straight
//    port.
//  - Time column formatting matches the Qt reference exactly (plain elapsed
//    seconds, 3 decimals, ignores settings.absolute_time) -- historylog.cc's
//    data() never routes through utils::formatSeconds().
//  - "Became visible" (switching from the Msg sub-tab back to Logs) is
//    approximated via an ImGui frame-count gap, since draw_history_log() has
//    no direct signal for it; this reproduces LogsWidget::showEvent()'s hard
//    clear-and-refetch without needing to touch the container's code.
//  - CSV export (LogsWidget::exportToCSV()) is not implemented: it needs a
//    Qt file dialog and utils/export.cc, neither of which exist in the
//    Qt-free imgui build (utils/export.cc isn't part of cabana_core).
//  - Scroll position: no explicit "stick to top" logic. New rows are always
//    inserted at index 0 like Qt's beginInsertRows({}, 0, n-1); with no
//    explicit scroll manipulation on either side, a viewer parked at
//    scroll-Y 0 keeps seeing the newest row on both UIs, and one scrolled
//    away sees the same pixel-offset "jump" on insert either way -- i.e. this
//    matches Qt's (unremarkable) default QTableView behavior, not a special
//    "pin to top" feature Qt doesn't have either.
//  - Added a raw-scan cap (kMaxScanPerCall) with no Qt equivalent, so a
//    restrictive filter over a huge route can't stall a frame; see the
//    comment above scan_backward().
void draw_history_log(AppState &app) {
  if (!app.selected_msg_id.has_value()) {
    ImGui::TextDisabled("No message selected.");
    return;
  }
  const MessageId msg_id = *app.selected_msg_id;

  // One-time wiring, mirrors LogsWidget's connections to seekedTo,
  // DBCFileChanged and UndoStack::indexChanged (all -> model->reset()).
  static AbstractStream *wired_stream = nullptr;
  if (wired_stream != app.stream.get()) {
    wired_stream = app.stream.get();
    wired_stream->seekedTo.connect([](double) { g_state.msg_id = {}; g_state.has_msg_id = false; });

    // g_state.rows holds raw CanEvent* allocated in the OLD stream's
    // MonotonicBuffer, which is freed along with that stream -- drop them
    // now instead of leaving dangling pointers around for the next draw.
    // has_msg_id=false additionally forces a fresh full_reset() below even
    // if the message selected on the new stream happens to share a
    // MessageId with the one last shown on the old stream (msg_changed only
    // looks at msg_id, so without this it would skip the reset).
    g_state.rows.clear();
    g_state.reached_end = false;
    g_state.has_msg_id = false;
  }
  static bool wired_globals = false;
  if (!wired_globals) {
    wired_globals = true;
    dbc()->DBCFileChanged.connect([]() { g_state.has_msg_id = false; });
    UndoStack::instance()->indexChanged.connect([](int) { g_state.has_msg_id = false; });
  }

  // Detect "became visible again" (switching Msg <-> Logs sub-tab while
  // staying on the same message) via a gap in per-frame calls, since
  // draw_history_log() is only invoked while the Logs tab is active.
  static int last_frame = -1;
  const int frame = ImGui::GetFrameCount();
  const bool became_visible = last_frame >= 0 && (frame - last_frame) > 1;
  last_frame = frame;

  const bool msg_changed = !g_state.has_msg_id || g_state.msg_id != msg_id;
  if (msg_changed) {
    g_state.msg_id = msg_id;
    g_state.has_msg_id = true;
    full_reset(g_state);
  } else if (became_visible) {
    soft_reset(g_state);
  } else if (!g_state.rows.empty()) {
    extend_newer(g_state, app);
  }

  draw_filter_toolbar(g_state, app);
  ImGui::Separator();

  // Synchronous first batch after any reset above, so the table isn't blank
  // for one frame (mirrors reset()/updateState(true) populating inline
  // before Qt's next paint).
  if (g_state.rows.empty() && !g_state.reached_end) {
    extend_older(g_state, app);
  }

  draw_table(g_state, app);
}
