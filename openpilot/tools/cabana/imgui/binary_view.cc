// Bit-grid view for the currently selected message -- ports
// tools/cabana/binaryview.{h,cc} (BinaryViewModel + BinaryItemDelegate::paint)
// to immediate-mode drawing. No persistent model: the item grid is rebuilt
// from dbc()->msg(id) + can->lastMessage(id) every call, per the "no caching
// beyond cheap per-frame locals" rule in the phase-2 spec.

#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "tools/cabana/commands.h"
#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/imgui/signal_state.h"
#include "tools/cabana/settings.h"

namespace {

constexpr int COLUMN_COUNT = 9;  // 8 bit columns + 1 hex byte column, matches BinaryViewModel::column_count
constexpr float ROW_HEADER_WIDTH = 26.0f;   // ~ Qt's VERTICAL_HEADER_WIDTH (30), trimmed for imgui's tighter chrome
constexpr float MIN_ROW_HEIGHT = 20.0f;     // ~ Qt's CELL_HEIGHT (36), scaled down for the smaller default UI font

// BinaryViewModel::Item::bg_color default (purple) -- the base tint for bits
// not covered by any defined signal; only its alpha animates with the
// bit-flip heatmap.
constexpr ColorRGBA DEFAULT_BIT_COLOR{102, 86, 169, 255};

ImU32 bv_color(const ColorRGBA &c) { return IM_COL32(c.r, c.g, c.b, c.a); }

// Approximates QColor::darker(125) (value *= 100/125) with a flat RGB scale --
// close enough for near-visual parity without pulling in HSV conversion.
ImU32 bv_darker(const ColorRGBA &c) {
  const auto scale = [](uint8_t v) { return static_cast<uint8_t>(std::clamp(v * 0.8f, 0.0f, 255.0f)); };
  return IM_COL32(scale(c.r), scale(c.g), scale(c.b), c.a);
}

// One binary-view cell, rebuilt every frame. Mirrors BinaryViewModel::Item;
// `sigs` is sorted largest-first when a cell has overlapping signals so the
// paint loop below draws (and hit-tests) the same way Qt does: biggest signal
// first, smaller/narrower ones layered on top.
struct BvItem {
  bool valid = false;
  bool is_msb = false;
  bool is_lsb = false;
  uint8_t val = 0;
  uint8_t heat_alpha = 0;           // bit-flip heatmap alpha (0..255); bit columns only
  ColorRGBA hex_color{0, 0, 0, 0};  // hex column: CanData::colors[byte] recent-change fade
  std::vector<const cabana::Signal *> sigs;
};

// Static (DBC-derived) half of the grid: which signal(s) cover each bit.
// Shared by draw_binary_view() and binary_view_overlapping_signals() so both
// use one source of truth for the bit->signal mapping, exactly like Qt's
// BinaryView and DetailWidget both read BinaryViewModel::items.
std::vector<BvItem> build_signal_coverage(const cabana::Msg *dbc_msg, int row_count) {
  std::vector<BvItem> items(static_cast<size_t>(row_count) * COLUMN_COUNT);
  if (dbc_msg == nullptr) return items;

  for (cabana::Signal *sig : dbc_msg->getSignals()) {
    for (int j = 0; j < sig->size; ++j) {
      const int pos = sig->is_little_endian ? flipBitPos(sig->start_bit + j) : flipBitPos(sig->start_bit) + j;
      const int idx = COLUMN_COUNT * (pos / 8) + pos % 8;
      if (idx < 0 || idx >= static_cast<int>(items.size())) break;  // out-of-bounds signal def, matches Qt's qWarning path
      if (j == 0) (sig->is_little_endian ? items[idx].is_lsb : items[idx].is_msb) = true;
      if (j == sig->size - 1) (sig->is_little_endian ? items[idx].is_msb : items[idx].is_lsb) = true;
      items[idx].sigs.push_back(sig);
    }
  }
  for (BvItem &item : items) {
    if (item.sigs.size() > 1) {
      std::sort(item.sigs.begin(), item.sigs.end(), [](const cabana::Signal *l, const cabana::Signal *r) { return l->size > r->size; });
    }
  }
  return items;
}

bool item_has_signal(const std::vector<BvItem> &items, int row_count, int row, int col, const cabana::Signal *sig) {
  if (sig == nullptr || row < 0 || row >= row_count || col < 0 || col >= 8) return false;
  const auto &sigs = items[row * COLUMN_COUNT + col].sigs;
  return std::find(sigs.begin(), sigs.end(), sig) != sigs.end();
}

// Hover/selection state shared with signal_view.cc -- see signal_state.h.
// Kept as local static state (not in AppState, which this workstream doesn't
// own); storage lives here since this is where it originated pre-Phase-3.
const cabana::Signal *g_hovered_sig = nullptr;
const cabana::Signal *g_selected_sig = nullptr;
bool g_selection_from_binary_view = false;

// -- heatmap mode: Live vs All -- ports DetailWidget's "Heatmap: Live/All"
// toolbar radio buttons + BinaryViewModel::heatmap_live_mode/
// getBitFlipChanges(). Reproduced inline here (rather than in detail_panel.cc)
// since it only ever affects this view. "Live" (default) reads
// CanData::bit_flip_counts, a rolling counter the stream core already
// maintains per message; "All" replays every event in the current time range
// (the whole route, or a chart zoom's sub-range) and counts byte-to-byte bit
// transitions, cached per-message until the range changes -- a map keyed by
// MessageId stands in for Qt's one-tracker-reset-on-setMessage() (same
// "starts fresh per message" effect, but a switch back to a previously-open
// tab keeps its own cache instead of recomputing).
bool g_heatmap_live_mode = true;

struct BitFlipTracker {
  std::optional<std::pair<double, double>> time_range;
  std::vector<std::array<uint32_t, 8>> flip_counts;
};
std::map<MessageId, BitFlipTracker> g_bit_flip_trackers;

const std::vector<std::array<uint32_t, 8>> &compute_bit_flip_changes(const MessageId &id, size_t msg_size) {
  BitFlipTracker &tracker = g_bit_flip_trackers[id];
  const auto time_range = can->timeRange();
  if (tracker.time_range == time_range && !tracker.flip_counts.empty()) return tracker.flip_counts;

  tracker.time_range = time_range;
  tracker.flip_counts.assign(msg_size, std::array<uint32_t, 8>{});

  auto [first, last] = can->eventsInRange(id, time_range);
  if (std::distance(first, last) <= 1) return tracker.flip_counts;

  std::vector<uint8_t> prev_values((*first)->dat, (*first)->dat + (*first)->size);
  for (auto it = std::next(first); it != last; ++it) {
    const CanEvent *event = *it;
    const int size = std::min<int>(static_cast<int>(msg_size), static_cast<int>(event->size));
    for (int i = 0; i < size; ++i) {
      const uint8_t diff = event->dat[i] ^ prev_values[static_cast<size_t>(i)];
      if (!diff) continue;

      auto &bit_flips = tracker.flip_counts[static_cast<size_t>(i)];
      for (int bit = 0; bit < 8; ++bit) {
        if (diff & (1u << bit)) ++bit_flips[static_cast<size_t>(7 - bit)];
      }
      prev_values[static_cast<size_t>(i)] = event->dat[i];
    }
  }
  return tracker.flip_counts;
}

// mirrors DetailWidget::createToolBar()'s heatmap_live/heatmap_all
// QRadioButtons + the can->timeRangeChanged connection that auto-selects
// "All" (labelled with the exact range) when a chart zoom sets a time range,
// and back to "Live" when the zoom clears -- edge-detected here per frame
// instead of an event subscription, same effect. The user can still click
// either button afterward, same as the Qt radio buttons.
void draw_heatmap_toggle() {
  static std::optional<std::pair<double, double>> last_seen_range;
  const auto time_range = can->timeRange();
  if (time_range != last_seen_range) {
    g_heatmap_live_mode = !time_range.has_value();
    last_seen_range = time_range;
  }

  ImGui::TextDisabled("Heatmap:");
  ImGui::SameLine();
  if (g_heatmap_live_mode) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  if (ImGui::SmallButton("Live")) g_heatmap_live_mode = true;
  if (g_heatmap_live_mode) ImGui::PopStyleColor();

  ImGui::SameLine();
  char all_label[64];
  if (time_range) {
    std::snprintf(all_label, sizeof(all_label), "%.3f - %.3f", time_range->first, time_range->second);
  } else {
    std::snprintf(all_label, sizeof(all_label), "All");
  }
  if (!g_heatmap_live_mode) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  if (ImGui::SmallButton(all_label)) g_heatmap_live_mode = false;
  if (!g_heatmap_live_mode) ImGui::PopStyleColor();
}

// -- drag-to-create / resize -- ports BinaryView::mousePressEvent() /
// mouseMoveEvent() / mouseReleaseEvent() + BinaryView::getSelection() from
// binaryview.cc. A drag spans multiple draw_binary_view() calls (press
// frame, N move frames, release frame), so -- like the hover/selection
// state above -- it has to live in a file static, not a per-frame local.
struct DragState {
  bool active = false;   // mouse went down inside the grid; drag in progress
  bool dragged = false;  // moved to a different cell since press -- distinguishes an
                          // actual drag (create/resize) from a plain click, which falls
                          // through untouched to the existing per-cell IsItemClicked()
                          // select-signal path below
  MessageId msg_id{};
  int press_row = -1, press_col = -1;    // raw press cell; used only for the `dragged` check
  int anchor_row = -1, anchor_col = -1;  // BinaryView::anchor_index -- reassigned to the
                                          // opposite edge when the press started a resize
  int cur_row = -1, cur_col = -1;        // last known mouse cell (BinaryView::indexAt on move)
  const cabana::Signal *resize_sig = nullptr;
};
DragState g_drag;

int grid_lin(int row, int col) { return row * 8 + col; }
int grid_bit_pos(int row, int col) { return flipBitPos(grid_lin(row, col)); }
// QModelIndex::operator< for indices from the same row-major model: mirrors the
// ordering BinaryView::getSelection() relies on via `index < anchor_index`.
bool grid_less(int row_a, int col_a, int row_b, int col_b) {
  return row_a < row_b || (row_a == row_b && col_a < col_b);
}

struct DragSelection {
  int start_bit;
  int size;
  bool is_little_endian;
};

// Port of BinaryView::getSelection(): turns an anchor/current grid-cell pair
// (+ optional resize target, which pins the endianness to the signal being
// resized) into the (start_bit, size, is_little_endian) triple for the
// new/resized signal, honoring settings.drag_direction exactly like Qt.
DragSelection compute_drag_selection(int anchor_row, int anchor_col, int cur_row, int cur_col,
                                      const cabana::Signal *resize_sig) {
  if (cur_col == 8) cur_col = 7;  // hex column -> last bit column, mirrors getSelection()

  bool is_lb;
  if (resize_sig != nullptr) {
    is_lb = resize_sig->is_little_endian;
  } else {
    const bool cur_before_anchor = grid_less(cur_row, cur_col, anchor_row, anchor_col);
    switch (settings.drag_direction) {
      case Settings::DragDirection::LsbFirst: is_lb = !cur_before_anchor; break;
      case Settings::DragDirection::AlwaysLE: is_lb = true; break;
      case Settings::DragDirection::AlwaysBE: is_lb = false; break;
      case Settings::DragDirection::MsbFirst:
      default: is_lb = cur_before_anchor; break;
    }
  }

  const int cur_bit_pos = grid_bit_pos(cur_row, cur_col);
  const int anchor_bit_pos = grid_bit_pos(anchor_row, anchor_col);
  DragSelection sel;
  sel.is_little_endian = is_lb;
  if (is_lb) {
    sel.start_bit = std::min(cur_bit_pos, anchor_bit_pos);
    sel.size = std::abs(cur_bit_pos - anchor_bit_pos) + 1;
  } else {
    // std::min(index, anchor_index) using grid_less as the operator<.
    const bool cur_is_min = !grid_less(anchor_row, anchor_col, cur_row, cur_col);
    sel.start_bit = grid_bit_pos(cur_is_min ? cur_row : anchor_row, cur_is_min ? cur_col : anchor_col);
    sel.size = std::abs(grid_lin(cur_row, cur_col) - grid_lin(anchor_row, anchor_col)) + 1;
  }
  return sel;
}

// Same start_bit/size/is_little_endian -> covered-bits walk used by
// build_signal_coverage() above, factored out so the live drag preview
// highlights exactly the cells the resulting signal would occupy.
template <typename F>
void for_each_selection_cell(int start_bit, int size, bool is_little_endian, int row_count, F &&fn) {
  for (int j = 0; j < size; ++j) {
    const int pos = is_little_endian ? flipBitPos(start_bit + j) : flipBitPos(start_bit) + j;
    const int row = pos / 8, col = pos % 8;
    if (row < 0 || row >= row_count || col < 0 || col > 7) continue;
    fn(row, col);
  }
}

}  // namespace

const cabana::Signal *hovered_signal() { return g_hovered_sig; }
void set_hovered_signal(const cabana::Signal *sig) { g_hovered_sig = sig; }

const cabana::Signal *selected_signal() { return g_selected_sig; }
void set_selected_signal(const cabana::Signal *sig, bool from_binary_view) {
  g_selected_sig = sig;
  if (from_binary_view) g_selection_from_binary_view = true;
}
bool consume_selection_from_binary_view() {
  bool v = g_selection_from_binary_view;
  g_selection_from_binary_view = false;
  return v;
}

std::set<const cabana::Signal *> binary_view_overlapping_signals(const MessageId &id) {
  std::set<const cabana::Signal *> overlapping;
  const cabana::Msg *dbc_msg = dbc()->msg(id);
  if (dbc_msg == nullptr) return overlapping;

  for (const BvItem &item : build_signal_coverage(dbc_msg, static_cast<int>(dbc_msg->size))) {
    if (item.sigs.size() > 1) {
      for (const cabana::Signal *s : item.sigs) {
        if (s->type == cabana::Signal::Type::Normal) overlapping.insert(s);
      }
    }
  }
  return overlapping;
}

void draw_binary_view(AppState &app) {
  if (!app.selected_msg_id) return;
  const MessageId id = *app.selected_msg_id;
  const CanData &last_msg = can->lastMessage(id);
  const cabana::Msg *dbc_msg = dbc()->msg(id);

  // A drag started on a different message tab (e.g. via a stray click
  // elsewhere while the button is somehow still held) is stale -- mirrors
  // BinaryView::refresh() clearing anchor_index/resize_sig on setMessage().
  if (g_drag.active && g_drag.msg_id != id) g_drag = DragState{};

  const int row_count = std::max(dbc_msg != nullptr ? static_cast<int>(dbc_msg->size) : static_cast<int>(last_msg.dat.size()), 0);
  if (row_count == 0) {
    ImGui::TextDisabled("No data");
    return;
  }
  std::vector<BvItem> items = build_signal_coverage(dbc_msg, row_count);

  draw_heatmap_toggle();

  // Bit-flip heatmap -- mirrors BinaryViewModel::updateState()'s
  // heatmap_live_mode branch: Live reads CanData::bit_flip_counts directly,
  // All recomputes from every event in the current time range (see
  // compute_bit_flip_changes() above).
  const auto &bit_flips =
      g_heatmap_live_mode ? last_msg.bit_flip_counts : compute_bit_flip_changes(id, last_msg.dat.size());
  uint32_t max_flip = 1;  // avoid div-by-zero, matches Qt's default
  for (const auto &row : bit_flips) {
    for (uint32_t c : row) max_flip = std::max(max_flip, c);
  }
  constexpr double kMaxAlpha = 255.0, kMinAlphaSig = 25.0, kMinAlphaNoSig = 10.0, kLogFactor = 1.2;
  const double log_scaler = kMaxAlpha / std::log2(kLogFactor * max_flip);

  const size_t data_rows = std::min<size_t>(last_msg.dat.size(), static_cast<size_t>(row_count));
  for (size_t i = 0; i < data_rows; ++i) {
    for (int j = 0; j < 8; ++j) {
      BvItem &item = items[i * COLUMN_COUNT + j];
      item.valid = true;
      item.val = (last_msg.dat[i] >> (7 - j)) & 1;

      double alpha = item.sigs.empty() ? 0.0 : kMinAlphaSig;
      const uint32_t flip_count = bit_flips[i][j];
      if (flip_count > 0) {
        const double normalized = std::log2(1.0 + flip_count * kLogFactor) * log_scaler;
        const double min_alpha = item.sigs.empty() ? kMinAlphaNoSig : kMinAlphaSig;
        alpha = std::clamp(normalized, min_alpha, kMaxAlpha);
      }
      item.heat_alpha = static_cast<uint8_t>(std::clamp(alpha, 0.0, 255.0));
    }
    BvItem &hex_item = items[i * COLUMN_COUNT + 8];
    hex_item.valid = true;
    hex_item.val = last_msg.dat[i];
    hex_item.hex_color = last_msg.colors[i];
  }

  const bool is_active = can->isMessageActive(id);
  push_mono_font();
  const float row_h = std::max(MIN_ROW_HEIGHT, ImGui::GetTextLineHeight() + 8.0f);
  const float avail_w = ImGui::GetContentRegionAvail().x;
  const float col_w = std::max(14.0f, (avail_w - ROW_HEADER_WIDTH) / COLUMN_COUNT);
  const float total_w = ROW_HEADER_WIDTH + col_w * COLUMN_COUNT;
  const float total_h = row_h * row_count;

  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImU32 text_color = ImGui::GetColorU32(is_active ? ImGuiCol_Text : ImGuiCol_TextDisabled);
  const ImU32 bright_text_color = ImGui::GetColorU32(ImGuiCol_Text);
  const ImU32 header_text_color = ImGui::GetColorU32(ImGuiCol_TextDisabled);
  const ImU32 hatch_color = ImGui::GetColorU32(ImVec4(0.5f, 0.5f, 0.5f, 0.35f));
  const ImU32 hover_outline = ImGui::GetColorU32(ImGuiCol_Text, 0.5f);

  // -- drag press / move -- mirrors BinaryView::mousePressEvent() and
  // mouseMoveEvent()'s indexAt() tracking. This uses raw mouse-position math
  // rather than the per-cell ImGui item hover used elsewhere in this file:
  // once a cell's InvisibleButton is held active, ImGui suppresses
  // IsItemHovered() on the *other* cells the drag passes over, so hit-testing
  // has to be geometry-based (same GetPlotMousePos()-driven pattern as
  // chart_view.cc's box-zoom drag).
  const ImGuiIO &io = ImGui::GetIO();
  const ImVec2 mouse_pos = io.MousePos;
  const bool mouse_in_grid = mouse_pos.x >= origin.x + ROW_HEADER_WIDTH && mouse_pos.x < origin.x + total_w &&
                             mouse_pos.y >= origin.y && mouse_pos.y < origin.y + total_h;
  int hit_row = -1, hit_col = -1;
  if (mouse_in_grid) {
    hit_col = std::clamp(static_cast<int>((mouse_pos.x - origin.x - ROW_HEADER_WIDTH) / col_w), 0, COLUMN_COUNT - 1);
    hit_row = std::clamp(static_cast<int>((mouse_pos.y - origin.y) / row_h), 0, row_count - 1);
  }

  if (!g_drag.active && mouse_in_grid && hit_col != 8 && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    g_drag = DragState{};
    g_drag.active = true;
    g_drag.msg_id = id;
    g_drag.press_row = g_drag.anchor_row = g_drag.cur_row = hit_row;
    g_drag.press_col = g_drag.anchor_col = g_drag.cur_col = hit_col;
    // Pressing on the msb/lsb edge bit of an existing signal starts a resize
    // instead of a create: the anchor snaps to that signal's *opposite* edge
    // so the drag extends/shrinks it from the fixed far end -- exactly like
    // BinaryView::mousePressEvent().
    const int bit_pos = grid_bit_pos(hit_row, hit_col);
    for (const cabana::Signal *s : items[hit_row * COLUMN_COUNT + hit_col].sigs) {
      if (bit_pos == s->lsb || bit_pos == s->msb) {
        const int idx = flipBitPos(bit_pos == s->lsb ? s->msb : s->lsb);
        g_drag.anchor_row = idx / 8;
        g_drag.anchor_col = idx % 8;
        g_drag.resize_sig = s;
        break;
      }
    }
  } else if (g_drag.active && g_drag.msg_id == id) {
    if (mouse_in_grid) {
      g_drag.cur_row = hit_row;
      g_drag.cur_col = hit_col;
    }
    if (g_drag.cur_row != g_drag.press_row || g_drag.cur_col != g_drag.press_col) g_drag.dragged = true;
  }

  // Resize-cursor hint on edge-bit hover -- approximates a resize affordance;
  // the frozen Qt source has no explicit setCursor() for this, so this is UX
  // polish per the phase-3 spec rather than a literal Qt port.
  if (mouse_in_grid && hit_col != 8 && hit_row >= 0) {
    bool on_edge = g_drag.active && g_drag.resize_sig != nullptr;
    if (!g_drag.active) {
      const int bit_pos = grid_bit_pos(hit_row, hit_col);
      for (const cabana::Signal *s : items[hit_row * COLUMN_COUNT + hit_col].sigs) {
        if (bit_pos == s->lsb || bit_pos == s->msb) { on_edge = true; break; }
      }
    }
    if (on_edge) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
  }

  char buf[16];
  for (int row = 0; row < row_count; ++row) {
    const float row_y = origin.y + row * row_h;

    snprintf(buf, sizeof(buf), "%d", row);
    const ImVec2 header_text_size = ImGui::CalcTextSize(buf);
    draw_list->AddText(ImVec2(origin.x + (ROW_HEADER_WIDTH - header_text_size.x) * 0.5f, row_y + (row_h - header_text_size.y) * 0.5f),
                       header_text_color, buf);

    for (int col = 0; col < COLUMN_COUNT; ++col) {
      const BvItem &item = items[row * COLUMN_COUNT + col];
      const ImVec2 cell_min(origin.x + ROW_HEADER_WIDTH + col * col_w, row_y);
      const ImVec2 cell_max(cell_min.x + col_w, row_y + row_h);
      const bool is_hex = (col == 8);
      const cabana::Signal *top_sig = item.sigs.empty() ? nullptr : item.sigs.back();

      ImGui::SetCursorScreenPos(cell_min);
      ImGui::PushID(row * COLUMN_COUNT + col);
      ImGui::InvisibleButton("cell", ImVec2(col_w, row_h));
      const bool hovered = ImGui::IsItemHovered();
      const bool clicked = ImGui::IsItemClicked();
      ImGui::PopID();

      // background -- mirrors BinaryItemDelegate::paint()'s non-selected branch.
      // The State_Selected branch (drag-select/resize preview) is handled by
      // a separate overlay pass after this loop instead of branching here --
      // see the "Live preview" block below.
      if (is_hex) {
        if (item.valid) draw_list->AddRectFilled(cell_min, cell_max, bv_color(item.hex_color));
      } else if (!item.sigs.empty()) {
        for (const cabana::Signal *sig : item.sigs) {
          if (sig == g_hovered_sig) {
            draw_list->AddRectFilled(cell_min, cell_max, bv_darker(sig->color));
          } else {
            ColorRGBA faded = sig->color;
            faded.a = item.heat_alpha;
            draw_list->AddRectFilled(cell_min, cell_max, bv_color(faded));
            // Border on edges where the same signal doesn't continue into the
            // neighboring cell -- simplified port of drawSignalCell() that
            // skips its corner-gap subtraction (a minor Qt-painter-specific
            // anti-double-border polish, not a data/behavior difference).
            const ImU32 edge_color = bv_darker(sig->color);
            if (!item_has_signal(items, row_count, row, col - 1, sig)) draw_list->AddLine(cell_min, ImVec2(cell_min.x, cell_max.y), edge_color);
            if (!item_has_signal(items, row_count, row - 1, col, sig)) draw_list->AddLine(cell_min, ImVec2(cell_max.x, cell_min.y), edge_color);
            if (!item_has_signal(items, row_count, row, col + 1, sig)) draw_list->AddLine(ImVec2(cell_max.x, cell_min.y), cell_max, edge_color);
            if (!item_has_signal(items, row_count, row + 1, col, sig)) draw_list->AddLine(ImVec2(cell_min.x, cell_max.y), cell_max, edge_color);
          }
        }
      } else if (item.valid && item.heat_alpha > 0) {
        ColorRGBA c = DEFAULT_BIT_COLOR;
        c.a = item.heat_alpha;
        draw_list->AddRectFilled(cell_min, cell_max, bv_color(c));
      }

      // overlapping-signal / invalid-cell hatch -- approximates Qt's
      // Dense7Pattern / BDiagPattern brushes with a few diagonal strokes.
      if (item.sigs.size() > 1 || !item.valid) {
        for (float x = cell_min.x - row_h; x < cell_max.x; x += 5.0f) {
          const ImVec2 p0(std::max(x, cell_min.x), cell_max.y);
          const ImVec2 p1(std::min(x + (cell_max.y - cell_min.y), cell_max.x), cell_min.y);
          draw_list->AddLine(p0, p1, hatch_color);
        }
      }

      // Phase-3 hook: outline the bits of the locally-selected signal.
      if (top_sig != nullptr && top_sig == g_selected_sig) {
        draw_list->AddRect(cell_min, cell_max, IM_COL32(255, 255, 255, 220), 0.0f, 0, 2.0f);
      }

      if (item.valid) {
        char text[4];
        if (is_hex) snprintf(text, sizeof(text), "%02X", item.val);
        else { text[0] = item.val ? '1' : '0'; text[1] = '\0'; }
        const ImU32 value_color = (top_sig != nullptr && top_sig == g_hovered_sig) ? bright_text_color : text_color;
        const ImVec2 text_size = ImGui::CalcTextSize(text);
        draw_list->AddText(ImVec2(cell_min.x + (col_w - text_size.x) * 0.5f, cell_min.y + (row_h - text_size.y) * 0.5f), value_color, text);
      }

      if (!is_hex && (item.is_msb || item.is_lsb)) {
        draw_list->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 0.65f, ImVec2(cell_max.x - 10.0f, cell_max.y - 12.0f), header_text_color,
                           item.is_msb ? "M" : "L");
      }

      if (hovered) {
        draw_list->AddRect(cell_min, cell_max, hover_outline);
        g_hovered_sig = top_sig;
        if (top_sig != nullptr) {
          ImGui::SetTooltip("%s\nStart Bit: %d  Size: %d\nMSB: %d  LSB: %d\nLittle Endian: %s  Signed: %s", top_sig->name.c_str(),
                            top_sig->start_bit, top_sig->size, top_sig->msb, top_sig->lsb, top_sig->is_little_endian ? "Y" : "N",
                            top_sig->is_signed ? "Y" : "N");
        }
        // mirrors BinaryView::mouseReleaseEvent()'s no-drag branch -> emit
        // signalClicked(sig) -> SignalView::selectSignal(sig, /*expand=*/true)
        if (clicked && top_sig != nullptr) set_selected_signal(top_sig, /*from_binary_view=*/true);
      }
    }
  }

  if (!ImGui::IsMouseHoveringRect(origin, ImVec2(origin.x + total_w, origin.y + total_h))) {
    g_hovered_sig = nullptr;  // mirrors BinaryView::leaveEvent() -> highlight(nullptr)
  }

  // Keyboard shortcuts on the hovered signal -- mirrors binaryview.cc's
  // createShortcuts(): x/Backspace/Delete removes it; e/s flip
  // endianness/signedness (via the same start_bit-flip-on-endian-change
  // SignalModel::saveSignal() applies before pushing EditSignalCommand);
  // p/g/c open a chart for it, always merge=false -- Qt's binaryview.cc
  // shortcut has no Shift/merge check, unlike the signal editor's plot
  // button (see signal_view.cc).
  if (!io.WantTextInput && g_hovered_sig != nullptr) {
    if (ImGui::IsKeyPressed(ImGuiKey_X, false) || ImGui::IsKeyPressed(ImGuiKey_Backspace, false) ||
        ImGui::IsKeyPressed(ImGuiKey_Delete, false)) {
      UndoStack::push(new RemoveSigCommand(id, g_hovered_sig));
      g_hovered_sig = nullptr;
    } else if (ImGui::IsKeyPressed(ImGuiKey_E, false)) {
      cabana::Signal s = *g_hovered_sig;
      s.is_little_endian = !s.is_little_endian;
      s.start_bit = flipBitPos(s.start_bit);
      UndoStack::push(new EditSignalCommand(id, g_hovered_sig, s));
    } else if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
      cabana::Signal s = *g_hovered_sig;
      s.is_signed = !s.is_signed;
      UndoStack::push(new EditSignalCommand(id, g_hovered_sig, s));
    } else if (ImGui::IsKeyPressed(ImGuiKey_P, false) || ImGui::IsKeyPressed(ImGuiKey_G, false) ||
               ImGui::IsKeyPressed(ImGuiKey_C, false)) {
      charts_show_signal(id, g_hovered_sig, true);
    }
  }

  // Live preview -- mirrors BinaryItemDelegate::paint()'s State_Selected
  // branch. Drawn as one overlay pass after the grid so it layers on top;
  // Qt achieves the same z-order within a single delegate paint() call by
  // filling the selection then drawing the cell's digit on top of it, so the
  // digit is redrawn here too rather than left hidden under the overlay.
  if (g_drag.active && g_drag.dragged && g_drag.msg_id == id) {
    const DragSelection sel = compute_drag_selection(g_drag.anchor_row, g_drag.anchor_col, g_drag.cur_row, g_drag.cur_col, g_drag.resize_sig);
    const ImU32 preview_color =
        g_drag.resize_sig != nullptr ? bv_color(g_drag.resize_sig->color) : ImGui::GetColorU32(ImGuiCol_HeaderActive);
    for_each_selection_cell(sel.start_bit, sel.size, sel.is_little_endian, row_count, [&](int r, int c) {
      const ImVec2 cell_min(origin.x + ROW_HEADER_WIDTH + c * col_w, origin.y + r * row_h);
      const ImVec2 cell_max(cell_min.x + col_w, cell_min.y + row_h);
      draw_list->AddRectFilled(cell_min, cell_max, preview_color);
      const BvItem &it = items[r * COLUMN_COUNT + c];
      if (it.valid) {
        char text[2] = {static_cast<char>(it.val ? '1' : '0'), '\0'};
        const ImVec2 text_size = ImGui::CalcTextSize(text);
        draw_list->AddText(ImVec2(cell_min.x + (col_w - text_size.x) * 0.5f, cell_min.y + (row_h - text_size.y) * 0.5f), IM_COL32_WHITE, text);
      }
    });
  }

  // Release -- mirrors BinaryView::mouseReleaseEvent(). A plain click (no
  // drag) is left untouched, handled entirely by the existing per-cell
  // IsItemClicked() select-signal path in the loop above.
  if (g_drag.active && g_drag.msg_id == id && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    if (g_drag.dragged) {
      const DragSelection sel = compute_drag_selection(g_drag.anchor_row, g_drag.anchor_col, g_drag.cur_row, g_drag.cur_col, g_drag.resize_sig);
      if (g_drag.resize_sig != nullptr) {
        // Guard against the signal having been removed from underneath the
        // held pointer mid-drag (e.g. the 'x' delete shortcut).
        bool sig_still_exists = false;
        if (const cabana::Msg *m = dbc()->msg(id)) {
          const auto &sigs = m->getSignals();
          sig_still_exists = std::find(sigs.begin(), sigs.end(), g_drag.resize_sig) != sigs.end();
        }
        if (sig_still_exists) {
          cabana::Signal new_sig = *g_drag.resize_sig;
          new_sig.start_bit = sel.start_bit;
          new_sig.size = sel.size;
          new_sig.is_little_endian = sel.is_little_endian;
          UndoStack::push(new EditSignalCommand(id, g_drag.resize_sig, new_sig));
        }
      } else {
        cabana::Signal new_sig{};
        new_sig.start_bit = sel.start_bit;
        new_sig.size = sel.size;
        new_sig.is_little_endian = sel.is_little_endian;
        UndoStack::push(new AddSigCommand(id, new_sig, row_count));
        // dbc()->signalAdded fired synchronously inside push() above and
        // already set selected_signal() to the new signal (see
        // signal_view.cc's ensure_connected() -> handleSignalAdded-style
        // hookup); re-flag it as a binary-view selection so the editor
        // force-expands it too, mirroring BinaryView::signalClicked ->
        // SignalView::selectSignal(sig, /*expand=*/true).
        set_selected_signal(selected_signal(), /*from_binary_view=*/true);
      }
    }
    g_drag = DragState{};
  }

  ImGui::SetCursorScreenPos(origin);
  ImGui::Dummy(ImVec2(total_w, total_h));  // register the grid's footprint for the enclosing scroll region
  pop_mono_font();
}
