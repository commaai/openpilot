// Bit-grid view for the currently selected message -- ports
// tools/cabana/binaryview.{h,cc} (BinaryViewModel + BinaryItemDelegate::paint)
// to immediate-mode drawing. No persistent model: the item grid is rebuilt
// from dbc()->msg(id) + can->lastMessage(id) every call, per the "no caching
// beyond cheap per-frame locals" rule in the phase-2 spec.

#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/dbc/dbcmanager.h"

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

// Phase-3 hook: hover/selection state the signal editor and drag-create will
// reuse. Kept as local static state (not in AppState, which this workstream
// doesn't own) -- structured as plain signal pointers so wiring in the editor
// is a matter of reading these two instead of re-deriving hit-testing.
const cabana::Signal *g_hovered_sig = nullptr;
const cabana::Signal *g_selected_sig = nullptr;

}  // namespace

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

  const int row_count = std::max(dbc_msg != nullptr ? static_cast<int>(dbc_msg->size) : static_cast<int>(last_msg.dat.size()), 0);
  if (row_count == 0) {
    ImGui::TextDisabled("No data");
    return;
  }
  std::vector<BvItem> items = build_signal_coverage(dbc_msg, row_count);

  // Bit-flip heatmap -- mirrors BinaryViewModel::updateState()'s live-mode
  // branch (heatmap_live_mode == true, CanData::bit_flip_counts). The "All"
  // time-range mode from DetailWidget's Live/All radio buttons isn't wired up
  // here -- see report.
  const auto &bit_flips = last_msg.bit_flip_counts;
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

      // background -- mirrors BinaryItemDelegate::paint()'s non-selected branch
      // (drag-select/resize is Phase 3, so that branch is skipped entirely).
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
        if (clicked && top_sig != nullptr) g_selected_sig = top_sig;
      }
    }
  }

  if (!ImGui::IsMouseHoveringRect(origin, ImVec2(origin.x + total_w, origin.y + total_h))) {
    g_hovered_sig = nullptr;  // mirrors BinaryView::leaveEvent() -> highlight(nullptr)
  }

  ImGui::SetCursorScreenPos(origin);
  ImGui::Dummy(ImVec2(total_w, total_h));  // register the grid's footprint for the enclosing scroll region
  pop_mono_font();
}
