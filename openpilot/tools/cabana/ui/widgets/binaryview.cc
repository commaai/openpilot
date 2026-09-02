#include "tools/cabana/ui/widgets/binaryview.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <utility>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"

namespace {

const int CELL_HEIGHT = 36;
const float SMALL_FONT_SIZE = 10.0f;  // Inter needs 10 px for a 7 px cap height
const int VERTICAL_HEADER_WIDTH = 30;
inline int get_bit_pos(const BinaryIndex &index) { return flipBitPos(index.row * 8 + index.column); }

inline ImU32 paletteHighlight() { return ImGui::GetColorU32(ImGuiCol_Header); }
inline ImU32 paletteBase() { return ImGui::GetColorU32(ImGuiCol_ChildBg); }
inline ImU32 paletteText(bool active) { return ImGui::GetColorU32(active ? ImGuiCol_Text : ImGuiCol_TextDisabled); }
const ImU32 DARK_GRAY = IM_COL32(128, 128, 128, 255);

// JetBrains Mono ships no bold variant, so emulate one by drawing the glyphs again a fraction of a
// pixel to the right. Keeps the monospace advance, unlike switching to the proportional bold face.
void drawBoldText(ImDrawList *p, const ImRect &r, const char *text, ImU32 col, ImFont *font, float font_size) {
  drawText(p, r, text, col, font, font_size);
  drawText(p, ImRect(ImVec2(r.Min.x + 0.6f, r.Min.y), ImVec2(r.Max.x + 0.6f, r.Max.y)), text, col, font, font_size);
}

// sparse dots
void fillDense7Pattern(ImDrawList *p, const ImRect &r, ImU32 col) {
  p->PushClipRect(r.Min, r.Max, true);
  for (float y = r.Min.y; y < r.Max.y; y += 4.0f) {
    for (float x = r.Min.x + (static_cast<int>((y - r.Min.y) / 4.0f) % 2) * 2.0f; x < r.Max.x; x += 4.0f) {
      p->AddRectFilled(ImVec2(x, y), ImVec2(x + 1.0f, y + 1.0f), col);
    }
  }
  p->PopClipRect();
}

// backward diagonal lines
void fillBDiagPattern(ImDrawList *p, const ImRect &r, ImU32 col) {
  p->PushClipRect(r.Min, r.Max, true);
  const float h = r.GetHeight();
  for (float x = r.Min.x - h; x < r.Max.x; x += 8.0f) {
    p->AddLine(ImVec2(x, r.Max.y), ImVec2(x + h, r.Min.y), col, 1.0f);
  }
  p->PopClipRect();
}

}  // namespace

BinaryView::BinaryView() {
  connections_.push_back(dbc()->fileChanged.connect([this]() { refresh(); }));
  connections_.push_back(UndoStack::instance()->indexChanged.connect([this]() { refresh(); }));
}

std::string BinaryView::whatsThis() const {
  return R"(
    <b>Binary View</b><br/>
    <span style="color:gray">Shortcuts</span><br />
    Delete Signal:
      <span style="background-color:lightGray;color:gray">&nbsp;x&nbsp;</span>,
      <span style="background-color:lightGray;color:gray">&nbsp;Backspace&nbsp;</span>,
      <span style="background-color:lightGray;color:gray">&nbsp;Delete&nbsp;</span><br />
    Change endianness: <span style="background-color:lightGray;color:gray">&nbsp;e&nbsp; </span><br />
    Change signedness: <span style="background-color:lightGray;color:gray">&nbsp;s&nbsp;</span><br />
    Open chart:
      <span style="background-color:lightGray;color:gray">&nbsp;c&nbsp;</span>,
      <span style="background-color:lightGray;color:gray">&nbsp;p&nbsp;</span>,
      <span style="background-color:lightGray;color:gray">&nbsp;g&nbsp;</span>
  )";
}

void BinaryView::addShortcuts() {
  const ImGuiIO &io = ImGui::GetIO();
  if (io.WantTextInput || io.KeyCtrl || io.KeySuper) return;
  if (ImGui::GetTopMostPopupModal() != nullptr) return;  // a modal dialog blocks the shortcuts

  if (ImGui::IsKeyPressed(ImGuiKey_X, false) || ImGui::IsKeyPressed(ImGuiKey_Backspace, false) || ImGui::IsKeyPressed(ImGuiKey_Delete, false)) {
    if (hovered_sig_ != nullptr) {
      UndoStack::instance()->push(new RemoveSigCommand(msg_id_, hovered_sig_));
      hovered_sig_ = nullptr;
    }
  }

  if (ImGui::IsKeyPressed(ImGuiKey_E, false)) {
    if (hovered_sig_ != nullptr) {
      cabana::Signal s = *hovered_sig_;
      s.is_little_endian = !s.is_little_endian;
      editSignal(hovered_sig_, s);
    }
  }

  if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
    if (hovered_sig_ != nullptr) {
      cabana::Signal s = *hovered_sig_;
      s.is_signed = !s.is_signed;
      editSignal(hovered_sig_, s);
    }
  }

  if (ImGui::IsKeyPressed(ImGuiKey_P, false) || ImGui::IsKeyPressed(ImGuiKey_G, false) || ImGui::IsKeyPressed(ImGuiKey_C, false)) {
    if (hovered_sig_ != nullptr) {
      showChart(msg_id_, hovered_sig_, true, false);
    }
  }
}

ImVec2 BinaryView::minimumSizeHint() const {
  // widest fixed-font glyph plus the header margins
  pushMonoFont();
  const float min_section_size = ImGui::CalcTextSize("W").x + 8.0f;
  popMonoFont();
  return {(min_section_size + 1) * 9 + VERTICAL_HEADER_WIDTH + 2,
          static_cast<float>(CELL_HEIGHT * std::min(row_count_, 10) + 2)};
}

void BinaryView::highlight(const cabana::Signal *sig) {
  if (sig != hovered_sig_) {
    hovered_sig_ = sig;
    signalHovered(hovered_sig_);
  }
}

void BinaryView::setSelection() {
  auto index = indexAt(last_mouse_pos_);
  if (!anchor_index_.isValid() || !index.isValid())
    return;

  std::set<BinaryIndex> selection;
  auto [start, size, is_lb] = getSelection(index);
  for (int i = 0; i < size; ++i) {
    int pos = is_lb ? flipBitPos(start + i) : flipBitPos(start) + i;
    selection.insert({pos / 8, pos % 8});
  }
  selection_ = std::move(selection);
}

void BinaryView::handleMousePress(const ImVec2 &pos) {
  resize_sig_ = nullptr;
  if (auto index = indexAt(last_mouse_pos_ = pos); index.isValid() && index.column != HEX_COLUMN) {
    anchor_index_ = index;
    auto item = &cellAt(anchor_index_);
    int bit_pos = get_bit_pos(anchor_index_);
    for (auto s : item->sigs) {
      if (bit_pos == s->lsb || bit_pos == s->msb) {
        int idx = flipBitPos(bit_pos == s->lsb ? s->msb : s->lsb);
        anchor_index_ = {idx / 8, idx % 8};
        resize_sig_ = s;
        break;
      }
    }
  }
}

void BinaryView::highlightPosition(const ImVec2 &pos) {
  if (auto index = indexAt(pos); index.isValid()) {
    auto item = &cellAt(index);
    const cabana::Signal *sig = item->sigs.empty() ? nullptr : item->sigs.back();
    highlight(sig);
  }
}

void BinaryView::handleMouseMove(const ImVec2 &pos) {
  highlightPosition(last_mouse_pos_ = pos);
  // drag selecting while the left button is down; the hex column is not selectable
  if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && indexAt(pos).column != HEX_COLUMN) setSelection();
}

void BinaryView::handleMouseRelease(const ImVec2 &pos) {
  auto release_index = indexAt(pos);
  if (release_index.isValid() && anchor_index_.isValid()) {
    if (hasSelection()) {
      auto sig = resize_sig_ ? *resize_sig_ : cabana::Signal{};
      std::tie(sig.start_bit, sig.size, sig.is_little_endian) = getSelection(release_index);
      resize_sig_ ? editSignal(resize_sig_, sig)
                 : UndoStack::instance()->push(new AddSigCommand(msg_id_, sig));
    } else {
      auto item = &cellAt(anchor_index_);
      if (item->sigs.size() > 0)
        signalClicked(item->sigs.back());
    }
  }
  selection_.clear();
  anchor_index_ = BinaryIndex();
  resize_sig_ = nullptr;
}

void BinaryView::setMessage(const MessageId &message_id) {
  msg_id_ = message_id;
  scroll_to_top_ = true;
  refresh();
}

void BinaryView::refresh() {
  selection_.clear();
  anchor_index_ = BinaryIndex();
  resize_sig_ = nullptr;
  hovered_sig_ = nullptr;
  bit_flip_tracker_ = {};
  cells_.clear();
  if (auto dbc_msg = dbc()->msg(msg_id_)) {
    row_count_ = dbc_msg->size;
    cells_.resize(row_count_ * COLUMN_COUNT);
    for (auto sig : dbc_msg->getSignals()) {
      for (int j = 0; j < sig->size; ++j) {
        int pos = sig->is_little_endian ? flipBitPos(sig->start_bit + j) : flipBitPos(sig->start_bit) + j;
        int idx = COLUMN_COUNT * (pos / 8) + pos % 8;
        if (idx >= cells_.size()) {
          fprintf(stderr, "signal %s out of bounds.start_bit: %d size: %d\n", sig->name.c_str(), sig->start_bit, sig->size);
          break;
        }
        if (j == 0) sig->is_little_endian ? cells_[idx].is_lsb = true : cells_[idx].is_msb = true;
        if (j == sig->size - 1) sig->is_little_endian ? cells_[idx].is_msb = true : cells_[idx].is_lsb = true;

        auto &sigs = cells_[idx].sigs;
        sigs.push_back(sig);
        if (sigs.size() > 1) {
          std::sort(sigs.begin(), sigs.end(), [](auto l, auto r) { return l->size > r->size; });
        }
      }
    }
  } else {
    row_count_ = can->lastMessage(msg_id_).dat.size();
    cells_.resize(row_count_ * COLUMN_COUNT);
  }
  updateState();
  if (under_mouse_) highlightPosition(last_mouse_pos_);
}


std::set<const cabana::Signal *> BinaryView::getOverlappingSignals() const {
  std::set<const cabana::Signal *> overlapping;
  for (const auto &item : cells_) {
    if (item.sigs.size() > 1) {
      for (auto s : item.sigs) {
        if (s->type == cabana::Signal::Type::Normal) overlapping.insert(s);
      }
    }
  }
  return overlapping;
}

std::tuple<int, int, bool> BinaryView::getSelection(BinaryIndex index) {
  if (index.column == HEX_COLUMN) {
    index = {index.row, 7};
  }
  bool is_lb = true;
  if (resize_sig_) {
    is_lb = resize_sig_->is_little_endian;
  } else if (settings.drag_direction == Settings::DragDirection::MsbFirst) {
    is_lb = index < anchor_index_;
  } else if (settings.drag_direction == Settings::DragDirection::LsbFirst) {
    is_lb = !(index < anchor_index_);
  } else if (settings.drag_direction == Settings::DragDirection::AlwaysLE) {
    is_lb = true;
  } else if (settings.drag_direction == Settings::DragDirection::AlwaysBE) {
    is_lb = false;
  }

  int cur_bit_pos = get_bit_pos(index);
  int anchor_bit_pos = get_bit_pos(anchor_index_);
  int start_bit = is_lb ? std::min(cur_bit_pos, anchor_bit_pos) : get_bit_pos(std::min(index, anchor_index_));
  int size = is_lb ? std::abs(cur_bit_pos - anchor_bit_pos) + 1 : std::abs(flipBitPos(cur_bit_pos) - flipBitPos(anchor_bit_pos)) + 1;
  return {start_bit, size, is_lb};
}

BinaryIndex BinaryView::indexAt(const ImVec2 &pos) const {
  if (column_width_ <= 0 || pos.x < grid_pos_.x + VERTICAL_HEADER_WIDTH || pos.y < grid_pos_.y) return {};
  int column = static_cast<int>((pos.x - grid_pos_.x - VERTICAL_HEADER_WIDTH) / column_width_);
  int row = static_cast<int>((pos.y - grid_pos_.y) / CELL_HEIGHT);
  if (column >= COLUMN_COUNT || row >= row_count_) return {};
  return {row, column};
}

ImRect BinaryView::visualRect(const BinaryIndex &index) const {
  // sections are integral: round the edges so neighboring cells share them exactly and the cells are
  // painted edge to edge with no grid line between them
  const float x0 = grid_pos_.x + VERTICAL_HEADER_WIDTH + IM_ROUND(index.column * column_width_);
  const float x1 = grid_pos_.x + VERTICAL_HEADER_WIDTH + IM_ROUND((index.column + 1) * column_width_);
  const float y = grid_pos_.y + index.row * CELL_HEIGHT;
  return ImRect(x0, y, x1, y + CELL_HEIGHT);
}

void BinaryView::draw() {
  is_message_active_ = can->isMessageActive(msg_id_);
  if (scroll_to_top_) {
    ImGui::SetScrollY(0.0f);
    scroll_to_top_ = false;
  }

  const int rows = row_count_;
  const float width = ImGui::GetContentRegionAvail().x;
  column_width_ = std::max(1.0f, (width - VERTICAL_HEADER_WIDTH) / COLUMN_COUNT);
  grid_pos_ = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##binary_view", ImVec2(std::max(width, 1.0f), std::max(static_cast<float>(rows * CELL_HEIGHT), 1.0f)));
  ImDrawList *painter = ImGui::GetWindowDrawList();

  for (int row = 0; row < rows; ++row) {
    const ImRect r(grid_pos_.x, grid_pos_.y + row * CELL_HEIGHT, grid_pos_.x + VERTICAL_HEADER_WIDTH, grid_pos_.y + (row + 1) * CELL_HEIGHT);
    painter->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_WindowBg));  // plain header background
    drawText(painter, r, std::to_string(row).c_str(), ImGui::GetColorU32(ImGuiCol_Text));
  }
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < COLUMN_COUNT; ++column) {
      const BinaryIndex index = {row, column};
      paintCell(painter, visualRect(index), index);
    }
  }

  const ImVec2 mouse = ImGui::GetMousePos();
  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();
  const bool under_mouse = (hovered || active) && ImGui::IsMouseHoveringRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), false);
  if (hovered || active) {
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) handleMousePress(mouse);
    const ImVec2 delta = ImGui::GetIO().MouseDelta;
    if (delta.x != 0.0f || delta.y != 0.0f) {
      handleMouseMove(mouse);
    } else {
      // imgui only reports a delta on the frames the mouse actually moves, so recompute the hovered
      // signal every frame the mouse is inside the widget, or the shortcuts stay inert after a click
      highlightPosition(last_mouse_pos_ = mouse);
    }
  }
  // the mouse left the widget rect, also while dragging
  if (std::exchange(under_mouse_, under_mouse) && !under_mouse) highlight(nullptr);
  if (ImGui::IsItemDeactivated()) handleMouseRelease(mouse);

  if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
    if (auto index = indexAt(mouse); index.isValid() && !cellAt(index).sigs.empty()) {
      ImGui::SetTooltip("%s", utils::stripHtml(utils::signalToolTip(cellAt(index).sigs.back())).c_str());
    }
  }

  addShortcuts();
}

void BinaryView::setCell(int row, int col, uint8_t val, const CabanaColor &color) {
  auto &item = cells_[row * COLUMN_COUNT + col];
  item.valid = true;
  item.val = val;
  item.bg_color = color;
}

void BinaryView::updateState() {
  const auto &last_msg = can->lastMessage(msg_id_);
  const auto &binary = last_msg.dat;
  if (binary.size() > row_count_) {
    row_count_ = binary.size();
    cells_.resize(row_count_ * COLUMN_COUNT);
  }

  auto &bit_flips = heatmap_live_mode_ ? last_msg.bit_flip_counts : bitFlipChanges(binary.size());
  uint32_t max_bit_flip_count = 1;  // 1 to avoid division by zero
  for (const auto &row : bit_flips) {
    for (uint32_t count : row) {
      max_bit_flip_count = std::max(max_bit_flip_count, count);
    }
  }

  const bool dark = isDarkTheme();
  const double max_alpha = 255.0;
  const double min_alpha_with_signal = dark ? 70.0 : 25.0;  // Base alpha for small flip counts
  const double min_alpha_no_signal = dark ? 28.0 : 10.0;    // Base alpha for small flip counts for no signal bits
  const double alpha_gamma = dark ? 0.6 : 1.0;
  const double log_factor = 1.0 + 0.2;
  const double log_scaler = max_alpha / log2(log_factor * max_bit_flip_count);

  for (size_t i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      auto &item = cells_[i * COLUMN_COUNT + j];
      int bit_val = (binary[i] >> (7 - j)) & 1;

      double alpha = item.sigs.empty() ? 0 : min_alpha_with_signal;
      uint32_t flip_count = bit_flips[i][j];
      if (flip_count > 0) {
        double normalized_alpha = log2(1.0 + flip_count * log_factor) * log_scaler;
        normalized_alpha = max_alpha * std::pow(std::clamp(normalized_alpha / max_alpha, 0.0, 1.0), alpha_gamma);
        double min_alpha = item.sigs.empty() ? min_alpha_no_signal : min_alpha_with_signal;
        alpha = std::clamp(normalized_alpha, min_alpha, max_alpha);
      }

      auto color = item.bg_color;
      color.a = static_cast<uint8_t>(alpha);
      setCell(i, j, bit_val, color);
    }
    setCell(i, HEX_COLUMN, binary[i], last_msg.colors[i]);
  }
}

const std::vector<std::array<uint32_t, 8>> &BinaryView::bitFlipChanges(size_t msg_size) {
  auto time_range = can->timeRange();
  if (bit_flip_tracker_.time_range == time_range && !bit_flip_tracker_.flip_counts.empty())
    return bit_flip_tracker_.flip_counts;

  bit_flip_tracker_.time_range = time_range;
  bit_flip_tracker_.flip_counts.assign(msg_size, std::array<uint32_t, 8>{});

  auto [first, last] = can->eventsInRange(msg_id_, time_range);
  if (std::distance(first, last) <= 1) return bit_flip_tracker_.flip_counts;

  std::vector<uint8_t> prev_values((*first)->dat, (*first)->dat + (*first)->size);
  for (auto it = std::next(first); it != last; ++it) {
    const CanEvent *event = *it;
    int size = std::min<int>(msg_size, event->size);
    for (int i = 0; i < size; ++i) {
      const uint8_t diff = event->dat[i] ^ prev_values[i];
      if (!diff) continue;

      auto &bit_flips = bit_flip_tracker_.flip_counts[i];
      for (int bit = 0; bit < 8; ++bit) {
        if (diff & (1u << bit)) ++bit_flips[7 - bit];
      }
      prev_values[i] = event->dat[i];
    }
  }

  return bit_flip_tracker_.flip_counts;
}

bool BinaryView::hasSignal(const BinaryIndex &index, int dx, int dy, const cabana::Signal *sig) const {
  if (!index.isValid()) return false;
  int idx = (index.row + dy) * COLUMN_COUNT + index.column + dx;
  if (idx < 0 || idx >= (int)cells_.size()) return false;
  auto &s = cells_[idx].sigs;
  return std::find(s.begin(), s.end(), sig) != s.end();
}

void BinaryView::paintCell(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index) const {
  auto item = &cellAt(index);
  ImFont *font = ImGui::GetFont();
  float font_size = ImGui::GetFontSize();
  ImU32 pen = paletteText(is_message_active_);

  if (index.column == HEX_COLUMN) {
    if (item->valid) {
      pushMonoFont();
      font = ImGui::GetFont();
      font_size = ImGui::GetFontSize();
      popMonoFont();
      painter->AddRectFilled(rect.Min, rect.Max, toImU32(item->bg_color));
    }
  } else if (isSelected(index)) {
    auto color = resize_sig_ ? toImU32(resize_sig_->color) : paletteHighlight();
    painter->AddRectFilled(rect.Min, rect.Max, color);
    pen = paletteBrightText();
  } else if (!hasSelection() || std::find(item->sigs.begin(), item->sigs.end(), resize_sig_) == item->sigs.end()) {  // not resizing
    if (item->sigs.size() > 0) {
      for (auto &s : item->sigs) {
        if (s == hovered_sig_) {
          painter->AddRectFilled(rect.Min, rect.Max, toImU32(s->color.darker(125)));  // 4/5x brightness
        } else {
          drawSignalCell(painter, rect, index, s);
        }
      }
    } else if (item->valid && item->bg_color.alpha() > 0) {
      painter->AddRectFilled(rect.Min, rect.Max, toImU32(item->bg_color));
    }
    bool bright = std::find(item->sigs.begin(), item->sigs.end(), hovered_sig_) != item->sigs.end();
    pen = bright ? paletteBrightText() : paletteText(is_message_active_);
  }

  if (item->sigs.size() > 1) {
    fillDense7Pattern(painter, rect, DARK_GRAY);
  } else if (!item->valid) {
    fillBDiagPattern(painter, rect, DARK_GRAY);
  }
  if (item->valid) {
    if (index.column == HEX_COLUMN) {
      drawBoldText(painter, rect, utils::hexByte(item->val), pen, font, font_size);
    } else {
      drawText(painter, rect, item->val ? "1" : "0", pen, font, font_size);
    }
  }
  if (item->is_msb || item->is_lsb) {
    const ImRect marker_rect(rect.Min, ImVec2(rect.Max.x - 8, rect.Max.y - 3));
    drawText(painter, marker_rect, item->is_msb ? "M" : "L", pen, nullptr, SMALL_FONT_SIZE, ImVec2(1.0f, 1.0f));
  }
}

// Draw border on edge of signal
void BinaryView::drawSignalCell(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index, const cabana::Signal *sig) const {
  bool draw_left = !hasSignal(index, -1, 0, sig);
  bool draw_top = !hasSignal(index, 0, -1, sig);
  bool draw_right = !hasSignal(index, 1, 0, sig);
  bool draw_bottom = !hasSignal(index, 0, 1, sig);

  const int spacing = 2;
  ImRect rc(rect.Min.x + draw_left * 3, rect.Min.y + draw_top * spacing, rect.Max.x - draw_right * 3, rect.Max.y - draw_bottom * spacing);
  std::vector<ImRect> subtract;
  if (!draw_top) {
    if (!draw_left && !hasSignal(index, -1, -1, sig)) {
      subtract.emplace_back(rc.Min.x, rc.Min.y, rc.Min.x + 3, rc.Min.y + spacing);
    } else if (!draw_right && !hasSignal(index, 1, -1, sig)) {
      subtract.emplace_back(rc.Max.x - 3, rc.Min.y, rc.Max.x, rc.Min.y + spacing);
    }
  }
  if (!draw_bottom) {
    if (!draw_left && !hasSignal(index, -1, 1, sig)) {
      subtract.emplace_back(rc.Min.x, rc.Max.y - spacing, rc.Min.x + 3, rc.Max.y);
    } else if (!draw_right && !hasSignal(index, 1, 1, sig)) {
      subtract.emplace_back(rc.Max.x - 3, rc.Max.y - spacing, rc.Max.x, rc.Max.y);
    }
  }
  // rc split into horizontal bands with the notch corners removed: at most one notch in the top band and
  // one in the bottom band
  const ImRect *top_notch = !subtract.empty() && subtract.front().Min.y == rc.Min.y ? &subtract.front() : nullptr;
  const ImRect *bottom_notch = !subtract.empty() && subtract.back().Min.y != rc.Min.y ? &subtract.back() : nullptr;
  std::vector<ImRect> region;
  auto band = [&](const ImRect *notch, float y0, float y1) {
    const float x0 = notch && notch->Min.x == rc.Min.x ? notch->Max.x : rc.Min.x;
    const float x1 = notch && notch->Min.x != rc.Min.x ? notch->Min.x : rc.Max.x;
    if (x1 > x0 && y1 > y0) region.emplace_back(x0, y0, x1, y1);
  };
  if (top_notch) band(top_notch, rc.Min.y, rc.Min.y + spacing);
  band(nullptr, rc.Min.y + (top_notch ? spacing : 0), rc.Max.y - (bottom_notch ? spacing : 0));
  if (bottom_notch) band(bottom_notch, rc.Max.y - spacing, rc.Max.y);

  auto item = &cellAt(index);
  CabanaColor color = sig->color;
  color.a = item->bg_color.alpha();
  const ImU32 edge = toImU32(sig->color.darker(125));

  for (const ImRect &clip : region) {
    painter->PushClipRect(clip.Min, clip.Max, true);
    // mix the signal color with the background to fade it
    painter->AddRectFilled(rc.Min, rc.Max, paletteBase());
    painter->AddRectFilled(rc.Min, rc.Max, toImU32(color));

    if (draw_left) painter->AddLine(ImVec2(rc.Min.x + 0.5f, rc.Min.y), ImVec2(rc.Min.x + 0.5f, rc.Max.y), edge, 1.0f);
    if (draw_right) painter->AddLine(ImVec2(rc.Max.x - 0.5f, rc.Min.y), ImVec2(rc.Max.x - 0.5f, rc.Max.y), edge, 1.0f);
    if (draw_bottom) painter->AddLine(ImVec2(rc.Min.x, rc.Max.y - 0.5f), ImVec2(rc.Max.x, rc.Max.y - 0.5f), edge, 1.0f);
    if (draw_top) painter->AddLine(ImVec2(rc.Min.x, rc.Min.y + 0.5f), ImVec2(rc.Max.x, rc.Min.y + 0.5f), edge, 1.0f);

    // fill gaps inside corners: the 2px stroke is clipped to the region, only the half outside the notch is painted
    for (auto &r : subtract) {
      painter->AddRect(r.Min, r.Max, edge, 0.0f, 0, 2.0f);
    }
    painter->PopClipRect();
  }
}
