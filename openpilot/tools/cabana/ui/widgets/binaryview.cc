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
#include "tools/cabana/ui/widgets/binarysignals.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"

namespace {

const int CELL_HEIGHT = 36;
const float CELL_FONT_SIZE = UI_FONT_SIZE + 2.0f;
const float SMALL_FONT_SIZE = 10.0f;  // Inter needs 10 px for a 7 px cap height
const int GRID_COLUMN_COUNT = BinaryView::COLUMN_COUNT + 1;
inline int get_bit_pos(const BinaryIndex &index) { return flipBitPos(index.row * 8 + index.column); }

inline ImU32 paletteBase() { return ImGui::GetColorU32(ImGuiCol_ChildBg); }
inline ImU32 paletteText() { return ImGui::GetColorU32(ImGuiCol_Text); }

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
  heatmap_live_mode_ = can->liveStreaming();
  connections_.push_back(can->eventsMerged.connect([this](const MessageEventsMap &events) {
    if (events.count(msg_id_)) {
      bit_flip_tracker_.valid = false;
      if (!heatmap_live_mode_) updateState();
    }
  }));
  connections_.push_back(dbc()->fileChanged.connect([this]() { refresh(); }));
  connections_.push_back(UndoStack::instance()->indexChanged.connect([this]() { refresh(); }));
}

std::string BinaryView::whatsThis() const {
  return R"(
    <b>Binary View</b><br/>
    All / selected range: brighter bits and bytes changed more often; uncolored cells did not change.<br/>
    Live: accumulated bit flips and fading byte changes during playback.<br/>
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
  // Match the enlarged hex font when reserving space for narrow panels.
  pushMonoFont(CELL_FONT_SIZE);
  const float min_section_size = std::ceil(ImGui::CalcTextSize("FF").x) + 10.0f;
  popMonoFont();
  return {min_section_size * GRID_COLUMN_COUNT + 2,
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
  } else {
    highlight(nullptr);
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
  visible_signals_.clear();
  const auto *msg = dbc()->msg(msg_id_);
  row_count_ = msg ? msg->size : can->lastMessage(msg_id_).dat.size();
  cells_.resize(row_count_ * COLUMN_COUNT);
  updateState();
  if (under_mouse_) highlightPosition(last_mouse_pos_);
}

void BinaryView::updateSignals() {
  const auto &data = can->lastMessage(msg_id_).dat;
  auto signals = binaryViewSignals(dbc()->msg(msg_id_), data.data(), data.size());
  if (signals == visible_signals_) return;
  visible_signals_ = std::move(signals);

  // A branch switch invalidates the old hover and any in-progress resize.
  selection_.clear();
  anchor_index_ = {};
  resize_sig_ = nullptr;
  highlight(nullptr);
  for (auto &cell : cells_) {
    cell.sigs.clear();
    cell.is_msb = cell.is_lsb = false;
  }
  for (auto sig : visible_signals_) {
    for (int j = 0; j < sig->size; ++j) {
      int pos = sig->is_little_endian ? flipBitPos(sig->start_bit + j) : flipBitPos(sig->start_bit) + j;
      int idx = COLUMN_COUNT * (pos / 8) + pos % 8;
      if (idx < 0 || idx >= cells_.size()) break;
      if (j == 0) sig->is_little_endian ? cells_[idx].is_lsb = true : cells_[idx].is_msb = true;
      if (j == sig->size - 1) sig->is_little_endian ? cells_[idx].is_msb = true : cells_[idx].is_lsb = true;
      cells_[idx].sigs.push_back(sig);
    }
  }
  for (auto &cell : cells_) {
    std::stable_sort(cell.sigs.begin(), cell.sigs.end(), [](auto l, auto r) { return l->size > r->size; });
  }
  if (under_mouse_) highlightPosition(last_mouse_pos_);
  signalsChanged();
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
  if (column_width_ <= 0 || pos.x < grid_pos_.x + IM_ROUND(column_width_) || pos.y < grid_pos_.y) return {};
  int column = 0;
  while (column < COLUMN_COUNT && pos.x >= grid_pos_.x + IM_ROUND((column + 2) * column_width_)) ++column;
  int row = static_cast<int>((pos.y - grid_pos_.y) / CELL_HEIGHT);
  if (column >= COLUMN_COUNT || row >= row_count_) return {};
  return {row, column};
}

ImRect BinaryView::visualRect(const BinaryIndex &index) const {
  // Round shared edges consistently across the byte index, bits, and hex column.
  const float x0 = grid_pos_.x + IM_ROUND((index.column + 1) * column_width_);
  const float x1 = grid_pos_.x + IM_ROUND((index.column + 2) * column_width_);
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
  // Keep hex bytes readable in narrow panels by scrolling instead of shrinking further.
  pushMonoFont(CELL_FONT_SIZE);
  const float min_column_width = std::ceil(ImGui::CalcTextSize("FF").x) + 10.0f;
  popMonoFont();
  const float width = std::max(ImGui::GetContentRegionAvail().x, min_column_width * GRID_COLUMN_COUNT);
  column_width_ = std::max(min_column_width, width / GRID_COLUMN_COUNT);
  grid_pos_ = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##binary_view", ImVec2(std::max(width, 1.0f), std::max(static_cast<float>(rows * CELL_HEIGHT), 1.0f)));
  ImDrawList *painter = ImGui::GetWindowDrawList();
  painter->AddRectFilled(grid_pos_, ImVec2(grid_pos_.x + width, grid_pos_.y + rows * CELL_HEIGHT), paletteBase());

  for (int row = 0; row < rows; ++row) {
    const ImRect r(grid_pos_.x, grid_pos_.y + row * CELL_HEIGHT, grid_pos_.x + IM_ROUND(column_width_), grid_pos_.y + (row + 1) * CELL_HEIGHT);
    drawText(painter, r, std::to_string(row).c_str(), paletteText(), nullptr, CELL_FONT_SIZE);
  }
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < COLUMN_COUNT; ++column) {
      const BinaryIndex index = {row, column};
      paintCell(painter, visualRect(index), index);
    }
  }

  // One consistent grid over every cell; hover never changes its geometry.
  const ImU32 grid_color = ImGui::GetColorU32(ImGuiCol_Border, 0.35f);
  const float right = grid_pos_.x + IM_ROUND(GRID_COLUMN_COUNT * column_width_);
  const float bottom = grid_pos_.y + rows * CELL_HEIGHT;
  for (int column = 0; column <= GRID_COLUMN_COUNT; ++column) {
    const float x = std::clamp(grid_pos_.x + IM_ROUND(column * column_width_), grid_pos_.x + 0.5f, right - 0.5f);
    painter->AddLine(ImVec2(x, grid_pos_.y), ImVec2(x, bottom), grid_color);
  }
  for (int row = 0; rows > 0 && row <= rows; ++row) {
    const float y = std::clamp(grid_pos_.y + row * CELL_HEIGHT, grid_pos_.y + 0.5f, bottom - 0.5f);
    painter->AddLine(ImVec2(grid_pos_.x, y), ImVec2(right, y), grid_color);
  }

  // Signal definitions are a separate layer from heatmap activity.
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < HEX_COLUMN; ++column) {
      const BinaryIndex index{row, column};
      for (const auto *sig : cellAt(index).sigs) {
        if (!hasSelection() || sig != resize_sig_) drawSignalOutline(painter, visualRect(index), index, sig);
      }
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

  for (auto &cell : cells_) cell.valid = false;
  updateSignals();

  auto &bit_flips = heatmap_live_mode_ ? last_msg.bit_flip_counts : bitFlipChanges(binary.size());
  uint32_t max_bit_flip_count = 1;  // 1 to avoid division by zero
  for (const auto &row : bit_flips) {
    for (uint32_t count : row) {
      max_bit_flip_count = std::max(max_bit_flip_count, count);
    }
  }

  uint32_t max_byte_flip_count = 1;
  for (auto count : bit_flip_tracker_.counts.bytes) max_byte_flip_count = std::max(max_byte_flip_count, count);

  // Use the same logarithmic intensity mapping in both themes.
  const double max_alpha = 255.0;
  const double min_alpha_with_signal = 25.0;  // Base alpha for small flip counts
  const double min_alpha_no_signal = 10.0;    // Base alpha for small flip counts for no signal bits
  const double log_factor = 1.0 + 0.2;
  const double log_scaler = max_alpha / log2(1.0 + log_factor * max_bit_flip_count);

  for (size_t i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      auto &item = cells_[i * COLUMN_COUNT + j];
      int bit_val = (binary[i] >> (7 - j)) & 1;

      double alpha = item.sigs.empty() ? 0 : min_alpha_with_signal;
      uint32_t flip_count = bit_flips[i][j];
      if (flip_count > 0) {
        double normalized_alpha = log2(1.0 + flip_count * log_factor) * log_scaler;
        double min_alpha = item.sigs.empty() ? min_alpha_no_signal : min_alpha_with_signal;
        alpha = std::clamp(normalized_alpha, min_alpha, max_alpha);
      }

      auto color = item.bg_color;
      color.a = static_cast<uint8_t>(alpha);
      setCell(i, j, bit_val, color);
    }
    auto byte_color = last_msg.colors[i];
    if (!heatmap_live_mode_) {
      const auto count = bit_flip_tracker_.counts.bytes[i];
      const double intensity = std::log2(1.0 + count * log_factor) / std::log2(1.0 + max_byte_flip_count * log_factor);
      byte_color = CabanaColor(102, 86, 169, static_cast<uint8_t>(max_alpha * intensity));
    }
    setCell(i, HEX_COLUMN, binary[i], byte_color);
  }
}

const std::vector<std::array<uint32_t, 8>> &BinaryView::bitFlipChanges(size_t msg_size) {
  auto time_range = can->timeRange();
  if (bit_flip_tracker_.valid && bit_flip_tracker_.time_range == time_range &&
      bit_flip_tracker_.counts.bits.size() == msg_size) return bit_flip_tracker_.counts.bits;

  bit_flip_tracker_.time_range = time_range;
  bit_flip_tracker_.counts = HeatmapCounts(msg_size);
  auto [first, last] = can->eventsInRange(msg_id_, time_range);
  for (auto it = first; it != last; ++it) bit_flip_tracker_.counts.add((*it)->dat, (*it)->size);
  bit_flip_tracker_.valid = true;
  return bit_flip_tracker_.counts.bits;
}

void BinaryView::paintCell(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index) const {
  auto item = &cellAt(index);
  ImFont *font = ImGui::GetFont();
  float font_size = CELL_FONT_SIZE;
  ImU32 pen = paletteText();
  const bool hovered = hovered_sig_ && std::find(item->sigs.begin(), item->sigs.end(), hovered_sig_) != item->sigs.end();

  if (index.column == HEX_COLUMN) {
    if (item->valid) {
      pushMonoFont(CELL_FONT_SIZE);
      font = ImGui::GetFont();
      font_size = ImGui::GetFontSize();
      popMonoFont();
      painter->AddRectFilled(rect.Min, rect.Max, toImU32(byteColor(item->bg_color)));
    }
  } else if (isSelected(index)) {
    painter->AddRectFilled(rect.Min, rect.Max, toImU32(signalHighlight(resize_sig_ ? resize_sig_->color : fromImVec4(palette().header))));
    pen = IM_COL32_WHITE;
  } else if (!hasSelection() || std::find(item->sigs.begin(), item->sigs.end(), resize_sig_) == item->sigs.end()) {  // not resizing
    if (item->sigs.size() > 0) {
      for (auto &s : item->sigs) {
        drawSignalCell(painter, rect, index, s);
      }
      // Hover covers the entire signal, including bits shared with another definition.
      if (hovered) painter->AddRectFilled(rect.Min, rect.Max, toImU32(signalHighlight(hovered_sig_->color)));
    } else if (item->valid) {
      if (item->bg_color.alpha() > 0) painter->AddRectFilled(rect.Min, rect.Max, toImU32(signalFill(item->bg_color, false)));
    }
    pen = hovered ? IM_COL32_WHITE : paletteText();
  }

  const auto text = fromImVec4(ImGui::ColorConvertU32ToFloat4(pen));
  const ImU32 pattern = toImU32(contrastColor({128, 128, 128}, text));
  if (item->sigs.size() > 1) {
    fillDense7Pattern(painter, rect, pattern);
  } else if (!item->valid) {
    fillBDiagPattern(painter, rect, pattern);
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

bool BinaryView::hasSignal(const BinaryIndex &index, int dx, int dy, const cabana::Signal *sig) const {
  const int row = index.row + dy;
  const int column = index.column + dx;
  if (row < 0 || row >= row_count_ || column < 0 || column >= HEX_COLUMN) return false;
  const auto &signals = cellAt({row, column}).sigs;
  return std::find(signals.begin(), signals.end(), sig) != signals.end();
}

void BinaryView::drawSignalOutline(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index,
                                   const cabana::Signal *sig) const {
  // Definition boundaries remain visible even at zero activity. Keep them above the
  // fills and grid, at the same fixed position before and during hover.
  const ImU32 edge = toImU32(signalOutline(sig->color, sig == hovered_sig_));
  const bool left = !hasSignal(index, -1, 0, sig);
  const bool right = !hasSignal(index, 1, 0, sig);
  const bool top = !hasSignal(index, 0, -1, sig);
  const bool bottom = !hasSignal(index, 0, 1, sig);
  // Extend one pixel into adjacent cells to join inward corners.
  auto offset = [&](bool boundary, int dx, int dy) {
    return boundary ? 1.0f : hasSignal(index, dx, dy, sig) ? -1.0f : 0.0f;
  };
  const ImVec2 min = rect.Min, max = rect.Max;
  if (left) painter->AddLine(ImVec2(min.x + 1, min.y + offset(top, -1, -1)), ImVec2(min.x + 1, max.y - offset(bottom, -1, 1)), edge);
  if (right) painter->AddLine(ImVec2(max.x - 1, min.y + offset(top, 1, -1)), ImVec2(max.x - 1, max.y - offset(bottom, 1, 1)), edge);
  if (top) painter->AddLine(ImVec2(min.x + offset(left, -1, -1), min.y + 1), ImVec2(max.x - offset(right, 1, -1), min.y + 1), edge);
  if (bottom) painter->AddLine(ImVec2(min.x + offset(left, -1, 1), max.y - 1), ImVec2(max.x - offset(right, 1, 1), max.y - 1), edge);
}

// Signal intensity and hover share the same full-cell geometry.
void BinaryView::drawSignalCell(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index, const cabana::Signal *sig) const {
  CabanaColor color = sig->color;
  color.a = 96 + cellAt(index).bg_color.alpha() * (255 - 96) / 255;
  painter->AddRectFilled(rect.Min, rect.Max, paletteBase());
  painter->AddRectFilled(rect.Min, rect.Max, toImU32(signalFill(color)));
}
