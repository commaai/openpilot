#include "tools/cabana/ui/widgets/binaryview.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <utility>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"

// Qt rendered the tooltip as rich text; drop the tags for the plain text imgui tooltip
static std::string stripHtml(const std::string &s) {
  std::string out;
  bool in_tag = false;
  for (char c : s) {
    if (c == '<') in_tag = true;
    else if (c == '>') in_tag = false;
    else if (!in_tag) out += c;
  }
  size_t b = out.find_first_not_of(" \n"), e = out.find_last_not_of(" \n");
  return b == std::string::npos ? std::string() : out.substr(b, e - b + 1);
}

// BinaryView

const int CELL_HEIGHT = 36;
const int VERTICAL_HEADER_WIDTH = 30;
inline int get_bit_pos(const BinaryIndex &index) { return flipBitPos(index.row * 8 + index.column); }

namespace {

inline ImU32 toImColor(const CabanaColor &color) {
  return IM_COL32(color.r, color.g, color.b, color.a);
}

// QPalette roles (see DarkTheme in utils/util.h and the light style in ui/style.cc)
inline ImU32 paletteHighlight() { return ImGui::GetColorU32(ImGuiCol_Header); }
inline ImU32 paletteBase() { return ImGui::GetColorU32(ImGuiCol_ChildBg); }
inline ImU32 paletteBrightText() {
  return settings.theme == DARK_THEME ? toImColor(DarkTheme::bright_text) : IM_COL32(255, 255, 255, 255);
}
inline ImU32 paletteText(bool active) { return ImGui::GetColorU32(active ? ImGuiCol_Text : ImGuiCol_TextDisabled); }
const ImU32 DARK_GRAY = IM_COL32(128, 128, 128, 255);  // Qt::darkGray

// utils::drawStaticText: text centered in r
void drawStaticText(ImDrawList *p, const ImRect &r, ImFont *font, float font_size, ImU32 col, const std::string &text) {
  const ImVec2 size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, text.c_str());
  p->AddText(font, font_size, ImVec2(r.Min.x + (r.GetWidth() - size.x) / 2, r.Min.y + (r.GetHeight() - size.y) / 2), col, text.c_str());
}

// QBrush(color, Qt::Dense7Pattern): sparse dots
void fillDense7Pattern(ImDrawList *p, const ImRect &r, ImU32 col) {
  p->PushClipRect(r.Min, r.Max, true);
  for (float y = r.Min.y; y < r.Max.y; y += 4.0f) {
    for (float x = r.Min.x + (static_cast<int>((y - r.Min.y) / 4.0f) % 2) * 2.0f; x < r.Max.x; x += 4.0f) {
      p->AddRectFilled(ImVec2(x, y), ImVec2(x + 1.0f, y + 1.0f), col);
    }
  }
  p->PopClipRect();
}

// QBrush(color, Qt::BDiagPattern): backward diagonal lines
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
  model = std::make_unique<BinaryViewModel>();
  delegate = std::make_unique<BinaryItemDelegate>(this);

  connections_.push_back(dbc()->fileChanged.connect([this]() { refresh(); }));
  connections_.push_back(UndoStack::instance()->indexChanged.connect([this]() { refresh(); }));
}

std::string BinaryView::whatsThis() const {
  return R"(
    <b>Binary View</b><br/>
    <!-- TODO: add description here -->
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
  if (ImGui::GetTopMostPopupModal() != nullptr) return;  // Qt::WindowShortcut is blocked by a modal dialog

  // Delete (x, backspace, delete)
  if (ImGui::IsKeyPressed(ImGuiKey_X, false) || ImGui::IsKeyPressed(ImGuiKey_Backspace, false) || ImGui::IsKeyPressed(ImGuiKey_Delete, false)) {
    if (hovered_sig != nullptr) {
      UndoStack::instance()->push(new RemoveSigCommand(model->msg_id, hovered_sig));
      hovered_sig = nullptr;
    }
  }

  // Change endianness (e)
  if (ImGui::IsKeyPressed(ImGuiKey_E, false)) {
    if (hovered_sig != nullptr) {
      cabana::Signal s = *hovered_sig;
      s.is_little_endian = !s.is_little_endian;
      editSignal(hovered_sig, s);
    }
  }

  // Change signedness (s)
  if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
    if (hovered_sig != nullptr) {
      cabana::Signal s = *hovered_sig;
      s.is_signed = !s.is_signed;
      editSignal(hovered_sig, s);
    }
  }

  // Open chart (c, p, g)
  if (ImGui::IsKeyPressed(ImGuiKey_P, false) || ImGui::IsKeyPressed(ImGuiKey_G, false) || ImGui::IsKeyPressed(ImGuiKey_C, false)) {
    if (hovered_sig != nullptr) {
      showChart(model->msg_id, hovered_sig, true, false);
    }
  }
}

ImVec2 BinaryView::minimumSizeHint() const {
  // horizontalHeader()->minimumSectionSize(): widest fixed-font glyph plus the header margins
  pushMonoFont();
  const float min_section_size = ImGui::CalcTextSize("W").x + 8.0f;
  popMonoFont();
  return {(min_section_size + 1) * 9 + VERTICAL_HEADER_WIDTH + 2,
          static_cast<float>(CELL_HEIGHT * std::min(model->rowCount(), 10) + 2)};
}

void BinaryView::highlight(const cabana::Signal *sig) {
  if (sig != hovered_sig) {
    // dataChanged for the affected cells is dropped, the view is redrawn every frame
    hovered_sig = sig;
    signalHovered(hovered_sig);
  }
}

void BinaryView::setSelection() {
  auto index = indexAt(last_mouse_pos);
  if (!anchor_index.isValid() || !index.isValid())
    return;

  std::set<BinaryIndex> selection;  // QItemSelectionModel::ClearAndSelect
  auto [start, size, is_lb] = getSelection(index);
  for (int i = 0; i < size; ++i) {
    int pos = is_lb ? flipBitPos(start + i) : flipBitPos(start) + i;
    selection.insert(model->index(pos / 8, pos % 8));
  }
  selection_ = std::move(selection);
}

void BinaryView::mousePressEvent(const ImVec2 &pos) {
  resize_sig = nullptr;
  if (auto index = indexAt(last_mouse_pos = pos); index.isValid() && index.column != 8) {
    anchor_index = index;
    auto item = &model->items[anchor_index.row * model->columnCount() + anchor_index.column];
    int bit_pos = get_bit_pos(anchor_index);
    for (auto s : item->sigs) {
      if (bit_pos == s->lsb || bit_pos == s->msb) {
        int idx = flipBitPos(bit_pos == s->lsb ? s->msb : s->lsb);
        anchor_index = model->index(idx / 8, idx % 8);
        resize_sig = s;
        break;
      }
    }
  }
}

void BinaryView::highlightPosition(const ImVec2 &pos) {
  if (auto index = indexAt(pos); index.isValid()) {
    auto item = &model->items[index.row * model->columnCount() + index.column];
    const cabana::Signal *sig = item->sigs.empty() ? nullptr : item->sigs.back();
    highlight(sig);
  }
}

void BinaryView::mouseMoveEvent(const ImVec2 &pos) {
  highlightPosition(last_mouse_pos = pos);
  // QAbstractItemView::mouseMoveEvent: drag selecting while the left button is down; flags() makes the hex column unselectable
  if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && model->isSelectable(indexAt(pos))) setSelection();
}

void BinaryView::mouseReleaseEvent(const ImVec2 &pos) {
  auto release_index = indexAt(pos);
  if (release_index.isValid() && anchor_index.isValid()) {
    if (hasSelection()) {
      auto sig = resize_sig ? *resize_sig : cabana::Signal{};
      std::tie(sig.start_bit, sig.size, sig.is_little_endian) = getSelection(release_index);
      resize_sig ? editSignal(resize_sig, sig)
                 : UndoStack::instance()->push(new AddSigCommand(model->msg_id, sig));
    } else {
      auto item = &model->items[anchor_index.row * model->columnCount() + anchor_index.column];
      if (item && item->sigs.size() > 0)
        signalClicked(item->sigs.back());
    }
  }
  clearSelection();
  anchor_index = BinaryIndex();
  resize_sig = nullptr;
}

void BinaryView::leaveEvent() {
  highlight(nullptr);
}

void BinaryView::setMessage(const MessageId &message_id) {
  model->msg_id = message_id;
  scroll_to_top_ = true;
  refresh();
}

void BinaryView::refresh() {
  clearSelection();
  anchor_index = BinaryIndex();
  resize_sig = nullptr;
  hovered_sig = nullptr;
  model->refresh();
  if (under_mouse_) highlightPosition(last_mouse_pos);
}

std::set<const cabana::Signal *> BinaryView::getOverlappingSignals() const {
  std::set<const cabana::Signal *> overlapping;
  for (const auto &item : model->items) {
    if (item.sigs.size() > 1) {
      for (auto s : item.sigs) {
        if (s->type == cabana::Signal::Type::Normal) overlapping.insert(s);
      }
    }
  }
  return overlapping;
}

std::tuple<int, int, bool> BinaryView::getSelection(BinaryIndex index) {
  if (index.column == 8) {
    index = model->index(index.row, 7);
  }
  bool is_lb = true;
  if (resize_sig) {
    is_lb = resize_sig->is_little_endian;
  } else if (settings.drag_direction == Settings::DragDirection::MsbFirst) {
    is_lb = index < anchor_index;
  } else if (settings.drag_direction == Settings::DragDirection::LsbFirst) {
    is_lb = !(index < anchor_index);
  } else if (settings.drag_direction == Settings::DragDirection::AlwaysLE) {
    is_lb = true;
  } else if (settings.drag_direction == Settings::DragDirection::AlwaysBE) {
    is_lb = false;
  }

  int cur_bit_pos = get_bit_pos(index);
  int anchor_bit_pos = get_bit_pos(anchor_index);
  int start_bit = is_lb ? std::min(cur_bit_pos, anchor_bit_pos) : get_bit_pos(std::min(index, anchor_index));
  int size = is_lb ? std::abs(cur_bit_pos - anchor_bit_pos) + 1 : std::abs(flipBitPos(cur_bit_pos) - flipBitPos(anchor_bit_pos)) + 1;
  return {start_bit, size, is_lb};
}

BinaryIndex BinaryView::indexAt(const ImVec2 &pos) const {
  if (column_width_ <= 0 || pos.x < grid_pos_.x + VERTICAL_HEADER_WIDTH || pos.y < grid_pos_.y) return {};
  int column = static_cast<int>((pos.x - grid_pos_.x - VERTICAL_HEADER_WIDTH) / column_width_);
  int row = static_cast<int>((pos.y - grid_pos_.y) / CELL_HEIGHT);
  if (column >= model->columnCount() || row >= model->rowCount()) return {};
  return {row, column};
}

ImRect BinaryView::visualRect(const BinaryIndex &index) const {
  const float x = grid_pos_.x + VERTICAL_HEADER_WIDTH + index.column * column_width_;
  const float y = grid_pos_.y + index.row * CELL_HEIGHT;
  // QTableView keeps a 1px grid line between the cells
  return ImRect(x, y, x + column_width_ - 1, y + CELL_HEIGHT - 1);
}

void BinaryView::draw() {
  // paintEvent
  is_message_active = can->isMessageActive(model->msg_id);
  if (scroll_to_top_) {
    ImGui::SetScrollY(0.0f);
    scroll_to_top_ = false;
  }

  const int rows = model->rowCount();
  const float width = ImGui::GetContentRegionAvail().x;
  column_width_ = std::max(1.0f, (width - VERTICAL_HEADER_WIDTH) / model->columnCount());
  grid_pos_ = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##binary_view", ImVec2(std::max(width, 1.0f), std::max(static_cast<float>(rows * CELL_HEIGHT), 1.0f)));
  ImDrawList *painter = ImGui::GetWindowDrawList();

  // verticalHeader()
  for (int row = 0; row < rows; ++row) {
    const ImRect r(grid_pos_.x, grid_pos_.y + row * CELL_HEIGHT, grid_pos_.x + VERTICAL_HEADER_WIDTH, grid_pos_.y + (row + 1) * CELL_HEIGHT);
    painter->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_WindowBg));  // plain header background
    drawStaticText(painter, r, ImGui::GetFont(), ImGui::GetFontSize(), ImGui::GetColorU32(ImGuiCol_Text), model->headerData(row));
  }
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < model->columnCount(); ++column) {
      const BinaryIndex index = model->index(row, column);
      delegate->paint(painter, visualRect(index), index);
    }
  }

  // mouse events
  const ImVec2 mouse = ImGui::GetMousePos();
  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();
  const bool under_mouse = (hovered || active) && ImGui::IsMouseHoveringRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), false);
  if (hovered || active) {
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) mousePressEvent(mouse);
    const ImVec2 delta = ImGui::GetIO().MouseDelta;
    if (delta.x != 0.0f || delta.y != 0.0f) mouseMoveEvent(mouse);
  }
  // leaveEvent: the mouse left the widget rect, also while dragging
  if (std::exchange(under_mouse_, under_mouse) && !under_mouse) leaveEvent();
  if (ImGui::IsItemDeactivated()) mouseReleaseEvent(mouse);

  // Qt::ToolTipRole
  if (hovered && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
    const std::string tip = model->data(indexAt(mouse));
    if (!tip.empty()) ImGui::SetTooltip("%s", stripHtml(tip).c_str());
  }

  addShortcuts();
}

// BinaryViewModel

void BinaryViewModel::refresh() {
  bit_flip_tracker = {};
  items.clear();
  if (auto dbc_msg = dbc()->msg(msg_id)) {
    row_count = dbc_msg->size;
    items.resize(row_count * column_count);
    for (auto sig : dbc_msg->getSignals()) {
      for (int j = 0; j < sig->size; ++j) {
        int pos = sig->is_little_endian ? flipBitPos(sig->start_bit + j) : flipBitPos(sig->start_bit) + j;
        int idx = column_count * (pos / 8) + pos % 8;
        if (idx >= items.size()) {
          fprintf(stderr, "signal %s out of bounds.start_bit: %d size: %d\n",
                  sig->name.c_str(), sig->start_bit, sig->size);
          break;
        }
        if (j == 0) sig->is_little_endian ? items[idx].is_lsb = true : items[idx].is_msb = true;
        if (j == sig->size - 1) sig->is_little_endian ? items[idx].is_msb = true : items[idx].is_lsb = true;

        auto &sigs = items[idx].sigs;
        sigs.push_back(sig);
        if (sigs.size() > 1) {
          std::sort(sigs.begin(), sigs.end(), [](auto l, auto r) { return l->size > r->size; });
        }
      }
    }
  } else {
    row_count = can->lastMessage(msg_id).dat.size();
    items.resize(row_count * column_count);
  }
  updateState();
}

void BinaryViewModel::updateItem(int row, int col, uint8_t val, const CabanaColor &color) {
  auto &item = items[row * column_count + col];
  item.valid = true;
  if (item.val != val || !(item.bg_color == color)) {
    item.val = val;
    item.bg_color = color;
  }
}

void BinaryViewModel::updateState() {
  const auto &last_msg = can->lastMessage(msg_id);
  const auto &binary = last_msg.dat;
  // Handle size changes in binary data
  if (binary.size() > row_count) {
    row_count = binary.size();
    items.resize(row_count * column_count);
  }

  auto &bit_flips = heatmap_live_mode ? last_msg.bit_flip_counts : getBitFlipChanges(binary.size());
  // Find the maximum bit flip count across the message
  uint32_t max_bit_flip_count = 1;  // Default to 1 to avoid division by zero
  for (const auto &row : bit_flips) {
    for (uint32_t count : row) {
      max_bit_flip_count = std::max(max_bit_flip_count, count);
    }
  }

  const double max_alpha = 255.0;
  const double min_alpha_with_signal = 25.0;  // Base alpha for small flip counts
  const double min_alpha_no_signal = 10.0;    // Base alpha for small flip counts for no signal bits
  const double log_factor = 1.0 + 0.2;        // Factor for logarithmic scaling
  const double log_scaler = max_alpha / log2(log_factor * max_bit_flip_count);

  for (size_t i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      auto &item = items[i * column_count + j];
      int bit_val = (binary[i] >> (7 - j)) & 1;

      double alpha = item.sigs.empty() ? 0 : min_alpha_with_signal;
      uint32_t flip_count = bit_flips[i][j];
      if (flip_count > 0) {
        double normalized_alpha = log2(1.0 + flip_count * log_factor) * log_scaler;
        double min_alpha = item.sigs.empty() ? min_alpha_no_signal : min_alpha_with_signal;
        alpha = std::clamp(normalized_alpha, min_alpha, max_alpha);
      }

      auto color = item.bg_color;
      color.a = static_cast<uint8_t>(alpha);  // QColor::setAlpha(int)
      updateItem(i, j, bit_val, color);
    }
    updateItem(i, 8, binary[i], last_msg.colors[i]);
  }
}

const std::vector<std::array<uint32_t, 8>> &BinaryViewModel::getBitFlipChanges(size_t msg_size) {
  // Return cached results if time range and data are unchanged
  auto time_range = can->timeRange();
  if (bit_flip_tracker.time_range == time_range && !bit_flip_tracker.flip_counts.empty())
    return bit_flip_tracker.flip_counts;

  bit_flip_tracker.time_range = time_range;
  bit_flip_tracker.flip_counts.assign(msg_size, std::array<uint32_t, 8>{});

  // Iterate over events within the specified time range and calculate bit flips
  auto [first, last] = can->eventsInRange(msg_id, time_range);
  if (std::distance(first, last) <= 1) return bit_flip_tracker.flip_counts;

  std::vector<uint8_t> prev_values((*first)->dat, (*first)->dat + (*first)->size);
  for (auto it = std::next(first); it != last; ++it) {
    const CanEvent *event = *it;
    int size = std::min<int>(msg_size, event->size);
    for (int i = 0; i < size; ++i) {
      const uint8_t diff = event->dat[i] ^ prev_values[i];
      if (!diff) continue;

      auto &bit_flips = bit_flip_tracker.flip_counts[i];
      for (int bit = 0; bit < 8; ++bit) {
        if (diff & (1u << bit)) ++bit_flips[7 - bit];
      }
      prev_values[i] = event->dat[i];
    }
  }

  return bit_flip_tracker.flip_counts;
}

std::string BinaryViewModel::headerData(int section) const {
  return std::to_string(section);
}

std::string BinaryViewModel::data(const BinaryIndex &index) const {
  auto item = index.isValid() ? &items[index.row * column_count + index.column] : nullptr;
  return item && !item->sigs.empty() ? utils::signalToolTip(item->sigs.back()) : std::string();
}

// BinaryItemDelegate

BinaryItemDelegate::BinaryItemDelegate(BinaryView *parent) : bin_view(parent) {
  bin_text_table[0] = "0";
  bin_text_table[1] = "1";
  for (int i = 0; i < 256; ++i) {
    char buf[8];
    snprintf(buf, sizeof(buf), "%02X", i);
    hex_text_table[i] = buf;
  }
}

bool BinaryItemDelegate::hasSignal(const BinaryIndex &index, int dx, int dy, const cabana::Signal *sig) const {
  if (!index.isValid()) return false;
  auto model = bin_view->model.get();
  int idx = (index.row + dy) * model->columnCount() + index.column + dx;
  if (idx < 0 || idx >= (int)model->items.size()) return false;
  auto &s = model->items[idx].sigs;
  return std::find(s.begin(), s.end(), sig) != s.end();
}

void BinaryItemDelegate::paint(ImDrawList *painter, const ImRect &rect, const BinaryIndex &index) const {
  auto item = &bin_view->model->items[index.row * bin_view->model->columnCount() + index.column];
  ImFont *font = ImGui::GetFont();
  float font_size = ImGui::GetFontSize();
  ImU32 pen = IM_COL32(0, 0, 0, 255);  // QPainter default pen, the Qt delegate never sets it for the hex column

  if (index.column == 8) {
    if (item->valid) {
      pushMonoFont();  // hex_font (bold variant is not available)
      font = ImGui::GetFont();
      font_size = ImGui::GetFontSize();
      popMonoFont();
      painter->AddRectFilled(rect.Min, rect.Max, toImColor(item->bg_color));
    }
  } else if (bin_view->isSelected(index)) {
    auto color = bin_view->resize_sig ? toImColor(bin_view->resize_sig->color) : paletteHighlight();
    painter->AddRectFilled(rect.Min, rect.Max, color);
    pen = paletteBrightText();
  } else if (!bin_view->hasSelection() || std::find(item->sigs.begin(), item->sigs.end(), bin_view->resize_sig) == item->sigs.end()) {  // not resizing
    if (item->sigs.size() > 0) {
      for (auto &s : item->sigs) {
        if (s == bin_view->hovered_sig) {
          painter->AddRectFilled(rect.Min, rect.Max, toImColor(s->color.darker(125)));  // 4/5x brightness
        } else {
          drawSignalCell(painter, rect, index, s);
        }
      }
    } else if (item->valid && item->bg_color.alpha() > 0) {
      painter->AddRectFilled(rect.Min, rect.Max, toImColor(item->bg_color));
    }
    bool bright = std::find(item->sigs.begin(), item->sigs.end(), bin_view->hovered_sig) != item->sigs.end();
    pen = bright ? paletteBrightText() : paletteText(bin_view->is_message_active);
  }

  if (item->sigs.size() > 1) {
    fillDense7Pattern(painter, rect, DARK_GRAY);
  } else if (!item->valid) {
    fillBDiagPattern(painter, rect, DARK_GRAY);
  }
  if (item->valid) {
    drawStaticText(painter, rect, font, font_size, pen, index.column == 8 ? hex_text_table[item->val] : bin_text_table[item->val]);
  }
  if (item->is_msb || item->is_lsb) {
    const char *text = item->is_msb ? "M" : "L";
    const ImVec2 size = ImGui::GetFont()->CalcTextSizeA(small_font_size, FLT_MAX, 0.0f, text);
    painter->AddText(ImGui::GetFont(), small_font_size, ImVec2(rect.Max.x - 8 - size.x, rect.Max.y - 3 - size.y), pen, text);
  }
}

// Draw border on edge of signal
void BinaryItemDelegate::drawSignalCell(ImDrawList *painter, const ImRect &rect,
                                        const BinaryIndex &index, const cabana::Signal *sig) const {
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
  // QRegion(rc).subtracted(subtract): rc split into horizontal bands with the notch corners removed
  std::vector<ImRect> region;
  {
    std::vector<float> ys{rc.Min.y, rc.Max.y};
    for (auto &r : subtract) {
      ys.push_back(r.Min.y);
      ys.push_back(r.Max.y);
    }
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    for (size_t i = 0; i + 1 < ys.size(); ++i) {
      const float y0 = ys[i], y1 = ys[i + 1];
      float x0 = rc.Min.x, x1 = rc.Max.x;
      for (auto &r : subtract) {
        if (r.Min.y <= y0 && r.Max.y >= y1) {
          if (r.Min.x <= x0) x0 = std::max(x0, r.Max.x);
          else x1 = std::min(x1, r.Min.x);
        }
      }
      if (x1 > x0) region.emplace_back(x0, y0, x1, y1);
    }
  }

  auto item = &bin_view->model->items[index.row * bin_view->model->columnCount() + index.column];
  CabanaColor color = sig->color;
  color.a = item->bg_color.alpha();
  const ImU32 edge = toImColor(sig->color.darker(125));

  for (const ImRect &clip : region) {
    painter->PushClipRect(clip.Min, clip.Max, true);
    // Mixing the signal color with the Base background color to fade it
    painter->AddRectFilled(rc.Min, rc.Max, paletteBase());
    painter->AddRectFilled(rc.Min, rc.Max, toImColor(color));

    // Draw edges
    if (draw_left) painter->AddLine(ImVec2(rc.Min.x + 0.5f, rc.Min.y), ImVec2(rc.Min.x + 0.5f, rc.Max.y), edge, 1.0f);
    if (draw_right) painter->AddLine(ImVec2(rc.Max.x - 0.5f, rc.Min.y), ImVec2(rc.Max.x - 0.5f, rc.Max.y), edge, 1.0f);
    if (draw_bottom) painter->AddLine(ImVec2(rc.Min.x, rc.Max.y - 0.5f), ImVec2(rc.Max.x, rc.Max.y - 0.5f), edge, 1.0f);
    if (draw_top) painter->AddLine(ImVec2(rc.Min.x, rc.Min.y + 0.5f), ImVec2(rc.Max.x, rc.Min.y + 0.5f), edge, 1.0f);

    // fill gaps inside corners: the 2px pen is clipped to the region, only the half outside the notch is painted
    for (auto &r : subtract) {
      painter->AddRect(r.Min, r.Max, edge, 0.0f, 0, 2.0f);
    }
    painter->PopClipRect();
  }
}
