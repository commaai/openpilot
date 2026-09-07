#include "tools/cabana/ui/widgets/historylog.h"

#include <algorithm>
#include <cstdio>
#include <iterator>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/ui/widgets/messagebytes.h"
#include "tools/cabana/utils/export.h"
#include "tools/cabana/utils/strings.h"

namespace {

constexpr int BATCH_SIZE = 50;
constexpr float DISPLAY_TYPE_WIDTH = 90.0f;
constexpr float SIGNALS_WIDTH = 160.0f;
constexpr float COMPARE_WIDTH = 50.0f;

std::string formatTime(uint64_t mono_time) {
  char buf[32] = {};
  snprintf(buf, sizeof(buf), "%.2f", can->toSeconds(mono_time));
  return buf;
}

}  // namespace

LogsWidget::LogsWidget() {
  connections_.push_back(can->seekedTo.connect([this](double) { reset(); }));
  connections_.push_back(dbc()->fileChanged.connect([this]() { reset(); }));
  connections_.push_back(UndoStack::instance()->indexChanged.connect([this]() { reset(); }));
}

void LogsWidget::setMessage(const MessageId &message_id) {
  msg_id_ = message_id;
  reset();
}

void LogsWidget::reset() {
  sigs_.clear();
  if (auto dbc_msg = dbc()->msg(msg_id_)) sigs_ = dbc_msg->getSignals();
  messages_.clear();
  hex_colors_ = {};
  signals_cb_ = comp_box_ = 0;
  value_edit_.clear();
  value_edit_modified_ = false;
  export_btn_enabled_ = false;
  selected_row_ = selected_col_ = -1;
  setFilter(0, "", nullptr);
}

void LogsWidget::setFilter(int sig_idx, const std::string &value, std::function<bool(double, double)> cmp) {
  filter_sig_idx_ = sig_idx;
  filter_value_ = utils::toDouble(value);
  filter_cmp_ = value.empty() ? nullptr : cmp;
  load(true);
}

void LogsWidget::load(bool clear) {
  if (clear && !messages_.empty()) {
    messages_.clear();
    selected_row_ = selected_col_ = -1;
  }
  const uint64_t current_time = can->toMonoTime(can->lastMessage(msg_id_).ts) + 1;
  fetch(messages_.begin(), current_time, messages_.empty() ? 0 : messages_.front().mono_time);
}

bool LogsWidget::canFetchMore() const {
  const auto &events = can->events(msg_id_);
  return !events.empty() && !messages_.empty() && messages_.back().mono_time > events.front()->mono_time;
}

void LogsWidget::fetch(std::deque<Message>::iterator insert_pos, uint64_t from_time, uint64_t min_time) {
  const auto &events = can->events(msg_id_);
  auto first = std::upper_bound(events.rbegin(), events.rend(), from_time, [](uint64_t ts, auto e) { return ts > e->mono_time; });

  std::vector<Message> msgs;
  std::vector<double> values(sigs_.size());
  msgs.reserve(BATCH_SIZE);
  for (; first != events.rend() && (*first)->mono_time > min_time; ++first) {
    const CanEvent *e = *first;
    for (int i = 0; i < sigs_.size(); ++i) {
      sigs_[i]->getValue(e->dat, e->size, &values[i]);
    }
    if (!filter_cmp_ || filter_cmp_(values[filter_sig_idx_], filter_value_)) {
      msgs.emplace_back(Message{e->mono_time, values, {e->dat, e->dat + e->size}});
      if (msgs.size() >= BATCH_SIZE && min_time == 0) break;
    }
  }
  if (msgs.empty()) return;

  if (hexMode() && (min_time > 0 || messages_.empty())) {
    const auto freq = can->lastMessage(msg_id_).freq;
    const std::vector<uint8_t> no_mask;
    for (auto &m : msgs) {
      hex_colors_.compute(msg_id_, m.data.data(), m.data.size(), m.mono_time / (double)1e9, can->getSpeed(), no_mask, freq);
      m.colors = hex_colors_.colors;
    }
  }
  const int pos = std::distance(messages_.begin(), insert_pos);
  messages_.insert(insert_pos, std::move_iterator(msgs.begin()), std::move_iterator(msgs.end()));
  export_btn_enabled_ = true;
  // the selection follows the message it was made on
  if (selected_row_ >= pos) selected_row_ += msgs.size();
}

void LogsWidget::filterChanged() {
  if (value_edit_.empty() && !value_edit_modified_) return;

  std::function<bool(double, double)> cmp = nullptr;
  switch (comp_box_) {
    case 0: cmp = std::greater<double>{}; break;
    case 1: cmp = std::equal_to<double>{}; break;
    case 2: cmp = [](double l, double r) { return l != r; }; break;
    case 3: cmp = std::less<double>{}; break;
  }
  setFilter(signals_cb_, value_edit_, cmp);
}

void LogsWidget::exportToCSV() {
  std::string dir = settings.last_dir + "/" + can->routeName() + "_" + msgName(msg_id_) + ".csv";
  FileDialog::getSaveFileName("Export " + msgName(msg_id_) + " to CSV file", dir, ".csv", [this](const std::string &fn) {
    if (!fn.empty()) {
      hexMode() ? utils::exportToCSV(fn, msg_id_) : utils::exportSignalsToCSV(fn, msg_id_);
    }
  });
}

void LogsWidget::draw() {
  const ImGuiStyle &style = ImGui::GetStyle();

  beginControlChild("toolbar", ImVec2(0, ImGui::GetFrameHeight() + CONTROL_OUTLINE_PADDING * 2),
                    ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

  // toolbar: the export button is right aligned and never clipped, the value input shrinks first
  const float export_w = iconButtonWidth();
  if (!sigs_.empty()) {
    const float fixed = DISPLAY_TYPE_WIDTH + SIGNALS_WIDTH + COMPARE_WIDTH + style.ItemSpacing.x * 4 + export_w;
    const float value_w = std::clamp(ImGui::GetContentRegionAvail().x - fixed, 30.0f, 120.0f);

    ImGui::SetNextItemWidth(DISPLAY_TYPE_WIDTH);
    if (ImGui::Combo("##display_type", &display_type_cb_, "Signal\0Hex\0")) {
      hex_mode_ = display_type_cb_;
      reset();
    }
    ImGui::SetItemTooltip("Display signal value or raw hex value");
    ImGui::SameLine();
    std::string sig_items;
    for (auto s : sigs_) {
      sig_items += s->name;
      sig_items += '\0';
    }
    sig_items += '\0';
    ImGui::SetNextItemWidth(SIGNALS_WIDTH);
    if (ImGui::Combo("##signals", &signals_cb_, sig_items.c_str())) filterChanged();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(COMPARE_WIDTH);
    if (ImGui::Combo("##comp", &comp_box_, ">\0=\0!=\0<\0")) filterChanged();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(value_w);
    if (clearableInput("##value", &value_edit_, "", doubleValidator)) {
      value_edit_modified_ = true;  // clearing the field still counts as modified
      filterChanged();
    }
  }
  alignRight(iconButtonWidth());
  ImGui::BeginDisabled(!export_btn_enabled_);
  if (iconButton("export_csv", icon::FILETYPE_CSV)) exportToCSV();
  ImGui::EndDisabled();
  disabledItemTooltip("Export to CSV file...");
  ImGui::EndChild();

  ImGui::Separator();
  drawTable();
}

std::string LogsWidget::headerText(int column) const {
  if (column == 0) return "Time";
  if (hexMode()) return "Data";
  std::string text = sigs_[column - 1]->name;
  if (!sigs_[column - 1]->unit.empty()) text += " (" + sigs_[column - 1]->unit + ")";
  std::replace(text.begin(), text.end(), '_', ' ');
  return text;
}

ImVec2 LogsWidget::headerSize(int column, float viewport_width) const {
  const ImVec2 time_text_size = ImGui::CalcTextSize("000000.00");
  const ImVec2 time_col_size(time_text_size.x + 10, time_text_size.y + 6);
  if (column == 0) return time_col_size;
  const int default_size = std::max(100, (int)((viewport_width - time_col_size.x) / (columnCount() - 1)));
  const ImVec2 rect = ImGui::CalcTextSize(headerText(column).c_str(), nullptr, false, default_size);
  return ImVec2{std::max(rect.x + 10, (float)default_size), rect.y + 6};
}

void LogsWidget::drawHeaderCell(ImDrawList *dl, const ImRect &rect, int column) const {
  if (column > 0 && !hexMode()) {
    CabanaColor bg = sigs_[column - 1]->color;
    bg.a = 128;
    dl->AddRectFilled(rect.Min, rect.Max, toImU32(bg), ImGui::GetStyle().FrameRounding);
  }
  const std::string text = headerText(column);
  const ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
  // right aligned and word wrapped, one line at a time
  const ImRect r(rect.Min.x + 5, rect.Min.y + 3, rect.Max.x - 5, rect.Max.y - 3);
  ImFont *font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const float wrap_width = std::max(1.0f, r.GetWidth());
  const char *s = text.c_str();
  const char *end = s + text.size();
  float y = r.Min.y;
  dl->PushClipRect(rect.Min, rect.Max, true);
  while (s < end) {
    const char *line_end = font->CalcWordWrapPosition(font_size, s, end, wrap_width);
    if (line_end == s) line_end = s + 1;
    const float w = ImGui::CalcTextSize(s, line_end).x;
    dl->AddText(font, font_size, ImVec2(r.Max.x - w, y), color, s, line_end);
    y += ImGui::GetTextLineHeight();
    s = line_end;
    while (s < end && ImCharIsBlankA(*s)) s++;
    if (s < end && *s == '\n') s++;
  }
  dl->PopClipRect();
}

void LogsWidget::drawTable() {
  const ImGuiStyle &style = ImGui::GetStyle();
  const int cols = columnCount();
  // the header viewport excludes the table's cell padding and the vertical scrollbar (the latter is one
  // frame behind, it is only known once shown)
  const float header_width = ImGui::GetContentRegionAvail().x - style.CellPadding.x * 2 * cols -
                             (vscrollbar_visible_ ? style.ScrollbarSize : 0.0f);

  std::vector<ImVec2> sizes(cols);
  float header_height = 0;
  for (int i = 0; i < cols; ++i) {
    sizes[i] = headerSize(i, header_width);
    header_height = std::max(header_height, sizes[i].y);
  }
  if (hexMode() && !messages_.empty()) {
    sizes[1].x = std::max(sizes[1].x, bytesCellSize(messages_.front().data.size(), false).x);
  }
  const float row_height = bytesCellSize(8, false).y;

  // fixed section sizes and a horizontal scrollbar, no alternating row colors; the grid is drawn between
  // rows and columns
  ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_Borders |
                          ImGuiTableFlags_SizingFixedFit;
  // an empty viewport draws no grid
  if (messages_.empty()) flags &= ~ImGuiTableFlags_BordersInnerV;
  // the sum of the fixed section sizes, so the columns keep their size and the table scrolls
  float inner_width = 0;
  for (int i = 0; i < cols; ++i) inner_width += sizes[i].x + style.CellPadding.x * 2;

  bool fetch_more = false;
  if (ImGui::BeginTable("logs", cols, flags, ImVec2(0, 0), inner_width)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    for (int i = 0; i < cols; ++i) {
      ImGui::TableSetupColumn(headerText(i).c_str(), ImGuiTableColumnFlags_WidthFixed, sizes[i].x);
    }
    ImGuiTable *table = ImGui::GetCurrentTable();
    ImDrawList *painter = ImGui::GetWindowDrawList();

    ImGui::TableNextRow(ImGuiTableRowFlags_Headers, header_height);
    for (int i = 0; i < cols; ++i) {
      if (!ImGui::TableSetColumnIndex(i)) continue;
      drawHeaderCell(painter, ImGui::TableGetCellBgRect(table, i), i);
      ImGui::Dummy(ImVec2(0, header_height - style.CellPadding.y * 2));
    }

    ImGuiListClipper clipper;
    clipper.Begin(messages_.size(), row_height);
    while (clipper.Step()) {
      for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
        const auto &m = messages_[row];
        ImGui::TableNextRow(0, row_height);
        // rows are prepended while the stream plays, so the id is the message, not the row index
        ImGui::PushID((void *)(uintptr_t)m.mono_time);
        for (int col = 0; col < cols; ++col) {
          if (!ImGui::TableSetColumnIndex(col)) continue;
          // cells are selected, not rows; there is no hover highlight, only the selection background
          const bool cell_selected = selected_row_ == row && selected_col_ == col;
          const ImVec2 pos = ImGui::GetCursorScreenPos();
          const ImRect rect(pos, ImVec2(pos.x + ImGui::GetContentRegionAvail().x, pos.y + row_height - style.CellPadding.y * 2));
          ImGui::PushID(col);
          if (viewSelectable("##cell", cell_selected, ImGuiSelectableFlags_AllowOverlap, ImVec2(0, rect.GetHeight()))) {
            selected_row_ = row;
            selected_col_ = col;
          }
          ImGui::PopID();
          if (col == 0) {
            drawTextCell(painter, rect, formatTime(m.mono_time), cell_selected, false);
          } else if (hexMode()) {
            drawBytesCell(painter, rect, m.data, &m.colors, cell_selected, false, false);
          } else {
            drawTextCell(painter, rect, sigs_[col - 1]->formatValue(m.sig_values[col - 1], false), cell_selected, false);
          }
        }
        ImGui::PopID();
      }
      // fetch more when the last row is visible or the scrollbar is at its maximum
      if (clipper.DisplayEnd >= (int)messages_.size()) fetch_more = true;
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) fetch_more = true;
    vscrollbar_visible_ = table->InnerWindow->ScrollbarY;
    ImGui::EndTable();
  }
  if (fetch_more && canFetchMore()) fetch(messages_.end(), messages_.back().mono_time, 0);
}
