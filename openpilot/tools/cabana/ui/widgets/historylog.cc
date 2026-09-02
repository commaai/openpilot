#include "tools/cabana/ui/widgets/historylog.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <iterator>

#include "imgui_internal.h"
#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/export.h"
#include "tools/cabana/utils/strings.h"

HistoryLogModel::HistoryLogModel() {
  connections_.push_back(can->seekedTo.connect([this](double) { reset(); }));
  connections_.push_back(dbc()->fileChanged.connect([this]() { reset(); }));
  connections_.push_back(UndoStack::instance()->indexChanged.connect([this]() { reset(); }));
}

std::string HistoryLogModel::data(int row, int col) const {
  const auto &m = messages[row];
  if (col == 0) {
    char buf[32] = {};
    snprintf(buf, sizeof(buf), "%.3f", can->toSeconds(m.mono_time));
    return buf;
  }
  if (!isHexMode()) return sigs[col - 1]->formatValue(m.sig_values[col - 1], false);
  return {};
}

void HistoryLogModel::setMessage(const MessageId &message_id) {
  msg_id = message_id;
  reset();
}

void HistoryLogModel::reset() {
  sigs.clear();
  if (auto dbc_msg = dbc()->msg(msg_id)) {
    sigs = dbc_msg->getSignals();
  }
  messages.clear();
  hex_colors = {};
  modelReset();
  setFilter(0, "", nullptr);
}

std::string HistoryLogModel::headerData(int section) const {
  if (section == 0) return "Time";
  if (isHexMode()) return "Data";

  const std::string &name = sigs[section - 1]->name;
  const std::string &unit = sigs[section - 1]->unit;
  return unit.empty() ? name : name + " (" + unit + ")";
}

std::optional<CabanaColor> HistoryLogModel::headerBackground(int section) const {
  if (section > 0 && !isHexMode()) {
    CabanaColor sigColor = sigs[section - 1]->color;
    sigColor.a = 128;
    return sigColor;
  }
  return std::nullopt;
}

void HistoryLogModel::setHexMode(bool hex) {
  hex_mode = hex;
  reset();
}

void HistoryLogModel::setFilter(int sig_idx, const std::string &value, std::function<bool(double, double)> cmp) {
  filter_sig_idx = sig_idx;
  filter_value = utils::toDouble(value);
  filter_cmp = value.empty() ? nullptr : cmp;
  updateState(true);
}

void HistoryLogModel::updateState(bool clear) {
  if (clear && !messages.empty()) {
    messages.clear();
    rowsRemoved();
  }
  uint64_t current_time = can->toMonoTime(can->lastMessage(msg_id).ts) + 1;
  fetchData(messages.begin(), current_time, messages.empty() ? 0 : messages.front().mono_time);
}

bool HistoryLogModel::canFetchMore() const {
  const auto &events = can->events(msg_id);
  return !events.empty() && !messages.empty() && messages.back().mono_time > events.front()->mono_time;
}

void HistoryLogModel::fetchMore() {
  if (!messages.empty())
    fetchData(messages.end(), messages.back().mono_time, 0);
}

void HistoryLogModel::fetchData(std::deque<Message>::iterator insert_pos, uint64_t from_time, uint64_t min_time) {
  const auto &events = can->events(msg_id);
  auto first = std::upper_bound(events.rbegin(), events.rend(), from_time, [](uint64_t ts, auto e) {
    return ts > e->mono_time;
  });

  std::vector<HistoryLogModel::Message> msgs;
  std::vector<double> values(sigs.size());
  msgs.reserve(batch_size);
  for (; first != events.rend() && (*first)->mono_time > min_time; ++first) {
    const CanEvent *e = *first;
    for (int i = 0; i < sigs.size(); ++i) {
      sigs[i]->getValue(e->dat, e->size, &values[i]);
    }
    if (!filter_cmp || filter_cmp(values[filter_sig_idx], filter_value)) {
      msgs.emplace_back(Message{e->mono_time, values, {e->dat, e->dat + e->size}});
      if (msgs.size() >= batch_size && min_time == 0) {
        break;
      }
    }
  }

  if (!msgs.empty()) {
    if (isHexMode() && (min_time > 0 || messages.empty())) {
      const auto freq = can->lastMessage(msg_id).freq;
      const std::vector<uint8_t> no_mask;
      for (auto &m : msgs) {
        hex_colors.compute(msg_id, m.data.data(), m.data.size(), m.mono_time / (double)1e9, can->getSpeed(), no_mask, freq);
        m.colors = hex_colors.colors;
      }
    }
    const int pos = std::distance(messages.begin(), insert_pos);
    messages.insert(insert_pos, std::move_iterator(msgs.begin()), std::move_iterator(msgs.end()));
    rowsInserted(pos, msgs.size());
  }
}

namespace {

constexpr float DISPLAY_TYPE_WIDTH = 90.0f;
constexpr float SIGNALS_WIDTH = 160.0f;
constexpr float COMPARE_WIDTH = 50.0f;

std::string headerText(std::string text) {
  std::replace(text.begin(), text.end(), '_', ' ');
  return text;
}

// `width`: the header viewport width
ImVec2 sectionSizeFromContents(const HistoryLogModel &model, int logicalIndex, float width) {
  const ImVec2 time_text_size = ImGui::CalcTextSize("000000.000");
  const ImVec2 time_col_size(time_text_size.x + 10, time_text_size.y + 6);
  if (logicalIndex == 0) {
    return time_col_size;
  } else {
    int default_size = std::max(100, (int)((width - time_col_size.x) / (model.columnCount() - 1)));
    std::string text = headerText(model.headerData(logicalIndex));
    const ImVec2 rect = ImGui::CalcTextSize(text.c_str(), nullptr, false, default_size);
    return ImVec2{std::max(rect.x + 10, (float)default_size), rect.y + 6};
  }
}

void paintSection(const HistoryLogModel &model, ImDrawList *painter, const ImRect &rect, int logicalIndex) {
  if (auto bg = model.headerBackground(logicalIndex)) {
    painter->AddRectFilled(rect.Min, rect.Max, toImU32(*bg));
  }
  std::string text = headerText(model.headerData(logicalIndex));
  const ImU32 color = isDarkTheme() ? toImU32(DarkTheme::bright_text) : ImGui::GetColorU32(ImGuiCol_Text);
  // right aligned and word wrapped, one line at a time
  const ImRect r(rect.Min.x + 5, rect.Min.y + 3, rect.Max.x - 5, rect.Max.y - 3);
  ImFont *font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const float wrap_width = std::max(1.0f, r.GetWidth());
  const char *s = text.c_str();
  const char *end = s + text.size();
  float y = r.Min.y;
  painter->PushClipRect(rect.Min, rect.Max, true);
  while (s < end) {
    const char *line_end = font->CalcWordWrapPosition(font_size, s, end, wrap_width);
    if (line_end == s) line_end = s + 1;
    const float w = ImGui::CalcTextSize(s, line_end).x;
    painter->AddText(font, font_size, ImVec2(r.Max.x - w, y), color, s, line_end);
    y += ImGui::GetTextLineHeight();
    s = line_end;
    while (s < end && ImCharIsBlankA(*s)) s++;
    if (s < end && *s == '\n') s++;
  }
  painter->PopClipRect();
}

}  // namespace

LogsWidget::LogsWidget() {
  connections_.push_back(model_.modelReset.connect([this]() { modelReset(); }));
  connections_.push_back(model_.rowsRemoved.connect([this]() { selected_row_ = selected_col_ = -1; }));
  connections_.push_back(model_.rowsInserted.connect([this](int pos, int count) {
    export_btn_enabled_ = true;
    // the selection follows the message it was made on, like QItemSelectionModel does
    if (selected_row_ >= pos) selected_row_ += count;
  }));
}

void LogsWidget::modelReset() {
  signals_cb_ = 0;
  export_btn_enabled_ = false;
  value_edit_.clear();
  value_edit_modified_ = false;
  comp_box_ = 0;
  selected_row_ = selected_col_ = -1;  // a model_ reset clears the selection
  filters_widget_visible_ = !model_.sigs.empty();
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
  model_.setFilter(signals_cb_, value_edit_, cmp);
}

void LogsWidget::exportToCSV() {
  std::string dir = settings.last_dir + "/" + can->routeName() + "_" + msgName(model_.msg_id) + ".csv";
  FileDialog::getSaveFileName("Export " + msgName(model_.msg_id) + " to CSV file", dir, ".csv", [this](const std::string &fn) {
    if (!fn.empty()) {
      model_.isHexMode() ? utils::exportToCSV(fn, model_.msg_id)
                        : utils::exportSignalsToCSV(fn, model_.msg_id);
    }
  });
}

void LogsWidget::draw() {
  delegate_.updateFontMetrics();
  const ImGuiStyle &style = ImGui::GetStyle();

  // toolbar: the export button is right aligned and never clipped, the value input shrinks first
  const float export_w = ImGui::CalcTextSize(icon::FILETYPE_CSV).x + style.FramePadding.x * 2;
  if (filters_widget_visible_) {
    const float clear_w = value_edit_.empty() ? 0.0f : ImGui::CalcTextSize(icon::X).x + style.FramePadding.x * 2;
    const float fixed = DISPLAY_TYPE_WIDTH + SIGNALS_WIDTH + COMPARE_WIDTH + clear_w + style.ItemSpacing.x * 4 + export_w;
    const float value_w = std::clamp(ImGui::GetContentRegionAvail().x - fixed, 30.0f, 120.0f);

    ImGui::SetNextItemWidth(DISPLAY_TYPE_WIDTH);
    if (ImGui::Combo("##display_type", &display_type_cb_, "Signal\0Hex\0")) model_.setHexMode(display_type_cb_);
    ImGui::SetItemTooltip("Display signal value or raw hex value");
    ImGui::SameLine();
    std::string sig_items;
    for (auto s : model_.sigs) {
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
  alignRight(export_w);
  ImGui::BeginDisabled(!export_btn_enabled_);
  if (ImGui::Button(icon::FILETYPE_CSV)) exportToCSV();
  ImGui::EndDisabled();
  disabledItemTooltip("Export to CSV file...");

  ImGui::Separator();
  drawTable();
}

void LogsWidget::drawTable() {
  const ImGuiStyle &style = ImGui::GetStyle();
  const int cols = model_.columnCount();
  // the header viewport excludes the table's cell padding and the vertical scrollbar (the latter is one
  // frame behind, it is only known once shown)
  const float header_width = ImGui::GetContentRegionAvail().x - style.CellPadding.x * 2 * cols -
                             (vscrollbar_visible_ ? style.ScrollbarSize : 0.0f);

  std::vector<ImVec2> sizes(cols);
  float header_height = 0;
  for (int i = 0; i < cols; ++i) {
    sizes[i] = sectionSizeFromContents(model_, i, header_width);
    header_height = std::max(header_height, sizes[i].y);
  }
  if (model_.isHexMode() && !model_.messages.empty()) {
    sizes[1].x = std::max(sizes[1].x, delegate_.sizeHint(&model_.messages.front().data).x);
  }
  const float row_height = delegate_.sizeForBytes(8).y;

  // fixed section sizes and a horizontal scrollbar, no alternating row colors; the grid is drawn between
  // rows and columns
  ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_BordersInner |
                          ImGuiTableFlags_SizingFixedFit;
  // an empty viewport draws no grid
  if (model_.rowCount() == 0) flags &= ~ImGuiTableFlags_BordersInnerV;
  // the sum of the fixed section sizes, so the columns keep their size and the table scrolls
  float inner_width = 0;
  for (int i = 0; i < cols; ++i) inner_width += sizes[i].x + style.CellPadding.x * 2;

  bool fetch_more = false;
  if (ImGui::BeginTable("logs", cols, flags, ImVec2(0, 0), inner_width)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    for (int i = 0; i < cols; ++i) {
      ImGui::TableSetupColumn(model_.headerData(i).c_str(), ImGuiTableColumnFlags_WidthFixed, sizes[i].x);
    }
    ImGuiTable *table = ImGui::GetCurrentTable();
    ImDrawList *painter = ImGui::GetWindowDrawList();

    ImGui::TableNextRow(ImGuiTableRowFlags_Headers, header_height);
    for (int i = 0; i < cols; ++i) {
      if (!ImGui::TableSetColumnIndex(i)) continue;
      paintSection(model_, painter, ImGui::TableGetCellBgRect(table, i), i);
      ImGui::Dummy(ImVec2(0, header_height - style.CellPadding.y * 2));
    }

    ImGuiListClipper clipper;
    clipper.Begin(model_.rowCount(), row_height);
    while (clipper.Step()) {
      for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
        const auto &m = model_.messages[row];
        ImGui::TableNextRow(0, row_height);
        // rows are prepended while the stream plays, so the id is the message, not the row index
        ImGui::PushID((void *)(uintptr_t)m.mono_time);
        for (int col = 0; col < cols; ++col) {
          if (!ImGui::TableSetColumnIndex(col)) continue;
          // cells are selected, not rows; there is no hover highlight, only the selection background
          const bool cell_selected = selected_row_ == row && selected_col_ == col;
          ImGui::PushID(col);
          if (viewSelectable("##cell", cell_selected, ImGuiSelectableFlags_AllowOverlap,
                             ImVec2(0, row_height - style.CellPadding.y * 2))) {
            selected_row_ = row;
            selected_col_ = col;
          }
          ImGui::PopID();
          const bool hex_cell = model_.isHexMode() && col == 1;
          delegate_.paint(painter, ImGui::TableGetCellBgRect(table, col), cell_selected, false,
                         hex_cell ? std::string() : model_.data(row, col), hex_cell ? &m.data : nullptr, hex_cell ? &m.colors : nullptr);
        }
        ImGui::PopID();
      }
      // fetch more when the last row is visible or the scrollbar is at its maximum
      if (clipper.DisplayEnd >= model_.rowCount()) fetch_more = true;
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) fetch_more = true;
    vscrollbar_visible_ = table->InnerWindow->ScrollbarY;
    ImGui::EndTable();
  }
  if (fetch_more && model_.canFetchMore()) {
    model_.fetchMore();
  }
}
