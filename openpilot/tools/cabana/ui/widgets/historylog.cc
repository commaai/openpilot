#define IMGUI_DEFINE_MATH_OPERATORS
#include "tools/cabana/ui/widgets/historylog.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>

#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/imgui_util.h"
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
    // alpha-blend the signal color with the background to ensure contrast
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
    messages.insert(insert_pos, std::move_iterator(msgs.begin()), std::move_iterator(msgs.end()));
    rowsInserted();
  }
}



static std::string headerText(std::string text) {
  std::replace(text.begin(), text.end(), '_', ' ');
  return text;
}

ImVec2 HeaderView::sectionSizeFromContents(int logicalIndex) const {
  const ImVec2 time_col_size = ImGui::CalcTextSize("000000.000") + ImVec2(10, 6);
  if (logicalIndex == 0) {
    return time_col_size;
  } else {
    int default_size = std::max(100, (int)((width - time_col_size.x) / (model->columnCount() - 1)));
    std::string text = headerText(model->headerData(logicalIndex));
    const ImVec2 rect = ImGui::CalcTextSize(text.c_str(), nullptr, false, default_size);
    ImVec2 size = rect + ImVec2{10, 6};
    return ImVec2{std::max(size.x, (float)default_size), size.y};
  }
}

void HeaderView::paintSection(ImDrawList *painter, const ImRect &rect, int logicalIndex) const {
  if (auto bg = model->headerBackground(logicalIndex)) {
    painter->AddRectFilled(rect.Min, rect.Max, IM_COL32(bg->r, bg->g, bg->b, bg->a));
  }
  std::string text = headerText(model->headerData(logicalIndex));
  const ImU32 color = isDarkTheme()
                          ? IM_COL32(DarkTheme::bright_text.r, DarkTheme::bright_text.g, DarkTheme::bright_text.b, 255)
                          : ImGui::GetColorU32(ImGuiCol_Text);
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


LogsWidget::LogsWidget() : header(&model) {
  connections_.push_back(model.modelReset.connect([this]() { modelReset(); }));
  connections_.push_back(model.rowsInserted.connect([this]() { export_btn_enabled = true; }));
}

void LogsWidget::modelReset() {
  signals_cb = 0;
  export_btn_enabled = false;
  value_edit.clear();
  value_edit_modified = false;
  comp_box = 0;
  selected_row = selected_col = -1;  // a model reset clears the selection
  filters_widget_visible = !model.sigs.empty();
}

void LogsWidget::filterChanged() {
  if (value_edit.empty() && !value_edit_modified) return;

  std::function<bool(double, double)> cmp = nullptr;
  switch (comp_box) {
    case 0: cmp = std::greater<double>{}; break;
    case 1: cmp = std::equal_to<double>{}; break;
    case 2: cmp = [](double l, double r) { return l != r; }; break;
    case 3: cmp = std::less<double>{}; break;
  }
  model.setFilter(signals_cb, value_edit, cmp);
}

void LogsWidget::exportToCSV() {
  std::string dir = settings.last_dir + "/" + can->routeName() + "_" + msgName(model.msg_id) + ".csv";
  FileDialog::getSaveFileName("Export " + msgName(model.msg_id) + " to CSV file", dir, ".csv", [this](const std::string &fn) {
    if (!fn.empty()) {
      model.isHexMode() ? utils::exportToCSV(fn, model.msg_id)
                        : utils::exportSignalsToCSV(fn, model.msg_id);
    }
  });
}

void LogsWidget::draw() {
  delegate.updateFontMetrics();
  const ImGuiStyle &style = ImGui::GetStyle();

  // toolbar: the export button is right aligned and never clipped, the value input shrinks first
  const float export_w = ImGui::CalcTextSize(icon::FILETYPE_CSV).x + style.FramePadding.x * 2;
  if (filters_widget_visible) {
    const float clear_w = value_edit.empty() ? 0.0f : ImGui::CalcTextSize(icon::X).x + style.FramePadding.x * 2;
    const float fixed = 90.0f + 160.0f + 50.0f + clear_w + style.ItemSpacing.x * 4 + export_w;
    const float value_w = std::clamp(ImGui::GetContentRegionAvail().x - fixed, 30.0f, 120.0f);

    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::Combo("##display_type", &display_type_cb, "Signal\0Hex\0")) model.setHexMode(display_type_cb);
    ImGui::SetItemTooltip("Display signal value or raw hex value");
    ImGui::SameLine();
    std::string sig_items;
    for (auto s : model.sigs) {
      sig_items += s->name;
      sig_items += '\0';
    }
    sig_items += '\0';
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::Combo("##signals", &signals_cb, sig_items.c_str())) filterChanged();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(50.0f);
    if (ImGui::Combo("##comp", &comp_box, ">\0=\0!=\0<\0")) filterChanged();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(value_w);
    std::string prev = value_edit;
    if (inputText("##value", &value_edit)) {
      value_edit = applyDoubleValidator(prev, value_edit);
      if (value_edit != prev) {
        value_edit_modified = true;
        filterChanged();
      }
    }
    // the clear button only shows when the field is non-empty
    if (!value_edit.empty()) {
      ImGui::SameLine(0, 0);
      if (ImGui::Button(icon::X)) {
        value_edit.clear();
        value_edit_modified = true;  // clearing the field still counts as modified
        filterChanged();
      }
    }
    ImGui::SameLine();
  }
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - export_w));
  ImGui::BeginDisabled(!export_btn_enabled);
  if (ImGui::Button(icon::FILETYPE_CSV)) exportToCSV();
  ImGui::EndDisabled();
  ImGui::SetItemTooltip("Export to CSV file...");

  ImGui::Separator();
  drawTable();
}

void LogsWidget::drawTable() {
  const ImGuiStyle &style = ImGui::GetStyle();
  const int cols = model.columnCount();
  // the header viewport excludes the table's cell padding and the vertical scrollbar (the latter is one
  // frame behind, it is only known once shown)
  header.width = ImGui::GetContentRegionAvail().x - style.CellPadding.x * 2 * cols -
                 (vscrollbar_visible_ ? style.ScrollbarSize : 0.0f);

  std::vector<ImVec2> sizes(cols);
  float header_height = 0;
  for (int i = 0; i < cols; ++i) {
    sizes[i] = header.sectionSizeFromContents(i);
    header_height = std::max(header_height, sizes[i].y);
  }
  if (model.isHexMode() && !model.messages.empty()) {
    sizes[1].x = std::max(sizes[1].x, delegate.sizeHint(&model.messages.front().data).x);
  }
  const float row_height = delegate.sizeForBytes(8).y;

  // fixed section sizes and a horizontal scrollbar, no alternating row colors; the grid is drawn between
  // rows and columns
  ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_BordersInner |
                          ImGuiTableFlags_SizingFixedFit;
  // an empty viewport draws no grid
  if (model.rowCount() == 0) flags &= ~ImGuiTableFlags_BordersInnerV;
  // the sum of the fixed section sizes, so the columns keep their size and the table scrolls
  float inner_width = 0;
  for (int i = 0; i < cols; ++i) inner_width += sizes[i].x + style.CellPadding.x * 2;

  bool fetch_more = false;
  if (ImGui::BeginTable("logs", cols, flags, ImVec2(0, 0), inner_width)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    for (int i = 0; i < cols; ++i) {
      ImGui::TableSetupColumn(model.headerData(i).c_str(), ImGuiTableColumnFlags_WidthFixed, sizes[i].x);
    }
    ImGuiTable *table = ImGui::GetCurrentTable();
    ImDrawList *painter = ImGui::GetWindowDrawList();

    ImGui::TableNextRow(ImGuiTableRowFlags_Headers, header_height);
    for (int i = 0; i < cols; ++i) {
      if (!ImGui::TableSetColumnIndex(i)) continue;
      header.paintSection(painter, ImGui::TableGetCellBgRect(table, i), i);
      ImGui::Dummy(ImVec2(0, header_height - style.CellPadding.y * 2));
    }

    ImGuiListClipper clipper;
    clipper.Begin(model.rowCount(), row_height);
    while (clipper.Step()) {
      for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
        const auto &m = model.messages[row];
        ImGui::TableNextRow(0, row_height);
        ImGui::PushID(row);
        for (int col = 0; col < cols; ++col) {
          if (!ImGui::TableSetColumnIndex(col)) continue;
          // cells are selected, not rows; there is no hover highlight, only the selection background
          const bool cell_selected = selected_row == row && selected_col == col;
          ImGui::PushStyleColor(ImGuiCol_HeaderHovered, cell_selected ? ImGui::GetColorU32(ImGuiCol_Header) : IM_COL32(0, 0, 0, 0));
          ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGui::GetColorU32(ImGuiCol_Header));
          ImGui::PushID(col);
          if (ImGui::Selectable("##cell", cell_selected, ImGuiSelectableFlags_AllowOverlap,
                                ImVec2(0, row_height - style.CellPadding.y * 2))) {
            selected_row = row;
            selected_col = col;
          }
          ImGui::PopID();
          ImGui::PopStyleColor(2);
          const bool hex_cell = model.isHexMode() && col == 1;
          delegate.paint(painter, ImGui::TableGetCellBgRect(table, col), cell_selected, false,
                         hex_cell ? std::string() : model.data(row, col), hex_cell ? &m.data : nullptr, hex_cell ? &m.colors : nullptr);
        }
        ImGui::PopID();
      }
      // fetch more when the last row is visible or the scrollbar is at its maximum
      if (clipper.DisplayEnd >= model.rowCount()) fetch_more = true;
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) fetch_more = true;
    vscrollbar_visible_ = table->InnerWindow->ScrollbarY;
    ImGui::EndTable();
  }
  if (fetch_more && model.canFetchMore()) {
    model.fetchMore();
  }
}
