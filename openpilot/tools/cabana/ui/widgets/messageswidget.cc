#include "tools/cabana/ui/widgets/messageswidget.h"

#include <cctype>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <utility>

#include "imgui_internal.h"
#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/strings.h"

MessagesWidget::MessagesWidget() : view(&view_), header(&header_), delegate(&delegate_), model(&model_), delegate_(settings.multiple_lines_hex) {
  // toolbar: createToolBar() is drawn every frame by draw()
  // message table
  view->setItemDelegate(delegate);
  view->setModel(model);
  view->setHeader(header);
  header->setModel(model);
  // setSortingEnabled: ImGuiTableFlags_Sortable
  model->sort(MessageListModel::Column::NAME, ImGuiSortDirection_Ascending);  // view->sortByColumn
  // setAllColumnsShowFocus: the row Selectable spans all columns

  // Must be called before setting any header parameters to avoid overriding
  restoreHeaderState(settings.message_header_state);
  // setSectionsMovable: ImGuiTableFlags_Reorderable; DATA is QHeaderView::Fixed: ImGuiTableColumnFlags_NoResize
  header->updateGeometries();

  // signals/slots
  connections_.push_back(model->modelReset.connect([this]() {
    if (current_msg_id) {
      selectMessage(*current_msg_id);
    }
    view->updateBytesSectionSize();
    updateTitle();
  }));
  connections_.push_back(view->currentChanged.connect([this](int current, int previous) {
    if (current >= 0 && current < (int)model->items_.size()) {
      const auto &id = model->items_[current].id;
      if (!current_msg_id || id != *current_msg_id) {
        current_msg_id = id;
        msgSelectionChanged(*current_msg_id);
      }
    }
  }));

  suppressHighlighted();  // Qt: called from createToolBar() with no sender
}

std::string MessagesWidget::whatsThis() const {
  return R"(
    <b>Message View</b><br/>
    <!-- TODO: add description here -->
    <span style="color:gray">Byte color</span><br />
    <span style="color:gray;">&#9632; </span> constant changing<br />
    <span style="color:blue;">&#9632; </span> increasing<br />
    <span style="color:red;">&#9632; </span> decreasing<br />
    <span style="color:gray">Shortcuts</span><br />
    Horizontal Scrolling: <span style="background-color:lightGray;color:gray">&nbsp;shift+wheel&nbsp;</span>
  )";
}

void MessagesWidget::createToolBar() {
  // layout->setContentsMargins(0, 9, 0, 0)
  ImGui::Dummy(ImVec2(0, std::max(0.0f, 9 - ImGui::GetStyle().ItemSpacing.y)));
  if (ImGui::Button("Suppress Highlighted")) suppressHighlighted(true);
  ImGui::SameLine();
  ImGui::BeginDisabled(!suppress_clear_enabled);
  const std::string clear_label = suppress_clear_text + "##suppress_clear";
  if (ImGui::Button(clear_label.c_str())) suppressHighlighted(false);
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Clear suppressed");

  // layout->addStretch(1): right-align the rest
  const ImGuiStyle &style = ImGui::GetStyle();
  const float checkbox_width = ImGui::CalcTextSize("Suppress Signals").x + ImGui::GetFrameHeight() + style.ItemInnerSpacing.x;
  const float view_button_width = ImGui::CalcTextSize(icon::THREE_DOTS).x + style.FramePadding.x * 2;
  ImGui::SameLine();
  const float x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - (checkbox_width + style.ItemSpacing.x + view_button_width);
  ImGui::SetCursorPosX(std::max(x, ImGui::GetCursorPosX()));

  bool suppress_defined_signals = settings.suppress_defined_signals;
  if (checkBox("Suppress Signals", &suppress_defined_signals)) can->suppressDefinedSignals(suppress_defined_signals);
  ImGui::SetItemTooltip("Suppress defined signals");
  ImGui::SameLine();

  // ToolButton("three-dots", "View...") with an instant popup menu
  if (toolButton(icon::THREE_DOTS, "View...")) ImGui::OpenPopup("menu");
}

void MessagesWidget::updateTitle() {
  auto stats = std::accumulate(
      model->items_.begin(), model->items_.end(), std::pair<size_t, size_t>(),
      [](const auto &pair, const auto &item) {
        auto m = dbc()->msg(item.id);
        return m ? std::make_pair(pair.first + 1, pair.second + m->sigs.size()) : pair;
      });
  char buf[128];
  snprintf(buf, sizeof(buf), "%zu Messages (%zu DBC Messages, %zu Signals)", model->items_.size(), stats.first, stats.second);
  title_ = buf;
  titleChanged(title_);
}

void MessagesWidget::selectMessage(const MessageId &msg_id) {
  auto it = std::find_if(model->items_.cbegin(), model->items_.cend(),
                         [&msg_id](auto &item) { return item.id == msg_id; });
  if (it != model->items_.cend()) {
    view->setCurrentIndex(std::distance(model->items_.cbegin(), it));
  }
}

void MessagesWidget::suppressHighlighted() {
  suppressHighlighted(false);
}

void MessagesWidget::suppressHighlighted(bool from_suppress_add) {
  int n = from_suppress_add ? can->suppressHighlighted() : (can->clearSuppressed(), 0);
  suppress_clear_text = n > 0 ? "Clear (" + std::to_string(n) + ")" : "Clear";
  suppress_clear_enabled = n > 0;
}

void MessagesWidget::headerContextMenuEvent() {
  ImGui::OpenPopup("menu");  // at the mouse position
}

void MessagesWidget::menuAboutToShow() {
  if (!ImGui::BeginPopup("menu")) return;
  for (int i = 0; i < header->count(); ++i) {
    int logical_index = header->logicalIndex(i);
    // Can't hide the name column
    if (ImGui::MenuItem(model->headerData(logical_index).c_str(), nullptr, !header->isSectionHidden(logical_index), logical_index > 0)) {
      header->setSectionHidden(logical_index, !header->isSectionHidden(logical_index));
    }
  }
  ImGui::Separator();
  if (ImGui::MenuItem("Multi-Line bytes", nullptr, settings.multiple_lines_hex)) {
    setMultiLineBytes(!settings.multiple_lines_hex);
  }
  if (ImGui::MenuItem("Show inactive messages", nullptr, model->show_inactive_messages)) {
    model->showInactiveMessages(!model->show_inactive_messages);
  }
  ImGui::EndPopup();
}

void MessagesWidget::setMultiLineBytes(bool multi) {
  settings.multiple_lines_hex = multi;
  delegate->setMultipleLines(multi);
  view->updateBytesSectionSize();
  // view->doItemsLayout(): imgui lays out every frame
}

void MessagesWidget::draw() {
  createToolBar();
  view->draw();
  if (std::exchange(header->customContextMenuRequested, false)) headerContextMenuEvent();
  menuAboutToShow();
}

// MessageListModel

MessageListModel::MessageListModel() {
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *msgs, bool has_new_ids) { msgsReceived(msgs, has_new_ids); }));
  connections_.push_back(dbc()->fileChanged.connect([this]() { dbcModified(); }));
  connections_.push_back(UndoStack::instance()->indexChanged.connect([this]() { dbcModified(); }));
}

std::string MessageListModel::headerData(int section) const {
  switch (section) {
    case Column::NAME: return "Name";
    case Column::SOURCE: return "Bus";
    case Column::ADDRESS: return "ID";
    case Column::NODE: return "Node";
    case Column::FREQ: return "Freq";
    case Column::COUNT: return "Count";
    case Column::DATA: return "Bytes";
  }
  return {};
}

std::string MessageListModel::data(int row, int column) const {
  if (row < 0 || row >= (int)items_.size()) return {};

  auto getFreq = [](float freq) -> std::string {
    if (freq > 0) {
      char buf[32];
      if (freq >= 0.95) {
        snprintf(buf, sizeof(buf), "%.0f", std::nearbyint(freq));
      } else {
        snprintf(buf, sizeof(buf), "%.2f", freq);
      }
      return buf;
    } else {
      return "--";
    }
  };

  const static std::string NA = "N/A";
  const auto &item = items_[row];
  switch (column) {
    case Column::NAME: return item.name;
    case Column::SOURCE: return item.id.source != INVALID_SOURCE ? std::to_string(item.id.source) : NA;
    case Column::ADDRESS: return utils::toHexString(item.id.address);
    case Column::NODE: return item.node;
    case Column::FREQ: return item.id.source != INVALID_SOURCE ? getFreq(can->lastMessage(item.id).freq) : NA;
    case Column::COUNT: return item.id.source != INVALID_SOURCE ? std::to_string(can->lastMessage(item.id).count) : NA;
    case Column::DATA: return item.id.source != INVALID_SOURCE ? "" : NA;
  }
  // ColorsRole/BytesRole: the view reads can->lastMessage(item.id).colors/dat directly
  return {};
}

std::string MessageListModel::toolTip(int row, int column) const {
  if (row < 0 || row >= (int)items_.size()) return {};
  const auto &item = items_[row];
  if (column == Column::NAME) {
    auto msg = dbc()->msg(item.id);
    auto tooltip = item.name;
    if (msg && !msg->comment.empty()) tooltip += "\n" + msg->comment;  // the comment is drawn in gray
    return tooltip;
  }
  return {};
}

void MessageListModel::setFilterStrings(const std::map<int, std::string> &filters) {
  filters_ = filters;
  filterAndSort();
}

void MessageListModel::showInactiveMessages(bool show) {
  show_inactive_messages = show;
  filterAndSort();
}

void MessageListModel::dbcModified() {
  dbc_messages_.clear();
  for (const auto &[_, m] : dbc()->getMessages(-1)) {
    dbc_messages_.insert(MessageId{.source = INVALID_SOURCE, .address = m.address});
  }
  filterAndSort();
}

void MessageListModel::sortItems(std::vector<MessageListModel::Item> &items) {
  auto compare = [this](const auto &l, const auto &r) {
    switch (sort_column) {
      case Column::NAME: return std::tie(l.name, l.id) < std::tie(r.name, r.id);
      case Column::SOURCE: return std::tie(l.id.source, l.id.address) < std::tie(r.id.source, r.id.address);
      case Column::ADDRESS: return std::tie(l.id.address, l.id.source) < std::tie(r.id.address, r.id.source);
      case Column::NODE: return std::tie(l.node, l.id) < std::tie(r.node, r.id);
      case Column::FREQ: return std::tie(can->lastMessage(l.id).freq, l.id) < std::tie(can->lastMessage(r.id).freq, r.id);
      case Column::COUNT: return std::tie(can->lastMessage(l.id).count, l.id) < std::tie(can->lastMessage(r.id).count, r.id);
      default: return false; // Default case to suppress compiler warning
    }
  };

  if (sort_order == ImGuiSortDirection_Descending)
    std::stable_sort(items.rbegin(), items.rend(), compare);
  else
    std::stable_sort(items.begin(), items.end(), compare);
}

// QString::contains(txt, Qt::CaseInsensitive)
static bool containsCI(const std::string &s, const std::string &txt) {
  auto it = std::search(s.begin(), s.end(), txt.begin(), txt.end(),
                        [](unsigned char a, unsigned char b) { return std::tolower(a) == std::tolower(b); });
  return it != s.end();
}

// QString::split('-')
static std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos; (pos = s.find(sep, start)) != std::string::npos; start = pos + 1) {
    parts.push_back(s.substr(start, pos - start));
  }
  parts.push_back(s.substr(start));
  return parts;
}

// QString::toUInt(&ok, base): surrounding whitespace is ignored; no sign, no 0x prefix
static unsigned int toUInt(const std::string &s, bool *ok, int base) {
  size_t begin = 0, end = s.size();
  while (begin < end && std::isspace((unsigned char)s[begin])) ++begin;
  while (end > begin && std::isspace((unsigned char)s[end - 1])) --end;
  unsigned long long v = 0;
  *ok = begin < end;
  for (size_t i = begin; i < end && *ok; ++i) {
    const unsigned char c = std::tolower((unsigned char)s[i]);
    int digit = -1;
    if (c >= '0' && c <= '9') digit = c - '0';
    else if (c >= 'a' && c <= 'z') digit = c - 'a' + 10;
    *ok = digit >= 0 && digit < base;
    v = v * base + std::max(digit, 0);
    *ok = *ok && v <= std::numeric_limits<unsigned int>::max();
  }
  return *ok ? (unsigned int)v : 0;
}

static bool parseRange(const std::string &filter, uint32_t value, int base = 10) {
  // Parse out filter string into a range (e.g. "1" -> {1, 1}, "1-3" -> {1, 3}, "1-" -> {1, inf})
  unsigned int min = std::numeric_limits<unsigned int>::min();
  unsigned int max = std::numeric_limits<unsigned int>::max();
  auto s = split(filter, '-');
  bool ok = s.size() >= 1 && s.size() <= 2;
  if (ok && !s[0].empty()) min = toUInt(s[0], &ok, base);
  if (ok && s.size() == 1) {
    max = min;
  } else if (ok && s.size() == 2 && !s[1].empty()) {
    max = toUInt(s[1], &ok, base);
  }
  return ok && value >= min && value <= max;
}

bool MessageListModel::match(const MessageListModel::Item &item) {
  if (filters_.empty())
    return true;

  bool match = true;
  const auto &data = can->lastMessage(item.id);
  for (auto it = filters_.cbegin(); it != filters_.cend() && match; ++it) {
    const std::string &txt = it->second;
    switch (it->first) {
      case Column::NAME: {
        match = containsCI(item.name, txt);
        if (!match) {
          const auto m = dbc()->msg(item.id);
          match = m && std::any_of(m->sigs.cbegin(), m->sigs.cend(),
                                   [&txt](const auto &s) { return containsCI(s->name, txt); });
        }
        break;
      }
      case Column::SOURCE:
        match = parseRange(txt, item.id.source);
        break;
      case Column::ADDRESS:
        match = containsCI(utils::toHexString(item.id.address), txt);
        match = match || parseRange(txt, item.id.address, 16);
        break;
      case Column::NODE:
        match = containsCI(item.node, txt);
        break;
      case Column::FREQ:
        match = parseRange(txt, data.freq);
        break;
      case Column::COUNT:
        match = parseRange(txt, data.count);
        break;
      case Column::DATA:
        match = containsCI(utils::toHex(data.dat), txt);
        break;
    }
  }
  return match;
}

bool MessageListModel::filterAndSort() {
  // merge CAN and DBC messages
  std::vector<MessageId> all_messages;
  all_messages.reserve(can->lastMessages().size() + dbc_messages_.size());
  auto dbc_msgs = dbc_messages_;
  for (const auto &[id, m] : can->lastMessages()) {
    all_messages.push_back(id);
    dbc_msgs.erase(MessageId{.source = INVALID_SOURCE, .address = id.address});
  }
  all_messages.insert(all_messages.end(), dbc_msgs.begin(), dbc_msgs.end());

  // filter and sort
  std::vector<Item> items;
  items.reserve(all_messages.size());
  for (const auto &id : all_messages) {
    if (show_inactive_messages || can->isMessageActive(id)) {
      auto msg = dbc()->msg(id);
      Item item = {.id = id,
                   .name = msg ? msg->name : UNTITLED,
                   .node = msg ? msg->transmitter : std::string()};
      if (match(item))
        items.emplace_back(item);
    }
  }
  sortItems(items);

  if (items_ != items) {
    // beginResetModel/endResetModel
    items_ = std::move(items);
    modelReset();
    return true;
  }
  return false;
}

void MessageListModel::msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids) {
  if (has_new_ids || ((filters_.count(Column::FREQ) || filters_.count(Column::COUNT) || filters_.count(Column::DATA)) &&
                      ++sort_threshold_ == STREAM_UPDATE_FPS)) {
    sort_threshold_ = 0;
    if (filterAndSort()) return;
  }

  // Update viewport: imgui redraws every frame
}

void MessageListModel::sort(int column, ImGuiSortDirection order) {
  if (column != Column::DATA) {
    sort_column = column;
    sort_order = order;
    filterAndSort();
  }
}

// MessageView

// QHeaderView::defaultSectionSize
static constexpr float DEFAULT_SECTION_SIZE = 100.0f;

// The Fusion style draws a down pointing arrow for AscendingOrder, imgui draws an up arrow for
// ImGuiSortDirection_Ascending. Feed imgui the opposite direction so the glyph matches Qt and flip it
// back before it reaches the model.
static inline ImGuiSortDirection flipSortDirection(ImGuiSortDirection dir) {
  return dir == ImGuiSortDirection_Ascending ? ImGuiSortDirection_Descending : ImGuiSortDirection_Ascending;
}

void MessageView::drawRow(int row) {
  const auto &item = model_->items_[row];
  const bool selected = row == current_row_;
  const bool inactive = !can->isMessageActive(item.id);
  const auto &m = can->lastMessage(item.id);
  const std::vector<uint8_t> *bytes = item.id.source != INVALID_SOURCE ? &m.dat : nullptr;
  const float row_height = delegate_->sizeHint(bytes).y - ImGui::GetStyle().CellPadding.y * 2;
  ImGui::TableNextRow();
  // inactive: palette Disabled Text, HighlightedText alpha 100 (applied by the delegate)
  ImGui::PushID(row);

  bool row_item_submitted = false;
  for (int column = MessageListModel::Column::NAME; column < model_->columnCount(); ++column) {
    if (!ImGui::TableSetColumnIndex(column)) continue;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    const ImRect rect(pos, ImVec2(pos.x + width, pos.y + row_height));

    const bool row_item = !row_item_submitted;
    if (row_item) {
      // the row selection spans all columns; submit it in the first visible column so that it is not
      // clipped away when the table is scrolled horizontally
      row_item_submitted = true;
      // QAbstractItemView selects on press. QTreeView has no hover highlight, only the selection background.
      // Selectable() prefers HeaderHovered over Header whenever the row is hovered, even when it is selected,
      // so the selected row has to keep the selection color as its hover color or it looks unselected.
      ImGui::PushStyleColor(ImGuiCol_HeaderHovered, selected ? ImGui::GetColorU32(ImGuiCol_Header) : IM_COL32(0, 0, 0, 0));
      ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGui::GetColorU32(ImGuiCol_Header));
      if (ImGui::Selectable("##row", selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_SelectOnClick, ImVec2(0, row_height))) {
        setCurrentIndex(row);
      }
      ImGui::PopStyleColor(2);
      if (selected && scroll_to_current_) {
        // scrollTo(EnsureVisible): only scroll when the row is outside the viewport, and only far enough
        const ImGuiTable *table = ImGui::GetCurrentTable();
        const ImGuiWindow *inner = table->InnerWindow;
        const float view_top = inner->InnerClipRect.Min.y + inner->DecoInnerSizeY1;
        const float view_bottom = inner->InnerClipRect.Max.y;
        const ImVec2 item_min = ImGui::GetItemRectMin();
        const ImVec2 item_max = ImGui::GetItemRectMax();
        if (item_min.y < view_top) {
          ImGui::SetScrollHereY(0.0f);
        } else if (item_max.y > view_bottom) {
          ImGui::SetScrollHereY(1.0f);
        }
        scroll_to_current_ = false;
      }
      // the tooltip is on the Name item in Qt, so only show it while the mouse is over the Name column
      const ImGuiTableColumn &name_col = ImGui::GetCurrentTable()->Columns[MessageListModel::Column::NAME];
      const float mouse_x = ImGui::GetIO().MousePos.x;
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip) && mouse_x >= name_col.MinX && mouse_x < name_col.MaxX) {
        const std::string tooltip = model_->toolTip(row, MessageListModel::Column::NAME);
        if (!tooltip.empty()) {
          const size_t nl = tooltip.find('\n');
          ImGui::BeginTooltip();
          ImGui::TextUnformatted(tooltip.c_str(), nl != std::string::npos ? tooltip.c_str() + nl : nullptr);
          if (nl != std::string::npos) ImGui::TextDisabled("%s", tooltip.c_str() + nl + 1);
          ImGui::EndTooltip();
        }
      }
    }

    if (column == MessageListModel::Column::DATA && bytes) {
      delegate_->paint(ImGui::GetWindowDrawList(), rect, selected, inactive, "", bytes, &m.colors);
    } else {
      delegate_->paint(ImGui::GetWindowDrawList(), rect, selected, inactive, model_->data(row, column));
    }
    // the Selectable already sized its cell
    if (!row_item) ImGui::Dummy(ImVec2(width, row_height));
  }

  ImGui::PopID();
  // the grid lines (row bottom border, column borders) are ImGuiTableFlags_Borders
}

void MessageView::setModel(MessageListModel *model) {
  model_ = model;
  // Bypass the slow call to QTreeView::dataChanged: imgui has no height cache, nothing to invalidate.
  // A model reset invalidates the current index like QAbstractItemView does.
  connections_.push_back(model_->modelReset.connect([this]() { current_row_ = -1; }));
}

void MessageView::setCurrentIndex(int row) {
  if (row < 0 || row >= model_->rowCount()) return;
  const int previous = std::exchange(current_row_, row);
  scroll_to_current_ = true;  // scrollTo(EnsureVisible)
  if (previous != row) currentChanged(row, previous);
}

void MessageView::updateBytesSectionSize() {
  auto delegate = itemDelegate();
  int max_bytes = 8;
  if (!delegate->multipleLines()) {
    for (const auto &[_, m] : can->lastMessages()) {
      max_bytes = std::max<int>(max_bytes, m.dat.size());
    }
  }
  // setUniformRowHeights(!multipleLines()): draw() only uses the clipper with uniform row heights
  // header()->resizeSection(DATA, ...): the minimum width of the stretched last section
  bytes_section_bytes_ = max_bytes;
}

void MessageView::keyPressEvent() {
  if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) || ImGui::IsAnyItemActive()) return;
  const int last = model_->rowCount() - 1;
  if (last < 0) return;
  if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && current_row_ > 0) {
    setCurrentIndex(current_row_ - 1);
  } else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) && current_row_ < last) {
    setCurrentIndex(current_row_ + 1);
  } else if (ImGui::IsKeyPressed(ImGuiKey_Home)) {
    setCurrentIndex(0);
  } else if (ImGui::IsKeyPressed(ImGuiKey_End)) {
    setCurrentIndex(last);
  } else if (ImGui::IsKeyPressed(ImGuiKey_PageUp)) {
    setCurrentIndex(std::max(current_row_ - visible_rows_, 0));
  } else if (ImGui::IsKeyPressed(ImGuiKey_PageDown)) {
    setCurrentIndex(std::min(current_row_ + visible_rows_, last));
  }
}

void MessageView::draw() {
  delegate_->updateFontMetrics();
  keyPressEvent();

  const ImGuiTableFlags flags = ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
                                ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Borders;
  // setStretchLastSection: with ScrollX a stretch column needs an explicit inner width
  const float bytes_width = delegate_->sizeForBytes(bytes_section_bytes_).x;
  const float avail_width = ImGui::GetContentRegionAvail().x - (has_scrollbar_y_ ? ImGui::GetStyle().ScrollbarSize : 0);
  const float inner_width = std::max(avail_width, fixed_columns_width_ + bytes_width);
  if (!ImGui::BeginTable("messages", model_->columnCount(), flags, ImVec2(0, 0), inner_width)) return;

  // QTreeView has no frozen column: only the header and the filter row stay put
  ImGui::TableSetupScrollFreeze(0, 2);
  // MessagesWidget shows its own header menu; imgui's default one is never drawn, so clear the flag
  // TableHeader() leaves behind (it would keep the column header highlighted forever)
  ImGuiTable *table = ImGui::GetCurrentTable();
  table->DisableDefaultContextMenu = true;
  table->IsContextPopupOpen = false;
  for (int i = 0; i < model_->columnCount(); ++i) {
    // PreferSortDescending: with the flipped direction the first click on a section sorts ascending, like QHeaderView
    ImGuiTableColumnFlags column_flags = ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending;
    float width = DEFAULT_SECTION_SIZE;
    if (i == MessageListModel::Column::NAME) {
      column_flags |= ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_NoHide;
    } else if (i == MessageListModel::Column::DATA) {
      column_flags = ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_NoResize;
      width = 0;
    }
    if (header_->isSectionHidden(i)) column_flags |= ImGuiTableColumnFlags_Disabled;
    ImGui::TableSetupColumn(model_->headerData(i).c_str(), column_flags, width);
  }

  if (ImGuiTableSortSpecs *specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsDirty) {
    if (specs->SpecsCount > 0) model_->sort(specs->Specs[0].ColumnIndex, flipSortDirection(specs->Specs[0].SortDirection));
    specs->SpecsDirty = false;
    if (current_row_ >= 0) scroll_to_current_ = true;  // QTreeView::sortByColumn: scrollTo(currentIndex)
  }

  header_->draw();

  if (!delegate_->multipleLines()) {
    ImGuiListClipper clipper;
    clipper.Begin(model_->rowCount());
    if (scroll_to_current_ && current_row_ >= 0) clipper.IncludeItemByIndex(current_row_);
    while (clipper.Step()) {
      for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
        drawRow(row);
      }
    }
  } else {
    // non-uniform row heights: no clipper
    for (int row = 0; row < model_->rowCount(); ++row) {
      drawRow(row);
    }
  }

  // for the next frame: the stretch inner width and the page size of keyPressEvent
  const ImGuiTableColumn &data_column = table->Columns[MessageListModel::Column::DATA];
  fixed_columns_width_ = data_column.IsEnabled ? table->ColumnsGivenWidth - data_column.WidthGiven : 0;
  has_scrollbar_y_ = table->InnerWindow->ScrollbarY;
  visible_rows_ = std::max(1, (int)(table->InnerWindow->InnerRect.GetHeight() / delegate_->sizeHint(nullptr).y) - 2);
  ImGui::EndTable();
}

// MessageViewHeader

MessageViewHeader::MessageViewHeader() {
  // sectionResized/sectionMoved -> updateHeaderPositions: done every frame in draw()
}

void MessageViewHeader::updateFilters() {
  std::map<int, std::string> filters;
  for (int i = 0; i < (int)editors.size(); i++) {
    if (!editors[i].empty()) {
      filters[i] = editors[i];
    }
  }
  model()->setFilterStrings(filters);
}

void MessageViewHeader::updateHeaderPositions() {
  // the editors live in the table's filter row; record the visual order of the sections for the menu
  ImGuiTable *table = ImGui::GetCurrentTable();
  for (int i = 0; i < (int)editors.size(); i++) {
    display_order_[i] = table->DisplayOrderToIndex[i];
  }
}

void MessageViewHeader::updateGeometries() {
  for (int i = (int)editors.size(); i < model()->columnCount(); i++) {
    std::string column_name = model()->headerData(i);
    editors.emplace_back();
    placeholders_.push_back("Filter " + column_name);
    hidden_.push_back(false);
    display_order_.push_back(i);
  }
  // setViewportMargins: the filter row is a frozen table row
}

void MessageViewHeader::draw() {
  updateGeometries();

  // TableHeadersRow, submitted manually to catch the right click
  ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
  for (int i = 0; i < count(); i++) {
    if (!ImGui::TableSetColumnIndex(i)) continue;
    ImGui::PushID(i);
    ImGui::TableHeader(ImGui::TableGetColumnName(i));
    // same timing as TableHeader's own TableOpenContextMenu, so MessagesWidget's menu is opened last
    // and replaces the (disabled) table context menu in the popup stack
    if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) customContextMenuRequested = true;
    ImGui::PopID();
  }
  updateHeaderPositions();

  // the filter editors under the header
  const ImGuiStyle &style = ImGui::GetStyle();
  const float clear_width = ImGui::CalcTextSize(icon::X).x + style.FramePadding.x * 2;
  ImGui::TableNextRow();
  for (int i = 0; i < count(); i++) {
    if (!ImGui::TableSetColumnIndex(i)) continue;
    ImGui::PushID(i);
    ImGui::SetNextItemWidth(editors[i].empty() ? -FLT_MIN : std::max(1.0f, ImGui::GetContentRegionAvail().x - clear_width));
    if (inputText("##filter", &editors[i], placeholders_[i].c_str())) updateFilters();
    // setClearButtonEnabled: the button only shows when the field is non-empty
    if (!editors[i].empty()) {
      ImGui::SameLine(0, 0);
      if (ImGui::Button(icon::X)) {
        editors[i].clear();
        updateFilters();
      }
    }
    ImGui::PopID();
  }
}
