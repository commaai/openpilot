#include "tools/cabana/ui/widgets/messageswidget.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <utility>

#include "imgui_internal.h"
#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/ui/widgets/messagebytes.h"
#include "tools/cabana/utils/strings.h"

namespace {

const char *COLUMN_TITLES[MessageList::COLUMN_COUNT] = {"Name", "Bus", "ID", "Node", "Freq", "Count", "Bytes"};
constexpr float DEFAULT_SECTION_SIZE = 100.0f;

// surrounding whitespace is ignored; no sign, no 0x prefix
unsigned int toUInt(const std::string &s, bool *ok, int base) {
  const char *b = s.data(), *e = b + s.size();
  while (b < e && std::isspace((unsigned char)*b)) ++b;
  while (e > b && std::isspace((unsigned char)e[-1])) --e;
  unsigned int v = 0;
  auto [p, ec] = std::from_chars(b, e, v, base);
  *ok = b < e && p == e && ec == std::errc();
  return *ok ? v : 0;
}

bool parseRange(const std::string &filter, uint32_t value, int base = 10) {
  // parse the filter string into a range: "1" -> {1, 1}, "1-3" -> {1, 3}, "1-" -> {1, inf}
  unsigned int min = std::numeric_limits<unsigned int>::min();
  unsigned int max = std::numeric_limits<unsigned int>::max();
  auto s = utils::split(filter, '-');
  bool ok = s.size() >= 1 && s.size() <= 2;
  if (ok && !s[0].empty()) min = toUInt(s[0], &ok, base);
  if (ok && s.size() == 1) {
    max = min;
  } else if (ok && s.size() == 2 && !s[1].empty()) {
    max = toUInt(s[1], &ok, base);
  }
  return ok && value >= min && value <= max;
}

// imgui draws an up arrow for ImGuiSortDirection_Ascending, cabana wants a down pointing one. Feed imgui
// the opposite direction and flip it back before it reaches the list.
inline ImGuiSortDirection flipSortDirection(ImGuiSortDirection dir) {
  return dir == ImGuiSortDirection_Ascending ? ImGuiSortDirection_Descending : ImGuiSortDirection_Ascending;
}

std::string formatFreq(float freq) {
  if (freq <= 0) return "--";
  char buf[32];
  snprintf(buf, sizeof(buf), freq >= 0.95 ? "%.0f" : "%.2f", freq >= 0.95 ? std::nearbyint(freq) : freq);
  return buf;
}

// the text of a cell; the DATA cell paints the bytes itself
std::string cellText(const MessageList::Item &item, int column) {
  const bool seen = item.id.source != INVALID_SOURCE;
  switch (column) {
    case MessageList::NAME: return item.name;
    case MessageList::SOURCE: return seen ? std::to_string(item.id.source) : "N/A";
    case MessageList::ADDRESS: return utils::toHexString(item.id.address);
    case MessageList::NODE: return item.node;
    case MessageList::FREQ: return seen ? formatFreq(can->lastMessage(item.id).freq) : "N/A";
    case MessageList::COUNT: return seen ? std::to_string(can->lastMessage(item.id).count) : "N/A";
    case MessageList::DATA: return seen ? "" : "N/A";
  }
  return {};
}

}  // namespace

// MessageList

MessageList::MessageList() {
  connections_.push_back(can->msgsReceived.connect([this](const std::set<MessageId> *msgs, bool has_new_ids) { msgsReceived(msgs, has_new_ids); }));
  connections_.push_back(dbc()->fileChanged.connect([this]() { dbcModified(); }));
  connections_.push_back(UndoStack::instance()->indexChanged.connect([this]() { dbcModified(); }));
}

void MessageList::setFilters(const std::map<int, std::string> &filters) {
  filters_ = filters;
  filterAndSort();
}

void MessageList::showInactiveMessages(bool show) {
  show_inactive_messages = show;
  filterAndSort();
}

void MessageList::dbcModified() {
  dbc_messages_.clear();
  for (const auto &[_, m] : dbc()->getMessages(-1)) {
    dbc_messages_.insert(MessageId{.source = INVALID_SOURCE, .address = m.address});
  }
  filterAndSort();
}

void MessageList::sortItems(std::vector<Item> &list) {
  auto compare = [this](const auto &l, const auto &r) {
    switch (sort_column_) {
      case NAME: return std::tie(l.name, l.id) < std::tie(r.name, r.id);
      case SOURCE: return std::tie(l.id.source, l.id.address) < std::tie(r.id.source, r.id.address);
      case ADDRESS: return std::tie(l.id.address, l.id.source) < std::tie(r.id.address, r.id.source);
      case NODE: return std::tie(l.node, l.id) < std::tie(r.node, r.id);
      case FREQ: return std::tie(can->lastMessage(l.id).freq, l.id) < std::tie(can->lastMessage(r.id).freq, r.id);
      case COUNT: return std::tie(can->lastMessage(l.id).count, l.id) < std::tie(can->lastMessage(r.id).count, r.id);
      default: return false;
    }
  };

  if (sort_order_ == ImGuiSortDirection_Descending)
    std::stable_sort(list.rbegin(), list.rend(), compare);
  else
    std::stable_sort(list.begin(), list.end(), compare);
}

bool MessageList::match(const Item &item) {
  if (filters_.empty()) return true;

  bool match = true;
  const auto &data = can->lastMessage(item.id);
  for (auto it = filters_.cbegin(); it != filters_.cend() && match; ++it) {
    const std::string &txt = it->second;
    switch (it->first) {
      case NAME: {
        match = utils::containsCI(item.name, txt);
        if (!match) {
          const auto m = dbc()->msg(item.id);
          match = m && std::any_of(m->sigs.cbegin(), m->sigs.cend(),
                                   [&txt](const auto &s) { return utils::containsCI(s->name, txt); });
        }
        break;
      }
      case SOURCE:
        match = parseRange(txt, item.id.source);
        break;
      case ADDRESS:
        match = utils::containsCI(utils::toHexString(item.id.address), txt);
        match = match || parseRange(txt, item.id.address, 16);
        break;
      case NODE:
        match = utils::containsCI(item.node, txt);
        break;
      case FREQ:
        match = parseRange(txt, data.freq);
        break;
      case COUNT:
        match = parseRange(txt, data.count);
        break;
      case DATA:
        match = utils::containsCI(utils::toHex(data.dat), txt);
        break;
    }
  }
  return match;
}

bool MessageList::filterAndSort() {
  // merge CAN and DBC messages
  std::vector<MessageId> all_messages;
  all_messages.reserve(can->lastMessages().size() + dbc_messages_.size());
  auto dbc_msgs = dbc_messages_;
  for (const auto &[id, m] : can->lastMessages()) {
    all_messages.push_back(id);
    dbc_msgs.erase(MessageId{.source = INVALID_SOURCE, .address = id.address});
  }
  all_messages.insert(all_messages.end(), dbc_msgs.begin(), dbc_msgs.end());

  std::vector<Item> new_items;
  new_items.reserve(all_messages.size());
  for (const auto &id : all_messages) {
    if (show_inactive_messages || can->isMessageActive(id)) {
      auto msg = dbc()->msg(id);
      Item item = {.id = id, .name = msg ? msg->name : UNTITLED, .node = msg ? msg->transmitter : std::string()};
      if (match(item)) new_items.emplace_back(item);
    }
  }
  sortItems(new_items);

  if (items != new_items) {
    items = std::move(new_items);
    changed();
    return true;
  }
  return false;
}

void MessageList::msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids) {
  if (has_new_ids || ((filters_.count(FREQ) || filters_.count(COUNT) || filters_.count(DATA)) &&
                      ++sort_threshold_ == STREAM_UPDATE_FPS)) {
    sort_threshold_ = 0;
    filterAndSort();
  }
}

void MessageList::sort(int column, ImGuiSortDirection order) {
  if (column != DATA) {
    sort_column_ = column;
    sort_order_ = order;
    filterAndSort();
  }
}

// MessagesWidget

MessagesWidget::MessagesWidget() {
  std::iota(display_order_.begin(), display_order_.end(), 0);
  list_.sort(MessageList::NAME, ImGuiSortDirection_Ascending);

  connections_.push_back(list_.changed.connect([this]() {
    current_row_ = -1;  // the rows moved
    if (current_msg_id_) selectMessage(*current_msg_id_);
    updateBytesSectionSize();
    updateTitle();
  }));

  suppressHighlighted();
}

std::string MessagesWidget::whatsThis() const {
  return R"(
    <b>Message View</b><br/>
    <span style="color:gray">Byte color</span><br />
    <span style="color:gray;">&#9632; </span> constantly changing<br />
    <span style="color:blue;">&#9632; </span> increasing<br />
    <span style="color:red;">&#9632; </span> decreasing<br />
    <span style="color:gray">Shortcuts</span><br />
    Horizontal Scrolling: <span style="background-color:lightGray;color:gray">&nbsp;Shift+Wheel&nbsp;</span>
  )";
}

void MessagesWidget::drawToolBar() {
  const ImGuiStyle &style = ImGui::GetStyle();
  // Reserve space for View so it remains accessible when other controls overflow.
  const std::string clear_label = suppress_clear_text_ + "##suppress_clear";
  std::vector<ToolbarItem> items;
  items.push_back({toolbarButtonWidth("Suppress Highlighted"), [this]() {
    if (ImGui::Button("Suppress Highlighted")) suppressHighlighted(true);
  }, "Suppress Highlighted", [this]() { suppressHighlighted(true); }});
  items.push_back({toolbarButtonWidth(suppress_clear_text_), [this, &clear_label]() {
    ImGui::BeginDisabled(!suppress_clear_enabled_);
    if (ImGui::Button(clear_label.c_str())) suppressHighlighted(false);
    ImGui::EndDisabled();
    disabledItemTooltip("Clear suppressed");
  }, suppress_clear_text_, [this]() { suppressHighlighted(false); }, suppress_clear_enabled_});
  const size_t spacer_index = items.size();
  items.push_back({ImGui::CalcTextSize("Suppress Signals").x + CHECKBOX_SIZE + style.ItemInnerSpacing.x, []() {
    bool suppress_defined_signals = settings.suppress_defined_signals;
    if (checkBox("Suppress Signals", &suppress_defined_signals)) can->suppressDefinedSignals(suppress_defined_signals);
    ImGui::SetItemTooltip("Suppress defined signals");
  }});

  const float reserved = iconButtonWidth() + style.ItemSpacing.x;
  drawToolbar(items, spacer_index, std::max(0.0f, ImGui::GetContentRegionAvail().x - reserved));
  ImGui::SameLine();
  if (iconButton("view_btn", icon::THREE_DOTS, "View...")) ImGui::OpenPopup("menu");
}

void MessagesWidget::updateTitle() {
  auto stats = std::accumulate(
      list_.items.begin(), list_.items.end(), std::pair<size_t, size_t>(),
      [](const auto &pair, const auto &item) {
        auto m = dbc()->msg(item.id);
        return m ? std::make_pair(pair.first + 1, pair.second + m->sigs.size()) : pair;
      });
  char buf[128];
  snprintf(buf, sizeof(buf), "%zu Messages (%zu DBC Messages, %zu Signals)", list_.items.size(), stats.first, stats.second);
  title_ = buf;
}

void MessagesWidget::selectMessage(const MessageId &msg_id) {
  auto it = std::find_if(list_.items.cbegin(), list_.items.cend(), [&msg_id](auto &item) { return item.id == msg_id; });
  if (it != list_.items.cend()) setCurrentRow(std::distance(list_.items.cbegin(), it));
}

void MessagesWidget::setCurrentRow(int row) {
  if (row < 0 || row >= (int)list_.items.size()) return;
  current_row_ = row;
  scroll_to_current_ = true;
  const auto &id = list_.items[row].id;
  if (!current_msg_id_ || id != *current_msg_id_) {
    current_msg_id_ = id;
    msgSelectionChanged(*current_msg_id_);
  }
}

void MessagesWidget::suppressHighlighted(bool from_suppress_add) {
  int n = from_suppress_add ? can->suppressHighlighted() : (can->clearSuppressed(), 0);
  suppress_clear_text_ = n > 0 ? "Clear (" + std::to_string(n) + ")" : "Clear";
  suppress_clear_enabled_ = n > 0;
}

void MessagesWidget::drawContextMenu() {
  if (!ImGui::BeginPopup("menu")) return;
  for (int i = 0; i < MessageList::COLUMN_COUNT; ++i) {
    const int column = display_order_[i];
    // can't hide the name column
    if (ImGui::MenuItem(COLUMN_TITLES[column], nullptr, !hidden_[column], column > 0)) {
      pending_hidden_.emplace_back(column, !hidden_[column]);
    }
  }
  ImGui::Separator();
  if (ImGui::MenuItem("Multiline Bytes", nullptr, settings.multiple_lines_hex)) {
    setMultiLineBytes(!settings.multiple_lines_hex);
  }
  if (ImGui::MenuItem("Show Inactive Messages", nullptr, list_.show_inactive_messages)) {
    list_.showInactiveMessages(!list_.show_inactive_messages);
  }
  ImGui::EndPopup();
}

void MessagesWidget::setMultiLineBytes(bool multi) {
  settings.multiple_lines_hex = multi;
  updateBytesSectionSize();
}

void MessagesWidget::updateBytesSectionSize() {
  int max_bytes = 8;
  if (!settings.multiple_lines_hex) {
    for (const auto &[_, m] : can->lastMessages()) {
      max_bytes = std::max<int>(max_bytes, m.dat.size());
    }
  }
  bytes_section_bytes_ = max_bytes;
}

void MessagesWidget::draw() {
  drawToolBar();
  drawTable();
  if (std::exchange(header_menu_requested_, false)) ImGui::OpenPopup("menu");  // at the mouse position
  drawContextMenu();
}

void MessagesWidget::handleKeys() {
  if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) || ImGui::IsAnyItemActive()) return;
  const int last = (int)list_.items.size() - 1;
  if (last < 0) return;
  if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && current_row_ > 0) {
    setCurrentRow(current_row_ - 1);
  } else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) && current_row_ < last) {
    setCurrentRow(current_row_ + 1);
  } else if (ImGui::IsKeyPressed(ImGuiKey_Home)) {
    setCurrentRow(0);
  } else if (ImGui::IsKeyPressed(ImGuiKey_End)) {
    setCurrentRow(last);
  } else if (ImGui::IsKeyPressed(ImGuiKey_PageUp)) {
    setCurrentRow(std::max(current_row_ - visible_rows_, 0));
  } else if (ImGui::IsKeyPressed(ImGuiKey_PageDown)) {
    setCurrentRow(std::min(current_row_ + visible_rows_, last));
  }
}

void MessagesWidget::drawTable() {
  handleKeys();
  const bool multiple_lines = settings.multiple_lines_hex;

  const ImGuiTableFlags flags = ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
                                ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Borders |
                                ImGuiTableFlags_Hideable;
  // with ScrollX a stretch column needs an explicit inner width
  const float bytes_width = bytesCellSize(bytes_section_bytes_, multiple_lines).x;
  const float avail_width = ImGui::GetContentRegionAvail().x - (has_scrollbar_y_ ? ImGui::GetStyle().ScrollbarSize : 0);
  const float inner_width = std::max(avail_width, fixed_columns_width_ + bytes_width);
  if (!ImGui::BeginTable("messages", MessageList::COLUMN_COUNT, flags, ImVec2(0, 0), inner_width)) return;

  // no frozen column: only the header and the filter row stay put
  ImGui::TableSetupScrollFreeze(0, 2);
  // the widget shows its own header menu; imgui's default one is never drawn, so clear the flag
  // TableHeader() leaves behind (it would keep the column header highlighted forever)
  ImGuiTable *table = ImGui::GetCurrentTable();
  table->DisableDefaultContextMenu = true;
  table->IsContextPopupOpen = false;
  for (int i = 0; i < MessageList::COLUMN_COUNT; ++i) {
    // with the flipped direction the first click on a section sorts ascending
    ImGuiTableColumnFlags column_flags = ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending;
    float width = DEFAULT_SECTION_SIZE;
    if (i == MessageList::NAME) {
      column_flags |= ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_NoHide;
    } else if (i == MessageList::DATA) {
      column_flags = ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_NoResize;
      width = 0;
    }
    ImGui::TableSetupColumn(COLUMN_TITLES[i], column_flags, width);
  }
  for (const auto &[column, hide] : pending_hidden_) ImGui::TableSetColumnEnabled(column, !hide);
  pending_hidden_.clear();

  if (ImGuiTableSortSpecs *specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsDirty) {
    if (specs->SpecsCount > 0) list_.sort(specs->Specs[0].ColumnIndex, flipSortDirection(specs->Specs[0].SortDirection));
    specs->SpecsDirty = false;
    if (current_row_ >= 0) scroll_to_current_ = true;  // keep the current row visible
  }

  drawHeader();

  const int rows = list_.items.size();
  if (!multiple_lines) {
    ImGuiListClipper clipper;
    clipper.Begin(rows);
    if (scroll_to_current_ && current_row_ >= 0) clipper.IncludeItemByIndex(current_row_);
    while (clipper.Step()) {
      for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) drawRow(row);
    }
  } else {
    // non-uniform row heights: no clipper
    for (int row = 0; row < rows; ++row) drawRow(row);
  }

  // for the next frame: the stretch inner width and the page size of handleKeys
  const ImGuiTableColumn &data_column = table->Columns[MessageList::DATA];
  fixed_columns_width_ = data_column.IsEnabled ? table->ColumnsGivenWidth - data_column.WidthGiven : 0;
  has_scrollbar_y_ = table->InnerWindow->ScrollbarY;
  visible_rows_ = std::max(1, (int)(table->InnerWindow->InnerRect.GetHeight() / bytesCellSize(0, multiple_lines).y) - 2);
  ImGui::EndTable();
}

void MessagesWidget::drawHeader() {
  if (tableHeadersRow() >= 0) header_menu_requested_ = true;
  // record the visual order and the visibility of the sections for the menu
  ImGuiTable *table = ImGui::GetCurrentTable();
  for (int i = 0; i < MessageList::COLUMN_COUNT; i++) {
    display_order_[i] = table->DisplayOrderToIndex[i];
    hidden_[i] = !(ImGui::TableGetColumnFlags(i) & ImGuiTableColumnFlags_IsEnabled);
  }

  // the filter editors under the header
  ImGui::TableNextRow();
  for (int i = 0; i < MessageList::COLUMN_COUNT; i++) {
    if (!ImGui::TableSetColumnIndex(i)) continue;
    ImGui::PushID(i);
    ImGui::SetNextItemWidth(-FLT_MIN);
    const std::string placeholder = std::string("Filter ") + COLUMN_TITLES[i];
    if (clearableInput("##filter", &filters_[i], placeholder.c_str())) {
      std::map<int, std::string> filters;
      for (int c = 0; c < MessageList::COLUMN_COUNT; ++c) {
        if (!filters_[c].empty()) filters[c] = filters_[c];
      }
      list_.setFilters(filters);
    }
    ImGui::PopID();
  }
}

void MessagesWidget::drawRow(int row) {
  const auto &item = list_.items[row];
  const bool selected = row == current_row_;
  const bool inactive = !can->isMessageActive(item.id);
  const auto &m = can->lastMessage(item.id);
  const bool seen = item.id.source != INVALID_SOURCE;
  const bool multiple_lines = settings.multiple_lines_hex;
  const float row_height = bytesCellSize(seen ? m.dat.size() : 0, multiple_lines).y - ImGui::GetStyle().CellPadding.y * 2;
  ImGui::TableNextRow();
  ImGui::PushID(row);

  bool row_item_submitted = false;
  for (int column = 0; column < MessageList::COLUMN_COUNT; ++column) {
    if (!ImGui::TableSetColumnIndex(column)) continue;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    const ImRect rect(pos, ImVec2(pos.x + width, pos.y + row_height));

    const bool row_item = !row_item_submitted;
    if (row_item) {
      // the row selection spans all columns; submit it in the first visible column so that it is not
      // clipped away when the table is scrolled horizontally
      row_item_submitted = true;
      // rows select on press
      if (viewSelectable("##row", selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_SelectOnClick, ImVec2(0, row_height))) {
        setCurrentRow(row);
      }
      if (selected && scroll_to_current_) {
        // only scroll when the row is outside the viewport, and only far enough
        const ImGuiWindow *inner = ImGui::GetCurrentTable()->InnerWindow;
        const float view_top = inner->InnerClipRect.Min.y + inner->DecoInnerSizeY1;
        const float view_bottom = inner->InnerClipRect.Max.y;
        if (ImGui::GetItemRectMin().y < view_top) {
          ImGui::SetScrollHereY(0.0f);
        } else if (ImGui::GetItemRectMax().y > view_bottom) {
          ImGui::SetScrollHereY(1.0f);
        }
        scroll_to_current_ = false;
      }
      // the tooltip belongs to the name, so only show it while the mouse is over the Name column
      const ImGuiTableColumn &name_col = ImGui::GetCurrentTable()->Columns[MessageList::NAME];
      const float mouse_x = ImGui::GetIO().MousePos.x;
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip) && mouse_x >= name_col.MinX && mouse_x < name_col.MaxX) {
        auto msg = dbc()->msg(item.id);
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(item.name.c_str());
        if (msg && !msg->comment.empty()) ImGui::TextDisabled("%s", msg->comment.c_str());
        ImGui::EndTooltip();
      }
    }

    if (column == MessageList::DATA && seen) {
      drawBytesCell(ImGui::GetWindowDrawList(), rect, m.dat, &m.colors, selected, inactive, multiple_lines);
    } else {
      drawTextCell(ImGui::GetWindowDrawList(), rect, cellText(item, column), selected, inactive);
    }
    // the Selectable already sized its cell
    if (!row_item) ImGui::Dummy(ImVec2(width, row_height));
  }

  ImGui::PopID();
}
