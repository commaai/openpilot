#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "imgui.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/widgets/messagebytes.h"

class MessageListModel {
public:
  enum Column {
    NAME = 0,
    SOURCE,
    ADDRESS,
    NODE,
    FREQ,
    COUNT,
    DATA,
  };

  MessageListModel();
  std::string headerData(int section) const;
  int columnCount() const { return Column::DATA + 1; }
  std::string data(int row, int column) const;  // Qt::DisplayRole
  std::string toolTip(int row, int column) const;  // Qt::ToolTipRole; name and comment separated by '\n'
  int rowCount() const { return items_.size(); }
  void sort(int column, ImGuiSortDirection order = ImGuiSortDirection_Ascending);
  void setFilterStrings(const std::map<int, std::string> &filters);
  void showInactiveMessages(bool show);
  void msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids);
  bool filterAndSort();
  void dbcModified();

  struct Item {
    MessageId id;
    std::string name;
    std::string node;
    bool operator==(const Item &other) const {
      return id == other.id && name == other.name && node == other.node;
    }
    bool operator!=(const Item &other) const { return !(*this == other); }
  };
  std::vector<Item> items_;
  bool show_inactive_messages = true;
  Observable<> modelReset;  // beginResetModel/endResetModel

private:
  void sortItems(std::vector<MessageListModel::Item> &items);
  bool match(const MessageListModel::Item &id);

  std::map<int, std::string> filters_;
  std::set<MessageId> dbc_messages_;
  int sort_column = 0;
  ImGuiSortDirection sort_order = ImGuiSortDirection_Ascending;
  int sort_threshold_ = 0;
  Connections connections_;
};

class MessageViewHeader;

class MessageView {
public:
  MessageView() {}
  void updateBytesSectionSize();
  void setModel(MessageListModel *model);
  void setHeader(MessageViewHeader *header) { header_ = header; }
  void setItemDelegate(MessageBytesDelegate *delegate) { delegate_ = delegate; }
  MessageBytesDelegate *itemDelegate() const { return delegate_; }
  MessageListModel *model() const { return model_; }
  MessageViewHeader *header() const { return header_; }
  void setCurrentIndex(int row);  // QAbstractItemView::setCurrentIndex, scrolls to the row
  int currentIndex() const { return current_row_; }
  void draw();  // the table: header, filter row, rows

  Observable<int, int> currentChanged;  // QItemSelectionModel::currentChanged(current, previous)

protected:
  void drawRow(int row);
  void keyPressEvent();  // up/down move the current row
  // wheelEvent: shift+wheel scrolls horizontally, imgui does this by default

  MessageListModel *model_ = nullptr;
  MessageViewHeader *header_ = nullptr;
  MessageBytesDelegate *delegate_ = nullptr;
  int current_row_ = -1;
  bool scroll_to_current_ = false;
  int bytes_section_bytes_ = 8;  // the DATA section width is applied inside the table
  bool bytes_section_dirty_ = true;
  Connections connections_;
};

class MessageViewHeader {
  // https://stackoverflow.com/a/44346317
public:
  MessageViewHeader();
  void setModel(MessageListModel *model) { model_ = model; }
  MessageListModel *model() const { return model_; }
  void updateHeaderPositions();
  void updateGeometries();
  void updateFilters();
  void draw();  // the header row and the filter editors row, inside the table

  // QHeaderView
  int count() const { return (int)editors.size(); }
  int logicalIndex(int visual_index) const { return display_order_[visual_index]; }
  bool isSectionHidden(int logical_index) const { return hidden_[logical_index]; }
  void setSectionHidden(int logical_index, bool hide) { hidden_[logical_index] = hide; }
  bool customContextMenuRequested = false;  // set for one frame by a right click on the header

  std::vector<std::string> editors;

private:
  MessageListModel *model_ = nullptr;
  std::vector<std::string> placeholders_;
  std::vector<bool> hidden_;
  std::vector<int> display_order_;
};

class MessagesWidget {
public:
  MessagesWidget();
  void draw();  // content only; MainWindow does ImGui::Begin/End with the dock title
  void selectMessage(const MessageId &message_id);
  std::vector<uint8_t> saveHeaderState() const {
    // TODO: Qt byte-array header state is out of scope for the port
    return {};
  }
  bool restoreHeaderState(const std::vector<uint8_t> &state) const {
    // TODO: Qt byte-array header state is out of scope for the port
    return false;
  }
  void suppressHighlighted();
  const std::string &title() const { return title_; }
  std::string whatsThis() const;

  Observable<const MessageId &> msgSelectionChanged;
  Observable<const std::string &> titleChanged;

protected:
  void createToolBar();  // drawn each frame
  void headerContextMenuEvent();
  void menuAboutToShow();
  void setMultiLineBytes(bool multi);
  void updateTitle();
  void suppressHighlighted(bool from_suppress_add);  // sender() == suppress_add

  MessageView *view;
  MessageViewHeader *header;
  MessageBytesDelegate *delegate;
  std::optional<MessageId> current_msg_id;
  MessageListModel *model;
  std::string suppress_clear_text;
  bool suppress_clear_enabled = false;
  std::string title_ = "MESSAGES";

private:
  MessageView view_;
  MessageViewHeader header_;
  MessageBytesDelegate delegate_;
  MessageListModel model_;
  Connections connections_;
};
