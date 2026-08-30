#pragma once

#include <algorithm>
#include <array>
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
  std::string data(int row, int column) const;
  std::string toolTip(int row, int column) const;  // name and comment separated by '\n'
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
  };
  std::vector<Item> items_;
  bool show_inactive_messages = true;
  Observable<> modelReset;

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
  explicit MessageView(MessageViewHeader *header) : header_(header) {}
  void updateBytesSectionSize();
  void setModel(MessageListModel *model);
  void setItemDelegate(MessageBytesDelegate *delegate) { delegate_ = delegate; }
  void setCurrentIndex(int row);  // scrolls to the row
  int currentIndex() const { return current_row_; }
  void draw();  // the table: header, filter row, rows

  Observable<int, int> currentChanged;  // (current, previous)

protected:
  void drawRow(int row);
  void keyPressEvent();  // up/down move the current row

  MessageListModel *model_ = nullptr;
  MessageViewHeader *header_ = nullptr;
  MessageBytesDelegate *delegate_ = nullptr;
  int current_row_ = -1;
  bool scroll_to_current_ = false;
  int bytes_section_bytes_ = 8;  // the minimum width of the stretched DATA section
  float fixed_columns_width_ = 0;  // width of everything but the stretched DATA section
  bool has_scrollbar_y_ = false;
  int visible_rows_ = 1;
  Connections connections_;
};

class MessageViewHeader {
  // https://stackoverflow.com/a/44346317
public:
  MessageViewHeader();
  void setModel(MessageListModel *model) { model_ = model; }
  MessageListModel *model() const { return model_; }
  void updateHeaderPositions();
  void updateFilters();
  void draw();  // the header row and the filter editors row, inside the table

  int count() const { return (int)editors.size(); }
  int logicalIndex(int visual_index) const { return display_order_[visual_index]; }
  bool isSectionHidden(int logical_index) const { return hidden_[logical_index]; }
  void setSectionHidden(int logical_index, bool hide) { hidden_[logical_index] = hide; }
  bool customContextMenuRequested = false;  // set for one frame by a right click on the header

  std::array<std::string, MessageListModel::Column::DATA + 1> editors;

private:
  MessageListModel *model_ = nullptr;
  std::array<bool, MessageListModel::Column::DATA + 1> hidden_ = {};
  std::array<int, MessageListModel::Column::DATA + 1> display_order_ = {};
};

class MessagesWidget {
public:
  MessagesWidget();
  void draw();  // content only; MainWindow does ImGui::Begin/End with the dock title
  void selectMessage(const MessageId &message_id);
  std::vector<uint8_t> saveHeaderState() const {
    // TODO: the persisted header state is not ported yet
    return {};
  }
  bool restoreHeaderState(const std::vector<uint8_t> &state) const {
    // TODO: the persisted header state is not ported yet
    return false;
  }
  void suppressHighlighted(bool from_suppress_add = false);
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

  MessageView view;
  MessageViewHeader header;
  MessageBytesDelegate delegate;
  std::optional<MessageId> current_msg_id;
  MessageListModel model;
  std::string suppress_clear_text;
  bool suppress_clear_enabled = false;
  std::string title_ = "MESSAGES";

private:
  Connections connections_;
};
