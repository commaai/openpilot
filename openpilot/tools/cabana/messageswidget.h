#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include <QAbstractTableModel>
#include <QHeaderView>
#include <QLineEdit>
#include <QMenu>
#include <QTreeView>
#include <QWheelEvent>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

class MessageListModel : public QAbstractTableModel {
Q_OBJECT

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

  MessageListModel(QObject *parent) : QAbstractTableModel(parent) {}
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return Column::DATA + 1; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return items_.size(); }
  void sort(int column, Qt::SortOrder order = Qt::AscendingOrder) override;
  void setFilterStrings(const std::map<int, QString> &filters);
  void showInactiveMessages(bool show);
  void msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids);
  bool filterAndSort();
  void dbcModified();

  struct Item {
    MessageId id;
    QString name;
    QString node;
    bool operator==(const Item &other) const {
      return id == other.id && name == other.name && node == other.node;
    }
  };
  std::vector<Item> items_;
  bool show_inactive_messages = true;

private:
  void sortItems(std::vector<MessageListModel::Item> &items);
  bool match(const MessageListModel::Item &id);

  std::map<int, QString> filters_;
  std::set<MessageId> dbc_messages_;
  int sort_column = 0;
  Qt::SortOrder sort_order = Qt::AscendingOrder;
  int sort_threshold_ = 0;
};

class MessageView : public QTreeView {
  Q_OBJECT
public:
  MessageView(QWidget *parent) : QTreeView(parent) {}
  void updateBytesSectionSize();
  void setModel(QAbstractItemModel *model) override;

protected:
  void drawRow(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  void drawBranches(QPainter *painter, const QRect &rect, const QModelIndex &index) const override {}
  void wheelEvent(QWheelEvent *event) override;
};

class MessageViewHeader : public QHeaderView {
  // https://stackoverflow.com/a/44346317
  Q_OBJECT
public:
  MessageViewHeader(QWidget *parent);
  void updateHeaderPositions();
  void updateGeometries() override;
  QSize sizeHint() const override;
  void updateFilters();

  std::vector<QLineEdit *> editors;
};

class MessagesWidget : public QWidget {
  Q_OBJECT

public:
  MessagesWidget(QWidget *parent);
  void selectMessage(const MessageId &message_id);
  std::vector<uint8_t> saveHeaderState() const {
    const auto state = view->header()->saveState();
    return {state.begin(), state.end()};
  }
  bool restoreHeaderState(const std::vector<uint8_t> &state) const {
    return view->header()->restoreState({(const char *)state.data(), (int)state.size()});
  }
  void suppressHighlighted();

signals:
  void msgSelectionChanged(const MessageId &message_id);
  void titleChanged(const QString &title);

protected:
  QWidget *createToolBar();
  void headerContextMenuEvent(const QPoint &pos);
  void menuAboutToShow();
  void setMultiLineBytes(bool multi);
  void updateTitle();

  MessageView *view;
  MessageViewHeader *header;
  MessageBytesDelegate *delegate;
  std::optional<MessageId> current_msg_id;
  MessageListModel *model;
  QPushButton *suppress_add;
  QPushButton *suppress_clear;
  QMenu *menu;
};
