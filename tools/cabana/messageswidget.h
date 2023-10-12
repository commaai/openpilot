#pragma once

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include <QAbstractTableModel>
#include <QHeaderView>
#include <QLineEdit>
#include <QMenu>
#include <QTimer>
#include <QToolBar>
#include <QTreeView>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

class MessageListModel : public QAbstractTableModel {
Q_OBJECT

public:
  enum Column {
    NAME = 0,
    SOURCE,
    ADDRESS,
    FREQ,
    COUNT,
    DATA,
  };

  MessageListModel(QObject *parent);
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return Column::DATA + 1; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return msgs.size(); }
  void sort(int column, Qt::SortOrder order = Qt::AscendingOrder) override;
  void setFilterStrings(const QMap<int, QString> &filters);
  void msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids);
  void filterAndSort();
  void dbcModified();
  std::vector<MessageId> msgs;

private:
  void sortMessages(std::vector<MessageId> &new_msgs);
  bool matchMessage(const MessageId &id, const CanData &data, const QMap<int, QString> &filters);

  QMap<int, QString> filter_str;
  std::set<uint32_t> dbc_address;
  int sort_column = 0;
  QTimer sort_timer;
  double last_sort_ts = 0;
  Qt::SortOrder sort_order = Qt::AscendingOrder;
};

class MessageView : public QTreeView {
  Q_OBJECT
public:
  MessageView(QWidget *parent) : QTreeView(parent) {}
  void drawRow(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  void drawBranches(QPainter *painter, const QRect &rect, const QModelIndex &index) const override {}
  void dataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles = QVector<int>()) override;
  void updateBytesSectionSize();
};

class MessageViewHeader : public QHeaderView {
  // https://stackoverflow.com/a/44346317
  Q_OBJECT
public:
  MessageViewHeader(QWidget *parent);
  void updateHeaderPositions();
  void updateGeometries() override;
  QSize sizeHint() const override;

signals:
  void filtersUpdated(const QMap<int, QString> &filters);

private:
  void updateFilters();

  QMap<int, QLineEdit *> editors;
};

class MessagesWidget : public QWidget {
  Q_OBJECT

public:
  MessagesWidget(QWidget *parent);
  QToolBar *createToolBar();
  void selectMessage(const MessageId &message_id);
  QByteArray saveHeaderState() const { return view->header()->saveState(); }
  bool restoreHeaderState(const QByteArray &state) const { return view->header()->restoreState(state); }
  void updateSuppressedButtons(size_t n);
  void updateTitle();

signals:
  void msgSelectionChanged(const MessageId &message_id);
  void titleChanged(const QString &title);

protected:
  void setMultiLineBytes(bool multi);
  void headerContextMenuEvent(const QPoint &pos);
  void menuAboutToShow();

  MessageView *view;
  MessageViewHeader *header;
  MessageBytesDelegate *delegate;
  std::optional<MessageId> current_msg_id;
  MessageListModel *model;
  QAction *suppress_clear;
  QMenu *menu;
};
