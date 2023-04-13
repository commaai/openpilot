#pragma once

#include <QAbstractTableModel>
#include <QCheckBox>
#include <QHeaderView>
#include <QLineEdit>
#include <QSet>
#include <QTreeView>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

class MessageListModel : public QAbstractTableModel {
Q_OBJECT

public:
  MessageListModel(QObject *parent) : QAbstractTableModel(parent) {}
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return 6; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return msgs.size(); }
  void sort(int column, Qt::SortOrder order = Qt::AscendingOrder) override;
  void setFilterString(const QString &string);
  void msgsReceived(const QHash<MessageId, CanData> *new_msgs = nullptr);
  void sortMessages();
  void suppress();
  void clearSuppress();
  void reset();
  QList<MessageId> msgs;
  QSet<std::pair<MessageId, int>> suppressed_bytes;

private:
  QString filter_str;
  int sort_column = 0;
  Qt::SortOrder sort_order = Qt::AscendingOrder;
};

class MessageView : public QTreeView {
  Q_OBJECT
public:
  MessageView(QWidget *parent) : QTreeView(parent) {}
  void drawRow(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
};

class MessagesWidget : public QWidget {
  Q_OBJECT

public:
  MessagesWidget(QWidget *parent);
  void selectMessage(const MessageId &message_id);
  QByteArray saveHeaderState() const { return view->header()->saveState(); }
  bool restoreHeaderState(const QByteArray &state) const { return view->header()->restoreState(state); }
  void updateSuppressedButtons();
  void reset();

signals:
  void msgSelectionChanged(const MessageId &message_id);

protected:
  MessageView *view;
  std::optional<MessageId> current_msg_id;
  QLineEdit *filter;
  QCheckBox *multiple_lines_bytes;
  MessageListModel *model;
  QPushButton *suppress_add;
  QPushButton *suppress_clear;
};
