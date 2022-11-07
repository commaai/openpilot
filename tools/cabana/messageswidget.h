#pragma once

#include <QAbstractTableModel>
#include <QTableView>
#include <QTimer>

#include "tools/cabana/canmessages.h"
class MessageListModel : public QAbstractTableModel {
Q_OBJECT

public:
  MessageListModel(QObject *parent);
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return 5; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return msgs.size(); }
  void sort(int column, Qt::SortOrder order = Qt::AscendingOrder) override;
  void setFilterString(const QString &string);
  void msgsUpdated(const QHash<QString, CanData> *new_msgs = nullptr);
  void sortMessages();
  QStringList msgs;

private:
  QString filter_str;
  int sort_column = 0;
  Qt::SortOrder sort_order = Qt::AscendingOrder;
  QTimer *sort_timer;
};

class MessagesWidget : public QWidget {
  Q_OBJECT

public:
  MessagesWidget(QWidget *parent);

signals:
  void msgSelectionChanged(const QString &message_id);

protected:
  QTableView *table_widget;
  QString current_msg_id;
  MessageListModel *model;
};
