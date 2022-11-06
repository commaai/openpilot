#pragma once

#include <QAbstractTableModel>
#include <QTableView>

#include "tools/cabana/canmessages.h"
class MessageListModel : public QAbstractTableModel {
Q_OBJECT

public:
  MessageListModel(QObject *parent) : QAbstractTableModel(parent) {}
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return 5; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return msgs.size(); }
  void sort(int column, Qt::SortOrder order = Qt::AscendingOrder) override;
  void updateState(bool sort = false);
  void setFilterString(const QString &string);

private:
  bool updateMessages(bool sort);

  struct Message {
    QString id, name;
  };
  std::vector<std::unique_ptr<Message>> msgs;
  QString filter_str;
  int sort_column = 0;
  Qt::SortOrder sort_order = Qt::AscendingOrder;
};

class MessagesWidget : public QWidget {
  Q_OBJECT

public:
  MessagesWidget(QWidget *parent);

signals:
  void msgSelectionChanged(const QString &message_id);

protected:
  QTableView *table_widget;
  MessageListModel *model;
};
