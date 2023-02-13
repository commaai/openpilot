#pragma once

#include <QAbstractTableModel>
#include <QHeaderView>
#include <QLineEdit>
#include <QSet>
#include <QStyledItemDelegate>
#include <QTableView>

#include "tools/cabana/streams/abstractstream.h"

class MessageListModel : public QAbstractTableModel {
Q_OBJECT

public:
  MessageListModel(QObject *parent) : QAbstractTableModel(parent) {}
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return 5; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return msgs.size(); }
  void sort(int column, Qt::SortOrder order = Qt::AscendingOrder) override;
  void setFilterString(const QString &string);
  void msgsReceived(const QHash<QString, CanData> *new_msgs = nullptr);
  void sortMessages();
  void suppress();
  void clearSuppress();
  void reset();
  QStringList msgs;
  QSet<std::pair<QString, int>> suppressed_bytes;

private:
  QString filter_str;
  int sort_column = 0;
  Qt::SortOrder sort_order = Qt::AscendingOrder;
};

class MessagesWidget : public QWidget {
  Q_OBJECT

public:
  MessagesWidget(QWidget *parent);
  void selectMessage(const QString &message_id);
  QByteArray saveHeaderState() const { return table_widget->horizontalHeader()->saveState(); }
  bool restoreHeaderState(const QByteArray &state) const { return table_widget->horizontalHeader()->restoreState(state); }
  void updateSuppressedButtons();
  void reset();

signals:
  void msgSelectionChanged(const QString &message_id);

protected:
  QTableView *table_widget;
  QString current_msg_id;
  QLineEdit *filter;
  MessageListModel *model;
  QPushButton *suppress_add;
  QPushButton *suppress_clear;

};
