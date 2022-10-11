#pragma once

#include <QAbstractTableModel>
#include <QLineEdit>
#include <QTableView>

#include "tools/cabana/canmessages.h"

class MessageListModel : public QAbstractTableModel {
Q_OBJECT

public:
  MessageListModel(QObject *parent) : QAbstractTableModel(parent) {}
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return 4; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return messages.size(); }
  void sort(int column, Qt::SortOrder order = Qt::AscendingOrder);
  void setNameFilter(const QString &filter) { name_filter = filter;};
  void updateState();

private:
  QString name_filter;
  int sort_column = 0;
  Qt::SortOrder sort_order = Qt::AscendingOrder;

  struct Data{
    QString name;
    QString id;
    uint64_t count;
  };
  std::vector<std::unique_ptr<Data>> messages;
};

class MessagesWidget : public QWidget {
  Q_OBJECT

public:
  MessagesWidget(QWidget *parent);

public slots:
  void dbcSelectionChanged(const QString &dbc_file);

signals:
  void msgSelectionChanged(const QString &message_id);

protected:
  QLineEdit *filter;
  QTableView *table_widget;
  MessageListModel *model;
};
