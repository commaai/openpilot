#pragma once

#include <QScrollArea>
#include <QStyledItemDelegate>
#include <QTableView>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/historylog.h"
#include "tools/cabana/signaledit.h"

class BinaryItemDelegate : public QStyledItemDelegate {
  Q_OBJECT

public:
  BinaryItemDelegate(QObject *parent) : QStyledItemDelegate(parent) {}
  void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const override {
    QSize sz = QStyledItemDelegate::sizeHint(option, index);
    return {sz.width(), 40};
  }
};

class BinaryViewModel : public QAbstractTableModel {
Q_OBJECT

public:
  BinaryViewModel(QObject *parent) : QAbstractTableModel(parent) {}
  void setMessage(const QString &message_id);
  void updateState();
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return column_count; }
  QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return row_count; }

struct Item {
    QColor bg_color = QColor(Qt::white);
    bool is_msb = false;
    bool is_lsb = false;
    QString val = "0";
  };

private:
  QString msg_id;
  int row_count = 0;
  const int column_count = 9;
  std::vector<Item> items;
};

class BinarySelectionModel : public QItemSelectionModel {
public:
  BinarySelectionModel(QAbstractItemModel *model = nullptr) : QItemSelectionModel(model) {}
  void select(const QItemSelection &selection, QItemSelectionModel::SelectionFlags command) override;
};

class BinaryView : public QTableView {
  Q_OBJECT
public:
  BinaryView(QWidget *parent = nullptr);
  void mouseReleaseEvent(QMouseEvent *event) override;
  void setMessage(const QString &message_id);
  void updateState();
signals:
  void cellsSelected(int start_bit, int size);

private:
  QString msg_id;
  BinaryViewModel *model;
};

class EditMessageDialog : public QDialog {
  Q_OBJECT

public:
  EditMessageDialog(const QString &msg_id, const QString &title, int size, QWidget *parent);

  QLineEdit *name_edit;
  QSpinBox *size_spin;
};

class ScrollArea : public QScrollArea {
  Q_OBJECT

public:
  ScrollArea(QWidget *parent) : QScrollArea(parent) {}
  bool eventFilter(QObject *obj, QEvent *ev) override;
  void setWidget(QWidget *w);
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  DetailWidget(QWidget *parent);
  void setMessage(const QString &message_id);
  void dbcMsgChanged();

signals:
  void showChart(const QString &msg_id, const Signal *sig);
  void removeChart(const Signal *sig);

private:
  void addSignal(int start_bit, int size);
  void saveSignal();
  void removeSignal();
  void editMsg();
  void showForm();
  void updateState();

  QString msg_id;
  QLabel *name_label, *time_label;
  QPushButton *edit_btn;
  QWidget *signals_container;
  HistoryLog *history_log;
  BinaryView *binary_view;
  ScrollArea *scroll;
};
