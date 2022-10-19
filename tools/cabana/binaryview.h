#pragma once

#include <QStyledItemDelegate>
#include <QTableView>

#include "tools/cabana/dbcmanager.h"

class BinaryItemDelegate : public QStyledItemDelegate {
  Q_OBJECT

public:
  BinaryItemDelegate(QObject *parent);
  void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const override;

private:
  QFont small_font, hex_font;
  QColor highlight_color;
};

class BinaryViewModel : public QAbstractTableModel {
  Q_OBJECT

public:
  BinaryViewModel(QObject *parent) : QAbstractTableModel(parent) {}
  void setMessage(const QString &message_id);
  void updateState();
  Qt::ItemFlags flags(const QModelIndex &index) const;
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const { return {}; }
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return row_count; }
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return column_count; }

  struct Item {
    QColor bg_color = QColor(Qt::white);
    bool is_msb = false;
    bool is_lsb = false;
    QString val = "0";
    const Signal *sig = nullptr;
  };

private:
  QString msg_id;
  const Msg *dbc_msg;
  int row_count = 0;
  const int column_count = 9;
  std::vector<Item> items;
};

// the default QItemSelectionModel does not support our selection mode.
class BinarySelectionModel : public QItemSelectionModel {
 public:
  BinarySelectionModel(QAbstractItemModel *model = nullptr) : QItemSelectionModel(model) {}
  void select(const QItemSelection &selection, QItemSelectionModel::SelectionFlags command) override;
};

class BinaryView : public QTableView {
  Q_OBJECT

public:
  BinaryView(QWidget *parent = nullptr);
  void setMessage(const QString &message_id);
  void updateState();
  void highlight(const Signal *sig);
  const Signal *hoveredSignal() const { return hovered_sig; }

signals:
  void cellsSelected(int start_bit, int size);
  void signalHovered(const Signal *sig);

private:
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void leaveEvent(QEvent *event) override;

  QString msg_id;
  BinaryViewModel *model;
  const Signal *hovered_sig = nullptr;
};
