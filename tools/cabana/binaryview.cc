#include "tools/cabana/binaryview.h"

#include <QApplication>
#include <QHeaderView>
#include <QPainter>

#include "tools/cabana/canmessages.h"

// BinaryView

const int CELL_HEIGHT = 35;

BinaryView::BinaryView(QWidget *parent) : QTableView(parent) {
  model = new BinaryViewModel(this);
  setModel(model);
  setItemDelegate(new BinaryItemDelegate(this));
  horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  horizontalHeader()->hide();
  verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  // replace selection model
  auto old_model = selectionModel();
  setSelectionModel(new BinarySelectionModel(model));
  delete old_model;

  QObject::connect(model, &QAbstractItemModel::modelReset, [this]() {
    setFixedHeight((CELL_HEIGHT + 1) * std::min(model->rowCount(), 8) + 2);
  });
}

void BinaryView::mouseReleaseEvent(QMouseEvent *event) {
  QTableView::mouseReleaseEvent(event);

  if (auto indexes = selectedIndexes(); !indexes.isEmpty()) {
    int start_bit = indexes.first().row() * 8 + indexes.first().column();
    int size = indexes.back().row() * 8 + indexes.back().column() - start_bit + 1;
    emit cellsSelected(start_bit, size);
  }
}

void BinaryView::setMessage(const QString &message_id) {
  msg_id = message_id;
  model->setMessage(message_id);
  resizeRowsToContents();
  clearSelection();
  updateState();
}

void BinaryView::updateState() {
  model->updateState();
}

// BinaryViewModel

void BinaryViewModel::setMessage(const QString &message_id) {
  msg_id = message_id;

  beginResetModel();
  items.clear();
  row_count = 0;

  dbc_msg = dbc()->msg(msg_id);
  if (dbc_msg) {
    row_count = dbc_msg->size;
    items.resize(row_count * column_count);
    for (int i = 0; i < dbc_msg->sigs.size(); ++i) {
      const auto &sig = dbc_msg->sigs[i];
      int start = sig.is_little_endian ? sig.start_bit : bigEndianBitIndex(sig.start_bit);
      for (int j = start; j <= start + sig.size - 1; ++j) {
        int idx = column_count * (j / (column_count - 1)) + j % (column_count - 1);
        if (j == sig.msb) {
          items[idx].is_msb = true;
        } else if (j == sig.lsb) {
          items[idx].is_lsb = true;
        }
        items[idx].bg_color = QColor(getColor(i));
      }
    }
  }

  endResetModel();
}

QModelIndex BinaryViewModel::index(int row, int column, const QModelIndex &parent) const {
  return createIndex(row, column, (void *)&items[row * column_count + column]);
}

Qt::ItemFlags BinaryViewModel::flags(const QModelIndex &index) const {
  return (index.column() == column_count - 1) ? Qt::ItemIsEnabled : Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

void BinaryViewModel::updateState() {
  char hex[3] = {'\0'};
  const auto &binary = can->lastMessage(msg_id).dat;
  // data size may changed.
  if (!dbc_msg && binary.size() != row_count) {
    beginResetModel();
    row_count = binary.size();
    items.clear();
    items.resize(row_count * column_count);
    endResetModel();
  }

  for (int i = 0; i < std::min(binary.size(), row_count); ++i) {
    for (int j = 0; j < column_count - 1; ++j) {
      items[i * column_count + j].val = QChar((binary[i] >> (7 - j)) & 1 ? '1' : '0');
    }
    hex[0] = toHex(binary[i] >> 4);
    hex[1] = toHex(binary[i] & 0xf);
    items[i * column_count + 8].val = hex;
  }

  emit dataChanged(index(0, 0), index(row_count - 1, column_count - 1));
}

QVariant BinaryViewModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Vertical) {
    switch (role) {
      case Qt::DisplayRole: return section + 1;
      case Qt::SizeHintRole: return QSize(30, CELL_HEIGHT);
      case Qt::TextAlignmentRole: return Qt::AlignCenter;
    }
  }
  return {};
}

// BinarySelectionModel

void BinarySelectionModel::select(const QItemSelection &selection, QItemSelectionModel::SelectionFlags command) {
  QItemSelection new_selection = selection;
  if (auto indexes = selection.indexes(); !indexes.isEmpty()) {
    auto [begin_idx, end_idx] = (QModelIndex[]){indexes.first(), indexes.back()};
    for (int row = begin_idx.row(); row <= end_idx.row(); ++row) {
      int left_col = (row == begin_idx.row()) ? begin_idx.column() : 0;
      int right_col = (row == end_idx.row()) ? end_idx.column() : 7;
      new_selection.merge({model()->index(row, left_col), model()->index(row, right_col)}, command);
    }
  }
  QItemSelectionModel::select(new_selection, command);
}

// BinaryItemDelegate

BinaryItemDelegate::BinaryItemDelegate(QObject *parent) : QStyledItemDelegate(parent) {
  // cache fonts and color
  small_font.setPointSize(7);
  bold_font.setBold(true);
  highlight_color = QApplication::style()->standardPalette().color(QPalette::Active, QPalette::Highlight);
}

QSize BinaryItemDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const {
  QSize sz = QStyledItemDelegate::sizeHint(option, index);
  return {sz.width(), CELL_HEIGHT};
}

void BinaryItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto item = (const BinaryViewModel::Item *)index.internalPointer();
  painter->save();

  painter->fillRect(option.rect, option.state & QStyle::State_Selected ? highlight_color : item->bg_color);
  if (index.column() == 8) {
    painter->setFont(bold_font);
  }
  painter->drawText(option.rect, Qt::AlignCenter, item->val);
  if (item->is_msb || item->is_lsb) {
    painter->setFont(small_font);
    painter->drawText(option.rect, Qt::AlignHCenter | Qt::AlignBottom, item->is_msb ? "MSB" : "LSB");
  }

  painter->restore();
}
