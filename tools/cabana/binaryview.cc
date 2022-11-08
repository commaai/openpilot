#include "tools/cabana/binaryview.h"

#include <QApplication>
#include <QFontDatabase>
#include <QHeaderView>
#include <QMouseEvent>
#include <QPainter>
#include <QToolTip>

#include "tools/cabana/canmessages.h"

// BinaryView

const int CELL_HEIGHT = 26;

inline int get_bit_index(const QModelIndex &index, bool little_endian) {
  return little_endian ? index.row() * 8 + 7 - index.column()
                       : index.row() * 8 + index.column();
}

BinaryView::BinaryView(QWidget *parent) : QTableView(parent) {
  model = new BinaryViewModel(this);
  setModel(model);
  delegate = new BinaryItemDelegate(this);
  setItemDelegate(delegate);
  horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  horizontalHeader()->hide();
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
  setFrameShape(QFrame::NoFrame);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
  setMouseTracking(true);
}

void BinaryView::highlight(const Signal *sig) {
  if (sig != hovered_sig) {
    hovered_sig = sig;
    model->dataChanged(model->index(0, 0), model->index(model->rowCount() - 1, model->columnCount() - 1));
    emit signalHovered(hovered_sig);
  }
}

void BinaryView::setSelection(const QRect &rect, QItemSelectionModel::SelectionFlags flags) {
  auto index = indexAt(viewport()->mapFromGlobal(QCursor::pos()));
  if (!anchor_index.isValid() || !index.isValid())
    return;

  auto tl = std::min(anchor_index, index);
  auto br = std::max(anchor_index, index);
  QItemSelection selection;
  if ((resize_sig && resize_sig->is_little_endian) || (!resize_sig && index < anchor_index)) {
    // little_endian selection
    if (tl.row() == br.row()) {
      selection.merge({model->index(tl.row(), tl.column()), model->index(tl.row(), br.column())}, flags);
    } else {
      for (int row = tl.row(); row <= br.row(); ++row) {
        int left_col = (row == br.row()) ? br.column() : 0;
        int right_col = (row == tl.row()) ? tl.column() : 7;
        selection.merge({model->index(row, left_col), model->index(row, right_col)}, flags);
      }
    }
  } else {
    // big endian selection
    for (int row = tl.row(); row <= br.row(); ++row) {
      int left_col = (row == tl.row()) ? tl.column() : 0;
      int right_col = (row == br.row()) ? br.column() : 7;
      selection.merge({model->index(row, left_col), model->index(row, right_col)}, flags);
    }
  }
  selectionModel()->select(selection, flags);
}

void BinaryView::mousePressEvent(QMouseEvent *event) {
  delegate->setSelectionColor(style()->standardPalette().color(QPalette::Active, QPalette::Highlight));
  anchor_index = indexAt(event->pos());
  if (resize_sig = getResizeSignal(); resize_sig != nullptr) {
    auto item = (const BinaryViewModel::Item *)anchor_index.internalPointer();
    delegate->setSelectionColor(item->bg_color);
  }
  QTableView::mousePressEvent(event);
}

void BinaryView::mouseMoveEvent(QMouseEvent *event) {
  if (auto index = indexAt(event->pos()); index.isValid()) {
    auto item = (BinaryViewModel::Item *)index.internalPointer();
    const Signal *sig = item->sigs.isEmpty() ? nullptr : item->sigs.back();
    highlight(sig);
    sig ? QToolTip::showText(event->globalPos(), sig->name.c_str(), this, rect())
        : QToolTip::hideText();
  }
  QTableView::mouseMoveEvent(event);
}

void BinaryView::mouseReleaseEvent(QMouseEvent *event) {
  QTableView::mouseReleaseEvent(event);

  auto release_index = indexAt(event->pos());
  if (release_index.isValid() && anchor_index.isValid()) {
    if (release_index.column() == 8) {
      release_index = model->index(release_index.row(), 7);
    }
    bool little_endian = (resize_sig && resize_sig->is_little_endian) || (!resize_sig && release_index < anchor_index);
    int release_bit_idx = get_bit_index(release_index, little_endian);
    int archor_bit_idx = get_bit_index(anchor_index, little_endian);
    if (resize_sig) {
      auto [sig_from, sig_to] = getSignalRange(resize_sig);
      int start_bit, end_bit;
      if (archor_bit_idx == sig_from) {
        start_bit = std::min(release_bit_idx, sig_to);
        end_bit = std::max(release_bit_idx, sig_to);
        if (start_bit >= sig_from && start_bit <= sig_to) {
          start_bit = std::min(start_bit + 1, sig_to);
        }
      } else {
        start_bit = std::min(release_bit_idx, sig_from);
        end_bit = std::max(release_bit_idx, sig_from);
        if (end_bit >= sig_from && end_bit <= sig_to) {
          end_bit = std::max(end_bit - 1, sig_from);
        }
      }
      emit resizeSignal(resize_sig, start_bit, end_bit - start_bit + 1);
    } else {
      if (archor_bit_idx > release_bit_idx) {
        std::swap(archor_bit_idx, release_bit_idx);
      }
      emit addSignal(archor_bit_idx, release_bit_idx - archor_bit_idx + 1, little_endian);
    }
  }
  clearSelection();
  anchor_index = QModelIndex();
  resize_sig = nullptr;
}

void BinaryView::leaveEvent(QEvent *event) {
  highlight(nullptr);
  QTableView::leaveEvent(event);
}

void BinaryView::setMessage(const QString &message_id) {
  model->setMessage(message_id);
  clearSelection();
  anchor_index = QModelIndex();
  resize_sig = nullptr;
  hovered_sig = nullptr;
  updateState();
}

const Signal *BinaryView::getResizeSignal() const {
  if (anchor_index.isValid()) {
    auto item = (const BinaryViewModel::Item *)anchor_index.internalPointer();
    if (item && item->sigs.size() > 0) {
      int archor_pos = anchor_index.row() * 8 + anchor_index.column();
      for (auto s : item->sigs) {
        auto [sig_from, sig_to] = getSignalRange(s);
        if (s->is_little_endian) {
          sig_from = bigEndianBitIndex(sig_from);
          sig_to = bigEndianBitIndex(sig_to);
        }
        if (archor_pos == sig_from || archor_pos == sig_to)
          return s;
      }
    }
  }
  return nullptr;
}

QSet<const Signal *> BinaryView::getOverlappingSignals() const {
  QSet<const Signal *> overlapping;
  for (int i = 0; i < model->rowCount(); ++i) {
    for (int j = 0; j < model->columnCount() - 1; ++j) {
      auto item = (const BinaryViewModel::Item *)model->index(i, j).internalPointer();
      if (item && item->sigs.size() > 1) {
        for (auto s : item->sigs)
          overlapping.insert(s);
      }
    }
  }
  return overlapping;
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
      auto [start, end] = getSignalRange(&sig);
      for (int j = start; j <= end; ++j) {
        int bit_index = sig.is_little_endian ? bigEndianBitIndex(j) : j;
        int idx = column_count * (bit_index / 8) + bit_index % 8;
        if (idx >= items.size()) {
          qWarning() << "signal " << sig.name.c_str() << "out of bounds.start_bit:" << sig.start_bit << "size:" << sig.size;
          break;
        }
        if (j == start) {
          sig.is_little_endian ? items[idx].is_lsb = true : items[idx].is_msb = true;
        }
        if (j == end) {
          sig.is_little_endian ? items[idx].is_msb = true : items[idx].is_lsb = true;
        }
        items[idx].bg_color = getColor(i);
        items[idx].sigs.push_back(&dbc_msg->sigs[i]);
      }
    }
  } else {
    row_count = can->lastMessage(msg_id).dat.size();
    items.resize(row_count * column_count);
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
  auto prev_items = items;

  const auto &binary = can->lastMessage(msg_id).dat;
  // data size may changed.
  if (!dbc_msg && binary.size() != row_count) {
    beginResetModel();
    row_count = binary.size();
    items.clear();
    items.resize(row_count * column_count);
    endResetModel();
  }

  char hex[3] = {'\0'};
  for (int i = 0; i < std::min(binary.size(), row_count); ++i) {
    for (int j = 0; j < column_count - 1; ++j) {
      items[i * column_count + j].val = QChar((binary[i] >> (7 - j)) & 1 ? '1' : '0');
    }
    hex[0] = toHex(binary[i] >> 4);
    hex[1] = toHex(binary[i] & 0xf);
    items[i * column_count + 8].val = hex;
  }

  for (int i = 0; i < items.size(); ++i) {
    if (i >= prev_items.size() || prev_items[i].val != items[i].val) {
      auto idx = index(i / column_count, i % column_count);
      emit dataChanged(idx, idx);
    }
  }
}

QVariant BinaryViewModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Vertical) {
    switch (role) {
      case Qt::DisplayRole: return section;
      case Qt::SizeHintRole: return QSize(30, CELL_HEIGHT);
      case Qt::TextAlignmentRole: return Qt::AlignCenter;
    }
  }
  return {};
}

// BinaryItemDelegate

BinaryItemDelegate::BinaryItemDelegate(QObject *parent) : QStyledItemDelegate(parent) {
  // cache fonts and color
  small_font.setPointSize(6);
  hex_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
  hex_font.setBold(true);
  selection_color = QApplication::style()->standardPalette().color(QPalette::Active, QPalette::Highlight);
}

QSize BinaryItemDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const {
  QSize sz = QStyledItemDelegate::sizeHint(option, index);
  return {sz.width(), CELL_HEIGHT};
}

void BinaryItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto item = (const BinaryViewModel::Item *)index.internalPointer();
  BinaryView *bin_view = (BinaryView *)parent();
  painter->save();

  bool hover = std::find_if(item->sigs.begin(), item->sigs.end(), [=](auto s) { return s == bin_view->hoveredSignal(); }) !=
               item->sigs.end();
  // background
  QColor bg_color = hover ? hoverColor(item->bg_color) : item->bg_color;
  if (option.state & QStyle::State_Selected) {
    bg_color = selection_color;
  }
  painter->fillRect(option.rect, bg_color);

  // text
  if (index.column() == 8) {  // hex column
    painter->setFont(hex_font);
  } else if (hover) {
    painter->setPen(Qt::white);
  }
  painter->drawText(option.rect, Qt::AlignCenter, item->val);
  if (item->is_msb || item->is_lsb) {
    painter->setFont(small_font);
    painter->drawText(option.rect, Qt::AlignHCenter | Qt::AlignBottom, item->is_msb ? "MSB" : "LSB");
  }

  painter->restore();
}
