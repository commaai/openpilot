#include "tools/cabana/binaryview.h"

#include <QFontDatabase>
#include <QHeaderView>
#include <QMouseEvent>
#include <QPainter>
#include <QScrollBar>
#include <QToolTip>

#include "tools/cabana/commands.h"
#include "tools/cabana/streams/abstractstream.h"

// BinaryView

const int CELL_HEIGHT = 36;
const int VERTICAL_HEADER_WIDTH = 30;

inline int get_bit_index(const QModelIndex &index, bool little_endian) {
  return index.row() * 8 + (little_endian ? 7 - index.column() : index.column());
}

BinaryView::BinaryView(QWidget *parent) : QTableView(parent) {
  model = new BinaryViewModel(this);
  setModel(model);
  delegate = new BinaryItemDelegate(this);
  setItemDelegate(delegate);
  horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  verticalHeader()->setSectionsClickable(false);
  verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
  verticalHeader()->setDefaultSectionSize(CELL_HEIGHT);
  horizontalHeader()->hide();
  setFrameShape(QFrame::NoFrame);
  setShowGrid(false);
  setMouseTracking(true);
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &BinaryView::refresh);
  QObject::connect(UndoStack::instance(), &QUndoStack::indexChanged, this, &BinaryView::refresh);
}

QSize BinaryView::minimumSizeHint() const {
  return {(horizontalHeader()->minimumSectionSize() + 1) * 9 + VERTICAL_HEADER_WIDTH,
          CELL_HEIGHT * std::min(model->rowCount(), 10)};
}

void BinaryView::highlight(const Signal *sig) {
  if (sig != hovered_sig) {
    for (int i = 0; i < model->items.size(); ++i) {
      auto &item_sigs = model->items[i].sigs;
      if ((sig && item_sigs.contains(sig)) || (hovered_sig && item_sigs.contains(hovered_sig))) {
        auto index = model->index(i / model->columnCount(), i % model->columnCount());
        emit model->dataChanged(index, index, {Qt::DisplayRole});
      }
    }
    hovered_sig = sig;
    emit signalHovered(hovered_sig);
  }
}

void BinaryView::setSelection(const QRect &rect, QItemSelectionModel::SelectionFlags flags) {
  auto index = indexAt(viewport()->mapFromGlobal(QCursor::pos()));
  if (!anchor_index.isValid() || !index.isValid())
    return;

  QItemSelection selection;
  auto [start, size, is_lb] = getSelection(index);
  for (int i = start; i < start + size; ++i) {
    auto idx = model->bitIndex(i, is_lb);
    selection.merge({idx, idx}, flags);
  }
  selectionModel()->select(selection, flags);
}

void BinaryView::mousePressEvent(QMouseEvent *event) {
  delegate->selection_color = (palette().color(QPalette::Active, QPalette::Highlight));
  if (auto index = indexAt(event->pos()); index.isValid() && index.column() != 8) {
    anchor_index = index;
    auto item = (const BinaryViewModel::Item *)anchor_index.internalPointer();
    int bit_idx = get_bit_index(anchor_index, true);
    for (auto s : item->sigs) {
      if (bit_idx == s->lsb || bit_idx == s->msb) {
        anchor_index = model->bitIndex(bit_idx == s->lsb ? s->msb : s->lsb, true);
        resize_sig = s;
        delegate->selection_color = item->bg_color;
        break;
      }
    }
  }
  event->accept();
}

void BinaryView::highlightPosition(const QPoint &pos) {
  if (auto index = indexAt(viewport()->mapFromGlobal(pos)); index.isValid()) {
    auto item = (BinaryViewModel::Item *)index.internalPointer();
    const Signal *sig = item->sigs.isEmpty() ? nullptr : item->sigs.back();
    highlight(sig);
    QToolTip::showText(pos, sig ? sig->name.c_str() : "", this, rect());
  }
}

void BinaryView::mouseMoveEvent(QMouseEvent *event) {
  highlightPosition(event->globalPos());
  QTableView::mouseMoveEvent(event);
}

void BinaryView::mouseReleaseEvent(QMouseEvent *event) {
  QTableView::mouseReleaseEvent(event);

  auto release_index = indexAt(event->pos());
  if (release_index.isValid() && anchor_index.isValid()) {
    if (selectionModel()->hasSelection()) {
      auto [start_bit, size, is_lb] = getSelection(release_index);
      resize_sig ? emit resizeSignal(resize_sig, start_bit, size)
                 : emit addSignal(start_bit, size, is_lb);
    } else {
      auto item = (const BinaryViewModel::Item *)anchor_index.internalPointer();
      if (item && item->sigs.size() > 0)
        emit signalClicked(item->sigs.back());
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
  model->msg_id = message_id;
  verticalScrollBar()->setValue(0);
  refresh();
}

void BinaryView::refresh() {
  if (model->msg_id.isEmpty()) return;

  clearSelection();
  anchor_index = QModelIndex();
  resize_sig = nullptr;
  hovered_sig = nullptr;
  highlightPosition(QCursor::pos());
  model->refresh();
}

QSet<const Signal *> BinaryView::getOverlappingSignals() const {
  QSet<const Signal *> overlapping;
  for (auto &item : model->items) {
    if (item.sigs.size() > 1)
      for (auto s : item.sigs) overlapping += s;
  }
  return overlapping;
}

std::tuple<int, int, bool> BinaryView::getSelection(QModelIndex index) {
  if (index.column() == 8) {
    index = model->index(index.row(), 7);
  }
  bool is_lb = (resize_sig && resize_sig->is_little_endian) || (!resize_sig && index < anchor_index);
  int cur_bit_idx = get_bit_index(index, is_lb);
  int anchor_bit_idx = get_bit_index(anchor_index, is_lb);
  auto [start_bit, end_bit] = std::minmax(cur_bit_idx, anchor_bit_idx);
  return {start_bit, end_bit - start_bit + 1, is_lb};
}

// BinaryViewModel

void BinaryViewModel::refresh() {
  beginResetModel();
  items.clear();
  if ((dbc_msg = dbc()->msg(msg_id))) {
    row_count = dbc_msg->size;
    items.resize(row_count * column_count);
    int i = 0;
    for (auto sig : dbc_msg->getSignals()) {
      auto [start, end] = getSignalRange(sig);
      for (int j = start; j <= end; ++j) {
        int bit_index = sig->is_little_endian ? bigEndianBitIndex(j) : j;
        int idx = column_count * (bit_index / 8) + bit_index % 8;
        if (idx >= items.size()) {
          qWarning() << "signal " << sig->name.c_str() << "out of bounds.start_bit:" << sig->start_bit << "size:" << sig->size;
          break;
        }
        if (j == start) sig->is_little_endian ? items[idx].is_lsb = true : items[idx].is_msb = true;
        if (j == end) sig->is_little_endian ? items[idx].is_msb = true : items[idx].is_lsb = true;
        items[idx].bg_color = getColor(i);
        items[idx].sigs.push_back(sig);
      }
      ++i;
    }
  } else {
    row_count = can->lastMessage(msg_id).dat.size();
    items.resize(row_count * column_count);
  }
  endResetModel();
  updateState();
}

void BinaryViewModel::updateState() {
  auto prev_items = items;
  const auto &last_msg = can->lastMessage(msg_id);
  const auto &binary = last_msg.dat;

  // data size may changed.
  if (binary.size() > row_count) {
    beginInsertRows({}, row_count, binary.size() - 1);
    row_count = binary.size();
    items.resize(row_count * column_count);
    endInsertRows();
  }
  char hex[3] = {'\0'};
  for (int i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      items[i * column_count + j].val = ((binary[i] >> (7 - j)) & 1) != 0 ? '1' : '0';
    }
    hex[0] = toHex(binary[i] >> 4);
    hex[1] = toHex(binary[i] & 0xf);
    items[i * column_count + 8].val = hex;
    items[i * column_count + 8].bg_color = last_msg.colors[i];
  }
  for (int i = binary.size(); i < row_count; ++i) {
    for (int j = 0; j < column_count; ++j) {
      items[i * column_count + j].val = "-";
    }
  }

  for (int i = 0; i < items.size(); ++i) {
    if (i >= prev_items.size() || prev_items[i].val != items[i].val || prev_items[i].bg_color != items[i].bg_color) {
      auto idx = index(i / column_count, i % column_count);
      emit dataChanged(idx, idx);
    }
  }
}

QVariant BinaryViewModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Vertical) {
    switch (role) {
      case Qt::DisplayRole: return section;
      case Qt::SizeHintRole: return QSize(VERTICAL_HEADER_WIDTH, 0);
      case Qt::TextAlignmentRole: return Qt::AlignCenter;
    }
  }
  return {};
}

// BinaryItemDelegate

BinaryItemDelegate::BinaryItemDelegate(QObject *parent) : QStyledItemDelegate(parent) {
  small_font.setPixelSize(8);
  hex_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
  hex_font.setBold(true);
}

void BinaryItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto item = (const BinaryViewModel::Item *)index.internalPointer();
  BinaryView *bin_view = (BinaryView *)parent();
  painter->save();

  if (index.column() == 8) {
    painter->setFont(hex_font);
    painter->fillRect(option.rect, item->bg_color);
  } else if (option.state & QStyle::State_Selected) {
    painter->fillRect(option.rect, selection_color);
    painter->setPen(option.palette.color(QPalette::BrightText));
  } else if (!item->sigs.isEmpty() && (!bin_view->selectionModel()->hasSelection() || !item->sigs.contains(bin_view->resize_sig))) {
    bool sig_hovered = item->sigs.contains(bin_view->hovered_sig);
    painter->fillRect(option.rect, sig_hovered ? item->bg_color.darker(125) : item->bg_color);  // 4/5x brightness
    painter->setPen(sig_hovered ? option.palette.color(QPalette::BrightText) : Qt::black);
  }

  painter->drawText(option.rect, Qt::AlignCenter, item->val);
  if (item->is_msb || item->is_lsb) {
    painter->setFont(small_font);
    painter->drawText(option.rect, Qt::AlignHCenter | Qt::AlignBottom, item->is_msb ? "MSB" : "LSB");
  }
  painter->restore();
}
