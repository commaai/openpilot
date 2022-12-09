#include "tools/cabana/historylog.h"

#include <QFontDatabase>
#include <QPainter>

// HistoryLogModel

QVariant HistoryLogModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    const auto &m = messages[index.row()];
    if (index.column() == 0) {
      return QString::number(m.ts, 'f', 2);
    }
    return !sigs.empty() ? QString::number(get_raw_value((uint8_t *)m.dat.data(), m.dat.size(), *sigs[index.column() - 1]))
                         : toHex(m.dat);
  } else if (role == Qt::FontRole && index.column() == 1 && sigs.empty()) {
    return QFontDatabase::systemFont(QFontDatabase::FixedFont);
  }
  return {};
}

void HistoryLogModel::setMessage(const QString &message_id) {
  beginResetModel();
  msg_id = message_id;
  sigs.clear();
  messages.clear();
  if (auto dbc_msg = dbc()->msg(message_id)) {
    sigs = dbc_msg->getSignals();
  }
  endResetModel();
  updateState();
}

QVariant HistoryLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal) {
    if (role == Qt::DisplayRole || role == Qt::ToolTipRole) {
      if (section == 0) {
        return "Time";
      }
      return !sigs.empty() ? QString::fromStdString(sigs[section - 1]->name).replace('_', ' ') : "Data";
    } else if (role == Qt::BackgroundRole && section > 0 && !sigs.empty()) {
      return QBrush(QColor(getColor(section - 1)));
    } else if (role == Qt::ForegroundRole && section > 0 && !sigs.empty()) {
      return QBrush(Qt::black);
    }
  }
  return {};
}

void HistoryLogModel::updateState() {
  int prev_row_count = messages.size();
  if (!msg_id.isEmpty()) {
    messages = can->messages(msg_id);
  }
  int delta = messages.size() - prev_row_count;
  if (delta > 0) {
    beginInsertRows({}, prev_row_count, messages.size() - 1);
    endInsertRows();
  } else if (delta < 0) {
    beginRemoveRows({}, messages.size(), prev_row_count - 1);
    endRemoveRows();
  }
  if (!messages.empty()) {
    emit dataChanged(index(0, 0), index(rowCount() - 1, columnCount() - 1), {Qt::DisplayRole});
  }
}

// HeaderView

QSize HeaderView::sectionSizeFromContents(int logicalIndex) const {
  int default_size = qMax(100, rect().width() / model()->columnCount());
  const QString text = model()->headerData(logicalIndex, this->orientation(), Qt::DisplayRole).toString();
  const QRect rect = fontMetrics().boundingRect({0, 0, default_size, 2000}, defaultAlignment(), text);
  QSize size = rect.size() + QSize{10, 6};
  return {qMax(size.width(), default_size), size.height()};
}

void HeaderView::paintSection(QPainter *painter, const QRect &rect, int logicalIndex) const {
  auto bg_role = model()->headerData(logicalIndex, Qt::Horizontal, Qt::BackgroundRole);
  if (bg_role.isValid()) {
    QPen pen(model()->headerData(logicalIndex, Qt::Horizontal, Qt::ForegroundRole).value<QBrush>(), 1);
    painter->setPen(pen);
    painter->fillRect(rect, bg_role.value<QBrush>());
  }
  QString text = model()->headerData(logicalIndex, Qt::Horizontal, Qt::DisplayRole).toString();
  painter->drawText(rect.adjusted(5, 3, -5, -3), defaultAlignment(), text);
}

// HistoryLog

HistoryLog::HistoryLog(QWidget *parent) : QTableView(parent) {
  model = new HistoryLogModel(this);
  setModel(model);
  setHorizontalHeader(new HeaderView(Qt::Horizontal, this));
  horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | (Qt::Alignment)Qt::TextWordWrap);
  horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  verticalHeader()->setVisible(false);
  setFrameShape(QFrame::NoFrame);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
}

int HistoryLog::sizeHintForColumn(int column) const {
  return -1;
}
