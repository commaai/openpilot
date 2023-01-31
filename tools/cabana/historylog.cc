#include "tools/cabana/historylog.h"

#include <QFontDatabase>

// HistoryLogModel

QVariant HistoryLogModel::data(const QModelIndex &index, int role) const {
  bool has_signal = dbc_msg && !dbc_msg->sigs.empty();
  if (role == Qt::DisplayRole) {
    const auto &m = messages[index.row()];
    if (index.column() == 0) {
      return QString::number(m.ts, 'f', 2);
    }
    return has_signal ? QString::number(get_raw_value((uint8_t *)m.dat.begin(), m.dat.size(), dbc_msg->sigs[index.column() - 1]))
                      : toHex(m.dat);
  } else if (role == Qt::FontRole && index.column() == 1 && !has_signal) {
    return QFontDatabase::systemFont(QFontDatabase::FixedFont);
  }
  return {};
}

void HistoryLogModel::setMessage(const QString &message_id) {
  beginResetModel();
  msg_id = message_id;
  dbc_msg = dbc()->msg(message_id);
  column_count = (dbc_msg && !dbc_msg->sigs.empty() ? dbc_msg->sigs.size() : 1) + 1;
  row_count = 0;
  endResetModel();

  updateState();
}

QVariant HistoryLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal) {
    bool has_signal = dbc_msg && !dbc_msg->sigs.empty();
    if (role == Qt::DisplayRole || role == Qt::ToolTipRole) {
      if (section == 0) {
        return "Time";
      }
      return has_signal ? QString::fromStdString(dbc_msg->sigs[section - 1].name).replace('_', ' ') : "Data";
    } else if (role == Qt::BackgroundRole && section > 0 && has_signal) {
      return QBrush(QColor(getColor(section - 1)));
    }
  }
  return {};
}

void HistoryLogModel::updateState() {
  if (msg_id.isEmpty()) return;

  int prev_row_count = row_count;
  messages = can->messages(msg_id);
  row_count = messages.size();
  int delta = row_count - prev_row_count;
  if (delta > 0) {
    beginInsertRows({}, prev_row_count, row_count - 1);
    endInsertRows();
  } else if (delta < 0) {
    beginRemoveRows({}, row_count, prev_row_count - 1);
    endRemoveRows();
  }
  if (row_count > 0) {
    emit dataChanged(index(0, 0), index(row_count - 1, column_count - 1), {Qt::DisplayRole});
  }
}

// HeaderView

QSize HeaderView::sectionSizeFromContents(int logicalIndex) const {
  const QString text = model()->headerData(logicalIndex, this->orientation(), Qt::DisplayRole).toString();
  const QRect rect = fontMetrics().boundingRect(QRect(0, 0, sectionSize(logicalIndex), 1000), defaultAlignment(), text);
  return rect.size() + QSize{10, 5};
}

// HistoryLog

HistoryLog::HistoryLog(QWidget *parent) : QTableView(parent) {
  model = new HistoryLogModel(this);
  setModel(model);
  setHorizontalHeader(new HeaderView(Qt::Horizontal, this));
  horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | (Qt::Alignment)Qt::TextWordWrap);
  horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
  verticalHeader()->setVisible(false);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
  setFrameShape(QFrame::NoFrame);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
  setStyleSheet("QTableView::item { border:0px; padding-left:5px; padding-right:5px; }");
}

int HistoryLog::sizeHintForColumn(int column) const {
  // sizeHintForColumn is only called for column 0 (ResizeToContents)
  return itemDelegate()->sizeHint(viewOptions(), model->index(0, 0)).width() + 1; // +1 for grid
}
