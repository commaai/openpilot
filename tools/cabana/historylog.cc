#include "tools/cabana/historylog.h"

#include <QHeaderView>
#include <QVBoxLayout>

QVariant HistoryLogModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    const auto &can_msgs = can->messages(msg_id);
    if (index.row() < can_msgs.size()) {
      const auto &can_data = can_msgs[index.row()];
      auto msg = dbc()->msg(msg_id);
      if (msg && index.column() < msg->sigs.size()) {
        return get_raw_value((uint8_t *)can_data.dat.begin(), can_data.dat.size(), msg->sigs[index.column()]);
      } else {
        return toHex(can_data.dat);
      }
    }
  }
  return {};
}

void HistoryLogModel::setMessage(const QString &message_id) {
  beginResetModel();
  msg_id = message_id;
  const auto msg = dbc()->msg(message_id);
  column_count = msg && !msg->sigs.empty() ? msg->sigs.size() : 1;
  row_count = 0;
  endResetModel();

  updateState();
}

QVariant HistoryLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal) {
    auto msg = dbc()->msg(msg_id);
    if (msg && section < msg->sigs.size()) {
      if (role == Qt::BackgroundRole) {
        return QBrush(QColor(getColor(section)));
      } else if (role == Qt::DisplayRole || role == Qt::ToolTipRole) {
        return QString::fromStdString(msg->sigs[section].name);
      }
    }
  } else if (role == Qt::DisplayRole) {
    const auto &can_msgs = can->messages(msg_id);
    if (section < can_msgs.size()) {
      return QString::number(can_msgs[section].ts, 'f', 2);
    }
  }
  return {};
}

void HistoryLogModel::updateState() {
  if (msg_id.isEmpty()) return;

  int prev_row_count = row_count;
  row_count = can->messages(msg_id).size();
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
    emit headerDataChanged(Qt::Vertical, 0, row_count - 1);
  }
}

HistoryLog::HistoryLog(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  model = new HistoryLogModel(this);
  table = new QTableView(this);
  table->setModel(model);
  table->horizontalHeader()->setStretchLastSection(true);
  table->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table->setStyleSheet("QTableView::item { border:0px; padding-left:5px; padding-right:5px; }");
  table->verticalHeader()->setStyleSheet("QHeaderView::section {padding-left: 5px; padding-right: 5px;min-width:40px;}");
  main_layout->addWidget(table);
}

void HistoryLog::setMessage(const QString &message_id) {
  model->setMessage(message_id);
}

void HistoryLog::updateState() {
  model->updateState();
}
