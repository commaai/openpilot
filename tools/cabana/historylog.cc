#include "tools/cabana/historylog.h"

#include <QHeaderView>
#include <QVBoxLayout>

QVariant HistoryLogModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    if (index.row() < values.size() && index.column() < values[index.row()].second.size())
      return values[index.row()].second[index.column()];
  }
  return {};
}

void HistoryLogModel::setMessage(const QString &message_id) {
  beginResetModel();
  msg_id = message_id;
  const auto msg = dbc()->msg(message_id);
  column_count = msg && !msg->sigs.empty() ? msg->sigs.size() : 1;
  values.clear();
  previous_count = 0;
  endResetModel();
}

QVariant HistoryLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal) {
    if (role == Qt::BackgroundRole) {
      auto msg = dbc()->msg(msg_id);
      if (msg && msg->sigs.size() > 0)
        return QBrush(QColor(getColor(section)));
    } else if (role == Qt::DisplayRole || role == Qt::ToolTipRole) {
      auto msg = dbc()->msg(msg_id);
      if (msg && section < msg->sigs.size())
        return QString::fromStdString(msg->sigs[section].name);
    }
  } else if (role == Qt::DisplayRole && section < values.size()) {
    return values[section].first;
  }
  return {};
}

void HistoryLogModel::clear() {
  values.clear();
  previous_count = 0;
  emit dataChanged(index(0, 0), index(CAN_MSG_LOG_SIZE - 1, column_count - 1));
}

void HistoryLogModel::updateState() {
  const auto &can_msgs = can->messages(msg_id);
  const auto msg = dbc()->msg(msg_id);
  uint64_t new_count = previous_count;
  for (const auto &can_data : can_msgs) {
    if (can_data.count <= previous_count)
      continue;

    if (values.size() >= CAN_MSG_LOG_SIZE)
      values.pop_back();

    QStringList data;
    if (msg && !msg->sigs.empty()) {
      for (const auto &sig : msg->sigs) {
        double value = get_raw_value((uint8_t *)can_data.dat.begin(), can_data.dat.size(), sig);
        data.append(QString::number(value));
      }
    } else {
      data.append(toHex(can_data.dat));
    }
    values.push_front({QString::number(can_data.ts, 'f', 2), data});
    new_count = can_data.count;
  }
  if (new_count != previous_count) {
    previous_count = new_count;
    emit dataChanged(index(0, 0), index(CAN_MSG_LOG_SIZE - 1, column_count - 1));
    emit headerDataChanged(Qt::Vertical, 0, CAN_MSG_LOG_SIZE - 1);
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

  QObject::connect(can, &CANMessages::rangeChanged, model, &HistoryLogModel::clear);
}

void HistoryLog::setMessage(const QString &message_id) {
  model->setMessage(message_id);
  model->updateState();
}

void HistoryLog::updateState() {
  model->updateState();
}
