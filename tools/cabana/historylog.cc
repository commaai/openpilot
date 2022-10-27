#include "tools/cabana/historylog.h"

#include <QFontDatabase>
#include <QHeaderView>
#include <QVBoxLayout>

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
      return has_signal ? dbc_msg->sigs[section - 1].name.c_str() : "Data";
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

HistoryLog::HistoryLog(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  model = new HistoryLogModel(this);
  table = new QTableView(this);
  table->setModel(model);
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
  table->setColumnWidth(0, 60);
  table->verticalHeader()->setVisible(false);
  table->setStyleSheet("QTableView::item { border:0px; padding-left:5px; padding-right:5px; }");
  main_layout->addWidget(table);
}
