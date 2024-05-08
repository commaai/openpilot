#include "tools/cabana/utils/signaltable.h"

#include <algorithm>
#include <QHeaderView>
#include <QScrollBar>
#include <QWheelEvent>

void SignalTableModel::setSignals(const std::set<std::pair<MessageId, QString>> &sigs) {
  signals_ = sigs;
  updateState(nullptr, true);
}

void SignalTableModel::updateState(const std::set<MessageId> *new_msgs, bool has_new_ids) {
  beginResetModel();
  if (new_msgs) {
    for (const auto &id : *new_msgs)
      updateMessage(id, can->lastMessage(id));
  } else {
    signals_map_.clear();
    for (const auto &[id, msg] : can->lastMessages())
      updateMessage(id, msg);
  }

  signal_items_.clear();
  signal_items_.reserve(signals_map_.size());
  for (const auto &[_, item] : signals_map_) {
    signal_items_.push_back(&item);
  }
  std::sort(signal_items_.rbegin(), signal_items_.rend(),
            [](const Item *l, const Item *r) { return l->seconds < r->seconds; });
  endResetModel();
}

void SignalTableModel::updateMessage(const MessageId &id, const CanData &data) {
  if (auto m = dbc()->msg(id); m && !data.dat.empty()) {
    for (auto signal : m->getSignals()) {
      double value;
      auto key = std::make_pair(id, signal->name);
      if ((display_all_ || signals_.count(key) > 0) &&
          signal->getValue(data.dat.data(), data.dat.size(), &value)) {
        auto &item = signals_map_[key];
        item.seconds = data.ts;
        item.msg_name = QString("%1 %2").arg(m->name, id.toString());
        item.sig_name = signal->name;
        item.unit = signal->unit;
        item.value = signal->formatValue(value);
      }
    }
  }
}

QVariant SignalTableModel::headerData(int section, Qt::Orientation orientation, int role) const {
  const static QString titles[] = {tr("Time"), tr("Signal"), tr("Message"), tr("Value")};
  if (orientation != Qt::Horizontal || role != Qt::DisplayRole) return {};
  return titles[section];
}

QVariant SignalTableModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    const Item *item = signal_items_.at(index.row());
    switch (index.column()) {
      case 0: return QString::number(item->seconds, 'f', 3);
      case 1: return item->sig_name;
      case 2: return item->msg_name;
      case 3: return item->value;
    }
  } else if (role == Qt::TextAlignmentRole && index.column() == 3) {
    return (uint32_t)(Qt::AlignRight | Qt::AlignVCenter);
  }
  return {};
}

SignalTable::SignalTable(QWidget *parent) : QTableView(parent) {
  setModel(model_ = new SignalTableModel(this));
  verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
  horizontalHeader()->setStretchLastSection(true);
  setSelectionMode(QAbstractItemView::NoSelection);
  QObject::connect(can, &AbstractStream::msgsReceived, this, [this](const std::set<MessageId> *new_msgs, bool has_new_ids) {
    if (isVisible()) model_->updateState(new_msgs, has_new_ids);
  });
}

void SignalTable::wheelEvent(QWheelEvent *event) {
  if (event->modifiers() == Qt::ShiftModifier) {
    QApplication::sendEvent(horizontalScrollBar(), event);
  } else {
    QTableView::wheelEvent(event);
  }
}
