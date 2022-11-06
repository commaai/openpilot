#include "tools/cabana/messageswidget.h"

#include <QFontDatabase>
#include <QHeaderView>
#include <QLineEdit>
#include <QVBoxLayout>

#include "tools/cabana/dbcmanager.h"

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // message filter
  QLineEdit *filter = new QLineEdit(this);
  filter->setClearButtonEnabled(true);
  filter->setPlaceholderText(tr("filter messages"));
  main_layout->addWidget(filter);

  // message table
  table_widget = new QTableView(this);
  model = new MessageListModel(this);
  table_widget->setModel(model);
  table_widget->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  table_widget->setSortingEnabled(true);
  table_widget->sortByColumn(0, Qt::AscendingOrder);
  table_widget->setColumnWidth(0, 250);
  table_widget->setColumnWidth(1, 80);
  table_widget->setColumnWidth(2, 80);
  table_widget->horizontalHeader()->setStretchLastSection(true);
  table_widget->verticalHeader()->hide();
  main_layout->addWidget(table_widget);

  // signals/slots
  QObject::connect(filter, &QLineEdit::textChanged, model, &MessageListModel::setFilterString);
  QObject::connect(can, &CANMessages::updated, [this]() { model->updateState(); });
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, [this]() { model->updateState(true); });
  QObject::connect(table_widget->selectionModel(), &QItemSelectionModel::currentChanged, [=](const QModelIndex &current, const QModelIndex &previous) {
    if (current.isValid()) {
      emit msgSelectionChanged(current.data(Qt::UserRole).toString());
    }
  });
}

// MessageListModel

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    return (QString[]){"Name", "ID", "Freq", "Count", "Bytes"}[section];
  return {};
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    const auto &m = msgs[index.row()];
    auto &can_data = can->lastMessage(m->id);
    switch (index.column()) {
      case 0: return m->name;
      case 1: return m->id;
      case 2: return can_data.freq;
      case 3: return can_data.count;
      case 4: return toHex(can_data.dat);
    }
  } else if (role == Qt::UserRole) {
    return msgs[index.row()]->id;
  } else if (role == Qt::FontRole) {
    if (index.column() == columnCount() - 1) {
      return QFontDatabase::systemFont(QFontDatabase::FixedFont);
    }
  }
  return {};
}

void MessageListModel::setFilterString(const QString &string) { 
  filter_str = string;
  updateState(true);
}

bool MessageListModel::updateMessages(bool sort) {
  if (msgs.size() == can->can_msgs.size() && filter_str.isEmpty() && !sort)
    return false;

  // update message list
  int i = 0;
  bool search_id = filter_str.contains(':');
  for (auto it = can->can_msgs.begin(); it != can->can_msgs.end(); ++it) {
    const Msg *msg = dbc()->msg(it.key());
    QString msg_name = msg ? msg->name.c_str() : "untitled";
    if (!filter_str.isEmpty() && !(search_id ? it.key() : msg_name).contains(filter_str, Qt::CaseInsensitive))
      continue;
    auto &m = i < msgs.size() ? msgs[i] : msgs.emplace_back(new Message);
    m->id = it.key();
    m->name = msg_name;
    ++i;
  }
  msgs.resize(i);

  if (sort_column == 0) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      bool ret = l->name < r->name || (l->name == r->name && l->id < r->id);
      return sort_order == Qt::AscendingOrder ? ret : !ret;
    });
  } else if (sort_column == 1) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      return sort_order == Qt::AscendingOrder ? l->id < r->id : l->id > r->id;
    });
  } else if (sort_column == 2) {
    // sort by frequency
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      uint32_t lfreq = can->lastMessage(l->id).freq;
      uint32_t rfreq = can->lastMessage(r->id).freq;
      bool ret = lfreq < rfreq || (lfreq == rfreq && l->id < r->id);
      return sort_order == Qt::AscendingOrder ? ret : !ret;
    });
  } else if (sort_column == 3) {
    // sort by count
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      uint32_t lcount = can->lastMessage(l->id).count;
      uint32_t rcount = can->lastMessage(r->id).count;
      bool ret = lcount < rcount || (lcount == rcount && l->id < r->id);
      return sort_order == Qt::AscendingOrder ? ret : !ret;
    });
  }
  return true;
}

void MessageListModel::updateState(bool sort) {
  int prev_row_count = msgs.size();
  auto prev_idx = persistentIndexList();
  QString selected_msg_id = prev_idx.empty() ? "" : prev_idx[0].data(Qt::UserRole).toString();

  bool msg_updated = updateMessages(sort);
  int delta = msgs.size() - prev_row_count;
  if (delta > 0) {
    beginInsertRows({}, prev_row_count, msgs.size() - 1);
    endInsertRows();
  } else if (delta < 0) {
    beginRemoveRows({}, msgs.size(), prev_row_count - 1);
    endRemoveRows();
  }

  if (!msgs.empty()) {
    if (msg_updated && !prev_idx.isEmpty()) {
      // keep selection
      auto it = std::find_if(msgs.begin(), msgs.end(), [&](auto &m) { return m->id == selected_msg_id; });
      if (it != msgs.end()) {
        for (auto &idx : prev_idx)
          changePersistentIndex(idx, index(std::distance(msgs.begin(), it), idx.column()));
      }
    }
    emit dataChanged(index(0, 0), index(msgs.size() - 1, columnCount() - 1), {Qt::DisplayRole});
  }
}

void MessageListModel::sort(int column, Qt::SortOrder order) {
  if (column != columnCount() - 1) {
    sort_column = column;
    sort_order = order;
    updateState(true);
  }
}
