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
  QObject::connect(can, &CANMessages::msgsUpdated, model, &MessageListModel::msgsUpdated);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, model, &MessageListModel::sortMessages);
  QObject::connect(dbc(), &DBCManager::msgUpdated, model, &MessageListModel::sortMessages);
  QObject::connect(table_widget->selectionModel(), &QItemSelectionModel::currentChanged, [=](const QModelIndex &current, const QModelIndex &previous) {
    if (current.isValid() && current.row() < model->msgs.size()) {
      current_msg_id = model->msgs[current.row()];
      emit msgSelectionChanged(current_msg_id);
    }
  });
  QObject::connect(model, &MessageListModel::modelReset, [this]() {
    if (int row = model->msgs.indexOf(current_msg_id); row != -1)
      table_widget->selectionModel()->select(model->index(row, 0), QItemSelectionModel::Rows | QItemSelectionModel::ClearAndSelect);
  });
}

// MessageListModel

MessageListModel::MessageListModel(QObject *parent) : QAbstractTableModel(parent) {
  sort_timer = new QTimer(this);
  sort_timer->setSingleShot(true);
  sort_timer->setInterval(100);
  sort_timer->callOnTimeout(this, &MessageListModel::sortMessages);
}

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    return (QString[]){"Name", "ID", "Freq", "Count", "Bytes"}[section];
  return {};
}

inline QString msg_name(const QString &id) {
  auto msg = dbc()->msg(id);
  return msg ? msg->name.c_str() : "untitled";
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    const auto &id = msgs[index.row()];
    auto &can_data = can->lastMessage(id);
    switch (index.column()) {
      case 0: return msg_name(id);
      case 1: return id;
      case 2: return can_data.freq;
      case 3: return can_data.count;
      case 4: return toHex(can_data.dat);
    }
  } else if (role == Qt::FontRole && index.column() == columnCount() - 1) {
    return QFontDatabase::systemFont(QFontDatabase::FixedFont);
  }
  return {};
}

void MessageListModel::setFilterString(const QString &string) {
  filter_str = string;
  bool search_id = filter_str.contains(':');
  msgs.clear();
  for (auto it = can->can_msgs.begin(); it != can->can_msgs.end(); ++it) {
    if ((search_id ? it.key() : msg_name(it.key())).contains(filter_str, Qt::CaseInsensitive))
      msgs.push_back(it.key());
  }
  sortMessages();
}

void MessageListModel::sortMessages() {
  beginResetModel();
  if (sort_column == 0) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto l_name = msg_name(l);
      auto r_name = msg_name(r);
      bool ret = l_name < r_name || (l_name == r_name && l < r);
      return sort_order == Qt::AscendingOrder ? ret : !ret;
    });
  } else if (sort_column == 1) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      return sort_order == Qt::AscendingOrder ? l < r : l > r;
    });
  } else if (sort_column == 2) {
    // sort by frequency
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      uint32_t lfreq = can->lastMessage(l).freq;
      uint32_t rfreq = can->lastMessage(r).freq;
      bool ret = lfreq < rfreq || (lfreq == rfreq && l < r);
      return sort_order == Qt::AscendingOrder ? ret : !ret;
    });
  } else if (sort_column == 3) {
    // sort by count
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      uint32_t lcount = can->lastMessage(l).count;
      uint32_t rcount = can->lastMessage(r).count;
      bool ret = lcount < rcount || (lcount == rcount && l < r);
      return sort_order == Qt::AscendingOrder ? ret : !ret;
    });
  }
  endResetModel();
}

void MessageListModel::msgsUpdated(const QHash<QString, CanData> *new_msgs) {
  int prev_row_count = msgs.size();
  if (filter_str.isEmpty() && msgs.size() != can->can_msgs.size()) {
    msgs = can->can_msgs.keys();
  }
  if (msgs.size() != prev_row_count) {
    sort_timer->start();
  } else {
    for (int i = 0; i < msgs.size(); ++i) {
      if (new_msgs->contains(msgs[i])) {
        for (int col = 2; col < columnCount(); ++col)
          dataChanged(index(i, col), index(i, col), {Qt::DisplayRole});
      }
    }
  }
}

void MessageListModel::sort(int column, Qt::SortOrder order) {
  if (column != columnCount() - 1) {
    sort_column = column;
    sort_order = order;
    sortMessages();
  }
}
