#include "tools/cabana/messageswidget.h"

#include <QFontDatabase>
#include <QHeaderView>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QPainter>
#include <QApplication>

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
  table_widget->setItemDelegateForColumn(4, new MessageBytesDelegate(table_widget));
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
  QObject::connect(can, &CANMessages::msgsReceived, model, &MessageListModel::msgsReceived);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, model, &MessageListModel::sortMessages);
  QObject::connect(dbc(), &DBCManager::msgUpdated, model, &MessageListModel::sortMessages);
  QObject::connect(dbc(), &DBCManager::msgRemoved, model, &MessageListModel::sortMessages);
  QObject::connect(model, &MessageListModel::modelReset, [this]() { selectMessage(current_msg_id); });
  QObject::connect(table_widget->selectionModel(), &QItemSelectionModel::currentChanged, [=](const QModelIndex &current, const QModelIndex &previous) {
    if (current.isValid() && current.row() < model->msgs.size()) {
      if (model->msgs[current.row()] != current_msg_id) {
        current_msg_id = model->msgs[current.row()];
        emit msgSelectionChanged(current_msg_id);
      }
    }
  });
}

void MessagesWidget::selectMessage(const QString &msg_id) {
  if (int row = model->msgs.indexOf(msg_id); row != -1) {
    table_widget->selectionModel()->setCurrentIndex(model->index(row, 0), QItemSelectionModel::Rows | QItemSelectionModel::ClearAndSelect);
  }
}

// MessageListModel

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    return (QString[]){"Name", "ID", "Freq", "Count", "Bytes"}[section];
  return {};
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  const auto &id = msgs[index.row()];
  auto &can_data = can->lastMessage(id);

  if (role == Qt::DisplayRole) {
    switch (index.column()) {
      case 0: return msgName(id);
      case 1: return id;
      case 2: return can_data.freq;
      case 3: return can_data.count;
      case 4: return toHex(can_data.dat);
    }
  } else if (role == Qt::FontRole && index.column() == columnCount() - 1) {
    return QFontDatabase::systemFont(QFontDatabase::FixedFont);
  } else if (role == Qt::UserRole && index.column() == 4) {

    QList<QVariant> colors;
    for (int i = 0; i < can_data.dat.size(); i++){
      colors.append(can_data.colors[i]);
    }
    return colors;
  }
  return {};
}

void MessageListModel::setFilterString(const QString &string) {
  filter_str = string;
  msgs.clear();
  for (auto it = can->can_msgs.begin(); it != can->can_msgs.end(); ++it) {
    bool found = false;

    // Search by message id or name
    if (it.key().contains(filter_str, Qt::CaseInsensitive) || msgName(it.key()).contains(filter_str, Qt::CaseInsensitive)) {
      found = true;
    }

    // Search by signal name
    const DBCMsg *msg = dbc()->msg(it.key());
    if (msg != nullptr) {
      for (auto &signal: msg->getSignals()) {
        if (QString::fromStdString(signal->name).contains(filter_str, Qt::CaseInsensitive)) {
          found = true;
        }
      }
    }

    if (found) {
      msgs.push_back(it.key());
    }
  }
  sortMessages();
}

void MessageListModel::sortMessages() {
  beginResetModel();
  if (sort_column == 0) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto ll = std::pair{msgName(l), l};
      auto rr = std::pair{msgName(r), r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == 1) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto ll = std::tuple{can->lastMessage(l).src, can->lastMessage(l).address, l};
      auto rr = std::tuple{can->lastMessage(r).src, can->lastMessage(r).address, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == 2) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto ll = std::pair{can->lastMessage(l).freq, l};
      auto rr = std::pair{can->lastMessage(r).freq, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == 3) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto ll = std::pair{can->lastMessage(l).count, l};
      auto rr = std::pair{can->lastMessage(r).count, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  }
  endResetModel();
}

void MessageListModel::msgsReceived(const QHash<QString, CanData> *new_msgs) {
  int prev_row_count = msgs.size();
  if (filter_str.isEmpty() && msgs.size() != can->can_msgs.size()) {
    msgs = can->can_msgs.keys();
  }
  if (msgs.size() != prev_row_count) {
    sortMessages();
    return;
  }
  for (int i = 0; i < msgs.size(); ++i) {
    if (new_msgs->contains(msgs[i])) {
      for (int col = 2; col < columnCount(); ++col)
        emit dataChanged(index(i, col), index(i, col), {Qt::DisplayRole});
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


MessageBytesDelegate::MessageBytesDelegate(QObject *parent) : QStyledItemDelegate(parent) {
}

void MessageBytesDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  QList<QVariant> colors = index.data(Qt::UserRole).toList();

  QStyleOptionViewItemV4 opt = option;
  initStyleOption(&opt, index);

  const QFont font = index.data(Qt::FontRole).value<QFont>();
  painter->setFont(font);

  QRect rect = opt.rect;
  QString bytes = QString(opt.text);

  QRect pos = rect;
  QRect space = painter->boundingRect(pos, opt.displayAlignment, " ");
  pos.setX(pos.x() + space.width());

  if ((option.state & QStyle::State_Selected) && (option.state & QStyle::State_Active)) {
    painter->setPen(option.palette.color(QPalette::HighlightedText));
  } else {
    painter->setPen(option.palette.color(QPalette::Text));
  }

  int i = 0;
  for (auto &byte : bytes.split(" ")) {
    QRect sz = painter->boundingRect(pos, opt.displayAlignment, byte);
    const int m = space.width() / 2;
    painter->fillRect(sz.marginsAdded(QMargins(m + 1, m, m, m)), colors[i].value<QColor>());
    painter->drawText(pos, opt.displayAlignment, byte);
    pos.setX(pos.x() + sz.width() + space.width());
    i++;
  }
}
