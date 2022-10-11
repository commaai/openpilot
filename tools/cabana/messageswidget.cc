#include "tools/cabana/messageswidget.h"

#include <QComboBox>
#include <QCompleter>
#include <QHeaderView>
#include <QPushButton>
#include <QVBoxLayout>

#include "tools/cabana/dbcmanager.h"

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *dbc_file_layout = new QHBoxLayout();
  QComboBox *combo = new QComboBox(this);
  auto dbc_names = dbc()->allDBCNames();
  for (const auto &name : dbc_names) {
    combo->addItem(QString::fromStdString(name));
  }
  combo->setEditable(true);
  combo->setCurrentText(QString());
  combo->setInsertPolicy(QComboBox::NoInsert);
  combo->completer()->setCompletionMode(QCompleter::PopupCompletion);
  QFont font;
  font.setBold(true);
  combo->lineEdit()->setFont(font);
  dbc_file_layout->addWidget(combo);

  dbc_file_layout->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save DBC"), this);
  dbc_file_layout->addWidget(save_btn);
  main_layout->addLayout(dbc_file_layout);

  filter = new QLineEdit(this);
  filter->setPlaceholderText(tr("filter messages"));
  main_layout->addWidget(filter);

  table_widget = new QTableView(this);
  model = new MessageListModel(this);
  table_widget->setModel(model);
  table_widget->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  table_widget->setSortingEnabled(true);
  table_widget->setColumnWidth(0, 250);
  table_widget->setColumnWidth(1, 80);
  table_widget->setColumnWidth(2, 80);
  table_widget->horizontalHeader()->setStretchLastSection(true);
  table_widget->sortByColumn(0, Qt::AscendingOrder);
  main_layout->addWidget(table_widget);

  QObject::connect(filter, &QLineEdit::textChanged, model, &MessageListModel::setNameFilter);
  QObject::connect(can, &CANMessages::updated, model, &MessageListModel::updateState);
  QObject::connect(combo, SIGNAL(activated(const QString &)), SLOT(dbcSelectionChanged(const QString &)));
  QObject::connect(save_btn, &QPushButton::clicked, [=]() {
    // TODO: save DBC to file
  });
  QObject::connect(table_widget->selectionModel(), &QItemSelectionModel::currentChanged, [=](const QModelIndex &current, const QModelIndex &previous) {
    if (current.isValid()) {
      emit msgSelectionChanged(table_widget->model()->data(current, Qt::UserRole).toString());
    }
  });

  // For test purpose
  combo->setCurrentText("toyota_nodsu_pt_generated");
}

void MessagesWidget::dbcSelectionChanged(const QString &dbc_file) {
  dbc()->open(dbc_file);
  // update detailwidget
  auto current = table_widget->selectionModel()->currentIndex();
  if (current.isValid()) {
    emit msgSelectionChanged(table_widget->model()->data(current, Qt::UserRole).toString());
  }
}

// MessageListModel

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    return (QString[]){"name", "id", "count", "data"}[section];
  else if (orientation == Qt::Vertical && role == Qt::DisplayRole) {
    return QString::number(section);
  }
  return {};
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    assert(index.row() < messages.size());
    const auto &d = messages[index.row()];
    switch (index.column()) {
      case 0: return d->name;
      case 1: return d->id;
      case 2: return QString::number(d->count);
      case 3: return toHex(can->can_msgs[d->id].back().dat);
    }
  } else if (role == Qt::UserRole) {
    return messages[index.row()]->id;
  }
  return {};
}

void MessageListModel::updateState() {
  int prev_row_count = messages.size();
  int row_count = 0;
  for (auto it = can->can_msgs.constBegin(); it != can->can_msgs.constEnd(); ++it) {
    auto msg = dbc()->msg(it.key());
    QString name = msg ? msg->name.c_str() : "untitled";
    if (name_filter.isEmpty() || name.contains(name_filter, Qt::CaseInsensitive)) {
      if (row_count >= messages.size()) {
        messages.emplace_back(new Data);
      }
      Data *d = messages[row_count].get();
      d->name = name;
      d->id = it.key();
      d->count = can->counters[it.key()];
      ++row_count;
    }
  }
  messages.resize(row_count);

  // sort messages
  if (sort_column == 0) {
    std::sort(messages.begin(), messages.end(), [&](auto &v1, auto &v2) {
      bool ret = v1->name < v2->name || (v1->name == v2->name && v1->id < v2->id);
      return sort_order == Qt::AscendingOrder ? ret : !ret;
    });
  } else if (sort_column == 1) {
    std::sort(messages.begin(), messages.end(), [=](auto &v1, auto &v2) {
      return sort_order == Qt::AscendingOrder ? v1->id < v2->id : v1->id > v2->id;
    });
  } else if (sort_column == 2) {
    std::sort(messages.begin(), messages.end(), [this](auto &v1, auto &v2) {
      bool ret = v1->count < v2->count || (v1->count == v2->count && v1->id < v2->id);
      return sort_order == Qt::AscendingOrder ? ret : !ret;
    });
  }

  // update view
  int delta = row_count - prev_row_count;
  if (delta > 0) {
    beginInsertRows({}, prev_row_count, row_count - 1);
    endInsertRows();
  } else if (delta < 0) {
    beginRemoveRows({}, row_count, prev_row_count - 1);
    endRemoveRows();
  }

  if (row_count > 0) {
    emit dataChanged(index(0, 0), index(row_count - 1, 3));
  }
}

void MessageListModel::sort(int column, Qt::SortOrder order) {
  if (column != 3) {
    sort_column = column;
    sort_order = order;
    updateState();
  }
}
