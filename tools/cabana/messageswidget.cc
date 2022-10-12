#include "tools/cabana/messageswidget.h"

#include <QComboBox>
#include <QCompleter>
#include <QHeaderView>
#include <QLineEdit>
#include <QPushButton>
#include <QSortFilterProxyModel>
#include <QVBoxLayout>

#include "tools/cabana/dbcmanager.h"

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  // DBC file selector
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

  // message filter
  QLineEdit *filter = new QLineEdit(this);
  filter->setPlaceholderText(tr("filter messages"));
  main_layout->addWidget(filter);

  // message table
  table_widget = new QTableView(this);
  model = new MessageListModel(this);
  QSortFilterProxyModel *proxy_model = new QSortFilterProxyModel(this);
  proxy_model->setSourceModel(model);
  proxy_model->setFilterCaseSensitivity(Qt::CaseInsensitive);
  proxy_model->setDynamicSortFilter(false);
  table_widget->setModel(proxy_model);
  table_widget->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  table_widget->setSortingEnabled(true);
  table_widget->setColumnWidth(0, 250);
  table_widget->setColumnWidth(1, 80);
  table_widget->setColumnWidth(2, 80);
  table_widget->horizontalHeader()->setStretchLastSection(true);
  table_widget->verticalHeader()->hide();
  table_widget->sortByColumn(0, Qt::AscendingOrder);
  main_layout->addWidget(table_widget);

  // signals/slots
  QObject::connect(filter, &QLineEdit::textChanged, proxy_model, &QSortFilterProxyModel::setFilterFixedString);
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
    return (QString[]){"Name", "ID", "Count", "Bytes"}[section];
  else if (orientation == Qt::Vertical && role == Qt::DisplayRole) {
    // return QString::number(section);
  }
  return {};
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    auto it = std::next(can->can_msgs.begin(), index.row());
    if (it != can->can_msgs.end() && !it.value().empty()) {
      const auto &d = it.value().front();
      const QString &msg_id = it.key();
      switch (index.column()) {
        case 0: {
          auto msg = dbc()->msg(msg_id);
          QString name = msg ? msg->name.c_str() : "untitled";
          return name;
        }
        case 1: return msg_id;
        case 2: return can->counters[msg_id];
        case 3: return toHex(d.dat);
      }
    }
  } else if (role == Qt::UserRole) {
    return std::next(can->can_msgs.begin(), index.row()).key();
  }
  return {};
}

void MessageListModel::updateState() {
  int prev_row_count = row_count;
  row_count = can->can_msgs.size();
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
