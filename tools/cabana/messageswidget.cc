#include "tools/cabana/messageswidget.h"

#include <QCompleter>
#include <QDialogButtonBox>
#include <QFontDatabase>
#include <QHeaderView>
#include <QLineEdit>
#include <QPushButton>
#include <QSortFilterProxyModel>
#include <QTextEdit>
#include <QVBoxLayout>

#include "tools/cabana/dbcmanager.h"

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  // DBC file selector
  QHBoxLayout *dbc_file_layout = new QHBoxLayout();
  dbc_combo = new QComboBox(this);
  auto dbc_names = dbc()->allDBCNames();
  for (const auto &name : dbc_names) {
    dbc_combo->addItem(QString::fromStdString(name));
  }
  dbc_combo->model()->sort(0);
  dbc_combo->setEditable(true);
  dbc_combo->setCurrentText(QString());
  dbc_combo->setInsertPolicy(QComboBox::NoInsert);
  dbc_combo->completer()->setCompletionMode(QCompleter::PopupCompletion);
  QFont font;
  font.setBold(true);
  dbc_combo->lineEdit()->setFont(font);
  dbc_file_layout->addWidget(dbc_combo);

  QPushButton *load_from_paste = new QPushButton(tr("Load from paste"), this);
  dbc_file_layout->addWidget(load_from_paste);

  dbc_file_layout->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save DBC"), this);
  dbc_file_layout->addWidget(save_btn);
  main_layout->addLayout(dbc_file_layout);

  // message filter
  QLineEdit *filter = new QLineEdit(this);
  filter->setClearButtonEnabled(true);
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
  QObject::connect(dbc_combo, SIGNAL(activated(const QString &)), SLOT(dbcSelectionChanged(const QString &)));
  QObject::connect(load_from_paste, &QPushButton::clicked, this, &MessagesWidget::loadFromPaste);
  QObject::connect(save_btn, &QPushButton::clicked, [=]() {
    // TODO: save DBC to file
  });
  QObject::connect(table_widget->selectionModel(), &QItemSelectionModel::currentChanged, [=](const QModelIndex &current, const QModelIndex &previous) {
    if (current.isValid()) {
      emit msgSelectionChanged(current.data(Qt::UserRole).toString());
    }
  });

  // For test purpose
  dbc_combo->setCurrentText("toyota_nodsu_pt_generated");
}

void MessagesWidget::dbcSelectionChanged(const QString &dbc_file) {
  dbc()->open(dbc_file);
  // TODO: reset model?
  table_widget->sortByColumn(0, Qt::AscendingOrder);
}

void MessagesWidget::loadFromPaste() {
  LoadDBCDialog dlg(this);
  if (dlg.exec()) {
    dbc()->open("from paste", dlg.dbc_edit->toPlainText());
    dbc_combo->setCurrentText("loaded from paste");
  }
}

// MessageListModel

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    return (QString[]){"Name", "ID", "Count", "Bytes"}[section];
  return {};
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    auto it = std::next(can->can_msgs.begin(), index.row());
    if (it != can->can_msgs.end() && !it.value().empty()) {
      const QString &msg_id = it.key();
      switch (index.column()) {
        case 0: {
          auto msg = dbc()->msg(msg_id);
          return msg ? msg->name.c_str() : "untitled";
        }
        case 1: return msg_id;
        case 2: return can->counters[msg_id];
        case 3: return toHex(it.value().front().dat);
      }
    }
  } else if (role == Qt::UserRole) {
    return std::next(can->can_msgs.begin(), index.row()).key();
  } else if (role == Qt::FontRole) {
    if (index.column() == 3) {
      return QFontDatabase::systemFont(QFontDatabase::FixedFont);
    }
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
    emit dataChanged(index(0, 0), index(row_count - 1, 3), {Qt::DisplayRole});
  }
}

LoadDBCDialog::LoadDBCDialog(QWidget *parent) : QDialog(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  dbc_edit = new QTextEdit(this);
  dbc_edit->setAcceptRichText(false);
  dbc_edit->setPlaceholderText(tr("paste DBC file here"));
  main_layout->addWidget(dbc_edit);
  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);

  setFixedWidth(640);
  connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}
