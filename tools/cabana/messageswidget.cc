#include "tools/cabana/messageswidget.h"

#include <QCompleter>
#include <QDialogButtonBox>
#include <QFontDatabase>
#include <QHeaderView>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>

#include "tools/cabana/dbcmanager.h"

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

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
  QObject::connect(can, &CANMessages::eventsMerged, this, &MessagesWidget::loadDBCFromFingerprint);
  QObject::connect(can, &CANMessages::updated, [this]() { model->updateState(); });
  QObject::connect(dbc_combo, SIGNAL(activated(const QString &)), SLOT(loadDBCFromName(const QString &)));
  QObject::connect(load_from_paste, &QPushButton::clicked, this, &MessagesWidget::loadDBCFromPaste);
  QObject::connect(save_btn, &QPushButton::clicked, [=]() {
    // TODO: save DBC to file
  });
  QObject::connect(table_widget->selectionModel(), &QItemSelectionModel::currentChanged, [=](const QModelIndex &current, const QModelIndex &previous) {
    if (current.isValid()) {
      emit msgSelectionChanged(current.data(Qt::UserRole).toString());
    }
  });

  QFile json_file("./car_fingerprint_to_dbc.json");
  if(json_file.open(QIODevice::ReadOnly)) {
    fingerprint_to_dbc = QJsonDocument::fromJson(json_file.readAll());
  }
}

void MessagesWidget::loadDBCFromName(const QString &name) {
  dbc()->open(name);
  dbc_combo->setCurrentText(name);
  // refresh model
  model->updateState();
}

void MessagesWidget::loadDBCFromPaste() {
  LoadDBCDialog dlg(this);
  if (dlg.exec()) {
    dbc()->open("from paste", dlg.dbc_edit->toPlainText());
    dbc_combo->setCurrentText("loaded from paste");
  }
}

void MessagesWidget::loadDBCFromFingerprint() {
  auto fingerprint = can->carFingerprint();
  if (!fingerprint.isEmpty() && dbc()->name().isEmpty())  {
    auto dbc_name = fingerprint_to_dbc[fingerprint];
    if (dbc_name !=  QJsonValue::Undefined) {
      loadDBCFromName(dbc_name.toString());
    }
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
    const auto &m = msgs[index.row()];
    switch (index.column()) {
      case 0: return m->name;
      case 1: return m->id;
      case 2: return can->counters[m->id];
      case 3: return toHex(can->lastMessage(m->id).dat);
    }
  } else if (role == Qt::UserRole) {
    return msgs[index.row()]->id;
  } else if (role == Qt::FontRole) {
    if (index.column() == 3) {
      return QFontDatabase::systemFont(QFontDatabase::FixedFont);
    }
  }
  return {};
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
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      uint32_t lcount = can->counters[l->id], rcount = can->counters[r->id];
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
    emit dataChanged(index(0, 0), index(msgs.size() - 1, 3), {Qt::DisplayRole});
  }
}

void MessageListModel::sort(int column, Qt::SortOrder order) {
  if (column != columnCount() - 1) {
    sort_column = column;
    sort_order = order;
    updateState(true);
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
