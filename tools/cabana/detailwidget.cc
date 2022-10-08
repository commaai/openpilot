
#include "tools/cabana/detailwidget.h"

#include <QDebug>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QTimer>
#include <QVBoxLayout>
#include <bitset>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

inline const QString &getColor(int i) {
  static const QString SIGNAL_COLORS[] = {"#9FE2BF", "#40E0D0", "#6495ED", "#CCCCFF", "#FF7F50", "#FFBF00"};
  return SIGNAL_COLORS[i % std::size(SIGNAL_COLORS)];
}

// DetailWidget

DetailWidget::DetailWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  // title
  QHBoxLayout *title_layout = new QHBoxLayout();
  title_layout->addWidget(new QLabel("time:"));
  time_label = new QLabel(this);
  title_layout->addWidget(time_label);
  time_label->setStyleSheet("font-weight:bold");
  title_layout->addStretch();
  name_label = new QLabel(this);
  name_label->setStyleSheet("font-weight:bold;");
  title_layout->addWidget(name_label);
  title_layout->addStretch();
  edit_btn = new QPushButton(tr("Edit"), this);
  edit_btn->setVisible(false);
  title_layout->addWidget(edit_btn);
  main_layout->addLayout(title_layout);

  // binary view
  binary_view = new BinaryView(this);
  main_layout->addWidget(binary_view, 0, Qt::AlignTop);

  // signal header
  signals_header = new QWidget(this);
  QHBoxLayout *signals_header_layout = new QHBoxLayout(signals_header);
  signals_header_layout->addWidget(new QLabel(tr("Signals")));
  signals_header_layout->addStretch();
  QPushButton *add_sig_btn = new QPushButton(tr("Add signal"), this);
  signals_header_layout->addWidget(add_sig_btn);
  signals_header->setVisible(false);
  main_layout->addWidget(signals_header);

  // scroll area
  QWidget *container = new QWidget(this);
  QVBoxLayout *container_layout = new QVBoxLayout(container);
  signal_edit_layout = new QVBoxLayout();
  signal_edit_layout->setSpacing(2);
  container_layout->addLayout(signal_edit_layout);

  history_log = new HistoryLog(this);
  container_layout->addWidget(history_log);

  QScrollArea *scroll = new QScrollArea(this);
  scroll->setWidget(container);
  scroll->setWidgetResizable(true);
  scroll->setFrameShape(QFrame::NoFrame);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scroll->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

  main_layout->addWidget(scroll);

  QObject::connect(add_sig_btn, &QPushButton::clicked, this, &DetailWidget::addSignal);
  QObject::connect(edit_btn, &QPushButton::clicked, this, &DetailWidget::editMsg);
  QObject::connect(can, &CANMessages::updated, this, &DetailWidget::updateState);
}

void DetailWidget::setMessage(const QString &message_id) {
  msg_id = message_id;
  clearLayout(signal_edit_layout);

  if (auto msg = dbc()->msg(msg_id)) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      auto edit = new SignalEdit(i, msg_id, msg->sigs[i], getColor(i));
      signal_edit_layout->addWidget(edit);
      QObject::connect(edit, &SignalEdit::showChart, this, &DetailWidget::showChart);
    }
    name_label->setText(msg->name.c_str());
    signals_header->setVisible(true);
  } else {
    name_label->setText(tr("untitled"));
    signals_header->setVisible(false);
  }
  edit_btn->setVisible(true);

  binary_view->setMessage(msg_id);
  history_log->setMessage(msg_id);
}

void DetailWidget::updateState() {
  time_label->setText(QString::number(can->currentSec(), 'f', 3));
  if (msg_id.isEmpty()) return;

  binary_view->updateState();
  history_log->updateState();
}

void DetailWidget::editMsg() {
  EditMessageDialog dlg(msg_id, this);
  if (dlg.exec()) {
    setMessage(msg_id);
  }
}

void DetailWidget::addSignal() {
  AddSignalDialog dlg(msg_id, this);
  if (dlg.exec()) {
    setMessage(msg_id);
  }
}

// BinaryView

BinaryView::BinaryView(QWidget *parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  table = new QTableWidget(this);
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->horizontalHeader()->hide();
  table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  table->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(table);
  table->setColumnCount(9);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
}

void BinaryView::setMessage(const QString &message_id) {
  msg_id = message_id;
  const Msg *msg = dbc()->msg(msg_id);
  int row_count = msg ? msg->size : can->lastMessage(msg_id).dat.size();

  table->setRowCount(row_count);
  table->setColumnCount(9);
  for (int i = 0; i < table->rowCount(); ++i) {
    for (int j = 0; j < table->columnCount(); ++j) {
      auto item = new QTableWidgetItem();
      item->setFlags(item->flags() ^ Qt::ItemIsEditable);
      item->setTextAlignment(Qt::AlignCenter);
      if (j == 8) {
        QFont font;
        font.setBold(true);
        item->setFont(font);
      }
      table->setItem(i, j, item);
    }
  }

  // set background color
  if (msg) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      const auto &sig = msg->sigs[i];
      int start = sig.is_little_endian ? sig.start_bit : bigEndianBitIndex(sig.start_bit);
      for (int j = start; j <= start + sig.size - 1; ++j) {
        table->item(j / 8, j % 8)->setBackground(QColor(getColor(i)));
      }
    }
  }

  table->setFixedHeight(table->rowHeight(0) * table->rowCount() + table->horizontalHeader()->height() + 2);
  updateState();
}

void BinaryView::updateState() {
  const auto &binary = can->lastMessage(msg_id).dat;
  std::string s;
  for (int j = 0; j < binary.size(); ++j) {
    s += std::bitset<8>(binary[j]).to_string();
  }

  setUpdatesEnabled(false);
  char hex[3] = {'\0'};
  for (int i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      table->item(i, j)->setText(QChar(s[i * 8 + j]));
    }
    sprintf(&hex[0], "%02X", (unsigned char)binary[i]);
    table->item(i, 8)->setText(hex);
  }
  setUpdatesEnabled(true);
}

// HistoryLog

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
    } else if (role == Qt::DisplayRole) {
      auto msg = dbc()->msg(msg_id);
      if (msg && section < msg->sigs.size())
        return QString::fromStdString(msg->sigs[section].name);
    }
  } else if (role == Qt::DisplayRole && section < values.size()) {
    return QString::number(values[section].first);
  }
  return {};
}

void HistoryLogModel::clear() {
  values.clear();
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
      for (int i = 0; i < msg->sigs.size(); ++i) {
        double value = get_raw_value((uint8_t *)can_data.dat.begin(), can_data.dat.size(), msg->sigs[i]);
        data.append(QString::number(value));
      }
    } else {
      data.append(toHex(can_data.dat));
    }
    values.push_front({can_data.ts, data});
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
  table->verticalHeader()->setStyleSheet("QHeaderView::section {padding-left: 5px; padding-right: 5px;}");
  main_layout->addWidget(table);

  QObject::connect(can, &CANMessages::rangeChanged, model, &HistoryLogModel::clear);
}

void HistoryLog::setMessage(const QString &message_id) {
  msg_id = message_id;
  model->setMessage(message_id);
  model->updateState();
}

void HistoryLog::updateState() {
  model->updateState();
}

// EditMessageDialog

EditMessageDialog::EditMessageDialog(const QString &msg_id, QWidget *parent) : msg_id(msg_id), QDialog(parent) {
  setWindowTitle(tr("Edit message"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QFormLayout *form_layout = new QFormLayout();
  form_layout->addRow("ID", new QLabel(msg_id));

  const auto msg = dbc()->msg(msg_id);
  name_edit = new QLineEdit(this);
  name_edit->setText(msg ? msg->name.c_str() : "untitled");
  form_layout->addRow(tr("Name"), name_edit);

  size_spin = new QSpinBox(this);
  size_spin->setValue(msg ? msg->size : can->lastMessage(msg_id).dat.size());
  form_layout->addRow(tr("Size"), size_spin);

  main_layout->addLayout(form_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, this, &EditMessageDialog::save);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void EditMessageDialog::save() {
  const QString name = name_edit->text();
  if (size_spin->value() <= 0 || name_edit->text().isEmpty() || name == tr("untitled"))
    return;

  dbc()->updateMsg(msg_id, name, size_spin->value());
  QDialog::accept();
}
