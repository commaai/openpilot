
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

  name_label = new QLabel(this);
  name_label->setStyleSheet("font-weight:bold;");
  name_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  name_label->setAlignment(Qt::AlignCenter);
  main_layout->addWidget(name_label);

  // title
  QHBoxLayout *title_layout = new QHBoxLayout();
  time_label = new QLabel(this);
  title_layout->addWidget(time_label);
  title_layout->addStretch();

  edit_btn = new QPushButton(tr("Edit"), this);
  edit_btn->setVisible(false);
  title_layout->addWidget(edit_btn);
  main_layout->addLayout(title_layout);

  // binary view
  binary_view = new BinaryView(this);
  main_layout->addWidget(binary_view);

  // signal header
  signals_header = new QWidget(this);
  QHBoxLayout *signals_header_layout = new QHBoxLayout(signals_header);
  signals_header_layout->addWidget(new QLabel(tr("Signals")));
  signals_header_layout->addStretch();
  QPushButton *add_sig_btn = new QPushButton(tr("Add signal"), this);
  add_sig_btn->setVisible(false);
  signals_header_layout->addWidget(add_sig_btn);
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
  QObject::connect(parser, &Parser::updated, this, &DetailWidget::updateState);
}

void DetailWidget::updateDBCMsg(const QString &message_id) {
  msg_id = message_id;
  clearLayout(signal_edit_layout);

  if (auto msg = parser->getDBCMsg(msg_id)) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      signal_edit_layout->addWidget(new SignalEdit(i, msg_id, msg->sigs[i], getColor(i)));
    }
    name_label->setText(msg->name.c_str());
    signals_header->setVisible(true);
  } else {
    name_label->setText(tr("untitled"));
    signals_header->setVisible(false);
  }
  edit_btn->setVisible(true);

  binary_view->updateDBCMsg(msg_id);
  history_log->updateDBCMsg(msg_id);
}

void DetailWidget::updateState() {
  if (msg_id.isEmpty()) return;

  const auto &can_data = parser->canMsgs(msg_id).back();
  time_label->setText(QString("time: %1").arg(can_data.ts, 0, 'f', 3));
  binary_view->setData(can_data.dat);
  history_log->updateState();
}

void DetailWidget::editMsg() {
  EditMessageDialog dlg(msg_id, this);
  if (dlg.exec()) {
    updateDBCMsg(msg_id);
  }
}

void DetailWidget::addSignal() {
  AddSignalDialog dlg(msg_id, this);
  if (dlg.exec()) {
    updateDBCMsg(msg_id);
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

void BinaryView::updateDBCMsg(const QString &message_id) {
  const Msg *msg = parser->getDBCMsg(message_id);
  int row_count = msg ? msg->size : parser->canMsgs(message_id).back().dat.size();

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

  setFixedHeight(table->rowHeight(0) * table->rowCount() + 25);
}

void BinaryView::setData(const QByteArray &binary) {
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

HistoryLog::HistoryLog(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  table = new QTableWidget(this);
  table->horizontalHeader()->setStretchLastSection(true);
  table->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table->setFocusPolicy(Qt::NoFocus);
  table->setSelectionMode(QAbstractItemView::NoSelection);
  table->setSelectionMode(QAbstractItemView::NoSelection);
  table->setStyleSheet("QTableView::item { border:0px; padding-left:5px; padding-right:5px; }");
  table->verticalHeader()->setStyleSheet("QHeaderView::section {padding-left: 5px; padding-right: 5px;}");
  main_layout->addWidget(table);
}

void HistoryLog::updateDBCMsg(const QString &message_id) {
  msg_id = message_id;
  previous_count = 0;
  table->clear();
  const auto msg = parser->getDBCMsg(msg_id);
  if (msg && !msg->sigs.empty()) {
    table->setColumnCount(msg->sigs.size());
    for (int i = 0; i < msg->sigs.size(); ++i) {
      auto item = new QTableWidgetItem(msg->sigs[i].name.c_str());
      item->setBackground(QColor(getColor(i)));
      table->setHorizontalHeaderItem(i, item);
    }
  } else {
    table->setColumnCount(1);
    table->setHorizontalHeaderItem(0, new QTableWidgetItem("data"));
  }
}

void HistoryLog::updateState() {
  auto model = table->model();
  const auto &can_msgs = parser->canMsgs(msg_id);
  const auto msg = parser->getDBCMsg(msg_id);
  for (const auto &can_data : can_msgs) {
    if (can_data.count <= previous_count)
      continue;

    table->insertRow(0);
    table->setVerticalHeaderItem(0, new QTableWidgetItem(QString::number(can_data.ts, 'f', 2)));
    if (msg && !msg->sigs.empty()) {
      for (int i = 0; i < msg->sigs.size(); ++i) {
        double value = get_raw_value((uint8_t *)can_data.dat.begin(), can_data.dat.size(), msg->sigs[i]);
        model->setData(model->index(0, i), QString::number(value));
      }
    } else {
      model->setData(model->index(0, 0), toHex(can_data.dat));
    }
    previous_count = can_data.count;
  }

  if (table->rowCount() > CAN_MSG_LOG_SIZE)
    table->setRowCount(CAN_MSG_LOG_SIZE);
}

// EditMessageDialog

EditMessageDialog::EditMessageDialog(const QString &msg_id, QWidget *parent) : msg_id(msg_id), QDialog(parent) {
  setWindowTitle(tr("Edit message"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QFormLayout *form_layout = new QFormLayout();
  form_layout->addRow("ID", new QLabel(msg_id));

  const auto msg = parser->getDBCMsg(msg_id);
  name_edit = new QLineEdit(this);
  name_edit->setText(msg ? msg->name.c_str() : "untitled");
  form_layout->addRow(tr("Name"), name_edit);

  size_spin = new QSpinBox(this);
  size_spin->setValue(msg ? msg->size : parser->canMsgs(msg_id).back().dat.size());
  form_layout->addRow(tr("Size"), size_spin);

  main_layout->addLayout(form_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, this, &EditMessageDialog::save);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void EditMessageDialog::save() {
  if (size_spin->value() <= 0 || name_edit->text().isEmpty()) return;

  if (auto msg = const_cast<Msg *>(parser->getDBCMsg(msg_id))) {
    msg->name = name_edit->text().toStdString();
    msg->size = size_spin->value();
  } else {
    Msg m = {};
    m.address = Parser::addressFromId(msg_id);
    m.name = name_edit->text().toStdString();
    m.size = size_spin->value();
    parser->addNewMsg(m);
  }
  QDialog::accept();
}
