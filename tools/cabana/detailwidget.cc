#include "tools/cabana/detailwidget.h"

#include <QDialogButtonBox>
#include <QFontDatabase>
#include <QFormLayout>
#include <QHeaderView>
#include <QScrollBar>
#include <QTimer>
#include <QVBoxLayout>

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
  scroll = new ScrollArea(this);
  QWidget *container = new QWidget(this);
  container->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  QVBoxLayout *container_layout = new QVBoxLayout(container);
  signal_edit_layout = new QVBoxLayout();
  signal_edit_layout->setSpacing(2);
  container_layout->addLayout(signal_edit_layout);

  scroll->setWidget(container);
  scroll->setWidgetResizable(true);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(scroll);

  history_log = new HistoryLog(this);
  main_layout->addWidget(history_log);

  QObject::connect(add_sig_btn, &QPushButton::clicked, this, &DetailWidget::addSignal);
  QObject::connect(edit_btn, &QPushButton::clicked, this, &DetailWidget::editMsg);
  QObject::connect(can, &CANMessages::updated, this, &DetailWidget::updateState);
}

void DetailWidget::setMessage(const QString &message_id) {
  msg_id = message_id;
  for (auto f : signal_forms) {
    f->deleteLater();
  }
  signal_forms.clear();

  if (auto msg = dbc()->msg(msg_id)) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      auto form = new SignalEdit(i, msg_id, msg->sigs[i], getColor(i));
      signal_edit_layout->addWidget(form);
      QObject::connect(form, &SignalEdit::showChart, this, &DetailWidget::showChart);
      QObject::connect(form, &SignalEdit::showFormClicked, this, &DetailWidget::showForm);
      signal_forms.push_back(form);
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

void DetailWidget::showForm() {
  SignalEdit *sender = qobject_cast<SignalEdit *>(QObject::sender());
  if (sender->isFormVisible()) {
    sender->setFormVisible(false);
  } else {
    for (auto f : signal_forms) {
      f->setFormVisible(f == sender);
      if (f == sender) {
        // scroll to header
        QTimer::singleShot(0, [=]() {
          const QPoint p = f->mapTo(scroll, QPoint(0, 0));
          scroll->verticalScrollBar()->setValue(p.y() + scroll->verticalScrollBar()->value());
        });
      }
    }
  }
}

// BinaryView

void BinaryViewModel::setMessage(const QString &message_id) {
  msg_id = message_id;
  beginResetModel();
  msg_id = message_id;
  const Msg *msg = dbc()->msg(msg_id);
  row_count = msg ? msg->size : can->lastMessage(msg_id).dat.size();
  items.clear();
  items.resize(row_count * column_count);
  if (msg) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      const auto &sig = msg->sigs[i];
      int start = sig.is_little_endian ? sig.start_bit : bigEndianBitIndex(sig.start_bit);
      for (int j = start; j <= start + sig.size - 1; ++j) {
        int idx = column_count * (j/(column_count-1)) +  j % (column_count-1);
        if (j == sig.msb) {
          items[idx].is_msb = true;
        } else if (j == sig.lsb) {
          items[idx].is_lsb = true;
        }
        items[idx].bg_color = QColor(getColor(i));
      }
    }
  }
  // table->setFixedHeight(table->rowHeight(0) * std::min(row_count, 8) + 2);
  endResetModel();
  updateState();
}

QModelIndex BinaryViewModel::index(int row, int column, const QModelIndex &parent) const {
  return createIndex(row, column, (void*)&items[row * column_count + column]);
}

void BinaryViewModel::updateState() {
  if (msg_id.isEmpty()) return;

  const auto &binary = can->lastMessage(msg_id).dat;
  char hex[3] = {'\0'};
  for (int i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < column_count - 1; ++j) {
      items[i * column_count + j].val = QChar((binary[i] >> (7 - j)) & 1 ? '1' : '0');
    }
    hex[0] = toHex(binary[i] >> 4);
    hex[1] = toHex(binary[i] & 0xf);
    items[i * column_count + 8].val = hex;
  }

  emit dataChanged(index(0, 0), index(row_count - 1, 8));
}

QVariant BinaryViewModel::headerData(int section, Qt::Orientation orientation, int role) const {
  return role == Qt::DisplayRole ? QVariant(section) : QVariant();
}

BinaryView::BinaryView(QWidget *parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  table = new QTableView(this);
  model = new BinaryViewModel(this);
  table->setModel(model);
  table->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->horizontalHeader()->hide();
  table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  table->setItemDelegate(new BinaryItemDelegate(this));
  main_layout->addWidget(table);
}

void BinaryView::setMessage(const QString &message_id) {
  model->setMessage(message_id);
}

void BinaryView::updateState() {
  model->updateState();
}

void BinaryItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  painter->save();
  QStyleOptionViewItem opt = option;
  auto item = (const BinaryViewModel::Item *)index.internalPointer();
  painter->fillRect(opt.rect, item->bg_color);
  if (index.column() == 8) {
    QFont f;
    f.setBold(true);
    painter->setFont(f);
  }
  painter->drawText(opt.rect, Qt::AlignCenter, item->val);
  if (item->is_msb || item->is_lsb) {
    QFont f;
    f.setPointSize(8);
    painter->setFont(f);
    painter->setPen(Qt::white);
    painter->drawText(opt.rect, item->is_msb ? "MSB" : "LSB");
  }
  painter->restore();
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

  setFixedWidth(parent->width() * 0.9);

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

// ScrollArea

bool ScrollArea::eventFilter(QObject *obj, QEvent *ev) {
  if (obj == widget() && ev->type() == QEvent::Resize) {
    int height = widget()->height() + 4;
    setMinimumHeight(height > 480 ? 480 : height);
    setMaximumHeight(height);
  }
  return QScrollArea::eventFilter(obj, ev);
}

void ScrollArea::setWidget(QWidget *w) {
  QScrollArea::setWidget(w);
  w->installEventFilter(this);
}
