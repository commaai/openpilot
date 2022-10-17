#include "tools/cabana/detailwidget.h"

#include <QDialogButtonBox>
#include <QFontDatabase>
#include <QFormLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QScrollBar>
#include <QTimer>

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

  // signals
  signals_container = new QWidget(this);
  signals_container->setLayout(new QVBoxLayout);
  signals_container->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

  scroll = new ScrollArea(this);
  scroll->setWidget(signals_container);
  scroll->setWidgetResizable(true);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(scroll);

  // history log
  history_log = new HistoryLog(this);
  main_layout->addWidget(history_log);

  QObject::connect(edit_btn, &QPushButton::clicked, this, &DetailWidget::editMsg);
  QObject::connect(binary_view, &BinaryView::cellsSelected, this, &DetailWidget::addSignal);
  QObject::connect(can, &CANMessages::updated, this, &DetailWidget::updateState);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &DetailWidget::dbcMsgChanged);
}

void DetailWidget::setMessage(const QString &message_id) {
  if (msg_id != message_id) {
    msg_id = message_id;
    dbcMsgChanged();
  }
}

void DetailWidget::dbcMsgChanged() {
  if (msg_id.isEmpty()) return;

  qDeleteAll(signals_container->findChildren<SignalEdit *>());
  QString msg_name = tr("untitled");
  if (auto msg = dbc()->msg(msg_id)) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      auto form = new SignalEdit(i, msg_id, msg->sigs[i]);
      signals_container->layout()->addWidget(form);
      QObject::connect(form, &SignalEdit::showChart, [this, sig = &msg->sigs[i]]() { emit showChart(msg_id, sig); });
      QObject::connect(form, &SignalEdit::showFormClicked, this, &DetailWidget::showForm);
      QObject::connect(form, &SignalEdit::remove, this, &DetailWidget::removeSignal);
      QObject::connect(form, &SignalEdit::save, this, &DetailWidget::saveSignal);
    }
    msg_name = msg->name.c_str();
  }
  edit_btn->setVisible(true);
  name_label->setText(msg_name);

  binary_view->setMessage(msg_id);
  history_log->setMessage(msg_id);
}

void DetailWidget::updateState() {
  time_label->setText(QString::number(can->currentSec(), 'f', 3));
  if (msg_id.isEmpty()) return;

  binary_view->updateState();
  history_log->updateState();
}

void DetailWidget::showForm() {
  SignalEdit *sender = qobject_cast<SignalEdit *>(QObject::sender());
  for (auto f : signals_container->findChildren<SignalEdit *>()) {
    f->setFormVisible(f == sender && !f->isFormVisible());
    if (f == sender) {
      QTimer::singleShot(0, [=]() { scroll->ensureWidgetVisible(f); });
    }
  }
}

void DetailWidget::editMsg() {
  auto msg = dbc()->msg(msg_id);
  QString name = msg ? msg->name.c_str() : "untitled";
  int size = msg ? msg->size : can->lastMessage(msg_id).dat.size();
  EditMessageDialog dlg(msg_id, name, size, this);
  if (dlg.exec()) {
    dbc()->updateMsg(msg_id, dlg.name_edit->text(), dlg.size_spin->value());
    dbcMsgChanged();
  }
}

void DetailWidget::addSignal(int start_bit, int size) {
  if (dbc()->msg(msg_id)) {
    AddSignalDialog dlg(msg_id, start_bit, size, this);
    if (dlg.exec()) {
      dbc()->addSignal(msg_id, dlg.form->getSignal());
      dbcMsgChanged();
    }
  }
}

void DetailWidget::saveSignal() {
  SignalEdit *sig_form = qobject_cast<SignalEdit *>(QObject::sender());
  auto s = sig_form->form->getSignal();
  dbc()->updateSignal(msg_id, sig_form->sig_name, s);
  // update binary view and history log
  binary_view->setMessage(msg_id);
  history_log->setMessage(msg_id);
}

void DetailWidget::removeSignal() {
  SignalEdit *sig_form = qobject_cast<SignalEdit *>(QObject::sender());
  QString text = tr("Are you sure you want to remove signal '%1'").arg(sig_form->sig_name);
  if (QMessageBox::Yes == QMessageBox::question(this, tr("Remove signal"), text)) {
    dbc()->removeSignal(msg_id, sig_form->sig_name);
    dbcMsgChanged();
  }
}

// BinaryView

BinaryView::BinaryView(QWidget *parent) : QTableWidget(parent) {
  horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  horizontalHeader()->hide();
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setColumnCount(9);

  // replace selection model
  auto old_model = selectionModel();
  setSelectionModel(new BinarySelectionModel(model()));
  delete old_model;
}

void BinaryView::mouseReleaseEvent(QMouseEvent *event) {
  QTableWidget::mouseReleaseEvent(event);

  if (auto items = selectedItems(); !items.isEmpty()) {
    int start_bit = items.first()->row() * 8 + items.first()->column();
    int size = items.back()->row() * 8 + items.back()->column() - start_bit + 1;
    emit cellsSelected(start_bit, size);
  }
}

void BinaryView::setMessage(const QString &message_id) {
  msg_id = message_id;
  if (msg_id.isEmpty()) return;

  const Msg *msg = dbc()->msg(msg_id);
  int row_count = msg ? msg->size : can->lastMessage(msg_id).dat.size();
  setRowCount(row_count);
  setColumnCount(9);
  for (int i = 0; i < rowCount(); ++i) {
    for (int j = 0; j < columnCount(); ++j) {
      auto item = new QTableWidgetItem();
      item->setFlags(item->flags() ^ Qt::ItemIsEditable);
      item->setTextAlignment(Qt::AlignCenter);
      if (j == 8) {
        QFont font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
        font.setBold(true);
        item->setFont(font);
        item->setFlags(item->flags() ^ Qt::ItemIsSelectable);
      }
      setItem(i, j, item);
    }
  }

  // set background color
  if (msg) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      const auto &sig = msg->sigs[i];
      int start = sig.is_little_endian ? sig.start_bit : bigEndianBitIndex(sig.start_bit);
      for (int j = start; j <= std::min(start + sig.size - 1, rowCount() * columnCount() - 1); ++j) {
        item(j / 8, j % 8)->setBackground(QColor(getColor(i)));
      }
    }
  }

  setFixedHeight(rowHeight(0) * std::min(row_count, 8) + 2);
  clearSelection();
  updateState();
}

void BinaryView::updateState() {
  const auto &binary = can->lastMessage(msg_id).dat;
  setUpdatesEnabled(false);
  char hex[3] = {'\0'};
  for (int i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      item(i, j)->setText(QChar((binary[i] >> (7 - j)) & 1 ? '1' : '0'));
    }
    hex[0] = toHex(binary[i] >> 4);
    hex[1] = toHex(binary[i] & 0xf);
    item(i, 8)->setText(hex);
  }
  setUpdatesEnabled(true);
}

void BinarySelectionModel::select(const QItemSelection &selection, QItemSelectionModel::SelectionFlags command) {
  QItemSelection new_selection = selection;
  if (auto indexes = selection.indexes(); !indexes.isEmpty()) {
    auto [begin_idx, end_idx] = (QModelIndex[]){indexes.first(), indexes.back()};
    for (int row = begin_idx.row(); row <= end_idx.row(); ++row) {
      int left_col = (row == begin_idx.row()) ? begin_idx.column() : 0;
      int right_col = (row == end_idx.row()) ? end_idx.column() : 7;
      new_selection.merge({model()->index(row, left_col), model()->index(row, right_col)}, command);
    }
  }
  QItemSelectionModel::select(new_selection, command);
}

// EditMessageDialog

EditMessageDialog::EditMessageDialog(const QString &msg_id, const QString &title, int size, QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Edit message"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QFormLayout *form_layout = new QFormLayout();
  form_layout->addRow("ID", new QLabel(msg_id));

  name_edit = new QLineEdit(title, this);
  form_layout->addRow(tr("Name"), name_edit);

  size_spin = new QSpinBox(this);
  // TODO: limit the maximum?
  size_spin->setMinimum(1);
  size_spin->setValue(size);
  form_layout->addRow(tr("Size"), size_spin);

  main_layout->addLayout(form_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);
  setFixedWidth(parent->width() * 0.9);

  connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
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
