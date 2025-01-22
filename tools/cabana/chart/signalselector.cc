#include "tools/cabana/chart/signalselector.h"

#include <QCompleter>
#include <QDialogButtonBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

#include "tools/cabana/streams/abstractstream.h"

SignalSelector::SignalSelector(QString title, QWidget *parent) : QDialog(parent) {
  setWindowTitle(title);
  QGridLayout *main_layout = new QGridLayout(this);

  // left column
  main_layout->addWidget(new QLabel(tr("Available Signals")), 0, 0);
  main_layout->addWidget(msgs_combo = new QComboBox(this), 1, 0);
  msgs_combo->setEditable(true);
  msgs_combo->lineEdit()->setPlaceholderText(tr("Select a msg..."));
  msgs_combo->setInsertPolicy(QComboBox::NoInsert);
  msgs_combo->completer()->setCompletionMode(QCompleter::PopupCompletion);
  msgs_combo->completer()->setFilterMode(Qt::MatchContains);

  main_layout->addWidget(available_list = new QListWidget(this), 2, 0);

  // buttons
  QVBoxLayout *btn_layout = new QVBoxLayout();
  QPushButton *add_btn = new QPushButton(utils::icon("chevron-right"), "", this);
  add_btn->setEnabled(false);
  QPushButton *remove_btn = new QPushButton(utils::icon("chevron-left"), "", this);
  remove_btn->setEnabled(false);
  btn_layout->addStretch(0);
  btn_layout->addWidget(add_btn);
  btn_layout->addWidget(remove_btn);
  btn_layout->addStretch(0);
  main_layout->addLayout(btn_layout, 0, 1, 3, 1);

  // right column
  main_layout->addWidget(new QLabel(tr("Selected Signals")), 0, 2);
  main_layout->addWidget(selected_list = new QListWidget(this), 1, 2, 2, 1);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox, 3, 2);

  for (const auto &[id, _] : can->lastMessages()) {
    if (auto m = dbc()->msg(id)) {
      msgs_combo->addItem(QString("%1 (%2)").arg(m->name).arg(id.toString()), QVariant::fromValue(id));
    }
  }
  msgs_combo->model()->sort(0);
  msgs_combo->setCurrentIndex(-1);

  QObject::connect(msgs_combo, qOverload<int>(&QComboBox::currentIndexChanged), this, &SignalSelector::updateAvailableList);
  QObject::connect(available_list, &QListWidget::currentRowChanged, [=](int row) { add_btn->setEnabled(row != -1); });
  QObject::connect(selected_list, &QListWidget::currentRowChanged, [=](int row) { remove_btn->setEnabled(row != -1); });
  QObject::connect(available_list, &QListWidget::itemDoubleClicked, this, &SignalSelector::add);
  QObject::connect(selected_list, &QListWidget::itemDoubleClicked, this, &SignalSelector::remove);
  QObject::connect(add_btn, &QPushButton::clicked, [this]() { if (auto item = available_list->currentItem()) add(item); });
  QObject::connect(remove_btn, &QPushButton::clicked, [this]() { if (auto item = selected_list->currentItem()) remove(item); });
  QObject::connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  QObject::connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void SignalSelector::add(QListWidgetItem *item) {
  auto it = (ListItem *)item;
  addItemToList(selected_list, it->msg_id, it->sig, true);
  delete item;
}

void SignalSelector::remove(QListWidgetItem *item) {
  auto it = (ListItem *)item;
  if (it->msg_id == msgs_combo->currentData().value<MessageId>()) {
    addItemToList(available_list, it->msg_id, it->sig);
  }
  delete item;
}

void SignalSelector::updateAvailableList(int index) {
  if (index == -1) return;
  available_list->clear();
  MessageId msg_id = msgs_combo->itemData(index).value<MessageId>();
  auto selected_items = seletedItems();
  for (auto s : dbc()->msg(msg_id)->getSignals()) {
    bool is_selected = std::any_of(selected_items.begin(), selected_items.end(),
                                   [sig = s, &msg_id](auto it) { return it->msg_id == msg_id && it->sig == sig; });
    if (!is_selected) {
      addItemToList(available_list, msg_id, s);
    }
  }
}

void SignalSelector::addItemToList(QListWidget *parent, const MessageId id, const cabana::Signal *sig, bool show_msg_name) {
  QString text = QString("<span style=\"color:%0;\">■ </span> %1").arg(sig->color.name(), sig->name);
  if (show_msg_name) text += QString(" <font color=\"gray\">%0 %1</font>").arg(msgName(id), id.toString());

  QLabel *label = new QLabel(text);
  label->setContentsMargins(5, 0, 5, 0);
  auto new_item = new ListItem(id, sig, parent);
  new_item->setSizeHint(label->sizeHint());
  parent->setItemWidget(new_item, label);
}

QList<SignalSelector::ListItem *> SignalSelector::seletedItems() {
  QList<SignalSelector::ListItem *> ret;
  for (int i = 0; i < selected_list->count(); ++i) ret.push_back((ListItem *)selected_list->item(i));
  return ret;
}
