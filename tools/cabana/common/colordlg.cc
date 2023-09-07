#include "tools/cabana/common/colordlg.h"

#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QVBoxLayout>

SignalColorDlg::SignalColorDlg(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Choose color for signal"));
  QHBoxLayout *hl = new QHBoxLayout;
  hl->addWidget(list = new QListWidget(this));
  list->setContentsMargins(9, 9, 9, 9);
  hl->addWidget(color_picker = new QColorDialog(this));
  color_picker->setOption(QColorDialog::NoButtons);
  color_picker->setWindowFlags(Qt::Widget);

  auto button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(hl);
  main_layout->addWidget(button_box);

  connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  connect(button_box, &QDialogButtonBox::accepted, [this]() {
    for (int i = 0; i < list->count(); ++i) {
      auto item = static_cast<Item *>(list->item(i));
      dbc()->updateSignalColor(item->msg_id, item->text(), item->color);
    }
    accept();
  });
  connect(color_picker, &QColorDialog::currentColorChanged, [this](const QColor &color) {
    if (auto current = static_cast<Item *>(list->currentItem())) {
      current->setColor(color);
    }
  });
  connect(list, &QListWidget::currentItemChanged, [this](QListWidgetItem *current, QListWidgetItem *) {
    color_picker->setCurrentColor(static_cast<Item *>(current)->color);
  });
}

void SignalColorDlg::addSignal(const MessageId &msg_id, const cabana::Signal *sig) {
  auto new_item = new Item(msg_id, sig, list);
  if (!list->currentItem()) list->setCurrentItem(new_item);
}
