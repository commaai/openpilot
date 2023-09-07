#include "tools/cabana/common/colordlg.h"

#include <QDebug>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>

#include "tools/cabana/dbc/dbcmanager.h"

SignalColorDlg::SignalColorDlg(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Choose color for signal"));
  QHBoxLayout *hl = new QHBoxLayout;
  hl->addWidget(signal_list = new QListWidget(this));
  signal_list->setSelectionMode(QAbstractItemView::SingleSelection);

  color_picker = new QColorDialog(this);
  color_picker->setOption(QColorDialog::NoButtons);
  color_picker->setWindowFlags(Qt::Widget);
  hl->addWidget(color_picker);

  auto button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(hl);
  main_layout->addWidget(button_box);

  QObject::connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(button_box, &QDialogButtonBox::accepted, [this]() {
    for (int i = 0; i < signal_list->count(); ++i) {
      auto item = static_cast<ListItem *>(signal_list->item(i));
      dbc()->updateSignalColor(item->msg_id, item->sig->name, item->color);
    }
    accept();
  });

  QObject::connect(color_picker, &QColorDialog::currentColorChanged, [this](const QColor &color) {
    if (auto current = static_cast<ListItem *>(signal_list->currentItem())) {
      current->color = color;
      QString text = QString("<span style=\"color:%0;\">■ </span> %1").arg(current->color.name(), current->sig->name);
      static_cast<QLabel *>(signal_list->itemWidget(current))->setText(text);
    }
  });

  QObject::connect(signal_list, &QListWidget::currentItemChanged, [this](QListWidgetItem *current, QListWidgetItem *) {
    color_picker->setCurrentColor(static_cast<ListItem *>(current)->color);
  });
}

void SignalColorDlg::addSignal(const MessageId &msg_id, const cabana::Signal *sig) {
  QString text = QString("<span style=\"color:%0;\">■ </span> %1").arg(sig->color.name(), sig->name);
  QLabel *label = new QLabel(text);
  label->setContentsMargins(5, 0, 5, 0);

  auto new_item = new ListItem(msg_id, sig, signal_list);
  new_item->setSizeHint(label->sizeHint());

  signal_list->setItemWidget(new_item, label);
  if (!signal_list->currentItem()) {
    signal_list->setCurrentItem(new_item);
  }
}
