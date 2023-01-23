
#include "tools/cabana/chart/seriesdlg.h"

#include <QCompleter>
#include <QHBoxLayout>
#include <QLabel>
#include <QDialogButtonBox>
#include <QLineEdit>

#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

SeriesSelector::SeriesSelector(QWidget *parent) {
  setWindowTitle(tr("Manage Chart Series"));
  QHBoxLayout *contents_layout = new QHBoxLayout();

  QVBoxLayout *left_layout = new QVBoxLayout();
  left_layout->addWidget(new QLabel(tr("Select Signals:")));

  msgs_combo = new QComboBox(this);
  msgs_combo->setEditable(true);
  msgs_combo->lineEdit()->setPlaceholderText(tr("Select Msg"));
  msgs_combo->setInsertPolicy(QComboBox::NoInsert);
  msgs_combo->completer()->setCompletionMode(QCompleter::PopupCompletion);
  msgs_combo->completer()->setFilterMode(Qt::MatchContains);

  left_layout->addWidget(msgs_combo);
  sig_list = new QListWidget(this);
  sig_list->setSortingEnabled(true);
  sig_list->setToolTip(tr("Double click on an item to add signal to chart"));
  left_layout->addWidget(sig_list);

  QVBoxLayout *right_layout = new QVBoxLayout();
  right_layout->addWidget(new QLabel(tr("Chart Signals:")));
  chart_series = new QListWidget(this);
  chart_series->setSortingEnabled(true);
  chart_series->setToolTip(tr("Double click on an item to remove signal from chart"));
  right_layout->addWidget(chart_series);
  contents_layout->addLayout(left_layout);
  contents_layout->addLayout(right_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(contents_layout);
  main_layout->addWidget(buttonBox);

  for (auto it = can->can_msgs.cbegin(); it != can->can_msgs.cend(); ++it) {
    if (auto m = dbc()->msg(it.key())) {
      msgs_combo->addItem(QString("%1 (%2)").arg(m->name).arg(it.key()), it.key());
    }
  }
  msgs_combo->model()->sort(0);

  QObject::connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  QObject::connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(msgs_combo, SIGNAL(currentIndexChanged(int)), SLOT(msgSelected(int)));
  QObject::connect(sig_list, &QListWidget::itemDoubleClicked, this, &SeriesSelector::addSignal);
  QObject::connect(chart_series, &QListWidget::itemDoubleClicked, [](QListWidgetItem *item) { delete item; });

  if (int index = msgs_combo->currentIndex(); index >= 0) {
    msgSelected(index);
  }
}

void SeriesSelector::msgSelected(int index) {
  QString msg_id = msgs_combo->itemData(index).toString();
  sig_list->clear();
  if (auto m = dbc()->msg(msg_id)) {
    for (auto &[name, s] : m->sigs) {
      QStringList data({msg_id, m->name, name});
      QListWidgetItem *item = new QListWidgetItem(name, sig_list);
      item->setData(Qt::UserRole, data);
      sig_list->addItem(item);
    }
  }
}

void SeriesSelector::addSignal(QListWidgetItem *item) {
  QStringList data = item->data(Qt::UserRole).toStringList();
  addSeries(data[0], data[1], data[2]);
}

void SeriesSelector::addSeries(const QString &id, const QString &msg_name, const QString &sig_name) {
  QStringList data({id, msg_name, sig_name});
  for (int i = 0; i < chart_series->count(); ++i) {
    if (chart_series->item(i)->data(Qt::UserRole).toStringList() == data) {
      return;
    }
  }
  QListWidgetItem *new_item = new QListWidgetItem(chart_series);
  new_item->setData(Qt::UserRole, data);
  chart_series->addItem(new_item);
  QLabel *label = new QLabel(QString("%0 <font color=\"gray\">%1 %2</font>").arg(data[2]).arg(data[1]).arg(data[0]), chart_series);
  label->setContentsMargins(5, 0, 5, 0);
  new_item->setSizeHint(label->sizeHint());
  chart_series->setItemWidget(new_item, label);
}

QList<QStringList> SeriesSelector::series() {
  QList<QStringList> ret;
  for (int i = 0; i < chart_series->count(); ++i) {
    ret.push_back(chart_series->item(i)->data(Qt::UserRole).toStringList());
  }
  return ret;
}
