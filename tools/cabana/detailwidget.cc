#include "tools/cabana/detailwidget.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QMenu>
#include <QMessageBox>
#include <QScrollBar>
#include <QTimer>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

// DetailWidget

DetailWidget::DetailWidget(ChartsWidget *charts, QWidget *parent) : charts(charts), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);

   // tabbar
  tabbar = new QTabBar(this);
  tabbar->setTabsClosable(true);
  tabbar->setDrawBase(false);
  tabbar->setUsesScrollButtons(true);
  tabbar->setAutoHide(true);
  tabbar->setContextMenuPolicy(Qt::CustomContextMenu);
  main_layout->addWidget(tabbar);

  QFrame *title_frame = new QFrame(this);
  QVBoxLayout *frame_layout = new QVBoxLayout(title_frame);
  title_frame->setFrameShape(QFrame::StyledPanel);

  // message title
  QHBoxLayout *title_layout = new QHBoxLayout();
  title_layout->addWidget(new QLabel("time:"));
  time_label = new QLabel(this);
  time_label->setStyleSheet("font-weight:bold");
  title_layout->addWidget(time_label);
  title_layout->addStretch();
  name_label = new QLabel(this);
  name_label->setStyleSheet("font-weight:bold;");
  title_layout->addWidget(name_label);
  title_layout->addStretch();
  edit_btn = new QPushButton(tr("Edit"), this);
  edit_btn->setVisible(false);
  title_layout->addWidget(edit_btn);
  frame_layout->addLayout(title_layout);

  // warning
  warning_widget = new QWidget(this);
  QHBoxLayout *warning_hlayout = new QHBoxLayout(warning_widget);
  warning_hlayout->setContentsMargins(0, 0, 0, 0);
  QLabel *warning_icon = new QLabel(this);
  warning_icon->setPixmap(style()->standardPixmap(QStyle::SP_MessageBoxWarning));
  warning_hlayout->addWidget(warning_icon, 0, Qt::AlignTop);
  warning_label = new QLabel(this);
  warning_hlayout->addWidget(warning_label, 1, Qt::AlignLeft);
  warning_widget->hide();
  frame_layout->addWidget(warning_widget);
  main_layout->addWidget(title_frame);

  QWidget *container = new QWidget(this);
  QVBoxLayout *container_layout = new QVBoxLayout(container);
  container_layout->setSpacing(0);
  container_layout->setContentsMargins(0, 0, 0, 0);

  scroll = new QScrollArea(this);
  scroll->setWidget(container);
  scroll->setWidgetResizable(true);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(scroll);

  // binary view
  binary_view = new BinaryView(this);
  container_layout->addWidget(binary_view);

  // signals
  signals_container = new QWidget(this);
  signals_container->setLayout(new QVBoxLayout);
  container_layout->addWidget(signals_container);

  // history log
  history_log = new HistoryLog(this);
  container_layout->addWidget(history_log);

  QObject::connect(edit_btn, &QPushButton::clicked, this, &DetailWidget::editMsg);
  QObject::connect(binary_view, &BinaryView::resizeSignal, this, &DetailWidget::resizeSignal);
  QObject::connect(binary_view, &BinaryView::addSignal, this, &DetailWidget::addSignal);
  QObject::connect(can, &CANMessages::updated, this, &DetailWidget::updateState);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, [this]() { dbcMsgChanged(); });
  QObject::connect(tabbar, &QTabBar::customContextMenuRequested, this, &DetailWidget::showTabBarContextMenu);
  QObject::connect(tabbar, &QTabBar::currentChanged, [this](int index) {
    if (index != -1 && tabbar->tabText(index) != msg_id) {
      setMessage(tabbar->tabText(index));
    }
  });
  QObject::connect(tabbar, &QTabBar::tabCloseRequested, tabbar, &QTabBar::removeTab);
  QObject::connect(charts, &ChartsWidget::chartOpened, [this](const QString &id, const Signal *sig) { updateChartState(id, sig, true); });
  QObject::connect(charts, &ChartsWidget::chartClosed, [this](const QString &id, const Signal *sig) { updateChartState(id, sig, false); });
}

void DetailWidget::showTabBarContextMenu(const QPoint &pt) {
  int index = tabbar->tabAt(pt);
  if (index >= 0) {
    QMenu menu(this);
    menu.addAction(tr("Close Other Tabs"));
    if (menu.exec(tabbar->mapToGlobal(pt))) {
      tabbar->setCurrentIndex(index);
      // remove all tabs before the one to keep
      for (int i = 0; i < index; ++i) {
        tabbar->removeTab(0);
      }
      // remove all tabs after the one to keep
      while (tabbar->count() > 1) {
        tabbar->removeTab(1);
      }
    }
  }
}

void DetailWidget::setMessage(const QString &message_id) {
  if (message_id.isEmpty()) return;

  int index = -1;
  for (int i = 0; i < tabbar->count(); ++i) {
    if (tabbar->tabText(i) == message_id) {
      index = i;
      break;
    }
  }
  if (index == -1) {
    index = tabbar->addTab(message_id);
    tabbar->setTabToolTip(index, msgName(message_id));
  }
  tabbar->setCurrentIndex(index);
  msg_id = message_id;
  dbcMsgChanged();

  scroll->verticalScrollBar()->setValue(0);
}

void DetailWidget::dbcMsgChanged(int show_form_idx) {
  if (msg_id.isEmpty()) return;

  setUpdatesEnabled(false);
  QStringList warnings;
  for (auto f : signal_list) f->hide();

  const Msg *msg = dbc()->msg(msg_id);
  if (msg) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      SignalEdit *form = i < signal_list.size() ? signal_list[i] : nullptr;
      if (!form) {
        form = new SignalEdit(i);
        QObject::connect(form, &SignalEdit::showFormClicked, this, &DetailWidget::showForm);
        QObject::connect(form, &SignalEdit::remove, this, &DetailWidget::removeSignal);
        QObject::connect(form, &SignalEdit::save, this, &DetailWidget::saveSignal);
        QObject::connect(form, &SignalEdit::highlight, binary_view, &BinaryView::highlight);
        QObject::connect(binary_view, &BinaryView::signalHovered, form, &SignalEdit::signalHovered);
        QObject::connect(form, &SignalEdit::showChart, charts, &ChartsWidget::showChart);
        signals_container->layout()->addWidget(form);
        signal_list.push_back(form);
      }
      form->setSignal(msg_id, &(msg->sigs[i]), i == show_form_idx);
      form->setChartOpened(charts->isChartOpened(msg_id, &(msg->sigs[i])));
      form->show();
    }
    if (msg->size != can->lastMessage(msg_id).dat.size())
      warnings.push_back(tr("Message size (%1) is incorrect.").arg(msg->size));
  }

  edit_btn->setVisible(true);
  name_label->setText(msgName(msg_id));

  binary_view->setMessage(msg_id);
  history_log->setMessage(msg_id);

  // Check overlapping bits
  if (auto overlapping = binary_view->getOverlappingSignals(); !overlapping.isEmpty()) {
    for (auto s : overlapping)
      warnings.push_back(tr("%1 has overlapping bits.").arg(s->name.c_str()));
  }

  warning_label->setText(warnings.join('\n'));
  warning_widget->setVisible(!warnings.isEmpty());
  setUpdatesEnabled(true);
}

void DetailWidget::updateState() {
  time_label->setText(QString::number(can->currentSec(), 'f', 3));
  if (msg_id.isEmpty()) return;

  binary_view->updateState();
  history_log->updateState();
}

void DetailWidget::showForm() {
  SignalEdit *sender = qobject_cast<SignalEdit *>(QObject::sender());
  setUpdatesEnabled(false);
  for (auto f : signal_list)
    f->setFormVisible(f == sender && !f->isFormVisible());
  QTimer::singleShot(1, [this]() { setUpdatesEnabled(true); });
}

void DetailWidget::updateChartState(const QString &id, const Signal *sig, bool opened) {
  for (auto f : signal_list)
    if (f->msg_id == id && f->sig == sig) f->setChartOpened(opened);
}

void DetailWidget::editMsg() {
  auto msg = dbc()->msg(msg_id);
  QString name = msgName(msg_id);
  int size = msg ? msg->size : can->lastMessage(msg_id).dat.size();
  EditMessageDialog dlg(msg_id, name, size, this);
  if (dlg.exec()) {
    dbc()->updateMsg(msg_id, dlg.name_edit->text(), dlg.size_spin->value());
    dbcMsgChanged();
  }
}

void DetailWidget::addSignal(int from, int to) {
  if (auto msg = dbc()->msg(msg_id)) {
    Signal sig = {};
    for (int i = 1; /**/; ++i) {
      sig.name = "NEW_SIGNAL_" + std::to_string(i);
      auto it = std::find_if(msg->sigs.begin(), msg->sigs.end(), [&](auto &s) { return sig.name == s.name; });
      if (it == msg->sigs.end()) break;
    }
    sig.is_little_endian = false,
    updateSigSizeParamsFromRange(sig, from, to);
    dbc()->addSignal(msg_id, sig);
    dbcMsgChanged(msg->sigs.size() - 1);
  }
}

void DetailWidget::resizeSignal(const Signal *sig, int from, int to) {
  Signal s = *sig;
  updateSigSizeParamsFromRange(s, from, to);
  saveSignal(sig, s);
}

void DetailWidget::saveSignal(const Signal *sig, const Signal &new_sig) {
  auto msg = dbc()->msg(msg_id);
  if (new_sig.name != sig->name) {
    auto it = std::find_if(msg->sigs.begin(), msg->sigs.end(), [&](auto &s) { return s.name == new_sig.name; });
    if (it != msg->sigs.end()) {
      QString warning_str = tr("There is already a signal with the same name '%1'").arg(new_sig.name.c_str());
      QMessageBox::warning(this, tr("Failed to save signal"), warning_str);
      return;
    }
  }

  auto [start, end] = getSignalRange(&new_sig);
  if (start < 0 || end >= msg->size * 8) {
    QString warning_str = tr("Signal size [%1] exceed limit").arg(new_sig.size);
    QMessageBox::warning(this, tr("Failed to save signal"), warning_str);
    return;
  }

  dbc()->updateSignal(msg_id, sig->name.c_str(), new_sig);
  // update binary view and history log
  updateState();
}

void DetailWidget::removeSignal(const Signal *sig) {
  QString text = tr("Are you sure you want to remove signal '%1'").arg(sig->name.c_str());
  if (QMessageBox::Yes == QMessageBox::question(this, tr("Remove signal"), text)) {
    dbc()->removeSignal(msg_id, sig->name.c_str());
    dbcMsgChanged();
  }
}

// EditMessageDialog

EditMessageDialog::EditMessageDialog(const QString &msg_id, const QString &title, int size, QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Edit message"));
  QFormLayout *form_layout = new QFormLayout(this);
  form_layout->addRow("ID", new QLabel(msg_id));

  name_edit = new QLineEdit(title, this);
  form_layout->addRow(tr("Name"), name_edit);

  size_spin = new QSpinBox(this);
  // TODO: limit the maximum?
  size_spin->setMinimum(1);
  size_spin->setValue(size);
  form_layout->addRow(tr("Size"), size_spin);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  form_layout->addRow(buttonBox);
  setFixedWidth(parent->width() * 0.9);

  connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}
