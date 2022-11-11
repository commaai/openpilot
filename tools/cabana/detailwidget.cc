#include "tools/cabana/detailwidget.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QMenu>
#include <QMessageBox>
#include <QScrollBar>
#include <QTimer>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/canmessages.h"
#include "tools/cabana/commands.h"
#include "tools/cabana/dbcmanager.h"

// DetailWidget

DetailWidget::DetailWidget(ChartsWidget *charts, QWidget *parent) : charts(charts), QWidget(parent) {
  undo_stack = new QUndoStack(this);

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
  toolbar = new QToolBar(this);
  toolbar->addWidget(new QLabel("time:"));
  time_label = new QLabel(this);
  time_label->setStyleSheet("font-weight:bold");
  toolbar->addWidget(time_label);
  name_label = new QLabel(this);
  name_label->setStyleSheet("font-weight:bold;");
  name_label->setAlignment(Qt::AlignCenter);
  name_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  toolbar->addWidget(name_label);
  toolbar->addAction("🖍", this, &DetailWidget::editMsg)->setToolTip(tr("Edit Message"));
  remove_msg_act = toolbar->addAction("X", this, &DetailWidget::removeMsg);
  remove_msg_act->setToolTip(tr("Remove Message"));
  toolbar->setVisible(false);
  frame_layout->addWidget(toolbar);

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
  signals_layout = new QVBoxLayout();
  container_layout->addLayout(signals_layout);

  // history log
  history_log = new HistoryLog(this);
  container_layout->addWidget(history_log);

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
  QObject::connect(undo_stack, &QUndoStack::indexChanged, [this]() {
    if (undo_stack->count() > 0)
      dbcMsgChanged();
  });
}

void DetailWidget::showTabBarContextMenu(const QPoint &pt) {
  int index = tabbar->tabAt(pt);
  if (index >= 0) {
    QMenu menu(this);
    menu.addAction(tr("Close Other Tabs"));
    if (menu.exec(tabbar->mapToGlobal(pt))) {
      tabbar->moveTab(index, 0);
      tabbar->setCurrentIndex(0);
      while (tabbar->count() > 1)
        tabbar->removeTab(1);
    }
  }
}

void DetailWidget::setMessage(const QString &message_id) {
  msg_id = message_id;
  int index = tabbar->count() - 1;
  for (/**/; index >= 0 && tabbar->tabText(index) != msg_id; --index) { /**/ }
  if (index == -1) {
    index = tabbar->addTab(message_id);
    tabbar->setTabToolTip(index, msgName(message_id));
  }
  tabbar->setCurrentIndex(index);
  dbcMsgChanged();
  scroll->verticalScrollBar()->setValue(0);
}

void DetailWidget::dbcMsgChanged(int show_form_idx) {
  if (msg_id.isEmpty()) return;

  setUpdatesEnabled(false);

  binary_view->setMessage(msg_id);
  history_log->setMessage(msg_id);

  int i = 0;
  QStringList warnings;
  const DBCMsg *msg = dbc()->msg(msg_id);
  if (msg) {
    for (auto &[name, sig] : msg->sigs) {
      SignalEdit *form = i < signal_list.size() ? signal_list[i] : nullptr;
      if (!form) {
        form = new SignalEdit(i);
        QObject::connect(form, &SignalEdit::remove, this, &DetailWidget::removeSignal);
        QObject::connect(form, &SignalEdit::save, this, &DetailWidget::saveSignal);
        QObject::connect(form, &SignalEdit::highlight, binary_view, &BinaryView::highlight);
        QObject::connect(binary_view, &BinaryView::signalHovered, form, &SignalEdit::signalHovered);
        QObject::connect(form, &SignalEdit::showChart, charts, &ChartsWidget::showChart);
        signals_layout->addWidget(form);
        signal_list.push_back(form);
      }
      form->setSignal(msg_id, &sig);
      form->setChartOpened(charts->isChartOpened(msg_id, &sig));
      ++i;
    }
    if (msg->size != can->lastMessage(msg_id).dat.size())
      warnings.push_back(tr("Message size (%1) is incorrect.").arg(msg->size));
  }
  for (/**/; i < signal_list.size(); ++i)
    signal_list[i]->hide();

  toolbar->setVisible(!msg_id.isEmpty());
  remove_msg_act->setEnabled(msg != nullptr);
  name_label->setText(msgName(msg_id));

  for (auto s : binary_view->getOverlappingSignals())
    warnings.push_back(tr("%1 has overlapping bits.").arg(s->name.c_str()));

  warning_label->setText(warnings.join('\n'));
  warning_widget->setVisible(!warnings.isEmpty());
  QTimer::singleShot(1, [this]() { setUpdatesEnabled(true); });
}

void DetailWidget::updateState() {
  time_label->setText(QString::number(can->currentSec(), 'f', 3));
  if (msg_id.isEmpty()) return;

  binary_view->updateState();
  history_log->updateState();
}

void DetailWidget::updateChartState(const QString &id, const Signal *sig, bool opened) {
  for (auto f : signal_list)
    if (f->msg_id == id && f->sig == sig) f->setChartOpened(opened);
}

void DetailWidget::editMsg() {
  QString id = msg_id;
  auto msg = dbc()->msg(id);
  int size = msg ? msg->size : can->lastMessage(id).dat.size();
  EditMessageDialog dlg(id, msgName(id), size, this);
  if (dlg.exec()) {
    undo_stack->push(new EditMsgCommand(msg_id, dlg.name_edit->text(), dlg.size_spin->value()));
  }
}

void DetailWidget::removeMsg() {
  undo_stack->push(new RemoveMsgCommand(msg_id));
}

void DetailWidget::addSignal(int start_bit, int size, bool little_endian) {
  auto msg = dbc()->msg(msg_id);
  if (!msg) {
    for (int i = 1; /**/; ++i) {
      QString name = QString("NEW_MSG_%1").arg(i);
      auto it = std::find_if(dbc()->messages().begin(), dbc()->messages().end(), [&](auto &m) { return m.second.name == name; });
      if (it == dbc()->messages().end()) {
        undo_stack->push(new EditMsgCommand(msg_id, name, can->lastMessage(msg_id).dat.size()));
        msg = dbc()->msg(msg_id);
        break;
      }
    }
  }
  Signal sig = {.is_little_endian = little_endian};
  for (int i = 1; /**/; ++i) {
    sig.name = "NEW_SIGNAL_" + std::to_string(i);
    if (msg->sigs.count(sig.name.c_str()) == 0) break;
  }
  updateSigSizeParamsFromRange(sig, start_bit, size);
  undo_stack->push(new AddSigCommand(msg_id, sig));
}

void DetailWidget::resizeSignal(const Signal *sig, int start_bit, int size) {
  Signal s = *sig;
  updateSigSizeParamsFromRange(s, start_bit, size);
  saveSignal(sig, s);
}

void DetailWidget::saveSignal(const Signal *sig, const Signal &new_sig) {
  auto msg = dbc()->msg(msg_id);
  if (new_sig.name != sig->name) {
    auto it = msg->sigs.find(new_sig.name.c_str());
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

  undo_stack->push(new EditSignalCommand(msg_id, sig, new_sig));
}

void DetailWidget::removeSignal(const Signal *sig) {
  undo_stack->push(new RemoveSigCommand(msg_id, sig));
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
