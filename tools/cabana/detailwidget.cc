#include "tools/cabana/detailwidget.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QMenu>
#include <QMessageBox>
#include <QScrollBar>
#include <QTimer>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/commands.h"
#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

// DetailWidget

DetailWidget::DetailWidget(ChartsWidget *charts, QWidget *parent) : charts(charts), QWidget(parent) {
  undo_stack = new QUndoStack(this);
  QWidget *main_widget = new QWidget(this);
  QVBoxLayout *main_layout = new QVBoxLayout(main_widget);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // tabbar
  tabbar = new QTabBar(this);
  tabbar->setTabsClosable(true);
  tabbar->setUsesScrollButtons(true);
  tabbar->setAutoHide(true);
  tabbar->setContextMenuPolicy(Qt::CustomContextMenu);
  main_layout->addWidget(tabbar);

  // message title
  toolbar = new QToolBar(this);
  toolbar->setIconSize({16, 16});
  toolbar->addWidget(new QLabel("time:"));
  time_label = new QLabel(this);
  time_label->setStyleSheet("font-weight:bold");
  toolbar->addWidget(time_label);
  name_label = new QLabel(this);
  name_label->setStyleSheet("font-weight:bold;");
  name_label->setAlignment(Qt::AlignCenter);
  name_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  toolbar->addWidget(name_label);
  toolbar->addAction(bootstrapPixmap("pencil"), "", this, &DetailWidget::editMsg)->setToolTip(tr("Edit Message"));
  remove_msg_act = toolbar->addAction(bootstrapPixmap("x-lg"), "", this, &DetailWidget::removeMsg);
  remove_msg_act->setToolTip(tr("Remove Message"));
  main_layout->addWidget(toolbar);

  // warning
  warning_widget = new QWidget(this);
  QHBoxLayout *warning_hlayout = new QHBoxLayout(warning_widget);
  warning_hlayout->addWidget(warning_icon = new QLabel(this), 0, Qt::AlignTop);
  warning_hlayout->addWidget(warning_label = new QLabel(this), 1, Qt::AlignLeft);
  warning_widget->hide();
  main_layout->addWidget(warning_widget);

  // msg widget
  QWidget *msg_widget = new QWidget(this);
  QVBoxLayout *msg_layout = new QVBoxLayout(msg_widget);
  msg_layout->setContentsMargins(0, 0, 0, 0);
  // binary view
  binary_view = new BinaryView(this);
  msg_layout->addWidget(binary_view);
  // signals
  signals_layout = new QVBoxLayout();
  signals_layout->setSpacing(0);
  msg_layout->addLayout(signals_layout);
  msg_layout->addStretch(0);

  scroll = new QScrollArea(this);
  scroll->setFrameShape(QFrame::NoFrame);
  scroll->setWidget(msg_widget);
  scroll->setWidgetResizable(true);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  tab_widget = new QTabWidget(this);
  tab_widget->setTabPosition(QTabWidget::South);
  tab_widget->addTab(scroll, bootstrapPixmap("file-earmark-ruled"), "&Msg");
  history_log = new LogsWidget(this);
  tab_widget->addTab(history_log, bootstrapPixmap("stopwatch"), "&Logs");
  main_layout->addWidget(tab_widget);

  stacked_layout = new QStackedLayout(this);
  stacked_layout->addWidget(new WelcomeWidget(this));
  stacked_layout->addWidget(main_widget);

  QObject::connect(binary_view, &BinaryView::signalClicked, this, &DetailWidget::showForm);
  QObject::connect(binary_view, &BinaryView::resizeSignal, this, &DetailWidget::resizeSignal);
  QObject::connect(binary_view, &BinaryView::addSignal, this, &DetailWidget::addSignal);
  QObject::connect(tab_widget, &QTabWidget::currentChanged, [this]() { updateState(); });
  QObject::connect(can, &AbstractStream::msgsReceived, this, &DetailWidget::updateState);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, [this]() { dbcMsgChanged(); });
  QObject::connect(tabbar, &QTabBar::customContextMenuRequested, this, &DetailWidget::showTabBarContextMenu);
  QObject::connect(tabbar, &QTabBar::currentChanged, [this](int index) {
    if (index != -1 && tabbar->tabText(index) != msg_id) {
      setMessage(tabbar->tabText(index));
    }
  });
  QObject::connect(tabbar, &QTabBar::tabCloseRequested, tabbar, &QTabBar::removeTab);
  QObject::connect(charts, &ChartsWidget::seriesChanged, this, &DetailWidget::updateChartState);
  QObject::connect(history_log, &LogsWidget::openChart, [this](const QString &id, const Signal *sig) {
    this->charts->showChart(id, sig, true, false);
  });
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
  stacked_layout->setCurrentIndex(1);
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
    for (auto sig : msg->getSignals()) {
      SignalEdit *form = i < signal_list.size() ? signal_list[i] : nullptr;
      if (!form) {
        form = new SignalEdit(i, this);
        QObject::connect(form, &SignalEdit::remove, this, &DetailWidget::removeSignal);
        QObject::connect(form, &SignalEdit::save, this, &DetailWidget::saveSignal);
        QObject::connect(form, &SignalEdit::showFormClicked, this, &DetailWidget::showForm);
        QObject::connect(form, &SignalEdit::highlight, binary_view, &BinaryView::highlight);
        QObject::connect(binary_view, &BinaryView::signalHovered, form, &SignalEdit::signalHovered);
        QObject::connect(form, &SignalEdit::showChart, charts, &ChartsWidget::showChart);
        signals_layout->addWidget(form);
        signal_list.push_back(form);
      }
      form->setSignal(msg_id, sig);
      form->setChartOpened(charts->hasSignal(msg_id, sig));
      ++i;
    }
    if (msg->size != can->lastMessage(msg_id).dat.size()) {
      warnings.push_back(tr("Message size (%1) is incorrect.").arg(msg->size));
    }
    for (auto s : binary_view->getOverlappingSignals()) {
      warnings.push_back(tr("%1 has overlapping bits.").arg(s->name.c_str()));
    }
  } else {
    warnings.push_back(tr("Drag-Select in binary view to create new signal."));
  }
  for (/**/; i < signal_list.size(); ++i)
    signal_list[i]->hide();

  toolbar->setVisible(!msg_id.isEmpty());
  remove_msg_act->setEnabled(msg != nullptr);
  name_label->setText(msgName(msg_id));

  if (!warnings.isEmpty()) {
    warning_label->setText(warnings.join('\n'));
    warning_icon->setPixmap(bootstrapPixmap(msg ? "exclamation-triangle" : "info-circle"));
  }
  warning_widget->setVisible(!warnings.isEmpty());
  setUpdatesEnabled(true);
}

void DetailWidget::updateState(const QHash<QString, CanData> * msgs) {
  time_label->setText(QString::number(can->currentSec(), 'f', 3));
  if (msg_id.isEmpty() || (msgs && !msgs->contains(msg_id)))
    return;

  if (tab_widget->currentIndex() == 0)
    binary_view->updateState();
  else
    history_log->updateState();
}

void DetailWidget::showForm(const Signal *sig) {
  setUpdatesEnabled(false);
  for (auto f : signal_list) {
    f->updateForm(f->sig == sig && !f->form->isVisible());
    if (f->sig == sig && f->form->isVisible()) {
      QTimer::singleShot(0, [=]() { scroll->ensureWidgetVisible(f); });
    }
  }
  setUpdatesEnabled(true);
}

void DetailWidget::updateChartState() {
  for (auto f : signal_list)
    f->setChartOpened(charts->hasSignal(f->msg_id, f->sig));
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
  Signal sig = {.is_little_endian = little_endian, .factor = 1};
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
  name_edit->setValidator(new NameValidator(name_edit));
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

// WelcomeWidget

WelcomeWidget::WelcomeWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addStretch(0);
  QLabel *logo = new QLabel("CABANA");
  logo->setAlignment(Qt::AlignCenter);
  logo->setStyleSheet("font-size:50px;font-weight:bold;");
  main_layout->addWidget(logo);

  auto newShortcutRow = [](const QString &title, const QString &key) {
    QHBoxLayout *hlayout = new QHBoxLayout();
    auto btn = new QToolButton();
    btn->setText(key);
    btn->setEnabled(false);
    hlayout->addWidget(new QLabel(title), 0, Qt::AlignRight);
    hlayout->addWidget(btn, 0, Qt::AlignLeft);
    return hlayout;
  };

  main_layout->addLayout(newShortcutRow("Pause", "Space"));
  main_layout->addLayout(newShortcutRow("Help", "Alt + H"));
  main_layout->addStretch(0);

  setStyleSheet("QLabel{color:darkGray;}");
}
