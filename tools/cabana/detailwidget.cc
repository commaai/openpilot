#include "tools/cabana/detailwidget.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QMenu>
#include <QMessageBox>
#include <QTimer>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

// DetailWidget

DetailWidget::DetailWidget(ChartsWidget *charts, QWidget *parent) : charts(charts), QWidget(parent) {
  main_layout = new QHBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  right_column = new QVBoxLayout();
  main_layout->addLayout(right_column);

  binary_view_container = new QWidget(this);
  binary_view_container->setMinimumWidth(500);
  binary_view_container->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
  QVBoxLayout *bin_layout = new QVBoxLayout(binary_view_container);
  bin_layout->setContentsMargins(0, 0, 0, 0);
  bin_layout->setSpacing(0);
  // tabbar
  tabbar = new QTabBar(this);
  tabbar->setTabsClosable(true);
  tabbar->setDrawBase(false);
  tabbar->setUsesScrollButtons(true);
  tabbar->setAutoHide(true);
  tabbar->setContextMenuPolicy(Qt::CustomContextMenu);
  bin_layout->addWidget(tabbar);

  TitleFrame *title_frame = new TitleFrame(this);
  title_frame->setFrameShape(QFrame::StyledPanel);
  title_frame->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
  QVBoxLayout *frame_layout = new QVBoxLayout(title_frame);

  // message title
  QHBoxLayout *title_layout = new QHBoxLayout();
  split_btn = new QPushButton("⬅", this);
  split_btn->setFixedSize(20, 20);
  split_btn->setToolTip(tr("Split to two columns"));
  title_layout->addWidget(split_btn);
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
  QLabel *warning_icon = new QLabel(this);
  warning_icon->setPixmap(style()->standardPixmap(QStyle::SP_MessageBoxWarning));
  warning_hlayout->addWidget(warning_icon, 0, Qt::AlignTop);
  warning_label = new QLabel(this);
  warning_hlayout->addWidget(warning_label, 1, Qt::AlignLeft);
  warning_widget->hide();
  frame_layout->addWidget(warning_widget);
  bin_layout->addWidget(title_frame);

  // binary view
  binary_view = new BinaryView(this);
  bin_layout->addWidget(binary_view);
  right_column->addWidget(binary_view_container);

  // signals
  signals_container = new QWidget(this);
  signals_container->setLayout(new QVBoxLayout);
  signals_container->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

  scroll = new ScrollArea(this);
  scroll->setWidget(signals_container);
  scroll->setWidgetResizable(true);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  right_column->addWidget(scroll);

  // history log
  history_log = new HistoryLog(this);
  right_column->addWidget(history_log);

  QObject::connect(split_btn, &QPushButton::clicked, this, &DetailWidget::moveBinaryView);
  QObject::connect(title_frame, &TitleFrame::doubleClicked, this, &DetailWidget::moveBinaryView);
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
  QObject::connect(tabbar, &QTabBar::tabCloseRequested, [=](int index) {
    if (tabbar->currentIndex() == index) {
      tabbar->setCurrentIndex(index == tabbar->count() - 1 ? index - 1 : index + 1);
    }
    tabbar->removeTab(index);
  });
  QObject::connect(charts, &ChartsWidget::chartOpened, [this](const QString &id, const Signal *sig) { updateChartState(id, sig, true); });
  QObject::connect(charts, &ChartsWidget::chartClosed, [this](const QString &id, const Signal *sig) { updateChartState(id, sig, false); });
}

void DetailWidget::showTabBarContextMenu(const QPoint &pt) {
  int index = tabbar->tabAt(pt);
  if (index >= 0) {
    QMenu menu(this);
    menu.addAction(tr("Close Other Tabs"));
    if (menu.exec(tabbar->mapToGlobal(pt))) {
      for (int i = tabbar->count() - 1; i >= 0; --i) {
        if (i != index)
          tabbar->removeTab(i);
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
    auto msg = dbc()->msg(message_id);
    tabbar->setTabToolTip(index, msg ? msg->name.c_str() : "untitled");
  }
  tabbar->setCurrentIndex(index);
  msg_id = message_id;
  dbcMsgChanged();
}

void DetailWidget::dbcMsgChanged(int show_form_idx) {
  if (msg_id.isEmpty()) return;

  warning_widget->hide();
  QStringList warnings;

  clearLayout(signals_container->layout());
  QString msg_name = tr("untitled");
  if (auto msg = dbc()->msg(msg_id)) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      auto form = new SignalEdit(i, msg_id, &(msg->sigs[i]));
      form->setChartOpened(charts->isChartOpened(msg_id, &(msg->sigs[i])));
      signals_container->layout()->addWidget(form);
      QObject::connect(form, &SignalEdit::showFormClicked, this, &DetailWidget::showForm);
      QObject::connect(form, &SignalEdit::remove, this, &DetailWidget::removeSignal);
      QObject::connect(form, &SignalEdit::save, this, &DetailWidget::saveSignal);
      QObject::connect(form, &SignalEdit::highlight, binary_view, &BinaryView::highlight);
      QObject::connect(binary_view, &BinaryView::signalHovered, form, &SignalEdit::signalHovered);
      QObject::connect(form, &SignalEdit::showChart, [this, sig = &msg->sigs[i]](bool show) { charts->showChart(msg_id, sig, show); });
      if (i == show_form_idx) {
        QTimer::singleShot(0, [=]() { emit form->showFormClicked(); });
      }
    }
    msg_name = msg->name.c_str();
    if (msg->size != can->lastMessage(msg_id).dat.size())
      warnings.push_back(tr("Message size (%1) is incorrect.").arg(msg->size));
  }
  edit_btn->setVisible(true);
  name_label->setText(msg_name);

  binary_view->setMessage(msg_id);
  history_log->setMessage(msg_id);

  // Check overlapping bits
  if (auto overlapping = binary_view->getOverlappingSignals(); !overlapping.isEmpty()) {
    for (auto s : overlapping)
      warnings.push_back(tr("%1 has overlapping bits.").arg(s->name.c_str()));
  }

  if (!warnings.isEmpty()) {
    warning_label->setText(warnings.join('\n'));
    warning_widget->show();
  }
}

void DetailWidget::updateState() {
  time_label->setText(QString::number(can->currentSec(), 'f', 3));
  if (msg_id.isEmpty()) return;

  binary_view->updateState();
  history_log->updateState();
}

void DetailWidget::moveBinaryView() {
  if (binview_in_left_col) {
    right_column->insertWidget(0, binary_view_container);
    emit binaryViewMoved(true);
  } else {
    main_layout->insertWidget(0, binary_view_container);
    emit binaryViewMoved(false);
  }
  split_btn->setText(binview_in_left_col ? "⬅" : "➡");
  split_btn->setToolTip(binview_in_left_col ? tr("Split to two columns") : tr("Move back"));
  binary_view->updateGeometry();
  binview_in_left_col = !binview_in_left_col;
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

void DetailWidget::updateChartState(const QString &id, const Signal *sig, bool opened) {
  if (id == msg_id) {
    for (auto f : signals_container->findChildren<SignalEdit *>())
      if (f->sig == sig) f->setChartOpened(opened);
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
  dbcMsgChanged();
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
