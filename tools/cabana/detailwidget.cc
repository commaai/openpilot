#include "tools/cabana/detailwidget.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QMenu>
#include <QMessageBox>
#include <QToolButton>

#include "tools/cabana/commands.h"
#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

// DetailWidget

DetailWidget::DetailWidget(ChartsWidget *charts, QWidget *parent) : charts(charts), QWidget(parent) {
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
  QToolBar *toolbar = new QToolBar(this);
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
  toolbar->addAction(utils::icon("pencil"), "", this, &DetailWidget::editMsg)->setToolTip(tr("Edit Message"));
  remove_msg_act = toolbar->addAction(utils::icon("x-lg"), "", this, &DetailWidget::removeMsg);
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
  splitter = new QSplitter(Qt::Vertical, this);
  splitter->setAutoFillBackground(true);
  splitter->addWidget(binary_view = new BinaryView(this));
  splitter->addWidget(signal_view = new SignalView(charts, this));
  binary_view->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
  signal_view->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  splitter->setStretchFactor(0, 0);
  splitter->setStretchFactor(1, 1);

  tab_widget = new QTabWidget(this);
  tab_widget->setTabPosition(QTabWidget::South);
  tab_widget->addTab(splitter, utils::icon("file-earmark-ruled"), "&Msg");
  tab_widget->addTab(history_log = new LogsWidget(this), utils::icon("stopwatch"), "&Logs");
  main_layout->addWidget(tab_widget);

  stacked_layout = new QStackedLayout(this);
  stacked_layout->addWidget(new WelcomeWidget(this));
  stacked_layout->addWidget(main_widget);

  QObject::connect(binary_view, &BinaryView::resizeSignal, signal_view->model, &SignalModel::resizeSignal);
  QObject::connect(binary_view, &BinaryView::addSignal, signal_view->model, &SignalModel::addSignal);
  QObject::connect(binary_view, &BinaryView::signalHovered, signal_view, &SignalView::signalHovered);
  QObject::connect(binary_view, &BinaryView::signalClicked, signal_view, &SignalView::expandSignal);
  QObject::connect(signal_view, &SignalView::showChart, charts, &ChartsWidget::showChart);
  QObject::connect(signal_view, &SignalView::highlight, binary_view, &BinaryView::highlight);
  QObject::connect(tab_widget, &QTabWidget::currentChanged, [this]() { updateState(); });
  QObject::connect(can, &AbstractStream::msgsReceived, this, &DetailWidget::updateState);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &DetailWidget::refresh);
  QObject::connect(UndoStack::instance(), &QUndoStack::indexChanged, this, &DetailWidget::refresh);
  QObject::connect(tabbar, &QTabBar::customContextMenuRequested, this, &DetailWidget::showTabBarContextMenu);
  QObject::connect(tabbar, &QTabBar::currentChanged, [this](int index) {
    if (index != -1 && tabbar->tabText(index) != msg_id) {
      setMessage(tabbar->tabText(index));
    }
  });
  QObject::connect(tabbar, &QTabBar::tabCloseRequested, tabbar, &QTabBar::removeTab);
  QObject::connect(charts, &ChartsWidget::seriesChanged, signal_view, &SignalView::updateChartState);
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

  setUpdatesEnabled(false);

  signal_view->setMessage(msg_id);
  binary_view->setMessage(msg_id);
  history_log->setMessage(msg_id);

  stacked_layout->setCurrentIndex(1);
  tabbar->setCurrentIndex(index);
  refresh();
  splitter->setSizes({1, 2});

  setUpdatesEnabled(true);
}

void DetailWidget::refresh() {
  if (msg_id.isEmpty()) return;

  QStringList warnings;
  const DBCMsg *msg = dbc()->msg(msg_id);
  if (msg) {
    if (msg->size != can->lastMessage(msg_id).dat.size()) {
      warnings.push_back(tr("Message size (%1) is incorrect.").arg(msg->size));
    }
    for (auto s : binary_view->getOverlappingSignals()) {
      warnings.push_back(tr("%1 has overlapping bits.").arg(s->name.c_str()));
    }
  } else {
    warnings.push_back(tr("Drag-Select in binary view to create new signal."));
  }
  remove_msg_act->setEnabled(msg != nullptr);
  name_label->setText(msgName(msg_id));

  if (!warnings.isEmpty()) {
    warning_label->setText(warnings.join('\n'));
    warning_icon->setPixmap(utils::icon(msg ? "exclamation-triangle" : "info-circle"));
  }
  warning_widget->setVisible(!warnings.isEmpty());
}

void DetailWidget::updateState(const QHash<QString, CanData> *msgs) {
  time_label->setText(QString::number(can->currentSec(), 'f', 3));
  if (msg_id.isEmpty() || (msgs && !msgs->contains(msg_id)))
    return;

  if (tab_widget->currentIndex() == 0)
    binary_view->updateState();
  else
    history_log->updateState();
}

void DetailWidget::editMsg() {
  QString id = msg_id;
  auto msg = dbc()->msg(id);
  int size = msg ? msg->size : can->lastMessage(id).dat.size();
  EditMessageDialog dlg(id, msgName(id), size, this);
  if (dlg.exec()) {
    UndoStack::push(new EditMsgCommand(msg_id, dlg.name_edit->text(), dlg.size_spin->value()));
  }
}

void DetailWidget::removeMsg() {
  UndoStack::push(new RemoveMsgCommand(msg_id));
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
