#include "tools/cabana/detailwidget.h"

#include <QFormLayout>
#include <QMenu>
#include <QRadioButton>
#include <QToolBar>

#include "tools/cabana/commands.h"
#include "tools/cabana/mainwin.h"

// DetailWidget

DetailWidget::DetailWidget(ChartsWidget *charts, QWidget *parent) : charts(charts), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // tabbar
  tabbar = new TabBar(this);
  tabbar->setUsesScrollButtons(true);
  tabbar->setAutoHide(true);
  tabbar->setContextMenuPolicy(Qt::CustomContextMenu);
  main_layout->addWidget(tabbar);

  createToolBar();

  // warning
  warning_widget = new QWidget(this);
  QHBoxLayout *warning_hlayout = new QHBoxLayout(warning_widget);
  warning_hlayout->addWidget(warning_icon = new QLabel(this), 0, Qt::AlignTop);
  warning_hlayout->addWidget(warning_label = new QLabel(this), 1, Qt::AlignLeft);
  warning_widget->hide();
  main_layout->addWidget(warning_widget);

  // msg widget
  splitter = new QSplitter(Qt::Vertical, this);
  splitter->addWidget(binary_view = new BinaryView(this));
  splitter->addWidget(signal_view = new SignalView(charts, this));
  binary_view->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
  signal_view->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  splitter->setStretchFactor(0, 0);
  splitter->setStretchFactor(1, 1);

  tab_widget = new QTabWidget(this);
  tab_widget->setStyleSheet("QTabWidget::pane {border: none; margin-bottom: -2px;}");
  tab_widget->setTabPosition(QTabWidget::South);
  tab_widget->addTab(splitter, utils::icon("file-earmark-ruled"), "&Msg");
  tab_widget->addTab(history_log = new LogsWidget(this), utils::icon("stopwatch"), "&Logs");
  main_layout->addWidget(tab_widget);

  QObject::connect(binary_view, &BinaryView::signalHovered, signal_view, &SignalView::signalHovered);
  QObject::connect(binary_view, &BinaryView::signalClicked, [this](const cabana::Signal *s) { signal_view->selectSignal(s, true); });
  QObject::connect(binary_view, &BinaryView::editSignal, signal_view->model, &SignalModel::saveSignal);
  QObject::connect(binary_view, &BinaryView::showChart, charts, &ChartsWidget::showChart);
  QObject::connect(signal_view, &SignalView::showChart, charts, &ChartsWidget::showChart);
  QObject::connect(signal_view, &SignalView::highlight, binary_view, &BinaryView::highlight);
  QObject::connect(tab_widget, &QTabWidget::currentChanged, [this]() { updateState(); });
  QObject::connect(can, &AbstractStream::msgsReceived, this, &DetailWidget::updateState);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &DetailWidget::refresh);
  QObject::connect(UndoStack::instance(), &QUndoStack::indexChanged, this, &DetailWidget::refresh);
  QObject::connect(tabbar, &QTabBar::customContextMenuRequested, this, &DetailWidget::showTabBarContextMenu);
  QObject::connect(tabbar, &QTabBar::currentChanged, [this](int index) {
    if (index != -1) {
      setMessage(tabbar->tabData(index).value<MessageId>());
    }
  });
  QObject::connect(tabbar, &QTabBar::tabCloseRequested, tabbar, &QTabBar::removeTab);
  QObject::connect(charts, &ChartsWidget::seriesChanged, signal_view, &SignalView::updateChartState);
}

void DetailWidget::createToolBar() {
  QToolBar *toolbar = new QToolBar(this);
  int icon_size = style()->pixelMetric(QStyle::PM_SmallIconSize);
  toolbar->setIconSize({icon_size, icon_size});
  toolbar->addWidget(name_label = new ElidedLabel(this));
  name_label->setStyleSheet("QLabel{font-weight:bold;}");

  QWidget *spacer = new QWidget();
  spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  toolbar->addWidget(spacer);

// Heatmap label and radio buttons
  toolbar->addWidget(new QLabel(tr("Heatmap:"), this));
  auto *heatmap_live = new QRadioButton(tr("Live"), this);
  auto *heatmap_all = new QRadioButton(tr("All"), this);
  heatmap_live->setChecked(true);

  toolbar->addWidget(heatmap_live);
  toolbar->addWidget(heatmap_all);

  // Edit and remove buttons
  toolbar->addSeparator();
  toolbar->addAction(utils::icon("pencil"), tr("Edit Message"), this, &DetailWidget::editMsg);
  action_remove_msg = toolbar->addAction(utils::icon("x-lg"), tr("Remove Message"), this, &DetailWidget::removeMsg);

  layout()->addWidget(toolbar);

  connect(heatmap_live, &QAbstractButton::toggled, this, [this](bool on) { binary_view->setHeatmapLiveMode(on); });
  connect(can, &AbstractStream::timeRangeChanged, this, [=](const std::optional<std::pair<double, double>> &range) {
    auto text = range ? QString("%1 - %2").arg(range->first, 0, 'f', 3).arg(range->second, 0, 'f', 3) : "All";
    heatmap_all->setText(text);
    (range ? heatmap_all : heatmap_live)->setChecked(true);
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
      while (tabbar->count() > 1) {
        tabbar->removeTab(1);
      }
    }
  }
}

void DetailWidget::setMessage(const MessageId &message_id) {
  if (std::exchange(msg_id, message_id) == message_id) return;

  tabbar->blockSignals(true);
  int index = tabbar->count() - 1;
  for (/**/; index >= 0; --index) {
    if (tabbar->tabData(index).value<MessageId>() == message_id) break;
  }
  if (index == -1) {
    index = tabbar->addTab(message_id.toString());
    tabbar->setTabData(index, QVariant::fromValue(message_id));
    tabbar->setTabToolTip(index, msgName(message_id));
  }
  tabbar->setCurrentIndex(index);
  tabbar->blockSignals(false);

  setUpdatesEnabled(false);
  signal_view->setMessage(msg_id);
  binary_view->setMessage(msg_id);
  history_log->setMessage(msg_id);
  refresh();
  setUpdatesEnabled(true);
}

void DetailWidget::refresh() {
  QStringList warnings;
  auto msg = dbc()->msg(msg_id);
  if (msg) {
    if (msg_id.source == INVALID_SOURCE) {
      warnings.push_back(tr("No messages received."));
    } else if (msg->size != can->lastMessage(msg_id).dat.size()) {
      warnings.push_back(tr("Message size (%1) is incorrect.").arg(msg->size));
    }
    for (auto s : binary_view->getOverlappingSignals()) {
      warnings.push_back(tr("%1 has overlapping bits.").arg(s->name));
    }
  }
  QString msg_name = msg ? QString("%1 (%2)").arg(msg->name, msg->transmitter) : msgName(msg_id);
  name_label->setText(msg_name);
  name_label->setToolTip(msg_name);
  action_remove_msg->setEnabled(msg != nullptr);

  if (!warnings.isEmpty()) {
    warning_label->setText(warnings.join('\n'));
    warning_icon->setPixmap(utils::icon(msg ? "exclamation-triangle" : "info-circle"));
  }
  warning_widget->setVisible(!warnings.isEmpty());
}

void DetailWidget::updateState(const std::set<MessageId> *msgs) {
  if ((msgs && !msgs->count(msg_id)))
    return;

  if (tab_widget->currentIndex() == 0)
    binary_view->updateState();
  else
    history_log->updateState();
}

void DetailWidget::editMsg() {
  auto msg = dbc()->msg(msg_id);
  int size = msg ? msg->size : can->lastMessage(msg_id).dat.size();
  EditMessageDialog dlg(msg_id, msgName(msg_id), size, this);
  if (dlg.exec()) {
    UndoStack::push(new EditMsgCommand(msg_id, dlg.name_edit->text().trimmed(), dlg.size_spin->value(),
                                       dlg.node->text().trimmed(), dlg.comment_edit->toPlainText().trimmed()));
  }
}

void DetailWidget::removeMsg() {
  UndoStack::push(new RemoveMsgCommand(msg_id));
}

// EditMessageDialog

EditMessageDialog::EditMessageDialog(const MessageId &msg_id, const QString &title, int size, QWidget *parent)
    : original_name(title), msg_id(msg_id), QDialog(parent) {
  setWindowTitle(tr("Edit message: %1").arg(msg_id.toString()));
  QFormLayout *form_layout = new QFormLayout(this);

  form_layout->addRow("", error_label = new QLabel);
  error_label->setVisible(false);
  form_layout->addRow(tr("Name"), name_edit = new QLineEdit(title, this));
  name_edit->setValidator(new NameValidator(name_edit));

  form_layout->addRow(tr("Size"), size_spin = new QSpinBox(this));
  size_spin->setRange(1, CAN_MAX_DATA_BYTES);
  size_spin->setValue(size);

  form_layout->addRow(tr("Node"), node = new QLineEdit(this));
  node->setValidator(new NameValidator(name_edit));
  form_layout->addRow(tr("Comment"), comment_edit = new QTextEdit(this));
  form_layout->addRow(btn_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel));

  if (auto msg = dbc()->msg(msg_id)) {
    node->setText(msg->transmitter);
    comment_edit->setText(msg->comment);
  }
  validateName(name_edit->text());
  setFixedWidth(parent->width() * 0.9);
  connect(name_edit, &QLineEdit::textEdited, this, &EditMessageDialog::validateName);
  connect(btn_box, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(btn_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void EditMessageDialog::validateName(const QString &text) {
  bool valid = text.compare(UNTITLED, Qt::CaseInsensitive) != 0;
  error_label->setVisible(false);
  if (!text.isEmpty() && valid && text != original_name) {
    valid = dbc()->msg(msg_id.source, text) == nullptr;
    if (!valid) {
      error_label->setText(tr("Name already exists"));
      error_label->setVisible(true);
    }
  }
  btn_box->button(QDialogButtonBox::Ok)->setEnabled(valid);
}

// CenterWidget

CenterWidget::CenterWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->addWidget(welcome_widget = createWelcomeWidget());
}

void CenterWidget::setMessage(const MessageId &msg_id) {
  if (!detail_widget) {
    delete welcome_widget;
    welcome_widget = nullptr;
    layout()->addWidget(detail_widget = new DetailWidget(((MainWindow*)parentWidget())->charts_widget, this));
  }
  detail_widget->setMessage(msg_id);
}

void CenterWidget::clear() {
  delete detail_widget;
  detail_widget = nullptr;
  if (!welcome_widget) {
    layout()->addWidget(welcome_widget = createWelcomeWidget());
  }
}

QWidget *CenterWidget::createWelcomeWidget() {
  QWidget *w = new QWidget(this);
  QVBoxLayout *main_layout = new QVBoxLayout(w);
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

  auto lb = new QLabel(tr("<-Select a message to view details"));
  lb->setAlignment(Qt::AlignHCenter);
  main_layout->addWidget(lb);
  main_layout->addLayout(newShortcutRow("Pause", "Space"));
  main_layout->addLayout(newShortcutRow("Help", "F1"));
  main_layout->addLayout(newShortcutRow("WhatsThis", "Shift+F1"));
  main_layout->addStretch(0);

  w->setStyleSheet("QLabel{color:darkGray;}");
  w->setBackgroundRole(QPalette::Base);
  w->setAutoFillBackground(true);
  return w;
}
