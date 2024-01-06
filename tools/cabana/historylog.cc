#include "tools/cabana/historylog.h"

#include <functional>

#include <QFileDialog>
#include <QPainter>
#include <QVBoxLayout>

#include "tools/cabana/commands.h"
#include "tools/cabana/utils/export.h"

QVariant HistoryLogModel::data(const QModelIndex &index, int role) const {
  const bool show_signals = display_signals_mode && sigs.size() > 0;
  const auto &m = messages[index.row()];
  if (role == Qt::DisplayRole) {
    if (index.column() == 0) {
      return QString::number((m.mono_time / (double)1e9) - can->routeStartTime(), 'f', 2);
    }
    int i = index.column() - 1;
    return show_signals ? QString::number(m.sig_values[i], 'f', sigs[i]->precision) : QString();
  } else if (role == ColorsRole) {
    return QVariant::fromValue((void *)(&m.colors));
  } else if (role == BytesRole) {
    return QVariant::fromValue((void *)(&m.data));
  } else if (role == Qt::TextAlignmentRole) {
    return (uint32_t)(Qt::AlignRight | Qt::AlignVCenter);
  }
  return {};
}

void HistoryLogModel::setMessage(const MessageId &message_id) {
  msg_id = message_id;
}

void HistoryLogModel::refresh(bool fetch_message) {
  beginResetModel();
  sigs.clear();
  if (auto dbc_msg = dbc()->msg(msg_id)) {
    sigs = dbc_msg->getSignals();
  }
  last_fetch_time = 0;
  has_more_data = true;
  messages.clear();
  hex_colors = {};
  if (fetch_message) {
    updateState();
  }
  endResetModel();
}

QVariant HistoryLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal) {
    const bool show_signals = display_signals_mode && !sigs.empty();
    if (role == Qt::DisplayRole || role == Qt::ToolTipRole) {
      if (section == 0) {
        return "Time";
      }
      if (show_signals) {
        QString name = sigs[section - 1]->name;
        if (!sigs[section - 1]->unit.isEmpty()) {
          name += QString(" (%1)").arg(sigs[section - 1]->unit);
        }
        return name;
      } else {
        return "Data";
      }
    } else if (role == Qt::BackgroundRole && section > 0 && show_signals) {
      // Alpha-blend the signal color with the background to ensure contrast
      QColor sigColor = sigs[section - 1]->color;
      sigColor.setAlpha(128);
      return QBrush(sigColor);
    }
  }
  return {};
}

void HistoryLogModel::setDynamicMode(int state) {
  dynamic_mode = state != 0;
  refresh();
}

void HistoryLogModel::setDisplayType(int type) {
  display_signals_mode = type == 0;
  refresh();
}

void HistoryLogModel::segmentsMerged() {
  if (!dynamic_mode) {
    has_more_data = true;
  }
}

void HistoryLogModel::setFilter(int sig_idx, const QString &value, std::function<bool(double, double)> cmp) {
  filter_sig_idx = sig_idx;
  filter_value = value.toDouble();
  filter_cmp = value.isEmpty() ? nullptr : cmp;
}

void HistoryLogModel::updateState() {
  uint64_t current_time = (can->lastMessage(msg_id).ts + can->routeStartTime()) * 1e9 + 1;
  auto new_msgs = dynamic_mode ? fetchData(current_time, last_fetch_time) : fetchData(0);
  if (!new_msgs.empty()) {
    beginInsertRows({}, 0, new_msgs.size() - 1);
    messages.insert(messages.begin(), std::move_iterator(new_msgs.begin()), std::move_iterator(new_msgs.end()));
    endInsertRows();
  }
  has_more_data = new_msgs.size() >= batch_size;
  last_fetch_time = current_time;
}

void HistoryLogModel::fetchMore(const QModelIndex &parent) {
  if (!messages.empty()) {
    auto new_msgs = fetchData(messages.back().mono_time);
    if (!new_msgs.empty()) {
      beginInsertRows({}, messages.size(), messages.size() + new_msgs.size() - 1);
      messages.insert(messages.end(), std::move_iterator(new_msgs.begin()), std::move_iterator(new_msgs.end()));
      endInsertRows();
    }
    has_more_data = new_msgs.size() >= batch_size;
  }
}

template <class InputIt>
std::deque<HistoryLogModel::Message> HistoryLogModel::fetchData(InputIt first, InputIt last, uint64_t min_time) {
  std::deque<HistoryLogModel::Message> msgs;
  std::vector<double> values(sigs.size());
  for (; first != last && (*first)->mono_time > min_time; ++first) {
    const CanEvent *e = *first;
    for (int i = 0; i < sigs.size(); ++i) {
      sigs[i]->getValue(e->dat, e->size, &values[i]);
    }
    if (!filter_cmp || filter_cmp(values[filter_sig_idx], filter_value)) {
      auto &m = msgs.emplace_back();
      m.mono_time = e->mono_time;
      m.data.assign(e->dat, e->dat + e->size);
      m.sig_values = values;
      if (msgs.size() >= batch_size && min_time == 0) {
        return msgs;
      }
    }
  }
  return msgs;
}

std::deque<HistoryLogModel::Message> HistoryLogModel::fetchData(uint64_t from_time, uint64_t min_time) {
  const auto &events = can->events(msg_id);
  const auto freq = can->lastMessage(msg_id).freq;
  const bool update_colors = !display_signals_mode || sigs.empty();
  const std::vector<uint8_t> no_mask;
  const auto speed = can->getSpeed();
  if (dynamic_mode) {
    auto first = std::upper_bound(events.rbegin(), events.rend(), from_time, [](uint64_t ts, auto e) {
      return ts > e->mono_time;
    });
    auto msgs = fetchData(first, events.rend(), min_time);
    if (update_colors && (min_time > 0 || messages.empty())) {
      for (auto it = msgs.rbegin(); it != msgs.rend(); ++it) {
        hex_colors.compute(msg_id, it->data.data(), it->data.size(), it->mono_time / (double)1e9, speed, no_mask, freq);
        it->colors = hex_colors.colors;
      }
    }
    return msgs;
  } else {
    assert(min_time == 0);
    auto first = std::upper_bound(events.cbegin(), events.cend(), from_time, CompareCanEvent());
    auto msgs = fetchData(first, events.cend(), 0);
    if (update_colors) {
      for (auto it = msgs.begin(); it != msgs.end(); ++it) {
        hex_colors.compute(msg_id, it->data.data(), it->data.size(), it->mono_time / (double)1e9, speed, no_mask, freq);
        it->colors = hex_colors.colors;
      }
    }
    return msgs;
  }
}

// HeaderView

QSize HeaderView::sectionSizeFromContents(int logicalIndex) const {
  static const QSize time_col_size = fontMetrics().boundingRect({0, 0, 200, 200}, defaultAlignment(), "000000.000").size() + QSize(10, 6);
  if (logicalIndex == 0) {
    return time_col_size;
  } else {
    int default_size = qMax(100, (rect().width() - time_col_size.width()) / (model()->columnCount() - 1));
    QString text = model()->headerData(logicalIndex, this->orientation(), Qt::DisplayRole).toString();
    const QRect rect = fontMetrics().boundingRect({0, 0, default_size, 2000}, defaultAlignment(), text.replace(QChar('_'), ' '));
    QSize size = rect.size() + QSize{10, 6};
    return QSize{qMax(size.width(), default_size), size.height()};
  }
}

void HeaderView::paintSection(QPainter *painter, const QRect &rect, int logicalIndex) const {
  auto bg_role = model()->headerData(logicalIndex, Qt::Horizontal, Qt::BackgroundRole);
  if (bg_role.isValid()) {
    painter->fillRect(rect, bg_role.value<QBrush>());
  }
  QString text = model()->headerData(logicalIndex, Qt::Horizontal, Qt::DisplayRole).toString();
  painter->setPen(palette().color(settings.theme == DARK_THEME ? QPalette::BrightText : QPalette::Text));
  painter->drawText(rect.adjusted(5, 3, -5, -3), defaultAlignment(), text.replace(QChar('_'), ' '));
}

// LogsWidget

LogsWidget::LogsWidget(QWidget *parent) : QFrame(parent) {
  setFrameStyle(QFrame::StyledPanel | QFrame::Plain);
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);

  QWidget *toolbar = new QWidget(this);
  toolbar->setAutoFillBackground(true);
  QHBoxLayout *h = new QHBoxLayout(toolbar);

  filters_widget = new QWidget(this);
  QHBoxLayout *filter_layout = new QHBoxLayout(filters_widget);
  filter_layout->setContentsMargins(0, 0, 0, 0);
  filter_layout->addWidget(display_type_cb = new QComboBox(this));
  filter_layout->addWidget(signals_cb = new QComboBox(this));
  filter_layout->addWidget(comp_box = new QComboBox(this));
  filter_layout->addWidget(value_edit = new QLineEdit(this));
  h->addWidget(filters_widget);
  h->addStretch(0);
  h->addWidget(dynamic_mode = new QCheckBox(tr("Dynamic")), 0, Qt::AlignRight);
  ToolButton *export_btn = new ToolButton("filetype-csv", tr("Export to CSV file..."));
  h->addWidget(export_btn, 0, Qt::AlignRight);

  display_type_cb->addItems({"Signal", "Hex"});
  display_type_cb->setToolTip(tr("Display signal value or raw hex value"));
  comp_box->addItems({">", "=", "!=", "<"});
  value_edit->setClearButtonEnabled(true);
  value_edit->setValidator(new DoubleValidator(this));
  dynamic_mode->setChecked(true);
  dynamic_mode->setEnabled(!can->liveStreaming());

  main_layout->addWidget(toolbar);
  QFrame *line = new QFrame(this);
  line->setFrameStyle(QFrame::HLine | QFrame::Sunken);
  main_layout->addWidget(line);
  main_layout->addWidget(logs = new QTableView(this));
  logs->setModel(model = new HistoryLogModel(this));
  delegate = new MessageBytesDelegate(this);
  logs->setHorizontalHeader(new HeaderView(Qt::Horizontal, this));
  logs->horizontalHeader()->setDefaultAlignment(Qt::AlignRight | (Qt::Alignment)Qt::TextWordWrap);
  logs->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  logs->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
  logs->verticalHeader()->setDefaultSectionSize(delegate->sizeForBytes(8).height());
  logs->verticalHeader()->setVisible(false);
  logs->setFrameShape(QFrame::NoFrame);

  QObject::connect(display_type_cb, qOverload<int>(&QComboBox::activated), [this](int index) {
    logs->setItemDelegateForColumn(1, index == 1 ? delegate : nullptr);
    model->setDisplayType(index);
  });
  QObject::connect(dynamic_mode, &QCheckBox::stateChanged, model, &HistoryLogModel::setDynamicMode);
  QObject::connect(signals_cb, SIGNAL(activated(int)), this, SLOT(setFilter()));
  QObject::connect(comp_box, SIGNAL(activated(int)), this, SLOT(setFilter()));
  QObject::connect(value_edit, &QLineEdit::textChanged, this, &LogsWidget::setFilter);
  QObject::connect(export_btn, &QToolButton::clicked, this, &LogsWidget::exportToCSV);
  QObject::connect(can, &AbstractStream::seekedTo, model, &HistoryLogModel::refresh);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &LogsWidget::refresh);
  QObject::connect(UndoStack::instance(), &QUndoStack::indexChanged, this, &LogsWidget::refresh);
  QObject::connect(can, &AbstractStream::eventsMerged, model, &HistoryLogModel::segmentsMerged);
}

void LogsWidget::setMessage(const MessageId &message_id) {
  model->setMessage(message_id);
  refresh();
}

void LogsWidget::refresh() {
  model->setFilter(0, "", nullptr);
  model->refresh(isVisible());
  bool has_signal = model->sigs.size();
  if (has_signal) {
    signals_cb->clear();
    for (auto s : model->sigs) {
      signals_cb->addItem(s->name);
    }
  }
  logs->setItemDelegateForColumn(1, !has_signal || display_type_cb->currentIndex() == 1 ? delegate : nullptr);
  value_edit->clear();
  comp_box->setCurrentIndex(0);
  filters_widget->setVisible(has_signal);
}

void LogsWidget::setFilter() {
  if (value_edit->text().isEmpty() && !value_edit->isModified()) return;

  std::function<bool(double, double)> cmp = nullptr;
  switch (comp_box->currentIndex()) {
    case 0: cmp = std::greater<double>{}; break;
    case 1: cmp = std::equal_to<double>{}; break;
    case 2: cmp = [](double l, double r) { return l != r; }; break; // not equal
    case 3: cmp = std::less<double>{}; break;
  }
  model->setFilter(signals_cb->currentIndex(), value_edit->text(), cmp);
  model->refresh();
}

void LogsWidget::updateState() {
  if (isVisible() && dynamic_mode->isChecked()) {
    model->updateState();
  }
}

void LogsWidget::showEvent(QShowEvent *event) {
  if (dynamic_mode->isChecked() || model->canFetchMore({}) && model->rowCount() == 0) {
    model->refresh();
  }
}

void LogsWidget::exportToCSV() {
  QString dir = QString("%1/%2_%3.csv").arg(settings.last_dir).arg(can->routeName()).arg(msgName(model->msg_id));
  QString fn = QFileDialog::getSaveFileName(this, QString("Export %1 to CSV file").arg(msgName(model->msg_id)),
                                            dir, tr("csv (*.csv)"));
  if (!fn.isEmpty()) {
    const bool export_signals = model->display_signals_mode && model->sigs.size() > 0;
    export_signals ? utils::exportSignalsToCSV(fn, model->msg_id) : utils::exportToCSV(fn, model->msg_id);
  }
}
