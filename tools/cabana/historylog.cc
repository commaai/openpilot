#include "tools/cabana/historylog.h"

#include <algorithm>
#include <functional>

#include <QPainter>
#include <QVBoxLayout>

#include "tools/cabana/commands.h"

QVariant HexLogModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid() || index.row() >= messages.size()) return {};

  const auto &m = messages[index.row()];
  switch (role) {
    case Qt::DisplayRole: return index.column() == 0 ? QString::number(can->toSeconds(m.mono_time), 'f', 2) : "";
    case ColorsRole: return QVariant::fromValue((void *)(&m.colors));
    case BytesRole: return QVariant::fromValue((void *)(&m.data));
    case Qt::TextAlignmentRole: return (uint32_t)(Qt::AlignRight | Qt::AlignVCenter);
    default: return {};
  }
}

QVariant HexLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  static const QVariant title[] = {tr("Time") , tr("Data")};
  return (orientation == Qt::Horizontal && role == Qt::DisplayRole) ? title[section] : QVariant();
}

QVariant SignalLogModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid() || index.row() >= messages.size()) return {};

  if (role == Qt::DisplayRole) {
    const auto &m = messages[index.row()];
    int n = index.column();
    return n == 0 ? QString::number(can->toSeconds(m.mono_time), 'f', 2)
                  : QString::number(m.sig_values[n - 1], 'f', sigs[n - 1]->precision);
  } else if (role == Qt::TextAlignmentRole) {
    return (uint32_t)(Qt::AlignRight | Qt::AlignVCenter);
  }
  return {};
}


QVariant SignalLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  auto signal_name = [](const cabana::Signal *s) {
    return s->name + (s->unit.isEmpty() ? "" : tr(" (%1)").arg(s->unit));
  };

  if (orientation == Qt::Horizontal) {
    if (role == Qt::DisplayRole || role == Qt::ToolTipRole) {
      return section == 0 ? tr("Time") : signal_name(sigs[section - 1]);
    } else if (role == Qt::BackgroundRole && section > 0) {
      // Alpha-blend the signal color with the background to ensure contrast
      QColor sigColor = sigs[section - 1]->color;
      sigColor.setAlpha(128);
      return QBrush(sigColor);
    }
  }
  return {};
}

// HistoryLogModel

HistoryLogModel::HistoryLogModel(const MessageId &id, QObject *parent) : msg_id(id), QAbstractTableModel(parent) {
  if (auto dbc_msg = dbc()->msg(msg_id))
    sigs = dbc_msg->getSignals();
}

void HistoryLogModel::refresh() {
  beginResetModel();
  messages.clear();
  last_fetch_time = 0;
  has_more_data = true;
  endResetModel();
  updateState();
}

void HistoryLogModel::setDynamicMode(int state) {
  dynamic_mode = state != 0;
  refresh();
}

void HistoryLogModel::setFilter(int sig_idx, const QString &value, std::function<bool(double, double)> cmp) {
  filter_sig_idx = sig_idx;
  filter_value = value.toDouble();
  filter_cmp = value.isEmpty() ? nullptr : cmp;
  refresh();
}

void HistoryLogModel::updateState() {
  uint64_t current_time = can->currentMonoTime() + 1;
  dynamic_mode ? fetchData(messages.begin(), current_time, last_fetch_time) : fetchData(messages.begin(), 0);
  last_fetch_time = current_time;
}

void HistoryLogModel::fetchMore(const QModelIndex &parent) {
  if (!messages.empty()) {
    int n = fetchData(messages.end(), messages.back().mono_time, 0);
    has_more_data = n >= batch_size;
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
    if (!filter_cmp || (filter_sig_idx < values.size() && filter_cmp(values[filter_sig_idx], filter_value))) {
      auto &m = msgs.emplace_back();
      m.mono_time = e->mono_time;
      m.data.assign(e->dat, e->dat + e->size);
      m.sig_values = values;
      if (msgs.size() >= batch_size && min_time == 0) break;
    }
  }
  return msgs;
}

size_t HistoryLogModel::fetchData(std::deque<HistoryLogModel::Message>::iterator insert_pos, uint64_t from_time, uint64_t min_time) {
  std::deque<HistoryLogModel::Message> msgs;
  const auto &events = can->events(msg_id);
  const auto speed = can->getSpeed();
  if (dynamic_mode) {
    auto first = std::upper_bound(events.rbegin(), events.rend(), from_time, [](uint64_t ts, auto e) {
      return ts > e->mono_time;
    });
    msgs = fetchData(first, events.rend(), min_time);
    if ((min_time > 0 || messages.empty())) {
      for (auto it = msgs.rbegin(); it != msgs.rend(); ++it) {
        hex_colors.update(msg_id, it->data.data(), it->data.size(), it->mono_time, speed, {});
        it->colors = hex_colors.colors;
      }
    }
  } else {
    auto first = std::upper_bound(events.cbegin(), events.cend(), from_time, CompareCanEvent());
    msgs = fetchData(first, events.cend(), 0);
    for (auto it = msgs.begin(); it != msgs.end(); ++it) {
      hex_colors.update(msg_id, it->data.data(), it->data.size(), it->mono_time, speed, {});
      it->colors = hex_colors.colors;
    }
  }

  if (!msgs.empty()) {
    int first = std::distance(messages.begin(), insert_pos);
    beginInsertRows({}, first, first + msgs.size() - 1);
    messages.insert(insert_pos, std::move_iterator(msgs.begin()), std::move_iterator(msgs.end()));
    endInsertRows();
  }
  return msgs.size();
}

// HeaderView

QSize HeaderView::sectionSizeFromContents(int logicalIndex) const {
  static QSize time_col_size = fontMetrics().boundingRect({0, 0, 200, 200}, defaultAlignment(), "000000.000").size() + QSize(10, 6);
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

  delegate = new MessageBytesDelegate(this);
  logs->setHorizontalHeader(new HeaderView(Qt::Horizontal, this));
  logs->horizontalHeader()->setDefaultAlignment(Qt::AlignRight | (Qt::Alignment)Qt::TextWordWrap);
  logs->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  logs->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
  logs->verticalHeader()->setDefaultSectionSize(delegate->sizeForBytes(8).height());
  logs->verticalHeader()->setVisible(false);
  logs->setFrameShape(QFrame::NoFrame);

  QObject::connect(display_type_cb, qOverload<int>(&QComboBox::activated), this, &LogsWidget::setModel);
  QObject::connect(signals_cb, SIGNAL(activated(int)), this, SLOT(setFilter()));
  QObject::connect(comp_box, SIGNAL(activated(int)), this, SLOT(setFilter()));
  QObject::connect(value_edit, &QLineEdit::textEdited, this, &LogsWidget::setFilter);
  QObject::connect(dbc(), &DBCManager::changed, this, &LogsWidget::msgChanged);
}

void LogsWidget::setMessage(const MessageId &message_id) {
  msg_id = message_id;
  msgChanged();
}

void LogsWidget::msgChanged() {
  auto msg = dbc()->msg(msg_id);
  bool has_signal = msg && msg->sigs.size() > 0;
  signals_cb->clear();
  if (has_signal) {
    for (auto s : msg->sigs) {
      signals_cb->addItem(s->name);
    }
  }
  value_edit->clear();
  filters_widget->setVisible(has_signal);

  setModel(!has_signal || display_type_cb->currentIndex() == 1);
}

void LogsWidget::setModel(bool hex_mode) {
  if (model) {
    model->deleteLater();
  }
  logs->setModel(hex_mode ? model = new HexLogModel(msg_id, this)
                          : model = new SignalLogModel(msg_id, this));
  logs->setItemDelegateForColumn(1, hex_mode ? delegate : nullptr);
  QObject::connect(can, &AbstractStream::seekedTo, model, &HistoryLogModel::refresh);
  QObject::connect(dynamic_mode, &QCheckBox::stateChanged, model, &HistoryLogModel::setDynamicMode);

  updateState();
}

void LogsWidget::setFilter() {
  std::function<bool(double, double)> cmp = nullptr;
  switch (comp_box->currentIndex()) {
    case 0: cmp = std::greater<double>{}; break;
    case 1: cmp = std::equal_to<double>{}; break;
    case 2: cmp = [](double l, double r) { return l != r; }; break; // not equal
    case 3: cmp = std::less<double>{}; break;
  }
  model->setFilter(signals_cb->currentIndex(), value_edit->text(), cmp);
}

void LogsWidget::updateState() {
  if (model && isVisible() && (dynamic_mode->isChecked() || model->rowCount() == 0))
    model->updateState();
}

void LogsWidget::showEvent(QShowEvent *event) {
  if (model && (dynamic_mode->isChecked() || model->rowCount() == 0))
    model->refresh();
}
