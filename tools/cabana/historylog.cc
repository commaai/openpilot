#include "tools/cabana/historylog.h"

#include <QFontDatabase>
#include <QPainter>
#include <QPushButton>
#include <QVBoxLayout>

// HistoryLogModel

void HistoryLogModel::setDisplayType(HistoryLogModel::DisplayType type) {
  if (display_type != type) {
    display_type = type;
    refresh();
  }
}

QVariant HistoryLogModel::data(const QModelIndex &index, int role) const {
  const bool display_signals = display_type == HistoryLogModel::Signals;
  if (role == Qt::DisplayRole) {
    const auto &m = messages[index.row()];
    if (index.column() == 0) {
      return QString::number((m.mono_time / (double)1e9) - can->routeStartTime(), 'f', 2);
    }
    return display_signals ? QString::number(m.sig_values[index.column() - 1]) : m.data;
  } else if (role == Qt::FontRole && index.column() == 1 && !display_signals) {
    return QFontDatabase::systemFont(QFontDatabase::FixedFont);
  } else if (role == Qt::ToolTipRole && index.column() > 0 && display_signals) {
    return tr("double click to open the chart");
  }
  return {};
}

void HistoryLogModel::setMessage(const QString &message_id) {
  msg_id = message_id;
  sigs.clear();
  if (auto dbc_msg = dbc()->msg(msg_id)) {
    sigs = dbc_msg->getSignals();
  }
  display_type = !sigs.empty() ? HistoryLogModel::Signals : HistoryLogModel::Hex;
  filter_cmp = nullptr;
  refresh();
}

void HistoryLogModel::refresh() {
  beginResetModel();
  last_fetch_time = 0;
  messages.clear();
  updateState();
  endResetModel();
}

QVariant HistoryLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  const bool display_signals = display_type == HistoryLogModel::Signals && !sigs.empty();
  if (orientation == Qt::Horizontal) {
    if (role == Qt::DisplayRole || role == Qt::ToolTipRole) {
      if (section == 0) {
        return "Time";
      }
      return display_signals ? QString::fromStdString(sigs[section - 1]->name).replace('_', ' ') : "Data";
    } else if (role == Qt::BackgroundRole && section > 0 && display_signals) {
      return QBrush(QColor(getColor(section - 1)));
    } else if (role == Qt::ForegroundRole && section > 0 && display_signals) {
      return QBrush(Qt::black);
    }
  }
  return {};
}

void HistoryLogModel::setDynamicMode(int state) {
  dynamic_mode = state != 0;
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
  refresh();
}

void HistoryLogModel::updateState() {
  if (!msg_id.isEmpty()) {
    uint64_t current_time = (can->currentSec() + can->routeStartTime()) * 1e9;
    auto new_msgs = dynamic_mode ? fetchData(current_time, last_fetch_time) : fetchData(0);
    if ((has_more_data = !new_msgs.empty())) {
      beginInsertRows({}, 0, new_msgs.size() - 1);
      messages.insert(messages.begin(), std::move_iterator(new_msgs.begin()), std::move_iterator(new_msgs.end()));
      endInsertRows();
    }
    last_fetch_time = current_time;
  }
}

void HistoryLogModel::fetchMore(const QModelIndex &parent) {
  if (!messages.empty()) {
    auto new_msgs = fetchData(messages.back().mono_time);
    if ((has_more_data = !new_msgs.empty())) {
      beginInsertRows({}, messages.size(), messages.size() + new_msgs.size() - 1);
      messages.insert(messages.end(), std::move_iterator(new_msgs.begin()), std::move_iterator(new_msgs.end()));
      endInsertRows();
    }
  }
}

template <class InputIt>
std::deque<HistoryLogModel::Message> HistoryLogModel::fetchData(InputIt first, InputIt last, uint64_t min_time) {
  std::deque<HistoryLogModel::Message> msgs;
  const auto [src, address] = DBCManager::parseId(msg_id);
  QVector<double> values(sigs.size());
  for (auto it = first; it != last && (*it)->mono_time > min_time; ++it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      for (const auto &c : (*it)->event.getCan()) {
        if (src == c.getSrc() && address == c.getAddress()) {
          const auto dat = c.getDat();
          for (int i = 0; i < sigs.size(); ++i) {
            values[i] = get_raw_value((uint8_t *)dat.begin(), dat.size(), *(sigs[i]));
          }
          if (!filter_cmp || filter_cmp(values[filter_sig_idx], filter_value)) {
            auto &m = msgs.emplace_back();
            m.mono_time = (*it)->mono_time;
            m.data = toHex(QByteArray((char *)dat.begin(), dat.size()));
            m.sig_values = values;
            if (msgs.size() >= batch_size && min_time == 0)
              return msgs;
          }
        }
      }
    }
  }
  return msgs;
}
template std::deque<HistoryLogModel::Message> HistoryLogModel::fetchData<>(std::vector<const Event*>::iterator first, std::vector<const Event*>::iterator last, uint64_t min_time);
template std::deque<HistoryLogModel::Message> HistoryLogModel::fetchData<>(std::vector<const Event*>::reverse_iterator first, std::vector<const Event*>::reverse_iterator last, uint64_t min_time);

std::deque<HistoryLogModel::Message> HistoryLogModel::fetchData(uint64_t from_time, uint64_t min_time) {
  auto events = can->events();
  if (dynamic_mode) {
    auto it = std::lower_bound(events->rbegin(), events->rend(), from_time, [=](auto &e, uint64_t ts) {
      return e->mono_time > ts;
    });
    if (it != events->rend()) ++it;
    return fetchData(it, events->rend(), min_time);
  } else {
    assert(min_time == 0);
    auto it = std::upper_bound(events->begin(), events->end(), from_time, [=](uint64_t ts, auto &e) {
      return ts < e->mono_time;
    });
    return fetchData(it, events->end(), 0);
  }
}

// HeaderView

QSize HeaderView::sectionSizeFromContents(int logicalIndex) const {
  int default_size = qMax(100, rect().width() / model()->columnCount());
  const QString text = model()->headerData(logicalIndex, this->orientation(), Qt::DisplayRole).toString();
  const QRect rect = fontMetrics().boundingRect({0, 0, default_size, 2000}, defaultAlignment(), text);
  QSize size = rect.size() + QSize{10, 6};
  return {qMax(size.width(), default_size), size.height()};
}

void HeaderView::paintSection(QPainter *painter, const QRect &rect, int logicalIndex) const {
  auto bg_role = model()->headerData(logicalIndex, Qt::Horizontal, Qt::BackgroundRole);
  if (bg_role.isValid()) {
    QPen pen(model()->headerData(logicalIndex, Qt::Horizontal, Qt::ForegroundRole).value<QBrush>(), 1);
    painter->setPen(pen);
    painter->fillRect(rect, bg_role.value<QBrush>());
  }
  QString text = model()->headerData(logicalIndex, Qt::Horizontal, Qt::DisplayRole).toString();
  painter->drawText(rect.adjusted(5, 3, -5, -3), defaultAlignment(), text);
}

// HistoryLog

HistoryLog::HistoryLog(QWidget *parent) : QTableView(parent) {
  setHorizontalHeader(new HeaderView(Qt::Horizontal, this));
  horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | (Qt::Alignment)Qt::TextWordWrap);
  horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  verticalHeader()->setVisible(false);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
}

// LogsWidget

LogsWidget::LogsWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *h = new QHBoxLayout();

  display_type_cb = new QComboBox();
  display_type_cb->addItems({"Signal value", "Hex value"});
  h->addWidget(display_type_cb);

  signals_cb = new QComboBox(this);
  signals_cb->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);
  h->addWidget(signals_cb);
  comp_box = new QComboBox();
  comp_box->addItems({">", "=", "!=", "<"});
  h->addWidget(comp_box);
  value_edit = new QLineEdit(this);
  value_edit->setClearButtonEnabled(true);
  value_edit->setValidator(new QDoubleValidator(-500000, 500000, 6, this));
  h->addWidget(value_edit);
  dynamic_mode = new QCheckBox(tr("Dynamic"));
  h->addWidget(dynamic_mode, 0, Qt::AlignRight);
  main_layout->addLayout(h);

  model = new HistoryLogModel(this);
  logs = new HistoryLog(this);
  logs->setModel(model);
  main_layout->addWidget(logs);

  QObject::connect(logs, &QTableView::doubleClicked, this, &LogsWidget::doubleClicked);
  QObject::connect(display_type_cb, SIGNAL(activated(int)), this, SLOT(displayTypeChanged()));
  QObject::connect(signals_cb, SIGNAL(activated(int)), this, SLOT(setFilter()));
  QObject::connect(comp_box, SIGNAL(activated(int)), this, SLOT(setFilter()));
  QObject::connect(value_edit, &QLineEdit::textChanged, this, &LogsWidget::setFilter);
  QObject::connect(dynamic_mode, &QCheckBox::stateChanged, model, &HistoryLogModel::setDynamicMode);
  QObject::connect(can, &AbstractStream::seekedTo, model, &HistoryLogModel::refresh);
  QObject::connect(can, &AbstractStream::eventsMerged, model, &HistoryLogModel::segmentsMerged);

  if (can->liveStreaming()) {
    dynamic_mode->setChecked(true);
    dynamic_mode->setEnabled(false);
  }
}

void LogsWidget::setMessage(const QString &message_id) {
  model->setMessage(message_id);
  cur_filter_text = "";
  value_edit->setText("");
  signals_cb->clear();
  comp_box->setCurrentIndex(0);
  bool has_signals = model->sigs.size() > 0;
  if (has_signals) {
    for (auto s : model->sigs) {
      signals_cb->addItem(s->name.c_str());
    }
  }
  display_type_cb->setCurrentIndex(has_signals ? 0 : 1);
  display_type_cb->setVisible(has_signals);
  comp_box->setVisible(has_signals);
  value_edit->setVisible(has_signals);
  signals_cb->setVisible(has_signals);
}

static bool not_equal(double l, double r) { return l != r; }

void LogsWidget::setFilter() {
  if (cur_filter_text.isEmpty() && value_edit->text().isEmpty()) {
    return;
  }

  std::function<bool(double, double)> cmp;
  switch (comp_box->currentIndex()) {
    case 0: cmp = std::greater<double>{}; break;
    case 1: cmp = std::equal_to<double>{}; break;
    case 2: cmp = not_equal; break;
    case 3: cmp = std::less<double>{}; break;
  }
  model->setFilter(signals_cb->currentIndex(), value_edit->text(), cmp);
  cur_filter_text = value_edit->text();
}

void LogsWidget::displayTypeChanged() {
  model->setDisplayType(display_type_cb->currentIndex() == 0 ? HistoryLogModel::Signals : HistoryLogModel::Hex);
}

void LogsWidget::showEvent(QShowEvent *event) {
  if (dynamic_mode->isChecked()) {
    model->refresh();
  }
}

void LogsWidget::updateState() {
  if (dynamic_mode->isChecked()) {
    model->updateState();
  }
}

void LogsWidget::doubleClicked(const QModelIndex &index) {
  if (index.isValid()) {
    if (model->display_type == HistoryLogModel::Signals && model->sigs.size() > 0 && index.column() > 0) {
      emit openChart(model->msg_id, model->sigs[index.column()-1]);
    }
    can->seekTo(model->messages[index.row()].mono_time / (double)1e9 - can->routeStartTime());
  }
}
