#include "tools/cabana/historylog.h"

#include <functional>

#include <QFileDialog>
#include <QPainter>
#include <QVBoxLayout>

#include "tools/cabana/commands.h"
#include "tools/cabana/utils/export.h"

QVariant HistoryLogModel::data(const QModelIndex &index, int role) const {
  const auto &m = messages[index.row()];
  const int col = index.column();
  if (role == Qt::DisplayRole) {
    if (col == 0) return QString::number(can->toSeconds(m.mono_time), 'f', 3);
    if (!isHexMode()) return sigs[col - 1]->formatValue(m.sig_values[col - 1], false);
  } else if (role == Qt::TextAlignmentRole) {
    return (uint32_t)(Qt::AlignRight | Qt::AlignVCenter);
  }

  if (isHexMode() && col == 1) {
    if (role == ColorsRole) return QVariant::fromValue((void *)(&m.colors));
    if (role == BytesRole) return QVariant::fromValue((void *)(&m.data));
  }
  return {};
}

void HistoryLogModel::setMessage(const MessageId &message_id) {
  msg_id = message_id;
  reset();
}

void HistoryLogModel::reset() {
  beginResetModel();
  sigs.clear();
  if (auto dbc_msg = dbc()->msg(msg_id)) {
    sigs = dbc_msg->getSignals();
  }
  messages.clear();
  hex_colors = {};
  endResetModel();
  setFilter(0, "", nullptr);
}

QVariant HistoryLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal) {
    if (role == Qt::DisplayRole || role == Qt::ToolTipRole) {
      if (section == 0) return "Time";
      if (isHexMode()) return "Data";

      QString name = sigs[section - 1]->name;
      QString unit = sigs[section - 1]->unit;
      return unit.isEmpty() ? name : QString("%1 (%2)").arg(name, unit);
    } else if (role == Qt::BackgroundRole && section > 0 && !isHexMode()) {
      // Alpha-blend the signal color with the background to ensure contrast
      QColor sigColor = sigs[section - 1]->color;
      sigColor.setAlpha(128);
      return QBrush(sigColor);
    }
  }
  return {};
}

void HistoryLogModel::setHexMode(bool hex) {
  hex_mode = hex;
  reset();
}

void HistoryLogModel::setFilter(int sig_idx, const QString &value, std::function<bool(double, double)> cmp) {
  filter_sig_idx = sig_idx;
  filter_value = value.toDouble();
  filter_cmp = value.isEmpty() ? nullptr : cmp;
  updateState(true);
}

void HistoryLogModel::updateState(bool clear) {
  if (clear && !messages.empty()) {
    beginRemoveRows({}, 0, messages.size() - 1);
    messages.clear();
    endRemoveRows();
  }
  uint64_t current_time = can->toMonoTime(can->lastMessage(msg_id).ts) + 1;
  fetchData(messages.begin(), current_time, messages.empty() ? 0 : messages.front().mono_time);
}

bool HistoryLogModel::canFetchMore(const QModelIndex &parent) const {
  const auto &events = can->events(msg_id);
  return !events.empty() && !messages.empty() && messages.back().mono_time > events.front()->mono_time;
}

void HistoryLogModel::fetchMore(const QModelIndex &parent) {
  if (!messages.empty())
    fetchData(messages.end(), messages.back().mono_time, 0);
}

void HistoryLogModel::fetchData(std::deque<Message>::iterator insert_pos, uint64_t from_time, uint64_t min_time) {
  const auto &events = can->events(msg_id);
  auto first = std::upper_bound(events.rbegin(), events.rend(), from_time, [](uint64_t ts, auto e) {
    return ts > e->mono_time;
  });

  std::vector<HistoryLogModel::Message> msgs;
  std::vector<double> values(sigs.size());
  msgs.reserve(batch_size);
  for (; first != events.rend() && (*first)->mono_time > min_time; ++first) {
    const CanEvent *e = *first;
    for (int i = 0; i < sigs.size(); ++i) {
      sigs[i]->getValue(e->dat, e->size, &values[i]);
    }
    if (!filter_cmp || filter_cmp(values[filter_sig_idx], filter_value)) {
       msgs.emplace_back(Message{e->mono_time, values, {e->dat, e->dat + e->size}});
      if (msgs.size() >= batch_size && min_time == 0) {
        break;
      }
    }
  }

  if (!msgs.empty()) {
    if (isHexMode() && (min_time > 0 || messages.empty())) {
      const auto freq = can->lastMessage(msg_id).freq;
      const std::vector<uint8_t> no_mask;
      for (auto &m : msgs) {
        hex_colors.compute(msg_id, m.data.data(), m.data.size(), m.mono_time / (double)1e9, can->getSpeed(), no_mask, freq);
        m.colors = hex_colors.colors;
      }
    }
    int pos = std::distance(messages.begin(), insert_pos);
    beginInsertRows({}, pos , pos + msgs.size() - 1);
    messages.insert(insert_pos, std::move_iterator(msgs.begin()), std::move_iterator(msgs.end()));
    endInsertRows();
  }
}

// HeaderView

QSize HeaderView::sectionSizeFromContents(int logicalIndex) const {
  static const QSize time_col_size = fontMetrics().size(Qt::TextSingleLine, "000000.000") + QSize(10, 6);
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
  export_btn = new ToolButton("filetype-csv", tr("Export to CSV file..."));
  h->addWidget(export_btn, 0, Qt::AlignRight);

  display_type_cb->addItems({"Signal", "Hex"});
  display_type_cb->setToolTip(tr("Display signal value or raw hex value"));
  comp_box->addItems({">", "=", "!=", "<"});
  value_edit->setClearButtonEnabled(true);
  value_edit->setValidator(new DoubleValidator(this));

  main_layout->addWidget(toolbar);
  QFrame *line = new QFrame(this);
  line->setFrameStyle(QFrame::HLine | QFrame::Sunken);
  main_layout->addWidget(line);
  main_layout->addWidget(logs = new QTableView(this));
  logs->setModel(model = new HistoryLogModel(this));
  logs->setItemDelegate(delegate = new MessageBytesDelegate(this));
  logs->setHorizontalHeader(new HeaderView(Qt::Horizontal, this));
  logs->horizontalHeader()->setDefaultAlignment(Qt::AlignRight | (Qt::Alignment)Qt::TextWordWrap);
  logs->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  logs->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
  logs->verticalHeader()->setDefaultSectionSize(delegate->sizeForBytes(8).height());
  logs->setFrameShape(QFrame::NoFrame);

  QObject::connect(display_type_cb, qOverload<int>(&QComboBox::activated), model, &HistoryLogModel::setHexMode);
  QObject::connect(signals_cb, SIGNAL(activated(int)), this, SLOT(filterChanged()));
  QObject::connect(comp_box, SIGNAL(activated(int)), this, SLOT(filterChanged()));
  QObject::connect(value_edit, &QLineEdit::textEdited, this, &LogsWidget::filterChanged);
  QObject::connect(export_btn, &QToolButton::clicked, this, &LogsWidget::exportToCSV);
  QObject::connect(can, &AbstractStream::seekedTo, model, &HistoryLogModel::reset);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, model, &HistoryLogModel::reset);
  QObject::connect(UndoStack::instance(), &QUndoStack::indexChanged, model, &HistoryLogModel::reset);
  QObject::connect(model, &HistoryLogModel::modelReset, this, &LogsWidget::modelReset);
  QObject::connect(model, &HistoryLogModel::rowsInserted, [this]() { export_btn->setEnabled(true); });
}

void LogsWidget::modelReset() {
  signals_cb->clear();
  for (auto s : model->sigs) {
    signals_cb->addItem(s->name);
  }
  export_btn->setEnabled(false);
  value_edit->clear();
  comp_box->setCurrentIndex(0);
  filters_widget->setVisible(!model->sigs.empty());
}

void LogsWidget::filterChanged() {
  if (value_edit->text().isEmpty() && !value_edit->isModified()) return;

  std::function<bool(double, double)> cmp = nullptr;
  switch (comp_box->currentIndex()) {
    case 0: cmp = std::greater<double>{}; break;
    case 1: cmp = std::equal_to<double>{}; break;
    case 2: cmp = [](double l, double r) { return l != r; }; break; // not equal
    case 3: cmp = std::less<double>{}; break;
  }
  model->setFilter(signals_cb->currentIndex(), value_edit->text(), cmp);
}

void LogsWidget::exportToCSV() {
  QString dir = QString("%1/%2_%3.csv").arg(settings.last_dir).arg(can->routeName()).arg(msgName(model->msg_id));
  QString fn = QFileDialog::getSaveFileName(this, QString("Export %1 to CSV file").arg(msgName(model->msg_id)),
                                            dir, tr("csv (*.csv)"));
  if (!fn.isEmpty()) {
    model->isHexMode() ? utils::exportToCSV(fn, model->msg_id)
                       : utils::exportSignalsToCSV(fn, model->msg_id);
  }
}
