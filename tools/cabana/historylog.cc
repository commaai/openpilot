#include "tools/cabana/historylog.h"

#include <QFontDatabase>
#include <QPainter>

// HistoryLogModel

HistoryLogModel::HistoryLogModel(QObject *parent) : QAbstractTableModel(parent) {
  QObject::connect(can, &CANMessages::seekedTo, [this]() {
    if (!msg_id.isEmpty()) setMessage(msg_id);
  });
}

QVariant HistoryLogModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole) {
    const auto &m = messages[index.row()];
    if (index.column() == 0) {
      return QString::number((m.mono_time / (double)1e9) - can->routeStartTime(), 'f', 2);
    }
    return !sigs.empty() ? QString::number(m.sig_values[index.column() - 1]) : toHex(m.data);
  } else if (role == Qt::FontRole && index.column() == 1 && sigs.empty()) {
    return QFontDatabase::systemFont(QFontDatabase::FixedFont);
  }
  return {};
}

void HistoryLogModel::setMessage(const QString &message_id) {
  beginResetModel();
  msg_id = message_id;
  sigs.clear();
  messages.clear();
  has_more_data = true;
  if (auto dbc_msg = dbc()->msg(message_id)) {
    sigs = dbc_msg->getSignals();
  }
  endResetModel();
  updateState();
}

QVariant HistoryLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal) {
    if (role == Qt::DisplayRole || role == Qt::ToolTipRole) {
      if (section == 0) {
        return "Time";
      }
      return !sigs.empty() ? QString::fromStdString(sigs[section - 1]->name).replace('_', ' ') : "Data";
    } else if (role == Qt::BackgroundRole && section > 0 && !sigs.empty()) {
      return QBrush(QColor(getColor(section - 1)));
    } else if (role == Qt::ForegroundRole && section > 0 && !sigs.empty()) {
      return QBrush(Qt::black);
    }
  }
  return {};
}

void HistoryLogModel::updateState() {
  if (!msg_id.isEmpty()) {
    uint64_t last_mono_time = messages.empty() ? 0 : messages.front().mono_time;
    auto new_msgs = fetchData(last_mono_time, (can->currentSec() + can->routeStartTime()) * 1e9);
    if ((has_more_data = !new_msgs.empty())) {
      beginInsertRows({}, 0, new_msgs.size() - 1);
      messages.insert(messages.begin(), std::move_iterator(new_msgs.begin()), std::move_iterator(new_msgs.end()));
      endInsertRows();
    }
  }
}

void HistoryLogModel::fetchMore(const QModelIndex &parent) {
  if (!messages.empty()) {
    auto new_msgs = fetchData(0, messages.back().mono_time);
    if ((has_more_data = !new_msgs.empty())) {
      beginInsertRows({}, messages.size(), messages.size() + new_msgs.size() - 1);
      messages.insert(messages.end(), std::move_iterator(new_msgs.begin()), std::move_iterator(new_msgs.end()));
      endInsertRows();
    }
  }
}

std::deque<HistoryLogModel::Message> HistoryLogModel::fetchData(uint64_t min_mono_time, uint64_t max_mono_time) {
  auto events = can->events();
  auto it = std::lower_bound(events->begin(), events->end(), max_mono_time, [=](auto &e, uint64_t ts) {
    return e->mono_time < ts;
  });
  if (it == events->end() || it == events->begin())
    return {};

  std::deque<HistoryLogModel::Message> msgs;
  const auto [src, address] = DBCManager::parseId(msg_id);
  uint32_t cnt = 0;
  for (--it; it != events->begin() && (*it)->mono_time > min_mono_time; --it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      for (const auto &c : (*it)->event.getCan()) {
        if (src == c.getSrc() && address == c.getAddress()) {
          const auto dat = c.getDat();
          auto &m = msgs.emplace_back();
          m.mono_time = (*it)->mono_time;
          m.data.append((char *)dat.begin(), dat.size());
          m.sig_values.reserve(sigs.size());
          for (const Signal *sig : sigs) {
            m.sig_values.push_back(get_raw_value((uint8_t *)dat.begin(), dat.size(), *sig));
          }
          if (++cnt >= batch_size && min_mono_time == 0)
            return msgs;
        }
      }
    }
  }
  return msgs;
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
  model = new HistoryLogModel(this);
  setModel(model);
  setHorizontalHeader(new HeaderView(Qt::Horizontal, this));
  horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | (Qt::Alignment)Qt::TextWordWrap);
  horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  verticalHeader()->setVisible(false);
  setFrameShape(QFrame::NoFrame);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
}
