#include "tools/cabana/chart/logsview.h"

#include "tools/cabana/historylog.h"
#include "tools/cabana/streams/abstractstream.h"

// MultipleSignalsLogModel

QVariant MultipleSignalsLogModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal) {
    if (role == Qt::DisplayRole) {
      auto s = sigs_[section - 1];
      return section == 0 ? tr("Time") : QString("%1(%2)\n%3").arg(msgName(s.msg_id), s.msg_id.toString(), s.sig->name);
    } else if (role == Qt::BackgroundRole && section > 0) {
      // Alpha-blend the signal color with the background to ensure contrast
      QColor color = sigs_[section - 1].sig->color;
      color.setAlpha(128);
      return QBrush(color);
    }
  }
  return {};
}

QVariant MultipleSignalsLogModel::data(const QModelIndex &index, int role) const {
  if (role == Qt::DisplayRole && index.isValid() && index.row() < values_.size()) {
    auto it = values_.crbegin();
    std::advance(it, index.row());
    int column = index.column();
    if (column == 0) {
      return QString::number((it->first / (double)1e9) - can->routeStartTime(), 'f', 2);
    }

    auto v = it->second[column - 1];
    return v ? QString::number(*v, 'f', sigs_[column - 1].sig->precision) : QStringLiteral("--");
  } else if (role == Qt::TextAlignmentRole) {
    return (uint32_t)(Qt::AlignRight | Qt::AlignVCenter);
  }
  return {};
}

void MultipleSignalsLogModel::updateState() {
  if (sigs_.empty()) return;

  uint64_t current_ts = (can->currentSec() + can->routeStartTime()) * 1e9;
  // delete data older than 30 seconds
  size_t prev_row_count = values_.size();
  for (auto it = values_.begin(); it != values_.end(); /**/) {
    it = (it->first < current_ts - 30 * 1e9) ? values_.erase(it) : ++it;
  }
  if (values_.size() < prev_row_count) {
    beginRemoveRows({}, values_.size(), prev_row_count - 1);
    endRemoveRows();
  }

  prev_row_count = values_.size();
  for (int i = 0; i < sigs_.size(); ++i) {
    const auto &s = sigs_[i];
    const auto &msgs = can->events(s.msg_id);
    auto first = std::lower_bound(msgs.crbegin(), msgs.crend(), last_ts_, [](auto e, uint64_t ts) { return e->mono_time > ts; });
    if (first != msgs.crend()) {
      // convert to forward iterator
      auto it = (first + 1).base();
      for (; it != msgs.cend() && (*it)->mono_time <= current_ts; ++it) {
        double value = 0;
        if (s.sig->getValue((*it)->dat, (*it)->size, &value)) {
          auto &v = values_[(*it)->mono_time];
          if (v.size() == 0) v.resize(sigs_.size());
          v[i] = value;
        }
      }
    }
  }
  last_ts_ = current_ts;

  if (values_.size() > prev_row_count) {
    beginInsertRows({}, 0, values_.size() - prev_row_count - 1);
    endInsertRows();
  }
}

void MultipleSignalsLogModel::setSignals(const std::vector<MultipleSignalsLogModel::Signal> &sigs) {
  beginResetModel();
  sigs_ = sigs;
  values_.clear();
  last_ts_ = (can->currentSec() + can->routeStartTime()) * 1e9;
  endResetModel();
  updateState();
}

void MultipleSignalsLogModel::signalUpdated(const cabana::Signal *sig) {
  if (std::any_of(sigs_.begin(), sigs_.end(), [sig](auto &s) { return s.sig == sig; })) {
    refresh();
  }
}

void MultipleSignalsLogModel::msgUpdated(MessageId id) {
  if (std::any_of(sigs_.begin(), sigs_.end(), [&id](auto &s) { return s.msg_id == id; })) {
    refresh();
  }
}

// MultipleSignalsLogView

MultipleSignalsLogView::MultipleSignalsLogView(QWidget *parent) : QTableView(parent) {
  setHorizontalHeader(new HeaderView(Qt::Horizontal, this));
  setModel(model = new MultipleSignalsLogModel(this));
  setSelectionMode(QAbstractItemView::NoSelection);
  horizontalHeader()->setDefaultAlignment(Qt::AlignRight | (Qt::Alignment)Qt::TextWordWrap);
  horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  verticalHeader()->setVisible(false);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
  setVisible(false);

  QObject::connect(can, &AbstractStream::updated, model, &MultipleSignalsLogModel::updateState);
  QObject::connect(can, &AbstractStream::seekedTo, model, &MultipleSignalsLogModel::refresh);
  QObject::connect(dbc(), &DBCManager::signalUpdated, model, &MultipleSignalsLogModel::signalUpdated);
  QObject::connect(dbc(), &DBCManager::msgUpdated, model, &MultipleSignalsLogModel::msgUpdated);
}

QSize MultipleSignalsLogView::minimumSizeHint() const {
  auto sz = QTableView::minimumSizeHint();
  return {sz.width(), 120};
}

void MultipleSignalsLogView::setSignals(const std::vector<MultipleSignalsLogModel::Signal> &sigs) {
  model->setSignals(sigs);
  setVisible(!sigs.empty());
}
