#include "tools/cabana/messageswidget.h"

#include <QHBoxLayout>
#include <QPainter>
#include <QPushButton>
#include <QVBoxLayout>

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0 ,0, 0, 0);

  // message filter
  QHBoxLayout *title_layout = new QHBoxLayout();
  title_layout->addWidget(filter = new QLineEdit(this));
  QRegularExpression re("\\S+");
  filter->setValidator(new QRegularExpressionValidator(re, this));
  filter->setClearButtonEnabled(true);
  filter->setPlaceholderText(tr("filter messages"));
  title_layout->addWidget(multiple_lines_bytes = new QCheckBox(tr("Multiple Lines Bytes"), this));
  multiple_lines_bytes->setToolTip(tr("Display bytes in multiple lines"));
  multiple_lines_bytes->setChecked(settings.multiple_lines_bytes);
  main_layout->addLayout(title_layout);

  // message table
  view = new MessageView(this);
  model = new MessageListModel(this);
  auto delegate = new MessageBytesDelegate(view, settings.multiple_lines_bytes);
  view->setItemDelegate(delegate);
  view->setModel(model);
  view->setSortingEnabled(true);
  view->sortByColumn(0, Qt::AscendingOrder);
  view->setAllColumnsShowFocus(true);
  view->setEditTriggers(QAbstractItemView::NoEditTriggers);
  view->setItemsExpandable(false);
  view->setIndentation(0);
  view->setRootIsDecorated(false);
  view->header()->setSectionsMovable(false);
  main_layout->addWidget(view);

  // suppress
  QHBoxLayout *suppress_layout = new QHBoxLayout();
  suppress_add = new QPushButton("Suppress Highlighted");
  suppress_clear = new QPushButton();
  suppress_layout->addWidget(suppress_add);
  suppress_layout->addWidget(suppress_clear);
  main_layout->addLayout(suppress_layout);

  // signals/slots
  QObject::connect(filter, &QLineEdit::textEdited, model, &MessageListModel::setFilterString);
  QObject::connect(multiple_lines_bytes, &QCheckBox::stateChanged, [=](int state) {
    settings.multiple_lines_bytes = (state == Qt::Checked);
    delegate->setMultipleLines(settings.multiple_lines_bytes);
    view->setUniformRowHeights(!settings.multiple_lines_bytes);
    model->sortMessages();
  });
  QObject::connect(can, &AbstractStream::msgsReceived, model, &MessageListModel::msgsReceived);
  QObject::connect(can, &AbstractStream::streamStarted, this, &MessagesWidget::reset);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, model, &MessageListModel::sortMessages);
  QObject::connect(dbc(), &DBCManager::msgUpdated, model, &MessageListModel::sortMessages);
  QObject::connect(dbc(), &DBCManager::msgRemoved, model, &MessageListModel::sortMessages);
  QObject::connect(model, &MessageListModel::modelReset, [this]() {
    if (current_msg_id) {
      selectMessage(*current_msg_id);
    }
  });
  QObject::connect(view->selectionModel(), &QItemSelectionModel::currentChanged, [=](const QModelIndex &current, const QModelIndex &previous) {
    if (current.isValid() && current.row() < model->msgs.size()) {
      auto &id = model->msgs[current.row()];
      if (!current_msg_id || id != *current_msg_id) {
        current_msg_id = id;
        emit msgSelectionChanged(*current_msg_id);
      }
    }
  });
  QObject::connect(suppress_add, &QPushButton::clicked, [=]() {
    model->suppress();
    updateSuppressedButtons();
  });
  QObject::connect(suppress_clear, &QPushButton::clicked, [=]() {
    model->clearSuppress();
    updateSuppressedButtons();
  });

  updateSuppressedButtons();

  setWhatsThis(tr(R"(
    <b>Message View</b><br/>
    <!-- TODO: add descprition here -->
    <span style="color:gray">Byte color</span><br />
    <span style="color:gray;">■ </span> constant changing<br />
    <span style="color:blue;">■ </span> increasing<br />
    <span style="color:red;">■ </span> decreasing
  )"));
}

void MessagesWidget::selectMessage(const MessageId &msg_id) {
  if (int row = model->msgs.indexOf(msg_id); row != -1) {
    view->selectionModel()->setCurrentIndex(model->index(row, 0), QItemSelectionModel::Rows | QItemSelectionModel::ClearAndSelect);
  }
}

void MessagesWidget::updateSuppressedButtons() {
  if (model->suppressed_bytes.empty()) {
    suppress_clear->setEnabled(false);
    suppress_clear->setText("Clear Suppressed");
  } else {
    suppress_clear->setEnabled(true);
    suppress_clear->setText(QString("Clear Suppressed (%1)").arg(model->suppressed_bytes.size()));
  }
}

void MessagesWidget::reset() {
  current_msg_id = std::nullopt;
  view->selectionModel()->clear();
  model->reset();
  filter->clear();
  updateSuppressedButtons();
}


// MessageListModel

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
    static const QString titles[] = {"Name", "Bus", "ID", "Freq", "Count", "Bytes"};
    return titles[section];
  }
  return {};
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  const auto &id = msgs[index.row()];
  auto &can_data = can->lastMessage(id);

  auto getFreq = [](const CanData &d) -> QString {
    if (d.freq > 0 && (can->currentSec() - d.ts - 1.0 / settings.fps) < (5.0 / d.freq)) {
      return d.freq >= 1 ? QString::number(std::nearbyint(d.freq)) : QString::number(d.freq, 'f', 2);
    } else {
      return "--";
    }
  };

  if (role == Qt::DisplayRole) {
    switch (index.column()) {
      case 0: return msgName(id);
      case 1: return id.source;
      case 2: return QString::number(id.address, 16);
      case 3: return getFreq(can_data);
      case 4: return can_data.count;
      case 5: return toHex(can_data.dat);
    }
  } else if (role == ColorsRole) {
    QVector<QColor> colors = can_data.colors;
    if (!suppressed_bytes.empty()) {
      for (int i = 0; i < colors.size(); i++) {
        if (suppressed_bytes.contains({id, i})) {
          colors[i] = QColor(255, 255, 255, 0);
        }
      }
    }
    return QVariant::fromValue(colors);
  } else if (role == BytesRole && index.column() == 5) {
    return can_data.dat;
  }
  return {};
}

void MessageListModel::setFilterString(const QString &string) {
  auto contains = [](const MessageId &id, const QString &txt) {
    auto cs = Qt::CaseInsensitive;
    if (id.toString().contains(txt, cs) || msgName(id).contains(txt, cs)) return true;
    // Search by signal name
    if (const auto msg = dbc()->msg(id)) {
      for (auto s : msg->getSignals()) {
        if (s->name.contains(txt, cs)) return true;
      }
    }
    return false;
  };

  filter_str = string;
  msgs.clear();
  for (auto it = can->last_msgs.begin(); it != can->last_msgs.end(); ++it) {
    if (filter_str.isEmpty() || contains(it.key(), filter_str)) {
      msgs.push_back(it.key());
    }
  }
  sortMessages();
}

void MessageListModel::sortMessages() {
  beginResetModel();
  if (sort_column == 0) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto ll = std::pair{msgName(l), l};
      auto rr = std::pair{msgName(r), r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == 1) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto ll = std::pair{l.source, l};
      auto rr = std::pair{r.source, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == 2) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto ll = std::pair{l.address, l};
      auto rr = std::pair{r.address, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == 3) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto ll = std::pair{can->lastMessage(l).freq, l};
      auto rr = std::pair{can->lastMessage(r).freq, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == 4) {
    std::sort(msgs.begin(), msgs.end(), [this](auto &l, auto &r) {
      auto ll = std::pair{can->lastMessage(l).count, l};
      auto rr = std::pair{can->lastMessage(r).count, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  }
  endResetModel();
}

void MessageListModel::msgsReceived(const QHash<MessageId, CanData> *new_msgs) {
  int prev_row_count = msgs.size();
  if (filter_str.isEmpty() && msgs.size() != can->last_msgs.size()) {
    msgs = can->last_msgs.keys();
  }
  if (msgs.size() != prev_row_count) {
    sortMessages();
    return;
  }
  for (int i = 0; i < msgs.size(); ++i) {
    if (new_msgs->contains(msgs[i])) {
      for (int col = 3; col < columnCount(); ++col)
        emit dataChanged(index(i, col), index(i, col), {Qt::DisplayRole});
    }
  }
}

void MessageListModel::sort(int column, Qt::SortOrder order) {
  if (column != columnCount() - 1) {
    sort_column = column;
    sort_order = order;
    sortMessages();
  }
}

void MessageListModel::suppress() {
  const double cur_ts = can->currentSec();

  for (auto &id : msgs) {
    auto &can_data = can->lastMessage(id);
    for (int i = 0; i < can_data.dat.size(); i++) {
      const double dt = cur_ts - can_data.last_change_t[i];
      if (dt < 2.0) {
        suppressed_bytes.insert({id, i});
      }
    }
  }
}

void MessageListModel::clearSuppress() {
  suppressed_bytes.clear();
}

void MessageListModel::reset() {
  beginResetModel();
  filter_str = "";
  msgs.clear();
  clearSuppress();
  endResetModel();
}

// MessageView

void MessageView::drawRow(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  QTreeView::drawRow(painter, option, index);
  const int gridHint = style()->styleHint(QStyle::SH_Table_GridLineColor, &option, this);
  const QColor gridColor = QColor::fromRgba(static_cast<QRgb>(gridHint));
  QPen old_pen = painter->pen();
  painter->setPen(gridColor);
  painter->drawLine(option.rect.left(), option.rect.bottom(), option.rect.right(), option.rect.bottom());

  auto y = option.rect.y();
  painter->translate(visualRect(model()->index(0, 0)).x() - indentation() - .5, -.5);
  for (int i = 0; i < header()->count(); ++i) {
    painter->translate(header()->sectionSize(i), 0);
    painter->drawLine(0, y, 0, y + option.rect.height());
  }
  painter->setPen(old_pen);
  painter->resetTransform();
}

void MessageView::dataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles) {
   // Bypass the slow call to QTreeView::dataChanged.
   // QTreeView::dataChanged will invalidate the height cache and that's what we don't need in MessageView.
   QAbstractItemView::dataChanged(topLeft, bottomRight, roles);
}
