#include "tools/cabana/messageswidget.h"

#include <algorithm>
#include <limits>

#include <QHBoxLayout>
#include <QPainter>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>

#include "tools/cabana/commands.h"

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // message table
  view = new MessageView(this);
  model = new MessageListModel(this);
  header = new MessageViewHeader(this);
  auto delegate = new MessageBytesDelegate(view, settings.multiple_lines_bytes);

  view->setItemDelegate(delegate);
  view->setUniformRowHeights(!settings.multiple_lines_bytes);
  view->setHeader(header);
  view->setModel(model);
  view->setHeader(header);
  view->setSortingEnabled(true);
  view->sortByColumn(MessageListModel::Column::NAME, Qt::AscendingOrder);
  view->setAllColumnsShowFocus(true);
  view->setEditTriggers(QAbstractItemView::NoEditTriggers);
  view->setItemsExpandable(false);
  view->setIndentation(0);
  view->setRootIsDecorated(false);

  // Must be called before setting any header parameters to avoid overriding
  restoreHeaderState(settings.message_header_state);
  view->header()->setSectionsMovable(true);
  view->header()->setSectionResizeMode(MessageListModel::Column::DATA, QHeaderView::Fixed);
  view->header()->setStretchLastSection(true);

  // Header context menu
  view->header()->setContextMenuPolicy(Qt::CustomContextMenu);
  QObject::connect(view->header(), &QHeaderView::customContextMenuRequested, view, &MessageView::headerContextMenuEvent);

  main_layout->addWidget(view);

  // bottom layout
  QHBoxLayout *bottom_layout = new QHBoxLayout();
  bottom_layout->addWidget(suppress_add = new QPushButton("&Suppress Highlighted"));
  suppress_add->setToolTip(tr("Suppress Highlighted bytes"));
  bottom_layout->addWidget(suppress_clear = new QPushButton());
  suppress_clear->setToolTip(tr("Clear suppressed bytes"));
  QCheckBox *suppress_defined_signals = new QCheckBox(tr("Suppress Signals"), this);
  suppress_defined_signals->setToolTip(tr("Suppress Defined Signals"));
  suppress_defined_signals->setChecked(settings.suppress_defined_signals);
  bottom_layout->addWidget(suppress_defined_signals);
  bottom_layout->addStretch();
  bottom_layout->addWidget(multiple_lines_bytes = new QCheckBox(tr("Multi-Line Bytes"), this));
  multiple_lines_bytes->setToolTip(tr("Display bytes in multiple lines"));
  multiple_lines_bytes->setChecked(settings.multiple_lines_bytes);

  main_layout->addLayout(bottom_layout);

  // signals/slots
  QObject::connect(header, &MessageViewHeader::filtersUpdated, model, &MessageListModel::setFilterStrings);
  QObject::connect(view->horizontalScrollBar(), &QScrollBar::valueChanged, header, &MessageViewHeader::updateHeaderPositions);
  QObject::connect(multiple_lines_bytes, &QCheckBox::stateChanged, [=](int state) {
    settings.multiple_lines_bytes = (state == Qt::Checked);
    delegate->setMultipleLines(settings.multiple_lines_bytes);
    view->setUniformRowHeights(!settings.multiple_lines_bytes);
    view->updateBytesSectionSize();
  });
  QObject::connect(can, &AbstractStream::msgsReceived, model, &MessageListModel::msgsReceived);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, model, &MessageListModel::dbcModified);
  QObject::connect(UndoStack::instance(), &QUndoStack::indexChanged, model, &MessageListModel::dbcModified);
  QObject::connect(model, &MessageListModel::modelReset, [this]() {
    if (current_msg_id) {
      selectMessage(*current_msg_id);
    }
    view->updateBytesSectionSize();
    updateTitle();
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

  QObject::connect(suppress_defined_signals, &QCheckBox::stateChanged, can, &AbstractStream::suppressDefinedSignals);
  QObject::connect(suppress_add, &QPushButton::clicked, [this]() {
    size_t cnt = can->suppressHighlighted();
    updateSuppressedButtons(cnt);
  });
  QObject::connect(suppress_clear, &QPushButton::clicked, [this]() {
    can->clearSuppressed();
    updateSuppressedButtons(0);
  });

  updateSuppressedButtons(0);

  setWhatsThis(tr(R"(
    <b>Message View</b><br/>
    <!-- TODO: add descprition here -->
    <span style="color:gray">Byte color</span><br />
    <span style="color:gray;">■ </span> constant changing<br />
    <span style="color:blue;">■ </span> increasing<br />
    <span style="color:red;">■ </span> decreasing
  )"));
}

void MessagesWidget::updateTitle() {
  size_t dbc_msg_count = 0;
  size_t signal_count = 0;
  for (const auto &msg_id : model->msgs) {
    if (auto m = dbc()->msg(msg_id)) {
      ++dbc_msg_count;
      signal_count += m->sigs.size();
    }
  }
  QString title = tr("%1 Messages (%2 DBC Messages, %3 Signals)")
                      .arg(model->msgs.size()).arg(dbc_msg_count).arg(signal_count);
  emit titleChanged(title);
}

void MessagesWidget::selectMessage(const MessageId &msg_id) {
  auto it = std::find(model->msgs.cbegin(), model->msgs.cend(), msg_id);
  if (it != model->msgs.cend()) {
    view->setCurrentIndex(model->index(std::distance(model->msgs.cbegin(), it), 0));
  }
}

void MessagesWidget::updateSuppressedButtons(size_t n) {
  if (!n) {
    suppress_clear->setEnabled(false);
    suppress_clear->setText("&Clear");
  } else {
    suppress_clear->setEnabled(true);
    suppress_clear->setText(QString("&Clear (%1)").arg(n));
  }
}

// MessageListModel
MessageListModel::MessageListModel(QObject *parent) : QAbstractTableModel(parent) {
  sort_timer.setSingleShot(true);
  sort_timer.callOnTimeout(this, &MessageListModel::filterAndSort);
}

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  static const QVariant titles[] = {tr("Name"), tr("Bus"), tr("Address"), tr("Freq"), tr("Count"), tr("Bytes")};
  return orientation == Qt::Horizontal && role == Qt::DisplayRole ? titles[section] : QVariant();
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  auto getFreq = [](const CanData &d) -> QString {
    if (d.freq > 0 && (can->currentSec() - can->toSeconds(d.mono_time) - 1.0 / settings.fps) < (5.0 / d.freq)) {
      return d.freq >= 0.95 ? QString::number(std::nearbyint(d.freq)) : QString::number(d.freq, 'f', 2);
    } else {
      return "--";
    }
  };

  const auto &id = msgs[index.row()];
  auto &can_data = can->lastMessage(id);
  if (role == Qt::DisplayRole) {
    switch (index.column()) {
      case Column::NAME: return msgName(id);
      case Column::SOURCE: return id.source != INVALID_SOURCE ? QString::number(id.source) : "N/A";
      case Column::ADDRESS: return QString::number(id.address, 16);
      case Column::FREQ: return id.source != INVALID_SOURCE ? getFreq(can_data) : "N/A";
      case Column::COUNT: return id.source != INVALID_SOURCE ? QString::number(can_data.count) : "N/A";
      case Column::DATA: return id.source != INVALID_SOURCE ? "" : "N/A";
    }
  } else if (role == ColorsRole) {
    return QVariant::fromValue((void*)(&can_data.colors));
  } else if (role == BytesRole && index.column() == Column::DATA && id.source != INVALID_SOURCE) {
    return QVariant::fromValue((void*)(&can_data.dat));
  } else if (role == Qt::ToolTipRole && index.column() == Column::NAME) {
    auto msg = dbc()->msg(id);
    auto tooltip = msg ? msg->name : UNTITLED;
    if (msg && !msg->comment.isEmpty()) tooltip += "<br /><span style=\"color:gray;\">" + msg->comment + "</span>";
    return tooltip;
  }
  return {};
}

void MessageListModel::setFilterStrings(const QMap<int, QString> &filters) {
  filter_str = filters;
  filterAndSort();
}

void MessageListModel::dbcModified() {
  dbc_address.clear();
  for (const auto &[_, m] : dbc()->getMessages(-1)) {
    dbc_address.insert(m.address);
  }
  filterAndSort();
}

void MessageListModel::sortMessages(std::vector<MessageId> &new_msgs) {
  if (sort_column == Column::NAME) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::pair{msgName(l), l};
      auto rr = std::pair{msgName(r), r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == Column::SOURCE) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::tie(l.source, l);
      auto rr = std::tie(r.source, r);
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == Column::ADDRESS) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::tie(l.address, l);
      auto rr = std::tie(r.address, r);
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == Column::FREQ) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::tie(can->lastMessage(l).freq, l);
      auto rr = std::tie(can->lastMessage(r).freq, r);
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == Column::COUNT) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::tie(can->lastMessage(l).count, l);
      auto rr = std::tie(can->lastMessage(r).count, r);
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  }
  last_sort_ts = millis_since_boot();
}

static bool parseRange(const QString &filter, uint32_t value, int base = 10) {
  // Parse out filter string into a range (e.g. "1" -> {1, 1}, "1-3" -> {1, 3}, "1-" -> {1, inf})
  unsigned int min = std::numeric_limits<unsigned int>::min();
  unsigned int max = std::numeric_limits<unsigned int>::max();
  auto s = filter.split('-');
  bool ok = s.size() >= 1 && s.size() <= 2;
  if (ok && !s[0].isEmpty()) min = s[0].toUInt(&ok, base);
  if (ok && s.size() == 1) {
    max = min;
  } else if (ok && s.size() == 2 && !s[1].isEmpty()) {
    max = s[1].toUInt(&ok, base);
  }
  return ok && value >= min && value <= max;
}

bool MessageListModel::matchMessage(const MessageId &id, const CanData &data, const QMap<int, QString> &filters) {
  bool match = true;
  for (auto it = filters.cbegin(); it != filters.cend() && match; ++it) {
    const QString &txt = it.value();
    QRegularExpression re(txt, QRegularExpression::CaseInsensitiveOption | QRegularExpression::DotMatchesEverythingOption);
    switch (it.key()) {
      case Column::NAME: {
        const auto msg = dbc()->msg(id);
        match = re.match(msg ? msg->name : UNTITLED).hasMatch();
        match = match || (msg && std::any_of(msg->sigs.cbegin(), msg->sigs.cend(),
                                             [&re](const auto &s) { return re.match(s->name).hasMatch(); }));
        break;
      }
      case Column::SOURCE:
        match = parseRange(txt, id.source);
        break;
      case Column::ADDRESS: {
        match = re.match(QString::number(id.address, 16)).hasMatch();
        match = match || parseRange(txt, id.address, 16);
        break;
      }
      case Column::FREQ:
        // TODO: Hide stale messages?
        match = parseRange(txt, data.freq);
        break;
      case Column::COUNT:
        match = parseRange(txt, data.count);
        break;
      case Column::DATA: {
        QString hex = utils::toHex(data.dat);
        match = hex.contains(txt, Qt::CaseInsensitive);
        match = match || re.match(hex).hasMatch();
        match = match || re.match(utils::toHex(data.dat, ' ')).hasMatch();
        break;
      }
    }
  }
  return match;
}

void MessageListModel::filterAndSort() {
  std::vector<MessageId> new_msgs;
  new_msgs.reserve(can->lastMessages().size() + dbc_address.size());

  auto address = dbc_address;
  for (const auto &[id, m] : can->lastMessages()) {
    if (filter_str.isEmpty() || matchMessage(id, m, filter_str)) {
      new_msgs.push_back(id);
    }
    address.remove(id.address);
  }

  // merge all DBC messages
  for (auto &addr : address) {
    MessageId id{.source = INVALID_SOURCE, .address = addr};
    if (filter_str.isEmpty() || matchMessage(id, {}, filter_str)) {
      new_msgs.push_back(id);
    }
  }

  sortMessages(new_msgs);

  if (msgs != new_msgs) {
    beginResetModel();
    msgs = std::move(new_msgs);
    endResetModel();
  }
}

void MessageListModel::msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids) {
  if (has_new_ids) {
    sort_timer.start(110);
  } else if (!filter_str.empty()) {
    bool resort = (filter_str.contains(Column::FREQ) || filter_str.contains(Column::COUNT) ||
                      filter_str.contains(Column::DATA));
    if (resort && ((millis_since_boot() - last_sort_ts) >= 1000)) {
      filterAndSort();
      return;
    }
  }

  for (int i = 0; i < msgs.size(); ++i) {
    if (!new_msgs || new_msgs->count(msgs[i])) {
      for (int col = Column::FREQ; col < columnCount(); ++col)
        emit dataChanged(index(i, col), index(i, col), {Qt::DisplayRole});
    }
  }
}

void MessageListModel::sort(int column, Qt::SortOrder order) {
  if (column != columnCount() - 1) {
    sort_column = column;
    sort_order = order;
    filterAndSort();
  }
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
    painter->translate(header()->sectionSize(header()->logicalIndex(i)), 0);
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

void MessageView::updateBytesSectionSize() {
  auto delegate = ((MessageBytesDelegate *)itemDelegate());
  int max_bytes = 8;
  if (!delegate->multipleLines()) {
    for (const auto &[_, m] : can->lastMessages()) {
      max_bytes = std::max<int>(max_bytes, m.dat.size());
    }
  }
  int width = delegate->sizeForBytes(max_bytes).width();
  if (header()->sectionSize(MessageListModel::Column::DATA) != width) {
    header()->resizeSection(MessageListModel::Column::DATA, width);
  }
}

void MessageView::headerContextMenuEvent(const QPoint &pos) {
  QMenu menu(this);
  int cur_index = header()->logicalIndexAt(pos);

  QAction *action;
  for (int visual_index = 0; visual_index < header()->count(); visual_index++) {
    int logical_index = header()->logicalIndex(visual_index);
    QString column_name = model()->headerData(logical_index, Qt::Horizontal).toString();

    // Hide show action
    if (header()->isSectionHidden(logical_index)) {
      action = menu.addAction(tr("  %1").arg(column_name), [=]() { header()->showSection(logical_index); });
    } else {
      action = menu.addAction(tr("✓ %1").arg(column_name), [=]() { header()->hideSection(logical_index); });
    }

    // Can't hide the name column
    action->setEnabled(logical_index > 0);

    // Make current column bold
    if (logical_index == cur_index) {
      QFont font = action->font();
      font.setBold(true);
      action->setFont(font);
    }
  }

  menu.exec(header()->mapToGlobal(pos));
}

MessageViewHeader::MessageViewHeader(QWidget *parent) : QHeaderView(Qt::Horizontal, parent) {
  QObject::connect(this, &QHeaderView::sectionResized, this, &MessageViewHeader::updateHeaderPositions);
  QObject::connect(this, &QHeaderView::sectionMoved, this, &MessageViewHeader::updateHeaderPositions);
}

void MessageViewHeader::updateFilters() {
  QMap<int, QString> filters;
  for (int i = 0; i < count(); i++) {
    if (editors[i] && !editors[i]->text().isEmpty()) {
      filters[i] = editors[i]->text();
    }
  }
  emit filtersUpdated(filters);
}

void MessageViewHeader::updateHeaderPositions() {
  QSize sz = QHeaderView::sizeHint();
  for (int i = 0; i < count(); i++) {
    if (editors[i]) {
      int h = editors[i]->sizeHint().height();
      editors[i]->move(sectionViewportPosition(i), sz.height());
      editors[i]->resize(sectionSize(i), h);
      editors[i]->setHidden(isSectionHidden(i));
    }
  }
}

void MessageViewHeader::updateGeometries() {
  for (int i = 0; i < count(); i++) {
    if (!editors[i]) {
      QString column_name = model()->headerData(i, Qt::Horizontal, Qt::DisplayRole).toString();
      editors[i] = new QLineEdit(this);
      editors[i]->setClearButtonEnabled(true);
      editors[i]->setPlaceholderText(tr("Filter %1").arg(column_name));

      QObject::connect(editors[i], &QLineEdit::textChanged, this, &MessageViewHeader::updateFilters);
    }
  }
  setViewportMargins(0, 0, 0, editors[0] ? editors[0]->sizeHint().height() : 0);

  QHeaderView::updateGeometries();
  updateHeaderPositions();
}

QSize MessageViewHeader::sizeHint() const {
  QSize sz = QHeaderView::sizeHint();
  if (editors[0])
    sz.setHeight(sz.height() + editors[0]->minimumSizeHint().height() + 1);
  return sz;
}
