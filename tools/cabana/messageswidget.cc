#include "tools/cabana/messageswidget.h"
#include <QHBoxLayout>
#include <QPainter>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0 ,0, 0, 0);

  QHBoxLayout *title_layout = new QHBoxLayout();
  num_msg_label = new QLabel(this);
  title_layout->addSpacing(10);
  title_layout->addWidget(num_msg_label);

  title_layout->addStretch();
  title_layout->addWidget(multiple_lines_bytes = new QCheckBox(tr("Multiple Lines Bytes"), this));
  multiple_lines_bytes->setToolTip(tr("Display bytes in multiple lines"));
  multiple_lines_bytes->setChecked(settings.multiple_lines_bytes);
  QPushButton *clear_filters = new QPushButton(tr("Clear Filters"));
  title_layout->addWidget(clear_filters);
  main_layout->addLayout(title_layout);

  // message table
  view = new MessageView(this);
  model = new MessageListModel(this);
  header = new MessageViewHeader(this, model);
  auto delegate = new MessageBytesDelegate(view, settings.multiple_lines_bytes);

  view->setItemDelegate(delegate);
  view->setModel(model);
  view->setSortingEnabled(true);
  view->sortByColumn(MessageListModel::Column::NAME, Qt::AscendingOrder);
  view->setAllColumnsShowFocus(true);
  view->setEditTriggers(QAbstractItemView::NoEditTriggers);
  view->setItemsExpandable(false);
  view->setIndentation(0);
  view->setRootIsDecorated(false);
  view->setHeader(header);

  // Must be called before setting any header parameters to avoid overriding
  restoreHeaderState(settings.message_header_state);
  view->header()->setSectionsMovable(true);

  // Header context menu
  view->header()->setContextMenuPolicy(Qt::CustomContextMenu);
  QObject::connect(view->header(), &QHeaderView::customContextMenuRequested, view, &MessageView::headerContextMenuEvent);

  main_layout->addWidget(view);

  // suppress
  QHBoxLayout *suppress_layout = new QHBoxLayout();
  suppress_add = new QPushButton("Suppress Highlighted");
  suppress_clear = new QPushButton();
  suppress_layout->addWidget(suppress_add);
  suppress_layout->addWidget(suppress_clear);
  QCheckBox *suppress_defined_signals = new QCheckBox(tr("Suppress Defined Signals"), this);
  suppress_defined_signals->setChecked(settings.suppress_defined_signals);
  suppress_layout->addWidget(suppress_defined_signals);
  main_layout->addLayout(suppress_layout);

  // signals/slots
  QObject::connect(header, &MessageViewHeader::filtersUpdated, model, &MessageListModel::setFilterStrings);
  QObject::connect(view->horizontalScrollBar(), &QScrollBar::valueChanged, header, &MessageViewHeader::updateHeaderPositions);
  QObject::connect(clear_filters, &QPushButton::clicked, header, &MessageViewHeader::clearFilters);
  QObject::connect(multiple_lines_bytes, &QCheckBox::stateChanged, [=](int state) {
    settings.multiple_lines_bytes = (state == Qt::Checked);
    delegate->setMultipleLines(settings.multiple_lines_bytes);
    view->setUniformRowHeights(!settings.multiple_lines_bytes);

    // Reset model to force recalculation of the width of the bytes column
    model->forceResetModel();
  });
  QObject::connect(suppress_defined_signals, &QCheckBox::stateChanged, [=](int state) {
    settings.suppress_defined_signals = (state == Qt::Checked);
  });
  QObject::connect(can, &AbstractStream::msgsReceived, model, &MessageListModel::msgsReceived);
  QObject::connect(can, &AbstractStream::streamStarted, this, &MessagesWidget::reset);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &MessagesWidget::dbcModified);
  QObject::connect(dbc(), &DBCManager::msgUpdated, this, &MessagesWidget::dbcModified);
  QObject::connect(dbc(), &DBCManager::msgRemoved, this, &MessagesWidget::dbcModified);
  QObject::connect(dbc(), &DBCManager::signalAdded, this, &MessagesWidget::dbcModified);
  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &MessagesWidget::dbcModified);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &MessagesWidget::dbcModified);
  QObject::connect(model, &MessageListModel::modelReset, [this]() {
    if (current_msg_id) {
      selectMessage(*current_msg_id);
    }
    view->updateBytesSectionSize();
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
  dbcModified();

  setWhatsThis(tr(R"(
    <b>Message View</b><br/>
    <!-- TODO: add descprition here -->
    <span style="color:gray">Byte color</span><br />
    <span style="color:gray;">■ </span> constant changing<br />
    <span style="color:blue;">■ </span> increasing<br />
    <span style="color:red;">■ </span> decreasing
  )"));
}

void MessagesWidget::dbcModified() {
  num_msg_label->setText(tr("%1 Messages, %2 Signals").arg(dbc()->msgCount()).arg(dbc()->signalCount()));
  model->fetchData();
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
  updateSuppressedButtons();
}


// MessageListModel

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
    switch (section) {
      case Column::NAME: return tr("Name");
      case Column::SOURCE: return tr("Bus");
      case Column::ADDRESS: return tr("ID");
      case Column::FREQ: return tr("Freq");
      case Column::COUNT: return tr("Count");
      case Column::DATA: return tr("Bytes");
    }
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
      case Column::NAME: return msgName(id);
      case Column::SOURCE: return id.source;
      case Column::ADDRESS: return QString::number(id.address, 16);
      case Column::FREQ: return getFreq(can_data);
      case Column::COUNT: return can_data.count;
      case Column::DATA: return toHex(can_data.dat);
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
  } else if (role == BytesRole && index.column() == Column::DATA) {
    return can_data.dat;
  }
  return {};
}

void MessageListModel::setFilterStrings(const QMap<int, QString> &filters) {
  filter_str = filters;
  fetchData();
}

void MessageListModel::sortMessages(Qt::SortOrder sort_order, int sort_column, QList<MessageId> &new_msgs) {
  if (sort_column == Column::NAME) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::pair{msgName(l), l};
      auto rr = std::pair{msgName(r), r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == Column::SOURCE) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::pair{l.source, l};
      auto rr = std::pair{r.source, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == Column::ADDRESS) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::pair{l.address, l};
      auto rr = std::pair{r.address, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == Column::FREQ) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::pair{can->lastMessage(l).freq, l};
      auto rr = std::pair{can->lastMessage(r).freq, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  } else if (sort_column == Column::COUNT) {
    std::sort(new_msgs.begin(), new_msgs.end(), [=](auto &l, auto &r) {
      auto ll = std::pair{can->lastMessage(l).count, l};
      auto rr = std::pair{can->lastMessage(r).count, r};
      return sort_order == Qt::AscendingOrder ? ll < rr : ll > rr;
    });
  }
}

static std::pair<unsigned int, unsigned int> parseRange(QString &filter, bool *ok = nullptr, int base = 10) {
  // Parse out filter string into a range (e.g. "1" -> {1, 1}, "1-3" -> {1, 3}, "1-" -> {1, inf})
  bool ok1 = true, ok2 = true;
  unsigned int parsed1 = std::numeric_limits<unsigned int>::min();
  unsigned int parsed2 = std::numeric_limits<unsigned int>::max();

  auto s = filter.split('-');
  if (s.size() == 1) {
    parsed1 = s[0].toUInt(ok, base);
    return {parsed1, parsed1};
  } else if (s.size() == 2) {
    if (!s[0].isEmpty()) parsed1 = s[0].toUInt(&ok1, base);
    if (!s[1].isEmpty()) parsed2 = s[1].toUInt(&ok2, base);

    *ok = ok1 & ok2;
    return {parsed1, parsed2};
  } else {
    *ok = false;
    return {0, 0};
  }
}

bool MessageListModel::matchMessage(const MessageId &id, const CanData &data, QMap<int, QString> &filters) {
  auto cs = Qt::CaseInsensitive;
  bool match = true;
  bool convert_ok;

  for (int column = Column::NAME; column <= Column::DATA; column++) {
    if (!filters.contains(column)) continue;
    QString txt = filters[column];

    QRegularExpression re(txt, QRegularExpression::CaseInsensitiveOption | QRegularExpression::DotMatchesEverythingOption);

    switch (column) {
      case Column::NAME:
        {
          bool name_match = re.match(msgName(id)).hasMatch();

          // Message signals
          if (const auto msg = dbc()->msg(id)) {
            for (auto s : msg->getSignals()) {
              if (re.match(s->name).hasMatch()) {
                name_match = true;
                break;
              }
            }
          }
          if (!name_match) match = false;
        }
        break;
      case Column::SOURCE:
        {
          auto source = parseRange(txt, &convert_ok);
          bool source_match = convert_ok && (id.source >= source.first && id.source <= source.second);
          if (!source_match) match = false;
        }
        break;
      case Column::ADDRESS:
        {
          QString address_str = QString::number(id.address, 16);
          bool address_re_match = re.match(address_str).capturedLength() == address_str.length();

          auto address = parseRange(txt, &convert_ok, 16);
          bool address_match = convert_ok && (id.address >= address.first && id.address <= address.second);

          if (!address_re_match && !address_match) match = false;
        }
        break;
      case Column::FREQ:
        {
          // TODO: Hide stale messages?
          auto freq = parseRange(txt, &convert_ok);
          bool freq_match = convert_ok && (data.freq >= freq.first && data.freq <= freq.second);
          if (!freq_match) match = false;
        }
        break;
      case Column::COUNT:
        {
          auto count = parseRange(txt, &convert_ok);
          bool count_match = convert_ok && (data.count >= count.first && data.count <= count.second);
          if (!count_match) match = false;
        }
        break;
      case Column::DATA:
        {
          bool data_match = false;
          data_match |= QString(data.dat.toHex()).contains(txt, cs);
          data_match |= re.match(QString(data.dat.toHex())).hasMatch();
          data_match |= re.match(QString(data.dat.toHex(' '))).hasMatch();

          if (!data_match) match = false;
        }
        break;
    }
  }
  return match;
}


void MessageListModel::fetchData() {
  QList<MessageId> new_msgs;
  for (auto it = can->last_msgs.begin(); it != can->last_msgs.end(); ++it) {
    if (matchMessage(it.key(), it.value(), filter_str)) {
      new_msgs.push_back(it.key());
    }
  }
  sortMessages(sort_order, sort_column, new_msgs);

  if (msgs != new_msgs) {
    beginResetModel();
    msgs = new_msgs;
    endResetModel();
  }
}

void MessageListModel::msgsReceived(const QHash<MessageId, CanData> *new_msgs) {
  QList<MessageId> prev_msgs = msgs;
  fetchData();

  for (int i = 0; i < msgs.size(); ++i) {
    if (new_msgs->contains(msgs[i])) {
      for (int col = Column::FREQ; col < columnCount(); ++col)
        emit dataChanged(index(i, col), index(i, col), {Qt::DisplayRole});
    }
  }
}

void MessageListModel::sort(int column, Qt::SortOrder order) {
  if (column != columnCount() - 1) {
    sort_column = column;
    sort_order = order;
    fetchData();
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
  filter_str.clear();
  msgs.clear();
  clearSuppress();
  endResetModel();
}

void MessageListModel::forceResetModel() {
  beginResetModel();
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
    for (auto it = can->last_msgs.constBegin(); it != can->last_msgs.constEnd(); ++it) {
      max_bytes = std::max(max_bytes, it.value().dat.size());
    }
  }
  int width = delegate->widthForBytes(max_bytes);
  if (header()->sectionSize(5) != width) {
    header()->resizeSection(5, width);
  }
}

void MessageView::headerContextMenuEvent(const QPoint &pos) {
  QMenu *menu = new QMenu(this);
  int cur_index = header()->logicalIndexAt(pos);

  QString column_name;
  QAction *action;
  for (int visual_index = 0; visual_index < header()->count(); visual_index++) {
    int logical_index = header()->logicalIndex(visual_index);
    column_name = model()->headerData(logical_index, Qt::Horizontal).toString();

    // Hide show action
    if (header()->isSectionHidden(logical_index)) {
      action = menu->addAction(tr("  %1").arg(column_name), [=]() { header()->showSection(logical_index); });
    } else {
      action = menu->addAction(tr("✓ %1").arg(column_name), [=]() { header()->hideSection(logical_index); });
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

  menu->popup(header()->mapToGlobal(pos));
}

MessageViewHeader::MessageViewHeader(QWidget *parent, MessageListModel *model) : model(model), QHeaderView(Qt::Horizontal, parent) {
  QObject::connect(this, &QHeaderView::sectionResized, this, &MessageViewHeader::updateHeaderPositions);
  QObject::connect(this, &QHeaderView::sectionMoved, this, &MessageViewHeader::updateHeaderPositions);
}

void MessageViewHeader::showEvent(QShowEvent *e) {

  for (int i = 0; i < count(); i++) {
    if (!editors[i]) {
      QString column_name = model->headerData(i, Qt::Horizontal, Qt::DisplayRole).toString();
      editors[i] = new QLineEdit(this);
      editors[i]->setClearButtonEnabled(true);
      editors[i]->setPlaceholderText(tr("Filter %1").arg(column_name));

      QObject::connect(editors[i], &QLineEdit::textChanged, this, &MessageViewHeader::updateFilters);
    }
    editors[i]->show();
  }
  QHeaderView::showEvent(e);
}

void MessageViewHeader::updateFilters() {
  QMap<int, QString> filters;
  for (int i = 0; i < count(); i++) {
    if (editors[i]) {
      QString filter = editors[i]->text();
      if (!filter.isEmpty()) {
        filters[i] = filter;
      }
    }
  }
  emit filtersUpdated(filters);
}

void MessageViewHeader::clearFilters() {
  for (QLineEdit *editor : editors) {
    editor->clear();
  }
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
  if (editors[0]) {
    setViewportMargins(0, 0, 0, editors[0]->sizeHint().height());
  } else {
    setViewportMargins(0, 0, 0, 0);
  }
  QHeaderView::updateGeometries();
  updateHeaderPositions();
}


QSize MessageViewHeader::sizeHint() const {
  QSize sz = QHeaderView::sizeHint();
  if (editors[0]) {
    sz.setHeight(sz.height() + editors[0]->minimumSizeHint().height());
  }
  return sz;
}
