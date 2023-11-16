#include "tools/cabana/messageswidget.h"

#include <limits>

#include <QCheckBox>
#include <QHBoxLayout>
#include <QPainter>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>

#include "tools/cabana/commands.h"

MessagesWidget::MessagesWidget(QWidget *parent) : menu(new QMenu(this)), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);
  // toolbar
  main_layout->addWidget(createToolBar());
  // message table
  main_layout->addWidget(view = new MessageView(this));
  view->setItemDelegate(delegate = new MessageBytesDelegate(view, settings.multiple_lines_hex));
  view->setModel(model = new MessageListModel(this));
  view->setHeader(header = new MessageViewHeader(this));
  view->setSortingEnabled(true);
  view->sortByColumn(MessageListModel::Column::NAME, Qt::AscendingOrder);
  view->setAllColumnsShowFocus(true);
  view->setEditTriggers(QAbstractItemView::NoEditTriggers);
  view->setItemsExpandable(false);
  view->setIndentation(0);
  view->setRootIsDecorated(false);
  view->setUniformRowHeights(!settings.multiple_lines_hex);

  // Must be called before setting any header parameters to avoid overriding
  restoreHeaderState(settings.message_header_state);
  header->setSectionsMovable(true);
  header->setSectionResizeMode(MessageListModel::Column::DATA, QHeaderView::Fixed);
  header->setStretchLastSection(true);
  header->setContextMenuPolicy(Qt::CustomContextMenu);

  // suppress
  QHBoxLayout *suppress_layout = new QHBoxLayout();
  suppress_layout->addWidget(suppress_add = new QPushButton("Suppress Highlighted"));
  suppress_layout->addWidget(suppress_clear = new QPushButton());
  suppress_clear->setToolTip(tr("Clear suppressed"));
  suppress_layout->addStretch(1);
  QCheckBox *suppress_defined_signals = new QCheckBox(tr("Suppress Signals"), this);
  suppress_defined_signals->setToolTip(tr("Suppress defined signals"));
  suppress_defined_signals->setChecked(settings.suppress_defined_signals);
  suppress_layout->addWidget(suppress_defined_signals);
  main_layout->addLayout(suppress_layout);

  // signals/slots
  QObject::connect(menu, &QMenu::aboutToShow, this, &MessagesWidget::menuAboutToShow);
  QObject::connect(header, &MessageViewHeader::filtersUpdated, model, &MessageListModel::setFilterStrings);
  QObject::connect(header, &MessageViewHeader::customContextMenuRequested, this, &MessagesWidget::headerContextMenuEvent);
  QObject::connect(view->horizontalScrollBar(), &QScrollBar::valueChanged, header, &MessageViewHeader::updateHeaderPositions);
  QObject::connect(suppress_defined_signals, &QCheckBox::stateChanged, can, &AbstractStream::suppressDefinedSignals);
  QObject::connect(can, &AbstractStream::msgsReceived, model, &MessageListModel::msgsReceived);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &MessagesWidget::dbcModified);
  QObject::connect(UndoStack::instance(), &QUndoStack::indexChanged, this, &MessagesWidget::dbcModified);
  QObject::connect(model, &MessageListModel::modelReset, [this]() {
    if (current_msg_id) {
      selectMessage(*current_msg_id);
    }
    view->updateBytesSectionSize();
  });
  QObject::connect(view->selectionModel(), &QItemSelectionModel::currentChanged, [=](const QModelIndex &current, const QModelIndex &previous) {
    if (current.isValid() && current.row() < model->items_.size()) {
      const auto &id = model->items_[current.row()].id;
      if (!current_msg_id || id != *current_msg_id) {
        current_msg_id = id;
        emit msgSelectionChanged(*current_msg_id);
      }
    }
  });
  QObject::connect(suppress_add, &QPushButton::clicked, this, &MessagesWidget::suppressHighlighted);
  QObject::connect(suppress_clear, &QPushButton::clicked, this, &MessagesWidget::suppressHighlighted);
  suppressHighlighted();

  setWhatsThis(tr(R"(
    <b>Message View</b><br/>
    <!-- TODO: add descprition here -->
    <span style="color:gray">Byte color</span><br />
    <span style="color:gray;">■ </span> constant changing<br />
    <span style="color:blue;">■ </span> increasing<br />
    <span style="color:red;">■ </span> decreasing
  )"));
}

QToolBar *MessagesWidget::createToolBar() {
  QToolBar *toolbar = new QToolBar(this);
  toolbar->setIconSize({12, 12});
  toolbar->addWidget(num_msg_label = new QLabel(this));
  num_msg_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

  auto views_btn = toolbar->addAction(utils::icon("three-dots"), tr("View..."));
  views_btn->setMenu(menu);
  auto view_button = qobject_cast<QToolButton *>(toolbar->widgetForAction(views_btn));
  view_button->setPopupMode(QToolButton::InstantPopup);
  view_button->setToolButtonStyle(Qt::ToolButtonIconOnly);
  view_button->setStyleSheet("QToolButton::menu-indicator { image: none; }");
  return toolbar;
}

void MessagesWidget::dbcModified() {
  num_msg_label->setText(tr("%1 Messages, %2 Signals").arg(dbc()->msgCount()).arg(dbc()->signalCount()));
  model->dbcModified();
}

void MessagesWidget::selectMessage(const MessageId &msg_id) {
  auto it = std::find_if(model->items_.cbegin(), model->items_.cend(),
                         [&msg_id](auto &item) { return item.id == msg_id; });
  if (it != model->items_.cend()) {
    view->setCurrentIndex(model->index(std::distance(model->items_.cbegin(), it), 0));
  }
}

void MessagesWidget::suppressHighlighted() {
  if (sender() == suppress_add) {
    size_t n = can->suppressHighlighted();
    suppress_clear->setText(tr("Clear (%1)").arg(n));
    suppress_clear->setEnabled(true);
  } else {
    can->clearSuppressed();
    suppress_clear->setText(tr("Clear"));
    suppress_clear->setEnabled(false);
  }
}

void MessagesWidget::headerContextMenuEvent(const QPoint &pos) {
  menu->exec(header->mapToGlobal(pos));
}

void MessagesWidget::menuAboutToShow() {
  menu->clear();
  for (int i = 0; i < header->count(); ++i) {
    int logical_index = header->logicalIndex(i);
    auto action = menu->addAction(model->headerData(logical_index, Qt::Horizontal).toString(),
                                  [=](bool checked) { header->setSectionHidden(logical_index, !checked); });
    action->setCheckable(true);
    action->setChecked(!header->isSectionHidden(logical_index));
    // Can't hide the name column
    action->setEnabled(logical_index > 0);
  }
  menu->addSeparator();
  auto action = menu->addAction(tr("Mutlti-Line bytes"), this, &MessagesWidget::setMultiLineBytes);
  action->setCheckable(true);
  action->setChecked(settings.multiple_lines_hex);
}

void MessagesWidget::setMultiLineBytes(bool multi) {
  settings.multiple_lines_hex = multi;
  delegate->setMultipleLines(multi);
  view->setUniformRowHeights(!multi);
  view->updateBytesSectionSize();
  view->doItemsLayout();
}

// MessageListModel

QVariant MessageListModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
    switch (section) {
      case Column::NAME: return tr("Name");
      case Column::SOURCE: return tr("Bus");
      case Column::ADDRESS: return tr("ID");
      case Column::NODE: return tr("Node");
      case Column::FREQ: return tr("Freq");
      case Column::COUNT: return tr("Count");
      case Column::DATA: return tr("Bytes");
    }
  }
  return {};
}

QVariant MessageListModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid() || index.row() >= items_.size()) return {};

  auto getFreq = [](const CanData &d) {
    if (d.freq > 0 && (can->currentSec() - d.ts - 1.0 / settings.fps) < (5.0 / d.freq)) {
      return d.freq >= 0.95 ? QString::number(std::nearbyint(d.freq)) : QString::number(d.freq, 'f', 2);
    } else {
      return QStringLiteral("--");
    }
  };

  const auto &item = items_[index.row()];
  const auto &data = can->lastMessage(item.id);
  if (role == Qt::DisplayRole) {
    switch (index.column()) {
      case Column::NAME: return item.name;
      case Column::SOURCE: return item.id.source != INVALID_SOURCE ? QString::number(item.id.source) : "N/A";
      case Column::ADDRESS: return QString::number(item.id.address, 16);
      case Column::NODE: return item.node;
      case Column::FREQ: return item.id.source != INVALID_SOURCE ? getFreq(data) : "N/A";
      case Column::COUNT: return item.id.source != INVALID_SOURCE ? QString::number(data.count) : "N/A";
      case Column::DATA: return item.id.source != INVALID_SOURCE ? "" : "N/A";
    }
  } else if (role == ColorsRole) {
    return QVariant::fromValue((void*)(&data.colors));
  } else if (role == BytesRole && index.column() == Column::DATA && item.id.source != INVALID_SOURCE) {
    return QVariant::fromValue((void*)(&data.dat));
  } else if (role == Qt::ToolTipRole && index.column() == Column::NAME) {
    auto msg = dbc()->msg(item.id);
    auto tooltip = item.name;
    if (msg && !msg->comment.isEmpty()) tooltip += "<br /><span style=\"color:gray;\">" + msg->comment + "</span>";
    return tooltip;
  }
  return {};
}

void MessageListModel::setFilterStrings(const QMap<int, QString> &filters) {
  filters_ = filters;
  filterAndSort();
}

void MessageListModel::dbcModified() {
  dbc_messages_.clear();
  for (const auto &[_, m] : dbc()->getMessages(-1)) {
    dbc_messages_.insert(MessageId{.source = INVALID_SOURCE, .address = m.address});
  }
  filterAndSort();
}

void MessageListModel::sortItems(std::vector<MessageListModel::Item> &items) {
  auto do_sort = [order = sort_order](std::vector<MessageListModel::Item> &m, auto proj) {
    std::stable_sort(m.begin(), m.end(), [order, proj = std::move(proj)](auto &l, auto &r) {
      return order == Qt::AscendingOrder ? proj(l) < proj(r) : proj(l) > proj(r);
    });
  };
  switch (sort_column) {
    case Column::NAME: do_sort(items, [](auto &item) { return std::tie(item.name, item.id); }); break;
    case Column::SOURCE: do_sort(items, [](auto &item) { return std::tie(item.id.source, item.id); }); break;
    case Column::ADDRESS: do_sort(items, [](auto &item) { return std::tie(item.id.address, item.id);}); break;
    case Column::NODE: do_sort(items, [](auto &item) { return std::tie(item.node, item.id);}); break;
    case Column::FREQ: do_sort(items, [](auto &item) { return std::make_pair(can->lastMessage(item.id).freq, item.id); }); break;
    case Column::COUNT: do_sort(items, [](auto &item) { return std::make_pair(can->lastMessage(item.id).count, item.id); }); break;
  }
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

bool MessageListModel::match(const MessageListModel::Item &item) {
  if (filters_.isEmpty())
    return true;

  bool match = true;
  const auto &data = can->lastMessage(item.id);
  for (auto it = filters_.cbegin(); it != filters_.cend() && match; ++it) {
    const QString &txt = it.value();
    switch (it.key()) {
      case Column::NAME: {
        match = item.name.contains(txt, Qt::CaseInsensitive);
        if (!match) {
          const auto m = dbc()->msg(item.id);
          match = m && std::any_of(m->sigs.cbegin(), m->sigs.cend(),
                                   [&txt](const auto &s) { return s->name.contains(txt, Qt::CaseInsensitive); });
        }
        break;
      }
      case Column::SOURCE:
        match = parseRange(txt, item.id.source);
        break;
      case Column::ADDRESS:
        match = QString::number(item.id.address, 16).contains(txt, Qt::CaseInsensitive);
        match = match || parseRange(txt, item.id.address, 16);
        break;
      case Column::NODE:
        match = item.node.contains(txt, Qt::CaseInsensitive);
        break;
      case Column::FREQ:
        // TODO: Hide stale messages?
        match = parseRange(txt, data.freq);
        break;
      case Column::COUNT:
        match = parseRange(txt, data.count);
        break;
      case Column::DATA:
        match = utils::toHex(data.dat).contains(txt, Qt::CaseInsensitive);
        break;
    }
  }
  return match;
}

void MessageListModel::filterAndSort() {
  // merge CAN and DBC messages
  std::vector<MessageId> all_messages;
  all_messages.reserve(can->lastMessages().size() + dbc_messages_.size());
  auto dbc_msgs = dbc_messages_;
  for (const auto &[id, m] : can->lastMessages()) {
    all_messages.push_back(id);
    dbc_msgs.erase(MessageId{.source = INVALID_SOURCE, .address = id.address});
  }
  all_messages.insert(all_messages.end(), dbc_msgs.begin(), dbc_msgs.end());

  // filter and sort
  std::vector<Item> items;
  for (const auto &id : all_messages) {
    auto msg = dbc()->msg(id);
    Item item = {.id = id,
                 .name = msg ? msg->name : UNTITLED,
                 .node = msg ? msg->transmitter : QString()};
    if (match(item))
      items.emplace_back(item);
  }
  sortItems(items);

  if (items_ != items) {
    beginResetModel();
    items_ = std::move(items);
    endResetModel();
  }
}

void MessageListModel::msgsReceived(const std::set<MessageId> *new_msgs, bool has_new_ids) {
  if (has_new_ids || filters_.contains(Column::FREQ) || filters_.contains(Column::COUNT) || filters_.contains(Column::DATA)) {
    filterAndSort();
  }
  for (int i = 0; i < items_.size(); ++i) {
    if (!new_msgs || new_msgs->count(items_[i].id)) {
      for (int col = Column::FREQ; col < columnCount(); ++col)
        emit dataChanged(index(i, col), index(i, col), {Qt::DisplayRole});
    }
  }
}

void MessageListModel::sort(int column, Qt::SortOrder order) {
  if (column != Column::DATA) {
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
  header()->resizeSection(MessageListModel::Column::DATA, delegate->sizeForBytes(max_bytes).width());
}

// MessageViewHeader

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
      editors[i]->setGeometry(sectionViewportPosition(i), sz.height(), sectionSize(i), h);
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
  return editors[0] ? QSize(sz.width(), sz.height() + editors[0]->height() + 1) : sz;
}
