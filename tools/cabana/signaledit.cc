#include "tools/cabana/signaledit.h"

#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>

#include "tools/cabana/commands.h"

// SignalModel

SignalModel::SignalModel(QObject *parent) : root(new Item), QAbstractItemModel(parent) {
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &SignalModel::refresh);
  QObject::connect(dbc(), &DBCManager::msgUpdated, this, &SignalModel::handleMsgChanged);
  QObject::connect(dbc(), &DBCManager::msgRemoved, this, &SignalModel::handleMsgChanged);
  QObject::connect(dbc(), &DBCManager::signalAdded, this, &SignalModel::handleSignalAdded);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &SignalModel::handleSignalUpdated);
  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &SignalModel::handleSignalRemoved);
  QObject::connect(can, &AbstractStream::msgsReceived, this, &SignalModel::updateState);
}

void SignalModel::insertItem(SignalModel::Item *parent_item, int pos, const Signal *sig) {
  Item *item = new Item{.sig = sig, .parent = parent_item, .title = sig->name.c_str(), .type = Item::Sig};
  parent_item->children.insert(pos, item);
  QString titles[]{"Name", "Size", "Little Endian", "Signed", "Offset", "Factor", "Extra Info", "Unit", "Comment", "Minimum", "Maximum", "Description"};
  for (int i = 0; i < std::size(titles); ++i) {
    item->children.push_back(new Item{.sig = sig, .parent = item, .title = titles[i], .type = (Item::Type)(i + Item::Name)});
  }
}

void SignalModel::setMessage(const QString &id) {
  msg_id = id;
  filter_str = "";
  refresh();
  updateState(nullptr);
}

void SignalModel::setFilter(const QString &txt) {
  filter_str = txt;
  refresh();
}

void SignalModel::refresh() {
  beginResetModel();
  root.reset(new SignalModel::Item);
  if (auto msg = dbc()->msg(msg_id)) {
    for (auto &s : msg->getSignals()) {
      if (filter_str.isEmpty() || QString::fromStdString(s->name).contains(filter_str, Qt::CaseInsensitive)) {
        insertItem(root.get(), root->children.size(), s);
      }
    }
  }
  endResetModel();
}

void SignalModel::updateState(const QHash<QString, CanData> *msgs) {
  if (!msgs || (msgs->contains(msg_id))) {
    auto &dat = can->lastMessage(msg_id).dat;
    int row = 0;
    for (auto item : root->children) {
      double value = get_raw_value((uint8_t *)dat.begin(), dat.size(), *item->sig);
      item->sig_val = QString::number(value);
      emit dataChanged(index(row, 1), index(row, 1), {Qt::DisplayRole});
      ++row;
    }
  }
}

int SignalModel::rowCount(const QModelIndex &parent) const {
  if (parent.column() > 0) return 0;

  auto parent_item = getItem(parent);
  int row_count = parent_item->children.size();
  if (parent_item->type == Item::Sig && !parent_item->extra_expanded) {
    row_count -= (Item::Desc - Item::ExtraInfo);
  }
  return row_count;
}

Qt::ItemFlags SignalModel::flags(const QModelIndex &index) const {
  if (!index.isValid()) return Qt::NoItemFlags;

  auto item = getItem(index);
  Qt::ItemFlags flags = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
  if (index.column() == 1  && item->type != Item::Sig && item->type != Item::ExtraInfo) {
    flags |= (item->type == Item::Endian || item->type == Item::Signed) ? Qt::ItemIsUserCheckable : Qt::ItemIsEditable;
  }
  return flags;
}

int SignalModel::signalRow(const Signal *sig) const {
  auto &children = root->children;
  for (int i = 0; i < children.size(); ++i) {
    if (children[i]->sig == sig) return i;
  }
  return -1;
}

QModelIndex SignalModel::index(int row, int column, const QModelIndex &parent) const {
  if (!hasIndex(row, column, parent)) return {};
  return createIndex(row, column, getItem(parent)->children[row]);
}

QModelIndex SignalModel::parent(const QModelIndex &index) const {
  if (!index.isValid()) return {};
  Item *parent_item = getItem(index)->parent;
  return parent_item == root.get() ? QModelIndex() : createIndex(parent_item->row(), 0, parent_item);
}

QVariant SignalModel::data(const QModelIndex &index, int role) const {
  if (index.isValid()) {
    const Item *item = getItem(index);
    if (role == Qt::DisplayRole || role == Qt::EditRole) {
      if (index.column() == 0) {
        return item->type == Item::Sig ? QString::fromStdString(item->sig->name) : item->title;
      } else {
        switch (item->type) {
          case Item::Sig: return item->sig_val;
          case Item::Name: return QString::fromStdString(item->sig->name);
          case Item::Size: return item->sig->size;
          case Item::Offset: return QString::number(item->sig->offset, 'f', 6);
          case Item::Factor: return QString::number(item->sig->factor, 'f', 6);
          default: break;
        }
      }
    } else if (role == Qt::CheckStateRole && index.column() == 1) {
      if (item->type == Item::Endian) return item->sig->is_little_endian ? Qt::Checked : Qt::Unchecked;
      if (item->type == Item::Signed) return item->sig->is_signed ? Qt::Checked : Qt::Unchecked;
    } else if (role == Qt::DecorationRole && index.column() == 0 && item->type == Item::ExtraInfo) {
      return utils::icon(item->parent->extra_expanded ? "chevron-compact-down" : "chevron-compact-up");
    }
  }
  return {};
}

bool SignalModel::setData(const QModelIndex &index, const QVariant &value, int role) {
  if (role != Qt::EditRole && role != Qt::CheckStateRole) return false;

  Item *item = getItem(index);
  Signal s = *item->sig;
  switch (item->type) {
    case Item::Name: s.name = value.toString().toStdString(); break;
    case Item::Size: s.size = value.toInt(); break;
    case Item::Endian: s.is_little_endian = value.toBool(); break;
    case Item::Signed: s.is_signed = value.toBool(); break;
    case Item::Offset: s.offset = value.toDouble(); break;
    case Item::Factor: s.factor = value.toDouble(); break;
    default: return false;
  }
  bool ret = saveSignal(item->sig, s);
  emit dataChanged(index, index, {Qt::DisplayRole, Qt::EditRole, Qt::CheckStateRole});
  return ret;
}

void SignalModel::showExtraInfo(const QModelIndex &index) {
  auto item = getItem(index);
  if (item->type == Item::ExtraInfo) {
    if (!item->parent->extra_expanded) {
      item->parent->extra_expanded = true;
      beginInsertRows(index.parent(), 7, 13);
      endInsertRows();
    } else {
      item->parent->extra_expanded = false;
      beginRemoveRows(index.parent(), 7, 13);
      endRemoveRows();
    }
  }
}

bool SignalModel::saveSignal(const Signal *origin_s, Signal &s) {
  auto msg = dbc()->msg(msg_id);
  if (s.name != origin_s->name && msg->sigs.count(s.name.c_str()) != 0) {
    QString text = tr("There is already a signal with the same name '%1'").arg(s.name.c_str());
    QMessageBox::warning(nullptr, tr("Failed to save signal"), text);
    return false;
  }

  if (s.is_little_endian != origin_s->is_little_endian) {
    int start = std::floor(s.start_bit / 8);
    if (s.is_little_endian) {
      int end = std::floor((s.start_bit - s.size + 1) / 8);
      s.start_bit = start == end ? s.start_bit - s.size + 1 : bigEndianStartBitsIndex(s.start_bit);
    } else {
      int end = std::floor((s.start_bit + s.size - 1) / 8);
      s.start_bit = start == end ? s.start_bit + s.size - 1 : bigEndianBitIndex(s.start_bit);
    }
  }
  if (s.is_little_endian) {
    s.lsb = s.start_bit;
    s.msb = s.start_bit + s.size - 1;
  } else {
    s.lsb = bigEndianStartBitsIndex(bigEndianBitIndex(s.start_bit) + s.size - 1);
    s.msb = s.start_bit;
  }

  UndoStack::push(new EditSignalCommand(msg_id, origin_s, s));
  return true;
}

void SignalModel::addSignal(int start_bit, int size, bool little_endian) {
  auto msg = dbc()->msg(msg_id);
  for (int i = 1; !msg; ++i) {
    QString name = QString("NEW_MSG_%1").arg(i);
    if (std::none_of(dbc()->messages().begin(), dbc()->messages().end(), [&](auto &m) { return m.second.name == name; })) {
      UndoStack::push(new EditMsgCommand(msg_id, name, can->lastMessage(msg_id).dat.size()));
      msg = dbc()->msg(msg_id);
    }
  }

  Signal sig = {.is_little_endian = little_endian, .factor = 1};
  for (int i = 1; /**/; ++i) {
    sig.name = "NEW_SIGNAL_" + std::to_string(i);
    if (msg->sigs.count(sig.name.c_str()) == 0) break;
  }
  updateSigSizeParamsFromRange(sig, start_bit, size);
  UndoStack::push(new AddSigCommand(msg_id, sig));
}

void SignalModel::resizeSignal(const Signal *sig, int start_bit, int size) {
  Signal s = *sig;
  updateSigSizeParamsFromRange(s, start_bit, size);
  saveSignal(sig, s);
}

void SignalModel::removeSignal(const Signal *sig) {
  UndoStack::push(new RemoveSigCommand(msg_id, sig));
}

void SignalModel::handleMsgChanged(uint32_t address) {
  if (address == DBCManager::parseId(msg_id).second) {
    refresh();
  }
}

void SignalModel::handleSignalAdded(uint32_t address, const Signal *sig) {
  if (address == DBCManager::parseId(msg_id).second) {
    int i = 0;
    for (; i < root->children.size(); ++i) {
      if (sig->start_bit < root->children[i]->sig->start_bit) break;
    }
    beginInsertRows({}, i, i);
    insertItem(root.get(), i, sig);
    endInsertRows();
  }
}

void SignalModel::handleSignalUpdated(const Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    emit dataChanged(index(row, 0), index(row, 1), {Qt::DisplayRole, Qt::EditRole, Qt::CheckStateRole});
  }
}

void SignalModel::handleSignalRemoved(const Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    beginRemoveRows({}, row, row);
    delete root->children.takeAt(row);
    endRemoveRows();
  }
}

// SignalItemDelegate

SignalItemDelegate::SignalItemDelegate(QObject *parent) {
  name_validator = new NameValidator(this);
  double_validator = new QDoubleValidator(this);
  double_validator->setLocale(QLocale::C);  // Match locale of QString::toDouble() instead of system
}

void SignalItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto item = (SignalModel::Item *)index.internalPointer();
  if (item && !index.parent().isValid() && index.column() == 0) {
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing);
    if (option.state & QStyle::State_Selected) {
      painter->fillRect(option.rect, option.palette.highlight());
    }

    // color label
    auto bg_color = QColor(getColor(item->row()));
    QRect rc{option.rect.left() + 3, option.rect.top(), 22, option.rect.height()};
    painter->setPen(Qt::NoPen);
    painter->setBrush(item->highlight ? bg_color.darker(125) : bg_color);
    painter->drawRoundedRect(rc.adjusted(0, 2, 0, -2), 5, 5);
    painter->setPen(item->highlight ? Qt::white : Qt::black);
    painter->drawText(rc, Qt::AlignCenter, QString::number(item->row() + 1));

    // signal name
    QFont font;
    font.setBold(true);
    painter->setFont(font);
    painter->setPen((option.state & QStyle::State_Selected ? option.palette.highlightedText() : option.palette.text()).color());
    painter->drawText(option.rect.adjusted(rc.width() + 9, 0, 0, 0), option.displayAlignment, index.data(Qt::DisplayRole).toString());
    painter->restore();
  } else {
    QStyledItemDelegate::paint(painter, option, index);
  }
}

QWidget *SignalItemDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto item = (SignalModel::Item *)index.internalPointer();
  if (item->type == SignalModel::Item::Name || item->type == SignalModel::Item::Offset || item->type == SignalModel::Item::Factor) {
    QLineEdit *e = new QLineEdit(parent);
    e->setFrame(false);
    e->setValidator(index.row() == 0 ? name_validator : double_validator);
    return e;
  } else if (item->type == SignalModel::Item::Size) {
    QSpinBox *spin = new QSpinBox(parent);
    spin->setFrame(false);
    spin->setRange(1, 64);
    return spin;
  }
  return QStyledItemDelegate::createEditor(parent, option, index);
}

// SignalView

SignalView::SignalView(ChartsWidget *charts, QWidget *parent) : charts(charts), QWidget(parent) {
  // title bar
  QWidget *title_bar = new QWidget(this);
  title_bar->setAutoFillBackground(true);
  QHBoxLayout *hl = new QHBoxLayout(title_bar);
  hl->addWidget(signal_count_lb = new QLabel());
  filter_edit = new QLineEdit(this);
  filter_edit->setClearButtonEnabled(true);
  filter_edit->setPlaceholderText(tr("filter signals by name"));
  hl->addWidget(filter_edit);
  hl->addStretch(1);
  auto collapse_btn = new QToolButton();
  collapse_btn->setIcon(utils::icon("dash-square"));
  collapse_btn->setIconSize({12, 12});
  collapse_btn->setAutoRaise(true);
  collapse_btn->setToolTip(tr("Collapse All"));
  hl->addWidget(collapse_btn);

  // tree view
  tree = new QTreeView(this);
  tree->setModel(model = new SignalModel(this));
  tree->setItemDelegate(new SignalItemDelegate(this));
  tree->setFrameShape(QFrame::NoFrame);
  tree->setHeaderHidden(true);
  tree->setMouseTracking(true);
  tree->setExpandsOnDoubleClick(false);
  tree->header()->setSectionResizeMode(QHeaderView::Stretch);
  tree->setMinimumHeight(300);
  tree->setStyleSheet("QSpinBox{background-color:white;border:none;} QLineEdit{background-color:white;}");

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);
  main_layout->addWidget(title_bar);
  main_layout->addWidget(tree);

  QObject::connect(filter_edit, &QLineEdit::textEdited, model, &SignalModel::setFilter);
  QObject::connect(collapse_btn, &QPushButton::clicked, tree, &QTreeView::collapseAll);
  QObject::connect(tree, &QAbstractItemView::clicked, this, &SignalView::rowClicked);
  QObject::connect(tree, &QTreeView::viewportEntered, [this]() { emit highlight(nullptr); });
  QObject::connect(tree, &QTreeView::entered, [this](const QModelIndex &index) { emit highlight(model->getItem(index)->sig); });
  QObject::connect(model, &QAbstractItemModel::modelReset, this, &SignalView::rowsChanged);
  QObject::connect(model, &QAbstractItemModel::rowsInserted, this, &SignalView::rowsChanged);
  QObject::connect(model, &QAbstractItemModel::rowsRemoved, this, &SignalView::rowsChanged);
  QObject::connect(dbc(), &DBCManager::signalAdded, [this](uint32_t address, const Signal *sig) { expandSignal(sig); });
}

void SignalView::setMessage(const QString &id) {
  msg_id = id;
  filter_edit->clear();
  model->setMessage(id);
}

void SignalView::rowsChanged() {
  auto create_btn = [](const QString &id, const QString &tooltip) {
    auto btn = new QToolButton();
    btn->setIcon(utils::icon(id));
    btn->setToolTip(tooltip);
    btn->setAutoRaise(true);
    return btn;
  };

  signal_count_lb->setText(tr("Signals: %1").arg(model->rowCount()));

  for (int i = 0; i < model->rowCount(); ++i) {
    auto index = model->index(i, 1);
    if (!tree->indexWidget(index)) {
      QWidget *w = new QWidget(this);
      QHBoxLayout *h = new QHBoxLayout(w);
      h->setContentsMargins(0, 2, 0, 2);
      h->addStretch(1);

      auto remove_btn = create_btn("x", tr("Remove signal"));
      auto plot_btn = create_btn("graph-up", "");
      plot_btn->setCheckable(true);
      h->addWidget(plot_btn);
      h->addWidget(remove_btn);

      tree->setIndexWidget(index, w);
      auto sig = model->getItem(index)->sig;
      QObject::connect(remove_btn, &QToolButton::clicked, [=]() { model->removeSignal(sig); });
      QObject::connect(plot_btn, &QToolButton::clicked, [=](bool checked) {
        emit showChart(msg_id, sig, checked, QGuiApplication::keyboardModifiers() & Qt::ShiftModifier);
      });
    }
  }
  updateChartState();
}

void SignalView::rowClicked(const QModelIndex &index) {
  auto item = model->getItem(index);
  if (item->type == SignalModel::Item::Sig) {
    auto sig_index = model->index(index.row(), 0, index.parent());
    tree->setExpanded(sig_index, !tree->isExpanded(sig_index));
  } else if (item->type == SignalModel::Item::ExtraInfo) {
    model->showExtraInfo(index);
  }
}

void SignalView::expandSignal(const Signal *sig) {
  if (int row = model->signalRow(sig); row != -1) {
    auto idx = model->index(row, 0);
    bool expand = !tree->isExpanded(idx);
    tree->setExpanded(idx, expand);
    tree->scrollTo(idx, QAbstractItemView::PositionAtTop);
    if (expand) tree->setCurrentIndex(idx);
  }
}

void SignalView::updateChartState() {
  int i = 0;
  for (auto item : model->root->children) {
    auto plot_btn = tree->indexWidget(model->index(i, 1))->findChildren<QToolButton*>()[0];
    bool chart_opened = charts->hasSignal(msg_id, item->sig);
    plot_btn->setChecked(chart_opened);
    plot_btn->setToolTip(chart_opened ? tr("Close Plot") : tr("Show Plot\nSHIFT click to add to previous opened plot"));
    ++i;
  }
}

void SignalView::signalHovered(const Signal *sig) {
  auto &children = model->root->children;
  for (int i = 0; i < children.size(); ++i) {
    bool highlight = children[i]->sig == sig;
    if (std::exchange(children[i]->highlight, highlight) != highlight) {
      emit model->dataChanged(model->index(i, 0), model->index(i, 0));
    }
  }
}
