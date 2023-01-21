#include "tools/cabana/signaledit.h"

#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>

#include "tools/cabana/commands.h"

#include "selfdrive/ui/qt/util.h"

// SignalModel

SignalModel::SignalModel(QObject *parent) : root(new Item), QAbstractItemModel(parent) {
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &SignalModel::refresh);
  QObject::connect(dbc(), &DBCManager::msgUpdated, this, &SignalModel::handleMsgChanged);
  QObject::connect(dbc(), &DBCManager::msgRemoved, this, &SignalModel::handleMsgChanged);
  QObject::connect(dbc(), &DBCManager::signalAdded, this, &SignalModel::handleSignalAdded);
  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &SignalModel::handleSignalRemoved);
}

void SignalModel::insertItem(SignalModel::Item *parent_item, int pos, const Signal *sig) {
  Item *item = new Item{.sig = sig, .parent = parent_item, .title = sig->name.c_str()};
  parent_item->children.insert(pos, item);
  QString titles[]{"Name", "Size", "Little Endian", "Signed", "Offset", "Factor"};
  for (int i = 0; i < std::size(titles); ++i) {
    item->children.push_back(new Item{.sig = sig, .parent = item, .title = titles[i], .data_id = i});
  }

  // Extra info
  Item *extra_item = new Item({.sig = sig, .parent = item, .title = "Extra Info"});
  item->children.push_back(extra_item);
  QString extra_titles[]{"Unit", "Comment", "Minimum", "Maxmum", "Description"};
  for (int i = 0; i < std::size(extra_titles); ++i) {
    extra_item->children.push_back(new Item{.sig = sig, .parent = extra_item, .title = extra_titles[i], .data_id = i + (int)std::size(titles)});
  }
}

void SignalModel::setMessage(const QString &id) {
  msg_id = id;
  filter_str = "";
  refresh();
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

int SignalModel::rowCount(const QModelIndex &parent) const {
  return (parent.column() > 0) ? 0 : getItem(parent)->children.size();
}

Qt::ItemFlags SignalModel::flags(const QModelIndex &index) const {
  if (!index.isValid()) return Qt::NoItemFlags;
  Qt::ItemFlags flags = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
  if (auto item = getItem(index); item->data_id != -1 && index.column() == 1) {
    flags |= (item->data_id == 2 || item->data_id == 3) ? Qt::ItemIsUserCheckable : Qt::ItemIsEditable;
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
        return item->parent == root.get() ? QString::fromStdString(item->sig->name) : item->title;
      } else if (item->data_id != -1) {
        switch (item->data_id) {
          case 0: return QString::fromStdString(item->sig->name);
          case 1: return item->sig->size;
          case 4: return QString::number(item->sig->offset, 'f', 6);
          case 5: return QString::number(item->sig->factor, 'f', 6);
        }
      }
    } else if (role == Qt::CheckStateRole && index.column() == 1) {
      if (item->data_id == 2) return item->sig->is_little_endian ? Qt::Checked : Qt::Unchecked;
      if (item->data_id == 3) return item->sig->is_signed ? Qt::Checked : Qt::Unchecked;
    }
  }
  return {};
}

bool SignalModel::setData(const QModelIndex &index, const QVariant &value, int role) {
  if (role != Qt::EditRole && role != Qt::CheckStateRole) return false;

  Item *item = getItem(index);
  Signal s = *item->sig;
  switch (item->data_id) {
    case 0: s.name = value.toString().toStdString(); break;
    case 1: s.size = value.toInt(); break;
    case 2: s.is_little_endian = value.toBool(); break;
    case 3: s.is_signed = value.toBool(); break;
    case 4: s.offset = value.toDouble(); break;
    case 5: s.factor = value.toDouble(); break;
  }
  bool ret = saveSignal(item->sig, s);
  emit dataChanged(index, index, {Qt::DisplayRole, Qt::EditRole, Qt::CheckStateRole});
  return ret;
}

bool SignalModel::saveSignal(const Signal *origin_s, Signal &s) {
  if (s.name.empty()) {
    return false;
  }
  auto msg = dbc()->msg(msg_id);
  if (s.name != origin_s->name) {
    if (msg->sigs.count(QString::fromStdString(s.name)) != 0) {
      QMessageBox::warning(nullptr, tr("Failed to save signal"),
                           tr("There is already a signal with the same name '%1'").arg(s.name.c_str()));
      return false;
    }
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

  auto [start, end] = getSignalRange(&s);
  if (start < 0 || end >= msg->size * 8) {
    QString warning_str = tr("Signal size [%1] exceed msg boundary").arg(s.size);
    QMessageBox::warning(nullptr, tr("Failed to save signal"), warning_str);
    return false;
  }

  Commands::push(new EditSignalCommand(msg_id, origin_s, s));
  return true;
}

void SignalModel::addSignal(int start_bit, int size, bool little_endian) {
  auto msg = dbc()->msg(msg_id);
  for (int i = 1; !msg; ++i) {
    QString name = QString("NEW_MSG_%1").arg(i);
    if (std::none_of(dbc()->messages().begin(), dbc()->messages().end(), [&](auto &m) { return m.second.name == name; })) {
      Commands::push(new EditMsgCommand(msg_id, name, can->lastMessage(msg_id).dat.size()));
      msg = dbc()->msg(msg_id);
    }
  }

  Signal sig = {.is_little_endian = little_endian, .factor = 1};
  for (int i = 1; /**/; ++i) {
    sig.name = "NEW_SIGNAL_" + std::to_string(i);
    if (msg->sigs.count(sig.name.c_str()) == 0) break;
  }
  updateSigSizeParamsFromRange(sig, start_bit, size);
  Commands::push(new AddSigCommand(msg_id, sig));
}

void SignalModel::resizeSignal(const Signal *sig, int start_bit, int size) {
  Signal s = *sig;
  updateSigSizeParamsFromRange(s, start_bit, size);
  saveSignal(sig, s);
}

void SignalModel::removeSignal(const Signal *sig) {
  Commands::push(new RemoveSigCommand(msg_id, sig));
}

void SignalModel::handleMsgChanged(uint32_t address) {
  if (address == DBCManager::parseId(msg_id).second) {
    refresh();
  }
}

void SignalModel::handleSignalRemoved(const Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    beginRemoveRows({}, row, row);
    delete root->children.takeAt(row);
    endRemoveRows();
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

// SignalItemDelegate

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

// SignalView

SignalView::SignalView(ChartsWidget *charts, QWidget *parent) : charts(charts), QWidget(parent) {
  // title bar
  QHBoxLayout *hl = new QHBoxLayout();
  hl->addWidget(signal_count_lb = new QLabel());
  filter_edit = new QLineEdit(this);
  filter_edit->setClearButtonEnabled(true);
  filter_edit->setPlaceholderText(tr("filter signals by name"));
  hl->addWidget(filter_edit);
  hl->addStretch(1);
  auto collapse_btn = new QToolButton();
  collapse_btn->setIcon(bootstrapPixmap("dash-square"));
  collapse_btn->setIconSize({12, 12});
  collapse_btn->setAutoRaise(true);
  collapse_btn->setToolTip(tr("Collapse All"));
  hl->addWidget(collapse_btn);

  // tree view
  tree = new QTreeView(this);
  tree->setModel(model = new SignalModel(this));
  tree->setItemDelegate(new SignalItemDelegate(this));
  tree->setHeaderHidden(true);
  tree->setMouseTracking(true);
  tree->header()->setSectionResizeMode(QHeaderView::Stretch);
  tree->setStyleSheet("QSpinBox{background-color:white;border:none;} QLineEdit{background-color:white;}");

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addWidget(tip_lb = new QLabel(tr("Drag-Select in binary view to create new signal.")));
  main_layout->addLayout(hl);
  main_layout->addWidget(tree);

  QObject::connect(filter_edit, &QLineEdit::textEdited, model, &SignalModel::setFilter);
  QObject::connect(collapse_btn, &QPushButton::clicked, tree, &QTreeView::collapseAll);
  QObject::connect(tree, &QTreeView::viewportEntered, [this]() { emit highlight(nullptr); });
  QObject::connect(tree, &QTreeView::entered, [this](const QModelIndex &index) { emit highlight(model->getItem(index)->sig); });
  QObject::connect(model, &QAbstractItemModel::modelReset, this, &SignalView::rowsChanged);
  QObject::connect(model, &QAbstractItemModel::rowsInserted, this, &SignalView::rowsChanged);
  QObject::connect(model, &QAbstractItemModel::rowsRemoved, this, &SignalView::rowsChanged);
}

void SignalView::setMessage(const QString &id) {
  msg_id = id;
  filter_edit->clear();
  model->setMessage(id);
}

void SignalView::rowsChanged() {
  auto create_btn = [](const QString &id, const QString &tooltip) {
    auto btn = new QToolButton();
    btn->setIcon(bootstrapPixmap(id));
    btn->setToolTip(tooltip);
    btn->setAutoRaise(true);
    return btn;
  };

  signal_count_lb->setText(tr("Signals: %1").arg(model->rowCount()));
  tip_lb->setVisible(model->rowCount() == 0);

  for (int i = 0; i < model->rowCount(); ++i) {
    auto index = model->index(i, 1);
    if (!tree->indexWidget(index)) {
      QWidget *w = new QWidget(this);
      QHBoxLayout *h = new QHBoxLayout(w);
      h->setContentsMargins(0, 2, 0, 2);
      h->setSpacing(3);
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
