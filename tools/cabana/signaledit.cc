#include "tools/cabana/signaledit.h"

#include <QApplication>
#include <QCompleter>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QPushButton>
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

void SignalModel::insertItem(SignalModel::Item *parent_item, int pos, const cabana::Signal *sig) {
  Item *item = new Item{.sig = sig, .parent = parent_item, .title = sig->name, .type = Item::Sig};
  parent_item->children.insert(pos, item);
  QString titles[]{"Name", "Size", "Little Endian", "Signed", "Offset", "Factor", "Extra Info", "Unit", "Comment", "Minimum Value", "Maximum Value", "Value Descriptions"};
  for (int i = 0; i < std::size(titles); ++i) {
    item->children.push_back(new Item{.sig = sig, .parent = item, .title = titles[i], .type = (Item::Type)(i + Item::Name)});
  }
}

void SignalModel::setMessage(const MessageId &id) {
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
    for (auto s : msg->getSignals()) {
      if (filter_str.isEmpty() || s->name.contains(filter_str, Qt::CaseInsensitive)) {
        insertItem(root.get(), root->children.size(), s);
      }
    }
  }
  endResetModel();
}

void SignalModel::updateState(const QHash<MessageId, CanData> *msgs) {
  if (!msgs || msgs->contains(msg_id)) {
    auto &dat = can->lastMessage(msg_id).dat;
    int row = 0;
    for (auto item : root->children) {
      QString value = QString::number(get_raw_value((uint8_t *)dat.begin(), dat.size(), *item->sig), 'f', item->sig->precision);
      if (!item->sig->unit.isEmpty()){
        value += " " + item->sig->unit;
      }
      if (value != item->sig_val) {
        item->sig_val = value;
        emit dataChanged(index(row, 1), index(row, 1), {Qt::DisplayRole});
      }
      ++row;
    }
  }
}

SignalModel::Item *SignalModel::getItem(const QModelIndex &index) const {
  SignalModel::Item *item = nullptr;
  if (index.isValid()) {
    item = (SignalModel::Item *)index.internalPointer();
  }
  return item ? item : root.get();
}

int SignalModel::rowCount(const QModelIndex &parent) const {
  if (parent.isValid() && parent.column() > 0) return 0;

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

int SignalModel::signalRow(const cabana::Signal *sig) const {
  for (int i = 0; i < root->children.size(); ++i) {
    if (root->children[i]->sig == sig) return i;
  }
  return -1;
}

QModelIndex SignalModel::index(int row, int column, const QModelIndex &parent) const {
  if (parent.isValid() && parent.column() != 0) return {};

  auto parent_item = getItem(parent);
  if (parent_item && row < parent_item->children.size()) {
    return createIndex(row, column, parent_item->children[row]);
  }
  return {};
}

QModelIndex SignalModel::parent(const QModelIndex &index) const {
  if (!index.isValid()) return {};
  Item *parent_item = getItem(index)->parent;
  return !parent_item || parent_item == root.get() ? QModelIndex() : createIndex(parent_item->row(), 0, parent_item);
}

QVariant SignalModel::data(const QModelIndex &index, int role) const {
  if (index.isValid()) {
    const Item *item = getItem(index);
    if (role == Qt::DisplayRole || role == Qt::EditRole) {
      if (index.column() == 0) {
        return item->type == Item::Sig ? item->sig->name : item->title;
      } else {
        switch (item->type) {
          case Item::Sig: return item->sig_val;
          case Item::Name: return item->sig->name;
          case Item::Size: return item->sig->size;
          case Item::Offset: return QString::number(item->sig->offset, 'f', 6);
          case Item::Factor: return QString::number(item->sig->factor, 'f', 6);
          case Item::Unit: return item->sig->unit;
          case Item::Comment: return item->sig->comment;
          case Item::Min: return item->sig->min;
          case Item::Max: return item->sig->max;
          case Item::Desc: {
            QStringList val_desc;
            for (auto &[val, desc] : item->sig->val_desc) {
              val_desc << QString("%1 \"%2\"").arg(val, desc);
            }
            return val_desc.join(" ");
          }
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
  cabana::Signal s = *item->sig;
  switch (item->type) {
    case Item::Name: s.name = value.toString(); break;
    case Item::Size: s.size = value.toInt(); break;
    case Item::Endian: s.is_little_endian = value.toBool(); break;
    case Item::Signed: s.is_signed = value.toBool(); break;
    case Item::Offset: s.offset = value.toDouble(); break;
    case Item::Factor: s.factor = value.toDouble(); break;
    case Item::Unit: s.unit = value.toString(); break;
    case Item::Comment: s.comment = value.toString(); break;
    case Item::Min: s.min = value.toString(); break;
    case Item::Max: s.max = value.toString(); break;
    case Item::Desc: s.val_desc = value.value<ValueDescription>(); break;
    default: return false;
  }
  s.updatePrecision();
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

bool SignalModel::saveSignal(const cabana::Signal *origin_s, cabana::Signal &s) {
  auto msg = dbc()->msg(msg_id);
  if (s.name != origin_s->name && msg->sig(s.name) != nullptr) {
    QString text = tr("There is already a signal with the same name '%1'").arg(s.name);
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
  for (int i = 0; !msg; ++i) {
    QString name = QString("NEW_MSG_") + QString::number(msg_id.address, 16).toUpper();
    if (!dbc()->msg(msg_id.source, name)) {
      UndoStack::push(new EditMsgCommand(msg_id, name, can->lastMessage(msg_id).dat.size()));
      msg = dbc()->msg(msg_id);
    }
  }

  cabana::Signal sig = {.is_little_endian = little_endian, .factor = 1, .min = "0", .max = QString::number(std::pow(2, size) - 1)};
  for (int i = 1; /**/; ++i) {
    sig.name = QString("NEW_SIGNAL_%1").arg(i);
    if (msg->sig(sig.name) == nullptr) break;
  }
  updateSigSizeParamsFromRange(sig, start_bit, size);
  UndoStack::push(new AddSigCommand(msg_id, sig));
}

void SignalModel::resizeSignal(const cabana::Signal *sig, int start_bit, int size) {
  cabana::Signal s = *sig;
  updateSigSizeParamsFromRange(s, start_bit, size);
  saveSignal(sig, s);
}

void SignalModel::removeSignal(const cabana::Signal *sig) {
  UndoStack::push(new RemoveSigCommand(msg_id, sig));
}

void SignalModel::handleMsgChanged(MessageId id) {
  if (id == msg_id) {
    refresh();
  }
}

void SignalModel::handleSignalAdded(MessageId id, const cabana::Signal *sig) {
  if (id == msg_id) {
    int i = 0;
    for (; i < root->children.size(); ++i) {
      if (sig->start_bit < root->children[i]->sig->start_bit) break;
    }
    beginInsertRows({}, i, i);
    insertItem(root.get(), i, sig);
    endInsertRows();
  }
}

void SignalModel::handleSignalUpdated(const cabana::Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    emit dataChanged(index(row, 0), index(row, 1), {Qt::DisplayRole, Qt::EditRole, Qt::CheckStateRole});
  }
}

void SignalModel::handleSignalRemoved(const cabana::Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    beginRemoveRows({}, row, row);
    delete root->children.takeAt(row);
    endRemoveRows();
  }
}

// SignalItemDelegate

SignalItemDelegate::SignalItemDelegate(QObject *parent) : QStyledItemDelegate(parent) {
  name_validator = new NameValidator(this);
  double_validator = new QDoubleValidator(this);
  double_validator->setLocale(QLocale::C);  // Match locale of QString::toDouble() instead of system
  small_font.setPointSize(8);
}

QSize SignalItemDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const {
  int width = option.widget->size().width() / 2;
  if (index.column() == 0) {
    auto text = index.data(Qt::DisplayRole).toString();
    auto it = width_cache.find(text);
    if (it == width_cache.end()) {
      int spacing = option.widget->style()->pixelMetric(QStyle::PM_TreeViewIndentation) + color_label_width + 8;
      it = width_cache.insert(text, option.fontMetrics.width(text) + spacing);
    }
    width = std::min(width, it.value());
  }
  return {width, QApplication::fontMetrics().height()};
}

void SignalItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto item = (SignalModel::Item *)index.internalPointer();
  if (index.column() == 0 && item && item->type == SignalModel::Item::Sig) {
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing);
    if (option.state & QStyle::State_Selected) {
      painter->fillRect(option.rect, option.palette.highlight());
    }

    // color label
    auto bg_color = getColor(item->sig);
    int h_margin = option.widget->style()->pixelMetric(QStyle::PM_FocusFrameHMargin) + 1;
    int v_margin = option.widget->style()->pixelMetric(QStyle::PM_FocusFrameVMargin);
    QRect rc{option.rect.left() + h_margin, option.rect.top(), color_label_width, option.rect.height()};
    painter->setPen(Qt::NoPen);
    painter->setBrush(item->highlight ? bg_color.darker(125) : bg_color);
    painter->drawRoundedRect(rc.adjusted(0, v_margin, 0, -v_margin), 3, 3);
    painter->setPen(item->highlight ? Qt::white : Qt::black);
    painter->setFont(small_font);
    painter->drawText(rc, Qt::AlignCenter, QString::number(item->row() + 1));

    // signal name
    painter->setFont(option.font);
    painter->setPen(option.palette.color(option.state & QStyle::State_Selected ? QPalette::HighlightedText : QPalette::Text));
    QString text = index.data(Qt::DisplayRole).toString();
    QRect text_rect = option.rect;
    text_rect.setLeft(rc.right() + h_margin * 2);
    text = painter->fontMetrics().elidedText(text, Qt::ElideRight, text_rect.width());
    painter->drawText(text_rect, option.displayAlignment, text);
    painter->restore();
  } else if (index.column() == 1 && item && item->type == SignalModel::Item::Sig) {
    // draw signal value
    if (option.state & QStyle::State_Selected) {
      painter->fillRect(option.rect, option.palette.highlight());
    }
    painter->setPen(option.palette.color(option.state & QStyle::State_Selected ? QPalette::HighlightedText : QPalette::Text));
    QRect rc = option.rect.adjusted(0, 0, -70, 0);
    auto text = painter->fontMetrics().elidedText(index.data(Qt::DisplayRole).toString(), Qt::ElideRight, rc.width());
    painter->drawText(rc, Qt::AlignRight | Qt::AlignVCenter, text);
  } else {
    QStyledItemDelegate::paint(painter, option, index);
  }
}

QWidget *SignalItemDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto item = (SignalModel::Item *)index.internalPointer();
  if (item->type == SignalModel::Item::Name || item->type == SignalModel::Item::Offset ||
      item->type == SignalModel::Item::Factor || item->type == SignalModel::Item::Min || item->type == SignalModel::Item::Max) {
    QLineEdit *e = new QLineEdit(parent);
    e->setFrame(false);
    e->setValidator(index.row() == 0 ? name_validator : double_validator);

    if (item->type == SignalModel::Item::Name) {
      QCompleter *completer = new QCompleter(dbc()->signalNames());
      completer->setCaseSensitivity(Qt::CaseInsensitive);
      completer->setFilterMode(Qt::MatchContains);
      e->setCompleter(completer);
    }

    return e;
  } else if (item->type == SignalModel::Item::Size) {
    QSpinBox *spin = new QSpinBox(parent);
    spin->setFrame(false);
    spin->setRange(1, 64);
    return spin;
  } else if (item->type == SignalModel::Item::Desc) {
    ValueDescriptionDlg dlg(item->sig->val_desc, parent);
    dlg.setWindowTitle(item->sig->name);
    if (dlg.exec()) {
      ((QAbstractItemModel *)index.model())->setData(index, QVariant::fromValue(dlg.val_desc));
    }
    return nullptr;
  }
  return QStyledItemDelegate::createEditor(parent, option, index);
}

// SignalView

SignalView::SignalView(ChartsWidget *charts, QWidget *parent) : charts(charts), QFrame(parent) {
  setFrameStyle(QFrame::StyledPanel | QFrame::Plain);
  // title bar
  QWidget *title_bar = new QWidget(this);
  QHBoxLayout *hl = new QHBoxLayout(title_bar);
  hl->addWidget(signal_count_lb = new QLabel());
  filter_edit = new QLineEdit(this);
  QRegularExpression re("\\S+");
  filter_edit->setValidator(new QRegularExpressionValidator(re, this));
  filter_edit->setClearButtonEnabled(true);
  filter_edit->setPlaceholderText(tr("filter signals"));
  hl->addWidget(filter_edit);
  hl->addStretch(1);
  auto collapse_btn = toolButton("dash-square", tr("Collapse All"));
  collapse_btn->setIconSize({12, 12});
  hl->addWidget(collapse_btn);

  // tree view
  tree = new QTreeView(this);
  tree->setModel(model = new SignalModel(this));
  tree->setItemDelegate(new SignalItemDelegate(this));
  tree->setFrameShape(QFrame::NoFrame);
  tree->setHeaderHidden(true);
  tree->setMouseTracking(true);
  tree->setExpandsOnDoubleClick(false);
  tree->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
  tree->header()->setStretchLastSection(true);
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
  QObject::connect(dbc(), &DBCManager::signalAdded, [this](MessageId id, const cabana::Signal *sig) { selectSignal(sig); });

  setWhatsThis(tr(R"(
    <b>Signal view</b><br />
    <!-- TODO: add descprition here -->
  )"));
}

void SignalView::setMessage(const MessageId &id) {
  msg_id = id;
  filter_edit->clear();
  model->setMessage(id);
}

void SignalView::rowsChanged() {
  for (int i = 0; i < model->rowCount(); ++i) {
    auto index = model->index(i, 1);
    if (!tree->indexWidget(index)) {
      QWidget *w = new QWidget(this);
      QHBoxLayout *h = new QHBoxLayout(w);
      int v_margin = style()->pixelMetric(QStyle::PM_FocusFrameVMargin);
      int h_margin = style()->pixelMetric(QStyle::PM_FocusFrameHMargin);
      h->setContentsMargins(0, v_margin, -h_margin, v_margin);
      h->setSpacing(style()->pixelMetric(QStyle::PM_ToolBarItemSpacing));
      h->addStretch(0);

      auto remove_btn = toolButton("x", tr("Remove signal"));
      auto plot_btn = toolButton("graph-up", "");
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
  signal_count_lb->setText(tr("Signals: %1").arg(model->rowCount()));
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

void SignalView::selectSignal(const cabana::Signal *sig, bool expand) {
  if (int row = model->signalRow(sig); row != -1) {
    auto idx = model->index(row, 0);
    if (expand) {
      tree->setExpanded(idx, !tree->isExpanded(idx));
    }
    tree->scrollTo(idx, QAbstractItemView::PositionAtTop);
    tree->setCurrentIndex(idx);
  }
}

void SignalView::updateChartState() {
  int i = 0;
  for (auto item : model->root->children) {
    bool chart_opened = charts->hasSignal(msg_id, item->sig);
    auto buttons = tree->indexWidget(model->index(i, 1))->findChildren<QToolButton *>();
    if (buttons.size() > 0) {
      buttons[0]->setChecked(chart_opened);
      buttons[0]->setToolTip(chart_opened ? tr("Close Plot") : tr("Show Plot\nSHIFT click to add to previous opened plot"));
    }
    ++i;
  }
}

void SignalView::signalHovered(const cabana::Signal *sig) {
  auto &children = model->root->children;
  for (int i = 0; i < children.size(); ++i) {
    bool highlight = children[i]->sig == sig;
    if (std::exchange(children[i]->highlight, highlight) != highlight) {
      emit model->dataChanged(model->index(i, 0), model->index(i, 0), {Qt::DecorationRole});
    }
  }
}

void SignalView::leaveEvent(QEvent *event) {
  emit highlight(nullptr);
  QWidget::leaveEvent(event);
}

// ValueDescriptionDlg

ValueDescriptionDlg::ValueDescriptionDlg(const ValueDescription &descriptions, QWidget *parent) : QDialog(parent) {
  QHBoxLayout *toolbar_layout = new QHBoxLayout();
  QPushButton *add = new QPushButton(utils::icon("plus"), "");
  QPushButton *remove = new QPushButton(utils::icon("dash"), "");
  remove->setEnabled(false);
  toolbar_layout->addWidget(add);
  toolbar_layout->addWidget(remove);
  toolbar_layout->addStretch(0);

  table = new QTableWidget(descriptions.size(), 2, this);
  table->setItemDelegate(new Delegate(this));
  table->setHorizontalHeaderLabels({"Value", "Description"});
  table->horizontalHeader()->setStretchLastSection(true);
  table->setSelectionBehavior(QAbstractItemView::SelectRows);
  table->setSelectionMode(QAbstractItemView::SingleSelection);
  table->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::EditKeyPressed);
  table->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  int row = 0;
  for (auto &[val, desc] : descriptions) {
    table->setItem(row, 0, new QTableWidgetItem(val));
    table->setItem(row, 1, new QTableWidgetItem(desc));
    ++row;
  }

  auto btn_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(toolbar_layout);
  main_layout->addWidget(table);
  main_layout->addWidget(btn_box);
  setMinimumWidth(500);

  QObject::connect(btn_box, &QDialogButtonBox::accepted, this, &ValueDescriptionDlg::save);
  QObject::connect(btn_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(add, &QPushButton::clicked, [this]() {
    table->setRowCount(table->rowCount() + 1);
    table->setItem(table->rowCount() - 1, 0, new QTableWidgetItem);
    table->setItem(table->rowCount() - 1, 1, new QTableWidgetItem);
  });
  QObject::connect(remove, &QPushButton::clicked, [this]() { table->removeRow(table->currentRow()); });
  QObject::connect(table, &QTableWidget::itemSelectionChanged, [=]() {
    remove->setEnabled(table->currentRow() != -1);
  });
}

void ValueDescriptionDlg::save() {
  for (int i = 0; i < table->rowCount(); ++i) {
    QString val = table->item(i, 0)->text().trimmed();
    QString desc = table->item(i, 1)->text().trimmed();
    if (!val.isEmpty() && !desc.isEmpty()) {
      val_desc.push_back({val, desc});
    }
  }
  QDialog::accept();
}

QWidget *ValueDescriptionDlg::Delegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  QLineEdit *edit = new QLineEdit(parent);
  edit->setFrame(false);
  if (index.column() == 0) {
    edit->setValidator(new QIntValidator(edit));
  }
  return edit;
}
