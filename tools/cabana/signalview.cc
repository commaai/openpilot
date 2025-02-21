#include "tools/cabana/signalview.h"

#include <algorithm>

#include <QCompleter>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QScrollBar>
#include <QtConcurrent>
#include <QVBoxLayout>

#include "tools/cabana/commands.h"

// SignalModel

static QString signalTypeToString(cabana::Signal::Type type) {
  if (type == cabana::Signal::Type::Multiplexor) return "Multiplexor Signal";
  else if (type == cabana::Signal::Type::Multiplexed) return "Multiplexed Signal";
  else return "Normal Signal";
}

SignalModel::SignalModel(QObject *parent) : root(new Item), QAbstractItemModel(parent) {
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &SignalModel::refresh);
  QObject::connect(dbc(), &DBCManager::msgUpdated, this, &SignalModel::handleMsgChanged);
  QObject::connect(dbc(), &DBCManager::msgRemoved, this, &SignalModel::handleMsgChanged);
  QObject::connect(dbc(), &DBCManager::signalAdded, this, &SignalModel::handleSignalAdded);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &SignalModel::handleSignalUpdated);
  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &SignalModel::handleSignalRemoved);
}

void SignalModel::insertItem(SignalModel::Item *root_item, int pos, const cabana::Signal *sig) {
  Item *parent_item = new Item{.sig = sig, .parent = root_item, .title = sig->name, .type = Item::Sig};
  root_item->children.insert(pos, parent_item);
  QString titles[]{"Name", "Size", "Receiver Nodes", "Little Endian", "Signed", "Offset", "Factor", "Type",
                   "Multiplex Value", "Extra Info", "Unit", "Comment", "Minimum Value", "Maximum Value", "Value Table"};
  for (int i = 0; i < std::size(titles); ++i) {
    auto item = new Item{.sig = sig, .parent = parent_item, .title = titles[i], .type = (Item::Type)(i + Item::Name)};
    parent_item->children.push_back(item);
    if (item->type == Item::ExtraInfo) {
      parent_item = item;
    }
  }
}

void SignalModel::setMessage(const MessageId &id) {
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
    for (auto s : msg->getSignals()) {
      if (filter_str.isEmpty() || s->name.contains(filter_str, Qt::CaseInsensitive)) {
        insertItem(root.get(), root->children.size(), s);
      }
    }
  }
  endResetModel();
}

SignalModel::Item *SignalModel::getItem(const QModelIndex &index) const {
  auto item = index.isValid() ? (SignalModel::Item *)index.internalPointer() : nullptr;
  return item ? item : root.get();
}

int SignalModel::rowCount(const QModelIndex &parent) const {
  if (parent.isValid() && parent.column() > 0) return 0;

  return getItem(parent)->children.size();
}

Qt::ItemFlags SignalModel::flags(const QModelIndex &index) const {
  if (!index.isValid()) return Qt::NoItemFlags;

  auto item = getItem(index);
  Qt::ItemFlags flags = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
  if (index.column() == 1  && item->children.empty()) {
    flags |= (item->type == Item::Endian || item->type == Item::Signed) ? Qt::ItemIsUserCheckable : Qt::ItemIsEditable;
  }
  if (item->type == Item::MultiplexValue && item->sig->type != cabana::Signal::Type::Multiplexed) {
    flags &= ~Qt::ItemIsEnabled;
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
          case Item::Node: return item->sig->receiver_name;
          case Item::SignalType: return signalTypeToString(item->sig->type);
          case Item::MultiplexValue: return item->sig->multiplex_value;
          case Item::Offset: return doubleToString(item->sig->offset);
          case Item::Factor: return doubleToString(item->sig->factor);
          case Item::Unit: return item->sig->unit;
          case Item::Comment: return item->sig->comment;
          case Item::Min: return doubleToString(item->sig->min);
          case Item::Max: return doubleToString(item->sig->max);
          case Item::Desc: {
            QStringList val_desc;
            for (auto &[val, desc] : item->sig->val_desc) {
              val_desc << QString("%1 \"%2\"").arg(val).arg(desc);
            }
            return val_desc.join(" ");
          }
          default: break;
        }
      }
    } else if (role == Qt::CheckStateRole && index.column() == 1) {
      if (item->type == Item::Endian) return item->sig->is_little_endian ? Qt::Checked : Qt::Unchecked;
      if (item->type == Item::Signed) return item->sig->is_signed ? Qt::Checked : Qt::Unchecked;
    } else if (role == Qt::ToolTipRole && item->type == Item::Sig) {
      return (index.column() == 0) ? signalToolTip(item->sig) : QString();
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
    case Item::Node: s.receiver_name = value.toString().trimmed(); break;
    case Item::SignalType: s.type = (cabana::Signal::Type)value.toInt(); break;
    case Item::MultiplexValue: s.multiplex_value = value.toInt(); break;
    case Item::Endian: s.is_little_endian = value.toBool(); break;
    case Item::Signed: s.is_signed = value.toBool(); break;
    case Item::Offset: s.offset = value.toDouble(); break;
    case Item::Factor: s.factor = value.toDouble(); break;
    case Item::Unit: s.unit = value.toString(); break;
    case Item::Comment: s.comment = value.toString(); break;
    case Item::Min: s.min = value.toDouble(); break;
    case Item::Max: s.max = value.toDouble(); break;
    case Item::Desc: s.val_desc = value.value<ValueDescription>(); break;
    default: return false;
  }
  bool ret = saveSignal(item->sig, s);
  emit dataChanged(index, index, {Qt::DisplayRole, Qt::EditRole, Qt::CheckStateRole});
  return ret;
}

bool SignalModel::saveSignal(const cabana::Signal *origin_s, cabana::Signal &s) {
  auto msg = dbc()->msg(msg_id);
  if (s.name != origin_s->name && msg->sig(s.name) != nullptr) {
    QString text = tr("There is already a signal with the same name '%1'").arg(s.name);
    QMessageBox::warning(nullptr, tr("Failed to save signal"), text);
    return false;
  }

  if (s.is_little_endian != origin_s->is_little_endian) {
    s.start_bit = flipBitPos(s.start_bit);
  }
  UndoStack::push(new EditSignalCommand(msg_id, origin_s, s));
  return true;
}

void SignalModel::handleMsgChanged(MessageId id) {
  if (id.address == msg_id.address) {
    refresh();
  }
}

void SignalModel::handleSignalAdded(MessageId id, const cabana::Signal *sig) {
  if (id == msg_id) {
    if (filter_str.isEmpty()) {
      int i = dbc()->msg(msg_id)->indexOf(sig);
      beginInsertRows({}, i, i);
      insertItem(root.get(), i, sig);
      endInsertRows();
    } else if (sig->name.contains(filter_str, Qt::CaseInsensitive)) {
      refresh();
    }
  }
}

void SignalModel::handleSignalUpdated(const cabana::Signal *sig) {
  if (int row = signalRow(sig); row != -1) {
    emit dataChanged(index(row, 0), index(row, 1), {Qt::DisplayRole, Qt::EditRole, Qt::CheckStateRole});

    if (filter_str.isEmpty()) {
      // move row when the order changes.
      int to = dbc()->msg(msg_id)->indexOf(sig);
      if (to != row) {
        beginMoveRows({}, row, row, {}, to > row ? to + 1 : to);
        root->children.move(row, to);
        endMoveRows();
      }
    }
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
  node_validator = new QRegExpValidator(QRegExp("^\\w+(,\\w+)*$"), this);
  double_validator = new DoubleValidator(this);

  label_font.setPointSize(8);
  minmax_font.setPixelSize(10);
}

QSize SignalItemDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const {
  int width = option.widget->size().width() / 2;
  if (index.column() == 0) {
    int spacing = option.widget->style()->pixelMetric(QStyle::PM_TreeViewIndentation) + color_label_width + 8;
    auto text = index.data(Qt::DisplayRole).toString();
    auto item = (SignalModel::Item *)index.internalPointer();
    if (item->type == SignalModel::Item::Sig && item->sig->type != cabana::Signal::Type::Normal) {
      text += item->sig->type == cabana::Signal::Type::Multiplexor ? QString(" M ") : QString(" m%1 ").arg(item->sig->multiplex_value);
      spacing += (option.widget->style()->pixelMetric(QStyle::PM_FocusFrameHMargin) + 1) * 2;
    }
    width = std::min<int>(option.widget->size().width() / 3.0, option.fontMetrics.width(text) + spacing);
  }
  return {width, option.fontMetrics.height() + option.widget->style()->pixelMetric(QStyle::PM_FocusFrameVMargin) * 2};
}

void SignalItemDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto item = (SignalModel::Item *)index.internalPointer();
  if (editor && item->type == SignalModel::Item::Sig && index.column() == 1) {
    QRect geom = option.rect;
    geom.setLeft(geom.right() - editor->sizeHint().width());
    editor->setGeometry(geom);
    button_size = geom.size();
    return;
  }
  QStyledItemDelegate::updateEditorGeometry(editor, option, index);
}

void SignalItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  const int h_margin = option.widget->style()->pixelMetric(QStyle::PM_FocusFrameHMargin) + 1;
  const int v_margin = option.widget->style()->pixelMetric(QStyle::PM_FocusFrameVMargin);
  auto item = static_cast<SignalModel::Item*>(index.internalPointer());

  QRect rect = option.rect.adjusted(h_margin, v_margin, -h_margin, -v_margin);
  painter->setRenderHint(QPainter::Antialiasing);
  if (option.state & QStyle::State_Selected) {
    painter->fillRect(option.rect, option.palette.brush(QPalette::Normal, QPalette::Highlight));
  }

  if (index.column() == 0) {
    if (item->type == SignalModel::Item::Sig) {
      // color label
      QPainterPath path;
      QRect icon_rect{rect.x(), rect.y(), color_label_width, rect.height()};
      path.addRoundedRect(icon_rect, 3, 3);
      painter->setPen(item->highlight ? Qt::white : Qt::black);
      painter->setFont(label_font);
      painter->fillPath(path, item->sig->color.darker(item->highlight ? 125 : 0));
      painter->drawText(icon_rect, Qt::AlignCenter, QString::number(item->row() + 1));

      rect.setLeft(icon_rect.right() + h_margin * 2);
      // multiplexer indicator
      if (item->sig->type != cabana::Signal::Type::Normal) {
        QString indicator = item->sig->type == cabana::Signal::Type::Multiplexor ? QString(" M ") : QString(" m%1 ").arg(item->sig->multiplex_value);
        QRect indicator_rect{rect.x(), rect.y(), option.fontMetrics.width(indicator), rect.height()};
        painter->setBrush(Qt::gray);
        painter->setPen(Qt::NoPen);
        painter->drawRoundedRect(indicator_rect, 3, 3);
        painter->setPen(Qt::white);
        painter->drawText(indicator_rect, Qt::AlignCenter, indicator);
        rect.setLeft(indicator_rect.right() + h_margin * 2);
      }
    } else {
      rect.setLeft(option.widget->style()->pixelMetric(QStyle::PM_TreeViewIndentation) + color_label_width + h_margin * 3);
    }

    // name
    auto text = option.fontMetrics.elidedText(index.data(Qt::DisplayRole).toString(), Qt::ElideRight, rect.width());
    painter->setPen(option.palette.color(option.state & QStyle::State_Selected ? QPalette::HighlightedText : QPalette::Text));
    painter->setFont(option.font);
    painter->drawText(rect, option.displayAlignment, text);
  } else if (index.column() == 1) {
    if (!item->sparkline.pixmap.isNull()) {
      QSize sparkline_size = item->sparkline.pixmap.size() / item->sparkline.pixmap.devicePixelRatio();
      painter->drawPixmap(QRect(rect.topLeft(), sparkline_size), item->sparkline.pixmap);
      // min-max value
      painter->setPen(option.palette.color(option.state & QStyle::State_Selected ? QPalette::HighlightedText : QPalette::Text));
      rect.adjust(sparkline_size.width() + 1, 0, 0, 0);
      int value_adjust = 10;
      if (!item->sparkline.isEmpty() && (item->highlight || option.state & QStyle::State_Selected)) {
        painter->drawLine(rect.topLeft(), rect.bottomLeft());
        rect.adjust(5, -v_margin, 0, v_margin);
        painter->setFont(minmax_font);
        QString min = QString::number(item->sparkline.min_val);
        QString max = QString::number(item->sparkline.max_val);
        painter->drawText(rect, Qt::AlignLeft | Qt::AlignTop, max);
        painter->drawText(rect, Qt::AlignLeft | Qt::AlignBottom, min);
        QFontMetrics fm(minmax_font);
        value_adjust = std::max(fm.width(min), fm.width(max)) + 5;
      } else if (!item->sparkline.isEmpty() && item->sig->type == cabana::Signal::Type::Multiplexed) {
        // display freq of multiplexed signal
        painter->setFont(label_font);
        QString freq = QString("%1 hz").arg(item->sparkline.freq(), 0, 'g', 2);
        painter->drawText(rect.adjusted(5, 0, 0, 0), Qt::AlignLeft | Qt::AlignVCenter, freq);
        value_adjust = QFontMetrics(label_font).width(freq) + 10;
      }
      // signal value
      painter->setFont(option.font);
      rect.adjust(value_adjust, 0, -button_size.width(), 0);
      auto text = option.fontMetrics.elidedText(index.data(Qt::DisplayRole).toString(), Qt::ElideRight, rect.width());
      painter->drawText(rect, Qt::AlignRight | Qt::AlignVCenter, text);
    } else {
      QStyledItemDelegate::paint(painter, option, index);
    }
  }
}

QWidget *SignalItemDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto item = (SignalModel::Item *)index.internalPointer();
  if (item->type == SignalModel::Item::Name || item->type == SignalModel::Item::Node || item->type == SignalModel::Item::Offset ||
      item->type == SignalModel::Item::Factor || item->type == SignalModel::Item::MultiplexValue ||
      item->type == SignalModel::Item::Min || item->type == SignalModel::Item::Max) {
    QLineEdit *e = new QLineEdit(parent);
    e->setFrame(false);
    if (item->type == SignalModel::Item::Name) e->setValidator(name_validator);
    else if (item->type == SignalModel::Item::Node) e->setValidator(node_validator);
    else e->setValidator(double_validator);

    if (item->type == SignalModel::Item::Name) {
      QCompleter *completer = new QCompleter(dbc()->signalNames(), e);
      completer->setCaseSensitivity(Qt::CaseInsensitive);
      completer->setFilterMode(Qt::MatchContains);
      e->setCompleter(completer);
    }
    return e;
  } else if (item->type == SignalModel::Item::Size) {
    QSpinBox *spin = new QSpinBox(parent);
    spin->setFrame(false);
    spin->setRange(1, CAN_MAX_DATA_BYTES);
    return spin;
  } else if (item->type == SignalModel::Item::SignalType) {
    QComboBox *c = new QComboBox(parent);
    c->addItem(signalTypeToString(cabana::Signal::Type::Normal), (int)cabana::Signal::Type::Normal);
    if (!dbc()->msg(((SignalModel *)index.model())->msg_id)->multiplexor) {
      c->addItem(signalTypeToString(cabana::Signal::Type::Multiplexor), (int)cabana::Signal::Type::Multiplexor);
    } else if (item->sig->type != cabana::Signal::Type::Multiplexor) {
      c->addItem(signalTypeToString(cabana::Signal::Type::Multiplexed), (int)cabana::Signal::Type::Multiplexed);
    }
    return c;
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

void SignalItemDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const {
  auto item = (SignalModel::Item *)index.internalPointer();
  if (item->type == SignalModel::Item::SignalType) {
    model->setData(index, ((QComboBox*)editor)->currentData().toInt());
    return;
  }
  QStyledItemDelegate::setModelData(editor, model, index);
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
  filter_edit->setPlaceholderText(tr("Filter Signal"));
  hl->addWidget(filter_edit);
  hl->addStretch(1);

  // WARNING: increasing the maximum range can result in severe performance degradation.
  // 30s is a reasonable value at present.
  const int max_range = 30; // 30s
  settings.sparkline_range = std::clamp(settings.sparkline_range, 1, max_range);
  hl->addWidget(sparkline_label = new QLabel());
  hl->addWidget(sparkline_range_slider = new QSlider(Qt::Horizontal, this));
  sparkline_range_slider->setRange(1, max_range);
  sparkline_range_slider->setValue(settings.sparkline_range);
  sparkline_range_slider->setToolTip(tr("Sparkline time range"));

  auto collapse_btn = new ToolButton("dash-square", tr("Collapse All"));
  collapse_btn->setIconSize({12, 12});
  hl->addWidget(collapse_btn);

  // tree view
  tree = new TreeView(this);
  tree->setModel(model = new SignalModel(this));
  tree->setItemDelegate(delegate = new SignalItemDelegate(this));
  tree->setFrameShape(QFrame::NoFrame);
  tree->setHeaderHidden(true);
  tree->setMouseTracking(true);
  tree->setExpandsOnDoubleClick(false);
  tree->setEditTriggers(QAbstractItemView::AllEditTriggers);
  tree->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
  tree->header()->setStretchLastSection(true);
  tree->setMinimumHeight(300);

  // Use a distinctive background for the whole row containing a QSpinBox or QLineEdit
  QString nodeBgColor = palette().color(QPalette::AlternateBase).name(QColor::HexArgb);
  tree->setStyleSheet(QString("QSpinBox{background-color:%1;border:none;} QLineEdit{background-color:%1;}").arg(nodeBgColor));

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);
  main_layout->addWidget(title_bar);
  main_layout->addWidget(tree);
  updateToolBar();

  QObject::connect(filter_edit, &QLineEdit::textEdited, model, &SignalModel::setFilter);
  QObject::connect(sparkline_range_slider, &QSlider::valueChanged, this, &SignalView::setSparklineRange);
  QObject::connect(collapse_btn, &QPushButton::clicked, tree, &QTreeView::collapseAll);
  QObject::connect(tree, &QAbstractItemView::clicked, this, &SignalView::rowClicked);
  QObject::connect(tree, &QTreeView::viewportEntered, [this]() { emit highlight(nullptr); });
  QObject::connect(tree, &QTreeView::entered, [this](const QModelIndex &index) { emit highlight(model->getItem(index)->sig); });
  QObject::connect(model, &QAbstractItemModel::modelReset, this, &SignalView::rowsChanged);
  QObject::connect(model, &QAbstractItemModel::rowsRemoved, this, &SignalView::rowsChanged);
  QObject::connect(dbc(), &DBCManager::signalAdded, this, &SignalView::handleSignalAdded);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &SignalView::handleSignalUpdated);
  QObject::connect(tree->verticalScrollBar(), &QScrollBar::valueChanged, [this]() { updateState(); });
  QObject::connect(tree->verticalScrollBar(), &QScrollBar::rangeChanged, [this]() { updateState(); });
  QObject::connect(can, &AbstractStream::msgsReceived, this, &SignalView::updateState);
  QObject::connect(tree->header(), &QHeaderView::sectionResized, [this](int logicalIndex, int oldSize, int newSize) {
    if (logicalIndex == 1) {
      value_column_width = newSize;
      updateState();
    }
  });

  setWhatsThis(tr(R"(
    <b>Signal view</b><br />
    <!-- TODO: add descprition here -->
  )"));
}

void SignalView::setMessage(const MessageId &id) {
  max_value_width = 0;
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

      auto remove_btn = new ToolButton("x", tr("Remove signal"));
      auto plot_btn = new ToolButton("graph-up", "");
      plot_btn->setCheckable(true);
      h->addWidget(plot_btn);
      h->addWidget(remove_btn);

      tree->setIndexWidget(index, w);
      auto sig = model->getItem(index)->sig;
      QObject::connect(remove_btn, &QToolButton::clicked, [=]() { UndoStack::push(new RemoveSigCommand(model->msg_id, sig)); });
      QObject::connect(plot_btn, &QToolButton::clicked, [=](bool checked) {
        emit showChart(model->msg_id, sig, checked, QGuiApplication::keyboardModifiers() & Qt::ShiftModifier);
      });
    }
  }
  updateToolBar();
  updateChartState();
  updateState();
}

void SignalView::rowClicked(const QModelIndex &index) {
  auto item = model->getItem(index);
  if (item->type == SignalModel::Item::Sig || item->type == SignalModel::Item::ExtraInfo) {
    auto expand_index = model->index(index.row(), 0, index.parent());
    tree->setExpanded(expand_index, !tree->isExpanded(expand_index));
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
    bool chart_opened = charts->hasSignal(model->msg_id, item->sig);
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
      emit model->dataChanged(model->index(i, 1), model->index(i, 1), {Qt::DisplayRole});
    }
  }
}

void SignalView::updateToolBar() {
  signal_count_lb->setText(tr("Signals: %1").arg(model->rowCount()));
  sparkline_label->setText(utils::formatSeconds(settings.sparkline_range));
}

void SignalView::setSparklineRange(int value) {
  settings.sparkline_range = value;
  updateToolBar();
  updateState();
}

void SignalView::handleSignalAdded(MessageId id, const cabana::Signal *sig) {
  if (id.address == model->msg_id.address) {
    selectSignal(sig);
  }
}

void SignalView::handleSignalUpdated(const cabana::Signal *sig) {
  if (int row = model->signalRow(sig); row != -1)
    updateState();
}

std::pair<QModelIndex, QModelIndex> SignalView::visibleSignalRange() {
  auto topLevelIndex = [](QModelIndex index) {
    while (index.isValid() && index.parent().isValid()) index = index.parent();
    return index;
  };

  const auto viewport_rect = tree->viewport()->rect();
  QModelIndex first_visible = tree->indexAt(viewport_rect.topLeft());
  if (first_visible.parent().isValid()) {
    first_visible = topLevelIndex(first_visible);
    first_visible = first_visible.siblingAtRow(first_visible.row() + 1);
  }

  QModelIndex last_visible = topLevelIndex(tree->indexAt(viewport_rect.bottomRight()));
  if (!last_visible.isValid()) {
    last_visible = model->index(model->rowCount() - 1, 0);
  }
  return {first_visible, last_visible};
}

void SignalView::updateState(const std::set<MessageId> *msgs) {
  const auto &last_msg = can->lastMessage(model->msg_id);
  if (model->rowCount() == 0 || (msgs && !msgs->count(model->msg_id)) || last_msg.dat.size() == 0) return;

  for (auto item : model->root->children) {
    double value = 0;
    if (item->sig->getValue(last_msg.dat.data(), last_msg.dat.size(), &value)) {
      item->sig_val = item->sig->formatValue(value);
      max_value_width = std::max(max_value_width, fontMetrics().width(item->sig_val));
    }
  }

  auto [first_visible, last_visible] = visibleSignalRange();
  if (first_visible.isValid() && last_visible.isValid()) {
    const static int min_max_width = QFontMetrics(delegate->minmax_font).width("-000.00") + 5;
    int available_width = value_column_width - delegate->button_size.width();
    int value_width = std::min<int>(max_value_width + min_max_width, available_width / 2);
    QSize size(available_width - value_width,
               delegate->button_size.height() - style()->pixelMetric(QStyle::PM_FocusFrameVMargin) * 2);

    QFutureSynchronizer<void> synchronizer;
    for (int i = first_visible.row(); i <= last_visible.row(); ++i) {
      auto item = model->getItem(model->index(i, 1));
      synchronizer.addFuture(QtConcurrent::run(
          &item->sparkline, &Sparkline::update, model->msg_id, item->sig, last_msg.ts, settings.sparkline_range, size));
    }
    synchronizer.waitForFinished();
  }

  for (int i = 0; i < model->rowCount(); ++i) {
    emit model->dataChanged(model->index(i, 1), model->index(i, 1), {Qt::DisplayRole});
  }
}

void SignalView::resizeEvent(QResizeEvent* event) {
  updateState();
  QFrame::resizeEvent(event);
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
    table->setItem(row, 0, new QTableWidgetItem(QString::number(val)));
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
      val_desc.push_back({val.toDouble(), desc});
    }
  }
  QDialog::accept();
}

QWidget *ValueDescriptionDlg::Delegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  QLineEdit *edit = new QLineEdit(parent);
  edit->setFrame(false);
  if (index.column() == 0) {
    edit->setValidator(new DoubleValidator(parent));
  }
  return edit;
}
