#pragma once

#include <QAbstractItemModel>
#include <QLabel>
#include <QLineEdit>
#include <QStyledItemDelegate>
#include <QTableWidget>
#include <QTreeView>

#include "tools/cabana/chartswidget.h"

class SignalModel : public QAbstractItemModel {
  Q_OBJECT
public:
  struct Item {
    enum Type {Root, Sig, Name, Size, Endian, Signed, Offset, Factor, ExtraInfo, Unit, Comment, Min, Max, Desc };
    ~Item() { qDeleteAll(children); }
    inline int row() { return parent->children.indexOf(this); }

    Type type = Type::Root;
    Item *parent = nullptr;
    QList<Item *> children;

    const cabana::Signal *sig = nullptr;
    QString title;
    bool highlight = false;
    bool extra_expanded = false;
    QString sig_val = "-";
  };

  SignalModel(QObject *parent);
  int rowCount(const QModelIndex &parent = QModelIndex()) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return 2; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const override;
  QModelIndex parent(const QModelIndex &index) const override;
  Qt::ItemFlags flags(const QModelIndex &index) const override;
  bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;
  void setMessage(const MessageId &id);
  void setFilter(const QString &txt);
  void addSignal(int start_bit, int size, bool little_endian);
  bool saveSignal(const cabana::Signal *origin_s, cabana::Signal &s);
  void resizeSignal(const cabana::Signal *sig, int start_bit, int size);
  void removeSignal(const cabana::Signal *sig);
  Item *getItem(const QModelIndex &index) const;
  int signalRow(const cabana::Signal *sig) const;
  void showExtraInfo(const QModelIndex &index);

private:
  void insertItem(SignalModel::Item *parent_item, int pos, const cabana::Signal *sig);
  void handleSignalAdded(MessageId id, const cabana::Signal *sig);
  void handleSignalUpdated(const cabana::Signal *sig);
  void handleSignalRemoved(const cabana::Signal *sig);
  void handleMsgChanged(MessageId id);
  void refresh();
  void updateState(const QHash<MessageId, CanData> *msgs);

  MessageId msg_id;
  QString filter_str;
  std::unique_ptr<Item> root;
  friend class SignalView;
};

class ValueDescriptionDlg : public QDialog {
public:
  ValueDescriptionDlg(const ValueDescription &descriptions, QWidget *parent);
  ValueDescription val_desc;

private:
  struct Delegate : public QStyledItemDelegate {
    Delegate(QWidget *parent) : QStyledItemDelegate(parent) {}
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  };

  void save();
  QTableWidget *table;
};

class SignalItemDelegate : public QStyledItemDelegate {
public:
  SignalItemDelegate(QObject *parent);
  void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const;
  QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  QValidator *name_validator, *double_validator;
  QFont small_font;
  const int color_label_width = 18;
  mutable QHash<QString, int> width_cache;
};

class SignalView : public QFrame {
  Q_OBJECT

public:
  SignalView(ChartsWidget *charts, QWidget *parent);
  void setMessage(const MessageId &id);
  void signalHovered(const cabana::Signal *sig);
  void updateChartState();
  void selectSignal(const cabana::Signal *sig, bool expand = false);
  void rowClicked(const QModelIndex &index);
  SignalModel *model = nullptr;

signals:
  void highlight(const cabana::Signal *sig);
  void showChart(const MessageId &id, const cabana::Signal *sig, bool show, bool merge);

private:
  void rowsChanged();
  void leaveEvent(QEvent *event);

  struct TreeView : public QTreeView {
    TreeView(QWidget *parent) : QTreeView(parent) {}
    void rowsInserted(const QModelIndex &parent, int start, int end) override {
      ((SignalView *)parentWidget())->rowsChanged();
      // update widget geometries in QTreeView::rowsInserted
      QTreeView::rowsInserted(parent, start, end);
    }
  };

  MessageId msg_id;
  TreeView *tree;
  QLineEdit *filter_edit;
  ChartsWidget *charts;
  QLabel *signal_count_lb;
};
