#pragma once

#include <QAbstractItemModel>
#include <QLabel>
#include <QLineEdit>
#include <QStyledItemDelegate>
#include <QTreeView>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/chartswidget.h"
#include "tools/cabana/dbcmanager.h"

class SignalModel : public QAbstractItemModel {
  Q_OBJECT
public:
  struct Item {
    ~Item() { qDeleteAll(children); }
    inline int row() { return parent->children.indexOf(this); }

    const Signal *sig = nullptr;
    bool highlight = false;
    QString title;
    int data_id = -1;
    Item *parent = nullptr;
    QList<Item *> children;
  };

  SignalModel(QObject *parent);
  int rowCount(const QModelIndex &parent = QModelIndex()) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return 2; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const override;
  QModelIndex parent(const QModelIndex &index) const override;
  Qt::ItemFlags flags(const QModelIndex &index) const override;
  bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;
  void setMessage(const QString &id);
  void setFilter(const QString &txt);
  void addSignal(int start_bit, int size, bool little_endian);
  bool saveSignal(const Signal *origin_s, Signal &s);
  void resizeSignal(const Signal *sig, int start_bit, int size);
  void removeSignal(const Signal *sig);
  inline Item *getItem(const QModelIndex &index) const { return index.isValid() ? (Item *)index.internalPointer() : root.get(); }
  int signalRow(const Signal *sig) const;

private:
  void insertItem(SignalModel::Item *parent_item, int pos, const Signal *sig);
  void handleSignalRemoved(const Signal *sig);
  void handleSignalAdded(uint32_t address, const Signal *sig);
  void handleMsgChanged(uint32_t address);
  void refresh();

  QString msg_id;
  QString filter_str;
  std::unique_ptr<Item> root;
  friend class SignalView;
};

class SignalItemDelegate : public QStyledItemDelegate {
public:
  SignalItemDelegate(QObject *parent) {}
  void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
};

class SignalView : public QWidget {
  Q_OBJECT

public:
  SignalView(ChartsWidget *charts, QWidget *parent);
  void setMessage(const QString &id);
  void signalHovered(const Signal *sig);
  void updateChartState();
  void expandSignal(const Signal *sig);
  SignalModel *model = nullptr;

signals:
  void highlight(const Signal *sig);
  void showChart(const QString &name, const Signal *sig, bool show, bool merge);

private:
  void rowsChanged();

  QString msg_id;
  QTreeView *tree;
  QLineEdit *filter_edit;
  ChartsWidget *charts;
  QLabel *signal_count_lb, *tip_lb;
};
