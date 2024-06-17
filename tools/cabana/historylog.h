#pragma once

#include <deque>
#include <vector>

#include <QComboBox>
#include <QHeaderView>
#include <QLineEdit>
#include <QTableView>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

class HeaderView : public QHeaderView {
public:
  HeaderView(Qt::Orientation orientation, QWidget *parent = nullptr) : QHeaderView(orientation, parent) {}
  QSize sectionSizeFromContents(int logicalIndex) const override;
  void paintSection(QPainter *painter, const QRect &rect, int logicalIndex) const;
};

class HistoryLogModel : public QAbstractTableModel {
  Q_OBJECT

public:
  HistoryLogModel(QObject *parent) : QAbstractTableModel(parent) {}
  void setMessage(const MessageId &message_id);
  void updateState(bool clear = false);
  void setFilter(int sig_idx, const QString &value, std::function<bool(double, double)> cmp);
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  void fetchMore(const QModelIndex &parent) override;
  bool canFetchMore(const QModelIndex &parent) const override;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return messages.size(); }
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return !isHexMode() ? sigs.size() + 1 : 2; }
  inline bool isHexMode() const { return sigs.empty() || hex_mode; }
  void reset();
  void setHexMode(bool hex_mode);

  struct Message {
    uint64_t mono_time = 0;
    std::vector<double> sig_values;
    std::vector<uint8_t> data;
    std::vector<QColor> colors;
  };

  void fetchData(std::deque<Message>::iterator insert_pos, uint64_t from_time, uint64_t min_time);

  MessageId msg_id;
  CanData hex_colors;
  const int batch_size = 50;
  int filter_sig_idx = -1;
  double filter_value = 0;
  std::function<bool(double, double)> filter_cmp = nullptr;
  std::deque<Message> messages;
  std::vector<cabana::Signal *> sigs;
  bool hex_mode = false;
};

class LogsWidget : public QFrame {
  Q_OBJECT

public:
  LogsWidget(QWidget *parent);
  void setMessage(const MessageId &message_id) { model->setMessage(message_id); }
  void updateState() { model->updateState(); }
  void showEvent(QShowEvent *event) override { model->updateState(true); }

private slots:
  void filterChanged();
  void exportToCSV();
  void modelReset();

private:
  QTableView *logs;
  HistoryLogModel *model;
  QComboBox *signals_cb, *comp_box, *display_type_cb;
  QLineEdit *value_edit;
  QWidget *filters_widget;
  ToolButton *export_btn;
  MessageBytesDelegate *delegate;
};
