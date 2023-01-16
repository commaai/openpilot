#pragma once

#include <deque>
#include <QCheckBox>
#include <QComboBox>
#include <QHeaderView>
#include <QLineEdit>
#include <QTableView>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

class HeaderView : public QHeaderView {
public:
  HeaderView(Qt::Orientation orientation, QWidget *parent = nullptr) : QHeaderView(orientation, parent) {}
  QSize sectionSizeFromContents(int logicalIndex) const override;
  void paintSection(QPainter *painter, const QRect &rect, int logicalIndex) const;
};

class HistoryLogModel : public QAbstractTableModel {
  Q_OBJECT

public:
  enum DisplayType {
    Signals,
    Hex
  };

  HistoryLogModel(QObject *parent) : QAbstractTableModel(parent) {}
  void setMessage(const QString &message_id);
  void updateState();
  void setFilter(int sig_idx, const QString &value, std::function<bool(double, double)> cmp);
  void setDisplayType(DisplayType type);
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  void fetchMore(const QModelIndex &parent) override;
  inline bool canFetchMore(const QModelIndex &parent) const override { return has_more_data; }
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return messages.size(); }
  int columnCount(const QModelIndex &parent = QModelIndex()) const override {
    return display_type == HistoryLogModel::Hex ? 2 : std::max(1ul, sigs.size()) + 1;
  }
  void setDynamicMode(int state);
  void segmentsMerged();
  void refresh();

  struct Message {
    uint64_t mono_time = 0;
    QVector<double> sig_values;
    QString data;
  };

  template <class InputIt>
  std::deque<HistoryLogModel::Message> fetchData(InputIt first, InputIt last, uint64_t min_time);
  std::deque<Message> fetchData(uint64_t from_time, uint64_t min_time = 0);

  QString msg_id;
  bool has_more_data = true;
  const int batch_size = 50;
  int filter_sig_idx = -1;
  double filter_value = 0;
  uint64_t last_fetch_time = 0;
  std::function<bool(double, double)> filter_cmp = nullptr;
  std::deque<Message> messages;
  std::vector<const Signal*> sigs;
  bool dynamic_mode = false;
  DisplayType display_type = HistoryLogModel::Signals;
};

class HistoryLog : public QTableView {
public:
  HistoryLog(QWidget *parent);
  int sizeHintForColumn(int column) const override { return -1; };
};

class LogsWidget : public QWidget {
  Q_OBJECT

public:
  LogsWidget(QWidget *parent);
  void setMessage(const QString &message_id);
  void updateState();

signals:
  void openChart(const QString &msg_id, const Signal *sig);

private slots:
  void setFilter();
  void displayTypeChanged();

private:
  void doubleClicked(const QModelIndex &index);
  void showEvent(QShowEvent *event) override;

  HistoryLog *logs;
  HistoryLogModel *model;
  QCheckBox *dynamic_mode;
  QComboBox *signals_cb, *comp_box, *display_type_cb;
  QLineEdit *value_edit;
  QString cur_filter_text;
};
