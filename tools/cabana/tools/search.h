#pragma once

#include <QDialog>
#include <QPushButton>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QDoubleValidator>
#include <QComboBox>
#include <QSpinBox>
#include <QLabel>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QMenu>

#include <iostream>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

#include "tools/cabana/tools/signal_filterer.h"

uint64_t getBitValue(uint64_t val, int offset, int size);

enum ScanType {
  ExactValue,
  BiggerThan,
  SmallerThan,
  ValueBetween,
  IncreasedValue,
  IncreasedValueBy,
  DecreasedValue,
  DecreasedValueBy,
  ChangedValue,
  UnchangedValue,
  UnknownInitialValue
};


class SearchDlg : public QDialog {
  Q_OBJECT

public:
  SearchDlg(QWidget *parent);

private:
  void firstScan();
  void nextScan();
  void undoScan();
  void reset();

  void showDataTableContextMenu(const QPoint &pt);

  void update();
  void updateRowData();
  void setRowData(int row, QString msgID, QString bitRange, QString currentValue, QString previousValue);

  std::vector<ScanType> enabledScanTypes();

  SignalFilterer* getCurrentFilterer();

  uint32_t scan_bits_range_min = 1;
  uint32_t scan_bits_range_max = 32;

  uint64_t scan_value1 = 0;
  uint64_t scan_value2 = 0;

  std::vector<SearchSignal> filteredSignals;

  ScanType selectedScanType;

  QLabel* numberOfSigsLabel;
  QComboBox *scan_type;
  QPushButton *first_scan_button;
  QPushButton *next_scan_button;
  QPushButton *undo_scan_button;

  QTableWidget *data_table;

  // Search history, at a specific time
  std::vector<std::shared_ptr<SignalFilterer>> searchHistory;

  bool scanningStarted();
};

