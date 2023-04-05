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

#include <iostream>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

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

struct CanEventLessThan
{
    bool operator() (const CanEvent & left, const CanEvent & right)
    {
        return left.mono_time < right.mono_time;
    }
    bool operator() (const CanEvent & left, double right)
    {
        return left.mono_time < right;
    }
    bool operator() (double left, const CanEvent & right)
    {
        return left < right.mono_time;
    }
};

class SearchSignal {
    public:
        SearchSignal(MessageId _messageID, size_t _offset, size_t _size) : messageID(_messageID), offset(_offset), size(_size) {}

        MessageId messageID;
        size_t offset;
        size_t size;

        uint64_t previousValue;

        uint64_t getValue(double ts){
            auto events = can->events().at(messageID);
            CanEventLessThan comp;
            std::lower_bound(events.begin(), events.end(), (ts - can->routeStartTime()) * 1e9, comp);

            auto msg = can->last_msgs[messageID];
            uint64_t* data = (uint64_t*)(msg.dat.data());
            return getBitValue(*data, offset, size);
        }
};

class SignalFilterer {
  public:
    std::vector<std::tuple<SignalFilterer*, double>> searchHistory;

    virtual bool signalMatches(SearchSignal sig) = 0;

    SignalFilterer(){

    }

    virtual ~SignalFilterer() = default;

    std::vector<SearchSignal> filter(std::vector<SearchSignal> in) {
      std::vector<SearchSignal> ret;

      std::copy_if(in.begin(), in.end(), std::back_inserter(ret), [=] (SearchSignal sig) { return signalMatches(sig); });

      return ret;
    }

    double currentTime(){
      return std::get<1>(searchHistory[searchHistory.size() - 1]);
    }
};

class SearchDlg : public QDialog {
  Q_OBJECT

public:
  SearchDlg(QWidget *parent);

private:
  void firstScan();
  void nextScan();
  void undoScan();

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
  std::vector<std::tuple<SignalFilterer*, double>> searchHistory;

  bool scanningStarted();
};

class ZeroInputSignalFilterer : public SignalFilterer {
  public:

    ZeroInputSignalFilterer(){

    }
};

class SingleInputSignalFilterer : public SignalFilterer {
  public:
    SingleInputSignalFilterer(uint64_t _value) : value(_value){

    }
  protected:
    uint64_t value;
};

class DoubleInputSignalFilterer : public SignalFilterer {
  public:
    DoubleInputSignalFilterer(uint64_t _value1, uint64_t _value2) : value1(_value1), value2(_value2){

    }
  protected:
    uint64_t value1;
    uint64_t value2;
};

class ExactValueSignalFilterer : public SingleInputSignalFilterer {
  using SingleInputSignalFilterer::SingleInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(currentTime()) == value;
  }
};

class BiggerThanSignalFilterer : public SingleInputSignalFilterer {
  using SingleInputSignalFilterer::SingleInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(currentTime()) > value;
  }
};

class SmallerThanSignalFilterer : public SingleInputSignalFilterer {
  using SingleInputSignalFilterer::SingleInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(currentTime()) < value;
  }
};

class UnknownInitialValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return true;
  }
};

class IncreasedValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(currentTime()) > sig.previousValue;
  }
};

class DecreasedValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(currentTime()) < sig.previousValue;
  }
};

class ChangedValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(currentTime()) != sig.previousValue;
  }
};

class UnchangedValueSignalFilter : public ZeroInputSignalFilterer {
  using ZeroInputSignalFilterer::ZeroInputSignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(currentTime()) == sig.previousValue;
  }
};