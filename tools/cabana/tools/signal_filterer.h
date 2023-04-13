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

class SearchSignal : public cabana::BaseSignal {
    public:
        SearchSignal(MessageId _messageID, int start_bit, int size, bool little_endian) : messageID(_messageID){
            this->is_little_endian = little_endian;
            this->factor = 1;

            updateSigSizeParamsFromRange(*this, start_bit, size);
        }

        MessageId messageID;

        int64_t previousValue;

        int64_t getValue(double ts){
            auto events = can->events().at(messageID);
            CanEventLessThan comp;
            auto event = std::lower_bound(events.begin(), events.end(), (ts - can->routeStartTime()) * 1e9, comp);

            return get_raw_value(event->dat, event->size, *this);
        }

        QString toString(){
            auto range = getSignalRange(this);
            return QString("%1:%2").arg(std::get<0>(range)).arg(std::get<1>(range));
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

     double previousTime(){
      return std::get<1>(searchHistory[searchHistory.size() - 2]);
    }
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