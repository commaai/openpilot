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
    bool operator() (const CanEvent *left, const CanEvent *right)
    {
        return left->mono_time < right->mono_time;
    }
    bool operator() (const CanEvent *left, uint64_t right)
    {
        return left->mono_time < right;
    }
    bool operator() (uint64_t left, const CanEvent *right)
    {
        return left < right->mono_time;
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

        int64_t getValue(double ts){
            auto events = can->events(messageID);
            CanEventLessThan comp;

            // get closest event to this ts
            auto event = std::lower_bound(events.begin(), events.end(), (can->routeStartTime() + ts) * 1e9, comp);

            if(event != events.begin()){
              event -= 1;
            }


            return get_raw_value((*event)->dat, (*event)->size, *this);
        }

        int64_t getCurrentValue(){
          return get_raw_value((const uint8_t*)can->last_msgs[messageID].dat.constData(), can->last_msgs[messageID].dat.size(), *this);
        }

        QString toString(){
            auto range = getSignalRange(this);
            return QString("%1:%2").arg(std::get<0>(range)).arg(std::get<1>(range));
        }
};

struct SignalFiltererParams{
    double ts_scan; // timestamp of current scan
    double ts_prev; // timestamp of previous scan

    uint64_t value1; // values from UI
    uint64_t value2;
};

class SignalFilterer {
  public:
    // signals that were filtered out, for the abilty to undo scans
    std::vector<SearchSignal> filteredSignals;

    SignalFiltererParams params;

    virtual bool signalMatches(SearchSignal sig) = 0;

    SignalFilterer(SignalFiltererParams params_in) : params(params_in) {}

    virtual ~SignalFilterer() = default;

    std::vector<SearchSignal> filter(std::vector<SearchSignal> &in) {
      std::vector<SearchSignal> ret;

      filteredSignals.clear();

      std::copy_if(in.begin(), in.end(), std::back_inserter(ret), [=] (SearchSignal sig) { return signalMatches(sig); });
      std::copy_if(in.begin(), in.end(), std::back_inserter(filteredSignals), [=] (SearchSignal sig) { return !signalMatches(sig); });

      return ret;
    }
};

class ExactValueSignalFilterer : public SignalFilterer {
  using SignalFilterer::SignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(params.ts_scan) == params.value1;
  }
};

class BiggerThanSignalFilterer : public SignalFilterer {
  using SignalFilterer::SignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(params.ts_scan) > params.value1;
  }
};

class SmallerThanSignalFilterer : public SignalFilterer {
  using SignalFilterer::SignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(params.ts_scan) < params.value1;
  }
};

class UnknownInitialValueSignalFilter : public SignalFilterer {
  using SignalFilterer::SignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return true;
  }
};

class IncreasedValueSignalFilter : public SignalFilterer {
  using SignalFilterer::SignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(params.ts_scan) > sig.getValue(params.ts_prev);
  }
};

class DecreasedValueSignalFilter : public SignalFilterer {
  using SignalFilterer::SignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(params.ts_scan) < sig.getValue(params.ts_prev);
  }
};

class ChangedValueSignalFilter : public SignalFilterer {
  using SignalFilterer::SignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(params.ts_scan) != sig.getValue(params.ts_prev);
  }
};

class UnchangedValueSignalFilter : public SignalFilterer {
  using SignalFilterer::SignalFilterer;

  bool signalMatches(SearchSignal sig) {
    return sig.getValue(params.ts_scan) == sig.getValue(params.ts_prev);
  }
};