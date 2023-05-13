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

        int64_t getValue(double ts) const {
            const auto &events = can->events(messageID);
            CanEventLessThan comp;

            // get closest event to this ts
            auto event = std::lower_bound(events.begin(), events.end(), (can->routeStartTime() + ts) * 1e9, comp);

            if(event != events.begin()){
              event -= 1;
            }

            return get_raw_value((*event)->dat, (*event)->size, *this);
        }

        int64_t getCurrentValue() const {
          return get_raw_value((const uint8_t*)can->last_msgs[messageID].dat.constData(), can->last_msgs[messageID].dat.size(), *this);
        }

        QString toString() const {
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

using filter_return = std::tuple<std::vector<SearchSignal>, std::vector<SearchSignal>>;
using filter_function = std::function<bool(const SearchSignal &sig, const SignalFiltererParams &params)>;

filter_return filter(std::vector<SearchSignal> &in, const SignalFiltererParams &params, filter_function signalMatches);

bool exactValueFilter(const SearchSignal &sig, const SignalFiltererParams& params);
bool biggerThanFilter(const SearchSignal &sig, const SignalFiltererParams& params);
bool smallerThanFilter(const SearchSignal &sig, const SignalFiltererParams& params);
bool unknownInitialValueFilter(const SearchSignal &sig, const SignalFiltererParams& params);
bool increasedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params);
bool decreasedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params);
bool changedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params);
bool unchangedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params);