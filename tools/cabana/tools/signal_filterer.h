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

class SearchSignal {
    public:
        int start_bit, msb, lsb, size;
        bool is_signed;
        double factor, offset;
        bool is_little_endian;

        SearchSignal(MessageId _messageID, int start_bit, int size, bool little_endian);
        MessageId messageID;
        int64_t getValue(double ts) const;
        int64_t getCurrentValue() const;
        QString toString() const;
};

struct SignalFiltererParams{
    double ts_scan; // timestamp of current scan
    double ts_prev; // timestamp of previous scan

    uint64_t value1; // values from UI
    uint64_t value2;
};

void updateSigSizeParamsFromRange(SearchSignal *s, int start_bit, int size);
std::pair<int, int> getSignalRange(const SearchSignal *s);

int64_t get_raw_value(const uint8_t *data, size_t data_size, const SearchSignal *sig);

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