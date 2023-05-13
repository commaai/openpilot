#include "tools/cabana/tools/signal_filterer.h"

filter_return filter(
  std::vector<SearchSignal> &in, const SignalFiltererParams &params,
  filter_function signalMatches) {

  std::vector<SearchSignal> filtered; // signals that pass
  std::vector<SearchSignal> removed; // signals that fail

  std::copy_if(in.begin(), in.end(), std::back_inserter(filtered), [=] (SearchSignal sig) { return signalMatches(sig, params); });
  std::copy_if(in.begin(), in.end(), std::back_inserter(removed), [=] (SearchSignal sig) { return !signalMatches(sig, params); });

  return std::tuple<std::vector<SearchSignal>, std::vector<SearchSignal>>(filtered, removed);
}

 SearchSignal::SearchSignal(MessageId _messageID, int start_bit, int size, bool little_endian) : messageID(_messageID){
  this->is_little_endian = little_endian;
  this->factor = 1;

  updateSigSizeParamsFromRange(this, start_bit, size);
}

int64_t  SearchSignal::getValue(double ts) const {
    const auto &events = can->events(messageID);
    CanEventLessThan comp;

    // get closest event to this ts
    auto event = std::lower_bound(events.begin(), events.end(), (can->routeStartTime() + ts) * 1e9, comp);

    if(event != events.begin()){
      event -= 1;
    }

    return get_raw_value((*event)->dat, (*event)->size, this);
}

int64_t  SearchSignal::getCurrentValue() const {
  return get_raw_value((const uint8_t*)can->last_msgs[messageID].dat.constData(), can->last_msgs[messageID].dat.size(), this);
}

QString  SearchSignal::toString() const {
    auto range = getSignalRange(this);
    return QString("%1:%2").arg(std::get<0>(range)).arg(std::get<1>(range));
}


void updateSigSizeParamsFromRange(SearchSignal *s, int start_bit, int size) {
  s->start_bit = s->is_little_endian ? start_bit : bigEndianBitIndex(start_bit);
  s->size = size;
  if (s->is_little_endian) {
    s->lsb = s->start_bit;
    s->msb = s->start_bit + s->size - 1;
  } else {
    s->lsb = bigEndianStartBitsIndex(bigEndianBitIndex(s->start_bit) + s->size - 1);
    s->msb = s->start_bit;
  }
}

std::pair<int, int> getSignalRange(const SearchSignal *s) {
  int from = s->is_little_endian ? s->start_bit : bigEndianBitIndex(s->start_bit);
  int to = from + s->size - 1;
  return {from, to};
}

int64_t get_raw_value(const uint8_t *data, size_t data_size, const SearchSignal *sig) {
  int64_t val = 0;

  int i = sig->msb / 8;
  int bits = sig->size;
  while (i >= 0 && i < data_size && bits > 0) {
    int lsb = (int)(sig->lsb / 8) == i ? sig->lsb : i * 8;
    int msb = (int)(sig->msb / 8) == i ? sig->msb : (i + 1) * 8 - 1;
    int size = msb - lsb + 1;

    uint64_t d = (data[i] >> (lsb - (i * 8))) & ((1ULL << size) - 1);
    val |= d << (bits - size);

    bits -= size;
    i = sig->is_little_endian ? i - 1 : i + 1;
  }
  if (sig->is_signed) {
    val -= ((val >> (sig->size - 1)) & 0x1) ? (1ULL << sig->size) : 0;
  }
  return val;
}

bool exactValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) == params.value1; }
bool biggerThanFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) > params.value1; }
bool smallerThanFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) < params.value1; }
bool unknownInitialValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return true; }
bool increasedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) > sig.getValue(params.ts_prev); }
bool decreasedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) < sig.getValue(params.ts_prev); }
bool changedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) != sig.getValue(params.ts_prev); }
bool unchangedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) == sig.getValue(params.ts_prev); }