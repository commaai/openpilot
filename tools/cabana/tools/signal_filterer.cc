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

bool exactValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) == params.value1; }
bool biggerThanFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) > params.value1; }
bool smallerThanFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) < params.value1; }
bool unknownInitialValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return true; }
bool increasedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) > sig.getValue(params.ts_prev); }
bool decreasedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) < sig.getValue(params.ts_prev); }
bool changedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) != sig.getValue(params.ts_prev); }
bool unchangedValueFilter(const SearchSignal &sig, const SignalFiltererParams& params) { return sig.getValue(params.ts_scan) == sig.getValue(params.ts_prev); }