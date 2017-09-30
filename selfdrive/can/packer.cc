#include <cassert>

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <map>

#include "common.h"

#define WARN printf

namespace {

class CANPacker {
public:
  CANPacker(const std::string& dbc_name) {
    dbc = dbc_lookup(dbc_name);
    assert(dbc);

    for (int i=0; i<dbc->num_msgs; i++) {
      const Msg* msg = &dbc->msgs[i];
      for (int j=0; j<msg->num_sigs; j++) {
        const Signal* sig = &msg->sigs[j];
        signal_lookup[std::make_pair(msg->address, std::string(sig->name))] = *sig;
      }
    }
  }

  uint64_t pack(uint32_t address, const std::vector<SignalPackValue> &signals) {
    uint64_t ret = 0;
    for (const auto& sigval : signals) {
      std::string name = std::string(sigval.name);
      double value = sigval.value;

      auto sig_it = signal_lookup.find(make_pair(address, name));
      if (sig_it == signal_lookup.end()) {
        WARN("undefined signal %s", name.c_str());
        continue;
      }
      auto sig = sig_it->second;

      int64_t ival = (int64_t)((value - sig.offset) / sig.factor);
      if (ival < 0) {
        WARN("signed pack unsupported right now");
        continue;
      }

      uint64_t mask = ((1ULL << sig.b2)-1) << sig.bo;
      uint64_t dat = (ival & ((1ULL << sig.b2)-1)) << sig.bo;
      ret &= ~mask;
      ret |= dat;
    }

    return ret;
  }



private:
  const DBC *dbc = NULL;
  std::map<std::pair<uint32_t, std::string>, Signal> signal_lookup;
};

}

extern "C" {

void* canpack_init(const char* dbc_name) {
  CANPacker *ret = new CANPacker(std::string(dbc_name));
  return (void*)ret;
}

uint64_t canpack_pack(void* inst, uint32_t address, size_t num_vals, const SignalPackValue *vals) {
  CANPacker *cp = (CANPacker*)inst;

  return cp->pack(address, std::vector<SignalPackValue>(vals, vals+num_vals));
}

}

