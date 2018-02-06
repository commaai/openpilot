#include <cassert>

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <map>
#include <cmath>

#include "common.h"

#define WARN printf

namespace {

  uint64_t set_value(uint64_t ret, Signal sig, int64_t ival){
    uint64_t mask = ((1ULL << sig.b2)-1) << sig.bo;
    uint64_t dat = (ival & ((1ULL << sig.b2)-1)) << sig.bo;
    ret &= ~mask;
    ret |= dat;
    return ret;
  }

  class CANPacker {
  public:
    CANPacker(const std::string& dbc_name) {
      dbc = dbc_lookup(dbc_name);
      assert(dbc);

      for (int i=0; i<dbc->num_msgs; i++) {
        const Msg* msg = &dbc->msgs[i];
        message_lookup[msg->address] = *msg;
        for (int j=0; j<msg->num_sigs; j++) {
          const Signal* sig = &msg->sigs[j];
          signal_lookup[std::make_pair(msg->address, std::string(sig->name))] = *sig;
        }
      }
    }

    uint64_t pack(uint32_t address, const std::vector<SignalPackValue> &signals, int counter) {
      uint64_t ret = 0;
      for (const auto& sigval : signals) {
        std::string name = std::string(sigval.name);
        double value = sigval.value;

        auto sig_it = signal_lookup.find(std::make_pair(address, name));
        if (sig_it == signal_lookup.end()) {
          WARN("undefined signal %s - %d\n", name.c_str(), address);
          continue;
        }
        auto sig = sig_it->second;

        int64_t ival = (int64_t)(round((value - sig.offset) / sig.factor));
        if (ival < 0) {
          ival = (1ULL << sig.b2) + ival;
        }

        ret = set_value(ret, sig, ival);
      }

      if (counter >= 0){
        auto sig_it = signal_lookup.find(std::make_pair(address, "COUNTER"));
        if (sig_it == signal_lookup.end()) {
          WARN("COUNTER not defined\n");
          return ret;
        }
        auto sig = sig_it->second;

        if (sig.type != SignalType::HONDA_COUNTER){
          WARN("COUNTER signal type not valid\n");
        }

        ret = set_value(ret, sig, counter);
      }

      auto sig_it = signal_lookup.find(std::make_pair(address, "CHECKSUM"));
      if (sig_it != signal_lookup.end()) {
        auto sig = sig_it->second;
        if (sig.type == SignalType::HONDA_CHECKSUM){
          unsigned int chksm = honda_checksum(address, ret, message_lookup[address].size);
          ret = set_value(ret, sig, chksm);
        }
        else if (sig.type == SignalType::TOYOTA_CHECKSUM){
          unsigned int chksm = toyota_checksum(address, ret, message_lookup[address].size);
          ret = set_value(ret, sig, chksm);
        } else {
          WARN("CHECKSUM signal type not valid\n");
        }
      }

      return ret;
    }



  private:
    const DBC *dbc = NULL;
    std::map<std::pair<uint32_t, std::string>, Signal> signal_lookup;
    std::map<uint32_t, Msg> message_lookup;
  };

}

extern "C" {
  void* canpack_init(const char* dbc_name) {
    CANPacker *ret = new CANPacker(std::string(dbc_name));
    return (void*)ret;
  }

  uint64_t canpack_pack(void* inst, uint32_t address, size_t num_vals, const SignalPackValue *vals, int counter, bool checksum) {
    CANPacker *cp = (CANPacker*)inst;

    return cp->pack(address, std::vector<SignalPackValue>(vals, vals+num_vals), counter);
  }

}
