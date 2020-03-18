#include <cassert>
#include <utility>
#include <algorithm>
#include <map>
#include <cmath>

#include "common.h"

#define WARN printf

// this is the same as read_u64_le, but uses uint64_t as in/out
uint64_t ReverseBytes(uint64_t x) {
  return ((x & 0xff00000000000000ull) >> 56) |
          ((x & 0x00ff000000000000ull) >> 40) |
          ((x & 0x0000ff0000000000ull) >> 24) |
          ((x & 0x000000ff00000000ull) >> 8) |
          ((x & 0x00000000ff000000ull) << 8) |
          ((x & 0x0000000000ff0000ull) << 24) |
          ((x & 0x000000000000ff00ull) << 40) |
          ((x & 0x00000000000000ffull) << 56);
}

uint64_t set_value(uint64_t ret, Signal sig, int64_t ival){
  int shift = sig.is_little_endian? sig.b1 : sig.bo;
  uint64_t mask = ((1ULL << sig.b2)-1) << shift;
  uint64_t dat = (ival & ((1ULL << sig.b2)-1)) << shift;
  if (sig.is_little_endian) {
    dat = ReverseBytes(dat);
    mask = ReverseBytes(mask);
  }
  ret &= ~mask;
  ret |= dat;
  return ret;
}

CANPacker::CANPacker(const std::string& dbc_name) {
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
  init_crc_lookup_tables();
}

uint64_t CANPacker::pack(uint32_t address, const std::vector<SignalPackValue> &signals, int counter) {
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

    if ((sig.type != SignalType::HONDA_COUNTER) && (sig.type != SignalType::VOLKSWAGEN_COUNTER)) {
      WARN("COUNTER signal type not valid\n");
    }

    ret = set_value(ret, sig, counter);
  }

  auto sig_it_checksum = signal_lookup.find(std::make_pair(address, "CHECKSUM"));
  if (sig_it_checksum != signal_lookup.end()) {
    auto sig = sig_it_checksum->second;
    if (sig.type == SignalType::HONDA_CHECKSUM) {
      unsigned int chksm = honda_checksum(address, ret, message_lookup[address].size);
      ret = set_value(ret, sig, chksm);
    } else if (sig.type == SignalType::TOYOTA_CHECKSUM) {
      unsigned int chksm = toyota_checksum(address, ret, message_lookup[address].size);
      ret = set_value(ret, sig, chksm);
    } else if (sig.type == SignalType::VOLKSWAGEN_CHECKSUM) {
      // FIXME: Hackish fix for an endianness issue. The message is in reverse byte order
      // until later in the pack process. Checksums can be run backwards, CRCs not so much.
      // The correct fix is unclear but this works for the moment.
      unsigned int chksm = volkswagen_crc(address, ReverseBytes(ret), message_lookup[address].size);
      ret = set_value(ret, sig, chksm);
    } else {
      //WARN("CHECKSUM signal type not valid\n");
    }
  }

  return ret;
}
