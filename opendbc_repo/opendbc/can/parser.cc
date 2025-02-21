#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <sstream>

#include "opendbc/can/common.h"

int64_t get_raw_value(const std::vector<uint8_t> &msg, const Signal &sig) {
  int64_t ret = 0;

  int i = sig.msb / 8;
  int bits = sig.size;
  while (i >= 0 && i < msg.size() && bits > 0) {
    int lsb = (int)(sig.lsb / 8) == i ? sig.lsb : i*8;
    int msb = (int)(sig.msb / 8) == i ? sig.msb : (i+1)*8 - 1;
    int size = msb - lsb + 1;

    uint64_t d = (msg[i] >> (lsb - (i*8))) & ((1ULL << size) - 1);
    ret |= d << (bits - size);

    bits -= size;
    i = sig.is_little_endian ? i-1 : i+1;
  }
  return ret;
}


bool MessageState::parse(uint64_t nanos, const std::vector<uint8_t> &dat) {
  std::vector<double> tmp_vals(parse_sigs.size());
  bool checksum_failed = false;
  bool counter_failed = false;

  for (int i = 0; i < parse_sigs.size(); i++) {
    const auto &sig = parse_sigs[i];

    int64_t tmp = get_raw_value(dat, sig);
    if (sig.is_signed) {
      tmp -= ((tmp >> (sig.size-1)) & 0x1) ? (1ULL << sig.size) : 0;
    }

    //DEBUG("parse 0x%X %s -> %ld\n", address, sig.name, tmp);

    if (!ignore_checksum) {
      if (sig.calc_checksum != nullptr && sig.calc_checksum(address, sig, dat) != tmp) {
        checksum_failed = true;
      }
    }

    if (!ignore_counter) {
      if (sig.type == SignalType::COUNTER && !update_counter_generic(tmp, sig.size)) {
        counter_failed = true;
      }
    }

    tmp_vals[i] = tmp * sig.factor + sig.offset;
  }

  // only update values if both checksum and counter are valid
  if (checksum_failed || counter_failed) {
    LOGE_100("0x%X message checks failed, checksum failed %d, counter failed %d", address, checksum_failed, counter_failed);
    return false;
  }

  for (int i = 0; i < parse_sigs.size(); i++) {
    vals[i] = tmp_vals[i];
    all_vals[i].push_back(vals[i]);
  }
  last_seen_nanos = nanos;

  return true;
}


bool MessageState::update_counter_generic(int64_t v, int cnt_size) {
  if (((counter + 1) & ((1 << cnt_size) -1)) != v) {
    counter_fail = std::min(counter_fail + 1, MAX_BAD_COUNTER);
    if (counter_fail > 1) {
      INFO("0x%X COUNTER FAIL #%d -- %d -> %d\n", address, counter_fail, counter, (int)v);
    }
  } else if (counter_fail > 0) {
    counter_fail--;
  }
  counter = v;
  return counter_fail < MAX_BAD_COUNTER;
}


CANParser::CANParser(int abus, const std::string& dbc_name, const std::vector<std::pair<uint32_t, int>> &messages)
  : bus(abus) {
  dbc = dbc_lookup(dbc_name);
  assert(dbc);

  bus_timeout_threshold = std::numeric_limits<uint64_t>::max();

  for (const auto& [address, frequency] : messages) {
    // disallow duplicate message checks
    if (message_states.find(address) != message_states.end()) {
      std::stringstream is;
      is << "Duplicate Message Check: " << address;
      throw std::runtime_error(is.str());
    }

    MessageState &state = message_states[address];
    state.address = address;
    // state.check_frequency = op.check_frequency,

    // msg is not valid if a message isn't received for 10 consecutive steps
    if (frequency > 0) {
      state.check_threshold = (1000000000ULL / frequency) * 10;

      // bus timeout threshold should be 10x the fastest msg
      bus_timeout_threshold = std::min(bus_timeout_threshold, state.check_threshold);
    }

    const Msg *msg = dbc->addr_to_msg.at(address);
    state.name = msg->name;
    state.size = msg->size;
    assert(state.size <= 64);  // max signal size is 64 bytes

    // track all signals for this message
    state.parse_sigs = msg->sigs;
    state.vals.resize(msg->sigs.size());
    state.all_vals.resize(msg->sigs.size());
  }
}

CANParser::CANParser(int abus, const std::string& dbc_name, bool ignore_checksum, bool ignore_counter)
  : bus(abus) {
  // Add all messages and signals

  dbc = dbc_lookup(dbc_name);
  assert(dbc);

  for (const auto& msg : dbc->msgs) {
    MessageState state = {
      .name = msg.name,
      .address = msg.address,
      .size = msg.size,
      .ignore_checksum = ignore_checksum,
      .ignore_counter = ignore_counter,
    };

    for (const auto& sig : msg.sigs) {
      state.parse_sigs.push_back(sig);
      state.vals.push_back(0);
      state.all_vals.push_back({});
    }

    message_states[state.address] = state;
  }
}

std::set<uint32_t> CANParser::update(const std::vector<CanData> &can_data) {
  // Clear all_values
  for (auto &state : message_states) {
    for (auto &vals : state.second.all_vals) vals.clear();
  }

  std::set<uint32_t> updated_addresses;
  for (const auto &c : can_data) {
    if (first_nanos == 0) {
      first_nanos = c.nanos;
    }

    UpdateCans(c, updated_addresses);
    UpdateValid(c.nanos);
  }
  return updated_addresses;
}

void CANParser::UpdateCans(const CanData &can, std::set<uint32_t> &updated_addresses) {
  //DEBUG("got %zu messages\n", can.frames.size());

  bool bus_empty = true;

  for (const auto &frame : can.frames) {
    if (frame.src != bus) {
      // DEBUG("skip %d: wrong bus\n", cmsg.getAddress());
      continue;
    }
    bus_empty = false;

    auto state_it = message_states.find(frame.address);
    if (state_it == message_states.end()) {
      // DEBUG("skip %d: not specified\n", cmsg.getAddress());
      continue;
    }
    if (frame.dat.size() > 64) {
      DEBUG("got message longer than 64 bytes: 0x%X %zu\n", frame.address, frame.dat.size());
      continue;
    }

    // TODO: this actually triggers for some cars. fix and enable this
    //if (dat.size() != state_it->second.size) {
    //  DEBUG("got message with unexpected length: expected %d, got %zu for %d", state_it->second.size, dat.size(), cmsg.getAddress());
    //  continue;
    //}

    if (state_it->second.parse(can.nanos, frame.dat)) {
      updated_addresses.insert(state_it->first);
    }
  }

  // update bus timeout
  if (!bus_empty) {
    last_nonempty_nanos = can.nanos;
  }
  bus_timeout = (can.nanos - last_nonempty_nanos) > bus_timeout_threshold;
}

void CANParser::UpdateValid(uint64_t nanos) {
  const bool show_missing = (nanos - first_nanos) > 8e9;

  bool _valid = true;
  bool _counters_valid = true;
  for (const auto& kv : message_states) {
    const auto& state = kv.second;

    if (state.counter_fail >= MAX_BAD_COUNTER) {
      _counters_valid = false;
    }

    const bool missing = state.last_seen_nanos == 0;
    const bool timed_out = (nanos - state.last_seen_nanos) > state.check_threshold;
    if (state.check_threshold > 0 && (missing || timed_out)) {
      if (show_missing && !bus_timeout) {
        if (missing) {
          LOGE_100("0x%X '%s' NOT SEEN", state.address, state.name.c_str());
        } else if (timed_out) {
          LOGE_100("0x%X '%s' TIMED OUT", state.address, state.name.c_str());
        }
      }
      _valid = false;
    }
  }
  can_invalid_cnt = _valid ? 0 : (can_invalid_cnt + 1);
  can_valid = (can_invalid_cnt < CAN_INVALID_CNT) && _counters_valid;
}
