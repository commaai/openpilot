#include <cassert>
#include <cstring>

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <algorithm>

#include "common.h"

#define DEBUG(...)
// #define DEBUG printf
#define INFO printf


bool MessageState::parse(uint64_t sec, uint16_t ts_, uint8_t * dat) {
  uint64_t dat_le = read_u64_le(dat);
  uint64_t dat_be = read_u64_be(dat);

  for (int i=0; i < parse_sigs.size(); i++) {
    auto& sig = parse_sigs[i];
    int64_t tmp;

    if (sig.is_little_endian){
      tmp = (dat_le >> sig.b1) & ((1ULL << sig.b2)-1);
    } else {
      tmp = (dat_be >> sig.bo) & ((1ULL << sig.b2)-1);
    }

    if (sig.is_signed) {
      tmp -= (tmp >> (sig.b2-1)) ? (1ULL << sig.b2) : 0; //signed
    }

    DEBUG("parse 0x%X %s -> %lld\n", address, sig.name, tmp);

    if (sig.type == SignalType::HONDA_CHECKSUM) {
      if (honda_checksum(address, dat_be, size) != tmp) {
        INFO("0x%X CHECKSUM FAIL\n", address);
        return false;
      }
    } else if (sig.type == SignalType::HONDA_COUNTER) {
      if (!update_counter_generic(tmp, sig.b2)) {
        return false;
      }
    } else if (sig.type == SignalType::TOYOTA_CHECKSUM) {
      if (toyota_checksum(address, dat_be, size) != tmp) {
        INFO("0x%X CHECKSUM FAIL\n", address);
        return false;
      }
    } else if (sig.type == SignalType::VOLKSWAGEN_CHECKSUM) {
      if (volkswagen_crc(address, dat_le, size) != tmp) {
        INFO("0x%X CRC FAIL\n", address);
        return false;
      }
    } else if (sig.type == SignalType::VOLKSWAGEN_COUNTER) {
        if (!update_counter_generic(tmp, sig.b2)) {
        return false;
      }
    } else if (sig.type == SignalType::PEDAL_CHECKSUM) {
      if (pedal_checksum(dat_be, size) != tmp) {
        INFO("0x%X PEDAL CHECKSUM FAIL\n", address);
        return false;
      }
    } else if (sig.type == SignalType::PEDAL_COUNTER) {
      if (!update_counter_generic(tmp, sig.b2)) {
        return false;
      }
    }

    vals[i] = tmp * sig.factor + sig.offset;
  }
  ts = ts_;
  seen = sec;

  return true;
}


bool MessageState::update_counter_generic(int64_t v, int cnt_size) {
  uint8_t old_counter = counter;
  counter = v;
  if (((old_counter+1) & ((1 << cnt_size) -1)) != v) {
    counter_fail += 1;
    if (counter_fail > 1) {
      INFO("0x%X COUNTER FAIL %d -- %d vs %d\n", address, counter_fail, old_counter, (int)v);
    }
    if (counter_fail >= MAX_BAD_COUNTER) {
      return false;
    }
  } else if (counter_fail > 0) {
    counter_fail--;
  }
  return true;
}


CANParser::CANParser(int abus, const std::string& dbc_name,
          const std::vector<MessageParseOptions> &options,
          const std::vector<SignalParseOptions> &sigoptions)
  : bus(abus) {

  dbc = dbc_lookup(dbc_name);
  assert(dbc);
  init_crc_lookup_tables();

  for (const auto& op : options) {
    MessageState state = {
      .address = op.address,
      // .check_frequency = op.check_frequency,
    };

    // msg is not valid if a message isn't received for 10 consecutive steps
    if (op.check_frequency > 0) {
      state.check_threshold = (1000000000ULL / op.check_frequency) * 10;
    }


    const Msg* msg = NULL;
    for (int i=0; i<dbc->num_msgs; i++) {
      if (dbc->msgs[i].address == op.address) {
        msg = &dbc->msgs[i];
        break;
      }
    }
    if (!msg) {
      fprintf(stderr, "CANParser: could not find message 0x%X in DBC %s\n", op.address, dbc_name.c_str());
      assert(false);
    }

    state.size = msg->size;

    // track checksums and counters for this message
    for (int i=0; i<msg->num_sigs; i++) {
      const Signal *sig = &msg->sigs[i];
      if (sig->type != SignalType::DEFAULT) {
        state.parse_sigs.push_back(*sig);
        state.vals.push_back(0);
      }
    }

    // track requested signals for this message
    for (const auto& sigop : sigoptions) {
      if (sigop.address != op.address) continue;

      for (int i=0; i<msg->num_sigs; i++) {
        const Signal *sig = &msg->sigs[i];
        if (strcmp(sig->name, sigop.name) == 0
            && sig->type == SignalType::DEFAULT) {
          state.parse_sigs.push_back(*sig);
          state.vals.push_back(sigop.default_value);
          break;
        }
      }

    }

    message_states[state.address] = state;
  }
}

void CANParser::UpdateCans(uint64_t sec, const capnp::List<cereal::CanData>::Reader& cans) {
    int msg_count = cans.size();
    uint64_t p;

    DEBUG("got %d messages\n", msg_count);

    // parse the messages
    for (int i = 0; i < msg_count; i++) {
      auto cmsg = cans[i];
      if (cmsg.getSrc() != bus) {
        // DEBUG("skip %d: wrong bus\n", cmsg.getAddress());
        continue;
      }
      auto state_it = message_states.find(cmsg.getAddress());
      if (state_it == message_states.end()) {
        // DEBUG("skip %d: not specified\n", cmsg.getAddress());
        continue;
      }

      if (cmsg.getDat().size() > 8) continue; //shouldnt ever happen
      uint8_t dat[8] = {0};
      memcpy(dat, cmsg.getDat().begin(), cmsg.getDat().size());

      state_it->second.parse(sec, cmsg.getBusTime(), dat);
    }
}

void CANParser::UpdateValid(uint64_t sec) {
  can_valid = true;
  for (const auto& kv : message_states) {
    const auto& state = kv.second;
    if (state.check_threshold > 0 && (sec - state.seen) > state.check_threshold) {
      if (state.seen > 0) {
        DEBUG("0x%X TIMEOUT\n", state.address);
      }
      can_valid = false;
    }
  }
}

void CANParser::update_string(std::string data) {
  // format for board, make copy due to alignment issues, will be freed on out of scope
  auto amsg = kj::heapArray<capnp::word>((data.length() / sizeof(capnp::word)) + 1);
  memcpy(amsg.begin(), data.data(), data.length());

  // extract the messages
  capnp::FlatArrayMessageReader cmsg(amsg);
  cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

  last_sec = event.getLogMonoTime();

  auto cans = event.getCan();
  UpdateCans(last_sec, cans);

  UpdateValid(last_sec);
}


std::vector<SignalValue> CANParser::query_latest() {
  std::vector<SignalValue> ret;

  for (const auto& kv : message_states) {
    const auto& state = kv.second;
    if (last_sec != 0 && state.seen != last_sec) continue;

    for (int i=0; i<state.parse_sigs.size(); i++) {
      const Signal &sig = state.parse_sigs[i];
      ret.push_back((SignalValue){
        .address = state.address,
        .ts = state.ts,
        .name = sig.name,
        .value = state.vals[i],
      });
    }
  }

  return ret;
}
