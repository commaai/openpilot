#include "cereal/messaging/messaging.h"
#include "selfdrive/pandad/panda.h"
#include "opendbc/can/common.h"

void can_list_to_can_capnp_cpp(const std::vector<can_frame> &can_list, std::string &out, bool sendCan, bool valid) {
  MessageBuilder msg;
  auto event = msg.initEvent(valid);

  auto canData = sendCan ? event.initSendcan(can_list.size()) : event.initCan(can_list.size());
  int j = 0;
  for (auto it = can_list.begin(); it != can_list.end(); it++, j++) {
    auto c = canData[j];
    c.setAddress(it->address);
    c.setBusTime(it->busTime);
    c.setDat(kj::arrayPtr((uint8_t*)it->dat.data(), it->dat.size()));
    c.setSrc(it->src);
  }
  const uint64_t msg_size = capnp::computeSerializedSizeInWords(msg) * sizeof(capnp::word);
  out.resize(msg_size);
  kj::ArrayOutputStream output_stream(kj::ArrayPtr<capnp::byte>((unsigned char *)out.data(), msg_size));
  capnp::writeMessage(output_stream, msg);
}

void can_capnp_to_can_list_cpp(const std::vector<std::string> &strings, std::vector<CanData> &can_data, bool sendcan) {
  kj::Array<capnp::word> aligned_buf;
  can_data.reserve(strings.size());

  for (const auto &s : strings) {
    const size_t buf_size = (s.length() / sizeof(capnp::word)) + 1;
    if (aligned_buf.size() < buf_size) {
      aligned_buf = kj::heapArray<capnp::word>(buf_size);
    }
    memcpy(aligned_buf.begin(), s.data(), s.length());

    // extract the messages
    capnp::FlatArrayMessageReader cmsg(aligned_buf.slice(0, buf_size));
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    auto &can = can_data.emplace_back();
    can.nanos = event.getLogMonoTime();

    auto cans = sendcan ? event.getSendcan() : event.getCan();
    can.frames.reserve(cans.size());
    for (const auto &c : cans) {
      auto &frame = can.frames.emplace_back();
      frame.src = c.getSrc();
      frame.address = c.getAddress();
      auto dat = c.getDat();
      frame.dat.assign(dat.begin(), dat.end());
    }
  }
}
