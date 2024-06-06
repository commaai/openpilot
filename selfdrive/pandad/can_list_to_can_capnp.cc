#include "cereal/messaging/messaging.h"
#include "selfdrive/pandad/panda.h"

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
