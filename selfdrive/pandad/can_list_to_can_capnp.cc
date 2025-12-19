#include "cereal/messaging/messaging.h"
#include "selfdrive/pandad/can_types.h"

void can_list_to_can_capnp_cpp(const std::vector<CanFrame> &can_list, std::string &out, bool sendcan, bool valid) {
  MessageBuilder msg;
  auto event = msg.initEvent(valid);

  auto canData = sendcan ? event.initSendcan(can_list.size()) : event.initCan(can_list.size());
  int j = 0;
  for (auto it = can_list.begin(); it != can_list.end(); it++, j++) {
    auto c = canData[j];
    c.setAddress(it->address);
    c.setDat(kj::arrayPtr((uint8_t*)it->dat.data(), it->dat.size()));
    c.setSrc(it->src);
  }
  const uint64_t msg_size = capnp::computeSerializedSizeInWords(msg) * sizeof(capnp::word);
  out.resize(msg_size);
  kj::ArrayOutputStream output_stream(kj::ArrayPtr<capnp::byte>((unsigned char *)out.data(), msg_size));
  capnp::writeMessage(output_stream, msg);
}

// Converts a vector of Cap'n Proto serialized can strings into a vector of CanData structures.
void can_capnp_to_can_list_cpp(const std::vector<std::string> &strings, std::vector<CanData> &can_list, bool sendcan) {
  AlignedBuffer aligned_buf;
  can_list.reserve(strings.size());

  for (const auto &str : strings) {
    // extract the messages
    capnp::FlatArrayMessageReader reader(aligned_buf.align(str.data(), str.size()));
    cereal::Event::Reader event = reader.getRoot<cereal::Event>();

    auto frames = sendcan ? event.getSendcan() : event.getCan();

    // Add new CanData entry
    CanData &can_data = can_list.emplace_back();
    can_data.nanos = event.getLogMonoTime();
    can_data.frames.reserve(frames.size());

    // Populate CAN frames
    for (const auto &frame : frames) {
      CanFrame &can_frame = can_data.frames.emplace_back();
      can_frame.src = frame.getSrc();
      can_frame.address = frame.getAddress();

      // Copy CAN data
      auto dat = frame.getDat();
      can_frame.dat.assign(dat.begin(), dat.end());
    }
  }
}
