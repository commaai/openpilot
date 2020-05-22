#include <vector>
#include <tuple>
#include <string>
#include "common/timing.h"
#include "messaging.hpp"
#include "cereal/gen/cpp/car.capnp.h"

typedef struct {
	long address;
	std::string dat;
	long busTime;
	long src;
} can_frame;

extern "C" {

void can_list_to_can_capnp_cpp(const std::vector<can_frame> &can_list, std::string &out, bool sendCan, bool valid) {
  MessageBuilder msg;
  auto canData = sendCan ? msg.initEvent(0, valid).initSendcan(can_list.size()) : msg.initEvent(0, valid).initCan(can_list.size());
  int j = 0;
  for (auto it = can_list.begin(); it != can_list.end(); it++, j++) {
    canData[j].setAddress(it->address);
    canData[j].setBusTime(it->busTime);
    canData[j].setDat(kj::arrayPtr((uint8_t*)it->dat.data(), it->dat.size()));
    canData[j].setSrc(it->src);
  }
  auto bytes = msg.toBytes();
  out.append((const char *)bytes.begin(), bytes.size());
}

}
