#include <cstring>
#include <vector>

#include <string>

#include "selfdrive/pandad/pandad.h"
#include "selfdrive/pandad/can_types.h"
#include "cereal/messaging/messaging.h"

extern void can_list_to_can_capnp_cpp(const std::vector<CanFrame> &can_list, std::string &out, bool sendcan, bool valid);
extern void can_capnp_to_can_list_cpp(const std::vector<std::string> &strings, std::vector<CanData> &can_list, bool sendcan);

extern "C" {

typedef struct {
  long address;
  const uint8_t* dat;
  size_t dat_len;
  long src;
} CanFrame_C;

typedef struct {
  uint64_t nanos;
  CanFrame_C* frames;
  size_t frames_len;
} CanData_C;

typedef struct {
  uint16_t voltage_pkt;
  uint16_t current_pkt;
  uint32_t uptime_pkt;
  uint32_t rx_buffer_overflow_pkt;
  uint32_t tx_buffer_overflow_pkt;
  uint32_t faults_pkt;
  uint8_t ignition_line_pkt;
  uint8_t ignition_can_pkt;
  uint8_t controls_allowed_pkt;
  uint8_t safety_mode_pkt;
  uint16_t safety_param_pkt;
  uint8_t fault_status_pkt;
  uint8_t power_save_enabled_pkt;
  uint8_t heartbeat_lost_pkt;
  uint16_t alternative_experience_pkt;
  uint8_t car_harness_status_pkt;
  uint8_t safety_tx_blocked_pkt;
  uint8_t safety_rx_invalid_pkt;
  uint8_t safety_rx_checks_invalid_pkt;
  uint32_t interrupt_load_pkt;
  uint16_t fan_power;
  uint32_t spi_error_count_pkt;
  uint16_t sbu1_voltage_mV;
  uint16_t sbu2_voltage_mV;
} PandaHealth_C;

} // extern C

// Helper to convert C array to std::vector<CanFrame> (C++)
std::vector<CanFrame> c_to_cpp_frames(const CanFrame_C* frames, size_t len) {
  std::vector<CanFrame> out;
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    CanFrame f;
    f.address = frames[i].address;
    f.dat.assign(frames[i].dat, frames[i].dat + frames[i].dat_len);
    f.src = frames[i].src;
    out.push_back(f);
  }
  return out;
}

extern "C" {


Panda* panda_create(const char* serial, uint32_t bus_offset) {
  try {
    return new Panda(serial ? std::string(serial) : "", bus_offset);
  } catch (...) {
    return nullptr;
  }
}

void panda_delete(Panda* p) {
  delete p;
}

bool panda_connected(Panda* p) {
  return p->connected();
}

bool panda_comms_healthy(Panda* p) {
  return p->comms_healthy();
}

const char* panda_get_serial(Panda* p) {
  std::string s = p->hw_serial();
  char* res = (char*)malloc(s.size() + 1);
  strcpy(res, s.c_str());
  return res;
}

void panda_free_str(char* s) {
  free(s);
}

void panda_set_safety_model(Panda* p, int safety_mode, uint16_t safety_param) {
  p->set_safety_model((cereal::CarParams::SafetyModel)safety_mode, safety_param);
}

void panda_set_alternative_experience(Panda* p, uint16_t alt_exp) {
  p->set_alternative_experience(alt_exp);
}



void panda_set_fan_speed(Panda* p, uint16_t speed) {
  p->set_fan_speed(speed);
}

uint16_t panda_get_fan_speed(Panda* p) {
  return p->get_fan_speed();
}

void panda_set_ir_pwr(Panda* p, uint16_t pwr) {
  p->set_ir_pwr(pwr);
}

bool panda_get_state(Panda* p, PandaHealth_C* out) {
  auto health_opt = p->get_state();
  if (!health_opt) return false;
  health_t h = *health_opt;

  out->voltage_pkt = h.voltage_pkt;
  out->current_pkt = h.current_pkt;
  out->uptime_pkt = h.uptime_pkt;
  out->rx_buffer_overflow_pkt = h.rx_buffer_overflow_pkt;
  out->tx_buffer_overflow_pkt = h.tx_buffer_overflow_pkt;
  out->faults_pkt = h.faults_pkt;
  out->ignition_line_pkt = h.ignition_line_pkt;
  out->ignition_can_pkt = h.ignition_can_pkt;
  out->controls_allowed_pkt = h.controls_allowed_pkt;
  out->safety_mode_pkt = h.safety_mode_pkt;
  out->safety_param_pkt = h.safety_param_pkt;
  out->fault_status_pkt = h.fault_status_pkt;
  out->power_save_enabled_pkt = h.power_save_enabled_pkt;
  out->heartbeat_lost_pkt = h.heartbeat_lost_pkt;
  out->alternative_experience_pkt = h.alternative_experience_pkt;
  out->car_harness_status_pkt = h.car_harness_status_pkt;
  out->safety_tx_blocked_pkt = h.safety_tx_blocked_pkt;
  out->safety_rx_invalid_pkt = h.safety_rx_invalid_pkt;
  out->safety_rx_checks_invalid_pkt = h.safety_rx_checks_invalid_pkt;
  out->interrupt_load_pkt = h.interrupt_load_pkt;
  out->fan_power = h.fan_power;
  out->spi_error_count_pkt = h.spi_error_count_pkt;
  out->sbu1_voltage_mV = h.sbu1_voltage_mV;
  out->sbu2_voltage_mV = h.sbu2_voltage_mV;

  return true;
}

void panda_set_loopback(Panda* p, bool loopback) {
  p->set_loopback(loopback);
}

void panda_set_power_saving(Panda* p, bool enable) {
  p->set_power_saving(enable);
}

void panda_send_heartbeat(Panda* p, bool engaged) {
  p->send_heartbeat(engaged);
}

uint16_t panda_get_type(Panda* p) {
  return (uint16_t)p->get_hw_type();
}

void panda_can_send(Panda* p, const CanFrame_C* frames, size_t len) {
  std::vector<CanFrame> cpp_frames = c_to_cpp_frames(frames, len);

  // Create a capnp message
  MessageBuilder msg;
  auto event = msg.initEvent();
  auto canData = event.initCan(len);

  for (size_t i = 0; i < len; ++i) {
    canData[i].setAddress(cpp_frames[i].address);
    canData[i].setDat(kj::arrayPtr(cpp_frames[i].dat.data(), cpp_frames[i].dat.size()));
    canData[i].setSrc(cpp_frames[i].src);
  }

  p->can_send(canData.asReader());
}

// Helper for receiving CAN frames
typedef struct {
  long address;
  uint8_t dat[64];
  size_t dat_len;
  long src;
} CanFrame_Flat;

int panda_can_receive(Panda* p, CanFrame_Flat* out_frames, size_t max_len) {
  std::vector<can_frame> raw_frames;
  if (!p->can_receive(raw_frames)) return -1; // Comms error

  size_t cnt = std::min(max_len, raw_frames.size());
  for (size_t i = 0; i < cnt; ++i) {
    out_frames[i].address = raw_frames[i].address;
    out_frames[i].src = raw_frames[i].src;
    size_t dlen = std::min(sizeof(out_frames[i].dat), raw_frames[i].dat.size());
    out_frames[i].dat_len = dlen;
    memcpy(out_frames[i].dat, raw_frames[i].dat.data(), dlen);
  }
  return (int)cnt; // Return number of frames read
}

// --- Helpers for can_list_to_can_capnp ---

void* can_list_to_capnp(const CanFrame_C* frames, size_t len, bool sendcan, bool valid, size_t* out_len) {
  std::vector<CanFrame> cpp_frames = c_to_cpp_frames(frames, len);
  std::string out;
  can_list_to_can_capnp_cpp(cpp_frames, out, sendcan, valid);

  *out_len = out.size();
  void* res = malloc(out.size());
  memcpy(res, out.data(), out.size());
  return res;
}

// capnp to list
// Helpers for can_capnp_to_list

void* can_capnp_to_list_create(const char* data, size_t len, bool sendcan) {
  std::vector<std::string> strings;
  strings.emplace_back(data, len);
  auto* ret = new std::vector<CanData>();
  can_capnp_to_can_list_cpp(strings, *ret, sendcan);
  return (void*)ret;
}

size_t can_capnp_handler_size(void* handler) {
  auto* v = (std::vector<CanData>*)handler;
  return v->size();
}

uint64_t can_capnp_handler_get_nanos(void* handler, size_t idx) {
  auto* v = (std::vector<CanData>*)handler;
  return (*v)[idx].nanos;
}

size_t can_capnp_handler_get_frame_count(void* handler, size_t idx) {
  auto* v = (std::vector<CanData>*)handler;
  return (*v)[idx].frames.size();
}

void can_capnp_handler_get_frame(void* handler, size_t idx, size_t frame_idx, CanFrame_Flat* out) {
  auto* v = (std::vector<CanData>*)handler;
  const auto& f = (*v)[idx].frames[frame_idx];
  out->address = f.address;
  out->src = f.src;
  size_t dlen = std::min(sizeof(out->dat), f.dat.size());
  out->dat_len = dlen;
  memcpy(out->dat, f.dat.data(), dlen);
}

void can_capnp_handler_free(void* handler) {
  delete (std::vector<CanData>*)handler;
}

} // extern C
