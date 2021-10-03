#pragma once

#include <optional>

#include "cereal/messaging/messaging.h"
#include "selfdrive/boardd/panda_comm.h"

// double the FIFO size
#define RECV_SIZE (0x1000)

// copied from panda/board/main.c
struct __attribute__((packed)) health_t {
  uint32_t uptime;
  uint32_t voltage;
  uint32_t current;
  uint32_t can_rx_errs;
  uint32_t can_send_errs;
  uint32_t can_fwd_errs;
  uint32_t gmlan_send_errs;
  uint32_t faults;
  uint8_t ignition_line;
  uint8_t ignition_can;
  uint8_t controls_allowed;
  uint8_t gas_interceptor_detected;
  uint8_t car_harness_status;
  uint8_t usb_power_mode;
  uint8_t safety_model;
  int16_t safety_param;
  uint8_t fault_status;
  uint8_t power_save_enabled;
  uint8_t heartbeat_lost;
};

class Panda : public PandaComm {
public:
  Panda(std::string serial = {});
  ~Panda();

  // std::string usb_serial;
  cereal::PandaState::PandaType hw_type = cereal::PandaState::PandaType::UNKNOWN;
  bool has_rtc = false;

  // Panda functionality
  cereal::PandaState::PandaType get_hw_type();
  void set_safety_model(cereal::CarParams::SafetyModel safety_model, int safety_param=0);
  void set_unsafe_mode(uint16_t unsafe_mode);
  void set_rtc(struct tm sys_time);
  struct tm get_rtc();
  void set_fan_speed(uint16_t fan_speed);
  uint16_t get_fan_speed();
  void set_ir_pwr(uint16_t ir_pwr);
  health_t get_state();
  void set_loopback(bool loopback);
  std::optional<std::vector<uint8_t>> get_firmware_version();
  std::optional<std::string> get_serial();
  void set_power_saving(bool power_saving);
  void set_usb_power_mode(cereal::PeripheralState::UsbPowerMode power_mode);
  void send_heartbeat();
  void can_send(capnp::List<cereal::CanData>::Reader can_data_list);
  int can_receive(kj::Array<capnp::word>& out_buf);
};
