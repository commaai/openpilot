#pragma once

#include <cstdint>
#include <ctime>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/gen/cpp/log.capnp.h"
#include "panda/board/health.h"
#include "panda/board/can.h"
#include "selfdrive/pandad/panda_comms.h"

#define USB_TX_SOFT_LIMIT   (0x100U)
#define USBPACKET_MAX_SIZE  (0x40)

#define RECV_SIZE (0x4000U)

#define CAN_REJECTED_BUS_OFFSET   0xC0U
#define CAN_RETURNED_BUS_OFFSET 0x80U

#define PANDA_BUS_OFFSET 4

struct __attribute__((packed)) can_header {
  uint8_t reserved : 1;
  uint8_t bus : 3;
  uint8_t data_len_code : 4;
  uint8_t rejected : 1;
  uint8_t returned : 1;
  uint8_t extended : 1;
  uint32_t addr : 29;
  uint8_t checksum : 8;
};

struct can_frame {
  long address;
  std::string dat;
  long src;
};


class Panda {
private:
  std::unique_ptr<PandaCommsHandle> handle;

public:
  Panda(std::string serial="", uint32_t bus_offset=0);

  cereal::PandaState::PandaType hw_type = cereal::PandaState::PandaType::UNKNOWN;
  const uint32_t bus_offset;

  bool connected();
  bool comms_healthy();
  std::string hw_serial();

  // Static functions
  static std::vector<std::string> list(bool usb_only=false);

  // Panda functionality
  cereal::PandaState::PandaType get_hw_type();
  void set_safety_model(cereal::CarParams::SafetyModel safety_model, uint16_t safety_param=0U);
  void set_alternative_experience(uint16_t alternative_experience, uint16_t safety_param_sp=0U);
  std::string serial_read(int port_number = 0);
  void set_uart_baud(int uart, int rate);
  void set_fan_speed(uint16_t fan_speed);
  uint16_t get_fan_speed();
  void set_ir_pwr(uint16_t ir_pwr);
  std::optional<health_t> get_state();
  std::optional<can_health_t> get_can_state(uint16_t can_number);
  void set_loopback(bool loopback);
  std::optional<std::vector<uint8_t>> get_firmware_version();
  bool up_to_date();
  std::optional<std::string> get_serial();
  void set_power_saving(bool power_saving);
  void enable_deepsleep();
  void send_heartbeat(bool engaged, bool engaged_mads);
  void set_can_speed_kbps(uint16_t bus, uint16_t speed);
  void set_can_fd_auto(uint16_t bus, bool enabled);
  void set_data_speed_kbps(uint16_t bus, uint16_t speed);
  void set_canfd_non_iso(uint16_t bus, bool non_iso);
  void can_send(const capnp::List<cereal::CanData>::Reader &can_data_list);
  bool can_receive(std::vector<can_frame>& out_vec);
  void can_reset_communications();

protected:
  // for unit tests
  uint8_t receive_buffer[RECV_SIZE + sizeof(can_header) + 64];
  uint32_t receive_buffer_size = 0;

  Panda(uint32_t bus_offset) : bus_offset(bus_offset) {}
  void pack_can_buffer(const capnp::List<cereal::CanData>::Reader &can_data_list,
                         std::function<void(uint8_t *, size_t)> write_func);
  bool unpack_can_buffer(uint8_t *data, uint32_t &size, std::vector<can_frame> &out_vec);
  uint8_t calculate_checksum(uint8_t *data, uint32_t len);
};
