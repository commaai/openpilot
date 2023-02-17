// When changing these structs, python/__init__.py needs to be kept up to date!

#define HEALTH_PACKET_VERSION 11

struct __attribute__((packed)) health_t {
  uint32_t uptime_pkt;
  uint32_t voltage_pkt;
  uint32_t current_pkt;
  uint32_t safety_tx_blocked_pkt;
  uint32_t safety_rx_invalid_pkt;
  uint32_t tx_buffer_overflow_pkt;
  uint32_t rx_buffer_overflow_pkt;
  uint32_t gmlan_send_errs_pkt;
  uint32_t faults_pkt;
  uint8_t ignition_line_pkt;
  uint8_t ignition_can_pkt;
  uint8_t controls_allowed_pkt;
  uint8_t gas_interceptor_detected_pkt;
  uint8_t car_harness_status_pkt;
  uint8_t safety_mode_pkt;
  uint16_t safety_param_pkt;
  uint8_t fault_status_pkt;
  uint8_t power_save_enabled_pkt;
  uint8_t heartbeat_lost_pkt;
  uint16_t alternative_experience_pkt;
  float interrupt_load;
  uint8_t fan_power;
  uint8_t safety_rx_checks_invalid;
};

#define CAN_HEALTH_PACKET_VERSION 4
typedef struct __attribute__((packed)) {
  uint8_t bus_off;
  uint32_t bus_off_cnt;
  uint8_t error_warning;
  uint8_t error_passive;
  uint8_t last_error; // real time LEC value
  uint8_t last_stored_error; // last LEC positive error code stored
  uint8_t last_data_error; // DLEC (for CANFD only)
  uint8_t last_data_stored_error; // last DLEC positive error code stored (for CANFD only)
  uint8_t receive_error_cnt; // REC
  uint8_t transmit_error_cnt; // TEC
  uint32_t total_error_cnt; // How many times any error interrupt were invoked
  uint32_t total_tx_lost_cnt; // Tx event FIFO element Lost
  uint32_t total_rx_lost_cnt; // Rx FIFO 0 message Lost
  uint32_t total_tx_cnt;
  uint32_t total_rx_cnt;
  uint32_t total_fwd_cnt; // Messages forwarded from one bus to another
  uint32_t total_tx_checksum_error_cnt;
  uint16_t can_speed;
  uint16_t can_data_speed;
  uint8_t canfd_enabled;
  uint8_t brs_enabled;
  uint8_t canfd_non_iso;
} can_health_t;
