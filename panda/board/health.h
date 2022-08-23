// When changing this struct, python/__init__.py needs to be kept up to date!
#define HEALTH_PACKET_VERSION 8
struct __attribute__((packed)) health_t {
  uint32_t uptime_pkt;
  uint32_t voltage_pkt;
  uint32_t current_pkt;
  uint32_t can_rx_errs_pkt;
  uint32_t can_send_errs_pkt;
  uint32_t can_fwd_errs_pkt;
  uint32_t gmlan_send_errs_pkt;
  uint32_t faults_pkt;
  uint8_t ignition_line_pkt;
  uint8_t ignition_can_pkt;
  uint8_t controls_allowed_pkt;
  uint8_t gas_interceptor_detected_pkt;
  uint8_t car_harness_status_pkt;
  uint8_t usb_power_mode_pkt;
  uint8_t safety_mode_pkt;
  uint16_t safety_param_pkt;
  uint8_t fault_status_pkt;
  uint8_t power_save_enabled_pkt;
  uint8_t heartbeat_lost_pkt;
  uint16_t alternative_experience_pkt;
  uint32_t blocked_msg_cnt_pkt;
  float interrupt_load;
  uint8_t fan_power;
};
