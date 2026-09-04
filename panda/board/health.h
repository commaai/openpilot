#pragma once

#define HEALTH_FLAG_IGNITION_LINE            (1U << 0)
#define HEALTH_FLAG_IGNITION_CAN             (1U << 1)
#define HEALTH_FLAG_CONTROLS_ALLOWED         (1U << 2)
#define HEALTH_FLAG_POWER_SAVE_ENABLED       (1U << 3)
#define HEALTH_FLAG_HEARTBEAT_LOST           (1U << 4)
#define HEALTH_FLAG_SAFETY_RX_CHECKS_INVALID (1U << 5)
#define HEALTH_FLAG_SOM_RESET_TRIGGERED      (1U << 6)

struct __attribute__((packed)) health_t {
  uint32_t uptime_pkt;
  uint16_t voltage_pkt;
  uint16_t current_pkt;
  uint32_t safety_tx_blocked_pkt;
  uint32_t safety_rx_invalid_pkt;
  uint32_t tx_buffer_overflow_pkt;
  uint32_t rx_buffer_overflow_pkt;
  uint32_t faults_pkt;
  uint16_t flags_pkt;
  uint8_t car_harness_status_pkt;
  uint8_t safety_mode_pkt;
  uint16_t safety_param_pkt;
  uint8_t fault_status_pkt;
  uint16_t alternative_experience_pkt;
  uint8_t interrupt_load_pkt;
  uint8_t fan_power;
  uint16_t spi_error_count_pkt;
  uint16_t sbu1_voltage_mV;
  uint16_t sbu2_voltage_mV;
  uint16_t sound_output_level_pkt;
  uint8_t temperature_pkt;
};

typedef struct __attribute__((packed)) {
  uint8_t bus_off;
  uint32_t bus_off_cnt;
  uint8_t error_warning;
  uint8_t error_passive;
  uint8_t last_error; // real time LEC value
  uint8_t last_stored_error; // last LEC positive error code stored
  uint8_t last_data_error; // DLEC (for CANFD only)
  uint8_t last_data_stored_error; // last DLEC positive error code stored (for CANFD only)
  uint8_t receive_error_cnt; // Actual state of the receive error counter, values between 0 and 127. FDCAN_ECR.REC
  uint8_t transmit_error_cnt; // Actual state of the transmit error counter, values between 0 and 255. FDCAN_ECR.TEC
  uint32_t total_error_cnt; // How many times any error interrupt was invoked
  uint32_t total_tx_lost_cnt; // Tx event FIFO element lost
  uint32_t total_rx_lost_cnt; // Rx FIFO 0 message lost due to FIFO full condition
  uint32_t total_tx_cnt;
  uint32_t total_rx_cnt;
  uint32_t total_fwd_cnt; // Messages forwarded from one bus to another
  uint32_t total_tx_checksum_error_cnt;
  uint16_t can_speed;
  uint16_t can_data_speed;
  uint8_t canfd_enabled;
  uint8_t brs_enabled;
  uint8_t canfd_non_iso;
  uint32_t irq0_call_rate;
  uint32_t irq1_call_rate;
  uint32_t irq2_call_rate;
  uint32_t can_core_reset_cnt;
} can_health_t;
