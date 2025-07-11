// When changing these structs, python/__init__.py needs to be kept up to date!

#define JUNGLE_HEALTH_PACKET_VERSION 1
struct __attribute__((packed)) jungle_health_t {
  uint32_t uptime_pkt;
  float ch1_power;
  float ch2_power;
  float ch3_power;
  float ch4_power;
  float ch5_power;
  float ch6_power;
  uint16_t ch1_sbu1_mV;
  uint16_t ch1_sbu2_mV;
  uint16_t ch2_sbu1_mV;
  uint16_t ch2_sbu2_mV;
  uint16_t ch3_sbu1_mV;
  uint16_t ch3_sbu2_mV;
  uint16_t ch4_sbu1_mV;
  uint16_t ch4_sbu2_mV;
  uint16_t ch5_sbu1_mV;
  uint16_t ch5_sbu2_mV;
  uint16_t ch6_sbu1_mV;
  uint16_t ch6_sbu2_mV;
};
