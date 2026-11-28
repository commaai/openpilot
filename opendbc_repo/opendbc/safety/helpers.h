#ifndef SAFETY_HELPERS_H
#define SAFETY_HELPERS_H

#include "opendbc/safety/declarations.h"

static bool GET_BIT(const CANPacket_t *msg, uint32_t b);
static uint32_t GET_BYTES_LE(const CANPacket_t *msg, int start, int len);
static uint32_t GET_BYTES_BE(const CANPacket_t *msg, int start, int len);
static uint64_t GET_BYTES_64_LE(const CANPacket_t *msg, int start, int len);
static int to_signed(uint32_t d, int bits);
static int sign_magnitude(uint32_t magnitude, bool negative);
static int from_offset(uint32_t raw, int offset);
static void update_sample(struct sample_t *sample, int sample_new);
static int ROUND(float val);
static void enter_controls(void);
static void exit_controls(void);
static void pcm_cruise_check(bool cruise_engaged);
static void speed_mismatch_check(const float speed_2);

static inline int safety_min_int(int a, int b) {
  return (a < b) ? a : b;
}

static inline uint8_t safety_min_uint8(uint8_t a, uint8_t b) {
  return (a < b) ? a : b;
}

static inline int safety_max_int(int a, int b) {
  return (a > b) ? a : b;
}

static inline float safety_max_float(float a, float b) {
  return (a > b) ? a : b;
}

static inline int safety_clamp_int(int x, int low, int high) {
  return (x > high) ? high : ((x < low) ? low : x);
}

#define SAFETY_ARRAY_LEN(a) ((int)sizeof(a) / (int)sizeof((a)[0]))

#define SAFETY_UNUSED(x) ((void)(x))

// Update an MSB-first CRC8 with one byte; callers choose the initial value and final XOR.
static uint8_t crc8_update(uint8_t initial_crc, uint8_t data, uint8_t poly) {
  uint8_t crc = initial_crc ^ data;
  for (int i = 0; i < 8; i++) {
    if ((crc & 0x80U) != 0U) {
      crc = (crc << 1) ^ poly;
    } else {
      crc <<= 1;
    }
  }
  return crc;
}

// Compare address and bus as one key: CANPacket_t has a 29-bit address and 3-bit bus.
static bool msg_matches_addr_bus(const CANPacket_t *msg, uint32_t addr, uint32_t bus) {
  uint32_t actual = ((uint32_t)msg->addr << 3) | (uint32_t)msg->bus;
  uint32_t expected = (addr << 3) | bus;
  return actual == expected;
}

// Compare address, bus, and decoded byte length as one key.
static bool msg_matches_addr_bus_len(const CANPacket_t *msg, uint32_t addr, uint32_t bus, uint32_t len) {
  uint64_t actual = ((uint64_t)msg->addr << 10) | ((uint64_t)msg->bus << 7) | (uint64_t)dlc_to_len[msg->data_len_code];
  uint64_t expected = ((uint64_t)addr << 10) | ((uint64_t)bus << 7) | (uint64_t)len;
  return actual == expected;
}

// the trailing 0 keeps the "..." of MSG_MATCHES_SELECT non-empty, as ISO C11 requires
#define MSG_MATCHES_SELECT(_msg, _addr, _bus, _len, NAME, ...) (NAME)
#define msg_matches(...) MSG_MATCHES_SELECT(__VA_ARGS__, msg_matches_addr_bus_len, msg_matches_addr_bus, 0)(__VA_ARGS__)

// compute the time elapsed (in microseconds) from 2 counter samples
// case where ts < ts_last is ok: overflow is properly re-casted into uint32_t
static inline uint32_t safety_get_ts_elapsed(uint32_t ts, uint32_t ts_last) {
  return ts - ts_last;
}

static bool safety_max_limit_check(int val, const int MAX_VAL, const int MIN_VAL) {
  return (val > MAX_VAL) || (val < MIN_VAL);
}

static float vehicle_speed_ms(int raw) {
  return (float)raw / VEHICLE_SPEED_FACTOR;
}

// convert a limit to CAN units and add a margin, truncating toward zero
static int limit_to_can(const float limit, const float to_can, const float margin) {
  const float limit_can = (limit * to_can) + margin;
  return (int)limit_can;
}

// interp function that holds extreme values
static float safety_interpolate(struct lookup_t xy, float x) {
  int size = SAFETY_ARRAY_LEN(xy.x);
  float ret = xy.y[size - 1];  // default output is last point

  // x is lower than the first point in the x array. Return the first point
  if (x <= xy.x[0]) {
    ret = xy.y[0];

  } else {
    // walk the segments, keeping x0 <= x
    float x0 = xy.x[0];
    float y0 = xy.y[0];
    for (int i = 1; i < size; i++) {
      float x1 = xy.x[i];
      float y1 = xy.y[i];
      if (x < x1) {
        // x0 <= x < x1, so x1 - x0 > 0
        ret = ((y1 - y0) * (x - x0) / (x1 - x0)) + y0;
        break;
      }
      x0 = x1;
      y0 = y1;
    }
  }
  return ret;
}

#endif  // SAFETY_HELPERS_H
