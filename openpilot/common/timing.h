#pragma once

#include <cstdint>
#include <ctime>

#if defined(__APPLE__) || defined(_WIN32)
#define CLOCK_BOOTTIME CLOCK_MONOTONIC
#endif
#ifdef _WIN32
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

static inline double millis_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec * 1e-6;
}

static inline double seconds_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

static inline uint64_t nanos_since_epoch() {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

static inline double seconds_since_epoch() {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

// you probably should use nanos_since_boot instead
static inline uint64_t nanos_monotonic() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

static inline uint64_t nanos_monotonic_raw() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}
