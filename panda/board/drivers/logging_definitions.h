
// Flash is writable in 32-byte lines, this struct is designed to fit in two lines.
// This also matches the USB transfer size.
typedef struct __attribute__((packed)) log_t {
  uint16_t id;
  timestamp_t timestamp;
  uint32_t uptime;
  char msg[50];
} log_t;
