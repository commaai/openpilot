#define PROVISION_CHUNK_LEN 0x20

// WiFi SSID     = 0x0  - 0x10
// WiFi password = 0x10 - 0x1C
// SHA1 checksum = 0x1C - 0x20

void get_provision_chunk(uint8_t *resp) {
  (void)memcpy(resp, (uint8_t *)0x1fff79e0, PROVISION_CHUNK_LEN);
  if (memcmp(resp, "\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff", 0x20) == 0) {
    (void)memcpy(resp, "unprovisioned\x00\x00\x00testing123\x00\x00\xa3\xa6\x99\xec", 0x20);
  }
}

