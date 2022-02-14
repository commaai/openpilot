#define PROVISION_CHUNK_LEN 0x20

void get_provision_chunk(uint8_t *resp) {
  (void)memcpy(resp, (uint8_t *)PROVISION_CHUNK_ADDRESS, PROVISION_CHUNK_LEN);
  if (memcmp(resp, "\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff", 0x20) == 0) {
    (void)memcpy(resp, "unprovisioned\x00\x00\x00testing123\x00\x00\xa3\xa6\x99\xec", 0x20);
  }
}

uint8_t chunk[PROVISION_CHUNK_LEN];
bool is_provisioned(void) {
  (void)memcpy(chunk, (uint8_t *)PROVISION_CHUNK_ADDRESS, PROVISION_CHUNK_LEN);
  return (memcmp(chunk, "\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff", 0x20) != 0);
}
