// this is where we manage the dongle ID assigned during our
// manufacturing. aside from this, there's a UID for the MCU

#define PROVISION_CHUNK_LEN 0x20

const unsigned char unprovisioned_text[] = "\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff";

void get_provision_chunk(uint8_t *resp) {
  (void)memcpy(resp, (uint8_t *)PROVISION_CHUNK_ADDRESS, PROVISION_CHUNK_LEN);
  if (memcmp(resp, unprovisioned_text, 0x20) == 0) {
    (void)memcpy(resp, "unprovisioned\x00\x00\x00testing123\x00\x00\xa3\xa6\x99\xec", 0x20);
  }
}
