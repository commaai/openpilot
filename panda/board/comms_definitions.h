typedef struct {
  uint8_t request;
  uint16_t param1;
  uint16_t param2;
  uint16_t length;
} __attribute__((packed)) ControlPacket_t;

int comms_control_handler(ControlPacket_t *req, uint8_t *resp);
void comms_endpoint2_write(uint8_t *data, uint32_t len);
void comms_can_write(uint8_t *data, uint32_t len);
int comms_can_read(uint8_t *data, uint32_t max_len);
void comms_can_reset(void);
