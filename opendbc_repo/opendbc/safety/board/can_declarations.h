#pragma once

#define CANPACKET_HEAD_SIZE 6U
#define CANPACKET_DATA_SIZE_MAX 64U

typedef struct {
  unsigned char fd : 1;
  unsigned char bus : 3;
  unsigned char data_len_code : 4;  // lookup length with dlc_to_len
  unsigned char rejected : 1;
  unsigned char returned : 1;
  unsigned char extended : 1;
  unsigned int addr : 29;
  unsigned char checksum;
  unsigned char data[CANPACKET_DATA_SIZE_MAX];
} __attribute__((packed, aligned(4))) CANPacket_t;

#define GET_LEN(msg) (dlc_to_len[(msg)->data_len_code])
