#pragma once

static const unsigned char dlc_to_len[] = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 12U, 16U, 20U, 24U, 32U, 48U, 64U};

#define CANPACKET_HEAD_SIZE 6U  // non-data portion of CANPacket_t
#define CANPACKET_DATA_SIZE_MAX 64U

// bump this when changing the CAN packet
#define CAN_PACKET_VERSION 4
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
