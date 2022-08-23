// Define to prevent recursive inclusion
#ifndef COMMS_H
#define COMMS_H

#define OFFSET            0x8U
#define BROADCAST_ADDR    0x7DFU
#define FALLBACK_ADDR     0x7E0U
#define FALLBACK_R_ADDR   (FALLBACK_ADDR + OFFSET)
#define ECU_ADDR          0x720U
#define ECU_R_ADDR        (ECU_ADDR + OFFSET)
#define DEBUG_ADDR      0x721U
#define DEBUG_R_ADDR    (DEBUG_ADDR + OFFSET)

#include "drivers/llbxcan.h"
#include "uds.h"

extern P rtP_Left;
extern P rtP_Right;
extern volatile int16_t cmdL;                    // global variable for Left Command
extern volatile int16_t cmdR;                    // global variable for Right Command
extern uint8_t hw_type;
extern board_t board;

extern uint32_t enter_bootloader_mode;
extern volatile uint32_t torque_cmd_timeout;

const uint8_t crc_poly = 0xD5U;  // standard crc8
uint32_t current_idx = 0;

typedef struct {
  volatile uint32_t w_ptr;
  volatile uint32_t r_ptr;
  uint32_t fifo_size;
  CAN_FIFOMailBox_TypeDef *elems;
} can_ring;

#define can_buffer(x, size) \
  CAN_FIFOMailBox_TypeDef elems_##x[size]; \
  can_ring can_##x = { .w_ptr = 0, .r_ptr = 0, .fifo_size = (size), .elems = (CAN_FIFOMailBox_TypeDef *)&(elems_##x) };

can_buffer(tx_q, 0x1A0)

bool can_pop(can_ring *q, CAN_FIFOMailBox_TypeDef *elem) {
  bool ret = 0;

  if (q->w_ptr != q->r_ptr) {
    *elem = q->elems[q->r_ptr];
    if ((q->r_ptr + 1U) == q->fifo_size) {
      q->r_ptr = 0;
    } else {
      q->r_ptr += 1U;
    }
    ret = 1;
  }
  return ret;
}

bool can_push(can_ring *q, CAN_FIFOMailBox_TypeDef *elem) {
  bool ret = false;
  uint32_t next_w_ptr;

  if ((q->w_ptr + 1U) == q->fifo_size) {
    next_w_ptr = 0;
  } else {
    next_w_ptr = q->w_ptr + 1U;
  }
  if (next_w_ptr != q->r_ptr) {
    q->elems[q->w_ptr] = *elem;
    q->w_ptr = next_w_ptr;
    ret = true;
  }
  return ret;
}

void can_send_msg(uint32_t addr, uint32_t dhr, uint32_t dlr, uint8_t len) {
  CAN_FIFOMailBox_TypeDef to_push;

  to_push.RDHR = dhr;
  to_push.RDLR = dlr;
  to_push.RDTR = len;
  to_push.RIR = ((addr >= 0x800U) ? ((addr << 3) | (1U << 2)) : (addr << 21)) | 1;

  can_push(&can_tx_q, &to_push);
}

void process_can(void) {
  CAN_FIFOMailBox_TypeDef to_send;
  if (board.CAN->TSR & CAN_TSR_TME0) {
    if (can_pop(&can_tx_q, &to_send)) {
      board.CAN->sTxMailBox[0].TDLR = to_send.RDLR;
      board.CAN->sTxMailBox[0].TDHR = to_send.RDHR;
      board.CAN->sTxMailBox[0].TDTR = to_send.RDTR;
      board.CAN->sTxMailBox[0].TIR = to_send.RIR;
    }
  }
}

void can_rx(void) {
  while ((board.CAN->RF0R & CAN_RF0R_FMP0) != 0) {
    int address = board.CAN->sFIFOMailBox[0].RIR >> 21;
    if (address == (int32_t)(0x250U + board.can_addr_offset)) {
      if ((GET_MAILBOX_BYTES_04(&board.CAN->sFIFOMailBox[0]) == 0xdeadface) && (GET_MAILBOX_BYTES_48(&board.CAN->sFIFOMailBox[0]) == 0x0ab00b1e)) {
        enter_bootloader_mode = ENTER_SOFTLOADER_MAGIC;
        NVIC_SystemReset();
      }
      #define MSG_TRQ_LEN 6
      uint8_t dat[MSG_TRQ_LEN];
      for (int i=0; i<MSG_TRQ_LEN; i++) {
        dat[i] = GET_MAILBOX_BYTE(&board.CAN->sFIFOMailBox[0], i);
      }
      uint16_t valueL = ((dat[0] << 8U) | dat[1]);
      uint16_t valueR = ((dat[2] << 8U) | dat[3]);

      uint8_t idx = dat[4] & 0xFU;
      if (crc_checksum(dat, 5, crc_poly) == dat[5]) {
        if (((current_idx + 1U) & 0xFU) == idx) {
          cmdL = valueL;
          cmdR = valueR;
          torque_cmd_timeout = 0;
        }
        current_idx = idx;
      }
      out_enable(LED_BLUE, true);
    } else if (address == (int32_t)(0x251U + board.can_addr_offset)) {
      #define MSG_SPD_LEN 5
      uint8_t dat[MSG_TRQ_LEN];
      for (int i=0; i<MSG_TRQ_LEN; i++) {
        dat[i] = GET_MAILBOX_BYTE(&board.CAN->sFIFOMailBox[0], i);
      }
      uint16_t valueL = ((dat[0] << 8U) | dat[1]);
      uint16_t valueR = ((dat[2] << 8U) | dat[3]);

      if (crc_checksum(dat, 4, crc_poly) == dat[4]) {
        if ((valueL == 0) || (valueR == 0)) {
          rtP_Left.n_max = rtP_Right.n_max = N_MOT_MAX << 4;
        } else {
          rtP_Left.n_max = valueL << 4;
          rtP_Right.n_max = valueR << 4;
        }
      }
      out_enable(LED_BLUE, true);
    } else if ((hw_type == HW_TYPE_BASE) && ((address == BROADCAST_ADDR) || (address == FALLBACK_ADDR) || (address == ECU_ADDR) || (address == DEBUG_ADDR))) { // Process UBS and OBD2 requests, ignore for knee
      process_uds(address, GET_MAILBOX_BYTES_04(&board.CAN->sFIFOMailBox[0]));
      out_enable(LED_BLUE, true);
    } else if ((hw_type == HW_TYPE_BASE) && (address == 0x203U + KNEE_ADDR_OFFSET)) { // detect knee by body and set flag for use with UDS message
      knee_detected = 1;
    }
    // next
    board.CAN->RF0R |= CAN_RF0R_RFOM0;
  }
}

void CAN1_TX_IRQHandler(void) {
  // clear interrupt
  board.CAN->TSR |= CAN_TSR_RQCP0;
  process_can();
}

void CAN1_SCE_IRQHandler(void) {
  llcan_clear_send(board.CAN);
}

void CAN1_RX0_IRQHandler(void) {
  can_rx();
}

void CAN2_TX_IRQHandler(void) {
  // clear interrupt
  board.CAN->TSR |= CAN_TSR_RQCP0;
  process_can();
}

void CAN2_SCE_IRQHandler(void) {
  llcan_clear_send(board.CAN);
}

void CAN2_RX0_IRQHandler(void) {
  can_rx();
}

#endif
