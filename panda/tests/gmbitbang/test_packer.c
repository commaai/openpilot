#include <stdio.h>
#include <stdint.h>

typedef struct {
  uint32_t RIR;  /*!< CAN receive FIFO mailbox identifier register */
  uint32_t RDTR; /*!< CAN receive FIFO mailbox data length control and time stamp register */
  uint32_t RDLR; /*!< CAN receive FIFO mailbox data low register */
  uint32_t RDHR; /*!< CAN receive FIFO mailbox data high register */
} CAN_FIFOMailBox_TypeDef;

#include "../../board/drivers/canbitbang.h"

int main() {
  char out[300];
  CAN_FIFOMailBox_TypeDef to_bang = {0};
  to_bang.RIR = 20 << 21;
  to_bang.RDTR = 1;
  to_bang.RDLR = 1;

  int len = get_bit_message(out, &to_bang);
  printf("T:");
  for (int i = 0; i < len; i++) {
    printf("%d", out[i]);
  }
  printf("\n");
  printf("R:0000010010100000100010000010011110111010100111111111111111");
  printf("\n");
  return 0;
}



