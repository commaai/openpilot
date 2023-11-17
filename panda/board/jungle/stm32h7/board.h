#include "boards/board_declarations.h"

#include "stm32h7/lladc.h"
#include "boards/board_v2.h"

void detect_board_type(void) {
  hw_type = HW_TYPE_V2;
  current_board = &board_v2;
}
