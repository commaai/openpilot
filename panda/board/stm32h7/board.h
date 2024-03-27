// ///////////////////////////////////////////////////////////// //
// Hardware abstraction layer for all different supported boards //
// ///////////////////////////////////////////////////////////// //
#include "boards/board_declarations.h"
#include "boards/unused_funcs.h"

// ///// Board definition and detection ///// //
#include "stm32h7/lladc.h"
#include "drivers/harness.h"
#include "drivers/fan.h"
#include "stm32h7/llfan.h"
#include "stm32h7/lldac.h"
#include "drivers/fake_siren.h"
#include "drivers/clock_source.h"
#include "boards/red.h"
#include "boards/red_chiplet.h"
#include "boards/tres.h"
#include "boards/cuatro.h"


uint8_t get_board_id(void) {
  return detect_with_pull(GPIOF, 7, PULL_UP) |
         (detect_with_pull(GPIOF, 8, PULL_UP) << 1U) |
         (detect_with_pull(GPIOF, 9, PULL_UP) << 2U) |
         (detect_with_pull(GPIOF, 10, PULL_UP) << 3U);
}

void detect_board_type(void) {
  const uint8_t board_id = get_board_id();

  if (board_id == 0U) {
    hw_type = HW_TYPE_RED_PANDA;
    current_board = &board_red;
  } else if (board_id == 1U) {
    // deprecated
    //hw_type = HW_TYPE_RED_PANDA_V2;
  } else if (board_id == 2U) {
    hw_type = HW_TYPE_TRES;
    current_board = &board_tres;
  } else if (board_id == 3U) {
    hw_type = HW_TYPE_CUATRO;
    current_board = &board_tres;
  } else {
    hw_type = HW_TYPE_UNKNOWN;
    print("Hardware type is UNKNOWN!\n");
  }

  // TODO: detect this live
#ifdef STM32H723
  hw_type = HW_TYPE_CUATRO;
  current_board = &board_cuatro;
#endif
}
