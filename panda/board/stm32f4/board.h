// ///////////////////////////////////////////////////////////// //
// Hardware abstraction layer for all different supported boards //
// ///////////////////////////////////////////////////////////// //
#include "board/boards/board_declarations.h"
#include "board/boards/unused_funcs.h"

// ///// Board definition and detection ///// //
#include "board/stm32f4/lladc.h"
#include "board/drivers/harness.h"
#include "board/drivers/fan.h"
#include "board/stm32f4/llfan.h"
#include "board/drivers/clock_source.h"
#include "board/boards/dos.h"

void detect_board_type(void) {
  set_gpio_output(GPIOC, 14, 1);
  set_gpio_output(GPIOC, 5, 1);
  if (!detect_with_pull(GPIOB, 1, PULL_UP) && !detect_with_pull(GPIOB, 7, PULL_UP)) {
    hw_type = HW_TYPE_DOS;
    current_board = &board_dos;
  }

  // Return A13 to the alt mode to fix SWD
  set_gpio_alternate(GPIOA, 13, GPIO_AF0_SWJ);
}
