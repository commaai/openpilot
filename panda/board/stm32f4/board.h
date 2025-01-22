// ///////////////////////////////////////////////////////////// //
// Hardware abstraction layer for all different supported boards //
// ///////////////////////////////////////////////////////////// //
#include "boards/board_declarations.h"
#include "boards/unused_funcs.h"

// ///// Board definition and detection ///// //
#include "stm32f4/lladc.h"
#include "drivers/harness.h"
#include "drivers/fan.h"
#include "stm32f4/llfan.h"
#include "drivers/clock_source.h"
#include "boards/white.h"
#include "boards/grey.h"
#include "boards/black.h"
#include "boards/uno.h"
#include "boards/dos.h"

// Unused functions on F4
void sound_tick(void) {}

void detect_board_type(void) {
  // SPI lines floating: white (TODO: is this reliable? Not really, we have to enable ESP/GPS to be able to detect this on the UART)
  set_gpio_output(GPIOC, 14, 1);
  set_gpio_output(GPIOC, 5, 1);
  if(!detect_with_pull(GPIOB, 1, PULL_UP) && !detect_with_pull(GPIOB, 7, PULL_UP)){
    hw_type = HW_TYPE_DOS;
    current_board = &board_dos;
  } else if((detect_with_pull(GPIOA, 4, PULL_DOWN)) || (detect_with_pull(GPIOA, 5, PULL_DOWN)) || (detect_with_pull(GPIOA, 6, PULL_DOWN)) || (detect_with_pull(GPIOA, 7, PULL_DOWN))){
    hw_type = HW_TYPE_WHITE_PANDA;
    current_board = &board_white;
  } else if(detect_with_pull(GPIOA, 13, PULL_DOWN)) { // Rev AB deprecated, so no pullup means black. In REV C, A13 is pulled up to 5V with a 10K
    hw_type = HW_TYPE_GREY_PANDA;
    current_board = &board_grey;
  } else if(!detect_with_pull(GPIOB, 15, PULL_UP)) {
    hw_type = HW_TYPE_UNO;
    current_board = &board_uno;
  } else {
    hw_type = HW_TYPE_BLACK_PANDA;
    current_board = &board_black;
  }
}
