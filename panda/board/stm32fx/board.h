// ///////////////////////////////////////////////////////////// //
// Hardware abstraction layer for all different supported boards //
// ///////////////////////////////////////////////////////////// //
#include "boards/board_declarations.h"
#include "boards/unused_funcs.h"

// ///// Board definition and detection ///// //
#include "drivers/harness.h"
#ifdef PANDA
  #include "drivers/fan.h"
  #include "stm32fx/llfan.h"
  #include "stm32fx/llrtc.h"
  #include "drivers/rtc.h"
  #include "stm32fx/clock_source.h"
  #include "boards/white.h"
  #include "boards/grey.h"
  #include "boards/black.h"
  #include "boards/uno.h"
  #include "boards/dos.h"
#else
  #include "boards/pedal.h"
#endif

void detect_board_type(void) {
  #ifdef PANDA
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
  #else
    #ifdef PEDAL
      hw_type = HW_TYPE_PEDAL;
      current_board = &board_pedal;
    #else
      hw_type = HW_TYPE_UNKNOWN;
      puts("Hardware type is UNKNOWN!\n");
    #endif
  #endif
}

bool has_external_debug_serial = 0;

void detect_external_debug_serial(void) {
  // detect if external serial debugging is present
  has_external_debug_serial = detect_with_pull(GPIOA, 3, PULL_DOWN);
}
