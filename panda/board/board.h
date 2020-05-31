// ///////////////////////////////////////////////////////////// //
// Hardware abstraction layer for all different supported boards //
// ///////////////////////////////////////////////////////////// //
#include "board_declarations.h"
#include "boards/common.h"

// ///// Board definition and detection ///// //
#include "drivers/harness.h"
#ifdef PANDA
  #include "drivers/fan.h"
  #include "drivers/rtc.h"
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
    if(!detect_with_pull(GPIOB, 1, PULL_UP)){
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


// ///// Configuration detection ///// //
bool has_external_debug_serial = 0;
bool is_entering_bootmode = 0;

void detect_configuration(void) {
  // detect if external serial debugging is present
  has_external_debug_serial = detect_with_pull(GPIOA, 3, PULL_DOWN);

  #ifdef PANDA
    if(hw_type == HW_TYPE_WHITE_PANDA) {
      // check if the ESP is trying to put me in boot mode
      is_entering_bootmode = !detect_with_pull(GPIOB, 0, PULL_UP);
    } else {
      is_entering_bootmode = 0;
    }
  #else
    is_entering_bootmode = 0;
  #endif
}

// ///// Board functions ///// //
// TODO: Make these config options in the board struct
bool board_has_gps(void) {
  return ((hw_type == HW_TYPE_GREY_PANDA) || (hw_type == HW_TYPE_BLACK_PANDA) || (hw_type == HW_TYPE_UNO));
}

bool board_has_gmlan(void) {
  return ((hw_type == HW_TYPE_WHITE_PANDA) || (hw_type == HW_TYPE_GREY_PANDA));
}

bool board_has_obd(void) {
  return ((hw_type == HW_TYPE_BLACK_PANDA) || (hw_type == HW_TYPE_UNO) || (hw_type == HW_TYPE_DOS));
}

bool board_has_lin(void) {
  return ((hw_type == HW_TYPE_WHITE_PANDA) || (hw_type == HW_TYPE_GREY_PANDA));
}

bool board_has_rtc(void) {
  return ((hw_type == HW_TYPE_UNO) || (hw_type == HW_TYPE_DOS));
}

bool board_has_relay(void) {
  return ((hw_type == HW_TYPE_BLACK_PANDA) || (hw_type == HW_TYPE_UNO) || (hw_type == HW_TYPE_DOS));
}
