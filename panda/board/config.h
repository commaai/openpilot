#pragma once

#include <stdbool.h>

//#define DEBUG
//#define DEBUG_UART
//#define DEBUG_USB
//#define DEBUG_SPI
//#define DEBUG_FAULTS
//#define DEBUG_COMMS
//#define DEBUG_FAN

#define CAN_INIT_TIMEOUT_MS 500U
#define DEEPSLEEP_WAKEUP_DELAY 3U
#define USBPACKET_MAX_SIZE 0x40U
#define MAX_CAN_MSGS_PER_USB_BULK_TRANSFER 51U
#define MAX_CAN_MSGS_PER_SPI_BULK_TRANSFER 170U

#define VIN_READOUT_DIVIDER 11U

// USB definitions
#define USB_VID 0xBBAAU

#ifdef BOOTSTUB
  #define USB_PID 0xDDEEU
#else
  #define USB_PID 0xDDCCU
#endif

// platform includes
#ifdef STM32H7
  #include "stm32h7/stm32h7_config.h"
#elif defined(STM32F2) || defined(STM32F4)
  #include "stm32fx/stm32fx_config.h"
#else
  // TODO: uncomment this, cppcheck complains
  // building for tests
  //#include "fake_stm.h"
#endif
