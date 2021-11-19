#ifndef PANDA_CONFIG_H
#define PANDA_CONFIG_H

//#define DEBUG
//#define DEBUG_UART
//#define DEBUG_USB
//#define DEBUG_SPI
//#define DEBUG_FAULTS

#define USB_VID 0xbbaaU

#ifdef BOOTSTUB
  #define USB_PID 0xddeeU
#else
  #define USB_PID 0xddccU
#endif

#define NULL ((void*)0)
#define COMPILE_TIME_ASSERT(pred) ((void)sizeof(char[1 - (2 * ((int)(!(pred))))]))

#define MIN(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   (_a < _b) ? _a : _b; })

#define MAX(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   (_a > _b) ? _a : _b; })

#define ABS(a) \
 ({ __typeof__ (a) _a = (a); \
   (_a > 0) ? _a : (-_a); })

#define MAX_RESP_LEN 0x40U

#include <stdbool.h>
#ifdef STM32H7
  #include "stm32h7/stm32h7_config.h"
#else
  #include "stm32fx/stm32fx_config.h"
#endif

#endif
