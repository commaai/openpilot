#ifndef PANDA_CONFIG_H
#define PANDA_CONFIG_H

//#define DEBUG
//#define DEBUG_USB
//#define DEBUG_SPI

#ifdef STM32F4
  #define PANDA
  #include "stm32f4xx.h"
#else
  #include "stm32f2xx.h"
#endif

#define USB_VID 0xbbaa

#ifdef BOOTSTUB
#define USB_PID 0xddee
#else
#define USB_PID 0xddcc
#endif

#include <stdbool.h>
#define NULL ((void*)0)
#define COMPILE_TIME_ASSERT(pred) ((void)sizeof(char[1 - (2 * (!(pred)))]))

#define MIN(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   (_a < _b) ? _a : _b; })

#define MAX(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   (_a > _b) ? _a : _b; })

#define MAX_RESP_LEN 0x40

#endif

