/*
 *
	Platform specific definitions
 *
 */
#ifndef PLATFORM_DEFS_H_
#define PLATFORM_DEFS_H_

#define SUSPEND_INTERRUPT \
	__asm volatile("mfmsr %0" : "=r" (msr) :);\
	if (msr & (uint32_t)0x00008000UL)\
		__asm(" wrteei 0");

#define RESUME_INTERRUPT \
	if (msr & (uint32_t)0x00008000UL)\
		__asm(" wrteei 1");

#define min(x, y) (x <= y ? x : y)
#define max(x, y) (x >= y ? x : y)

#define NULL ((void *)0)

#ifdef DEBUG_FLEXRAY
#include <console.h>
#define DBG_PRINT(fmt, ...) \
	do {\
		(void)PRINTF("\r\n%s:%u ", __FILE__, __LINE__);\
		(void)PRINTF(fmt, ##__VA_ARGS__);\
	} while(0);
#else
#define DBG_PRINT(fmt, ...)
#endif

void sleep(uint32_t ms);

#endif /* PLATFORM_DEFS_H_ */
