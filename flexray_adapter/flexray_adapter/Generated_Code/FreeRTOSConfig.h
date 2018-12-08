/*
    FreeRTOS V9.0.0 - Copyright (C) 2016 Real Time Engineers Ltd.
    All rights reserved

    VISIT http://www.FreeRTOS.org TO ENSURE YOU ARE USING THE LATEST VERSION.

    This file is part of the FreeRTOS distribution.

    FreeRTOS is free software; you can redistribute it and/or modify it under
    the terms of the GNU General Public License (version 2) as published by the
    Free Software Foundation >>>> AND MODIFIED BY <<<< the FreeRTOS exception.

    ***************************************************************************
    >>!   NOTE: The modification to the GPL is included to allow you to     !<<
    >>!   distribute a combined work that includes FreeRTOS without being   !<<
    >>!   obliged to provide the source code for proprietary components     !<<
    >>!   outside of the FreeRTOS kernel.                                   !<<
    ***************************************************************************

    FreeRTOS is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  Full license text is available on the following
    link: http://www.freertos.org/a00114.html

    ***************************************************************************
     *                                                                       *
     *    FreeRTOS provides completely free yet professionally developed,    *
     *    robust, strictly quality controlled, supported, and cross          *
     *    platform software that is more than just the market leader, it     *
     *    is the industry's de facto standard.                               *
     *                                                                       *
     *    Help yourself get started quickly while simultaneously helping     *
     *    to support the FreeRTOS project by purchasing a FreeRTOS           *
     *    tutorial book, reference manual, or both:                          *
     *    http://www.FreeRTOS.org/Documentation                              *
     *                                                                       *
    ***************************************************************************

    http://www.FreeRTOS.org/FAQHelp.html - Having a problem?  Start by reading
    the FAQ page "My application does not run, what could be wrong?".  Have you
    defined configASSERT()?

    http://www.FreeRTOS.org/support - In return for receiving this top quality
    embedded software for free we request you assist our global community by
    participating in the support forum.

    http://www.FreeRTOS.org/training - Investing in training allows your team to
    be as productive as possible as early as possible.  Now you can receive
    FreeRTOS training directly from Richard Barry, CEO of Real Time Engineers
    Ltd, and the world's leading authority on the world's leading RTOS.

    http://www.FreeRTOS.org/plus - A selection of FreeRTOS ecosystem products,
    including FreeRTOS+Trace - an indispensable productivity tool, a DOS
    compatible FAT file system, and our tiny thread aware UDP/IP stack.

    http://www.FreeRTOS.org/labs - Where new FreeRTOS products go to incubate.
    Come and try FreeRTOS+TCP, our new open source TCP/IP stack for FreeRTOS.

    http://www.OpenRTOS.com - Real Time Engineers ltd. license FreeRTOS to High
    Integrity Systems ltd. to sell under the OpenRTOS brand.  Low cost OpenRTOS
    licenses offer ticketed support, indemnification and commercial middleware.

    http://www.SafeRTOS.com - High Integrity Systems also provide a safety
    engineered and independently SIL3 certified version for use in safety and
    mission critical applications that require provable dependability.

    1 tab == 4 spaces!
*/


#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H

/**
 * @page misra_violations MISRA-C:2012 violations
 *
 * @section [global]
 * Violates MISRA 2012 Advisory Directive 4.9, Function-like macro defined.
 * The macro is used for development validation.
 *
 * @section [global]
 * Violates MISRA 2012 Required Rule 3.1, C comment contains C++ comment.
 * Detections are URL links from FreeRTOS header text.
 *
 */

/*-----------------------------------------------------------
 * Application specific definitions.
 *
 * These definitions should be adjusted for your particular hardware and
 * application requirements.
 *
 * THESE PARAMETERS ARE DESCRIBED WITHIN THE 'CONFIGURATION' SECTION OF THE
 * FreeRTOS API DOCUMENTATION AVAILABLE ON THE FreeRTOS.org WEB SITE.
 *
 * See http://www.freertos.org/a00110.html.
 *----------------------------------------------------------*/

#define configUSE_PREEMPTION                     1
#define configUSE_IDLE_HOOK                      0
#define configUSE_TICK_HOOK                      0
#define configCPU_CLOCK_HZ                       ( 40000000UL )
#define configTICK_RATE_HZ                       ( ( TickType_t ) 1000 )
#define configMAX_PRIORITIES                     ( 8 )
#define configMINIMAL_STACK_SIZE                 ( ( unsigned short ) 512 )
#define configTOTAL_HEAP_SIZE                    ( ( size_t ) 49152 )
#define configMAX_TASK_NAME_LEN                  ( 12 )
#define configUSE_TRACE_FACILITY                 0
#define configUSE_16_BIT_TICKS                   0
#define configIDLE_SHOULD_YIELD                  1
#define configUSE_MUTEXES                        1
#define configQUEUE_REGISTRY_SIZE                0
#define configCHECK_FOR_STACK_OVERFLOW           0
#define configUSE_RECURSIVE_MUTEXES              0
#define configUSE_MALLOC_FAILED_HOOK             0
#define configUSE_APPLICATION_TASK_TAG           0
#define configUSE_COUNTING_SEMAPHORES            1

/* Co-routine definitions. */
#define configUSE_CO_ROUTINES                    0
#define configMAX_CO_ROUTINE_PRIORITIES          ( 2 )

/* Software timer definitions. */
#define configUSE_TIMERS                         0

/* API functions definitions */
#define INCLUDE_vTaskPrioritySet                 0
#define INCLUDE_uxTaskPriorityGet                0
#define INCLUDE_vTaskDelete                      1
#define INCLUDE_vTaskSuspend                     0
#define INCLUDE_vTaskDelayUntil                  0
#define INCLUDE_vTaskDelay                       1
#define INCLUDE_eTaskGetState                    0
#define INCLUDE_uxTaskGetStackHighWaterMark      1
#define INCLUDE_xTaskGetSchedulerState           0
#define INCLUDE_xQueueGetMutexHolder             1
#define INCLUDE_xTaskGetCurrentTaskHandle        1
#define INCLUDE_xTaskGetIdleTaskHandle           0
#define INCLUDE_pcTaskGetTaskName                0
#define INCLUDE_xEventGroupSetBitFromISR         0
#define INCLUDE_xTimerPendFunctionCall           0

/* Additional settings can be defined in the property Settings > User settings > Definitions of the FreeRTOS component */
#define configNUM_THREAD_LOCAL_STORAGE_POINTERS       2
#if defined(__GNUC__) && !defined(__ASSEMBLER__)
#include "device_registers.h"
#include "devassert.h"
#endif

/* This demo makes use of one or more example stats formatting functions.  These
format the raw data provided by the uxTaskGetSystemState() function in to human
readable ASCII form.  See the notes in the implementation of vTaskList() within
FreeRTOS/Source/tasks.c for limitations. */
#define configUSE_STATS_FORMATTING_FUNCTIONS     0

/* Run time stats gathering definitions. */
#ifdef __ICCARM__
	/* The #ifdef just prevents this C specific syntax from being included in
	assembly files. */
	void vMainConfigureTimerForRunTimeStats( void );
	unsigned long ulMainGetRunTimeCounterValue( void );
#endif
#if defined(__GNUC__) && !defined(__ASSEMBLER__)
	/* The #ifdef just prevents this C specific syntax from being included in
	assembly files. */
	void vMainConfigureTimerForRunTimeStats( void );
	unsigned long ulMainGetRunTimeCounterValue( void );
#endif


#define configGENERATE_RUN_TIME_STATS            0 /* 1: generate runtime statistics; 0: no runtime statistics */

/* Calypso specific: pit channel to use 0-15 */
#define configUSE_PIT_CHANNEL                    (15)

/* The highest interrupt priority that can be used by any interrupt service
routine that makes calls to interrupt safe FreeRTOS API functions.  DO NOT CALL
INTERRUPT SAFE FREERTOS API FUNCTIONS FROM ANY INTERRUPT THAT HAS A HIGHER
PRIORITY THAN THIS! (higher priorities are lower numeric values. */
#define configMAX_API_CALL_INTERRUPT_PRIORITY    (15)

/* Definition assert() function. */
#define configASSERT(x)                          DEV_ASSERT(x)


/* Tickless Idle Mode */
#define configUSE_TICKLESS_IDLE                  0 
#define configEXPECTED_IDLE_TIME_BEFORE_SLEEP    2 
#define configUSE_TICKLESS_IDLE_DECISION_HOOK    0 
#endif /* FREERTOS_CONFIG_H */
