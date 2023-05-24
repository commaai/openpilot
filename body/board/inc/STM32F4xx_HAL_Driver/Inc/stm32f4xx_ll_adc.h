/**
  ******************************************************************************
  * @file    stm32f4xx_ll_adc.h
  * @author  MCD Application Team
  * @brief   Header file of ADC LL module.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM32F4xx_LL_ADC_H
#define __STM32F4xx_LL_ADC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (ADC1) || defined (ADC2) || defined (ADC3)

/** @defgroup ADC_LL ADC
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/

/* Private constants ---------------------------------------------------------*/
/** @defgroup ADC_LL_Private_Constants ADC Private Constants
  * @{
  */

/* Internal mask for ADC group regular sequencer:                             */
/* To select into literal LL_ADC_REG_RANK_x the relevant bits for:            */
/* - sequencer register offset                                                */
/* - sequencer rank bits position into the selected register                  */

/* Internal register offset for ADC group regular sequencer configuration */
/* (offset placed into a spare area of literal definition) */
#define ADC_SQR1_REGOFFSET                 0x00000000UL
#define ADC_SQR2_REGOFFSET                 0x00000100UL
#define ADC_SQR3_REGOFFSET                 0x00000200UL
#define ADC_SQR4_REGOFFSET                 0x00000300UL

#define ADC_REG_SQRX_REGOFFSET_MASK        (ADC_SQR1_REGOFFSET | ADC_SQR2_REGOFFSET | ADC_SQR3_REGOFFSET | ADC_SQR4_REGOFFSET)
#define ADC_REG_RANK_ID_SQRX_MASK          (ADC_CHANNEL_ID_NUMBER_MASK_POSBIT0)

/* Definition of ADC group regular sequencer bits information to be inserted  */
/* into ADC group regular sequencer ranks literals definition.                */
#define ADC_REG_RANK_1_SQRX_BITOFFSET_POS  ( 0UL) /* Value equivalent to POSITION_VAL(ADC_SQR3_SQ1) */
#define ADC_REG_RANK_2_SQRX_BITOFFSET_POS  ( 5UL) /* Value equivalent to POSITION_VAL(ADC_SQR3_SQ2) */
#define ADC_REG_RANK_3_SQRX_BITOFFSET_POS  (10UL) /* Value equivalent to POSITION_VAL(ADC_SQR3_SQ3) */
#define ADC_REG_RANK_4_SQRX_BITOFFSET_POS  (15UL) /* Value equivalent to POSITION_VAL(ADC_SQR3_SQ4) */
#define ADC_REG_RANK_5_SQRX_BITOFFSET_POS  (20UL) /* Value equivalent to POSITION_VAL(ADC_SQR3_SQ5) */
#define ADC_REG_RANK_6_SQRX_BITOFFSET_POS  (25UL) /* Value equivalent to POSITION_VAL(ADC_SQR3_SQ6) */
#define ADC_REG_RANK_7_SQRX_BITOFFSET_POS  ( 0UL) /* Value equivalent to POSITION_VAL(ADC_SQR2_SQ7) */
#define ADC_REG_RANK_8_SQRX_BITOFFSET_POS  ( 5UL) /* Value equivalent to POSITION_VAL(ADC_SQR2_SQ8) */
#define ADC_REG_RANK_9_SQRX_BITOFFSET_POS  (10UL) /* Value equivalent to POSITION_VAL(ADC_SQR2_SQ9) */
#define ADC_REG_RANK_10_SQRX_BITOFFSET_POS (15UL) /* Value equivalent to POSITION_VAL(ADC_SQR2_SQ10) */
#define ADC_REG_RANK_11_SQRX_BITOFFSET_POS (20UL) /* Value equivalent to POSITION_VAL(ADC_SQR2_SQ11) */
#define ADC_REG_RANK_12_SQRX_BITOFFSET_POS (25UL) /* Value equivalent to POSITION_VAL(ADC_SQR2_SQ12) */
#define ADC_REG_RANK_13_SQRX_BITOFFSET_POS ( 0UL) /* Value equivalent to POSITION_VAL(ADC_SQR1_SQ13) */
#define ADC_REG_RANK_14_SQRX_BITOFFSET_POS ( 5UL) /* Value equivalent to POSITION_VAL(ADC_SQR1_SQ14) */
#define ADC_REG_RANK_15_SQRX_BITOFFSET_POS (10UL) /* Value equivalent to POSITION_VAL(ADC_SQR1_SQ15) */
#define ADC_REG_RANK_16_SQRX_BITOFFSET_POS (15UL) /* Value equivalent to POSITION_VAL(ADC_SQR1_SQ16) */

/* Internal mask for ADC group injected sequencer:                            */
/* To select into literal LL_ADC_INJ_RANK_x the relevant bits for:            */
/* - data register offset                                                     */
/* - offset register offset                                                   */
/* - sequencer rank bits position into the selected register                  */

/* Internal register offset for ADC group injected data register */
/* (offset placed into a spare area of literal definition) */
#define ADC_JDR1_REGOFFSET                 0x00000000UL
#define ADC_JDR2_REGOFFSET                 0x00000100UL
#define ADC_JDR3_REGOFFSET                 0x00000200UL
#define ADC_JDR4_REGOFFSET                 0x00000300UL

/* Internal register offset for ADC group injected offset configuration */
/* (offset placed into a spare area of literal definition) */
#define ADC_JOFR1_REGOFFSET                0x00000000UL
#define ADC_JOFR2_REGOFFSET                0x00001000UL
#define ADC_JOFR3_REGOFFSET                0x00002000UL
#define ADC_JOFR4_REGOFFSET                0x00003000UL

#define ADC_INJ_JDRX_REGOFFSET_MASK        (ADC_JDR1_REGOFFSET | ADC_JDR2_REGOFFSET | ADC_JDR3_REGOFFSET | ADC_JDR4_REGOFFSET)
#define ADC_INJ_JOFRX_REGOFFSET_MASK       (ADC_JOFR1_REGOFFSET | ADC_JOFR2_REGOFFSET | ADC_JOFR3_REGOFFSET | ADC_JOFR4_REGOFFSET)
#define ADC_INJ_RANK_ID_JSQR_MASK          (ADC_CHANNEL_ID_NUMBER_MASK_POSBIT0)

/* Internal mask for ADC group regular trigger:                               */
/* To select into literal LL_ADC_REG_TRIG_x the relevant bits for:            */
/* - regular trigger source                                                   */
/* - regular trigger edge                                                     */
#define ADC_REG_TRIG_EXT_EDGE_DEFAULT       (ADC_CR2_EXTEN_0) /* Trigger edge set to rising edge (default setting for compatibility with some ADC on other STM32 families having this setting set by HW default value) */

/* Mask containing trigger source masks for each of possible                  */
/* trigger edge selection duplicated with shifts [0; 4; 8; 12]                */
/* corresponding to {SW start; ext trigger; ext trigger; ext trigger}.        */
#define ADC_REG_TRIG_SOURCE_MASK            (((LL_ADC_REG_TRIG_SOFTWARE & ADC_CR2_EXTSEL) >> (4UL * 0UL)) | \
                                             ((ADC_CR2_EXTSEL)                            >> (4UL * 1UL)) | \
                                             ((ADC_CR2_EXTSEL)                            >> (4UL * 2UL)) | \
                                             ((ADC_CR2_EXTSEL)                            >> (4UL * 3UL)))

/* Mask containing trigger edge masks for each of possible                    */
/* trigger edge selection duplicated with shifts [0; 4; 8; 12]                */
/* corresponding to {SW start; ext trigger; ext trigger; ext trigger}.        */
#define ADC_REG_TRIG_EDGE_MASK              (((LL_ADC_REG_TRIG_SOFTWARE & ADC_CR2_EXTEN) >> (4UL * 0UL)) | \
                                             ((ADC_REG_TRIG_EXT_EDGE_DEFAULT)            >> (4UL * 1UL)) | \
                                             ((ADC_REG_TRIG_EXT_EDGE_DEFAULT)            >> (4UL * 2UL)) | \
                                             ((ADC_REG_TRIG_EXT_EDGE_DEFAULT)            >> (4UL * 3UL)))

/* Definition of ADC group regular trigger bits information.                  */
#define ADC_REG_TRIG_EXTSEL_BITOFFSET_POS  (24UL) /* Value equivalent to POSITION_VAL(ADC_CR2_EXTSEL) */
#define ADC_REG_TRIG_EXTEN_BITOFFSET_POS   (28UL) /* Value equivalent to POSITION_VAL(ADC_CR2_EXTEN) */



/* Internal mask for ADC group injected trigger:                              */
/* To select into literal LL_ADC_INJ_TRIG_x the relevant bits for:            */
/* - injected trigger source                                                  */
/* - injected trigger edge                                                    */
#define ADC_INJ_TRIG_EXT_EDGE_DEFAULT      (ADC_CR2_JEXTEN_0) /* Trigger edge set to rising edge (default setting for compatibility with some ADC on other STM32 families having this setting set by HW default value) */

/* Mask containing trigger source masks for each of possible                  */
/* trigger edge selection duplicated with shifts [0; 4; 8; 12]                */
/* corresponding to {SW start; ext trigger; ext trigger; ext trigger}.        */
#define ADC_INJ_TRIG_SOURCE_MASK            (((LL_ADC_REG_TRIG_SOFTWARE & ADC_CR2_JEXTSEL) >> (4UL * 0UL)) | \
                                             ((ADC_CR2_JEXTSEL)                            >> (4UL * 1UL)) | \
                                             ((ADC_CR2_JEXTSEL)                            >> (4UL * 2UL)) | \
                                             ((ADC_CR2_JEXTSEL)                            >> (4UL * 3UL)))

/* Mask containing trigger edge masks for each of possible                    */
/* trigger edge selection duplicated with shifts [0; 4; 8; 12]                */
/* corresponding to {SW start; ext trigger; ext trigger; ext trigger}.        */
#define ADC_INJ_TRIG_EDGE_MASK              (((LL_ADC_INJ_TRIG_SOFTWARE & ADC_CR2_JEXTEN) >> (4UL * 0UL)) | \
                                             ((ADC_INJ_TRIG_EXT_EDGE_DEFAULT)             >> (4UL * 1UL)) | \
                                             ((ADC_INJ_TRIG_EXT_EDGE_DEFAULT)             >> (4UL * 2UL)) | \
                                             ((ADC_INJ_TRIG_EXT_EDGE_DEFAULT)             >> (4UL * 3UL)))

/* Definition of ADC group injected trigger bits information.                 */
#define ADC_INJ_TRIG_EXTSEL_BITOFFSET_POS  (16UL) /* Value equivalent to POSITION_VAL(ADC_CR2_JEXTSEL) */
#define ADC_INJ_TRIG_EXTEN_BITOFFSET_POS   (20UL) /* Value equivalent to POSITION_VAL(ADC_CR2_JEXTEN) */

/* Internal mask for ADC channel:                                             */
/* To select into literal LL_ADC_CHANNEL_x the relevant bits for:             */
/* - channel identifier defined by number                                     */
/* - channel differentiation between external channels (connected to          */
/*   GPIO pins) and internal channels (connected to internal paths)           */
/* - channel sampling time defined by SMPRx register offset                   */
/*   and SMPx bits positions into SMPRx register                              */
#define ADC_CHANNEL_ID_NUMBER_MASK         (ADC_CR1_AWDCH)
#define ADC_CHANNEL_ID_NUMBER_BITOFFSET_POS ( 0UL)/* Value equivalent to POSITION_VAL(ADC_CHANNEL_ID_NUMBER_MASK) */
#define ADC_CHANNEL_ID_MASK                (ADC_CHANNEL_ID_NUMBER_MASK | ADC_CHANNEL_ID_INTERNAL_CH_MASK)
/* Equivalent mask of ADC_CHANNEL_NUMBER_MASK aligned on register LSB (bit 0) */
#define ADC_CHANNEL_ID_NUMBER_MASK_POSBIT0 0x0000001FU /* Equivalent to shift: (ADC_CHANNEL_NUMBER_MASK >> POSITION_VAL(ADC_CHANNEL_NUMBER_MASK)) */

/* Channel differentiation between external and internal channels */
#define ADC_CHANNEL_ID_INTERNAL_CH         0x80000000UL   /* Marker of internal channel */
#define ADC_CHANNEL_ID_INTERNAL_CH_2       0x40000000UL   /* Marker of internal channel for other ADC instances, in case of different ADC internal channels mapped on same channel number on different ADC instances */
#define ADC_CHANNEL_DIFFERENCIATION_TEMPSENSOR_VBAT 0x10000000U  /* Dummy bit for driver internal usage, not used in ADC channel setting registers CR1 or SQRx */
#define ADC_CHANNEL_ID_INTERNAL_CH_MASK    (ADC_CHANNEL_ID_INTERNAL_CH | ADC_CHANNEL_ID_INTERNAL_CH_2 | ADC_CHANNEL_DIFFERENCIATION_TEMPSENSOR_VBAT)

/* Internal register offset for ADC channel sampling time configuration */
/* (offset placed into a spare area of literal definition) */
#define ADC_SMPR1_REGOFFSET                0x00000000UL
#define ADC_SMPR2_REGOFFSET                0x02000000UL
#define ADC_CHANNEL_SMPRX_REGOFFSET_MASK   (ADC_SMPR1_REGOFFSET | ADC_SMPR2_REGOFFSET)

#define ADC_CHANNEL_SMPx_BITOFFSET_MASK    0x01F00000UL
#define ADC_CHANNEL_SMPx_BITOFFSET_POS     (20UL)           /* Value equivalent to POSITION_VAL(ADC_CHANNEL_SMPx_BITOFFSET_MASK) */

/* Definition of channels ID number information to be inserted into           */
/* channels literals definition.                                              */
#define ADC_CHANNEL_0_NUMBER               0x00000000UL
#define ADC_CHANNEL_1_NUMBER               (                                                                        ADC_CR1_AWDCH_0)
#define ADC_CHANNEL_2_NUMBER               (                                                      ADC_CR1_AWDCH_1                  )
#define ADC_CHANNEL_3_NUMBER               (                                                      ADC_CR1_AWDCH_1 | ADC_CR1_AWDCH_0)
#define ADC_CHANNEL_4_NUMBER               (                                    ADC_CR1_AWDCH_2                                    )
#define ADC_CHANNEL_5_NUMBER               (                                    ADC_CR1_AWDCH_2                   | ADC_CR1_AWDCH_0)
#define ADC_CHANNEL_6_NUMBER               (                                    ADC_CR1_AWDCH_2 | ADC_CR1_AWDCH_1                  )
#define ADC_CHANNEL_7_NUMBER               (                                    ADC_CR1_AWDCH_2 | ADC_CR1_AWDCH_1 | ADC_CR1_AWDCH_0)
#define ADC_CHANNEL_8_NUMBER               (                  ADC_CR1_AWDCH_3                                                      )
#define ADC_CHANNEL_9_NUMBER               (                  ADC_CR1_AWDCH_3                                     | ADC_CR1_AWDCH_0)
#define ADC_CHANNEL_10_NUMBER              (                  ADC_CR1_AWDCH_3                   | ADC_CR1_AWDCH_1                  )
#define ADC_CHANNEL_11_NUMBER              (                  ADC_CR1_AWDCH_3                   | ADC_CR1_AWDCH_1 | ADC_CR1_AWDCH_0)
#define ADC_CHANNEL_12_NUMBER              (                  ADC_CR1_AWDCH_3 | ADC_CR1_AWDCH_2                                    )
#define ADC_CHANNEL_13_NUMBER              (                  ADC_CR1_AWDCH_3 | ADC_CR1_AWDCH_2                   | ADC_CR1_AWDCH_0)
#define ADC_CHANNEL_14_NUMBER              (                  ADC_CR1_AWDCH_3 | ADC_CR1_AWDCH_2 | ADC_CR1_AWDCH_1                  )
#define ADC_CHANNEL_15_NUMBER              (                  ADC_CR1_AWDCH_3 | ADC_CR1_AWDCH_2 | ADC_CR1_AWDCH_1 | ADC_CR1_AWDCH_0)
#define ADC_CHANNEL_16_NUMBER              (ADC_CR1_AWDCH_4                                                                        )
#define ADC_CHANNEL_17_NUMBER              (ADC_CR1_AWDCH_4                                                       | ADC_CR1_AWDCH_0)
#define ADC_CHANNEL_18_NUMBER              (ADC_CR1_AWDCH_4                                     | ADC_CR1_AWDCH_1                  )

/* Definition of channels sampling time information to be inserted into       */
/* channels literals definition.                                              */
#define ADC_CHANNEL_0_SMP                  (ADC_SMPR2_REGOFFSET | (( 0UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP0) */
#define ADC_CHANNEL_1_SMP                  (ADC_SMPR2_REGOFFSET | (( 3UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP1) */
#define ADC_CHANNEL_2_SMP                  (ADC_SMPR2_REGOFFSET | (( 6UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP2) */
#define ADC_CHANNEL_3_SMP                  (ADC_SMPR2_REGOFFSET | (( 9UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP3) */
#define ADC_CHANNEL_4_SMP                  (ADC_SMPR2_REGOFFSET | ((12UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP4) */
#define ADC_CHANNEL_5_SMP                  (ADC_SMPR2_REGOFFSET | ((15UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP5) */
#define ADC_CHANNEL_6_SMP                  (ADC_SMPR2_REGOFFSET | ((18UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP6) */
#define ADC_CHANNEL_7_SMP                  (ADC_SMPR2_REGOFFSET | ((21UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP7) */
#define ADC_CHANNEL_8_SMP                  (ADC_SMPR2_REGOFFSET | ((24UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP8) */
#define ADC_CHANNEL_9_SMP                  (ADC_SMPR2_REGOFFSET | ((27UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR2_SMP9) */
#define ADC_CHANNEL_10_SMP                 (ADC_SMPR1_REGOFFSET | (( 0UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR1_SMP10) */
#define ADC_CHANNEL_11_SMP                 (ADC_SMPR1_REGOFFSET | (( 3UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR1_SMP11) */
#define ADC_CHANNEL_12_SMP                 (ADC_SMPR1_REGOFFSET | (( 6UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR1_SMP12) */
#define ADC_CHANNEL_13_SMP                 (ADC_SMPR1_REGOFFSET | (( 9UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR1_SMP13) */
#define ADC_CHANNEL_14_SMP                 (ADC_SMPR1_REGOFFSET | ((12UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR1_SMP14) */
#define ADC_CHANNEL_15_SMP                 (ADC_SMPR1_REGOFFSET | ((15UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR1_SMP15) */
#define ADC_CHANNEL_16_SMP                 (ADC_SMPR1_REGOFFSET | ((18UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR1_SMP16) */
#define ADC_CHANNEL_17_SMP                 (ADC_SMPR1_REGOFFSET | ((21UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR1_SMP17) */
#define ADC_CHANNEL_18_SMP                 (ADC_SMPR1_REGOFFSET | ((24UL) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) /* Value shifted is equivalent to POSITION_VAL(ADC_SMPR1_SMP18) */

/* Internal mask for ADC analog watchdog:                                     */
/* To select into literals LL_ADC_AWD_CHANNELx_xxx the relevant bits for:     */
/* (concatenation of multiple bits used in different analog watchdogs,        */
/* (feature of several watchdogs not available on all STM32 families)).       */
/* - analog watchdog 1: monitored channel defined by number,                  */
/*   selection of ADC group (ADC groups regular and-or injected).             */

/* Internal register offset for ADC analog watchdog channel configuration */
#define ADC_AWD_CR1_REGOFFSET              0x00000000UL

#define ADC_AWD_CRX_REGOFFSET_MASK         (ADC_AWD_CR1_REGOFFSET)

#define ADC_AWD_CR1_CHANNEL_MASK           (ADC_CR1_AWDCH | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL)
#define ADC_AWD_CR_ALL_CHANNEL_MASK        (ADC_AWD_CR1_CHANNEL_MASK)

/* Internal register offset for ADC analog watchdog threshold configuration */
#define ADC_AWD_TR1_HIGH_REGOFFSET         0x00000000UL
#define ADC_AWD_TR1_LOW_REGOFFSET          0x00000001UL
#define ADC_AWD_TRX_REGOFFSET_MASK         (ADC_AWD_TR1_HIGH_REGOFFSET | ADC_AWD_TR1_LOW_REGOFFSET)

/* ADC registers bits positions */
#define ADC_CR1_RES_BITOFFSET_POS          (24UL) /* Value equivalent to POSITION_VAL(ADC_CR1_RES) */
#define ADC_TR_HT_BITOFFSET_POS            (16UL) /* Value equivalent to POSITION_VAL(ADC_TR_HT) */

/* ADC internal channels related definitions */
/* Internal voltage reference VrefInt */
#define VREFINT_CAL_ADDR                   ((uint16_t*) (0x1FFF7A2AU)) /* Internal voltage reference, address of parameter VREFINT_CAL: VrefInt ADC raw data acquired at temperature 30 DegC (tolerance: +-5 DegC), Vref+ = 3.3 V (tolerance: +-10 mV). */
#define VREFINT_CAL_VREF                   ( 3300UL)                    /* Analog voltage reference (Vref+) value with which temperature sensor has been calibrated in production (tolerance: +-10 mV) (unit: mV). */
/* Temperature sensor */
#define TEMPSENSOR_CAL1_ADDR               ((uint16_t*) (0x1FFF7A2CU)) /* Internal temperature sensor, address of parameter TS_CAL1: On STM32F4, temperature sensor ADC raw data acquired at temperature  30 DegC (tolerance: +-5 DegC), Vref+ = 3.3 V (tolerance: +-10 mV). */
#define TEMPSENSOR_CAL2_ADDR               ((uint16_t*) (0x1FFF7A2EU)) /* Internal temperature sensor, address of parameter TS_CAL2: On STM32F4, temperature sensor ADC raw data acquired at temperature 110 DegC (tolerance: +-5 DegC), Vref+ = 3.3 V (tolerance: +-10 mV). */
#define TEMPSENSOR_CAL1_TEMP               (( int32_t)   30)           /* Internal temperature sensor, temperature at which temperature sensor has been calibrated in production for data into TEMPSENSOR_CAL1_ADDR (tolerance: +-5 DegC) (unit: DegC). */
#define TEMPSENSOR_CAL2_TEMP               (( int32_t)  110)           /* Internal temperature sensor, temperature at which temperature sensor has been calibrated in production for data into TEMPSENSOR_CAL2_ADDR (tolerance: +-5 DegC) (unit: DegC). */
#define TEMPSENSOR_CAL_VREFANALOG          ( 3300UL)                    /* Analog voltage reference (Vref+) voltage with which temperature sensor has been calibrated in production (+-10 mV) (unit: mV). */

/**
  * @}
  */


/* Private macros ------------------------------------------------------------*/
/** @defgroup ADC_LL_Private_Macros ADC Private Macros
  * @{
  */

/**
  * @brief  Driver macro reserved for internal use: isolate bits with the
  *         selected mask and shift them to the register LSB
  *         (shift mask on register position bit 0).
  * @param  __BITS__ Bits in register 32 bits
  * @param  __MASK__ Mask in register 32 bits
  * @retval Bits in register 32 bits
  */
#define __ADC_MASK_SHIFT(__BITS__, __MASK__)                                   \
  (((__BITS__) & (__MASK__)) >> POSITION_VAL((__MASK__)))

/**
  * @brief  Driver macro reserved for internal use: set a pointer to
  *         a register from a register basis from which an offset
  *         is applied.
  * @param  __REG__ Register basis from which the offset is applied.
  * @param  __REG_OFFFSET__ Offset to be applied (unit number of registers).
  * @retval Pointer to register address
  */
#define __ADC_PTR_REG_OFFSET(__REG__, __REG_OFFFSET__)                         \
 ((__IO uint32_t *)((uint32_t) ((uint32_t)(&(__REG__)) + ((__REG_OFFFSET__) << 2UL))))

/**
  * @}
  */


/* Exported types ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup ADC_LL_ES_INIT ADC Exported Init structure
  * @{
  */

/**
  * @brief  Structure definition of some features of ADC common parameters
  *         and multimode
  *         (all ADC instances belonging to the same ADC common instance).
  * @note   The setting of these parameters by function @ref LL_ADC_CommonInit()
  *         is conditioned to ADC instances state (all ADC instances
  *         sharing the same ADC common instance):
  *         All ADC instances sharing the same ADC common instance must be
  *         disabled.
  */
typedef struct
{
  uint32_t CommonClock;                 /*!< Set parameter common to several ADC: Clock source and prescaler.
                                             This parameter can be a value of @ref ADC_LL_EC_COMMON_CLOCK_SOURCE
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_SetCommonClock(). */

#if defined(ADC_MULTIMODE_SUPPORT)
  uint32_t Multimode;                   /*!< Set ADC multimode configuration to operate in independent mode or multimode (for devices with several ADC instances).
                                             This parameter can be a value of @ref ADC_LL_EC_MULTI_MODE
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_SetMultimode(). */

  uint32_t MultiDMATransfer;            /*!< Set ADC multimode conversion data transfer: no transfer or transfer by DMA.
                                             This parameter can be a value of @ref ADC_LL_EC_MULTI_DMA_TRANSFER
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_SetMultiDMATransfer(). */

  uint32_t MultiTwoSamplingDelay;       /*!< Set ADC multimode delay between 2 sampling phases.
                                             This parameter can be a value of @ref ADC_LL_EC_MULTI_TWOSMP_DELAY
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_SetMultiTwoSamplingDelay(). */
#endif /* ADC_MULTIMODE_SUPPORT */

} LL_ADC_CommonInitTypeDef;

/**
  * @brief  Structure definition of some features of ADC instance.
  * @note   These parameters have an impact on ADC scope: ADC instance.
  *         Affects both group regular and group injected (availability
  *         of ADC group injected depends on STM32 families).
  *         Refer to corresponding unitary functions into
  *         @ref ADC_LL_EF_Configuration_ADC_Instance .
  * @note   The setting of these parameters by function @ref LL_ADC_Init()
  *         is conditioned to ADC state:
  *         ADC instance must be disabled.
  *         This condition is applied to all ADC features, for efficiency
  *         and compatibility over all STM32 families. However, the different
  *         features can be set under different ADC state conditions
  *         (setting possible with ADC enabled without conversion on going,
  *         ADC enabled with conversion on going, ...)
  *         Each feature can be updated afterwards with a unitary function
  *         and potentially with ADC in a different state than disabled,
  *         refer to description of each function for setting
  *         conditioned to ADC state.
  */
typedef struct
{
  uint32_t Resolution;                  /*!< Set ADC resolution.
                                             This parameter can be a value of @ref ADC_LL_EC_RESOLUTION
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_SetResolution(). */

  uint32_t DataAlignment;               /*!< Set ADC conversion data alignment.
                                             This parameter can be a value of @ref ADC_LL_EC_DATA_ALIGN
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_SetDataAlignment(). */

  uint32_t SequencersScanMode;          /*!< Set ADC scan selection.
                                             This parameter can be a value of @ref ADC_LL_EC_SCAN_SELECTION
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_SetSequencersScanMode(). */

} LL_ADC_InitTypeDef;

/**
  * @brief  Structure definition of some features of ADC group regular.
  * @note   These parameters have an impact on ADC scope: ADC group regular.
  *         Refer to corresponding unitary functions into
  *         @ref ADC_LL_EF_Configuration_ADC_Group_Regular
  *         (functions with prefix "REG").
  * @note   The setting of these parameters by function @ref LL_ADC_REG_Init()
  *         is conditioned to ADC state:
  *         ADC instance must be disabled.
  *         This condition is applied to all ADC features, for efficiency
  *         and compatibility over all STM32 families. However, the different
  *         features can be set under different ADC state conditions
  *         (setting possible with ADC enabled without conversion on going,
  *         ADC enabled with conversion on going, ...)
  *         Each feature can be updated afterwards with a unitary function
  *         and potentially with ADC in a different state than disabled,
  *         refer to description of each function for setting
  *         conditioned to ADC state.
  */
typedef struct
{
  uint32_t TriggerSource;               /*!< Set ADC group regular conversion trigger source: internal (SW start) or from external IP (timer event, external interrupt line).
                                             This parameter can be a value of @ref ADC_LL_EC_REG_TRIGGER_SOURCE
                                             @note On this STM32 series, setting of external trigger edge is performed
                                                   using function @ref LL_ADC_REG_StartConversionExtTrig().
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_REG_SetTriggerSource(). */

  uint32_t SequencerLength;             /*!< Set ADC group regular sequencer length.
                                             This parameter can be a value of @ref ADC_LL_EC_REG_SEQ_SCAN_LENGTH
                                             @note This parameter is discarded if scan mode is disabled (refer to parameter 'ADC_SequencersScanMode').
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_REG_SetSequencerLength(). */

  uint32_t SequencerDiscont;            /*!< Set ADC group regular sequencer discontinuous mode: sequence subdivided and scan conversions interrupted every selected number of ranks.
                                             This parameter can be a value of @ref ADC_LL_EC_REG_SEQ_DISCONT_MODE
                                             @note This parameter has an effect only if group regular sequencer is enabled
                                                   (scan length of 2 ranks or more).
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_REG_SetSequencerDiscont(). */

  uint32_t ContinuousMode;              /*!< Set ADC continuous conversion mode on ADC group regular, whether ADC conversions are performed in single mode (one conversion per trigger) or in continuous mode (after the first trigger, following conversions launched successively automatically).
                                             This parameter can be a value of @ref ADC_LL_EC_REG_CONTINUOUS_MODE
                                             Note: It is not possible to enable both ADC group regular continuous mode and discontinuous mode.
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_REG_SetContinuousMode(). */

  uint32_t DMATransfer;                 /*!< Set ADC group regular conversion data transfer: no transfer or transfer by DMA, and DMA requests mode.
                                             This parameter can be a value of @ref ADC_LL_EC_REG_DMA_TRANSFER
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_REG_SetDMATransfer(). */

} LL_ADC_REG_InitTypeDef;

/**
  * @brief  Structure definition of some features of ADC group injected.
  * @note   These parameters have an impact on ADC scope: ADC group injected.
  *         Refer to corresponding unitary functions into
  *         @ref ADC_LL_EF_Configuration_ADC_Group_Regular
  *         (functions with prefix "INJ").
  * @note   The setting of these parameters by function @ref LL_ADC_INJ_Init()
  *         is conditioned to ADC state:
  *         ADC instance must be disabled.
  *         This condition is applied to all ADC features, for efficiency
  *         and compatibility over all STM32 families. However, the different
  *         features can be set under different ADC state conditions
  *         (setting possible with ADC enabled without conversion on going,
  *         ADC enabled with conversion on going, ...)
  *         Each feature can be updated afterwards with a unitary function
  *         and potentially with ADC in a different state than disabled,
  *         refer to description of each function for setting
  *         conditioned to ADC state.
  */
typedef struct
{
  uint32_t TriggerSource;               /*!< Set ADC group injected conversion trigger source: internal (SW start) or from external IP (timer event, external interrupt line).
                                             This parameter can be a value of @ref ADC_LL_EC_INJ_TRIGGER_SOURCE
                                             @note On this STM32 series, setting of external trigger edge is performed
                                                   using function @ref LL_ADC_INJ_StartConversionExtTrig().
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_INJ_SetTriggerSource(). */

  uint32_t SequencerLength;             /*!< Set ADC group injected sequencer length.
                                             This parameter can be a value of @ref ADC_LL_EC_INJ_SEQ_SCAN_LENGTH
                                             @note This parameter is discarded if scan mode is disabled (refer to parameter 'ADC_SequencersScanMode').
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_INJ_SetSequencerLength(). */

  uint32_t SequencerDiscont;            /*!< Set ADC group injected sequencer discontinuous mode: sequence subdivided and scan conversions interrupted every selected number of ranks.
                                             This parameter can be a value of @ref ADC_LL_EC_INJ_SEQ_DISCONT_MODE
                                             @note This parameter has an effect only if group injected sequencer is enabled
                                                   (scan length of 2 ranks or more).
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_INJ_SetSequencerDiscont(). */

  uint32_t TrigAuto;                    /*!< Set ADC group injected conversion trigger: independent or from ADC group regular.
                                             This parameter can be a value of @ref ADC_LL_EC_INJ_TRIG_AUTO
                                             Note: This parameter must be set to set to independent trigger if injected trigger source is set to an external trigger. 
                                             
                                             This feature can be modified afterwards using unitary function @ref LL_ADC_INJ_SetTrigAuto(). */

} LL_ADC_INJ_InitTypeDef;

/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/* Exported constants --------------------------------------------------------*/
/** @defgroup ADC_LL_Exported_Constants ADC Exported Constants
  * @{
  */

/** @defgroup ADC_LL_EC_FLAG ADC flags
  * @brief    Flags defines which can be used with LL_ADC_ReadReg function
  * @{
  */
#define LL_ADC_FLAG_STRT                   ADC_SR_STRT        /*!< ADC flag ADC group regular conversion start */
#define LL_ADC_FLAG_EOCS                   ADC_SR_EOC         /*!< ADC flag ADC group regular end of unitary conversion or sequence conversions (to configure flag of end of conversion, use function @ref LL_ADC_REG_SetFlagEndOfConversion() ) */
#define LL_ADC_FLAG_OVR                    ADC_SR_OVR         /*!< ADC flag ADC group regular overrun */
#define LL_ADC_FLAG_JSTRT                  ADC_SR_JSTRT       /*!< ADC flag ADC group injected conversion start */
#define LL_ADC_FLAG_JEOS                   ADC_SR_JEOC        /*!< ADC flag ADC group injected end of sequence conversions (Note: on this STM32 series, there is no flag ADC group injected end of unitary conversion. Flag noted as "JEOC" is corresponding to flag "JEOS" in other STM32 families) */
#define LL_ADC_FLAG_AWD1                   ADC_SR_AWD         /*!< ADC flag ADC analog watchdog 1 */
#if defined(ADC_MULTIMODE_SUPPORT)
#define LL_ADC_FLAG_EOCS_MST               ADC_CSR_EOC1       /*!< ADC flag ADC multimode master group regular end of unitary conversion or sequence conversions (to configure flag of end of conversion, use function @ref LL_ADC_REG_SetFlagEndOfConversion() ) */
#define LL_ADC_FLAG_EOCS_SLV1              ADC_CSR_EOC2       /*!< ADC flag ADC multimode slave 1 group regular end of unitary conversion or sequence conversions (to configure flag of end of conversion, use function @ref LL_ADC_REG_SetFlagEndOfConversion() ) */
#define LL_ADC_FLAG_EOCS_SLV2              ADC_CSR_EOC3       /*!< ADC flag ADC multimode slave 2 group regular end of unitary conversion or sequence conversions (to configure flag of end of conversion, use function @ref LL_ADC_REG_SetFlagEndOfConversion() ) */
#define LL_ADC_FLAG_OVR_MST                ADC_CSR_OVR1    /*!< ADC flag ADC multimode master group regular overrun */ 
#define LL_ADC_FLAG_OVR_SLV1               ADC_CSR_OVR2   /*!< ADC flag ADC multimode slave 1 group regular overrun */
#define LL_ADC_FLAG_OVR_SLV2               ADC_CSR_OVR3   /*!< ADC flag ADC multimode slave 2 group regular overrun */
#define LL_ADC_FLAG_JEOS_MST               ADC_CSR_JEOC1     /*!< ADC flag ADC multimode master group injected end of sequence conversions (Note: on this STM32 series, there is no flag ADC group injected end of unitary conversion. Flag noted as "JEOC" is corresponding to flag "JEOS" in other STM32 families) */
#define LL_ADC_FLAG_JEOS_SLV1              ADC_CSR_JEOC2  /*!< ADC flag ADC multimode slave 1 group injected end of sequence conversions (Note: on this STM32 series, there is no flag ADC group injected end of unitary conversion. Flag noted as "JEOC" is corresponding to flag "JEOS" in other STM32 families) */
#define LL_ADC_FLAG_JEOS_SLV2              ADC_CSR_JEOC3  /*!< ADC flag ADC multimode slave 2 group injected end of sequence conversions (Note: on this STM32 series, there is no flag ADC group injected end of unitary conversion. Flag noted as "JEOC" is corresponding to flag "JEOS" in other STM32 families) */
#define LL_ADC_FLAG_AWD1_MST               ADC_CSR_AWD1       /*!< ADC flag ADC multimode master analog watchdog 1 of the ADC master */
#define LL_ADC_FLAG_AWD1_SLV1              ADC_CSR_AWD2       /*!< ADC flag ADC multimode slave 1 analog watchdog 1 */
#define LL_ADC_FLAG_AWD1_SLV2              ADC_CSR_AWD3       /*!< ADC flag ADC multimode slave 2 analog watchdog 1 */
#endif
/**
  * @}
  */

/** @defgroup ADC_LL_EC_IT ADC interruptions for configuration (interruption enable or disable)
  * @brief    IT defines which can be used with LL_ADC_ReadReg and  LL_ADC_WriteReg functions
  * @{
  */
#define LL_ADC_IT_EOCS                     ADC_CR1_EOCIE      /*!< ADC interruption ADC group regular end of unitary conversion or sequence conversions (to configure flag of end of conversion, use function @ref LL_ADC_REG_SetFlagEndOfConversion() ) */
#define LL_ADC_IT_OVR                      ADC_CR1_OVRIE      /*!< ADC interruption ADC group regular overrun */
#define LL_ADC_IT_JEOS                     ADC_CR1_JEOCIE     /*!< ADC interruption ADC group injected end of sequence conversions (Note: on this STM32 series, there is no flag ADC group injected end of unitary conversion. Flag noted as "JEOC" is corresponding to flag "JEOS" in other STM32 families) */
#define LL_ADC_IT_AWD1                     ADC_CR1_AWDIE      /*!< ADC interruption ADC analog watchdog 1 */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_REGISTERS  ADC registers compliant with specific purpose
  * @{
  */
/* List of ADC registers intended to be used (most commonly) with             */
/* DMA transfer.                                                              */
/* Refer to function @ref LL_ADC_DMA_GetRegAddr().                            */
#define LL_ADC_DMA_REG_REGULAR_DATA          0x00000000UL   /* ADC group regular conversion data register (corresponding to register DR) to be used with ADC configured in independent mode. Without DMA transfer, register accessed by LL function @ref LL_ADC_REG_ReadConversionData32() and other functions @ref LL_ADC_REG_ReadConversionDatax() */
#if defined(ADC_MULTIMODE_SUPPORT)
#define LL_ADC_DMA_REG_REGULAR_DATA_MULTI    0x00000001UL   /* ADC group regular conversion data register (corresponding to register CDR) to be used with ADC configured in multimode (available on STM32 devices with several ADC instances). Without DMA transfer, register accessed by LL function @ref LL_ADC_REG_ReadMultiConversionData32() */
#endif
/**
  * @}
  */

/** @defgroup ADC_LL_EC_COMMON_CLOCK_SOURCE  ADC common - Clock source
  * @{
  */
#define LL_ADC_CLOCK_SYNC_PCLK_DIV2        0x00000000UL                                           /*!< ADC synchronous clock derived from AHB clock with prescaler division by 2 */
#define LL_ADC_CLOCK_SYNC_PCLK_DIV4        (                   ADC_CCR_ADCPRE_0)                 /*!< ADC synchronous clock derived from AHB clock with prescaler division by 4 */
#define LL_ADC_CLOCK_SYNC_PCLK_DIV6        (ADC_CCR_ADCPRE_1                   )                 /*!< ADC synchronous clock derived from AHB clock with prescaler division by 6 */
#define LL_ADC_CLOCK_SYNC_PCLK_DIV8        (ADC_CCR_ADCPRE_1 | ADC_CCR_ADCPRE_0)                 /*!< ADC synchronous clock derived from AHB clock with prescaler division by 8 */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_COMMON_PATH_INTERNAL  ADC common - Measurement path to internal channels
  * @{
  */
/* Note: Other measurement paths to internal channels may be available        */
/*       (connections to other peripherals).                                  */
/*       If they are not listed below, they do not require any specific       */
/*       path enable. In this case, Access to measurement path is done        */
/*       only by selecting the corresponding ADC internal channel.            */
#define LL_ADC_PATH_INTERNAL_NONE          0x00000000UL            /*!< ADC measurement paths all disabled */
#define LL_ADC_PATH_INTERNAL_VREFINT       (ADC_CCR_TSVREFE)      /*!< ADC measurement path to internal channel VrefInt */
#define LL_ADC_PATH_INTERNAL_TEMPSENSOR    (ADC_CCR_TSVREFE)      /*!< ADC measurement path to internal channel temperature sensor */
#define LL_ADC_PATH_INTERNAL_VBAT          (ADC_CCR_VBATE)        /*!< ADC measurement path to internal channel Vbat */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_RESOLUTION  ADC instance - Resolution
  * @{
  */
#define LL_ADC_RESOLUTION_12B              0x00000000UL                         /*!< ADC resolution 12 bits */
#define LL_ADC_RESOLUTION_10B              (                ADC_CR1_RES_0)     /*!< ADC resolution 10 bits */
#define LL_ADC_RESOLUTION_8B               (ADC_CR1_RES_1                )     /*!< ADC resolution  8 bits */
#define LL_ADC_RESOLUTION_6B               (ADC_CR1_RES_1 | ADC_CR1_RES_0)     /*!< ADC resolution  6 bits */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_DATA_ALIGN  ADC instance - Data alignment
  * @{
  */
#define LL_ADC_DATA_ALIGN_RIGHT            0x00000000UL            /*!< ADC conversion data alignment: right aligned (alignment on data register LSB bit 0)*/
#define LL_ADC_DATA_ALIGN_LEFT             (ADC_CR2_ALIGN)        /*!< ADC conversion data alignment: left aligned (alignment on data register MSB bit 15)*/
/**
  * @}
  */

/** @defgroup ADC_LL_EC_SCAN_SELECTION ADC instance - Scan selection
  * @{
  */
#define LL_ADC_SEQ_SCAN_DISABLE            0x00000000UL    /*!< ADC conversion is performed in unitary conversion mode (one channel converted, that defined in rank 1). Configuration of both groups regular and injected sequencers (sequence length, ...) is discarded: equivalent to length of 1 rank.*/
#define LL_ADC_SEQ_SCAN_ENABLE             (ADC_CR1_SCAN) /*!< ADC conversions are performed in sequence conversions mode, according to configuration of both groups regular and injected sequencers (sequence length, ...). */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_GROUPS  ADC instance - Groups
  * @{
  */
#define LL_ADC_GROUP_REGULAR               0x00000001UL   /*!< ADC group regular (available on all STM32 devices) */
#define LL_ADC_GROUP_INJECTED              0x00000002UL   /*!< ADC group injected (not available on all STM32 devices)*/
#define LL_ADC_GROUP_REGULAR_INJECTED      0x00000003UL   /*!< ADC both groups regular and injected */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_CHANNEL  ADC instance - Channel number
  * @{
  */
#define LL_ADC_CHANNEL_0                   (ADC_CHANNEL_0_NUMBER  | ADC_CHANNEL_0_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN0  */
#define LL_ADC_CHANNEL_1                   (ADC_CHANNEL_1_NUMBER  | ADC_CHANNEL_1_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN1  */
#define LL_ADC_CHANNEL_2                   (ADC_CHANNEL_2_NUMBER  | ADC_CHANNEL_2_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN2  */
#define LL_ADC_CHANNEL_3                   (ADC_CHANNEL_3_NUMBER  | ADC_CHANNEL_3_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN3  */
#define LL_ADC_CHANNEL_4                   (ADC_CHANNEL_4_NUMBER  | ADC_CHANNEL_4_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN4  */
#define LL_ADC_CHANNEL_5                   (ADC_CHANNEL_5_NUMBER  | ADC_CHANNEL_5_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN5  */
#define LL_ADC_CHANNEL_6                   (ADC_CHANNEL_6_NUMBER  | ADC_CHANNEL_6_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN6  */
#define LL_ADC_CHANNEL_7                   (ADC_CHANNEL_7_NUMBER  | ADC_CHANNEL_7_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN7  */
#define LL_ADC_CHANNEL_8                   (ADC_CHANNEL_8_NUMBER  | ADC_CHANNEL_8_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN8  */
#define LL_ADC_CHANNEL_9                   (ADC_CHANNEL_9_NUMBER  | ADC_CHANNEL_9_SMP)  /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN9  */
#define LL_ADC_CHANNEL_10                  (ADC_CHANNEL_10_NUMBER | ADC_CHANNEL_10_SMP) /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN10 */
#define LL_ADC_CHANNEL_11                  (ADC_CHANNEL_11_NUMBER | ADC_CHANNEL_11_SMP) /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN11 */
#define LL_ADC_CHANNEL_12                  (ADC_CHANNEL_12_NUMBER | ADC_CHANNEL_12_SMP) /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN12 */
#define LL_ADC_CHANNEL_13                  (ADC_CHANNEL_13_NUMBER | ADC_CHANNEL_13_SMP) /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN13 */
#define LL_ADC_CHANNEL_14                  (ADC_CHANNEL_14_NUMBER | ADC_CHANNEL_14_SMP) /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN14 */
#define LL_ADC_CHANNEL_15                  (ADC_CHANNEL_15_NUMBER | ADC_CHANNEL_15_SMP) /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN15 */
#define LL_ADC_CHANNEL_16                  (ADC_CHANNEL_16_NUMBER | ADC_CHANNEL_16_SMP) /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN16 */
#define LL_ADC_CHANNEL_17                  (ADC_CHANNEL_17_NUMBER | ADC_CHANNEL_17_SMP) /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN17 */
#define LL_ADC_CHANNEL_18                  (ADC_CHANNEL_18_NUMBER | ADC_CHANNEL_18_SMP) /*!< ADC external channel (channel connected to GPIO pin) ADCx_IN18 */
#define LL_ADC_CHANNEL_VREFINT             (LL_ADC_CHANNEL_17 | ADC_CHANNEL_ID_INTERNAL_CH) /*!< ADC internal channel connected to VrefInt: Internal voltage reference. On STM32F4, ADC channel available only on ADC instance: ADC1. */
#define LL_ADC_CHANNEL_VBAT                (LL_ADC_CHANNEL_18 | ADC_CHANNEL_ID_INTERNAL_CH) /*!< ADC internal channel connected to Vbat/3: Vbat voltage through a divider ladder of factor 1/3 to have Vbat always below Vdda. On STM32F4, ADC channel available only on ADC instance: ADC1. */
#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F405xx) || defined(STM32F407xx) || defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F415xx) || defined(STM32F417xx)
#define LL_ADC_CHANNEL_TEMPSENSOR          (LL_ADC_CHANNEL_16 | ADC_CHANNEL_ID_INTERNAL_CH) /*!< ADC internal channel connected to Temperature sensor. On STM32F4, ADC channel available only on ADC instance: ADC1. */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F401xC || STM32F401xE || STM32F410xx */
#if defined(STM32F411xE) || defined(STM32F412Cx) || defined(STM32F412Rx) || defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx) || defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define LL_ADC_CHANNEL_TEMPSENSOR          (LL_ADC_CHANNEL_18 | ADC_CHANNEL_ID_INTERNAL_CH | ADC_CHANNEL_DIFFERENCIATION_TEMPSENSOR_VBAT) /*!< ADC internal channel connected to Temperature sensor. On STM32F4, ADC channel available only on ADC instance: ADC1. This internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled. */
#endif /* STM32F411xE || STM32F412Cx || STM32F412Rx || STM32F412Vx || STM32F412Zx || STM32F413xx || STM32F423xx || STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_REG_TRIGGER_SOURCE  ADC group regular - Trigger source
  * @{
  */
#define LL_ADC_REG_TRIG_SOFTWARE           0x00000000UL                                                                                                 /*!< ADC group regular conversion trigger internal: SW start. */
#define LL_ADC_REG_TRIG_EXT_TIM1_CH1       (ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                                                             /*!< ADC group regular conversion trigger from external IP: TIM1 channel 1 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM1_CH2       (ADC_CR2_EXTSEL_0 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                                          /*!< ADC group regular conversion trigger from external IP: TIM1 channel 2 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM1_CH3       (ADC_CR2_EXTSEL_1 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                                          /*!< ADC group regular conversion trigger from external IP: TIM1 channel 3 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM2_CH2       (ADC_CR2_EXTSEL_1 | ADC_CR2_EXTSEL_0 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                       /*!< ADC group regular conversion trigger from external IP: TIM2 channel 2 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM2_CH3       (ADC_CR2_EXTSEL_2 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                                          /*!< ADC group regular conversion trigger from external IP: TIM2 channel 3 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM2_CH4       (ADC_CR2_EXTSEL_2 | ADC_CR2_EXTSEL_0 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                       /*!< ADC group regular conversion trigger from external IP: TIM2 channel 4 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM2_TRGO      (ADC_CR2_EXTSEL_2 | ADC_CR2_EXTSEL_1 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                       /*!< ADC group regular conversion trigger from external IP: TIM2 TRGO. Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM3_CH1       (ADC_CR2_EXTSEL_2 | ADC_CR2_EXTSEL_1 | ADC_CR2_EXTSEL_0 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                    /*!< ADC group regular conversion trigger from external IP: TIM3 channel 1 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM3_TRGO      (ADC_CR2_EXTSEL_3 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                                          /*!< ADC group regular conversion trigger from external IP: TIM3 TRGO. Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM4_CH4       (ADC_CR2_EXTSEL_3 | ADC_CR2_EXTSEL_0 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                       /*!< ADC group regular conversion trigger from external IP: TIM4 channel 4 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM5_CH1       (ADC_CR2_EXTSEL_3 | ADC_CR2_EXTSEL_1 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                       /*!< ADC group regular conversion trigger from external IP: TIM5 channel 1 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM5_CH2       (ADC_CR2_EXTSEL_3 | ADC_CR2_EXTSEL_1 | ADC_CR2_EXTSEL_0 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                    /*!< ADC group regular conversion trigger from external IP: TIM5 channel 2 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM5_CH3       (ADC_CR2_EXTSEL_3 | ADC_CR2_EXTSEL_2 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                                       /*!< ADC group regular conversion trigger from external IP: TIM5 channel 3 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM8_CH1       (ADC_CR2_EXTSEL_3 | ADC_CR2_EXTSEL_2 | ADC_CR2_EXTSEL_0 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                    /*!< ADC group regular conversion trigger from external IP: TIM8 channel 1 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_TIM8_TRGO      (ADC_CR2_EXTSEL_3 | ADC_CR2_EXTSEL_2 | ADC_CR2_EXTSEL_1 | ADC_REG_TRIG_EXT_EDGE_DEFAULT)                    /*!< ADC group regular conversion trigger from external IP: TIM8 TRGO. Trigger edge set to rising edge (default setting). */
#define LL_ADC_REG_TRIG_EXT_EXTI_LINE11    (ADC_CR2_EXTSEL_3 | ADC_CR2_EXTSEL_2 | ADC_CR2_EXTSEL_1 | ADC_CR2_EXTSEL_0 | ADC_REG_TRIG_EXT_EDGE_DEFAULT) /*!< ADC group regular conversion trigger from external IP: external interrupt line 11. Trigger edge set to rising edge (default setting). */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_REG_TRIGGER_EDGE  ADC group regular - Trigger edge
  * @{
  */
#define LL_ADC_REG_TRIG_EXT_RISING         (                  ADC_CR2_EXTEN_0)     /*!< ADC group regular conversion trigger polarity set to rising edge */
#define LL_ADC_REG_TRIG_EXT_FALLING        (ADC_CR2_EXTEN_1                  )     /*!< ADC group regular conversion trigger polarity set to falling edge */
#define LL_ADC_REG_TRIG_EXT_RISINGFALLING  (ADC_CR2_EXTEN_1 | ADC_CR2_EXTEN_0)     /*!< ADC group regular conversion trigger polarity set to both rising and falling edges */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_REG_CONTINUOUS_MODE  ADC group regular - Continuous mode
* @{
*/
#define LL_ADC_REG_CONV_SINGLE             0x00000000UL             /*!< ADC conversions are performed in single mode: one conversion per trigger */
#define LL_ADC_REG_CONV_CONTINUOUS         (ADC_CR2_CONT)          /*!< ADC conversions are performed in continuous mode: after the first trigger, following conversions launched successively automatically */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_REG_DMA_TRANSFER  ADC group regular - DMA transfer of ADC conversion data
  * @{
  */
#define LL_ADC_REG_DMA_TRANSFER_NONE       0x00000000UL              /*!< ADC conversions are not transferred by DMA */
#define LL_ADC_REG_DMA_TRANSFER_LIMITED    (              ADC_CR2_DMA)          /*!< ADC conversion data are transferred by DMA, in limited mode (one shot mode): DMA transfer requests are stopped when number of DMA data transfers (number of ADC conversions) is reached. This ADC mode is intended to be used with DMA mode non-circular. */
#define LL_ADC_REG_DMA_TRANSFER_UNLIMITED  (ADC_CR2_DDS | ADC_CR2_DMA)          /*!< ADC conversion data are transferred by DMA, in unlimited mode: DMA transfer requests are unlimited, whatever number of DMA data transferred (number of ADC conversions). This ADC mode is intended to be used with DMA mode circular. */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_REG_FLAG_EOC_SELECTION ADC group regular - Flag EOC selection (unitary or sequence conversions)
  * @{
  */
#define LL_ADC_REG_FLAG_EOC_SEQUENCE_CONV       0x00000000UL    /*!< ADC flag EOC (end of unitary conversion) selected */
#define LL_ADC_REG_FLAG_EOC_UNITARY_CONV        (ADC_CR2_EOCS) /*!< ADC flag EOS (end of sequence conversions) selected */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_REG_SEQ_SCAN_LENGTH  ADC group regular - Sequencer scan length
  * @{
  */
#define LL_ADC_REG_SEQ_SCAN_DISABLE        0x00000000UL                                                 /*!< ADC group regular sequencer disable (equivalent to sequencer of 1 rank: ADC conversion on only 1 channel) */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_2RANKS  (                                             ADC_SQR1_L_0) /*!< ADC group regular sequencer enable with 2 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_3RANKS  (                              ADC_SQR1_L_1               ) /*!< ADC group regular sequencer enable with 3 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_4RANKS  (                              ADC_SQR1_L_1 | ADC_SQR1_L_0) /*!< ADC group regular sequencer enable with 4 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_5RANKS  (               ADC_SQR1_L_2                              ) /*!< ADC group regular sequencer enable with 5 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_6RANKS  (               ADC_SQR1_L_2                | ADC_SQR1_L_0) /*!< ADC group regular sequencer enable with 6 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_7RANKS  (               ADC_SQR1_L_2 | ADC_SQR1_L_1               ) /*!< ADC group regular sequencer enable with 7 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_8RANKS  (               ADC_SQR1_L_2 | ADC_SQR1_L_1 | ADC_SQR1_L_0) /*!< ADC group regular sequencer enable with 8 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_9RANKS  (ADC_SQR1_L_3                                             ) /*!< ADC group regular sequencer enable with 9 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_10RANKS (ADC_SQR1_L_3                               | ADC_SQR1_L_0) /*!< ADC group regular sequencer enable with 10 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_11RANKS (ADC_SQR1_L_3                | ADC_SQR1_L_1               ) /*!< ADC group regular sequencer enable with 11 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_12RANKS (ADC_SQR1_L_3                | ADC_SQR1_L_1 | ADC_SQR1_L_0) /*!< ADC group regular sequencer enable with 12 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_13RANKS (ADC_SQR1_L_3 | ADC_SQR1_L_2                              ) /*!< ADC group regular sequencer enable with 13 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_14RANKS (ADC_SQR1_L_3 | ADC_SQR1_L_2                | ADC_SQR1_L_0) /*!< ADC group regular sequencer enable with 14 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_15RANKS (ADC_SQR1_L_3 | ADC_SQR1_L_2 | ADC_SQR1_L_1               ) /*!< ADC group regular sequencer enable with 15 ranks in the sequence */
#define LL_ADC_REG_SEQ_SCAN_ENABLE_16RANKS (ADC_SQR1_L_3 | ADC_SQR1_L_2 | ADC_SQR1_L_1 | ADC_SQR1_L_0) /*!< ADC group regular sequencer enable with 16 ranks in the sequence */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_REG_SEQ_DISCONT_MODE  ADC group regular - Sequencer discontinuous mode
  * @{
  */
#define LL_ADC_REG_SEQ_DISCONT_DISABLE     0x00000000UL                                                                  /*!< ADC group regular sequencer discontinuous mode disable */
#define LL_ADC_REG_SEQ_DISCONT_1RANK       (                                                            ADC_CR1_DISCEN) /*!< ADC group regular sequencer discontinuous mode enable with sequence interruption every rank */
#define LL_ADC_REG_SEQ_DISCONT_2RANKS      (                                        ADC_CR1_DISCNUM_0 | ADC_CR1_DISCEN) /*!< ADC group regular sequencer discontinuous mode enabled with sequence interruption every 2 ranks */
#define LL_ADC_REG_SEQ_DISCONT_3RANKS      (                    ADC_CR1_DISCNUM_1                     | ADC_CR1_DISCEN) /*!< ADC group regular sequencer discontinuous mode enable with sequence interruption every 3 ranks */
#define LL_ADC_REG_SEQ_DISCONT_4RANKS      (                    ADC_CR1_DISCNUM_1 | ADC_CR1_DISCNUM_0 | ADC_CR1_DISCEN) /*!< ADC group regular sequencer discontinuous mode enable with sequence interruption every 4 ranks */
#define LL_ADC_REG_SEQ_DISCONT_5RANKS      (ADC_CR1_DISCNUM_2                                         | ADC_CR1_DISCEN) /*!< ADC group regular sequencer discontinuous mode enable with sequence interruption every 5 ranks */
#define LL_ADC_REG_SEQ_DISCONT_6RANKS      (ADC_CR1_DISCNUM_2                     | ADC_CR1_DISCNUM_0 | ADC_CR1_DISCEN) /*!< ADC group regular sequencer discontinuous mode enable with sequence interruption every 6 ranks */
#define LL_ADC_REG_SEQ_DISCONT_7RANKS      (ADC_CR1_DISCNUM_2 | ADC_CR1_DISCNUM_1                     | ADC_CR1_DISCEN) /*!< ADC group regular sequencer discontinuous mode enable with sequence interruption every 7 ranks */
#define LL_ADC_REG_SEQ_DISCONT_8RANKS      (ADC_CR1_DISCNUM_2 | ADC_CR1_DISCNUM_1 | ADC_CR1_DISCNUM_0 | ADC_CR1_DISCEN) /*!< ADC group regular sequencer discontinuous mode enable with sequence interruption every 8 ranks */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_REG_SEQ_RANKS  ADC group regular - Sequencer ranks
  * @{
  */
#define LL_ADC_REG_RANK_1                  (ADC_SQR3_REGOFFSET | ADC_REG_RANK_1_SQRX_BITOFFSET_POS)  /*!< ADC group regular sequencer rank 1 */
#define LL_ADC_REG_RANK_2                  (ADC_SQR3_REGOFFSET | ADC_REG_RANK_2_SQRX_BITOFFSET_POS)  /*!< ADC group regular sequencer rank 2 */
#define LL_ADC_REG_RANK_3                  (ADC_SQR3_REGOFFSET | ADC_REG_RANK_3_SQRX_BITOFFSET_POS)  /*!< ADC group regular sequencer rank 3 */
#define LL_ADC_REG_RANK_4                  (ADC_SQR3_REGOFFSET | ADC_REG_RANK_4_SQRX_BITOFFSET_POS)  /*!< ADC group regular sequencer rank 4 */
#define LL_ADC_REG_RANK_5                  (ADC_SQR3_REGOFFSET | ADC_REG_RANK_5_SQRX_BITOFFSET_POS)  /*!< ADC group regular sequencer rank 5 */
#define LL_ADC_REG_RANK_6                  (ADC_SQR3_REGOFFSET | ADC_REG_RANK_6_SQRX_BITOFFSET_POS)  /*!< ADC group regular sequencer rank 6 */
#define LL_ADC_REG_RANK_7                  (ADC_SQR2_REGOFFSET | ADC_REG_RANK_7_SQRX_BITOFFSET_POS)  /*!< ADC group regular sequencer rank 7 */
#define LL_ADC_REG_RANK_8                  (ADC_SQR2_REGOFFSET | ADC_REG_RANK_8_SQRX_BITOFFSET_POS)  /*!< ADC group regular sequencer rank 8 */
#define LL_ADC_REG_RANK_9                  (ADC_SQR2_REGOFFSET | ADC_REG_RANK_9_SQRX_BITOFFSET_POS)  /*!< ADC group regular sequencer rank 9 */
#define LL_ADC_REG_RANK_10                 (ADC_SQR2_REGOFFSET | ADC_REG_RANK_10_SQRX_BITOFFSET_POS) /*!< ADC group regular sequencer rank 10 */
#define LL_ADC_REG_RANK_11                 (ADC_SQR2_REGOFFSET | ADC_REG_RANK_11_SQRX_BITOFFSET_POS) /*!< ADC group regular sequencer rank 11 */
#define LL_ADC_REG_RANK_12                 (ADC_SQR2_REGOFFSET | ADC_REG_RANK_12_SQRX_BITOFFSET_POS) /*!< ADC group regular sequencer rank 12 */
#define LL_ADC_REG_RANK_13                 (ADC_SQR1_REGOFFSET | ADC_REG_RANK_13_SQRX_BITOFFSET_POS) /*!< ADC group regular sequencer rank 13 */
#define LL_ADC_REG_RANK_14                 (ADC_SQR1_REGOFFSET | ADC_REG_RANK_14_SQRX_BITOFFSET_POS) /*!< ADC group regular sequencer rank 14 */
#define LL_ADC_REG_RANK_15                 (ADC_SQR1_REGOFFSET | ADC_REG_RANK_15_SQRX_BITOFFSET_POS) /*!< ADC group regular sequencer rank 15 */
#define LL_ADC_REG_RANK_16                 (ADC_SQR1_REGOFFSET | ADC_REG_RANK_16_SQRX_BITOFFSET_POS) /*!< ADC group regular sequencer rank 16 */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_INJ_TRIGGER_SOURCE  ADC group injected - Trigger source
  * @{
  */
#define LL_ADC_INJ_TRIG_SOFTWARE           0x00000000UL                                                                                                     /*!< ADC group injected conversion trigger internal: SW start. */
#define LL_ADC_INJ_TRIG_EXT_TIM1_CH4       (ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                                                                 /*!< ADC group injected conversion trigger from external IP: TIM1 channel 4 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM1_TRGO      (ADC_CR2_JEXTSEL_0 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                                             /*!< ADC group injected conversion trigger from external IP: TIM1 TRGO. Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM2_CH1       (ADC_CR2_JEXTSEL_1 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                                             /*!< ADC group injected conversion trigger from external IP: TIM2 channel 1 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM2_TRGO      (ADC_CR2_JEXTSEL_1 | ADC_CR2_JEXTSEL_0 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                         /*!< ADC group injected conversion trigger from external IP: TIM2 TRGO. Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM3_CH2       (ADC_CR2_JEXTSEL_2 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                                             /*!< ADC group injected conversion trigger from external IP: TIM3 channel 2 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM3_CH4       (ADC_CR2_JEXTSEL_2 | ADC_CR2_JEXTSEL_0 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                         /*!< ADC group injected conversion trigger from external IP: TIM3 channel 4 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM4_CH1       (ADC_CR2_JEXTSEL_2 | ADC_CR2_JEXTSEL_1 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                         /*!< ADC group injected conversion trigger from external IP: TIM4 channel 1 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM4_CH2       (ADC_CR2_JEXTSEL_2 | ADC_CR2_JEXTSEL_1 | ADC_CR2_JEXTSEL_0 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                     /*!< ADC group injected conversion trigger from external IP: TIM4 channel 2 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM4_CH3       (ADC_CR2_JEXTSEL_3 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                                             /*!< ADC group injected conversion trigger from external IP: TIM4 channel 3 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM4_TRGO      (ADC_CR2_JEXTSEL_3 | ADC_CR2_JEXTSEL_0 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                         /*!< ADC group injected conversion trigger from external IP: TIM4 TRGO. Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM5_CH4       (ADC_CR2_JEXTSEL_3 | ADC_CR2_JEXTSEL_1 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                         /*!< ADC group injected conversion trigger from external IP: TIM5 channel 4 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM5_TRGO      (ADC_CR2_JEXTSEL_3 | ADC_CR2_JEXTSEL_1 | ADC_CR2_JEXTSEL_0 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                     /*!< ADC group injected conversion trigger from external IP: TIM5 TRGO. Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM8_CH2       (ADC_CR2_JEXTSEL_3 | ADC_CR2_JEXTSEL_2 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                                         /*!< ADC group injected conversion trigger from external IP: TIM8 channel 2 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM8_CH3       (ADC_CR2_JEXTSEL_3 | ADC_CR2_JEXTSEL_2 | ADC_CR2_JEXTSEL_0 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                     /*!< ADC group injected conversion trigger from external IP: TIM8 channel 3 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_TIM8_CH4       (ADC_CR2_JEXTSEL_3 | ADC_CR2_JEXTSEL_2 | ADC_CR2_JEXTSEL_1 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                     /*!< ADC group injected conversion trigger from external IP: TIM8 channel 4 event (capture compare: input capture or output capture). Trigger edge set to rising edge (default setting). */
#define LL_ADC_INJ_TRIG_EXT_EXTI_LINE15    (ADC_CR2_JEXTSEL_3 | ADC_CR2_JEXTSEL_2 | ADC_CR2_JEXTSEL_1 | ADC_CR2_JEXTSEL_0 | ADC_INJ_TRIG_EXT_EDGE_DEFAULT) /*!< ADC group injected conversion trigger from external IP: external interrupt line 15. Trigger edge set to rising edge (default setting). */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_INJ_TRIGGER_EDGE  ADC group injected - Trigger edge
  * @{
  */
#define LL_ADC_INJ_TRIG_EXT_RISING         (                   ADC_CR2_JEXTEN_0)   /*!< ADC group injected conversion trigger polarity set to rising edge */
#define LL_ADC_INJ_TRIG_EXT_FALLING        (ADC_CR2_JEXTEN_1                   )   /*!< ADC group injected conversion trigger polarity set to falling edge */
#define LL_ADC_INJ_TRIG_EXT_RISINGFALLING  (ADC_CR2_JEXTEN_1 | ADC_CR2_JEXTEN_0)   /*!< ADC group injected conversion trigger polarity set to both rising and falling edges */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_INJ_TRIG_AUTO  ADC group injected - Automatic trigger mode
* @{
*/
#define LL_ADC_INJ_TRIG_INDEPENDENT        0x00000000UL            /*!< ADC group injected conversion trigger independent. Setting mandatory if ADC group injected injected trigger source is set to an external trigger. */
#define LL_ADC_INJ_TRIG_FROM_GRP_REGULAR   (ADC_CR1_JAUTO)        /*!< ADC group injected conversion trigger from ADC group regular. Setting compliant only with group injected trigger source set to SW start, without any further action on  ADC group injected conversion start or stop: in this case, ADC group injected is controlled only from ADC group regular. */
/**
  * @}
  */


/** @defgroup ADC_LL_EC_INJ_SEQ_SCAN_LENGTH  ADC group injected - Sequencer scan length
  * @{
  */
#define LL_ADC_INJ_SEQ_SCAN_DISABLE        0x00000000UL                     /*!< ADC group injected sequencer disable (equivalent to sequencer of 1 rank: ADC conversion on only 1 channel) */
#define LL_ADC_INJ_SEQ_SCAN_ENABLE_2RANKS  (                ADC_JSQR_JL_0) /*!< ADC group injected sequencer enable with 2 ranks in the sequence */
#define LL_ADC_INJ_SEQ_SCAN_ENABLE_3RANKS  (ADC_JSQR_JL_1                ) /*!< ADC group injected sequencer enable with 3 ranks in the sequence */
#define LL_ADC_INJ_SEQ_SCAN_ENABLE_4RANKS  (ADC_JSQR_JL_1 | ADC_JSQR_JL_0) /*!< ADC group injected sequencer enable with 4 ranks in the sequence */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_INJ_SEQ_DISCONT_MODE  ADC group injected - Sequencer discontinuous mode
  * @{
  */
#define LL_ADC_INJ_SEQ_DISCONT_DISABLE     0x00000000UL            /*!< ADC group injected sequencer discontinuous mode disable */
#define LL_ADC_INJ_SEQ_DISCONT_1RANK       (ADC_CR1_JDISCEN)      /*!< ADC group injected sequencer discontinuous mode enable with sequence interruption every rank */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_INJ_SEQ_RANKS  ADC group injected - Sequencer ranks
  * @{
  */
#define LL_ADC_INJ_RANK_1                  (ADC_JDR1_REGOFFSET | ADC_JOFR1_REGOFFSET | 0x00000001UL) /*!< ADC group injected sequencer rank 1 */
#define LL_ADC_INJ_RANK_2                  (ADC_JDR2_REGOFFSET | ADC_JOFR2_REGOFFSET | 0x00000002UL) /*!< ADC group injected sequencer rank 2 */
#define LL_ADC_INJ_RANK_3                  (ADC_JDR3_REGOFFSET | ADC_JOFR3_REGOFFSET | 0x00000003UL) /*!< ADC group injected sequencer rank 3 */
#define LL_ADC_INJ_RANK_4                  (ADC_JDR4_REGOFFSET | ADC_JOFR4_REGOFFSET | 0x00000004UL) /*!< ADC group injected sequencer rank 4 */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_CHANNEL_SAMPLINGTIME  Channel - Sampling time
  * @{
  */
#define LL_ADC_SAMPLINGTIME_3CYCLES        0x00000000UL                                              /*!< Sampling time 3 ADC clock cycles */
#define LL_ADC_SAMPLINGTIME_15CYCLES       (ADC_SMPR1_SMP10_0)                                      /*!< Sampling time 15 ADC clock cycles */
#define LL_ADC_SAMPLINGTIME_28CYCLES       (ADC_SMPR1_SMP10_1)                                      /*!< Sampling time 28 ADC clock cycles */
#define LL_ADC_SAMPLINGTIME_56CYCLES       (ADC_SMPR1_SMP10_1 | ADC_SMPR1_SMP10_0)                  /*!< Sampling time 56 ADC clock cycles */
#define LL_ADC_SAMPLINGTIME_84CYCLES       (ADC_SMPR1_SMP10_2)                                      /*!< Sampling time 84 ADC clock cycles */
#define LL_ADC_SAMPLINGTIME_112CYCLES      (ADC_SMPR1_SMP10_2 | ADC_SMPR1_SMP10_0)                  /*!< Sampling time 112 ADC clock cycles */
#define LL_ADC_SAMPLINGTIME_144CYCLES      (ADC_SMPR1_SMP10_2 | ADC_SMPR1_SMP10_1)                  /*!< Sampling time 144 ADC clock cycles */
#define LL_ADC_SAMPLINGTIME_480CYCLES      (ADC_SMPR1_SMP10)                                        /*!< Sampling time 480 ADC clock cycles */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_AWD_NUMBER Analog watchdog - Analog watchdog number
  * @{
  */
#define LL_ADC_AWD1                        (ADC_AWD_CR1_CHANNEL_MASK  | ADC_AWD_CR1_REGOFFSET) /*!< ADC analog watchdog number 1 */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_AWD_CHANNELS  Analog watchdog - Monitored channels
  * @{
  */
#define LL_ADC_AWD_DISABLE                 0x00000000UL                                                                                   /*!< ADC analog watchdog monitoring disabled */
#define LL_ADC_AWD_ALL_CHANNELS_REG        (                                                             ADC_CR1_AWDEN                 ) /*!< ADC analog watchdog monitoring of all channels, converted by group regular only */
#define LL_ADC_AWD_ALL_CHANNELS_INJ        (                                            ADC_CR1_JAWDEN                                 ) /*!< ADC analog watchdog monitoring of all channels, converted by group injected only */
#define LL_ADC_AWD_ALL_CHANNELS_REG_INJ    (                                            ADC_CR1_JAWDEN | ADC_CR1_AWDEN                 ) /*!< ADC analog watchdog monitoring of all channels, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_0_REG           ((LL_ADC_CHANNEL_0  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN0, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_0_INJ           ((LL_ADC_CHANNEL_0  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN0, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_0_REG_INJ       ((LL_ADC_CHANNEL_0  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN0, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_1_REG           ((LL_ADC_CHANNEL_1  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN1, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_1_INJ           ((LL_ADC_CHANNEL_1  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN1, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_1_REG_INJ       ((LL_ADC_CHANNEL_1  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN1, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_2_REG           ((LL_ADC_CHANNEL_2  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN2, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_2_INJ           ((LL_ADC_CHANNEL_2  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN2, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_2_REG_INJ       ((LL_ADC_CHANNEL_2  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN2, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_3_REG           ((LL_ADC_CHANNEL_3  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN3, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_3_INJ           ((LL_ADC_CHANNEL_3  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN3, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_3_REG_INJ       ((LL_ADC_CHANNEL_3  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN3, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_4_REG           ((LL_ADC_CHANNEL_4  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN4, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_4_INJ           ((LL_ADC_CHANNEL_4  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN4, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_4_REG_INJ       ((LL_ADC_CHANNEL_4  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN4, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_5_REG           ((LL_ADC_CHANNEL_5  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN5, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_5_INJ           ((LL_ADC_CHANNEL_5  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN5, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_5_REG_INJ       ((LL_ADC_CHANNEL_5  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN5, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_6_REG           ((LL_ADC_CHANNEL_6  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN6, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_6_INJ           ((LL_ADC_CHANNEL_6  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN6, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_6_REG_INJ       ((LL_ADC_CHANNEL_6  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN6, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_7_REG           ((LL_ADC_CHANNEL_7  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN7, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_7_INJ           ((LL_ADC_CHANNEL_7  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN7, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_7_REG_INJ       ((LL_ADC_CHANNEL_7  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN7, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_8_REG           ((LL_ADC_CHANNEL_8  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN8, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_8_INJ           ((LL_ADC_CHANNEL_8  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN8, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_8_REG_INJ       ((LL_ADC_CHANNEL_8  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN8, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_9_REG           ((LL_ADC_CHANNEL_9  & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN9, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_9_INJ           ((LL_ADC_CHANNEL_9  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN9, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_9_REG_INJ       ((LL_ADC_CHANNEL_9  & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN9, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_10_REG          ((LL_ADC_CHANNEL_10 & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN10, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_10_INJ          ((LL_ADC_CHANNEL_10 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN10, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_10_REG_INJ      ((LL_ADC_CHANNEL_10 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN10, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_11_REG          ((LL_ADC_CHANNEL_11 & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN11, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_11_INJ          ((LL_ADC_CHANNEL_11 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN11, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_11_REG_INJ      ((LL_ADC_CHANNEL_11 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN11, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_12_REG          ((LL_ADC_CHANNEL_12 & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN12, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_12_INJ          ((LL_ADC_CHANNEL_12 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN12, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_12_REG_INJ      ((LL_ADC_CHANNEL_12 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN12, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_13_REG          ((LL_ADC_CHANNEL_13 & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN13, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_13_INJ          ((LL_ADC_CHANNEL_13 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN13, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_13_REG_INJ      ((LL_ADC_CHANNEL_13 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN13, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_14_REG          ((LL_ADC_CHANNEL_14 & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN14, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_14_INJ          ((LL_ADC_CHANNEL_14 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN14, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_14_REG_INJ      ((LL_ADC_CHANNEL_14 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN14, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_15_REG          ((LL_ADC_CHANNEL_15 & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN15, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_15_INJ          ((LL_ADC_CHANNEL_15 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN15, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_15_REG_INJ      ((LL_ADC_CHANNEL_15 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN15, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_16_REG          ((LL_ADC_CHANNEL_16 & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN16, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_16_INJ          ((LL_ADC_CHANNEL_16 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN16, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_16_REG_INJ      ((LL_ADC_CHANNEL_16 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN16, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_17_REG          ((LL_ADC_CHANNEL_17 & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN17, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_17_INJ          ((LL_ADC_CHANNEL_17 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN17, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_17_REG_INJ      ((LL_ADC_CHANNEL_17 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN17, converted by either group regular or injected */
#define LL_ADC_AWD_CHANNEL_18_REG          ((LL_ADC_CHANNEL_18 & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN18, converted by group regular only */
#define LL_ADC_AWD_CHANNEL_18_INJ          ((LL_ADC_CHANNEL_18 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN18, converted by group injected only */
#define LL_ADC_AWD_CHANNEL_18_REG_INJ      ((LL_ADC_CHANNEL_18 & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC external channel (channel connected to GPIO pin) ADCx_IN18, converted by either group regular or injected */
#define LL_ADC_AWD_CH_VREFINT_REG          ((LL_ADC_CHANNEL_VREFINT    & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to VrefInt: Internal voltage reference, converted by group regular only */
#define LL_ADC_AWD_CH_VREFINT_INJ          ((LL_ADC_CHANNEL_VREFINT    & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to VrefInt: Internal voltage reference, converted by group injected only */
#define LL_ADC_AWD_CH_VREFINT_REG_INJ      ((LL_ADC_CHANNEL_VREFINT    & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to VrefInt: Internal voltage reference, converted by either group regular or injected */
#define LL_ADC_AWD_CH_VBAT_REG             ((LL_ADC_CHANNEL_VBAT       & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to Vbat/3: Vbat voltage through a divider ladder of factor 1/3 to have Vbat always below Vdda, converted by group regular only */
#define LL_ADC_AWD_CH_VBAT_INJ             ((LL_ADC_CHANNEL_VBAT       & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to Vbat/3: Vbat voltage through a divider ladder of factor 1/3 to have Vbat always below Vdda, converted by group injected only */
#define LL_ADC_AWD_CH_VBAT_REG_INJ         ((LL_ADC_CHANNEL_VBAT       & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to Vbat/3: Vbat voltage through a divider ladder of factor 1/3 to have Vbat always below Vdda */
#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F405xx) || defined(STM32F407xx) || defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F415xx) || defined(STM32F417xx)
#define LL_ADC_AWD_CH_TEMPSENSOR_REG       ((LL_ADC_CHANNEL_TEMPSENSOR & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to Temperature sensor, converted by group regular only */
#define LL_ADC_AWD_CH_TEMPSENSOR_INJ       ((LL_ADC_CHANNEL_TEMPSENSOR & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to Temperature sensor, converted by group injected only */
#define LL_ADC_AWD_CH_TEMPSENSOR_REG_INJ   ((LL_ADC_CHANNEL_TEMPSENSOR & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to Temperature sensor, converted by either group regular or injected */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F401xC || STM32F401xE || STM32F410xx */
#if defined(STM32F411xE) || defined(STM32F412Cx) || defined(STM32F412Rx) || defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx) || defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define LL_ADC_AWD_CH_TEMPSENSOR_REG       ((LL_ADC_CHANNEL_TEMPSENSOR & ADC_CHANNEL_ID_MASK)                  | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to Temperature sensor, converted by group regular only. This internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled. */
#define LL_ADC_AWD_CH_TEMPSENSOR_INJ       ((LL_ADC_CHANNEL_TEMPSENSOR & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN                 | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to Temperature sensor, converted by group injected only. This internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled. */
#define LL_ADC_AWD_CH_TEMPSENSOR_REG_INJ   ((LL_ADC_CHANNEL_TEMPSENSOR & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL) /*!< ADC analog watchdog monitoring of ADC internal channel connected to Temperature sensor, converted by either group regular or injected. This internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled. */
#endif /* STM32F411xE || STM32F412Cx || STM32F412Rx || STM32F412Vx || STM32F412Zx || STM32F413xx || STM32F423xx || STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_AWD_THRESHOLDS  Analog watchdog - Thresholds
  * @{
  */
#define LL_ADC_AWD_THRESHOLD_HIGH          (ADC_AWD_TR1_HIGH_REGOFFSET) /*!< ADC analog watchdog threshold high */
#define LL_ADC_AWD_THRESHOLD_LOW           (ADC_AWD_TR1_LOW_REGOFFSET)  /*!< ADC analog watchdog threshold low */
/**
  * @}
  */

#if defined(ADC_MULTIMODE_SUPPORT)
/** @defgroup ADC_LL_EC_MULTI_MODE  Multimode - Mode
  * @{
  */
#define LL_ADC_MULTI_INDEPENDENT           0x00000000UL                                                             /*!< ADC dual mode disabled (ADC independent mode) */
#define LL_ADC_MULTI_DUAL_REG_SIMULT       (                  ADC_CCR_MULTI_2 | ADC_CCR_MULTI_1                  ) /*!< ADC dual mode enabled: group regular simultaneous */
#define LL_ADC_MULTI_DUAL_REG_INTERL       (                  ADC_CCR_MULTI_2 | ADC_CCR_MULTI_1 | ADC_CCR_MULTI_0) /*!< ADC dual mode enabled: Combined group regular interleaved */
#define LL_ADC_MULTI_DUAL_INJ_SIMULT       (                  ADC_CCR_MULTI_2                   | ADC_CCR_MULTI_0) /*!< ADC dual mode enabled: group injected simultaneous */
#define LL_ADC_MULTI_DUAL_INJ_ALTERN       (ADC_CCR_MULTI_3                                     | ADC_CCR_MULTI_0) /*!< ADC dual mode enabled: group injected alternate trigger. Works only with external triggers (not internal SW start) */
#define LL_ADC_MULTI_DUAL_REG_SIM_INJ_SIM  (                                                      ADC_CCR_MULTI_0) /*!< ADC dual mode enabled: Combined group regular simultaneous + group injected simultaneous */
#define LL_ADC_MULTI_DUAL_REG_SIM_INJ_ALT  (                                    ADC_CCR_MULTI_1                  ) /*!< ADC dual mode enabled: Combined group regular simultaneous + group injected alternate trigger */
#define LL_ADC_MULTI_DUAL_REG_INT_INJ_SIM  (                                    ADC_CCR_MULTI_1 | ADC_CCR_MULTI_0) /*!< ADC dual mode enabled: Combined group regular interleaved + group injected simultaneous */
#if defined(ADC3)
#define LL_ADC_MULTI_TRIPLE_REG_SIM_INJ_SIM  (ADC_CCR_MULTI_4                                                       | ADC_CCR_MULTI_0) /*!< ADC triple mode enabled: Combined group regular simultaneous + group injected simultaneous */
#define LL_ADC_MULTI_TRIPLE_REG_SIM_INJ_ALT  (ADC_CCR_MULTI_4                                     | ADC_CCR_MULTI_1                  ) /*!< ADC triple mode enabled: Combined group regular simultaneous + group injected alternate trigger */
#define LL_ADC_MULTI_TRIPLE_INJ_SIMULT       (ADC_CCR_MULTI_4                   | ADC_CCR_MULTI_2                   | ADC_CCR_MULTI_0) /*!< ADC triple mode enabled: group injected simultaneous */
#define LL_ADC_MULTI_TRIPLE_REG_SIMULT       (ADC_CCR_MULTI_4                   | ADC_CCR_MULTI_2 | ADC_CCR_MULTI_1                  ) /*!< ADC triple mode enabled: group regular simultaneous */
#define LL_ADC_MULTI_TRIPLE_REG_INTERL       (ADC_CCR_MULTI_4                   | ADC_CCR_MULTI_2 | ADC_CCR_MULTI_1 | ADC_CCR_MULTI_0) /*!< ADC triple mode enabled: Combined group regular interleaved */
#define LL_ADC_MULTI_TRIPLE_INJ_ALTERN       (ADC_CCR_MULTI_4                                                       | ADC_CCR_MULTI_0) /*!< ADC triple mode enabled: group injected alternate trigger. Works only with external triggers (not internal SW start) */
#endif
/**
  * @}
  */

/** @defgroup ADC_LL_EC_MULTI_DMA_TRANSFER  Multimode - DMA transfer
  * @{
  */
#define LL_ADC_MULTI_REG_DMA_EACH_ADC        0x00000000UL                                   /*!< ADC multimode group regular conversions are transferred by DMA: each ADC uses its own DMA channel, with its individual DMA transfer settings */
#define LL_ADC_MULTI_REG_DMA_LIMIT_1         (                              ADC_CCR_DMA_0) /*!< ADC multimode group regular conversions are transferred by DMA, one DMA channel for all ADC instances (DMA of ADC master), in limited mode (one shot mode): DMA transfer requests are stopped when number of DMA data transfers (number of ADC conversions) is reached. This ADC mode is intended to be used with DMA mode non-circular. Setting of DMA mode 1: 2 or 3 (dual or triple mode) half-words one by one, ADC1 then ADC2 then ADC3. */
#define LL_ADC_MULTI_REG_DMA_LIMIT_2         (              ADC_CCR_DMA_1                ) /*!< ADC multimode group regular conversions are transferred by DMA, one DMA channel for all ADC instances (DMA of ADC master), in limited mode (one shot mode): DMA transfer requests are stopped when number of DMA data transfers (number of ADC conversions) is reached. This ADC mode is intended to be used with DMA mode non-circular. Setting of DMA mode 2: 2 or 3 (dual or triple mode) half-words one by one, ADC2&1 then ADC1&3 then ADC3&2. */
#define LL_ADC_MULTI_REG_DMA_LIMIT_3         (              ADC_CCR_DMA_1 | ADC_CCR_DMA_0) /*!< ADC multimode group regular conversions are transferred by DMA, one DMA channel for all ADC instances (DMA of ADC master), in limited mode (one shot mode): DMA transfer requests are stopped when number of DMA data transfers (number of ADC conversions) is reached. This ADC mode is intended to be used with DMA mode non-circular. Setting of DMA mode 3: 2 or 3 (dual or triple mode) bytes one by one, ADC2&1 then ADC1&3 then ADC3&2. */
#define LL_ADC_MULTI_REG_DMA_UNLMT_1         (ADC_CCR_DDS |                 ADC_CCR_DMA_0) /*!< ADC multimode group regular conversions are transferred by DMA, one DMA channel for all ADC instances (DMA of ADC master), in unlimited mode: DMA transfer requests are unlimited, whatever number of DMA data transferred (number of ADC conversions) is reached. This ADC mode is intended to be used with DMA mode non-circular. Setting of DMA mode 1: 2 or 3 (dual or triple mode) half-words one by one, ADC1 then ADC2 then ADC3. */
#define LL_ADC_MULTI_REG_DMA_UNLMT_2         (ADC_CCR_DDS | ADC_CCR_DMA_1                ) /*!< ADC multimode group regular conversions are transferred by DMA, one DMA channel for all ADC instances (DMA of ADC master), in unlimited mode: DMA transfer requests are unlimited, whatever number of DMA data transferred (number of ADC conversions) is reached. This ADC mode is intended to be used with DMA mode non-circular. Setting of DMA mode 2: 2 or 3 (dual or triple mode) half-words by pairs, ADC2&1 then ADC1&3 then ADC3&2. */
#define LL_ADC_MULTI_REG_DMA_UNLMT_3         (ADC_CCR_DDS | ADC_CCR_DMA_1 | ADC_CCR_DMA_0) /*!< ADC multimode group regular conversions are transferred by DMA, one DMA channel for all ADC instances (DMA of ADC master), in unlimited mode: DMA transfer requests are unlimited, whatever number of DMA data transferred (number of ADC conversions) is reached. This ADC mode is intended to be used with DMA mode non-circular. Setting of DMA mode 3: 2 or 3 (dual or triple mode) bytes one by one, ADC2&1 then ADC1&3 then ADC3&2. */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_MULTI_TWOSMP_DELAY  Multimode - Delay between two sampling phases
  * @{
  */
#define LL_ADC_MULTI_TWOSMP_DELAY_5CYCLES  0x00000000UL                                                             /*!< ADC multimode delay between two sampling phases: 5 ADC clock cycles*/
#define LL_ADC_MULTI_TWOSMP_DELAY_6CYCLES  (                                                      ADC_CCR_DELAY_0) /*!< ADC multimode delay between two sampling phases: 6 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_7CYCLES  (                                    ADC_CCR_DELAY_1                  ) /*!< ADC multimode delay between two sampling phases: 7 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_8CYCLES  (                                    ADC_CCR_DELAY_1 | ADC_CCR_DELAY_0) /*!< ADC multimode delay between two sampling phases: 8 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_9CYCLES  (                  ADC_CCR_DELAY_2                                    ) /*!< ADC multimode delay between two sampling phases: 9 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_10CYCLES (                  ADC_CCR_DELAY_2                   | ADC_CCR_DELAY_0) /*!< ADC multimode delay between two sampling phases: 10 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_11CYCLES (                  ADC_CCR_DELAY_2 | ADC_CCR_DELAY_1                  ) /*!< ADC multimode delay between two sampling phases: 11 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_12CYCLES (                  ADC_CCR_DELAY_2 | ADC_CCR_DELAY_1 | ADC_CCR_DELAY_0) /*!< ADC multimode delay between two sampling phases: 12 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_13CYCLES (ADC_CCR_DELAY_3                                                      ) /*!< ADC multimode delay between two sampling phases: 13 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_14CYCLES (ADC_CCR_DELAY_3                                     | ADC_CCR_DELAY_0) /*!< ADC multimode delay between two sampling phases: 14 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_15CYCLES (ADC_CCR_DELAY_3                   | ADC_CCR_DELAY_1                  ) /*!< ADC multimode delay between two sampling phases: 15 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_16CYCLES (ADC_CCR_DELAY_3                   | ADC_CCR_DELAY_1 | ADC_CCR_DELAY_0) /*!< ADC multimode delay between two sampling phases: 16 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_17CYCLES (ADC_CCR_DELAY_3 | ADC_CCR_DELAY_2                                    ) /*!< ADC multimode delay between two sampling phases: 17 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_18CYCLES (ADC_CCR_DELAY_3 | ADC_CCR_DELAY_2                   | ADC_CCR_DELAY_0) /*!< ADC multimode delay between two sampling phases: 18 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_19CYCLES (ADC_CCR_DELAY_3 | ADC_CCR_DELAY_2 | ADC_CCR_DELAY_1                  ) /*!< ADC multimode delay between two sampling phases: 19 ADC clock cycles */
#define LL_ADC_MULTI_TWOSMP_DELAY_20CYCLES (ADC_CCR_DELAY_3 | ADC_CCR_DELAY_2 | ADC_CCR_DELAY_1 | ADC_CCR_DELAY_0) /*!< ADC multimode delay between two sampling phases: 20 ADC clock cycles */
/**
  * @}
  */

/** @defgroup ADC_LL_EC_MULTI_MASTER_SLAVE  Multimode - ADC master or slave
  * @{
  */
#define LL_ADC_MULTI_MASTER                (                    ADC_CDR_RDATA_MST) /*!< In multimode, selection among several ADC instances: ADC master */
#define LL_ADC_MULTI_SLAVE                 (ADC_CDR_RDATA_SLV                    ) /*!< In multimode, selection among several ADC instances: ADC slave */
#define LL_ADC_MULTI_MASTER_SLAVE          (ADC_CDR_RDATA_SLV | ADC_CDR_RDATA_MST) /*!< In multimode, selection among several ADC instances: both ADC master and ADC slave */
/**
  * @}
  */

#endif /* ADC_MULTIMODE_SUPPORT */


/** @defgroup ADC_LL_EC_HW_DELAYS  Definitions of ADC hardware constraints delays
  * @note   Only ADC IP HW delays are defined in ADC LL driver driver,
  *         not timeout values.
  *         For details on delays values, refer to descriptions in source code
  *         above each literal definition.
  * @{
  */
  
/* Note: Only ADC IP HW delays are defined in ADC LL driver driver,           */
/*       not timeout values.                                                  */
/*       Timeout values for ADC operations are dependent to device clock      */
/*       configuration (system clock versus ADC clock),                       */
/*       and therefore must be defined in user application.                   */
/*       Indications for estimation of ADC timeout delays, for this           */
/*       STM32 series:                                                        */
/*       - ADC enable time: maximum delay is 2us                              */
/*         (refer to device datasheet, parameter "tSTAB")                     */
/*       - ADC conversion time: duration depending on ADC clock and ADC       */
/*         configuration.                                                     */
/*         (refer to device reference manual, section "Timing")               */

/* Delay for internal voltage reference stabilization time.                   */
/* Delay set to maximum value (refer to device datasheet,                     */
/* parameter "tSTART").                                                       */
/* Unit: us                                                                   */
#define LL_ADC_DELAY_VREFINT_STAB_US       (  10UL)  /*!< Delay for internal voltage reference stabilization time */

/* Delay for temperature sensor stabilization time.                           */
/* Literal set to maximum value (refer to device datasheet,                   */
/* parameter "tSTART").                                                       */
/* Unit: us                                                                   */
#define LL_ADC_DELAY_TEMPSENSOR_STAB_US    (  10UL)  /*!< Delay for internal voltage reference stabilization time */

/**
  * @}
  */

/**
  * @}
  */


/* Exported macro ------------------------------------------------------------*/
/** @defgroup ADC_LL_Exported_Macros ADC Exported Macros
  * @{
  */

/** @defgroup ADC_LL_EM_WRITE_READ Common write and read registers Macros
  * @{
  */

/**
  * @brief  Write a value in ADC register
  * @param  __INSTANCE__ ADC Instance
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_ADC_WriteReg(__INSTANCE__, __REG__, __VALUE__) WRITE_REG(__INSTANCE__->__REG__, (__VALUE__))

/**
  * @brief  Read a value in ADC register
  * @param  __INSTANCE__ ADC Instance
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_ADC_ReadReg(__INSTANCE__, __REG__) READ_REG(__INSTANCE__->__REG__)
/**
  * @}
  */

/** @defgroup ADC_LL_EM_HELPER_MACRO ADC helper macro
  * @{
  */

/**
  * @brief  Helper macro to get ADC channel number in decimal format
  *         from literals LL_ADC_CHANNEL_x.
  * @note   Example:
  *           __LL_ADC_CHANNEL_TO_DECIMAL_NB(LL_ADC_CHANNEL_4)
  *           will return decimal number "4".
  * @note   The input can be a value from functions where a channel
  *         number is returned, either defined with number
  *         or with bitfield (only one bit must be set).
  * @param  __CHANNEL__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  * @retval Value between Min_Data=0 and Max_Data=18
  */
#define __LL_ADC_CHANNEL_TO_DECIMAL_NB(__CHANNEL__)                                        \
  (((__CHANNEL__) & ADC_CHANNEL_ID_NUMBER_MASK) >> ADC_CHANNEL_ID_NUMBER_BITOFFSET_POS)

/**
  * @brief  Helper macro to get ADC channel in literal format LL_ADC_CHANNEL_x
  *         from number in decimal format.
  * @note   Example:
  *           __LL_ADC_DECIMAL_NB_TO_CHANNEL(4)
  *           will return a data equivalent to "LL_ADC_CHANNEL_4".
  * @param  __DECIMAL_NB__ Value between Min_Data=0 and Max_Data=18
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.\n
  *         (1) For ADC channel read back from ADC register,
  *             comparison with internal channel parameter to be done
  *             using helper macro @ref __LL_ADC_CHANNEL_INTERNAL_TO_EXTERNAL().
  */
#define __LL_ADC_DECIMAL_NB_TO_CHANNEL(__DECIMAL_NB__)                                                          \
  (((__DECIMAL_NB__) <= 9UL)                                                                                     \
    ? (                                                                                                         \
       ((__DECIMAL_NB__) << ADC_CHANNEL_ID_NUMBER_BITOFFSET_POS)                                       |        \
       (ADC_SMPR2_REGOFFSET | (((uint32_t) (3UL * (__DECIMAL_NB__))) << ADC_CHANNEL_SMPx_BITOFFSET_POS))         \
      )                                                                                                         \
      :                                                                                                         \
      (                                                                                                         \
       ((__DECIMAL_NB__) << ADC_CHANNEL_ID_NUMBER_BITOFFSET_POS)                                              | \
       (ADC_SMPR1_REGOFFSET | (((uint32_t) (3UL * ((__DECIMAL_NB__) - 10UL))) << ADC_CHANNEL_SMPx_BITOFFSET_POS)) \
      )                                                                                                         \
  )

/**
  * @brief  Helper macro to determine whether the selected channel
  *         corresponds to literal definitions of driver.
  * @note   The different literal definitions of ADC channels are:
  *         - ADC internal channel:
  *           LL_ADC_CHANNEL_VREFINT, LL_ADC_CHANNEL_TEMPSENSOR, ...
  *         - ADC external channel (channel connected to a GPIO pin):
  *           LL_ADC_CHANNEL_1, LL_ADC_CHANNEL_2, ...
  * @note   The channel parameter must be a value defined from literal
  *         definition of a ADC internal channel (LL_ADC_CHANNEL_VREFINT,
  *         LL_ADC_CHANNEL_TEMPSENSOR, ...),
  *         ADC external channel (LL_ADC_CHANNEL_1, LL_ADC_CHANNEL_2, ...),
  *         must not be a value from functions where a channel number is
  *         returned from ADC registers,
  *         because internal and external channels share the same channel
  *         number in ADC registers. The differentiation is made only with
  *         parameters definitions of driver.
  * @param  __CHANNEL__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  * @retval Value "0" if the channel corresponds to a parameter definition of a ADC external channel (channel connected to a GPIO pin).
  *         Value "1" if the channel corresponds to a parameter definition of a ADC internal channel.
  */
#define __LL_ADC_IS_CHANNEL_INTERNAL(__CHANNEL__)                              \
  (((__CHANNEL__) & ADC_CHANNEL_ID_INTERNAL_CH_MASK) != 0UL)

/**
  * @brief  Helper macro to convert a channel defined from parameter
  *         definition of a ADC internal channel (LL_ADC_CHANNEL_VREFINT,
  *         LL_ADC_CHANNEL_TEMPSENSOR, ...),
  *         to its equivalent parameter definition of a ADC external channel
  *         (LL_ADC_CHANNEL_1, LL_ADC_CHANNEL_2, ...).
  * @note   The channel parameter can be, additionally to a value
  *         defined from parameter definition of a ADC internal channel
  *         (LL_ADC_CHANNEL_VREFINT, LL_ADC_CHANNEL_TEMPSENSOR, ...),
  *         a value defined from parameter definition of
  *         ADC external channel (LL_ADC_CHANNEL_1, LL_ADC_CHANNEL_2, ...)
  *         or a value from functions where a channel number is returned
  *         from ADC registers.
  * @param  __CHANNEL__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  */
#define __LL_ADC_CHANNEL_INTERNAL_TO_EXTERNAL(__CHANNEL__)                     \
  ((__CHANNEL__) & ~ADC_CHANNEL_ID_INTERNAL_CH_MASK)

/**
  * @brief  Helper macro to determine whether the internal channel
  *         selected is available on the ADC instance selected.
  * @note   The channel parameter must be a value defined from parameter
  *         definition of a ADC internal channel (LL_ADC_CHANNEL_VREFINT,
  *         LL_ADC_CHANNEL_TEMPSENSOR, ...),
  *         must not be a value defined from parameter definition of
  *         ADC external channel (LL_ADC_CHANNEL_1, LL_ADC_CHANNEL_2, ...)
  *         or a value from functions where a channel number is
  *         returned from ADC registers,
  *         because internal and external channels share the same channel
  *         number in ADC registers. The differentiation is made only with
  *         parameters definitions of driver.
  * @param  __ADC_INSTANCE__ ADC instance
  * @param  __CHANNEL__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  * @retval Value "0" if the internal channel selected is not available on the ADC instance selected.
  *         Value "1" if the internal channel selected is available on the ADC instance selected.
  */
#define __LL_ADC_IS_CHANNEL_INTERNAL_AVAILABLE(__ADC_INSTANCE__, __CHANNEL__)  \
  (                                                                            \
   ((__CHANNEL__) == LL_ADC_CHANNEL_VREFINT)    ||                             \
   ((__CHANNEL__) == LL_ADC_CHANNEL_TEMPSENSOR) ||                             \
   ((__CHANNEL__) == LL_ADC_CHANNEL_VBAT)                                      \
  )
/**
  * @brief  Helper macro to define ADC analog watchdog parameter:
  *         define a single channel to monitor with analog watchdog
  *         from sequencer channel and groups definition.
  * @note   To be used with function @ref LL_ADC_SetAnalogWDMonitChannels().
  *         Example:
  *           LL_ADC_SetAnalogWDMonitChannels(
  *             ADC1, LL_ADC_AWD1,
  *             __LL_ADC_ANALOGWD_CHANNEL_GROUP(LL_ADC_CHANNEL4, LL_ADC_GROUP_REGULAR))
  * @param  __CHANNEL__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.\n
  *         (1) For ADC channel read back from ADC register,
  *             comparison with internal channel parameter to be done
  *             using helper macro @ref __LL_ADC_CHANNEL_INTERNAL_TO_EXTERNAL().
  * @param  __GROUP__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_GROUP_REGULAR
  *         @arg @ref LL_ADC_GROUP_INJECTED
  *         @arg @ref LL_ADC_GROUP_REGULAR_INJECTED
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_AWD_DISABLE
  *         @arg @ref LL_ADC_AWD_ALL_CHANNELS_REG
  *         @arg @ref LL_ADC_AWD_ALL_CHANNELS_INJ
  *         @arg @ref LL_ADC_AWD_ALL_CHANNELS_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_0_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_0_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_0_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_1_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_1_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_1_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_2_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_2_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_2_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_3_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_3_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_3_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_4_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_4_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_4_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_5_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_5_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_5_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_6_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_6_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_6_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_7_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_7_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_7_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_8_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_8_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_8_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_9_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_9_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_9_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_10_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_10_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_10_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_11_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_11_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_11_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_12_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_12_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_12_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_13_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_13_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_13_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_14_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_14_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_14_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_15_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_15_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_15_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_16_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_16_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_16_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_17_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_17_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_17_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_18_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_18_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_18_REG_INJ
  *         @arg @ref LL_ADC_AWD_CH_VREFINT_REG          (1)
  *         @arg @ref LL_ADC_AWD_CH_VREFINT_INJ          (1)
  *         @arg @ref LL_ADC_AWD_CH_VREFINT_REG_INJ      (1)
  *         @arg @ref LL_ADC_AWD_CH_TEMPSENSOR_REG       (1)(2)
  *         @arg @ref LL_ADC_AWD_CH_TEMPSENSOR_INJ       (1)(2)
  *         @arg @ref LL_ADC_AWD_CH_TEMPSENSOR_REG_INJ   (1)(2)
  *         @arg @ref LL_ADC_AWD_CH_VBAT_REG             (1)
  *         @arg @ref LL_ADC_AWD_CH_VBAT_INJ             (1)
  *         @arg @ref LL_ADC_AWD_CH_VBAT_REG_INJ         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  */
#define __LL_ADC_ANALOGWD_CHANNEL_GROUP(__CHANNEL__, __GROUP__)                                           \
  (((__GROUP__) == LL_ADC_GROUP_REGULAR)                                                                  \
    ? (((__CHANNEL__) & ADC_CHANNEL_ID_MASK) | ADC_CR1_AWDEN | ADC_CR1_AWDSGL)                            \
      :                                                                                                   \
      ((__GROUP__) == LL_ADC_GROUP_INJECTED)                                                              \
       ? (((__CHANNEL__) & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDSGL)                        \
         :                                                                                                \
         (((__CHANNEL__) & ADC_CHANNEL_ID_MASK) | ADC_CR1_JAWDEN | ADC_CR1_AWDEN | ADC_CR1_AWDSGL)        \
  )

/**
  * @brief  Helper macro to set the value of ADC analog watchdog threshold high
  *         or low in function of ADC resolution, when ADC resolution is
  *         different of 12 bits.
  * @note   To be used with function @ref LL_ADC_SetAnalogWDThresholds().
  *         Example, with a ADC resolution of 8 bits, to set the value of
  *         analog watchdog threshold high (on 8 bits):
  *           LL_ADC_SetAnalogWDThresholds
  *            (< ADCx param >,
  *             __LL_ADC_ANALOGWD_SET_THRESHOLD_RESOLUTION(LL_ADC_RESOLUTION_8B, <threshold_value_8_bits>)
  *            );
  * @param  __ADC_RESOLUTION__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @param  __AWD_THRESHOLD__ Value between Min_Data=0x000 and Max_Data=0xFFF
  * @retval Value between Min_Data=0x000 and Max_Data=0xFFF
  */
#define __LL_ADC_ANALOGWD_SET_THRESHOLD_RESOLUTION(__ADC_RESOLUTION__, __AWD_THRESHOLD__) \
  ((__AWD_THRESHOLD__) << ((__ADC_RESOLUTION__) >> (ADC_CR1_RES_BITOFFSET_POS - 1UL )))

/**
  * @brief  Helper macro to get the value of ADC analog watchdog threshold high
  *         or low in function of ADC resolution, when ADC resolution is 
  *         different of 12 bits.
  * @note   To be used with function @ref LL_ADC_GetAnalogWDThresholds().
  *         Example, with a ADC resolution of 8 bits, to get the value of
  *         analog watchdog threshold high (on 8 bits):
  *           < threshold_value_6_bits > = __LL_ADC_ANALOGWD_GET_THRESHOLD_RESOLUTION
  *            (LL_ADC_RESOLUTION_8B,
  *             LL_ADC_GetAnalogWDThresholds(<ADCx param>, LL_ADC_AWD_THRESHOLD_HIGH)
  *            );
  * @param  __ADC_RESOLUTION__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @param  __AWD_THRESHOLD_12_BITS__ Value between Min_Data=0x000 and Max_Data=0xFFF
  * @retval Value between Min_Data=0x000 and Max_Data=0xFFF
  */
#define __LL_ADC_ANALOGWD_GET_THRESHOLD_RESOLUTION(__ADC_RESOLUTION__, __AWD_THRESHOLD_12_BITS__) \
  ((__AWD_THRESHOLD_12_BITS__) >> ((__ADC_RESOLUTION__) >> (ADC_CR1_RES_BITOFFSET_POS - 1UL )))

#if defined(ADC_MULTIMODE_SUPPORT)
/**
  * @brief  Helper macro to get the ADC multimode conversion data of ADC master
  *         or ADC slave from raw value with both ADC conversion data concatenated.
  * @note   This macro is intended to be used when multimode transfer by DMA
  *         is enabled: refer to function @ref LL_ADC_SetMultiDMATransfer().
  *         In this case the transferred data need to processed with this macro
  *         to separate the conversion data of ADC master and ADC slave.
  * @param  __ADC_MULTI_MASTER_SLAVE__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_MULTI_MASTER
  *         @arg @ref LL_ADC_MULTI_SLAVE
  * @param  __ADC_MULTI_CONV_DATA__ Value between Min_Data=0x000 and Max_Data=0xFFF
  * @retval Value between Min_Data=0x000 and Max_Data=0xFFF
  */
#define __LL_ADC_MULTI_CONV_DATA_MASTER_SLAVE(__ADC_MULTI_MASTER_SLAVE__, __ADC_MULTI_CONV_DATA__)  \
  (((__ADC_MULTI_CONV_DATA__) >> POSITION_VAL((__ADC_MULTI_MASTER_SLAVE__))) & ADC_CDR_RDATA_MST)
#endif

/**
  * @brief  Helper macro to select the ADC common instance
  *         to which is belonging the selected ADC instance.
  * @note   ADC common register instance can be used for:
  *         - Set parameters common to several ADC instances
  *         - Multimode (for devices with several ADC instances)
  *         Refer to functions having argument "ADCxy_COMMON" as parameter.
  * @param  __ADCx__ ADC instance
  * @retval ADC common register instance
  */
#if defined(ADC1) && defined(ADC2) && defined(ADC3)
#define __LL_ADC_COMMON_INSTANCE(__ADCx__)                                     \
  (ADC123_COMMON)
#elif defined(ADC1) && defined(ADC2)
#define __LL_ADC_COMMON_INSTANCE(__ADCx__)                                     \
  (ADC12_COMMON)
#else
#define __LL_ADC_COMMON_INSTANCE(__ADCx__)                                     \
  (ADC1_COMMON)
#endif

/**
  * @brief  Helper macro to check if all ADC instances sharing the same
  *         ADC common instance are disabled.
  * @note   This check is required by functions with setting conditioned to
  *         ADC state:
  *         All ADC instances of the ADC common group must be disabled.
  *         Refer to functions having argument "ADCxy_COMMON" as parameter.
  * @note   On devices with only 1 ADC common instance, parameter of this macro
  *         is useless and can be ignored (parameter kept for compatibility
  *         with devices featuring several ADC common instances).
  * @param  __ADCXY_COMMON__ ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval Value "0" if all ADC instances sharing the same ADC common instance
  *         are disabled.
  *         Value "1" if at least one ADC instance sharing the same ADC common instance
  *         is enabled.
  */
#if defined(ADC1) && defined(ADC2) && defined(ADC3)
#define __LL_ADC_IS_ENABLED_ALL_COMMON_INSTANCE(__ADCXY_COMMON__)              \
  (LL_ADC_IsEnabled(ADC1) |                                                    \
   LL_ADC_IsEnabled(ADC2) |                                                    \
   LL_ADC_IsEnabled(ADC3)  )
#elif defined(ADC1) && defined(ADC2)
#define __LL_ADC_IS_ENABLED_ALL_COMMON_INSTANCE(__ADCXY_COMMON__)              \
  (LL_ADC_IsEnabled(ADC1) |                                                    \
   LL_ADC_IsEnabled(ADC2)  )
#else
#define __LL_ADC_IS_ENABLED_ALL_COMMON_INSTANCE(__ADCXY_COMMON__)              \
  (LL_ADC_IsEnabled(ADC1))
#endif

/**
  * @brief  Helper macro to define the ADC conversion data full-scale digital
  *         value corresponding to the selected ADC resolution.
  * @note   ADC conversion data full-scale corresponds to voltage range
  *         determined by analog voltage references Vref+ and Vref-
  *         (refer to reference manual).
  * @param  __ADC_RESOLUTION__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @retval ADC conversion data equivalent voltage value (unit: mVolt)
  */
#define __LL_ADC_DIGITAL_SCALE(__ADC_RESOLUTION__)                             \
  (0xFFFU >> ((__ADC_RESOLUTION__) >> (ADC_CR1_RES_BITOFFSET_POS - 1UL)))

/**
  * @brief  Helper macro to convert the ADC conversion data from
  *         a resolution to another resolution.
  * @param  __DATA__ ADC conversion data to be converted 
  * @param  __ADC_RESOLUTION_CURRENT__ Resolution of to the data to be converted
  *         This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @param  __ADC_RESOLUTION_TARGET__ Resolution of the data after conversion
  *         This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @retval ADC conversion data to the requested resolution
  */
#define __LL_ADC_CONVERT_DATA_RESOLUTION(__DATA__, __ADC_RESOLUTION_CURRENT__, __ADC_RESOLUTION_TARGET__) \
  (((__DATA__)                                                                 \
    << ((__ADC_RESOLUTION_CURRENT__) >> (ADC_CR1_RES_BITOFFSET_POS - 1UL)))     \
   >> ((__ADC_RESOLUTION_TARGET__) >> (ADC_CR1_RES_BITOFFSET_POS - 1UL))        \
  )

/**
  * @brief  Helper macro to calculate the voltage (unit: mVolt)
  *         corresponding to a ADC conversion data (unit: digital value).
  * @note   Analog reference voltage (Vref+) must be either known from
  *         user board environment or can be calculated using ADC measurement
  *         and ADC helper macro @ref __LL_ADC_CALC_VREFANALOG_VOLTAGE().
  * @param  __VREFANALOG_VOLTAGE__ Analog reference voltage (unit mV)
  * @param  __ADC_DATA__ ADC conversion data (resolution 12 bits)
  *                       (unit: digital value).
  * @param  __ADC_RESOLUTION__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @retval ADC conversion data equivalent voltage value (unit: mVolt)
  */
#define __LL_ADC_CALC_DATA_TO_VOLTAGE(__VREFANALOG_VOLTAGE__,\
                                      __ADC_DATA__,\
                                      __ADC_RESOLUTION__)                      \
  ((__ADC_DATA__) * (__VREFANALOG_VOLTAGE__)                                   \
   / __LL_ADC_DIGITAL_SCALE(__ADC_RESOLUTION__)                                \
  )

/**
  * @brief  Helper macro to calculate analog reference voltage (Vref+)
  *         (unit: mVolt) from ADC conversion data of internal voltage
  *         reference VrefInt.
  * @note   Computation is using VrefInt calibration value
  *         stored in system memory for each device during production.
  * @note   This voltage depends on user board environment: voltage level
  *         connected to pin Vref+.
  *         On devices with small package, the pin Vref+ is not present
  *         and internally bonded to pin Vdda.
  * @note   On this STM32 series, calibration data of internal voltage reference
  *         VrefInt corresponds to a resolution of 12 bits,
  *         this is the recommended ADC resolution to convert voltage of
  *         internal voltage reference VrefInt.
  *         Otherwise, this macro performs the processing to scale
  *         ADC conversion data to 12 bits.
  * @param  __VREFINT_ADC_DATA__ ADC conversion data (resolution 12 bits)
  *         of internal voltage reference VrefInt (unit: digital value).
  * @param  __ADC_RESOLUTION__ This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @retval Analog reference voltage (unit: mV)
  */
#define __LL_ADC_CALC_VREFANALOG_VOLTAGE(__VREFINT_ADC_DATA__,\
                                         __ADC_RESOLUTION__)                   \
  (((uint32_t)(*VREFINT_CAL_ADDR) * VREFINT_CAL_VREF)                          \
   / __LL_ADC_CONVERT_DATA_RESOLUTION((__VREFINT_ADC_DATA__),                  \
                                      (__ADC_RESOLUTION__),                    \
                                      LL_ADC_RESOLUTION_12B))

/* Note: On device STM32F4x9, calibration parameter TS_CAL2 is not available. */
/*       Therefore, helper macro __LL_ADC_CALC_TEMPERATURE() is not available.*/
/*       Use helper macro @ref __LL_ADC_CALC_TEMPERATURE_TYP_PARAMS().        */
#if !defined(STM32F469) && !defined(STM32F479xx) && !defined(STM32F429xx) && !defined(STM32F439xx)
/**
  * @brief  Helper macro to calculate the temperature (unit: degree Celsius)
  *         from ADC conversion data of internal temperature sensor.
  * @note   Computation is using temperature sensor calibration values
  *         stored in system memory for each device during production.
  * @note   Calculation formula:
  *           Temperature = ((TS_ADC_DATA - TS_CAL1)
  *                           * (TS_CAL2_TEMP - TS_CAL1_TEMP))
  *                         / (TS_CAL2 - TS_CAL1) + TS_CAL1_TEMP
  *           with TS_ADC_DATA = temperature sensor raw data measured by ADC
  *                Avg_Slope = (TS_CAL2 - TS_CAL1)
  *                            / (TS_CAL2_TEMP - TS_CAL1_TEMP)
  *                TS_CAL1   = equivalent TS_ADC_DATA at temperature
  *                            TEMP_DEGC_CAL1 (calibrated in factory)
  *                TS_CAL2   = equivalent TS_ADC_DATA at temperature
  *                            TEMP_DEGC_CAL2 (calibrated in factory)
  *         Caution: Calculation relevancy under reserve that calibration
  *                  parameters are correct (address and data).
  *                  To calculate temperature using temperature sensor
  *                  datasheet typical values (generic values less, therefore
  *                  less accurate than calibrated values),
  *                  use helper macro @ref __LL_ADC_CALC_TEMPERATURE_TYP_PARAMS().
  * @note   As calculation input, the analog reference voltage (Vref+) must be
  *         defined as it impacts the ADC LSB equivalent voltage.
  * @note   Analog reference voltage (Vref+) must be either known from
  *         user board environment or can be calculated using ADC measurement
  *         and ADC helper macro @ref __LL_ADC_CALC_VREFANALOG_VOLTAGE().
  * @note   On this STM32 series, calibration data of temperature sensor
  *         corresponds to a resolution of 12 bits,
  *         this is the recommended ADC resolution to convert voltage of
  *         temperature sensor.
  *         Otherwise, this macro performs the processing to scale
  *         ADC conversion data to 12 bits.
  * @param  __VREFANALOG_VOLTAGE__  Analog reference voltage (unit mV)
  * @param  __TEMPSENSOR_ADC_DATA__ ADC conversion data of internal
  *                                 temperature sensor (unit: digital value).
  * @param  __ADC_RESOLUTION__      ADC resolution at which internal temperature
  *                                 sensor voltage has been measured.
  *         This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @retval Temperature (unit: degree Celsius)
  */
#define __LL_ADC_CALC_TEMPERATURE(__VREFANALOG_VOLTAGE__,\
                                  __TEMPSENSOR_ADC_DATA__,\
                                  __ADC_RESOLUTION__)                              \
  (((( ((int32_t)((__LL_ADC_CONVERT_DATA_RESOLUTION((__TEMPSENSOR_ADC_DATA__),     \
                                                    (__ADC_RESOLUTION__),          \
                                                    LL_ADC_RESOLUTION_12B)         \
                   * (__VREFANALOG_VOLTAGE__))                                     \
                  / TEMPSENSOR_CAL_VREFANALOG)                                     \
        - (int32_t) *TEMPSENSOR_CAL1_ADDR)                                         \
     ) * (int32_t)(TEMPSENSOR_CAL2_TEMP - TEMPSENSOR_CAL1_TEMP)                    \
    ) / (int32_t)((int32_t)*TEMPSENSOR_CAL2_ADDR - (int32_t)*TEMPSENSOR_CAL1_ADDR) \
   ) + TEMPSENSOR_CAL1_TEMP                                                        \
  )
#endif

/**
  * @brief  Helper macro to calculate the temperature (unit: degree Celsius)
  *         from ADC conversion data of internal temperature sensor.
  * @note   Computation is using temperature sensor typical values
  *         (refer to device datasheet).
  * @note   Calculation formula:
  *           Temperature = (TS_TYP_CALx_VOLT(uV) - TS_ADC_DATA * Conversion_uV)
  *                         / Avg_Slope + CALx_TEMP
  *           with TS_ADC_DATA      = temperature sensor raw data measured by ADC
  *                                   (unit: digital value)
  *                Avg_Slope        = temperature sensor slope
  *                                   (unit: uV/Degree Celsius)
  *                TS_TYP_CALx_VOLT = temperature sensor digital value at
  *                                   temperature CALx_TEMP (unit: mV)
  *         Caution: Calculation relevancy under reserve the temperature sensor
  *                  of the current device has characteristics in line with
  *                  datasheet typical values.
  *                  If temperature sensor calibration values are available on
  *                  on this device (presence of macro __LL_ADC_CALC_TEMPERATURE()),
  *                  temperature calculation will be more accurate using
  *                  helper macro @ref __LL_ADC_CALC_TEMPERATURE().
  * @note   As calculation input, the analog reference voltage (Vref+) must be
  *         defined as it impacts the ADC LSB equivalent voltage.
  * @note   Analog reference voltage (Vref+) must be either known from
  *         user board environment or can be calculated using ADC measurement
  *         and ADC helper macro @ref __LL_ADC_CALC_VREFANALOG_VOLTAGE().
  * @note   ADC measurement data must correspond to a resolution of 12bits
  *         (full scale digital value 4095). If not the case, the data must be
  *         preliminarily rescaled to an equivalent resolution of 12 bits.
  * @param  __TEMPSENSOR_TYP_AVGSLOPE__   Device datasheet data Temperature sensor slope typical value (unit uV/DegCelsius).
  *                                       On STM32F4, refer to device datasheet parameter "Avg_Slope".
  * @param  __TEMPSENSOR_TYP_CALX_V__     Device datasheet data Temperature sensor voltage typical value (at temperature and Vref+ defined in parameters below) (unit mV).
  *                                       On STM32F4, refer to device datasheet parameter "V25".
  * @param  __TEMPSENSOR_CALX_TEMP__      Device datasheet data Temperature at which temperature sensor voltage (see parameter above) is corresponding (unit mV)
  * @param  __VREFANALOG_VOLTAGE__        Analog voltage reference (Vref+) voltage (unit mV)
  * @param  __TEMPSENSOR_ADC_DATA__       ADC conversion data of internal temperature sensor (unit digital value).
  * @param  __ADC_RESOLUTION__            ADC resolution at which internal temperature sensor voltage has been measured.
  *         This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @retval Temperature (unit: degree Celsius)
  */
#define __LL_ADC_CALC_TEMPERATURE_TYP_PARAMS(__TEMPSENSOR_TYP_AVGSLOPE__,\
                                             __TEMPSENSOR_TYP_CALX_V__,\
                                             __TEMPSENSOR_CALX_TEMP__,\
                                             __VREFANALOG_VOLTAGE__,\
                                             __TEMPSENSOR_ADC_DATA__,\
                                             __ADC_RESOLUTION__)               \
  ((( (                                                                        \
       (int32_t)(((__TEMPSENSOR_TYP_CALX_V__))                                 \
                 * 1000)                                                       \
       -                                                                       \
       (int32_t)((((__TEMPSENSOR_ADC_DATA__) * (__VREFANALOG_VOLTAGE__))       \
                  / __LL_ADC_DIGITAL_SCALE(__ADC_RESOLUTION__))                \
                 * 1000)                                                       \
      )                                                                        \
    ) / (__TEMPSENSOR_TYP_AVGSLOPE__)                                          \
   ) + (__TEMPSENSOR_CALX_TEMP__)                                              \
  )

/**
  * @}
  */

/**
  * @}
  */


/* Exported functions --------------------------------------------------------*/
/** @defgroup ADC_LL_Exported_Functions ADC Exported Functions
  * @{
  */

/** @defgroup ADC_LL_EF_DMA_Management ADC DMA management
  * @{
  */
/* Note: LL ADC functions to set DMA transfer are located into sections of    */
/*       configuration of ADC instance, groups and multimode (if available):  */
/*       @ref LL_ADC_REG_SetDMATransfer(), ...                                */

/**
  * @brief  Function to help to configure DMA transfer from ADC: retrieve the
  *         ADC register address from ADC instance and a list of ADC registers
  *         intended to be used (most commonly) with DMA transfer.
  * @note   These ADC registers are data registers:
  *         when ADC conversion data is available in ADC data registers,
  *         ADC generates a DMA transfer request.
  * @note   This macro is intended to be used with LL DMA driver, refer to
  *         function "LL_DMA_ConfigAddresses()".
  *         Example:
  *           LL_DMA_ConfigAddresses(DMA1,
  *                                  LL_DMA_CHANNEL_1,
  *                                  LL_ADC_DMA_GetRegAddr(ADC1, LL_ADC_DMA_REG_REGULAR_DATA),
  *                                  (uint32_t)&< array or variable >,
  *                                  LL_DMA_DIRECTION_PERIPH_TO_MEMORY);
  * @note   For devices with several ADC: in multimode, some devices
  *         use a different data register outside of ADC instance scope
  *         (common data register). This macro manages this register difference,
  *         only ADC instance has to be set as parameter.
  * @rmtoll DR       RDATA          LL_ADC_DMA_GetRegAddr\n
  *         CDR      RDATA_MST      LL_ADC_DMA_GetRegAddr\n
  *         CDR      RDATA_SLV      LL_ADC_DMA_GetRegAddr
  * @param  ADCx ADC instance
  * @param  Register This parameter can be one of the following values:
  *         @arg @ref LL_ADC_DMA_REG_REGULAR_DATA
  *         @arg @ref LL_ADC_DMA_REG_REGULAR_DATA_MULTI (1)
  *         
  *         (1) Available on devices with several ADC instances.
  * @retval ADC register address
  */
#if defined(ADC_MULTIMODE_SUPPORT)
__STATIC_INLINE uint32_t LL_ADC_DMA_GetRegAddr(ADC_TypeDef *ADCx, uint32_t Register)
{
  uint32_t data_reg_addr = 0UL;
  UNUSED(Register);
  
  if (Register == LL_ADC_DMA_REG_REGULAR_DATA)
  {
    /* Retrieve address of register DR */
    data_reg_addr = (uint32_t)&(ADCx->DR);
  }
  else /* (Register == LL_ADC_DMA_REG_REGULAR_DATA_MULTI) */
  {
    /* Retrieve address of register CDR */
    data_reg_addr = (uint32_t)&((__LL_ADC_COMMON_INSTANCE(ADCx))->CDR);
  }
  
  return data_reg_addr;
}
#else
__STATIC_INLINE uint32_t LL_ADC_DMA_GetRegAddr(ADC_TypeDef *ADCx, uint32_t Register)
{
  UNUSED(Register);
  /* Retrieve address of register DR */
  return (uint32_t)&(ADCx->DR);
}
#endif

/**
  * @}
  */

/** @defgroup ADC_LL_EF_Configuration_ADC_Common Configuration of ADC hierarchical scope: common to several ADC instances
  * @{
  */

/**
  * @brief  Set parameter common to several ADC: Clock source and prescaler.
  * @rmtoll CCR      ADCPRE         LL_ADC_SetCommonClock
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @param  CommonClock This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CLOCK_SYNC_PCLK_DIV2
  *         @arg @ref LL_ADC_CLOCK_SYNC_PCLK_DIV4
  *         @arg @ref LL_ADC_CLOCK_SYNC_PCLK_DIV6
  *         @arg @ref LL_ADC_CLOCK_SYNC_PCLK_DIV8
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetCommonClock(ADC_Common_TypeDef *ADCxy_COMMON, uint32_t CommonClock)
{
  MODIFY_REG(ADCxy_COMMON->CCR, ADC_CCR_ADCPRE, CommonClock);
}

/**
  * @brief  Get parameter common to several ADC: Clock source and prescaler.
  * @rmtoll CCR      ADCPRE         LL_ADC_GetCommonClock
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_CLOCK_SYNC_PCLK_DIV2
  *         @arg @ref LL_ADC_CLOCK_SYNC_PCLK_DIV4
  *         @arg @ref LL_ADC_CLOCK_SYNC_PCLK_DIV6
  *         @arg @ref LL_ADC_CLOCK_SYNC_PCLK_DIV8
  */
__STATIC_INLINE uint32_t LL_ADC_GetCommonClock(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (uint32_t)(READ_BIT(ADCxy_COMMON->CCR, ADC_CCR_ADCPRE));
}

/**
  * @brief  Set parameter common to several ADC: measurement path to internal
  *         channels (VrefInt, temperature sensor, ...).
  * @note   One or several values can be selected.
  *         Example: (LL_ADC_PATH_INTERNAL_VREFINT |
  *                   LL_ADC_PATH_INTERNAL_TEMPSENSOR)
  * @note   Stabilization time of measurement path to internal channel:
  *         After enabling internal paths, before starting ADC conversion,
  *         a delay is required for internal voltage reference and
  *         temperature sensor stabilization time.
  *         Refer to device datasheet.
  *         Refer to literal @ref LL_ADC_DELAY_VREFINT_STAB_US.
  *         Refer to literal @ref LL_ADC_DELAY_TEMPSENSOR_STAB_US.
  * @note   ADC internal channel sampling time constraint:
  *         For ADC conversion of internal channels,
  *         a sampling time minimum value is required.
  *         Refer to device datasheet.
  * @rmtoll CCR      TSVREFE        LL_ADC_SetCommonPathInternalCh\n
  *         CCR      VBATE          LL_ADC_SetCommonPathInternalCh
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @param  PathInternal This parameter can be a combination of the following values:
  *         @arg @ref LL_ADC_PATH_INTERNAL_NONE
  *         @arg @ref LL_ADC_PATH_INTERNAL_VREFINT
  *         @arg @ref LL_ADC_PATH_INTERNAL_TEMPSENSOR
  *         @arg @ref LL_ADC_PATH_INTERNAL_VBAT
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetCommonPathInternalCh(ADC_Common_TypeDef *ADCxy_COMMON, uint32_t PathInternal)
{
  MODIFY_REG(ADCxy_COMMON->CCR, ADC_CCR_TSVREFE | ADC_CCR_VBATE, PathInternal);
}

/**
  * @brief  Get parameter common to several ADC: measurement path to internal
  *         channels (VrefInt, temperature sensor, ...).
  * @note   One or several values can be selected.
  *         Example: (LL_ADC_PATH_INTERNAL_VREFINT |
  *                   LL_ADC_PATH_INTERNAL_TEMPSENSOR)
  * @rmtoll CCR      TSVREFE        LL_ADC_GetCommonPathInternalCh\n
  *         CCR      VBATE          LL_ADC_GetCommonPathInternalCh
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval Returned value can be a combination of the following values:
  *         @arg @ref LL_ADC_PATH_INTERNAL_NONE
  *         @arg @ref LL_ADC_PATH_INTERNAL_VREFINT
  *         @arg @ref LL_ADC_PATH_INTERNAL_TEMPSENSOR
  *         @arg @ref LL_ADC_PATH_INTERNAL_VBAT
  */
__STATIC_INLINE uint32_t LL_ADC_GetCommonPathInternalCh(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (uint32_t)(READ_BIT(ADCxy_COMMON->CCR, ADC_CCR_TSVREFE | ADC_CCR_VBATE));
}

/**
  * @}
  */

/** @defgroup ADC_LL_EF_Configuration_ADC_Instance Configuration of ADC hierarchical scope: ADC instance
  * @{
  */

/**
  * @brief  Set ADC resolution.
  *         Refer to reference manual for alignments formats
  *         dependencies to ADC resolutions.
  * @rmtoll CR1      RES            LL_ADC_SetResolution
  * @param  ADCx ADC instance
  * @param  Resolution This parameter can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetResolution(ADC_TypeDef *ADCx, uint32_t Resolution)
{
  MODIFY_REG(ADCx->CR1, ADC_CR1_RES, Resolution);
}

/**
  * @brief  Get ADC resolution.
  *         Refer to reference manual for alignments formats
  *         dependencies to ADC resolutions.
  * @rmtoll CR1      RES            LL_ADC_GetResolution
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_RESOLUTION_12B
  *         @arg @ref LL_ADC_RESOLUTION_10B
  *         @arg @ref LL_ADC_RESOLUTION_8B
  *         @arg @ref LL_ADC_RESOLUTION_6B
  */
__STATIC_INLINE uint32_t LL_ADC_GetResolution(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR1, ADC_CR1_RES));
}

/**
  * @brief  Set ADC conversion data alignment.
  * @note   Refer to reference manual for alignments formats
  *         dependencies to ADC resolutions.
  * @rmtoll CR2      ALIGN          LL_ADC_SetDataAlignment
  * @param  ADCx ADC instance
  * @param  DataAlignment This parameter can be one of the following values:
  *         @arg @ref LL_ADC_DATA_ALIGN_RIGHT
  *         @arg @ref LL_ADC_DATA_ALIGN_LEFT
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetDataAlignment(ADC_TypeDef *ADCx, uint32_t DataAlignment)
{
  MODIFY_REG(ADCx->CR2, ADC_CR2_ALIGN, DataAlignment);
}

/**
  * @brief  Get ADC conversion data alignment.
  * @note   Refer to reference manual for alignments formats
  *         dependencies to ADC resolutions.
  * @rmtoll CR2      ALIGN          LL_ADC_SetDataAlignment
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_DATA_ALIGN_RIGHT
  *         @arg @ref LL_ADC_DATA_ALIGN_LEFT
  */
__STATIC_INLINE uint32_t LL_ADC_GetDataAlignment(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR2, ADC_CR2_ALIGN));
}

/**
  * @brief  Set ADC sequencers scan mode, for all ADC groups
  *         (group regular, group injected).
  * @note  According to sequencers scan mode :
  *         - If disabled: ADC conversion is performed in unitary conversion
  *           mode (one channel converted, that defined in rank 1).
  *           Configuration of sequencers of all ADC groups
  *           (sequencer scan length, ...) is discarded: equivalent to
  *           scan length of 1 rank.
  *         - If enabled: ADC conversions are performed in sequence conversions
  *           mode, according to configuration of sequencers of
  *           each ADC group (sequencer scan length, ...).
  *           Refer to function @ref LL_ADC_REG_SetSequencerLength()
  *           and to function @ref LL_ADC_INJ_SetSequencerLength().
  * @rmtoll CR1      SCAN           LL_ADC_SetSequencersScanMode
  * @param  ADCx ADC instance
  * @param  ScanMode This parameter can be one of the following values:
  *         @arg @ref LL_ADC_SEQ_SCAN_DISABLE
  *         @arg @ref LL_ADC_SEQ_SCAN_ENABLE
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetSequencersScanMode(ADC_TypeDef *ADCx, uint32_t ScanMode)
{
  MODIFY_REG(ADCx->CR1, ADC_CR1_SCAN, ScanMode);
}

/**
  * @brief  Get ADC sequencers scan mode, for all ADC groups
  *         (group regular, group injected).
  * @note  According to sequencers scan mode :
  *         - If disabled: ADC conversion is performed in unitary conversion
  *           mode (one channel converted, that defined in rank 1).
  *           Configuration of sequencers of all ADC groups
  *           (sequencer scan length, ...) is discarded: equivalent to
  *           scan length of 1 rank.
  *         - If enabled: ADC conversions are performed in sequence conversions
  *           mode, according to configuration of sequencers of
  *           each ADC group (sequencer scan length, ...).
  *           Refer to function @ref LL_ADC_REG_SetSequencerLength()
  *           and to function @ref LL_ADC_INJ_SetSequencerLength().
  * @rmtoll CR1      SCAN           LL_ADC_GetSequencersScanMode
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_SEQ_SCAN_DISABLE
  *         @arg @ref LL_ADC_SEQ_SCAN_ENABLE
  */
__STATIC_INLINE uint32_t LL_ADC_GetSequencersScanMode(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR1, ADC_CR1_SCAN));
}

/**
  * @}
  */

/** @defgroup ADC_LL_EF_Configuration_ADC_Group_Regular Configuration of ADC hierarchical scope: group regular
  * @{
  */

/**
  * @brief  Set ADC group regular conversion trigger source:
  *         internal (SW start) or from external IP (timer event,
  *         external interrupt line).
  * @note   On this STM32 series, setting of external trigger edge is performed
  *         using function @ref LL_ADC_REG_StartConversionExtTrig().
  * @note   Availability of parameters of trigger sources from timer 
  *         depends on timers availability on the selected device.
  * @rmtoll CR2      EXTSEL         LL_ADC_REG_SetTriggerSource\n
  *         CR2      EXTEN          LL_ADC_REG_SetTriggerSource
  * @param  ADCx ADC instance
  * @param  TriggerSource This parameter can be one of the following values:
  *         @arg @ref LL_ADC_REG_TRIG_SOFTWARE
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM1_CH1
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM1_CH2
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM1_CH3
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM2_CH2
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM2_CH3
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM2_CH4
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM2_TRGO
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM3_CH1
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM3_TRGO
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM4_CH4
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM5_CH1
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM5_CH2
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM5_CH3
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM8_CH1
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM8_TRGO
  *         @arg @ref LL_ADC_REG_TRIG_EXT_EXTI_LINE11
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_SetTriggerSource(ADC_TypeDef *ADCx, uint32_t TriggerSource)
{
/* Note: On this STM32 series, ADC group regular external trigger edge        */
/*       is used to perform a ADC conversion start.                           */
/*       This function does not set external trigger edge.                    */
/*       This feature is set using function                                   */
/*       @ref LL_ADC_REG_StartConversionExtTrig().                            */
  MODIFY_REG(ADCx->CR2, ADC_CR2_EXTSEL, (TriggerSource & ADC_CR2_EXTSEL));
}

/**
  * @brief  Get ADC group regular conversion trigger source:
  *         internal (SW start) or from external IP (timer event,
  *         external interrupt line).
  * @note   To determine whether group regular trigger source is
  *         internal (SW start) or external, without detail
  *         of which peripheral is selected as external trigger,
  *         (equivalent to 
  *         "if(LL_ADC_REG_GetTriggerSource(ADC1) == LL_ADC_REG_TRIG_SOFTWARE)")
  *         use function @ref LL_ADC_REG_IsTriggerSourceSWStart.
  * @note   Availability of parameters of trigger sources from timer 
  *         depends on timers availability on the selected device.
  * @rmtoll CR2      EXTSEL         LL_ADC_REG_GetTriggerSource\n
  *         CR2      EXTEN          LL_ADC_REG_GetTriggerSource
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_REG_TRIG_SOFTWARE
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM1_CH1
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM1_CH2
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM1_CH3
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM2_CH2
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM2_CH3
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM2_CH4
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM2_TRGO
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM3_CH1
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM3_TRGO
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM4_CH4
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM5_CH1
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM5_CH2
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM5_CH3
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM8_CH1
  *         @arg @ref LL_ADC_REG_TRIG_EXT_TIM8_TRGO
  *         @arg @ref LL_ADC_REG_TRIG_EXT_EXTI_LINE11
  */
__STATIC_INLINE uint32_t LL_ADC_REG_GetTriggerSource(ADC_TypeDef *ADCx)
{
  uint32_t TriggerSource = READ_BIT(ADCx->CR2, ADC_CR2_EXTSEL | ADC_CR2_EXTEN);
  
  /* Value for shift of {0; 4; 8; 12} depending on value of bitfield          */
  /* corresponding to ADC_CR2_EXTEN {0; 1; 2; 3}.                             */
  uint32_t ShiftExten = ((TriggerSource & ADC_CR2_EXTEN) >> (ADC_REG_TRIG_EXTEN_BITOFFSET_POS - 2UL));
  
  /* Set bitfield corresponding to ADC_CR2_EXTEN and ADC_CR2_EXTSEL           */
  /* to match with triggers literals definition.                              */
  return ((TriggerSource
           & (ADC_REG_TRIG_SOURCE_MASK << ShiftExten) & ADC_CR2_EXTSEL)
          | ((ADC_REG_TRIG_EDGE_MASK << ShiftExten) & ADC_CR2_EXTEN)
         );
}

/**
  * @brief  Get ADC group regular conversion trigger source internal (SW start)
            or external.
  * @note   In case of group regular trigger source set to external trigger,
  *         to determine which peripheral is selected as external trigger,
  *         use function @ref LL_ADC_REG_GetTriggerSource().
  * @rmtoll CR2      EXTEN          LL_ADC_REG_IsTriggerSourceSWStart
  * @param  ADCx ADC instance
  * @retval Value "0" if trigger source external trigger
  *         Value "1" if trigger source SW start.
  */
__STATIC_INLINE uint32_t LL_ADC_REG_IsTriggerSourceSWStart(ADC_TypeDef *ADCx)
{
  return (READ_BIT(ADCx->CR2, ADC_CR2_EXTEN) == (LL_ADC_REG_TRIG_SOFTWARE & ADC_CR2_EXTEN));
}

/**
  * @brief  Get ADC group regular conversion trigger polarity.
  * @note   Applicable only for trigger source set to external trigger.
  * @note   On this STM32 series, setting of external trigger edge is performed
  *         using function @ref LL_ADC_REG_StartConversionExtTrig().
  * @rmtoll CR2      EXTEN          LL_ADC_REG_GetTriggerEdge
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_REG_TRIG_EXT_RISING
  *         @arg @ref LL_ADC_REG_TRIG_EXT_FALLING
  *         @arg @ref LL_ADC_REG_TRIG_EXT_RISINGFALLING
  */
__STATIC_INLINE uint32_t LL_ADC_REG_GetTriggerEdge(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR2, ADC_CR2_EXTEN));
}


/**
  * @brief  Set ADC group regular sequencer length and scan direction.
  * @note   Description of ADC group regular sequencer features:
  *         - For devices with sequencer fully configurable
  *           (function "LL_ADC_REG_SetSequencerRanks()" available):
  *           sequencer length and each rank affectation to a channel
  *           are configurable.
  *           This function performs configuration of:
  *           - Sequence length: Number of ranks in the scan sequence.
  *           - Sequence direction: Unless specified in parameters, sequencer
  *             scan direction is forward (from rank 1 to rank n).
  *           Sequencer ranks are selected using
  *           function "LL_ADC_REG_SetSequencerRanks()".
  *         - For devices with sequencer not fully configurable
  *           (function "LL_ADC_REG_SetSequencerChannels()" available):
  *           sequencer length and each rank affectation to a channel
  *           are defined by channel number.
  *           This function performs configuration of:
  *           - Sequence length: Number of ranks in the scan sequence is
  *             defined by number of channels set in the sequence,
  *             rank of each channel is fixed by channel HW number.
  *             (channel 0 fixed on rank 0, channel 1 fixed on rank1, ...).
  *           - Sequence direction: Unless specified in parameters, sequencer
  *             scan direction is forward (from lowest channel number to
  *             highest channel number).
  *           Sequencer ranks are selected using
  *           function "LL_ADC_REG_SetSequencerChannels()".
  * @note   On this STM32 series, group regular sequencer configuration
  *         is conditioned to ADC instance sequencer mode.
  *         If ADC instance sequencer mode is disabled, sequencers of
  *         all groups (group regular, group injected) can be configured
  *         but their execution is disabled (limited to rank 1).
  *         Refer to function @ref LL_ADC_SetSequencersScanMode().
  * @note   Sequencer disabled is equivalent to sequencer of 1 rank:
  *         ADC conversion on only 1 channel.
  * @rmtoll SQR1     L              LL_ADC_REG_SetSequencerLength
  * @param  ADCx ADC instance
  * @param  SequencerNbRanks This parameter can be one of the following values:
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_DISABLE
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_2RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_3RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_4RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_5RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_6RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_7RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_8RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_9RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_10RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_11RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_12RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_13RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_14RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_15RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_16RANKS
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_SetSequencerLength(ADC_TypeDef *ADCx, uint32_t SequencerNbRanks)
{
  MODIFY_REG(ADCx->SQR1, ADC_SQR1_L, SequencerNbRanks);
}

/**
  * @brief  Get ADC group regular sequencer length and scan direction.
  * @note   Description of ADC group regular sequencer features:
  *         - For devices with sequencer fully configurable
  *           (function "LL_ADC_REG_SetSequencerRanks()" available):
  *           sequencer length and each rank affectation to a channel
  *           are configurable.
  *           This function retrieves:
  *           - Sequence length: Number of ranks in the scan sequence.
  *           - Sequence direction: Unless specified in parameters, sequencer
  *             scan direction is forward (from rank 1 to rank n).
  *           Sequencer ranks are selected using
  *           function "LL_ADC_REG_SetSequencerRanks()".
  *         - For devices with sequencer not fully configurable
  *           (function "LL_ADC_REG_SetSequencerChannels()" available):
  *           sequencer length and each rank affectation to a channel
  *           are defined by channel number.
  *           This function retrieves:
  *           - Sequence length: Number of ranks in the scan sequence is
  *             defined by number of channels set in the sequence,
  *             rank of each channel is fixed by channel HW number.
  *             (channel 0 fixed on rank 0, channel 1 fixed on rank1, ...).
  *           - Sequence direction: Unless specified in parameters, sequencer
  *             scan direction is forward (from lowest channel number to
  *             highest channel number).
  *           Sequencer ranks are selected using
  *           function "LL_ADC_REG_SetSequencerChannels()".
  * @note   On this STM32 series, group regular sequencer configuration
  *         is conditioned to ADC instance sequencer mode.
  *         If ADC instance sequencer mode is disabled, sequencers of
  *         all groups (group regular, group injected) can be configured
  *         but their execution is disabled (limited to rank 1).
  *         Refer to function @ref LL_ADC_SetSequencersScanMode().
  * @note   Sequencer disabled is equivalent to sequencer of 1 rank:
  *         ADC conversion on only 1 channel.
  * @rmtoll SQR1     L              LL_ADC_REG_SetSequencerLength
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_DISABLE
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_2RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_3RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_4RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_5RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_6RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_7RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_8RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_9RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_10RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_11RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_12RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_13RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_14RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_15RANKS
  *         @arg @ref LL_ADC_REG_SEQ_SCAN_ENABLE_16RANKS
  */
__STATIC_INLINE uint32_t LL_ADC_REG_GetSequencerLength(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->SQR1, ADC_SQR1_L));
}

/**
  * @brief  Set ADC group regular sequencer discontinuous mode:
  *         sequence subdivided and scan conversions interrupted every selected
  *         number of ranks.
  * @note   It is not possible to enable both ADC group regular 
  *         continuous mode and sequencer discontinuous mode.
  * @note   It is not possible to enable both ADC auto-injected mode
  *         and ADC group regular sequencer discontinuous mode.
  * @rmtoll CR1      DISCEN         LL_ADC_REG_SetSequencerDiscont\n
  *         CR1      DISCNUM        LL_ADC_REG_SetSequencerDiscont
  * @param  ADCx ADC instance
  * @param  SeqDiscont This parameter can be one of the following values:
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_DISABLE
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_1RANK
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_2RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_3RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_4RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_5RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_6RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_7RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_8RANKS
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_SetSequencerDiscont(ADC_TypeDef *ADCx, uint32_t SeqDiscont)
{
  MODIFY_REG(ADCx->CR1, ADC_CR1_DISCEN | ADC_CR1_DISCNUM, SeqDiscont);
}

/**
  * @brief  Get ADC group regular sequencer discontinuous mode:
  *         sequence subdivided and scan conversions interrupted every selected
  *         number of ranks.
  * @rmtoll CR1      DISCEN         LL_ADC_REG_GetSequencerDiscont\n
  *         CR1      DISCNUM        LL_ADC_REG_GetSequencerDiscont
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_DISABLE
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_1RANK
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_2RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_3RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_4RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_5RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_6RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_7RANKS
  *         @arg @ref LL_ADC_REG_SEQ_DISCONT_8RANKS
  */
__STATIC_INLINE uint32_t LL_ADC_REG_GetSequencerDiscont(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR1, ADC_CR1_DISCEN | ADC_CR1_DISCNUM));
}

/**
  * @brief  Set ADC group regular sequence: channel on the selected
  *         scan sequence rank.
  * @note   This function performs configuration of:
  *         - Channels ordering into each rank of scan sequence:
  *           whatever channel can be placed into whatever rank.
  * @note   On this STM32 series, ADC group regular sequencer is
  *         fully configurable: sequencer length and each rank
  *         affectation to a channel are configurable.
  *         Refer to description of function @ref LL_ADC_REG_SetSequencerLength().
  * @note   Depending on devices and packages, some channels may not be available.
  *         Refer to device datasheet for channels availability.
  * @note   On this STM32 series, to measure internal channels (VrefInt,
  *         TempSensor, ...), measurement paths to internal channels must be
  *         enabled separately.
  *         This can be done using function @ref LL_ADC_SetCommonPathInternalCh().
  * @rmtoll SQR3     SQ1            LL_ADC_REG_SetSequencerRanks\n
  *         SQR3     SQ2            LL_ADC_REG_SetSequencerRanks\n
  *         SQR3     SQ3            LL_ADC_REG_SetSequencerRanks\n
  *         SQR3     SQ4            LL_ADC_REG_SetSequencerRanks\n
  *         SQR3     SQ5            LL_ADC_REG_SetSequencerRanks\n
  *         SQR3     SQ6            LL_ADC_REG_SetSequencerRanks\n
  *         SQR2     SQ7            LL_ADC_REG_SetSequencerRanks\n
  *         SQR2     SQ8            LL_ADC_REG_SetSequencerRanks\n
  *         SQR2     SQ9            LL_ADC_REG_SetSequencerRanks\n
  *         SQR2     SQ10           LL_ADC_REG_SetSequencerRanks\n
  *         SQR2     SQ11           LL_ADC_REG_SetSequencerRanks\n
  *         SQR2     SQ12           LL_ADC_REG_SetSequencerRanks\n
  *         SQR1     SQ13           LL_ADC_REG_SetSequencerRanks\n
  *         SQR1     SQ14           LL_ADC_REG_SetSequencerRanks\n
  *         SQR1     SQ15           LL_ADC_REG_SetSequencerRanks\n
  *         SQR1     SQ16           LL_ADC_REG_SetSequencerRanks
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_REG_RANK_1
  *         @arg @ref LL_ADC_REG_RANK_2
  *         @arg @ref LL_ADC_REG_RANK_3
  *         @arg @ref LL_ADC_REG_RANK_4
  *         @arg @ref LL_ADC_REG_RANK_5
  *         @arg @ref LL_ADC_REG_RANK_6
  *         @arg @ref LL_ADC_REG_RANK_7
  *         @arg @ref LL_ADC_REG_RANK_8
  *         @arg @ref LL_ADC_REG_RANK_9
  *         @arg @ref LL_ADC_REG_RANK_10
  *         @arg @ref LL_ADC_REG_RANK_11
  *         @arg @ref LL_ADC_REG_RANK_12
  *         @arg @ref LL_ADC_REG_RANK_13
  *         @arg @ref LL_ADC_REG_RANK_14
  *         @arg @ref LL_ADC_REG_RANK_15
  *         @arg @ref LL_ADC_REG_RANK_16
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_SetSequencerRanks(ADC_TypeDef *ADCx, uint32_t Rank, uint32_t Channel)
{
  /* Set bits with content of parameter "Channel" with bits position          */
  /* in register and register position depending on parameter "Rank".         */
  /* Parameters "Rank" and "Channel" are used with masks because containing   */
  /* other bits reserved for other purpose.                                   */
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->SQR1, __ADC_MASK_SHIFT(Rank, ADC_REG_SQRX_REGOFFSET_MASK));
  
  MODIFY_REG(*preg,
             ADC_CHANNEL_ID_NUMBER_MASK << (Rank & ADC_REG_RANK_ID_SQRX_MASK),
             (Channel & ADC_CHANNEL_ID_NUMBER_MASK) << (Rank & ADC_REG_RANK_ID_SQRX_MASK));
}

/**
  * @brief  Get ADC group regular sequence: channel on the selected
  *         scan sequence rank.
  * @note   On this STM32 series, ADC group regular sequencer is
  *         fully configurable: sequencer length and each rank
  *         affectation to a channel are configurable.
  *         Refer to description of function @ref LL_ADC_REG_SetSequencerLength().
  * @note   Depending on devices and packages, some channels may not be available.
  *         Refer to device datasheet for channels availability.
  * @note   Usage of the returned channel number:
  *         - To reinject this channel into another function LL_ADC_xxx:
  *           the returned channel number is only partly formatted on definition
  *           of literals LL_ADC_CHANNEL_x. Therefore, it has to be compared
  *           with parts of literals LL_ADC_CHANNEL_x or using
  *           helper macro @ref __LL_ADC_CHANNEL_TO_DECIMAL_NB().
  *           Then the selected literal LL_ADC_CHANNEL_x can be used
  *           as parameter for another function.
  *         - To get the channel number in decimal format:
  *           process the returned value with the helper macro
  *           @ref __LL_ADC_CHANNEL_TO_DECIMAL_NB().
  * @rmtoll SQR3     SQ1            LL_ADC_REG_GetSequencerRanks\n
  *         SQR3     SQ2            LL_ADC_REG_GetSequencerRanks\n
  *         SQR3     SQ3            LL_ADC_REG_GetSequencerRanks\n
  *         SQR3     SQ4            LL_ADC_REG_GetSequencerRanks\n
  *         SQR3     SQ5            LL_ADC_REG_GetSequencerRanks\n
  *         SQR3     SQ6            LL_ADC_REG_GetSequencerRanks\n
  *         SQR2     SQ7            LL_ADC_REG_GetSequencerRanks\n
  *         SQR2     SQ8            LL_ADC_REG_GetSequencerRanks\n
  *         SQR2     SQ9            LL_ADC_REG_GetSequencerRanks\n
  *         SQR2     SQ10           LL_ADC_REG_GetSequencerRanks\n
  *         SQR2     SQ11           LL_ADC_REG_GetSequencerRanks\n
  *         SQR2     SQ12           LL_ADC_REG_GetSequencerRanks\n
  *         SQR1     SQ13           LL_ADC_REG_GetSequencerRanks\n
  *         SQR1     SQ14           LL_ADC_REG_GetSequencerRanks\n
  *         SQR1     SQ15           LL_ADC_REG_GetSequencerRanks\n
  *         SQR1     SQ16           LL_ADC_REG_GetSequencerRanks
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_REG_RANK_1
  *         @arg @ref LL_ADC_REG_RANK_2
  *         @arg @ref LL_ADC_REG_RANK_3
  *         @arg @ref LL_ADC_REG_RANK_4
  *         @arg @ref LL_ADC_REG_RANK_5
  *         @arg @ref LL_ADC_REG_RANK_6
  *         @arg @ref LL_ADC_REG_RANK_7
  *         @arg @ref LL_ADC_REG_RANK_8
  *         @arg @ref LL_ADC_REG_RANK_9
  *         @arg @ref LL_ADC_REG_RANK_10
  *         @arg @ref LL_ADC_REG_RANK_11
  *         @arg @ref LL_ADC_REG_RANK_12
  *         @arg @ref LL_ADC_REG_RANK_13
  *         @arg @ref LL_ADC_REG_RANK_14
  *         @arg @ref LL_ADC_REG_RANK_15
  *         @arg @ref LL_ADC_REG_RANK_16
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.\n
  *         (1) For ADC channel read back from ADC register,
  *             comparison with internal channel parameter to be done
  *             using helper macro @ref __LL_ADC_CHANNEL_INTERNAL_TO_EXTERNAL().
  */
__STATIC_INLINE uint32_t LL_ADC_REG_GetSequencerRanks(ADC_TypeDef *ADCx, uint32_t Rank)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->SQR1, __ADC_MASK_SHIFT(Rank, ADC_REG_SQRX_REGOFFSET_MASK));
  
  return (uint32_t) (READ_BIT(*preg,
                              ADC_CHANNEL_ID_NUMBER_MASK << (Rank & ADC_REG_RANK_ID_SQRX_MASK))
                     >> (Rank & ADC_REG_RANK_ID_SQRX_MASK)
                    );
}

/**
  * @brief  Set ADC continuous conversion mode on ADC group regular.
  * @note   Description of ADC continuous conversion mode:
  *         - single mode: one conversion per trigger
  *         - continuous mode: after the first trigger, following
  *           conversions launched successively automatically.
  * @note   It is not possible to enable both ADC group regular 
  *         continuous mode and sequencer discontinuous mode.
  * @rmtoll CR2      CONT           LL_ADC_REG_SetContinuousMode
  * @param  ADCx ADC instance
  * @param  Continuous This parameter can be one of the following values:
  *         @arg @ref LL_ADC_REG_CONV_SINGLE
  *         @arg @ref LL_ADC_REG_CONV_CONTINUOUS
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_SetContinuousMode(ADC_TypeDef *ADCx, uint32_t Continuous)
{
  MODIFY_REG(ADCx->CR2, ADC_CR2_CONT, Continuous);
}

/**
  * @brief  Get ADC continuous conversion mode on ADC group regular.
  * @note   Description of ADC continuous conversion mode:
  *         - single mode: one conversion per trigger
  *         - continuous mode: after the first trigger, following
  *           conversions launched successively automatically.
  * @rmtoll CR2      CONT           LL_ADC_REG_GetContinuousMode
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_REG_CONV_SINGLE
  *         @arg @ref LL_ADC_REG_CONV_CONTINUOUS
  */
__STATIC_INLINE uint32_t LL_ADC_REG_GetContinuousMode(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR2, ADC_CR2_CONT));
}

/**
  * @brief  Set ADC group regular conversion data transfer: no transfer or
  *         transfer by DMA, and DMA requests mode.
  * @note   If transfer by DMA selected, specifies the DMA requests
  *         mode:
  *         - Limited mode (One shot mode): DMA transfer requests are stopped
  *           when number of DMA data transfers (number of
  *           ADC conversions) is reached.
  *           This ADC mode is intended to be used with DMA mode non-circular.
  *         - Unlimited mode: DMA transfer requests are unlimited,
  *           whatever number of DMA data transfers (number of
  *           ADC conversions).
  *           This ADC mode is intended to be used with DMA mode circular.
  * @note   If ADC DMA requests mode is set to unlimited and DMA is set to
  *         mode non-circular:
  *         when DMA transfers size will be reached, DMA will stop transfers of
  *         ADC conversions data ADC will raise an overrun error
  *        (overrun flag and interruption if enabled).
  * @note   For devices with several ADC instances: ADC multimode DMA
  *         settings are available using function @ref LL_ADC_SetMultiDMATransfer().
  * @note   To configure DMA source address (peripheral address),
  *         use function @ref LL_ADC_DMA_GetRegAddr().
  * @rmtoll CR2      DMA            LL_ADC_REG_SetDMATransfer\n
  *         CR2      DDS            LL_ADC_REG_SetDMATransfer
  * @param  ADCx ADC instance
  * @param  DMATransfer This parameter can be one of the following values:
  *         @arg @ref LL_ADC_REG_DMA_TRANSFER_NONE
  *         @arg @ref LL_ADC_REG_DMA_TRANSFER_LIMITED
  *         @arg @ref LL_ADC_REG_DMA_TRANSFER_UNLIMITED
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_SetDMATransfer(ADC_TypeDef *ADCx, uint32_t DMATransfer)
{
  MODIFY_REG(ADCx->CR2, ADC_CR2_DMA | ADC_CR2_DDS, DMATransfer);
}

/**
  * @brief  Get ADC group regular conversion data transfer: no transfer or
  *         transfer by DMA, and DMA requests mode.
  * @note   If transfer by DMA selected, specifies the DMA requests
  *         mode:
  *         - Limited mode (One shot mode): DMA transfer requests are stopped
  *           when number of DMA data transfers (number of
  *           ADC conversions) is reached.
  *           This ADC mode is intended to be used with DMA mode non-circular.
  *         - Unlimited mode: DMA transfer requests are unlimited,
  *           whatever number of DMA data transfers (number of
  *           ADC conversions).
  *           This ADC mode is intended to be used with DMA mode circular.
  * @note   If ADC DMA requests mode is set to unlimited and DMA is set to
  *         mode non-circular:
  *         when DMA transfers size will be reached, DMA will stop transfers of
  *         ADC conversions data ADC will raise an overrun error
  *         (overrun flag and interruption if enabled).
  * @note   For devices with several ADC instances: ADC multimode DMA
  *         settings are available using function @ref LL_ADC_GetMultiDMATransfer().
  * @note   To configure DMA source address (peripheral address),
  *         use function @ref LL_ADC_DMA_GetRegAddr().
  * @rmtoll CR2      DMA            LL_ADC_REG_GetDMATransfer\n
  *         CR2      DDS            LL_ADC_REG_GetDMATransfer
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_REG_DMA_TRANSFER_NONE
  *         @arg @ref LL_ADC_REG_DMA_TRANSFER_LIMITED
  *         @arg @ref LL_ADC_REG_DMA_TRANSFER_UNLIMITED
  */
__STATIC_INLINE uint32_t LL_ADC_REG_GetDMATransfer(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR2, ADC_CR2_DMA | ADC_CR2_DDS));
}

/**
  * @brief  Specify which ADC flag between EOC (end of unitary conversion)
  *         or EOS (end of sequence conversions) is used to indicate
  *         the end of conversion.
  * @note   This feature is aimed to be set when using ADC with
  *         programming model by polling or interruption
  *         (programming model by DMA usually uses DMA interruptions
  *         to indicate end of conversion and data transfer).
  * @note   For ADC group injected, end of conversion (flag&IT) is raised
  *         only at the end of the sequence.
  * @rmtoll CR2      EOCS           LL_ADC_REG_SetFlagEndOfConversion
  * @param  ADCx ADC instance
  * @param  EocSelection This parameter can be one of the following values:
  *         @arg @ref LL_ADC_REG_FLAG_EOC_SEQUENCE_CONV
  *         @arg @ref LL_ADC_REG_FLAG_EOC_UNITARY_CONV
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_SetFlagEndOfConversion(ADC_TypeDef *ADCx, uint32_t EocSelection)
{
  MODIFY_REG(ADCx->CR2, ADC_CR2_EOCS, EocSelection);
}

/**
  * @brief  Get which ADC flag between EOC (end of unitary conversion)
  *         or EOS (end of sequence conversions) is used to indicate
  *         the end of conversion.
  * @rmtoll CR2      EOCS           LL_ADC_REG_GetFlagEndOfConversion
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_REG_FLAG_EOC_SEQUENCE_CONV
  *         @arg @ref LL_ADC_REG_FLAG_EOC_UNITARY_CONV
  */
__STATIC_INLINE uint32_t LL_ADC_REG_GetFlagEndOfConversion(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR2, ADC_CR2_EOCS));
}

/**
  * @}
  */

/** @defgroup ADC_LL_EF_Configuration_ADC_Group_Injected Configuration of ADC hierarchical scope: group injected
  * @{
  */

/**
  * @brief  Set ADC group injected conversion trigger source:
  *         internal (SW start) or from external IP (timer event,
  *         external interrupt line).
  * @note   On this STM32 series, setting of external trigger edge is performed
  *         using function @ref LL_ADC_INJ_StartConversionExtTrig().
  * @note   Availability of parameters of trigger sources from timer 
  *         depends on timers availability on the selected device.
  * @rmtoll CR2      JEXTSEL        LL_ADC_INJ_SetTriggerSource\n
  *         CR2      JEXTEN         LL_ADC_INJ_SetTriggerSource
  * @param  ADCx ADC instance
  * @param  TriggerSource This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_TRIG_SOFTWARE
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM1_CH4
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM1_TRGO
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM2_CH1
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM2_TRGO
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM3_CH2
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM3_CH4
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM4_CH1
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM4_CH2
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM4_CH3
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM4_TRGO
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM5_CH4
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM5_TRGO
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM8_CH2
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM8_CH3
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM8_CH4
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_EXTI_LINE15
  * @retval None
  */
__STATIC_INLINE void LL_ADC_INJ_SetTriggerSource(ADC_TypeDef *ADCx, uint32_t TriggerSource)
{
/* Note: On this STM32 series, ADC group injected external trigger edge       */
/*       is used to perform a ADC conversion start.                           */
/*       This function does not set external trigger edge.                    */
/*       This feature is set using function                                   */
/*       @ref LL_ADC_INJ_StartConversionExtTrig().                            */
  MODIFY_REG(ADCx->CR2, ADC_CR2_JEXTSEL, (TriggerSource & ADC_CR2_JEXTSEL));
}

/**
  * @brief  Get ADC group injected conversion trigger source:
  *         internal (SW start) or from external IP (timer event,
  *         external interrupt line).
  * @note   To determine whether group injected trigger source is
  *         internal (SW start) or external, without detail
  *         of which peripheral is selected as external trigger,
  *         (equivalent to 
  *         "if(LL_ADC_INJ_GetTriggerSource(ADC1) == LL_ADC_INJ_TRIG_SOFTWARE)")
  *         use function @ref LL_ADC_INJ_IsTriggerSourceSWStart.
  * @note   Availability of parameters of trigger sources from timer 
  *         depends on timers availability on the selected device.
  * @rmtoll CR2      JEXTSEL        LL_ADC_INJ_GetTriggerSource\n
  *         CR2      JEXTEN         LL_ADC_INJ_GetTriggerSource
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_INJ_TRIG_SOFTWARE
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM1_CH4
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM1_TRGO
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM2_CH1
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM2_TRGO
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM3_CH2
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM3_CH4
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM4_CH1
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM4_CH2
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM4_CH3
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM4_TRGO
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM5_CH4
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM5_TRGO
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM8_CH2
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM8_CH3
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_TIM8_CH4
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_EXTI_LINE15
  */
__STATIC_INLINE uint32_t LL_ADC_INJ_GetTriggerSource(ADC_TypeDef *ADCx)
{
  uint32_t TriggerSource = READ_BIT(ADCx->CR2, ADC_CR2_JEXTSEL | ADC_CR2_JEXTEN);
  
  /* Value for shift of {0; 4; 8; 12} depending on value of bitfield          */
  /* corresponding to ADC_CR2_JEXTEN {0; 1; 2; 3}.                            */
  uint32_t ShiftExten = ((TriggerSource & ADC_CR2_JEXTEN) >> (ADC_INJ_TRIG_EXTEN_BITOFFSET_POS - 2UL));
  
  /* Set bitfield corresponding to ADC_CR2_JEXTEN and ADC_CR2_JEXTSEL         */
  /* to match with triggers literals definition.                              */
  return ((TriggerSource
           & (ADC_INJ_TRIG_SOURCE_MASK << ShiftExten) & ADC_CR2_JEXTSEL)
          | ((ADC_INJ_TRIG_EDGE_MASK << ShiftExten) & ADC_CR2_JEXTEN)
         );
}

/**
  * @brief  Get ADC group injected conversion trigger source internal (SW start)
            or external
  * @note   In case of group injected trigger source set to external trigger,
  *         to determine which peripheral is selected as external trigger,
  *         use function @ref LL_ADC_INJ_GetTriggerSource.
  * @rmtoll CR2      JEXTEN         LL_ADC_INJ_IsTriggerSourceSWStart
  * @param  ADCx ADC instance
  * @retval Value "0" if trigger source external trigger
  *         Value "1" if trigger source SW start.
  */
__STATIC_INLINE uint32_t LL_ADC_INJ_IsTriggerSourceSWStart(ADC_TypeDef *ADCx)
{
  return (READ_BIT(ADCx->CR2, ADC_CR2_JEXTEN) == (LL_ADC_INJ_TRIG_SOFTWARE & ADC_CR2_JEXTEN));
}

/**
  * @brief  Get ADC group injected conversion trigger polarity.
  *         Applicable only for trigger source set to external trigger.
  * @rmtoll CR2      JEXTEN         LL_ADC_INJ_GetTriggerEdge
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_RISING
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_FALLING
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_RISINGFALLING
  */
__STATIC_INLINE uint32_t LL_ADC_INJ_GetTriggerEdge(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR2, ADC_CR2_JEXTEN));
}

/**
  * @brief  Set ADC group injected sequencer length and scan direction.
  * @note   This function performs configuration of:
  *         - Sequence length: Number of ranks in the scan sequence.
  *         - Sequence direction: Unless specified in parameters, sequencer
  *           scan direction is forward (from rank 1 to rank n).
  * @note   On this STM32 series, group injected sequencer configuration
  *         is conditioned to ADC instance sequencer mode.
  *         If ADC instance sequencer mode is disabled, sequencers of
  *         all groups (group regular, group injected) can be configured
  *         but their execution is disabled (limited to rank 1).
  *         Refer to function @ref LL_ADC_SetSequencersScanMode().
  * @note   Sequencer disabled is equivalent to sequencer of 1 rank:
  *         ADC conversion on only 1 channel.
  * @rmtoll JSQR     JL             LL_ADC_INJ_SetSequencerLength
  * @param  ADCx ADC instance
  * @param  SequencerNbRanks This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_SEQ_SCAN_DISABLE
  *         @arg @ref LL_ADC_INJ_SEQ_SCAN_ENABLE_2RANKS
  *         @arg @ref LL_ADC_INJ_SEQ_SCAN_ENABLE_3RANKS
  *         @arg @ref LL_ADC_INJ_SEQ_SCAN_ENABLE_4RANKS
  * @retval None
  */
__STATIC_INLINE void LL_ADC_INJ_SetSequencerLength(ADC_TypeDef *ADCx, uint32_t SequencerNbRanks)
{
  MODIFY_REG(ADCx->JSQR, ADC_JSQR_JL, SequencerNbRanks);
}

/**
  * @brief  Get ADC group injected sequencer length and scan direction.
  * @note   This function retrieves:
  *         - Sequence length: Number of ranks in the scan sequence.
  *         - Sequence direction: Unless specified in parameters, sequencer
  *           scan direction is forward (from rank 1 to rank n).
  * @note   On this STM32 series, group injected sequencer configuration
  *         is conditioned to ADC instance sequencer mode.
  *         If ADC instance sequencer mode is disabled, sequencers of
  *         all groups (group regular, group injected) can be configured
  *         but their execution is disabled (limited to rank 1).
  *         Refer to function @ref LL_ADC_SetSequencersScanMode().
  * @note   Sequencer disabled is equivalent to sequencer of 1 rank:
  *         ADC conversion on only 1 channel.
  * @rmtoll JSQR     JL             LL_ADC_INJ_GetSequencerLength
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_INJ_SEQ_SCAN_DISABLE
  *         @arg @ref LL_ADC_INJ_SEQ_SCAN_ENABLE_2RANKS
  *         @arg @ref LL_ADC_INJ_SEQ_SCAN_ENABLE_3RANKS
  *         @arg @ref LL_ADC_INJ_SEQ_SCAN_ENABLE_4RANKS
  */
__STATIC_INLINE uint32_t LL_ADC_INJ_GetSequencerLength(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->JSQR, ADC_JSQR_JL));
}

/**
  * @brief  Set ADC group injected sequencer discontinuous mode:
  *         sequence subdivided and scan conversions interrupted every selected
  *         number of ranks.
  * @note   It is not possible to enable both ADC group injected
  *         auto-injected mode and sequencer discontinuous mode.
  * @rmtoll CR1      DISCEN         LL_ADC_INJ_SetSequencerDiscont
  * @param  ADCx ADC instance
  * @param  SeqDiscont This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_SEQ_DISCONT_DISABLE
  *         @arg @ref LL_ADC_INJ_SEQ_DISCONT_1RANK
  * @retval None
  */
__STATIC_INLINE void LL_ADC_INJ_SetSequencerDiscont(ADC_TypeDef *ADCx, uint32_t SeqDiscont)
{
  MODIFY_REG(ADCx->CR1, ADC_CR1_JDISCEN, SeqDiscont);
}

/**
  * @brief  Get ADC group injected sequencer discontinuous mode:
  *         sequence subdivided and scan conversions interrupted every selected
  *         number of ranks.
  * @rmtoll CR1      DISCEN         LL_ADC_REG_GetSequencerDiscont
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_INJ_SEQ_DISCONT_DISABLE
  *         @arg @ref LL_ADC_INJ_SEQ_DISCONT_1RANK
  */
__STATIC_INLINE uint32_t LL_ADC_INJ_GetSequencerDiscont(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR1, ADC_CR1_JDISCEN));
}

/**
  * @brief  Set ADC group injected sequence: channel on the selected
  *         sequence rank.
  * @note   Depending on devices and packages, some channels may not be available.
  *         Refer to device datasheet for channels availability.
  * @note   On this STM32 series, to measure internal channels (VrefInt,
  *         TempSensor, ...), measurement paths to internal channels must be
  *         enabled separately.
  *         This can be done using function @ref LL_ADC_SetCommonPathInternalCh().
  * @rmtoll JSQR     JSQ1           LL_ADC_INJ_SetSequencerRanks\n
  *         JSQR     JSQ2           LL_ADC_INJ_SetSequencerRanks\n
  *         JSQR     JSQ3           LL_ADC_INJ_SetSequencerRanks\n
  *         JSQR     JSQ4           LL_ADC_INJ_SetSequencerRanks
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_RANK_1
  *         @arg @ref LL_ADC_INJ_RANK_2
  *         @arg @ref LL_ADC_INJ_RANK_3
  *         @arg @ref LL_ADC_INJ_RANK_4
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  * @retval None
  */
__STATIC_INLINE void LL_ADC_INJ_SetSequencerRanks(ADC_TypeDef *ADCx, uint32_t Rank, uint32_t Channel)
{
  /* Set bits with content of parameter "Channel" with bits position          */
  /* in register depending on parameter "Rank".                               */
  /* Parameters "Rank" and "Channel" are used with masks because containing   */
  /* other bits reserved for other purpose.                                   */
  uint32_t tmpreg1 = (READ_BIT(ADCx->JSQR, ADC_JSQR_JL) >> ADC_JSQR_JL_Pos) + 1UL;
  
  MODIFY_REG(ADCx->JSQR,
             ADC_CHANNEL_ID_NUMBER_MASK << (5UL * (uint8_t)(((Rank) + 3UL) - (tmpreg1))),
             (Channel & ADC_CHANNEL_ID_NUMBER_MASK) << (5UL * (uint8_t)(((Rank) + 3UL) - (tmpreg1))));
}

/**
  * @brief  Get ADC group injected sequence: channel on the selected
  *         sequence rank.
  * @note   Depending on devices and packages, some channels may not be available.
  *         Refer to device datasheet for channels availability.
  * @note   Usage of the returned channel number:
  *         - To reinject this channel into another function LL_ADC_xxx:
  *           the returned channel number is only partly formatted on definition
  *           of literals LL_ADC_CHANNEL_x. Therefore, it has to be compared
  *           with parts of literals LL_ADC_CHANNEL_x or using
  *           helper macro @ref __LL_ADC_CHANNEL_TO_DECIMAL_NB().
  *           Then the selected literal LL_ADC_CHANNEL_x can be used
  *           as parameter for another function.
  *         - To get the channel number in decimal format:
  *           process the returned value with the helper macro
  *           @ref __LL_ADC_CHANNEL_TO_DECIMAL_NB().
  * @rmtoll JSQR     JSQ1           LL_ADC_INJ_SetSequencerRanks\n
  *         JSQR     JSQ2           LL_ADC_INJ_SetSequencerRanks\n
  *         JSQR     JSQ3           LL_ADC_INJ_SetSequencerRanks\n
  *         JSQR     JSQ4           LL_ADC_INJ_SetSequencerRanks
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_RANK_1
  *         @arg @ref LL_ADC_INJ_RANK_2
  *         @arg @ref LL_ADC_INJ_RANK_3
  *         @arg @ref LL_ADC_INJ_RANK_4
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.\n
  *         (1) For ADC channel read back from ADC register,
  *             comparison with internal channel parameter to be done
  *             using helper macro @ref __LL_ADC_CHANNEL_INTERNAL_TO_EXTERNAL().
  */
__STATIC_INLINE uint32_t LL_ADC_INJ_GetSequencerRanks(ADC_TypeDef *ADCx, uint32_t Rank)
{
  uint32_t tmpreg1 = (READ_BIT(ADCx->JSQR, ADC_JSQR_JL) >> ADC_JSQR_JL_Pos)  + 1UL;
  
  return (uint32_t)(READ_BIT(ADCx->JSQR,
                             ADC_CHANNEL_ID_NUMBER_MASK << (5UL * (uint8_t)(((Rank) + 3UL) - (tmpreg1))))
                    >> (5UL * (uint8_t)(((Rank) + 3UL) - (tmpreg1)))
                   );
}

/**
  * @brief  Set ADC group injected conversion trigger:
  *         independent or from ADC group regular.
  * @note   This mode can be used to extend number of data registers
  *         updated after one ADC conversion trigger and with data 
  *         permanently kept (not erased by successive conversions of scan of
  *         ADC sequencer ranks), up to 5 data registers:
  *         1 data register on ADC group regular, 4 data registers
  *         on ADC group injected.            
  * @note   If ADC group injected injected trigger source is set to an
  *         external trigger, this feature must be must be set to
  *         independent trigger.
  *         ADC group injected automatic trigger is compliant only with 
  *         group injected trigger source set to SW start, without any 
  *         further action on  ADC group injected conversion start or stop: 
  *         in this case, ADC group injected is controlled only 
  *         from ADC group regular.
  * @note   It is not possible to enable both ADC group injected
  *         auto-injected mode and sequencer discontinuous mode.
  * @rmtoll CR1      JAUTO          LL_ADC_INJ_SetTrigAuto
  * @param  ADCx ADC instance
  * @param  TrigAuto This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_TRIG_INDEPENDENT
  *         @arg @ref LL_ADC_INJ_TRIG_FROM_GRP_REGULAR
  * @retval None
  */
__STATIC_INLINE void LL_ADC_INJ_SetTrigAuto(ADC_TypeDef *ADCx, uint32_t TrigAuto)
{
  MODIFY_REG(ADCx->CR1, ADC_CR1_JAUTO, TrigAuto);
}

/**
  * @brief  Get ADC group injected conversion trigger:
  *         independent or from ADC group regular.
  * @rmtoll CR1      JAUTO          LL_ADC_INJ_GetTrigAuto
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_INJ_TRIG_INDEPENDENT
  *         @arg @ref LL_ADC_INJ_TRIG_FROM_GRP_REGULAR
  */
__STATIC_INLINE uint32_t LL_ADC_INJ_GetTrigAuto(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR1, ADC_CR1_JAUTO));
}

/**
  * @brief  Set ADC group injected offset.
  * @note   It sets:
  *         - ADC group injected rank to which the offset programmed
  *           will be applied
  *         - Offset level (offset to be subtracted from the raw
  *           converted data).
  *         Caution: Offset format is dependent to ADC resolution:
  *         offset has to be left-aligned on bit 11, the LSB (right bits)
  *         are set to 0.
  * @note   Offset cannot be enabled or disabled.
  *         To emulate offset disabled, set an offset value equal to 0.
  * @rmtoll JOFR1    JOFFSET1       LL_ADC_INJ_SetOffset\n
  *         JOFR2    JOFFSET2       LL_ADC_INJ_SetOffset\n
  *         JOFR3    JOFFSET3       LL_ADC_INJ_SetOffset\n
  *         JOFR4    JOFFSET4       LL_ADC_INJ_SetOffset
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_RANK_1
  *         @arg @ref LL_ADC_INJ_RANK_2
  *         @arg @ref LL_ADC_INJ_RANK_3
  *         @arg @ref LL_ADC_INJ_RANK_4
  * @param  OffsetLevel Value between Min_Data=0x000 and Max_Data=0xFFF
  * @retval None
  */
__STATIC_INLINE void LL_ADC_INJ_SetOffset(ADC_TypeDef *ADCx, uint32_t Rank, uint32_t OffsetLevel)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->JOFR1, __ADC_MASK_SHIFT(Rank, ADC_INJ_JOFRX_REGOFFSET_MASK));
  
  MODIFY_REG(*preg,
             ADC_JOFR1_JOFFSET1,
             OffsetLevel);
}

/**
  * @brief  Get ADC group injected offset.
  * @note   It gives offset level (offset to be subtracted from the raw converted data).
  *         Caution: Offset format is dependent to ADC resolution:
  *         offset has to be left-aligned on bit 11, the LSB (right bits)
  *         are set to 0.
  * @rmtoll JOFR1    JOFFSET1       LL_ADC_INJ_GetOffset\n
  *         JOFR2    JOFFSET2       LL_ADC_INJ_GetOffset\n
  *         JOFR3    JOFFSET3       LL_ADC_INJ_GetOffset\n
  *         JOFR4    JOFFSET4       LL_ADC_INJ_GetOffset
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_RANK_1
  *         @arg @ref LL_ADC_INJ_RANK_2
  *         @arg @ref LL_ADC_INJ_RANK_3
  *         @arg @ref LL_ADC_INJ_RANK_4
  * @retval Value between Min_Data=0x000 and Max_Data=0xFFF
  */
__STATIC_INLINE uint32_t LL_ADC_INJ_GetOffset(ADC_TypeDef *ADCx, uint32_t Rank)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->JOFR1, __ADC_MASK_SHIFT(Rank, ADC_INJ_JOFRX_REGOFFSET_MASK));
  
  return (uint32_t)(READ_BIT(*preg,
                             ADC_JOFR1_JOFFSET1)
                   );
}

/**
  * @}
  */

/** @defgroup ADC_LL_EF_Configuration_Channels Configuration of ADC hierarchical scope: channels
  * @{
  */

/**
  * @brief  Set sampling time of the selected ADC channel
  *         Unit: ADC clock cycles.
  * @note   On this device, sampling time is on channel scope: independently
  *         of channel mapped on ADC group regular or injected.
  * @note   In case of internal channel (VrefInt, TempSensor, ...) to be
  *         converted:
  *         sampling time constraints must be respected (sampling time can be
  *         adjusted in function of ADC clock frequency and sampling time
  *         setting).
  *         Refer to device datasheet for timings values (parameters TS_vrefint,
  *         TS_temp, ...).
  * @note   Conversion time is the addition of sampling time and processing time.
  *         Refer to reference manual for ADC processing time of
  *         this STM32 series.
  * @note   In case of ADC conversion of internal channel (VrefInt,
  *         temperature sensor, ...), a sampling time minimum value
  *         is required.
  *         Refer to device datasheet.
  * @rmtoll SMPR1    SMP18          LL_ADC_SetChannelSamplingTime\n
  *         SMPR1    SMP17          LL_ADC_SetChannelSamplingTime\n
  *         SMPR1    SMP16          LL_ADC_SetChannelSamplingTime\n
  *         SMPR1    SMP15          LL_ADC_SetChannelSamplingTime\n
  *         SMPR1    SMP14          LL_ADC_SetChannelSamplingTime\n
  *         SMPR1    SMP13          LL_ADC_SetChannelSamplingTime\n
  *         SMPR1    SMP12          LL_ADC_SetChannelSamplingTime\n
  *         SMPR1    SMP11          LL_ADC_SetChannelSamplingTime\n
  *         SMPR1    SMP10          LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP9           LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP8           LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP7           LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP6           LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP5           LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP4           LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP3           LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP2           LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP1           LL_ADC_SetChannelSamplingTime\n
  *         SMPR2    SMP0           LL_ADC_SetChannelSamplingTime
  * @param  ADCx ADC instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  * @param  SamplingTime This parameter can be one of the following values:
  *         @arg @ref LL_ADC_SAMPLINGTIME_3CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_15CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_28CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_56CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_84CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_112CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_144CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_480CYCLES
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetChannelSamplingTime(ADC_TypeDef *ADCx, uint32_t Channel, uint32_t SamplingTime)
{
  /* Set bits with content of parameter "SamplingTime" with bits position     */
  /* in register and register position depending on parameter "Channel".      */
  /* Parameter "Channel" is used with masks because containing                */
  /* other bits reserved for other purpose.                                   */
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->SMPR1, __ADC_MASK_SHIFT(Channel, ADC_CHANNEL_SMPRX_REGOFFSET_MASK));
  
  MODIFY_REG(*preg,
             ADC_SMPR2_SMP0 << __ADC_MASK_SHIFT(Channel, ADC_CHANNEL_SMPx_BITOFFSET_MASK),
             SamplingTime   << __ADC_MASK_SHIFT(Channel, ADC_CHANNEL_SMPx_BITOFFSET_MASK));
}

/**
  * @brief  Get sampling time of the selected ADC channel
  *         Unit: ADC clock cycles.
  * @note   On this device, sampling time is on channel scope: independently
  *         of channel mapped on ADC group regular or injected.
  * @note   Conversion time is the addition of sampling time and processing time.
  *         Refer to reference manual for ADC processing time of
  *         this STM32 series.
  * @rmtoll SMPR1    SMP18          LL_ADC_GetChannelSamplingTime\n
  *         SMPR1    SMP17          LL_ADC_GetChannelSamplingTime\n
  *         SMPR1    SMP16          LL_ADC_GetChannelSamplingTime\n
  *         SMPR1    SMP15          LL_ADC_GetChannelSamplingTime\n
  *         SMPR1    SMP14          LL_ADC_GetChannelSamplingTime\n
  *         SMPR1    SMP13          LL_ADC_GetChannelSamplingTime\n
  *         SMPR1    SMP12          LL_ADC_GetChannelSamplingTime\n
  *         SMPR1    SMP11          LL_ADC_GetChannelSamplingTime\n
  *         SMPR1    SMP10          LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP9           LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP8           LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP7           LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP6           LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP5           LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP4           LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP3           LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP2           LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP1           LL_ADC_GetChannelSamplingTime\n
  *         SMPR2    SMP0           LL_ADC_GetChannelSamplingTime
  * @param  ADCx ADC instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_ADC_CHANNEL_0
  *         @arg @ref LL_ADC_CHANNEL_1
  *         @arg @ref LL_ADC_CHANNEL_2
  *         @arg @ref LL_ADC_CHANNEL_3
  *         @arg @ref LL_ADC_CHANNEL_4
  *         @arg @ref LL_ADC_CHANNEL_5
  *         @arg @ref LL_ADC_CHANNEL_6
  *         @arg @ref LL_ADC_CHANNEL_7
  *         @arg @ref LL_ADC_CHANNEL_8
  *         @arg @ref LL_ADC_CHANNEL_9
  *         @arg @ref LL_ADC_CHANNEL_10
  *         @arg @ref LL_ADC_CHANNEL_11
  *         @arg @ref LL_ADC_CHANNEL_12
  *         @arg @ref LL_ADC_CHANNEL_13
  *         @arg @ref LL_ADC_CHANNEL_14
  *         @arg @ref LL_ADC_CHANNEL_15
  *         @arg @ref LL_ADC_CHANNEL_16
  *         @arg @ref LL_ADC_CHANNEL_17
  *         @arg @ref LL_ADC_CHANNEL_18
  *         @arg @ref LL_ADC_CHANNEL_VREFINT      (1)
  *         @arg @ref LL_ADC_CHANNEL_TEMPSENSOR   (1)(2)
  *         @arg @ref LL_ADC_CHANNEL_VBAT         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_SAMPLINGTIME_3CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_15CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_28CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_56CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_84CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_112CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_144CYCLES
  *         @arg @ref LL_ADC_SAMPLINGTIME_480CYCLES
  */
__STATIC_INLINE uint32_t LL_ADC_GetChannelSamplingTime(ADC_TypeDef *ADCx, uint32_t Channel)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->SMPR1, __ADC_MASK_SHIFT(Channel, ADC_CHANNEL_SMPRX_REGOFFSET_MASK));
  
  return (uint32_t)(READ_BIT(*preg,
                             ADC_SMPR2_SMP0 << __ADC_MASK_SHIFT(Channel, ADC_CHANNEL_SMPx_BITOFFSET_MASK))
                    >> __ADC_MASK_SHIFT(Channel, ADC_CHANNEL_SMPx_BITOFFSET_MASK)
                   );
}

/**
  * @}
  */

/** @defgroup ADC_LL_EF_Configuration_ADC_AnalogWatchdog Configuration of ADC transversal scope: analog watchdog
  * @{
  */

/**
  * @brief  Set ADC analog watchdog monitored channels:
  *         a single channel or all channels,
  *         on ADC groups regular and-or injected.
  * @note   Once monitored channels are selected, analog watchdog
  *         is enabled.
  * @note   In case of need to define a single channel to monitor
  *         with analog watchdog from sequencer channel definition,
  *         use helper macro @ref __LL_ADC_ANALOGWD_CHANNEL_GROUP().
  * @note   On this STM32 series, there is only 1 kind of analog watchdog
  *         instance:
  *         - AWD standard (instance AWD1):
  *           - channels monitored: can monitor 1 channel or all channels.
  *           - groups monitored: ADC groups regular and-or injected.
  *           - resolution: resolution is not limited (corresponds to
  *             ADC resolution configured).
  * @rmtoll CR1      AWD1CH         LL_ADC_SetAnalogWDMonitChannels\n
  *         CR1      AWD1SGL        LL_ADC_SetAnalogWDMonitChannels\n
  *         CR1      AWD1EN         LL_ADC_SetAnalogWDMonitChannels
  * @param  ADCx ADC instance
  * @param  AWDChannelGroup This parameter can be one of the following values:
  *         @arg @ref LL_ADC_AWD_DISABLE
  *         @arg @ref LL_ADC_AWD_ALL_CHANNELS_REG
  *         @arg @ref LL_ADC_AWD_ALL_CHANNELS_INJ
  *         @arg @ref LL_ADC_AWD_ALL_CHANNELS_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_0_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_0_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_0_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_1_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_1_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_1_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_2_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_2_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_2_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_3_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_3_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_3_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_4_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_4_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_4_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_5_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_5_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_5_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_6_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_6_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_6_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_7_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_7_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_7_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_8_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_8_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_8_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_9_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_9_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_9_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_10_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_10_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_10_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_11_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_11_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_11_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_12_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_12_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_12_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_13_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_13_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_13_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_14_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_14_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_14_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_15_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_15_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_15_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_16_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_16_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_16_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_17_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_17_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_17_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_18_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_18_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_18_REG_INJ
  *         @arg @ref LL_ADC_AWD_CH_VREFINT_REG          (1)
  *         @arg @ref LL_ADC_AWD_CH_VREFINT_INJ          (1)
  *         @arg @ref LL_ADC_AWD_CH_VREFINT_REG_INJ      (1)
  *         @arg @ref LL_ADC_AWD_CH_TEMPSENSOR_REG       (1)(2)
  *         @arg @ref LL_ADC_AWD_CH_TEMPSENSOR_INJ       (1)(2)
  *         @arg @ref LL_ADC_AWD_CH_TEMPSENSOR_REG_INJ   (1)(2)
  *         @arg @ref LL_ADC_AWD_CH_VBAT_REG             (1)
  *         @arg @ref LL_ADC_AWD_CH_VBAT_INJ             (1)
  *         @arg @ref LL_ADC_AWD_CH_VBAT_REG_INJ         (1)
  *         
  *         (1) On STM32F4, parameter available only on ADC instance: ADC1.\n
  *         (2) On devices STM32F42x and STM32F43x, limitation: this internal channel is shared between temperature sensor and Vbat, only 1 measurement path must be enabled.
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetAnalogWDMonitChannels(ADC_TypeDef *ADCx, uint32_t AWDChannelGroup)
{
  MODIFY_REG(ADCx->CR1,
             (ADC_CR1_AWDEN | ADC_CR1_JAWDEN | ADC_CR1_AWDSGL | ADC_CR1_AWDCH),
             AWDChannelGroup);
}

/**
  * @brief  Get ADC analog watchdog monitored channel.
  * @note   Usage of the returned channel number:
  *         - To reinject this channel into another function LL_ADC_xxx:
  *           the returned channel number is only partly formatted on definition
  *           of literals LL_ADC_CHANNEL_x. Therefore, it has to be compared
  *           with parts of literals LL_ADC_CHANNEL_x or using
  *           helper macro @ref __LL_ADC_CHANNEL_TO_DECIMAL_NB().
  *           Then the selected literal LL_ADC_CHANNEL_x can be used
  *           as parameter for another function.
  *         - To get the channel number in decimal format:
  *           process the returned value with the helper macro
  *           @ref __LL_ADC_CHANNEL_TO_DECIMAL_NB().
  *           Applicable only when the analog watchdog is set to monitor
  *           one channel.
  * @note   On this STM32 series, there is only 1 kind of analog watchdog
  *         instance:
  *         - AWD standard (instance AWD1):
  *           - channels monitored: can monitor 1 channel or all channels.
  *           - groups monitored: ADC groups regular and-or injected.
  *           - resolution: resolution is not limited (corresponds to
  *             ADC resolution configured).
  * @rmtoll CR1      AWD1CH         LL_ADC_GetAnalogWDMonitChannels\n
  *         CR1      AWD1SGL        LL_ADC_GetAnalogWDMonitChannels\n
  *         CR1      AWD1EN         LL_ADC_GetAnalogWDMonitChannels
  * @param  ADCx ADC instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_AWD_DISABLE
  *         @arg @ref LL_ADC_AWD_ALL_CHANNELS_REG
  *         @arg @ref LL_ADC_AWD_ALL_CHANNELS_INJ
  *         @arg @ref LL_ADC_AWD_ALL_CHANNELS_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_0_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_0_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_0_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_1_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_1_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_1_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_2_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_2_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_2_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_3_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_3_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_3_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_4_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_4_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_4_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_5_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_5_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_5_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_6_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_6_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_6_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_7_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_7_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_7_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_8_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_8_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_8_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_9_REG 
  *         @arg @ref LL_ADC_AWD_CHANNEL_9_INJ 
  *         @arg @ref LL_ADC_AWD_CHANNEL_9_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_10_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_10_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_10_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_11_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_11_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_11_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_12_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_12_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_12_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_13_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_13_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_13_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_14_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_14_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_14_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_15_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_15_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_15_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_16_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_16_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_16_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_17_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_17_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_17_REG_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_18_REG
  *         @arg @ref LL_ADC_AWD_CHANNEL_18_INJ
  *         @arg @ref LL_ADC_AWD_CHANNEL_18_REG_INJ
  */
__STATIC_INLINE uint32_t LL_ADC_GetAnalogWDMonitChannels(ADC_TypeDef *ADCx)
{
  return (uint32_t)(READ_BIT(ADCx->CR1, (ADC_CR1_AWDEN | ADC_CR1_JAWDEN | ADC_CR1_AWDSGL | ADC_CR1_AWDCH)));
}

/**
  * @brief  Set ADC analog watchdog threshold value of threshold
  *         high or low.
  * @note   In case of ADC resolution different of 12 bits,
  *         analog watchdog thresholds data require a specific shift.
  *         Use helper macro @ref __LL_ADC_ANALOGWD_SET_THRESHOLD_RESOLUTION().
  * @note   On this STM32 series, there is only 1 kind of analog watchdog
  *         instance:
  *         - AWD standard (instance AWD1):
  *           - channels monitored: can monitor 1 channel or all channels.
  *           - groups monitored: ADC groups regular and-or injected.
  *           - resolution: resolution is not limited (corresponds to
  *             ADC resolution configured).
  * @rmtoll HTR      HT             LL_ADC_SetAnalogWDThresholds\n
  *         LTR      LT             LL_ADC_SetAnalogWDThresholds
  * @param  ADCx ADC instance
  * @param  AWDThresholdsHighLow This parameter can be one of the following values:
  *         @arg @ref LL_ADC_AWD_THRESHOLD_HIGH
  *         @arg @ref LL_ADC_AWD_THRESHOLD_LOW
  * @param  AWDThresholdValue Value between Min_Data=0x000 and Max_Data=0xFFF
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetAnalogWDThresholds(ADC_TypeDef *ADCx, uint32_t AWDThresholdsHighLow, uint32_t AWDThresholdValue)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->HTR, AWDThresholdsHighLow);
  
  MODIFY_REG(*preg,
             ADC_HTR_HT,
             AWDThresholdValue);
}

/**
  * @brief  Get ADC analog watchdog threshold value of threshold high or
  *         threshold low.
  * @note   In case of ADC resolution different of 12 bits,
  *         analog watchdog thresholds data require a specific shift.
  *         Use helper macro @ref __LL_ADC_ANALOGWD_GET_THRESHOLD_RESOLUTION().
  * @rmtoll HTR      HT             LL_ADC_GetAnalogWDThresholds\n
  *         LTR      LT             LL_ADC_GetAnalogWDThresholds
  * @param  ADCx ADC instance
  * @param  AWDThresholdsHighLow This parameter can be one of the following values:
  *         @arg @ref LL_ADC_AWD_THRESHOLD_HIGH
  *         @arg @ref LL_ADC_AWD_THRESHOLD_LOW
  * @retval Value between Min_Data=0x000 and Max_Data=0xFFF
*/
__STATIC_INLINE uint32_t LL_ADC_GetAnalogWDThresholds(ADC_TypeDef *ADCx, uint32_t AWDThresholdsHighLow)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->HTR, AWDThresholdsHighLow);
  
  return (uint32_t)(READ_BIT(*preg, ADC_HTR_HT));
}

/**
  * @}
  */

/** @defgroup ADC_LL_EF_Configuration_ADC_Multimode Configuration of ADC hierarchical scope: multimode
  * @{
  */

#if defined(ADC_MULTIMODE_SUPPORT)
/**
  * @brief  Set ADC multimode configuration to operate in independent mode
  *         or multimode (for devices with several ADC instances).
  * @note   If multimode configuration: the selected ADC instance is
  *         either master or slave depending on hardware.
  *         Refer to reference manual.
  * @rmtoll CCR      MULTI          LL_ADC_SetMultimode
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @param  Multimode This parameter can be one of the following values:
  *         @arg @ref LL_ADC_MULTI_INDEPENDENT
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_SIMULT
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_INTERL
  *         @arg @ref LL_ADC_MULTI_DUAL_INJ_SIMULT
  *         @arg @ref LL_ADC_MULTI_DUAL_INJ_ALTERN
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_SIM_INJ_SIM
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_SIM_INJ_ALT
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_INT_INJ_SIM
  *         @arg @ref LL_ADC_MULTI_TRIPLE_REG_SIM_INJ_SIM
  *         @arg @ref LL_ADC_MULTI_TRIPLE_REG_SIM_INJ_ALT
  *         @arg @ref LL_ADC_MULTI_TRIPLE_INJ_SIMULT
  *         @arg @ref LL_ADC_MULTI_TRIPLE_REG_SIMULT
  *         @arg @ref LL_ADC_MULTI_TRIPLE_REG_INTERL
  *         @arg @ref LL_ADC_MULTI_TRIPLE_INJ_ALTERN
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetMultimode(ADC_Common_TypeDef *ADCxy_COMMON, uint32_t Multimode)
{
  MODIFY_REG(ADCxy_COMMON->CCR, ADC_CCR_MULTI, Multimode);
}

/**
  * @brief  Get ADC multimode configuration to operate in independent mode
  *         or multimode (for devices with several ADC instances).
  * @note   If multimode configuration: the selected ADC instance is
  *         either master or slave depending on hardware.
  *         Refer to reference manual.
  * @rmtoll CCR      MULTI          LL_ADC_GetMultimode
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_MULTI_INDEPENDENT
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_SIMULT
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_INTERL
  *         @arg @ref LL_ADC_MULTI_DUAL_INJ_SIMULT
  *         @arg @ref LL_ADC_MULTI_DUAL_INJ_ALTERN
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_SIM_INJ_SIM
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_SIM_INJ_ALT
  *         @arg @ref LL_ADC_MULTI_DUAL_REG_INT_INJ_SIM
  *         @arg @ref LL_ADC_MULTI_TRIPLE_REG_SIM_INJ_SIM
  *         @arg @ref LL_ADC_MULTI_TRIPLE_REG_SIM_INJ_ALT
  *         @arg @ref LL_ADC_MULTI_TRIPLE_INJ_SIMULT
  *         @arg @ref LL_ADC_MULTI_TRIPLE_REG_SIMULT
  *         @arg @ref LL_ADC_MULTI_TRIPLE_REG_INTERL
  *         @arg @ref LL_ADC_MULTI_TRIPLE_INJ_ALTERN
  */
__STATIC_INLINE uint32_t LL_ADC_GetMultimode(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (uint32_t)(READ_BIT(ADCxy_COMMON->CCR, ADC_CCR_MULTI));
}

/**
  * @brief  Set ADC multimode conversion data transfer: no transfer
  *         or transfer by DMA.
  * @note   If ADC multimode transfer by DMA is not selected:
  *         each ADC uses its own DMA channel, with its individual
  *         DMA transfer settings.
  *         If ADC multimode transfer by DMA is selected:
  *         One DMA channel is used for both ADC (DMA of ADC master)
  *         Specifies the DMA requests mode:
  *         - Limited mode (One shot mode): DMA transfer requests are stopped
  *           when number of DMA data transfers (number of
  *           ADC conversions) is reached.
  *           This ADC mode is intended to be used with DMA mode non-circular.
  *         - Unlimited mode: DMA transfer requests are unlimited,
  *           whatever number of DMA data transfers (number of
  *           ADC conversions).
  *           This ADC mode is intended to be used with DMA mode circular.
  * @note   If ADC DMA requests mode is set to unlimited and DMA is set to
  *         mode non-circular:
  *         when DMA transfers size will be reached, DMA will stop transfers of
  *         ADC conversions data ADC will raise an overrun error
  *         (overrun flag and interruption if enabled).
  * @note   How to retrieve multimode conversion data:
  *         Whatever multimode transfer by DMA setting: using function
  *         @ref LL_ADC_REG_ReadMultiConversionData32().
  *         If ADC multimode transfer by DMA is selected: conversion data
  *         is a raw data with ADC master and slave concatenated.
  *         A macro is available to get the conversion data of
  *         ADC master or ADC slave: see helper macro
  *         @ref __LL_ADC_MULTI_CONV_DATA_MASTER_SLAVE().
  * @rmtoll CCR      MDMA           LL_ADC_SetMultiDMATransfer\n
  *         CCR      DDS            LL_ADC_SetMultiDMATransfer
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @param  MultiDMATransfer This parameter can be one of the following values:
  *         @arg @ref LL_ADC_MULTI_REG_DMA_EACH_ADC
  *         @arg @ref LL_ADC_MULTI_REG_DMA_LIMIT_1
  *         @arg @ref LL_ADC_MULTI_REG_DMA_LIMIT_2
  *         @arg @ref LL_ADC_MULTI_REG_DMA_LIMIT_3
  *         @arg @ref LL_ADC_MULTI_REG_DMA_UNLMT_1
  *         @arg @ref LL_ADC_MULTI_REG_DMA_UNLMT_2
  *         @arg @ref LL_ADC_MULTI_REG_DMA_UNLMT_3
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetMultiDMATransfer(ADC_Common_TypeDef *ADCxy_COMMON, uint32_t MultiDMATransfer)
{
  MODIFY_REG(ADCxy_COMMON->CCR, ADC_CCR_DMA | ADC_CCR_DDS, MultiDMATransfer);
}

/**
  * @brief  Get ADC multimode conversion data transfer: no transfer
  *         or transfer by DMA.
  * @note   If ADC multimode transfer by DMA is not selected:
  *         each ADC uses its own DMA channel, with its individual
  *         DMA transfer settings.
  *         If ADC multimode transfer by DMA is selected:
  *         One DMA channel is used for both ADC (DMA of ADC master)
  *         Specifies the DMA requests mode:
  *         - Limited mode (One shot mode): DMA transfer requests are stopped
  *           when number of DMA data transfers (number of
  *           ADC conversions) is reached.
  *           This ADC mode is intended to be used with DMA mode non-circular.
  *         - Unlimited mode: DMA transfer requests are unlimited,
  *           whatever number of DMA data transfers (number of
  *           ADC conversions).
  *           This ADC mode is intended to be used with DMA mode circular.
  * @note   If ADC DMA requests mode is set to unlimited and DMA is set to
  *         mode non-circular:
  *         when DMA transfers size will be reached, DMA will stop transfers of
  *         ADC conversions data ADC will raise an overrun error
  *         (overrun flag and interruption if enabled).
  * @note   How to retrieve multimode conversion data:
  *         Whatever multimode transfer by DMA setting: using function
  *         @ref LL_ADC_REG_ReadMultiConversionData32().
  *         If ADC multimode transfer by DMA is selected: conversion data
  *         is a raw data with ADC master and slave concatenated.
  *         A macro is available to get the conversion data of
  *         ADC master or ADC slave: see helper macro
  *         @ref __LL_ADC_MULTI_CONV_DATA_MASTER_SLAVE().
  * @rmtoll CCR      MDMA           LL_ADC_GetMultiDMATransfer\n
  *         CCR      DDS            LL_ADC_GetMultiDMATransfer
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_MULTI_REG_DMA_EACH_ADC
  *         @arg @ref LL_ADC_MULTI_REG_DMA_LIMIT_1
  *         @arg @ref LL_ADC_MULTI_REG_DMA_LIMIT_2
  *         @arg @ref LL_ADC_MULTI_REG_DMA_LIMIT_3
  *         @arg @ref LL_ADC_MULTI_REG_DMA_UNLMT_1
  *         @arg @ref LL_ADC_MULTI_REG_DMA_UNLMT_2
  *         @arg @ref LL_ADC_MULTI_REG_DMA_UNLMT_3
  */
__STATIC_INLINE uint32_t LL_ADC_GetMultiDMATransfer(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (uint32_t)(READ_BIT(ADCxy_COMMON->CCR, ADC_CCR_DMA | ADC_CCR_DDS));
}

/**
  * @brief  Set ADC multimode delay between 2 sampling phases.
  * @note   The sampling delay range depends on ADC resolution:
  *         - ADC resolution 12 bits can have maximum delay of 12 cycles.
  *         - ADC resolution 10 bits can have maximum delay of 10 cycles.
  *         - ADC resolution  8 bits can have maximum delay of  8 cycles.
  *         - ADC resolution  6 bits can have maximum delay of  6 cycles.
  * @rmtoll CCR      DELAY          LL_ADC_SetMultiTwoSamplingDelay
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @param  MultiTwoSamplingDelay This parameter can be one of the following values:
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_5CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_6CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_7CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_8CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_9CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_10CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_11CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_12CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_13CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_14CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_15CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_16CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_17CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_18CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_19CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_20CYCLES
  * @retval None
  */
__STATIC_INLINE void LL_ADC_SetMultiTwoSamplingDelay(ADC_Common_TypeDef *ADCxy_COMMON, uint32_t MultiTwoSamplingDelay)
{
  MODIFY_REG(ADCxy_COMMON->CCR, ADC_CCR_DELAY, MultiTwoSamplingDelay);
}

/**
  * @brief  Get ADC multimode delay between 2 sampling phases.
  * @rmtoll CCR      DELAY          LL_ADC_GetMultiTwoSamplingDelay
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_5CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_6CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_7CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_8CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_9CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_10CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_11CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_12CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_13CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_14CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_15CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_16CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_17CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_18CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_19CYCLES
  *         @arg @ref LL_ADC_MULTI_TWOSMP_DELAY_20CYCLES
  */
__STATIC_INLINE uint32_t LL_ADC_GetMultiTwoSamplingDelay(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (uint32_t)(READ_BIT(ADCxy_COMMON->CCR, ADC_CCR_DELAY));
}
#endif /* ADC_MULTIMODE_SUPPORT */

/**
  * @}
  */
/** @defgroup ADC_LL_EF_Operation_ADC_Instance Operation on ADC hierarchical scope: ADC instance
  * @{
  */

/**
  * @brief  Enable the selected ADC instance.
  * @note   On this STM32 series, after ADC enable, a delay for 
  *         ADC internal analog stabilization is required before performing a
  *         ADC conversion start.
  *         Refer to device datasheet, parameter tSTAB.
  * @rmtoll CR2      ADON           LL_ADC_Enable
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_Enable(ADC_TypeDef *ADCx)
{
  SET_BIT(ADCx->CR2, ADC_CR2_ADON);
}

/**
  * @brief  Disable the selected ADC instance.
  * @rmtoll CR2      ADON           LL_ADC_Disable
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_Disable(ADC_TypeDef *ADCx)
{
  CLEAR_BIT(ADCx->CR2, ADC_CR2_ADON);
}

/**
  * @brief  Get the selected ADC instance enable state.
  * @rmtoll CR2      ADON           LL_ADC_IsEnabled
  * @param  ADCx ADC instance
  * @retval 0: ADC is disabled, 1: ADC is enabled.
  */
__STATIC_INLINE uint32_t LL_ADC_IsEnabled(ADC_TypeDef *ADCx)
{
  return (READ_BIT(ADCx->CR2, ADC_CR2_ADON) == (ADC_CR2_ADON));
}

/**
  * @}
  */

/** @defgroup ADC_LL_EF_Operation_ADC_Group_Regular Operation on ADC hierarchical scope: group regular
  * @{
  */

/**
  * @brief  Start ADC group regular conversion.
  * @note   On this STM32 series, this function is relevant only for
  *         internal trigger (SW start), not for external trigger:
  *         - If ADC trigger has been set to software start, ADC conversion
  *           starts immediately.
  *         - If ADC trigger has been set to external trigger, ADC conversion
  *           start must be performed using function 
  *           @ref LL_ADC_REG_StartConversionExtTrig().
  *           (if external trigger edge would have been set during ADC other 
  *           settings, ADC conversion would start at trigger event
  *           as soon as ADC is enabled).
  * @rmtoll CR2      SWSTART        LL_ADC_REG_StartConversionSWStart
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_StartConversionSWStart(ADC_TypeDef *ADCx)
{
  SET_BIT(ADCx->CR2, ADC_CR2_SWSTART);
}

/**
  * @brief  Start ADC group regular conversion from external trigger.
  * @note   ADC conversion will start at next trigger event (on the selected
  *         trigger edge) following the ADC start conversion command.
  * @note   On this STM32 series, this function is relevant for 
  *         ADC conversion start from external trigger.
  *         If internal trigger (SW start) is needed, perform ADC conversion
  *         start using function @ref LL_ADC_REG_StartConversionSWStart().
  * @rmtoll CR2      EXTEN          LL_ADC_REG_StartConversionExtTrig
  * @param  ExternalTriggerEdge This parameter can be one of the following values:
  *         @arg @ref LL_ADC_REG_TRIG_EXT_RISING
  *         @arg @ref LL_ADC_REG_TRIG_EXT_FALLING
  *         @arg @ref LL_ADC_REG_TRIG_EXT_RISINGFALLING
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_StartConversionExtTrig(ADC_TypeDef *ADCx, uint32_t ExternalTriggerEdge)
{
  SET_BIT(ADCx->CR2, ExternalTriggerEdge);
}

/**
  * @brief  Stop ADC group regular conversion from external trigger.
  * @note   No more ADC conversion will start at next trigger event
  *         following the ADC stop conversion command.
  *         If a conversion is on-going, it will be completed.
  * @note   On this STM32 series, there is no specific command
  *         to stop a conversion on-going or to stop ADC converting
  *         in continuous mode. These actions can be performed
  *         using function @ref LL_ADC_Disable().
  * @rmtoll CR2      EXTEN          LL_ADC_REG_StopConversionExtTrig
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_REG_StopConversionExtTrig(ADC_TypeDef *ADCx)
{
  CLEAR_BIT(ADCx->CR2, ADC_CR2_EXTEN);
}

/**
  * @brief  Get ADC group regular conversion data, range fit for
  *         all ADC configurations: all ADC resolutions and
  *         all oversampling increased data width (for devices
  *         with feature oversampling).
  * @rmtoll DR       RDATA          LL_ADC_REG_ReadConversionData32
  * @param  ADCx ADC instance
  * @retval Value between Min_Data=0x00000000 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t LL_ADC_REG_ReadConversionData32(ADC_TypeDef *ADCx)
{
  return (uint16_t)(READ_BIT(ADCx->DR, ADC_DR_DATA));
}

/**
  * @brief  Get ADC group regular conversion data, range fit for
  *         ADC resolution 12 bits.
  * @note   For devices with feature oversampling: Oversampling
  *         can increase data width, function for extended range
  *         may be needed: @ref LL_ADC_REG_ReadConversionData32.
  * @rmtoll DR       RDATA          LL_ADC_REG_ReadConversionData12
  * @param  ADCx ADC instance
  * @retval Value between Min_Data=0x000 and Max_Data=0xFFF
  */
__STATIC_INLINE uint16_t LL_ADC_REG_ReadConversionData12(ADC_TypeDef *ADCx)
{
  return (uint16_t)(READ_BIT(ADCx->DR, ADC_DR_DATA));
}

/**
  * @brief  Get ADC group regular conversion data, range fit for
  *         ADC resolution 10 bits.
  * @note   For devices with feature oversampling: Oversampling
  *         can increase data width, function for extended range
  *         may be needed: @ref LL_ADC_REG_ReadConversionData32.
  * @rmtoll DR       RDATA          LL_ADC_REG_ReadConversionData10
  * @param  ADCx ADC instance
  * @retval Value between Min_Data=0x000 and Max_Data=0x3FF
  */
__STATIC_INLINE uint16_t LL_ADC_REG_ReadConversionData10(ADC_TypeDef *ADCx)
{
  return (uint16_t)(READ_BIT(ADCx->DR, ADC_DR_DATA));
}

/**
  * @brief  Get ADC group regular conversion data, range fit for
  *         ADC resolution 8 bits.
  * @note   For devices with feature oversampling: Oversampling
  *         can increase data width, function for extended range
  *         may be needed: @ref LL_ADC_REG_ReadConversionData32.
  * @rmtoll DR       RDATA          LL_ADC_REG_ReadConversionData8
  * @param  ADCx ADC instance
  * @retval Value between Min_Data=0x00 and Max_Data=0xFF
  */
__STATIC_INLINE uint8_t LL_ADC_REG_ReadConversionData8(ADC_TypeDef *ADCx)
{
  return (uint16_t)(READ_BIT(ADCx->DR, ADC_DR_DATA));
}

/**
  * @brief  Get ADC group regular conversion data, range fit for
  *         ADC resolution 6 bits.
  * @note   For devices with feature oversampling: Oversampling
  *         can increase data width, function for extended range
  *         may be needed: @ref LL_ADC_REG_ReadConversionData32.
  * @rmtoll DR       RDATA          LL_ADC_REG_ReadConversionData6
  * @param  ADCx ADC instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x3F
  */
__STATIC_INLINE uint8_t LL_ADC_REG_ReadConversionData6(ADC_TypeDef *ADCx)
{
  return (uint16_t)(READ_BIT(ADCx->DR, ADC_DR_DATA));
}

#if defined(ADC_MULTIMODE_SUPPORT)
/**
  * @brief  Get ADC multimode conversion data of ADC master, ADC slave
  *         or raw data with ADC master and slave concatenated.
  * @note   If raw data with ADC master and slave concatenated is retrieved,
  *         a macro is available to get the conversion data of
  *         ADC master or ADC slave: see helper macro
  *         @ref __LL_ADC_MULTI_CONV_DATA_MASTER_SLAVE().
  *         (however this macro is mainly intended for multimode
  *         transfer by DMA, because this function can do the same
  *         by getting multimode conversion data of ADC master or ADC slave
  *         separately).
  * @rmtoll CDR      DATA1          LL_ADC_REG_ReadMultiConversionData32\n
  *         CDR      DATA2          LL_ADC_REG_ReadMultiConversionData32
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @param  ConversionData This parameter can be one of the following values:
  *         @arg @ref LL_ADC_MULTI_MASTER
  *         @arg @ref LL_ADC_MULTI_SLAVE
  *         @arg @ref LL_ADC_MULTI_MASTER_SLAVE
  * @retval Value between Min_Data=0x00000000 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t LL_ADC_REG_ReadMultiConversionData32(ADC_Common_TypeDef *ADCxy_COMMON, uint32_t ConversionData)
{
  return (uint32_t)(READ_BIT(ADCxy_COMMON->CDR,
                             ADC_DR_ADC2DATA)
                    >> POSITION_VAL(ConversionData)
                   );
}
#endif /* ADC_MULTIMODE_SUPPORT */

/**
  * @}
  */

/** @defgroup ADC_LL_EF_Operation_ADC_Group_Injected Operation on ADC hierarchical scope: group injected
  * @{
  */

/**
  * @brief  Start ADC group injected conversion.
  * @note   On this STM32 series, this function is relevant only for
  *         internal trigger (SW start), not for external trigger:
  *         - If ADC trigger has been set to software start, ADC conversion
  *           starts immediately.
  *         - If ADC trigger has been set to external trigger, ADC conversion
  *           start must be performed using function 
  *           @ref LL_ADC_INJ_StartConversionExtTrig().
  *           (if external trigger edge would have been set during ADC other 
  *           settings, ADC conversion would start at trigger event
  *           as soon as ADC is enabled).
  * @rmtoll CR2      JSWSTART       LL_ADC_INJ_StartConversionSWStart
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_INJ_StartConversionSWStart(ADC_TypeDef *ADCx)
{
  SET_BIT(ADCx->CR2, ADC_CR2_JSWSTART);
}

/**
  * @brief  Start ADC group injected conversion from external trigger.
  * @note   ADC conversion will start at next trigger event (on the selected
  *         trigger edge) following the ADC start conversion command.
  * @note   On this STM32 series, this function is relevant for 
  *         ADC conversion start from external trigger.
  *         If internal trigger (SW start) is needed, perform ADC conversion
  *         start using function @ref LL_ADC_INJ_StartConversionSWStart().
  * @rmtoll CR2      JEXTEN         LL_ADC_INJ_StartConversionExtTrig
  * @param  ExternalTriggerEdge This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_RISING
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_FALLING
  *         @arg @ref LL_ADC_INJ_TRIG_EXT_RISINGFALLING
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_INJ_StartConversionExtTrig(ADC_TypeDef *ADCx, uint32_t ExternalTriggerEdge)
{
  SET_BIT(ADCx->CR2, ExternalTriggerEdge);
}

/**
  * @brief  Stop ADC group injected conversion from external trigger.
  * @note   No more ADC conversion will start at next trigger event
  *         following the ADC stop conversion command.
  *         If a conversion is on-going, it will be completed.
  * @note   On this STM32 series, there is no specific command
  *         to stop a conversion on-going or to stop ADC converting
  *         in continuous mode. These actions can be performed
  *         using function @ref LL_ADC_Disable().
  * @rmtoll CR2      JEXTEN         LL_ADC_INJ_StopConversionExtTrig
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_INJ_StopConversionExtTrig(ADC_TypeDef *ADCx)
{
  CLEAR_BIT(ADCx->CR2, ADC_CR2_JEXTEN);
}

/**
  * @brief  Get ADC group regular conversion data, range fit for
  *         all ADC configurations: all ADC resolutions and
  *         all oversampling increased data width (for devices
  *         with feature oversampling).
  * @rmtoll JDR1     JDATA          LL_ADC_INJ_ReadConversionData32\n
  *         JDR2     JDATA          LL_ADC_INJ_ReadConversionData32\n
  *         JDR3     JDATA          LL_ADC_INJ_ReadConversionData32\n
  *         JDR4     JDATA          LL_ADC_INJ_ReadConversionData32
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_RANK_1
  *         @arg @ref LL_ADC_INJ_RANK_2
  *         @arg @ref LL_ADC_INJ_RANK_3
  *         @arg @ref LL_ADC_INJ_RANK_4
  * @retval Value between Min_Data=0x00000000 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t LL_ADC_INJ_ReadConversionData32(ADC_TypeDef *ADCx, uint32_t Rank)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->JDR1, __ADC_MASK_SHIFT(Rank, ADC_INJ_JDRX_REGOFFSET_MASK));
  
  return (uint32_t)(READ_BIT(*preg,
                             ADC_JDR1_JDATA)
                   );
}

/**
  * @brief  Get ADC group injected conversion data, range fit for
  *         ADC resolution 12 bits.
  * @note   For devices with feature oversampling: Oversampling
  *         can increase data width, function for extended range
  *         may be needed: @ref LL_ADC_INJ_ReadConversionData32.
  * @rmtoll JDR1     JDATA          LL_ADC_INJ_ReadConversionData12\n
  *         JDR2     JDATA          LL_ADC_INJ_ReadConversionData12\n
  *         JDR3     JDATA          LL_ADC_INJ_ReadConversionData12\n
  *         JDR4     JDATA          LL_ADC_INJ_ReadConversionData12
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_RANK_1
  *         @arg @ref LL_ADC_INJ_RANK_2
  *         @arg @ref LL_ADC_INJ_RANK_3
  *         @arg @ref LL_ADC_INJ_RANK_4
  * @retval Value between Min_Data=0x000 and Max_Data=0xFFF
  */
__STATIC_INLINE uint16_t LL_ADC_INJ_ReadConversionData12(ADC_TypeDef *ADCx, uint32_t Rank)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->JDR1, __ADC_MASK_SHIFT(Rank, ADC_INJ_JDRX_REGOFFSET_MASK));
  
  return (uint16_t)(READ_BIT(*preg,
                             ADC_JDR1_JDATA)
                   );
}

/**
  * @brief  Get ADC group injected conversion data, range fit for
  *         ADC resolution 10 bits.
  * @note   For devices with feature oversampling: Oversampling
  *         can increase data width, function for extended range
  *         may be needed: @ref LL_ADC_INJ_ReadConversionData32.
  * @rmtoll JDR1     JDATA          LL_ADC_INJ_ReadConversionData10\n
  *         JDR2     JDATA          LL_ADC_INJ_ReadConversionData10\n
  *         JDR3     JDATA          LL_ADC_INJ_ReadConversionData10\n
  *         JDR4     JDATA          LL_ADC_INJ_ReadConversionData10
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_RANK_1
  *         @arg @ref LL_ADC_INJ_RANK_2
  *         @arg @ref LL_ADC_INJ_RANK_3
  *         @arg @ref LL_ADC_INJ_RANK_4
  * @retval Value between Min_Data=0x000 and Max_Data=0x3FF
  */
__STATIC_INLINE uint16_t LL_ADC_INJ_ReadConversionData10(ADC_TypeDef *ADCx, uint32_t Rank)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->JDR1, __ADC_MASK_SHIFT(Rank, ADC_INJ_JDRX_REGOFFSET_MASK));
  
  return (uint16_t)(READ_BIT(*preg,
                             ADC_JDR1_JDATA)
                   );
}

/**
  * @brief  Get ADC group injected conversion data, range fit for
  *         ADC resolution 8 bits.
  * @note   For devices with feature oversampling: Oversampling
  *         can increase data width, function for extended range
  *         may be needed: @ref LL_ADC_INJ_ReadConversionData32.
  * @rmtoll JDR1     JDATA          LL_ADC_INJ_ReadConversionData8\n
  *         JDR2     JDATA          LL_ADC_INJ_ReadConversionData8\n
  *         JDR3     JDATA          LL_ADC_INJ_ReadConversionData8\n
  *         JDR4     JDATA          LL_ADC_INJ_ReadConversionData8
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_RANK_1
  *         @arg @ref LL_ADC_INJ_RANK_2
  *         @arg @ref LL_ADC_INJ_RANK_3
  *         @arg @ref LL_ADC_INJ_RANK_4
  * @retval Value between Min_Data=0x00 and Max_Data=0xFF
  */
__STATIC_INLINE uint8_t LL_ADC_INJ_ReadConversionData8(ADC_TypeDef *ADCx, uint32_t Rank)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->JDR1, __ADC_MASK_SHIFT(Rank, ADC_INJ_JDRX_REGOFFSET_MASK));
  
  return (uint8_t)(READ_BIT(*preg,
                            ADC_JDR1_JDATA)
                  );
}

/**
  * @brief  Get ADC group injected conversion data, range fit for
  *         ADC resolution 6 bits.
  * @note   For devices with feature oversampling: Oversampling
  *         can increase data width, function for extended range
  *         may be needed: @ref LL_ADC_INJ_ReadConversionData32.
  * @rmtoll JDR1     JDATA          LL_ADC_INJ_ReadConversionData6\n
  *         JDR2     JDATA          LL_ADC_INJ_ReadConversionData6\n
  *         JDR3     JDATA          LL_ADC_INJ_ReadConversionData6\n
  *         JDR4     JDATA          LL_ADC_INJ_ReadConversionData6
  * @param  ADCx ADC instance
  * @param  Rank This parameter can be one of the following values:
  *         @arg @ref LL_ADC_INJ_RANK_1
  *         @arg @ref LL_ADC_INJ_RANK_2
  *         @arg @ref LL_ADC_INJ_RANK_3
  *         @arg @ref LL_ADC_INJ_RANK_4
  * @retval Value between Min_Data=0x00 and Max_Data=0x3F
  */
__STATIC_INLINE uint8_t LL_ADC_INJ_ReadConversionData6(ADC_TypeDef *ADCx, uint32_t Rank)
{
  __IO uint32_t *preg = __ADC_PTR_REG_OFFSET(ADCx->JDR1, __ADC_MASK_SHIFT(Rank, ADC_INJ_JDRX_REGOFFSET_MASK));
  
  return (uint8_t)(READ_BIT(*preg,
                            ADC_JDR1_JDATA)
                  );
}

/**
  * @}
  */

/** @defgroup ADC_LL_EF_FLAG_Management ADC flag management
  * @{
  */

/**
  * @brief  Get flag ADC group regular end of unitary conversion
  *         or end of sequence conversions, depending on
  *         ADC configuration.
  * @note   To configure flag of end of conversion,
  *         use function @ref LL_ADC_REG_SetFlagEndOfConversion().
  * @rmtoll SR       EOC            LL_ADC_IsActiveFlag_EOCS
  * @param  ADCx ADC instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_EOCS(ADC_TypeDef *ADCx)
{
  return (READ_BIT(ADCx->SR, LL_ADC_FLAG_EOCS) == (LL_ADC_FLAG_EOCS));
}

/**
  * @brief  Get flag ADC group regular overrun.
  * @rmtoll SR       OVR            LL_ADC_IsActiveFlag_OVR
  * @param  ADCx ADC instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_OVR(ADC_TypeDef *ADCx)
{
  return (READ_BIT(ADCx->SR, LL_ADC_FLAG_OVR) == (LL_ADC_FLAG_OVR));
}


/**
  * @brief  Get flag ADC group injected end of sequence conversions.
  * @rmtoll SR       JEOC           LL_ADC_IsActiveFlag_JEOS
  * @param  ADCx ADC instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_JEOS(ADC_TypeDef *ADCx)
{
  /* Note: on this STM32 series, there is no flag ADC group injected          */
  /*       end of unitary conversion.                                         */
  /*       Flag noted as "JEOC" is corresponding to flag "JEOS"               */
  /*       in other STM32 families).                                          */
  return (READ_BIT(ADCx->SR, LL_ADC_FLAG_JEOS) == (LL_ADC_FLAG_JEOS));
}

/**
  * @brief  Get flag ADC analog watchdog 1 flag
  * @rmtoll SR       AWD            LL_ADC_IsActiveFlag_AWD1
  * @param  ADCx ADC instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_AWD1(ADC_TypeDef *ADCx)
{
  return (READ_BIT(ADCx->SR, LL_ADC_FLAG_AWD1) == (LL_ADC_FLAG_AWD1));
}

/**
  * @brief  Clear flag ADC group regular end of unitary conversion
  *         or end of sequence conversions, depending on
  *         ADC configuration.
  * @note   To configure flag of end of conversion,
  *         use function @ref LL_ADC_REG_SetFlagEndOfConversion().
  * @rmtoll SR       EOC            LL_ADC_ClearFlag_EOCS
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_ClearFlag_EOCS(ADC_TypeDef *ADCx)
{
  WRITE_REG(ADCx->SR, ~LL_ADC_FLAG_EOCS);
}

/**
  * @brief  Clear flag ADC group regular overrun.
  * @rmtoll SR       OVR            LL_ADC_ClearFlag_OVR
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_ClearFlag_OVR(ADC_TypeDef *ADCx)
{
  WRITE_REG(ADCx->SR, ~LL_ADC_FLAG_OVR);
}


/**
  * @brief  Clear flag ADC group injected end of sequence conversions.
  * @rmtoll SR       JEOC           LL_ADC_ClearFlag_JEOS
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_ClearFlag_JEOS(ADC_TypeDef *ADCx)
{
  /* Note: on this STM32 series, there is no flag ADC group injected          */
  /*       end of unitary conversion.                                         */
  /*       Flag noted as "JEOC" is corresponding to flag "JEOS"               */
  /*       in other STM32 families).                                          */
  WRITE_REG(ADCx->SR, ~LL_ADC_FLAG_JEOS);
}

/**
  * @brief  Clear flag ADC analog watchdog 1.
  * @rmtoll SR       AWD            LL_ADC_ClearFlag_AWD1
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_ClearFlag_AWD1(ADC_TypeDef *ADCx)
{
  WRITE_REG(ADCx->SR, ~LL_ADC_FLAG_AWD1);
}

#if defined(ADC_MULTIMODE_SUPPORT)
/**
  * @brief  Get flag multimode ADC group regular end of unitary conversion
  *         or end of sequence conversions, depending on
  *         ADC configuration, of the ADC master.
  * @note   To configure flag of end of conversion,
  *         use function @ref LL_ADC_REG_SetFlagEndOfConversion().
  * @rmtoll CSR      EOC1           LL_ADC_IsActiveFlag_MST_EOCS
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_MST_EOCS(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (READ_BIT(ADC1->SR, LL_ADC_FLAG_EOCS) == (LL_ADC_FLAG_EOCS));
}

/**
  * @brief  Get flag multimode ADC group regular end of unitary conversion
  *         or end of sequence conversions, depending on
  *         ADC configuration, of the ADC slave 1.
  * @note   To configure flag of end of conversion,
  *         use function @ref LL_ADC_REG_SetFlagEndOfConversion().
  * @rmtoll CSR      EOC2           LL_ADC_IsActiveFlag_SLV1_EOCS
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_SLV1_EOCS(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (READ_BIT(ADCxy_COMMON->CSR, LL_ADC_FLAG_EOCS_SLV1) == (LL_ADC_FLAG_EOCS_SLV1));
}

/**
  * @brief  Get flag multimode ADC group regular end of unitary conversion
  *         or end of sequence conversions, depending on
  *         ADC configuration, of the ADC slave 2.
  * @note   To configure flag of end of conversion,
  *         use function @ref LL_ADC_REG_SetFlagEndOfConversion().
  * @rmtoll CSR      EOC3           LL_ADC_IsActiveFlag_SLV2_EOCS
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_SLV2_EOCS(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (READ_BIT(ADCxy_COMMON->CSR, LL_ADC_FLAG_EOCS_SLV2) == (LL_ADC_FLAG_EOCS_SLV2));
}
/**
  * @brief  Get flag multimode ADC group regular overrun of the ADC master.
  * @rmtoll CSR      OVR1           LL_ADC_IsActiveFlag_MST_OVR
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_MST_OVR(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (READ_BIT(ADCxy_COMMON->CSR, LL_ADC_FLAG_OVR_MST) == (LL_ADC_FLAG_OVR_MST));
}

/**
  * @brief  Get flag multimode ADC group regular overrun of the ADC slave 1.
  * @rmtoll CSR      OVR2           LL_ADC_IsActiveFlag_SLV1_OVR
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_SLV1_OVR(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (READ_BIT(ADCxy_COMMON->CSR, LL_ADC_FLAG_OVR_SLV1) == (LL_ADC_FLAG_OVR_SLV1));
}

/**
  * @brief  Get flag multimode ADC group regular overrun of the ADC slave 2.
  * @rmtoll CSR      OVR3           LL_ADC_IsActiveFlag_SLV2_OVR
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_SLV2_OVR(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (READ_BIT(ADCxy_COMMON->CSR, LL_ADC_FLAG_OVR_SLV2) == (LL_ADC_FLAG_OVR_SLV2));
}


/**
  * @brief  Get flag multimode ADC group injected end of sequence conversions of the ADC master.
  * @rmtoll CSR      JEOC           LL_ADC_IsActiveFlag_MST_EOCS
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_MST_JEOS(ADC_Common_TypeDef *ADCxy_COMMON)
{
  /* Note: on this STM32 series, there is no flag ADC group injected          */
  /*       end of unitary conversion.                                         */
  /*       Flag noted as "JEOC" is corresponding to flag "JEOS"               */
  /*       in other STM32 families).                                          */
  return (READ_BIT(ADCxy_COMMON->CSR, ADC_CSR_JEOC1) == (ADC_CSR_JEOC1));
}

/**
  * @brief  Get flag multimode ADC group injected end of sequence conversions of the ADC slave 1.
  * @rmtoll CSR      JEOC2          LL_ADC_IsActiveFlag_SLV1_JEOS
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_SLV1_JEOS(ADC_Common_TypeDef *ADCxy_COMMON)
{
  /* Note: on this STM32 series, there is no flag ADC group injected          */
  /*       end of unitary conversion.                                         */
  /*       Flag noted as "JEOC" is corresponding to flag "JEOS"               */
  /*       in other STM32 families).                                          */
  return (READ_BIT(ADCxy_COMMON->CSR, ADC_CSR_JEOC2) == (ADC_CSR_JEOC2));
}

/**
  * @brief  Get flag multimode ADC group injected end of sequence conversions of the ADC slave 2.
  * @rmtoll CSR      JEOC3          LL_ADC_IsActiveFlag_SLV2_JEOS
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_SLV2_JEOS(ADC_Common_TypeDef *ADCxy_COMMON)
{
  /* Note: on this STM32 series, there is no flag ADC group injected          */
  /*       end of unitary conversion.                                         */
  /*       Flag noted as "JEOC" is corresponding to flag "JEOS"               */
  /*       in other STM32 families).                                          */
  return (READ_BIT(ADCxy_COMMON->CSR, ADC_CSR_JEOC3) == (ADC_CSR_JEOC3));
}

/**
  * @brief  Get flag multimode ADC analog watchdog 1 of the ADC master.
  * @rmtoll CSR      AWD1           LL_ADC_IsActiveFlag_MST_AWD1
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_MST_AWD1(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (READ_BIT(ADCxy_COMMON->CSR, LL_ADC_FLAG_AWD1_MST) == (LL_ADC_FLAG_AWD1_MST));
}

/**
  * @brief  Get flag multimode analog watchdog 1 of the ADC slave 1.
  * @rmtoll CSR      AWD2           LL_ADC_IsActiveFlag_SLV1_AWD1
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_SLV1_AWD1(ADC_Common_TypeDef *ADCxy_COMMON)
{
  return (READ_BIT(ADCxy_COMMON->CSR, LL_ADC_FLAG_AWD1_SLV1) == (LL_ADC_FLAG_AWD1_SLV1));
}

/**
  * @brief  Get flag multimode analog watchdog 1 of the ADC slave 2.
  * @rmtoll CSR      AWD3           LL_ADC_IsActiveFlag_SLV2_AWD1
  * @param  ADCxy_COMMON ADC common instance
  *         (can be set directly from CMSIS definition or by using helper macro @ref __LL_ADC_COMMON_INSTANCE() )
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsActiveFlag_SLV2_AWD1(ADC_Common_TypeDef *ADCxy_COMMON)
{
    return (READ_BIT(ADCxy_COMMON->CSR, LL_ADC_FLAG_AWD1_SLV2) == (LL_ADC_FLAG_AWD1_SLV2));
}

#endif /* ADC_MULTIMODE_SUPPORT */

/**
  * @}
  */

/** @defgroup ADC_LL_EF_IT_Management ADC IT management
  * @{
  */

/**
  * @brief  Enable interruption ADC group regular end of unitary conversion
  *         or end of sequence conversions, depending on
  *         ADC configuration.
  * @note   To configure flag of end of conversion,
  *         use function @ref LL_ADC_REG_SetFlagEndOfConversion().
  * @rmtoll CR1      EOCIE          LL_ADC_EnableIT_EOCS
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_EnableIT_EOCS(ADC_TypeDef *ADCx)
{
  SET_BIT(ADCx->CR1, LL_ADC_IT_EOCS);
}

/**
  * @brief  Enable ADC group regular interruption overrun.
  * @rmtoll CR1      OVRIE          LL_ADC_EnableIT_OVR
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_EnableIT_OVR(ADC_TypeDef *ADCx)
{
  SET_BIT(ADCx->CR1, LL_ADC_IT_OVR);
}


/**
  * @brief  Enable interruption ADC group injected end of sequence conversions.
  * @rmtoll CR1      JEOCIE         LL_ADC_EnableIT_JEOS
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_EnableIT_JEOS(ADC_TypeDef *ADCx)
{
  /* Note: on this STM32 series, there is no flag ADC group injected          */
  /*       end of unitary conversion.                                         */
  /*       Flag noted as "JEOC" is corresponding to flag "JEOS"               */
  /*       in other STM32 families).                                          */
  SET_BIT(ADCx->CR1, LL_ADC_IT_JEOS);
}

/**
  * @brief  Enable interruption ADC analog watchdog 1.
  * @rmtoll CR1      AWDIE          LL_ADC_EnableIT_AWD1
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_EnableIT_AWD1(ADC_TypeDef *ADCx)
{
  SET_BIT(ADCx->CR1, LL_ADC_IT_AWD1);
}

/**
  * @brief  Disable interruption ADC group regular end of unitary conversion
  *         or end of sequence conversions, depending on
  *         ADC configuration.
  * @note   To configure flag of end of conversion,
  *         use function @ref LL_ADC_REG_SetFlagEndOfConversion().
  * @rmtoll CR1      EOCIE          LL_ADC_DisableIT_EOCS
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_DisableIT_EOCS(ADC_TypeDef *ADCx)
{
  CLEAR_BIT(ADCx->CR1, LL_ADC_IT_EOCS);
}

/**
  * @brief  Disable interruption ADC group regular overrun.
  * @rmtoll CR1      OVRIE          LL_ADC_DisableIT_OVR
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_DisableIT_OVR(ADC_TypeDef *ADCx)
{
  CLEAR_BIT(ADCx->CR1, LL_ADC_IT_OVR);
}


/**
  * @brief  Disable interruption ADC group injected end of sequence conversions.
  * @rmtoll CR1      JEOCIE         LL_ADC_EnableIT_JEOS
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_DisableIT_JEOS(ADC_TypeDef *ADCx)
{
  /* Note: on this STM32 series, there is no flag ADC group injected          */
  /*       end of unitary conversion.                                         */
  /*       Flag noted as "JEOC" is corresponding to flag "JEOS"               */
  /*       in other STM32 families).                                          */
  CLEAR_BIT(ADCx->CR1, LL_ADC_IT_JEOS);
}

/**
  * @brief  Disable interruption ADC analog watchdog 1.
  * @rmtoll CR1      AWDIE          LL_ADC_EnableIT_AWD1
  * @param  ADCx ADC instance
  * @retval None
  */
__STATIC_INLINE void LL_ADC_DisableIT_AWD1(ADC_TypeDef *ADCx)
{
  CLEAR_BIT(ADCx->CR1, LL_ADC_IT_AWD1);
}

/**
  * @brief  Get state of interruption ADC group regular end of unitary conversion
  *         or end of sequence conversions, depending on
  *         ADC configuration.
  * @note   To configure flag of end of conversion,
  *         use function @ref LL_ADC_REG_SetFlagEndOfConversion().
  *         (0: interrupt disabled, 1: interrupt enabled)
  * @rmtoll CR1      EOCIE          LL_ADC_IsEnabledIT_EOCS
  * @param  ADCx ADC instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsEnabledIT_EOCS(ADC_TypeDef *ADCx)
{
  return (READ_BIT(ADCx->CR1, LL_ADC_IT_EOCS) == (LL_ADC_IT_EOCS));
}

/**
  * @brief  Get state of interruption ADC group regular overrun
  *         (0: interrupt disabled, 1: interrupt enabled).
  * @rmtoll CR1      OVRIE          LL_ADC_IsEnabledIT_OVR
  * @param  ADCx ADC instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsEnabledIT_OVR(ADC_TypeDef *ADCx)
{
  return (READ_BIT(ADCx->CR1, LL_ADC_IT_OVR) == (LL_ADC_IT_OVR));
}


/**
  * @brief  Get state of interruption ADC group injected end of sequence conversions
  *         (0: interrupt disabled, 1: interrupt enabled).
  * @rmtoll CR1      JEOCIE         LL_ADC_EnableIT_JEOS
  * @param  ADCx ADC instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsEnabledIT_JEOS(ADC_TypeDef *ADCx)
{
  /* Note: on this STM32 series, there is no flag ADC group injected          */
  /*       end of unitary conversion.                                         */
  /*       Flag noted as "JEOC" is corresponding to flag "JEOS"               */
  /*       in other STM32 families).                                          */
  return (READ_BIT(ADCx->CR1, LL_ADC_IT_JEOS) == (LL_ADC_IT_JEOS));
}

/**
  * @brief  Get state of interruption ADC analog watchdog 1
  *         (0: interrupt disabled, 1: interrupt enabled).
  * @rmtoll CR1      AWDIE          LL_ADC_EnableIT_AWD1
  * @param  ADCx ADC instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_ADC_IsEnabledIT_AWD1(ADC_TypeDef *ADCx)
{
  return (READ_BIT(ADCx->CR1, LL_ADC_IT_AWD1) == (LL_ADC_IT_AWD1));
}

/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup ADC_LL_EF_Init Initialization and de-initialization functions
  * @{
  */

/* Initialization of some features of ADC common parameters and multimode */
ErrorStatus LL_ADC_CommonDeInit(ADC_Common_TypeDef *ADCxy_COMMON);
ErrorStatus LL_ADC_CommonInit(ADC_Common_TypeDef *ADCxy_COMMON, LL_ADC_CommonInitTypeDef *ADC_CommonInitStruct);
void        LL_ADC_CommonStructInit(LL_ADC_CommonInitTypeDef *ADC_CommonInitStruct);

/* De-initialization of ADC instance, ADC group regular and ADC group injected */
/* (availability of ADC group injected depends on STM32 families) */
ErrorStatus LL_ADC_DeInit(ADC_TypeDef *ADCx);

/* Initialization of some features of ADC instance */
ErrorStatus LL_ADC_Init(ADC_TypeDef *ADCx, LL_ADC_InitTypeDef *ADC_InitStruct);
void        LL_ADC_StructInit(LL_ADC_InitTypeDef *ADC_InitStruct);

/* Initialization of some features of ADC instance and ADC group regular */
ErrorStatus LL_ADC_REG_Init(ADC_TypeDef *ADCx, LL_ADC_REG_InitTypeDef *ADC_REG_InitStruct);
void        LL_ADC_REG_StructInit(LL_ADC_REG_InitTypeDef *ADC_REG_InitStruct);

/* Initialization of some features of ADC instance and ADC group injected */
ErrorStatus LL_ADC_INJ_Init(ADC_TypeDef *ADCx, LL_ADC_INJ_InitTypeDef *ADC_INJ_InitStruct);
void        LL_ADC_INJ_StructInit(LL_ADC_INJ_InitTypeDef *ADC_INJ_InitStruct);

/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/**
  * @}
  */

/**
  * @}
  */

#endif /* ADC1 || ADC2 || ADC3 */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_LL_ADC_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
