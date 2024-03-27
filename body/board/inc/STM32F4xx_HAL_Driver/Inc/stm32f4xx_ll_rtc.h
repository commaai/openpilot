/**
  ******************************************************************************
  * @file    stm32f4xx_ll_rtc.h
  * @author  MCD Application Team
  * @brief   Header file of RTC LL module.
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
#ifndef __STM32F4xx_LL_RTC_H
#define __STM32F4xx_LL_RTC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined(RTC)

/** @defgroup RTC_LL RTC
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup RTC_LL_Private_Constants RTC Private Constants
  * @{
  */
/* Masks Definition */
#define RTC_INIT_MASK                 0xFFFFFFFFU
#define RTC_RSF_MASK                  0xFFFFFF5FU

/* Write protection defines */
#define RTC_WRITE_PROTECTION_DISABLE  ((uint8_t)0xFFU)
#define RTC_WRITE_PROTECTION_ENABLE_1 ((uint8_t)0xCAU)
#define RTC_WRITE_PROTECTION_ENABLE_2 ((uint8_t)0x53U)

/* Defines used to combine date & time */
#define RTC_OFFSET_WEEKDAY            24U
#define RTC_OFFSET_DAY                16U
#define RTC_OFFSET_MONTH              8U
#define RTC_OFFSET_HOUR               16U
#define RTC_OFFSET_MINUTE             8U

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup RTC_LL_Private_Macros RTC Private Macros
  * @{
  */
/**
  * @}
  */
#endif /*USE_FULL_LL_DRIVER*/

/* Exported types ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup RTC_LL_ES_INIT RTC Exported Init structure
  * @{
  */

/**
  * @brief  RTC Init structures definition
  */
typedef struct
{
  uint32_t HourFormat;   /*!< Specifies the RTC Hours Format.
                              This parameter can be a value of @ref RTC_LL_EC_HOURFORMAT
                              
                              This feature can be modified afterwards using unitary function
                              @ref LL_RTC_SetHourFormat(). */

  uint32_t AsynchPrescaler; /*!< Specifies the RTC Asynchronous Predivider value.
                              This parameter must be a number between Min_Data = 0x00 and Max_Data = 0x7F
                              
                              This feature can be modified afterwards using unitary function
                              @ref LL_RTC_SetAsynchPrescaler(). */

  uint32_t SynchPrescaler;  /*!< Specifies the RTC Synchronous Predivider value.
                              This parameter must be a number between Min_Data = 0x00 and Max_Data = 0x7FFF
                              
                              This feature can be modified afterwards using unitary function
                              @ref LL_RTC_SetSynchPrescaler(). */
} LL_RTC_InitTypeDef;

/**
  * @brief  RTC Time structure definition
  */
typedef struct
{
  uint32_t TimeFormat; /*!< Specifies the RTC AM/PM Time.
                            This parameter can be a value of @ref RTC_LL_EC_TIME_FORMAT

                            This feature can be modified afterwards using unitary function @ref LL_RTC_TIME_SetFormat(). */

  uint8_t Hours;       /*!< Specifies the RTC Time Hours.
                            This parameter must be a number between Min_Data = 0 and Max_Data = 12 if the @ref LL_RTC_TIME_FORMAT_PM is selected.
                            This parameter must be a number between Min_Data = 0 and Max_Data = 23 if the @ref LL_RTC_TIME_FORMAT_AM_OR_24 is selected.

                            This feature can be modified afterwards using unitary function @ref LL_RTC_TIME_SetHour(). */

  uint8_t Minutes;     /*!< Specifies the RTC Time Minutes.
                            This parameter must be a number between Min_Data = 0 and Max_Data = 59

                            This feature can be modified afterwards using unitary function @ref LL_RTC_TIME_SetMinute(). */

  uint8_t Seconds;     /*!< Specifies the RTC Time Seconds.
                            This parameter must be a number between Min_Data = 0 and Max_Data = 59

                            This feature can be modified afterwards using unitary function @ref LL_RTC_TIME_SetSecond(). */
} LL_RTC_TimeTypeDef;

/**
  * @brief  RTC Date structure definition
  */
typedef struct
{
  uint8_t WeekDay;  /*!< Specifies the RTC Date WeekDay.
                         This parameter can be a value of @ref RTC_LL_EC_WEEKDAY

                         This feature can be modified afterwards using unitary function @ref LL_RTC_DATE_SetWeekDay(). */

  uint8_t Month;    /*!< Specifies the RTC Date Month.
                         This parameter can be a value of @ref RTC_LL_EC_MONTH

                         This feature can be modified afterwards using unitary function @ref LL_RTC_DATE_SetMonth(). */

  uint8_t Day;      /*!< Specifies the RTC Date Day.
                         This parameter must be a number between Min_Data = 1 and Max_Data = 31

                         This feature can be modified afterwards using unitary function @ref LL_RTC_DATE_SetDay(). */

  uint8_t Year;     /*!< Specifies the RTC Date Year.
                         This parameter must be a number between Min_Data = 0 and Max_Data = 99

                         This feature can be modified afterwards using unitary function @ref LL_RTC_DATE_SetYear(). */
} LL_RTC_DateTypeDef;

/**
  * @brief  RTC Alarm structure definition
  */
typedef struct
{
  LL_RTC_TimeTypeDef AlarmTime;  /*!< Specifies the RTC Alarm Time members. */

  uint32_t AlarmMask;            /*!< Specifies the RTC Alarm Masks.
                                      This parameter can be a value of @ref RTC_LL_EC_ALMA_MASK for ALARM A or @ref RTC_LL_EC_ALMB_MASK for ALARM B.

                                      This feature can be modified afterwards using unitary function @ref LL_RTC_ALMA_SetMask() for ALARM A 
                                      or @ref LL_RTC_ALMB_SetMask() for ALARM B
                                 */

  uint32_t AlarmDateWeekDaySel;  /*!< Specifies the RTC Alarm is on day or WeekDay.
                                      This parameter can be a value of @ref RTC_LL_EC_ALMA_WEEKDAY_SELECTION for ALARM A or @ref RTC_LL_EC_ALMB_WEEKDAY_SELECTION for ALARM B

                                      This feature can be modified afterwards using unitary function @ref LL_RTC_ALMA_EnableWeekday() or @ref LL_RTC_ALMA_DisableWeekday()
                                      for ALARM A or @ref LL_RTC_ALMB_EnableWeekday() or @ref LL_RTC_ALMB_DisableWeekday() for ALARM B
                                 */

  uint8_t AlarmDateWeekDay;      /*!< Specifies the RTC Alarm Day/WeekDay.
                                      If AlarmDateWeekDaySel set to day, this parameter  must be a number between Min_Data = 1 and Max_Data = 31.

                                      This feature can be modified afterwards using unitary function @ref LL_RTC_ALMA_SetDay()
                                      for ALARM A or @ref LL_RTC_ALMB_SetDay() for ALARM B.

                                      If AlarmDateWeekDaySel set to Weekday, this parameter can be a value of @ref RTC_LL_EC_WEEKDAY.

                                      This feature can be modified afterwards using unitary function @ref LL_RTC_ALMA_SetWeekDay()
                                      for ALARM A or @ref LL_RTC_ALMB_SetWeekDay() for ALARM B.
                                 */
} LL_RTC_AlarmTypeDef;

/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/* Exported constants --------------------------------------------------------*/
/** @defgroup RTC_LL_Exported_Constants RTC Exported Constants
  * @{
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup RTC_LL_EC_FORMAT FORMAT
  * @{
  */
#define LL_RTC_FORMAT_BIN                  0x000000000U /*!< Binary data format */
#define LL_RTC_FORMAT_BCD                  0x000000001U /*!< BCD data format */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_ALMA_WEEKDAY_SELECTION RTC Alarm A Date WeekDay
  * @{
  */
#define LL_RTC_ALMA_DATEWEEKDAYSEL_DATE    0x00000000U             /*!< Alarm A Date is selected */
#define LL_RTC_ALMA_DATEWEEKDAYSEL_WEEKDAY RTC_ALRMAR_WDSEL        /*!< Alarm A WeekDay is selected */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_ALMB_WEEKDAY_SELECTION RTC Alarm B Date WeekDay
  * @{
  */
#define LL_RTC_ALMB_DATEWEEKDAYSEL_DATE    0x00000000U             /*!< Alarm B Date is selected */
#define LL_RTC_ALMB_DATEWEEKDAYSEL_WEEKDAY RTC_ALRMBR_WDSEL        /*!< Alarm B WeekDay is selected */
/**
  * @}
  */

#endif /* USE_FULL_LL_DRIVER */

/** @defgroup RTC_LL_EC_GET_FLAG Get Flags Defines
  * @brief    Flags defines which can be used with LL_RTC_ReadReg function
  * @{
  */
#define LL_RTC_ISR_RECALPF                 RTC_ISR_RECALPF
#define LL_RTC_ISR_TAMP3F                  RTC_ISR_TAMP3F
#define LL_RTC_ISR_TAMP2F                  RTC_ISR_TAMP2F
#define LL_RTC_ISR_TAMP1F                  RTC_ISR_TAMP1F
#define LL_RTC_ISR_TSOVF                   RTC_ISR_TSOVF
#define LL_RTC_ISR_TSF                     RTC_ISR_TSF
#define LL_RTC_ISR_WUTF                    RTC_ISR_WUTF
#define LL_RTC_ISR_ALRBF                   RTC_ISR_ALRBF
#define LL_RTC_ISR_ALRAF                   RTC_ISR_ALRAF
#define LL_RTC_ISR_INITF                   RTC_ISR_INITF
#define LL_RTC_ISR_RSF                     RTC_ISR_RSF
#define LL_RTC_ISR_INITS                   RTC_ISR_INITS
#define LL_RTC_ISR_SHPF                    RTC_ISR_SHPF
#define LL_RTC_ISR_WUTWF                   RTC_ISR_WUTWF
#define LL_RTC_ISR_ALRBWF                  RTC_ISR_ALRBWF
#define LL_RTC_ISR_ALRAWF                  RTC_ISR_ALRAWF
/**
  * @}
  */

/** @defgroup RTC_LL_EC_IT IT Defines
  * @brief    IT defines which can be used with LL_RTC_ReadReg and  LL_RTC_WriteReg functions
  * @{
  */
#define LL_RTC_CR_TSIE                     RTC_CR_TSIE
#define LL_RTC_CR_WUTIE                    RTC_CR_WUTIE
#define LL_RTC_CR_ALRBIE                   RTC_CR_ALRBIE
#define LL_RTC_CR_ALRAIE                   RTC_CR_ALRAIE
#define LL_RTC_TAFCR_TAMPIE               RTC_TAFCR_TAMPIE
/**
  * @}
  */

/** @defgroup RTC_LL_EC_WEEKDAY  WEEK DAY
  * @{
  */
#define LL_RTC_WEEKDAY_MONDAY              ((uint8_t)0x01U) /*!< Monday    */
#define LL_RTC_WEEKDAY_TUESDAY             ((uint8_t)0x02U) /*!< Tuesday   */
#define LL_RTC_WEEKDAY_WEDNESDAY           ((uint8_t)0x03U) /*!< Wednesday */
#define LL_RTC_WEEKDAY_THURSDAY            ((uint8_t)0x04U) /*!< Thrusday  */
#define LL_RTC_WEEKDAY_FRIDAY              ((uint8_t)0x05U) /*!< Friday    */
#define LL_RTC_WEEKDAY_SATURDAY            ((uint8_t)0x06U) /*!< Saturday  */
#define LL_RTC_WEEKDAY_SUNDAY              ((uint8_t)0x07U) /*!< Sunday    */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_MONTH  MONTH
  * @{
  */
#define LL_RTC_MONTH_JANUARY               ((uint8_t)0x01U)  /*!< January   */
#define LL_RTC_MONTH_FEBRUARY              ((uint8_t)0x02U)  /*!< February  */
#define LL_RTC_MONTH_MARCH                 ((uint8_t)0x03U)  /*!< March     */
#define LL_RTC_MONTH_APRIL                 ((uint8_t)0x04U)  /*!< April     */
#define LL_RTC_MONTH_MAY                   ((uint8_t)0x05U)  /*!< May       */
#define LL_RTC_MONTH_JUNE                  ((uint8_t)0x06U)  /*!< June      */
#define LL_RTC_MONTH_JULY                  ((uint8_t)0x07U)  /*!< July      */
#define LL_RTC_MONTH_AUGUST                ((uint8_t)0x08U)  /*!< August    */
#define LL_RTC_MONTH_SEPTEMBER             ((uint8_t)0x09U)  /*!< September */
#define LL_RTC_MONTH_OCTOBER               ((uint8_t)0x10U)  /*!< October   */
#define LL_RTC_MONTH_NOVEMBER              ((uint8_t)0x11U)  /*!< November  */
#define LL_RTC_MONTH_DECEMBER              ((uint8_t)0x12U)  /*!< December  */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_HOURFORMAT  HOUR FORMAT
  * @{
  */
#define LL_RTC_HOURFORMAT_24HOUR           0x00000000U           /*!< 24 hour/day format */
#define LL_RTC_HOURFORMAT_AMPM             RTC_CR_FMT            /*!< AM/PM hour format */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_ALARMOUT  ALARM OUTPUT
  * @{
  */
#define LL_RTC_ALARMOUT_DISABLE            0x00000000U             /*!< Output disabled */
#define LL_RTC_ALARMOUT_ALMA               RTC_CR_OSEL_0           /*!< Alarm A output enabled */
#define LL_RTC_ALARMOUT_ALMB               RTC_CR_OSEL_1           /*!< Alarm B output enabled */
#define LL_RTC_ALARMOUT_WAKEUP             RTC_CR_OSEL             /*!< Wakeup output enabled */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_ALARM_OUTPUTTYPE  ALARM OUTPUT TYPE
  * @{
  */
#define LL_RTC_ALARM_OUTPUTTYPE_OPENDRAIN  0x00000000U                          /*!< RTC_ALARM, when mapped on PC13, is open-drain output */
#define LL_RTC_ALARM_OUTPUTTYPE_PUSHPULL   RTC_TAFCR_ALARMOUTTYPE /*!< RTC_ALARM, when mapped on PC13, is push-pull output */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_PIN PIN
  * @{
  */
#define LL_RTC_PIN_PC13                    RTC_TAFCR_PC13MODE    /*!< PC13 is forced to push-pull output if all RTC alternate functions are disabled */
#define LL_RTC_PIN_PC14                    RTC_TAFCR_PC14MODE    /*!< PC14 is forced to push-pull output if LSE is disabled */
#define LL_RTC_PIN_PC15                    RTC_TAFCR_PC15MODE    /*!< PC15 is forced to push-pull output if LSE is disabled */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_OUTPUTPOLARITY_PIN  OUTPUT POLARITY PIN
  * @{
  */
#define LL_RTC_OUTPUTPOLARITY_PIN_HIGH     0x00000000U           /*!< Pin is high when ALRAF/ALRBF/WUTF is asserted (depending on OSEL)*/
#define LL_RTC_OUTPUTPOLARITY_PIN_LOW      RTC_CR_POL            /*!< Pin is low when ALRAF/ALRBF/WUTF is asserted (depending on OSEL) */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_TIME_FORMAT TIME FORMAT
  * @{
  */
#define LL_RTC_TIME_FORMAT_AM_OR_24        0x00000000U           /*!< AM or 24-hour format */
#define LL_RTC_TIME_FORMAT_PM              RTC_TR_PM             /*!< PM */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_SHIFT_SECOND  SHIFT SECOND
  * @{
  */
#define LL_RTC_SHIFT_SECOND_DELAY          0x00000000U           /* Delay (seconds) = SUBFS / (PREDIV_S + 1) */
#define LL_RTC_SHIFT_SECOND_ADVANCE        RTC_SHIFTR_ADD1S      /* Advance (seconds) = (1 - (SUBFS / (PREDIV_S + 1))) */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_ALMA_MASK  ALARMA MASK
  * @{
  */
#define LL_RTC_ALMA_MASK_NONE              0x00000000U             /*!< No masks applied on Alarm A*/
#define LL_RTC_ALMA_MASK_DATEWEEKDAY       RTC_ALRMAR_MSK4         /*!< Date/day do not care in Alarm A comparison */
#define LL_RTC_ALMA_MASK_HOURS             RTC_ALRMAR_MSK3         /*!< Hours do not care in Alarm A comparison */
#define LL_RTC_ALMA_MASK_MINUTES           RTC_ALRMAR_MSK2         /*!< Minutes do not care in Alarm A comparison */
#define LL_RTC_ALMA_MASK_SECONDS           RTC_ALRMAR_MSK1         /*!< Seconds do not care in Alarm A comparison */
#define LL_RTC_ALMA_MASK_ALL               (RTC_ALRMAR_MSK4 | RTC_ALRMAR_MSK3 | RTC_ALRMAR_MSK2 | RTC_ALRMAR_MSK1) /*!< Masks all */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_ALMA_TIME_FORMAT  ALARMA TIME FORMAT
  * @{
  */
#define LL_RTC_ALMA_TIME_FORMAT_AM         0x00000000U           /*!< AM or 24-hour format */
#define LL_RTC_ALMA_TIME_FORMAT_PM         RTC_ALRMAR_PM         /*!< PM */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_ALMB_MASK  ALARMB MASK
  * @{
  */
#define LL_RTC_ALMB_MASK_NONE              0x00000000U             /*!< No masks applied on Alarm B*/
#define LL_RTC_ALMB_MASK_DATEWEEKDAY       RTC_ALRMBR_MSK4         /*!< Date/day do not care in Alarm B comparison */
#define LL_RTC_ALMB_MASK_HOURS             RTC_ALRMBR_MSK3         /*!< Hours do not care in Alarm B comparison */
#define LL_RTC_ALMB_MASK_MINUTES           RTC_ALRMBR_MSK2         /*!< Minutes do not care in Alarm B comparison */
#define LL_RTC_ALMB_MASK_SECONDS           RTC_ALRMBR_MSK1         /*!< Seconds do not care in Alarm B comparison */
#define LL_RTC_ALMB_MASK_ALL               (RTC_ALRMBR_MSK4 | RTC_ALRMBR_MSK3 | RTC_ALRMBR_MSK2 | RTC_ALRMBR_MSK1) /*!< Masks all */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_ALMB_TIME_FORMAT  ALARMB TIME FORMAT
  * @{
  */
#define LL_RTC_ALMB_TIME_FORMAT_AM         0x00000000U           /*!< AM or 24-hour format */
#define LL_RTC_ALMB_TIME_FORMAT_PM         RTC_ALRMBR_PM         /*!< PM */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_TIMESTAMP_EDGE  TIMESTAMP EDGE
  * @{
  */
#define LL_RTC_TIMESTAMP_EDGE_RISING       0x00000000U           /*!< RTC_TS input rising edge generates a time-stamp event */
#define LL_RTC_TIMESTAMP_EDGE_FALLING      RTC_CR_TSEDGE         /*!< RTC_TS input falling edge generates a time-stamp even */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_TS_TIME_FORMAT  TIMESTAMP TIME FORMAT
  * @{
  */
#define LL_RTC_TS_TIME_FORMAT_AM           0x00000000U           /*!< AM or 24-hour format */
#define LL_RTC_TS_TIME_FORMAT_PM           RTC_TSTR_PM           /*!< PM */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_TAMPER  TAMPER
  * @{
  */
#define LL_RTC_TAMPER_1                    RTC_TAFCR_TAMP1E /*!< RTC_TAMP1 input detection */
#if defined(RTC_TAMPER2_SUPPORT)
#define LL_RTC_TAMPER_2                    RTC_TAFCR_TAMP2E /*!< RTC_TAMP2 input detection */
#endif /* RTC_TAMPER2_SUPPORT */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_TAMPER_MASK  TAMPER MASK
  * @{
  */
#define LL_RTC_TAMPER_MASK_TAMPER1         RTC_TAFCR_TAMP1MF /*!< Tamper 1 event generates a trigger event. TAMP1F is masked and internally cleared by hardware.The backup registers are not erased */
#if defined(RTC_TAMPER2_SUPPORT)
#define LL_RTC_TAMPER_MASK_TAMPER2         RTC_TAFCR_TAMP2MF /*!< Tamper 2 event generates a trigger event. TAMP2F is masked and internally cleared by hardware. The backup registers are not erased. */
#endif /* RTC_TAMPER2_SUPPORT */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_TAMPER_NOERASE  TAMPER NO ERASE
  * @{
  */
#define LL_RTC_TAMPER_NOERASE_TAMPER1      RTC_TAFCR_TAMP1NOERASE /*!< Tamper 1 event does not erase the backup registers. */
#if defined(RTC_TAMPER2_SUPPORT)
#define LL_RTC_TAMPER_NOERASE_TAMPER2      RTC_TAFCR_TAMP2NOERASE /*!< Tamper 2 event does not erase the backup registers. */
#endif /* RTC_TAMPER2_SUPPORT */
/**
  * @}
  */

#if defined(RTC_TAFCR_TAMPPRCH)
/** @defgroup RTC_LL_EC_TAMPER_DURATION  TAMPER DURATION
  * @{
  */
#define LL_RTC_TAMPER_DURATION_1RTCCLK     0x00000000U                             /*!< Tamper pins are pre-charged before sampling during 1 RTCCLK cycle  */
#define LL_RTC_TAMPER_DURATION_2RTCCLK     RTC_TAFCR_TAMPPRCH_0  /*!< Tamper pins are pre-charged before sampling during 2 RTCCLK cycles */
#define LL_RTC_TAMPER_DURATION_4RTCCLK     RTC_TAFCR_TAMPPRCH_1  /*!< Tamper pins are pre-charged before sampling during 4 RTCCLK cycles */
#define LL_RTC_TAMPER_DURATION_8RTCCLK     RTC_TAFCR_TAMPPRCH    /*!< Tamper pins are pre-charged before sampling during 8 RTCCLK cycles */
/**
  * @}
  */
#endif /* RTC_TAFCR_TAMPPRCH */

#if defined(RTC_TAFCR_TAMPFLT)
/** @defgroup RTC_LL_EC_TAMPER_FILTER  TAMPER FILTER
  * @{
  */
#define LL_RTC_TAMPER_FILTER_DISABLE       0x00000000U                              /*!< Tamper filter is disabled */
#define LL_RTC_TAMPER_FILTER_2SAMPLE       RTC_TAFCR_TAMPFLT_0    /*!< Tamper is activated after 2 consecutive samples at the active level */
#define LL_RTC_TAMPER_FILTER_4SAMPLE       RTC_TAFCR_TAMPFLT_1    /*!< Tamper is activated after 4 consecutive samples at the active level */
#define LL_RTC_TAMPER_FILTER_8SAMPLE       RTC_TAFCR_TAMPFLT      /*!< Tamper is activated after 8 consecutive samples at the active level. */
/**
  * @}
  */
#endif /* RTC_TAFCR_TAMPFLT */

#if defined(RTC_TAFCR_TAMPFREQ)
/** @defgroup RTC_LL_EC_TAMPER_SAMPLFREQDIV  TAMPER SAMPLING FREQUENCY DIVIDER
  * @{
  */
#define LL_RTC_TAMPER_SAMPLFREQDIV_32768   0x00000000U                                                      /*!< Each of the tamper inputs are sampled with a frequency =  RTCCLK / 32768 */
#define LL_RTC_TAMPER_SAMPLFREQDIV_16384   RTC_TAFCR_TAMPFREQ_0                           /*!< Each of the tamper inputs are sampled with a frequency =  RTCCLK / 16384 */
#define LL_RTC_TAMPER_SAMPLFREQDIV_8192    RTC_TAFCR_TAMPFREQ_1                           /*!< Each of the tamper inputs are sampled with a frequency =  RTCCLK / 8192 */
#define LL_RTC_TAMPER_SAMPLFREQDIV_4096    (RTC_TAFCR_TAMPFREQ_1 | RTC_TAFCR_TAMPFREQ_0) /*!< Each of the tamper inputs are sampled with a frequency =  RTCCLK / 4096 */
#define LL_RTC_TAMPER_SAMPLFREQDIV_2048    RTC_TAFCR_TAMPFREQ_2                           /*!< Each of the tamper inputs are sampled with a frequency =  RTCCLK / 2048 */
#define LL_RTC_TAMPER_SAMPLFREQDIV_1024    (RTC_TAFCR_TAMPFREQ_2 | RTC_TAFCR_TAMPFREQ_0) /*!< Each of the tamper inputs are sampled with a frequency =  RTCCLK / 1024 */
#define LL_RTC_TAMPER_SAMPLFREQDIV_512     (RTC_TAFCR_TAMPFREQ_2 | RTC_TAFCR_TAMPFREQ_1) /*!< Each of the tamper inputs are sampled with a frequency =  RTCCLK / 512 */
#define LL_RTC_TAMPER_SAMPLFREQDIV_256     RTC_TAFCR_TAMPFREQ                             /*!< Each of the tamper inputs are sampled with a frequency =  RTCCLK / 256 */
/**
  * @}
  */
#endif /* RTC_TAFCR_TAMPFREQ */

/** @defgroup RTC_LL_EC_TAMPER_ACTIVELEVEL  TAMPER ACTIVE LEVEL
  * @{
  */
#define LL_RTC_TAMPER_ACTIVELEVEL_TAMP1    RTC_TAFCR_TAMP1TRG /*!< RTC_TAMP1 input falling edge (if TAMPFLT = 00) or staying high (if TAMPFLT != 00) triggers a tamper detection event*/
#if defined(RTC_TAMPER2_SUPPORT)
#define LL_RTC_TAMPER_ACTIVELEVEL_TAMP2    RTC_TAFCR_TAMP2TRG /*!< RTC_TAMP2 input falling edge (if TAMPFLT = 00) or staying high (if TAMPFLT != 00) triggers a tamper detection event*/
#endif /* RTC_TAMPER2_SUPPORT */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_WAKEUPCLOCK_DIV  WAKEUP CLOCK DIV
  * @{
  */
#define LL_RTC_WAKEUPCLOCK_DIV_16          0x00000000U                           /*!< RTC/16 clock is selected */
#define LL_RTC_WAKEUPCLOCK_DIV_8           (RTC_CR_WUCKSEL_0)                    /*!< RTC/8 clock is selected */
#define LL_RTC_WAKEUPCLOCK_DIV_4           (RTC_CR_WUCKSEL_1)                    /*!< RTC/4 clock is selected */
#define LL_RTC_WAKEUPCLOCK_DIV_2           (RTC_CR_WUCKSEL_1 | RTC_CR_WUCKSEL_0) /*!< RTC/2 clock is selected */
#define LL_RTC_WAKEUPCLOCK_CKSPRE          (RTC_CR_WUCKSEL_2)                    /*!< ck_spre (usually 1 Hz) clock is selected */
#define LL_RTC_WAKEUPCLOCK_CKSPRE_WUT      (RTC_CR_WUCKSEL_2 | RTC_CR_WUCKSEL_1) /*!< ck_spre (usually 1 Hz) clock is selected and 2exp16 is added to the WUT counter value*/
/**
  * @}
  */

/** @defgroup RTC_LL_EC_BKP  BACKUP
  * @{
  */
#define LL_RTC_BKP_DR0                     0x00000000U
#define LL_RTC_BKP_DR1                     0x00000001U
#define LL_RTC_BKP_DR2                     0x00000002U
#define LL_RTC_BKP_DR3                     0x00000003U
#define LL_RTC_BKP_DR4                     0x00000004U
#if RTC_BKP_NUMBER > 5
#define LL_RTC_BKP_DR5                     0x00000005U
#define LL_RTC_BKP_DR6                     0x00000006U
#define LL_RTC_BKP_DR7                     0x00000007U
#define LL_RTC_BKP_DR8                     0x00000008U
#define LL_RTC_BKP_DR9                     0x00000009U
#define LL_RTC_BKP_DR10                    0x0000000AU
#define LL_RTC_BKP_DR11                    0x0000000BU
#define LL_RTC_BKP_DR12                    0x0000000CU
#define LL_RTC_BKP_DR13                    0x0000000DU
#define LL_RTC_BKP_DR14                    0x0000000EU
#define LL_RTC_BKP_DR15                    0x0000000FU
#endif /* RTC_BKP_NUMBER > 5 */

#if RTC_BKP_NUMBER > 16
#define LL_RTC_BKP_DR16                    0x00000010U
#define LL_RTC_BKP_DR17                    0x00000011U
#define LL_RTC_BKP_DR18                    0x00000012U
#define LL_RTC_BKP_DR19                    0x00000013U
#endif /* RTC_BKP_NUMBER > 16 */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_CALIB_OUTPUT  Calibration output
  * @{
  */
#define LL_RTC_CALIB_OUTPUT_NONE           0x00000000U                 /*!< Calibration output disabled */
#define LL_RTC_CALIB_OUTPUT_1HZ            (RTC_CR_COE | RTC_CR_COSEL) /*!< Calibration output is 1 Hz */
#define LL_RTC_CALIB_OUTPUT_512HZ          (RTC_CR_COE)                /*!< Calibration output is 512 Hz */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_CALIB_SIGN Coarse digital calibration sign
  * @{
  */
#define LL_RTC_CALIB_SIGN_POSITIVE         0x00000000U           /*!< Positive calibration: calendar update frequency is increased */
#define LL_RTC_CALIB_SIGN_NEGATIVE         RTC_CALIBR_DCS        /*!< Negative calibration: calendar update frequency is decreased */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_CALIB_INSERTPULSE  Calibration pulse insertion 
  * @{
  */
#define LL_RTC_CALIB_INSERTPULSE_NONE      0x00000000U           /*!< No RTCCLK pulses are added */
#define LL_RTC_CALIB_INSERTPULSE_SET       RTC_CALR_CALP         /*!< One RTCCLK pulse is effectively inserted every 2exp11 pulses (frequency increased by 488.5 ppm) */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_CALIB_PERIOD  Calibration period
  * @{
  */
#define LL_RTC_CALIB_PERIOD_32SEC          0x00000000U           /*!< Use a 32-second calibration cycle period */
#define LL_RTC_CALIB_PERIOD_16SEC          RTC_CALR_CALW16       /*!< Use a 16-second calibration cycle period */
#define LL_RTC_CALIB_PERIOD_8SEC           RTC_CALR_CALW8        /*!< Use a 8-second calibration cycle period */
/**
  * @}
  */

/** @defgroup RTC_LL_EC_TSINSEL  TIMESTAMP mapping
  * @{
  */
#define LL_RTC_TimeStampPin_Default        0x00000000U           /*!< Use RTC_AF1 as TIMESTAMP */
#if defined(RTC_AF2_SUPPORT)
#define LL_RTC_TimeStampPin_Pos1           RTC_TAFCR_TSINSEL     /*!< Use RTC_AF2 as TIMESTAMP */
#endif
/**
  * @}
  */

/** @defgroup RTC_LL_EC_TAMP1INSEL  TAMPER1 mapping
  * @{
  */
#define LL_RTC_TamperPin_Default           0x00000000U           /*!< Use RTC_AF1 as TAMPER1 */
#if defined(RTC_AF2_SUPPORT)
#define LL_RTC_TamperPin_Pos1              RTC_TAFCR_TAMP1INSEL  /*!< Use RTC_AF2 as TAMPER1 */
#endif
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup RTC_LL_Exported_Macros RTC Exported Macros
  * @{
  */

/** @defgroup RTC_LL_EM_WRITE_READ Common Write and read registers Macros
  * @{
  */

/**
  * @brief  Write a value in RTC register
  * @param  __INSTANCE__ RTC Instance
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_RTC_WriteReg(__INSTANCE__, __REG__, __VALUE__) WRITE_REG(__INSTANCE__->__REG__, (__VALUE__))

/**
  * @brief  Read a value in RTC register
  * @param  __INSTANCE__ RTC Instance
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_RTC_ReadReg(__INSTANCE__, __REG__) READ_REG(__INSTANCE__->__REG__)
/**
  * @}
  */

/** @defgroup RTC_LL_EM_Convert Convert helper Macros
  * @{
  */

/**
  * @brief  Helper macro to convert a value from 2 digit decimal format to BCD format
  * @param  __VALUE__ Byte to be converted
  * @retval Converted byte
  */
#define __LL_RTC_CONVERT_BIN2BCD(__VALUE__) (uint8_t)((((__VALUE__) / 10U) << 4U) | ((__VALUE__) % 10U))

/**
  * @brief  Helper macro to convert a value from BCD format to 2 digit decimal format
  * @param  __VALUE__ BCD value to be converted
  * @retval Converted byte
  */
#define __LL_RTC_CONVERT_BCD2BIN(__VALUE__) (uint8_t)(((uint8_t)((__VALUE__) & (uint8_t)0xF0U) >> (uint8_t)0x4U) * 10U + ((__VALUE__) & (uint8_t)0x0FU))

/**
  * @}
  */

/** @defgroup RTC_LL_EM_Date Date helper Macros
  * @{
  */

/**
  * @brief  Helper macro to retrieve weekday.
  * @param  __RTC_DATE__ Date returned by @ref  LL_RTC_DATE_Get function.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_WEEKDAY_MONDAY
  *         @arg @ref LL_RTC_WEEKDAY_TUESDAY
  *         @arg @ref LL_RTC_WEEKDAY_WEDNESDAY
  *         @arg @ref LL_RTC_WEEKDAY_THURSDAY
  *         @arg @ref LL_RTC_WEEKDAY_FRIDAY
  *         @arg @ref LL_RTC_WEEKDAY_SATURDAY
  *         @arg @ref LL_RTC_WEEKDAY_SUNDAY
  */
#define __LL_RTC_GET_WEEKDAY(__RTC_DATE__) (((__RTC_DATE__) >> RTC_OFFSET_WEEKDAY) & 0x000000FFU)

/**
  * @brief  Helper macro to retrieve Year in BCD format
  * @param  __RTC_DATE__ Value returned by @ref  LL_RTC_DATE_Get
  * @retval Year in BCD format (0x00 . . . 0x99)
  */
#define __LL_RTC_GET_YEAR(__RTC_DATE__) ((__RTC_DATE__) & 0x000000FFU)

/**
  * @brief  Helper macro to retrieve Month in BCD format
  * @param  __RTC_DATE__ Value returned by @ref  LL_RTC_DATE_Get
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_MONTH_JANUARY
  *         @arg @ref LL_RTC_MONTH_FEBRUARY
  *         @arg @ref LL_RTC_MONTH_MARCH
  *         @arg @ref LL_RTC_MONTH_APRIL
  *         @arg @ref LL_RTC_MONTH_MAY
  *         @arg @ref LL_RTC_MONTH_JUNE
  *         @arg @ref LL_RTC_MONTH_JULY
  *         @arg @ref LL_RTC_MONTH_AUGUST
  *         @arg @ref LL_RTC_MONTH_SEPTEMBER
  *         @arg @ref LL_RTC_MONTH_OCTOBER
  *         @arg @ref LL_RTC_MONTH_NOVEMBER
  *         @arg @ref LL_RTC_MONTH_DECEMBER
  */
#define __LL_RTC_GET_MONTH(__RTC_DATE__) (((__RTC_DATE__) >>RTC_OFFSET_MONTH) & 0x000000FFU)

/**
  * @brief  Helper macro to retrieve Day in BCD format
  * @param  __RTC_DATE__ Value returned by @ref  LL_RTC_DATE_Get
  * @retval Day in BCD format (0x01 . . . 0x31)
  */
#define __LL_RTC_GET_DAY(__RTC_DATE__) (((__RTC_DATE__) >>RTC_OFFSET_DAY) & 0x000000FFU)

/**
  * @}
  */

/** @defgroup RTC_LL_EM_Time Time helper Macros
  * @{
  */

/**
  * @brief  Helper macro to retrieve hour in BCD format
  * @param  __RTC_TIME__ RTC time returned by @ref LL_RTC_TIME_Get function
  * @retval Hours in BCD format (0x01. . .0x12 or between Min_Data=0x00 and Max_Data=0x23)
  */
#define __LL_RTC_GET_HOUR(__RTC_TIME__) (((__RTC_TIME__) >> RTC_OFFSET_HOUR) & 0x000000FFU)

/**
  * @brief  Helper macro to retrieve minute in BCD format
  * @param  __RTC_TIME__ RTC time returned by @ref LL_RTC_TIME_Get function
  * @retval Minutes in BCD format (0x00. . .0x59)
  */
#define __LL_RTC_GET_MINUTE(__RTC_TIME__) (((__RTC_TIME__) >> RTC_OFFSET_MINUTE) & 0x000000FFU)

/**
  * @brief  Helper macro to retrieve second in BCD format
  * @param  __RTC_TIME__ RTC time returned by @ref LL_RTC_TIME_Get function
  * @retval Seconds in  format (0x00. . .0x59)
  */
#define __LL_RTC_GET_SECOND(__RTC_TIME__) ((__RTC_TIME__) & 0x000000FFU)

/**
  * @}
  */

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup RTC_LL_Exported_Functions RTC Exported Functions
  * @{
  */

/** @defgroup RTC_LL_EF_Configuration Configuration
  * @{
  */

/**
  * @brief  Set Hours format (24 hour/day or AM/PM hour format)
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @rmtoll CR           FMT           LL_RTC_SetHourFormat
  * @param  RTCx RTC Instance
  * @param  HourFormat This parameter can be one of the following values:
  *         @arg @ref LL_RTC_HOURFORMAT_24HOUR
  *         @arg @ref LL_RTC_HOURFORMAT_AMPM
  * @retval None
  */
__STATIC_INLINE void LL_RTC_SetHourFormat(RTC_TypeDef *RTCx, uint32_t HourFormat)
{
  MODIFY_REG(RTCx->CR, RTC_CR_FMT, HourFormat);
}

/**
  * @brief  Get Hours format (24 hour/day or AM/PM hour format)
  * @rmtoll CR           FMT           LL_RTC_GetHourFormat
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_HOURFORMAT_24HOUR
  *         @arg @ref LL_RTC_HOURFORMAT_AMPM
  */
__STATIC_INLINE uint32_t LL_RTC_GetHourFormat(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CR, RTC_CR_FMT));
}

/**
  * @brief  Select the flag to be routed to RTC_ALARM output
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           OSEL          LL_RTC_SetAlarmOutEvent
  * @param  RTCx RTC Instance
  * @param  AlarmOutput This parameter can be one of the following values:
  *         @arg @ref LL_RTC_ALARMOUT_DISABLE
  *         @arg @ref LL_RTC_ALARMOUT_ALMA
  *         @arg @ref LL_RTC_ALARMOUT_ALMB
  *         @arg @ref LL_RTC_ALARMOUT_WAKEUP
  * @retval None
  */
__STATIC_INLINE void LL_RTC_SetAlarmOutEvent(RTC_TypeDef *RTCx, uint32_t AlarmOutput)
{
  MODIFY_REG(RTCx->CR, RTC_CR_OSEL, AlarmOutput);
}

/**
  * @brief  Get the flag to be routed to RTC_ALARM output
  * @rmtoll CR           OSEL          LL_RTC_GetAlarmOutEvent
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_ALARMOUT_DISABLE
  *         @arg @ref LL_RTC_ALARMOUT_ALMA
  *         @arg @ref LL_RTC_ALARMOUT_ALMB
  *         @arg @ref LL_RTC_ALARMOUT_WAKEUP
  */
__STATIC_INLINE uint32_t LL_RTC_GetAlarmOutEvent(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CR, RTC_CR_OSEL));
}

/**
  * @brief  Set RTC_ALARM output type (ALARM in push-pull or open-drain output)
  * @note   Used only when RTC_ALARM is mapped on PC13
  * @note   If all RTC alternate functions are disabled and PC13MODE = 1, PC13VALUE configures the
  *         PC13 output data
  * @rmtoll TAFCR        ALARMOUTTYPE  LL_RTC_SetAlarmOutputType
  * @param  RTCx RTC Instance
  * @param  Output This parameter can be one of the following values:
  *         @arg @ref LL_RTC_ALARM_OUTPUTTYPE_OPENDRAIN
  *         @arg @ref LL_RTC_ALARM_OUTPUTTYPE_PUSHPULL
  * @retval None
  */
__STATIC_INLINE void LL_RTC_SetAlarmOutputType(RTC_TypeDef *RTCx, uint32_t Output)
{
  MODIFY_REG(RTCx->TAFCR, RTC_TAFCR_ALARMOUTTYPE, Output);
}

/**
  * @brief  Get RTC_ALARM output type (ALARM in push-pull or open-drain output)
  * @note   used only when RTC_ALARM is mapped on PC13
  * @note   If all RTC alternate functions are disabled and PC13MODE = 1, PC13VALUE configures the
  *         PC13 output data
  * @rmtoll TAFCR        ALARMOUTTYPE  LL_RTC_GetAlarmOutputType
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_ALARM_OUTPUTTYPE_OPENDRAIN
  *         @arg @ref LL_RTC_ALARM_OUTPUTTYPE_PUSHPULL
  */
__STATIC_INLINE uint32_t LL_RTC_GetAlarmOutputType(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TAFCR, RTC_TAFCR_ALARMOUTTYPE));
}

/**
  * @brief  Enable push-pull output on PC13, PC14 and/or PC15
  * @note   PC13 forced to push-pull output if all RTC alternate functions are disabled
  * @note   PC14 and PC15 forced to push-pull output if LSE is disabled
  * @rmtoll TAFCR        PC13MODE  LL_RTC_EnablePushPullMode\n
  * @rmtoll TAFCR        PC14MODE  LL_RTC_EnablePushPullMode\n
  * @rmtoll TAFCR        PC15MODE  LL_RTC_EnablePushPullMode
  * @param  RTCx RTC Instance
  * @param  PinMask This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_PIN_PC13
  *         @arg @ref LL_RTC_PIN_PC14
  *         @arg @ref LL_RTC_PIN_PC15
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnablePushPullMode(RTC_TypeDef *RTCx, uint32_t PinMask)
{
  SET_BIT(RTCx->TAFCR, PinMask);
}

/**
  * @brief  Disable push-pull output on PC13, PC14 and/or PC15
  * @note   PC13, PC14 and/or PC15 are controlled by the GPIO configuration registers.
  *         Consequently PC13, PC14 and/or PC15 are floating in Standby mode.
  * @rmtoll TAFCR        PC13MODE      LL_RTC_DisablePushPullMode\n
  *         TAFCR        PC14MODE      LL_RTC_DisablePushPullMode\n
  *         TAFCR        PC15MODE      LL_RTC_DisablePushPullMode
  * @param  RTCx RTC Instance
  * @param  PinMask This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_PIN_PC13
  *         @arg @ref LL_RTC_PIN_PC14
  *         @arg @ref LL_RTC_PIN_PC15
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisablePushPullMode(RTC_TypeDef* RTCx, uint32_t PinMask)
{
  CLEAR_BIT(RTCx->TAFCR, PinMask);
}

/**
  * @brief  Set PC14 and/or PC15 to high level.
  * @note   Output data configuration is possible if the LSE is disabled and PushPull output is enabled (through @ref LL_RTC_EnablePushPullMode) 
  * @rmtoll TAFCR        PC14VALUE     LL_RTC_SetOutputPin\n
  *         TAFCR        PC15VALUE     LL_RTC_SetOutputPin
  * @param  RTCx RTC Instance
  * @param  PinMask This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_PIN_PC14
  *         @arg @ref LL_RTC_PIN_PC15
  * @retval None
  */
__STATIC_INLINE void LL_RTC_SetOutputPin(RTC_TypeDef* RTCx, uint32_t PinMask)
{
  SET_BIT(RTCx->TAFCR, (PinMask >> 1));
}

/**
  * @brief  Set PC14 and/or PC15 to low level.
  * @note   Output data configuration is possible if the LSE is disabled and PushPull output is enabled (through @ref LL_RTC_EnablePushPullMode) 
  * @rmtoll TAFCR        PC14VALUE     LL_RTC_ResetOutputPin\n
  *         TAFCR        PC15VALUE     LL_RTC_ResetOutputPin
  * @param  RTCx RTC Instance
  * @param  PinMask This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_PIN_PC14
  *         @arg @ref LL_RTC_PIN_PC15
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ResetOutputPin(RTC_TypeDef* RTCx, uint32_t PinMask)
{
  CLEAR_BIT(RTCx->TAFCR, (PinMask >> 1));
}

/**
  * @brief  Enable initialization mode
  * @note   Initialization mode is used to program time and date register (RTC_TR and RTC_DR)
  *         and prescaler register (RTC_PRER).
  *         Counters are stopped and start counting from the new value when INIT is reset.
  * @rmtoll ISR          INIT          LL_RTC_EnableInitMode
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnableInitMode(RTC_TypeDef *RTCx)
{
  /* Set the Initialization mode */
  WRITE_REG(RTCx->ISR, RTC_INIT_MASK);
}

/**
  * @brief  Disable initialization mode (Free running mode)
  * @rmtoll ISR          INIT          LL_RTC_DisableInitMode
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisableInitMode(RTC_TypeDef *RTCx)
{
  /* Exit Initialization mode */
  WRITE_REG(RTCx->ISR, (uint32_t)~RTC_ISR_INIT);
}

/**
  * @brief  Set Output polarity (pin is low when ALRAF/ALRBF/WUTF is asserted)
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           POL           LL_RTC_SetOutputPolarity
  * @param  RTCx RTC Instance
  * @param  Polarity This parameter can be one of the following values:
  *         @arg @ref LL_RTC_OUTPUTPOLARITY_PIN_HIGH
  *         @arg @ref LL_RTC_OUTPUTPOLARITY_PIN_LOW
  * @retval None
  */
__STATIC_INLINE void LL_RTC_SetOutputPolarity(RTC_TypeDef *RTCx, uint32_t Polarity)
{
  MODIFY_REG(RTCx->CR, RTC_CR_POL, Polarity);
}

/**
  * @brief  Get Output polarity
  * @rmtoll CR           POL           LL_RTC_GetOutputPolarity
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_OUTPUTPOLARITY_PIN_HIGH
  *         @arg @ref LL_RTC_OUTPUTPOLARITY_PIN_LOW
  */
__STATIC_INLINE uint32_t LL_RTC_GetOutputPolarity(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CR, RTC_CR_POL));
}

/**
  * @brief  Enable Bypass the shadow registers
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           BYPSHAD       LL_RTC_EnableShadowRegBypass
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnableShadowRegBypass(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_BYPSHAD);
}

/**
  * @brief  Disable Bypass the shadow registers
  * @rmtoll CR           BYPSHAD       LL_RTC_DisableShadowRegBypass
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisableShadowRegBypass(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_BYPSHAD);
}

/**
  * @brief  Check if Shadow registers bypass is enabled or not.
  * @rmtoll CR           BYPSHAD       LL_RTC_IsShadowRegBypassEnabled
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsShadowRegBypassEnabled(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->CR, RTC_CR_BYPSHAD) == (RTC_CR_BYPSHAD));
}

/**
  * @brief  Enable RTC_REFIN reference clock detection (50 or 60 Hz)
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @rmtoll CR           REFCKON       LL_RTC_EnableRefClock
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnableRefClock(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_REFCKON);
}

/**
  * @brief  Disable RTC_REFIN reference clock detection (50 or 60 Hz)
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @rmtoll CR           REFCKON       LL_RTC_DisableRefClock
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisableRefClock(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_REFCKON);
}

/**
  * @brief  Set Asynchronous prescaler factor
  * @rmtoll PRER         PREDIV_A      LL_RTC_SetAsynchPrescaler
  * @param  RTCx RTC Instance
  * @param  AsynchPrescaler Value between Min_Data = 0 and Max_Data = 0x7F
  * @retval None
  */
__STATIC_INLINE void LL_RTC_SetAsynchPrescaler(RTC_TypeDef *RTCx, uint32_t AsynchPrescaler)
{
  MODIFY_REG(RTCx->PRER, RTC_PRER_PREDIV_A, AsynchPrescaler << RTC_PRER_PREDIV_A_Pos);
}

/**
  * @brief  Set Synchronous prescaler factor
  * @rmtoll PRER         PREDIV_S      LL_RTC_SetSynchPrescaler
  * @param  RTCx RTC Instance
  * @param  SynchPrescaler Value between Min_Data = 0 and Max_Data = 0x7FFF
  * @retval None
  */
__STATIC_INLINE void LL_RTC_SetSynchPrescaler(RTC_TypeDef *RTCx, uint32_t SynchPrescaler)
{
  MODIFY_REG(RTCx->PRER, RTC_PRER_PREDIV_S, SynchPrescaler);
}

/**
  * @brief  Get Asynchronous prescaler factor
  * @rmtoll PRER         PREDIV_A      LL_RTC_GetAsynchPrescaler
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data = 0 and Max_Data = 0x7F
  */
__STATIC_INLINE uint32_t LL_RTC_GetAsynchPrescaler(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->PRER, RTC_PRER_PREDIV_A) >> RTC_PRER_PREDIV_A_Pos);
}

/**
  * @brief  Get Synchronous prescaler factor
  * @rmtoll PRER         PREDIV_S      LL_RTC_GetSynchPrescaler
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data = 0 and Max_Data = 0x7FFF
  */
__STATIC_INLINE uint32_t LL_RTC_GetSynchPrescaler(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->PRER, RTC_PRER_PREDIV_S));
}

/**
  * @brief  Enable the write protection for RTC registers.
  * @rmtoll WPR          KEY           LL_RTC_EnableWriteProtection
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnableWriteProtection(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->WPR, RTC_WRITE_PROTECTION_DISABLE);
}

/**
  * @brief  Disable the write protection for RTC registers.
  * @rmtoll WPR          KEY           LL_RTC_DisableWriteProtection
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisableWriteProtection(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->WPR, RTC_WRITE_PROTECTION_ENABLE_1);
  WRITE_REG(RTCx->WPR, RTC_WRITE_PROTECTION_ENABLE_2);
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_Time Time
  * @{
  */

/**
  * @brief  Set time format (AM/24-hour or PM notation)
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @rmtoll TR           PM            LL_RTC_TIME_SetFormat
  * @param  RTCx RTC Instance
  * @param  TimeFormat This parameter can be one of the following values:
  *         @arg @ref LL_RTC_TIME_FORMAT_AM_OR_24
  *         @arg @ref LL_RTC_TIME_FORMAT_PM
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_SetFormat(RTC_TypeDef *RTCx, uint32_t TimeFormat)
{
  MODIFY_REG(RTCx->TR, RTC_TR_PM, TimeFormat);
}

/**
  * @brief  Get time format (AM or PM notation)
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @note Read either RTC_SSR or RTC_TR locks the values in the higher-order calendar
  *       shadow registers until RTC_DR is read (LL_RTC_ReadReg(RTC, DR)).
  * @rmtoll TR           PM            LL_RTC_TIME_GetFormat
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_TIME_FORMAT_AM_OR_24
  *         @arg @ref LL_RTC_TIME_FORMAT_PM
  */
__STATIC_INLINE uint32_t LL_RTC_TIME_GetFormat(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TR, RTC_TR_PM));
}

/**
  * @brief  Set Hours in BCD format
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert hour from binary to BCD format
  * @rmtoll TR           HT            LL_RTC_TIME_SetHour\n
  *         TR           HU            LL_RTC_TIME_SetHour
  * @param  RTCx RTC Instance
  * @param  Hours Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_SetHour(RTC_TypeDef *RTCx, uint32_t Hours)
{
  MODIFY_REG(RTCx->TR, (RTC_TR_HT | RTC_TR_HU),
             (((Hours & 0xF0U) << (RTC_TR_HT_Pos - 4U)) | ((Hours & 0x0FU) << RTC_TR_HU_Pos)));
}

/**
  * @brief  Get Hours in BCD format
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @note Read either RTC_SSR or RTC_TR locks the values in the higher-order calendar
  *       shadow registers until RTC_DR is read (LL_RTC_ReadReg(RTC, DR)).
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert hour from BCD to
  *       Binary format
  * @rmtoll TR           HT            LL_RTC_TIME_GetHour\n
  *         TR           HU            LL_RTC_TIME_GetHour
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  */
__STATIC_INLINE uint32_t LL_RTC_TIME_GetHour(RTC_TypeDef *RTCx)
{
  return (uint32_t)((READ_BIT(RTCx->TR, (RTC_TR_HT | RTC_TR_HU))) >> RTC_TR_HU_Pos);
}

/**
  * @brief  Set Minutes in BCD format
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Minutes from binary to BCD format
  * @rmtoll TR           MNT           LL_RTC_TIME_SetMinute\n
  *         TR           MNU           LL_RTC_TIME_SetMinute
  * @param  RTCx RTC Instance
  * @param  Minutes Value between Min_Data=0x00 and Max_Data=0x59
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_SetMinute(RTC_TypeDef *RTCx, uint32_t Minutes)
{
  MODIFY_REG(RTCx->TR, (RTC_TR_MNT | RTC_TR_MNU),
             (((Minutes & 0xF0U) << (RTC_TR_MNT_Pos - 4U)) | ((Minutes & 0x0FU) << RTC_TR_MNU_Pos)));
}

/**
  * @brief  Get Minutes in BCD format
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @note Read either RTC_SSR or RTC_TR locks the values in the higher-order calendar
  *       shadow registers until RTC_DR is read (LL_RTC_ReadReg(RTC, DR)).
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert minute from BCD
  *       to Binary format
  * @rmtoll TR           MNT           LL_RTC_TIME_GetMinute\n
  *         TR           MNU           LL_RTC_TIME_GetMinute
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x59
  */
__STATIC_INLINE uint32_t LL_RTC_TIME_GetMinute(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TR, (RTC_TR_MNT | RTC_TR_MNU))>> RTC_TR_MNU_Pos);
}

/**
  * @brief  Set Seconds in BCD format
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Seconds from binary to BCD format
  * @rmtoll TR           ST            LL_RTC_TIME_SetSecond\n
  *         TR           SU            LL_RTC_TIME_SetSecond
  * @param  RTCx RTC Instance
  * @param  Seconds Value between Min_Data=0x00 and Max_Data=0x59
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_SetSecond(RTC_TypeDef *RTCx, uint32_t Seconds)
{
  MODIFY_REG(RTCx->TR, (RTC_TR_ST | RTC_TR_SU),
             (((Seconds & 0xF0U) << (RTC_TR_ST_Pos - 4U)) | ((Seconds & 0x0FU) << RTC_TR_SU_Pos)));
}

/**
  * @brief  Get Seconds in BCD format
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @note Read either RTC_SSR or RTC_TR locks the values in the higher-order calendar
  *       shadow registers until RTC_DR is read (LL_RTC_ReadReg(RTC, DR)).
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Seconds from BCD
  *       to Binary format
  * @rmtoll TR           ST            LL_RTC_TIME_GetSecond\n
  *         TR           SU            LL_RTC_TIME_GetSecond
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x59
  */
__STATIC_INLINE uint32_t LL_RTC_TIME_GetSecond(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TR, (RTC_TR_ST | RTC_TR_SU)) >> RTC_TR_SU_Pos);
}

/**
  * @brief  Set time (hour, minute and second) in BCD format
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @note TimeFormat and Hours should follow the same format
  * @rmtoll TR           PM            LL_RTC_TIME_Config\n
  *         TR           HT            LL_RTC_TIME_Config\n
  *         TR           HU            LL_RTC_TIME_Config\n
  *         TR           MNT           LL_RTC_TIME_Config\n
  *         TR           MNU           LL_RTC_TIME_Config\n
  *         TR           ST            LL_RTC_TIME_Config\n
  *         TR           SU            LL_RTC_TIME_Config
  * @param  RTCx RTC Instance
  * @param  Format12_24 This parameter can be one of the following values:
  *         @arg @ref LL_RTC_TIME_FORMAT_AM_OR_24
  *         @arg @ref LL_RTC_TIME_FORMAT_PM
  * @param  Hours Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  * @param  Minutes Value between Min_Data=0x00 and Max_Data=0x59
  * @param  Seconds Value between Min_Data=0x00 and Max_Data=0x59
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_Config(RTC_TypeDef *RTCx, uint32_t Format12_24, uint32_t Hours, uint32_t Minutes, uint32_t Seconds)
{
  register uint32_t temp = 0U;

  temp = Format12_24                                                                                    | \
         (((Hours & 0xF0U) << (RTC_TR_HT_Pos - 4U)) | ((Hours & 0x0FU) << RTC_TR_HU_Pos))     | \
         (((Minutes & 0xF0U) << (RTC_TR_MNT_Pos - 4U)) | ((Minutes & 0x0FU) << RTC_TR_MNU_Pos)) | \
         (((Seconds & 0xF0U) << (RTC_TR_ST_Pos - 4U)) | ((Seconds & 0x0FU) << RTC_TR_SU_Pos));
  MODIFY_REG(RTCx->TR, (RTC_TR_PM | RTC_TR_HT | RTC_TR_HU | RTC_TR_MNT | RTC_TR_MNU | RTC_TR_ST | RTC_TR_SU), temp);
}

/**
  * @brief  Get time (hour, minute and second) in BCD format
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @note Read either RTC_SSR or RTC_TR locks the values in the higher-order calendar
  *       shadow registers until RTC_DR is read (LL_RTC_ReadReg(RTC, DR)).
  * @note helper macros __LL_RTC_GET_HOUR, __LL_RTC_GET_MINUTE and __LL_RTC_GET_SECOND
  *       are available to get independently each parameter.
  * @rmtoll TR           HT            LL_RTC_TIME_Get\n
  *         TR           HU            LL_RTC_TIME_Get\n
  *         TR           MNT           LL_RTC_TIME_Get\n
  *         TR           MNU           LL_RTC_TIME_Get\n
  *         TR           ST            LL_RTC_TIME_Get\n
  *         TR           SU            LL_RTC_TIME_Get
  * @param  RTCx RTC Instance
  * @retval Combination of hours, minutes and seconds (Format: 0x00HHMMSS).
  */
__STATIC_INLINE uint32_t LL_RTC_TIME_Get(RTC_TypeDef *RTCx)
{
  register uint32_t temp = 0U;
  
  temp = READ_BIT(RTCx->TR, (RTC_TR_HT | RTC_TR_HU | RTC_TR_MNT | RTC_TR_MNU | RTC_TR_ST | RTC_TR_SU));
  return (uint32_t)((((((temp & RTC_TR_HT) >> RTC_TR_HT_Pos) << 4U) | ((temp & RTC_TR_HU) >> RTC_TR_HU_Pos)) << RTC_OFFSET_HOUR) |  \
                    (((((temp & RTC_TR_MNT) >> RTC_TR_MNT_Pos) << 4U) | ((temp & RTC_TR_MNU) >> RTC_TR_MNU_Pos)) << RTC_OFFSET_MINUTE) | \
                    ((((temp & RTC_TR_ST) >> RTC_TR_ST_Pos) << 4U) | ((temp & RTC_TR_SU) >> RTC_TR_SU_Pos)));
}

/**
  * @brief  Memorize whether the daylight saving time change has been performed
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           BKP           LL_RTC_TIME_EnableDayLightStore
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_EnableDayLightStore(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_BKP);
}

/**
  * @brief  Disable memorization whether the daylight saving time change has been performed.
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           BKP           LL_RTC_TIME_DisableDayLightStore
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_DisableDayLightStore(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_BKP);
}

/**
  * @brief  Check if RTC Day Light Saving stored operation has been enabled or not
  * @rmtoll CR           BKP           LL_RTC_TIME_IsDayLightStoreEnabled
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_TIME_IsDayLightStoreEnabled(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->CR, RTC_CR_BKP) == (RTC_CR_BKP));
}

/**
  * @brief  Subtract 1 hour (winter time change)
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           SUB1H         LL_RTC_TIME_DecHour
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_DecHour(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_SUB1H);
}

/**
  * @brief  Add 1 hour (summer time change)
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           ADD1H         LL_RTC_TIME_IncHour
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_IncHour(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_ADD1H);
}

/**
  * @brief  Get Sub second value in the synchronous prescaler counter.
  * @note  You can use both SubSeconds value and SecondFraction (PREDIV_S through
  *        LL_RTC_GetSynchPrescaler function) terms returned to convert Calendar
  *        SubSeconds value in second fraction ratio with time unit following
  *        generic formula:
  *          ==> Seconds fraction ratio * time_unit= [(SecondFraction-SubSeconds)/(SecondFraction+1)] * time_unit
  *        This conversion can be performed only if no shift operation is pending
  *        (ie. SHFP=0) when PREDIV_S >= SS.
  * @rmtoll SSR          SS            LL_RTC_TIME_GetSubSecond
  * @param  RTCx RTC Instance
  * @retval Sub second value (number between 0 and 65535)
  */
__STATIC_INLINE uint32_t LL_RTC_TIME_GetSubSecond(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->SSR, RTC_SSR_SS));
}

/**
  * @brief  Synchronize to a remote clock with a high degree of precision.
  * @note   This operation effectively subtracts from (delays) or advance the clock of a fraction of a second.
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   When REFCKON is set, firmware must not write to Shift control register.
  * @rmtoll SHIFTR       ADD1S         LL_RTC_TIME_Synchronize\n
  *         SHIFTR       SUBFS         LL_RTC_TIME_Synchronize
  * @param  RTCx RTC Instance
  * @param  ShiftSecond This parameter can be one of the following values:
  *         @arg @ref LL_RTC_SHIFT_SECOND_DELAY
  *         @arg @ref LL_RTC_SHIFT_SECOND_ADVANCE
  * @param  Fraction Number of Seconds Fractions (any value from 0 to 0x7FFF)
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TIME_Synchronize(RTC_TypeDef *RTCx, uint32_t ShiftSecond, uint32_t Fraction)
{
  WRITE_REG(RTCx->SHIFTR, ShiftSecond | Fraction);
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_Date Date
  * @{
  */

/**
  * @brief  Set Year in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Year from binary to BCD format
  * @rmtoll DR           YT            LL_RTC_DATE_SetYear\n
  *         DR           YU            LL_RTC_DATE_SetYear
  * @param  RTCx RTC Instance
  * @param  Year Value between Min_Data=0x00 and Max_Data=0x99
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DATE_SetYear(RTC_TypeDef *RTCx, uint32_t Year)
{
  MODIFY_REG(RTCx->DR, (RTC_DR_YT | RTC_DR_YU),
             (((Year & 0xF0U) << (RTC_DR_YT_Pos - 4U)) | ((Year & 0x0FU) << RTC_DR_YU_Pos)));
}

/**
  * @brief  Get Year in BCD format
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Year from BCD to Binary format
  * @rmtoll DR           YT            LL_RTC_DATE_GetYear\n
  *         DR           YU            LL_RTC_DATE_GetYear
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x99
  */
__STATIC_INLINE uint32_t LL_RTC_DATE_GetYear(RTC_TypeDef *RTCx)
{
  return (uint32_t)((READ_BIT(RTCx->DR, (RTC_DR_YT | RTC_DR_YU))) >> RTC_DR_YU_Pos);
}

/**
  * @brief  Set Week day
  * @rmtoll DR           WDU           LL_RTC_DATE_SetWeekDay
  * @param  RTCx RTC Instance
  * @param  WeekDay This parameter can be one of the following values:
  *         @arg @ref LL_RTC_WEEKDAY_MONDAY
  *         @arg @ref LL_RTC_WEEKDAY_TUESDAY
  *         @arg @ref LL_RTC_WEEKDAY_WEDNESDAY
  *         @arg @ref LL_RTC_WEEKDAY_THURSDAY
  *         @arg @ref LL_RTC_WEEKDAY_FRIDAY
  *         @arg @ref LL_RTC_WEEKDAY_SATURDAY
  *         @arg @ref LL_RTC_WEEKDAY_SUNDAY
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DATE_SetWeekDay(RTC_TypeDef *RTCx, uint32_t WeekDay)
{
  MODIFY_REG(RTCx->DR, RTC_DR_WDU, WeekDay << RTC_DR_WDU_Pos);
}

/**
  * @brief  Get Week day
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @rmtoll DR           WDU           LL_RTC_DATE_GetWeekDay
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_WEEKDAY_MONDAY
  *         @arg @ref LL_RTC_WEEKDAY_TUESDAY
  *         @arg @ref LL_RTC_WEEKDAY_WEDNESDAY
  *         @arg @ref LL_RTC_WEEKDAY_THURSDAY
  *         @arg @ref LL_RTC_WEEKDAY_FRIDAY
  *         @arg @ref LL_RTC_WEEKDAY_SATURDAY
  *         @arg @ref LL_RTC_WEEKDAY_SUNDAY
  */
__STATIC_INLINE uint32_t LL_RTC_DATE_GetWeekDay(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->DR, RTC_DR_WDU) >> RTC_DR_WDU_Pos);
}

/**
  * @brief  Set Month in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Month from binary to BCD format
  * @rmtoll DR           MT            LL_RTC_DATE_SetMonth\n
  *         DR           MU            LL_RTC_DATE_SetMonth
  * @param  RTCx RTC Instance
  * @param  Month This parameter can be one of the following values:
  *         @arg @ref LL_RTC_MONTH_JANUARY
  *         @arg @ref LL_RTC_MONTH_FEBRUARY
  *         @arg @ref LL_RTC_MONTH_MARCH
  *         @arg @ref LL_RTC_MONTH_APRIL
  *         @arg @ref LL_RTC_MONTH_MAY
  *         @arg @ref LL_RTC_MONTH_JUNE
  *         @arg @ref LL_RTC_MONTH_JULY
  *         @arg @ref LL_RTC_MONTH_AUGUST
  *         @arg @ref LL_RTC_MONTH_SEPTEMBER
  *         @arg @ref LL_RTC_MONTH_OCTOBER
  *         @arg @ref LL_RTC_MONTH_NOVEMBER
  *         @arg @ref LL_RTC_MONTH_DECEMBER
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DATE_SetMonth(RTC_TypeDef *RTCx, uint32_t Month)
{
  MODIFY_REG(RTCx->DR, (RTC_DR_MT | RTC_DR_MU),
             (((Month & 0xF0U) << (RTC_DR_MT_Pos - 4U)) | ((Month & 0x0FU) << RTC_DR_MU_Pos)));
}

/**
  * @brief  Get Month in BCD format
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Month from BCD to Binary format
  * @rmtoll DR           MT            LL_RTC_DATE_GetMonth\n
  *         DR           MU            LL_RTC_DATE_GetMonth
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_MONTH_JANUARY
  *         @arg @ref LL_RTC_MONTH_FEBRUARY
  *         @arg @ref LL_RTC_MONTH_MARCH
  *         @arg @ref LL_RTC_MONTH_APRIL
  *         @arg @ref LL_RTC_MONTH_MAY
  *         @arg @ref LL_RTC_MONTH_JUNE
  *         @arg @ref LL_RTC_MONTH_JULY
  *         @arg @ref LL_RTC_MONTH_AUGUST
  *         @arg @ref LL_RTC_MONTH_SEPTEMBER
  *         @arg @ref LL_RTC_MONTH_OCTOBER
  *         @arg @ref LL_RTC_MONTH_NOVEMBER
  *         @arg @ref LL_RTC_MONTH_DECEMBER
  */
__STATIC_INLINE uint32_t LL_RTC_DATE_GetMonth(RTC_TypeDef *RTCx)
{
  return (uint32_t)((READ_BIT(RTCx->DR, (RTC_DR_MT | RTC_DR_MU)))>> RTC_DR_MU_Pos);
}

/**
  * @brief  Set Day in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Day from binary to BCD format
  * @rmtoll DR           DT            LL_RTC_DATE_SetDay\n
  *         DR           DU            LL_RTC_DATE_SetDay
  * @param  RTCx RTC Instance
  * @param  Day Value between Min_Data=0x01 and Max_Data=0x31
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DATE_SetDay(RTC_TypeDef *RTCx, uint32_t Day)
{
  MODIFY_REG(RTCx->DR, (RTC_DR_DT | RTC_DR_DU),
             (((Day & 0xF0U) << (RTC_DR_DT_Pos - 4U)) | ((Day & 0x0FU) << RTC_DR_DU_Pos)));
}

/**
  * @brief  Get Day in BCD format
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Day from BCD to Binary format
  * @rmtoll DR           DT            LL_RTC_DATE_GetDay\n
  *         DR           DU            LL_RTC_DATE_GetDay
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x01 and Max_Data=0x31
  */
__STATIC_INLINE uint32_t LL_RTC_DATE_GetDay(RTC_TypeDef *RTCx)
{
  return (uint32_t)((READ_BIT(RTCx->DR, (RTC_DR_DT | RTC_DR_DU))) >> RTC_DR_DU_Pos);
}

/**
  * @brief  Set date (WeekDay, Day, Month and Year) in BCD format
  * @rmtoll DR           WDU           LL_RTC_DATE_Config\n
  *         DR           MT            LL_RTC_DATE_Config\n
  *         DR           MU            LL_RTC_DATE_Config\n
  *         DR           DT            LL_RTC_DATE_Config\n
  *         DR           DU            LL_RTC_DATE_Config\n
  *         DR           YT            LL_RTC_DATE_Config\n
  *         DR           YU            LL_RTC_DATE_Config
  * @param  RTCx RTC Instance
  * @param  WeekDay This parameter can be one of the following values:
  *         @arg @ref LL_RTC_WEEKDAY_MONDAY
  *         @arg @ref LL_RTC_WEEKDAY_TUESDAY
  *         @arg @ref LL_RTC_WEEKDAY_WEDNESDAY
  *         @arg @ref LL_RTC_WEEKDAY_THURSDAY
  *         @arg @ref LL_RTC_WEEKDAY_FRIDAY
  *         @arg @ref LL_RTC_WEEKDAY_SATURDAY
  *         @arg @ref LL_RTC_WEEKDAY_SUNDAY
  * @param  Day Value between Min_Data=0x01 and Max_Data=0x31
  * @param  Month This parameter can be one of the following values:
  *         @arg @ref LL_RTC_MONTH_JANUARY
  *         @arg @ref LL_RTC_MONTH_FEBRUARY
  *         @arg @ref LL_RTC_MONTH_MARCH
  *         @arg @ref LL_RTC_MONTH_APRIL
  *         @arg @ref LL_RTC_MONTH_MAY
  *         @arg @ref LL_RTC_MONTH_JUNE
  *         @arg @ref LL_RTC_MONTH_JULY
  *         @arg @ref LL_RTC_MONTH_AUGUST
  *         @arg @ref LL_RTC_MONTH_SEPTEMBER
  *         @arg @ref LL_RTC_MONTH_OCTOBER
  *         @arg @ref LL_RTC_MONTH_NOVEMBER
  *         @arg @ref LL_RTC_MONTH_DECEMBER
  * @param  Year Value between Min_Data=0x00 and Max_Data=0x99
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DATE_Config(RTC_TypeDef *RTCx, uint32_t WeekDay, uint32_t Day, uint32_t Month, uint32_t Year)
{
  register uint32_t temp = 0U;

  temp = (WeekDay << RTC_DR_WDU_Pos)                                                        | \
         (((Year & 0xF0U) << (RTC_DR_YT_Pos - 4U)) | ((Year & 0x0FU) << RTC_DR_YU_Pos))   | \
         (((Month & 0xF0U) << (RTC_DR_MT_Pos - 4U)) | ((Month & 0x0FU) << RTC_DR_MU_Pos)) | \
         (((Day & 0xF0U) << (RTC_DR_DT_Pos - 4U)) | ((Day & 0x0FU) << RTC_DR_DU_Pos));

  MODIFY_REG(RTCx->DR, (RTC_DR_WDU | RTC_DR_MT | RTC_DR_MU | RTC_DR_DT | RTC_DR_DU | RTC_DR_YT | RTC_DR_YU), temp);
}

/**
  * @brief  Get date (WeekDay, Day, Month and Year) in BCD format
  * @note if shadow mode is disabled (BYPSHAD=0), need to check if RSF flag is set
  *       before reading this bit
  * @note helper macros __LL_RTC_GET_WEEKDAY, __LL_RTC_GET_YEAR, __LL_RTC_GET_MONTH,
  * and __LL_RTC_GET_DAY are available to get independently each parameter.
  * @rmtoll DR           WDU           LL_RTC_DATE_Get\n
  *         DR           MT            LL_RTC_DATE_Get\n
  *         DR           MU            LL_RTC_DATE_Get\n
  *         DR           DT            LL_RTC_DATE_Get\n
  *         DR           DU            LL_RTC_DATE_Get\n
  *         DR           YT            LL_RTC_DATE_Get\n
  *         DR           YU            LL_RTC_DATE_Get
  * @param  RTCx RTC Instance
  * @retval Combination of WeekDay, Day, Month and Year (Format: 0xWWDDMMYY).
  */
__STATIC_INLINE uint32_t LL_RTC_DATE_Get(RTC_TypeDef *RTCx)
{
  register uint32_t temp = 0U;
  
  temp = READ_BIT(RTCx->DR, (RTC_DR_WDU | RTC_DR_MT | RTC_DR_MU | RTC_DR_DT | RTC_DR_DU | RTC_DR_YT | RTC_DR_YU));
  return (uint32_t)((((temp & RTC_DR_WDU) >> RTC_DR_WDU_Pos) << RTC_OFFSET_WEEKDAY) | \
                    (((((temp & RTC_DR_DT) >> RTC_DR_DT_Pos) << 4U) | ((temp & RTC_DR_DU) >> RTC_DR_DU_Pos)) << RTC_OFFSET_DAY) | \
                    (((((temp & RTC_DR_MT) >> RTC_DR_MT_Pos) << 4U) | ((temp & RTC_DR_MU) >> RTC_DR_MU_Pos)) << RTC_OFFSET_MONTH) | \
                    ((((temp & RTC_DR_YT) >> RTC_DR_YT_Pos) << 4U) | ((temp & RTC_DR_YU) >> RTC_DR_YU_Pos)));
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_ALARMA ALARMA
  * @{
  */

/**
  * @brief  Enable Alarm A
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           ALRAE         LL_RTC_ALMA_Enable
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_Enable(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_ALRAE);
}

/**
  * @brief  Disable Alarm A
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           ALRAE         LL_RTC_ALMA_Disable
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_Disable(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_ALRAE);
}

/**
  * @brief  Specify the Alarm A masks.
  * @rmtoll ALRMAR       MSK4          LL_RTC_ALMA_SetMask\n
  *         ALRMAR       MSK3          LL_RTC_ALMA_SetMask\n
  *         ALRMAR       MSK2          LL_RTC_ALMA_SetMask\n
  *         ALRMAR       MSK1          LL_RTC_ALMA_SetMask
  * @param  RTCx RTC Instance
  * @param  Mask This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_ALMA_MASK_NONE
  *         @arg @ref LL_RTC_ALMA_MASK_DATEWEEKDAY
  *         @arg @ref LL_RTC_ALMA_MASK_HOURS
  *         @arg @ref LL_RTC_ALMA_MASK_MINUTES
  *         @arg @ref LL_RTC_ALMA_MASK_SECONDS
  *         @arg @ref LL_RTC_ALMA_MASK_ALL
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_SetMask(RTC_TypeDef *RTCx, uint32_t Mask)
{
  MODIFY_REG(RTCx->ALRMAR, RTC_ALRMAR_MSK4 | RTC_ALRMAR_MSK3 | RTC_ALRMAR_MSK2 | RTC_ALRMAR_MSK1, Mask);
}

/**
  * @brief  Get the Alarm A masks.
  * @rmtoll ALRMAR       MSK4          LL_RTC_ALMA_GetMask\n
  *         ALRMAR       MSK3          LL_RTC_ALMA_GetMask\n
  *         ALRMAR       MSK2          LL_RTC_ALMA_GetMask\n
  *         ALRMAR       MSK1          LL_RTC_ALMA_GetMask
  * @param  RTCx RTC Instance
  * @retval Returned value can be can be a combination of the following values:
  *         @arg @ref LL_RTC_ALMA_MASK_NONE
  *         @arg @ref LL_RTC_ALMA_MASK_DATEWEEKDAY
  *         @arg @ref LL_RTC_ALMA_MASK_HOURS
  *         @arg @ref LL_RTC_ALMA_MASK_MINUTES
  *         @arg @ref LL_RTC_ALMA_MASK_SECONDS
  *         @arg @ref LL_RTC_ALMA_MASK_ALL
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetMask(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMAR, RTC_ALRMAR_MSK4 | RTC_ALRMAR_MSK3 | RTC_ALRMAR_MSK2 | RTC_ALRMAR_MSK1));
}

/**
  * @brief  Enable AlarmA Week day selection (DU[3:0] represents the week day. DT[1:0] is do not care)
  * @rmtoll ALRMAR       WDSEL         LL_RTC_ALMA_EnableWeekday
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_EnableWeekday(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->ALRMAR, RTC_ALRMAR_WDSEL);
}

/**
  * @brief  Disable AlarmA Week day selection (DU[3:0] represents the date )
  * @rmtoll ALRMAR       WDSEL         LL_RTC_ALMA_DisableWeekday
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_DisableWeekday(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->ALRMAR, RTC_ALRMAR_WDSEL);
}

/**
  * @brief  Set ALARM A Day in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Day from binary to BCD format
  * @rmtoll ALRMAR       DT            LL_RTC_ALMA_SetDay\n
  *         ALRMAR       DU            LL_RTC_ALMA_SetDay
  * @param  RTCx RTC Instance
  * @param  Day Value between Min_Data=0x01 and Max_Data=0x31
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_SetDay(RTC_TypeDef *RTCx, uint32_t Day)
{
  MODIFY_REG(RTCx->ALRMAR, (RTC_ALRMAR_DT | RTC_ALRMAR_DU),
             (((Day & 0xF0U) << (RTC_ALRMAR_DT_Pos - 4U)) | ((Day & 0x0FU) << RTC_ALRMAR_DU_Pos)));
}

/**
  * @brief  Get ALARM A Day in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Day from BCD to Binary format
  * @rmtoll ALRMAR       DT            LL_RTC_ALMA_GetDay\n
  *         ALRMAR       DU            LL_RTC_ALMA_GetDay
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x01 and Max_Data=0x31
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetDay(RTC_TypeDef *RTCx)
{
  return (uint32_t)((READ_BIT(RTCx->ALRMAR, (RTC_ALRMAR_DT | RTC_ALRMAR_DU))) >> RTC_ALRMAR_DU_Pos);
}

/**
  * @brief  Set ALARM A Weekday
  * @rmtoll ALRMAR       DU            LL_RTC_ALMA_SetWeekDay
  * @param  RTCx RTC Instance
  * @param  WeekDay This parameter can be one of the following values:
  *         @arg @ref LL_RTC_WEEKDAY_MONDAY
  *         @arg @ref LL_RTC_WEEKDAY_TUESDAY
  *         @arg @ref LL_RTC_WEEKDAY_WEDNESDAY
  *         @arg @ref LL_RTC_WEEKDAY_THURSDAY
  *         @arg @ref LL_RTC_WEEKDAY_FRIDAY
  *         @arg @ref LL_RTC_WEEKDAY_SATURDAY
  *         @arg @ref LL_RTC_WEEKDAY_SUNDAY
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_SetWeekDay(RTC_TypeDef *RTCx, uint32_t WeekDay)
{
  MODIFY_REG(RTCx->ALRMAR, RTC_ALRMAR_DU, WeekDay << RTC_ALRMAR_DU_Pos);
}

/**
  * @brief  Get ALARM A Weekday
  * @rmtoll ALRMAR       DU            LL_RTC_ALMA_GetWeekDay
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_WEEKDAY_MONDAY
  *         @arg @ref LL_RTC_WEEKDAY_TUESDAY
  *         @arg @ref LL_RTC_WEEKDAY_WEDNESDAY
  *         @arg @ref LL_RTC_WEEKDAY_THURSDAY
  *         @arg @ref LL_RTC_WEEKDAY_FRIDAY
  *         @arg @ref LL_RTC_WEEKDAY_SATURDAY
  *         @arg @ref LL_RTC_WEEKDAY_SUNDAY
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetWeekDay(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMAR, RTC_ALRMAR_DU) >> RTC_ALRMAR_DU_Pos);
}

/**
  * @brief  Set Alarm A time format (AM/24-hour or PM notation)
  * @rmtoll ALRMAR       PM            LL_RTC_ALMA_SetTimeFormat
  * @param  RTCx RTC Instance
  * @param  TimeFormat This parameter can be one of the following values:
  *         @arg @ref LL_RTC_ALMA_TIME_FORMAT_AM
  *         @arg @ref LL_RTC_ALMA_TIME_FORMAT_PM
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_SetTimeFormat(RTC_TypeDef *RTCx, uint32_t TimeFormat)
{
  MODIFY_REG(RTCx->ALRMAR, RTC_ALRMAR_PM, TimeFormat);
}

/**
  * @brief  Get Alarm A time format (AM or PM notation)
  * @rmtoll ALRMAR       PM            LL_RTC_ALMA_GetTimeFormat
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_ALMA_TIME_FORMAT_AM
  *         @arg @ref LL_RTC_ALMA_TIME_FORMAT_PM
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetTimeFormat(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMAR, RTC_ALRMAR_PM));
}

/**
  * @brief  Set ALARM A Hours in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Hours from binary to BCD format
  * @rmtoll ALRMAR       HT            LL_RTC_ALMA_SetHour\n
  *         ALRMAR       HU            LL_RTC_ALMA_SetHour
  * @param  RTCx RTC Instance
  * @param  Hours Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_SetHour(RTC_TypeDef *RTCx, uint32_t Hours)
{
  MODIFY_REG(RTCx->ALRMAR, (RTC_ALRMAR_HT | RTC_ALRMAR_HU),
             (((Hours & 0xF0U) << (RTC_ALRMAR_HT_Pos - 4U)) | ((Hours & 0x0FU) << RTC_ALRMAR_HU_Pos)));
}

/**
  * @brief  Get ALARM A Hours in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Hours from BCD to Binary format
  * @rmtoll ALRMAR       HT            LL_RTC_ALMA_GetHour\n
  *         ALRMAR       HU            LL_RTC_ALMA_GetHour
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetHour(RTC_TypeDef *RTCx)
{
  return (uint32_t)(( READ_BIT(RTCx->ALRMAR, (RTC_ALRMAR_HT | RTC_ALRMAR_HU))) >> RTC_ALRMAR_HU_Pos);
}

/**
  * @brief  Set ALARM A Minutes in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Minutes from binary to BCD format
  * @rmtoll ALRMAR       MNT           LL_RTC_ALMA_SetMinute\n
  *         ALRMAR       MNU           LL_RTC_ALMA_SetMinute
  * @param  RTCx RTC Instance
  * @param  Minutes Value between Min_Data=0x00 and Max_Data=0x59
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_SetMinute(RTC_TypeDef *RTCx, uint32_t Minutes)
{
  MODIFY_REG(RTCx->ALRMAR, (RTC_ALRMAR_MNT | RTC_ALRMAR_MNU),
             (((Minutes & 0xF0U) << (RTC_ALRMAR_MNT_Pos - 4U)) | ((Minutes & 0x0FU) << RTC_ALRMAR_MNU_Pos)));
}

/**
  * @brief  Get ALARM A Minutes in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Minutes from BCD to Binary format
  * @rmtoll ALRMAR       MNT           LL_RTC_ALMA_GetMinute\n
  *         ALRMAR       MNU           LL_RTC_ALMA_GetMinute
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x59
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetMinute(RTC_TypeDef *RTCx)
{
  return (uint32_t)((READ_BIT(RTCx->ALRMAR, (RTC_ALRMAR_MNT | RTC_ALRMAR_MNU))) >> RTC_ALRMAR_MNU_Pos);
}

/**
  * @brief  Set ALARM A Seconds in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Seconds from binary to BCD format
  * @rmtoll ALRMAR       ST            LL_RTC_ALMA_SetSecond\n
  *         ALRMAR       SU            LL_RTC_ALMA_SetSecond
  * @param  RTCx RTC Instance
  * @param  Seconds Value between Min_Data=0x00 and Max_Data=0x59
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_SetSecond(RTC_TypeDef *RTCx, uint32_t Seconds)
{
  MODIFY_REG(RTCx->ALRMAR, (RTC_ALRMAR_ST | RTC_ALRMAR_SU),
             (((Seconds & 0xF0U) << (RTC_ALRMAR_ST_Pos - 4U)) | ((Seconds & 0x0FU) << RTC_ALRMAR_SU_Pos)));
}

/**
  * @brief  Get ALARM A Seconds in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Seconds from BCD to Binary format
  * @rmtoll ALRMAR       ST            LL_RTC_ALMA_GetSecond\n
  *         ALRMAR       SU            LL_RTC_ALMA_GetSecond
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x59
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetSecond(RTC_TypeDef *RTCx)
{
  return (uint32_t)((READ_BIT(RTCx->ALRMAR, (RTC_ALRMAR_ST | RTC_ALRMAR_SU))) >> RTC_ALRMAR_SU_Pos);
}

/**
  * @brief  Set Alarm A Time (hour, minute and second) in BCD format
  * @rmtoll ALRMAR       PM            LL_RTC_ALMA_ConfigTime\n
  *         ALRMAR       HT            LL_RTC_ALMA_ConfigTime\n
  *         ALRMAR       HU            LL_RTC_ALMA_ConfigTime\n
  *         ALRMAR       MNT           LL_RTC_ALMA_ConfigTime\n
  *         ALRMAR       MNU           LL_RTC_ALMA_ConfigTime\n
  *         ALRMAR       ST            LL_RTC_ALMA_ConfigTime\n
  *         ALRMAR       SU            LL_RTC_ALMA_ConfigTime
  * @param  RTCx RTC Instance
  * @param  Format12_24 This parameter can be one of the following values:
  *         @arg @ref LL_RTC_ALMA_TIME_FORMAT_AM
  *         @arg @ref LL_RTC_ALMA_TIME_FORMAT_PM
  * @param  Hours Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  * @param  Minutes Value between Min_Data=0x00 and Max_Data=0x59
  * @param  Seconds Value between Min_Data=0x00 and Max_Data=0x59
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_ConfigTime(RTC_TypeDef *RTCx, uint32_t Format12_24, uint32_t Hours, uint32_t Minutes, uint32_t Seconds)
{
  register uint32_t temp = 0U;

  temp = Format12_24 | (((Hours & 0xF0U) << (RTC_ALRMAR_HT_Pos - 4U)) | ((Hours & 0x0FU) << RTC_ALRMAR_HU_Pos))    | \
         (((Minutes & 0xF0U) << (RTC_ALRMAR_MNT_Pos - 4U)) | ((Minutes & 0x0FU) << RTC_ALRMAR_MNU_Pos)) | \
         (((Seconds & 0xF0U) << (RTC_ALRMAR_ST_Pos - 4U)) | ((Seconds & 0x0FU) << RTC_ALRMAR_SU_Pos));

  MODIFY_REG(RTCx->ALRMAR, RTC_ALRMAR_PM | RTC_ALRMAR_HT | RTC_ALRMAR_HU | RTC_ALRMAR_MNT | RTC_ALRMAR_MNU | RTC_ALRMAR_ST | RTC_ALRMAR_SU, temp);
}

/**
  * @brief  Get Alarm B Time (hour, minute and second) in BCD format
  * @note helper macros __LL_RTC_GET_HOUR, __LL_RTC_GET_MINUTE and __LL_RTC_GET_SECOND
  * are available to get independently each parameter.
  * @rmtoll ALRMAR       HT            LL_RTC_ALMA_GetTime\n
  *         ALRMAR       HU            LL_RTC_ALMA_GetTime\n
  *         ALRMAR       MNT           LL_RTC_ALMA_GetTime\n
  *         ALRMAR       MNU           LL_RTC_ALMA_GetTime\n
  *         ALRMAR       ST            LL_RTC_ALMA_GetTime\n
  *         ALRMAR       SU            LL_RTC_ALMA_GetTime
  * @param  RTCx RTC Instance
  * @retval Combination of hours, minutes and seconds.
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetTime(RTC_TypeDef *RTCx)
{
  return (uint32_t)((LL_RTC_ALMA_GetHour(RTCx) << RTC_OFFSET_HOUR) | (LL_RTC_ALMA_GetMinute(RTCx) << RTC_OFFSET_MINUTE) | LL_RTC_ALMA_GetSecond(RTCx));
}

/**
  * @brief  Set Alarm A Mask the most-significant bits starting at this bit
  * @note This register can be written only when ALRAE is reset in RTC_CR register,
  *       or in initialization mode.
  * @rmtoll ALRMASSR     MASKSS        LL_RTC_ALMA_SetSubSecondMask
  * @param  RTCx RTC Instance
  * @param  Mask Value between Min_Data=0x00 and Max_Data=0xF
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_SetSubSecondMask(RTC_TypeDef *RTCx, uint32_t Mask)
{
  MODIFY_REG(RTCx->ALRMASSR, RTC_ALRMASSR_MASKSS, Mask << RTC_ALRMASSR_MASKSS_Pos);
}

/**
  * @brief  Get Alarm A Mask the most-significant bits starting at this bit
  * @rmtoll ALRMASSR     MASKSS        LL_RTC_ALMA_GetSubSecondMask
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0xF
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetSubSecondMask(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMASSR, RTC_ALRMASSR_MASKSS) >> RTC_ALRMASSR_MASKSS_Pos);
}

/**
  * @brief  Set Alarm A Sub seconds value
  * @rmtoll ALRMASSR     SS            LL_RTC_ALMA_SetSubSecond
  * @param  RTCx RTC Instance
  * @param  Subsecond Value between Min_Data=0x00 and Max_Data=0x7FFF
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMA_SetSubSecond(RTC_TypeDef *RTCx, uint32_t Subsecond)
{
  MODIFY_REG(RTCx->ALRMASSR, RTC_ALRMASSR_SS, Subsecond);
}

/**
  * @brief  Get Alarm A Sub seconds value
  * @rmtoll ALRMASSR     SS            LL_RTC_ALMA_GetSubSecond
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x7FFF
  */
__STATIC_INLINE uint32_t LL_RTC_ALMA_GetSubSecond(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMASSR, RTC_ALRMASSR_SS));
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_ALARMB ALARMB
  * @{
  */

/**
  * @brief  Enable Alarm B
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           ALRBE         LL_RTC_ALMB_Enable
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_Enable(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_ALRBE);
}

/**
  * @brief  Disable Alarm B
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           ALRBE         LL_RTC_ALMB_Disable
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_Disable(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_ALRBE);
}

/**
  * @brief  Specify the Alarm B masks.
  * @rmtoll ALRMBR       MSK4          LL_RTC_ALMB_SetMask\n
  *         ALRMBR       MSK3          LL_RTC_ALMB_SetMask\n
  *         ALRMBR       MSK2          LL_RTC_ALMB_SetMask\n
  *         ALRMBR       MSK1          LL_RTC_ALMB_SetMask
  * @param  RTCx RTC Instance
  * @param  Mask This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_ALMB_MASK_NONE
  *         @arg @ref LL_RTC_ALMB_MASK_DATEWEEKDAY
  *         @arg @ref LL_RTC_ALMB_MASK_HOURS
  *         @arg @ref LL_RTC_ALMB_MASK_MINUTES
  *         @arg @ref LL_RTC_ALMB_MASK_SECONDS
  *         @arg @ref LL_RTC_ALMB_MASK_ALL
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_SetMask(RTC_TypeDef *RTCx, uint32_t Mask)
{
  MODIFY_REG(RTCx->ALRMBR, RTC_ALRMBR_MSK4 | RTC_ALRMBR_MSK3 | RTC_ALRMBR_MSK2 | RTC_ALRMBR_MSK1, Mask);
}

/**
  * @brief  Get the Alarm B masks.
  * @rmtoll ALRMBR       MSK4          LL_RTC_ALMB_GetMask\n
  *         ALRMBR       MSK3          LL_RTC_ALMB_GetMask\n
  *         ALRMBR       MSK2          LL_RTC_ALMB_GetMask\n
  *         ALRMBR       MSK1          LL_RTC_ALMB_GetMask
  * @param  RTCx RTC Instance
  * @retval Returned value can be can be a combination of the following values:
  *         @arg @ref LL_RTC_ALMB_MASK_NONE
  *         @arg @ref LL_RTC_ALMB_MASK_DATEWEEKDAY
  *         @arg @ref LL_RTC_ALMB_MASK_HOURS
  *         @arg @ref LL_RTC_ALMB_MASK_MINUTES
  *         @arg @ref LL_RTC_ALMB_MASK_SECONDS
  *         @arg @ref LL_RTC_ALMB_MASK_ALL
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetMask(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMBR, RTC_ALRMBR_MSK4 | RTC_ALRMBR_MSK3 | RTC_ALRMBR_MSK2 | RTC_ALRMBR_MSK1));
}

/**
  * @brief  Enable AlarmB Week day selection (DU[3:0] represents the week day. DT[1:0] is do not care)
  * @rmtoll ALRMBR       WDSEL         LL_RTC_ALMB_EnableWeekday
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_EnableWeekday(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->ALRMBR, RTC_ALRMBR_WDSEL);
}

/**
  * @brief  Disable AlarmB Week day selection (DU[3:0] represents the date )
  * @rmtoll ALRMBR       WDSEL         LL_RTC_ALMB_DisableWeekday
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_DisableWeekday(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->ALRMBR, RTC_ALRMBR_WDSEL);
}

/**
  * @brief  Set ALARM B Day in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Day from binary to BCD format
  * @rmtoll ALRMBR       DT            LL_RTC_ALMB_SetDay\n
  *         ALRMBR       DU            LL_RTC_ALMB_SetDay
  * @param  RTCx RTC Instance
  * @param  Day Value between Min_Data=0x01 and Max_Data=0x31
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_SetDay(RTC_TypeDef *RTCx, uint32_t Day)
{
  MODIFY_REG(RTC->ALRMBR, (RTC_ALRMBR_DT | RTC_ALRMBR_DU),
             (((Day & 0xF0U) << (RTC_ALRMBR_DT_Pos - 4U)) | ((Day & 0x0FU) << RTC_ALRMBR_DU_Pos)));
}

/**
  * @brief  Get ALARM B Day in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Day from BCD to Binary format
  * @rmtoll ALRMBR       DT            LL_RTC_ALMB_GetDay\n
  *         ALRMBR       DU            LL_RTC_ALMB_GetDay
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x01 and Max_Data=0x31
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetDay(RTC_TypeDef *RTCx)
{
  return (uint32_t)(( READ_BIT(RTCx->ALRMBR, (RTC_ALRMBR_DT | RTC_ALRMBR_DU))) >> RTC_ALRMBR_DU_Pos);
}

/**
  * @brief  Set ALARM B Weekday
  * @rmtoll ALRMBR       DU            LL_RTC_ALMB_SetWeekDay
  * @param  RTCx RTC Instance
  * @param  WeekDay This parameter can be one of the following values:
  *         @arg @ref LL_RTC_WEEKDAY_MONDAY
  *         @arg @ref LL_RTC_WEEKDAY_TUESDAY
  *         @arg @ref LL_RTC_WEEKDAY_WEDNESDAY
  *         @arg @ref LL_RTC_WEEKDAY_THURSDAY
  *         @arg @ref LL_RTC_WEEKDAY_FRIDAY
  *         @arg @ref LL_RTC_WEEKDAY_SATURDAY
  *         @arg @ref LL_RTC_WEEKDAY_SUNDAY
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_SetWeekDay(RTC_TypeDef *RTCx, uint32_t WeekDay)
{
  MODIFY_REG(RTCx->ALRMBR, RTC_ALRMBR_DU, WeekDay << RTC_ALRMBR_DU_Pos);
}

/**
  * @brief  Get ALARM B Weekday
  * @rmtoll ALRMBR       DU            LL_RTC_ALMB_GetWeekDay
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_WEEKDAY_MONDAY
  *         @arg @ref LL_RTC_WEEKDAY_TUESDAY
  *         @arg @ref LL_RTC_WEEKDAY_WEDNESDAY
  *         @arg @ref LL_RTC_WEEKDAY_THURSDAY
  *         @arg @ref LL_RTC_WEEKDAY_FRIDAY
  *         @arg @ref LL_RTC_WEEKDAY_SATURDAY
  *         @arg @ref LL_RTC_WEEKDAY_SUNDAY
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetWeekDay(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMBR, RTC_ALRMBR_DU) >> RTC_ALRMBR_DU_Pos);
}

/**
  * @brief  Set ALARM B time format (AM/24-hour or PM notation)
  * @rmtoll ALRMBR       PM            LL_RTC_ALMB_SetTimeFormat
  * @param  RTCx RTC Instance
  * @param  TimeFormat This parameter can be one of the following values:
  *         @arg @ref LL_RTC_ALMB_TIME_FORMAT_AM
  *         @arg @ref LL_RTC_ALMB_TIME_FORMAT_PM
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_SetTimeFormat(RTC_TypeDef *RTCx, uint32_t TimeFormat)
{
  MODIFY_REG(RTCx->ALRMBR, RTC_ALRMBR_PM, TimeFormat);
}

/**
  * @brief  Get ALARM B time format (AM or PM notation)
  * @rmtoll ALRMBR       PM            LL_RTC_ALMB_GetTimeFormat
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_ALMB_TIME_FORMAT_AM
  *         @arg @ref LL_RTC_ALMB_TIME_FORMAT_PM
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetTimeFormat(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMBR, RTC_ALRMBR_PM));
}

/**
  * @brief  Set ALARM B Hours in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Hours from binary to BCD format
  * @rmtoll ALRMBR       HT            LL_RTC_ALMB_SetHour\n
  *         ALRMBR       HU            LL_RTC_ALMB_SetHour
  * @param  RTCx RTC Instance
  * @param  Hours Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_SetHour(RTC_TypeDef *RTCx, uint32_t Hours)
{
  MODIFY_REG(RTCx->ALRMBR, (RTC_ALRMBR_HT | RTC_ALRMBR_HU),
             (((Hours & 0xF0U) << (RTC_ALRMBR_HT_Pos - 4U)) | ((Hours & 0x0FU) << RTC_ALRMBR_HU_Pos)));
}

/**
  * @brief  Get ALARM B Hours in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Hours from BCD to Binary format
  * @rmtoll ALRMBR       HT            LL_RTC_ALMB_GetHour\n
  *         ALRMBR       HU            LL_RTC_ALMB_GetHour
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetHour(RTC_TypeDef *RTCx)
{
  return (uint32_t)((READ_BIT(RTCx->ALRMBR, (RTC_ALRMBR_HT | RTC_ALRMBR_HU))) >> RTC_ALRMBR_HU_Pos);
}

/**
  * @brief  Set ALARM B Minutes in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Minutes from binary to BCD format
  * @rmtoll ALRMBR       MNT           LL_RTC_ALMB_SetMinute\n
  *         ALRMBR       MNU           LL_RTC_ALMB_SetMinute
  * @param  RTCx RTC Instance
  * @param  Minutes between Min_Data=0x00 and Max_Data=0x59
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_SetMinute(RTC_TypeDef *RTCx, uint32_t Minutes)
{
  MODIFY_REG(RTCx->ALRMBR, (RTC_ALRMBR_MNT | RTC_ALRMBR_MNU),
             (((Minutes & 0xF0U) << (RTC_ALRMBR_MNT_Pos - 4U)) | ((Minutes & 0x0FU) << RTC_ALRMBR_MNU_Pos)));
}

/**
  * @brief  Get ALARM B Minutes in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Minutes from BCD to Binary format
  * @rmtoll ALRMBR       MNT           LL_RTC_ALMB_GetMinute\n
  *         ALRMBR       MNU           LL_RTC_ALMB_GetMinute
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x59
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetMinute(RTC_TypeDef *RTCx)
{
  return (uint32_t)((READ_BIT(RTCx->ALRMBR, (RTC_ALRMBR_MNT | RTC_ALRMBR_MNU))) >> RTC_ALRMBR_MNU_Pos);
}

/**
  * @brief  Set ALARM B Seconds in BCD format
  * @note helper macro __LL_RTC_CONVERT_BIN2BCD is available to convert Seconds from binary to BCD format
  * @rmtoll ALRMBR       ST            LL_RTC_ALMB_SetSecond\n
  *         ALRMBR       SU            LL_RTC_ALMB_SetSecond
  * @param  RTCx RTC Instance
  * @param  Seconds Value between Min_Data=0x00 and Max_Data=0x59
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_SetSecond(RTC_TypeDef *RTCx, uint32_t Seconds)
{
  MODIFY_REG(RTCx->ALRMBR, (RTC_ALRMBR_ST | RTC_ALRMBR_SU),
             (((Seconds & 0xF0U) << (RTC_ALRMBR_ST_Pos - 4U)) | ((Seconds & 0x0FU) << RTC_ALRMBR_SU_Pos)));
}

/**
  * @brief  Get ALARM B Seconds in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Seconds from BCD to Binary format
  * @rmtoll ALRMBR       ST            LL_RTC_ALMB_GetSecond\n
  *         ALRMBR       SU            LL_RTC_ALMB_GetSecond
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x59
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetSecond(RTC_TypeDef *RTCx)
{
  register uint32_t temp = 0U;

  temp = READ_BIT(RTCx->ALRMBR, (RTC_ALRMBR_ST | RTC_ALRMBR_SU));
  return (uint32_t)((((temp & RTC_ALRMBR_ST) >> RTC_ALRMBR_ST_Pos) << 4U) | ((temp & RTC_ALRMBR_SU) >> RTC_ALRMBR_SU_Pos));
}

/**
  * @brief  Set Alarm B Time (hour, minute and second) in BCD format
  * @rmtoll ALRMBR       PM            LL_RTC_ALMB_ConfigTime\n
  *         ALRMBR       HT            LL_RTC_ALMB_ConfigTime\n
  *         ALRMBR       HU            LL_RTC_ALMB_ConfigTime\n
  *         ALRMBR       MNT           LL_RTC_ALMB_ConfigTime\n
  *         ALRMBR       MNU           LL_RTC_ALMB_ConfigTime\n
  *         ALRMBR       ST            LL_RTC_ALMB_ConfigTime\n
  *         ALRMBR       SU            LL_RTC_ALMB_ConfigTime
  * @param  RTCx RTC Instance
  * @param  Format12_24 This parameter can be one of the following values:
  *         @arg @ref LL_RTC_ALMB_TIME_FORMAT_AM
  *         @arg @ref LL_RTC_ALMB_TIME_FORMAT_PM
  * @param  Hours Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  * @param  Minutes Value between Min_Data=0x00 and Max_Data=0x59
  * @param  Seconds Value between Min_Data=0x00 and Max_Data=0x59
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_ConfigTime(RTC_TypeDef *RTCx, uint32_t Format12_24, uint32_t Hours, uint32_t Minutes, uint32_t Seconds)
{
  register uint32_t temp = 0U;

  temp = Format12_24 | (((Hours & 0xF0U) << (RTC_ALRMBR_HT_Pos - 4U)) | ((Hours & 0x0FU) << RTC_ALRMBR_HU_Pos))    | \
         (((Minutes & 0xF0U) << (RTC_ALRMBR_MNT_Pos - 4U)) | ((Minutes & 0x0FU) << RTC_ALRMBR_MNU_Pos)) | \
         (((Seconds & 0xF0U) << (RTC_ALRMBR_ST_Pos - 4U)) | ((Seconds & 0x0FU) << RTC_ALRMBR_SU_Pos));

  MODIFY_REG(RTCx->ALRMBR, RTC_ALRMBR_PM| RTC_ALRMBR_HT | RTC_ALRMBR_HU | RTC_ALRMBR_MNT | RTC_ALRMBR_MNU | RTC_ALRMBR_ST | RTC_ALRMBR_SU, temp);
}

/**
  * @brief  Get Alarm B Time (hour, minute and second) in BCD format
  * @note helper macros __LL_RTC_GET_HOUR, __LL_RTC_GET_MINUTE and __LL_RTC_GET_SECOND
  * are available to get independently each parameter.
  * @rmtoll ALRMBR       HT            LL_RTC_ALMB_GetTime\n
  *         ALRMBR       HU            LL_RTC_ALMB_GetTime\n
  *         ALRMBR       MNT           LL_RTC_ALMB_GetTime\n
  *         ALRMBR       MNU           LL_RTC_ALMB_GetTime\n
  *         ALRMBR       ST            LL_RTC_ALMB_GetTime\n
  *         ALRMBR       SU            LL_RTC_ALMB_GetTime
  * @param  RTCx RTC Instance
  * @retval Combination of hours, minutes and seconds.
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetTime(RTC_TypeDef *RTCx)
{
  return (uint32_t)((LL_RTC_ALMB_GetHour(RTCx) << RTC_OFFSET_HOUR) | (LL_RTC_ALMB_GetMinute(RTCx) << RTC_OFFSET_MINUTE) | LL_RTC_ALMB_GetSecond(RTCx));
}

/**
  * @brief  Set Alarm B Mask the most-significant bits starting at this bit
  * @note This register can be written only when ALRBE is reset in RTC_CR register,
  *       or in initialization mode.
  * @rmtoll ALRMBSSR     MASKSS        LL_RTC_ALMB_SetSubSecondMask
  * @param  RTCx RTC Instance
  * @param  Mask Value between Min_Data=0x00 and Max_Data=0xF
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_SetSubSecondMask(RTC_TypeDef *RTCx, uint32_t Mask)
{
  MODIFY_REG(RTCx->ALRMBSSR, RTC_ALRMBSSR_MASKSS, Mask << RTC_ALRMBSSR_MASKSS_Pos);
}

/**
  * @brief  Get Alarm B Mask the most-significant bits starting at this bit
  * @rmtoll ALRMBSSR     MASKSS        LL_RTC_ALMB_GetSubSecondMask
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0xF
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetSubSecondMask(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMBSSR, RTC_ALRMBSSR_MASKSS)  >> RTC_ALRMBSSR_MASKSS_Pos);
}

/**
  * @brief  Set Alarm B Sub seconds value
  * @rmtoll ALRMBSSR     SS            LL_RTC_ALMB_SetSubSecond
  * @param  RTCx RTC Instance
  * @param  Subsecond Value between Min_Data=0x00 and Max_Data=0x7FFF
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ALMB_SetSubSecond(RTC_TypeDef *RTCx, uint32_t Subsecond)
{
  MODIFY_REG(RTCx->ALRMBSSR, RTC_ALRMBSSR_SS, Subsecond);
}

/**
  * @brief  Get Alarm B Sub seconds value
  * @rmtoll ALRMBSSR     SS            LL_RTC_ALMB_GetSubSecond
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x7FFF
  */
__STATIC_INLINE uint32_t LL_RTC_ALMB_GetSubSecond(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->ALRMBSSR, RTC_ALRMBSSR_SS));
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_Timestamp Timestamp
  * @{
  */

/**
  * @brief  Enable Timestamp
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           TSE           LL_RTC_TS_Enable
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TS_Enable(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_TSE);
}

/**
  * @brief  Disable Timestamp
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           TSE           LL_RTC_TS_Disable
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TS_Disable(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_TSE);
}

/**
  * @brief  Set Time-stamp event active edge
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note TSE must be reset when TSEDGE is changed to avoid unwanted TSF setting
  * @rmtoll CR           TSEDGE        LL_RTC_TS_SetActiveEdge
  * @param  RTCx RTC Instance
  * @param  Edge This parameter can be one of the following values:
  *         @arg @ref LL_RTC_TIMESTAMP_EDGE_RISING
  *         @arg @ref LL_RTC_TIMESTAMP_EDGE_FALLING
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TS_SetActiveEdge(RTC_TypeDef *RTCx, uint32_t Edge)
{
  MODIFY_REG(RTCx->CR, RTC_CR_TSEDGE, Edge);
}

/**
  * @brief  Get Time-stamp event active edge
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           TSEDGE        LL_RTC_TS_GetActiveEdge
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_TIMESTAMP_EDGE_RISING
  *         @arg @ref LL_RTC_TIMESTAMP_EDGE_FALLING
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetActiveEdge(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CR, RTC_CR_TSEDGE));
}

/**
  * @brief  Get Timestamp AM/PM notation (AM or 24-hour format)
  * @rmtoll TSTR         PM            LL_RTC_TS_GetTimeFormat
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_TS_TIME_FORMAT_AM
  *         @arg @ref LL_RTC_TS_TIME_FORMAT_PM
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetTimeFormat(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSTR, RTC_TSTR_PM));
}

/**
  * @brief  Get Timestamp Hours in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Hours from BCD to Binary format
  * @rmtoll TSTR         HT            LL_RTC_TS_GetHour\n
  *         TSTR         HU            LL_RTC_TS_GetHour
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x01 and Max_Data=0x12 or between Min_Data=0x00 and Max_Data=0x23
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetHour(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSTR, RTC_TSTR_HT | RTC_TSTR_HU) >> RTC_TSTR_HU_Pos);
}

/**
  * @brief  Get Timestamp Minutes in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Minutes from BCD to Binary format
  * @rmtoll TSTR         MNT           LL_RTC_TS_GetMinute\n
  *         TSTR         MNU           LL_RTC_TS_GetMinute
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x59
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetMinute(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSTR, RTC_TSTR_MNT | RTC_TSTR_MNU) >> RTC_TSTR_MNU_Pos);
}

/**
  * @brief  Get Timestamp Seconds in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Seconds from BCD to Binary format
  * @rmtoll TSTR         ST            LL_RTC_TS_GetSecond\n
  *         TSTR         SU            LL_RTC_TS_GetSecond
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0x59
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetSecond(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSTR, RTC_TSTR_ST | RTC_TSTR_SU));
}

/**
  * @brief  Get Timestamp time (hour, minute and second) in BCD format
  * @note helper macros __LL_RTC_GET_HOUR, __LL_RTC_GET_MINUTE and __LL_RTC_GET_SECOND
  * are available to get independently each parameter.
  * @rmtoll TSTR         HT            LL_RTC_TS_GetTime\n
  *         TSTR         HU            LL_RTC_TS_GetTime\n
  *         TSTR         MNT           LL_RTC_TS_GetTime\n
  *         TSTR         MNU           LL_RTC_TS_GetTime\n
  *         TSTR         ST            LL_RTC_TS_GetTime\n
  *         TSTR         SU            LL_RTC_TS_GetTime
  * @param  RTCx RTC Instance
  * @retval Combination of hours, minutes and seconds.
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetTime(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSTR,
                             RTC_TSTR_HT | RTC_TSTR_HU | RTC_TSTR_MNT | RTC_TSTR_MNU | RTC_TSTR_ST | RTC_TSTR_SU));
}

/**
  * @brief  Get Timestamp Week day
  * @rmtoll TSDR         WDU           LL_RTC_TS_GetWeekDay
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_WEEKDAY_MONDAY
  *         @arg @ref LL_RTC_WEEKDAY_TUESDAY
  *         @arg @ref LL_RTC_WEEKDAY_WEDNESDAY
  *         @arg @ref LL_RTC_WEEKDAY_THURSDAY
  *         @arg @ref LL_RTC_WEEKDAY_FRIDAY
  *         @arg @ref LL_RTC_WEEKDAY_SATURDAY
  *         @arg @ref LL_RTC_WEEKDAY_SUNDAY
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetWeekDay(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSDR, RTC_TSDR_WDU) >> RTC_TSDR_WDU_Pos);
}

/**
  * @brief  Get Timestamp Month in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Month from BCD to Binary format
  * @rmtoll TSDR         MT            LL_RTC_TS_GetMonth\n
  *         TSDR         MU            LL_RTC_TS_GetMonth
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_MONTH_JANUARY
  *         @arg @ref LL_RTC_MONTH_FEBRUARY
  *         @arg @ref LL_RTC_MONTH_MARCH
  *         @arg @ref LL_RTC_MONTH_APRIL
  *         @arg @ref LL_RTC_MONTH_MAY
  *         @arg @ref LL_RTC_MONTH_JUNE
  *         @arg @ref LL_RTC_MONTH_JULY
  *         @arg @ref LL_RTC_MONTH_AUGUST
  *         @arg @ref LL_RTC_MONTH_SEPTEMBER
  *         @arg @ref LL_RTC_MONTH_OCTOBER
  *         @arg @ref LL_RTC_MONTH_NOVEMBER
  *         @arg @ref LL_RTC_MONTH_DECEMBER
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetMonth(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSDR, RTC_TSDR_MT | RTC_TSDR_MU) >> RTC_TSDR_MU_Pos);
}

/**
  * @brief  Get Timestamp Day in BCD format
  * @note helper macro __LL_RTC_CONVERT_BCD2BIN is available to convert Day from BCD to Binary format
  * @rmtoll TSDR         DT            LL_RTC_TS_GetDay\n
  *         TSDR         DU            LL_RTC_TS_GetDay
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x01 and Max_Data=0x31
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetDay(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSDR, RTC_TSDR_DT | RTC_TSDR_DU));
}

/**
  * @brief  Get Timestamp date (WeekDay, Day and Month) in BCD format
  * @note helper macros __LL_RTC_GET_WEEKDAY, __LL_RTC_GET_MONTH,
  * and __LL_RTC_GET_DAY are available to get independently each parameter.
  * @rmtoll TSDR         WDU           LL_RTC_TS_GetDate\n
  *         TSDR         MT            LL_RTC_TS_GetDate\n
  *         TSDR         MU            LL_RTC_TS_GetDate\n
  *         TSDR         DT            LL_RTC_TS_GetDate\n
  *         TSDR         DU            LL_RTC_TS_GetDate
  * @param  RTCx RTC Instance
  * @retval Combination of Weekday, Day and Month
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetDate(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSDR, RTC_TSDR_WDU | RTC_TSDR_MT | RTC_TSDR_MU | RTC_TSDR_DT | RTC_TSDR_DU));
}

/**
  * @brief  Get time-stamp sub second value
  * @rmtoll TSSSR        SS            LL_RTC_TS_GetSubSecond
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0xFFFF
  */
__STATIC_INLINE uint32_t LL_RTC_TS_GetSubSecond(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TSSSR, RTC_TSSSR_SS));
}

#if defined(RTC_TAFCR_TAMPTS)
/**
  * @brief  Activate timestamp on tamper detection event
  * @rmtoll TAFCR       TAMPTS        LL_RTC_TS_EnableOnTamper
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TS_EnableOnTamper(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->TAFCR, RTC_TAFCR_TAMPTS);
}

/**
  * @brief  Disable timestamp on tamper detection event
  * @rmtoll TAFCR       TAMPTS        LL_RTC_TS_DisableOnTamper
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TS_DisableOnTamper(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->TAFCR, RTC_TAFCR_TAMPTS);
}
#endif /* RTC_TAFCR_TAMPTS */

/**
  * @brief  Set timestamp Pin
  * @rmtoll TAFCR       TSINSEL      LL_RTC_TS_SetPin
  * @param  RTCx RTC Instance
  * @param  TSPin specifies the RTC TimeStamp Pin.
  *          This parameter can be one of the following values:
  *            @arg LL_RTC_TimeStampPin_Default: RTC_AF1 is used as RTC TimeStamp.
  *            @arg LL_RTC_TimeStampPin_Pos1: RTC_AF2 is selected as RTC TimeStamp. (*)
  *            
  *            (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TS_SetPin(RTC_TypeDef *RTCx, uint32_t TSPin)
{
  MODIFY_REG(RTCx->TAFCR, RTC_TAFCR_TSINSEL , TSPin); 
}

/**
  * @brief  Get timestamp Pin
  * @rmtoll TAFCR       TSINSEL      LL_RTC_TS_GetPin
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *            @arg LL_RTC_TimeStampPin_Default: RTC_AF1 is used as RTC TimeStamp Pin.
  *            @arg LL_RTC_TimeStampPin_Pos1: RTC_AF2 is selected as RTC TimeStamp Pin. (*)
  *            
  *            (*) value not defined in all devices.
  * @retval None
  */

__STATIC_INLINE uint32_t LL_RTC_TS_GetPin(RTC_TypeDef *RTCx)
{
   return (uint32_t)(READ_BIT(RTCx->TAFCR, RTC_TAFCR_TSINSEL));
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_Tamper Tamper
  * @{
  */

/**
  * @brief  Enable RTC_TAMPx input detection
  * @rmtoll TAFCR       TAMP1E        LL_RTC_TAMPER_Enable\n
  *         TAFCR       TAMP2E        LL_RTC_TAMPER_Enable\n
  * @param  RTCx RTC Instance
  * @param  Tamper This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_TAMPER_1 
  *         @arg @ref LL_RTC_TAMPER_2 (*)
  *         
  *         (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_Enable(RTC_TypeDef *RTCx, uint32_t Tamper)
{
  SET_BIT(RTCx->TAFCR, Tamper);
}

/**
  * @brief  Clear RTC_TAMPx input detection
  * @rmtoll TAFCR       TAMP1E        LL_RTC_TAMPER_Disable\n
  *         TAFCR       TAMP2E        LL_RTC_TAMPER_Disable\n
  * @param  RTCx RTC Instance
  * @param  Tamper This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_TAMPER_1 
  *         @arg @ref LL_RTC_TAMPER_2 (*)
  *         
  *         (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_Disable(RTC_TypeDef *RTCx, uint32_t Tamper)
{
  CLEAR_BIT(RTCx->TAFCR, Tamper);
}

#if defined(RTC_TAFCR_TAMPPUDIS)
/**
  * @brief  Disable RTC_TAMPx pull-up disable (Disable precharge of RTC_TAMPx pins)
  * @rmtoll TAFCR       TAMPPUDIS     LL_RTC_TAMPER_DisablePullUp
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_DisablePullUp(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->TAFCR, RTC_TAFCR_TAMPPUDIS);
}

/**
  * @brief  Enable RTC_TAMPx pull-up disable ( Precharge RTC_TAMPx pins before sampling)
  * @rmtoll TAFCR       TAMPPUDIS     LL_RTC_TAMPER_EnablePullUp
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_EnablePullUp(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->TAFCR, RTC_TAFCR_TAMPPUDIS);
}
#endif /* RTC_TAFCR_TAMPPUDIS */

#if defined(RTC_TAFCR_TAMPPRCH)
/**
  * @brief  Set RTC_TAMPx precharge duration
  * @rmtoll TAFCR       TAMPPRCH      LL_RTC_TAMPER_SetPrecharge
  * @param  RTCx RTC Instance
  * @param  Duration This parameter can be one of the following values:
  *         @arg @ref LL_RTC_TAMPER_DURATION_1RTCCLK
  *         @arg @ref LL_RTC_TAMPER_DURATION_2RTCCLK
  *         @arg @ref LL_RTC_TAMPER_DURATION_4RTCCLK
  *         @arg @ref LL_RTC_TAMPER_DURATION_8RTCCLK
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_SetPrecharge(RTC_TypeDef *RTCx, uint32_t Duration)
{
  MODIFY_REG(RTCx->TAFCR, RTC_TAFCR_TAMPPRCH, Duration);
}

/**
  * @brief  Get RTC_TAMPx precharge duration
  * @rmtoll TAFCR       TAMPPRCH      LL_RTC_TAMPER_GetPrecharge
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_TAMPER_DURATION_1RTCCLK
  *         @arg @ref LL_RTC_TAMPER_DURATION_2RTCCLK
  *         @arg @ref LL_RTC_TAMPER_DURATION_4RTCCLK
  *         @arg @ref LL_RTC_TAMPER_DURATION_8RTCCLK
  */
__STATIC_INLINE uint32_t LL_RTC_TAMPER_GetPrecharge(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TAFCR, RTC_TAFCR_TAMPPRCH));
}
#endif /* RTC_TAFCR_TAMPPRCH */

#if defined(RTC_TAFCR_TAMPFLT)
/**
  * @brief  Set RTC_TAMPx filter count
  * @rmtoll TAFCR       TAMPFLT       LL_RTC_TAMPER_SetFilterCount
  * @param  RTCx RTC Instance
  * @param  FilterCount This parameter can be one of the following values:
  *         @arg @ref LL_RTC_TAMPER_FILTER_DISABLE
  *         @arg @ref LL_RTC_TAMPER_FILTER_2SAMPLE
  *         @arg @ref LL_RTC_TAMPER_FILTER_4SAMPLE
  *         @arg @ref LL_RTC_TAMPER_FILTER_8SAMPLE
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_SetFilterCount(RTC_TypeDef *RTCx, uint32_t FilterCount)
{
  MODIFY_REG(RTCx->TAFCR, RTC_TAFCR_TAMPFLT, FilterCount);
}

/**
  * @brief  Get RTC_TAMPx filter count
  * @rmtoll TAFCR       TAMPFLT       LL_RTC_TAMPER_GetFilterCount
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_TAMPER_FILTER_DISABLE
  *         @arg @ref LL_RTC_TAMPER_FILTER_2SAMPLE
  *         @arg @ref LL_RTC_TAMPER_FILTER_4SAMPLE
  *         @arg @ref LL_RTC_TAMPER_FILTER_8SAMPLE
  */
__STATIC_INLINE uint32_t LL_RTC_TAMPER_GetFilterCount(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TAFCR, RTC_TAFCR_TAMPFLT));
}
#endif /* RTC_TAFCR_TAMPFLT */

#if defined(RTC_TAFCR_TAMPFREQ)
/**
  * @brief  Set Tamper sampling frequency
  * @rmtoll TAFCR       TAMPFREQ      LL_RTC_TAMPER_SetSamplingFreq
  * @param  RTCx RTC Instance
  * @param  SamplingFreq This parameter can be one of the following values:
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_32768
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_16384
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_8192
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_4096
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_2048
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_1024
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_512
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_256
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_SetSamplingFreq(RTC_TypeDef *RTCx, uint32_t SamplingFreq)
{
  MODIFY_REG(RTCx->TAFCR, RTC_TAFCR_TAMPFREQ, SamplingFreq);
}

/**
  * @brief  Get Tamper sampling frequency
  * @rmtoll TAFCR       TAMPFREQ      LL_RTC_TAMPER_GetSamplingFreq
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_32768
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_16384
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_8192
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_4096
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_2048
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_1024
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_512
  *         @arg @ref LL_RTC_TAMPER_SAMPLFREQDIV_256
  */
__STATIC_INLINE uint32_t LL_RTC_TAMPER_GetSamplingFreq(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->TAFCR, RTC_TAFCR_TAMPFREQ));
}
#endif /* RTC_TAFCR_TAMPFREQ */

/**
  * @brief  Enable Active level for Tamper input
  * @rmtoll TAFCR       TAMP1TRG      LL_RTC_TAMPER_EnableActiveLevel\n
  *         TAFCR       TAMP2TRG      LL_RTC_TAMPER_EnableActiveLevel\n
  * @param  RTCx RTC Instance
  * @param  Tamper This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_TAMPER_ACTIVELEVEL_TAMP1 
  *         @arg @ref LL_RTC_TAMPER_ACTIVELEVEL_TAMP2 (*)
  *         
  *         (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_EnableActiveLevel(RTC_TypeDef *RTCx, uint32_t Tamper)
{
  SET_BIT(RTCx->TAFCR, Tamper);
}

/**
  * @brief  Disable Active level for Tamper input
  * @rmtoll TAFCR       TAMP1TRG      LL_RTC_TAMPER_DisableActiveLevel\n
  *         TAFCR       TAMP2TRG      LL_RTC_TAMPER_DisableActiveLevel\n
  * @param  RTCx RTC Instance
  * @param  Tamper This parameter can be a combination of the following values:
  *         @arg @ref LL_RTC_TAMPER_ACTIVELEVEL_TAMP1 
  *         @arg @ref LL_RTC_TAMPER_ACTIVELEVEL_TAMP2 (*)
  *         
  *         (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_DisableActiveLevel(RTC_TypeDef *RTCx, uint32_t Tamper)
{
  CLEAR_BIT(RTCx->TAFCR, Tamper);
}

/**
  * @brief  Set Tamper Pin
  * @rmtoll TAFCR       TAMP1INSEL      LL_RTC_TAMPER_SetPin
  * @param  RTCx RTC Instance
  * @param  TamperPin specifies the RTC Tamper Pin.
  *          This parameter can be one of the following values:
  *            @arg LL_RTC_TamperPin_Default: RTC_AF1 is used as RTC Tamper.
  *            @arg LL_RTC_TamperPin_Pos1: RTC_AF2 is selected as RTC Tamper. (*)
  *            
  *            (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_RTC_TAMPER_SetPin(RTC_TypeDef *RTCx, uint32_t TamperPin)
{
  MODIFY_REG(RTCx->TAFCR, RTC_TAFCR_TAMP1INSEL , TamperPin); 
}

/**
  * @brief  Get Tamper Pin
  * @rmtoll TAFCR       TAMP1INSEL      LL_RTC_TAMPER_GetPin
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *            @arg LL_RTC_TamperPin_Default: RTC_AF1 is used as RTC Tamper Pin.
  *            @arg LL_RTC_TamperPin_Pos1: RTC_AF2 is selected as RTC Tamper Pin. (*)
  *            
  *            (*) value not defined in all devices.
  * @retval None
  */

__STATIC_INLINE uint32_t LL_RTC_TAMPER_GetPin(RTC_TypeDef *RTCx)
{
   return (uint32_t)(READ_BIT(RTCx->TAFCR, RTC_TAFCR_TAMP1INSEL));
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_Wakeup Wakeup
  * @{
  */

/**
  * @brief  Enable Wakeup timer
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           WUTE          LL_RTC_WAKEUP_Enable
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_WAKEUP_Enable(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_WUTE);
}

/**
  * @brief  Disable Wakeup timer
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           WUTE          LL_RTC_WAKEUP_Disable
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_WAKEUP_Disable(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_WUTE);
}

/**
  * @brief  Check if Wakeup timer is enabled or not
  * @rmtoll CR           WUTE          LL_RTC_WAKEUP_IsEnabled
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_WAKEUP_IsEnabled(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->CR, RTC_CR_WUTE) == (RTC_CR_WUTE));
}

/**
  * @brief  Select Wakeup clock
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note Bit can be written only when RTC_CR WUTE bit = 0 and RTC_ISR WUTWF bit = 1
  * @rmtoll CR           WUCKSEL       LL_RTC_WAKEUP_SetClock
  * @param  RTCx RTC Instance
  * @param  WakeupClock This parameter can be one of the following values:
  *         @arg @ref LL_RTC_WAKEUPCLOCK_DIV_16
  *         @arg @ref LL_RTC_WAKEUPCLOCK_DIV_8
  *         @arg @ref LL_RTC_WAKEUPCLOCK_DIV_4
  *         @arg @ref LL_RTC_WAKEUPCLOCK_DIV_2
  *         @arg @ref LL_RTC_WAKEUPCLOCK_CKSPRE
  *         @arg @ref LL_RTC_WAKEUPCLOCK_CKSPRE_WUT
  * @retval None
  */
__STATIC_INLINE void LL_RTC_WAKEUP_SetClock(RTC_TypeDef *RTCx, uint32_t WakeupClock)
{
  MODIFY_REG(RTCx->CR, RTC_CR_WUCKSEL, WakeupClock);
}

/**
  * @brief  Get Wakeup clock
  * @rmtoll CR           WUCKSEL       LL_RTC_WAKEUP_GetClock
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_WAKEUPCLOCK_DIV_16
  *         @arg @ref LL_RTC_WAKEUPCLOCK_DIV_8
  *         @arg @ref LL_RTC_WAKEUPCLOCK_DIV_4
  *         @arg @ref LL_RTC_WAKEUPCLOCK_DIV_2
  *         @arg @ref LL_RTC_WAKEUPCLOCK_CKSPRE
  *         @arg @ref LL_RTC_WAKEUPCLOCK_CKSPRE_WUT
  */
__STATIC_INLINE uint32_t LL_RTC_WAKEUP_GetClock(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CR, RTC_CR_WUCKSEL));
}

/**
  * @brief  Set Wakeup auto-reload value
  * @note Bit can be written only when WUTWF is set to 1 in RTC_ISR
  * @rmtoll WUTR         WUT           LL_RTC_WAKEUP_SetAutoReload
  * @param  RTCx RTC Instance
  * @param  Value Value between Min_Data=0x00 and Max_Data=0xFFFF
  * @retval None
  */
__STATIC_INLINE void LL_RTC_WAKEUP_SetAutoReload(RTC_TypeDef *RTCx, uint32_t Value)
{
  MODIFY_REG(RTCx->WUTR, RTC_WUTR_WUT, Value);
}

/**
  * @brief  Get Wakeup auto-reload value
  * @rmtoll WUTR         WUT           LL_RTC_WAKEUP_GetAutoReload
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data=0xFFFF
  */
__STATIC_INLINE uint32_t LL_RTC_WAKEUP_GetAutoReload(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->WUTR, RTC_WUTR_WUT));
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_Backup_Registers Backup_Registers
  * @{
  */

/**
  * @brief  Writes a data in a specified RTC Backup data register.
  * @rmtoll BKPxR        BKP           LL_RTC_BAK_SetRegister
  * @param  RTCx RTC Instance
  * @param  BackupRegister This parameter can be one of the following values:
  *         @arg @ref LL_RTC_BKP_DR0
  *         @arg @ref LL_RTC_BKP_DR1
  *         @arg @ref LL_RTC_BKP_DR2
  *         @arg @ref LL_RTC_BKP_DR3
  *         @arg @ref LL_RTC_BKP_DR4
  *         @arg @ref LL_RTC_BKP_DR5
  *         @arg @ref LL_RTC_BKP_DR6
  *         @arg @ref LL_RTC_BKP_DR7
  *         @arg @ref LL_RTC_BKP_DR8
  *         @arg @ref LL_RTC_BKP_DR9
  *         @arg @ref LL_RTC_BKP_DR10
  *         @arg @ref LL_RTC_BKP_DR11
  *         @arg @ref LL_RTC_BKP_DR12
  *         @arg @ref LL_RTC_BKP_DR13
  *         @arg @ref LL_RTC_BKP_DR14
  *         @arg @ref LL_RTC_BKP_DR15
  *         @arg @ref LL_RTC_BKP_DR16
  *         @arg @ref LL_RTC_BKP_DR17
  *         @arg @ref LL_RTC_BKP_DR18
  *         @arg @ref LL_RTC_BKP_DR19
  * @param  Data Value between Min_Data=0x00 and Max_Data=0xFFFFFFFF
  * @retval None
  */
__STATIC_INLINE void LL_RTC_BAK_SetRegister(RTC_TypeDef *RTCx, uint32_t BackupRegister, uint32_t Data)
{
  register uint32_t tmp = 0U;

  tmp = (uint32_t)(&(RTCx->BKP0R));
  tmp += (BackupRegister * 4U);

  /* Write the specified register */
  *(__IO uint32_t *)tmp = (uint32_t)Data;
}

/**
  * @brief  Reads data from the specified RTC Backup data Register.
  * @rmtoll BKPxR        BKP           LL_RTC_BAK_GetRegister
  * @param  RTCx RTC Instance
  * @param  BackupRegister This parameter can be one of the following values:
  *         @arg @ref LL_RTC_BKP_DR0
  *         @arg @ref LL_RTC_BKP_DR1
  *         @arg @ref LL_RTC_BKP_DR2
  *         @arg @ref LL_RTC_BKP_DR3
  *         @arg @ref LL_RTC_BKP_DR4
  *         @arg @ref LL_RTC_BKP_DR5
  *         @arg @ref LL_RTC_BKP_DR6
  *         @arg @ref LL_RTC_BKP_DR7
  *         @arg @ref LL_RTC_BKP_DR8
  *         @arg @ref LL_RTC_BKP_DR9
  *         @arg @ref LL_RTC_BKP_DR10
  *         @arg @ref LL_RTC_BKP_DR11
  *         @arg @ref LL_RTC_BKP_DR12
  *         @arg @ref LL_RTC_BKP_DR13
  *         @arg @ref LL_RTC_BKP_DR14
  *         @arg @ref LL_RTC_BKP_DR15
  *         @arg @ref LL_RTC_BKP_DR16
  *         @arg @ref LL_RTC_BKP_DR17
  *         @arg @ref LL_RTC_BKP_DR18
  *         @arg @ref LL_RTC_BKP_DR19
  * @retval Value between Min_Data=0x00 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t LL_RTC_BAK_GetRegister(RTC_TypeDef *RTCx, uint32_t BackupRegister)
{
  register uint32_t tmp = 0U;

  tmp = (uint32_t)(&(RTCx->BKP0R));
  tmp += (BackupRegister * 4U);

  /* Read the specified register */
  return (*(__IO uint32_t *)tmp);
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_Calibration Calibration
  * @{
  */

/**
  * @brief  Set Calibration output frequency (1 Hz or 512 Hz)
  * @note Bits are write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           COE           LL_RTC_CAL_SetOutputFreq\n
  *         CR           COSEL         LL_RTC_CAL_SetOutputFreq
  * @param  RTCx RTC Instance
  * @param  Frequency This parameter can be one of the following values:
  *         @arg @ref LL_RTC_CALIB_OUTPUT_NONE
  *         @arg @ref LL_RTC_CALIB_OUTPUT_1HZ
  *         @arg @ref LL_RTC_CALIB_OUTPUT_512HZ
  * @retval None
  */
__STATIC_INLINE void LL_RTC_CAL_SetOutputFreq(RTC_TypeDef *RTCx, uint32_t Frequency)
{
  MODIFY_REG(RTCx->CR, RTC_CR_COE | RTC_CR_COSEL, Frequency);
}

/**
  * @brief  Get Calibration output frequency (1 Hz or 512 Hz)
  * @rmtoll CR           COE           LL_RTC_CAL_GetOutputFreq\n
  *         CR           COSEL         LL_RTC_CAL_GetOutputFreq
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_CALIB_OUTPUT_NONE
  *         @arg @ref LL_RTC_CALIB_OUTPUT_1HZ
  *         @arg @ref LL_RTC_CALIB_OUTPUT_512HZ
  */
__STATIC_INLINE uint32_t LL_RTC_CAL_GetOutputFreq(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CR, RTC_CR_COE | RTC_CR_COSEL));
}

/**
  * @brief  Enable Coarse digital calibration
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @rmtoll CR           DCE           LL_RTC_CAL_EnableCoarseDigital
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_CAL_EnableCoarseDigital(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_DCE);
}

/**
  * @brief  Disable Coarse digital calibration
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @rmtoll CR           DCE           LL_RTC_CAL_DisableCoarseDigital
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_CAL_DisableCoarseDigital(RTC_TypeDef  *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_DCE);
}

/**
  * @brief  Set the coarse digital calibration
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   It can be written in initialization mode only (@ref LL_RTC_EnableInitMode function)
  * @rmtoll CALIBR       DCS           LL_RTC_CAL_ConfigCoarseDigital\n
  *         CALIBR       DC            LL_RTC_CAL_ConfigCoarseDigital
  * @param  RTCx RTC Instance
  * @param  Sign This parameter can be one of the following values:
  *         @arg @ref LL_RTC_CALIB_SIGN_POSITIVE
  *         @arg @ref LL_RTC_CALIB_SIGN_NEGATIVE
  * @param  Value value of coarse calibration expressed in ppm (coded on 5 bits)
  * @note   This Calibration value should be between 0 and 63 when using negative sign with a 2-ppm step.
  * @note   This Calibration value should be between 0 and 126 when using positive sign with a 4-ppm step.
  * @retval None
  */
__STATIC_INLINE void LL_RTC_CAL_ConfigCoarseDigital(RTC_TypeDef* RTCx, uint32_t Sign, uint32_t Value)
{
  MODIFY_REG(RTCx->CALIBR, RTC_CALIBR_DCS | RTC_CALIBR_DC, Sign | Value);
}

/**
  * @brief  Get the coarse digital calibration value
  * @rmtoll CALIBR       DC            LL_RTC_CAL_GetCoarseDigitalValue
  * @param  RTCx RTC Instance
  * @retval value of coarse calibration expressed in ppm (coded on 5 bits)
  */
__STATIC_INLINE uint32_t LL_RTC_CAL_GetCoarseDigitalValue(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CALIBR, RTC_CALIBR_DC));
}

/**
  * @brief  Get the coarse digital calibration sign
  * @rmtoll CALIBR       DCS           LL_RTC_CAL_GetCoarseDigitalSign
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_CALIB_SIGN_POSITIVE
  *         @arg @ref LL_RTC_CALIB_SIGN_NEGATIVE
  */
__STATIC_INLINE uint32_t LL_RTC_CAL_GetCoarseDigitalSign(RTC_TypeDef* RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CALIBR, RTC_CALIBR_DCS));
}

/**
  * @brief  Insert or not One RTCCLK pulse every 2exp11 pulses (frequency increased by 488.5 ppm)
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note Bit can be written only when RECALPF is set to 0 in RTC_ISR
  * @rmtoll CALR         CALP          LL_RTC_CAL_SetPulse
  * @param  RTCx RTC Instance
  * @param  Pulse This parameter can be one of the following values:
  *         @arg @ref LL_RTC_CALIB_INSERTPULSE_NONE
  *         @arg @ref LL_RTC_CALIB_INSERTPULSE_SET
  * @retval None
  */
__STATIC_INLINE void LL_RTC_CAL_SetPulse(RTC_TypeDef *RTCx, uint32_t Pulse)
{
  MODIFY_REG(RTCx->CALR, RTC_CALR_CALP, Pulse);
}

/**
  * @brief  Check if one RTCCLK has been inserted or not every 2exp11 pulses (frequency increased by 488.5 ppm)
  * @rmtoll CALR         CALP          LL_RTC_CAL_IsPulseInserted
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_CAL_IsPulseInserted(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->CALR, RTC_CALR_CALP) == (RTC_CALR_CALP));
}

/**
  * @brief  Set the calibration cycle period
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   Bit can be written only when RECALPF is set to 0 in RTC_ISR
  * @rmtoll CALR         CALW8         LL_RTC_CAL_SetPeriod\n
  *         CALR         CALW16        LL_RTC_CAL_SetPeriod
  * @param  RTCx RTC Instance
  * @param  Period This parameter can be one of the following values:
  *         @arg @ref LL_RTC_CALIB_PERIOD_32SEC
  *         @arg @ref LL_RTC_CALIB_PERIOD_16SEC
  *         @arg @ref LL_RTC_CALIB_PERIOD_8SEC
  * @retval None
  */
__STATIC_INLINE void LL_RTC_CAL_SetPeriod(RTC_TypeDef *RTCx, uint32_t Period)
{
  MODIFY_REG(RTCx->CALR, RTC_CALR_CALW8 | RTC_CALR_CALW16, Period);
}

/**
  * @brief  Get the calibration cycle period
  * @rmtoll CALR         CALW8         LL_RTC_CAL_GetPeriod\n
  *         CALR         CALW16        LL_RTC_CAL_GetPeriod
  * @param  RTCx RTC Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_RTC_CALIB_PERIOD_32SEC
  *         @arg @ref LL_RTC_CALIB_PERIOD_16SEC
  *         @arg @ref LL_RTC_CALIB_PERIOD_8SEC
  */
__STATIC_INLINE uint32_t LL_RTC_CAL_GetPeriod(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CALR, RTC_CALR_CALW8 | RTC_CALR_CALW16));
}

/**
  * @brief  Set Calibration minus
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @note   Bit can be written only when RECALPF is set to 0 in RTC_ISR
  * @rmtoll CALR         CALM          LL_RTC_CAL_SetMinus
  * @param  RTCx RTC Instance
  * @param  CalibMinus Value between Min_Data=0x00 and Max_Data=0x1FF
  * @retval None
  */
__STATIC_INLINE void LL_RTC_CAL_SetMinus(RTC_TypeDef *RTCx, uint32_t CalibMinus)
{
  MODIFY_REG(RTCx->CALR, RTC_CALR_CALM, CalibMinus);
}

/**
  * @brief  Get Calibration minus
  * @rmtoll CALR         CALM          LL_RTC_CAL_GetMinus
  * @param  RTCx RTC Instance
  * @retval Value between Min_Data=0x00 and Max_Data= 0x1FF
  */
__STATIC_INLINE uint32_t LL_RTC_CAL_GetMinus(RTC_TypeDef *RTCx)
{
  return (uint32_t)(READ_BIT(RTCx->CALR, RTC_CALR_CALM));
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_FLAG_Management FLAG_Management
  * @{
  */

/**
  * @brief  Get Recalibration pending Flag
  * @rmtoll ISR          RECALPF       LL_RTC_IsActiveFlag_RECALP
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_RECALP(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_RECALPF) == (RTC_ISR_RECALPF));
}


#if defined(RTC_TAMPER2_SUPPORT)
/**
  * @brief  Get RTC_TAMP2 detection flag
  * @rmtoll ISR          TAMP2F        LL_RTC_IsActiveFlag_TAMP2
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_TAMP2(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_TAMP2F) == (RTC_ISR_TAMP2F));
}
#endif /* RTC_TAMPER2_SUPPORT */

/**
  * @brief  Get RTC_TAMP1 detection flag
  * @rmtoll ISR          TAMP1F        LL_RTC_IsActiveFlag_TAMP1
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_TAMP1(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_TAMP1F) == (RTC_ISR_TAMP1F));
}

/**
  * @brief  Get Time-stamp overflow flag
  * @rmtoll ISR          TSOVF         LL_RTC_IsActiveFlag_TSOV
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_TSOV(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_TSOVF) == (RTC_ISR_TSOVF));
}

/**
  * @brief  Get Time-stamp flag
  * @rmtoll ISR          TSF           LL_RTC_IsActiveFlag_TS
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_TS(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_TSF) == (RTC_ISR_TSF));
}

/**
  * @brief  Get Wakeup timer flag
  * @rmtoll ISR          WUTF          LL_RTC_IsActiveFlag_WUT
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_WUT(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_WUTF) == (RTC_ISR_WUTF));
}

/**
  * @brief  Get Alarm B flag
  * @rmtoll ISR          ALRBF         LL_RTC_IsActiveFlag_ALRB
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_ALRB(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_ALRBF) == (RTC_ISR_ALRBF));
}

/**
  * @brief  Get Alarm A flag
  * @rmtoll ISR          ALRAF         LL_RTC_IsActiveFlag_ALRA
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_ALRA(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_ALRAF) == (RTC_ISR_ALRAF));
}


#if defined(RTC_TAMPER2_SUPPORT)
/**
  * @brief  Clear RTC_TAMP2 detection flag
  * @rmtoll ISR          TAMP2F        LL_RTC_ClearFlag_TAMP2
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ClearFlag_TAMP2(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->ISR, (~((RTC_ISR_TAMP2F | RTC_ISR_INIT) & 0x0000FFFFU) | (RTCx->ISR & RTC_ISR_INIT)));
}
#endif /* RTC_TAMPER2_SUPPORT */

/**
  * @brief  Clear RTC_TAMP1 detection flag
  * @rmtoll ISR          TAMP1F        LL_RTC_ClearFlag_TAMP1
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ClearFlag_TAMP1(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->ISR, (~((RTC_ISR_TAMP1F | RTC_ISR_INIT) & 0x0000FFFFU) | (RTCx->ISR & RTC_ISR_INIT)));
}

/**
  * @brief  Clear Time-stamp overflow flag
  * @rmtoll ISR          TSOVF         LL_RTC_ClearFlag_TSOV
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ClearFlag_TSOV(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->ISR, (~((RTC_ISR_TSOVF | RTC_ISR_INIT) & 0x0000FFFFU) | (RTCx->ISR & RTC_ISR_INIT)));
}

/**
  * @brief  Clear Time-stamp flag
  * @rmtoll ISR          TSF           LL_RTC_ClearFlag_TS
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ClearFlag_TS(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->ISR, (~((RTC_ISR_TSF | RTC_ISR_INIT) & 0x0000FFFFU) | (RTCx->ISR & RTC_ISR_INIT)));
}

/**
  * @brief  Clear Wakeup timer flag
  * @rmtoll ISR          WUTF          LL_RTC_ClearFlag_WUT
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ClearFlag_WUT(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->ISR, (~((RTC_ISR_WUTF | RTC_ISR_INIT) & 0x0000FFFFU) | (RTCx->ISR & RTC_ISR_INIT)));
}

/**
  * @brief  Clear Alarm B flag
  * @rmtoll ISR          ALRBF         LL_RTC_ClearFlag_ALRB
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ClearFlag_ALRB(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->ISR, (~((RTC_ISR_ALRBF | RTC_ISR_INIT) & 0x0000FFFFU) | (RTCx->ISR & RTC_ISR_INIT)));
}

/**
  * @brief  Clear Alarm A flag
  * @rmtoll ISR          ALRAF         LL_RTC_ClearFlag_ALRA
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ClearFlag_ALRA(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->ISR, (~((RTC_ISR_ALRAF | RTC_ISR_INIT) & 0x0000FFFFU) | (RTCx->ISR & RTC_ISR_INIT)));
}

/**
  * @brief  Get Initialization flag
  * @rmtoll ISR          INITF         LL_RTC_IsActiveFlag_INIT
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_INIT(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_INITF) == (RTC_ISR_INITF));
}

/**
  * @brief  Get Registers synchronization flag
  * @rmtoll ISR          RSF           LL_RTC_IsActiveFlag_RS
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_RS(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_RSF) == (RTC_ISR_RSF));
}

/**
  * @brief  Clear Registers synchronization flag
  * @rmtoll ISR          RSF           LL_RTC_ClearFlag_RS
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_ClearFlag_RS(RTC_TypeDef *RTCx)
{
  WRITE_REG(RTCx->ISR, (~((RTC_ISR_RSF | RTC_ISR_INIT) & 0x0000FFFFU) | (RTCx->ISR & RTC_ISR_INIT)));
}

/**
  * @brief  Get Initialization status flag
  * @rmtoll ISR          INITS         LL_RTC_IsActiveFlag_INITS
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_INITS(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_INITS) == (RTC_ISR_INITS));
}

/**
  * @brief  Get Shift operation pending flag
  * @rmtoll ISR          SHPF          LL_RTC_IsActiveFlag_SHP
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_SHP(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_SHPF) == (RTC_ISR_SHPF));
}

/**
  * @brief  Get Wakeup timer write flag
  * @rmtoll ISR          WUTWF         LL_RTC_IsActiveFlag_WUTW
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_WUTW(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_WUTWF) == (RTC_ISR_WUTWF));
}

/**
  * @brief  Get Alarm B write flag
  * @rmtoll ISR          ALRBWF        LL_RTC_IsActiveFlag_ALRBW
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_ALRBW(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_ALRBWF) == (RTC_ISR_ALRBWF));
}

/**
  * @brief  Get Alarm A write flag
  * @rmtoll ISR          ALRAWF        LL_RTC_IsActiveFlag_ALRAW
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsActiveFlag_ALRAW(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->ISR, RTC_ISR_ALRAWF) == (RTC_ISR_ALRAWF));
}

/**
  * @}
  */

/** @defgroup RTC_LL_EF_IT_Management IT_Management
  * @{
  */

/**
  * @brief  Enable Time-stamp interrupt
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           TSIE          LL_RTC_EnableIT_TS
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnableIT_TS(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_TSIE);
}

/**
  * @brief  Disable Time-stamp interrupt
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           TSIE          LL_RTC_DisableIT_TS
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisableIT_TS(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_TSIE);
}

/**
  * @brief  Enable Wakeup timer interrupt
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           WUTIE         LL_RTC_EnableIT_WUT
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnableIT_WUT(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_WUTIE);
}

/**
  * @brief  Disable Wakeup timer interrupt
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           WUTIE         LL_RTC_DisableIT_WUT
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisableIT_WUT(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_WUTIE);
}

/**
  * @brief  Enable Alarm B interrupt
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           ALRBIE        LL_RTC_EnableIT_ALRB
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnableIT_ALRB(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_ALRBIE);
}

/**
  * @brief  Disable Alarm B interrupt
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           ALRBIE        LL_RTC_DisableIT_ALRB
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisableIT_ALRB(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_ALRBIE);
}

/**
  * @brief  Enable Alarm A interrupt
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           ALRAIE        LL_RTC_EnableIT_ALRA
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnableIT_ALRA(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->CR, RTC_CR_ALRAIE);
}

/**
  * @brief  Disable Alarm A interrupt
  * @note   Bit is write-protected. @ref LL_RTC_DisableWriteProtection function should be called before.
  * @rmtoll CR           ALRAIE        LL_RTC_DisableIT_ALRA
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisableIT_ALRA(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->CR, RTC_CR_ALRAIE);
}

/**
  * @brief  Enable all Tamper Interrupt
  * @rmtoll TAFCR       TAMPIE        LL_RTC_EnableIT_TAMP
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_EnableIT_TAMP(RTC_TypeDef *RTCx)
{
  SET_BIT(RTCx->TAFCR, RTC_TAFCR_TAMPIE);
}

/**
  * @brief  Disable all Tamper Interrupt
  * @rmtoll TAFCR       TAMPIE        LL_RTC_DisableIT_TAMP
  * @param  RTCx RTC Instance
  * @retval None
  */
__STATIC_INLINE void LL_RTC_DisableIT_TAMP(RTC_TypeDef *RTCx)
{
  CLEAR_BIT(RTCx->TAFCR, RTC_TAFCR_TAMPIE);
}

/**
  * @brief  Check if  Time-stamp interrupt is enabled or not
  * @rmtoll CR           TSIE          LL_RTC_IsEnabledIT_TS
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsEnabledIT_TS(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->CR, RTC_CR_TSIE) == (RTC_CR_TSIE));
}

/**
  * @brief  Check if  Wakeup timer interrupt is enabled or not
  * @rmtoll CR           WUTIE         LL_RTC_IsEnabledIT_WUT
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsEnabledIT_WUT(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->CR, RTC_CR_WUTIE) == (RTC_CR_WUTIE));
}

/**
  * @brief  Check if  Alarm B interrupt is enabled or not
  * @rmtoll CR           ALRBIE        LL_RTC_IsEnabledIT_ALRB
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsEnabledIT_ALRB(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->CR, RTC_CR_ALRBIE) == (RTC_CR_ALRBIE));
}

/**
  * @brief  Check if  Alarm A interrupt is enabled or not
  * @rmtoll CR           ALRAIE        LL_RTC_IsEnabledIT_ALRA
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsEnabledIT_ALRA(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->CR, RTC_CR_ALRAIE) == (RTC_CR_ALRAIE));
}

/**
  * @brief  Check if all the TAMPER interrupts are enabled or not
  * @rmtoll TAFCR       TAMPIE        LL_RTC_IsEnabledIT_TAMP
  * @param  RTCx RTC Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_RTC_IsEnabledIT_TAMP(RTC_TypeDef *RTCx)
{
  return (READ_BIT(RTCx->TAFCR,
                   RTC_TAFCR_TAMPIE) == (RTC_TAFCR_TAMPIE));
}

/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup RTC_LL_EF_Init Initialization and de-initialization functions
  * @{
  */

ErrorStatus LL_RTC_DeInit(RTC_TypeDef *RTCx);
ErrorStatus LL_RTC_Init(RTC_TypeDef *RTCx, LL_RTC_InitTypeDef *RTC_InitStruct);
void        LL_RTC_StructInit(LL_RTC_InitTypeDef *RTC_InitStruct);
ErrorStatus LL_RTC_TIME_Init(RTC_TypeDef *RTCx, uint32_t RTC_Format, LL_RTC_TimeTypeDef *RTC_TimeStruct);
void        LL_RTC_TIME_StructInit(LL_RTC_TimeTypeDef *RTC_TimeStruct);
ErrorStatus LL_RTC_DATE_Init(RTC_TypeDef *RTCx, uint32_t RTC_Format, LL_RTC_DateTypeDef *RTC_DateStruct);
void        LL_RTC_DATE_StructInit(LL_RTC_DateTypeDef *RTC_DateStruct);
ErrorStatus LL_RTC_ALMA_Init(RTC_TypeDef *RTCx, uint32_t RTC_Format, LL_RTC_AlarmTypeDef *RTC_AlarmStruct);
ErrorStatus LL_RTC_ALMB_Init(RTC_TypeDef *RTCx, uint32_t RTC_Format, LL_RTC_AlarmTypeDef *RTC_AlarmStruct);
void        LL_RTC_ALMA_StructInit(LL_RTC_AlarmTypeDef *RTC_AlarmStruct);
void        LL_RTC_ALMB_StructInit(LL_RTC_AlarmTypeDef *RTC_AlarmStruct);
ErrorStatus LL_RTC_EnterInitMode(RTC_TypeDef *RTCx);
ErrorStatus LL_RTC_ExitInitMode(RTC_TypeDef *RTCx);
ErrorStatus LL_RTC_WaitForSynchro(RTC_TypeDef *RTCx);

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

#endif /* defined(RTC) */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_LL_RTC_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
