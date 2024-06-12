/**
  ******************************************************************************
  * @file    stm32f4xx_ll_rtc.c
  * @author  MCD Application Team
  * @brief   RTC LL module driver.
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
#if defined(USE_FULL_LL_DRIVER)

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_ll_rtc.h"
#include "stm32f4xx_ll_cortex.h"
#ifdef  USE_FULL_ASSERT
#include "stm32_assert.h"
#else
#define assert_param(expr) ((void)0U)
#endif

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined(RTC)

/** @addtogroup RTC_LL
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @addtogroup RTC_LL_Private_Constants
  * @{
  */
/* Default values used for prescaler */
#define RTC_ASYNCH_PRESC_DEFAULT     0x0000007FU
#define RTC_SYNCH_PRESC_DEFAULT      0x000000FFU

/* Values used for timeout */
#define RTC_INITMODE_TIMEOUT         1000U /* 1s when tick set to 1ms */
#define RTC_SYNCHRO_TIMEOUT          1000U /* 1s when tick set to 1ms */
/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @addtogroup RTC_LL_Private_Macros
  * @{
  */

#define IS_LL_RTC_HOURFORMAT(__VALUE__) (((__VALUE__) == LL_RTC_HOURFORMAT_24HOUR) \
                                      || ((__VALUE__) == LL_RTC_HOURFORMAT_AMPM))

#define IS_LL_RTC_ASYNCH_PREDIV(__VALUE__)   ((__VALUE__) <= 0x7FU)

#define IS_LL_RTC_SYNCH_PREDIV(__VALUE__)    ((__VALUE__) <= 0x7FFFU)

#define IS_LL_RTC_FORMAT(__VALUE__) (((__VALUE__) == LL_RTC_FORMAT_BIN) \
                                  || ((__VALUE__) == LL_RTC_FORMAT_BCD))

#define IS_LL_RTC_TIME_FORMAT(__VALUE__) (((__VALUE__) == LL_RTC_TIME_FORMAT_AM_OR_24) \
                                       || ((__VALUE__) == LL_RTC_TIME_FORMAT_PM))

#define IS_LL_RTC_HOUR12(__HOUR__)            (((__HOUR__) > 0U) && ((__HOUR__) <= 12U))
#define IS_LL_RTC_HOUR24(__HOUR__)            ((__HOUR__) <= 23U)
#define IS_LL_RTC_MINUTES(__MINUTES__)        ((__MINUTES__) <= 59U)
#define IS_LL_RTC_SECONDS(__SECONDS__)        ((__SECONDS__) <= 59U)

#define IS_LL_RTC_WEEKDAY(__VALUE__) (((__VALUE__) == LL_RTC_WEEKDAY_MONDAY) \
                                   || ((__VALUE__) == LL_RTC_WEEKDAY_TUESDAY) \
                                   || ((__VALUE__) == LL_RTC_WEEKDAY_WEDNESDAY) \
                                   || ((__VALUE__) == LL_RTC_WEEKDAY_THURSDAY) \
                                   || ((__VALUE__) == LL_RTC_WEEKDAY_FRIDAY) \
                                   || ((__VALUE__) == LL_RTC_WEEKDAY_SATURDAY) \
                                   || ((__VALUE__) == LL_RTC_WEEKDAY_SUNDAY))

#define IS_LL_RTC_DAY(__DAY__)    (((__DAY__) >= 1U) && ((__DAY__) <= 31U))

#define IS_LL_RTC_MONTH(__MONTH__) (((__MONTH__) >= 1U) && ((__MONTH__) <= 12U))

#define IS_LL_RTC_YEAR(__YEAR__) ((__YEAR__) <= 99U)

#define IS_LL_RTC_ALMA_MASK(__VALUE__) (((__VALUE__) == LL_RTC_ALMA_MASK_NONE) \
                                     || ((__VALUE__) == LL_RTC_ALMA_MASK_DATEWEEKDAY) \
                                     || ((__VALUE__) == LL_RTC_ALMA_MASK_HOURS) \
                                     || ((__VALUE__) == LL_RTC_ALMA_MASK_MINUTES) \
                                     || ((__VALUE__) == LL_RTC_ALMA_MASK_SECONDS) \
                                     || ((__VALUE__) == LL_RTC_ALMA_MASK_ALL))

#define IS_LL_RTC_ALMB_MASK(__VALUE__) (((__VALUE__) == LL_RTC_ALMB_MASK_NONE) \
                                     || ((__VALUE__) == LL_RTC_ALMB_MASK_DATEWEEKDAY) \
                                     || ((__VALUE__) == LL_RTC_ALMB_MASK_HOURS) \
                                     || ((__VALUE__) == LL_RTC_ALMB_MASK_MINUTES) \
                                     || ((__VALUE__) == LL_RTC_ALMB_MASK_SECONDS) \
                                     || ((__VALUE__) == LL_RTC_ALMB_MASK_ALL))


#define IS_LL_RTC_ALMA_DATE_WEEKDAY_SEL(__SEL__) (((__SEL__) == LL_RTC_ALMA_DATEWEEKDAYSEL_DATE) || \
                                                  ((__SEL__) == LL_RTC_ALMA_DATEWEEKDAYSEL_WEEKDAY))

#define IS_LL_RTC_ALMB_DATE_WEEKDAY_SEL(__SEL__) (((__SEL__) == LL_RTC_ALMB_DATEWEEKDAYSEL_DATE) || \
                                                  ((__SEL__) == LL_RTC_ALMB_DATEWEEKDAYSEL_WEEKDAY))


/**
  * @}
  */
/* Private function prototypes -----------------------------------------------*/
/* Exported functions --------------------------------------------------------*/
/** @addtogroup RTC_LL_Exported_Functions
  * @{
  */

/** @addtogroup RTC_LL_EF_Init
  * @{
  */

/**
  * @brief  De-Initializes the RTC registers to their default reset values.
  * @note   This function doesn't reset the RTC Clock source and RTC Backup Data
  *         registers.
  * @param  RTCx RTC Instance
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: RTC registers are de-initialized
  *          - ERROR: RTC registers are not de-initialized
  */
ErrorStatus LL_RTC_DeInit(RTC_TypeDef *RTCx)
{
  ErrorStatus status = ERROR;

  /* Check the parameter */
  assert_param(IS_RTC_ALL_INSTANCE(RTCx));

  /* Disable the write protection for RTC registers */
  LL_RTC_DisableWriteProtection(RTCx);

  /* Set Initialization mode */
  if (LL_RTC_EnterInitMode(RTCx) != ERROR)
  {
    /* Reset TR, DR and CR registers */
    LL_RTC_WriteReg(RTCx, TR,       0x00000000U);
#if defined(RTC_WAKEUP_SUPPORT)
    LL_RTC_WriteReg(RTCx, WUTR,     RTC_WUTR_WUT);
#endif /* RTC_WAKEUP_SUPPORT */
    LL_RTC_WriteReg(RTCx, DR  ,     (RTC_DR_WDU_0 | RTC_DR_MU_0 | RTC_DR_DU_0));
    /* Reset All CR bits except CR[2:0] */
#if defined(RTC_WAKEUP_SUPPORT)
    LL_RTC_WriteReg(RTCx, CR, (LL_RTC_ReadReg(RTCx, CR) & RTC_CR_WUCKSEL));
#else
    LL_RTC_WriteReg(RTCx, CR, 0x00000000U);
#endif /* RTC_WAKEUP_SUPPORT */
    LL_RTC_WriteReg(RTCx, PRER,     (RTC_PRER_PREDIV_A | RTC_SYNCH_PRESC_DEFAULT));
    LL_RTC_WriteReg(RTCx, ALRMAR,   0x00000000U);
    LL_RTC_WriteReg(RTCx, ALRMBR,   0x00000000U);
    LL_RTC_WriteReg(RTCx, SHIFTR,   0x00000000U);
    LL_RTC_WriteReg(RTCx, CALR,     0x00000000U);
    LL_RTC_WriteReg(RTCx, ALRMASSR, 0x00000000U);
    LL_RTC_WriteReg(RTCx, ALRMBSSR, 0x00000000U);

    /* Reset ISR register and exit initialization mode */
    LL_RTC_WriteReg(RTCx, ISR,      0x00000000U);

    /* Reset Tamper and alternate functions configuration register */
    LL_RTC_WriteReg(RTCx, TAFCR, 0x00000000U);

    /* Wait till the RTC RSF flag is set */
    status = LL_RTC_WaitForSynchro(RTCx);
  }

  /* Enable the write protection for RTC registers */
  LL_RTC_EnableWriteProtection(RTCx);

  return status;
}

/**
  * @brief  Initializes the RTC registers according to the specified parameters
  *         in RTC_InitStruct.
  * @param  RTCx RTC Instance
  * @param  RTC_InitStruct pointer to a @ref LL_RTC_InitTypeDef structure that contains
  *         the configuration information for the RTC peripheral.
  * @note   The RTC Prescaler register is write protected and can be written in
  *         initialization mode only.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: RTC registers are initialized
  *          - ERROR: RTC registers are not initialized
  */
ErrorStatus LL_RTC_Init(RTC_TypeDef *RTCx, LL_RTC_InitTypeDef *RTC_InitStruct)
{
  ErrorStatus status = ERROR;

  /* Check the parameters */
  assert_param(IS_RTC_ALL_INSTANCE(RTCx));
  assert_param(IS_LL_RTC_HOURFORMAT(RTC_InitStruct->HourFormat));
  assert_param(IS_LL_RTC_ASYNCH_PREDIV(RTC_InitStruct->AsynchPrescaler));
  assert_param(IS_LL_RTC_SYNCH_PREDIV(RTC_InitStruct->SynchPrescaler));

  /* Disable the write protection for RTC registers */
  LL_RTC_DisableWriteProtection(RTCx);

  /* Set Initialization mode */
  if (LL_RTC_EnterInitMode(RTCx) != ERROR)
  {
    /* Set Hour Format */
    LL_RTC_SetHourFormat(RTCx, RTC_InitStruct->HourFormat);

    /* Configure Synchronous and Asynchronous prescaler factor */
    LL_RTC_SetSynchPrescaler(RTCx, RTC_InitStruct->SynchPrescaler);
    LL_RTC_SetAsynchPrescaler(RTCx, RTC_InitStruct->AsynchPrescaler);

    /* Exit Initialization mode */
    LL_RTC_DisableInitMode(RTCx);

    status = SUCCESS;
  }
  /* Enable the write protection for RTC registers */
  LL_RTC_EnableWriteProtection(RTCx);

  return status;
}

/**
  * @brief  Set each @ref LL_RTC_InitTypeDef field to default value.
  * @param  RTC_InitStruct pointer to a @ref LL_RTC_InitTypeDef structure which will be initialized.
  * @retval None
  */
void LL_RTC_StructInit(LL_RTC_InitTypeDef *RTC_InitStruct)
{
  /* Set RTC_InitStruct fields to default values */
  RTC_InitStruct->HourFormat      = LL_RTC_HOURFORMAT_24HOUR;
  RTC_InitStruct->AsynchPrescaler = RTC_ASYNCH_PRESC_DEFAULT;
  RTC_InitStruct->SynchPrescaler  = RTC_SYNCH_PRESC_DEFAULT;
}

/**
  * @brief  Set the RTC current time.
  * @param  RTCx RTC Instance
  * @param  RTC_Format This parameter can be one of the following values:
  *         @arg @ref LL_RTC_FORMAT_BIN
  *         @arg @ref LL_RTC_FORMAT_BCD
  * @param  RTC_TimeStruct pointer to a RTC_TimeTypeDef structure that contains
  *                        the time configuration information for the RTC.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: RTC Time register is configured
  *          - ERROR: RTC Time register is not configured
  */
ErrorStatus LL_RTC_TIME_Init(RTC_TypeDef *RTCx, uint32_t RTC_Format, LL_RTC_TimeTypeDef *RTC_TimeStruct)
{
  ErrorStatus status = ERROR;

  /* Check the parameters */
  assert_param(IS_RTC_ALL_INSTANCE(RTCx));
  assert_param(IS_LL_RTC_FORMAT(RTC_Format));

  if (RTC_Format == LL_RTC_FORMAT_BIN)
  {
    if (LL_RTC_GetHourFormat(RTCx) != LL_RTC_HOURFORMAT_24HOUR)
    {
      assert_param(IS_LL_RTC_HOUR12(RTC_TimeStruct->Hours));
      assert_param(IS_LL_RTC_TIME_FORMAT(RTC_TimeStruct->TimeFormat));
    }
    else
    {
      RTC_TimeStruct->TimeFormat = 0x00U;
      assert_param(IS_LL_RTC_HOUR24(RTC_TimeStruct->Hours));
    }
    assert_param(IS_LL_RTC_MINUTES(RTC_TimeStruct->Minutes));
    assert_param(IS_LL_RTC_SECONDS(RTC_TimeStruct->Seconds));
  }
  else
  {
    if (LL_RTC_GetHourFormat(RTCx) != LL_RTC_HOURFORMAT_24HOUR)
    {
      assert_param(IS_LL_RTC_HOUR12(__LL_RTC_CONVERT_BCD2BIN(RTC_TimeStruct->Hours)));
      assert_param(IS_LL_RTC_TIME_FORMAT(RTC_TimeStruct->TimeFormat));
    }
    else
    {
      RTC_TimeStruct->TimeFormat = 0x00U;
      assert_param(IS_LL_RTC_HOUR24(__LL_RTC_CONVERT_BCD2BIN(RTC_TimeStruct->Hours)));
    }
    assert_param(IS_LL_RTC_MINUTES(__LL_RTC_CONVERT_BCD2BIN(RTC_TimeStruct->Minutes)));
    assert_param(IS_LL_RTC_SECONDS(__LL_RTC_CONVERT_BCD2BIN(RTC_TimeStruct->Seconds)));
  }

  /* Disable the write protection for RTC registers */
  LL_RTC_DisableWriteProtection(RTCx);

  /* Set Initialization mode */
  if (LL_RTC_EnterInitMode(RTCx) != ERROR)
  {
    /* Check the input parameters format */
    if (RTC_Format != LL_RTC_FORMAT_BIN)
    {
      LL_RTC_TIME_Config(RTCx, RTC_TimeStruct->TimeFormat, RTC_TimeStruct->Hours,
                         RTC_TimeStruct->Minutes, RTC_TimeStruct->Seconds);
    }
    else
    {
      LL_RTC_TIME_Config(RTCx, RTC_TimeStruct->TimeFormat, __LL_RTC_CONVERT_BIN2BCD(RTC_TimeStruct->Hours),
                         __LL_RTC_CONVERT_BIN2BCD(RTC_TimeStruct->Minutes),
                         __LL_RTC_CONVERT_BIN2BCD(RTC_TimeStruct->Seconds));
    }

    /* Exit Initialization mode */
    LL_RTC_DisableInitMode(RTCx);

    /* If  RTC_CR_BYPSHAD bit = 0, wait for synchro else this check is not needed */
    if (LL_RTC_IsShadowRegBypassEnabled(RTCx) == 0U)
    {
      status = LL_RTC_WaitForSynchro(RTCx);
    }
    else
    {
      status = SUCCESS;
    }
  }
  /* Enable the write protection for RTC registers */
  LL_RTC_EnableWriteProtection(RTCx);

  return status;
}

/**
  * @brief  Set each @ref LL_RTC_TimeTypeDef field to default value (Time = 00h:00min:00sec).
  * @param  RTC_TimeStruct pointer to a @ref LL_RTC_TimeTypeDef structure which will be initialized.
  * @retval None
  */
void LL_RTC_TIME_StructInit(LL_RTC_TimeTypeDef *RTC_TimeStruct)
{
  /* Time = 00h:00min:00sec */
  RTC_TimeStruct->TimeFormat = LL_RTC_TIME_FORMAT_AM_OR_24;
  RTC_TimeStruct->Hours      = 0U;
  RTC_TimeStruct->Minutes    = 0U;
  RTC_TimeStruct->Seconds    = 0U;
}

/**
  * @brief  Set the RTC current date.
  * @param  RTCx RTC Instance
  * @param  RTC_Format This parameter can be one of the following values:
  *         @arg @ref LL_RTC_FORMAT_BIN
  *         @arg @ref LL_RTC_FORMAT_BCD
  * @param  RTC_DateStruct pointer to a RTC_DateTypeDef structure that contains
  *                         the date configuration information for the RTC.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: RTC Day register is configured
  *          - ERROR: RTC Day register is not configured
  */
ErrorStatus LL_RTC_DATE_Init(RTC_TypeDef *RTCx, uint32_t RTC_Format, LL_RTC_DateTypeDef *RTC_DateStruct)
{
  ErrorStatus status = ERROR;

  /* Check the parameters */
  assert_param(IS_RTC_ALL_INSTANCE(RTCx));
  assert_param(IS_LL_RTC_FORMAT(RTC_Format));

  if ((RTC_Format == LL_RTC_FORMAT_BIN) && ((RTC_DateStruct->Month & 0x10U) == 0x10U))
  {
    RTC_DateStruct->Month = (RTC_DateStruct->Month & (uint32_t)~(0x10U)) + 0x0AU;
  }
  if (RTC_Format == LL_RTC_FORMAT_BIN)
  {
    assert_param(IS_LL_RTC_YEAR(RTC_DateStruct->Year));
    assert_param(IS_LL_RTC_MONTH(RTC_DateStruct->Month));
    assert_param(IS_LL_RTC_DAY(RTC_DateStruct->Day));
  }
  else
  {
    assert_param(IS_LL_RTC_YEAR(__LL_RTC_CONVERT_BCD2BIN(RTC_DateStruct->Year)));
    assert_param(IS_LL_RTC_MONTH(__LL_RTC_CONVERT_BCD2BIN(RTC_DateStruct->Month)));
    assert_param(IS_LL_RTC_DAY(__LL_RTC_CONVERT_BCD2BIN(RTC_DateStruct->Day)));
  }
  assert_param(IS_LL_RTC_WEEKDAY(RTC_DateStruct->WeekDay));

  /* Disable the write protection for RTC registers */
  LL_RTC_DisableWriteProtection(RTCx);

  /* Set Initialization mode */
  if (LL_RTC_EnterInitMode(RTCx) != ERROR)
  {
    /* Check the input parameters format */
    if (RTC_Format != LL_RTC_FORMAT_BIN)
    {
      LL_RTC_DATE_Config(RTCx, RTC_DateStruct->WeekDay, RTC_DateStruct->Day, RTC_DateStruct->Month, RTC_DateStruct->Year);
    }
    else
    {
      LL_RTC_DATE_Config(RTCx, RTC_DateStruct->WeekDay, __LL_RTC_CONVERT_BIN2BCD(RTC_DateStruct->Day),
                         __LL_RTC_CONVERT_BIN2BCD(RTC_DateStruct->Month), __LL_RTC_CONVERT_BIN2BCD(RTC_DateStruct->Year));
    }

    /* Exit Initialization mode */
    LL_RTC_DisableInitMode(RTCx);

    /* If  RTC_CR_BYPSHAD bit = 0, wait for synchro else this check is not needed */
    if (LL_RTC_IsShadowRegBypassEnabled(RTCx) == 0U)
    {
      status = LL_RTC_WaitForSynchro(RTCx);
    }
    else
    {
      status = SUCCESS;
    }
  }
  /* Enable the write protection for RTC registers */
  LL_RTC_EnableWriteProtection(RTCx);

  return status;
}

/**
  * @brief  Set each @ref LL_RTC_DateTypeDef field to default value (date = Monday, January 01 xx00)
  * @param  RTC_DateStruct pointer to a @ref LL_RTC_DateTypeDef structure which will be initialized.
  * @retval None
  */
void LL_RTC_DATE_StructInit(LL_RTC_DateTypeDef *RTC_DateStruct)
{
  /* Monday, January 01 xx00 */
  RTC_DateStruct->WeekDay = LL_RTC_WEEKDAY_MONDAY;
  RTC_DateStruct->Day     = 1U;
  RTC_DateStruct->Month   = LL_RTC_MONTH_JANUARY;
  RTC_DateStruct->Year    = 0U;
}

/**
  * @brief  Set the RTC Alarm A.
  * @note   The Alarm register can only be written when the corresponding Alarm
  *         is disabled (Use @ref LL_RTC_ALMA_Disable function).
  * @param  RTCx RTC Instance
  * @param  RTC_Format This parameter can be one of the following values:
  *         @arg @ref LL_RTC_FORMAT_BIN
  *         @arg @ref LL_RTC_FORMAT_BCD
  * @param  RTC_AlarmStruct pointer to a @ref LL_RTC_AlarmTypeDef structure that
  *                         contains the alarm configuration parameters.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: ALARMA registers are configured
  *          - ERROR: ALARMA registers are not configured
  */
ErrorStatus LL_RTC_ALMA_Init(RTC_TypeDef *RTCx, uint32_t RTC_Format, LL_RTC_AlarmTypeDef *RTC_AlarmStruct)
{
  /* Check the parameters */
  assert_param(IS_RTC_ALL_INSTANCE(RTCx));
  assert_param(IS_LL_RTC_FORMAT(RTC_Format));
  assert_param(IS_LL_RTC_ALMA_MASK(RTC_AlarmStruct->AlarmMask));
  assert_param(IS_LL_RTC_ALMA_DATE_WEEKDAY_SEL(RTC_AlarmStruct->AlarmDateWeekDaySel));

  if (RTC_Format == LL_RTC_FORMAT_BIN)
  {
    if (LL_RTC_GetHourFormat(RTCx) != LL_RTC_HOURFORMAT_24HOUR)
    {
      assert_param(IS_LL_RTC_HOUR12(RTC_AlarmStruct->AlarmTime.Hours));
      assert_param(IS_LL_RTC_TIME_FORMAT(RTC_AlarmStruct->AlarmTime.TimeFormat));
    }
    else
    {
      RTC_AlarmStruct->AlarmTime.TimeFormat = 0x00U;
      assert_param(IS_LL_RTC_HOUR24(RTC_AlarmStruct->AlarmTime.Hours));
    }
    assert_param(IS_LL_RTC_MINUTES(RTC_AlarmStruct->AlarmTime.Minutes));
    assert_param(IS_LL_RTC_SECONDS(RTC_AlarmStruct->AlarmTime.Seconds));

    if (RTC_AlarmStruct->AlarmDateWeekDaySel == LL_RTC_ALMA_DATEWEEKDAYSEL_DATE)
    {
      assert_param(IS_LL_RTC_DAY(RTC_AlarmStruct->AlarmDateWeekDay));
    }
    else
    {
      assert_param(IS_LL_RTC_WEEKDAY(RTC_AlarmStruct->AlarmDateWeekDay));
    }
  }
  else
  {
    if (LL_RTC_GetHourFormat(RTCx) != LL_RTC_HOURFORMAT_24HOUR)
    {
      assert_param(IS_LL_RTC_HOUR12(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmTime.Hours)));
      assert_param(IS_LL_RTC_TIME_FORMAT(RTC_AlarmStruct->AlarmTime.TimeFormat));
    }
    else
    {
      RTC_AlarmStruct->AlarmTime.TimeFormat = 0x00U;
      assert_param(IS_LL_RTC_HOUR24(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmTime.Hours)));
    }

    assert_param(IS_LL_RTC_MINUTES(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmTime.Minutes)));
    assert_param(IS_LL_RTC_SECONDS(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmTime.Seconds)));

    if (RTC_AlarmStruct->AlarmDateWeekDaySel == LL_RTC_ALMA_DATEWEEKDAYSEL_DATE)
    {
      assert_param(IS_LL_RTC_DAY(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmDateWeekDay)));
    }
    else
    {
      assert_param(IS_LL_RTC_WEEKDAY(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmDateWeekDay)));
    }
  }

  /* Disable the write protection for RTC registers */
  LL_RTC_DisableWriteProtection(RTCx);

  /* Select weekday selection */
  if (RTC_AlarmStruct->AlarmDateWeekDaySel == LL_RTC_ALMA_DATEWEEKDAYSEL_DATE)
  {
    /* Set the date for ALARM */
    LL_RTC_ALMA_DisableWeekday(RTCx);
    if (RTC_Format != LL_RTC_FORMAT_BIN)
    {
      LL_RTC_ALMA_SetDay(RTCx, RTC_AlarmStruct->AlarmDateWeekDay);
    }
    else
    {
      LL_RTC_ALMA_SetDay(RTCx, __LL_RTC_CONVERT_BIN2BCD(RTC_AlarmStruct->AlarmDateWeekDay));
    }
  }
  else
  {
    /* Set the week day for ALARM */
    LL_RTC_ALMA_EnableWeekday(RTCx);
    LL_RTC_ALMA_SetWeekDay(RTCx, RTC_AlarmStruct->AlarmDateWeekDay);
  }

  /* Configure the Alarm register */
  if (RTC_Format != LL_RTC_FORMAT_BIN)
  {
    LL_RTC_ALMA_ConfigTime(RTCx, RTC_AlarmStruct->AlarmTime.TimeFormat, RTC_AlarmStruct->AlarmTime.Hours,
                           RTC_AlarmStruct->AlarmTime.Minutes, RTC_AlarmStruct->AlarmTime.Seconds);
  }
  else
  {
    LL_RTC_ALMA_ConfigTime(RTCx, RTC_AlarmStruct->AlarmTime.TimeFormat,
                           __LL_RTC_CONVERT_BIN2BCD(RTC_AlarmStruct->AlarmTime.Hours),
                           __LL_RTC_CONVERT_BIN2BCD(RTC_AlarmStruct->AlarmTime.Minutes),
                           __LL_RTC_CONVERT_BIN2BCD(RTC_AlarmStruct->AlarmTime.Seconds));
  }
  /* Set ALARM mask */
  LL_RTC_ALMA_SetMask(RTCx, RTC_AlarmStruct->AlarmMask);

  /* Enable the write protection for RTC registers */
  LL_RTC_EnableWriteProtection(RTCx);

  return SUCCESS;
}

/**
  * @brief  Set the RTC Alarm B.
  * @note   The Alarm register can only be written when the corresponding Alarm
  *         is disabled (@ref LL_RTC_ALMB_Disable function).
  * @param  RTCx RTC Instance
  * @param  RTC_Format This parameter can be one of the following values:
  *         @arg @ref LL_RTC_FORMAT_BIN
  *         @arg @ref LL_RTC_FORMAT_BCD
  * @param  RTC_AlarmStruct pointer to a @ref LL_RTC_AlarmTypeDef structure that
  *                         contains the alarm configuration parameters.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: ALARMB registers are configured
  *          - ERROR: ALARMB registers are not configured
  */
ErrorStatus LL_RTC_ALMB_Init(RTC_TypeDef *RTCx, uint32_t RTC_Format, LL_RTC_AlarmTypeDef *RTC_AlarmStruct)
{
  /* Check the parameters */
  assert_param(IS_RTC_ALL_INSTANCE(RTCx));
  assert_param(IS_LL_RTC_FORMAT(RTC_Format));
  assert_param(IS_LL_RTC_ALMB_MASK(RTC_AlarmStruct->AlarmMask));
  assert_param(IS_LL_RTC_ALMB_DATE_WEEKDAY_SEL(RTC_AlarmStruct->AlarmDateWeekDaySel));

  if (RTC_Format == LL_RTC_FORMAT_BIN)
  {
    if (LL_RTC_GetHourFormat(RTCx) != LL_RTC_HOURFORMAT_24HOUR)
    {
      assert_param(IS_LL_RTC_HOUR12(RTC_AlarmStruct->AlarmTime.Hours));
      assert_param(IS_LL_RTC_TIME_FORMAT(RTC_AlarmStruct->AlarmTime.TimeFormat));
    }
    else
    {
      RTC_AlarmStruct->AlarmTime.TimeFormat = 0x00U;
      assert_param(IS_LL_RTC_HOUR24(RTC_AlarmStruct->AlarmTime.Hours));
    }
    assert_param(IS_LL_RTC_MINUTES(RTC_AlarmStruct->AlarmTime.Minutes));
    assert_param(IS_LL_RTC_SECONDS(RTC_AlarmStruct->AlarmTime.Seconds));

    if (RTC_AlarmStruct->AlarmDateWeekDaySel == LL_RTC_ALMB_DATEWEEKDAYSEL_DATE)
    {
      assert_param(IS_LL_RTC_DAY(RTC_AlarmStruct->AlarmDateWeekDay));
    }
    else
    {
      assert_param(IS_LL_RTC_WEEKDAY(RTC_AlarmStruct->AlarmDateWeekDay));
    }
  }
  else
  {
    if (LL_RTC_GetHourFormat(RTCx) != LL_RTC_HOURFORMAT_24HOUR)
    {
      assert_param(IS_LL_RTC_HOUR12(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmTime.Hours)));
      assert_param(IS_LL_RTC_TIME_FORMAT(RTC_AlarmStruct->AlarmTime.TimeFormat));
    }
    else
    {
      RTC_AlarmStruct->AlarmTime.TimeFormat = 0x00U;
      assert_param(IS_LL_RTC_HOUR24(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmTime.Hours)));
    }

    assert_param(IS_LL_RTC_MINUTES(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmTime.Minutes)));
    assert_param(IS_LL_RTC_SECONDS(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmTime.Seconds)));

    if (RTC_AlarmStruct->AlarmDateWeekDaySel == LL_RTC_ALMB_DATEWEEKDAYSEL_DATE)
    {
      assert_param(IS_LL_RTC_DAY(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmDateWeekDay)));
    }
    else
    {
      assert_param(IS_LL_RTC_WEEKDAY(__LL_RTC_CONVERT_BCD2BIN(RTC_AlarmStruct->AlarmDateWeekDay)));
    }
  }

  /* Disable the write protection for RTC registers */
  LL_RTC_DisableWriteProtection(RTCx);

  /* Select weekday selection */
  if (RTC_AlarmStruct->AlarmDateWeekDaySel == LL_RTC_ALMB_DATEWEEKDAYSEL_DATE)
  {
    /* Set the date for ALARM */
    LL_RTC_ALMB_DisableWeekday(RTCx);
    if (RTC_Format != LL_RTC_FORMAT_BIN)
    {
      LL_RTC_ALMB_SetDay(RTCx, RTC_AlarmStruct->AlarmDateWeekDay);
    }
    else
    {
      LL_RTC_ALMB_SetDay(RTCx, __LL_RTC_CONVERT_BIN2BCD(RTC_AlarmStruct->AlarmDateWeekDay));
    }
  }
  else
  {
    /* Set the week day for ALARM */
    LL_RTC_ALMB_EnableWeekday(RTCx);
    LL_RTC_ALMB_SetWeekDay(RTCx, RTC_AlarmStruct->AlarmDateWeekDay);
  }

  /* Configure the Alarm register */
  if (RTC_Format != LL_RTC_FORMAT_BIN)
  {
    LL_RTC_ALMB_ConfigTime(RTCx, RTC_AlarmStruct->AlarmTime.TimeFormat, RTC_AlarmStruct->AlarmTime.Hours,
                           RTC_AlarmStruct->AlarmTime.Minutes, RTC_AlarmStruct->AlarmTime.Seconds);
  }
  else
  {
    LL_RTC_ALMB_ConfigTime(RTCx, RTC_AlarmStruct->AlarmTime.TimeFormat,
                           __LL_RTC_CONVERT_BIN2BCD(RTC_AlarmStruct->AlarmTime.Hours),
                           __LL_RTC_CONVERT_BIN2BCD(RTC_AlarmStruct->AlarmTime.Minutes),
                           __LL_RTC_CONVERT_BIN2BCD(RTC_AlarmStruct->AlarmTime.Seconds));
  }
  /* Set ALARM mask */
  LL_RTC_ALMB_SetMask(RTCx, RTC_AlarmStruct->AlarmMask);

  /* Enable the write protection for RTC registers */
  LL_RTC_EnableWriteProtection(RTCx);

  return SUCCESS;
}

/**
  * @brief  Set each @ref LL_RTC_AlarmTypeDef of ALARMA field to default value (Time = 00h:00mn:00sec /
  *         Day = 1st day of the month/Mask = all fields are masked).
  * @param  RTC_AlarmStruct pointer to a @ref LL_RTC_AlarmTypeDef structure which will be initialized.
  * @retval None
  */
void LL_RTC_ALMA_StructInit(LL_RTC_AlarmTypeDef *RTC_AlarmStruct)
{
  /* Alarm Time Settings : Time = 00h:00mn:00sec */
  RTC_AlarmStruct->AlarmTime.TimeFormat = LL_RTC_ALMA_TIME_FORMAT_AM;
  RTC_AlarmStruct->AlarmTime.Hours      = 0U;
  RTC_AlarmStruct->AlarmTime.Minutes    = 0U;
  RTC_AlarmStruct->AlarmTime.Seconds    = 0U;

  /* Alarm Day Settings : Day = 1st day of the month */
  RTC_AlarmStruct->AlarmDateWeekDaySel = LL_RTC_ALMA_DATEWEEKDAYSEL_DATE;
  RTC_AlarmStruct->AlarmDateWeekDay    = 1U;

  /* Alarm Masks Settings : Mask =  all fields are not masked */
  RTC_AlarmStruct->AlarmMask           = LL_RTC_ALMA_MASK_NONE;
}

/**
  * @brief  Set each @ref LL_RTC_AlarmTypeDef of ALARMA field to default value (Time = 00h:00mn:00sec /
  *         Day = 1st day of the month/Mask = all fields are masked).
  * @param  RTC_AlarmStruct pointer to a @ref LL_RTC_AlarmTypeDef structure which will be initialized.
  * @retval None
  */
void LL_RTC_ALMB_StructInit(LL_RTC_AlarmTypeDef *RTC_AlarmStruct)
{
  /* Alarm Time Settings : Time = 00h:00mn:00sec */
  RTC_AlarmStruct->AlarmTime.TimeFormat = LL_RTC_ALMB_TIME_FORMAT_AM;
  RTC_AlarmStruct->AlarmTime.Hours      = 0U;
  RTC_AlarmStruct->AlarmTime.Minutes    = 0U;
  RTC_AlarmStruct->AlarmTime.Seconds    = 0U;

  /* Alarm Day Settings : Day = 1st day of the month */
  RTC_AlarmStruct->AlarmDateWeekDaySel = LL_RTC_ALMB_DATEWEEKDAYSEL_DATE;
  RTC_AlarmStruct->AlarmDateWeekDay    = 1U;

  /* Alarm Masks Settings : Mask =  all fields are not masked */
  RTC_AlarmStruct->AlarmMask           = LL_RTC_ALMB_MASK_NONE;
}

/**
  * @brief  Enters the RTC Initialization mode.
  * @note   The RTC Initialization mode is write protected, use the
  *         @ref LL_RTC_DisableWriteProtection before calling this function.
  * @param  RTCx RTC Instance
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: RTC is in Init mode
  *          - ERROR: RTC is not in Init mode
  */
ErrorStatus LL_RTC_EnterInitMode(RTC_TypeDef *RTCx)
{
  __IO uint32_t timeout = RTC_INITMODE_TIMEOUT;
  ErrorStatus status = SUCCESS;
  uint32_t tmp = 0U;

  /* Check the parameter */
  assert_param(IS_RTC_ALL_INSTANCE(RTCx));

  /* Check if the Initialization mode is set */
  if (LL_RTC_IsActiveFlag_INIT(RTCx) == 0U)
  {
    /* Set the Initialization mode */
    LL_RTC_EnableInitMode(RTCx);

    /* Wait till RTC is in INIT state and if Time out is reached exit */
    tmp = LL_RTC_IsActiveFlag_INIT(RTCx);
    while ((timeout != 0U) && (tmp != 1U))
    {
      if (LL_SYSTICK_IsActiveCounterFlag() == 1U)
      {
        timeout --;
      }
      tmp = LL_RTC_IsActiveFlag_INIT(RTCx);
      if (timeout == 0U)
      {
        status = ERROR;
      }
    }
  }
  return status;
}

/**
  * @brief  Exit the RTC Initialization mode.
  * @note   When the initialization sequence is complete, the calendar restarts
  *         counting after 4 RTCCLK cycles.
  * @note   The RTC Initialization mode is write protected, use the
  *         @ref LL_RTC_DisableWriteProtection before calling this function.
  * @param  RTCx RTC Instance
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: RTC exited from in Init mode
  *          - ERROR: Not applicable
  */
ErrorStatus LL_RTC_ExitInitMode(RTC_TypeDef *RTCx)
{
  /* Check the parameter */
  assert_param(IS_RTC_ALL_INSTANCE(RTCx));

  /* Disable initialization mode */
  LL_RTC_DisableInitMode(RTCx);

  return SUCCESS;
}

/**
  * @brief  Waits until the RTC Time and Day registers (RTC_TR and RTC_DR) are
  *         synchronized with RTC APB clock.
  * @note   The RTC Resynchronization mode is write protected, use the
  *         @ref LL_RTC_DisableWriteProtection before calling this function.
  * @note   To read the calendar through the shadow registers after Calendar
  *         initialization, calendar update or after wakeup from low power modes
  *         the software must first clear the RSF flag.
  *         The software must then wait until it is set again before reading
  *         the calendar, which means that the calendar registers have been
  *         correctly copied into the RTC_TR and RTC_DR shadow registers.
  * @param  RTCx RTC Instance
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: RTC registers are synchronised
  *          - ERROR: RTC registers are not synchronised
  */
ErrorStatus LL_RTC_WaitForSynchro(RTC_TypeDef *RTCx)
{
  __IO uint32_t timeout = RTC_SYNCHRO_TIMEOUT;
  ErrorStatus status = SUCCESS;
  uint32_t tmp = 0U;

  /* Check the parameter */
  assert_param(IS_RTC_ALL_INSTANCE(RTCx));

  /* Clear RSF flag */
  LL_RTC_ClearFlag_RS(RTCx);

  /* Wait the registers to be synchronised */
  tmp = LL_RTC_IsActiveFlag_RS(RTCx);
  while ((timeout != 0U) && (tmp != 0U))
  {
    if (LL_SYSTICK_IsActiveCounterFlag() == 1U)
    {
      timeout--;
    }
    tmp = LL_RTC_IsActiveFlag_RS(RTCx);
    if (timeout == 0U)
    {
      status = ERROR;
    }
  }

  if (status != ERROR)
  {
    timeout = RTC_SYNCHRO_TIMEOUT;
    tmp = LL_RTC_IsActiveFlag_RS(RTCx);
    while ((timeout != 0U) && (tmp != 1U))
    {
      if (LL_SYSTICK_IsActiveCounterFlag() == 1U)
      {
        timeout--;
      }
      tmp = LL_RTC_IsActiveFlag_RS(RTCx);
      if (timeout == 0U)
      {
        status = ERROR;
      }
    }
  }

  return (status);
}

/**
  * @}
  */

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

#endif /* USE_FULL_LL_DRIVER */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
