/**
  ******************************************************************************
  * @file    stm32f4xx_ll_rcc.c
  * @author  MCD Application Team
  * @brief   RCC LL module driver.
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
#include "stm32f4xx_ll_rcc.h"
#ifdef  USE_FULL_ASSERT
  #include "stm32_assert.h"
#else
  #define assert_param(expr) ((void)0U)
#endif
/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined(RCC)

/** @addtogroup RCC_LL
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/** @addtogroup RCC_LL_Private_Macros
  * @{
  */
#if defined(FMPI2C1)
#define IS_LL_RCC_FMPI2C_CLKSOURCE(__VALUE__)     ((__VALUE__) == LL_RCC_FMPI2C1_CLKSOURCE)
#endif /* FMPI2C1 */

#if defined(LPTIM1)
#define IS_LL_RCC_LPTIM_CLKSOURCE(__VALUE__)  (((__VALUE__) == LL_RCC_LPTIM1_CLKSOURCE))
#endif /* LPTIM1 */

#if defined(SAI1)
#if defined(RCC_DCKCFGR_SAI1SRC)
#define IS_LL_RCC_SAI_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_SAI1_CLKSOURCE) \
                                            || ((__VALUE__) == LL_RCC_SAI2_CLKSOURCE))
#elif defined(RCC_DCKCFGR_SAI1ASRC)
#define IS_LL_RCC_SAI_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_SAI1_A_CLKSOURCE) \
                                            || ((__VALUE__) == LL_RCC_SAI1_B_CLKSOURCE))
#endif /* RCC_DCKCFGR_SAI1SRC */
#endif /* SAI1 */

#if defined(SDIO)
#define IS_LL_RCC_SDIO_CLKSOURCE(__VALUE__)  (((__VALUE__) == LL_RCC_SDIO_CLKSOURCE))
#endif /* SDIO */

#if defined(RNG)
#define IS_LL_RCC_RNG_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_RNG_CLKSOURCE))
#endif /* RNG */

#if defined(USB_OTG_FS) || defined(USB_OTG_HS)
#define IS_LL_RCC_USB_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_USB_CLKSOURCE))
#endif /* USB_OTG_FS || USB_OTG_HS */

#if defined(DFSDM2_Channel0)
#define IS_LL_RCC_DFSDM_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_DFSDM1_CLKSOURCE))

#define IS_LL_RCC_DFSDM_AUDIO_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_DFSDM1_AUDIO_CLKSOURCE) \
                                                    || ((__VALUE__) == LL_RCC_DFSDM2_AUDIO_CLKSOURCE))
#elif defined(DFSDM1_Channel0)
#define IS_LL_RCC_DFSDM_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_DFSDM1_CLKSOURCE))

#define IS_LL_RCC_DFSDM_AUDIO_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_DFSDM1_AUDIO_CLKSOURCE))
#endif /* DFSDM2_Channel0 */

#if defined(RCC_DCKCFGR_I2S2SRC)
#define IS_LL_RCC_I2S_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_I2S1_CLKSOURCE) \
                                            || ((__VALUE__) == LL_RCC_I2S2_CLKSOURCE))
#else
#define IS_LL_RCC_I2S_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_I2S1_CLKSOURCE))
#endif /* RCC_DCKCFGR_I2S2SRC */

#if defined(CEC)
#define IS_LL_RCC_CEC_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_CEC_CLKSOURCE))
#endif /* CEC */

#if defined(DSI)
#define IS_LL_RCC_DSI_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_DSI_CLKSOURCE))
#endif /* DSI */

#if defined(LTDC)
#define IS_LL_RCC_LTDC_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_LTDC_CLKSOURCE))
#endif /* LTDC */

#if defined(SPDIFRX)
#define IS_LL_RCC_SPDIFRX_CLKSOURCE(__VALUE__)    (((__VALUE__) == LL_RCC_SPDIFRX1_CLKSOURCE))
#endif /* SPDIFRX */
/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/
/** @defgroup RCC_LL_Private_Functions RCC Private functions
  * @{
  */
uint32_t RCC_GetSystemClockFreq(void);
uint32_t RCC_GetHCLKClockFreq(uint32_t SYSCLK_Frequency);
uint32_t RCC_GetPCLK1ClockFreq(uint32_t HCLK_Frequency);
uint32_t RCC_GetPCLK2ClockFreq(uint32_t HCLK_Frequency);
uint32_t RCC_PLL_GetFreqDomain_SYS(uint32_t SYSCLK_Source);
uint32_t RCC_PLL_GetFreqDomain_48M(void);
#if defined(RCC_DCKCFGR_I2SSRC) || defined(RCC_DCKCFGR_I2S1SRC)
uint32_t RCC_PLL_GetFreqDomain_I2S(void);
#endif /* RCC_DCKCFGR_I2SSRC || RCC_DCKCFGR_I2S1SRC */
#if defined(SPDIFRX)
uint32_t RCC_PLL_GetFreqDomain_SPDIFRX(void);
#endif /* SPDIFRX */
#if defined(RCC_PLLCFGR_PLLR)
#if defined(SAI1)
uint32_t RCC_PLL_GetFreqDomain_SAI(void);
#endif /* SAI1 */
#endif /* RCC_PLLCFGR_PLLR */
#if defined(DSI)
uint32_t RCC_PLL_GetFreqDomain_DSI(void);
#endif /* DSI */
#if defined(RCC_PLLSAI_SUPPORT)
uint32_t RCC_PLLSAI_GetFreqDomain_SAI(void);
#if defined(RCC_PLLSAICFGR_PLLSAIP)
uint32_t RCC_PLLSAI_GetFreqDomain_48M(void);
#endif /* RCC_PLLSAICFGR_PLLSAIP */
#if defined(LTDC)
uint32_t RCC_PLLSAI_GetFreqDomain_LTDC(void);
#endif /* LTDC */
#endif /* RCC_PLLSAI_SUPPORT */
#if defined(RCC_PLLI2S_SUPPORT)
uint32_t RCC_PLLI2S_GetFreqDomain_I2S(void);
#if defined(RCC_PLLI2SCFGR_PLLI2SQ) && !defined(RCC_DCKCFGR_PLLI2SDIVQ)
uint32_t RCC_PLLI2S_GetFreqDomain_48M(void);
#endif /* RCC_PLLI2SCFGR_PLLI2SQ && !RCC_DCKCFGR_PLLI2SDIVQ */
#if defined(SAI1)
uint32_t RCC_PLLI2S_GetFreqDomain_SAI(void);
#endif /* SAI1 */
#if defined(SPDIFRX)
uint32_t RCC_PLLI2S_GetFreqDomain_SPDIFRX(void);
#endif /* SPDIFRX */
#endif /* RCC_PLLI2S_SUPPORT */
/**
  * @}
  */


/* Exported functions --------------------------------------------------------*/
/** @addtogroup RCC_LL_Exported_Functions
  * @{
  */

/** @addtogroup RCC_LL_EF_Init
  * @{
  */

/**
  * @brief  Reset the RCC clock configuration to the default reset state.
  * @note   The default reset state of the clock configuration is given below:
  *         - HSI ON and used as system clock source
  *         - HSE and PLL OFF
  *         - AHB, APB1 and APB2 prescaler set to 1.
  *         - CSS, MCO OFF
  *         - All interrupts disabled
  * @note   This function doesn't modify the configuration of the
  *         - Peripheral clocks
  *         - LSI, LSE and RTC clocks
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: RCC registers are de-initialized
  *          - ERROR: not applicable
  */
ErrorStatus LL_RCC_DeInit(void)
{
  __IO uint32_t vl_mask;

  /* Set HSION bit */
  LL_RCC_HSI_Enable();

  /* Wait for HSI READY bit */
  while(LL_RCC_HSI_IsReady() != 1U)
  {}

  /* Reset CFGR register */
  LL_RCC_WriteReg(CFGR, 0x00000000U);

  /* Read CR register */
  vl_mask = LL_RCC_ReadReg(CR);

  /* Reset HSEON, HSEBYP, PLLON, CSSON bits */
  CLEAR_BIT(vl_mask,
            (RCC_CR_HSEON | RCC_CR_HSEBYP | RCC_CR_PLLON | RCC_CR_CSSON));

#if defined(RCC_PLLSAI_SUPPORT)
  /* Reset PLLSAION bit */
  CLEAR_BIT(vl_mask, RCC_CR_PLLSAION);
#endif /* RCC_PLLSAI_SUPPORT */

#if defined(RCC_PLLI2S_SUPPORT)
  /* Reset PLLI2SON bit */
  CLEAR_BIT(vl_mask, RCC_CR_PLLI2SON);
#endif /* RCC_PLLI2S_SUPPORT */

  /* Write new value in CR register */
  LL_RCC_WriteReg(CR, vl_mask);

  /* Set HSITRIM bits to the reset value*/
  LL_RCC_HSI_SetCalibTrimming(0x10U);

  /* Wait for PLL READY bit to be reset */
  while(LL_RCC_PLL_IsReady() != 0U)
  {}

  /* Reset PLLCFGR register */
  LL_RCC_WriteReg(PLLCFGR, RCC_PLLCFGR_RST_VALUE);

#if defined(RCC_PLLI2S_SUPPORT)
  /* Reset PLLI2SCFGR register */
  LL_RCC_WriteReg(PLLI2SCFGR, RCC_PLLI2SCFGR_RST_VALUE);
#endif /* RCC_PLLI2S_SUPPORT */

#if defined(RCC_PLLSAI_SUPPORT)
  /* Reset PLLSAICFGR register */
  LL_RCC_WriteReg(PLLSAICFGR, RCC_PLLSAICFGR_RST_VALUE);
#endif /* RCC_PLLSAI_SUPPORT */

  /* Disable all interrupts */
  CLEAR_BIT(RCC->CIR, RCC_CIR_LSIRDYIE | RCC_CIR_LSERDYIE | RCC_CIR_HSIRDYIE | RCC_CIR_HSERDYIE | RCC_CIR_PLLRDYIE);

#if defined(RCC_CIR_PLLI2SRDYIE)
  CLEAR_BIT(RCC->CIR, RCC_CIR_PLLI2SRDYIE);
#endif /* RCC_CIR_PLLI2SRDYIE */

#if defined(RCC_CIR_PLLSAIRDYIE)
  CLEAR_BIT(RCC->CIR, RCC_CIR_PLLSAIRDYIE);
#endif /* RCC_CIR_PLLSAIRDYIE */

  /* Clear all interrupt flags */
  SET_BIT(RCC->CIR, RCC_CIR_LSIRDYC | RCC_CIR_LSERDYC | RCC_CIR_HSIRDYC | RCC_CIR_HSERDYC | RCC_CIR_PLLRDYC | RCC_CIR_CSSC);

#if defined(RCC_CIR_PLLI2SRDYC)
  SET_BIT(RCC->CIR, RCC_CIR_PLLI2SRDYC);
#endif /* RCC_CIR_PLLI2SRDYC */

#if defined(RCC_CIR_PLLSAIRDYC)
  SET_BIT(RCC->CIR, RCC_CIR_PLLSAIRDYC);
#endif /* RCC_CIR_PLLSAIRDYC */

  /* Clear LSION bit */
  CLEAR_BIT(RCC->CSR, RCC_CSR_LSION);

  /* Reset all CSR flags */
  SET_BIT(RCC->CSR, RCC_CSR_RMVF);

  return SUCCESS;
}

/**
  * @}
  */

/** @addtogroup RCC_LL_EF_Get_Freq
  * @brief  Return the frequencies of different on chip clocks;  System, AHB, APB1 and APB2 buses clocks
  *         and different peripheral clocks available on the device.
  * @note   If SYSCLK source is HSI, function returns values based on HSI_VALUE(**)
  * @note   If SYSCLK source is HSE, function returns values based on HSE_VALUE(***)
  * @note   If SYSCLK source is PLL, function returns values based on HSE_VALUE(***)
  *         or HSI_VALUE(**) multiplied/divided by the PLL factors.
  * @note   (**) HSI_VALUE is a constant defined in this file (default value
  *              16 MHz) but the real value may vary depending on the variations
  *              in voltage and temperature.
  * @note   (***) HSE_VALUE is a constant defined in this file (default value
  *               25 MHz), user has to ensure that HSE_VALUE is same as the real
  *               frequency of the crystal used. Otherwise, this function may
  *               have wrong result.
  * @note   The result of this function could be incorrect when using fractional
  *         value for HSE crystal.
  * @note   This function can be used by the user application to compute the
  *         baud-rate for the communication peripherals or configure other parameters.
  * @{
  */

/**
  * @brief  Return the frequencies of different on chip clocks;  System, AHB, APB1 and APB2 buses clocks
  * @note   Each time SYSCLK, HCLK, PCLK1 and/or PCLK2 clock changes, this function
  *         must be called to update structure fields. Otherwise, any
  *         configuration based on this function will be incorrect.
  * @param  RCC_Clocks pointer to a @ref LL_RCC_ClocksTypeDef structure which will hold the clocks frequencies
  * @retval None
  */
void LL_RCC_GetSystemClocksFreq(LL_RCC_ClocksTypeDef *RCC_Clocks)
{
  /* Get SYSCLK frequency */
  RCC_Clocks->SYSCLK_Frequency = RCC_GetSystemClockFreq();

  /* HCLK clock frequency */
  RCC_Clocks->HCLK_Frequency   = RCC_GetHCLKClockFreq(RCC_Clocks->SYSCLK_Frequency);

  /* PCLK1 clock frequency */
  RCC_Clocks->PCLK1_Frequency  = RCC_GetPCLK1ClockFreq(RCC_Clocks->HCLK_Frequency);

  /* PCLK2 clock frequency */
  RCC_Clocks->PCLK2_Frequency  = RCC_GetPCLK2ClockFreq(RCC_Clocks->HCLK_Frequency);
}

#if defined(FMPI2C1)
/**
  * @brief  Return FMPI2Cx clock frequency
  * @param  FMPI2CxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_FMPI2C1_CLKSOURCE
  * @retval FMPI2C clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that HSI oscillator is not ready
  */
uint32_t LL_RCC_GetFMPI2CClockFreq(uint32_t FMPI2CxSource)
{
  uint32_t FMPI2C_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_FMPI2C_CLKSOURCE(FMPI2CxSource));

  if (FMPI2CxSource == LL_RCC_FMPI2C1_CLKSOURCE)
  {
    /* FMPI2C1 CLK clock frequency */
    switch (LL_RCC_GetFMPI2CClockSource(FMPI2CxSource))
    {
      case LL_RCC_FMPI2C1_CLKSOURCE_SYSCLK: /* FMPI2C1 Clock is System Clock */
        FMPI2C_frequency = RCC_GetSystemClockFreq();
        break;

      case LL_RCC_FMPI2C1_CLKSOURCE_HSI:    /* FMPI2C1 Clock is HSI Osc. */
        if (LL_RCC_HSI_IsReady())
        {
          FMPI2C_frequency = HSI_VALUE;
        }
        break;

      case LL_RCC_FMPI2C1_CLKSOURCE_PCLK1:  /* FMPI2C1 Clock is PCLK1 */
      default:
        FMPI2C_frequency = RCC_GetPCLK1ClockFreq(RCC_GetHCLKClockFreq(RCC_GetSystemClockFreq()));
        break;
    }
  }

  return FMPI2C_frequency;
}
#endif /* FMPI2C1 */

/**
  * @brief  Return I2Sx clock frequency
  * @param  I2SxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_I2S1_CLKSOURCE
  *         @arg @ref LL_RCC_I2S2_CLKSOURCE (*)
  *
  *         (*) value not defined in all devices.
  * @retval I2S clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator is not ready
  */
uint32_t LL_RCC_GetI2SClockFreq(uint32_t I2SxSource)
{
  uint32_t i2s_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_I2S_CLKSOURCE(I2SxSource));

  if (I2SxSource == LL_RCC_I2S1_CLKSOURCE)
  {
    /* I2S1 CLK clock frequency */
    switch (LL_RCC_GetI2SClockSource(I2SxSource))
    {
#if defined(RCC_PLLI2S_SUPPORT)
      case LL_RCC_I2S1_CLKSOURCE_PLLI2S:       /* I2S1 Clock is PLLI2S */
        if (LL_RCC_PLLI2S_IsReady())
        {
          i2s_frequency = RCC_PLLI2S_GetFreqDomain_I2S();
        }
        break;
#endif /* RCC_PLLI2S_SUPPORT */

#if defined(RCC_DCKCFGR_I2SSRC) || defined(RCC_DCKCFGR_I2S1SRC)
      case LL_RCC_I2S1_CLKSOURCE_PLL:          /* I2S1 Clock is PLL */
        if (LL_RCC_PLL_IsReady())
        {
          i2s_frequency = RCC_PLL_GetFreqDomain_I2S();
        }
        break;

      case LL_RCC_I2S1_CLKSOURCE_PLLSRC:       /* I2S1 Clock is PLL Main source */
        switch (LL_RCC_PLL_GetMainSource())
        {
           case LL_RCC_PLLSOURCE_HSE:          /* I2S1 Clock is HSE Osc. */
             if (LL_RCC_HSE_IsReady())
             {
               i2s_frequency = HSE_VALUE;
             }
             break;

           case LL_RCC_PLLSOURCE_HSI:          /* I2S1 Clock is HSI Osc. */
           default:
             if (LL_RCC_HSI_IsReady())
             {
               i2s_frequency = HSI_VALUE;
             }
             break;
        }
        break;
#endif /* RCC_DCKCFGR_I2SSRC || RCC_DCKCFGR_I2S1SRC */

      case LL_RCC_I2S1_CLKSOURCE_PIN:          /* I2S1 Clock is External clock */
      default:
        i2s_frequency = EXTERNAL_CLOCK_VALUE;
        break;
    }
  }
#if defined(RCC_DCKCFGR_I2S2SRC)
  else
  {
    /* I2S2 CLK clock frequency */
    switch (LL_RCC_GetI2SClockSource(I2SxSource))
    {
      case LL_RCC_I2S2_CLKSOURCE_PLLI2S:       /* I2S2 Clock is PLLI2S */
        if (LL_RCC_PLLI2S_IsReady())
        {
          i2s_frequency = RCC_PLLI2S_GetFreqDomain_I2S();
        }
        break;

      case LL_RCC_I2S2_CLKSOURCE_PLL:          /* I2S2 Clock is PLL */
        if (LL_RCC_PLL_IsReady())
        {
          i2s_frequency = RCC_PLL_GetFreqDomain_I2S();
        }
        break;

      case LL_RCC_I2S2_CLKSOURCE_PLLSRC:       /* I2S2 Clock is PLL Main source */
        switch (LL_RCC_PLL_GetMainSource())
        {
           case LL_RCC_PLLSOURCE_HSE:          /* I2S2 Clock is HSE Osc. */
             if (LL_RCC_HSE_IsReady())
             {
               i2s_frequency = HSE_VALUE;
             }
             break;

           case LL_RCC_PLLSOURCE_HSI:          /* I2S2 Clock is HSI Osc. */
           default:
             if (LL_RCC_HSI_IsReady())
             {
               i2s_frequency = HSI_VALUE;
             }
             break;
        }
        break;

      case LL_RCC_I2S2_CLKSOURCE_PIN:          /* I2S2 Clock is External clock */
      default:
        i2s_frequency = EXTERNAL_CLOCK_VALUE;
        break;
    } 
  }
#endif /* RCC_DCKCFGR_I2S2SRC */

  return i2s_frequency;
}

#if defined(LPTIM1)
/**
  * @brief  Return LPTIMx clock frequency
  * @param  LPTIMxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_LPTIM1_CLKSOURCE
  * @retval LPTIM clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator (HSI, LSI or LSE) is not ready
  */
uint32_t LL_RCC_GetLPTIMClockFreq(uint32_t LPTIMxSource)
{
  uint32_t lptim_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_LPTIM_CLKSOURCE(LPTIMxSource));

  if (LPTIMxSource == LL_RCC_LPTIM1_CLKSOURCE)
  {
    /* LPTIM1CLK clock frequency */
    switch (LL_RCC_GetLPTIMClockSource(LPTIMxSource))
    {
      case LL_RCC_LPTIM1_CLKSOURCE_LSI:    /* LPTIM1 Clock is LSI Osc. */
        if (LL_RCC_LSI_IsReady())
        {
          lptim_frequency = LSI_VALUE;
        }
        break;

      case LL_RCC_LPTIM1_CLKSOURCE_HSI:    /* LPTIM1 Clock is HSI Osc. */
        if (LL_RCC_HSI_IsReady())
        {
          lptim_frequency = HSI_VALUE;
        }
        break;

      case LL_RCC_LPTIM1_CLKSOURCE_LSE:    /* LPTIM1 Clock is LSE Osc. */
        if (LL_RCC_LSE_IsReady())
        {
          lptim_frequency = LSE_VALUE;
        }
        break;

      case LL_RCC_LPTIM1_CLKSOURCE_PCLK1:  /* LPTIM1 Clock is PCLK1 */
      default:
        lptim_frequency = RCC_GetPCLK1ClockFreq(RCC_GetHCLKClockFreq(RCC_GetSystemClockFreq()));
        break;
    }
  }

  return lptim_frequency;
}
#endif /* LPTIM1 */

#if defined(SAI1)
/**
  * @brief  Return SAIx clock frequency
  * @param  SAIxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_SAI1_CLKSOURCE (*)
  *         @arg @ref LL_RCC_SAI2_CLKSOURCE (*)
  *         @arg @ref LL_RCC_SAI1_A_CLKSOURCE (*)
  *         @arg @ref LL_RCC_SAI1_B_CLKSOURCE (*)
  *
  *         (*) value not defined in all devices.
  * @retval SAI clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator is not ready
  */
uint32_t LL_RCC_GetSAIClockFreq(uint32_t SAIxSource)
{
  uint32_t sai_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_SAI_CLKSOURCE(SAIxSource));

#if defined(RCC_DCKCFGR_SAI1SRC)
  if ((SAIxSource == LL_RCC_SAI1_CLKSOURCE) || (SAIxSource == LL_RCC_SAI2_CLKSOURCE))
  {
    /* SAI1CLK clock frequency */
    switch (LL_RCC_GetSAIClockSource(SAIxSource))
    {
      case LL_RCC_SAI1_CLKSOURCE_PLLSAI:     /* PLLSAI clock used as SAI1 clock source */
      case LL_RCC_SAI2_CLKSOURCE_PLLSAI:     /* PLLSAI clock used as SAI2 clock source */
        if (LL_RCC_PLLSAI_IsReady())
        {
          sai_frequency = RCC_PLLSAI_GetFreqDomain_SAI();
        }
        break;

      case LL_RCC_SAI1_CLKSOURCE_PLLI2S:     /* PLLI2S clock used as SAI1 clock source */
      case LL_RCC_SAI2_CLKSOURCE_PLLI2S:     /* PLLI2S clock used as SAI2 clock source */
        if (LL_RCC_PLLI2S_IsReady())
        {
          sai_frequency = RCC_PLLI2S_GetFreqDomain_SAI();
        }
        break;

      case LL_RCC_SAI1_CLKSOURCE_PLL:        /* PLL clock used as SAI1 clock source */
      case LL_RCC_SAI2_CLKSOURCE_PLL:        /* PLL clock used as SAI2 clock source */
        if (LL_RCC_PLL_IsReady())
        {
          sai_frequency = RCC_PLL_GetFreqDomain_SAI();
        }
        break;

      case LL_RCC_SAI2_CLKSOURCE_PLLSRC:
        switch (LL_RCC_PLL_GetMainSource())
        {
           case LL_RCC_PLLSOURCE_HSE:        /* HSE clock used as SAI2 clock source */
             if (LL_RCC_HSE_IsReady())
             {
               sai_frequency = HSE_VALUE;
             }
             break;

           case LL_RCC_PLLSOURCE_HSI:        /* HSI clock used as SAI2 clock source */
           default:
             if (LL_RCC_HSI_IsReady())
             {
               sai_frequency = HSI_VALUE;
             }
             break;
        }
        break;

      case LL_RCC_SAI1_CLKSOURCE_PIN:        /* External input clock used as SAI1 clock source */
      default:
        sai_frequency = EXTERNAL_CLOCK_VALUE;
        break;
    }
  }
#endif /* RCC_DCKCFGR_SAI1SRC */
#if defined(RCC_DCKCFGR_SAI1ASRC)
  if ((SAIxSource == LL_RCC_SAI1_A_CLKSOURCE) || (SAIxSource == LL_RCC_SAI1_B_CLKSOURCE))
  {
    /* SAI1CLK clock frequency */
    switch (LL_RCC_GetSAIClockSource(SAIxSource))
    {
#if defined(RCC_PLLSAI_SUPPORT)
      case LL_RCC_SAI1_A_CLKSOURCE_PLLSAI:     /* PLLSAI clock used as SAI1 Block A clock source */
      case LL_RCC_SAI1_B_CLKSOURCE_PLLSAI:     /* PLLSAI clock used as SAI1 Block B clock source */
        if (LL_RCC_PLLSAI_IsReady())
        {
          sai_frequency = RCC_PLLSAI_GetFreqDomain_SAI();
        }
        break;
#endif /* RCC_PLLSAI_SUPPORT */

      case LL_RCC_SAI1_A_CLKSOURCE_PLLI2S:     /* PLLI2S clock used as SAI1 Block A clock source */
      case LL_RCC_SAI1_B_CLKSOURCE_PLLI2S:     /* PLLI2S clock used as SAI1 Block B clock source */
        if (LL_RCC_PLLI2S_IsReady())
        {
          sai_frequency = RCC_PLLI2S_GetFreqDomain_SAI();
        }
        break;

#if defined(RCC_SAI1A_PLLSOURCE_SUPPORT)
      case LL_RCC_SAI1_A_CLKSOURCE_PLL:        /* PLL clock used as SAI1 Block A clock source */
      case LL_RCC_SAI1_B_CLKSOURCE_PLL:        /* PLL clock used as SAI1 Block B clock source */
        if (LL_RCC_PLL_IsReady())
        {
          sai_frequency = RCC_PLL_GetFreqDomain_SAI();
        }
        break;

      case LL_RCC_SAI1_A_CLKSOURCE_PLLSRC:
      case LL_RCC_SAI1_B_CLKSOURCE_PLLSRC:
        switch (LL_RCC_PLL_GetMainSource())
        {
           case LL_RCC_PLLSOURCE_HSE:          /* HSE clock used as SAI1 Block A or B clock source */
             if (LL_RCC_HSE_IsReady())
             {
               sai_frequency = HSE_VALUE;
             }
             break;

           case LL_RCC_PLLSOURCE_HSI:          /* HSI clock used as SAI1 Block A or B clock source */
           default:
             if (LL_RCC_HSI_IsReady())
             {
               sai_frequency = HSI_VALUE;
             }
             break;
        }
        break;
#endif /* RCC_SAI1A_PLLSOURCE_SUPPORT */

      case LL_RCC_SAI1_A_CLKSOURCE_PIN:        /* External input clock used as SAI1 Block A clock source */
      case LL_RCC_SAI1_B_CLKSOURCE_PIN:        /* External input clock used as SAI1 Block B clock source */
      default:
        sai_frequency = EXTERNAL_CLOCK_VALUE;
        break;
    }
  }
#endif /* RCC_DCKCFGR_SAI1ASRC */

  return sai_frequency;
}
#endif /* SAI1 */

#if defined(SDIO)
/**
  * @brief  Return SDIOx clock frequency
  * @param  SDIOxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_SDIO_CLKSOURCE
  * @retval SDIO clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator is not ready
  */
uint32_t LL_RCC_GetSDIOClockFreq(uint32_t SDIOxSource)
{
  uint32_t SDIO_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_SDIO_CLKSOURCE(SDIOxSource));

  if (SDIOxSource == LL_RCC_SDIO_CLKSOURCE)
  {
#if defined(RCC_DCKCFGR_SDIOSEL) || defined(RCC_DCKCFGR2_SDIOSEL)
    /* SDIOCLK clock frequency */
    switch (LL_RCC_GetSDIOClockSource(SDIOxSource))
    {
      case LL_RCC_SDIO_CLKSOURCE_PLL48CLK:         /* PLL48M clock used as SDIO clock source */
        switch (LL_RCC_GetCK48MClockSource(LL_RCC_CK48M_CLKSOURCE))
        {
          case LL_RCC_CK48M_CLKSOURCE_PLL:         /* PLL clock used as 48Mhz domain clock */
            if (LL_RCC_PLL_IsReady())
            {
              SDIO_frequency = RCC_PLL_GetFreqDomain_48M();
            }
          break;

#if defined(RCC_PLLSAI_SUPPORT)
          case LL_RCC_CK48M_CLKSOURCE_PLLSAI:      /* PLLSAI clock used as 48Mhz domain clock */
          default:
            if (LL_RCC_PLLSAI_IsReady())
            {
              SDIO_frequency = RCC_PLLSAI_GetFreqDomain_48M();
            }
            break;
#endif /* RCC_PLLSAI_SUPPORT */

#if defined(RCC_PLLI2SCFGR_PLLI2SQ) && !defined(RCC_DCKCFGR_PLLI2SDIVQ)
          case LL_RCC_CK48M_CLKSOURCE_PLLI2S:      /* PLLI2S clock used as 48Mhz domain clock */
          default:
            if (LL_RCC_PLLI2S_IsReady())
            {
              SDIO_frequency = RCC_PLLI2S_GetFreqDomain_48M();
            }
            break;
#endif /* RCC_PLLI2SCFGR_PLLI2SQ && !RCC_DCKCFGR_PLLI2SDIVQ */
        }
        break;

      case LL_RCC_SDIO_CLKSOURCE_SYSCLK:           /* PLL clock used as SDIO clock source */
      default:
      SDIO_frequency = RCC_GetSystemClockFreq();
      break;
    }
#else
    /* PLL clock used as 48Mhz domain clock */
    if (LL_RCC_PLL_IsReady())
    {
      SDIO_frequency = RCC_PLL_GetFreqDomain_48M();
    }
#endif /* RCC_DCKCFGR_SDIOSEL || RCC_DCKCFGR2_SDIOSEL */
  }

  return SDIO_frequency;
}
#endif /* SDIO */

#if defined(RNG)
/**
  * @brief  Return RNGx clock frequency
  * @param  RNGxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_RNG_CLKSOURCE
  * @retval RNG clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator is not ready
  */
uint32_t LL_RCC_GetRNGClockFreq(uint32_t RNGxSource)
{
  uint32_t rng_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_RNG_CLKSOURCE(RNGxSource));

#if defined(RCC_DCKCFGR_CK48MSEL) || defined(RCC_DCKCFGR2_CK48MSEL)
  /* RNGCLK clock frequency */
  switch (LL_RCC_GetRNGClockSource(RNGxSource))
  {
#if defined(RCC_PLLI2SCFGR_PLLI2SQ) && !defined(RCC_DCKCFGR_PLLI2SDIVQ)
    case LL_RCC_RNG_CLKSOURCE_PLLI2S:        /* PLLI2S clock used as RNG clock source */
      if (LL_RCC_PLLI2S_IsReady())
      {
        rng_frequency = RCC_PLLI2S_GetFreqDomain_48M();
      }
      break;
#endif /* RCC_PLLI2SCFGR_PLLI2SQ && !RCC_DCKCFGR_PLLI2SDIVQ */

#if defined(RCC_PLLSAI_SUPPORT)
    case LL_RCC_RNG_CLKSOURCE_PLLSAI:        /* PLLSAI clock used as RNG clock source */
      if (LL_RCC_PLLSAI_IsReady())
      {
        rng_frequency = RCC_PLLSAI_GetFreqDomain_48M();
      }
      break;
#endif /* RCC_PLLSAI_SUPPORT */

    case LL_RCC_RNG_CLKSOURCE_PLL:           /* PLL clock used as RNG clock source */
    default:
      if (LL_RCC_PLL_IsReady())
      {
        rng_frequency = RCC_PLL_GetFreqDomain_48M();
      }
      break;
  }
#else
  /* PLL clock used as RNG clock source */
  if (LL_RCC_PLL_IsReady())
  {
    rng_frequency = RCC_PLL_GetFreqDomain_48M();
  }
#endif /* RCC_DCKCFGR_CK48MSEL || RCC_DCKCFGR2_CK48MSEL */

  return rng_frequency;
}
#endif /* RNG */

#if defined(CEC)
/**
  * @brief  Return CEC clock frequency
  * @param  CECxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_CEC_CLKSOURCE
  * @retval CEC clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator (HSI or LSE) is not ready
  */
uint32_t LL_RCC_GetCECClockFreq(uint32_t CECxSource)
{
  uint32_t cec_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_CEC_CLKSOURCE(CECxSource));

  /* CECCLK clock frequency */
  switch (LL_RCC_GetCECClockSource(CECxSource))
  {
    case LL_RCC_CEC_CLKSOURCE_LSE:           /* CEC Clock is LSE Osc. */
      if (LL_RCC_LSE_IsReady())
      {
        cec_frequency = LSE_VALUE;
      }
      break;

    case LL_RCC_CEC_CLKSOURCE_HSI_DIV488:    /* CEC Clock is HSI Osc. */
    default:
      if (LL_RCC_HSI_IsReady())
      {
        cec_frequency = HSI_VALUE/488U;
      }
      break;
  }

  return cec_frequency;
}
#endif /* CEC */

#if defined(USB_OTG_FS) || defined(USB_OTG_HS)
/**
  * @brief  Return USBx clock frequency
  * @param  USBxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_USB_CLKSOURCE
  * @retval USB clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator is not ready
  */
uint32_t LL_RCC_GetUSBClockFreq(uint32_t USBxSource)
{
  uint32_t usb_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_USB_CLKSOURCE(USBxSource));

#if defined(RCC_DCKCFGR_CK48MSEL) || defined(RCC_DCKCFGR2_CK48MSEL)
  /* USBCLK clock frequency */
  switch (LL_RCC_GetUSBClockSource(USBxSource))
  {
#if defined(RCC_PLLI2SCFGR_PLLI2SQ) && !defined(RCC_DCKCFGR_PLLI2SDIVQ)
    case LL_RCC_USB_CLKSOURCE_PLLI2S:       /* PLLI2S clock used as USB clock source */
      if (LL_RCC_PLLI2S_IsReady())
      {
        usb_frequency = RCC_PLLI2S_GetFreqDomain_48M();
      }
      break;

#endif /* RCC_PLLI2SCFGR_PLLI2SQ && !RCC_DCKCFGR_PLLI2SDIVQ */

#if defined(RCC_PLLSAI_SUPPORT)
    case LL_RCC_USB_CLKSOURCE_PLLSAI:       /* PLLSAI clock used as USB clock source */
      if (LL_RCC_PLLSAI_IsReady())
      {
        usb_frequency = RCC_PLLSAI_GetFreqDomain_48M();
      }
      break;
#endif /* RCC_PLLSAI_SUPPORT */

    case LL_RCC_USB_CLKSOURCE_PLL:          /* PLL clock used as USB clock source */
    default:
      if (LL_RCC_PLL_IsReady())
      {
        usb_frequency = RCC_PLL_GetFreqDomain_48M();
      }
      break;
  }
#else
  /* PLL clock used as USB clock source */
  if (LL_RCC_PLL_IsReady())
  {
    usb_frequency = RCC_PLL_GetFreqDomain_48M();
  }
#endif /* RCC_DCKCFGR_CK48MSEL || RCC_DCKCFGR2_CK48MSEL */

  return usb_frequency;
}
#endif /* USB_OTG_FS || USB_OTG_HS */

#if defined(DFSDM1_Channel0)
/**
  * @brief  Return DFSDMx clock frequency
  * @param  DFSDMxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_DFSDM1_CLKSOURCE
  *         @arg @ref LL_RCC_DFSDM2_CLKSOURCE (*)
  *
  *         (*) value not defined in all devices.
  * @retval DFSDM clock frequency (in Hz)
  */
uint32_t LL_RCC_GetDFSDMClockFreq(uint32_t DFSDMxSource)
{
  uint32_t dfsdm_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_DFSDM_CLKSOURCE(DFSDMxSource));

  if (DFSDMxSource == LL_RCC_DFSDM1_CLKSOURCE)
  {
    /* DFSDM1CLK clock frequency */
    switch (LL_RCC_GetDFSDMClockSource(DFSDMxSource))
    {
      case LL_RCC_DFSDM1_CLKSOURCE_SYSCLK:      /* DFSDM1 Clock is SYSCLK */
        dfsdm_frequency = RCC_GetSystemClockFreq();
        break;

      case LL_RCC_DFSDM1_CLKSOURCE_PCLK2:       /* DFSDM1 Clock is PCLK2 */
      default:
        dfsdm_frequency = RCC_GetPCLK2ClockFreq(RCC_GetHCLKClockFreq(RCC_GetSystemClockFreq()));
        break;
    }
  }
#if defined(DFSDM2_Channel0)
  else
  {
    /* DFSDM2CLK clock frequency */
    switch (LL_RCC_GetDFSDMClockSource(DFSDMxSource))
    {
      case LL_RCC_DFSDM2_CLKSOURCE_SYSCLK:      /* DFSDM2 Clock is SYSCLK */
        dfsdm_frequency = RCC_GetSystemClockFreq();
        break;

      case LL_RCC_DFSDM2_CLKSOURCE_PCLK2:       /* DFSDM2 Clock is PCLK2 */
      default:
        dfsdm_frequency = RCC_GetPCLK2ClockFreq(RCC_GetHCLKClockFreq(RCC_GetSystemClockFreq()));
        break;
    }
  }
#endif /* DFSDM2_Channel0 */

  return dfsdm_frequency;
}

/**
  * @brief  Return DFSDMx Audio clock frequency
  * @param  DFSDMxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_DFSDM1_AUDIO_CLKSOURCE
  *         @arg @ref LL_RCC_DFSDM2_AUDIO_CLKSOURCE (*)
  *
  *         (*) value not defined in all devices.
  * @retval DFSDM clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator is not ready
  */
uint32_t LL_RCC_GetDFSDMAudioClockFreq(uint32_t DFSDMxSource)
{
  uint32_t dfsdm_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_DFSDM_AUDIO_CLKSOURCE(DFSDMxSource));

  if (DFSDMxSource == LL_RCC_DFSDM1_AUDIO_CLKSOURCE)
  {
    /* DFSDM1CLK clock frequency */
    switch (LL_RCC_GetDFSDMAudioClockSource(DFSDMxSource))
    {
      case LL_RCC_DFSDM1_AUDIO_CLKSOURCE_I2S1:     /* I2S1 clock used as DFSDM1 clock */
        dfsdm_frequency = LL_RCC_GetI2SClockFreq(LL_RCC_I2S1_CLKSOURCE);
        break;

      case LL_RCC_DFSDM1_AUDIO_CLKSOURCE_I2S2:     /* I2S2 clock used as DFSDM1 clock */
      default:
        dfsdm_frequency = LL_RCC_GetI2SClockFreq(LL_RCC_I2S2_CLKSOURCE);
        break;
    }
  }
#if defined(DFSDM2_Channel0)
  else
  {
    /* DFSDM2CLK clock frequency */
    switch (LL_RCC_GetDFSDMAudioClockSource(DFSDMxSource))
    {
      case LL_RCC_DFSDM2_AUDIO_CLKSOURCE_I2S1:     /* I2S1 clock used as DFSDM2 clock */
        dfsdm_frequency = LL_RCC_GetI2SClockFreq(LL_RCC_I2S1_CLKSOURCE);
        break;

      case LL_RCC_DFSDM2_AUDIO_CLKSOURCE_I2S2:     /* I2S2 clock used as DFSDM2 clock */
      default:
        dfsdm_frequency = LL_RCC_GetI2SClockFreq(LL_RCC_I2S2_CLKSOURCE);
        break;
    }
  }
#endif /* DFSDM2_Channel0 */

  return dfsdm_frequency;
}
#endif /* DFSDM1_Channel0 */

#if defined(DSI)
/**
  * @brief  Return DSI clock frequency
  * @param  DSIxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_DSI_CLKSOURCE
  * @retval DSI clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator is not ready
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NA indicates that external clock is used
  */
uint32_t LL_RCC_GetDSIClockFreq(uint32_t DSIxSource)
{
  uint32_t dsi_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_DSI_CLKSOURCE(DSIxSource));

  /* DSICLK clock frequency */
  switch (LL_RCC_GetDSIClockSource(DSIxSource))
  {
    case LL_RCC_DSI_CLKSOURCE_PLL:     /* DSI Clock is PLL Osc. */
      if (LL_RCC_PLL_IsReady())
      {
        dsi_frequency = RCC_PLL_GetFreqDomain_DSI();
      }
      break;

    case LL_RCC_DSI_CLKSOURCE_PHY:    /* DSI Clock is DSI physical clock. */
    default:
      dsi_frequency = LL_RCC_PERIPH_FREQUENCY_NA;
      break;
  }

  return dsi_frequency;
}
#endif /* DSI */

#if defined(LTDC)
/**
  * @brief  Return LTDC clock frequency
  * @param  LTDCxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_LTDC_CLKSOURCE
  * @retval LTDC clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator PLLSAI is not ready
  */
uint32_t LL_RCC_GetLTDCClockFreq(uint32_t LTDCxSource)
{
  uint32_t ltdc_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_LTDC_CLKSOURCE(LTDCxSource));

  if (LL_RCC_PLLSAI_IsReady())
  {
     ltdc_frequency = RCC_PLLSAI_GetFreqDomain_LTDC();
  }

  return ltdc_frequency;
}
#endif /* LTDC */

#if defined(SPDIFRX)
/**
  * @brief  Return SPDIFRX clock frequency
  * @param  SPDIFRXxSource This parameter can be one of the following values:
  *         @arg @ref LL_RCC_SPDIFRX1_CLKSOURCE
  * @retval SPDIFRX clock frequency (in Hz)
  *         - @ref  LL_RCC_PERIPH_FREQUENCY_NO indicates that oscillator is not ready
  */
uint32_t LL_RCC_GetSPDIFRXClockFreq(uint32_t SPDIFRXxSource)
{
  uint32_t spdifrx_frequency = LL_RCC_PERIPH_FREQUENCY_NO;

  /* Check parameter */
  assert_param(IS_LL_RCC_SPDIFRX_CLKSOURCE(SPDIFRXxSource));

  /* SPDIFRX1CLK clock frequency */
  switch (LL_RCC_GetSPDIFRXClockSource(SPDIFRXxSource))
  {
    case LL_RCC_SPDIFRX1_CLKSOURCE_PLLI2S:  /* SPDIFRX Clock is PLLI2S Osc. */
      if (LL_RCC_PLLI2S_IsReady())
      {
        spdifrx_frequency = RCC_PLLI2S_GetFreqDomain_SPDIFRX();
      }
      break;

    case LL_RCC_SPDIFRX1_CLKSOURCE_PLL:     /* SPDIFRX Clock is PLL Osc. */
    default:
      if (LL_RCC_PLL_IsReady())
      {
        spdifrx_frequency = RCC_PLL_GetFreqDomain_SPDIFRX();
      }
      break;
  }

  return spdifrx_frequency;
}
#endif /* SPDIFRX */

/**
  * @}
  */

/**
  * @}
  */

/** @addtogroup RCC_LL_Private_Functions
  * @{
  */

/**
  * @brief  Return SYSTEM clock frequency
  * @retval SYSTEM clock frequency (in Hz)
  */
uint32_t RCC_GetSystemClockFreq(void)
{
  uint32_t frequency = 0U;

  /* Get SYSCLK source -------------------------------------------------------*/
  switch (LL_RCC_GetSysClkSource())
  {
    case LL_RCC_SYS_CLKSOURCE_STATUS_HSI:  /* HSI used as system clock  source */
      frequency = HSI_VALUE;
      break;

    case LL_RCC_SYS_CLKSOURCE_STATUS_HSE:  /* HSE used as system clock  source */
      frequency = HSE_VALUE;
      break;

    case LL_RCC_SYS_CLKSOURCE_STATUS_PLL:  /* PLL used as system clock  source */
      frequency = RCC_PLL_GetFreqDomain_SYS(LL_RCC_SYS_CLKSOURCE_STATUS_PLL);
      break;

#if defined(RCC_PLLR_SYSCLK_SUPPORT)
    case LL_RCC_SYS_CLKSOURCE_STATUS_PLLR: /* PLLR used as system clock  source */
      frequency = RCC_PLL_GetFreqDomain_SYS(LL_RCC_SYS_CLKSOURCE_STATUS_PLLR);
      break;
#endif /* RCC_PLLR_SYSCLK_SUPPORT */

    default:
      frequency = HSI_VALUE;
      break;
  }

  return frequency;
}

/**
  * @brief  Return HCLK clock frequency
  * @param  SYSCLK_Frequency SYSCLK clock frequency
  * @retval HCLK clock frequency (in Hz)
  */
uint32_t RCC_GetHCLKClockFreq(uint32_t SYSCLK_Frequency)
{
  /* HCLK clock frequency */
  return __LL_RCC_CALC_HCLK_FREQ(SYSCLK_Frequency, LL_RCC_GetAHBPrescaler());
}

/**
  * @brief  Return PCLK1 clock frequency
  * @param  HCLK_Frequency HCLK clock frequency
  * @retval PCLK1 clock frequency (in Hz)
  */
uint32_t RCC_GetPCLK1ClockFreq(uint32_t HCLK_Frequency)
{
  /* PCLK1 clock frequency */
  return __LL_RCC_CALC_PCLK1_FREQ(HCLK_Frequency, LL_RCC_GetAPB1Prescaler());
}

/**
  * @brief  Return PCLK2 clock frequency
  * @param  HCLK_Frequency HCLK clock frequency
  * @retval PCLK2 clock frequency (in Hz)
  */
uint32_t RCC_GetPCLK2ClockFreq(uint32_t HCLK_Frequency)
{
  /* PCLK2 clock frequency */
  return __LL_RCC_CALC_PCLK2_FREQ(HCLK_Frequency, LL_RCC_GetAPB2Prescaler());
}

/**
  * @brief  Return PLL clock frequency used for system domain
  * @param  SYSCLK_Source System clock source
  * @retval PLL clock frequency (in Hz)
  */
uint32_t RCC_PLL_GetFreqDomain_SYS(uint32_t SYSCLK_Source)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U, plloutputfreq = 0U;

  /* PLL_VCO = (HSE_VALUE or HSI_VALUE / PLLM) * PLLN
     SYSCLK = PLL_VCO / (PLLP or PLLR)
  */
  pllsource = LL_RCC_PLL_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLL clock source */
      pllinputfreq = HSI_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLL clock source */
      pllinputfreq = HSE_VALUE;
      break;

    default:
      pllinputfreq = HSI_VALUE;
      break;
  }

  if (SYSCLK_Source == LL_RCC_SYS_CLKSOURCE_STATUS_PLL)
  {
    plloutputfreq = __LL_RCC_CALC_PLLCLK_FREQ(pllinputfreq, LL_RCC_PLL_GetDivider(),
                                        LL_RCC_PLL_GetN(), LL_RCC_PLL_GetP());
  }
#if defined(RCC_PLLR_SYSCLK_SUPPORT)
  else
  {
    plloutputfreq = __LL_RCC_CALC_PLLRCLK_FREQ(pllinputfreq, LL_RCC_PLL_GetDivider(),
                                        LL_RCC_PLL_GetN(), LL_RCC_PLL_GetR());
  }
#endif /* RCC_PLLR_SYSCLK_SUPPORT */

  return plloutputfreq;
}

/**
  * @brief  Return PLL clock frequency used for 48 MHz domain
  * @retval PLL clock frequency (in Hz)
  */
uint32_t RCC_PLL_GetFreqDomain_48M(void)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U;

  /* PLL_VCO = (HSE_VALUE or HSI_VALUE / PLLM ) * PLLN
     48M Domain clock = PLL_VCO / PLLQ
  */
  pllsource = LL_RCC_PLL_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLL clock source */
      pllinputfreq = HSI_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLL clock source */
      pllinputfreq = HSE_VALUE;
      break;

    default:
      pllinputfreq = HSI_VALUE;
      break;
  }
  return __LL_RCC_CALC_PLLCLK_48M_FREQ(pllinputfreq, LL_RCC_PLL_GetDivider(),
                                        LL_RCC_PLL_GetN(), LL_RCC_PLL_GetQ());
}

#if defined(DSI)
/**
  * @brief  Return PLL clock frequency used for DSI clock
  * @retval PLL clock frequency (in Hz)
  */
uint32_t RCC_PLL_GetFreqDomain_DSI(void)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U;

  /* PLL_VCO = (HSE_VALUE or HSI_VALUE / PLLM) * PLLN
     DSICLK = PLL_VCO / PLLR
  */
  pllsource = LL_RCC_PLL_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLL clock source */
      pllinputfreq = HSE_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLL clock source */
    default:
      pllinputfreq = HSI_VALUE;
      break;
  }
  return __LL_RCC_CALC_PLLCLK_DSI_FREQ(pllinputfreq, LL_RCC_PLL_GetDivider(),
                                        LL_RCC_PLL_GetN(), LL_RCC_PLL_GetR());
}
#endif /* DSI */

#if defined(RCC_DCKCFGR_I2SSRC) || defined(RCC_DCKCFGR_I2S1SRC)
/**
  * @brief  Return PLL clock frequency used for I2S clock
  * @retval PLL clock frequency (in Hz)
  */
uint32_t RCC_PLL_GetFreqDomain_I2S(void)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U;

  /* PLL_VCO = (HSE_VALUE or HSI_VALUE / PLLM) * PLLN
     I2SCLK = PLL_VCO / PLLR
  */
  pllsource = LL_RCC_PLL_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLL clock source */
      pllinputfreq = HSE_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLL clock source */
    default:
      pllinputfreq = HSI_VALUE;
      break;
  }
  return __LL_RCC_CALC_PLLCLK_I2S_FREQ(pllinputfreq, LL_RCC_PLL_GetDivider(),
                                        LL_RCC_PLL_GetN(), LL_RCC_PLL_GetR());
}
#endif /* RCC_DCKCFGR_I2SSRC || RCC_DCKCFGR_I2S1SRC */

#if defined(SPDIFRX)
/**
  * @brief  Return PLL clock frequency used for SPDIFRX clock
  * @retval PLL clock frequency (in Hz)
  */
uint32_t RCC_PLL_GetFreqDomain_SPDIFRX(void)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U;

  /* PLL_VCO = (HSE_VALUE or HSI_VALUE / PLLM) * PLLN
     SPDIFRXCLK = PLL_VCO / PLLR
  */
  pllsource = LL_RCC_PLL_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLL clock source */
      pllinputfreq = HSE_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLL clock source */
    default:
      pllinputfreq = HSI_VALUE;
      break;
  }
  return __LL_RCC_CALC_PLLCLK_SPDIFRX_FREQ(pllinputfreq, LL_RCC_PLL_GetDivider(),
                                        LL_RCC_PLL_GetN(), LL_RCC_PLL_GetR());
}
#endif /* SPDIFRX */

#if defined(RCC_PLLCFGR_PLLR)
#if defined(SAI1)
/**
  * @brief  Return PLL clock frequency used for SAI clock
  * @retval PLL clock frequency (in Hz)
  */
uint32_t RCC_PLL_GetFreqDomain_SAI(void)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U, plloutputfreq = 0U;

  /* PLL_VCO = (HSE_VALUE or HSI_VALUE / PLLM) * PLLN
     SAICLK = (PLL_VCO / PLLR) / PLLDIVR
     or
     SAICLK = PLL_VCO / PLLR
  */
  pllsource = LL_RCC_PLL_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLL clock source */
      pllinputfreq = HSE_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLL clock source */
    default:
      pllinputfreq = HSI_VALUE;
      break;
  }

#if defined(RCC_DCKCFGR_PLLDIVR)
  plloutputfreq = __LL_RCC_CALC_PLLCLK_SAI_FREQ(pllinputfreq, LL_RCC_PLL_GetDivider(),
                                        LL_RCC_PLL_GetN(), LL_RCC_PLL_GetR(), LL_RCC_PLL_GetDIVR());
#else
  plloutputfreq = __LL_RCC_CALC_PLLCLK_SAI_FREQ(pllinputfreq, LL_RCC_PLL_GetDivider(),
                                        LL_RCC_PLL_GetN(), LL_RCC_PLL_GetR());
#endif /* RCC_DCKCFGR_PLLDIVR */

  return plloutputfreq;
}
#endif /* SAI1 */
#endif /* RCC_PLLCFGR_PLLR */

#if defined(RCC_PLLSAI_SUPPORT)
/**
  * @brief  Return PLLSAI clock frequency used for SAI domain
  * @retval PLLSAI clock frequency (in Hz)
  */
uint32_t RCC_PLLSAI_GetFreqDomain_SAI(void)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U;

  /* PLLSAI_VCO = (HSE_VALUE or HSI_VALUE / PLLSAIM) * PLLSAIN
     SAI domain clock  = (PLLSAI_VCO / PLLSAIQ) / PLLSAIDIVQ
  */
  pllsource = LL_RCC_PLL_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLLSAI clock source */
      pllinputfreq = HSI_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLLSAI clock source */
      pllinputfreq = HSE_VALUE;
      break;

    default:
      pllinputfreq = HSI_VALUE;
      break;
  }
  return __LL_RCC_CALC_PLLSAI_SAI_FREQ(pllinputfreq, LL_RCC_PLLSAI_GetDivider(),
                                        LL_RCC_PLLSAI_GetN(), LL_RCC_PLLSAI_GetQ(), LL_RCC_PLLSAI_GetDIVQ());
}

#if defined(RCC_PLLSAICFGR_PLLSAIP)
/**
  * @brief  Return PLLSAI clock frequency used for 48Mhz domain
  * @retval PLLSAI clock frequency (in Hz)
  */
uint32_t RCC_PLLSAI_GetFreqDomain_48M(void)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U;

  /* PLLSAI_VCO = (HSE_VALUE or HSI_VALUE / PLLSAIM) * PLLSAIN
     48M Domain clock  = PLLSAI_VCO / PLLSAIP
  */
  pllsource = LL_RCC_PLL_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLLSAI clock source */
      pllinputfreq = HSI_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLLSAI clock source */
      pllinputfreq = HSE_VALUE;
      break;

    default:
      pllinputfreq = HSI_VALUE;
      break;
  }
  return __LL_RCC_CALC_PLLSAI_48M_FREQ(pllinputfreq, LL_RCC_PLLSAI_GetDivider(),
                                        LL_RCC_PLLSAI_GetN(), LL_RCC_PLLSAI_GetP());
}
#endif /* RCC_PLLSAICFGR_PLLSAIP */

#if defined(LTDC)
/**
  * @brief  Return PLLSAI clock frequency used for LTDC domain
  * @retval PLLSAI clock frequency (in Hz)
  */
uint32_t RCC_PLLSAI_GetFreqDomain_LTDC(void)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U;

  /* PLLSAI_VCO = (HSE_VALUE or HSI_VALUE / PLLSAIM) * PLLSAIN
     LTDC Domain clock  = (PLLSAI_VCO / PLLSAIR) / PLLSAIDIVR
  */
  pllsource = LL_RCC_PLL_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLLSAI clock source */
      pllinputfreq = HSI_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLLSAI clock source */
      pllinputfreq = HSE_VALUE;
      break;

    default:
      pllinputfreq = HSI_VALUE;
      break;
  }
  return __LL_RCC_CALC_PLLSAI_LTDC_FREQ(pllinputfreq, LL_RCC_PLLSAI_GetDivider(),
                                        LL_RCC_PLLSAI_GetN(), LL_RCC_PLLSAI_GetR(), LL_RCC_PLLSAI_GetDIVR());
}
#endif /* LTDC */
#endif /* RCC_PLLSAI_SUPPORT */

#if defined(RCC_PLLI2S_SUPPORT)
#if defined(SAI1)
/**
  * @brief  Return PLLI2S clock frequency used for SAI domains
  * @retval PLLI2S clock frequency (in Hz)
  */
uint32_t RCC_PLLI2S_GetFreqDomain_SAI(void)
{
  uint32_t plli2sinputfreq = 0U, plli2ssource = 0U, plli2soutputfreq = 0U;

  /* PLLI2S_VCO = (HSE_VALUE or HSI_VALUE / PLLI2SM) * PLLI2SN
     SAI domain clock  = (PLLI2S_VCO / PLLI2SQ) / PLLI2SDIVQ
     or
     SAI domain clock  = (PLLI2S_VCO / PLLI2SR) / PLLI2SDIVR
  */
  plli2ssource = LL_RCC_PLLI2S_GetMainSource();

  switch (plli2ssource)
  {
    case LL_RCC_PLLSOURCE_HSE:     /* HSE used as PLLI2S clock source */
      plli2sinputfreq = HSE_VALUE;
      break;

#if defined(RCC_PLLI2SCFGR_PLLI2SSRC)
    case LL_RCC_PLLI2SSOURCE_PIN:  /* External pin input clock used as PLLI2S clock source */
      plli2sinputfreq = EXTERNAL_CLOCK_VALUE;
      break;
#endif /* RCC_PLLI2SCFGR_PLLI2SSRC */

    case LL_RCC_PLLSOURCE_HSI:     /* HSI used as PLLI2S clock source */
    default:
      plli2sinputfreq = HSI_VALUE;
      break;
  }

#if defined(RCC_DCKCFGR_PLLI2SDIVQ)
  plli2soutputfreq = __LL_RCC_CALC_PLLI2S_SAI_FREQ(plli2sinputfreq, LL_RCC_PLLI2S_GetDivider(),
                                          LL_RCC_PLLI2S_GetN(), LL_RCC_PLLI2S_GetQ(), LL_RCC_PLLI2S_GetDIVQ());
#else
  plli2soutputfreq = __LL_RCC_CALC_PLLI2S_SAI_FREQ(plli2sinputfreq, LL_RCC_PLLI2S_GetDivider(),
                                          LL_RCC_PLLI2S_GetN(), LL_RCC_PLLI2S_GetR(), LL_RCC_PLLI2S_GetDIVR());
#endif /* RCC_DCKCFGR_PLLI2SDIVQ */

  return plli2soutputfreq;
}
#endif /* SAI1 */

#if defined(SPDIFRX)
/**
  * @brief  Return PLLI2S clock frequency used for SPDIFRX domain
  * @retval PLLI2S clock frequency (in Hz)
  */
uint32_t RCC_PLLI2S_GetFreqDomain_SPDIFRX(void)
{
  uint32_t pllinputfreq = 0U, pllsource = 0U;

  /* PLLI2S_VCO = (HSE_VALUE or HSI_VALUE / PLLI2SM) * PLLI2SN
     SPDIFRX Domain clock  = PLLI2S_VCO / PLLI2SP
  */
  pllsource = LL_RCC_PLLI2S_GetMainSource();

  switch (pllsource)
  {
    case LL_RCC_PLLSOURCE_HSE:  /* HSE used as PLLI2S clock source */
      pllinputfreq = HSE_VALUE;
      break;

    case LL_RCC_PLLSOURCE_HSI:  /* HSI used as PLLI2S clock source */
    default:
      pllinputfreq = HSI_VALUE;
      break;
  }

  return __LL_RCC_CALC_PLLI2S_SPDIFRX_FREQ(pllinputfreq, LL_RCC_PLLI2S_GetDivider(),
                                           LL_RCC_PLLI2S_GetN(), LL_RCC_PLLI2S_GetP());
}
#endif /* SPDIFRX */

/**
  * @brief  Return PLLI2S clock frequency used for I2S domain
  * @retval PLLI2S clock frequency (in Hz)
  */
uint32_t RCC_PLLI2S_GetFreqDomain_I2S(void)
{
  uint32_t plli2sinputfreq = 0U, plli2ssource = 0U, plli2soutputfreq = 0U;

  /* PLLI2S_VCO = (HSE_VALUE or HSI_VALUE / PLLI2SM) * PLLI2SN
     I2S Domain clock  = PLLI2S_VCO / PLLI2SR
  */
  plli2ssource = LL_RCC_PLLI2S_GetMainSource();

  switch (plli2ssource)
  {
    case LL_RCC_PLLSOURCE_HSE:     /* HSE used as PLLI2S clock source */
      plli2sinputfreq = HSE_VALUE;
      break;

#if defined(RCC_PLLI2SCFGR_PLLI2SSRC)
    case LL_RCC_PLLI2SSOURCE_PIN:  /* External pin input clock used as PLLI2S clock source */
      plli2sinputfreq = EXTERNAL_CLOCK_VALUE;
      break;
#endif /* RCC_PLLI2SCFGR_PLLI2SSRC */

    case LL_RCC_PLLSOURCE_HSI:     /* HSI used as PLLI2S clock source */
    default:
      plli2sinputfreq = HSI_VALUE;
      break;
  }

  plli2soutputfreq = __LL_RCC_CALC_PLLI2S_I2S_FREQ(plli2sinputfreq, LL_RCC_PLLI2S_GetDivider(),
                                                   LL_RCC_PLLI2S_GetN(), LL_RCC_PLLI2S_GetR());

  return plli2soutputfreq;
}

#if defined(RCC_PLLI2SCFGR_PLLI2SQ) && !defined(RCC_DCKCFGR_PLLI2SDIVQ)
/**
  * @brief  Return PLLI2S clock frequency used for 48Mhz domain
  * @retval PLLI2S clock frequency (in Hz)
  */
uint32_t RCC_PLLI2S_GetFreqDomain_48M(void)
{
  uint32_t plli2sinputfreq = 0U, plli2ssource = 0U, plli2soutputfreq = 0U;

  /* PLL48M_VCO = (HSE_VALUE or HSI_VALUE / PLLI2SM) * PLLI2SN
     48M Domain clock  = PLLI2S_VCO / PLLI2SQ
  */
  plli2ssource = LL_RCC_PLLI2S_GetMainSource();

  switch (plli2ssource)
  {
    case LL_RCC_PLLSOURCE_HSE:     /* HSE used as PLLI2S clock source */
      plli2sinputfreq = HSE_VALUE;
      break;

#if defined(RCC_PLLI2SCFGR_PLLI2SSRC)
    case LL_RCC_PLLI2SSOURCE_PIN:  /* External pin input clock used as PLLI2S clock source */
      plli2sinputfreq = EXTERNAL_CLOCK_VALUE;
      break;
#endif /* RCC_PLLI2SCFGR_PLLI2SSRC */

    case LL_RCC_PLLSOURCE_HSI:     /* HSI used as PLLI2S clock source */
    default:
      plli2sinputfreq = HSI_VALUE;
      break;
  }

  plli2soutputfreq = __LL_RCC_CALC_PLLI2S_48M_FREQ(plli2sinputfreq, LL_RCC_PLLI2S_GetDivider(),
                                                   LL_RCC_PLLI2S_GetN(), LL_RCC_PLLI2S_GetQ());

  return plli2soutputfreq;
}
#endif /* RCC_PLLI2SCFGR_PLLI2SQ && !RCC_DCKCFGR_PLLI2SDIVQ */
#endif /* RCC_PLLI2S_SUPPORT */
/**
  * @}
  */

/**
  * @}
  */

#endif /* defined(RCC) */

/**
  * @}
  */

#endif /* USE_FULL_LL_DRIVER */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
