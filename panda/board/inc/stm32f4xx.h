/**
  ******************************************************************************
  * @file    stm32f4xx.h
  * @author  MCD Application Team
  * @version V2.6.0
  * @date    04-November-2016
  * @brief   CMSIS STM32F4xx Device Peripheral Access Layer Header File.
  *
  *          The file is the unique include file that the application programmer
  *          is using in the C source code, usually in main.c. This file contains:
  *           - Configuration section that allows to select:
  *              - The STM32F4xx device used in the target application
  *              - To use or not the peripheral’s drivers in application code(i.e.
  *                code will be based on direct access to peripheral’s registers
  *                rather than drivers API), this option is controlled by
  *                "#define USE_HAL_DRIVER"
  *
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT(c) 2016 STMicroelectronics</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *   1. Redistributions of source code must retain the above copyright notice,
  *      this list of conditions and the following disclaimer.
  *   2. Redistributions in binary form must reproduce the above copyright notice,
  *      this list of conditions and the following disclaimer in the documentation
  *      and/or other materials provided with the distribution.
  *   3. Neither the name of STMicroelectronics nor the names of its contributors
  *      may be used to endorse or promote products derived from this software
  *      without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */

/** @addtogroup CMSIS
  * @{
  */

/** @addtogroup stm32f4xx
  * @{
  */

#ifndef __STM32F4xx_H
#define __STM32F4xx_H

#ifdef __cplusplus
 extern "C" {
#endif /* __cplusplus */

/** @addtogroup Library_configuration_section
  * @{
  */

/**
  * @brief STM32 Family
  */
#if !defined  (STM32F4)
#define STM32F4
#endif /* STM32F4 */

/* Uncomment the line below according to the target STM32 device used in your
   application
  */
#if !defined (STM32F405xx) && !defined (STM32F415xx) && !defined (STM32F407xx) && !defined (STM32F417xx) && \
    !defined (STM32F427xx) && !defined (STM32F437xx) && !defined (STM32F429xx) && !defined (STM32F439xx) && \
    !defined (STM32F401xC) && !defined (STM32F401xE) && !defined (STM32F410Tx) && !defined (STM32F410Cx) && \
    !defined (STM32F410Rx) && !defined (STM32F411xE) && !defined (STM32F446xx) && !defined (STM32F469xx) && \
    !defined (STM32F479xx) && !defined (STM32F412Cx) && !defined (STM32F412Rx) && !defined (STM32F412Vx) && \
    !defined (STM32F412Zx) && !defined (STM32F413xx) && !defined (STM32F423xx)
  /* #define STM32F405xx */   /*!< STM32F405RG, STM32F405VG and STM32F405ZG Devices */
  /* #define STM32F415xx */   /*!< STM32F415RG, STM32F415VG and STM32F415ZG Devices */
  /* #define STM32F407xx */   /*!< STM32F407VG, STM32F407VE, STM32F407ZG, STM32F407ZE, STM32F407IG  and STM32F407IE Devices */
  /* #define STM32F417xx */   /*!< STM32F417VG, STM32F417VE, STM32F417ZG, STM32F417ZE, STM32F417IG and STM32F417IE Devices */
  /* #define STM32F427xx */   /*!< STM32F427VG, STM32F427VI, STM32F427ZG, STM32F427ZI, STM32F427IG and STM32F427II Devices */
  /* #define STM32F437xx */   /*!< STM32F437VG, STM32F437VI, STM32F437ZG, STM32F437ZI, STM32F437IG and STM32F437II Devices */
  /* #define STM32F429xx */   /*!< STM32F429VG, STM32F429VI, STM32F429ZG, STM32F429ZI, STM32F429BG, STM32F429BI, STM32F429NG,
                                   STM32F439NI, STM32F429IG  and STM32F429II Devices */
  /* #define STM32F439xx */   /*!< STM32F439VG, STM32F439VI, STM32F439ZG, STM32F439ZI, STM32F439BG, STM32F439BI, STM32F439NG,
                                   STM32F439NI, STM32F439IG and STM32F439II Devices */
  /* #define STM32F401xC */   /*!< STM32F401CB, STM32F401CC, STM32F401RB, STM32F401RC, STM32F401VB and STM32F401VC Devices */
  /* #define STM32F401xE */   /*!< STM32F401CD, STM32F401RD, STM32F401VD, STM32F401CE, STM32F401RE and STM32F401VE Devices */
  /* #define STM32F410Tx */   /*!< STM32F410T8 and STM32F410TB Devices */
  /* #define STM32F410Cx */   /*!< STM32F410C8 and STM32F410CB Devices */
  /* #define STM32F410Rx */   /*!< STM32F410R8 and STM32F410RB Devices */
  /* #define STM32F411xE */   /*!< STM32F411CC, STM32F411RC, STM32F411VC, STM32F411CE, STM32F411RE and STM32F411VE Devices */
  /* #define STM32F446xx */   /*!< STM32F446MC, STM32F446ME, STM32F446RC, STM32F446RE, STM32F446VC, STM32F446VE, STM32F446ZC,
                                   and STM32F446ZE Devices */
  /* #define STM32F469xx */   /*!< STM32F469AI, STM32F469II, STM32F469BI, STM32F469NI, STM32F469AG, STM32F469IG, STM32F469BG,
                                   STM32F469NG, STM32F469AE, STM32F469IE, STM32F469BE and STM32F469NE Devices */
  /* #define STM32F479xx */   /*!< STM32F479AI, STM32F479II, STM32F479BI, STM32F479NI, STM32F479AG, STM32F479IG, STM32F479BG
                                   and STM32F479NG Devices */
  /* #define STM32F412Cx */   /*!< STM32F412CEU and STM32F412CGU Devices */
  /* #define STM32F412Zx */   /*!< STM32F412ZET, STM32F412ZGT, STM32F412ZEJ and STM32F412ZGJ Devices */
  /* #define STM32F412Vx */   /*!< STM32F412VET, STM32F412VGT, STM32F412VEH and STM32F412VGH Devices */
  /* #define STM32F412Rx */   /*!< STM32F412RET, STM32F412RGT, STM32F412REY and STM32F412RGY Devices */
  /* #define STM32F413xx */   /*!< STM32F413CH, STM32F413MH, STM32F413RH, STM32F413VH, STM32F413ZH, STM32F413CG, STM32F413MG,
                                   STM32F413RG, STM32F413VG and STM32F413ZG Devices */
  /* #define STM32F423xx */   /*!< STM32F423CH, STM32F423RH, STM32F423VH and STM32F423ZH Devices */
#endif

/*  Tip: To avoid modifying this file each time you need to switch between these
        devices, you can define the device in your toolchain compiler preprocessor.
  */
#if !defined  (USE_HAL_DRIVER)
/**
 * @brief Comment the line below if you will not use the peripherals drivers.
   In this case, these drivers will not be included and the application code will
   be based on direct access to peripherals registers
   */
  /*#define USE_HAL_DRIVER */
#endif /* USE_HAL_DRIVER */

/**
  * @brief CMSIS version number V2.6.0
  */
#define __STM32F4xx_CMSIS_VERSION_MAIN   (0x02U) /*!< [31:24] main version */
#define __STM32F4xx_CMSIS_VERSION_SUB1   (0x06U) /*!< [23:16] sub1 version */
#define __STM32F4xx_CMSIS_VERSION_SUB2   (0x00U) /*!< [15:8]  sub2 version */
#define __STM32F4xx_CMSIS_VERSION_RC     (0x00U) /*!< [7:0]  release candidate */
#define __STM32F4xx_CMSIS_VERSION        ((__STM32F4xx_CMSIS_VERSION_MAIN << 24)\
                                         |(__STM32F4xx_CMSIS_VERSION_SUB1 << 16)\
                                         |(__STM32F4xx_CMSIS_VERSION_SUB2 << 8 )\
                                         |(__STM32F4xx_CMSIS_VERSION))

/**
  * @}
  */

/** @addtogroup Device_Included
  * @{
  */

#if defined(STM32F405xx)
  #include "stm32f405xx.h"
#elif defined(STM32F415xx)
  #include "stm32f415xx.h"
#elif defined(STM32F407xx)
  #include "stm32f407xx.h"
#elif defined(STM32F417xx)
  #include "stm32f417xx.h"
#elif defined(STM32F427xx)
  #include "stm32f427xx.h"
#elif defined(STM32F437xx)
  #include "stm32f437xx.h"
#elif defined(STM32F429xx)
  #include "stm32f429xx.h"
#elif defined(STM32F439xx)
  #include "stm32f439xx.h"
#elif defined(STM32F401xC)
  #include "stm32f401xc.h"
#elif defined(STM32F401xE)
  #include "stm32f401xe.h"
#elif defined(STM32F410Tx)
  #include "stm32f410tx.h"
#elif defined(STM32F410Cx)
  #include "stm32f410cx.h"
#elif defined(STM32F410Rx)
  #include "stm32f410rx.h"
#elif defined(STM32F411xE)
  #include "stm32f411xe.h"
#elif defined(STM32F446xx)
  #include "stm32f446xx.h"
#elif defined(STM32F469xx)
  #include "stm32f469xx.h"
#elif defined(STM32F479xx)
  #include "stm32f479xx.h"
#elif defined(STM32F412Cx)
  #include "stm32f412cx.h"
#elif defined(STM32F412Zx)
  #include "stm32f412zx.h"
#elif defined(STM32F412Rx)
  #include "stm32f412rx.h"
#elif defined(STM32F412Vx)
  #include "stm32f412vx.h"
#elif defined(STM32F413xx)
  #include "stm32f413xx.h"
#elif defined(STM32F423xx)
  #include "stm32f423xx.h"
#else
 #error "Please select first the target STM32F4xx device used in your application (in stm32f4xx.h file)"
#endif

/**
  * @}
  */

/** @addtogroup Exported_types
  * @{
  */
typedef enum
{
  RESET = 0U,
  SET = !RESET
} FlagStatus, ITStatus;

typedef enum
{
  DISABLE = 0U,
  ENABLE = !DISABLE
} FunctionalState;
#define IS_FUNCTIONAL_STATE(STATE) (((STATE) == DISABLE) || ((STATE) == ENABLE))

typedef enum
{
  ERROR = 0U,
  SUCCESS = !ERROR
} ErrorStatus;

/**
  * @}
  */


/** @addtogroup Exported_macro
  * @{
  */
#define SET_BIT(REG, BIT)     ((REG) |= (BIT))

#define CLEAR_BIT(REG, BIT)   ((REG) &= ~(BIT))

#define READ_BIT(REG, BIT)    ((REG) & (BIT))

#define CLEAR_REG(REG)        ((REG) = (0x0))

#define WRITE_REG(REG, VAL)   ((REG) = (VAL))

#define READ_REG(REG)         ((REG))

#define MODIFY_REG(REG, CLEARMASK, SETMASK)  WRITE_REG((REG), (((READ_REG(REG)) & (~(CLEARMASK))) | (SETMASK)))

#define POSITION_VAL(VAL)     (__CLZ(__RBIT(VAL)))


/**
  * @}
  */

#if defined (USE_HAL_DRIVER)
 #include "stm32f4xx_hal.h"
#endif /* USE_HAL_DRIVER */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __STM32F4xx_H */
/**
  * @}
  */

/**
  * @}
  */




/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
