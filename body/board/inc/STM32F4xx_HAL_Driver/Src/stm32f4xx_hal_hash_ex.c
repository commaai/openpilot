/**
  ******************************************************************************
  * @file    stm32f4xx_hal_hash_ex.c
  * @author  MCD Application Team
  * @brief   Extended HASH HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the HASH peripheral for SHA-224 and SHA-256
  *          algorithms:
  *           + HASH or HMAC processing in polling mode
  *           + HASH or HMAC processing in interrupt mode
  *           + HASH or HMAC processing in DMA mode
  *         Additionally, this file provides functions to manage HMAC
  *         multi-buffer DMA-based processing for MD-5, SHA-1, SHA-224
  *         and SHA-256.
  *
  *
  @verbatim
 ===============================================================================
                     ##### HASH peripheral extended features  #####
 ===============================================================================
    [..]
    The SHA-224 and SHA-256 HASH and HMAC processing can be carried out exactly
    the same way as for SHA-1 or MD-5 algorithms.
    (#) Three modes are available.
        (##) Polling mode: processing APIs are blocking functions
             i.e. they process the data and wait till the digest computation is finished,
             e.g. HAL_HASHEx_xxx_Start()
        (##) Interrupt mode: processing APIs are not blocking functions
                i.e. they process the data under interrupt,
                e.g. HAL_HASHEx_xxx_Start_IT()
        (##) DMA mode: processing APIs are not blocking functions and the CPU is
             not used for data transfer i.e. the data transfer is ensured by DMA,
                e.g. HAL_HASHEx_xxx_Start_DMA(). Note that in DMA mode, a call to
                HAL_HASHEx_xxx_Finish() is then required to retrieve the digest.

   (#)Multi-buffer processing is possible in polling, interrupt and DMA modes.
        (##) In polling mode, only multi-buffer HASH processing is possible.
             API HAL_HASHEx_xxx_Accumulate() must be called for each input buffer, except for the last one.
             User must resort to HAL_HASHEx_xxx_Accumulate_End() to enter the last one and retrieve as
             well the computed digest.

        (##) In interrupt mode, API HAL_HASHEx_xxx_Accumulate_IT() must be called for each input buffer,
             except for the last one.
             User must resort to HAL_HASHEx_xxx_Accumulate_End_IT() to enter the last one and retrieve as
             well the computed digest.

        (##) In DMA mode, multi-buffer HASH and HMAC processing are possible.

              (+++) HASH processing: once initialization is done, MDMAT bit must be set through
               __HAL_HASH_SET_MDMAT() macro.
             From that point, each buffer can be fed to the Peripheral through HAL_HASHEx_xxx_Start_DMA() API.
             Before entering the last buffer, reset the MDMAT bit with __HAL_HASH_RESET_MDMAT()
             macro then wrap-up the HASH processing in feeding the last input buffer through the
             same API HAL_HASHEx_xxx_Start_DMA(). The digest can then be retrieved with a call to
             API HAL_HASHEx_xxx_Finish().

             (+++) HMAC processing (MD-5, SHA-1, SHA-224 and SHA-256 must all resort to
             extended functions): after initialization, the key and the first input buffer are entered
             in the Peripheral with the API HAL_HMACEx_xxx_Step1_2_DMA(). This carries out HMAC step 1 and
             starts step 2.
             The following buffers are next entered with the API  HAL_HMACEx_xxx_Step2_DMA(). At this
             point, the HMAC processing is still carrying out step 2.
             Then, step 2 for the last input buffer and step 3 are carried out by a single call
             to HAL_HMACEx_xxx_Step2_3_DMA().

             The digest can finally be retrieved with a call to API HAL_HASH_xxx_Finish() for
             MD-5 and SHA-1, to HAL_HASHEx_xxx_Finish() for SHA-224 and SHA-256.


  @endverbatim
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2016 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"




/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */
#if defined (HASH)

/** @defgroup HASHEx HASHEx
  * @brief HASH HAL extended module driver.
  * @{
  */
#ifdef HAL_HASH_MODULE_ENABLED
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
#if defined (HASH_CR_MDMAT)

/** @defgroup HASHEx_Exported_Functions HASH Extended Exported Functions
  * @{
  */

/** @defgroup HASHEx_Exported_Functions_Group1 HASH extended processing functions in polling mode
  *  @brief   HASH extended processing functions using polling mode.
  *
@verbatim
 ===============================================================================
               ##### Polling mode HASH extended processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in polling mode
          the hash value using one of the following algorithms:
      (+) SHA224
         (++) HAL_HASHEx_SHA224_Start()
         (++) HAL_HASHEx_SHA224_Accmlt()
         (++) HAL_HASHEx_SHA224_Accmlt_End()
      (+) SHA256
         (++) HAL_HASHEx_SHA256_Start()
         (++) HAL_HASHEx_SHA256_Accmlt()
         (++) HAL_HASHEx_SHA256_Accmlt_End()

    [..] For a single buffer to be hashed, user can resort to HAL_HASH_xxx_Start().

    [..]  In case of multi-buffer HASH processing (a single digest is computed while
          several buffers are fed to the Peripheral), the user can resort to successive calls
          to HAL_HASHEx_xxx_Accumulate() and wrap-up the digest computation by a call
          to HAL_HASHEx_xxx_Accumulate_End().

@endverbatim
  * @{
  */


/**
  * @brief  Initialize the HASH peripheral in SHA224 mode, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 28 bytes.
  * @param  Timeout Timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA224_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                          uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  If not already done, initialize the HASH peripheral in SHA224 mode then
  *         processes pInBuffer.
  * @note   Consecutive calls to HAL_HASHEx_SHA224_Accmlt() can be used to feed
  *         several input buffers back-to-back to the Peripheral that will yield a single
  *         HASH signature once all buffers have been entered. Wrap-up of input
  *         buffers feeding and retrieval of digest is done by a call to
  *         HAL_HASHEx_SHA224_Accmlt_End().
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   Digest is not retrieved by this API, user must resort to HAL_HASHEx_SHA224_Accmlt_End()
  *         to read it, feeding at the same time the last input buffer to the Peripheral.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted. Only HAL_HASHEx_SHA224_Accmlt_End() is able
  *         to manage the ending buffer with a length in bytes not a multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA224_Accmlt(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HASH_Accumulate(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  End computation of a single HASH signature after several calls to HAL_HASHEx_SHA224_Accmlt() API.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 28 bytes.
  * @param  Timeout Timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA224_Accmlt_End(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                               uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  Initialize the HASH peripheral in SHA256 mode, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 32 bytes.
  * @param  Timeout Timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA256_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                          uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_SHA256);
}

/**
  * @brief  If not already done, initialize the HASH peripheral in SHA256 mode then
  *         processes pInBuffer.
  * @note   Consecutive calls to HAL_HASHEx_SHA256_Accmlt() can be used to feed
  *         several input buffers back-to-back to the Peripheral that will yield a single
  *         HASH signature once all buffers have been entered. Wrap-up of input
  *         buffers feeding and retrieval of digest is done by a call to
  *         HAL_HASHEx_SHA256_Accmlt_End().
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   Digest is not retrieved by this API, user must resort to HAL_HASHEx_SHA256_Accmlt_End()
  *         to read it, feeding at the same time the last input buffer to the Peripheral.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted. Only HAL_HASHEx_SHA256_Accmlt_End() is able
  *         to manage the ending buffer with a length in bytes not a multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA256_Accmlt(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HASH_Accumulate(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA256);
}

/**
  * @brief  End computation of a single HASH signature after several calls to HAL_HASHEx_SHA256_Accmlt() API.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 32 bytes.
  * @param  Timeout Timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA256_Accmlt_End(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                               uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_SHA256);
}

/**
  * @}
  */

/** @defgroup HASHEx_Exported_Functions_Group2 HASH extended processing functions in interrupt mode
  *  @brief   HASH extended processing functions using interrupt mode.
  *
@verbatim
 ===============================================================================
          ##### Interruption mode HASH extended processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in interrupt mode
          the hash value using one of the following algorithms:
      (+) SHA224
         (++) HAL_HASHEx_SHA224_Start_IT()
         (++) HAL_HASHEx_SHA224_Accmlt_IT()
         (++) HAL_HASHEx_SHA224_Accmlt_End_IT()
      (+) SHA256
         (++) HAL_HASHEx_SHA256_Start_IT()
         (++) HAL_HASHEx_SHA256_Accmlt_IT()
         (++) HAL_HASHEx_SHA256_Accmlt_End_IT()

@endverbatim
  * @{
  */


/**
  * @brief  Initialize the HASH peripheral in SHA224 mode, next process pInBuffer then
  *         read the computed digest in interruption mode.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 28 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA224_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                             uint8_t *pOutBuffer)
{
  return HASH_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  If not already done, initialize the HASH peripheral in SHA224 mode then
  *         processes pInBuffer in interruption mode.
  * @note   Consecutive calls to HAL_HASHEx_SHA224_Accmlt_IT() can be used to feed
  *         several input buffers back-to-back to the Peripheral that will yield a single
  *         HASH signature once all buffers have been entered. Wrap-up of input
  *         buffers feeding and retrieval of digest is done by a call to
  *         HAL_HASHEx_SHA224_Accmlt_End_IT().
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted. Only HAL_HASHEx_SHA224_Accmlt_End_IT() is able
  *         to manage the ending buffer with a length in bytes not a multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA224_Accmlt_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HASH_Accumulate_IT(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  End computation of a single HASH signature after several calls to HAL_HASHEx_SHA224_Accmlt_IT() API.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 28 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA224_Accmlt_End_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                                  uint8_t *pOutBuffer)
{
  return HASH_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  Initialize the HASH peripheral in SHA256 mode, next process pInBuffer then
  *         read the computed digest in interruption mode.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 32 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA256_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                             uint8_t *pOutBuffer)
{
  return HASH_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_SHA256);
}

/**
  * @brief  If not already done, initialize the HASH peripheral in SHA256 mode then
  *         processes pInBuffer in interruption mode.
  * @note   Consecutive calls to HAL_HASHEx_SHA256_Accmlt_IT() can be used to feed
  *         several input buffers back-to-back to the Peripheral that will yield a single
  *         HASH signature once all buffers have been entered. Wrap-up of input
  *         buffers feeding and retrieval of digest is done by a call to
  *         HAL_HASHEx_SHA256_Accmlt_End_IT().
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted. Only HAL_HASHEx_SHA256_Accmlt_End_IT() is able
  *         to manage the ending buffer with a length in bytes not a multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA256_Accmlt_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HASH_Accumulate_IT(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA256);
}

/**
  * @brief  End computation of a single HASH signature after several calls to HAL_HASHEx_SHA256_Accmlt_IT() API.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 32 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA256_Accmlt_End_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                                  uint8_t *pOutBuffer)
{
  return HASH_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_SHA256);
}

/**
  * @}
  */

/** @defgroup HASHEx_Exported_Functions_Group3 HASH extended processing functions in DMA mode
  *  @brief   HASH extended processing functions using DMA mode.
  *
@verbatim
 ===============================================================================
                ##### DMA mode HASH extended  processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in DMA mode
          the hash value using one of the following algorithms:
      (+) SHA224
         (++) HAL_HASHEx_SHA224_Start_DMA()
         (++) HAL_HASHEx_SHA224_Finish()
      (+) SHA256
         (++) HAL_HASHEx_SHA256_Start_DMA()
         (++) HAL_HASHEx_SHA256_Finish()

    [..]  When resorting to DMA mode to enter the data in the Peripheral, user must resort
          to  HAL_HASHEx_xxx_Start_DMA() then read the resulting digest with
          HAL_HASHEx_xxx_Finish().

    [..]  In case of multi-buffer HASH processing, MDMAT bit must first be set before
          the successive calls to HAL_HASHEx_xxx_Start_DMA(). Then, MDMAT bit needs to be
          reset before the last call to HAL_HASHEx_xxx_Start_DMA(). Digest is finally
          retrieved thanks to HAL_HASHEx_xxx_Finish().

@endverbatim
  * @{
  */




/**
  * @brief  Initialize the HASH peripheral in SHA224 mode then initiate a DMA transfer
  *         to feed the input buffer to the Peripheral.
  * @note   Once the DMA transfer is finished, HAL_HASHEx_SHA224_Finish() API must
  *         be called to retrieve the computed digest.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA224_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return HASH_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  Return the computed digest in SHA224 mode.
  * @note   The API waits for DCIS to be set then reads the computed digest.
  * @note   HAL_HASHEx_SHA224_Finish() can be used as well to retrieve the digest in
  *         HMAC SHA224 mode.
  * @param  hhash HASH handle.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 28 bytes.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA224_Finish(HASH_HandleTypeDef *hhash, uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Finish(hhash, pOutBuffer, Timeout);
}

/**
  * @brief  Initialize the HASH peripheral in SHA256 mode then initiate a DMA transfer
  *         to feed the input buffer to the Peripheral.
  * @note   Once the DMA transfer is finished, HAL_HASHEx_SHA256_Finish() API must
  *         be called to retrieve the computed digest.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA256_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return HASH_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA256);
}

/**
  * @brief  Return the computed digest in SHA256 mode.
  * @note   The API waits for DCIS to be set then reads the computed digest.
  * @note   HAL_HASHEx_SHA256_Finish() can be used as well to retrieve the digest in
  *         HMAC SHA256 mode.
  * @param  hhash HASH handle.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 32 bytes.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASHEx_SHA256_Finish(HASH_HandleTypeDef *hhash, uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Finish(hhash, pOutBuffer, Timeout);
}

/**
  * @}
  */

/** @defgroup HASHEx_Exported_Functions_Group4 HMAC extended processing functions in polling mode
  *  @brief   HMAC extended processing functions using polling mode.
  *
@verbatim
 ===============================================================================
             ##### Polling mode HMAC extended processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in polling mode
          the HMAC value using one of the following algorithms:
      (+) SHA224
         (++) HAL_HMACEx_SHA224_Start()
      (+) SHA256
         (++) HAL_HMACEx_SHA256_Start()

@endverbatim
  * @{
  */



/**
  * @brief  Initialize the HASH peripheral in HMAC SHA224 mode, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 28 bytes.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA224_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                          uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HMAC_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  Initialize the HASH peripheral in HMAC SHA256 mode, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 32 bytes.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA256_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                          uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HMAC_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_SHA256);
}

/**
  * @}
  */


/** @defgroup HASHEx_Exported_Functions_Group5 HMAC extended processing functions in interrupt mode
  *  @brief   HMAC extended processing functions using interruption mode.
  *
@verbatim
 ===============================================================================
             ##### Interrupt mode HMAC extended processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in interrupt mode
          the HMAC value using one of the following algorithms:
      (+) SHA224
         (++) HAL_HMACEx_SHA224_Start_IT()
      (+) SHA256
         (++) HAL_HMACEx_SHA256_Start_IT()

@endverbatim
  * @{
  */



/**
  * @brief  Initialize the HASH peripheral in HMAC SHA224 mode, next process pInBuffer then
  *         read the computed digest in interrupt mode.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 28 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA224_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                             uint8_t *pOutBuffer)
{
  return  HMAC_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  Initialize the HASH peripheral in HMAC SHA256 mode, next process pInBuffer then
  *         read the computed digest in interrupt mode.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 32 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA256_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                             uint8_t *pOutBuffer)
{
  return  HMAC_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_SHA256);
}




/**
  * @}
  */


/** @defgroup HASHEx_Exported_Functions_Group6 HMAC extended processing functions in DMA mode
  *  @brief   HMAC extended processing functions using DMA mode.
  *
@verbatim
 ===============================================================================
              ##### DMA mode HMAC extended processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in DMA mode
          the HMAC value using one of the following algorithms:
      (+) SHA224
         (++) HAL_HMACEx_SHA224_Start_DMA()
      (+) SHA256
         (++) HAL_HMACEx_SHA256_Start_DMA()

    [..]  When resorting to DMA mode to enter the data in the Peripheral for HMAC processing,
          user must resort to  HAL_HMACEx_xxx_Start_DMA() then read the resulting digest
          with HAL_HASHEx_xxx_Finish().


@endverbatim
  * @{
  */



/**
  * @brief  Initialize the HASH peripheral in HMAC SHA224 mode then initiate the required
  *         DMA transfers to feed the key and the input buffer to the Peripheral.
  * @note   Once the DMA transfers are finished (indicated by hhash->State set back
  *         to HAL_HASH_STATE_READY), HAL_HASHEx_SHA224_Finish() API must be called to retrieve
  *         the computed digest.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   If MDMAT bit is set before calling this function (multi-buffer
  *          HASH processing case), the input buffer size (in bytes) must be
  *          a multiple of 4 otherwise, the HASH digest computation is corrupted.
  *          For the processing of the last buffer of the thread, MDMAT bit must
  *          be reset and the buffer length (in bytes) doesn't have to be a
  *          multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA224_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  Initialize the HASH peripheral in HMAC SHA224 mode then initiate the required
  *         DMA transfers to feed the key and the input buffer to the Peripheral.
  * @note   Once the DMA transfers are finished (indicated by hhash->State set back
  *         to HAL_HASH_STATE_READY), HAL_HASHEx_SHA256_Finish() API must be called to retrieve
  *         the computed digest.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   If MDMAT bit is set before calling this function (multi-buffer
  *          HASH processing case), the input buffer size (in bytes) must be
  *          a multiple of 4 otherwise, the HASH digest computation is corrupted.
  *          For the processing of the last buffer of the thread, MDMAT bit must
  *          be reset and the buffer length (in bytes) doesn't have to be a
  *          multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA256_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA256);
}


/**
  * @}
  */

/** @defgroup HASHEx_Exported_Functions_Group7 Multi-buffer HMAC extended processing functions in DMA mode
  *  @brief   HMAC extended processing functions in multi-buffer DMA mode.
  *
@verbatim
 ===============================================================================
      ##### Multi-buffer DMA mode HMAC extended processing functions #####
 ===============================================================================
    [..]  This section provides functions to manage HMAC multi-buffer
          DMA-based processing for MD5, SHA1, SHA224 and SHA256 algorithms.
      (+) MD5
         (++) HAL_HMACEx_MD5_Step1_2_DMA()
         (++) HAL_HMACEx_MD5_Step2_DMA()
         (++) HAL_HMACEx_MD5_Step2_3_DMA()
      (+) SHA1
         (++) HAL_HMACEx_SHA1_Step1_2_DMA()
         (++) HAL_HMACEx_SHA1_Step2_DMA()
         (++) HAL_HMACEx_SHA1_Step2_3_DMA()

      (+) SHA256
         (++) HAL_HMACEx_SHA224_Step1_2_DMA()
         (++) HAL_HMACEx_SHA224_Step2_DMA()
         (++) HAL_HMACEx_SHA224_Step2_3_DMA()
      (+) SHA256
         (++) HAL_HMACEx_SHA256_Step1_2_DMA()
         (++) HAL_HMACEx_SHA256_Step2_DMA()
         (++) HAL_HMACEx_SHA256_Step2_3_DMA()

    [..]  User must first start-up the multi-buffer DMA-based HMAC computation in
          calling HAL_HMACEx_xxx_Step1_2_DMA(). This carries out HMAC step 1 and
          intiates step 2 with the first input buffer.

    [..]  The following buffers are next fed to the Peripheral with a call to the API
          HAL_HMACEx_xxx_Step2_DMA(). There may be several consecutive calls
          to this API.

    [..]  Multi-buffer DMA-based HMAC computation is wrapped up by a call to
          HAL_HMACEx_xxx_Step2_3_DMA(). This finishes step 2 in feeding the last input
          buffer to the Peripheral then carries out step 3.

    [..]  Digest is retrieved by a call to HAL_HASH_xxx_Finish() for MD-5 or
          SHA-1, to HAL_HASHEx_xxx_Finish() for SHA-224 or SHA-256.

    [..]  If only two buffers need to be consecutively processed, a call to
          HAL_HMACEx_xxx_Step1_2_DMA() followed by a call to HAL_HMACEx_xxx_Step2_3_DMA()
          is sufficient.

@endverbatim
  * @{
  */

/**
  * @brief  MD5 HMAC step 1 completion and step 2 start in multi-buffer DMA mode.
  * @note   Step 1 consists in writing the inner hash function key in the Peripheral,
  *         step 2 consists in writing the message text.
  * @note   The API carries out the HMAC step 1 then starts step 2 with
  *         the first buffer entered to the Peripheral. DCAL bit is not automatically set after
  *         the message buffer feeding, allowing other messages DMA transfers to occur.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_MD5_Step1_2_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  hhash->DigestCalculationDisable = SET;
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  MD5 HMAC step 2 in multi-buffer DMA mode.
  * @note   Step 2 consists in writing the message text in the Peripheral.
  * @note   The API carries on the HMAC step 2, applied to the buffer entered as input
  *         parameter. DCAL bit is not automatically set after the message buffer feeding,
  *         allowing other messages DMA transfers to occur.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_MD5_Step2_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  if (hhash->DigestCalculationDisable != SET)
  {
    return HAL_ERROR;
  }
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  MD5 HMAC step 2 wrap-up and step 3 completion in multi-buffer DMA mode.
  * @note   Step 2 consists in writing the message text in the Peripheral,
  *         step 3 consists in writing the outer hash function key.
  * @note   The API wraps up the HMAC step 2 in processing the buffer entered as input
  *         parameter (the input buffer must be the last one of the multi-buffer thread)
  *         then carries out HMAC step 3.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   Once the DMA transfers are finished (indicated by hhash->State set back
  *         to HAL_HASH_STATE_READY), HAL_HASHEx_SHA256_Finish() API must be called to retrieve
  *         the computed digest.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_MD5_Step2_3_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  hhash->DigestCalculationDisable = RESET;
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_MD5);
}


/**
  * @brief  SHA1 HMAC step 1 completion and step 2 start in multi-buffer DMA mode.
  * @note   Step 1 consists in writing the inner hash function key in the Peripheral,
  *         step 2 consists in writing the message text.
  * @note   The API carries out the HMAC step 1 then starts step 2 with
  *         the first buffer entered to the Peripheral. DCAL bit is not automatically set after
  *         the message buffer feeding, allowing other messages DMA transfers to occur.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA1_Step1_2_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  hhash->DigestCalculationDisable = SET;
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA1);
}

/**
  * @brief  SHA1 HMAC step 2 in multi-buffer DMA mode.
  * @note   Step 2 consists in writing the message text in the Peripheral.
  * @note   The API carries on the HMAC step 2, applied to the buffer entered as input
  *         parameter. DCAL bit is not automatically set after the message buffer feeding,
  *         allowing other messages DMA transfers to occur.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA1_Step2_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  if (hhash->DigestCalculationDisable != SET)
  {
    return HAL_ERROR;
  }
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA1);
}

/**
  * @brief  SHA1 HMAC step 2 wrap-up and step 3 completion in multi-buffer DMA mode.
  * @note   Step 2 consists in writing the message text in the Peripheral,
  *         step 3 consists in writing the outer hash function key.
  * @note   The API wraps up the HMAC step 2 in processing the buffer entered as input
  *         parameter (the input buffer must be the last one of the multi-buffer thread)
  *         then carries out HMAC step 3.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   Once the DMA transfers are finished (indicated by hhash->State set back
  *         to HAL_HASH_STATE_READY), HAL_HASHEx_SHA256_Finish() API must be called to retrieve
  *         the computed digest.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA1_Step2_3_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  hhash->DigestCalculationDisable = RESET;
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA1);
}

/**
  * @brief  SHA224 HMAC step 1 completion and step 2 start in multi-buffer DMA mode.
  * @note   Step 1 consists in writing the inner hash function key in the Peripheral,
  *         step 2 consists in writing the message text.
  * @note   The API carries out the HMAC step 1 then starts step 2 with
  *         the first buffer entered to the Peripheral. DCAL bit is not automatically set after
  *         the message buffer feeding, allowing other messages DMA transfers to occur.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA224_Step1_2_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  hhash->DigestCalculationDisable = SET;
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  SHA224 HMAC step 2 in multi-buffer DMA mode.
  * @note   Step 2 consists in writing the message text in the Peripheral.
  * @note   The API carries on the HMAC step 2, applied to the buffer entered as input
  *         parameter. DCAL bit is not automatically set after the message buffer feeding,
  *         allowing other messages DMA transfers to occur.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA224_Step2_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  if (hhash->DigestCalculationDisable != SET)
  {
    return HAL_ERROR;
  }
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  SHA224 HMAC step 2 wrap-up and step 3 completion in multi-buffer DMA mode.
  * @note   Step 2 consists in writing the message text in the Peripheral,
  *         step 3 consists in writing the outer hash function key.
  * @note   The API wraps up the HMAC step 2 in processing the buffer entered as input
  *         parameter (the input buffer must be the last one of the multi-buffer thread)
  *         then carries out HMAC step 3.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   Once the DMA transfers are finished (indicated by hhash->State set back
  *         to HAL_HASH_STATE_READY), HAL_HASHEx_SHA256_Finish() API must be called to retrieve
  *         the computed digest.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA224_Step2_3_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  hhash->DigestCalculationDisable = RESET;
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA224);
}

/**
  * @brief  SHA256 HMAC step 1 completion and step 2 start in multi-buffer DMA mode.
  * @note   Step 1 consists in writing the inner hash function key in the Peripheral,
  *         step 2 consists in writing the message text.
  * @note   The API carries out the HMAC step 1 then starts step 2 with
  *         the first buffer entered to the Peripheral. DCAL bit is not automatically set after
  *         the message buffer feeding, allowing other messages DMA transfers to occur.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA256_Step1_2_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  hhash->DigestCalculationDisable = SET;
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA256);
}

/**
  * @brief  SHA256 HMAC step 2 in multi-buffer DMA mode.
  * @note   Step 2 consists in writing the message text in the Peripheral.
  * @note   The API carries on the HMAC step 2, applied to the buffer entered as input
  *         parameter. DCAL bit is not automatically set after the message buffer feeding,
  *         allowing other messages DMA transfers to occur.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA256_Step2_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  if (hhash->DigestCalculationDisable != SET)
  {
    return HAL_ERROR;
  }
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA256);
}

/**
  * @brief  SHA256 HMAC step 2 wrap-up and step 3 completion in multi-buffer DMA mode.
  * @note   Step 2 consists in writing the message text in the Peripheral,
  *         step 3 consists in writing the outer hash function key.
  * @note   The API wraps up the HMAC step 2 in processing the buffer entered as input
  *         parameter (the input buffer must be the last one of the multi-buffer thread)
  *         then carries out HMAC step 3.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   Once the DMA transfers are finished (indicated by hhash->State set back
  *         to HAL_HASH_STATE_READY), HAL_HASHEx_SHA256_Finish() API must be called to retrieve
  *         the computed digest.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (message buffer).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMACEx_SHA256_Step2_3_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  hhash->DigestCalculationDisable = RESET;
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA256);
}

/**
  * @}
  */

#endif /* MDMA defined*/
/**
  * @}
  */
#endif /* HAL_HASH_MODULE_ENABLED */

/**
  * @}
  */
#endif /*  HASH*/
/**
  * @}
  */



/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
