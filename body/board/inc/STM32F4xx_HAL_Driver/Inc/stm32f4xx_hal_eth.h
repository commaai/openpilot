/**
  ******************************************************************************
  * @file    stm32f4xx_hal_eth.h
  * @author  MCD Application Team
  * @brief   Header file of ETH HAL module.
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
#ifndef __STM32F4xx_HAL_ETH_H
#define __STM32F4xx_HAL_ETH_H

#ifdef __cplusplus
 extern "C" {
#endif

#if defined(STM32F407xx) || defined(STM32F417xx) || defined(STM32F427xx) || defined(STM32F437xx) ||\
    defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup ETH
  * @{
  */ 
  
/** @addtogroup ETH_Private_Macros
  * @{
  */
#define IS_ETH_PHY_ADDRESS(ADDRESS) ((ADDRESS) <= 0x20U)
#define IS_ETH_AUTONEGOTIATION(CMD) (((CMD) == ETH_AUTONEGOTIATION_ENABLE) || \
                                     ((CMD) == ETH_AUTONEGOTIATION_DISABLE))
#define IS_ETH_SPEED(SPEED) (((SPEED) == ETH_SPEED_10M) || \
                             ((SPEED) == ETH_SPEED_100M))
#define IS_ETH_DUPLEX_MODE(MODE)  (((MODE) == ETH_MODE_FULLDUPLEX) || \
                                  ((MODE) == ETH_MODE_HALFDUPLEX))
#define IS_ETH_RX_MODE(MODE)    (((MODE) == ETH_RXPOLLING_MODE) || \
                                 ((MODE) == ETH_RXINTERRUPT_MODE)) 
#define IS_ETH_CHECKSUM_MODE(MODE)    (((MODE) == ETH_CHECKSUM_BY_HARDWARE) || \
                                      ((MODE) == ETH_CHECKSUM_BY_SOFTWARE))
#define IS_ETH_MEDIA_INTERFACE(MODE)         (((MODE) == ETH_MEDIA_INTERFACE_MII) || \
                                              ((MODE) == ETH_MEDIA_INTERFACE_RMII))
#define IS_ETH_WATCHDOG(CMD) (((CMD) == ETH_WATCHDOG_ENABLE) || \
                              ((CMD) == ETH_WATCHDOG_DISABLE))
#define IS_ETH_JABBER(CMD) (((CMD) == ETH_JABBER_ENABLE) || \
                            ((CMD) == ETH_JABBER_DISABLE))
#define IS_ETH_INTER_FRAME_GAP(GAP) (((GAP) == ETH_INTERFRAMEGAP_96BIT) || \
                                     ((GAP) == ETH_INTERFRAMEGAP_88BIT) || \
                                     ((GAP) == ETH_INTERFRAMEGAP_80BIT) || \
                                     ((GAP) == ETH_INTERFRAMEGAP_72BIT) || \
                                     ((GAP) == ETH_INTERFRAMEGAP_64BIT) || \
                                     ((GAP) == ETH_INTERFRAMEGAP_56BIT) || \
                                     ((GAP) == ETH_INTERFRAMEGAP_48BIT) || \
                                     ((GAP) == ETH_INTERFRAMEGAP_40BIT))
#define IS_ETH_CARRIER_SENSE(CMD) (((CMD) == ETH_CARRIERSENCE_ENABLE) || \
                                   ((CMD) == ETH_CARRIERSENCE_DISABLE))
#define IS_ETH_RECEIVE_OWN(CMD) (((CMD) == ETH_RECEIVEOWN_ENABLE) || \
                                 ((CMD) == ETH_RECEIVEOWN_DISABLE))
#define IS_ETH_LOOPBACK_MODE(CMD) (((CMD) == ETH_LOOPBACKMODE_ENABLE) || \
                                   ((CMD) == ETH_LOOPBACKMODE_DISABLE))
#define IS_ETH_CHECKSUM_OFFLOAD(CMD) (((CMD) == ETH_CHECKSUMOFFLAOD_ENABLE) || \
                                      ((CMD) == ETH_CHECKSUMOFFLAOD_DISABLE))
#define IS_ETH_RETRY_TRANSMISSION(CMD) (((CMD) == ETH_RETRYTRANSMISSION_ENABLE) || \
                                        ((CMD) == ETH_RETRYTRANSMISSION_DISABLE))
#define IS_ETH_AUTOMATIC_PADCRC_STRIP(CMD) (((CMD) == ETH_AUTOMATICPADCRCSTRIP_ENABLE) || \
                                            ((CMD) == ETH_AUTOMATICPADCRCSTRIP_DISABLE))
#define IS_ETH_BACKOFF_LIMIT(LIMIT) (((LIMIT) == ETH_BACKOFFLIMIT_10) || \
                                     ((LIMIT) == ETH_BACKOFFLIMIT_8) || \
                                     ((LIMIT) == ETH_BACKOFFLIMIT_4) || \
                                     ((LIMIT) == ETH_BACKOFFLIMIT_1))
#define IS_ETH_DEFERRAL_CHECK(CMD) (((CMD) == ETH_DEFFERRALCHECK_ENABLE) || \
                                    ((CMD) == ETH_DEFFERRALCHECK_DISABLE))
#define IS_ETH_RECEIVE_ALL(CMD) (((CMD) == ETH_RECEIVEALL_ENABLE) || \
                                 ((CMD) == ETH_RECEIVEAll_DISABLE))
#define IS_ETH_SOURCE_ADDR_FILTER(CMD) (((CMD) == ETH_SOURCEADDRFILTER_NORMAL_ENABLE) || \
                                        ((CMD) == ETH_SOURCEADDRFILTER_INVERSE_ENABLE) || \
                                        ((CMD) == ETH_SOURCEADDRFILTER_DISABLE))
#define IS_ETH_CONTROL_FRAMES(PASS) (((PASS) == ETH_PASSCONTROLFRAMES_BLOCKALL) || \
                                     ((PASS) == ETH_PASSCONTROLFRAMES_FORWARDALL) || \
                                     ((PASS) == ETH_PASSCONTROLFRAMES_FORWARDPASSEDADDRFILTER))
#define IS_ETH_BROADCAST_FRAMES_RECEPTION(CMD) (((CMD) == ETH_BROADCASTFRAMESRECEPTION_ENABLE) || \
                                                ((CMD) == ETH_BROADCASTFRAMESRECEPTION_DISABLE))
#define IS_ETH_DESTINATION_ADDR_FILTER(FILTER) (((FILTER) == ETH_DESTINATIONADDRFILTER_NORMAL) || \
                                                ((FILTER) == ETH_DESTINATIONADDRFILTER_INVERSE))
#define IS_ETH_PROMISCUOUS_MODE(CMD) (((CMD) == ETH_PROMISCUOUS_MODE_ENABLE) || \
                                      ((CMD) == ETH_PROMISCUOUS_MODE_DISABLE))
#define IS_ETH_MULTICAST_FRAMES_FILTER(FILTER) (((FILTER) == ETH_MULTICASTFRAMESFILTER_PERFECTHASHTABLE) || \
                                                ((FILTER) == ETH_MULTICASTFRAMESFILTER_HASHTABLE) || \
                                                ((FILTER) == ETH_MULTICASTFRAMESFILTER_PERFECT) || \
                                                ((FILTER) == ETH_MULTICASTFRAMESFILTER_NONE))
#define IS_ETH_UNICAST_FRAMES_FILTER(FILTER) (((FILTER) == ETH_UNICASTFRAMESFILTER_PERFECTHASHTABLE) || \
                                              ((FILTER) == ETH_UNICASTFRAMESFILTER_HASHTABLE) || \
                                              ((FILTER) == ETH_UNICASTFRAMESFILTER_PERFECT))
#define IS_ETH_PAUSE_TIME(TIME) ((TIME) <= 0xFFFFU)
#define IS_ETH_ZEROQUANTA_PAUSE(CMD)   (((CMD) == ETH_ZEROQUANTAPAUSE_ENABLE) || \
                                        ((CMD) == ETH_ZEROQUANTAPAUSE_DISABLE))
#define IS_ETH_PAUSE_LOW_THRESHOLD(THRESHOLD) (((THRESHOLD) == ETH_PAUSELOWTHRESHOLD_MINUS4) || \
                                               ((THRESHOLD) == ETH_PAUSELOWTHRESHOLD_MINUS28) || \
                                               ((THRESHOLD) == ETH_PAUSELOWTHRESHOLD_MINUS144) || \
                                               ((THRESHOLD) == ETH_PAUSELOWTHRESHOLD_MINUS256))
#define IS_ETH_UNICAST_PAUSE_FRAME_DETECT(CMD) (((CMD) == ETH_UNICASTPAUSEFRAMEDETECT_ENABLE) || \
                                                ((CMD) == ETH_UNICASTPAUSEFRAMEDETECT_DISABLE))
#define IS_ETH_RECEIVE_FLOWCONTROL(CMD) (((CMD) == ETH_RECEIVEFLOWCONTROL_ENABLE) || \
                                         ((CMD) == ETH_RECEIVEFLOWCONTROL_DISABLE))
#define IS_ETH_TRANSMIT_FLOWCONTROL(CMD) (((CMD) == ETH_TRANSMITFLOWCONTROL_ENABLE) || \
                                          ((CMD) == ETH_TRANSMITFLOWCONTROL_DISABLE))
#define IS_ETH_VLAN_TAG_COMPARISON(COMPARISON) (((COMPARISON) == ETH_VLANTAGCOMPARISON_12BIT) || \
                                                ((COMPARISON) == ETH_VLANTAGCOMPARISON_16BIT))
#define IS_ETH_VLAN_TAG_IDENTIFIER(IDENTIFIER) ((IDENTIFIER) <= 0xFFFFU)
#define IS_ETH_MAC_ADDRESS0123(ADDRESS) (((ADDRESS) == ETH_MAC_ADDRESS0) || \
                                         ((ADDRESS) == ETH_MAC_ADDRESS1) || \
                                         ((ADDRESS) == ETH_MAC_ADDRESS2) || \
                                         ((ADDRESS) == ETH_MAC_ADDRESS3))
#define IS_ETH_MAC_ADDRESS123(ADDRESS) (((ADDRESS) == ETH_MAC_ADDRESS1) || \
                                        ((ADDRESS) == ETH_MAC_ADDRESS2) || \
                                        ((ADDRESS) == ETH_MAC_ADDRESS3))
#define IS_ETH_MAC_ADDRESS_FILTER(FILTER) (((FILTER) == ETH_MAC_ADDRESSFILTER_SA) || \
                                           ((FILTER) == ETH_MAC_ADDRESSFILTER_DA))
#define IS_ETH_MAC_ADDRESS_MASK(MASK) (((MASK) == ETH_MAC_ADDRESSMASK_BYTE6) || \
                                       ((MASK) == ETH_MAC_ADDRESSMASK_BYTE5) || \
                                       ((MASK) == ETH_MAC_ADDRESSMASK_BYTE4) || \
                                       ((MASK) == ETH_MAC_ADDRESSMASK_BYTE3) || \
                                       ((MASK) == ETH_MAC_ADDRESSMASK_BYTE2) || \
                                       ((MASK) == ETH_MAC_ADDRESSMASK_BYTE1))
#define IS_ETH_DROP_TCPIP_CHECKSUM_FRAME(CMD) (((CMD) == ETH_DROPTCPIPCHECKSUMERRORFRAME_ENABLE) || \
                                               ((CMD) == ETH_DROPTCPIPCHECKSUMERRORFRAME_DISABLE))
#define IS_ETH_RECEIVE_STORE_FORWARD(CMD) (((CMD) == ETH_RECEIVESTOREFORWARD_ENABLE) || \
                                           ((CMD) == ETH_RECEIVESTOREFORWARD_DISABLE))
#define IS_ETH_FLUSH_RECEIVE_FRAME(CMD) (((CMD) == ETH_FLUSHRECEIVEDFRAME_ENABLE) || \
                                         ((CMD) == ETH_FLUSHRECEIVEDFRAME_DISABLE))
#define IS_ETH_TRANSMIT_STORE_FORWARD(CMD) (((CMD) == ETH_TRANSMITSTOREFORWARD_ENABLE) || \
                                            ((CMD) == ETH_TRANSMITSTOREFORWARD_DISABLE))
#define IS_ETH_TRANSMIT_THRESHOLD_CONTROL(THRESHOLD) (((THRESHOLD) == ETH_TRANSMITTHRESHOLDCONTROL_64BYTES) || \
                                                      ((THRESHOLD) == ETH_TRANSMITTHRESHOLDCONTROL_128BYTES) || \
                                                      ((THRESHOLD) == ETH_TRANSMITTHRESHOLDCONTROL_192BYTES) || \
                                                      ((THRESHOLD) == ETH_TRANSMITTHRESHOLDCONTROL_256BYTES) || \
                                                      ((THRESHOLD) == ETH_TRANSMITTHRESHOLDCONTROL_40BYTES) || \
                                                      ((THRESHOLD) == ETH_TRANSMITTHRESHOLDCONTROL_32BYTES) || \
                                                      ((THRESHOLD) == ETH_TRANSMITTHRESHOLDCONTROL_24BYTES) || \
                                                      ((THRESHOLD) == ETH_TRANSMITTHRESHOLDCONTROL_16BYTES))
#define IS_ETH_FORWARD_ERROR_FRAMES(CMD) (((CMD) == ETH_FORWARDERRORFRAMES_ENABLE) || \
                                          ((CMD) == ETH_FORWARDERRORFRAMES_DISABLE))
#define IS_ETH_FORWARD_UNDERSIZED_GOOD_FRAMES(CMD) (((CMD) == ETH_FORWARDUNDERSIZEDGOODFRAMES_ENABLE) || \
                                                    ((CMD) == ETH_FORWARDUNDERSIZEDGOODFRAMES_DISABLE))
#define IS_ETH_RECEIVE_THRESHOLD_CONTROL(THRESHOLD) (((THRESHOLD) == ETH_RECEIVEDTHRESHOLDCONTROL_64BYTES) || \
                                                     ((THRESHOLD) == ETH_RECEIVEDTHRESHOLDCONTROL_32BYTES) || \
                                                     ((THRESHOLD) == ETH_RECEIVEDTHRESHOLDCONTROL_96BYTES) || \
                                                     ((THRESHOLD) == ETH_RECEIVEDTHRESHOLDCONTROL_128BYTES))
#define IS_ETH_SECOND_FRAME_OPERATE(CMD) (((CMD) == ETH_SECONDFRAMEOPERARTE_ENABLE) || \
                                          ((CMD) == ETH_SECONDFRAMEOPERARTE_DISABLE))
#define IS_ETH_ADDRESS_ALIGNED_BEATS(CMD) (((CMD) == ETH_ADDRESSALIGNEDBEATS_ENABLE) || \
                                           ((CMD) == ETH_ADDRESSALIGNEDBEATS_DISABLE))
#define IS_ETH_FIXED_BURST(CMD) (((CMD) == ETH_FIXEDBURST_ENABLE) || \
                                 ((CMD) == ETH_FIXEDBURST_DISABLE))
#define IS_ETH_RXDMA_BURST_LENGTH(LENGTH) (((LENGTH) == ETH_RXDMABURSTLENGTH_1BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_2BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_4BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_8BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_16BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_32BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_4XPBL_4BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_4XPBL_8BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_4XPBL_16BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_4XPBL_32BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_4XPBL_64BEAT) || \
                                           ((LENGTH) == ETH_RXDMABURSTLENGTH_4XPBL_128BEAT))
#define IS_ETH_TXDMA_BURST_LENGTH(LENGTH) (((LENGTH) == ETH_TXDMABURSTLENGTH_1BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_2BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_4BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_8BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_16BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_32BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_4XPBL_4BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_4XPBL_8BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_4XPBL_16BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_4XPBL_32BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_4XPBL_64BEAT) || \
                                           ((LENGTH) == ETH_TXDMABURSTLENGTH_4XPBL_128BEAT))
#define IS_ETH_DMA_DESC_SKIP_LENGTH(LENGTH) ((LENGTH) <= 0x1FU)
#define IS_ETH_DMA_ARBITRATION_ROUNDROBIN_RXTX(RATIO) (((RATIO) == ETH_DMAARBITRATION_ROUNDROBIN_RXTX_1_1) || \
                                                       ((RATIO) == ETH_DMAARBITRATION_ROUNDROBIN_RXTX_2_1) || \
                                                       ((RATIO) == ETH_DMAARBITRATION_ROUNDROBIN_RXTX_3_1) || \
                                                       ((RATIO) == ETH_DMAARBITRATION_ROUNDROBIN_RXTX_4_1) || \
                                                       ((RATIO) == ETH_DMAARBITRATION_RXPRIORTX))
#define IS_ETH_DMATXDESC_GET_FLAG(FLAG) (((FLAG) == ETH_DMATXDESC_OWN) || \
                                         ((FLAG) == ETH_DMATXDESC_IC) || \
                                         ((FLAG) == ETH_DMATXDESC_LS) || \
                                         ((FLAG) == ETH_DMATXDESC_FS) || \
                                         ((FLAG) == ETH_DMATXDESC_DC) || \
                                         ((FLAG) == ETH_DMATXDESC_DP) || \
                                         ((FLAG) == ETH_DMATXDESC_TTSE) || \
                                         ((FLAG) == ETH_DMATXDESC_TER) || \
                                         ((FLAG) == ETH_DMATXDESC_TCH) || \
                                         ((FLAG) == ETH_DMATXDESC_TTSS) || \
                                         ((FLAG) == ETH_DMATXDESC_IHE) || \
                                         ((FLAG) == ETH_DMATXDESC_ES) || \
                                         ((FLAG) == ETH_DMATXDESC_JT) || \
                                         ((FLAG) == ETH_DMATXDESC_FF) || \
                                         ((FLAG) == ETH_DMATXDESC_PCE) || \
                                         ((FLAG) == ETH_DMATXDESC_LCA) || \
                                         ((FLAG) == ETH_DMATXDESC_NC) || \
                                         ((FLAG) == ETH_DMATXDESC_LCO) || \
                                         ((FLAG) == ETH_DMATXDESC_EC) || \
                                         ((FLAG) == ETH_DMATXDESC_VF) || \
                                         ((FLAG) == ETH_DMATXDESC_CC) || \
                                         ((FLAG) == ETH_DMATXDESC_ED) || \
                                         ((FLAG) == ETH_DMATXDESC_UF) || \
                                         ((FLAG) == ETH_DMATXDESC_DB))
#define IS_ETH_DMA_TXDESC_SEGMENT(SEGMENT) (((SEGMENT) == ETH_DMATXDESC_LASTSEGMENTS) || \
                                            ((SEGMENT) == ETH_DMATXDESC_FIRSTSEGMENT))
#define IS_ETH_DMA_TXDESC_CHECKSUM(CHECKSUM) (((CHECKSUM) == ETH_DMATXDESC_CHECKSUMBYPASS) || \
                                              ((CHECKSUM) == ETH_DMATXDESC_CHECKSUMIPV4HEADER) || \
                                              ((CHECKSUM) == ETH_DMATXDESC_CHECKSUMTCPUDPICMPSEGMENT) || \
                                              ((CHECKSUM) == ETH_DMATXDESC_CHECKSUMTCPUDPICMPFULL))
#define IS_ETH_DMATXDESC_BUFFER_SIZE(SIZE) ((SIZE) <= 0x1FFFU)
#define IS_ETH_DMARXDESC_GET_FLAG(FLAG) (((FLAG) == ETH_DMARXDESC_OWN) || \
                                         ((FLAG) == ETH_DMARXDESC_AFM) || \
                                         ((FLAG) == ETH_DMARXDESC_ES) || \
                                         ((FLAG) == ETH_DMARXDESC_DE) || \
                                         ((FLAG) == ETH_DMARXDESC_SAF) || \
                                         ((FLAG) == ETH_DMARXDESC_LE) || \
                                         ((FLAG) == ETH_DMARXDESC_OE) || \
                                         ((FLAG) == ETH_DMARXDESC_VLAN) || \
                                         ((FLAG) == ETH_DMARXDESC_FS) || \
                                         ((FLAG) == ETH_DMARXDESC_LS) || \
                                         ((FLAG) == ETH_DMARXDESC_IPV4HCE) || \
                                         ((FLAG) == ETH_DMARXDESC_LC) || \
                                         ((FLAG) == ETH_DMARXDESC_FT) || \
                                         ((FLAG) == ETH_DMARXDESC_RWT) || \
                                         ((FLAG) == ETH_DMARXDESC_RE) || \
                                         ((FLAG) == ETH_DMARXDESC_DBE) || \
                                         ((FLAG) == ETH_DMARXDESC_CE) || \
                                         ((FLAG) == ETH_DMARXDESC_MAMPCE))
#define IS_ETH_DMA_RXDESC_BUFFER(BUFFER) (((BUFFER) == ETH_DMARXDESC_BUFFER1) || \
                                          ((BUFFER) == ETH_DMARXDESC_BUFFER2))
#define IS_ETH_PMT_GET_FLAG(FLAG) (((FLAG) == ETH_PMT_FLAG_WUFR) || \
                                   ((FLAG) == ETH_PMT_FLAG_MPR))
#define IS_ETH_DMA_FLAG(FLAG) ((((FLAG) & 0xC7FE1800U) == 0x00U) && ((FLAG) != 0x00U)) 
#define IS_ETH_DMA_GET_FLAG(FLAG) (((FLAG) == ETH_DMA_FLAG_TST) || ((FLAG) == ETH_DMA_FLAG_PMT) || \
                                   ((FLAG) == ETH_DMA_FLAG_MMC) || ((FLAG) == ETH_DMA_FLAG_DATATRANSFERERROR) || \
                                   ((FLAG) == ETH_DMA_FLAG_READWRITEERROR) || ((FLAG) == ETH_DMA_FLAG_ACCESSERROR) || \
                                   ((FLAG) == ETH_DMA_FLAG_NIS) || ((FLAG) == ETH_DMA_FLAG_AIS) || \
                                   ((FLAG) == ETH_DMA_FLAG_ER) || ((FLAG) == ETH_DMA_FLAG_FBE) || \
                                   ((FLAG) == ETH_DMA_FLAG_ET) || ((FLAG) == ETH_DMA_FLAG_RWT) || \
                                   ((FLAG) == ETH_DMA_FLAG_RPS) || ((FLAG) == ETH_DMA_FLAG_RBU) || \
                                   ((FLAG) == ETH_DMA_FLAG_R) || ((FLAG) == ETH_DMA_FLAG_TU) || \
                                   ((FLAG) == ETH_DMA_FLAG_RO) || ((FLAG) == ETH_DMA_FLAG_TJT) || \
                                   ((FLAG) == ETH_DMA_FLAG_TBU) || ((FLAG) == ETH_DMA_FLAG_TPS) || \
                                   ((FLAG) == ETH_DMA_FLAG_T))
#define IS_ETH_MAC_IT(IT) ((((IT) & 0xFFFFFDF1U) == 0x00U) && ((IT) != 0x00U))
#define IS_ETH_MAC_GET_IT(IT) (((IT) == ETH_MAC_IT_TST) || ((IT) == ETH_MAC_IT_MMCT) || \
                               ((IT) == ETH_MAC_IT_MMCR) || ((IT) == ETH_MAC_IT_MMC) || \
                               ((IT) == ETH_MAC_IT_PMT))
#define IS_ETH_MAC_GET_FLAG(FLAG) (((FLAG) == ETH_MAC_FLAG_TST) || ((FLAG) == ETH_MAC_FLAG_MMCT) || \
                                   ((FLAG) == ETH_MAC_FLAG_MMCR) || ((FLAG) == ETH_MAC_FLAG_MMC) || \
                                   ((FLAG) == ETH_MAC_FLAG_PMT))
#define IS_ETH_DMA_IT(IT) ((((IT) & 0xC7FE1800U) == 0x00U) && ((IT) != 0x00U))
#define IS_ETH_DMA_GET_IT(IT) (((IT) == ETH_DMA_IT_TST) || ((IT) == ETH_DMA_IT_PMT) || \
                               ((IT) == ETH_DMA_IT_MMC) || ((IT) == ETH_DMA_IT_NIS) || \
                               ((IT) == ETH_DMA_IT_AIS) || ((IT) == ETH_DMA_IT_ER) || \
                               ((IT) == ETH_DMA_IT_FBE) || ((IT) == ETH_DMA_IT_ET) || \
                               ((IT) == ETH_DMA_IT_RWT) || ((IT) == ETH_DMA_IT_RPS) || \
                               ((IT) == ETH_DMA_IT_RBU) || ((IT) == ETH_DMA_IT_R) || \
                               ((IT) == ETH_DMA_IT_TU) || ((IT) == ETH_DMA_IT_RO) || \
                               ((IT) == ETH_DMA_IT_TJT) || ((IT) == ETH_DMA_IT_TBU) || \
                               ((IT) == ETH_DMA_IT_TPS) || ((IT) == ETH_DMA_IT_T))
#define IS_ETH_DMA_GET_OVERFLOW(OVERFLOW) (((OVERFLOW) == ETH_DMA_OVERFLOW_RXFIFOCOUNTER) || \
                                           ((OVERFLOW) == ETH_DMA_OVERFLOW_MISSEDFRAMECOUNTER))
#define IS_ETH_MMC_IT(IT) (((((IT) & 0xFFDF3FFFU) == 0x00U) || (((IT) & 0xEFFDFF9FU) == 0x00U)) && \
                           ((IT) != 0x00U))
#define IS_ETH_MMC_GET_IT(IT) (((IT) == ETH_MMC_IT_TGF) || ((IT) == ETH_MMC_IT_TGFMSC) || \
                               ((IT) == ETH_MMC_IT_TGFSC) || ((IT) == ETH_MMC_IT_RGUF) || \
                               ((IT) == ETH_MMC_IT_RFAE) || ((IT) == ETH_MMC_IT_RFCE))
#define IS_ETH_ENHANCED_DESCRIPTOR_FORMAT(CMD) (((CMD) == ETH_DMAENHANCEDDESCRIPTOR_ENABLE) || \
                                                ((CMD) == ETH_DMAENHANCEDDESCRIPTOR_DISABLE))

/**
  * @}
  */

/** @addtogroup ETH_Private_Defines
  * @{
  */
/* Delay to wait when writing to some Ethernet registers */
#define ETH_REG_WRITE_DELAY     0x00000001U

/* ETHERNET Errors */
#define  ETH_SUCCESS            0U
#define  ETH_ERROR              1U

/* ETHERNET DMA Tx descriptors Collision Count Shift */
#define  ETH_DMATXDESC_COLLISION_COUNTSHIFT        3U

/* ETHERNET DMA Tx descriptors Buffer2 Size Shift */
#define  ETH_DMATXDESC_BUFFER2_SIZESHIFT           16U

/* ETHERNET DMA Rx descriptors Frame Length Shift */
#define  ETH_DMARXDESC_FRAME_LENGTHSHIFT           16U

/* ETHERNET DMA Rx descriptors Buffer2 Size Shift */
#define  ETH_DMARXDESC_BUFFER2_SIZESHIFT           16U

/* ETHERNET DMA Rx descriptors Frame length Shift */
#define  ETH_DMARXDESC_FRAMELENGTHSHIFT            16U

/* ETHERNET MAC address offsets */
#define ETH_MAC_ADDR_HBASE    (uint32_t)(ETH_MAC_BASE + 0x40U)  /* ETHERNET MAC address high offset */
#define ETH_MAC_ADDR_LBASE    (uint32_t)(ETH_MAC_BASE + 0x44U)  /* ETHERNET MAC address low offset */

/* ETHERNET MACMIIAR register Mask */
#define ETH_MACMIIAR_CR_MASK    0xFFFFFFE3U

/* ETHERNET MACCR register Mask */
#define ETH_MACCR_CLEAR_MASK    0xFF20810FU

/* ETHERNET MACFCR register Mask */
#define ETH_MACFCR_CLEAR_MASK   0x0000FF41U

/* ETHERNET DMAOMR register Mask */
#define ETH_DMAOMR_CLEAR_MASK   0xF8DE3F23U

/* ETHERNET Remote Wake-up frame register length */
#define ETH_WAKEUP_REGISTER_LENGTH      8U

/* ETHERNET Missed frames counter Shift */
#define  ETH_DMA_RX_OVERFLOW_MISSEDFRAMES_COUNTERSHIFT     17U
 /**
  * @}
  */

/* Exported types ------------------------------------------------------------*/ 
/** @defgroup ETH_Exported_Types ETH Exported Types
  * @{
  */

/** 
  * @brief  HAL State structures definition  
  */ 
typedef enum
{
  HAL_ETH_STATE_RESET             = 0x00U,    /*!< Peripheral not yet Initialized or disabled         */
  HAL_ETH_STATE_READY             = 0x01U,    /*!< Peripheral Initialized and ready for use           */
  HAL_ETH_STATE_BUSY              = 0x02U,    /*!< an internal process is ongoing                     */
  HAL_ETH_STATE_BUSY_TX           = 0x12U,    /*!< Data Transmission process is ongoing               */
  HAL_ETH_STATE_BUSY_RX           = 0x22U,    /*!< Data Reception process is ongoing                  */
  HAL_ETH_STATE_BUSY_TX_RX        = 0x32U,    /*!< Data Transmission and Reception process is ongoing */
  HAL_ETH_STATE_BUSY_WR           = 0x42U,    /*!< Write process is ongoing                           */
  HAL_ETH_STATE_BUSY_RD           = 0x82U,    /*!< Read process is ongoing                            */
  HAL_ETH_STATE_TIMEOUT           = 0x03U,    /*!< Timeout state                                      */
  HAL_ETH_STATE_ERROR             = 0x04U     /*!< Reception process is ongoing                       */
}HAL_ETH_StateTypeDef;

/** 
  * @brief  ETH Init Structure definition  
  */

typedef struct
{
  uint32_t             AutoNegotiation;           /*!< Selects or not the AutoNegotiation mode for the external PHY
                                                           The AutoNegotiation allows an automatic setting of the Speed (10/100Mbps)
                                                           and the mode (half/full-duplex).
                                                           This parameter can be a value of @ref ETH_AutoNegotiation */

  uint32_t             Speed;                     /*!< Sets the Ethernet speed: 10/100 Mbps.
                                                           This parameter can be a value of @ref ETH_Speed */

  uint32_t             DuplexMode;                /*!< Selects the MAC duplex mode: Half-Duplex or Full-Duplex mode
                                                           This parameter can be a value of @ref ETH_Duplex_Mode */
  
  uint16_t             PhyAddress;                /*!< Ethernet PHY address.
                                                           This parameter must be a number between Min_Data = 0 and Max_Data = 32 */
  
  uint8_t             *MACAddr;                   /*!< MAC Address of used Hardware: must be pointer on an array of 6 bytes */
  
  uint32_t             RxMode;                    /*!< Selects the Ethernet Rx mode: Polling mode, Interrupt mode.
                                                           This parameter can be a value of @ref ETH_Rx_Mode */
  
  uint32_t             ChecksumMode;              /*!< Selects if the checksum is check by hardware or by software. 
                                                         This parameter can be a value of @ref ETH_Checksum_Mode */
  
  uint32_t             MediaInterface;            /*!< Selects the media-independent interface or the reduced media-independent interface. 
                                                         This parameter can be a value of @ref ETH_Media_Interface */

} ETH_InitTypeDef;


 /** 
  * @brief  ETH MAC Configuration Structure definition  
  */

typedef struct
{
  uint32_t             Watchdog;                  /*!< Selects or not the Watchdog timer
                                                           When enabled, the MAC allows no more then 2048 bytes to be received.
                                                           When disabled, the MAC can receive up to 16384 bytes.
                                                           This parameter can be a value of @ref ETH_Watchdog */  

  uint32_t             Jabber;                    /*!< Selects or not Jabber timer
                                                           When enabled, the MAC allows no more then 2048 bytes to be sent.
                                                           When disabled, the MAC can send up to 16384 bytes.
                                                           This parameter can be a value of @ref ETH_Jabber */

  uint32_t             InterFrameGap;             /*!< Selects the minimum IFG between frames during transmission.
                                                           This parameter can be a value of @ref ETH_Inter_Frame_Gap */   

  uint32_t             CarrierSense;              /*!< Selects or not the Carrier Sense.
                                                           This parameter can be a value of @ref ETH_Carrier_Sense */

  uint32_t             ReceiveOwn;                /*!< Selects or not the ReceiveOwn,
                                                           ReceiveOwn allows the reception of frames when the TX_EN signal is asserted
                                                           in Half-Duplex mode.
                                                           This parameter can be a value of @ref ETH_Receive_Own */  

  uint32_t             LoopbackMode;              /*!< Selects or not the internal MAC MII Loopback mode.
                                                           This parameter can be a value of @ref ETH_Loop_Back_Mode */  

  uint32_t             ChecksumOffload;           /*!< Selects or not the IPv4 checksum checking for received frame payloads' TCP/UDP/ICMP headers.
                                                           This parameter can be a value of @ref ETH_Checksum_Offload */    

  uint32_t             RetryTransmission;         /*!< Selects or not the MAC attempt retries transmission, based on the settings of BL,
                                                           when a collision occurs (Half-Duplex mode).
                                                           This parameter can be a value of @ref ETH_Retry_Transmission */

  uint32_t             AutomaticPadCRCStrip;      /*!< Selects or not the Automatic MAC Pad/CRC Stripping.
                                                           This parameter can be a value of @ref ETH_Automatic_Pad_CRC_Strip */ 

  uint32_t             BackOffLimit;              /*!< Selects the BackOff limit value.
                                                           This parameter can be a value of @ref ETH_Back_Off_Limit */

  uint32_t             DeferralCheck;             /*!< Selects or not the deferral check function (Half-Duplex mode).
                                                           This parameter can be a value of @ref ETH_Deferral_Check */                                                                                                        

  uint32_t             ReceiveAll;                /*!< Selects or not all frames reception by the MAC (No filtering).
                                                           This parameter can be a value of @ref ETH_Receive_All */   

  uint32_t             SourceAddrFilter;          /*!< Selects the Source Address Filter mode.
                                                           This parameter can be a value of @ref ETH_Source_Addr_Filter */

  uint32_t             PassControlFrames;         /*!< Sets the forwarding mode of the control frames (including unicast and multicast PAUSE frames)
                                                           This parameter can be a value of @ref ETH_Pass_Control_Frames */ 

  uint32_t             BroadcastFramesReception;  /*!< Selects or not the reception of Broadcast Frames.
                                                           This parameter can be a value of @ref ETH_Broadcast_Frames_Reception */

  uint32_t             DestinationAddrFilter;     /*!< Sets the destination filter mode for both unicast and multicast frames.
                                                           This parameter can be a value of @ref ETH_Destination_Addr_Filter */ 

  uint32_t             PromiscuousMode;           /*!< Selects or not the Promiscuous Mode
                                                           This parameter can be a value of @ref ETH_Promiscuous_Mode */

  uint32_t             MulticastFramesFilter;     /*!< Selects the Multicast Frames filter mode: None/HashTableFilter/PerfectFilter/PerfectHashTableFilter.
                                                           This parameter can be a value of @ref ETH_Multicast_Frames_Filter */ 

  uint32_t             UnicastFramesFilter;       /*!< Selects the Unicast Frames filter mode: HashTableFilter/PerfectFilter/PerfectHashTableFilter.
                                                           This parameter can be a value of @ref ETH_Unicast_Frames_Filter */ 

  uint32_t             HashTableHigh;             /*!< This field holds the higher 32 bits of Hash table.
                                                           This parameter must be a number between Min_Data = 0x0 and Max_Data = 0xFFFFFFFFU */

  uint32_t             HashTableLow;              /*!< This field holds the lower 32 bits of Hash table.
                                                           This parameter must be a number between Min_Data = 0x0 and Max_Data = 0xFFFFFFFFU  */    

  uint32_t             PauseTime;                 /*!< This field holds the value to be used in the Pause Time field in the transmit control frame. 
                                                           This parameter must be a number between Min_Data = 0x0 and Max_Data = 0xFFFFU */

  uint32_t             ZeroQuantaPause;           /*!< Selects or not the automatic generation of Zero-Quanta Pause Control frames.
                                                           This parameter can be a value of @ref ETH_Zero_Quanta_Pause */  

  uint32_t             PauseLowThreshold;         /*!< This field configures the threshold of the PAUSE to be checked for
                                                           automatic retransmission of PAUSE Frame.
                                                           This parameter can be a value of @ref ETH_Pause_Low_Threshold */

  uint32_t             UnicastPauseFrameDetect;   /*!< Selects or not the MAC detection of the Pause frames (with MAC Address0
                                                           unicast address and unique multicast address).
                                                           This parameter can be a value of @ref ETH_Unicast_Pause_Frame_Detect */  

  uint32_t             ReceiveFlowControl;        /*!< Enables or disables the MAC to decode the received Pause frame and
                                                           disable its transmitter for a specified time (Pause Time)
                                                           This parameter can be a value of @ref ETH_Receive_Flow_Control */

  uint32_t             TransmitFlowControl;       /*!< Enables or disables the MAC to transmit Pause frames (Full-Duplex mode)
                                                           or the MAC back-pressure operation (Half-Duplex mode)
                                                           This parameter can be a value of @ref ETH_Transmit_Flow_Control */     

  uint32_t             VLANTagComparison;         /*!< Selects the 12-bit VLAN identifier or the complete 16-bit VLAN tag for
                                                           comparison and filtering.
                                                           This parameter can be a value of @ref ETH_VLAN_Tag_Comparison */ 

  uint32_t             VLANTagIdentifier;         /*!< Holds the VLAN tag identifier for receive frames */

} ETH_MACInitTypeDef;

/** 
  * @brief  ETH DMA Configuration Structure definition  
  */

typedef struct
{
 uint32_t              DropTCPIPChecksumErrorFrame; /*!< Selects or not the Dropping of TCP/IP Checksum Error Frames.
                                                             This parameter can be a value of @ref ETH_Drop_TCP_IP_Checksum_Error_Frame */ 

  uint32_t             ReceiveStoreForward;         /*!< Enables or disables the Receive store and forward mode.
                                                             This parameter can be a value of @ref ETH_Receive_Store_Forward */ 

  uint32_t             FlushReceivedFrame;          /*!< Enables or disables the flushing of received frames.
                                                             This parameter can be a value of @ref ETH_Flush_Received_Frame */ 

  uint32_t             TransmitStoreForward;        /*!< Enables or disables Transmit store and forward mode.
                                                             This parameter can be a value of @ref ETH_Transmit_Store_Forward */ 

  uint32_t             TransmitThresholdControl;    /*!< Selects or not the Transmit Threshold Control.
                                                             This parameter can be a value of @ref ETH_Transmit_Threshold_Control */

  uint32_t             ForwardErrorFrames;          /*!< Selects or not the forward to the DMA of erroneous frames.
                                                             This parameter can be a value of @ref ETH_Forward_Error_Frames */

  uint32_t             ForwardUndersizedGoodFrames; /*!< Enables or disables the Rx FIFO to forward Undersized frames (frames with no Error
                                                             and length less than 64 bytes) including pad-bytes and CRC)
                                                             This parameter can be a value of @ref ETH_Forward_Undersized_Good_Frames */

  uint32_t             ReceiveThresholdControl;     /*!< Selects the threshold level of the Receive FIFO.
                                                             This parameter can be a value of @ref ETH_Receive_Threshold_Control */

  uint32_t             SecondFrameOperate;          /*!< Selects or not the Operate on second frame mode, which allows the DMA to process a second
                                                             frame of Transmit data even before obtaining the status for the first frame.
                                                             This parameter can be a value of @ref ETH_Second_Frame_Operate */

  uint32_t             AddressAlignedBeats;         /*!< Enables or disables the Address Aligned Beats.
                                                             This parameter can be a value of @ref ETH_Address_Aligned_Beats */

  uint32_t             FixedBurst;                  /*!< Enables or disables the AHB Master interface fixed burst transfers.
                                                             This parameter can be a value of @ref ETH_Fixed_Burst */
                       
  uint32_t             RxDMABurstLength;            /*!< Indicates the maximum number of beats to be transferred in one Rx DMA transaction.
                                                             This parameter can be a value of @ref ETH_Rx_DMA_Burst_Length */ 

  uint32_t             TxDMABurstLength;            /*!< Indicates the maximum number of beats to be transferred in one Tx DMA transaction.
                                                             This parameter can be a value of @ref ETH_Tx_DMA_Burst_Length */
  
  uint32_t             EnhancedDescriptorFormat;    /*!< Enables the enhanced descriptor format.
                                                             This parameter can be a value of @ref ETH_DMA_Enhanced_descriptor_format */

  uint32_t             DescriptorSkipLength;        /*!< Specifies the number of word to skip between two unchained descriptors (Ring mode)
                                                             This parameter must be a number between Min_Data = 0 and Max_Data = 32 */

  uint32_t             DMAArbitration;              /*!< Selects the DMA Tx/Rx arbitration.
                                                             This parameter can be a value of @ref ETH_DMA_Arbitration */  
} ETH_DMAInitTypeDef;


/** 
  * @brief  ETH DMA Descriptors data structure definition
  */ 

typedef struct  
{
  __IO uint32_t   Status;           /*!< Status */
  
  uint32_t   ControlBufferSize;     /*!< Control and Buffer1, Buffer2 lengths */
  
  uint32_t   Buffer1Addr;           /*!< Buffer1 address pointer */
  
  uint32_t   Buffer2NextDescAddr;   /*!< Buffer2 or next descriptor address pointer */
  
  /*!< Enhanced ETHERNET DMA PTP Descriptors */
  uint32_t   ExtendedStatus;        /*!< Extended status for PTP receive descriptor */
  
  uint32_t   Reserved1;             /*!< Reserved */
  
  uint32_t   TimeStampLow;          /*!< Time Stamp Low value for transmit and receive */
  
  uint32_t   TimeStampHigh;         /*!< Time Stamp High value for transmit and receive */

} ETH_DMADescTypeDef;

/** 
  * @brief  Received Frame Informations structure definition
  */ 
typedef struct  
{
  ETH_DMADescTypeDef *FSRxDesc;          /*!< First Segment Rx Desc */
  
  ETH_DMADescTypeDef *LSRxDesc;          /*!< Last Segment Rx Desc */
  
  uint32_t  SegCount;                    /*!< Segment count */
  
  uint32_t length;                       /*!< Frame length */
  
  uint32_t buffer;                       /*!< Frame buffer */

} ETH_DMARxFrameInfos;

/** 
  * @brief  ETH Handle Structure definition  
  */
  
#if (USE_HAL_ETH_REGISTER_CALLBACKS == 1)
typedef struct __ETH_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_ETH_REGISTER_CALLBACKS */
{
  ETH_TypeDef                *Instance;     /*!< Register base address       */
  
  ETH_InitTypeDef            Init;          /*!< Ethernet Init Configuration */
  
  uint32_t                   LinkStatus;    /*!< Ethernet link status        */
  
  ETH_DMADescTypeDef         *RxDesc;       /*!< Rx descriptor to Get        */
  
  ETH_DMADescTypeDef         *TxDesc;       /*!< Tx descriptor to Set        */
  
  ETH_DMARxFrameInfos        RxFrameInfos;  /*!< last Rx frame infos         */
  
  __IO HAL_ETH_StateTypeDef  State;         /*!< ETH communication state     */
  
  HAL_LockTypeDef            Lock;          /*!< ETH Lock                    */

#if (USE_HAL_ETH_REGISTER_CALLBACKS == 1)

  void    (* TxCpltCallback)     ( struct __ETH_HandleTypeDef * heth);  /*!< ETH Tx Complete Callback   */
  void    (* RxCpltCallback)     ( struct __ETH_HandleTypeDef * heth);  /*!< ETH Rx  Complete Callback   */
  void    (* DMAErrorCallback)   ( struct __ETH_HandleTypeDef * heth);  /*!< DMA Error Callback      */
  void    (* MspInitCallback)    ( struct __ETH_HandleTypeDef * heth);  /*!< ETH Msp Init callback       */
  void    (* MspDeInitCallback)  ( struct __ETH_HandleTypeDef * heth);  /*!< ETH Msp DeInit callback     */

#endif  /* USE_HAL_ETH_REGISTER_CALLBACKS */

} ETH_HandleTypeDef;

#if (USE_HAL_ETH_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL ETH Callback ID enumeration definition
  */
typedef enum
{
  HAL_ETH_MSPINIT_CB_ID            = 0x00U,    /*!< ETH MspInit callback ID            */
  HAL_ETH_MSPDEINIT_CB_ID          = 0x01U,    /*!< ETH MspDeInit callback ID          */
  HAL_ETH_TX_COMPLETE_CB_ID        = 0x02U,    /*!< ETH Tx Complete Callback ID        */
  HAL_ETH_RX_COMPLETE_CB_ID        = 0x03U,    /*!< ETH Rx Complete Callback ID        */
  HAL_ETH_DMA_ERROR_CB_ID          = 0x04U,    /*!< ETH DMA Error Callback ID          */

}HAL_ETH_CallbackIDTypeDef;

/**
  * @brief  HAL ETH Callback pointer definition
  */
typedef  void (*pETH_CallbackTypeDef)(ETH_HandleTypeDef * heth); /*!< pointer to an ETH callback function */

#endif /* USE_HAL_ETH_REGISTER_CALLBACKS */

 /**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup ETH_Exported_Constants ETH Exported Constants
  * @{
  */

/** @defgroup ETH_Buffers_setting ETH Buffers setting
  * @{
  */ 
#define ETH_MAX_PACKET_SIZE       1524U    /*!< ETH_HEADER + ETH_EXTRA + ETH_VLAN_TAG + ETH_MAX_ETH_PAYLOAD + ETH_CRC */
#define ETH_HEADER                14U      /*!< 6 byte Dest addr, 6 byte Src addr, 2 byte length/type */
#define ETH_CRC                   4U       /*!< Ethernet CRC */
#define ETH_EXTRA                 2U       /*!< Extra bytes in some cases */   
#define ETH_VLAN_TAG              4U       /*!< optional 802.1q VLAN Tag */
#define ETH_MIN_ETH_PAYLOAD       46U      /*!< Minimum Ethernet payload size */
#define ETH_MAX_ETH_PAYLOAD       1500U    /*!< Maximum Ethernet payload size */
#define ETH_JUMBO_FRAME_PAYLOAD   9000U    /*!< Jumbo frame payload size */      

 /* Ethernet driver receive buffers are organized in a chained linked-list, when
    an ethernet packet is received, the Rx-DMA will transfer the packet from RxFIFO
    to the driver receive buffers memory.

    Depending on the size of the received ethernet packet and the size of 
    each ethernet driver receive buffer, the received packet can take one or more
    ethernet driver receive buffer. 

    In below are defined the size of one ethernet driver receive buffer ETH_RX_BUF_SIZE 
    and the total count of the driver receive buffers ETH_RXBUFNB.

    The configured value for ETH_RX_BUF_SIZE and ETH_RXBUFNB are only provided as 
    example, they can be reconfigured in the application layer to fit the application 
    needs */ 

/* Here we configure each Ethernet driver receive buffer to fit the Max size Ethernet
   packet */
#ifndef ETH_RX_BUF_SIZE
 #define ETH_RX_BUF_SIZE         ETH_MAX_PACKET_SIZE 
#endif

/* 5 Ethernet driver receive buffers are used (in a chained linked list)*/ 
#ifndef ETH_RXBUFNB
 #define ETH_RXBUFNB             5U     /*  5 Rx buffers of size ETH_RX_BUF_SIZE */
#endif


 /* Ethernet driver transmit buffers are organized in a chained linked-list, when
    an ethernet packet is transmitted, Tx-DMA will transfer the packet from the 
    driver transmit buffers memory to the TxFIFO.

    Depending on the size of the Ethernet packet to be transmitted and the size of 
    each ethernet driver transmit buffer, the packet to be transmitted can take 
    one or more ethernet driver transmit buffer. 

    In below are defined the size of one ethernet driver transmit buffer ETH_TX_BUF_SIZE 
    and the total count of the driver transmit buffers ETH_TXBUFNB.

    The configured value for ETH_TX_BUF_SIZE and ETH_TXBUFNB are only provided as 
    example, they can be reconfigured in the application layer to fit the application 
    needs */ 

/* Here we configure each Ethernet driver transmit buffer to fit the Max size Ethernet
   packet */
#ifndef ETH_TX_BUF_SIZE 
 #define ETH_TX_BUF_SIZE         ETH_MAX_PACKET_SIZE
#endif

/* 5 ethernet driver transmit buffers are used (in a chained linked list)*/ 
#ifndef ETH_TXBUFNB
 #define ETH_TXBUFNB             5U      /* 5  Tx buffers of size ETH_TX_BUF_SIZE */
#endif

 /**
  * @}
  */

/** @defgroup ETH_DMA_TX_Descriptor ETH DMA TX Descriptor
  * @{
  */

/*
   DMA Tx Descriptor
  -----------------------------------------------------------------------------------------------
  TDES0 | OWN(31) | CTRL[30:26] | Reserved[25:24] | CTRL[23:20] | Reserved[19:17] | Status[16:0] |
  -----------------------------------------------------------------------------------------------
  TDES1 | Reserved[31:29] | Buffer2 ByteCount[28:16] | Reserved[15:13] | Buffer1 ByteCount[12:0] |
  -----------------------------------------------------------------------------------------------
  TDES2 |                         Buffer1 Address [31:0]                                         |
  -----------------------------------------------------------------------------------------------
  TDES3 |                   Buffer2 Address [31:0] / Next Descriptor Address [31:0]              |
  -----------------------------------------------------------------------------------------------
*/

/** 
  * @brief  Bit definition of TDES0 register: DMA Tx descriptor status register
  */ 
#define ETH_DMATXDESC_OWN                     0x80000000U  /*!< OWN bit: descriptor is owned by DMA engine */
#define ETH_DMATXDESC_IC                      0x40000000U  /*!< Interrupt on Completion */
#define ETH_DMATXDESC_LS                      0x20000000U  /*!< Last Segment */
#define ETH_DMATXDESC_FS                      0x10000000U  /*!< First Segment */
#define ETH_DMATXDESC_DC                      0x08000000U  /*!< Disable CRC */
#define ETH_DMATXDESC_DP                      0x04000000U  /*!< Disable Padding */
#define ETH_DMATXDESC_TTSE                    0x02000000U  /*!< Transmit Time Stamp Enable */
#define ETH_DMATXDESC_CIC                     0x00C00000U  /*!< Checksum Insertion Control: 4 cases */
#define ETH_DMATXDESC_CIC_BYPASS              0x00000000U  /*!< Do Nothing: Checksum Engine is bypassed */ 
#define ETH_DMATXDESC_CIC_IPV4HEADER          0x00400000U  /*!< IPV4 header Checksum Insertion */ 
#define ETH_DMATXDESC_CIC_TCPUDPICMP_SEGMENT  0x00800000U  /*!< TCP/UDP/ICMP Checksum Insertion calculated over segment only */ 
#define ETH_DMATXDESC_CIC_TCPUDPICMP_FULL     0x00C00000U  /*!< TCP/UDP/ICMP Checksum Insertion fully calculated */ 
#define ETH_DMATXDESC_TER                     0x00200000U  /*!< Transmit End of Ring */
#define ETH_DMATXDESC_TCH                     0x00100000U  /*!< Second Address Chained */
#define ETH_DMATXDESC_TTSS                    0x00020000U  /*!< Tx Time Stamp Status */
#define ETH_DMATXDESC_IHE                     0x00010000U  /*!< IP Header Error */
#define ETH_DMATXDESC_ES                      0x00008000U  /*!< Error summary: OR of the following bits: UE || ED || EC || LCO || NC || LCA || FF || JT */
#define ETH_DMATXDESC_JT                      0x00004000U  /*!< Jabber Timeout */
#define ETH_DMATXDESC_FF                      0x00002000U  /*!< Frame Flushed: DMA/MTL flushed the frame due to SW flush */
#define ETH_DMATXDESC_PCE                     0x00001000U  /*!< Payload Checksum Error */
#define ETH_DMATXDESC_LCA                     0x00000800U  /*!< Loss of Carrier: carrier lost during transmission */
#define ETH_DMATXDESC_NC                      0x00000400U  /*!< No Carrier: no carrier signal from the transceiver */
#define ETH_DMATXDESC_LCO                     0x00000200U  /*!< Late Collision: transmission aborted due to collision */
#define ETH_DMATXDESC_EC                      0x00000100U  /*!< Excessive Collision: transmission aborted after 16 collisions */
#define ETH_DMATXDESC_VF                      0x00000080U  /*!< VLAN Frame */
#define ETH_DMATXDESC_CC                      0x00000078U  /*!< Collision Count */
#define ETH_DMATXDESC_ED                      0x00000004U  /*!< Excessive Deferral */
#define ETH_DMATXDESC_UF                      0x00000002U  /*!< Underflow Error: late data arrival from the memory */
#define ETH_DMATXDESC_DB                      0x00000001U  /*!< Deferred Bit */

/** 
  * @brief  Bit definition of TDES1 register
  */ 
#define ETH_DMATXDESC_TBS2  0x1FFF0000U  /*!< Transmit Buffer2 Size */
#define ETH_DMATXDESC_TBS1  0x00001FFFU  /*!< Transmit Buffer1 Size */

/** 
  * @brief  Bit definition of TDES2 register
  */ 
#define ETH_DMATXDESC_B1AP  0xFFFFFFFFU  /*!< Buffer1 Address Pointer */

/** 
  * @brief  Bit definition of TDES3 register
  */ 
#define ETH_DMATXDESC_B2AP  0xFFFFFFFFU  /*!< Buffer2 Address Pointer */

  /*---------------------------------------------------------------------------------------------
  TDES6 |                         Transmit Time Stamp Low [31:0]                                 |
  -----------------------------------------------------------------------------------------------
  TDES7 |                         Transmit Time Stamp High [31:0]                                |
  ----------------------------------------------------------------------------------------------*/

/* Bit definition of TDES6 register */
 #define ETH_DMAPTPTXDESC_TTSL  0xFFFFFFFFU  /* Transmit Time Stamp Low */

/* Bit definition of TDES7 register */
 #define ETH_DMAPTPTXDESC_TTSH  0xFFFFFFFFU  /* Transmit Time Stamp High */

/**
  * @}
  */ 
/** @defgroup ETH_DMA_RX_Descriptor ETH DMA RX Descriptor
  * @{
  */

/*
  DMA Rx Descriptor
  --------------------------------------------------------------------------------------------------------------------
  RDES0 | OWN(31) |                                             Status [30:0]                                          |
  ---------------------------------------------------------------------------------------------------------------------
  RDES1 | CTRL(31) | Reserved[30:29] | Buffer2 ByteCount[28:16] | CTRL[15:14] | Reserved(13) | Buffer1 ByteCount[12:0] |
  ---------------------------------------------------------------------------------------------------------------------
  RDES2 |                                       Buffer1 Address [31:0]                                                 |
  ---------------------------------------------------------------------------------------------------------------------
  RDES3 |                          Buffer2 Address [31:0] / Next Descriptor Address [31:0]                             |
  ---------------------------------------------------------------------------------------------------------------------
*/

/** 
  * @brief  Bit definition of RDES0 register: DMA Rx descriptor status register
  */ 
#define ETH_DMARXDESC_OWN         0x80000000U  /*!< OWN bit: descriptor is owned by DMA engine  */
#define ETH_DMARXDESC_AFM         0x40000000U  /*!< DA Filter Fail for the rx frame  */
#define ETH_DMARXDESC_FL          0x3FFF0000U  /*!< Receive descriptor frame length  */
#define ETH_DMARXDESC_ES          0x00008000U  /*!< Error summary: OR of the following bits: DE || OE || IPC || LC || RWT || RE || CE */
#define ETH_DMARXDESC_DE          0x00004000U  /*!< Descriptor error: no more descriptors for receive frame  */
#define ETH_DMARXDESC_SAF         0x00002000U  /*!< SA Filter Fail for the received frame */
#define ETH_DMARXDESC_LE          0x00001000U  /*!< Frame size not matching with length field */
#define ETH_DMARXDESC_OE          0x00000800U  /*!< Overflow Error: Frame was damaged due to buffer overflow */
#define ETH_DMARXDESC_VLAN        0x00000400U  /*!< VLAN Tag: received frame is a VLAN frame */
#define ETH_DMARXDESC_FS          0x00000200U  /*!< First descriptor of the frame  */
#define ETH_DMARXDESC_LS          0x00000100U  /*!< Last descriptor of the frame  */ 
#define ETH_DMARXDESC_IPV4HCE     0x00000080U  /*!< IPC Checksum Error: Rx Ipv4 header checksum error   */    
#define ETH_DMARXDESC_LC          0x00000040U  /*!< Late collision occurred during reception   */
#define ETH_DMARXDESC_FT          0x00000020U  /*!< Frame type - Ethernet, otherwise 802.3    */
#define ETH_DMARXDESC_RWT         0x00000010U  /*!< Receive Watchdog Timeout: watchdog timer expired during reception    */
#define ETH_DMARXDESC_RE          0x00000008U  /*!< Receive error: error reported by MII interface  */
#define ETH_DMARXDESC_DBE         0x00000004U  /*!< Dribble bit error: frame contains non int multiple of 8 bits  */
#define ETH_DMARXDESC_CE          0x00000002U  /*!< CRC error */
#define ETH_DMARXDESC_MAMPCE      0x00000001U  /*!< Rx MAC Address/Payload Checksum Error: Rx MAC address matched/ Rx Payload Checksum Error */

/** 
  * @brief  Bit definition of RDES1 register
  */ 
#define ETH_DMARXDESC_DIC   0x80000000U  /*!< Disable Interrupt on Completion */
#define ETH_DMARXDESC_RBS2  0x1FFF0000U  /*!< Receive Buffer2 Size */
#define ETH_DMARXDESC_RER   0x00008000U  /*!< Receive End of Ring */
#define ETH_DMARXDESC_RCH   0x00004000U  /*!< Second Address Chained */
#define ETH_DMARXDESC_RBS1  0x00001FFFU  /*!< Receive Buffer1 Size */

/** 
  * @brief  Bit definition of RDES2 register  
  */ 
#define ETH_DMARXDESC_B1AP  0xFFFFFFFFU  /*!< Buffer1 Address Pointer */

/** 
  * @brief  Bit definition of RDES3 register  
  */ 
#define ETH_DMARXDESC_B2AP  0xFFFFFFFFU  /*!< Buffer2 Address Pointer */

/*---------------------------------------------------------------------------------------------------------------------
  RDES4 |                   Reserved[31:15]              |             Extended Status [14:0]                          |
  ---------------------------------------------------------------------------------------------------------------------
  RDES5 |                                            Reserved[31:0]                                                    |
  ---------------------------------------------------------------------------------------------------------------------
  RDES6 |                                       Receive Time Stamp Low [31:0]                                          |
  ---------------------------------------------------------------------------------------------------------------------
  RDES7 |                                       Receive Time Stamp High [31:0]                                         |
  --------------------------------------------------------------------------------------------------------------------*/

/* Bit definition of RDES4 register */
#define ETH_DMAPTPRXDESC_PTPV     0x00002000U  /* PTP Version */
#define ETH_DMAPTPRXDESC_PTPFT    0x00001000U  /* PTP Frame Type */
#define ETH_DMAPTPRXDESC_PTPMT    0x00000F00U  /* PTP Message Type */
  #define ETH_DMAPTPRXDESC_PTPMT_SYNC                      0x00000100U  /* SYNC message (all clock types) */
  #define ETH_DMAPTPRXDESC_PTPMT_FOLLOWUP                  0x00000200U  /* FollowUp message (all clock types) */ 
  #define ETH_DMAPTPRXDESC_PTPMT_DELAYREQ                  0x00000300U  /* DelayReq message (all clock types) */ 
  #define ETH_DMAPTPRXDESC_PTPMT_DELAYRESP                 0x00000400U  /* DelayResp message (all clock types) */ 
  #define ETH_DMAPTPRXDESC_PTPMT_PDELAYREQ_ANNOUNCE        0x00000500U  /* PdelayReq message (peer-to-peer transparent clock) or Announce message (Ordinary or Boundary clock) */ 
  #define ETH_DMAPTPRXDESC_PTPMT_PDELAYRESP_MANAG          0x00000600U  /* PdelayResp message (peer-to-peer transparent clock) or Management message (Ordinary or Boundary clock)  */ 
  #define ETH_DMAPTPRXDESC_PTPMT_PDELAYRESPFOLLOWUP_SIGNAL 0x00000700U  /* PdelayRespFollowUp message (peer-to-peer transparent clock) or Signaling message (Ordinary or Boundary clock) */           
#define ETH_DMAPTPRXDESC_IPV6PR   0x00000080U  /* IPv6 Packet Received */
#define ETH_DMAPTPRXDESC_IPV4PR   0x00000040U  /* IPv4 Packet Received */
#define ETH_DMAPTPRXDESC_IPCB     0x00000020U  /* IP Checksum Bypassed */
#define ETH_DMAPTPRXDESC_IPPE     0x00000010U  /* IP Payload Error */
#define ETH_DMAPTPRXDESC_IPHE     0x00000008U  /* IP Header Error */
#define ETH_DMAPTPRXDESC_IPPT     0x00000007U  /* IP Payload Type */
  #define ETH_DMAPTPRXDESC_IPPT_UDP                 0x00000001U  /* UDP payload encapsulated in the IP datagram */
  #define ETH_DMAPTPRXDESC_IPPT_TCP                 0x00000002U  /* TCP payload encapsulated in the IP datagram */ 
  #define ETH_DMAPTPRXDESC_IPPT_ICMP                0x00000003U  /* ICMP payload encapsulated in the IP datagram */

/* Bit definition of RDES6 register */
#define ETH_DMAPTPRXDESC_RTSL  0xFFFFFFFFU  /* Receive Time Stamp Low */

/* Bit definition of RDES7 register */
#define ETH_DMAPTPRXDESC_RTSH  0xFFFFFFFFU  /* Receive Time Stamp High */
/**
  * @}
  */
 /** @defgroup ETH_AutoNegotiation ETH AutoNegotiation 
  * @{
  */ 
#define ETH_AUTONEGOTIATION_ENABLE     0x00000001U
#define ETH_AUTONEGOTIATION_DISABLE    0x00000000U

/**
  * @}
  */
/** @defgroup ETH_Speed ETH Speed 
  * @{
  */ 
#define ETH_SPEED_10M        0x00000000U
#define ETH_SPEED_100M       0x00004000U

/**
  * @}
  */
/** @defgroup ETH_Duplex_Mode ETH Duplex Mode
  * @{
  */ 
#define ETH_MODE_FULLDUPLEX       0x00000800U
#define ETH_MODE_HALFDUPLEX       0x00000000U
/**
  * @}
  */
/** @defgroup ETH_Rx_Mode ETH Rx Mode
  * @{
  */ 
#define ETH_RXPOLLING_MODE      0x00000000U
#define ETH_RXINTERRUPT_MODE    0x00000001U
/**
  * @}
  */

/** @defgroup ETH_Checksum_Mode ETH Checksum Mode
  * @{
  */ 
#define ETH_CHECKSUM_BY_HARDWARE      0x00000000U
#define ETH_CHECKSUM_BY_SOFTWARE      0x00000001U
/**
  * @}
  */

/** @defgroup ETH_Media_Interface ETH Media Interface
  * @{
  */ 
#define ETH_MEDIA_INTERFACE_MII       0x00000000U
#define ETH_MEDIA_INTERFACE_RMII      ((uint32_t)SYSCFG_PMC_MII_RMII_SEL)
/**
  * @}
  */

/** @defgroup ETH_Watchdog ETH Watchdog 
  * @{
  */ 
#define ETH_WATCHDOG_ENABLE       0x00000000U
#define ETH_WATCHDOG_DISABLE      0x00800000U
/**
  * @}
  */

/** @defgroup ETH_Jabber ETH Jabber
  * @{
  */ 
#define ETH_JABBER_ENABLE    0x00000000U
#define ETH_JABBER_DISABLE   0x00400000U
/**
  * @}
  */

/** @defgroup ETH_Inter_Frame_Gap ETH Inter Frame Gap 
  * @{
  */ 
#define ETH_INTERFRAMEGAP_96BIT   0x00000000U  /*!< minimum IFG between frames during transmission is 96Bit */
#define ETH_INTERFRAMEGAP_88BIT   0x00020000U  /*!< minimum IFG between frames during transmission is 88Bit */
#define ETH_INTERFRAMEGAP_80BIT   0x00040000U  /*!< minimum IFG between frames during transmission is 80Bit */
#define ETH_INTERFRAMEGAP_72BIT   0x00060000U  /*!< minimum IFG between frames during transmission is 72Bit */
#define ETH_INTERFRAMEGAP_64BIT   0x00080000U  /*!< minimum IFG between frames during transmission is 64Bit */
#define ETH_INTERFRAMEGAP_56BIT   0x000A0000U  /*!< minimum IFG between frames during transmission is 56Bit */
#define ETH_INTERFRAMEGAP_48BIT   0x000C0000U  /*!< minimum IFG between frames during transmission is 48Bit */
#define ETH_INTERFRAMEGAP_40BIT   0x000E0000U  /*!< minimum IFG between frames during transmission is 40Bit */
/**
  * @}
  */

/** @defgroup ETH_Carrier_Sense ETH Carrier Sense
  * @{
  */ 
#define ETH_CARRIERSENCE_ENABLE   0x00000000U
#define ETH_CARRIERSENCE_DISABLE  0x00010000U
/**
  * @}
  */

/** @defgroup ETH_Receive_Own ETH Receive Own 
  * @{
  */ 
#define ETH_RECEIVEOWN_ENABLE     0x00000000U
#define ETH_RECEIVEOWN_DISABLE    0x00002000U
/**
  * @}
  */

/** @defgroup ETH_Loop_Back_Mode ETH Loop Back Mode 
  * @{
  */ 
#define ETH_LOOPBACKMODE_ENABLE        0x00001000U
#define ETH_LOOPBACKMODE_DISABLE       0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Checksum_Offload ETH Checksum Offload
  * @{
  */ 
#define ETH_CHECKSUMOFFLAOD_ENABLE     0x00000400U
#define ETH_CHECKSUMOFFLAOD_DISABLE    0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Retry_Transmission ETH Retry Transmission
  * @{
  */ 
#define ETH_RETRYTRANSMISSION_ENABLE   0x00000000U
#define ETH_RETRYTRANSMISSION_DISABLE  0x00000200U
/**
  * @}
  */

/** @defgroup ETH_Automatic_Pad_CRC_Strip ETH Automatic Pad CRC Strip
  * @{
  */ 
#define ETH_AUTOMATICPADCRCSTRIP_ENABLE     0x00000080U
#define ETH_AUTOMATICPADCRCSTRIP_DISABLE    0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Back_Off_Limit ETH Back Off Limit
  * @{
  */ 
#define ETH_BACKOFFLIMIT_10  0x00000000U
#define ETH_BACKOFFLIMIT_8   0x00000020U
#define ETH_BACKOFFLIMIT_4   0x00000040U
#define ETH_BACKOFFLIMIT_1   0x00000060U
/**
  * @}
  */

/** @defgroup ETH_Deferral_Check ETH Deferral Check
  * @{
  */
#define ETH_DEFFERRALCHECK_ENABLE       0x00000010U
#define ETH_DEFFERRALCHECK_DISABLE      0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Receive_All ETH Receive All
  * @{
  */ 
#define ETH_RECEIVEALL_ENABLE     0x80000000U
#define ETH_RECEIVEAll_DISABLE    0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Source_Addr_Filter ETH Source Addr Filter
  * @{
  */ 
#define ETH_SOURCEADDRFILTER_NORMAL_ENABLE       0x00000200U
#define ETH_SOURCEADDRFILTER_INVERSE_ENABLE      0x00000300U
#define ETH_SOURCEADDRFILTER_DISABLE             0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Pass_Control_Frames ETH Pass Control Frames
  * @{
  */ 
#define ETH_PASSCONTROLFRAMES_BLOCKALL                0x00000040U  /*!< MAC filters all control frames from reaching the application */
#define ETH_PASSCONTROLFRAMES_FORWARDALL              0x00000080U  /*!< MAC forwards all control frames to application even if they fail the Address Filter */
#define ETH_PASSCONTROLFRAMES_FORWARDPASSEDADDRFILTER 0x000000C0U  /*!< MAC forwards control frames that pass the Address Filter. */ 
/**
  * @}
  */

/** @defgroup ETH_Broadcast_Frames_Reception ETH Broadcast Frames Reception
  * @{
  */ 
#define ETH_BROADCASTFRAMESRECEPTION_ENABLE     0x00000000U
#define ETH_BROADCASTFRAMESRECEPTION_DISABLE    0x00000020U
/**
  * @}
  */

/** @defgroup ETH_Destination_Addr_Filter ETH Destination Addr Filter
  * @{
  */ 
#define ETH_DESTINATIONADDRFILTER_NORMAL    0x00000000U
#define ETH_DESTINATIONADDRFILTER_INVERSE   0x00000008U
/**
  * @}
  */

/** @defgroup ETH_Promiscuous_Mode ETH Promiscuous Mode
  * @{
  */ 
#define ETH_PROMISCUOUS_MODE_ENABLE     0x00000001U
#define ETH_PROMISCUOUS_MODE_DISABLE    0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Multicast_Frames_Filter ETH Multicast Frames Filter
  * @{
  */ 
#define ETH_MULTICASTFRAMESFILTER_PERFECTHASHTABLE    0x00000404U
#define ETH_MULTICASTFRAMESFILTER_HASHTABLE           0x00000004U
#define ETH_MULTICASTFRAMESFILTER_PERFECT             0x00000000U
#define ETH_MULTICASTFRAMESFILTER_NONE                0x00000010U
/**
  * @}
  */

/** @defgroup ETH_Unicast_Frames_Filter ETH Unicast Frames Filter
  * @{
  */ 
#define ETH_UNICASTFRAMESFILTER_PERFECTHASHTABLE 0x00000402U
#define ETH_UNICASTFRAMESFILTER_HASHTABLE        0x00000002U
#define ETH_UNICASTFRAMESFILTER_PERFECT          0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Zero_Quanta_Pause ETH Zero Quanta Pause 
  * @{
  */ 
#define ETH_ZEROQUANTAPAUSE_ENABLE     0x00000000U
#define ETH_ZEROQUANTAPAUSE_DISABLE    0x00000080U
/**
  * @}
  */

/** @defgroup ETH_Pause_Low_Threshold ETH Pause Low Threshold
  * @{
  */ 
#define ETH_PAUSELOWTHRESHOLD_MINUS4        0x00000000U  /*!< Pause time minus 4 slot times */
#define ETH_PAUSELOWTHRESHOLD_MINUS28       0x00000010U  /*!< Pause time minus 28 slot times */
#define ETH_PAUSELOWTHRESHOLD_MINUS144      0x00000020U  /*!< Pause time minus 144 slot times */
#define ETH_PAUSELOWTHRESHOLD_MINUS256      0x00000030U  /*!< Pause time minus 256 slot times */
/**
  * @}
  */

/** @defgroup ETH_Unicast_Pause_Frame_Detect ETH Unicast Pause Frame Detect
  * @{
  */ 
#define ETH_UNICASTPAUSEFRAMEDETECT_ENABLE  0x00000008U
#define ETH_UNICASTPAUSEFRAMEDETECT_DISABLE 0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Receive_Flow_Control ETH Receive Flow Control
  * @{
  */ 
#define ETH_RECEIVEFLOWCONTROL_ENABLE       0x00000004U
#define ETH_RECEIVEFLOWCONTROL_DISABLE      0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Transmit_Flow_Control ETH Transmit Flow Control
  * @{
  */ 
#define ETH_TRANSMITFLOWCONTROL_ENABLE      0x00000002U
#define ETH_TRANSMITFLOWCONTROL_DISABLE     0x00000000U
/**
  * @}
  */

/** @defgroup ETH_VLAN_Tag_Comparison ETH VLAN Tag Comparison
  * @{
  */ 
#define ETH_VLANTAGCOMPARISON_12BIT    0x00010000U
#define ETH_VLANTAGCOMPARISON_16BIT    0x00000000U
/**
  * @}
  */

/** @defgroup ETH_MAC_addresses ETH MAC addresses
  * @{
  */ 
#define ETH_MAC_ADDRESS0     0x00000000U
#define ETH_MAC_ADDRESS1     0x00000008U
#define ETH_MAC_ADDRESS2     0x00000010U
#define ETH_MAC_ADDRESS3     0x00000018U
/**
  * @}
  */

/** @defgroup ETH_MAC_addresses_filter_SA_DA ETH MAC addresses filter SA DA 
  * @{
  */ 
#define ETH_MAC_ADDRESSFILTER_SA       0x00000000U
#define ETH_MAC_ADDRESSFILTER_DA       0x00000008U
/**
  * @}
  */

/** @defgroup ETH_MAC_addresses_filter_Mask_bytes ETH MAC addresses filter Mask bytes
  * @{
  */ 
#define ETH_MAC_ADDRESSMASK_BYTE6      0x20000000U  /*!< Mask MAC Address high reg bits [15:8] */
#define ETH_MAC_ADDRESSMASK_BYTE5      0x10000000U  /*!< Mask MAC Address high reg bits [7:0] */
#define ETH_MAC_ADDRESSMASK_BYTE4      0x08000000U  /*!< Mask MAC Address low reg bits [31:24] */
#define ETH_MAC_ADDRESSMASK_BYTE3      0x04000000U  /*!< Mask MAC Address low reg bits [23:16] */
#define ETH_MAC_ADDRESSMASK_BYTE2      0x02000000U  /*!< Mask MAC Address low reg bits [15:8] */
#define ETH_MAC_ADDRESSMASK_BYTE1      0x01000000U  /*!< Mask MAC Address low reg bits [70] */
/**
  * @}
  */

/** @defgroup ETH_Drop_TCP_IP_Checksum_Error_Frame ETH Drop TCP IP Checksum Error Frame
  * @{
  */ 
#define ETH_DROPTCPIPCHECKSUMERRORFRAME_ENABLE   0x00000000U
#define ETH_DROPTCPIPCHECKSUMERRORFRAME_DISABLE  0x04000000U
/**
  * @}
  */

/** @defgroup ETH_Receive_Store_Forward ETH Receive Store Forward
  * @{
  */ 
#define ETH_RECEIVESTOREFORWARD_ENABLE      0x02000000U
#define ETH_RECEIVESTOREFORWARD_DISABLE     0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Flush_Received_Frame ETH Flush Received Frame
  * @{
  */ 
#define ETH_FLUSHRECEIVEDFRAME_ENABLE       0x00000000U
#define ETH_FLUSHRECEIVEDFRAME_DISABLE      0x01000000U
/**
  * @}
  */

/** @defgroup ETH_Transmit_Store_Forward ETH Transmit Store Forward
  * @{
  */ 
#define ETH_TRANSMITSTOREFORWARD_ENABLE     0x00200000U
#define ETH_TRANSMITSTOREFORWARD_DISABLE    0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Transmit_Threshold_Control ETH Transmit Threshold Control
  * @{
  */ 
#define ETH_TRANSMITTHRESHOLDCONTROL_64BYTES     0x00000000U  /*!< threshold level of the MTL Transmit FIFO is 64 Bytes */
#define ETH_TRANSMITTHRESHOLDCONTROL_128BYTES    0x00004000U  /*!< threshold level of the MTL Transmit FIFO is 128 Bytes */
#define ETH_TRANSMITTHRESHOLDCONTROL_192BYTES    0x00008000U  /*!< threshold level of the MTL Transmit FIFO is 192 Bytes */
#define ETH_TRANSMITTHRESHOLDCONTROL_256BYTES    0x0000C000U  /*!< threshold level of the MTL Transmit FIFO is 256 Bytes */
#define ETH_TRANSMITTHRESHOLDCONTROL_40BYTES     0x00010000U  /*!< threshold level of the MTL Transmit FIFO is 40 Bytes */
#define ETH_TRANSMITTHRESHOLDCONTROL_32BYTES     0x00014000U  /*!< threshold level of the MTL Transmit FIFO is 32 Bytes */
#define ETH_TRANSMITTHRESHOLDCONTROL_24BYTES     0x00018000U  /*!< threshold level of the MTL Transmit FIFO is 24 Bytes */
#define ETH_TRANSMITTHRESHOLDCONTROL_16BYTES     0x0001C000U  /*!< threshold level of the MTL Transmit FIFO is 16 Bytes */
/**
  * @}
  */

/** @defgroup ETH_Forward_Error_Frames ETH Forward Error Frames
  * @{
  */ 
#define ETH_FORWARDERRORFRAMES_ENABLE       0x00000080U
#define ETH_FORWARDERRORFRAMES_DISABLE      0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Forward_Undersized_Good_Frames ETH Forward Undersized Good Frames
  * @{
  */ 
#define ETH_FORWARDUNDERSIZEDGOODFRAMES_ENABLE   0x00000040U
#define ETH_FORWARDUNDERSIZEDGOODFRAMES_DISABLE  0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Receive_Threshold_Control ETH Receive Threshold Control
  * @{
  */ 
#define ETH_RECEIVEDTHRESHOLDCONTROL_64BYTES      0x00000000U  /*!< threshold level of the MTL Receive FIFO is 64 Bytes */
#define ETH_RECEIVEDTHRESHOLDCONTROL_32BYTES      0x00000008U  /*!< threshold level of the MTL Receive FIFO is 32 Bytes */
#define ETH_RECEIVEDTHRESHOLDCONTROL_96BYTES      0x00000010U  /*!< threshold level of the MTL Receive FIFO is 96 Bytes */
#define ETH_RECEIVEDTHRESHOLDCONTROL_128BYTES     0x00000018U  /*!< threshold level of the MTL Receive FIFO is 128 Bytes */
/**
  * @}
  */

/** @defgroup ETH_Second_Frame_Operate ETH Second Frame Operate
  * @{
  */ 
#define ETH_SECONDFRAMEOPERARTE_ENABLE       0x00000004U
#define ETH_SECONDFRAMEOPERARTE_DISABLE      0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Address_Aligned_Beats ETH Address Aligned Beats 
  * @{
  */ 
#define ETH_ADDRESSALIGNEDBEATS_ENABLE      0x02000000U
#define ETH_ADDRESSALIGNEDBEATS_DISABLE     0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Fixed_Burst ETH Fixed Burst
  * @{
  */ 
#define ETH_FIXEDBURST_ENABLE     0x00010000U
#define ETH_FIXEDBURST_DISABLE    0x00000000U
/**
  * @}
  */

/** @defgroup ETH_Rx_DMA_Burst_Length ETH Rx DMA Burst Length
  * @{
  */ 
#define ETH_RXDMABURSTLENGTH_1BEAT          0x00020000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 1 */
#define ETH_RXDMABURSTLENGTH_2BEAT          0x00040000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 2 */
#define ETH_RXDMABURSTLENGTH_4BEAT          0x00080000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 4 */
#define ETH_RXDMABURSTLENGTH_8BEAT          0x00100000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 8 */
#define ETH_RXDMABURSTLENGTH_16BEAT         0x00200000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 16 */
#define ETH_RXDMABURSTLENGTH_32BEAT         0x00400000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 32 */
#define ETH_RXDMABURSTLENGTH_4XPBL_4BEAT    0x01020000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 4 */
#define ETH_RXDMABURSTLENGTH_4XPBL_8BEAT    0x01040000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 8 */
#define ETH_RXDMABURSTLENGTH_4XPBL_16BEAT   0x01080000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 16 */
#define ETH_RXDMABURSTLENGTH_4XPBL_32BEAT   0x01100000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 32 */
#define ETH_RXDMABURSTLENGTH_4XPBL_64BEAT   0x01200000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 64 */
#define ETH_RXDMABURSTLENGTH_4XPBL_128BEAT  0x01400000U  /*!< maximum number of beats to be transferred in one RxDMA transaction is 128 */
/**
  * @}
  */

/** @defgroup ETH_Tx_DMA_Burst_Length ETH Tx DMA Burst Length
  * @{
  */ 
#define ETH_TXDMABURSTLENGTH_1BEAT          0x00000100U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 1 */
#define ETH_TXDMABURSTLENGTH_2BEAT          0x00000200U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 2 */
#define ETH_TXDMABURSTLENGTH_4BEAT          0x00000400U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 4 */
#define ETH_TXDMABURSTLENGTH_8BEAT          0x00000800U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 8 */
#define ETH_TXDMABURSTLENGTH_16BEAT         0x00001000U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 16 */
#define ETH_TXDMABURSTLENGTH_32BEAT         0x00002000U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 32 */
#define ETH_TXDMABURSTLENGTH_4XPBL_4BEAT    0x01000100U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 4 */
#define ETH_TXDMABURSTLENGTH_4XPBL_8BEAT    0x01000200U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 8 */
#define ETH_TXDMABURSTLENGTH_4XPBL_16BEAT   0x01000400U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 16 */
#define ETH_TXDMABURSTLENGTH_4XPBL_32BEAT   0x01000800U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 32 */
#define ETH_TXDMABURSTLENGTH_4XPBL_64BEAT   0x01001000U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 64 */
#define ETH_TXDMABURSTLENGTH_4XPBL_128BEAT  0x01002000U  /*!< maximum number of beats to be transferred in one TxDMA (or both) transaction is 128 */
/**
  * @}
  */

/** @defgroup ETH_DMA_Enhanced_descriptor_format ETH DMA Enhanced descriptor format
  * @{
  */  
#define ETH_DMAENHANCEDDESCRIPTOR_ENABLE              0x00000080U
#define ETH_DMAENHANCEDDESCRIPTOR_DISABLE             0x00000000U
/**
  * @}
  */

/** @defgroup ETH_DMA_Arbitration ETH DMA Arbitration
  * @{
  */ 
#define ETH_DMAARBITRATION_ROUNDROBIN_RXTX_1_1   0x00000000U
#define ETH_DMAARBITRATION_ROUNDROBIN_RXTX_2_1   0x00004000U
#define ETH_DMAARBITRATION_ROUNDROBIN_RXTX_3_1   0x00008000U
#define ETH_DMAARBITRATION_ROUNDROBIN_RXTX_4_1   0x0000C000U
#define ETH_DMAARBITRATION_RXPRIORTX             0x00000002U
/**
  * @}
  */

/** @defgroup ETH_DMA_Tx_descriptor_segment ETH DMA Tx descriptor segment
  * @{
  */ 
#define ETH_DMATXDESC_LASTSEGMENTS      0x40000000U  /*!< Last Segment */
#define ETH_DMATXDESC_FIRSTSEGMENT      0x20000000U  /*!< First Segment */
/**
  * @}
  */

/** @defgroup ETH_DMA_Tx_descriptor_Checksum_Insertion_Control ETH DMA Tx descriptor Checksum Insertion Control
  * @{
  */ 
#define ETH_DMATXDESC_CHECKSUMBYPASS             0x00000000U   /*!< Checksum engine bypass */
#define ETH_DMATXDESC_CHECKSUMIPV4HEADER         0x00400000U   /*!< IPv4 header checksum insertion  */
#define ETH_DMATXDESC_CHECKSUMTCPUDPICMPSEGMENT  0x00800000U   /*!< TCP/UDP/ICMP checksum insertion. Pseudo header checksum is assumed to be present */
#define ETH_DMATXDESC_CHECKSUMTCPUDPICMPFULL     0x00C00000U   /*!< TCP/UDP/ICMP checksum fully in hardware including pseudo header */
/**
  * @}
  */

/** @defgroup ETH_DMA_Rx_descriptor_buffers ETH DMA Rx descriptor buffers 
  * @{
  */ 
#define ETH_DMARXDESC_BUFFER1     0x00000000U  /*!< DMA Rx Desc Buffer1 */
#define ETH_DMARXDESC_BUFFER2     0x00000001U  /*!< DMA Rx Desc Buffer2 */
/**
  * @}
  */

/** @defgroup ETH_PMT_Flags ETH PMT Flags
  * @{
  */ 
#define ETH_PMT_FLAG_WUFFRPR      0x80000000U  /*!< Wake-Up Frame Filter Register Pointer Reset */
#define ETH_PMT_FLAG_WUFR         0x00000040U  /*!< Wake-Up Frame Received */
#define ETH_PMT_FLAG_MPR          0x00000020U  /*!< Magic Packet Received */
/**
  * @}
  */

/** @defgroup ETH_MMC_Tx_Interrupts ETH MMC Tx Interrupts
  * @{
  */ 
#define ETH_MMC_IT_TGF       0x00200000U  /*!< When Tx good frame counter reaches half the maximum value */
#define ETH_MMC_IT_TGFMSC    0x00008000U  /*!< When Tx good multi col counter reaches half the maximum value */
#define ETH_MMC_IT_TGFSC     0x00004000U  /*!< When Tx good single col counter reaches half the maximum value */
/**
  * @}
  */

/** @defgroup ETH_MMC_Rx_Interrupts ETH MMC Rx Interrupts
  * @{
  */
#define ETH_MMC_IT_RGUF      0x10020000U  /*!< When Rx good unicast frames counter reaches half the maximum value */
#define ETH_MMC_IT_RFAE      0x10000040U  /*!< When Rx alignment error counter reaches half the maximum value */
#define ETH_MMC_IT_RFCE      0x10000020U  /*!< When Rx crc error counter reaches half the maximum value */
/**
  * @}
  */

/** @defgroup ETH_MAC_Flags ETH MAC Flags
  * @{
  */ 
#define ETH_MAC_FLAG_TST     0x00000200U  /*!< Time stamp trigger flag (on MAC) */
#define ETH_MAC_FLAG_MMCT    0x00000040U  /*!< MMC transmit flag  */
#define ETH_MAC_FLAG_MMCR    0x00000020U  /*!< MMC receive flag */
#define ETH_MAC_FLAG_MMC     0x00000010U  /*!< MMC flag (on MAC) */
#define ETH_MAC_FLAG_PMT     0x00000008U  /*!< PMT flag (on MAC) */
/**
  * @}
  */

/** @defgroup ETH_DMA_Flags ETH DMA Flags
  * @{
  */ 
#define ETH_DMA_FLAG_TST               0x20000000U  /*!< Time-stamp trigger interrupt (on DMA) */
#define ETH_DMA_FLAG_PMT               0x10000000U  /*!< PMT interrupt (on DMA) */
#define ETH_DMA_FLAG_MMC               0x08000000U  /*!< MMC interrupt (on DMA) */
#define ETH_DMA_FLAG_DATATRANSFERERROR 0x00800000U  /*!< Error bits 0-Rx DMA, 1-Tx DMA */
#define ETH_DMA_FLAG_READWRITEERROR    0x01000000U  /*!< Error bits 0-write transfer, 1-read transfer */
#define ETH_DMA_FLAG_ACCESSERROR       0x02000000U  /*!< Error bits 0-data buffer, 1-desc. access */
#define ETH_DMA_FLAG_NIS               0x00010000U  /*!< Normal interrupt summary flag */
#define ETH_DMA_FLAG_AIS               0x00008000U  /*!< Abnormal interrupt summary flag */
#define ETH_DMA_FLAG_ER                0x00004000U  /*!< Early receive flag */
#define ETH_DMA_FLAG_FBE               0x00002000U  /*!< Fatal bus error flag */
#define ETH_DMA_FLAG_ET                0x00000400U  /*!< Early transmit flag */
#define ETH_DMA_FLAG_RWT               0x00000200U  /*!< Receive watchdog timeout flag */
#define ETH_DMA_FLAG_RPS               0x00000100U  /*!< Receive process stopped flag */
#define ETH_DMA_FLAG_RBU               0x00000080U  /*!< Receive buffer unavailable flag */
#define ETH_DMA_FLAG_R                 0x00000040U  /*!< Receive flag */
#define ETH_DMA_FLAG_TU                0x00000020U  /*!< Underflow flag */
#define ETH_DMA_FLAG_RO                0x00000010U  /*!< Overflow flag */
#define ETH_DMA_FLAG_TJT               0x00000008U  /*!< Transmit jabber timeout flag */
#define ETH_DMA_FLAG_TBU               0x00000004U  /*!< Transmit buffer unavailable flag */
#define ETH_DMA_FLAG_TPS               0x00000002U  /*!< Transmit process stopped flag */
#define ETH_DMA_FLAG_T                 0x00000001U  /*!< Transmit flag */
/**
  * @}
  */

/** @defgroup ETH_MAC_Interrupts ETH MAC Interrupts 
  * @{
  */ 
#define ETH_MAC_IT_TST       0x00000200U  /*!< Time stamp trigger interrupt (on MAC) */
#define ETH_MAC_IT_MMCT      0x00000040U  /*!< MMC transmit interrupt */
#define ETH_MAC_IT_MMCR      0x00000020U  /*!< MMC receive interrupt */
#define ETH_MAC_IT_MMC       0x00000010U  /*!< MMC interrupt (on MAC) */
#define ETH_MAC_IT_PMT       0x00000008U  /*!< PMT interrupt (on MAC) */
/**
  * @}
  */

/** @defgroup ETH_DMA_Interrupts ETH DMA Interrupts 
  * @{
  */ 
#define ETH_DMA_IT_TST       0x20000000U  /*!< Time-stamp trigger interrupt (on DMA) */
#define ETH_DMA_IT_PMT       0x10000000U  /*!< PMT interrupt (on DMA) */
#define ETH_DMA_IT_MMC       0x08000000U  /*!< MMC interrupt (on DMA) */
#define ETH_DMA_IT_NIS       0x00010000U  /*!< Normal interrupt summary */
#define ETH_DMA_IT_AIS       0x00008000U  /*!< Abnormal interrupt summary */
#define ETH_DMA_IT_ER        0x00004000U  /*!< Early receive interrupt */
#define ETH_DMA_IT_FBE       0x00002000U  /*!< Fatal bus error interrupt */
#define ETH_DMA_IT_ET        0x00000400U  /*!< Early transmit interrupt */
#define ETH_DMA_IT_RWT       0x00000200U  /*!< Receive watchdog timeout interrupt */
#define ETH_DMA_IT_RPS       0x00000100U  /*!< Receive process stopped interrupt */
#define ETH_DMA_IT_RBU       0x00000080U  /*!< Receive buffer unavailable interrupt */
#define ETH_DMA_IT_R         0x00000040U  /*!< Receive interrupt */
#define ETH_DMA_IT_TU        0x00000020U  /*!< Underflow interrupt */
#define ETH_DMA_IT_RO        0x00000010U  /*!< Overflow interrupt */
#define ETH_DMA_IT_TJT       0x00000008U  /*!< Transmit jabber timeout interrupt */
#define ETH_DMA_IT_TBU       0x00000004U  /*!< Transmit buffer unavailable interrupt */
#define ETH_DMA_IT_TPS       0x00000002U  /*!< Transmit process stopped interrupt */
#define ETH_DMA_IT_T         0x00000001U  /*!< Transmit interrupt */
/**
  * @}
  */

/** @defgroup ETH_DMA_transmit_process_state ETH DMA transmit process state 
  * @{
  */ 
#define ETH_DMA_TRANSMITPROCESS_STOPPED     0x00000000U  /*!< Stopped - Reset or Stop Tx Command issued */
#define ETH_DMA_TRANSMITPROCESS_FETCHING    0x00100000U  /*!< Running - fetching the Tx descriptor */
#define ETH_DMA_TRANSMITPROCESS_WAITING     0x00200000U  /*!< Running - waiting for status */
#define ETH_DMA_TRANSMITPROCESS_READING     0x00300000U  /*!< Running - reading the data from host memory */
#define ETH_DMA_TRANSMITPROCESS_SUSPENDED   0x00600000U  /*!< Suspended - Tx Descriptor unavailable */
#define ETH_DMA_TRANSMITPROCESS_CLOSING     0x00700000U  /*!< Running - closing Rx descriptor */

/**
  * @}
  */ 


/** @defgroup ETH_DMA_receive_process_state ETH DMA receive process state 
  * @{
  */ 
#define ETH_DMA_RECEIVEPROCESS_STOPPED      0x00000000U  /*!< Stopped - Reset or Stop Rx Command issued */
#define ETH_DMA_RECEIVEPROCESS_FETCHING     0x00020000U  /*!< Running - fetching the Rx descriptor */
#define ETH_DMA_RECEIVEPROCESS_WAITING      0x00060000U  /*!< Running - waiting for packet */
#define ETH_DMA_RECEIVEPROCESS_SUSPENDED    0x00080000U  /*!< Suspended - Rx Descriptor unavailable */
#define ETH_DMA_RECEIVEPROCESS_CLOSING      0x000A0000U  /*!< Running - closing descriptor */
#define ETH_DMA_RECEIVEPROCESS_QUEUING      0x000E0000U  /*!< Running - queuing the receive frame into host memory */

/**
  * @}
  */

/** @defgroup ETH_DMA_overflow ETH DMA overflow
  * @{
  */ 
#define ETH_DMA_OVERFLOW_RXFIFOCOUNTER      0x10000000U  /*!< Overflow bit for FIFO overflow counter */
#define ETH_DMA_OVERFLOW_MISSEDFRAMECOUNTER 0x00010000U  /*!< Overflow bit for missed frame counter */
/**
  * @}
  */ 

/** @defgroup ETH_EXTI_LINE_WAKEUP ETH EXTI LINE WAKEUP
  * @{
  */ 
#define ETH_EXTI_LINE_WAKEUP              0x00080000U  /*!< External interrupt line 19 Connected to the ETH EXTI Line */

/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup ETH_Exported_Macros ETH Exported Macros
 *  @brief macros to handle interrupts and specific clock configurations
 * @{
 */
 
/** @brief Reset ETH handle state
  * @param  __HANDLE__ specifies the ETH handle.
  * @retval None
  */
#if (USE_HAL_ETH_REGISTER_CALLBACKS == 1)
#define __HAL_ETH_RESET_HANDLE_STATE(__HANDLE__)  do{                                                 \
                                                       (__HANDLE__)->State = HAL_ETH_STATE_RESET;     \
                                                       (__HANDLE__)->MspInitCallback = NULL;          \
                                                       (__HANDLE__)->MspDeInitCallback = NULL;        \
                                                     } while(0)
#else
#define __HAL_ETH_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_ETH_STATE_RESET)
#endif /*USE_HAL_ETH_REGISTER_CALLBACKS */

/** 
  * @brief  Checks whether the specified ETHERNET DMA Tx Desc flag is set or not.
  * @param  __HANDLE__ ETH Handle
  * @param  __FLAG__ specifies the flag of TDES0 to check.
  * @retval the ETH_DMATxDescFlag (SET or RESET).
  */
#define __HAL_ETH_DMATXDESC_GET_FLAG(__HANDLE__, __FLAG__)             ((__HANDLE__)->TxDesc->Status & (__FLAG__) == (__FLAG__))

/**
  * @brief  Checks whether the specified ETHERNET DMA Rx Desc flag is set or not.
  * @param  __HANDLE__ ETH Handle
  * @param  __FLAG__ specifies the flag of RDES0 to check.
  * @retval the ETH_DMATxDescFlag (SET or RESET).
  */
#define __HAL_ETH_DMARXDESC_GET_FLAG(__HANDLE__, __FLAG__)             ((__HANDLE__)->RxDesc->Status & (__FLAG__) == (__FLAG__))

/**
  * @brief  Enables the specified DMA Rx Desc receive interrupt.
  * @param  __HANDLE__ ETH Handle
  * @retval None
  */
#define __HAL_ETH_DMARXDESC_ENABLE_IT(__HANDLE__)                          ((__HANDLE__)->RxDesc->ControlBufferSize &=(~(uint32_t)ETH_DMARXDESC_DIC))

/**
  * @brief  Disables the specified DMA Rx Desc receive interrupt.
  * @param  __HANDLE__ ETH Handle
  * @retval None
  */
#define __HAL_ETH_DMARXDESC_DISABLE_IT(__HANDLE__)                         ((__HANDLE__)->RxDesc->ControlBufferSize |= ETH_DMARXDESC_DIC)

/**
  * @brief  Set the specified DMA Rx Desc Own bit.
  * @param  __HANDLE__ ETH Handle
  * @retval None
  */
#define __HAL_ETH_DMARXDESC_SET_OWN_BIT(__HANDLE__)                           ((__HANDLE__)->RxDesc->Status |= ETH_DMARXDESC_OWN)

/**
  * @brief  Returns the specified ETHERNET DMA Tx Desc collision count.
  * @param  __HANDLE__ ETH Handle                     
  * @retval The Transmit descriptor collision counter value.
  */
#define __HAL_ETH_DMATXDESC_GET_COLLISION_COUNT(__HANDLE__)                   (((__HANDLE__)->TxDesc->Status & ETH_DMATXDESC_CC) >> ETH_DMATXDESC_COLLISION_COUNTSHIFT)

/**
  * @brief  Set the specified DMA Tx Desc Own bit.
  * @param  __HANDLE__ ETH Handle
  * @retval None
  */
#define __HAL_ETH_DMATXDESC_SET_OWN_BIT(__HANDLE__)                       ((__HANDLE__)->TxDesc->Status |= ETH_DMATXDESC_OWN)

/**
  * @brief  Enables the specified DMA Tx Desc Transmit interrupt.
  * @param  __HANDLE__ ETH Handle                   
  * @retval None
  */
#define __HAL_ETH_DMATXDESC_ENABLE_IT(__HANDLE__)                          ((__HANDLE__)->TxDesc->Status |= ETH_DMATXDESC_IC)

/**
  * @brief  Disables the specified DMA Tx Desc Transmit interrupt.
  * @param  __HANDLE__ ETH Handle             
  * @retval None
  */
#define __HAL_ETH_DMATXDESC_DISABLE_IT(__HANDLE__)                          ((__HANDLE__)->TxDesc->Status &= ~ETH_DMATXDESC_IC)

/**
  * @brief  Selects the specified ETHERNET DMA Tx Desc Checksum Insertion.
  * @param  __HANDLE__ ETH Handle  
  * @param  __CHECKSUM__ specifies is the DMA Tx desc checksum insertion.
  *   This parameter can be one of the following values:
  *     @arg ETH_DMATXDESC_CHECKSUMBYPASS : Checksum bypass
  *     @arg ETH_DMATXDESC_CHECKSUMIPV4HEADER : IPv4 header checksum
  *     @arg ETH_DMATXDESC_CHECKSUMTCPUDPICMPSEGMENT : TCP/UDP/ICMP checksum. Pseudo header checksum is assumed to be present
  *     @arg ETH_DMATXDESC_CHECKSUMTCPUDPICMPFULL : TCP/UDP/ICMP checksum fully in hardware including pseudo header
  * @retval None
  */
#define __HAL_ETH_DMATXDESC_CHECKSUM_INSERTION(__HANDLE__, __CHECKSUM__)     ((__HANDLE__)->TxDesc->Status |= (__CHECKSUM__))

/**
  * @brief  Enables the DMA Tx Desc CRC.
  * @param  __HANDLE__ ETH Handle 
  * @retval None
  */
#define __HAL_ETH_DMATXDESC_CRC_ENABLE(__HANDLE__)                          ((__HANDLE__)->TxDesc->Status &= ~ETH_DMATXDESC_DC)

/**
  * @brief  Disables the DMA Tx Desc CRC.
  * @param  __HANDLE__ ETH Handle 
  * @retval None
  */
#define __HAL_ETH_DMATXDESC_CRC_DISABLE(__HANDLE__)                         ((__HANDLE__)->TxDesc->Status |= ETH_DMATXDESC_DC)

/**
  * @brief  Enables the DMA Tx Desc padding for frame shorter than 64 bytes.
  * @param  __HANDLE__ ETH Handle 
  * @retval None
  */
#define __HAL_ETH_DMATXDESC_SHORT_FRAME_PADDING_ENABLE(__HANDLE__)            ((__HANDLE__)->TxDesc->Status &= ~ETH_DMATXDESC_DP)

/**
  * @brief  Disables the DMA Tx Desc padding for frame shorter than 64 bytes.
  * @param  __HANDLE__ ETH Handle 
  * @retval None
  */
#define __HAL_ETH_DMATXDESC_SHORT_FRAME_PADDING_DISABLE(__HANDLE__)           ((__HANDLE__)->TxDesc->Status |= ETH_DMATXDESC_DP)

/** 
 * @brief  Enables the specified ETHERNET MAC interrupts.
  * @param  __HANDLE__    ETH Handle
  * @param  __INTERRUPT__ specifies the ETHERNET MAC interrupt sources to be
  *   enabled or disabled.
  *   This parameter can be any combination of the following values:
  *     @arg ETH_MAC_IT_TST : Time stamp trigger interrupt 
  *     @arg ETH_MAC_IT_PMT : PMT interrupt 
  * @retval None
  */
#define __HAL_ETH_MAC_ENABLE_IT(__HANDLE__, __INTERRUPT__)                 ((__HANDLE__)->Instance->MACIMR |= (__INTERRUPT__))

/**
  * @brief  Disables the specified ETHERNET MAC interrupts.
  * @param  __HANDLE__    ETH Handle
  * @param  __INTERRUPT__ specifies the ETHERNET MAC interrupt sources to be
  *   enabled or disabled.
  *   This parameter can be any combination of the following values:
  *     @arg ETH_MAC_IT_TST : Time stamp trigger interrupt 
  *     @arg ETH_MAC_IT_PMT : PMT interrupt
  * @retval None
  */
#define __HAL_ETH_MAC_DISABLE_IT(__HANDLE__, __INTERRUPT__)                ((__HANDLE__)->Instance->MACIMR &= ~(__INTERRUPT__))

/**
  * @brief  Initiate a Pause Control Frame (Full-duplex only).
  * @param  __HANDLE__ ETH Handle
  * @retval None
  */
#define __HAL_ETH_INITIATE_PAUSE_CONTROL_FRAME(__HANDLE__)              ((__HANDLE__)->Instance->MACFCR |= ETH_MACFCR_FCBBPA)

/**
  * @brief  Checks whether the ETHERNET flow control busy bit is set or not.
  * @param  __HANDLE__ ETH Handle
  * @retval The new state of flow control busy status bit (SET or RESET).
  */
#define __HAL_ETH_GET_FLOW_CONTROL_BUSY_STATUS(__HANDLE__)               (((__HANDLE__)->Instance->MACFCR & ETH_MACFCR_FCBBPA) == ETH_MACFCR_FCBBPA)

/**
  * @brief  Enables the MAC Back Pressure operation activation (Half-duplex only).
  * @param  __HANDLE__ ETH Handle
  * @retval None
  */
#define __HAL_ETH_BACK_PRESSURE_ACTIVATION_ENABLE(__HANDLE__)          ((__HANDLE__)->Instance->MACFCR |= ETH_MACFCR_FCBBPA)

/**
  * @brief  Disables the MAC BackPressure operation activation (Half-duplex only).
  * @param  __HANDLE__ ETH Handle
  * @retval None
  */
#define __HAL_ETH_BACK_PRESSURE_ACTIVATION_DISABLE(__HANDLE__)         ((__HANDLE__)->Instance->MACFCR &= ~ETH_MACFCR_FCBBPA)

/**
  * @brief  Checks whether the specified ETHERNET MAC flag is set or not.
  * @param  __HANDLE__ ETH Handle
  * @param  __FLAG__ specifies the flag to check.
  *   This parameter can be one of the following values:
  *     @arg ETH_MAC_FLAG_TST  : Time stamp trigger flag   
  *     @arg ETH_MAC_FLAG_MMCT : MMC transmit flag  
  *     @arg ETH_MAC_FLAG_MMCR : MMC receive flag   
  *     @arg ETH_MAC_FLAG_MMC  : MMC flag  
  *     @arg ETH_MAC_FLAG_PMT  : PMT flag  
  * @retval The state of ETHERNET MAC flag.
  */
#define __HAL_ETH_MAC_GET_FLAG(__HANDLE__, __FLAG__)                   (((__HANDLE__)->Instance->MACSR &( __FLAG__)) == ( __FLAG__))

/** 
  * @brief  Enables the specified ETHERNET DMA interrupts.
  * @param  __HANDLE__    ETH Handle
  * @param  __INTERRUPT__ specifies the ETHERNET DMA interrupt sources to be
  *   enabled @ref ETH_DMA_Interrupts
  * @retval None
  */
#define __HAL_ETH_DMA_ENABLE_IT(__HANDLE__, __INTERRUPT__)                 ((__HANDLE__)->Instance->DMAIER |= (__INTERRUPT__))

/**
  * @brief  Disables the specified ETHERNET DMA interrupts.
  * @param  __HANDLE__    ETH Handle
  * @param  __INTERRUPT__ specifies the ETHERNET DMA interrupt sources to be
  *   disabled. @ref ETH_DMA_Interrupts
  * @retval None
  */
#define __HAL_ETH_DMA_DISABLE_IT(__HANDLE__, __INTERRUPT__)                ((__HANDLE__)->Instance->DMAIER &= ~(__INTERRUPT__))

/**
  * @brief  Clears the ETHERNET DMA IT pending bit.
  * @param  __HANDLE__    ETH Handle
  * @param  __INTERRUPT__ specifies the interrupt pending bit to clear. @ref ETH_DMA_Interrupts
  * @retval None
  */
#define __HAL_ETH_DMA_CLEAR_IT(__HANDLE__, __INTERRUPT__)      ((__HANDLE__)->Instance->DMASR =(__INTERRUPT__))

/**
  * @brief  Checks whether the specified ETHERNET DMA flag is set or not.
* @param  __HANDLE__ ETH Handle
  * @param  __FLAG__ specifies the flag to check. @ref ETH_DMA_Flags
  * @retval The new state of ETH_DMA_FLAG (SET or RESET).
  */
#define __HAL_ETH_DMA_GET_FLAG(__HANDLE__, __FLAG__)                   (((__HANDLE__)->Instance->DMASR &( __FLAG__)) == ( __FLAG__))

/**
  * @brief  Checks whether the specified ETHERNET DMA flag is set or not.
  * @param  __HANDLE__ ETH Handle
  * @param  __FLAG__ specifies the flag to clear. @ref ETH_DMA_Flags
  * @retval The new state of ETH_DMA_FLAG (SET or RESET).
  */
#define __HAL_ETH_DMA_CLEAR_FLAG(__HANDLE__, __FLAG__)                 ((__HANDLE__)->Instance->DMASR = (__FLAG__))

/**
  * @brief  Checks whether the specified ETHERNET DMA overflow flag is set or not.
  * @param  __HANDLE__ ETH Handle
  * @param  __OVERFLOW__ specifies the DMA overflow flag to check.
  *   This parameter can be one of the following values:
  *     @arg ETH_DMA_OVERFLOW_RXFIFOCOUNTER : Overflow for FIFO Overflows Counter
  *     @arg ETH_DMA_OVERFLOW_MISSEDFRAMECOUNTER : Overflow for Buffer Unavailable Missed Frame Counter
  * @retval The state of ETHERNET DMA overflow Flag (SET or RESET).
  */
#define __HAL_ETH_GET_DMA_OVERFLOW_STATUS(__HANDLE__, __OVERFLOW__)       (((__HANDLE__)->Instance->DMAMFBOCR & (__OVERFLOW__)) == (__OVERFLOW__))

/**
  * @brief  Set the DMA Receive status watchdog timer register value
  * @param  __HANDLE__ ETH Handle
  * @param  __VALUE__ DMA Receive status watchdog timer register value   
  * @retval None
  */
#define __HAL_ETH_SET_RECEIVE_WATCHDOG_TIMER(__HANDLE__, __VALUE__)       ((__HANDLE__)->Instance->DMARSWTR = (__VALUE__))

/** 
  * @brief  Enables any unicast packet filtered by the MAC address
  *   recognition to be a wake-up frame.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_GLOBAL_UNICAST_WAKEUP_ENABLE(__HANDLE__)               ((__HANDLE__)->Instance->MACPMTCSR |= ETH_MACPMTCSR_GU)

/**
  * @brief  Disables any unicast packet filtered by the MAC address
  *   recognition to be a wake-up frame.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_GLOBAL_UNICAST_WAKEUP_DISABLE(__HANDLE__)              ((__HANDLE__)->Instance->MACPMTCSR &= ~ETH_MACPMTCSR_GU)

/**
  * @brief  Enables the MAC Wake-Up Frame Detection.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_FRAME_DETECTION_ENABLE(__HANDLE__)              ((__HANDLE__)->Instance->MACPMTCSR |= ETH_MACPMTCSR_WFE)

/**
  * @brief  Disables the MAC Wake-Up Frame Detection.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_FRAME_DETECTION_DISABLE(__HANDLE__)             ((__HANDLE__)->Instance->MACPMTCSR &= ~ETH_MACPMTCSR_WFE)

/**
  * @brief  Enables the MAC Magic Packet Detection.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_MAGIC_PACKET_DETECTION_ENABLE(__HANDLE__)              ((__HANDLE__)->Instance->MACPMTCSR |= ETH_MACPMTCSR_MPE)

/**
  * @brief  Disables the MAC Magic Packet Detection.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_MAGIC_PACKET_DETECTION_DISABLE(__HANDLE__)             ((__HANDLE__)->Instance->MACPMTCSR &= ~ETH_MACPMTCSR_WFE)

/**
  * @brief  Enables the MAC Power Down.
  * @param  __HANDLE__ ETH Handle
  * @retval None
  */
#define __HAL_ETH_POWER_DOWN_ENABLE(__HANDLE__)                         ((__HANDLE__)->Instance->MACPMTCSR |= ETH_MACPMTCSR_PD)

/**
  * @brief  Disables the MAC Power Down.
  * @param  __HANDLE__ ETH Handle
  * @retval None
  */
#define __HAL_ETH_POWER_DOWN_DISABLE(__HANDLE__)                        ((__HANDLE__)->Instance->MACPMTCSR &= ~ETH_MACPMTCSR_PD)

/**
  * @brief  Checks whether the specified ETHERNET PMT flag is set or not.
  * @param  __HANDLE__ ETH Handle.
  * @param  __FLAG__ specifies the flag to check.
  *   This parameter can be one of the following values:
  *     @arg ETH_PMT_FLAG_WUFFRPR : Wake-Up Frame Filter Register Pointer Reset 
  *     @arg ETH_PMT_FLAG_WUFR    : Wake-Up Frame Received 
  *     @arg ETH_PMT_FLAG_MPR     : Magic Packet Received
  * @retval The new state of ETHERNET PMT Flag (SET or RESET).
  */
#define __HAL_ETH_GET_PMT_FLAG_STATUS(__HANDLE__, __FLAG__)               (((__HANDLE__)->Instance->MACPMTCSR &( __FLAG__)) == ( __FLAG__))

/** 
  * @brief  Preset and Initialize the MMC counters to almost-full value: 0xFFFF_FFF0 (full - 16)
  * @param   __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_MMC_COUNTER_FULL_PRESET(__HANDLE__)                     ((__HANDLE__)->Instance->MMCCR |= (ETH_MMCCR_MCFHP | ETH_MMCCR_MCP))

/**
  * @brief  Preset and Initialize the MMC counters to almost-half value: 0x7FFF_FFF0 (half - 16)
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_MMC_COUNTER_HALF_PRESET(__HANDLE__)                     do{(__HANDLE__)->Instance->MMCCR &= ~ETH_MMCCR_MCFHP;\
                                                                          (__HANDLE__)->Instance->MMCCR |= ETH_MMCCR_MCP;} while (0)

/**
  * @brief  Enables the MMC Counter Freeze.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_MMC_COUNTER_FREEZE_ENABLE(__HANDLE__)                  ((__HANDLE__)->Instance->MMCCR |= ETH_MMCCR_MCF)

/**
  * @brief  Disables the MMC Counter Freeze.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_MMC_COUNTER_FREEZE_DISABLE(__HANDLE__)                 ((__HANDLE__)->Instance->MMCCR &= ~ETH_MMCCR_MCF)

/**
  * @brief  Enables the MMC Reset On Read.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_ETH_MMC_RESET_ONREAD_ENABLE(__HANDLE__)                ((__HANDLE__)->Instance->MMCCR |= ETH_MMCCR_ROR)

/**
  * @brief  Disables the MMC Reset On Read.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_ETH_MMC_RESET_ONREAD_DISABLE(__HANDLE__)               ((__HANDLE__)->Instance->MMCCR &= ~ETH_MMCCR_ROR)

/**
  * @brief  Enables the MMC Counter Stop Rollover.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_ETH_MMC_COUNTER_ROLLOVER_ENABLE(__HANDLE__)            ((__HANDLE__)->Instance->MMCCR &= ~ETH_MMCCR_CSR)

/**
  * @brief  Disables the MMC Counter Stop Rollover.
  * @param  __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_ETH_MMC_COUNTER_ROLLOVER_DISABLE(__HANDLE__)           ((__HANDLE__)->Instance->MMCCR |= ETH_MMCCR_CSR)

/**
  * @brief  Resets the MMC Counters.
  * @param   __HANDLE__ ETH Handle.
  * @retval None
  */
#define __HAL_ETH_MMC_COUNTERS_RESET(__HANDLE__)                         ((__HANDLE__)->Instance->MMCCR |= ETH_MMCCR_CR)

/**
  * @brief  Enables the specified ETHERNET MMC Rx interrupts.
  * @param   __HANDLE__ ETH Handle.
  * @param  __INTERRUPT__ specifies the ETHERNET MMC interrupt sources to be enabled or disabled.
  *   This parameter can be one of the following values:  
  *     @arg ETH_MMC_IT_RGUF  : When Rx good unicast frames counter reaches half the maximum value 
  *     @arg ETH_MMC_IT_RFAE  : When Rx alignment error counter reaches half the maximum value 
  *     @arg ETH_MMC_IT_RFCE  : When Rx crc error counter reaches half the maximum value
  * @retval None
  */
#define __HAL_ETH_MMC_RX_IT_ENABLE(__HANDLE__, __INTERRUPT__)               (__HANDLE__)->Instance->MMCRIMR &= ~((__INTERRUPT__) & 0xEFFFFFFFU)
/**
  * @brief  Disables the specified ETHERNET MMC Rx interrupts.
  * @param   __HANDLE__ ETH Handle.
  * @param  __INTERRUPT__ specifies the ETHERNET MMC interrupt sources to be enabled or disabled.
  *   This parameter can be one of the following values: 
  *     @arg ETH_MMC_IT_RGUF  : When Rx good unicast frames counter reaches half the maximum value 
  *     @arg ETH_MMC_IT_RFAE  : When Rx alignment error counter reaches half the maximum value 
  *     @arg ETH_MMC_IT_RFCE  : When Rx crc error counter reaches half the maximum value
  * @retval None
  */
#define __HAL_ETH_MMC_RX_IT_DISABLE(__HANDLE__, __INTERRUPT__)              (__HANDLE__)->Instance->MMCRIMR |= ((__INTERRUPT__) & 0xEFFFFFFFU)
/**
  * @brief  Enables the specified ETHERNET MMC Tx interrupts.
  * @param   __HANDLE__ ETH Handle.
  * @param  __INTERRUPT__ specifies the ETHERNET MMC interrupt sources to be enabled or disabled.
  *   This parameter can be one of the following values:  
  *     @arg ETH_MMC_IT_TGF   : When Tx good frame counter reaches half the maximum value 
  *     @arg ETH_MMC_IT_TGFMSC: When Tx good multi col counter reaches half the maximum value 
  *     @arg ETH_MMC_IT_TGFSC : When Tx good single col counter reaches half the maximum value 
  * @retval None
  */
#define __HAL_ETH_MMC_TX_IT_ENABLE(__HANDLE__, __INTERRUPT__)            ((__HANDLE__)->Instance->MMCRIMR &= ~ (__INTERRUPT__))

/**
  * @brief  Disables the specified ETHERNET MMC Tx interrupts.
  * @param   __HANDLE__ ETH Handle.
  * @param  __INTERRUPT__ specifies the ETHERNET MMC interrupt sources to be enabled or disabled.
  *   This parameter can be one of the following values:  
  *     @arg ETH_MMC_IT_TGF   : When Tx good frame counter reaches half the maximum value 
  *     @arg ETH_MMC_IT_TGFMSC: When Tx good multi col counter reaches half the maximum value 
  *     @arg ETH_MMC_IT_TGFSC : When Tx good single col counter reaches half the maximum value 
  * @retval None
  */
#define __HAL_ETH_MMC_TX_IT_DISABLE(__HANDLE__, __INTERRUPT__)           ((__HANDLE__)->Instance->MMCRIMR |= (__INTERRUPT__))

/**
  * @brief  Enables the ETH External interrupt line.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_EXTI_ENABLE_IT()    EXTI->IMR |= (ETH_EXTI_LINE_WAKEUP)

/**
  * @brief  Disables the ETH External interrupt line.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_EXTI_DISABLE_IT()   EXTI->IMR &= ~(ETH_EXTI_LINE_WAKEUP)

/**
  * @brief Enable event on ETH External event line.
  * @retval None.
  */
#define __HAL_ETH_WAKEUP_EXTI_ENABLE_EVENT()  EXTI->EMR |= (ETH_EXTI_LINE_WAKEUP)

/**
  * @brief Disable event on ETH External event line
  * @retval None.
  */
#define __HAL_ETH_WAKEUP_EXTI_DISABLE_EVENT() EXTI->EMR &= ~(ETH_EXTI_LINE_WAKEUP)

/**
  * @brief  Get flag of the ETH External interrupt line.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_EXTI_GET_FLAG()     EXTI->PR & (ETH_EXTI_LINE_WAKEUP)

/**
  * @brief  Clear flag of the ETH External interrupt line.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_EXTI_CLEAR_FLAG()   EXTI->PR = (ETH_EXTI_LINE_WAKEUP)

/**
  * @brief  Enables rising edge trigger to the ETH External interrupt line.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_EXTI_ENABLE_RISING_EDGE_TRIGGER()  EXTI->RTSR |= ETH_EXTI_LINE_WAKEUP
                                                            
/**
  * @brief  Disables the rising edge trigger to the ETH External interrupt line.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_EXTI_DISABLE_RISING_EDGE_TRIGGER()  EXTI->RTSR &= ~(ETH_EXTI_LINE_WAKEUP)

/**
  * @brief  Enables falling edge trigger to the ETH External interrupt line.
  * @retval None
  */                                                      
#define __HAL_ETH_WAKEUP_EXTI_ENABLE_FALLING_EDGE_TRIGGER()  EXTI->FTSR |= (ETH_EXTI_LINE_WAKEUP)

/**
  * @brief  Disables falling edge trigger to the ETH External interrupt line.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_EXTI_DISABLE_FALLING_EDGE_TRIGGER()  EXTI->FTSR &= ~(ETH_EXTI_LINE_WAKEUP)

/**
  * @brief  Enables rising/falling edge trigger to the ETH External interrupt line.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_EXTI_ENABLE_FALLINGRISING_TRIGGER()  do{EXTI->RTSR |= ETH_EXTI_LINE_WAKEUP;\
                                                                 EXTI->FTSR |= ETH_EXTI_LINE_WAKEUP;\
                                                                }while(0U)

/**
  * @brief  Disables rising/falling edge trigger to the ETH External interrupt line.
  * @retval None
  */
#define __HAL_ETH_WAKEUP_EXTI_DISABLE_FALLINGRISING_TRIGGER() do{EXTI->RTSR &= ~(ETH_EXTI_LINE_WAKEUP);\
                                                                 EXTI->FTSR &= ~(ETH_EXTI_LINE_WAKEUP);\
                                                                }while(0U)

/**
  * @brief Generate a Software interrupt on selected EXTI line.
  * @retval None.
  */
#define __HAL_ETH_WAKEUP_EXTI_GENERATE_SWIT()                  EXTI->SWIER|= ETH_EXTI_LINE_WAKEUP

/**
  * @}
  */
/* Exported functions --------------------------------------------------------*/

/** @addtogroup ETH_Exported_Functions
  * @{
  */

/* Initialization and de-initialization functions  ****************************/

/** @addtogroup ETH_Exported_Functions_Group1
  * @{
  */
HAL_StatusTypeDef HAL_ETH_Init(ETH_HandleTypeDef *heth);
HAL_StatusTypeDef HAL_ETH_DeInit(ETH_HandleTypeDef *heth);
void HAL_ETH_MspInit(ETH_HandleTypeDef *heth);
void HAL_ETH_MspDeInit(ETH_HandleTypeDef *heth);
HAL_StatusTypeDef HAL_ETH_DMATxDescListInit(ETH_HandleTypeDef *heth, ETH_DMADescTypeDef *DMATxDescTab, uint8_t* TxBuff, uint32_t TxBuffCount);
HAL_StatusTypeDef HAL_ETH_DMARxDescListInit(ETH_HandleTypeDef *heth, ETH_DMADescTypeDef *DMARxDescTab, uint8_t *RxBuff, uint32_t RxBuffCount);
/* Callbacks Register/UnRegister functions  ***********************************/
#if (USE_HAL_ETH_REGISTER_CALLBACKS == 1)
HAL_StatusTypeDef HAL_ETH_RegisterCallback(ETH_HandleTypeDef *heth, HAL_ETH_CallbackIDTypeDef CallbackID, pETH_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_ETH_UnRegisterCallback(ETH_HandleTypeDef *heth, HAL_ETH_CallbackIDTypeDef CallbackID);
#endif /* USE_HAL_ETH_REGISTER_CALLBACKS */

/**
  * @}
  */
/* IO operation functions  ****************************************************/

/** @addtogroup ETH_Exported_Functions_Group2
  * @{
  */
HAL_StatusTypeDef HAL_ETH_TransmitFrame(ETH_HandleTypeDef *heth, uint32_t FrameLength);
HAL_StatusTypeDef HAL_ETH_GetReceivedFrame(ETH_HandleTypeDef *heth);
/* Communication with PHY functions*/
HAL_StatusTypeDef HAL_ETH_ReadPHYRegister(ETH_HandleTypeDef *heth, uint16_t PHYReg, uint32_t *RegValue);
HAL_StatusTypeDef HAL_ETH_WritePHYRegister(ETH_HandleTypeDef *heth, uint16_t PHYReg, uint32_t RegValue);
/* Non-Blocking mode: Interrupt */
HAL_StatusTypeDef HAL_ETH_GetReceivedFrame_IT(ETH_HandleTypeDef *heth);
void HAL_ETH_IRQHandler(ETH_HandleTypeDef *heth);
/* Callback in non blocking modes (Interrupt) */
void HAL_ETH_TxCpltCallback(ETH_HandleTypeDef *heth);
void HAL_ETH_RxCpltCallback(ETH_HandleTypeDef *heth);
void HAL_ETH_ErrorCallback(ETH_HandleTypeDef *heth);
/**
  * @}
  */

/* Peripheral Control functions  **********************************************/

/** @addtogroup ETH_Exported_Functions_Group3
  * @{
  */

HAL_StatusTypeDef HAL_ETH_Start(ETH_HandleTypeDef *heth);
HAL_StatusTypeDef HAL_ETH_Stop(ETH_HandleTypeDef *heth);
HAL_StatusTypeDef HAL_ETH_ConfigMAC(ETH_HandleTypeDef *heth, ETH_MACInitTypeDef *macconf);
HAL_StatusTypeDef HAL_ETH_ConfigDMA(ETH_HandleTypeDef *heth, ETH_DMAInitTypeDef *dmaconf);
/**
  * @}
  */ 

/* Peripheral State functions  ************************************************/

/** @addtogroup ETH_Exported_Functions_Group4
  * @{
  */
HAL_ETH_StateTypeDef HAL_ETH_GetState(ETH_HandleTypeDef *heth);
/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F427xx ||\
          STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */
  
#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_ETH_H */


/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
