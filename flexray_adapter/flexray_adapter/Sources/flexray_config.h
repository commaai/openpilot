#ifndef FLEXRAY_CONFIG_H
#define FLEXRAY_CONFIG_H

#ifdef __cplusplus
extern "C"{
#endif

#include <stdint.h>

#define FR_CHANNEL_A 0x00000000U
#define FR_CHANNEL_B 0x00000001U
#define FR_CHANNEL_AB 0x00000002U

#define MSG_BUF_TYPE_TX_FLAG (0x00000001)
#define MSG_BUF_CHANNEL_A_ENABLED_FLAG (0x00000001 << 1)
#define MSG_BUF_CHANNEL_B_ENABLED_FLAG (0x00000001 << 2)
#define MSG_BUF_CYCLE_COUNTER_FILTER_ENABLED_FLAG (0x00000001 << 3)
#define MSG_BUF_PAYLOAD_PREAMBLE_INDICATOR_FLAG (0x00000001 << 4)
#define MSG_BUF_DYNAMIC_PAYLOAD_LENGTH_ENABLED_FLAG (0x00000001 << 5)
#define EXTRACT_MSG_BUF_CYCLE_COUNTER_FILTER_VALUE(flags) ((flags & 0x00FF0000) >> 16)
#define EXTRACT_MSG_BUF_CYCLE_COUNTER_FILTER_MASK(flags) ((flags & 0xFF000000) >> 24)

typedef struct
{
	/*
		Bits 0: Type, 0 for Rx, 1 for tx
		Bits 1: Channel A enable/disable
		Bits 2: Channel B enable/disable
		Bits 3: Cycle counter filter enable/disable
		Bits 4: Tx only: Payload preamble indicator
		Bits 5: Tx only: Dynamic payload length enable/disable, for slots in dynamic segment only.
		Bits 6-15: Reserved
		bits 16-23: Cycle counter filter value
		bits 24-31: Cycle counter filter mask
	*/
    uint32_t flags;
    /* Slot id */
    uint16_t frame_id;
    /* Tx only: max payload length in words, for slots in dynamic segment only, should be equal or less than pPayloadLengthDynMax */
    uint16_t payload_length_max;
} fr_msg_buffer;

/*
 * Enable bit: 0 disable, 1 enable
 * Mode bit: 0 Acceptance, 1 Rejection
 * */
#define FR_FIFO_FILTER0_ENABLE_MASK (0x0001)
#define FR_FIFO_FILTER0_MODE_MASK (0x0001 << 1)
#define FR_FIFO_FILTER1_ENABLE_MASK (0x0001 << 2)
#define FR_FIFO_FILTER1_MODE_MASK (0x0001 << 3)
#define FR_FIFO_FILTER2_ENABLE_MASK (0x0001 << 4)
#define FR_FIFO_FILTER2_MODE_MASK (0x0001 << 5)
#define FR_FIFO_FILTER3_ENABLE_MASK (0x0001 << 6)
#define FR_FIFO_FILTER3_MODE_MASK (0x0001 << 7)

typedef struct
{
    /* The number of elements in FIFO queue. */
    uint16_t depth;
    /* Message ID filter mask value */
    /* FlexRay spec V3.0.1 4.3.2 Message ID (16 bits) */
    /* Message ID filter match value */
    uint16_t message_id_filter_value;
    uint16_t message_id_filter_mask;
    /* Bits 0: Filter 0 enable/disable
     * Bits 1: Filter 0 mode
     * Bits 2: Filter 1 enable/disable
     * Bits 3: Filter 1 mode
     * Bits 4: Filter 2 enable/disable
     * Bits 5: Filter 2 mode
     * Bits 6: Filter 3 enable/disable
     * Bits 7: Filter 3 mode
     *      * */
    uint16_t flags;
    /* MPC5748G Reference Manual 46.8.9.9.2 RX FIFO Frame ID Range Rejection Filter, 46.8.9.9.3 RX FIFO Frame ID Range Acceptance filter */
    uint16_t filter0_slot_id_lower;
    uint16_t filter0_slot_id_upper;
    uint16_t filter1_slot_id_lower;
    uint16_t filter1_slot_id_upper;
    uint16_t filter2_slot_id_lower;
    uint16_t filter2_slot_id_upper;
    uint16_t filter3_slot_id_lower;
    uint16_t filter3_slot_id_upper;
} fr_fifo_queue_config;

#define MAX_MSG_BUFS 128

/* Single channel mode enabled bit: 0 is disabled, 1 is enabled. */
#define FR_CONFIG_FLAG_SINGLE_CHANNEL_MODE_ENABLED_MASK (0x0001)
/* Clock source type bit: 0 is crystal oscillator, 1 is PLL. */
#define FR_CONFIG_FLAG_CLOCK_SOURCE_TYPE_MASK (0x0001 << 1)
/* FIFOA enabled bit: 0 is disabled, 1 is enabled. */
#define FR_CONFIG_FLAG_FIFOA_ENABLED_MASK (0x0001 << 2)
/* FIFOB enabled bit: 0 is disabled, 1 is enabled. */
#define FR_CONFIG_FLAG_FIFOB_ENABLED_MASK (0x0001 << 3)
/* Enable logging of status data (FlexRay spec 2.1, Section 9.3.1.3 Protocol status data), send to client in health message. */
#define FR_CONFIG_FLAG_LOG_STATUS_DATA_MASK (0x0001 << 4)

typedef struct
{
	/* Cluster configuration, using naming convention in FlexRay Protocol Specification Version 2.1 Revision A */
	uint32_t gdMacrotick;
	uint32_t gPayloadLengthStatic;
	uint32_t gNumberOfStaticSlots;
	uint32_t gdStaticSlot;
	uint32_t gdActionPointOffset;
	uint32_t gNumberOfMinislots;
	uint32_t gdMinislot;
	uint32_t gdMiniSlotActionPointOffset;
	uint32_t gdSymbolWindow;
	uint32_t gdNIT;
	uint32_t gOffsetCorrectionStart;
	uint32_t gdWakeupRxWindow;
	uint32_t gColdStartAttempts;
	uint32_t gListenNoise;
	uint32_t gMaxWithoutClockCorrectionFatal;
	uint32_t gMaxWithoutClockCorrectionPassive;
	uint32_t gNetworkManagementVectorLength;
	uint32_t gSyncFrameIDCountMax;
	uint32_t gdCasRxLowMax;
	uint32_t gdDynamicSlotIdlePhase;
	uint32_t gdTSSTransmitter;
	uint32_t gdWakeupSymbolRxIdle;
	uint32_t gdWakeupSymbolRxLow;
	uint32_t gdWakeupSymbolTxActive;
	uint32_t gdWakeupSymbolTxIdle;
	/* Node configuration */
	uint32_t pChannels;
	uint32_t pWakeupChannel;
	uint32_t pWakeupPattern;
	uint32_t pPayloadLengthDynMax;
	uint32_t pMicroPerCycle;
	uint32_t pdListenTimeout;
	uint32_t pRateCorrectionOut;
	uint32_t pKeySlotId;
	uint32_t pKeySlotOnlyEnabled;
	uint32_t pKeySlotUsedForStartup;
	uint32_t pKeySlotUsedForSync;
	uint32_t pLatestTx;
	uint32_t pOffsetCorrectionOut;
	uint32_t pdAcceptedStartupRange;
	uint32_t pAllowPassiveToActive;
	uint32_t pClusterDriftDamping;
	uint32_t pDecodingCorrection;
	uint32_t pDelayCompensationA;
	uint32_t pDelayCompensationB;
	uint32_t pMacroInitialOffsetA;
	uint32_t pMacroInitialOffsetB;
	uint32_t pMicroInitialOffsetA;
	uint32_t pMicroInitialOffsetB;
	uint32_t pAllowHaltDueToClock;
	uint32_t pdMaxDrift;
	/* Board configuration flags: Bit 0: Single channel mode enable, Bit 1: Clock Source Choose, Bit 2: FIFOA enable, Bit 3: FIFOB enable */
	uint16_t flags;
    uint16_t bit_rate;
    fr_fifo_queue_config fifoa_config;
    fr_fifo_queue_config fifob_config;
	uint16_t individual_rx_msg_buf_count;
	uint16_t individual_tx_msg_buf_count;
	/* Individual Rx/Tx message buffers configurations, Rx bufs first, then Tx bufs, total count is individual_rx_msg_buf_count + individual_tx_msg_buf_count. */
	fr_msg_buffer msg_bufs[MAX_MSG_BUFS];
} fr_config;

#ifdef __cplusplus
}
#endif

#endif /* FLEXRAY_CONFIG_H */
