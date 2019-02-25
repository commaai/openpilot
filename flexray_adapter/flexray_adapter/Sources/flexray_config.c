#include "flexray_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default flexray config, should never be used. */
fr_config g_fr_config ={
	.gPayloadLengthStatic  = 8U,
	.gNumberOfStaticSlots  = 60U,
	.gdStaticSlot  = 65U,
	.gdActionPointOffset  = 6U,
	.gNumberOfMinislots  = 163U,
	.gdMinislot  = 6U,
	.gdMiniSlotActionPointOffset  = 3U,
	.gdSymbolWindow  = 19U,
	.gdNIT  = 103U,
	.gOffsetCorrectionStart  = 4920U,
	.gdWakeupRxWindow  = 301U,
	.gColdStartAttempts  = 10U,
	.gListenNoise  = 2U,
	.gMaxWithoutClockCorrectionFatal  = 14U,
	.gMaxWithoutClockCorrectionPassive  = 10U,
	.gNetworkManagementVectorLength  = 2U,
	.gSyncFrameIDCountMax  = 5U,
	.gdCasRxLowMax  = 91U,
	.gdDynamicSlotIdlePhase  = 1U,
	.gdTSSTransmitter  = 11U,
	.gdWakeupSymbolRxIdle  = 59U,
	.gdWakeupSymbolRxLow  = 55U,
	.gdWakeupSymbolTxActive  = 60U,
	.gdWakeupSymbolTxIdle  = 180U,

	.pChannels = 0U, /* FR_CHANNEL_A */
	.pWakeupChannel = 0U, /* FR_CHANNEL_A */
	.pWakeupPattern  = 63U,
	.pPayloadLengthDynMax  = 8U,
	.pMicroPerCycle  = 200000U,
	.pdListenTimeout  = 401202U,
	.pRateCorrectionOut  = 962U,
	.pKeySlotId = 1U,
	.pKeySlotOnlyEnabled  = 0U,
	.pKeySlotUsedForStartup  = 1U,
	.pKeySlotUsedForSync  = 1U,
	.pLatestTx = 157U,
	.pOffsetCorrectionOut  = 120U,
	.pdAcceptedStartupRange  = 110U,
	.pAllowPassiveToActive  = 0U,
	.pClusterDriftDamping  = 1U,
	.pDecodingCorrection  = 36U,
	.pDelayCompensationA  = 0U,
	.pDelayCompensationB  = 0U,
	.pMacroInitialOffsetA  = 8U,
	.pMacroInitialOffsetB  = 8U,
	.pMicroInitialOffsetA  = 24U,
	.pMicroInitialOffsetB  = 24U,
	.pAllowHaltDueToClock  = 1U,
	.pdMaxDrift = 600U,
	.flags = 0,
	.bit_rate = 0U,  /* Bus speed: 10 Mb/s */
	{},
	{},
	.individual_rx_msg_buf_count = 4U,
	.individual_tx_msg_buf_count = 3U,
	.msg_bufs = {
		{
			MSG_BUF_CHANNEL_A_ENABLED_FLAG,
			2U, /* Receive Frame ID */
			8U, /* Data Length in Words */
		},
		{
			MSG_BUF_CHANNEL_A_ENABLED_FLAG,
			4U, /* Receive Frame ID */
			8U, /* Data Length in Words */
		},
		{
			MSG_BUF_CHANNEL_A_ENABLED_FLAG,
			6U, /* Receive Frame ID */
			8U, /* Data Length in Words */
		},
		{
			MSG_BUF_CHANNEL_A_ENABLED_FLAG,
			100U, /* Receive Frame ID */
			8U, /* Data Length in Words */
		},
		{
			MSG_BUF_TYPE_TX_FLAG|MSG_BUF_CHANNEL_A_ENABLED_FLAG,
			1U, /* Transmit Frame ID, it is the Key Slot */
			8U, /* Data Length in Words */
		},
		{
			MSG_BUF_TYPE_TX_FLAG|MSG_BUF_CHANNEL_A_ENABLED_FLAG,
			5U, /* Transmit Frame ID */
			8U, /* Data Length in Words */
		},
		{
			MSG_BUF_TYPE_TX_FLAG|MSG_BUF_CHANNEL_A_ENABLED_FLAG,
			7U, /* Transmit Frame ID */
			8U, /* Data Length in Words */
		},
	}
};

#ifdef __cplusplus
}
#endif
