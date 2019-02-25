#ifdef __cplusplus
extern "C"{
#endif

#include <math.h>
#include "flexray_driver.h"
#include "flexray_registers.h"
#include "platform_defs.h"

#define MPC5748G_MVR_VALUE 0xA568U
#define MPC5748G_FR_CC_ADDRESS 0xffe50000U

#define READ_FR_REGISTER16(offset) (*((volatile uint16_t *)MPC5748G_FR_CC_ADDRESS + (uint16_t)offset))
#define READ_FR_REGISTER32(offset) (*((volatile uint32_t *)MPC5748G_FR_CC_ADDRESS + ((uint16_t)offset) / 2U))
#define WRITE_FR_REGISTER16(offset, value) do { ( *( (volatile uint16_t*)MPC5748G_FR_CC_ADDRESS + offset)) = value; }while(0)
#define WRITE_FR_REGISTER32(offset, value) do { ( *( (volatile uint32_t*)MPC5748G_FR_CC_ADDRESS + (offset / 2U))) = value; } while(0)

#define SYNC_FRAME_TABLE_SIZE (2 * 120)
/* Message buffer header length in words */
#define FR_MESSAGE_BUFFER_HEADER_SIZE                        ((uint8_t)4U)
#define FR_MESSAGE_BUFFER_HEADER_STATUS_OFFSET ((uint8_t)3U)
#define MAX_SHADOW_BUF_COUNT 4
#define MAX_FIFO_DEPTH 255U
#define ROUND_UP_TO_MULTILE_OF_4(input) ((input + 3) & ~0x03)

#define CC_ENABLED ( (READ_FR_REGISTER16(FR_MCR_OFFSET) & FR_MCR_MEN_U16) == FR_MCR_MEN_U16)

typedef struct
{
	uint32_t segment1_msg_buf_count;
	uint32_t segment2_msg_buf_count;
	uint32_t segment1_shadow_buf_count;
	uint32_t segment2_shadow_buf_count;
    uint32_t segment1_msg_buf_data_size; /* In words */
    uint32_t segment2_msg_buf_data_size; /* In words */
    uint32_t segment1_channel_a_shadow_buf_header_idx;
    uint32_t segment1_channel_b_shadow_buf_header_idx;
    uint32_t segment2_channel_a_shadow_buf_header_idx;
    uint32_t segment2_channel_b_shadow_buf_header_idx;
    uint32_t segment1_last_msg_buf_num; /* The index of last message data buffer in segment 1 */
    uint32_t last_msg_buf_idx;/*  The index of last message data buffer */
    uint16_t fifo_buffer_data_size; /* In words */
    uint32_t fifoa_msg_buf_start_offset;
    uint32_t fifoa_first_msg_buf_header_index;
    uint32_t fifob_msg_buf_start_offset;
    uint32_t fifob_first_msg_buf_header_index;
	uint16_t sftor;
    uint8_t individual_msg_buf_numbers[MAX_MSG_BUFS];/* Map g_fr_config.msg_bufs index to message buffer number */
    uint16_t msg_buf_offsets[MAX_MSG_BUFS + MAX_SHADOW_BUF_COUNT + MAX_FIFO_DEPTH * 2];/* Map message buffer number to offset in memory area */
} fr_memory_layout;
fr_memory_layout g_mem_layout;

/* MPC5748G Reference Manual 46.8.4.1 FlexRay Memory Area Layout (FR_MCR[FAM] = 0)
In this mode, the following set of rules applies to the layout of the FlexRay memory area:
	The FlexRay memory area is one contiguous region.
	The FlexRay memory area size is maximum 64 KB.
	The FlexRay memory area starts at a 16 byte boundary
*/
#define FR_MEMORY_SIZE (64*1024)

__attribute__(( aligned(16U) )) uint8_t g_fr_memory[FR_MEMORY_SIZE];

static uint8_t init_memory_area_layout();
static uint8_t switch_to_poc_config_state();
static uint8_t leave_poc_config_state();
static void read_message_buffer (uint8_t * buf_ptr, uint8_t payload_len, uint8_t msg_buf_num);
static uint16_t calc_header_crc(uint16_t sync, uint16_t startup, uint16_t frame_id, uint16_t payload_len);
static uint8_t init_msg_bufs();
static uint8_t init_tx_msg_buf(const fr_msg_buffer * msg_buf_cfg, uint8_t msg_buf_num);
static uint8_t init_rx_msg_buf(const fr_msg_buffer * msg_buf_cfg, uint8_t msg_buf_num);
static uint8_t clear_all_interrupts();
static uint8_t send_chi_command(uint16_t chiCommand);
static uint8_t wait_for_chi_cmd_completed();
static uint16_t get_poc_state();
static uint8_t wait_for_poc_state(const uint16_t pocState);
static void init_fifo(const fr_fifo_queue_config * const p_fifo_queue_cfg, uint8_t fifo_queue_index, uint16_t msg_buf_start_offset, uint16_t first_msg_buf_header_index);

uint8_t flexray_driver_init() {
    uint16_t reg_val = 0U;
    uint32_t reg_val32 = 0U;
    if (READ_FR_REGISTER16(FR_MVR_OFFSET) != MPC5748G_MVR_VALUE) {
    	DBG_PRINT("Invalid MVR, wrong hardware?");
		return FAILED;
	}

    if(CC_ENABLED) {
        if(SUCCESS == send_chi_command(FR_POCR_CMD_FREEZE_U16) &&
			SUCCESS == wait_for_poc_state(FR_PSR0_PROTSTATE_HALT_U16) &&
			SUCCESS == send_chi_command(FR_POCR_CMD_DEFAULTCONFIG_U16) &&
			SUCCESS == wait_for_poc_state(FR_PSR0_PROTSTATE_DEFAULT_CONFIG_U16)) {
    		/* Disable FlexRay CC */
    		WRITE_FR_REGISTER16(FR_MCR_OFFSET, 0U);
        }
	}

    if(CC_ENABLED) {
    	DBG_PRINT("Disable CC failed");
    	return FAILED;
    }

	if (g_fr_config.flags & FR_CONFIG_FLAG_SINGLE_CHANNEL_MODE_ENABLED_MASK) {
		reg_val |= FR_MCR_SCM_U16;
		if(g_fr_config.pChannels == FR_CHANNEL_AB) {
	    	DBG_PRINT("In single channel device mode, node can't connect to both channel A&B.");
			return FAILED;
		}
	}

	/* MPC5748G Reference Manual Table 46-7. FlexRay channel selection */
	switch(g_fr_config.pChannels) {
		case FR_CHANNEL_A:
			reg_val |= FR_MCR_CHA_U16;
			break;
		case FR_CHANNEL_B:
			reg_val |= FR_MCR_CHB_U16;
			break;
		case FR_CHANNEL_AB:
			reg_val |= (uint16_t)(FR_MCR_CHA_U16 | FR_MCR_CHB_U16);
			break;
		default:
			break;
	}

	if(g_fr_config.flags & FR_CONFIG_FLAG_CLOCK_SOURCE_TYPE_MASK)
		reg_val |= FR_MCR_CLKSEL_U16; /* PLL */
	else
		reg_val &= ~FR_MCR_CLKSEL_U16; /* Crystal oscillator*/

	reg_val |= (uint16_t)((g_fr_config.bit_rate << 1) & FR_MCR_BITRATE_MASK_U16);
	WRITE_FR_REGISTER16(FR_MCR_OFFSET, reg_val);
	WRITE_FR_REGISTER16(FR_SYMATOR_OFFSET, 10U);
	WRITE_FR_REGISTER16(FR_SYMBADHR_OFFSET, (uint16_t)(((uint32_t)&g_fr_memory[0]) >> 16U));
	WRITE_FR_REGISTER16(FR_SYMBADLR_OFFSET, (uint16_t)(((uint32_t)&g_fr_memory[0]) & FR_SYMBADLR_SMBA_MASK_U16));
	reg_val |= FR_MCR_MEN_U16;
	/* FAM is 0*/
	WRITE_FR_REGISTER16(FR_MCR_OFFSET, reg_val);

	if(switch_to_poc_config_state() == FAILED) {
		DBG_PRINT("Switch to poc config state failed");
		return FAILED;
	}

	/* FlexRay cluster & node configuration, many PCR registers */
	reg_val32 = ((g_fr_config.gdActionPointOffset - 1) << 10) | (g_fr_config.gdStaticSlot & 0x000003FFU);
	reg_val32 = (reg_val32 << 16) | (gMacroPerCycle - g_fr_config.gdStaticSlot);
	WRITE_FR_REGISTER32(FR_PCR0_OFFSET, reg_val32);
	reg_val32 = ((g_fr_config.gdMinislot - g_fr_config.gdMiniSlotActionPointOffset - 1) << 10) | (g_fr_config.gNumberOfStaticSlots & 0x000003FFU);
	reg_val32 = (reg_val32 << 16) | ((((g_fr_config.gdWakeupSymbolRxLow << 5) | ((g_fr_config.gdMiniSlotActionPointOffset & 0x0000001FU) - 1)) << 5) | (g_fr_config.gColdStartAttempts & 0x0000001FU));
	WRITE_FR_REGISTER32(FR_PCR2_OFFSET, reg_val32);
	reg_val32 = ((g_fr_config.gdCasRxLowMax - 1) << 9) | (g_fr_config.gdWakeupRxWindow & 0x000001FFU);
	reg_val32 = (reg_val32 << 16) | ((((g_fr_config.gdTSSTransmitter << 6) | (g_fr_config.gdWakeupSymbolTxActive & 0x0000003FU)) << 6) | (g_fr_config.gdWakeupSymbolRxIdle & 0x0000003FU));
	WRITE_FR_REGISTER32(FR_PCR4_OFFSET, reg_val32);
	if(g_fr_config.gdSymbolWindow != 0)
		reg_val32 = (g_fr_config.gdSymbolWindow - g_fr_config.gdActionPointOffset - 1) << 7;
	else
		reg_val32 = 0;
	reg_val32 |= (g_fr_config.pMacroInitialOffsetA & 0x0000007FU);
	reg_val32 = (reg_val32 << 16) | ((g_fr_config.pDecodingCorrection + g_fr_config.pDelayCompensationB + 2) << 7) | (g_fr_config.pMicroPerCycle / gMacroPerCycle / 2);
	WRITE_FR_REGISTER32(FR_PCR6_OFFSET, reg_val32);
	reg_val32 = (((g_fr_config.gMaxWithoutClockCorrectionFatal << 4) | (g_fr_config.gMaxWithoutClockCorrectionPassive & 0x0000000FU)) << 8) | (g_fr_config.gdWakeupSymbolTxIdle & 0x0000000FFU);
	reg_val32 = (reg_val32 << 16) | ((((((g_fr_config.gNumberOfMinislots != 0) ? 0x00000001U : 0U) << 1) | ((g_fr_config.gdSymbolWindow != 0) ? 0x00000001U : 0U)) << 14) | (g_fr_config.pOffsetCorrectionOut & 0x00003FFFU));
	WRITE_FR_REGISTER32(FR_PCR8_OFFSET, reg_val32);
	reg_val32 = (((((g_fr_config.pKeySlotOnlyEnabled != 0)? 0x00000001U : 0U) << 1) | g_fr_config.pWakeupChannel) << 14) | (gMacroPerCycle & 0x00003FFFU);
	reg_val32 = (reg_val32 << 16) | ((((g_fr_config.pKeySlotUsedForStartup << 1) | g_fr_config.pKeySlotUsedForSync) << 14 ) | (g_fr_config.gOffsetCorrectionStart & 0x00003FFFU));
	WRITE_FR_REGISTER32(FR_PCR10_OFFSET, reg_val32);
	if(g_fr_config.pKeySlotId > 0)
		reg_val32 = calc_header_crc(g_fr_config.pKeySlotUsedForSync, g_fr_config.pKeySlotUsedForStartup, g_fr_config.pKeySlotId, g_fr_config.gPayloadLengthStatic);
	else
		reg_val32 = 0;
	reg_val32 |= (g_fr_config.pAllowPassiveToActive << 11);
	reg_val32 = (reg_val32 << 16) | ((max(g_fr_config.gdMiniSlotActionPointOffset - 1, g_fr_config.gdActionPointOffset - 1) << 10) | ((g_fr_config.gdStaticSlot - g_fr_config.gdActionPointOffset - 1) & 0x000003FFU));
	WRITE_FR_REGISTER32(FR_PCR12_OFFSET, reg_val32);
	reg_val32 = (g_fr_config.pRateCorrectionOut << 5) | (((g_fr_config.pdListenTimeout - 1) & 0x001F0000U) >> 16);
	reg_val32 = (reg_val32 << 16) | (g_fr_config.pdListenTimeout - 1);
	WRITE_FR_REGISTER32(FR_PCR14_OFFSET, reg_val32);
	reg_val32 = (g_fr_config.pMacroInitialOffsetB << 9) | (((g_fr_config.pdListenTimeout * g_fr_config.gListenNoise - 1) & 0x01FF0000U) >> 16);
	reg_val32 = (reg_val32 << 16) | (g_fr_config.pdListenTimeout * g_fr_config.gListenNoise - 1);
	WRITE_FR_REGISTER32(FR_PCR16_OFFSET, reg_val32);
	reg_val32 = (g_fr_config.pWakeupPattern << 10) | g_fr_config.pKeySlotId;
	reg_val32 = (reg_val32 << 16) | (((g_fr_config.pDecodingCorrection + g_fr_config.pDelayCompensationA + 2) << 7) | (g_fr_config.gPayloadLengthStatic & 0x0000007FU));
	WRITE_FR_REGISTER32(FR_PCR18_OFFSET, reg_val32);
	reg_val32 = (g_fr_config.pMicroInitialOffsetB << 8) | g_fr_config.pMicroInitialOffsetA;
	reg_val32 = (reg_val32 << 16) | ((g_fr_config.gNumberOfMinislots - g_fr_config.pLatestTx) & 0x00001FFFU);
	WRITE_FR_REGISTER32(FR_PCR20_OFFSET, reg_val32);
	reg_val32 = ((g_fr_config.pdAcceptedStartupRange - g_fr_config.pDelayCompensationA) << 4) | ((g_fr_config.pMicroPerCycle & 0x000F0000U) >> 16);
	reg_val32 = (reg_val32 << 16) | g_fr_config.pMicroPerCycle;
	WRITE_FR_REGISTER32(FR_PCR22_OFFSET, reg_val32);
	reg_val32 = (((g_fr_config.pClusterDriftDamping << 7) | (g_fr_config.pPayloadLengthDynMax & 0x0000007FU)) << 4) | (((g_fr_config.pMicroPerCycle - g_fr_config.pdMaxDrift) & 0x000F0000U) >> 16);
	reg_val32 = (reg_val32 << 16) | (g_fr_config.pMicroPerCycle - g_fr_config.pdMaxDrift);
	WRITE_FR_REGISTER32(FR_PCR24_OFFSET, reg_val32);
	reg_val32 = (((g_fr_config.pAllowHaltDueToClock << 11) | ((g_fr_config.pdAcceptedStartupRange - g_fr_config.pDelayCompensationB) & 0x000007FFU)) << 4) |
		(((g_fr_config.pMicroPerCycle + g_fr_config.pdMaxDrift) & 0x000F0000U) >> 16);
	reg_val32 = (reg_val32 << 16) | (g_fr_config.pMicroPerCycle + g_fr_config.pdMaxDrift);
	WRITE_FR_REGISTER32(FR_PCR26_OFFSET, reg_val32);
	reg_val32 = (g_fr_config.gdDynamicSlotIdlePhase << 14) |((gMacroPerCycle - g_fr_config.gOffsetCorrectionStart) & 0x00003FFFU);
	reg_val32 = (reg_val32 << 16) | ((g_fr_config.gNumberOfMinislots -1 ) & 0x00001FFFU);
	WRITE_FR_REGISTER32(FR_PCR28_OFFSET, reg_val32);
	WRITE_FR_REGISTER16(FR_PCR30_OFFSET, (uint16_t)g_fr_config.gSyncFrameIDCountMax);
	WRITE_FR_REGISTER16(FR_NMVLR_OFFSET, (uint16_t)g_fr_config.gNetworkManagementVectorLength);
	/* Request CC to write Sync Frame ID/Deviation table if logging enabled. */
	if(g_fr_config.flags & FR_CONFIG_FLAG_LOG_STATUS_DATA_MASK)
		WRITE_FR_REGISTER16(FR_SFTCCSR_OFFSET, FR_SFTCCSR_SIDEN_U16 | FR_SFTCCSR_SDVEN_U16);

	if(FAILED == init_memory_area_layout())
		return FAILED;
	WRITE_FR_REGISTER16(FR_SFTOR_OFFSET, g_mem_layout.sftor);

	if(FAILED == init_msg_bufs())
		return FAILED;

	if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOA_ENABLED_MASK) {
		DBG_PRINT("Initialize FIFOA");
		init_fifo(&g_fr_config.fifoa_config, 0, g_mem_layout.fifoa_msg_buf_start_offset, g_mem_layout.fifoa_first_msg_buf_header_index);
	}
	if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOB_ENABLED_MASK) {
		DBG_PRINT("Initialize FIFOB");
		init_fifo(&g_fr_config.fifob_config, 1, g_mem_layout.fifob_msg_buf_start_offset, g_mem_layout.fifob_first_msg_buf_header_index);
	}

	WRITE_FR_REGISTER16(FR_TICCR_OFFSET, (uint16_t)(FR_TICCR_T2SP_U16 | FR_TICCR_T1SP_U16));
	if(FAILED == leave_poc_config_state())
		return FAILED;

	if(FAILED == clear_all_interrupts())
		return FAILED;

	return SUCCESS;
}

/* FlexRay spec 2.1: 4.5 CRC calculation details */
static uint16_t calc_header_crc(uint16_t sync, uint16_t startup, uint16_t frame_id, uint16_t payload_len)
{
    uint16_t crc;
    uint16_t next_bit;
    uint16_t crc_next_bit;
    int8_t i;
    crc = cHCrcInit;
    next_bit = (sync & 0x0001U) ^ ((crc >> 10U) & 0x0001U);
    crc = crc << 1;
    if(next_bit != 0)
    	crc ^= cHCrcPolynomial;
    next_bit = (startup & 0x0001U) ^ ((crc >> 10U) & 0x0001U);
    crc = crc << 1;
    if(next_bit != 0)
    	crc ^= cHCrcPolynomial;
    /* Frame ID */
    for(i = 10; i >= 0; i--)
    {
        next_bit = (uint16_t)((frame_id >> i) & 0x0001U);
        crc_next_bit = (uint16_t)(next_bit ^ ((crc >> 10U) & 0x0001U));
        crc = (uint16_t)(crc << 1U);
        if(0U != crc_next_bit)
            crc ^= cHCrcPolynomial;
    }
    /* Payload Length */
    for(i = 6; i >= 0; i--)
    {
        next_bit = (uint16_t)((payload_len >> i) & 0x0001U);
        crc_next_bit = (uint16_t)(next_bit ^ ((crc >> 10U) & 0x0001U));
        crc = (uint16_t)(crc << 1U);
        if(0U != crc_next_bit)
            crc ^= cHCrcPolynomial;
    }
    return  (uint16_t)(crc & 0x7FFU);
}

uint8_t flexray_driver_start_communication() {
	return send_chi_command(FR_POCR_CMD_RUN_U16);
}

uint8_t flexray_driver_allow_coldstart() {
	return send_chi_command(FR_POCR_CMDALLOWCOLDSTART_U16);
}

uint8_t flexray_driver_abort_communication() {
	return send_chi_command(FR_POCR_CMD_FREEZE_U16);
}

uint8_t flexray_driver_set_wakeup_channel(uint32_t channel_index)
{
	uint16_t reg_val;
    if(send_chi_command(FR_POCR_CMD_CONFIG_U16) != SUCCESS)
    	return FAILED;
	/* Wait till FlexRay CC is not in POC:Config */
	if(wait_for_poc_state(FR_PSR0_PROTSTATE_CONFIG_U16) != SUCCESS)
		return FAILED;
	/* Load the PCR10 */
	reg_val = READ_FR_REGISTER16(FR_PCR10_OFFSET);
	if(FR_CHANNEL_A == channel_index)
		 reg_val &= ~FR_PCR10_WUP_CH_U16;
	else
		reg_val |= FR_PCR10_WUP_CH_U16;
	WRITE_FR_REGISTER16(FR_PCR10_OFFSET, reg_val);
	if(send_chi_command(FR_POCR_CMDCONFIGCOMPLETE_U16) != SUCCESS)
		return FAILED;
	if(wait_for_poc_state(FR_PSR0_PROTSTATE_READY_U16) != SUCCESS)
		return FAILED;
	return SUCCESS;
}

uint8_t flexray_driver_get_poc_status(fr_poc_status * poc_status_ptr) {
	uint32_t reg_val;
	if(!CC_ENABLED) {
		DBG_PRINT("CC disabled!");
		return FAILED;
	}
	/* Read PSR0 & PSR1. */
	reg_val = READ_FR_REGISTER32(FR_PSR0_OFFSET);
	if(FR_PSR1_FRZ_U16 == ((uint16_t)(reg_val) & FR_PSR1_FRZ_U16)) {
		poc_status_ptr->state = FR_PSR0_PROTSTATE_HALT_U16;
		poc_status_ptr->freeze = 1;
	} else {
		poc_status_ptr->state = ((uint16_t)(reg_val >> 16U)) & FR_PSR0_PROTSTATE_MASK_U16;
		poc_status_ptr->freeze = 0;
	}
	if (poc_status_ptr->state == FR_PSR0_PROTSTATE_STARTUP_U16)
		poc_status_ptr->startup_status = ((uint16_t)(reg_val >> 16U)) & FR_PSR0_STARTUP_MASK_U16;
	else
		poc_status_ptr->startup_status = 0U;
	poc_status_ptr->wakeup_status = ((uint16_t)(reg_val >> 16U)) & FR_PSR0_WUP_MASK_U16;

	if(FR_PSR1_HHR_U16 == ((uint16_t)(reg_val) & FR_PSR1_HHR_U16))
		poc_status_ptr->halt_request = 1U;
	else
		poc_status_ptr->halt_request = 0U;
	if(((uint16_t)(reg_val) & FR_PSR1_CPN_U16) != 0)
		poc_status_ptr->coldstart_noise = 1;
	else
		poc_status_ptr->coldstart_noise = 0;
	return SUCCESS;
}

uint8_t flexray_driver_get_global_time(uint8_t * cycle_ptr, uint16_t * macrotick_ptr) {
    uint32_t reg_val;
	if(!CC_ENABLED) {
		DBG_PRINT("CC disabled");
		return FAILED;
	}
	/* Read both MTCTR and CYCTR */
	reg_val = READ_FR_REGISTER32(FR_MTCTR_OFFSET);
	/* Lower part contains CYCTR - cycle count */
	*cycle_ptr = (uint8_t)(reg_val);
	/* Upper part contains MTCTR - macrotick */
	*macrotick_ptr = (uint16_t)(reg_val >> 16U);
	/* Check whether the read value does not exceed configured one */
	if((uint16_t)(gMacroPerCycle) <= *macrotick_ptr) {
		DBG_PRINT("macrotick error");
		*macrotick_ptr = (uint16_t)((uint16_t)(gMacroPerCycle) - 1U);
		/* Report error */
		return FAILED;
	}
    return SUCCESS;
}

uint8_t flexray_driver_set_abs_timer(uint8_t timer_index, uint8_t cycle, uint16_t offset) {
	uint16_t start_bit;
	uint16_t stop_bit;
	uint16_t cysr_reg_val;
	uint16_t mtor_reg_val;
	uint32_t msr;
	SUSPEND_INTERRUPT;
	if(0U == timer_index) {
		stop_bit = (uint16_t)(READ_FR_REGISTER16(FR_TICCR_OFFSET) | FR_TICCR_T1SP_U16);
		start_bit = (uint16_t)(READ_FR_REGISTER16(FR_TICCR_OFFSET) | FR_TICCR_T1TR_U16);
		cysr_reg_val = FR_TI1CYSR_OFFSET;
		mtor_reg_val = FR_TI1MTOR_OFFSET;
	} else {
		stop_bit = (uint16_t)(READ_FR_REGISTER16(FR_TICCR_OFFSET) | FR_TICCR_T2SP_U16);
		start_bit = (uint16_t)(READ_FR_REGISTER16(FR_TICCR_OFFSET) | FR_TICCR_T2TR_U16);
		cysr_reg_val = FR_TI2CR0_OFFSET;
		mtor_reg_val = FR_TI2CR1_OFFSET;
	}
	WRITE_FR_REGISTER16(FR_TICCR_OFFSET, stop_bit);
	/* store cycle value */
	WRITE_FR_REGISTER16(cysr_reg_val, (uint16_t)((uint16_t)(((uint16_t)cycle) << 8U) | FR_TI1CYSR_T1_CYC_MSK_U16));
	/* store the macro tick */
	WRITE_FR_REGISTER16(mtor_reg_val, offset);
	/* start the timer */
	WRITE_FR_REGISTER16(FR_TICCR_OFFSET, start_bit);
	RESUME_INTERRUPT;
    return SUCCESS;
}

uint8_t flexray_driver_cancel_abs_timer(uint8_t timer_index) {
    uint16_t reg_val;
    uint32_t msr;
    if(!CC_ENABLED)
    	return FAILED;

    SUSPEND_INTERRUPT;
    reg_val = (uint16_t)(READ_FR_REGISTER16(FR_TICCR_OFFSET) & FR_TICCR_CONFIG_MASK_U16);
    if(0U == timer_index)
        WRITE_FR_REGISTER16(FR_TICCR_OFFSET, (uint16_t)(reg_val | FR_TICCR_T1SP_U16));
    else
        WRITE_FR_REGISTER16(FR_TICCR_OFFSET, (uint16_t)(reg_val | FR_TICCR_T2SP_U16));
    RESUME_INTERRUPT;
    return SUCCESS;
}

uint8_t flexray_driver_ack_abs_timer(uint8_t timer_index) {
    if(!CC_ENABLED)
    	return FAILED;
	if(0U == timer_index)
		WRITE_FR_REGISTER16(FR_PIFR0_OFFSET, FR_PIFR0_TI1_IF_U16); /* Clear flag */
	else
		WRITE_FR_REGISTER16(FR_PIFR0_OFFSET, FR_PIFR0_TI2_IF_U16); /* Clear flag */
    return SUCCESS;
}

uint8_t flexray_driver_get_timer_irq_status(uint8_t timer_index, uint8_t *status_ptr) {
    uint16_t reg_val;
    if(!CC_ENABLED) {
    	return FAILED;
    }
	reg_val = READ_FR_REGISTER16(FR_PIFR0_OFFSET);
	if(0U == timer_index) {   /* Timer 1 */
		if(0U != (reg_val & FR_PIFR0_TI1_IF_U16))
		{   /* Interrupt flag has been set */
			*status_ptr = 1;
		}
	} else {   /* Timer 2 */
		if(0U != (reg_val & FR_PIFR0_TI2_IF_U16))
			*status_ptr = 0;
	}
    return SUCCESS;
}

uint8_t flexray_driver_write_tx_buffer(uint8_t tx_msg_buf_idx, const uint8_t * buf_ptr, uint8_t data_len) {
    uint16_t reg_offset;
    uint8_t msg_buf_num = 0;
    volatile uint16_t * msg_buf_header_ptr;
    volatile uint16_t * msg_buf_data_ptr16;
    volatile uint8_t * msg_buf_data_ptr8;
    volatile uint32_t * msg_buf_data_ptr32;
    uint16_t reg_val;
    uint8_t i;
    const fr_msg_buffer *msg_buf_cfg;
    uint8_t words;
    uint32_t msr;

	if(tx_msg_buf_idx > g_fr_config.individual_tx_msg_buf_count) {
		DBG_PRINT("Invalid tx msg buf idx: %u", tx_msg_buf_idx);
		return FAILED;
	}
    msg_buf_num = g_mem_layout.individual_msg_buf_numbers[g_fr_config.individual_rx_msg_buf_count + tx_msg_buf_idx];
	msg_buf_cfg = &g_fr_config.msg_bufs[g_fr_config.individual_rx_msg_buf_count + tx_msg_buf_idx];
	/* Avoid buffer overflow */
	if((msg_buf_cfg->frame_id) > g_fr_config.gNumberOfStaticSlots) {
		if(data_len > msg_buf_cfg->payload_length_max * 2)
			data_len = msg_buf_cfg->payload_length_max * 2;
	} else {
		if(data_len > (uint16_t)(g_fr_config.gPayloadLengthStatic) * 2)
			data_len = (uint16_t)(g_fr_config.gPayloadLengthStatic) * 2;
	}
	reg_offset = (((uint16_t)msg_buf_num) * 4U);
	/* This is not necessary, we don't support buffer reconfiguration currently. */
    msg_buf_num = (uint8_t)(READ_FR_REGISTER16(FR_MBIDXR0_OFFSET + reg_offset));
	reg_val = READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset);
	if(FR_MBCCSR_MTD_U16 != (reg_val & FR_MBCCSR_MTD_U16)) {
		DBG_PRINT("Tx msg buffer not configured yet, idx: %u", msg_buf_num);
		return FAILED;
	}
	/* Lock the msg buf */
	WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, FR_MBCCSR_LCKT_U16);
	if(FR_MBCCSR_LCKS_U16 != (READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR_LCKS_U16)) {
		DBG_PRINT("Lock buffer failed.");
		return FAILED;
	}
	msg_buf_header_ptr = (volatile uint16_t *)((uint32_t)&g_fr_memory[0]) + (msg_buf_num * FR_MESSAGE_BUFFER_HEADER_SIZE);
	msg_buf_data_ptr32 = (volatile uint32_t *)((uint32_t)&g_fr_memory[g_mem_layout.msg_buf_offsets[msg_buf_num]]);
	/* Copy 4-bytes block */
	for(i = 0U; i < (data_len >> 2U); i++)
		msg_buf_data_ptr32[i] = ((const volatile uint32_t *)buf_ptr)[i];
	i = (uint8_t)((i * 2U) & 0xFFU);
	/* Copy remained 2-bytes block */
	if(i < (data_len >> 1U)) {
		msg_buf_data_ptr16 = (volatile uint16_t *)((uint32_t)&g_fr_memory[g_mem_layout.msg_buf_offsets[msg_buf_num]]);
		msg_buf_data_ptr16[i] = ((const volatile uint16_t*)buf_ptr)[i];
		i++;
	}
	i = (uint8_t)((i * 2U) & 0xFFU);
	/* Copy remained 1-byte block */
	if(i < (data_len)) {
		msg_buf_data_ptr8 = (volatile uint8_t *)(&g_fr_memory[g_mem_layout.msg_buf_offsets[msg_buf_num]]);
		msg_buf_data_ptr8[i] = ((const volatile uint8_t*)buf_ptr)[i];
	}
	/* dynamic payload length enabled? */
	if(msg_buf_cfg->flags & MSG_BUF_DYNAMIC_PAYLOAD_LENGTH_ENABLED_FLAG)
	{
	   /* Bytes to words and round up*/
		words = (data_len >> 1U) + (data_len % 2U);
		if((msg_buf_header_ptr[1U] & FR_FRAMEHEADER1_PAYLOAD_LEN_MASK) != (uint16_t)words) {
			/* Payload data length parameter and header CRC have to be recalculated */
			msg_buf_header_ptr[1U] = words;
			msg_buf_header_ptr[2U] = calc_header_crc(0, 0, READ_FR_REGISTER16(FR_MBFIDR0_OFFSET + reg_offset), (uint16_t)words);
		}
	}
	/* Set MB to commit */
	/* Clear Transmit Buffer Interrupt Flag */
	/* Load FR_MBCCSRn register and select only necessary bits */
	SUSPEND_INTERRUPT;
	reg_val = (uint16_t)(READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR0_CONFIG_MASK_U16);
	reg_val |= (uint16_t)(FR_MBCCSR_CMT_U16 | FR_MBCCSR_MBIF_U16);
	WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, reg_val);
	RESUME_INTERRUPT;
	/* Unlock message buffer */
	WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, FR_MBCCSR_LCKT_U16);
	return SUCCESS;        /* API call was successful */
}

uint8_t flexray_driver_tx_buffer_idle(uint8_t tx_msg_buf_idx) {
    uint8_t msg_buf_num;
    uint16_t u16mbRegOffset_13;  /* Temporary offset address of MB registers */
    uint16_t u16tmpRegVal_13;

	if(tx_msg_buf_idx > g_fr_config.individual_tx_msg_buf_count) {
		DBG_PRINT("Invalid tx msg buf idx: %u", tx_msg_buf_idx);
		return 0;
	}
    msg_buf_num = g_mem_layout.individual_msg_buf_numbers[g_fr_config.individual_rx_msg_buf_count + tx_msg_buf_idx];
    u16mbRegOffset_13 = (((uint16_t)msg_buf_num) * 4U);
    u16tmpRegVal_13 = READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + u16mbRegOffset_13);
    if(FR_MBCCSR_MTD_U16 != (u16tmpRegVal_13 & FR_MBCCSR_MTD_U16)) {
		DBG_PRINT("Tx msg buf not configured yet: %u", tx_msg_buf_idx);
		return 0;
    }
    /* MPC5748G Reference Manual 46.7.114
     * CMT bit: Commit for Transmission. This bit indicates if the transmit message buffer data are ready for transmission */
	if(FR_MBCCSR_CMT_U16 != (u16tmpRegVal_13 & FR_MBCCSR_CMT_U16))
		return 1;
	else
		return 0;
}

uint8_t flexray_driver_deinit(void)
{
    uint8_t i;
    uint16_t max_msg_buf_count;
    uint8_t cycle_counter = 0U;
	if(!CC_ENABLED)
		return FAILED;
	if(SUCCESS != send_chi_command(FR_POCR_CMD_FREEZE_U16))
		return FAILED;
	if(SUCCESS != wait_for_poc_state(FR_PSR0_PROTSTATE_HALT_U16)) {
		DBG_PRINT("Wait poc state halt 1 failed");
		return FAILED;
	}
	if(SUCCESS != send_chi_command(FR_POCR_CMD_DEFAULTCONFIG_U16))
		return FAILED;
	if(SUCCESS != wait_for_poc_state(FR_PSR0_PROTSTATE_DEFAULT_CONFIG_U16)) {
		DBG_PRINT("Wait poc state default config 1 failed");
		return FAILED;
	}
	if(SUCCESS != switch_to_poc_config_state())
		return FAILED;
	for(i = 0U; i < FR_NUMBER_PCR_U8; i++) {
		WRITE_FR_REGISTER16((FR_PCR0_OFFSET + (i)), FR_RESET_VAL_U16);
	}
	/* Clear all other registers that can be cleared */
	WRITE_FR_REGISTER16(FR_STBSCR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_MBDSR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_PEDRDR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_PIER0_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_PIER1_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SFTOR_OFFSET , FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SFTOR_OFFSET , FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SFIDAFVR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SFIDAFMR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_NMVLR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_TICCR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SFTCCSR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_TI1CYSR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_TI1MTOR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_TI2CR0_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_TI2CR1_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SSSR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SSCCR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_MTSACFR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_MTSBCFR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RSBIR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFWMSR_OFFSET , FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFSIR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFDSR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFMIDAFVR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFMIDAFMR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFFIDRFVR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFFIDRFMR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFRFCFR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFRFCTR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFSDOR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFPTR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFFLPCR_OFFSET, FR_RESET_VAL_U16);

	while(((uint16_t)(READ_FR_REGISTER16(FR_EERICR_OFFSET)) & FR_EERICR_BSY_U16) == FR_EERICR_BSY_U16) {
		sleep(SLEEP_MILISECONDS);
		cycle_counter++;
		if(cycle_counter >= MAX_WAIT_POC_STATE_CHANGE_CYCLES) break;
	}
	if(cycle_counter >= MAX_WAIT_POC_STATE_CHANGE_CYCLES) {
		DBG_PRINT("Wait FR_EERICR busy bit timeout");
		return FAILED;
	}
	WRITE_FR_REGISTER16(FR_EERICR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_EEIAR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_EEIDR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_EEICR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_MBSSUTR_OFFSET, FR_MBSSUTR_RESET_VAL_U16);
	max_msg_buf_count = (uint16_t)((READ_FR_REGISTER16(FR_MBSSUTR_OFFSET)) & ((uint16_t)0x7FU));
	for (i = 0U; i <= max_msg_buf_count; i++)
	{
		if (((READ_FR_REGISTER16((FR_MBCCSR0_OFFSET + ((uint16_t)i * 4U)))) & FR_MBCCSR_EDS_U16) == FR_MBCCSR_EDS_U16)
			WRITE_FR_REGISTER16((FR_MBCCSR0_OFFSET + ((uint16_t)i * 4U)) , FR_MBCCSR_EDT_U16);
		if ((READ_FR_REGISTER16((FR_MBCCSR0_OFFSET + ((uint16_t)i * 4U))) & FR_MBCCSR_EDS_U16) == FR_MBCCSR_EDS_U16) {
			DBG_PRINT("Disable message buffer %u failed.", i);
			return FAILED;
		}
		WRITE_FR_REGISTER16((FR_MBCCSR0_OFFSET + ((uint16_t)i * 4U)), FR_RESET_VAL_1_U16);
		WRITE_FR_REGISTER16((FR_MBCCFR0_OFFSET + ((uint16_t)i * 4U)), FR_RESET_VAL_U16);
		WRITE_FR_REGISTER16((FR_MBFIDR0_OFFSET + ((uint16_t)i * 4U)), FR_RESET_VAL_U16);
		WRITE_FR_REGISTER16((FR_MBIDXR0_OFFSET + ((uint16_t)i * 4U)), FR_RESET_VAL_U16);
		WRITE_FR_REGISTER16((FR_MBDOR0_OFFSET + (uint16_t)i), FR_RESET_VAL_U16);
	}
	WRITE_FR_REGISTER16((FR_MBDOR0_OFFSET + (i)), FR_RESET_VAL_U16);
	i++;
	WRITE_FR_REGISTER16((FR_MBDOR0_OFFSET + (i)), FR_RESET_VAL_U16);
	i++;
	WRITE_FR_REGISTER16((FR_MBDOR0_OFFSET + (i)), FR_RESET_VAL_U16);
	i++;
	WRITE_FR_REGISTER16((FR_MBDOR0_OFFSET + (i)), FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_GIFER_OFFSET, FR_GIFER_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_PIFR0_OFFSET, FR_PIFR0_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_PIFR1_OFFSET, FR_PIFR1_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_PSR1_OFFSET, FR_PSR1_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_PSR3_OFFSET, FR_PSR3_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_EEIFER_OFFSET, FR_EEIFER_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_CHIERFR_OFFSET, FR_CHIERFR_RESET_VAL_U16);
	if(SUCCESS != send_chi_command(FR_POCR_CMD_FREEZE_U16)) {
		DBG_PRINT("Go to freeze failed.");
		return FAILED;
	}
	if(SUCCESS != wait_for_poc_state(FR_PSR0_PROTSTATE_HALT_U16)) {
		DBG_PRINT("Wait poc state halt 2 failed");
		return FAILED;
	}
	if(SUCCESS != send_chi_command(FR_POCR_CMD_DEFAULTCONFIG_U16)) {
		DBG_PRINT("Go to default config failed.");
		return FAILED;
	}
	if(SUCCESS != wait_for_poc_state(FR_PSR0_PROTSTATE_DEFAULT_CONFIG_U16))
		return FAILED;
	WRITE_FR_REGISTER16(FR_MCR_OFFSET, FR_RESET_VAL_U16);
	if((((READ_FR_REGISTER16(FR_MCR_OFFSET)) & FR_MCR_MEN_U16) != 0U)) {
		DBG_PRINT("Disable CC failed.");
		return FAILED;
	}
	WRITE_FR_REGISTER16(FR_MCR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SYMATOR_OFFSET, FR_SYMATOR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SYMBADHR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_SYMBADLR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFSYMBADHR_OFFSET, FR_RESET_VAL_U16);
	WRITE_FR_REGISTER16(FR_RFSYMBADLR_OFFSET, FR_RESET_VAL_U16);
	return SUCCESS;
}


uint8_t flexray_driver_receive_fifoa(uint8_t *buf_ptr, fr_rx_status * rx_status_ptr, uint8_t* payload_len_ptr, uint16_t *frame_id_ptr)
{
    uint16_t reg_val;
    uint8_t msg_buf_num;
    volatile uint16_t * msg_buf_header_ptr;
    uint8_t payload_len;  /* in Words */
 	uint32_t msr;
   /* Check whether the FIFO A is empty or not */
    if(FR_GIFER_FAFAIF_U16 == (READ_FR_REGISTER16(FR_GIFER_OFFSET) & FR_GIFER_FAFAIF_U16))
    {   /* FIFO A is not empty */
        msg_buf_num = (uint8_t)(READ_FR_REGISTER16(FR_RFARIR_OFFSET));
        msg_buf_header_ptr = (volatile uint16_t *)((uint32_t)&g_fr_memory[0]) + (msg_buf_num * FR_MESSAGE_BUFFER_HEADER_SIZE);
        *frame_id_ptr = (uint16_t)(msg_buf_header_ptr[0U] & FR_FRAMEHEADER0_FRAME_ID_MASK);
        payload_len = (uint8_t)(msg_buf_header_ptr[1U] & FR_FRAMEHEADER1_PAYLOAD_LEN_MASK);
        /* Payload data length is also limited by Rx MB configuration */
        if(payload_len != g_mem_layout.fifo_buffer_data_size)
        {   /* received payload length does not match configured payload length */
            if(payload_len > g_mem_layout.fifo_buffer_data_size)
                payload_len = (uint8_t)g_mem_layout.fifo_buffer_data_size;
            if(0x0080U == (msg_buf_header_ptr[FR_MESSAGE_BUFFER_HEADER_STATUS_OFFSET] & 0x0080U)) {
            	read_message_buffer(buf_ptr, payload_len, msg_buf_num);
                *payload_len_ptr = (uint8_t)(payload_len << 1U);
                *rx_status_ptr = FR_RX_STATUS_RECEIVED_MORE_DATA_AVAILABLE;
            } else {
                *payload_len_ptr = 0U;
                *rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
            }
        } /* received payload length does not match configured payload length */
        else /* received payload length exactly matches configured payload length */
        {
            if(0x0080U == (msg_buf_header_ptr[FR_MESSAGE_BUFFER_HEADER_STATUS_OFFSET] & 0x0080U)) { /* Stringent check passed */
            	read_message_buffer(buf_ptr, payload_len, msg_buf_num);
                *payload_len_ptr = (uint8_t)(payload_len << 1U);
                *rx_status_ptr = FR_RX_STATUS_RECEIVED_MORE_DATA_AVAILABLE;
            } else {
                *payload_len_ptr = 0U;
                *rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
            }
        }
        /* Clear FIFO A not empty flag */
        SUSPEND_INTERRUPT;
        reg_val = (uint16_t)(READ_FR_REGISTER16(FR_GIFER_OFFSET) & (uint16_t)(~FR_GIFER_INT_FLAGS_MASK_U16));
        reg_val |= FR_GIFER_FAFAIF_U16;
        WRITE_FR_REGISTER16(FR_GIFER_OFFSET, reg_val);  /* Clear the flag */
        RESUME_INTERRUPT;
    } else {   /* FIFO A is empty */
        *payload_len_ptr = 0U;
        *rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
    }
	return SUCCESS;
}

uint8_t flexray_driver_receive_fifob(uint8_t *buf_ptr, fr_rx_status * rx_status_ptr, uint8_t* payload_len_ptr, uint16_t *frame_id_ptr)
{
    uint16_t reg_val;
    uint8_t msg_buf_num;
    volatile uint16_t * msg_buf_header_ptr;
    uint8_t payload_len;
	uint32_t msr;
    if(FR_GIFER_FAFBIF_U16 == (READ_FR_REGISTER16(FR_GIFER_OFFSET) & FR_GIFER_FAFBIF_U16)) {
        msg_buf_num = (uint8_t)(READ_FR_REGISTER16(FR_RFBRIR_OFFSET));
        msg_buf_header_ptr = (volatile uint16_t *)((uint32_t)&g_fr_memory[0]) + (msg_buf_num * FR_MESSAGE_BUFFER_HEADER_SIZE);
        msg_buf_header_ptr = (volatile uint16_t *)((uint32_t)&g_fr_memory[0]) + (msg_buf_num * FR_MESSAGE_BUFFER_HEADER_SIZE);
        payload_len = (uint8_t)(msg_buf_header_ptr[1U] & FR_FRAMEHEADER1_PAYLOAD_LEN_MASK);
        if(payload_len != g_mem_layout.fifo_buffer_data_size)
        {
            if(payload_len > g_mem_layout.fifo_buffer_data_size)
                payload_len = (uint8_t)g_mem_layout.fifo_buffer_data_size;
            if(0x8000U == (msg_buf_header_ptr[FR_MESSAGE_BUFFER_HEADER_STATUS_OFFSET] & 0x8000U)) { /* stringent check passed */
            	read_message_buffer(buf_ptr, payload_len, msg_buf_num);
                *payload_len_ptr = (uint8_t)(payload_len << 1U);
                *rx_status_ptr = FR_RX_STATUS_RECEIVED_MORE_DATA_AVAILABLE;
            } else {
                *payload_len_ptr = 0U;
                *rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
            }
        } else {
            if(0x8000U == (msg_buf_header_ptr[FR_MESSAGE_BUFFER_HEADER_STATUS_OFFSET] & 0x8000U)) { /* stringent check passed */
            	read_message_buffer(buf_ptr, payload_len, msg_buf_num);
                *payload_len_ptr = (uint8_t)(payload_len << 1U);
                *rx_status_ptr = FR_RX_STATUS_RECEIVED_MORE_DATA_AVAILABLE;
            } else {
                *payload_len_ptr = 0U;
                *rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
            }
        }
        SUSPEND_INTERRUPT;
        reg_val = (uint16_t)(READ_FR_REGISTER16(FR_GIFER_OFFSET) & (uint16_t)(~FR_GIFER_INT_FLAGS_MASK_U16));
        reg_val |= FR_GIFER_FAFBIF_U16;
        WRITE_FR_REGISTER16(FR_GIFER_OFFSET, reg_val);  /* Clear the flag */
        RESUME_INTERRUPT;
    } else {
        *payload_len_ptr = 0U;
        *rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
    }
	return SUCCESS;
}

uint8_t flexray_driver_read_rx_buffer(uint8_t rx_msg_buf_idx, uint8_t * buf_ptr, fr_rx_status *rx_status_ptr, uint8_t * payload_len_ptr) {
    uint16_t reg_val;
    uint16_t status;
    uint8_t msg_buf_num;
    uint16_t reg_offset;
    volatile uint16_t * msg_header_ptr;
    uint8_t payload_len;  /* In Words */
	uint32_t msr;

    msg_buf_num = g_mem_layout.individual_msg_buf_numbers[rx_msg_buf_idx];
    reg_offset = (((uint16_t)msg_buf_num) * 4U);
    /* Due to shadow buffer mechanism, we must read the msg buf num from FR_MBIDX. */
    msg_buf_num = (uint8_t)(READ_FR_REGISTER16(FR_MBIDXR0_OFFSET + reg_offset));
    if(msg_buf_num > g_mem_layout.segment2_channel_b_shadow_buf_header_idx) {
    	DBG_PRINT("Invalid msg buf num: %u, rx msg buf idx %u", msg_buf_num, rx_msg_buf_idx);
    	return FAILED;
    }
	reg_val = READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset);
	/* MPC5748G Reference Manual 46.7.114, MTD bit is 0 for Rx msg buf */
	if(FR_MBCCSR_MTD_U16 == (reg_val & FR_MBCCSR_MTD_U16)) {
		DBG_PRINT("msg buf not configured for Rx: %u, rx msg buf idx %u", msg_buf_num, rx_msg_buf_idx);
		return FAILED;
	}
	if(((uint8_t)(FR_MBCCSR_DUP_U16 | FR_MBCCSR_MBIF_U16)) == (reg_val & ((uint16_t) (FR_MBCCSR_DUP_U16 | FR_MBCCSR_MBIF_U16)))) {
		WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, FR_MBCCSR_LCKT_U16);
		reg_val = READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset);
		if(FR_MBCCSR_LCKS_U16 == (reg_val & FR_MBCCSR_LCKS_U16)) {   /* MB is locked */
			msg_header_ptr = (volatile uint16_t *)((uint32_t)&g_fr_memory[0]) + (msg_buf_num * FR_MESSAGE_BUFFER_HEADER_SIZE);
			payload_len = (uint8_t)(msg_header_ptr[1U] & FR_FRAMEHEADER1_PAYLOAD_LEN_MASK);
			reg_val = READ_FR_REGISTER16(FR_MBCCFR0_OFFSET + reg_offset);
			if((FR_MBCCFR_CHA_U16 | FR_MBCCFR_CHB_U16) == (reg_val & (FR_MBCCFR_CHA_U16 | FR_MBCCFR_CHB_U16))) {
				status = msg_header_ptr[FR_MESSAGE_BUFFER_HEADER_STATUS_OFFSET];
				if((0x8080U == (status & 0x8080U)) ||
				   (0x0080U == (status & 0x0080U)) ||
				   (0x8000U == (status & 0x8000U)))
				{
					/* Stringent check pass */
					read_message_buffer (buf_ptr, payload_len, msg_buf_num);
					*rx_status_ptr = FR_RX_STATUS_RECEIVED;
					*payload_len_ptr = (uint8_t)(payload_len << 1U);
				}
				else /* Stringent check did not pass */
				{
					*rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
					*payload_len_ptr = 0U;
				}
			} else if(FR_MBCCFR_CHA_U16 == (reg_val & FR_MBCCFR_CHA_U16)) {
				/* Check Slot Status information */
				if(0x0080U == (msg_header_ptr[FR_MESSAGE_BUFFER_HEADER_STATUS_OFFSET] & 0x0080U))
				{
					/* Stringent check pass */
					read_message_buffer (buf_ptr, payload_len, msg_buf_num);
					/* Store the reception status */
					*rx_status_ptr = FR_RX_STATUS_RECEIVED;
					/* Store the number of copied bytes */
					*payload_len_ptr = (uint8_t)(payload_len << 1U);
				}
				else /* Stringent check did not pass */
				{
					/* Store the reception status */
					*rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
					/* Store the number of copied bytes */
					*payload_len_ptr = 0U;
				}
			} else {
				if(0x8000U == (msg_header_ptr[FR_MESSAGE_BUFFER_HEADER_STATUS_OFFSET] & 0x8000U)) {
					/* Stringent check passed */
					read_message_buffer(buf_ptr, payload_len, msg_buf_num);
					/* Store the reception status */
					*rx_status_ptr = FR_RX_STATUS_RECEIVED;
					/* Store the number of copied bytes */
					*payload_len_ptr = (uint8_t)(payload_len << 1U);
				} else {
					*rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
					*payload_len_ptr = 0U;
				}
			}
			SUSPEND_INTERRUPT;
			reg_val = (uint16_t)(READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR0_CONFIG_MASK_U16);
			reg_val |= FR_MBCCSR_MBIF_U16;    /*Clear flag*/
			WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, reg_val);
			RESUME_INTERRUPT;
			WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, FR_MBCCSR_LCKT_U16);
		}
	} else {
		*payload_len_ptr = 0U;
		*rx_status_ptr = FR_RX_STATUS_NOT_RECEIVED;
		SUSPEND_INTERRUPT;
		reg_val = (uint16_t)(READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR0_CONFIG_MASK_U16);
		reg_val |= FR_MBCCSR_MBIF_U16;    /* Clear flag */
		WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, reg_val);
		RESUME_INTERRUPT;
	}
	return SUCCESS;
}

uint8_t flexray_driver_get_clock_correction(int16_t * rate_correction, int16_t * offset_correction) {
    *rate_correction = (int16_t)READ_FR_REGISTER16(FR_RTCORVR_OFFSET);
    *offset_correction = (int16_t)READ_FR_REGISTER16(FR_OFCORVR_OFFSET);
	return SUCCESS;
}

uint8_t flexray_driver_get_sync_frame_table (
		uint16_t *frame_table_ptr,
		uint8_t *a_even_cnt,
		uint8_t *b_even_cnt,
		uint8_t *a_odd_cnt,
		uint8_t *b_odd_cnt,
		uint16_t *sfcntr
) {
    uint8_t i, table_i = 0;
    *sfcntr = 0U;
    if((FR_SFTCCSR_EVAL_U16 | FR_SFTCCSR_OVAL_U16) == ((READ_FR_REGISTER16(FR_SFTCCSR_OFFSET)) & (FR_SFTCCSR_EVAL_U16 | FR_SFTCCSR_OVAL_U16))) {
    	WRITE_FR_REGISTER16(FR_SFTCCSR_OFFSET, (FR_SFTCCSR_OLKT_U16 | FR_SFTCCSR_ELKT_U16));
        if((FR_SFTCCSR_ELKS_U16 | FR_SFTCCSR_OLKS_U16) == ((READ_FR_REGISTER16(FR_SFTCCSR_OFFSET)) & (FR_SFTCCSR_ELKS_U16 | FR_SFTCCSR_OLKS_U16 | FR_SFTCCSR_SIDEN_U16))) {
        	*sfcntr = READ_FR_REGISTER16(FR_SFCNTR_OFFSET);
            *a_even_cnt = (uint8_t)(((*sfcntr) & 0x0F00U) >> 8U);
            *b_even_cnt = (uint8_t)(((*sfcntr) & 0xF000U) >> 12U);
            *a_odd_cnt = (uint8_t)((*sfcntr) & 0x000FU);
            *b_odd_cnt = (uint8_t)(((*sfcntr) & 0x00F0U) >> 4U);
            if((*a_even_cnt) + (*b_even_cnt) + (*a_odd_cnt) + (*b_odd_cnt) > 60) {
            	DBG_PRINT("Invalid frame count.");
				return FAILED;
            }
            for(i = 0U; i < *a_even_cnt; i++) {
            	frame_table_ptr[table_i++] = *(((volatile uint16_t *)(&g_fr_memory[g_mem_layout.sftor])) + i);
            	frame_table_ptr[table_i++] = *(((volatile uint16_t *)(&g_fr_memory[g_mem_layout.sftor + 120U])) + i);
            }
            for(i = 0U; i < *b_even_cnt; i++) {
            	frame_table_ptr[table_i++] = *(((volatile uint16_t *)(&g_fr_memory[g_mem_layout.sftor + 30U])) + i);
            	frame_table_ptr[table_i++] = *(((volatile uint16_t *)(&g_fr_memory[g_mem_layout.sftor + 150U])) + i);
            }
            for(i = 0U; i < *a_odd_cnt; i++) {
				frame_table_ptr[table_i++] = *(((volatile uint16_t *)(&g_fr_memory[g_mem_layout.sftor + 60U])) + i);
				frame_table_ptr[table_i++] = *(((volatile uint16_t *)(&g_fr_memory[g_mem_layout.sftor + 180U])) + i);
			}
            for(i = 0U; i < *b_odd_cnt; i++) {
				frame_table_ptr[table_i++] = *(((volatile uint16_t *)(&g_fr_memory[g_mem_layout.sftor + 90U])) + i);
				frame_table_ptr[table_i++] = *(((volatile uint16_t *)(&g_fr_memory[g_mem_layout.sftor + 210U])) + i);
            }
            WRITE_FR_REGISTER16(FR_SFTCCSR_OFFSET, (FR_SFTCCSR_ELKT_U16 | FR_SFTCCSR_OLKT_U16 | FR_SFTCCSR_SIDEN_U16));
        }
    }
    return SUCCESS;
}

uint8_t flexray_driver_get_status_registers(uint16_t *psr0, uint16_t *psr1, uint16_t *psr2,  uint16_t *psr3, uint16_t *pifr0) {
    if(!CC_ENABLED) {
    	return FAILED;
    }
	/* Protocol Status */
	*psr0 = READ_FR_REGISTER16(FR_PSR0_OFFSET);
	/* Protocol Status */
	*psr1 = READ_FR_REGISTER16(FR_PSR1_OFFSET);
	/*  a snapshot of status information about the Network Idle Time NIT, the Symbol Window and the clock synchronization */
	*psr2 = READ_FR_REGISTER16(FR_PSR2_OFFSET);
	/* aggregated channel status information */
	*psr2 = READ_FR_REGISTER16(FR_PSR3_OFFSET);
	*pifr0 = READ_FR_REGISTER16(FR_PIFR0_OFFSET);
	return SUCCESS;
}

static uint8_t init_memory_area_layout() {
	uint32_t static_slot_msg_buf_count = 0U, dynamic_slot_msg_buf_count = 0U;
	uint32_t fifoa_depth = 0U, fifob_depth = 0U, fifo_msg_buf_begin_offset = 0U, fifo_msg_buf_begin_header_index = 0U;
	uint32_t msg_buf_headers_size = 0U, fifo_data_size_in_bytes = 0U, segment1_size = 0U, segment2_size = 0U;
	uint8_t i = 0U, static_buf_idx = 0U, dynamic_buf_idx = 0U;
	if(g_fr_config.individual_rx_msg_buf_count == 0 && !(g_fr_config.flags & FR_CONFIG_FLAG_FIFOA_ENABLED_MASK) && !(g_fr_config.flags & FR_CONFIG_FLAG_FIFOB_ENABLED_MASK)) {
		DBG_PRINT("Invalid msg buf config");
		return FAILED;
	}
	/* MPC5748G Reference Manual 46.8.4.3 Message Buffer Header Area (FR_MCR[FAM] = 0)
	 * FlexRay memory area layout:
	 * 	Message buffer headers, in this order: segment 1, segment 2, shadow for segment 1, shadow for segment 2, fifoa, fifob
	 * 	Individual message buffers in segment 1
	 * 	Individual message buffers in segment 2 (Optional)
	 * 	Shadow buffers for message buffers in segment 1
	 * 	Shadow buffers for message buffers in segment 2 (Optional)
	 * 	FIFO buffers for channel A (Optional)
	 * 	FIFO buffers for channel B (Optional)
	 * 	Sync frame table : 2 * 120 Bytes
	 * */
	/* Decide segment count, buffer count in every segment, shadow buffer count, and message buffer data size. */
	for(;i < (uint8_t)g_fr_config.individual_rx_msg_buf_count + (uint8_t)g_fr_config.individual_tx_msg_buf_count;i++) {
		if(g_fr_config.msg_bufs[i].frame_id > g_fr_config.gNumberOfStaticSlots)
			dynamic_slot_msg_buf_count++;
		else
			static_slot_msg_buf_count++;
	}
	/* If for OpenPilot we always use both of the two FlexRay channels, we can remove single channel mode support and simplify the logic. */
	if(static_slot_msg_buf_count > 0U) {
		/* segment 1 is for static slot*/
		g_mem_layout.segment1_msg_buf_count = static_slot_msg_buf_count;
		g_mem_layout.segment1_msg_buf_data_size = g_fr_config.gPayloadLengthStatic;
		if(g_fr_config.flags & FR_CONFIG_FLAG_SINGLE_CHANNEL_MODE_ENABLED_MASK)
			g_mem_layout.segment1_shadow_buf_count = 1;
		else
			g_mem_layout.segment1_shadow_buf_count = 2;
		if(dynamic_slot_msg_buf_count > 0) {
			/* segment 2 is for dynamic slot*/
			g_mem_layout.segment2_msg_buf_count = dynamic_slot_msg_buf_count;
			g_mem_layout.segment2_msg_buf_data_size = g_fr_config.pPayloadLengthDynMax;
			if(g_fr_config.flags & FR_CONFIG_FLAG_SINGLE_CHANNEL_MODE_ENABLED_MASK) {
				g_mem_layout.segment2_shadow_buf_count = 1;
				g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
				g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx + 1;
				g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment2_channel_a_shadow_buf_header_idx; /* Unused */
			} else {
				g_mem_layout.segment2_shadow_buf_count = 2;
				if(g_fr_config.pChannels == FR_CHANNEL_AB) {
					g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
					g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx + 1;
					g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx + 2;
					g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx + 3;
				} else if(g_fr_config.pChannels == FR_CHANNEL_A) {
					g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
					g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count; /* Unused */
					g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx + 1;
					g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx + 1; /* Unused */
				} else if(g_fr_config.pChannels == FR_CHANNEL_B) {
					g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count; /* Unused */
					g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
					g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx + 1; /* Unused */
					g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx + 1;
				}
			}
		} else {
			/* one segment only */
			g_mem_layout.segment2_msg_buf_count = 0;
			g_mem_layout.segment2_msg_buf_data_size = 0U;
			g_mem_layout.segment2_shadow_buf_count = 0U;
			if(g_fr_config.flags & FR_CONFIG_FLAG_SINGLE_CHANNEL_MODE_ENABLED_MASK) {
				g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
				g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
			} else {
				if(g_fr_config.pChannels == FR_CHANNEL_AB) {
					g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
					g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				} else if(g_fr_config.pChannels == FR_CHANNEL_A) {
					g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
					g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count; /* Unused */
					g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
					g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				} else if(g_fr_config.pChannels == FR_CHANNEL_B) {
					g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count; /* Unused */
					g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
					g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
					g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				}
			}
		}
	} else {
		/* segment 1 is for dynamic slot, one segment only */
		g_mem_layout.segment2_msg_buf_count = 0U;
		g_mem_layout.segment2_msg_buf_data_size = 0U;
		g_mem_layout.segment2_shadow_buf_count = 0U;
		if(dynamic_slot_msg_buf_count > 0) {
			g_mem_layout.segment1_msg_buf_count = dynamic_slot_msg_buf_count;
			g_mem_layout.segment1_msg_buf_data_size = g_fr_config.pPayloadLengthDynMax;
			if(g_fr_config.flags & FR_CONFIG_FLAG_SINGLE_CHANNEL_MODE_ENABLED_MASK) {
				g_mem_layout.segment1_shadow_buf_count = 1;
				g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
				g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
			} else {
				g_mem_layout.segment1_shadow_buf_count = 2;
				if(g_fr_config.pChannels == FR_CHANNEL_AB) {
					g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
					g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx + 1;
					g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
					g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				} else if(g_fr_config.pChannels == FR_CHANNEL_A) {
					g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
					g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count; /* Unused */
					g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
					g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				} else if(g_fr_config.pChannels == FR_CHANNEL_B) {
					g_mem_layout.segment1_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count; /* Unused */
					g_mem_layout.segment1_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count;
					g_mem_layout.segment2_channel_a_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
					g_mem_layout.segment2_channel_b_shadow_buf_header_idx = g_mem_layout.segment1_channel_a_shadow_buf_header_idx; /* Unused */
				}
			}
		} else {
			/* No segment */
			g_mem_layout.segment1_msg_buf_count = 0;
			g_mem_layout.segment1_msg_buf_data_size = 0U;
			g_mem_layout.segment1_shadow_buf_count = 0U;
		}
	}
	if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOA_ENABLED_MASK)
		fifoa_depth = g_fr_config.fifoa_config.depth;
	if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOB_ENABLED_MASK)
		fifob_depth = g_fr_config.fifob_config.depth;
	msg_buf_headers_size = (
		g_mem_layout.segment1_msg_buf_count + \
		g_mem_layout.segment2_msg_buf_count + \
		g_mem_layout.segment1_shadow_buf_count + \
		g_mem_layout.segment2_shadow_buf_count + \
		fifoa_depth + \
		fifob_depth ) * FR_MESSAGE_BUFFER_HEADER_SIZE * 2;
	msg_buf_headers_size = ROUND_UP_TO_MULTILE_OF_4(msg_buf_headers_size);
	segment1_size = ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment1_msg_buf_data_size * 2) * g_mem_layout.segment1_msg_buf_count;
	segment2_size = ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment2_msg_buf_data_size * 2) * g_mem_layout.segment2_msg_buf_count;
	/* Assign msg buf number for every individual msg buf, also calculate offsets in memory area */
	for(i = 0;i < (uint8_t)g_fr_config.individual_rx_msg_buf_count + (uint8_t)g_fr_config.individual_tx_msg_buf_count;i++) {
		if(g_fr_config.msg_bufs[i].frame_id > g_fr_config.gNumberOfStaticSlots) {
			g_mem_layout.individual_msg_buf_numbers[i] = static_slot_msg_buf_count + dynamic_buf_idx;
			if(static_slot_msg_buf_count > 0) /* Dynamic msg buf is in segment 2 */
				g_mem_layout.msg_buf_offsets[static_slot_msg_buf_count + dynamic_buf_idx] = \
						msg_buf_headers_size + \
						segment1_size + \
						ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment2_msg_buf_data_size * 2) * dynamic_buf_idx;
			else /* Dynamic msg buf is in segment 1 */
				g_mem_layout.msg_buf_offsets[dynamic_buf_idx] = \
						msg_buf_headers_size + \
						ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment1_msg_buf_data_size * 2) * dynamic_buf_idx;
			dynamic_buf_idx++;
		} else {
			g_mem_layout.individual_msg_buf_numbers[i] = static_buf_idx;
			g_mem_layout.msg_buf_offsets[static_buf_idx] = \
					msg_buf_headers_size + \
					ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment1_msg_buf_data_size * 2) * static_buf_idx;
			static_buf_idx++;
		}
	}
	/* Calculate shadow buffers' offsets in memory area. */
	if(g_mem_layout.segment1_shadow_buf_count > 0) {
		for(i = 0;i < g_mem_layout.segment1_shadow_buf_count; i++)
			g_mem_layout.msg_buf_offsets[static_buf_idx + dynamic_buf_idx + i] = \
				msg_buf_headers_size + \
				segment1_size + \
				segment2_size + \
				ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment1_msg_buf_data_size * 2) * i;
	}
	if(g_mem_layout.segment2_shadow_buf_count > 0) {
		for(i = 0;i < g_mem_layout.segment2_shadow_buf_count; i++)
			g_mem_layout.msg_buf_offsets[static_buf_idx + dynamic_buf_idx + g_mem_layout.segment1_shadow_buf_count + i] = \
				msg_buf_headers_size + \
				segment1_size + \
				segment1_size + \
				ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment1_msg_buf_data_size * 2) * g_mem_layout.segment1_shadow_buf_count + \
				ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment2_msg_buf_data_size * 2) * i;
	}
	/* Ensure that the FIFO entry size is big enough for static slot frame and dynamic slot frames*/
	if((g_fr_config.flags & FR_CONFIG_FLAG_FIFOA_ENABLED_MASK) || (g_fr_config.flags & FR_CONFIG_FLAG_FIFOB_ENABLED_MASK)) {
		g_mem_layout.fifo_buffer_data_size = max(g_fr_config.gPayloadLengthStatic, g_fr_config.pPayloadLengthDynMax);
		fifo_data_size_in_bytes = ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.fifo_buffer_data_size * 2);
	}
	fifo_msg_buf_begin_offset = msg_buf_headers_size + \
			segment1_size + segment2_size + \
			ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment1_msg_buf_data_size * 2) * g_mem_layout.segment1_shadow_buf_count + \
			ROUND_UP_TO_MULTILE_OF_4(g_mem_layout.segment2_msg_buf_data_size * 2) * g_mem_layout.segment2_shadow_buf_count;
	fifo_msg_buf_begin_header_index = g_mem_layout.segment1_msg_buf_count + \
			g_mem_layout.segment2_msg_buf_count + \
			g_mem_layout.segment1_shadow_buf_count + \
			g_mem_layout.segment2_shadow_buf_count;
	if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOA_ENABLED_MASK) {
		g_mem_layout.fifoa_msg_buf_start_offset = fifo_msg_buf_begin_offset;
		g_mem_layout.fifoa_first_msg_buf_header_index = fifo_msg_buf_begin_header_index;
		if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOB_ENABLED_MASK) {
			g_mem_layout.fifob_msg_buf_start_offset = g_mem_layout.fifoa_msg_buf_start_offset + fifoa_depth * fifo_data_size_in_bytes;
			g_mem_layout.fifob_first_msg_buf_header_index = g_mem_layout.fifoa_first_msg_buf_header_index + fifoa_depth;
		}
	} else {
		if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOB_ENABLED_MASK) {
			g_mem_layout.fifob_msg_buf_start_offset = fifo_msg_buf_begin_offset;
			g_mem_layout.fifob_first_msg_buf_header_index = fifo_msg_buf_begin_header_index;
		}
	}
	/* Calculate offsets in memory area for FIFO buffers. */
	for(i = 0;i < fifoa_depth + fifob_depth;i++)
		g_mem_layout.msg_buf_offsets[static_buf_idx + dynamic_buf_idx + g_mem_layout.segment1_shadow_buf_count + g_mem_layout.segment2_shadow_buf_count + i] = \
			fifo_msg_buf_begin_offset + fifo_data_size_in_bytes * i;
	/* Sync frame table offset*/
	g_mem_layout.sftor = fifo_msg_buf_begin_offset + (fifoa_depth + fifob_depth) * fifo_data_size_in_bytes;
	DBG_PRINT("sftor: %u", g_mem_layout.sftor);
	if(g_mem_layout.sftor + SYNC_FRAME_TABLE_SIZE > sizeof(g_fr_memory)) {
		DBG_PRINT("FlexRay memory area space not enough: %u > %u", g_mem_layout.sftor + SYNC_FRAME_TABLE_SIZE, sizeof(g_fr_memory));
		return FAILED;
	}

	g_mem_layout.segment1_last_msg_buf_num = g_mem_layout.segment1_msg_buf_count - 1;
	g_mem_layout.last_msg_buf_idx = g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count - 1;
#if 0
	DBG_PRINT("segment1_msg_buf_count : %u", g_mem_layout.segment1_msg_buf_count);
	DBG_PRINT("segment1_shadow_buf_count : %u", g_mem_layout.segment1_shadow_buf_count);
	DBG_PRINT("segment1_msg_buf_data_size : %u", g_mem_layout.segment1_msg_buf_data_size);
	DBG_PRINT("segment1_channel_a_shadow_buf_header_idx : %u", g_mem_layout.segment1_channel_a_shadow_buf_header_idx);
	DBG_PRINT("segment1_channel_b_shadow_buf_header_idx : %u", g_mem_layout.segment1_channel_b_shadow_buf_header_idx);
	DBG_PRINT("segment2_msg_buf_count : %u", g_mem_layout.segment2_msg_buf_count);
	DBG_PRINT("segment2_shadow_buf_count : %u", g_mem_layout.segment2_shadow_buf_count);
	DBG_PRINT("segment2_msg_buf_data_size : %u", g_mem_layout.segment2_msg_buf_data_size);
	DBG_PRINT("segment2_channel_a_shadow_buf_header_idx : %u", g_mem_layout.segment2_channel_a_shadow_buf_header_idx);
	DBG_PRINT("segment2_channel_b_shadow_buf_header_idx : %u", g_mem_layout.segment2_channel_b_shadow_buf_header_idx);
	if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOA_ENABLED_MASK) {
		DBG_PRINT("fifoa_msg_buf_start_offset : %u", g_mem_layout.fifoa_msg_buf_start_offset);

	}
	if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOB_ENABLED_MASK) {
		DBG_PRINT("fifob_msg_buf_start_offset : %u", g_mem_layout.fifob_msg_buf_start_offset);
	}
	DBG_PRINT("FlexRay memory Area Layout: %u bytes", g_mem_layout.sftor + SYNC_FRAME_TABLE_SIZE);
	DBG_PRINT("Index\tStart - End");
	DBG_PRINT("\t0 - %u\tMessage Headers", msg_buf_headers_size);
	for(i = 0;i < g_mem_layout.segment1_msg_buf_count;i++)
		DBG_PRINT("%u\t%u - %u\tSegment 1 Msg Buf %u", i, g_mem_layout.msg_buf_offsets[i], g_mem_layout.msg_buf_offsets[i] + g_mem_layout.segment1_msg_buf_data_size, i);
	for(i = 0;i < g_mem_layout.segment2_msg_buf_count;i++)
		DBG_PRINT("%u\t%u - %u\tSegment 2 Msg Buf %u", g_mem_layout.segment1_msg_buf_count + i, g_mem_layout.msg_buf_offsets[g_mem_layout.segment1_msg_buf_count + i], g_mem_layout.msg_buf_offsets[g_mem_layout.segment1_msg_buf_count + i] + g_mem_layout.segment2_msg_buf_data_size, i);
	for(i = 0;i < g_mem_layout.segment1_shadow_buf_count;i++)
		DBG_PRINT("%u\t%u - %u\tSegment 1 Shadow Buf %u", g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count + i, g_mem_layout.msg_buf_offsets[g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count + i], g_mem_layout.msg_buf_offsets[g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count + i] + g_mem_layout.segment1_msg_buf_data_size, i);
	for(i = 0;i < g_mem_layout.segment2_shadow_buf_count;i++)
		DBG_PRINT("%u\t%u - %u\tSegment 2 Shadow Buf %u", g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count + g_mem_layout.segment1_shadow_buf_count + i, g_mem_layout.msg_buf_offsets[g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count + g_mem_layout.segment1_shadow_buf_count + i], g_mem_layout.msg_buf_offsets[g_mem_layout.segment1_msg_buf_count + g_mem_layout.segment2_msg_buf_count + g_mem_layout.segment1_shadow_buf_count + i] + g_mem_layout.segment2_msg_buf_data_size, i);
	if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOA_ENABLED_MASK) {
		for(i = 0;i < fifoa_depth;i++)
			DBG_PRINT("%u\t%u - %u\tFIFOA\tMsg Buf %u", g_mem_layout.fifoa_first_msg_buf_header_index + i, g_mem_layout.msg_buf_offsets[g_mem_layout.fifoa_first_msg_buf_header_index + i], g_mem_layout.msg_buf_offsets[g_mem_layout.fifoa_first_msg_buf_header_index + i] + g_mem_layout.fifo_buffer_data_size, i);
	}
	if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOB_ENABLED_MASK) {
		for(i = 0;i < fifob_depth;i++)
			DBG_PRINT("%u\t%u - %u\tFIFOB\tMsg Buf %u", g_mem_layout.fifob_first_msg_buf_header_index + i, g_mem_layout.msg_buf_offsets[g_mem_layout.fifob_first_msg_buf_header_index + i], g_mem_layout.msg_buf_offsets[g_mem_layout.fifob_first_msg_buf_header_index + i] + g_mem_layout.fifo_buffer_data_size, i);
	}
	DBG_PRINT("\t%u - %u\tSync Frame Table", g_mem_layout.sftor, g_mem_layout.sftor + SYNC_FRAME_TABLE_SIZE);
#endif
	return SUCCESS;
}

static uint8_t init_rx_msg_buf(const fr_msg_buffer * msg_buf_cfg, uint8_t msg_buf_num) {
    volatile uint16_t reg_offset = (((uint16_t)msg_buf_num) * 4U);
    volatile uint16_t * msg_buf_header_ptr;
    uint16_t reg_val;
	uint32_t msr;
    /* Disable the msg buf */
    if(FR_MBCCSR_EDS_U16 == (READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR_EDS_U16))
        WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, FR_MBCCSR_EDT_U16);
    if(FR_MBCCSR_EDS_U16 == (READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR_EDS_U16)) {
    	DBG_PRINT("Disable msg buf failed");
        return FAILED;
    }
	reg_val = 0U;
	WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, reg_val);
	reg_val = 0U;
	if(msg_buf_cfg->flags & MSG_BUF_CHANNEL_A_ENABLED_FLAG)
		reg_val |= FR_MBCCFR_CHA_U16;      /* Channel A */
	if(msg_buf_cfg->flags & MSG_BUF_CHANNEL_B_ENABLED_FLAG)
		reg_val |= FR_MBCCFR_CHB_U16;      /* Channel B */
	/* Cycle counter filter should be enabled? */
	if(msg_buf_cfg->flags & MSG_BUF_CYCLE_COUNTER_FILTER_ENABLED_FLAG) {
		reg_val |= (uint16_t)(EXTRACT_MSG_BUF_CYCLE_COUNTER_FILTER_VALUE(msg_buf_cfg->flags)) & FR_MBCCFR_CCFVAL_MASK_U16;
		reg_val |= (uint16_t)((uint16_t)(EXTRACT_MSG_BUF_CYCLE_COUNTER_FILTER_MASK(msg_buf_cfg->flags)) << 6U) & FR_MBCCFR_CCFMSK_MASK_U16;
		reg_val |= FR_MBCCFR_CCFE_U16;
	}
	WRITE_FR_REGISTER16(FR_MBCCFR0_OFFSET + reg_offset, reg_val);
	WRITE_FR_REGISTER16(FR_MBFIDR0_OFFSET + reg_offset, msg_buf_cfg->frame_id);
	msg_buf_header_ptr = (volatile uint16_t *)((uint32_t)&g_fr_memory[0]) + (msg_buf_num * FR_MESSAGE_BUFFER_HEADER_SIZE);
	/* Configure Frame Header registers */
	reg_val = 0U;
	reg_val |= g_mem_layout.msg_buf_offsets[msg_buf_num];
	WRITE_FR_REGISTER16(FR_MBDOR0_OFFSET + (msg_buf_num), reg_val);
	msg_buf_header_ptr[FR_MESSAGE_BUFFER_HEADER_SIZE - 1U] = 0x00U;   /* Clear slot status */
	/* Message Buffer Index Registers initialization */
	WRITE_FR_REGISTER16(FR_MBIDXR0_OFFSET + reg_offset, (uint16_t)msg_buf_num);
	/* Enable message buffer */
	SUSPEND_INTERRUPT;
	reg_val = (uint16_t)(READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) | FR_MBCCSR_EDT_U16);
	WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, reg_val);
	RESUME_INTERRUPT;
	if(FR_MBCCSR_EDS_U16 != (READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR_EDS_U16))
	{
    	DBG_PRINT("Enable msg buf failed");
		return FAILED;
	}
    return SUCCESS;
}

static uint8_t init_tx_msg_buf(const fr_msg_buffer * msg_buf_cfg, uint8_t msg_buf_num) {
    uint16_t reg_val;
    volatile uint16_t reg_offset = (((uint16_t)msg_buf_num) * 4U);
    volatile uint16_t * msg_buf_header_ptr;
	uint32_t msr;
    if(FR_MBCCSR_EDS_U16 == (READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR_EDS_U16))
        WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, FR_MBCCSR_EDT_U16);
    if(FR_MBCCSR_EDS_U16 == (READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR_EDS_U16))
    {
    	DBG_PRINT("Disable msg buf failed");
        return FAILED;
    }
    /* MPC5748G Reference Manual Table 46-38: MTD will be on for Tx buf */
	reg_val = FR_MBCCSR_MTD_U16;
	WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, reg_val);
	reg_val = 0U;
	/* Tx channels configuration. */
	if(msg_buf_cfg->flags & MSG_BUF_CHANNEL_A_ENABLED_FLAG)
		reg_val |= FR_MBCCFR_CHA_U16;
	if(msg_buf_cfg->flags & MSG_BUF_CHANNEL_B_ENABLED_FLAG)
		reg_val |= FR_MBCCFR_CHB_U16;
	/* Dynamic segment buffer can only be on one channel. */
	if((msg_buf_cfg->frame_id) > g_fr_config.gNumberOfStaticSlots) {
		if((msg_buf_cfg->flags & MSG_BUF_CHANNEL_A_ENABLED_FLAG) && (msg_buf_cfg->flags & MSG_BUF_CHANNEL_B_ENABLED_FLAG)) {
			reg_val = FR_MBCCFR_CHA_U16;
		}
	}
	/* Cycle counter filter configuration. */
	if(msg_buf_cfg->flags & MSG_BUF_CYCLE_COUNTER_FILTER_ENABLED_FLAG)
	{
		reg_val |= (uint16_t)(EXTRACT_MSG_BUF_CYCLE_COUNTER_FILTER_VALUE(msg_buf_cfg->flags)) & FR_MBCCFR_CCFVAL_MASK_U16;
		reg_val |= (uint16_t)((uint16_t)(EXTRACT_MSG_BUF_CYCLE_COUNTER_FILTER_MASK(msg_buf_cfg->flags)) << 6U) & FR_MBCCFR_CCFMSK_MASK_U16;
		reg_val |= FR_MBCCFR_CCFE_U16;
	}
	WRITE_FR_REGISTER16(FR_MBCCFR0_OFFSET + reg_offset, reg_val);
	WRITE_FR_REGISTER16(FR_MBFIDR0_OFFSET + reg_offset, (uint16_t)(msg_buf_cfg->frame_id & FR_MBFIDR_FID_MASK_U16));
	msg_buf_header_ptr = (volatile uint16_t *)((uint32_t)&g_fr_memory[0]) + (msg_buf_num * FR_MESSAGE_BUFFER_HEADER_SIZE);
	reg_val = (uint16_t)(msg_buf_cfg->frame_id & 0x07FFU);
	if(msg_buf_cfg->flags & MSG_BUF_PAYLOAD_PREAMBLE_INDICATOR_FLAG)
		reg_val |= FR_FRAMEHEADER0_PPI_U16;
	msg_buf_header_ptr[0U] = reg_val;
	reg_val = 0U;
	if((msg_buf_cfg->frame_id) > g_fr_config.gNumberOfStaticSlots) {
		/* Per buffer max payload length for frames in dynamic segment */
		reg_val |= msg_buf_cfg->payload_length_max;
		msg_buf_header_ptr[1U] = reg_val;
		reg_val = calc_header_crc(0, 0, msg_buf_cfg->frame_id, msg_buf_cfg->payload_length_max);
		msg_buf_header_ptr[2U] = reg_val;
	} else {
		reg_val |= (uint16_t)(g_fr_config.gPayloadLengthStatic);
		msg_buf_header_ptr[1U] = reg_val;
		if(g_fr_config.pKeySlotId > 0 && g_fr_config.pKeySlotId == msg_buf_cfg->frame_id)
			reg_val = calc_header_crc(g_fr_config.pKeySlotUsedForSync, g_fr_config.pKeySlotUsedForStartup, msg_buf_cfg->frame_id, (uint16_t)(g_fr_config.gPayloadLengthStatic));
		else
			reg_val = calc_header_crc(0, 0, msg_buf_cfg->frame_id, (uint16_t)(g_fr_config.gPayloadLengthStatic));
		msg_buf_header_ptr[2U] = reg_val;
	}
	reg_val = 0U;
	reg_val |= g_mem_layout.msg_buf_offsets[msg_buf_num];
	WRITE_FR_REGISTER16(FR_MBDOR0_OFFSET + (msg_buf_num), reg_val);
	WRITE_FR_REGISTER16(FR_MBIDXR0_OFFSET + reg_offset, (uint16_t)(msg_buf_num));
	SUSPEND_INTERRUPT;
	reg_val = (uint16_t)(READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) | FR_MBCCSR_EDT_U16);
	/* Enable message buffer */
	WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset, reg_val);
	RESUME_INTERRUPT;
	if(FR_MBCCSR_EDS_U16 != (READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + reg_offset) & FR_MBCCSR_EDS_U16)) {
		DBG_PRINT("Enable msg buf failed");
		return FAILED;
	}
    return SUCCESS;
}

static uint8_t init_msg_bufs() {
    uint16_t i = 0;
    uint16_t reg_val = 0U;
    uint8_t ret = SUCCESS;
    /* MPC5748G Reference Manual 46.8.3.1.1 Individual Message Buffer Segments
		The set of the individual message buffers can be split up into two message buffer segments using the Message Buffer Segment Size and Utilization Register(FR_MBSSUTR).
		All individual message buffers with a message buffer number n <= FR_MBSSUTR[LAST_MB_SEG1] belong to the first message buffer segment.
		All individual message buffers with a message buffer number n > FR_MBSSUTR[LAST_MB_SEG1] belong to the second message buffer segment. */
	/* MPC5748G Reference Manual 46.7.6 Message Buffer Data Size Register (FR_MBDSR) */
    reg_val = (uint16_t)((uint16_t)(g_mem_layout.segment1_msg_buf_data_size) & FR_MBDSR_MBSEG1DS_MASK_U16);
    reg_val |= (uint16_t)((uint16_t)(g_mem_layout.segment2_msg_buf_data_size) << 8U) & FR_MBDSR_MBSEG2DS_MASK_U16;
    WRITE_FR_REGISTER16(FR_MBDSR_OFFSET, reg_val);
    reg_val = (uint16_t)((uint16_t)(g_mem_layout.last_msg_buf_idx) & FR_MBSSUTR_LAST_MB_UTIL_MASK_U16);
    reg_val |= (uint16_t)((uint16_t)(g_mem_layout.segment1_last_msg_buf_num) << 8U) & FR_MBSSUTR_LAST_MB_SEG1_MASK_U16;
    /* Configure FR_MBSSUTR_OFFSET */
    WRITE_FR_REGISTER16(FR_MBSSUTR_OFFSET, reg_val);
    /* MPC5748G Reference Manual 46.8.3.2 Receive Shadow Buffers
		The receive shadow buffers are required for the frame reception process for individual message buffers.
		The CC provides four receive shadow buffers, one receive shadow buffer per channel and per message buffer segment. */
    /* Configuration of receive shadow buffer for channel A and segment 1 */
    reg_val = (uint16_t)g_mem_layout.segment1_channel_a_shadow_buf_header_idx;
    /* Store settings for shadow buffer, ch A, seg 1 */
    WRITE_FR_REGISTER16(FR_RSBIR_OFFSET, reg_val);
    reg_val = 0U;            /* Clear temporary variable */
    reg_val |= g_mem_layout.msg_buf_offsets[g_mem_layout.segment1_channel_a_shadow_buf_header_idx];
    WRITE_FR_REGISTER16(FR_MBDOR0_OFFSET + (g_mem_layout.segment1_channel_a_shadow_buf_header_idx), reg_val);

    /* Configuration of receive shadow buffer for channel A and segment 2 */
    reg_val = (uint16_t)(g_mem_layout.segment2_channel_a_shadow_buf_header_idx);
    reg_val |= FR_RSBIR_SEL_RSBIR_A2_U16;     /* Selector field - ch A, seg 2 */
    WRITE_FR_REGISTER16(FR_RSBIR_OFFSET, reg_val);
    reg_val = 0U;            /* Clear temporary variable */
    reg_val |= g_mem_layout.msg_buf_offsets[g_mem_layout.segment2_channel_a_shadow_buf_header_idx];
    WRITE_FR_REGISTER16(FR_MBDOR0_OFFSET + (g_mem_layout.segment2_channel_a_shadow_buf_header_idx), reg_val);
    /* Configuration of receive shadow buffer for channel B and segment 1 */
    reg_val = (uint16_t)(g_mem_layout.segment1_channel_b_shadow_buf_header_idx);
    reg_val |= FR_RSBIR_SEL_RSBIR_B1_U16;     /* Selector field - ch B, seg 1 */
    WRITE_FR_REGISTER16(FR_RSBIR_OFFSET, reg_val);
    reg_val = 0U;
    reg_val |= g_mem_layout.msg_buf_offsets[g_mem_layout.segment1_channel_b_shadow_buf_header_idx];
    WRITE_FR_REGISTER16(FR_MBDOR0_OFFSET + (g_mem_layout.segment1_channel_b_shadow_buf_header_idx), reg_val);
    /* Configuration of receive shadow buffer for channel B and segment 2 */
    reg_val = (uint16_t)(g_mem_layout.segment2_channel_b_shadow_buf_header_idx);
    reg_val |= FR_RSBIR_SEL_RSBIR_B2_U16;     /* Selector field - ch B, seg 2 */
    /* Store settings for shadow buffer, ch B, seg 2 */
    WRITE_FR_REGISTER16(FR_RSBIR_OFFSET, reg_val);
    reg_val = 0U;
    reg_val |= g_mem_layout.msg_buf_offsets[g_mem_layout.segment2_channel_b_shadow_buf_header_idx];
    WRITE_FR_REGISTER16(FR_MBDOR0_OFFSET + (g_mem_layout.segment2_channel_b_shadow_buf_header_idx), reg_val);

    for(; i < (uint8_t)g_fr_config.individual_rx_msg_buf_count + g_fr_config.individual_tx_msg_buf_count; i++) {
		if(i < (uint8_t)g_fr_config.individual_rx_msg_buf_count)
			ret = init_rx_msg_buf(&g_fr_config.msg_bufs[i], g_mem_layout.individual_msg_buf_numbers[i]);
		else {
			if(g_fr_config.msg_bufs[i].payload_length_max > g_fr_config.pPayloadLengthDynMax)
				g_fr_config.msg_bufs[i].payload_length_max = g_fr_config.pPayloadLengthDynMax;
			ret = init_tx_msg_buf(&g_fr_config.msg_bufs[i], g_mem_layout.individual_msg_buf_numbers[i]);
		}
        if(FAILED == ret)
            return ret;
    }
    return SUCCESS;
}

static uint8_t switch_to_poc_config_state() {
    uint16_t cycle_counter = 0U;
    uint16_t poc_state = get_poc_state();
    if(FR_PSR0_PROTSTATE_READY_U16 != poc_state && FR_PSR0_PROTSTATE_DEFAULT_CONFIG_U16 != poc_state) {
		if(SUCCESS == send_chi_command(FR_POCR_CMD_FREEZE_U16)) {
			if(SUCCESS == wait_for_poc_state(FR_PSR0_PROTSTATE_HALT_U16)) {
				while(cycle_counter < MAX_WAIT_POC_STATE_CHANGE_CYCLES) {
					if(SUCCESS == wait_for_chi_cmd_completed())
						WRITE_FR_REGISTER16(FR_POCR_OFFSET,((uint16_t)(FR_POCR_CMD_DEFAULTCONFIG_U16 | FR_POCR_WME_U16)));
					poc_state = get_poc_state();
					if(FR_PSR0_PROTSTATE_DEFAULT_CONFIG_U16 == poc_state)
						break;
					sleep(SLEEP_MILISECONDS);
					cycle_counter++;
				}
				if(cycle_counter >= MAX_WAIT_POC_STATE_CHANGE_CYCLES) {
					DBG_PRINT("Wait poc state default config timeout");
					return FAILED;
				}
			}
		}
    }

	if(wait_for_chi_cmd_completed() == FAILED)
		return FAILED;
	WRITE_FR_REGISTER16(FR_POCR_OFFSET, ((uint16_t)(FR_POCR_CMD_CONFIG_U16 | FR_POCR_WME_U16)));
	return wait_for_poc_state(FR_PSR0_PROTSTATE_CONFIG_U16);
}

static uint8_t leave_poc_config_state() {
    uint8_t ret = FAILED;
    if(SUCCESS == wait_for_chi_cmd_completed()) {
        WRITE_FR_REGISTER16(FR_POCR_OFFSET, ((uint16_t)(FR_POCR_CMDCONFIGCOMPLETE_U16 | FR_POCR_WME_U16)));
        if(SUCCESS == wait_for_chi_cmd_completed()) {
            ret = wait_for_poc_state(FR_PSR0_PROTSTATE_READY_U16);
        }
    }
    return ret;
}

static void read_message_buffer(uint8_t * buf, uint8_t payload_len_in_words, uint8_t msg_buf_num) {
    uint8_t i;
    volatile uint32_t * msg_buf_ptr32 = (volatile uint32_t *)((uint32_t)&g_fr_memory[g_mem_layout.msg_buf_offsets[msg_buf_num]]);
    volatile uint16_t * msg_buf_ptr16 = (volatile uint16_t *)msg_buf_ptr32;
    /* Copy 4 bytes blocks */
    for(i = 0U; i < (payload_len_in_words >> 1U); i++)
        ((volatile uint32_t *)buf)[i] = msg_buf_ptr32[i];
    /* Copy remained 2 bytes block */
    for(i = (uint8_t)((i * 2U) & 0xFFU); i < payload_len_in_words; i++)
        ((volatile uint16_t *)buf)[i] = msg_buf_ptr16[i];
}


static uint8_t clear_all_interrupts() {
    uint16_t reg_val;
    uint16_t cycle_counter = 0U;
    uint16_t i;
	uint32_t msr;

	SUSPEND_INTERRUPT;
    for(i = 0U; i < MAX_MSG_BUFS; i++)
    {
        reg_val = READ_FR_REGISTER16(FR_MBCCSR0_OFFSET + (i * 4U));
        reg_val &= ~(FR_MBCCSR_CMT_U16 | FR_MBCCSR_MBIE_U16);
        reg_val |= FR_MBCCSR_MBIF_U16;
        WRITE_FR_REGISTER16(FR_MBCCSR0_OFFSET + (i * 4U), reg_val);
    }
	/* MPC5748G Reference Manual 46.7.11 Global Interrupt Flag and Enable Register (FR_GIFER) */
    while(1) {
        reg_val = READ_FR_REGISTER16(FR_GIFER_OFFSET);
        if((FR_GIFER_FAFAIF_U16 == (reg_val & FR_GIFER_FAFAIF_U16)) || (FR_GIFER_FAFBIF_U16 == (reg_val & FR_GIFER_FAFBIF_U16))) {
            reg_val |= (FR_GIFER_FAFAIF_U16 | FR_GIFER_FAFBIF_U16);
            WRITE_FR_REGISTER16(FR_GIFER_OFFSET, reg_val);
        } else
            break;
        cycle_counter++;
        if(cycle_counter >= 255U)
        	break;
    }
    RESUME_INTERRUPT;
    if(cycle_counter >= 255U) {
    	DBG_PRINT("Clear FAFAIF FAFBIF timeout");
    	return FAILED;
    }
	/* Clear all interrupt flags */
	WRITE_FR_REGISTER16(FR_GIFER_OFFSET,(uint16_t)(FR_GIFER_WUPIF_U16 | ((uint16_t)(FR_GIFER_FAFBIF_U16 | FR_GIFER_FAFAIF_U16))));
	WRITE_FR_REGISTER16(FR_PIFR0_OFFSET, 0xFFFFU);
	WRITE_FR_REGISTER16(FR_PIFR1_OFFSET, 0xFFFFU);
	WRITE_FR_REGISTER16(FR_CHIERFR_OFFSET, 0xFFFFU);
	/* Disable all FlexRay interrupts */
	WRITE_FR_REGISTER16(FR_GIFER_OFFSET, 0U);
	if(0U != READ_FR_REGISTER16(FR_GIFER_OFFSET)) {
		DBG_PRINT("Clear FR_GIFER failed");
		return FAILED;
	}
	WRITE_FR_REGISTER16(FR_PIER0_OFFSET, 0U);
	if(0U != READ_FR_REGISTER16(FR_PIER0_OFFSET)) {
		DBG_PRINT("Clear FR_PIER0 failed");
		return FAILED;
	}
	WRITE_FR_REGISTER16(FR_PIER1_OFFSET, 0U);
	if(0U != READ_FR_REGISTER16(FR_PIER1_OFFSET)) {
		DBG_PRINT("Clear FR_PIER1 failed");
		return FAILED;
	}
    return SUCCESS;
}

static void init_fifo(const fr_fifo_queue_config * const p_fifo_queue_cfg, uint8_t fifo_queue_index, uint16_t msg_buf_start_offset, uint16_t first_msg_buf_header_index)
{
    uint16_t reg_val = 0U;
    if(fifo_queue_index == 1) /* 0: FIFOA, 1: FIFOB */
        reg_val |= FR_RFWMSR_SEL_U16;

    WRITE_FR_REGISTER16(FR_RFWMSR_OFFSET, reg_val);
    WRITE_FR_REGISTER16(FR_RFSDOR_OFFSET, msg_buf_start_offset);
    WRITE_FR_REGISTER16(FR_RFSIR_OFFSET, (uint16_t)(first_msg_buf_header_index & FR_RFSIR_SIDX_MASK_U16));

    reg_val = (uint16_t)((uint16_t)(g_mem_layout.fifo_buffer_data_size) & FR_RFDSR_ENTRY_SIZE_MASK_U16); /*in words */
    reg_val |= (uint16_t)((uint16_t)(p_fifo_queue_cfg->depth) << 8U) & FR_RFDSR_FIFO_DEPTH_MASK_U16;
    WRITE_FR_REGISTER16(FR_RFDSR_OFFSET, reg_val);

    WRITE_FR_REGISTER16(FR_RFMIDAFVR_OFFSET, (uint16_t)(p_fifo_queue_cfg->message_id_filter_value));
    WRITE_FR_REGISTER16(FR_RFMIDAFMR_OFFSET, (uint16_t)(p_fifo_queue_cfg->message_id_filter_mask));
    WRITE_FR_REGISTER16(FR_RFFIDRFVR_OFFSET, 0U);
    WRITE_FR_REGISTER16(FR_RFFIDRFMR_OFFSET, 0x07FFU);

    /* Range filter 0 enabled? */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER0_ENABLE_MASK)
    {
        reg_val = FR_RFRFCFR_SEL_F0_U16;
        reg_val |= (p_fifo_queue_cfg->filter0_slot_id_lower & FR_RFRFCFR_SID_MASK_U16);
        WRITE_FR_REGISTER16(FR_RFRFCFR_OFFSET, reg_val);
        reg_val = FR_RFRFCFR_SEL_F0_U16;
        reg_val |= FR_RFRFCFR_IBD_UPPINT_U16;
        reg_val |= (p_fifo_queue_cfg->filter0_slot_id_upper & FR_RFRFCFR_SID_MASK_U16);
        WRITE_FR_REGISTER16(FR_RFRFCFR_OFFSET, reg_val);
    }

    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER1_ENABLE_MASK)
    {
        reg_val = FR_RFRFCFR_SEL_F1_U16;
        reg_val |= (p_fifo_queue_cfg->filter1_slot_id_lower & FR_RFRFCFR_SID_MASK_U16);
        WRITE_FR_REGISTER16(FR_RFRFCFR_OFFSET, reg_val);
        reg_val = FR_RFRFCFR_SEL_F1_U16;
        reg_val |= FR_RFRFCFR_IBD_UPPINT_U16;
        reg_val |= (p_fifo_queue_cfg->filter1_slot_id_upper & FR_RFRFCFR_SID_MASK_U16);
        WRITE_FR_REGISTER16(FR_RFRFCFR_OFFSET, reg_val);
    }

    /* Range filter 2 enabled? */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER2_ENABLE_MASK)
    {
        reg_val = FR_RFRFCFR_SEL_F2_U16;
        reg_val |= (p_fifo_queue_cfg->filter2_slot_id_lower & FR_RFRFCFR_SID_MASK_U16);
        WRITE_FR_REGISTER16(FR_RFRFCFR_OFFSET, reg_val);
        reg_val = FR_RFRFCFR_SEL_F2_U16;
        reg_val |= FR_RFRFCFR_IBD_UPPINT_U16;
        reg_val |= (p_fifo_queue_cfg->filter2_slot_id_upper & FR_RFRFCFR_SID_MASK_U16);
        WRITE_FR_REGISTER16(FR_RFRFCFR_OFFSET, reg_val);
    }

    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER3_ENABLE_MASK)
    {
        reg_val = FR_RFRFCFR_SEL_F3_U16;
        reg_val |= (p_fifo_queue_cfg->filter3_slot_id_lower & FR_RFRFCFR_SID_MASK_U16);
        WRITE_FR_REGISTER16(FR_RFRFCFR_OFFSET, reg_val);
        reg_val = FR_RFRFCFR_SEL_F3_U16;
        reg_val |= FR_RFRFCFR_IBD_UPPINT_U16;
        reg_val |= (p_fifo_queue_cfg->filter3_slot_id_upper & FR_RFRFCFR_SID_MASK_U16);
        WRITE_FR_REGISTER16(FR_RFRFCFR_OFFSET, reg_val);
    }

    reg_val = 0U;            /* Clear temporary variable */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER0_ENABLE_MASK)
        reg_val |= FR_RFRFCTR_F0EN_U16;             /* Enable Range Filter 0 */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER1_ENABLE_MASK)
        reg_val |= FR_RFRFCTR_F1EN_U16;             /* Enable Range Filter 1 */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER2_ENABLE_MASK)
        reg_val |= FR_RFRFCTR_F2EN_U16;             /* Enable Range Filter 2 */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER3_ENABLE_MASK)
        reg_val |= FR_RFRFCTR_F3EN_U16;             /* Enable Range Filter 3  */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER0_MODE_MASK)
        reg_val |= FR_RFRFCTR_F0MD_U16;             /* Range filter 0 as rejection filter */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER1_MODE_MASK)
        reg_val |= FR_RFRFCTR_F1MD_U16;             /* Range filter 1 as rejection filter */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER2_MODE_MASK)
        reg_val |= FR_RFRFCTR_F2MD_U16;             /* Range filter 2 as rejection filter */
    if(p_fifo_queue_cfg->flags & FR_FIFO_FILTER3_MODE_MASK)
        reg_val |= FR_RFRFCTR_F3MD_U16;             /* Range filter 3 as rejection filter */
    WRITE_FR_REGISTER16(FR_RFRFCTR_OFFSET, reg_val);
}

static uint16_t get_poc_state() {
    uint32_t reg_val;
    uint16_t poc_state;
    reg_val = READ_FR_REGISTER32(FR_PSR0_OFFSET);
    poc_state = (uint16_t)(reg_val);
    if ((poc_state & FR_PSR1_FRZ_U16) == FR_PSR1_FRZ_U16)
        poc_state = FR_PSR0_PROTSTATE_HALT_U16;
    else
        poc_state = (uint16_t)((uint16_t)(reg_val >> 16U) & FR_PSR0_PROTSTATE_MASK_U16);
    return poc_state;
}

static uint8_t send_chi_command(uint16_t chiCommand) {
    if(SUCCESS != wait_for_chi_cmd_completed())
    	return FAILED;
	WRITE_FR_REGISTER16(FR_POCR_OFFSET, ((uint16_t)(chiCommand | FR_POCR_WME_U16)));
	return SUCCESS;
}

static uint8_t wait_for_chi_cmd_completed() {
    uint16_t cycle_counter = 0U;
    while(cycle_counter < MAX_WAIT_POC_STATE_CHANGE_CYCLES && FR_POCR_BSY_U16 == (READ_FR_REGISTER16(FR_POCR_OFFSET) & FR_POCR_BSY_U16)) {
    	sleep(SLEEP_MILISECONDS);
        cycle_counter++;
    }

    if(cycle_counter >= MAX_WAIT_POC_STATE_CHANGE_CYCLES) {
    	DBG_PRINT("Wait chi time out!");
        return FAILED;
    }
    else
    	return SUCCESS;
}

static uint8_t wait_for_poc_state(const uint16_t poc_state)
{
	uint16_t cycle_counter = 0U;
	while(cycle_counter < MAX_WAIT_POC_STATE_CHANGE_CYCLES && get_poc_state() != poc_state) {
		sleep(SLEEP_MILISECONDS);
		cycle_counter++;
	}
	if(cycle_counter >= MAX_WAIT_POC_STATE_CHANGE_CYCLES) {
		DBG_PRINT("Wait poc state changing to 0x%X time out!", poc_state);
		return FAILED;
	} else
		return SUCCESS;
}

#ifdef __cplusplus
}
#endif
/* End of file */
