#ifndef FLEXRAY_DRIVER_H_
#define FLEXRAY_DRIVER_H_

#ifdef __cplusplus
extern "C"{
#endif

#include "flexray_config.h"

#define SUCCESS ((uint8_t)0x00)
#define FAILED ((uint8_t)0x01)

#define MAX_WAIT_POC_STATE_CHANGE_CYCLES 512U
#define SLEEP_MILISECONDS 10U

/* Naming convention comes from FlexRay Communications System Protocol Specification Version 2.1 Revision A */
#define cHCrcInit 0x001AU
#define cHCrcPolynomial 0x0385U
#define cSlotIDMax 2047
#define cStaticSlotIdMax 1023U
#define cClockDeviationMax 0.0015
#define cSamplesPerBit 8
#define cCycleCountMax 63U
/* Max FlexRay frame payload length in words */
#define cPayloadLengthMax 127

extern fr_config g_fr_config;

#define adActionPointDifference ( (g_fr_config.gdActionPointOffset <= g_fr_config.gdMiniSlotActionPointOffset || g_fr_config.gNumberOfMinislots == 0) ? 0 : \
		(g_fr_config.gdActionPointOffset - g_fr_config.gdMiniSlotActionPointOffset) )
#define gMacroPerCycle ( g_fr_config.gdStaticSlot * g_fr_config.gNumberOfStaticSlots + \
		adActionPointDifference + \
		g_fr_config.gdMinislot * g_fr_config.gNumberOfMinislots + \
		g_fr_config.gdSymbolWindow + g_fr_config.gdNIT )

typedef enum
{
    FR_RX_STATUS_RECEIVED = 0U,
    FR_RX_STATUS_NOT_RECEIVED,
    FR_RX_STATUS_RECEIVED_MORE_DATA_AVAILABLE
} fr_rx_status;

typedef struct
{
    uint16_t wakeup_status;
    uint16_t error_mode;
    uint16_t startup_status;
    uint16_t state;
	uint8_t coldstart_noise;
	uint8_t halt_request;
	uint8_t freeze;
} fr_poc_status;

uint8_t flexray_driver_init();
uint8_t flexray_driver_deinit(void);

uint8_t flexray_driver_get_poc_status(fr_poc_status * poc_status_ptr);
uint8_t flexray_driver_allow_coldstart();
uint8_t flexray_driver_start_communication();
uint8_t flexray_driver_abort_communication();
uint8_t flexray_driver_set_wakeup_channel(uint32_t channel_index);
uint8_t flexray_driver_get_clock_correction(int16_t * rate_correction, int16_t * offset_correction);
uint8_t flexray_driver_get_sync_frame_table (
		uint16_t *frame_table_ptr,
		uint8_t *a_even_cnt,
		uint8_t *b_even_cnt,
		uint8_t *a_odd_cnt,
		uint8_t *b_odd_cnt,
		uint16_t *sfcntr
);
uint8_t flexray_driver_get_status_registers(uint16_t *psr0, uint16_t *psr1, uint16_t *psr2, uint16_t *psr3, uint16_t *pifr0);

uint8_t flexray_driver_receive_fifoa(uint8_t *buf_ptr, fr_rx_status * rx_status_ptr, uint8_t* data_len_ptr, uint16_t *frame_id_ptr);
uint8_t flexray_driver_receive_fifob(uint8_t *buf_ptr, fr_rx_status * rx_status_ptr, uint8_t* data_len_ptr, uint16_t *frame_id_ptr);
uint8_t flexray_driver_read_rx_buffer(uint8_t rx_msg_buf_idx,uint8_t *buf, fr_rx_status * rx_status_ptr, uint8_t * payload_len_ptr);
uint8_t flexray_driver_write_tx_buffer(uint8_t tx_msg_buf_idx, const uint8_t *buf_ptr, uint8_t data_len);
uint8_t flexray_driver_tx_buffer_idle(uint8_t tx_msg_buf_idx);

uint8_t flexray_driver_get_global_time(uint8_t * cycle_ptr, uint16_t * macrotick_ptr);
uint8_t flexray_driver_set_abs_timer(uint8_t timer_index, uint8_t cycle, uint16_t offset);
uint8_t flexray_driver_ack_abs_timer(uint8_t timer_index);
uint8_t flexray_driver_cancel_abs_timer(uint8_t timer_index);
uint8_t flexray_driver_get_timer_irq_status(uint8_t timer_index, uint8_t * status_ptr);

#ifdef __cplusplus
}
#endif

#endif /* FLEXRAY_DRIVER_H_ */
