#include <stdint.h>
#include <math.h>
#include "FreeRTOS.h"
#include <queue.h>
#include <task.h>
#include <event_groups.h>
#include "platform_defs.h"
#include "flexray_registers.h"
#include "flexray_driver.h"
#include "flexray_state_machine.h"
#include "packet.h"
#include "tcp_interface.h"
#include "event.h"

flexray_data g_flexray_data = {.state = FLEXRAY_WAITING_CLIENT_CONNECTION};

typedef struct {
	packet_header hdr;
	uint8_t payload[cPayloadLengthMax * 2];
}frame_packet;
frame_packet s_frame_packet;

static uint8_t handle_rx() {
    uint8_t ret = SUCCESS;
    fr_rx_status RxStatus;
    uint8_t u8RxLength = 0U;
    uint8_t i = 0U;
    uint16_t frame_id = 0;
    /* Receive on all individual rx msg bufs and FIFOs */
	for(; i < (uint8_t)g_fr_config.individual_rx_msg_buf_count; i++) {
		ret = flexray_driver_read_rx_buffer(i, &s_frame_packet.payload[0], &RxStatus, &u8RxLength);
		/* Check the success of the call */
		if(SUCCESS != ret) {
			DBG_PRINT("flexray_driver_read_rx_buffer error %u", ret);
			return ret;
		} else {
			if(FR_RX_STATUS_RECEIVED == RxStatus) {
				SET_PACKET_FLAG_FRAME_ID(s_frame_packet.hdr.flags, g_fr_config.msg_bufs[i].frame_id);
				tcp_interface_send_packet(PACKET_TYPE_FLEXRAY_FRAME, &s_frame_packet.hdr, u8RxLength);
			}
		}
	}

    if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOA_ENABLED_MASK) {
		while(1) {
			flexray_driver_receive_fifoa(&s_frame_packet.payload[0], &RxStatus, &u8RxLength, &frame_id);
			if(RxStatus != FR_RX_STATUS_RECEIVED && RxStatus != FR_RX_STATUS_RECEIVED_MORE_DATA_AVAILABLE)
				break;
			SET_PACKET_FLAG_FRAME_ID(s_frame_packet.hdr.flags, frame_id);
			tcp_interface_send_packet(PACKET_TYPE_FLEXRAY_FRAME, &s_frame_packet.hdr, u8RxLength);
		}
    }

    if(g_fr_config.flags & FR_CONFIG_FLAG_FIFOB_ENABLED_MASK) {
    	while(1) {
			flexray_driver_receive_fifob(&s_frame_packet.payload[0], &RxStatus, &u8RxLength, &frame_id);
			if(RxStatus != FR_RX_STATUS_RECEIVED && RxStatus != FR_RX_STATUS_RECEIVED_MORE_DATA_AVAILABLE)
				break;
			SET_PACKET_FLAG_FRAME_ID(s_frame_packet.hdr.flags, frame_id);
			tcp_interface_send_packet(PACKET_TYPE_FLEXRAY_FRAME, &s_frame_packet.hdr, u8RxLength);
		}
	}
    return(ret);
}

#if 0
/* 40 bytes per task. */
static char task_stat_buf[40 * 10 + 2] = "\r\n";
static void print_task_statistics() {
	vTaskGetRunTimeStats(&task_stat_buf[1]);
	DBG_PRINT(task_stat_buf);
}
#endif

uint8_t set_abs_timer() {
    uint8_t ret, desired_timer_expire_cycle;
    uint8_t cur_cycle_counter = 0U;
    uint16_t cur_macro_tick = 0U;
    int16_t rate_correction, offset_correction;
    /* Handle read/write before symbol window */
    uint16_t offset_macroticks = (uint16_t)gMacroPerCycle - (g_fr_config.gdSymbolWindow + g_fr_config.gdNIT);
	/* DEBUG: Track min/max rate/offset corrections */
    if(g_fr_config.flags & FR_CONFIG_FLAG_LOG_STATUS_DATA_MASK) {
		flexray_driver_get_clock_correction(&rate_correction, &offset_correction);
		if(rate_correction > g_flexray_data.max_rate_correction)
			g_flexray_data.max_rate_correction = rate_correction;
		if(rate_correction < g_flexray_data.min_rate_correction)
			g_flexray_data.min_rate_correction = rate_correction;
		if(offset_correction > g_flexray_data.max_offset_correction)
			g_flexray_data.max_offset_correction = offset_correction;
		if(offset_correction < g_flexray_data.max_offset_correction)
			g_flexray_data.min_offset_correction = offset_correction;
    }
    ret = flexray_driver_get_global_time(&cur_cycle_counter, &cur_macro_tick);
    if(SUCCESS != ret)
    	return ret;

	if(cur_cycle_counter < cCycleCountMax) {
		desired_timer_expire_cycle = cur_cycle_counter + 1U;
	} else {
		desired_timer_expire_cycle = 0U;
	}
	/* Set the timer to expire in the next cycle */
	ret = flexray_driver_set_abs_timer(0U, desired_timer_expire_cycle, offset_macroticks);
	if(SUCCESS != ret) {
		return ret;
		DBG_PRINT("flexray_driver_set_abs_timer error %u", ret);
	}
	/* We sleep for a while, give cpu to other tasks, wait the timer to expire.
	 * FlexRay spec 2.1, B.4.11: max value of gMacroPerCycle is 16000
	*/
	sleep( floor((double)(gMacroPerCycle + offset_macroticks  - cur_macro_tick) * ((double)g_fr_config.gdMacrotick) / 1000.0) );
    return ret;
}

void flexray_run()
{
    uint8_t ret;
    uint8_t timer_expired = 0;
    fr_poc_status poc_state;
    EventBits_t event_bits;
    packet_header msg_hdr;

    /* Check in event group, start/stop driver on demand */
    event_bits =  xEventGroupWaitBits(
    		 g_flexray_data.in_event_group,
			 EVENT_GROUP_START_FLEXRAY_STATE_MACHINE_BIT | EVENT_GROUP_STOP_FLEXRAY_STATE_MACHINE_BIT,
    		pdTRUE, pdFALSE, 0 );
	if((event_bits & EVENT_GROUP_START_FLEXRAY_STATE_MACHINE_BIT) != 0) {
		DBG_PRINT("Client request me to start the driver");
		if(g_flexray_data.state != FLEXRAY_WAITING_CLIENT_CONNECTION) {
			DBG_PRINT("Invalid state while starting driver.");
			return;
		}
		g_flexray_data.wait_poc_ready_cycles_counter = MAX_WAIT_POC_STATE_CHANGE_CYCLES;
		g_flexray_data.state = FLEXRAY_INITIALIZED;
		g_flexray_data.max_rate_correction = g_flexray_data.min_rate_correction = g_flexray_data.max_offset_correction = g_flexray_data.min_offset_correction = 0;
		xEventGroupSetBits(g_flexray_data.out_event_group, EVENT_GROUP_FLEXRAY_STATE_MACHINE_STARTED_BIT);
	}
	else if((event_bits & EVENT_GROUP_STOP_FLEXRAY_STATE_MACHINE_BIT) != 0) {
		DBG_PRINT("Stop the state machine");
#if 0
		print_task_statistics();
#endif
		g_flexray_data.state = FLEXRAY_WAITING_CLIENT_CONNECTION;
		xEventGroupSetBits(g_flexray_data.out_event_group, EVENT_GROUP_FLEXRAY_STATE_MACHINE_STOPPED_BIT);
    }

    switch(g_flexray_data.state)
    {
		case FLEXRAY_WAITING_CLIENT_CONNECTION:
			break;
        case FLEXRAY_INITIALIZED:
            /* The FlexRay configuration was set - wait until it reaches the POC:Ready state */
            ret = flexray_driver_get_poc_status(&poc_state);
            /* Check the error status */
            if(SUCCESS != ret) {   /* The call was not successful - go to error state */
            	g_flexray_data.state = FLEXRAY_ERROR;
            	DBG_PRINT("flexray_driver_get_poc_status error %u at FLEXRAY_CONFIGURED", ret);
            	break;
            }
			if(FR_PSR0_PROTSTATE_READY_U16 == poc_state.state) {
				/* Allow the node to be a coldstart node */
				ret = flexray_driver_allow_coldstart();
				/* Check the status */
				if(SUCCESS != ret) {   /* An error has occurred - go to the error state */
					g_flexray_data.state = FLEXRAY_ERROR;
					DBG_PRINT("flexray_driver_allow_coldstart error %u", ret);
				} else {    /* No error, so join the cluster */
					 ret = flexray_driver_start_communication();
					 /* Check success of the call (not of the integration to cluster) */
					 if(SUCCESS != ret) {   /* An error has occurred - go to the error state */
						 g_flexray_data.state = FLEXRAY_ERROR;
						 DBG_PRINT("flexray_driver_start_communication error %u", ret);
					 } else {   /* No error, the controller started joining the cluster - go to the next state */
						 g_flexray_data.state = FLEXRAY_JOINING_CLUSTER;
						 DBG_PRINT("Joining cluster...");
					 }
				}
			} else {   /* The FlexRay controller has not reached the POC:Ready yet */
				if(0 < g_flexray_data.wait_poc_ready_cycles_counter) {
					g_flexray_data.wait_poc_ready_cycles_counter--;
					g_flexray_data.state = FLEXRAY_INITIALIZED;
				} else {   /* Timeout has expired - this is an error */
					g_flexray_data.state = FLEXRAY_ERROR;
					DBG_PRINT("Waiting poc:ready timeout");
				}
			}
            break;
        case FLEXRAY_JOINING_CLUSTER:
            ret = flexray_driver_get_poc_status(&poc_state);
            if(SUCCESS != ret) {
            	g_flexray_data.state = FLEXRAY_ERROR;
            	DBG_PRINT("flexray_driver_get_poc_status error %u", ret);
            	break;
            }
			if(FR_PSR0_PROTSTATE_NORMAL_ACTIVE_U16 == poc_state.state) {
				DBG_PRINT("Joining cluster succeeded.");
				tcp_interface_send_packet(PACKET_TYPE_FLEXRAY_JOINED_CLUSTER, &msg_hdr, 0U);
				ret = set_abs_timer();
				if(SUCCESS != ret)
					g_flexray_data.state = FLEXRAY_ERROR;
				else
					g_flexray_data.state = FLEXRAY_CHECK_TIMER_STATUS;
			} else if(FR_PSR0_PROTSTATE_HALT_U16 == poc_state.state) {
				tcp_interface_send_packet(PACKET_TYPE_FLEXRAY_JOIN_CLUSTER_FAILED, &msg_hdr, 0);
				g_flexray_data.state = FLEXRAY_ERROR;
			} else
				sleep(0);
            break;
        case FLEXRAY_CHECK_TIMER_STATUS:
            ret = flexray_driver_get_poc_status(&poc_state);
            if(SUCCESS != ret) {
            	g_flexray_data.state = FLEXRAY_ERROR;
            	DBG_PRINT("flexray_driver_get_poc_status error %u", ret);
            	break;
            }
			if((FR_PSR0_PROTSTATE_NORMAL_PASSIVE_U16 != poc_state.state) && FR_PSR0_PROTSTATE_NORMAL_ACTIVE_U16 != poc_state.state) {
				DBG_PRINT("Invalid poc state 0x%X before checking abs timer", poc_state.state);
				g_flexray_data.state = FLEXRAY_DISCONNECT_FROM_CLUSTER;
				break;
			}
			ret = flexray_driver_get_timer_irq_status(0U, &timer_expired);
			if(SUCCESS != ret) {
				g_flexray_data.state = FLEXRAY_ERROR;
				DBG_PRINT("Error in flexray_driver_get_timer_irq_status %u", ret);
			} else {
				if(1 == timer_expired) {
					ret = handle_rx();
					if(SUCCESS != ret) {
						g_flexray_data.state = FLEXRAY_ERROR;
						DBG_PRINT("Error in handle_rx");
						break;
					}
					ret = flexray_driver_ack_abs_timer(0U);
					if(SUCCESS != ret) {
						g_flexray_data.state = FLEXRAY_ERROR;
						DBG_PRINT("Error in flexray_driver_ack_abs_timer %u", ret);
						break;
					}
					ret = set_abs_timer(); /* Schedule next timer */
					if(SUCCESS != ret)
						g_flexray_data.state = FLEXRAY_ERROR;
					else
						g_flexray_data.state = FLEXRAY_CHECK_TIMER_STATUS;
					break;
				} else
					sleep(0);
			}
            break;
        case FLEXRAY_DISCONNECT_FROM_CLUSTER:
			tcp_interface_send_packet(PACKET_TYPE_FLEXRAY_DISCONNECTED_FROM_CLUSTER, &msg_hdr, 0);
            g_flexray_data.state = FLEXRAY_ERROR_FINAL;
            break;
        case FLEXRAY_ERROR:
        	/* Fatal error, should not happen. */
			tcp_interface_send_packet(PACKET_TYPE_FLEXRAY_FATAL_ERROR, &msg_hdr, 0);
            g_flexray_data.state = FLEXRAY_ERROR_FINAL;
            break;
        case FLEXRAY_ERROR_FINAL:
        	sleep(1);
            g_flexray_data.state = FLEXRAY_ERROR_FINAL;
            break;
        default:
        	DBG_PRINT("Unknown state");
            break;

    }
    return;
}

uint8_t flexray_write_tx_msg_buf(uint16_t frame_id, uint8_t *payload, uint16_t payload_length) {
	uint8_t ret = SUCCESS;
	uint8_t tx_msg_buf_idx =  (uint8_t)frame_id;
	if(tx_msg_buf_idx >= g_fr_config.individual_tx_msg_buf_count) {
		DBG_PRINT("Invalid transmit msg buf index: %u", tx_msg_buf_idx );
		return FAILED;
	}
	if(flexray_driver_tx_buffer_idle(tx_msg_buf_idx) == 1) {
		ret = flexray_driver_write_tx_buffer(tx_msg_buf_idx, payload, payload_length);
		if(SUCCESS != ret) {
			DBG_PRINT("flexray_driver_write_tx_buffer on msg buf %u error %u", tx_msg_buf_idx, ret);
			return FAILED;
		}
	} else {
		/* The slot is busy, drop it */
	}
	return(ret);
}
