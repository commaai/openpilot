/*
 * FlexRay TCP interface, forward/receive FlexRay frames from/to TCP client.
 *
 */
#include "lwip/opt.h"
#include "lwip/sys.h"
#include "lwip/api.h"
#include <string.h>
#include <event_groups.h>
#include "platform_defs.h"
#include "packet.h"
#include "tcp_interface.h"
#include "flexray_driver.h"
#include "flexray_state_machine.h"
#include "event.h"

/* In milliseconds */
#define RECV_TIMEOUT 2000U
#define MAX_RECV_IDLE_TIME 4000U

typedef struct {
	packet_header msg_hdr;
	uint16_t reg_vals[6];
	int16_t min_max_corrections[4];
	/* Sync frame ID/Deviation tables. */
	uint16_t sync_frame_table[60];
}status_data_msg;
static status_data_msg s_status_data_packet;

/* The biggest message is MESSAGE_TYPE_START_DRIVER, with fr_config embedded */
uint8_t s_packet_parse_buf[sizeof(packet_header) + sizeof(fr_config)];
size_t s_bytes_in_pkt_parse_buf = 0U;
static uint8_t s_flexray_started = 0;
static SemaphoreHandle_t s_mutex;
tcp_interface_data g_tcp_data;

static void tcp_server_thread(void *arg);
static void flexray_loop(void *param);
static err_t write_all_to_netconn(struct netconn *conn, unsigned char *data, size_t len);
static void process_packet(const packet_header *pkt_header);
static size_t packet_parser_process_data(unsigned char *incoming_data, u16_t incoming_data_len);


static err_t write_all_to_netconn(struct netconn *conn, unsigned char *data, size_t len) {
	err_t err = ERR_OK;
	size_t bytes_written = 0, bytes_written_total = 0;
	while(bytes_written_total < len) {
		err = netconn_write_partly(conn, data + bytes_written_total, len - bytes_written_total, NETCONN_COPY, &bytes_written);
		if (err != ERR_OK) {
			DBG_PRINT("netconn_write: error \"%s\"", lwip_strerr(err));
			break;
		}
		bytes_written_total += bytes_written;
	}
	return err;
}

void tcp_interface_send_packet(uint16_t type, packet_header *msg_hdr, uint16_t data_len) {
	SET_PACKET_FLAG_TYPE(msg_hdr->flags, type);
	msg_hdr->length = sizeof(packet_header) + data_len;
	xSemaphoreTake( s_mutex, portMAX_DELAY );
	if(write_all_to_netconn(g_tcp_data.conn,  (uint8_t *)msg_hdr, msg_hdr->length) != ERR_OK)
		DBG_PRINT("Send msg to tcp client error");
	xSemaphoreGive( s_mutex );
	return;
}

static void process_packet(const packet_header *pkt_header) {
	uint16_t payload_length = 0U;
	uint8_t ret = FAILED;
	uint8_t a_even_cnt = 0, b_even_cnt = 0, a_odd_cnt = 0, b_odd_cnt = 0;
	packet_header send_pkt_header;
	switch(EXTRACT_PACKET_FLAG_TYPE(pkt_header->flags)) {
		case PACKET_TYPE_START_DRIVER:
			if(s_flexray_started) {
				DBG_PRINT("Recved MESSAGE_TYPE_START_DRIVER after flexray driver started" );
				break;
			}
			if(s_bytes_in_pkt_parse_buf > sizeof(packet_header) + sizeof(fr_config) ||
					s_bytes_in_pkt_parse_buf < sizeof(packet_header) + sizeof(fr_config) - sizeof(g_fr_config.msg_bufs)) {
				DBG_PRINT("Invalid MESSAGE_TYPE_START_DRIVER msg length: %u", s_bytes_in_pkt_parse_buf );
			}
			memcpy(&g_fr_config, pkt_header + 1, s_bytes_in_pkt_parse_buf - sizeof(packet_header));
            ret = flexray_driver_init();
            if(SUCCESS != ret) {
            	DBG_PRINT("flexray_driver_init error %u", ret);
            	flexray_driver_deinit();
            	break;
            }
			xEventGroupSetBits(g_tcp_data.flexray_in_event_group, EVENT_GROUP_START_FLEXRAY_STATE_MACHINE_BIT);
			xEventGroupWaitBits(g_tcp_data.event_group, EVENT_GROUP_FLEXRAY_STATE_MACHINE_STARTED_BIT,
					pdTRUE, pdTRUE, portMAX_DELAY );
			s_flexray_started = 1;
			break;
		case PACKET_TYPE_FLEXRAY_FRAME:
			if(!s_flexray_started) {
				DBG_PRINT("Recved MESSAGE_TYPE_FLEXRAY_FRAME before flexray driver started" );
				break;
			}
			if(s_bytes_in_pkt_parse_buf <= sizeof(packet_header)) {
				DBG_PRINT("Invalid MESSAGE_TYPE_FLEXRAY_FRAME msg length: %u", s_bytes_in_pkt_parse_buf );
			}
			payload_length = pkt_header->length - sizeof(packet_header);
			if(payload_length <= 0 || payload_length > cPayloadLengthMax * 2){
				DBG_PRINT("Invalid frame payload len: %u", payload_length );
			} else {
				flexray_write_tx_msg_buf(EXTRACT_PACKET_FLAG_FRAME_ID(pkt_header->flags), (uint8_t *)(pkt_header + 1), payload_length);
			}
			break;
		case PACKET_TYPE_HEALTH:
			if(!s_flexray_started || (g_fr_config.flags & FR_CONFIG_FLAG_LOG_STATUS_DATA_MASK) == 0) {
				tcp_interface_send_packet(PACKET_TYPE_HEALTH, &send_pkt_header, 0);
				break;
			}
			/* FlexRay spec 2.1: Section 9.3.1.3 Protocol status data */
			flexray_driver_get_status_registers(
					&s_status_data_packet.reg_vals[0], &s_status_data_packet.reg_vals[1], &s_status_data_packet.reg_vals[2], &s_status_data_packet.reg_vals[3], &s_status_data_packet.reg_vals[4]);
			flexray_driver_get_sync_frame_table(&s_status_data_packet.sync_frame_table[0], &a_even_cnt, &b_even_cnt, &a_odd_cnt, &b_odd_cnt, &s_status_data_packet.reg_vals[5]);
			memcpy(&s_status_data_packet.min_max_corrections[0], &g_flexray_data.max_rate_correction, sizeof(s_status_data_packet.min_max_corrections));
			tcp_interface_send_packet(PACKET_TYPE_HEALTH,
					(packet_header *)&s_status_data_packet,
					sizeof(s_status_data_packet.reg_vals) + sizeof(s_status_data_packet.min_max_corrections) + (a_even_cnt + b_even_cnt + a_odd_cnt + b_odd_cnt) * sizeof(uint16_t) * 2);

	}
	s_bytes_in_pkt_parse_buf = 0;
}

static size_t packet_parser_process_data(unsigned char *incoming_data, u16_t incoming_data_len) {
	size_t bytes_consumed = 0;
	packet_header *pkt_header = (packet_header *)&s_packet_parse_buf[0];
	if(s_bytes_in_pkt_parse_buf < sizeof(packet_header))
		bytes_consumed = min(sizeof(packet_header) - s_bytes_in_pkt_parse_buf, incoming_data_len );
	else
		bytes_consumed = min(pkt_header->length - s_bytes_in_pkt_parse_buf, incoming_data_len );
	memcpy(s_packet_parse_buf + s_bytes_in_pkt_parse_buf, incoming_data, bytes_consumed);
	s_bytes_in_pkt_parse_buf += bytes_consumed;
	if(sizeof(packet_header) == s_bytes_in_pkt_parse_buf) {
		/* Packet header sanity check */
		if(pkt_header->length > sizeof(s_packet_parse_buf) || pkt_header->length < sizeof(packet_header)) {
			DBG_PRINT("Invalid packet len: %u", pkt_header->length );
			return incoming_data_len;
		}
	}
	/* Packet payload reception completed? */
	if(s_bytes_in_pkt_parse_buf == pkt_header->length) {
		process_packet(pkt_header);
	}
	return bytes_consumed;
}

static void tcp_server_thread(void *arg) {
	struct netconn *conn = NULL;
	err_t err = ERR_OK;
	struct netbuf *buf = NULL;
	void *data = NULL;
	u16_t len = 0;
	size_t bytes_consumed = 0;
	int recv_idle_time = 0U;

#if LWIP_IPV6
	conn = netconn_new(NETCONN_TCP_IPV6);
	netconn_bind(conn, IP6_ADDR_ANY, LISTEN_PORT);
#else /* LWIP_IPV6 */
	conn = netconn_new(NETCONN_TCP);
	netconn_bind(conn, IP_ADDR_ANY, LISTEN_PORT);
#endif /* LWIP_IPV6 */
	if(conn == NULL) {
		DBG_PRINT("Create connection failed.");
		return;
	}
	netconn_listen(conn);
	s_flexray_started = 0;
	while (1) {
		DBG_PRINT("Listening on %u...", LISTEN_PORT);
		err = netconn_accept(conn, &g_tcp_data.conn);
		if (err == ERR_OK) {
			DBG_PRINT("Accepted new TCP connection");
			netconn_set_recvtimeout(g_tcp_data.conn, RECV_TIMEOUT);/* 2 second timeout*/
			recv_idle_time = 0;
			s_bytes_in_pkt_parse_buf = 0;
			while (1) {
				err = netconn_recv(g_tcp_data.conn, &buf);
				if(err == ERR_OK) {
					recv_idle_time = 0;
					do {
						netbuf_data(buf, &data, &len);
						bytes_consumed = 0;
						while(bytes_consumed < len)
							bytes_consumed += packet_parser_process_data((u8_t *)data + bytes_consumed, len - bytes_consumed);
					} while (netbuf_next(buf) >= 0);
					netbuf_delete(buf);
				} else if(err == ERR_TIMEOUT) {
					recv_idle_time += RECV_TIMEOUT;
					if(recv_idle_time >= MAX_RECV_IDLE_TIME) {
						DBG_PRINT("TCP connection health check failed, disconnect now");
						break;
					}
				} else {
					DBG_PRINT("TCP connection lost ");
					break;
				}
			}
			if(s_flexray_started) {
				xEventGroupSetBits(g_tcp_data.flexray_in_event_group, EVENT_GROUP_STOP_FLEXRAY_STATE_MACHINE_BIT);
				xEventGroupWaitBits(g_tcp_data.event_group, EVENT_GROUP_FLEXRAY_STATE_MACHINE_STOPPED_BIT,
						pdTRUE, pdTRUE, portMAX_DELAY );
	    		flexray_driver_deinit();
				s_flexray_started = 0;
			}
			netconn_close(g_tcp_data.conn);
			netconn_delete(g_tcp_data.conn);
		}
	}
}

static void flexray_loop(void *param) {
	while (1)
		flexray_run();
}

void tcp_interface_init()
{
	s_mutex = xSemaphoreCreateMutex();
	g_tcp_data.flexray_in_event_group = (void *)xEventGroupCreate();
	g_tcp_data.event_group = (void *)xEventGroupCreate();
	g_flexray_data.in_event_group = g_tcp_data.flexray_in_event_group;
	g_flexray_data.out_event_group = g_tcp_data.event_group;
	/* FlexRay state machine will call LWIP API which rely on netconn thread sem, so we must run it in LWIP threads. */
	sys_thread_new("flexray", flexray_loop, NULL, 2048U, DEFAULT_THREAD_PRIO);
	sys_thread_new("tcpserver", tcp_server_thread, NULL, 2048U, DEFAULT_THREAD_PRIO);
}
