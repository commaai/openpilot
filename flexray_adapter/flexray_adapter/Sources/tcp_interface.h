/*
 * tcp_interface.h
 *
 */

#ifndef TCP_INTERFACE_H_
#define TCP_INTERFACE_H_

#define LISTEN_PORT 3888

struct netconn;

typedef struct{
	void *flexray_in_event_group;/* For events sent from TCP/USB task to FR task */
	void *event_group;
	struct netconn *conn;
} tcp_interface_data;

extern tcp_interface_data g_tcp_data;

void tcp_interface_init();
void tcp_interface_send_packet(uint16_t type, packet_header *msg_hdr, uint16_t data_len);

#endif /* TCP_INTERFACE_H_ */
