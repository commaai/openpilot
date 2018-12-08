/*
 * packet.h
	Packet exchanged between FlexRay adapter and Eon/PC
 */

#ifndef PACKET_H_
#define PACKET_H_

#define PACKET_TYPE_MIN 0

#define PACKET_TYPE_MAX 0x1FU

#define PACKET_TYPE_START_DRIVER 0U
/* Tx/Rx frame */
#define PACKET_TYPE_FLEXRAY_FRAME 1U
/* Health check msg */
#define PACKET_TYPE_HEALTH 2U
/* Joined into cluster successfully, config matched. */
#define PACKET_TYPE_FLEXRAY_JOINED_CLUSTER 3U
/* Joined into cluster failed. */
#define PACKET_TYPE_FLEXRAY_JOIN_CLUSTER_FAILED 4U
/* Disconnected from cluster. */
#define PACKET_TYPE_FLEXRAY_DISCONNECTED_FROM_CLUSTER 5U
/* Fatal error, should not happen. */
#define PACKET_TYPE_FLEXRAY_FATAL_ERROR 6U

#define EXTRACT_PACKET_FLAG_TYPE(flags) (flags >> 11)
#define SET_PACKET_FLAG_TYPE(flags, type) do {flags = ((flags & 0x07FFU) | (((uint16_t)type) << 11));} while(0);
#define EXTRACT_PACKET_FLAG_FRAME_ID(flags) ((uint16_t)(flags & 0x07FFU))
#define SET_PACKET_FLAG_FRAME_ID(flags, frame_id) do {flags = ((flags & 0xF800U) | ((uint16_t)frame_id));} while(0);

/* Packet exchanged between adapter and TCP client */
typedef struct {
	uint16_t length; /* Packet length in bytes, including the header*/
	/*
	 * Bit 0 - Bit 10: frame ID (for frame message only), FlexRay cSlotIDMax is 2047 == 0x7FF
	 * Bit 11 - Bit 15: packet type, 0 - 31
	*/
	uint16_t flags; /* Packet type and frame ID */
}packet_header;

#endif /* PACKET_H_ */
