#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <assert.h>
#include <pthread.h>

#include <zmq.h>
#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"

#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"

#include <algorithm>
#include "packet.h"

#define DBG

#ifdef DBG
#define DBG_PRINT(fmt, ...) \
	do {\
		(void)fprintf(stderr, fmt, ##__VA_ARGS__);\
	} while(0);
#else
#define DBG_PRINT(fmt, ...)
#endif

namespace {
volatile int do_exit = 0; // Flag for process exit on signal
volatile int stop_pending = 0; // Flag for notifying all threads to exit
const char *ADAPTER_IP_ADDR = "192.168.5.10";
const in_port_t ADAPTER_TCP_PORT = 3888U;
const time_t CONNECT_TIMEOUT = 5U; // In seconds
const unsigned int SEND_HEALTH_INTERVAL = 1U; // In seconds
const time_t SELECT_TIMEOUT = 2U; // In seconds, should be greater than SEND_HEALTH_INTERVAL
const time_t MAX_RECV_IDLE_TIME = 4U; // In seconds, should be greater than SELECT_TIMEOUT
const long ZMQ_POLL_TIMEOUT = 1000; // In miliseconds
const ssize_t MAX_FRAME_LENGTH = 128U; // In words
int socket_fd = -1;
pthread_mutex_t send_lock;
uint8_t s_recv_packet_buffer[sizeof(packet_header) + MAX_FRAME_LENGTH];
size_t s_bytes_in_recv_packet_buffer = 0U;


bool tcp_connect() {
  int err;
	if((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		LOGE_100("socket create error %s in %s", strerror(err), __FUNCTION__);
		return false;
	}
	struct sockaddr_in server_addr = {
		.sin_family = AF_INET, .sin_addr = {.s_addr = inet_addr(ADAPTER_IP_ADDR)}, .sin_port = htons(ADAPTER_TCP_PORT) 
	};
	// Set non-blocking 
  long arg = fcntl(socket_fd, F_GETFL, NULL);
  arg |= O_NONBLOCK;
  if( fcntl(socket_fd, F_SETFL, arg) < 0) {
		LOGE_100("fcntl error %s in %s", strerror(errno), __FUNCTION__);
		goto fail;
  }
  err = connect(socket_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
	if(err >= 0)
		return true;
	if (errno == EINPROGRESS) {
		DBG_PRINT("connecting\n");
		struct timeval tv = {.tv_sec = CONNECT_TIMEOUT, .tv_usec = 0};
		fd_set myset;
		FD_ZERO(&myset);
		FD_SET(socket_fd, &myset);
		err = select(socket_fd + 1, NULL, &myset, NULL, &tv);
		if (err < 0 && errno != EINTR) {
			DBG_PRINT("select error while connecting: %d - %s\n", errno, strerror(errno));
			goto fail;
		} else if (err > 0) {
			int opt;
			socklen_t len = sizeof(opt);
			if (getsockopt(socket_fd, SOL_SOCKET, SO_ERROR, (void*)(&opt), &len) < 0) { 
				DBG_PRINT("error in getsockopt while connecting: %d - %s\n", errno, strerror(errno)); 
				goto fail;
			} 
			if (opt) { 
				DBG_PRINT("socket error while connecting: %d - %s\n", opt, strerror(opt));
				goto fail;
			}
			return true;
		 } else {
			goto fail;
		}
	} else {
		DBG_PRINT("connect error %d - %s\n", errno, strerror(errno));
	}
fail:
	close(socket_fd);
  return false;
}

bool tcp_sendall(const uint8_t *data, size_t len) {
	ssize_t sent = 0, cur_sent = 0;
	while(sent < len && !stop_pending) {
		struct timeval tv = {.tv_sec = SELECT_TIMEOUT, .tv_usec = 0};
		fd_set myset;
		FD_ZERO(&myset);
		FD_SET(socket_fd, &myset);
		int err = select(socket_fd + 1, NULL, &myset, NULL, &tv);
		if (err < 0 && errno != EINTR) {
			DBG_PRINT("select error while sending data: %d - %s\n", errno, strerror(errno));
			return false;
		} else if (err > 0) {
			cur_sent = send(socket_fd, data + sent, len - sent, 0);
			if(cur_sent < 0) {
				DBG_PRINT("send error: %d - %s\n", errno, strerror(errno));
				return false;
			}
			sent += cur_sent;
		}
	}
	return true;
}

void tcp_send_msg(uint16_t msg_type, const uint8_t *packet_body, size_t packet_body_len) {
	packet_header hdr;
	SET_PACKET_FLAG_TYPE(hdr.flags, msg_type);
	hdr.flags = htons(hdr.flags);
	hdr.length = htons(((uint16_t)sizeof(packet_header)) + packet_body_len);
	pthread_mutex_lock(&send_lock);
	if(tcp_sendall((uint8_t *)&hdr, sizeof(hdr)) && packet_body_len > 0)
		tcp_sendall(packet_body, packet_body_len);
	pthread_mutex_unlock(&send_lock);
}

void flexray_send(void *s) {
  int err;
	zmq_pollitem_t item = {.socket = s, .events = ZMQ_POLLIN};
	err = zmq_poll (&item, 1, ZMQ_POLL_TIMEOUT);
	assert (err >= 0);
	if(err < 0) {
		LOGE_100("zmq_poll error %s in %s", strerror(errno ), __FUNCTION__);
		return;
	} else if(err == 0) {
		return;
	}
  zmq_msg_t msg;
  zmq_msg_init(&msg);
  err = zmq_msg_recv(&msg, s, 0);
  assert(err >= 0);

  // format for board, make copy due to alignment issues, will be freed on out of scope
  auto amsg = kj::heapArray<capnp::word>((zmq_msg_size(&msg) / sizeof(capnp::word)) + 1);
  memcpy(amsg.begin(), zmq_msg_data(&msg), zmq_msg_size(&msg));

  capnp::FlatArrayMessageReader cmsg(amsg);
  cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
  int msg_count = event.getSendFlexRay().size();
  for (int i = 0; i < msg_count; i++) {
    auto cmsg = event.getSendFlexRay()[i];
		packet_header hdr;
		SET_PACKET_FLAG_TYPE(hdr.flags, PACKET_TYPE_FLEXRAY_FRAME);
		SET_PACKET_FLAG_FRAME_ID(hdr.flags, cmsg.getFrameId());
		hdr.flags = htons(hdr.flags);
		hdr.length = htons(sizeof(packet_header) + cmsg.getDat().size());
		pthread_mutex_lock(&send_lock);
		if(!stop_pending) tcp_sendall((uint8_t *)&hdr, sizeof(hdr));
		if(!stop_pending) tcp_sendall(cmsg.getDat().begin(), cmsg.getDat().size());
		pthread_mutex_unlock(&send_lock);
	}
  zmq_msg_close(&msg);
}

void *flexray_send_thread(void *crap) {
  LOGD("start send thread");
	char *value;
  size_t value_sz = 0;
	LOGW("waiting for flexray params");
  while (!stop_pending) {
    const int result = read_db_value(NULL, "FlexRayParams", &value, &value_sz);
    if (value_sz > 0) break;
    usleep(100*1000);
  }
	if(stop_pending) return NULL;
  LOGW("got %d bytes of flexray params", value_sz);
	tcp_send_msg(PACKET_TYPE_START_DRIVER, (const uint8_t *)value, value_sz);
	free(value);
	
  // sendFlexRay = 8066
  void *context = zmq_ctx_new();
  void *subscriber = zmq_socket(context, ZMQ_SUB);
  zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);
  zmq_connect(subscriber, "tcp://127.0.0.1:8066");

  while (!stop_pending) {
    flexray_send(subscriber);
  }
	zmq_close(subscriber);
	zmq_ctx_destroy(context);
	DBG_PRINT("%s exit.\n", __FUNCTION__);
  return NULL;
}

void *flexray_health_thread(void *crap) {
  // run at 1hz
  while (!stop_pending) {
		// Send a health message to adapter, to keep the connection alive.
		tcp_send_msg(PACKET_TYPE_HEALTH, NULL, 0);
    sleep(SEND_HEALTH_INTERVAL);
  }
	DBG_PRINT("%s exit.\n", __FUNCTION__);
  return NULL;
}

int set_realtime_priority(int level) {
  // should match python using chrt
  struct sched_param sa;
  memset(&sa, 0, sizeof(sa));
  sa.sched_priority = level;
  return sched_setscheduler(getpid(), SCHED_FIFO, &sa);
}

/* Return false in case FlexRay adapter report error */
bool process_packet(void *publisher, const packet_header *pkt_hdr) {
	switch(EXTRACT_PACKET_FLAG_TYPE(ntohs(pkt_hdr->flags))) {
		case PACKET_TYPE_FLEXRAY_JOINED_CLUSTER:
			DBG_PRINT("Joined into cluster.\n");
			break;
		case PACKET_TYPE_FLEXRAY_JOIN_CLUSTER_FAILED:
			DBG_PRINT("Join cluster failed.\n");
			break;
		case PACKET_TYPE_FLEXRAY_DISCONNECTED_FROM_CLUSTER:
			DBG_PRINT("Disconnect from cluster.\n");
			return false;
		case PACKET_TYPE_FLEXRAY_FATAL_ERROR:
			DBG_PRINT("Fatal error!\n");
			return false;
		case PACKET_TYPE_HEALTH:
			break;
		case PACKET_TYPE_FLEXRAY_FRAME:
		{
			uint16_t msg_len = ntohs(pkt_hdr->length);
			if(msg_len <= sizeof(packet_header)){
				DBG_PRINT("Invalid frame msg len: %u\n", msg_len );
				break;
			}
			uint16_t payload_len = msg_len - sizeof(packet_header);
			if(payload_len <= 0 || payload_len > MAX_FRAME_LENGTH * 2) {
				DBG_PRINT("Invalid frame payload len: %u\n", payload_len );
				break;
			}
			capnp::MallocMessageBuilder msg;
			cereal::Event::Builder event = msg.initRoot<cereal::Event>();
			event.setLogMonoTime(nanos_since_boot());
			auto flexRayData = event.initFlexRay(1);
			flexRayData[0].setFrameId(EXTRACT_PACKET_FLAG_FRAME_ID(ntohs(pkt_hdr->flags)));
			flexRayData[0].setDat(kj::arrayPtr((uint8_t *)(pkt_hdr + 1), payload_len));
			auto words = capnp::messageToFlatArray(msg);
			auto bytes = words.asBytes();
			zmq_send(publisher, bytes.begin(), bytes.size(), 0);
			break;
		}
		default:
			LOGW("Unknown packet type: %u", EXTRACT_PACKET_FLAG_TYPE(ntohs(pkt_hdr->flags)));
			break;
	}
	return true;
}

bool do_tcp_recv(void *publisher) {
	size_t required_len = 0;
	if(s_bytes_in_recv_packet_buffer < sizeof(packet_header))
		required_len = sizeof(packet_header) - s_bytes_in_recv_packet_buffer;
	else {
		required_len = ntohs(((packet_header *)s_recv_packet_buffer)->length) - s_bytes_in_recv_packet_buffer;
	}
	ssize_t recved = recv(socket_fd, s_recv_packet_buffer + s_bytes_in_recv_packet_buffer, required_len, 0);
	if(recved < 0) {
		DBG_PRINT("recv error: %d - %s\n", errno, strerror(errno));
		return false;
	} else if(recved == 0) {
		DBG_PRINT("peer shutdown\n");
		return false;
	} else {
		s_bytes_in_recv_packet_buffer += recved;
		if(s_bytes_in_recv_packet_buffer < sizeof(packet_header))
			return true;
		size_t packet_len = ntohs(((packet_header *)s_recv_packet_buffer)->length);
		if(s_bytes_in_recv_packet_buffer == sizeof(packet_header)) { // Header reception completed.
			if(packet_len > sizeof(s_recv_packet_buffer) || packet_len < sizeof(packet_header)) {
				DBG_PRINT("Invalid msg length: %u\n", ntohs(((packet_header *)s_recv_packet_buffer)->length) );
				return false;
			}
		}
		if(s_bytes_in_recv_packet_buffer == packet_len) { // Packet reception completed.
			if(!process_packet(publisher, (packet_header *)s_recv_packet_buffer))
				return false;
			s_bytes_in_recv_packet_buffer = 0;
		}
		return true;
	}
}

void set_do_exit(int sig) {
  do_exit = 1;
}

}

int main() {
  int err;
  LOGW("starting flexrayd");
	signal(SIGINT, (sighandler_t) set_do_exit);
  signal(SIGTERM, (sighandler_t) set_do_exit);
  // set process priority
  err = set_realtime_priority(4);
  LOG("setpriority returns %d", err);
  void *context = zmq_ctx_new();
  void *publisher = zmq_socket(context, ZMQ_PUB);
  // flexRay = 8065
  zmq_bind(publisher, "tcp://*:8065");
	// Do connect & recv in main thread
	while(!do_exit) {
		LOG("Attempting to connect");
		while (!tcp_connect() && !do_exit) { usleep(100*1000); }
		if(do_exit) break;
		stop_pending = 0;
		// Sending & health check are in dedicate worker threads
		pthread_t flexray_send_thread_handle;
		err = pthread_create(&flexray_send_thread_handle, NULL,
												 flexray_send_thread, NULL);
		assert(err == 0);
		pthread_t flexray_health_thread_handle;
		err = pthread_create(&flexray_health_thread_handle, NULL,
												 flexray_health_thread, NULL);
		assert(err == 0);
		s_bytes_in_recv_packet_buffer = 0;
		bool recv_err = false;
		unsigned int recv_idle_seconds = 0;
		while(!do_exit && !recv_err) {
			struct timeval tv = {.tv_sec = SELECT_TIMEOUT, .tv_usec = 0};
			fd_set myset;
			FD_ZERO(&myset);
			FD_SET(socket_fd, &myset);
			int err = select(socket_fd + 1, &myset, NULL, NULL, &tv);
			if (err < 0 && errno != EINTR) {
				DBG_PRINT("select error while receiving data: %d - %s\n", errno, strerror(errno));
				recv_err = true;
			} else if (err > 0) {
				recv_idle_seconds = 0;
				if(!do_tcp_recv(publisher))
					recv_err = true;
			} else if(err == 0) {
				recv_idle_seconds += SELECT_TIMEOUT;
				if(recv_idle_seconds >= MAX_RECV_IDLE_TIME) {
					DBG_PRINT("Health check failed, ethernet cable may be unplugged, reconnect now\n");
					recv_err = true;
				}
			}
		}
		stop_pending = 1; // Notify all worker threads to exit
		err = pthread_join(flexray_send_thread_handle, NULL);
		assert(err == 0);
		err = pthread_join(flexray_health_thread_handle, NULL);
		assert(err == 0);
		close(socket_fd);
	}
	zmq_close(publisher);
	zmq_ctx_destroy(context);
	DBG_PRINT("Exit\n");
}
