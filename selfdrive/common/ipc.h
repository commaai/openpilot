#ifndef IPC_H
#define IPC_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int ipc_connect(const char* socket_path);
int ipc_bind(const char* socket_path);
int ipc_sendrecv_with_fds(bool send, int fd, void *buf, size_t buf_size, int* fds, int num_fds,
                          int *out_num_fds);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif