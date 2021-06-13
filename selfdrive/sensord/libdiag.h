#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DIAG_MAX_RX_PKT_SIZ 4096

bool Diag_LSM_Init(uint8_t* pIEnv);
bool Diag_LSM_DeInit(void);

// DCI

#define DIAG_CON_APSS 0x001
#define DIAG_CON_MPSS 0x002
#define DIAG_CON_LPASS 0x004
#define DIAG_CON_WCNSS 0x008

enum {
  DIAG_DCI_NO_ERROR = 1001,
} diag_dci_error_type;

int diag_register_dci_client(int*, uint16_t*, int, void*);
int diag_log_stream_config(int client_id, int set_mask, uint16_t log_codes_array[], int num_codes);
int diag_register_dci_stream(void (*func_ptr_logs)(unsigned char *ptr, int len), void (*func_ptr_events)(unsigned char *ptr, int len));
int diag_release_dci_client(int*);

int diag_send_dci_async_req(int client_id, unsigned char buf[], int bytes, unsigned char *rsp_ptr, int rsp_len,
                            void (*func_ptr)(unsigned char *ptr, int len, void *data_ptr), void *data_ptr);


#ifdef __cplusplus
}
#endif
