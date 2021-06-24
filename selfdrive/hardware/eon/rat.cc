#include <semaphore.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "json11.hpp"

#include "libdiag.h"
#include "selfdrive/common/swaglog.h"


#define DIAG_NV_READ_F 38
#define DIAG_NV_WRITE_F 39

#define DIAG_SUBSYS_CMD 75
#define DIAG_SUBSYS_CMD_VER_2 128

#define DIAG_SUBSYS_FS 19

#define EFS2_DIAG_SYNC_NO_WAIT 48


struct __attribute__((packed)) NvPacket {
  uint8_t cmd_code;
  uint16_t nv_id;
  uint8_t data[128];
  uint16_t status;
};


enum NvStatus {
  NV_DONE,
  NV_BUSY,
  NV_FULL,
  NV_FAIL,
  NV_NOTACTIVE,
  NV_BADPARAM,
  NV_READONLY,
  NV_BADRG,
  NV_NOMEM,
  NV_NOTALLOC,
};

struct __attribute__((packed)) Efs2DiagSyncReq {
  uint8_t cmd_code;
  uint8_t subsys_id;
  uint16_t subsys_cmd_code;
  uint16_t sequence_num;
  char path[8];
};

struct __attribute__((packed)) Efs2DiagSyncResp {
  uint8_t cmd_code;
  uint8_t subsys_id;
  uint16_t subsys_cmd_code;
  uint16_t sequence_num;
  uint32_t sync_token;
  int32_t diag_errno;
};


struct SendDiagSyncState {
  sem_t sem;
  int len;
};

static void diag_send_sync_cb(unsigned char *ptr, int len, void *data_ptr) {
  SendDiagSyncState *s = (SendDiagSyncState*)data_ptr;
  s->len = len;
  sem_post(&s->sem);
}

static int diag_send_sync(int client_id, unsigned char* req_pkt, size_t pkt_len,
                           unsigned char* res_pkt, size_t res_pkt_size) {

  SendDiagSyncState s = {0};
  sem_init(&s.sem, 0, 0);

  int err = diag_send_dci_async_req(client_id, req_pkt, pkt_len, res_pkt, res_pkt_size,
                                    diag_send_sync_cb, &s);
  assert(err == DIAG_DCI_NO_ERROR);

  sem_wait(&s.sem);
  return s.len;
}

static void efs_sync(int client_id) {
  unsigned char res_pkt[DIAG_MAX_RX_PKT_SIZ];

  Efs2DiagSyncReq req_pkt = {
    .cmd_code = DIAG_SUBSYS_CMD,
    .subsys_id = DIAG_SUBSYS_FS,
    .subsys_cmd_code = EFS2_DIAG_SYNC_NO_WAIT,
    .sequence_num = (uint16_t)(rand() % 100),
  };
  req_pkt.path[0] = '/';
  req_pkt.path[1] = 0;

  int res_len = diag_send_sync(client_id, (unsigned char*)&req_pkt, sizeof(req_pkt),
                                res_pkt, sizeof(res_pkt));
  Efs2DiagSyncResp *resp = (Efs2DiagSyncResp*)res_pkt;

  if (res_len != sizeof(Efs2DiagSyncResp)
      || resp->cmd_code != DIAG_SUBSYS_CMD
      || resp->subsys_id != DIAG_SUBSYS_FS
      || resp->subsys_cmd_code != EFS2_DIAG_SYNC_NO_WAIT) {
    LOGW("efs_sync: bad response!");
    return;
  }
  if (resp->diag_errno != 0) {
    LOGW("efs_sync: error %d", resp->diag_errno);
  }
}

static int nv_read(int client_id, uint16_t nv_id, NvPacket &resp) {
  NvPacket req = {
    .cmd_code = DIAG_NV_READ_F,
    .nv_id = nv_id,
  };

  int res_len = diag_send_sync(client_id, (unsigned char*)&req, sizeof(req),
                               (unsigned char*)&resp, sizeof(resp));

  // hexdump((uint8_t*)&resp, res_len);

  if (resp.cmd_code != DIAG_NV_READ_F || resp.nv_id != nv_id) {
    LOGW("nv_read: diag command failed");
    return -1;
  }

  if (resp.status != NV_DONE) {
    LOGW("nv_read: read failed: %d", resp.status);
    return -1;
  }
  return 0;
}

static uint32_t nv_read_u32(int client_id, uint16_t nv_id) {
  NvPacket resp = {0};
  if (nv_read(client_id, nv_id, resp) < 0) {
    return 0;
  }
  return *(uint32_t*)resp.data;
}



int main() {
  int err;
  int client_id = 0;

  // setup diag
  bool ok = Diag_LSM_Init(NULL);
  assert(ok);

  uint16_t list = DIAG_CON_APSS | DIAG_CON_MPSS;
  int signal_type = SIGCONT;
  err = diag_register_dci_client(&client_id, &list, 0, &signal_type);
  assert(err == DIAG_DCI_NO_ERROR);

  // log some stuff
  std::map<std::string, uint16_t> to_log = {
    {"B13", 6502},
    {"B7", 6553},
    {"B17", 6606},
    {"B40", 6658},
    {"B1", 6710},
  };
  auto log = json11::Json::object {};
  for (auto const &kv : to_log) {
    NvPacket resp = {0};
    nv_read(client_id, kv.second, resp);
    log[kv.first] = *(uint8_t*)resp.data;
  }

  printf("%s\n", json11::Json(log).dump().c_str());

  // cleanup
  err = diag_release_dci_client(&client_id);
  assert(err == DIAG_DCI_NO_ERROR);
  Diag_LSM_DeInit();
}
