# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
enum_nvme_subsys_type: dict[int, str] = {(NVME_NQN_DISC:=1): 'NVME_NQN_DISC', (NVME_NQN_NVME:=2): 'NVME_NQN_NVME', (NVME_NQN_CURR:=3): 'NVME_NQN_CURR'}
enum_nvme_ctrl_type: dict[int, str] = {(NVME_CTRL_IO:=1): 'NVME_CTRL_IO', (NVME_CTRL_DISC:=2): 'NVME_CTRL_DISC', (NVME_CTRL_ADMIN:=3): 'NVME_CTRL_ADMIN'}
enum_nvme_dctype: dict[int, str] = {(NVME_DCTYPE_NOT_REPORTED:=0): 'NVME_DCTYPE_NOT_REPORTED', (NVME_DCTYPE_DDC:=1): 'NVME_DCTYPE_DDC', (NVME_DCTYPE_CDC:=2): 'NVME_DCTYPE_CDC'}
_anonenum0: dict[int, str] = {(NVMF_ADDR_FAMILY_PCI:=0): 'NVMF_ADDR_FAMILY_PCI', (NVMF_ADDR_FAMILY_IP4:=1): 'NVMF_ADDR_FAMILY_IP4', (NVMF_ADDR_FAMILY_IP6:=2): 'NVMF_ADDR_FAMILY_IP6', (NVMF_ADDR_FAMILY_IB:=3): 'NVMF_ADDR_FAMILY_IB', (NVMF_ADDR_FAMILY_FC:=4): 'NVMF_ADDR_FAMILY_FC', (NVMF_ADDR_FAMILY_LOOP:=254): 'NVMF_ADDR_FAMILY_LOOP', (NVMF_ADDR_FAMILY_MAX:=255): 'NVMF_ADDR_FAMILY_MAX'}
_anonenum1: dict[int, str] = {(NVMF_TRTYPE_PCI:=0): 'NVMF_TRTYPE_PCI', (NVMF_TRTYPE_RDMA:=1): 'NVMF_TRTYPE_RDMA', (NVMF_TRTYPE_FC:=2): 'NVMF_TRTYPE_FC', (NVMF_TRTYPE_TCP:=3): 'NVMF_TRTYPE_TCP', (NVMF_TRTYPE_LOOP:=254): 'NVMF_TRTYPE_LOOP', (NVMF_TRTYPE_MAX:=255): 'NVMF_TRTYPE_MAX'}
_anonenum2: dict[int, str] = {(NVMF_TREQ_NOT_SPECIFIED:=0): 'NVMF_TREQ_NOT_SPECIFIED', (NVMF_TREQ_REQUIRED:=1): 'NVMF_TREQ_REQUIRED', (NVMF_TREQ_NOT_REQUIRED:=2): 'NVMF_TREQ_NOT_REQUIRED', (NVMF_TREQ_DISABLE_SQFLOW:=4): 'NVMF_TREQ_DISABLE_SQFLOW'}
_anonenum3: dict[int, str] = {(NVMF_RDMA_QPTYPE_CONNECTED:=1): 'NVMF_RDMA_QPTYPE_CONNECTED', (NVMF_RDMA_QPTYPE_DATAGRAM:=2): 'NVMF_RDMA_QPTYPE_DATAGRAM', (NVMF_RDMA_QPTYPE_INVALID:=255): 'NVMF_RDMA_QPTYPE_INVALID'}
_anonenum4: dict[int, str] = {(NVMF_RDMA_PRTYPE_NOT_SPECIFIED:=1): 'NVMF_RDMA_PRTYPE_NOT_SPECIFIED', (NVMF_RDMA_PRTYPE_IB:=2): 'NVMF_RDMA_PRTYPE_IB', (NVMF_RDMA_PRTYPE_ROCE:=3): 'NVMF_RDMA_PRTYPE_ROCE', (NVMF_RDMA_PRTYPE_ROCEV2:=4): 'NVMF_RDMA_PRTYPE_ROCEV2', (NVMF_RDMA_PRTYPE_IWARP:=5): 'NVMF_RDMA_PRTYPE_IWARP'}
_anonenum5: dict[int, str] = {(NVMF_RDMA_CMS_RDMA_CM:=1): 'NVMF_RDMA_CMS_RDMA_CM'}
_anonenum6: dict[int, str] = {(NVMF_TCP_SECTYPE_NONE:=0): 'NVMF_TCP_SECTYPE_NONE', (NVMF_TCP_SECTYPE_TLS12:=1): 'NVMF_TCP_SECTYPE_TLS12', (NVMF_TCP_SECTYPE_TLS13:=2): 'NVMF_TCP_SECTYPE_TLS13', (NVMF_TCP_SECTYPE_INVALID:=255): 'NVMF_TCP_SECTYPE_INVALID'}
_anonenum7: dict[int, str] = {(NVME_REG_CAP:=0): 'NVME_REG_CAP', (NVME_REG_VS:=8): 'NVME_REG_VS', (NVME_REG_INTMS:=12): 'NVME_REG_INTMS', (NVME_REG_INTMC:=16): 'NVME_REG_INTMC', (NVME_REG_CC:=20): 'NVME_REG_CC', (NVME_REG_CSTS:=28): 'NVME_REG_CSTS', (NVME_REG_NSSR:=32): 'NVME_REG_NSSR', (NVME_REG_AQA:=36): 'NVME_REG_AQA', (NVME_REG_ASQ:=40): 'NVME_REG_ASQ', (NVME_REG_ACQ:=48): 'NVME_REG_ACQ', (NVME_REG_CMBLOC:=56): 'NVME_REG_CMBLOC', (NVME_REG_CMBSZ:=60): 'NVME_REG_CMBSZ', (NVME_REG_BPINFO:=64): 'NVME_REG_BPINFO', (NVME_REG_BPRSEL:=68): 'NVME_REG_BPRSEL', (NVME_REG_BPMBL:=72): 'NVME_REG_BPMBL', (NVME_REG_CMBMSC:=80): 'NVME_REG_CMBMSC', (NVME_REG_CRTO:=104): 'NVME_REG_CRTO', (NVME_REG_PMRCAP:=3584): 'NVME_REG_PMRCAP', (NVME_REG_PMRCTL:=3588): 'NVME_REG_PMRCTL', (NVME_REG_PMRSTS:=3592): 'NVME_REG_PMRSTS', (NVME_REG_PMREBS:=3596): 'NVME_REG_PMREBS', (NVME_REG_PMRSWTP:=3600): 'NVME_REG_PMRSWTP', (NVME_REG_DBS:=4096): 'NVME_REG_DBS'}
_anonenum8: dict[int, str] = {(NVME_CMBSZ_SQS:=1): 'NVME_CMBSZ_SQS', (NVME_CMBSZ_CQS:=2): 'NVME_CMBSZ_CQS', (NVME_CMBSZ_LISTS:=4): 'NVME_CMBSZ_LISTS', (NVME_CMBSZ_RDS:=8): 'NVME_CMBSZ_RDS', (NVME_CMBSZ_WDS:=16): 'NVME_CMBSZ_WDS', (NVME_CMBSZ_SZ_SHIFT:=12): 'NVME_CMBSZ_SZ_SHIFT', (NVME_CMBSZ_SZ_MASK:=1048575): 'NVME_CMBSZ_SZ_MASK', (NVME_CMBSZ_SZU_SHIFT:=8): 'NVME_CMBSZ_SZU_SHIFT', (NVME_CMBSZ_SZU_MASK:=15): 'NVME_CMBSZ_SZU_MASK'}
_anonenum9: dict[int, str] = {(NVME_CC_ENABLE:=1): 'NVME_CC_ENABLE', (NVME_CC_EN_SHIFT:=0): 'NVME_CC_EN_SHIFT', (NVME_CC_CSS_SHIFT:=4): 'NVME_CC_CSS_SHIFT', (NVME_CC_CSS_MASK:=112): 'NVME_CC_CSS_MASK', (NVME_CC_CSS_NVM:=0): 'NVME_CC_CSS_NVM', (NVME_CC_CSS_CSI:=96): 'NVME_CC_CSS_CSI', (NVME_CC_MPS_SHIFT:=7): 'NVME_CC_MPS_SHIFT', (NVME_CC_MPS_MASK:=1920): 'NVME_CC_MPS_MASK', (NVME_CC_AMS_SHIFT:=11): 'NVME_CC_AMS_SHIFT', (NVME_CC_AMS_MASK:=14336): 'NVME_CC_AMS_MASK', (NVME_CC_AMS_RR:=0): 'NVME_CC_AMS_RR', (NVME_CC_AMS_WRRU:=2048): 'NVME_CC_AMS_WRRU', (NVME_CC_AMS_VS:=14336): 'NVME_CC_AMS_VS', (NVME_CC_SHN_SHIFT:=14): 'NVME_CC_SHN_SHIFT', (NVME_CC_SHN_MASK:=49152): 'NVME_CC_SHN_MASK', (NVME_CC_SHN_NONE:=0): 'NVME_CC_SHN_NONE', (NVME_CC_SHN_NORMAL:=16384): 'NVME_CC_SHN_NORMAL', (NVME_CC_SHN_ABRUPT:=32768): 'NVME_CC_SHN_ABRUPT', (NVME_CC_IOSQES_SHIFT:=16): 'NVME_CC_IOSQES_SHIFT', (NVME_CC_IOSQES_MASK:=983040): 'NVME_CC_IOSQES_MASK', (NVME_CC_IOSQES:=393216): 'NVME_CC_IOSQES', (NVME_CC_IOCQES_SHIFT:=20): 'NVME_CC_IOCQES_SHIFT', (NVME_CC_IOCQES_MASK:=15728640): 'NVME_CC_IOCQES_MASK', (NVME_CC_IOCQES:=4194304): 'NVME_CC_IOCQES', (NVME_CC_CRIME:=16777216): 'NVME_CC_CRIME'}
_anonenum10: dict[int, str] = {(NVME_CSTS_RDY:=1): 'NVME_CSTS_RDY', (NVME_CSTS_CFS:=2): 'NVME_CSTS_CFS', (NVME_CSTS_NSSRO:=16): 'NVME_CSTS_NSSRO', (NVME_CSTS_PP:=32): 'NVME_CSTS_PP', (NVME_CSTS_SHST_NORMAL:=0): 'NVME_CSTS_SHST_NORMAL', (NVME_CSTS_SHST_OCCUR:=4): 'NVME_CSTS_SHST_OCCUR', (NVME_CSTS_SHST_CMPLT:=8): 'NVME_CSTS_SHST_CMPLT', (NVME_CSTS_SHST_MASK:=12): 'NVME_CSTS_SHST_MASK'}
_anonenum11: dict[int, str] = {(NVME_CMBMSC_CRE:=1): 'NVME_CMBMSC_CRE', (NVME_CMBMSC_CMSE:=2): 'NVME_CMBMSC_CMSE'}
_anonenum12: dict[int, str] = {(NVME_CAP_CSS_NVM:=1): 'NVME_CAP_CSS_NVM', (NVME_CAP_CSS_CSI:=64): 'NVME_CAP_CSS_CSI'}
_anonenum13: dict[int, str] = {(NVME_CAP_CRMS_CRWMS:=576460752303423488): 'NVME_CAP_CRMS_CRWMS', (NVME_CAP_CRMS_CRIMS:=1152921504606846976): 'NVME_CAP_CRMS_CRIMS'}
@c.record
class struct_nvme_id_power_state(c.Struct):
  SIZE = 32
  max_power: int
  rsvd2: int
  flags: int
  entry_lat: int
  exit_lat: int
  read_tput: int
  read_lat: int
  write_tput: int
  write_lat: int
  idle_power: int
  idle_scale: int
  rsvd19: int
  active_power: int
  active_work_scale: int
  rsvd23: c.Array[ctypes.c_ubyte, Literal[9]]
__le16: TypeAlias = ctypes.c_uint16
__u8: TypeAlias = ctypes.c_ubyte
__le32: TypeAlias = ctypes.c_uint32
struct_nvme_id_power_state.register_fields([('max_power', ctypes.c_uint16, 0), ('rsvd2', ctypes.c_ubyte, 2), ('flags', ctypes.c_ubyte, 3), ('entry_lat', ctypes.c_uint32, 4), ('exit_lat', ctypes.c_uint32, 8), ('read_tput', ctypes.c_ubyte, 12), ('read_lat', ctypes.c_ubyte, 13), ('write_tput', ctypes.c_ubyte, 14), ('write_lat', ctypes.c_ubyte, 15), ('idle_power', ctypes.c_uint16, 16), ('idle_scale', ctypes.c_ubyte, 18), ('rsvd19', ctypes.c_ubyte, 19), ('active_power', ctypes.c_uint16, 20), ('active_work_scale', ctypes.c_ubyte, 22), ('rsvd23', c.Array[ctypes.c_ubyte, Literal[9]], 23)])
_anonenum14: dict[int, str] = {(NVME_PS_FLAGS_MAX_POWER_SCALE:=1): 'NVME_PS_FLAGS_MAX_POWER_SCALE', (NVME_PS_FLAGS_NON_OP_STATE:=2): 'NVME_PS_FLAGS_NON_OP_STATE'}
enum_nvme_ctrl_attr: dict[int, str] = {(NVME_CTRL_ATTR_HID_128_BIT:=1): 'NVME_CTRL_ATTR_HID_128_BIT', (NVME_CTRL_ATTR_TBKAS:=64): 'NVME_CTRL_ATTR_TBKAS', (NVME_CTRL_ATTR_ELBAS:=32768): 'NVME_CTRL_ATTR_ELBAS', (NVME_CTRL_ATTR_RHII:=262144): 'NVME_CTRL_ATTR_RHII', (NVME_CTRL_ATTR_FDPS:=524288): 'NVME_CTRL_ATTR_FDPS'}
@c.record
class struct_nvme_id_ctrl(c.Struct):
  SIZE = 4096
  vid: int
  ssvid: int
  sn: c.Array[ctypes.c_char, Literal[20]]
  mn: c.Array[ctypes.c_char, Literal[40]]
  fr: c.Array[ctypes.c_char, Literal[8]]
  rab: int
  ieee: c.Array[ctypes.c_ubyte, Literal[3]]
  cmic: int
  mdts: int
  cntlid: int
  ver: int
  rtd3r: int
  rtd3e: int
  oaes: int
  ctratt: int
  rsvd100: c.Array[ctypes.c_ubyte, Literal[11]]
  cntrltype: int
  fguid: c.Array[ctypes.c_ubyte, Literal[16]]
  crdt1: int
  crdt2: int
  crdt3: int
  rsvd134: c.Array[ctypes.c_ubyte, Literal[122]]
  oacs: int
  acl: int
  aerl: int
  frmw: int
  lpa: int
  elpe: int
  npss: int
  avscc: int
  apsta: int
  wctemp: int
  cctemp: int
  mtfa: int
  hmpre: int
  hmmin: int
  tnvmcap: c.Array[ctypes.c_ubyte, Literal[16]]
  unvmcap: c.Array[ctypes.c_ubyte, Literal[16]]
  rpmbs: int
  edstt: int
  dsto: int
  fwug: int
  kas: int
  hctma: int
  mntmt: int
  mxtmt: int
  sanicap: int
  hmminds: int
  hmmaxd: int
  nvmsetidmax: int
  endgidmax: int
  anatt: int
  anacap: int
  anagrpmax: int
  nanagrpid: int
  rsvd352: c.Array[ctypes.c_ubyte, Literal[160]]
  sqes: int
  cqes: int
  maxcmd: int
  nn: int
  oncs: int
  fuses: int
  fna: int
  vwc: int
  awun: int
  awupf: int
  nvscc: int
  nwpc: int
  acwu: int
  rsvd534: c.Array[ctypes.c_ubyte, Literal[2]]
  sgls: int
  mnan: int
  rsvd544: c.Array[ctypes.c_ubyte, Literal[224]]
  subnqn: c.Array[ctypes.c_char, Literal[256]]
  rsvd1024: c.Array[ctypes.c_ubyte, Literal[768]]
  ioccsz: int
  iorcsz: int
  icdoff: int
  ctrattr: int
  msdbd: int
  rsvd1804: c.Array[ctypes.c_ubyte, Literal[2]]
  dctype: int
  rsvd1807: c.Array[ctypes.c_ubyte, Literal[241]]
  psd: c.Array[struct_nvme_id_power_state, Literal[32]]
  vs: c.Array[ctypes.c_ubyte, Literal[1024]]
struct_nvme_id_ctrl.register_fields([('vid', ctypes.c_uint16, 0), ('ssvid', ctypes.c_uint16, 2), ('sn', c.Array[ctypes.c_char, Literal[20]], 4), ('mn', c.Array[ctypes.c_char, Literal[40]], 24), ('fr', c.Array[ctypes.c_char, Literal[8]], 64), ('rab', ctypes.c_ubyte, 72), ('ieee', c.Array[ctypes.c_ubyte, Literal[3]], 73), ('cmic', ctypes.c_ubyte, 76), ('mdts', ctypes.c_ubyte, 77), ('cntlid', ctypes.c_uint16, 78), ('ver', ctypes.c_uint32, 80), ('rtd3r', ctypes.c_uint32, 84), ('rtd3e', ctypes.c_uint32, 88), ('oaes', ctypes.c_uint32, 92), ('ctratt', ctypes.c_uint32, 96), ('rsvd100', c.Array[ctypes.c_ubyte, Literal[11]], 100), ('cntrltype', ctypes.c_ubyte, 111), ('fguid', c.Array[ctypes.c_ubyte, Literal[16]], 112), ('crdt1', ctypes.c_uint16, 128), ('crdt2', ctypes.c_uint16, 130), ('crdt3', ctypes.c_uint16, 132), ('rsvd134', c.Array[ctypes.c_ubyte, Literal[122]], 134), ('oacs', ctypes.c_uint16, 256), ('acl', ctypes.c_ubyte, 258), ('aerl', ctypes.c_ubyte, 259), ('frmw', ctypes.c_ubyte, 260), ('lpa', ctypes.c_ubyte, 261), ('elpe', ctypes.c_ubyte, 262), ('npss', ctypes.c_ubyte, 263), ('avscc', ctypes.c_ubyte, 264), ('apsta', ctypes.c_ubyte, 265), ('wctemp', ctypes.c_uint16, 266), ('cctemp', ctypes.c_uint16, 268), ('mtfa', ctypes.c_uint16, 270), ('hmpre', ctypes.c_uint32, 272), ('hmmin', ctypes.c_uint32, 276), ('tnvmcap', c.Array[ctypes.c_ubyte, Literal[16]], 280), ('unvmcap', c.Array[ctypes.c_ubyte, Literal[16]], 296), ('rpmbs', ctypes.c_uint32, 312), ('edstt', ctypes.c_uint16, 316), ('dsto', ctypes.c_ubyte, 318), ('fwug', ctypes.c_ubyte, 319), ('kas', ctypes.c_uint16, 320), ('hctma', ctypes.c_uint16, 322), ('mntmt', ctypes.c_uint16, 324), ('mxtmt', ctypes.c_uint16, 326), ('sanicap', ctypes.c_uint32, 328), ('hmminds', ctypes.c_uint32, 332), ('hmmaxd', ctypes.c_uint16, 336), ('nvmsetidmax', ctypes.c_uint16, 338), ('endgidmax', ctypes.c_uint16, 340), ('anatt', ctypes.c_ubyte, 342), ('anacap', ctypes.c_ubyte, 343), ('anagrpmax', ctypes.c_uint32, 344), ('nanagrpid', ctypes.c_uint32, 348), ('rsvd352', c.Array[ctypes.c_ubyte, Literal[160]], 352), ('sqes', ctypes.c_ubyte, 512), ('cqes', ctypes.c_ubyte, 513), ('maxcmd', ctypes.c_uint16, 514), ('nn', ctypes.c_uint32, 516), ('oncs', ctypes.c_uint16, 520), ('fuses', ctypes.c_uint16, 522), ('fna', ctypes.c_ubyte, 524), ('vwc', ctypes.c_ubyte, 525), ('awun', ctypes.c_uint16, 526), ('awupf', ctypes.c_uint16, 528), ('nvscc', ctypes.c_ubyte, 530), ('nwpc', ctypes.c_ubyte, 531), ('acwu', ctypes.c_uint16, 532), ('rsvd534', c.Array[ctypes.c_ubyte, Literal[2]], 534), ('sgls', ctypes.c_uint32, 536), ('mnan', ctypes.c_uint32, 540), ('rsvd544', c.Array[ctypes.c_ubyte, Literal[224]], 544), ('subnqn', c.Array[ctypes.c_char, Literal[256]], 768), ('rsvd1024', c.Array[ctypes.c_ubyte, Literal[768]], 1024), ('ioccsz', ctypes.c_uint32, 1792), ('iorcsz', ctypes.c_uint32, 1796), ('icdoff', ctypes.c_uint16, 1800), ('ctrattr', ctypes.c_ubyte, 1802), ('msdbd', ctypes.c_ubyte, 1803), ('rsvd1804', c.Array[ctypes.c_ubyte, Literal[2]], 1804), ('dctype', ctypes.c_ubyte, 1806), ('rsvd1807', c.Array[ctypes.c_ubyte, Literal[241]], 1807), ('psd', c.Array[struct_nvme_id_power_state, Literal[32]], 2048), ('vs', c.Array[ctypes.c_ubyte, Literal[1024]], 3072)])
_anonenum15: dict[int, str] = {(NVME_CTRL_CMIC_MULTI_PORT:=1): 'NVME_CTRL_CMIC_MULTI_PORT', (NVME_CTRL_CMIC_MULTI_CTRL:=2): 'NVME_CTRL_CMIC_MULTI_CTRL', (NVME_CTRL_CMIC_ANA:=8): 'NVME_CTRL_CMIC_ANA', (NVME_CTRL_ONCS_COMPARE:=1): 'NVME_CTRL_ONCS_COMPARE', (NVME_CTRL_ONCS_WRITE_UNCORRECTABLE:=2): 'NVME_CTRL_ONCS_WRITE_UNCORRECTABLE', (NVME_CTRL_ONCS_DSM:=4): 'NVME_CTRL_ONCS_DSM', (NVME_CTRL_ONCS_WRITE_ZEROES:=8): 'NVME_CTRL_ONCS_WRITE_ZEROES', (NVME_CTRL_ONCS_RESERVATIONS:=32): 'NVME_CTRL_ONCS_RESERVATIONS', (NVME_CTRL_ONCS_TIMESTAMP:=64): 'NVME_CTRL_ONCS_TIMESTAMP', (NVME_CTRL_VWC_PRESENT:=1): 'NVME_CTRL_VWC_PRESENT', (NVME_CTRL_OACS_SEC_SUPP:=1): 'NVME_CTRL_OACS_SEC_SUPP', (NVME_CTRL_OACS_NS_MNGT_SUPP:=8): 'NVME_CTRL_OACS_NS_MNGT_SUPP', (NVME_CTRL_OACS_DIRECTIVES:=32): 'NVME_CTRL_OACS_DIRECTIVES', (NVME_CTRL_OACS_DBBUF_SUPP:=256): 'NVME_CTRL_OACS_DBBUF_SUPP', (NVME_CTRL_LPA_CMD_EFFECTS_LOG:=2): 'NVME_CTRL_LPA_CMD_EFFECTS_LOG', (NVME_CTRL_CTRATT_128_ID:=1): 'NVME_CTRL_CTRATT_128_ID', (NVME_CTRL_CTRATT_NON_OP_PSP:=2): 'NVME_CTRL_CTRATT_NON_OP_PSP', (NVME_CTRL_CTRATT_NVM_SETS:=4): 'NVME_CTRL_CTRATT_NVM_SETS', (NVME_CTRL_CTRATT_READ_RECV_LVLS:=8): 'NVME_CTRL_CTRATT_READ_RECV_LVLS', (NVME_CTRL_CTRATT_ENDURANCE_GROUPS:=16): 'NVME_CTRL_CTRATT_ENDURANCE_GROUPS', (NVME_CTRL_CTRATT_PREDICTABLE_LAT:=32): 'NVME_CTRL_CTRATT_PREDICTABLE_LAT', (NVME_CTRL_CTRATT_NAMESPACE_GRANULARITY:=128): 'NVME_CTRL_CTRATT_NAMESPACE_GRANULARITY', (NVME_CTRL_CTRATT_UUID_LIST:=512): 'NVME_CTRL_CTRATT_UUID_LIST', (NVME_CTRL_SGLS_BYTE_ALIGNED:=1): 'NVME_CTRL_SGLS_BYTE_ALIGNED', (NVME_CTRL_SGLS_DWORD_ALIGNED:=2): 'NVME_CTRL_SGLS_DWORD_ALIGNED', (NVME_CTRL_SGLS_KSDBDS:=4): 'NVME_CTRL_SGLS_KSDBDS', (NVME_CTRL_SGLS_MSDS:=524288): 'NVME_CTRL_SGLS_MSDS', (NVME_CTRL_SGLS_SAOS:=1048576): 'NVME_CTRL_SGLS_SAOS'}
@c.record
class struct_nvme_lbaf(c.Struct):
  SIZE = 4
  ms: int
  ds: int
  rp: int
struct_nvme_lbaf.register_fields([('ms', ctypes.c_uint16, 0), ('ds', ctypes.c_ubyte, 2), ('rp', ctypes.c_ubyte, 3)])
@c.record
class struct_nvme_id_ns(c.Struct):
  SIZE = 4096
  nsze: int
  ncap: int
  nuse: int
  nsfeat: int
  nlbaf: int
  flbas: int
  mc: int
  dpc: int
  dps: int
  nmic: int
  rescap: int
  fpi: int
  dlfeat: int
  nawun: int
  nawupf: int
  nacwu: int
  nabsn: int
  nabo: int
  nabspf: int
  noiob: int
  nvmcap: c.Array[ctypes.c_ubyte, Literal[16]]
  npwg: int
  npwa: int
  npdg: int
  npda: int
  nows: int
  rsvd74: c.Array[ctypes.c_ubyte, Literal[18]]
  anagrpid: int
  rsvd96: c.Array[ctypes.c_ubyte, Literal[3]]
  nsattr: int
  nvmsetid: int
  endgid: int
  nguid: c.Array[ctypes.c_ubyte, Literal[16]]
  eui64: c.Array[ctypes.c_ubyte, Literal[8]]
  lbaf: c.Array[struct_nvme_lbaf, Literal[64]]
  vs: c.Array[ctypes.c_ubyte, Literal[3712]]
__le64: TypeAlias = ctypes.c_uint64
struct_nvme_id_ns.register_fields([('nsze', ctypes.c_uint64, 0), ('ncap', ctypes.c_uint64, 8), ('nuse', ctypes.c_uint64, 16), ('nsfeat', ctypes.c_ubyte, 24), ('nlbaf', ctypes.c_ubyte, 25), ('flbas', ctypes.c_ubyte, 26), ('mc', ctypes.c_ubyte, 27), ('dpc', ctypes.c_ubyte, 28), ('dps', ctypes.c_ubyte, 29), ('nmic', ctypes.c_ubyte, 30), ('rescap', ctypes.c_ubyte, 31), ('fpi', ctypes.c_ubyte, 32), ('dlfeat', ctypes.c_ubyte, 33), ('nawun', ctypes.c_uint16, 34), ('nawupf', ctypes.c_uint16, 36), ('nacwu', ctypes.c_uint16, 38), ('nabsn', ctypes.c_uint16, 40), ('nabo', ctypes.c_uint16, 42), ('nabspf', ctypes.c_uint16, 44), ('noiob', ctypes.c_uint16, 46), ('nvmcap', c.Array[ctypes.c_ubyte, Literal[16]], 48), ('npwg', ctypes.c_uint16, 64), ('npwa', ctypes.c_uint16, 66), ('npdg', ctypes.c_uint16, 68), ('npda', ctypes.c_uint16, 70), ('nows', ctypes.c_uint16, 72), ('rsvd74', c.Array[ctypes.c_ubyte, Literal[18]], 74), ('anagrpid', ctypes.c_uint32, 92), ('rsvd96', c.Array[ctypes.c_ubyte, Literal[3]], 96), ('nsattr', ctypes.c_ubyte, 99), ('nvmsetid', ctypes.c_uint16, 100), ('endgid', ctypes.c_uint16, 102), ('nguid', c.Array[ctypes.c_ubyte, Literal[16]], 104), ('eui64', c.Array[ctypes.c_ubyte, Literal[8]], 120), ('lbaf', c.Array[struct_nvme_lbaf, Literal[64]], 128), ('vs', c.Array[ctypes.c_ubyte, Literal[3712]], 384)])
@c.record
class struct_nvme_id_ns_cs_indep(c.Struct):
  SIZE = 4096
  nsfeat: int
  nmic: int
  rescap: int
  fpi: int
  anagrpid: int
  nsattr: int
  rsvd9: int
  nvmsetid: int
  endgid: int
  nstat: int
  rsvd15: c.Array[ctypes.c_ubyte, Literal[4081]]
struct_nvme_id_ns_cs_indep.register_fields([('nsfeat', ctypes.c_ubyte, 0), ('nmic', ctypes.c_ubyte, 1), ('rescap', ctypes.c_ubyte, 2), ('fpi', ctypes.c_ubyte, 3), ('anagrpid', ctypes.c_uint32, 4), ('nsattr', ctypes.c_ubyte, 8), ('rsvd9', ctypes.c_ubyte, 9), ('nvmsetid', ctypes.c_uint16, 10), ('endgid', ctypes.c_uint16, 12), ('nstat', ctypes.c_ubyte, 14), ('rsvd15', c.Array[ctypes.c_ubyte, Literal[4081]], 15)])
@c.record
class struct_nvme_zns_lbafe(c.Struct):
  SIZE = 16
  zsze: int
  zdes: int
  rsvd9: c.Array[ctypes.c_ubyte, Literal[7]]
struct_nvme_zns_lbafe.register_fields([('zsze', ctypes.c_uint64, 0), ('zdes', ctypes.c_ubyte, 8), ('rsvd9', c.Array[ctypes.c_ubyte, Literal[7]], 9)])
@c.record
class struct_nvme_id_ns_zns(c.Struct):
  SIZE = 4096
  zoc: int
  ozcs: int
  mar: int
  mor: int
  rrl: int
  frl: int
  rsvd20: c.Array[ctypes.c_ubyte, Literal[2796]]
  lbafe: c.Array[struct_nvme_zns_lbafe, Literal[64]]
  vs: c.Array[ctypes.c_ubyte, Literal[256]]
struct_nvme_id_ns_zns.register_fields([('zoc', ctypes.c_uint16, 0), ('ozcs', ctypes.c_uint16, 2), ('mar', ctypes.c_uint32, 4), ('mor', ctypes.c_uint32, 8), ('rrl', ctypes.c_uint32, 12), ('frl', ctypes.c_uint32, 16), ('rsvd20', c.Array[ctypes.c_ubyte, Literal[2796]], 20), ('lbafe', c.Array[struct_nvme_zns_lbafe, Literal[64]], 2816), ('vs', c.Array[ctypes.c_ubyte, Literal[256]], 3840)])
@c.record
class struct_nvme_id_ctrl_zns(c.Struct):
  SIZE = 4096
  zasl: int
  rsvd1: c.Array[ctypes.c_ubyte, Literal[4095]]
struct_nvme_id_ctrl_zns.register_fields([('zasl', ctypes.c_ubyte, 0), ('rsvd1', c.Array[ctypes.c_ubyte, Literal[4095]], 1)])
@c.record
class struct_nvme_id_ns_nvm(c.Struct):
  SIZE = 4096
  lbstm: int
  pic: int
  rsvd9: c.Array[ctypes.c_ubyte, Literal[3]]
  elbaf: c.Array[ctypes.c_uint32, Literal[64]]
  rsvd268: c.Array[ctypes.c_ubyte, Literal[3828]]
struct_nvme_id_ns_nvm.register_fields([('lbstm', ctypes.c_uint64, 0), ('pic', ctypes.c_ubyte, 8), ('rsvd9', c.Array[ctypes.c_ubyte, Literal[3]], 9), ('elbaf', c.Array[ctypes.c_uint32, Literal[64]], 12), ('rsvd268', c.Array[ctypes.c_ubyte, Literal[3828]], 268)])
_anonenum16: dict[int, str] = {(NVME_ID_NS_NVM_STS_MASK:=127): 'NVME_ID_NS_NVM_STS_MASK', (NVME_ID_NS_NVM_GUARD_SHIFT:=7): 'NVME_ID_NS_NVM_GUARD_SHIFT', (NVME_ID_NS_NVM_GUARD_MASK:=3): 'NVME_ID_NS_NVM_GUARD_MASK', (NVME_ID_NS_NVM_QPIF_SHIFT:=9): 'NVME_ID_NS_NVM_QPIF_SHIFT', (NVME_ID_NS_NVM_QPIF_MASK:=15): 'NVME_ID_NS_NVM_QPIF_MASK', (NVME_ID_NS_NVM_QPIFS:=8): 'NVME_ID_NS_NVM_QPIFS'}
@c.record
class struct_nvme_id_ctrl_nvm(c.Struct):
  SIZE = 4096
  vsl: int
  wzsl: int
  wusl: int
  dmrl: int
  dmrsl: int
  dmsl: int
  rsvd16: c.Array[ctypes.c_ubyte, Literal[4080]]
struct_nvme_id_ctrl_nvm.register_fields([('vsl', ctypes.c_ubyte, 0), ('wzsl', ctypes.c_ubyte, 1), ('wusl', ctypes.c_ubyte, 2), ('dmrl', ctypes.c_ubyte, 3), ('dmrsl', ctypes.c_uint32, 4), ('dmsl', ctypes.c_uint64, 8), ('rsvd16', c.Array[ctypes.c_ubyte, Literal[4080]], 16)])
_anonenum17: dict[int, str] = {(NVME_ID_CNS_NS:=0): 'NVME_ID_CNS_NS', (NVME_ID_CNS_CTRL:=1): 'NVME_ID_CNS_CTRL', (NVME_ID_CNS_NS_ACTIVE_LIST:=2): 'NVME_ID_CNS_NS_ACTIVE_LIST', (NVME_ID_CNS_NS_DESC_LIST:=3): 'NVME_ID_CNS_NS_DESC_LIST', (NVME_ID_CNS_CS_NS:=5): 'NVME_ID_CNS_CS_NS', (NVME_ID_CNS_CS_CTRL:=6): 'NVME_ID_CNS_CS_CTRL', (NVME_ID_CNS_NS_ACTIVE_LIST_CS:=7): 'NVME_ID_CNS_NS_ACTIVE_LIST_CS', (NVME_ID_CNS_NS_CS_INDEP:=8): 'NVME_ID_CNS_NS_CS_INDEP', (NVME_ID_CNS_NS_PRESENT_LIST:=16): 'NVME_ID_CNS_NS_PRESENT_LIST', (NVME_ID_CNS_NS_PRESENT:=17): 'NVME_ID_CNS_NS_PRESENT', (NVME_ID_CNS_CTRL_NS_LIST:=18): 'NVME_ID_CNS_CTRL_NS_LIST', (NVME_ID_CNS_CTRL_LIST:=19): 'NVME_ID_CNS_CTRL_LIST', (NVME_ID_CNS_SCNDRY_CTRL_LIST:=21): 'NVME_ID_CNS_SCNDRY_CTRL_LIST', (NVME_ID_CNS_NS_GRANULARITY:=22): 'NVME_ID_CNS_NS_GRANULARITY', (NVME_ID_CNS_UUID_LIST:=23): 'NVME_ID_CNS_UUID_LIST', (NVME_ID_CNS_ENDGRP_LIST:=25): 'NVME_ID_CNS_ENDGRP_LIST'}
_anonenum18: dict[int, str] = {(NVME_CSI_NVM:=0): 'NVME_CSI_NVM', (NVME_CSI_ZNS:=2): 'NVME_CSI_ZNS'}
_anonenum19: dict[int, str] = {(NVME_DIR_IDENTIFY:=0): 'NVME_DIR_IDENTIFY', (NVME_DIR_STREAMS:=1): 'NVME_DIR_STREAMS', (NVME_DIR_SND_ID_OP_ENABLE:=1): 'NVME_DIR_SND_ID_OP_ENABLE', (NVME_DIR_SND_ST_OP_REL_ID:=1): 'NVME_DIR_SND_ST_OP_REL_ID', (NVME_DIR_SND_ST_OP_REL_RSC:=2): 'NVME_DIR_SND_ST_OP_REL_RSC', (NVME_DIR_RCV_ID_OP_PARAM:=1): 'NVME_DIR_RCV_ID_OP_PARAM', (NVME_DIR_RCV_ST_OP_PARAM:=1): 'NVME_DIR_RCV_ST_OP_PARAM', (NVME_DIR_RCV_ST_OP_STATUS:=2): 'NVME_DIR_RCV_ST_OP_STATUS', (NVME_DIR_RCV_ST_OP_RESOURCE:=3): 'NVME_DIR_RCV_ST_OP_RESOURCE', (NVME_DIR_ENDIR:=1): 'NVME_DIR_ENDIR'}
_anonenum20: dict[int, str] = {(NVME_NS_FEAT_THIN:=1): 'NVME_NS_FEAT_THIN', (NVME_NS_FEAT_ATOMICS:=2): 'NVME_NS_FEAT_ATOMICS', (NVME_NS_FEAT_IO_OPT:=16): 'NVME_NS_FEAT_IO_OPT', (NVME_NS_ATTR_RO:=1): 'NVME_NS_ATTR_RO', (NVME_NS_FLBAS_LBA_MASK:=15): 'NVME_NS_FLBAS_LBA_MASK', (NVME_NS_FLBAS_LBA_UMASK:=96): 'NVME_NS_FLBAS_LBA_UMASK', (NVME_NS_FLBAS_LBA_SHIFT:=1): 'NVME_NS_FLBAS_LBA_SHIFT', (NVME_NS_FLBAS_META_EXT:=16): 'NVME_NS_FLBAS_META_EXT', (NVME_NS_NMIC_SHARED:=1): 'NVME_NS_NMIC_SHARED', (NVME_NS_ROTATIONAL:=16): 'NVME_NS_ROTATIONAL', (NVME_NS_VWC_NOT_PRESENT:=32): 'NVME_NS_VWC_NOT_PRESENT', (NVME_LBAF_RP_BEST:=0): 'NVME_LBAF_RP_BEST', (NVME_LBAF_RP_BETTER:=1): 'NVME_LBAF_RP_BETTER', (NVME_LBAF_RP_GOOD:=2): 'NVME_LBAF_RP_GOOD', (NVME_LBAF_RP_DEGRADED:=3): 'NVME_LBAF_RP_DEGRADED', (NVME_NS_DPC_PI_LAST:=16): 'NVME_NS_DPC_PI_LAST', (NVME_NS_DPC_PI_FIRST:=8): 'NVME_NS_DPC_PI_FIRST', (NVME_NS_DPC_PI_TYPE3:=4): 'NVME_NS_DPC_PI_TYPE3', (NVME_NS_DPC_PI_TYPE2:=2): 'NVME_NS_DPC_PI_TYPE2', (NVME_NS_DPC_PI_TYPE1:=1): 'NVME_NS_DPC_PI_TYPE1', (NVME_NS_DPS_PI_FIRST:=8): 'NVME_NS_DPS_PI_FIRST', (NVME_NS_DPS_PI_MASK:=7): 'NVME_NS_DPS_PI_MASK', (NVME_NS_DPS_PI_TYPE1:=1): 'NVME_NS_DPS_PI_TYPE1', (NVME_NS_DPS_PI_TYPE2:=2): 'NVME_NS_DPS_PI_TYPE2', (NVME_NS_DPS_PI_TYPE3:=3): 'NVME_NS_DPS_PI_TYPE3'}
_anonenum21: dict[int, str] = {(NVME_NSTAT_NRDY:=1): 'NVME_NSTAT_NRDY'}
_anonenum22: dict[int, str] = {(NVME_NVM_NS_16B_GUARD:=0): 'NVME_NVM_NS_16B_GUARD', (NVME_NVM_NS_32B_GUARD:=1): 'NVME_NVM_NS_32B_GUARD', (NVME_NVM_NS_64B_GUARD:=2): 'NVME_NVM_NS_64B_GUARD', (NVME_NVM_NS_QTYPE_GUARD:=3): 'NVME_NVM_NS_QTYPE_GUARD'}
_anonenum23: dict[int, str] = {(NVME_MC_EXTENDED_LBA:=1): 'NVME_MC_EXTENDED_LBA', (NVME_MC_METADATA_PTR:=2): 'NVME_MC_METADATA_PTR'}
@c.record
class struct_nvme_ns_id_desc(c.Struct):
  SIZE = 4
  nidt: int
  nidl: int
  reserved: int
struct_nvme_ns_id_desc.register_fields([('nidt', ctypes.c_ubyte, 0), ('nidl', ctypes.c_ubyte, 1), ('reserved', ctypes.c_uint16, 2)])
_anonenum24: dict[int, str] = {(NVME_NIDT_EUI64:=1): 'NVME_NIDT_EUI64', (NVME_NIDT_NGUID:=2): 'NVME_NIDT_NGUID', (NVME_NIDT_UUID:=3): 'NVME_NIDT_UUID', (NVME_NIDT_CSI:=4): 'NVME_NIDT_CSI'}
@c.record
class struct_nvme_endurance_group_log(c.Struct):
  SIZE = 512
  egcw: int
  egfeat: int
  rsvd2: int
  avsp: int
  avspt: int
  pused: int
  did: int
  rsvd8: c.Array[ctypes.c_ubyte, Literal[24]]
  ee: c.Array[ctypes.c_ubyte, Literal[16]]
  dur: c.Array[ctypes.c_ubyte, Literal[16]]
  duw: c.Array[ctypes.c_ubyte, Literal[16]]
  muw: c.Array[ctypes.c_ubyte, Literal[16]]
  hrc: c.Array[ctypes.c_ubyte, Literal[16]]
  hwc: c.Array[ctypes.c_ubyte, Literal[16]]
  mdie: c.Array[ctypes.c_ubyte, Literal[16]]
  neile: c.Array[ctypes.c_ubyte, Literal[16]]
  tegcap: c.Array[ctypes.c_ubyte, Literal[16]]
  uegcap: c.Array[ctypes.c_ubyte, Literal[16]]
  rsvd192: c.Array[ctypes.c_ubyte, Literal[320]]
struct_nvme_endurance_group_log.register_fields([('egcw', ctypes.c_ubyte, 0), ('egfeat', ctypes.c_ubyte, 1), ('rsvd2', ctypes.c_ubyte, 2), ('avsp', ctypes.c_ubyte, 3), ('avspt', ctypes.c_ubyte, 4), ('pused', ctypes.c_ubyte, 5), ('did', ctypes.c_uint16, 6), ('rsvd8', c.Array[ctypes.c_ubyte, Literal[24]], 8), ('ee', c.Array[ctypes.c_ubyte, Literal[16]], 32), ('dur', c.Array[ctypes.c_ubyte, Literal[16]], 48), ('duw', c.Array[ctypes.c_ubyte, Literal[16]], 64), ('muw', c.Array[ctypes.c_ubyte, Literal[16]], 80), ('hrc', c.Array[ctypes.c_ubyte, Literal[16]], 96), ('hwc', c.Array[ctypes.c_ubyte, Literal[16]], 112), ('mdie', c.Array[ctypes.c_ubyte, Literal[16]], 128), ('neile', c.Array[ctypes.c_ubyte, Literal[16]], 144), ('tegcap', c.Array[ctypes.c_ubyte, Literal[16]], 160), ('uegcap', c.Array[ctypes.c_ubyte, Literal[16]], 176), ('rsvd192', c.Array[ctypes.c_ubyte, Literal[320]], 192)])
@c.record
class struct_nvme_rotational_media_log(c.Struct):
  SIZE = 512
  endgid: int
  numa: int
  nrs: int
  rsvd6: c.Array[ctypes.c_ubyte, Literal[2]]
  spinc: int
  fspinc: int
  ldc: int
  fldc: int
  rsvd24: c.Array[ctypes.c_ubyte, Literal[488]]
struct_nvme_rotational_media_log.register_fields([('endgid', ctypes.c_uint16, 0), ('numa', ctypes.c_uint16, 2), ('nrs', ctypes.c_uint16, 4), ('rsvd6', c.Array[ctypes.c_ubyte, Literal[2]], 6), ('spinc', ctypes.c_uint32, 8), ('fspinc', ctypes.c_uint32, 12), ('ldc', ctypes.c_uint32, 16), ('fldc', ctypes.c_uint32, 20), ('rsvd24', c.Array[ctypes.c_ubyte, Literal[488]], 24)])
@c.record
class struct_nvme_fdp_config(c.Struct):
  SIZE = 4
  flags: int
  fdpcidx: int
  reserved: int
struct_nvme_fdp_config.register_fields([('flags', ctypes.c_ubyte, 0), ('fdpcidx', ctypes.c_ubyte, 1), ('reserved', ctypes.c_uint16, 2)])
@c.record
class struct_nvme_fdp_ruh_desc(c.Struct):
  SIZE = 4
  ruht: int
  reserved: c.Array[ctypes.c_ubyte, Literal[3]]
struct_nvme_fdp_ruh_desc.register_fields([('ruht', ctypes.c_ubyte, 0), ('reserved', c.Array[ctypes.c_ubyte, Literal[3]], 1)])
@c.record
class struct_nvme_fdp_config_desc(c.Struct):
  SIZE = 64
  dsze: int
  fdpa: int
  vss: int
  nrg: int
  nruh: int
  maxpids: int
  nns: int
  runs: int
  erutl: int
  rsvd28: c.Array[ctypes.c_ubyte, Literal[36]]
  ruhs: c.Array[struct_nvme_fdp_ruh_desc, Literal[0]]
struct_nvme_fdp_config_desc.register_fields([('dsze', ctypes.c_uint16, 0), ('fdpa', ctypes.c_ubyte, 2), ('vss', ctypes.c_ubyte, 3), ('nrg', ctypes.c_uint32, 4), ('nruh', ctypes.c_uint16, 8), ('maxpids', ctypes.c_uint16, 10), ('nns', ctypes.c_uint32, 12), ('runs', ctypes.c_uint64, 16), ('erutl', ctypes.c_uint32, 24), ('rsvd28', c.Array[ctypes.c_ubyte, Literal[36]], 28), ('ruhs', c.Array[struct_nvme_fdp_ruh_desc, Literal[0]], 64)])
@c.record
class struct_nvme_fdp_config_log(c.Struct):
  SIZE = 16
  numfdpc: int
  ver: int
  rsvd3: int
  sze: int
  rsvd8: c.Array[ctypes.c_ubyte, Literal[8]]
struct_nvme_fdp_config_log.register_fields([('numfdpc', ctypes.c_uint16, 0), ('ver', ctypes.c_ubyte, 2), ('rsvd3', ctypes.c_ubyte, 3), ('sze', ctypes.c_uint32, 4), ('rsvd8', c.Array[ctypes.c_ubyte, Literal[8]], 8)])
@c.record
class struct_nvme_smart_log(c.Struct):
  SIZE = 512
  critical_warning: int
  temperature: c.Array[ctypes.c_ubyte, Literal[2]]
  avail_spare: int
  spare_thresh: int
  percent_used: int
  endu_grp_crit_warn_sumry: int
  rsvd7: c.Array[ctypes.c_ubyte, Literal[25]]
  data_units_read: c.Array[ctypes.c_ubyte, Literal[16]]
  data_units_written: c.Array[ctypes.c_ubyte, Literal[16]]
  host_reads: c.Array[ctypes.c_ubyte, Literal[16]]
  host_writes: c.Array[ctypes.c_ubyte, Literal[16]]
  ctrl_busy_time: c.Array[ctypes.c_ubyte, Literal[16]]
  power_cycles: c.Array[ctypes.c_ubyte, Literal[16]]
  power_on_hours: c.Array[ctypes.c_ubyte, Literal[16]]
  unsafe_shutdowns: c.Array[ctypes.c_ubyte, Literal[16]]
  media_errors: c.Array[ctypes.c_ubyte, Literal[16]]
  num_err_log_entries: c.Array[ctypes.c_ubyte, Literal[16]]
  warning_temp_time: int
  critical_comp_time: int
  temp_sensor: c.Array[ctypes.c_uint16, Literal[8]]
  thm_temp1_trans_count: int
  thm_temp2_trans_count: int
  thm_temp1_total_time: int
  thm_temp2_total_time: int
  rsvd232: c.Array[ctypes.c_ubyte, Literal[280]]
struct_nvme_smart_log.register_fields([('critical_warning', ctypes.c_ubyte, 0), ('temperature', c.Array[ctypes.c_ubyte, Literal[2]], 1), ('avail_spare', ctypes.c_ubyte, 3), ('spare_thresh', ctypes.c_ubyte, 4), ('percent_used', ctypes.c_ubyte, 5), ('endu_grp_crit_warn_sumry', ctypes.c_ubyte, 6), ('rsvd7', c.Array[ctypes.c_ubyte, Literal[25]], 7), ('data_units_read', c.Array[ctypes.c_ubyte, Literal[16]], 32), ('data_units_written', c.Array[ctypes.c_ubyte, Literal[16]], 48), ('host_reads', c.Array[ctypes.c_ubyte, Literal[16]], 64), ('host_writes', c.Array[ctypes.c_ubyte, Literal[16]], 80), ('ctrl_busy_time', c.Array[ctypes.c_ubyte, Literal[16]], 96), ('power_cycles', c.Array[ctypes.c_ubyte, Literal[16]], 112), ('power_on_hours', c.Array[ctypes.c_ubyte, Literal[16]], 128), ('unsafe_shutdowns', c.Array[ctypes.c_ubyte, Literal[16]], 144), ('media_errors', c.Array[ctypes.c_ubyte, Literal[16]], 160), ('num_err_log_entries', c.Array[ctypes.c_ubyte, Literal[16]], 176), ('warning_temp_time', ctypes.c_uint32, 192), ('critical_comp_time', ctypes.c_uint32, 196), ('temp_sensor', c.Array[ctypes.c_uint16, Literal[8]], 200), ('thm_temp1_trans_count', ctypes.c_uint32, 216), ('thm_temp2_trans_count', ctypes.c_uint32, 220), ('thm_temp1_total_time', ctypes.c_uint32, 224), ('thm_temp2_total_time', ctypes.c_uint32, 228), ('rsvd232', c.Array[ctypes.c_ubyte, Literal[280]], 232)])
@c.record
class struct_nvme_fw_slot_info_log(c.Struct):
  SIZE = 512
  afi: int
  rsvd1: c.Array[ctypes.c_ubyte, Literal[7]]
  frs: c.Array[ctypes.c_uint64, Literal[7]]
  rsvd64: c.Array[ctypes.c_ubyte, Literal[448]]
struct_nvme_fw_slot_info_log.register_fields([('afi', ctypes.c_ubyte, 0), ('rsvd1', c.Array[ctypes.c_ubyte, Literal[7]], 1), ('frs', c.Array[ctypes.c_uint64, Literal[7]], 8), ('rsvd64', c.Array[ctypes.c_ubyte, Literal[448]], 64)])
_anonenum25: dict[int, str] = {(NVME_CMD_EFFECTS_CSUPP:=1): 'NVME_CMD_EFFECTS_CSUPP', (NVME_CMD_EFFECTS_LBCC:=2): 'NVME_CMD_EFFECTS_LBCC', (NVME_CMD_EFFECTS_NCC:=4): 'NVME_CMD_EFFECTS_NCC', (NVME_CMD_EFFECTS_NIC:=8): 'NVME_CMD_EFFECTS_NIC', (NVME_CMD_EFFECTS_CCC:=16): 'NVME_CMD_EFFECTS_CCC', (NVME_CMD_EFFECTS_CSER_MASK:=49152): 'NVME_CMD_EFFECTS_CSER_MASK', (NVME_CMD_EFFECTS_CSE_MASK:=458752): 'NVME_CMD_EFFECTS_CSE_MASK', (NVME_CMD_EFFECTS_UUID_SEL:=524288): 'NVME_CMD_EFFECTS_UUID_SEL', (NVME_CMD_EFFECTS_SCOPE_MASK:=4293918720): 'NVME_CMD_EFFECTS_SCOPE_MASK'}
@c.record
class struct_nvme_effects_log(c.Struct):
  SIZE = 4096
  acs: c.Array[ctypes.c_uint32, Literal[256]]
  iocs: c.Array[ctypes.c_uint32, Literal[256]]
  resv: c.Array[ctypes.c_ubyte, Literal[2048]]
struct_nvme_effects_log.register_fields([('acs', c.Array[ctypes.c_uint32, Literal[256]], 0), ('iocs', c.Array[ctypes.c_uint32, Literal[256]], 1024), ('resv', c.Array[ctypes.c_ubyte, Literal[2048]], 2048)])
enum_nvme_ana_state: dict[int, str] = {(NVME_ANA_OPTIMIZED:=1): 'NVME_ANA_OPTIMIZED', (NVME_ANA_NONOPTIMIZED:=2): 'NVME_ANA_NONOPTIMIZED', (NVME_ANA_INACCESSIBLE:=3): 'NVME_ANA_INACCESSIBLE', (NVME_ANA_PERSISTENT_LOSS:=4): 'NVME_ANA_PERSISTENT_LOSS', (NVME_ANA_CHANGE:=15): 'NVME_ANA_CHANGE'}
@c.record
class struct_nvme_ana_group_desc(c.Struct):
  SIZE = 32
  grpid: int
  nnsids: int
  chgcnt: int
  state: int
  rsvd17: c.Array[ctypes.c_ubyte, Literal[15]]
  nsids: c.Array[ctypes.c_uint32, Literal[0]]
struct_nvme_ana_group_desc.register_fields([('grpid', ctypes.c_uint32, 0), ('nnsids', ctypes.c_uint32, 4), ('chgcnt', ctypes.c_uint64, 8), ('state', ctypes.c_ubyte, 16), ('rsvd17', c.Array[ctypes.c_ubyte, Literal[15]], 17), ('nsids', c.Array[ctypes.c_uint32, Literal[0]], 32)])
@c.record
class struct_nvme_ana_rsp_hdr(c.Struct):
  SIZE = 16
  chgcnt: int
  ngrps: int
  rsvd10: c.Array[ctypes.c_uint16, Literal[3]]
struct_nvme_ana_rsp_hdr.register_fields([('chgcnt', ctypes.c_uint64, 0), ('ngrps', ctypes.c_uint16, 8), ('rsvd10', c.Array[ctypes.c_uint16, Literal[3]], 10)])
@c.record
class struct_nvme_zone_descriptor(c.Struct):
  SIZE = 64
  zt: int
  zs: int
  za: int
  rsvd3: c.Array[ctypes.c_ubyte, Literal[5]]
  zcap: int
  zslba: int
  wp: int
  rsvd32: c.Array[ctypes.c_ubyte, Literal[32]]
struct_nvme_zone_descriptor.register_fields([('zt', ctypes.c_ubyte, 0), ('zs', ctypes.c_ubyte, 1), ('za', ctypes.c_ubyte, 2), ('rsvd3', c.Array[ctypes.c_ubyte, Literal[5]], 3), ('zcap', ctypes.c_uint64, 8), ('zslba', ctypes.c_uint64, 16), ('wp', ctypes.c_uint64, 24), ('rsvd32', c.Array[ctypes.c_ubyte, Literal[32]], 32)])
_anonenum26: dict[int, str] = {(NVME_ZONE_TYPE_SEQWRITE_REQ:=2): 'NVME_ZONE_TYPE_SEQWRITE_REQ'}
@c.record
class struct_nvme_zone_report(c.Struct):
  SIZE = 64
  nr_zones: int
  resv8: c.Array[ctypes.c_ubyte, Literal[56]]
  entries: c.Array[struct_nvme_zone_descriptor, Literal[0]]
struct_nvme_zone_report.register_fields([('nr_zones', ctypes.c_uint64, 0), ('resv8', c.Array[ctypes.c_ubyte, Literal[56]], 8), ('entries', c.Array[struct_nvme_zone_descriptor, Literal[0]], 64)])
_anonenum27: dict[int, str] = {(NVME_SMART_CRIT_SPARE:=1): 'NVME_SMART_CRIT_SPARE', (NVME_SMART_CRIT_TEMPERATURE:=2): 'NVME_SMART_CRIT_TEMPERATURE', (NVME_SMART_CRIT_RELIABILITY:=4): 'NVME_SMART_CRIT_RELIABILITY', (NVME_SMART_CRIT_MEDIA:=8): 'NVME_SMART_CRIT_MEDIA', (NVME_SMART_CRIT_VOLATILE_MEMORY:=16): 'NVME_SMART_CRIT_VOLATILE_MEMORY'}
_anonenum28: dict[int, str] = {(NVME_AER_ERROR:=0): 'NVME_AER_ERROR', (NVME_AER_SMART:=1): 'NVME_AER_SMART', (NVME_AER_NOTICE:=2): 'NVME_AER_NOTICE', (NVME_AER_CSS:=6): 'NVME_AER_CSS', (NVME_AER_VS:=7): 'NVME_AER_VS'}
_anonenum29: dict[int, str] = {(NVME_AER_ERROR_PERSIST_INT_ERR:=3): 'NVME_AER_ERROR_PERSIST_INT_ERR'}
_anonenum30: dict[int, str] = {(NVME_AER_NOTICE_NS_CHANGED:=0): 'NVME_AER_NOTICE_NS_CHANGED', (NVME_AER_NOTICE_FW_ACT_STARTING:=1): 'NVME_AER_NOTICE_FW_ACT_STARTING', (NVME_AER_NOTICE_ANA:=3): 'NVME_AER_NOTICE_ANA', (NVME_AER_NOTICE_DISC_CHANGED:=240): 'NVME_AER_NOTICE_DISC_CHANGED'}
_anonenum31: dict[int, str] = {(NVME_AEN_BIT_NS_ATTR:=8): 'NVME_AEN_BIT_NS_ATTR', (NVME_AEN_BIT_FW_ACT:=9): 'NVME_AEN_BIT_FW_ACT', (NVME_AEN_BIT_ANA_CHANGE:=11): 'NVME_AEN_BIT_ANA_CHANGE', (NVME_AEN_BIT_DISC_CHANGE:=31): 'NVME_AEN_BIT_DISC_CHANGE'}
_anonenum32: dict[int, str] = {(NVME_AEN_CFG_NS_ATTR:=256): 'NVME_AEN_CFG_NS_ATTR', (NVME_AEN_CFG_FW_ACT:=512): 'NVME_AEN_CFG_FW_ACT', (NVME_AEN_CFG_ANA_CHANGE:=2048): 'NVME_AEN_CFG_ANA_CHANGE', (NVME_AEN_CFG_DISC_CHANGE:=-2147483648): 'NVME_AEN_CFG_DISC_CHANGE'}
@c.record
class struct_nvme_lba_range_type(c.Struct):
  SIZE = 64
  type: int
  attributes: int
  rsvd2: c.Array[ctypes.c_ubyte, Literal[14]]
  slba: int
  nlb: int
  guid: c.Array[ctypes.c_ubyte, Literal[16]]
  rsvd48: c.Array[ctypes.c_ubyte, Literal[16]]
struct_nvme_lba_range_type.register_fields([('type', ctypes.c_ubyte, 0), ('attributes', ctypes.c_ubyte, 1), ('rsvd2', c.Array[ctypes.c_ubyte, Literal[14]], 2), ('slba', ctypes.c_uint64, 16), ('nlb', ctypes.c_uint64, 24), ('guid', c.Array[ctypes.c_ubyte, Literal[16]], 32), ('rsvd48', c.Array[ctypes.c_ubyte, Literal[16]], 48)])
_anonenum33: dict[int, str] = {(NVME_LBART_TYPE_FS:=1): 'NVME_LBART_TYPE_FS', (NVME_LBART_TYPE_RAID:=2): 'NVME_LBART_TYPE_RAID', (NVME_LBART_TYPE_CACHE:=3): 'NVME_LBART_TYPE_CACHE', (NVME_LBART_TYPE_SWAP:=4): 'NVME_LBART_TYPE_SWAP', (NVME_LBART_ATTRIB_TEMP:=1): 'NVME_LBART_ATTRIB_TEMP', (NVME_LBART_ATTRIB_HIDE:=2): 'NVME_LBART_ATTRIB_HIDE'}
enum_nvme_pr_type: dict[int, str] = {(NVME_PR_WRITE_EXCLUSIVE:=1): 'NVME_PR_WRITE_EXCLUSIVE', (NVME_PR_EXCLUSIVE_ACCESS:=2): 'NVME_PR_EXCLUSIVE_ACCESS', (NVME_PR_WRITE_EXCLUSIVE_REG_ONLY:=3): 'NVME_PR_WRITE_EXCLUSIVE_REG_ONLY', (NVME_PR_EXCLUSIVE_ACCESS_REG_ONLY:=4): 'NVME_PR_EXCLUSIVE_ACCESS_REG_ONLY', (NVME_PR_WRITE_EXCLUSIVE_ALL_REGS:=5): 'NVME_PR_WRITE_EXCLUSIVE_ALL_REGS', (NVME_PR_EXCLUSIVE_ACCESS_ALL_REGS:=6): 'NVME_PR_EXCLUSIVE_ACCESS_ALL_REGS'}
enum_nvme_eds: dict[int, str] = {(NVME_EXTENDED_DATA_STRUCT:=1): 'NVME_EXTENDED_DATA_STRUCT'}
@c.record
class struct_nvme_registered_ctrl(c.Struct):
  SIZE = 24
  cntlid: int
  rcsts: int
  rsvd3: c.Array[ctypes.c_ubyte, Literal[5]]
  hostid: int
  rkey: int
struct_nvme_registered_ctrl.register_fields([('cntlid', ctypes.c_uint16, 0), ('rcsts', ctypes.c_ubyte, 2), ('rsvd3', c.Array[ctypes.c_ubyte, Literal[5]], 3), ('hostid', ctypes.c_uint64, 8), ('rkey', ctypes.c_uint64, 16)])
@c.record
class struct_nvme_reservation_status(c.Struct):
  SIZE = 24
  gen: int
  rtype: int
  regctl: c.Array[ctypes.c_ubyte, Literal[2]]
  resv5: c.Array[ctypes.c_ubyte, Literal[2]]
  ptpls: int
  resv10: c.Array[ctypes.c_ubyte, Literal[14]]
  regctl_ds: c.Array[struct_nvme_registered_ctrl, Literal[0]]
struct_nvme_reservation_status.register_fields([('gen', ctypes.c_uint32, 0), ('rtype', ctypes.c_ubyte, 4), ('regctl', c.Array[ctypes.c_ubyte, Literal[2]], 5), ('resv5', c.Array[ctypes.c_ubyte, Literal[2]], 7), ('ptpls', ctypes.c_ubyte, 9), ('resv10', c.Array[ctypes.c_ubyte, Literal[14]], 10), ('regctl_ds', c.Array[struct_nvme_registered_ctrl, Literal[0]], 24)])
@c.record
class struct_nvme_registered_ctrl_ext(c.Struct):
  SIZE = 64
  cntlid: int
  rcsts: int
  rsvd3: c.Array[ctypes.c_ubyte, Literal[5]]
  rkey: int
  hostid: c.Array[ctypes.c_ubyte, Literal[16]]
  rsvd32: c.Array[ctypes.c_ubyte, Literal[32]]
struct_nvme_registered_ctrl_ext.register_fields([('cntlid', ctypes.c_uint16, 0), ('rcsts', ctypes.c_ubyte, 2), ('rsvd3', c.Array[ctypes.c_ubyte, Literal[5]], 3), ('rkey', ctypes.c_uint64, 8), ('hostid', c.Array[ctypes.c_ubyte, Literal[16]], 16), ('rsvd32', c.Array[ctypes.c_ubyte, Literal[32]], 32)])
@c.record
class struct_nvme_reservation_status_ext(c.Struct):
  SIZE = 64
  gen: int
  rtype: int
  regctl: c.Array[ctypes.c_ubyte, Literal[2]]
  resv5: c.Array[ctypes.c_ubyte, Literal[2]]
  ptpls: int
  resv10: c.Array[ctypes.c_ubyte, Literal[14]]
  rsvd24: c.Array[ctypes.c_ubyte, Literal[40]]
  regctl_eds: c.Array[struct_nvme_registered_ctrl_ext, Literal[0]]
struct_nvme_reservation_status_ext.register_fields([('gen', ctypes.c_uint32, 0), ('rtype', ctypes.c_ubyte, 4), ('regctl', c.Array[ctypes.c_ubyte, Literal[2]], 5), ('resv5', c.Array[ctypes.c_ubyte, Literal[2]], 7), ('ptpls', ctypes.c_ubyte, 9), ('resv10', c.Array[ctypes.c_ubyte, Literal[14]], 10), ('rsvd24', c.Array[ctypes.c_ubyte, Literal[40]], 24), ('regctl_eds', c.Array[struct_nvme_registered_ctrl_ext, Literal[0]], 64)])
enum_nvme_opcode: dict[int, str] = {(nvme_cmd_flush:=0): 'nvme_cmd_flush', (nvme_cmd_write:=1): 'nvme_cmd_write', (nvme_cmd_read:=2): 'nvme_cmd_read', (nvme_cmd_write_uncor:=4): 'nvme_cmd_write_uncor', (nvme_cmd_compare:=5): 'nvme_cmd_compare', (nvme_cmd_write_zeroes:=8): 'nvme_cmd_write_zeroes', (nvme_cmd_dsm:=9): 'nvme_cmd_dsm', (nvme_cmd_verify:=12): 'nvme_cmd_verify', (nvme_cmd_resv_register:=13): 'nvme_cmd_resv_register', (nvme_cmd_resv_report:=14): 'nvme_cmd_resv_report', (nvme_cmd_resv_acquire:=17): 'nvme_cmd_resv_acquire', (nvme_cmd_io_mgmt_recv:=18): 'nvme_cmd_io_mgmt_recv', (nvme_cmd_resv_release:=21): 'nvme_cmd_resv_release', (nvme_cmd_zone_mgmt_send:=121): 'nvme_cmd_zone_mgmt_send', (nvme_cmd_zone_mgmt_recv:=122): 'nvme_cmd_zone_mgmt_recv', (nvme_cmd_zone_append:=125): 'nvme_cmd_zone_append', (nvme_cmd_vendor_start:=128): 'nvme_cmd_vendor_start'}
_anonenum34: dict[int, str] = {(NVME_SGL_FMT_ADDRESS:=0): 'NVME_SGL_FMT_ADDRESS', (NVME_SGL_FMT_OFFSET:=1): 'NVME_SGL_FMT_OFFSET', (NVME_SGL_FMT_TRANSPORT_A:=10): 'NVME_SGL_FMT_TRANSPORT_A', (NVME_SGL_FMT_INVALIDATE:=15): 'NVME_SGL_FMT_INVALIDATE'}
_anonenum35: dict[int, str] = {(NVME_SGL_FMT_DATA_DESC:=0): 'NVME_SGL_FMT_DATA_DESC', (NVME_SGL_FMT_SEG_DESC:=2): 'NVME_SGL_FMT_SEG_DESC', (NVME_SGL_FMT_LAST_SEG_DESC:=3): 'NVME_SGL_FMT_LAST_SEG_DESC', (NVME_KEY_SGL_FMT_DATA_DESC:=4): 'NVME_KEY_SGL_FMT_DATA_DESC', (NVME_TRANSPORT_SGL_DATA_DESC:=5): 'NVME_TRANSPORT_SGL_DATA_DESC'}
@c.record
class struct_nvme_sgl_desc(c.Struct):
  SIZE = 16
  addr: int
  length: int
  rsvd: c.Array[ctypes.c_ubyte, Literal[3]]
  type: int
struct_nvme_sgl_desc.register_fields([('addr', ctypes.c_uint64, 0), ('length', ctypes.c_uint32, 8), ('rsvd', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('type', ctypes.c_ubyte, 15)])
@c.record
class struct_nvme_keyed_sgl_desc(c.Struct):
  SIZE = 16
  addr: int
  length: c.Array[ctypes.c_ubyte, Literal[3]]
  key: c.Array[ctypes.c_ubyte, Literal[4]]
  type: int
struct_nvme_keyed_sgl_desc.register_fields([('addr', ctypes.c_uint64, 0), ('length', c.Array[ctypes.c_ubyte, Literal[3]], 8), ('key', c.Array[ctypes.c_ubyte, Literal[4]], 11), ('type', ctypes.c_ubyte, 15)])
@c.record
class union_nvme_data_ptr(c.Struct):
  SIZE = 16
  prp1: int
  prp2: int
  sgl: struct_nvme_sgl_desc
  ksgl: struct_nvme_keyed_sgl_desc
union_nvme_data_ptr.register_fields([('prp1', ctypes.c_uint64, 0), ('prp2', ctypes.c_uint64, 8), ('sgl', struct_nvme_sgl_desc, 0), ('ksgl', struct_nvme_keyed_sgl_desc, 0)])
_anonenum36: dict[int, str] = {(NVME_CMD_FUSE_FIRST:=1): 'NVME_CMD_FUSE_FIRST', (NVME_CMD_FUSE_SECOND:=2): 'NVME_CMD_FUSE_SECOND', (NVME_CMD_SGL_METABUF:=64): 'NVME_CMD_SGL_METABUF', (NVME_CMD_SGL_METASEG:=128): 'NVME_CMD_SGL_METASEG', (NVME_CMD_SGL_ALL:=192): 'NVME_CMD_SGL_ALL'}
@c.record
class struct_nvme_common_command(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  cdw2: c.Array[ctypes.c_uint32, Literal[2]]
  metadata: int
  dptr: union_nvme_data_ptr
  cdw10: int
  cdw11: int
  cdw12: int
  cdw13: int
  cdw14: int
  cdw15: int
  cdws: struct_nvme_common_command_cdws
__u16: TypeAlias = ctypes.c_uint16
@c.record
class struct_nvme_common_command_cdws(c.Struct):
  SIZE = 24
  cdw10: int
  cdw11: int
  cdw12: int
  cdw13: int
  cdw14: int
  cdw15: int
struct_nvme_common_command_cdws.register_fields([('cdw10', ctypes.c_uint32, 0), ('cdw11', ctypes.c_uint32, 4), ('cdw12', ctypes.c_uint32, 8), ('cdw13', ctypes.c_uint32, 12), ('cdw14', ctypes.c_uint32, 16), ('cdw15', ctypes.c_uint32, 20)])
struct_nvme_common_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('cdw2', c.Array[ctypes.c_uint32, Literal[2]], 8), ('metadata', ctypes.c_uint64, 16), ('dptr', union_nvme_data_ptr, 24), ('cdw10', ctypes.c_uint32, 40), ('cdw11', ctypes.c_uint32, 44), ('cdw12', ctypes.c_uint32, 48), ('cdw13', ctypes.c_uint32, 52), ('cdw14', ctypes.c_uint32, 56), ('cdw15', ctypes.c_uint32, 60), ('cdws', struct_nvme_common_command_cdws, 40)])
@c.record
class struct_nvme_rw_command(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  cdw2: int
  cdw3: int
  metadata: int
  dptr: union_nvme_data_ptr
  slba: int
  length: int
  control: int
  dsmgmt: int
  reftag: int
  lbat: int
  lbatm: int
struct_nvme_rw_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('cdw2', ctypes.c_uint32, 8), ('cdw3', ctypes.c_uint32, 12), ('metadata', ctypes.c_uint64, 16), ('dptr', union_nvme_data_ptr, 24), ('slba', ctypes.c_uint64, 40), ('length', ctypes.c_uint16, 48), ('control', ctypes.c_uint16, 50), ('dsmgmt', ctypes.c_uint32, 52), ('reftag', ctypes.c_uint32, 56), ('lbat', ctypes.c_uint16, 60), ('lbatm', ctypes.c_uint16, 62)])
_anonenum37: dict[int, str] = {(NVME_RW_LR:=32768): 'NVME_RW_LR', (NVME_RW_FUA:=16384): 'NVME_RW_FUA', (NVME_RW_APPEND_PIREMAP:=512): 'NVME_RW_APPEND_PIREMAP', (NVME_RW_DSM_FREQ_UNSPEC:=0): 'NVME_RW_DSM_FREQ_UNSPEC', (NVME_RW_DSM_FREQ_TYPICAL:=1): 'NVME_RW_DSM_FREQ_TYPICAL', (NVME_RW_DSM_FREQ_RARE:=2): 'NVME_RW_DSM_FREQ_RARE', (NVME_RW_DSM_FREQ_READS:=3): 'NVME_RW_DSM_FREQ_READS', (NVME_RW_DSM_FREQ_WRITES:=4): 'NVME_RW_DSM_FREQ_WRITES', (NVME_RW_DSM_FREQ_RW:=5): 'NVME_RW_DSM_FREQ_RW', (NVME_RW_DSM_FREQ_ONCE:=6): 'NVME_RW_DSM_FREQ_ONCE', (NVME_RW_DSM_FREQ_PREFETCH:=7): 'NVME_RW_DSM_FREQ_PREFETCH', (NVME_RW_DSM_FREQ_TEMP:=8): 'NVME_RW_DSM_FREQ_TEMP', (NVME_RW_DSM_LATENCY_NONE:=0): 'NVME_RW_DSM_LATENCY_NONE', (NVME_RW_DSM_LATENCY_IDLE:=16): 'NVME_RW_DSM_LATENCY_IDLE', (NVME_RW_DSM_LATENCY_NORM:=32): 'NVME_RW_DSM_LATENCY_NORM', (NVME_RW_DSM_LATENCY_LOW:=48): 'NVME_RW_DSM_LATENCY_LOW', (NVME_RW_DSM_SEQ_REQ:=64): 'NVME_RW_DSM_SEQ_REQ', (NVME_RW_DSM_COMPRESSED:=128): 'NVME_RW_DSM_COMPRESSED', (NVME_RW_PRINFO_PRCHK_REF:=1024): 'NVME_RW_PRINFO_PRCHK_REF', (NVME_RW_PRINFO_PRCHK_APP:=2048): 'NVME_RW_PRINFO_PRCHK_APP', (NVME_RW_PRINFO_PRCHK_GUARD:=4096): 'NVME_RW_PRINFO_PRCHK_GUARD', (NVME_RW_PRINFO_PRACT:=8192): 'NVME_RW_PRINFO_PRACT', (NVME_RW_DTYPE_STREAMS:=16): 'NVME_RW_DTYPE_STREAMS', (NVME_RW_DTYPE_DPLCMT:=32): 'NVME_RW_DTYPE_DPLCMT', (NVME_WZ_DEAC:=512): 'NVME_WZ_DEAC'}
@c.record
class struct_nvme_dsm_cmd(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  rsvd2: c.Array[ctypes.c_uint64, Literal[2]]
  dptr: union_nvme_data_ptr
  nr: int
  attributes: int
  rsvd12: c.Array[ctypes.c_uint32, Literal[4]]
__u64: TypeAlias = ctypes.c_uint64
__u32: TypeAlias = ctypes.c_uint32
struct_nvme_dsm_cmd.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint64, Literal[2]], 8), ('dptr', union_nvme_data_ptr, 24), ('nr', ctypes.c_uint32, 40), ('attributes', ctypes.c_uint32, 44), ('rsvd12', c.Array[ctypes.c_uint32, Literal[4]], 48)])
_anonenum38: dict[int, str] = {(NVME_DSMGMT_IDR:=1): 'NVME_DSMGMT_IDR', (NVME_DSMGMT_IDW:=2): 'NVME_DSMGMT_IDW', (NVME_DSMGMT_AD:=4): 'NVME_DSMGMT_AD'}
@c.record
class struct_nvme_dsm_range(c.Struct):
  SIZE = 16
  cattr: int
  nlb: int
  slba: int
struct_nvme_dsm_range.register_fields([('cattr', ctypes.c_uint32, 0), ('nlb', ctypes.c_uint32, 4), ('slba', ctypes.c_uint64, 8)])
@c.record
class struct_nvme_write_zeroes_cmd(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  rsvd2: int
  metadata: int
  dptr: union_nvme_data_ptr
  slba: int
  length: int
  control: int
  dsmgmt: int
  reftag: int
  lbat: int
  lbatm: int
struct_nvme_write_zeroes_cmd.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('rsvd2', ctypes.c_uint64, 8), ('metadata', ctypes.c_uint64, 16), ('dptr', union_nvme_data_ptr, 24), ('slba', ctypes.c_uint64, 40), ('length', ctypes.c_uint16, 48), ('control', ctypes.c_uint16, 50), ('dsmgmt', ctypes.c_uint32, 52), ('reftag', ctypes.c_uint32, 56), ('lbat', ctypes.c_uint16, 60), ('lbatm', ctypes.c_uint16, 62)])
enum_nvme_zone_mgmt_action: dict[int, str] = {(NVME_ZONE_CLOSE:=1): 'NVME_ZONE_CLOSE', (NVME_ZONE_FINISH:=2): 'NVME_ZONE_FINISH', (NVME_ZONE_OPEN:=3): 'NVME_ZONE_OPEN', (NVME_ZONE_RESET:=4): 'NVME_ZONE_RESET', (NVME_ZONE_OFFLINE:=5): 'NVME_ZONE_OFFLINE', (NVME_ZONE_SET_DESC_EXT:=16): 'NVME_ZONE_SET_DESC_EXT'}
@c.record
class struct_nvme_zone_mgmt_send_cmd(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  cdw2: c.Array[ctypes.c_uint32, Literal[2]]
  metadata: int
  dptr: union_nvme_data_ptr
  slba: int
  cdw12: int
  zsa: int
  select_all: int
  rsvd13: c.Array[ctypes.c_ubyte, Literal[2]]
  cdw14: c.Array[ctypes.c_uint32, Literal[2]]
struct_nvme_zone_mgmt_send_cmd.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('cdw2', c.Array[ctypes.c_uint32, Literal[2]], 8), ('metadata', ctypes.c_uint64, 16), ('dptr', union_nvme_data_ptr, 24), ('slba', ctypes.c_uint64, 40), ('cdw12', ctypes.c_uint32, 48), ('zsa', ctypes.c_ubyte, 52), ('select_all', ctypes.c_ubyte, 53), ('rsvd13', c.Array[ctypes.c_ubyte, Literal[2]], 54), ('cdw14', c.Array[ctypes.c_uint32, Literal[2]], 56)])
@c.record
class struct_nvme_zone_mgmt_recv_cmd(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  rsvd2: c.Array[ctypes.c_uint64, Literal[2]]
  dptr: union_nvme_data_ptr
  slba: int
  numd: int
  zra: int
  zrasf: int
  pr: int
  rsvd13: int
  cdw14: c.Array[ctypes.c_uint32, Literal[2]]
struct_nvme_zone_mgmt_recv_cmd.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint64, Literal[2]], 8), ('dptr', union_nvme_data_ptr, 24), ('slba', ctypes.c_uint64, 40), ('numd', ctypes.c_uint32, 48), ('zra', ctypes.c_ubyte, 52), ('zrasf', ctypes.c_ubyte, 53), ('pr', ctypes.c_ubyte, 54), ('rsvd13', ctypes.c_ubyte, 55), ('cdw14', c.Array[ctypes.c_uint32, Literal[2]], 56)])
@c.record
class struct_nvme_io_mgmt_recv_cmd(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  rsvd2: c.Array[ctypes.c_uint64, Literal[2]]
  dptr: union_nvme_data_ptr
  mo: int
  rsvd11: int
  mos: int
  numd: int
  cdw12: c.Array[ctypes.c_uint32, Literal[4]]
struct_nvme_io_mgmt_recv_cmd.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint64, Literal[2]], 8), ('dptr', union_nvme_data_ptr, 24), ('mo', ctypes.c_ubyte, 40), ('rsvd11', ctypes.c_ubyte, 41), ('mos', ctypes.c_uint16, 42), ('numd', ctypes.c_uint32, 44), ('cdw12', c.Array[ctypes.c_uint32, Literal[4]], 48)])
_anonenum39: dict[int, str] = {(NVME_IO_MGMT_RECV_MO_RUHS:=1): 'NVME_IO_MGMT_RECV_MO_RUHS'}
@c.record
class struct_nvme_fdp_ruh_status_desc(c.Struct):
  SIZE = 32
  pid: int
  ruhid: int
  earutr: int
  ruamw: int
  reserved: c.Array[ctypes.c_ubyte, Literal[16]]
struct_nvme_fdp_ruh_status_desc.register_fields([('pid', ctypes.c_uint16, 0), ('ruhid', ctypes.c_uint16, 2), ('earutr', ctypes.c_uint32, 4), ('ruamw', ctypes.c_uint64, 8), ('reserved', c.Array[ctypes.c_ubyte, Literal[16]], 16)])
@c.record
class struct_nvme_fdp_ruh_status(c.Struct):
  SIZE = 16
  rsvd0: c.Array[ctypes.c_ubyte, Literal[14]]
  nruhsd: int
  ruhsd: c.Array[struct_nvme_fdp_ruh_status_desc, Literal[0]]
struct_nvme_fdp_ruh_status.register_fields([('rsvd0', c.Array[ctypes.c_ubyte, Literal[14]], 0), ('nruhsd', ctypes.c_uint16, 14), ('ruhsd', c.Array[struct_nvme_fdp_ruh_status_desc, Literal[0]], 16)])
_anonenum40: dict[int, str] = {(NVME_ZRA_ZONE_REPORT:=0): 'NVME_ZRA_ZONE_REPORT', (NVME_ZRASF_ZONE_REPORT_ALL:=0): 'NVME_ZRASF_ZONE_REPORT_ALL', (NVME_ZRASF_ZONE_STATE_EMPTY:=1): 'NVME_ZRASF_ZONE_STATE_EMPTY', (NVME_ZRASF_ZONE_STATE_IMP_OPEN:=2): 'NVME_ZRASF_ZONE_STATE_IMP_OPEN', (NVME_ZRASF_ZONE_STATE_EXP_OPEN:=3): 'NVME_ZRASF_ZONE_STATE_EXP_OPEN', (NVME_ZRASF_ZONE_STATE_CLOSED:=4): 'NVME_ZRASF_ZONE_STATE_CLOSED', (NVME_ZRASF_ZONE_STATE_READONLY:=5): 'NVME_ZRASF_ZONE_STATE_READONLY', (NVME_ZRASF_ZONE_STATE_FULL:=6): 'NVME_ZRASF_ZONE_STATE_FULL', (NVME_ZRASF_ZONE_STATE_OFFLINE:=7): 'NVME_ZRASF_ZONE_STATE_OFFLINE', (NVME_REPORT_ZONE_PARTIAL:=1): 'NVME_REPORT_ZONE_PARTIAL'}
_anonenum41: dict[int, str] = {(NVME_TEMP_THRESH_MASK:=65535): 'NVME_TEMP_THRESH_MASK', (NVME_TEMP_THRESH_SELECT_SHIFT:=16): 'NVME_TEMP_THRESH_SELECT_SHIFT', (NVME_TEMP_THRESH_TYPE_UNDER:=1048576): 'NVME_TEMP_THRESH_TYPE_UNDER'}
@c.record
class struct_nvme_feat_auto_pst(c.Struct):
  SIZE = 256
  entries: c.Array[ctypes.c_uint64, Literal[32]]
struct_nvme_feat_auto_pst.register_fields([('entries', c.Array[ctypes.c_uint64, Literal[32]], 0)])
_anonenum42: dict[int, str] = {(NVME_HOST_MEM_ENABLE:=1): 'NVME_HOST_MEM_ENABLE', (NVME_HOST_MEM_RETURN:=2): 'NVME_HOST_MEM_RETURN'}
@c.record
class struct_nvme_feat_host_behavior(c.Struct):
  SIZE = 512
  acre: int
  etdas: int
  lbafee: int
  resv1: c.Array[ctypes.c_ubyte, Literal[509]]
struct_nvme_feat_host_behavior.register_fields([('acre', ctypes.c_ubyte, 0), ('etdas', ctypes.c_ubyte, 1), ('lbafee', ctypes.c_ubyte, 2), ('resv1', c.Array[ctypes.c_ubyte, Literal[509]], 3)])
_anonenum43: dict[int, str] = {(NVME_ENABLE_ACRE:=1): 'NVME_ENABLE_ACRE', (NVME_ENABLE_LBAFEE:=1): 'NVME_ENABLE_LBAFEE'}
enum_nvme_admin_opcode: dict[int, str] = {(nvme_admin_delete_sq:=0): 'nvme_admin_delete_sq', (nvme_admin_create_sq:=1): 'nvme_admin_create_sq', (nvme_admin_get_log_page:=2): 'nvme_admin_get_log_page', (nvme_admin_delete_cq:=4): 'nvme_admin_delete_cq', (nvme_admin_create_cq:=5): 'nvme_admin_create_cq', (nvme_admin_identify:=6): 'nvme_admin_identify', (nvme_admin_abort_cmd:=8): 'nvme_admin_abort_cmd', (nvme_admin_set_features:=9): 'nvme_admin_set_features', (nvme_admin_get_features:=10): 'nvme_admin_get_features', (nvme_admin_async_event:=12): 'nvme_admin_async_event', (nvme_admin_ns_mgmt:=13): 'nvme_admin_ns_mgmt', (nvme_admin_activate_fw:=16): 'nvme_admin_activate_fw', (nvme_admin_download_fw:=17): 'nvme_admin_download_fw', (nvme_admin_dev_self_test:=20): 'nvme_admin_dev_self_test', (nvme_admin_ns_attach:=21): 'nvme_admin_ns_attach', (nvme_admin_keep_alive:=24): 'nvme_admin_keep_alive', (nvme_admin_directive_send:=25): 'nvme_admin_directive_send', (nvme_admin_directive_recv:=26): 'nvme_admin_directive_recv', (nvme_admin_virtual_mgmt:=28): 'nvme_admin_virtual_mgmt', (nvme_admin_nvme_mi_send:=29): 'nvme_admin_nvme_mi_send', (nvme_admin_nvme_mi_recv:=30): 'nvme_admin_nvme_mi_recv', (nvme_admin_dbbuf:=124): 'nvme_admin_dbbuf', (nvme_admin_format_nvm:=128): 'nvme_admin_format_nvm', (nvme_admin_security_send:=129): 'nvme_admin_security_send', (nvme_admin_security_recv:=130): 'nvme_admin_security_recv', (nvme_admin_sanitize_nvm:=132): 'nvme_admin_sanitize_nvm', (nvme_admin_get_lba_status:=134): 'nvme_admin_get_lba_status', (nvme_admin_vendor_start:=192): 'nvme_admin_vendor_start'}
_anonenum44: dict[int, str] = {(NVME_QUEUE_PHYS_CONTIG:=1): 'NVME_QUEUE_PHYS_CONTIG', (NVME_CQ_IRQ_ENABLED:=2): 'NVME_CQ_IRQ_ENABLED', (NVME_SQ_PRIO_URGENT:=0): 'NVME_SQ_PRIO_URGENT', (NVME_SQ_PRIO_HIGH:=2): 'NVME_SQ_PRIO_HIGH', (NVME_SQ_PRIO_MEDIUM:=4): 'NVME_SQ_PRIO_MEDIUM', (NVME_SQ_PRIO_LOW:=6): 'NVME_SQ_PRIO_LOW', (NVME_FEAT_ARBITRATION:=1): 'NVME_FEAT_ARBITRATION', (NVME_FEAT_POWER_MGMT:=2): 'NVME_FEAT_POWER_MGMT', (NVME_FEAT_LBA_RANGE:=3): 'NVME_FEAT_LBA_RANGE', (NVME_FEAT_TEMP_THRESH:=4): 'NVME_FEAT_TEMP_THRESH', (NVME_FEAT_ERR_RECOVERY:=5): 'NVME_FEAT_ERR_RECOVERY', (NVME_FEAT_VOLATILE_WC:=6): 'NVME_FEAT_VOLATILE_WC', (NVME_FEAT_NUM_QUEUES:=7): 'NVME_FEAT_NUM_QUEUES', (NVME_FEAT_IRQ_COALESCE:=8): 'NVME_FEAT_IRQ_COALESCE', (NVME_FEAT_IRQ_CONFIG:=9): 'NVME_FEAT_IRQ_CONFIG', (NVME_FEAT_WRITE_ATOMIC:=10): 'NVME_FEAT_WRITE_ATOMIC', (NVME_FEAT_ASYNC_EVENT:=11): 'NVME_FEAT_ASYNC_EVENT', (NVME_FEAT_AUTO_PST:=12): 'NVME_FEAT_AUTO_PST', (NVME_FEAT_HOST_MEM_BUF:=13): 'NVME_FEAT_HOST_MEM_BUF', (NVME_FEAT_TIMESTAMP:=14): 'NVME_FEAT_TIMESTAMP', (NVME_FEAT_KATO:=15): 'NVME_FEAT_KATO', (NVME_FEAT_HCTM:=16): 'NVME_FEAT_HCTM', (NVME_FEAT_NOPSC:=17): 'NVME_FEAT_NOPSC', (NVME_FEAT_RRL:=18): 'NVME_FEAT_RRL', (NVME_FEAT_PLM_CONFIG:=19): 'NVME_FEAT_PLM_CONFIG', (NVME_FEAT_PLM_WINDOW:=20): 'NVME_FEAT_PLM_WINDOW', (NVME_FEAT_HOST_BEHAVIOR:=22): 'NVME_FEAT_HOST_BEHAVIOR', (NVME_FEAT_SANITIZE:=23): 'NVME_FEAT_SANITIZE', (NVME_FEAT_FDP:=29): 'NVME_FEAT_FDP', (NVME_FEAT_SW_PROGRESS:=128): 'NVME_FEAT_SW_PROGRESS', (NVME_FEAT_HOST_ID:=129): 'NVME_FEAT_HOST_ID', (NVME_FEAT_RESV_MASK:=130): 'NVME_FEAT_RESV_MASK', (NVME_FEAT_RESV_PERSIST:=131): 'NVME_FEAT_RESV_PERSIST', (NVME_FEAT_WRITE_PROTECT:=132): 'NVME_FEAT_WRITE_PROTECT', (NVME_FEAT_VENDOR_START:=192): 'NVME_FEAT_VENDOR_START', (NVME_FEAT_VENDOR_END:=255): 'NVME_FEAT_VENDOR_END', (NVME_LOG_SUPPORTED:=0): 'NVME_LOG_SUPPORTED', (NVME_LOG_ERROR:=1): 'NVME_LOG_ERROR', (NVME_LOG_SMART:=2): 'NVME_LOG_SMART', (NVME_LOG_FW_SLOT:=3): 'NVME_LOG_FW_SLOT', (NVME_LOG_CHANGED_NS:=4): 'NVME_LOG_CHANGED_NS', (NVME_LOG_CMD_EFFECTS:=5): 'NVME_LOG_CMD_EFFECTS', (NVME_LOG_DEVICE_SELF_TEST:=6): 'NVME_LOG_DEVICE_SELF_TEST', (NVME_LOG_TELEMETRY_HOST:=7): 'NVME_LOG_TELEMETRY_HOST', (NVME_LOG_TELEMETRY_CTRL:=8): 'NVME_LOG_TELEMETRY_CTRL', (NVME_LOG_ENDURANCE_GROUP:=9): 'NVME_LOG_ENDURANCE_GROUP', (NVME_LOG_ANA:=12): 'NVME_LOG_ANA', (NVME_LOG_FEATURES:=18): 'NVME_LOG_FEATURES', (NVME_LOG_RMI:=22): 'NVME_LOG_RMI', (NVME_LOG_FDP_CONFIGS:=32): 'NVME_LOG_FDP_CONFIGS', (NVME_LOG_DISC:=112): 'NVME_LOG_DISC', (NVME_LOG_RESERVATION:=128): 'NVME_LOG_RESERVATION', (NVME_FWACT_REPL:=0): 'NVME_FWACT_REPL', (NVME_FWACT_REPL_ACTV:=8): 'NVME_FWACT_REPL_ACTV', (NVME_FWACT_ACTV:=16): 'NVME_FWACT_ACTV'}
@c.record
class struct_nvme_supported_log(c.Struct):
  SIZE = 1024
  lids: c.Array[ctypes.c_uint32, Literal[256]]
struct_nvme_supported_log.register_fields([('lids', c.Array[ctypes.c_uint32, Literal[256]], 0)])
_anonenum45: dict[int, str] = {(NVME_LIDS_LSUPP:=1): 'NVME_LIDS_LSUPP'}
@c.record
class struct_nvme_supported_features_log(c.Struct):
  SIZE = 1024
  fis: c.Array[ctypes.c_uint32, Literal[256]]
struct_nvme_supported_features_log.register_fields([('fis', c.Array[ctypes.c_uint32, Literal[256]], 0)])
_anonenum46: dict[int, str] = {(NVME_FIS_FSUPP:=1): 'NVME_FIS_FSUPP', (NVME_FIS_NSCPE:=1048576): 'NVME_FIS_NSCPE', (NVME_FIS_CSCPE:=2097152): 'NVME_FIS_CSCPE'}
_anonenum47: dict[int, str] = {(NVME_NS_NO_WRITE_PROTECT:=0): 'NVME_NS_NO_WRITE_PROTECT', (NVME_NS_WRITE_PROTECT:=1): 'NVME_NS_WRITE_PROTECT', (NVME_NS_WRITE_PROTECT_POWER_CYCLE:=2): 'NVME_NS_WRITE_PROTECT_POWER_CYCLE', (NVME_NS_WRITE_PROTECT_PERMANENT:=3): 'NVME_NS_WRITE_PROTECT_PERMANENT'}
@c.record
class struct_nvme_identify(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  rsvd2: c.Array[ctypes.c_uint64, Literal[2]]
  dptr: union_nvme_data_ptr
  cns: int
  rsvd3: int
  ctrlid: int
  cnssid: int
  rsvd11: int
  csi: int
  rsvd12: c.Array[ctypes.c_uint32, Literal[4]]
struct_nvme_identify.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint64, Literal[2]], 8), ('dptr', union_nvme_data_ptr, 24), ('cns', ctypes.c_ubyte, 40), ('rsvd3', ctypes.c_ubyte, 41), ('ctrlid', ctypes.c_uint16, 42), ('cnssid', ctypes.c_uint16, 44), ('rsvd11', ctypes.c_ubyte, 46), ('csi', ctypes.c_ubyte, 47), ('rsvd12', c.Array[ctypes.c_uint32, Literal[4]], 48)])
@c.record
class struct_nvme_features(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  rsvd2: c.Array[ctypes.c_uint64, Literal[2]]
  dptr: union_nvme_data_ptr
  fid: int
  dword11: int
  dword12: int
  dword13: int
  dword14: int
  dword15: int
struct_nvme_features.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint64, Literal[2]], 8), ('dptr', union_nvme_data_ptr, 24), ('fid', ctypes.c_uint32, 40), ('dword11', ctypes.c_uint32, 44), ('dword12', ctypes.c_uint32, 48), ('dword13', ctypes.c_uint32, 52), ('dword14', ctypes.c_uint32, 56), ('dword15', ctypes.c_uint32, 60)])
@c.record
class struct_nvme_host_mem_buf_desc(c.Struct):
  SIZE = 16
  addr: int
  size: int
  rsvd: int
struct_nvme_host_mem_buf_desc.register_fields([('addr', ctypes.c_uint64, 0), ('size', ctypes.c_uint32, 8), ('rsvd', ctypes.c_uint32, 12)])
@c.record
class struct_nvme_create_cq(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  rsvd1: c.Array[ctypes.c_uint32, Literal[5]]
  prp1: int
  rsvd8: int
  cqid: int
  qsize: int
  cq_flags: int
  irq_vector: int
  rsvd12: c.Array[ctypes.c_uint32, Literal[4]]
struct_nvme_create_cq.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('rsvd1', c.Array[ctypes.c_uint32, Literal[5]], 4), ('prp1', ctypes.c_uint64, 24), ('rsvd8', ctypes.c_uint64, 32), ('cqid', ctypes.c_uint16, 40), ('qsize', ctypes.c_uint16, 42), ('cq_flags', ctypes.c_uint16, 44), ('irq_vector', ctypes.c_uint16, 46), ('rsvd12', c.Array[ctypes.c_uint32, Literal[4]], 48)])
@c.record
class struct_nvme_create_sq(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  rsvd1: c.Array[ctypes.c_uint32, Literal[5]]
  prp1: int
  rsvd8: int
  sqid: int
  qsize: int
  sq_flags: int
  cqid: int
  rsvd12: c.Array[ctypes.c_uint32, Literal[4]]
struct_nvme_create_sq.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('rsvd1', c.Array[ctypes.c_uint32, Literal[5]], 4), ('prp1', ctypes.c_uint64, 24), ('rsvd8', ctypes.c_uint64, 32), ('sqid', ctypes.c_uint16, 40), ('qsize', ctypes.c_uint16, 42), ('sq_flags', ctypes.c_uint16, 44), ('cqid', ctypes.c_uint16, 46), ('rsvd12', c.Array[ctypes.c_uint32, Literal[4]], 48)])
@c.record
class struct_nvme_delete_queue(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  rsvd1: c.Array[ctypes.c_uint32, Literal[9]]
  qid: int
  rsvd10: int
  rsvd11: c.Array[ctypes.c_uint32, Literal[5]]
struct_nvme_delete_queue.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('rsvd1', c.Array[ctypes.c_uint32, Literal[9]], 4), ('qid', ctypes.c_uint16, 40), ('rsvd10', ctypes.c_uint16, 42), ('rsvd11', c.Array[ctypes.c_uint32, Literal[5]], 44)])
@c.record
class struct_nvme_abort_cmd(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  rsvd1: c.Array[ctypes.c_uint32, Literal[9]]
  sqid: int
  cid: int
  rsvd11: c.Array[ctypes.c_uint32, Literal[5]]
struct_nvme_abort_cmd.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('rsvd1', c.Array[ctypes.c_uint32, Literal[9]], 4), ('sqid', ctypes.c_uint16, 40), ('cid', ctypes.c_uint16, 42), ('rsvd11', c.Array[ctypes.c_uint32, Literal[5]], 44)])
@c.record
class struct_nvme_download_firmware(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  rsvd1: c.Array[ctypes.c_uint32, Literal[5]]
  dptr: union_nvme_data_ptr
  numd: int
  offset: int
  rsvd12: c.Array[ctypes.c_uint32, Literal[4]]
struct_nvme_download_firmware.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('rsvd1', c.Array[ctypes.c_uint32, Literal[5]], 4), ('dptr', union_nvme_data_ptr, 24), ('numd', ctypes.c_uint32, 40), ('offset', ctypes.c_uint32, 44), ('rsvd12', c.Array[ctypes.c_uint32, Literal[4]], 48)])
@c.record
class struct_nvme_format_cmd(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  rsvd2: c.Array[ctypes.c_uint64, Literal[4]]
  cdw10: int
  rsvd11: c.Array[ctypes.c_uint32, Literal[5]]
struct_nvme_format_cmd.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint64, Literal[4]], 8), ('cdw10', ctypes.c_uint32, 40), ('rsvd11', c.Array[ctypes.c_uint32, Literal[5]], 44)])
@c.record
class struct_nvme_get_log_page_command(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  rsvd2: c.Array[ctypes.c_uint64, Literal[2]]
  dptr: union_nvme_data_ptr
  lid: int
  lsp: int
  numdl: int
  numdu: int
  lsi: int
  lpol: int
  lpou: int
  lpo: int
  rsvd14: c.Array[ctypes.c_ubyte, Literal[3]]
  csi: int
  rsvd15: int
struct_nvme_get_log_page_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint64, Literal[2]], 8), ('dptr', union_nvme_data_ptr, 24), ('lid', ctypes.c_ubyte, 40), ('lsp', ctypes.c_ubyte, 41), ('numdl', ctypes.c_uint16, 42), ('numdu', ctypes.c_uint16, 44), ('lsi', ctypes.c_uint16, 46), ('lpol', ctypes.c_uint32, 48), ('lpou', ctypes.c_uint32, 52), ('lpo', ctypes.c_uint64, 48), ('rsvd14', c.Array[ctypes.c_ubyte, Literal[3]], 56), ('csi', ctypes.c_ubyte, 59), ('rsvd15', ctypes.c_uint32, 60)])
@c.record
class struct_nvme_directive_cmd(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  nsid: int
  rsvd2: c.Array[ctypes.c_uint64, Literal[2]]
  dptr: union_nvme_data_ptr
  numd: int
  doper: int
  dtype: int
  dspec: int
  endir: int
  tdtype: int
  rsvd15: int
  rsvd16: c.Array[ctypes.c_uint32, Literal[3]]
struct_nvme_directive_cmd.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('nsid', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint64, Literal[2]], 8), ('dptr', union_nvme_data_ptr, 24), ('numd', ctypes.c_uint32, 40), ('doper', ctypes.c_ubyte, 44), ('dtype', ctypes.c_ubyte, 45), ('dspec', ctypes.c_uint16, 46), ('endir', ctypes.c_ubyte, 48), ('tdtype', ctypes.c_ubyte, 49), ('rsvd15', ctypes.c_uint16, 50), ('rsvd16', c.Array[ctypes.c_uint32, Literal[3]], 52)])
enum_nvmf_fabrics_opcode: dict[int, str] = {(nvme_fabrics_command:=127): 'nvme_fabrics_command'}
enum_nvmf_capsule_command: dict[int, str] = {(nvme_fabrics_type_property_set:=0): 'nvme_fabrics_type_property_set', (nvme_fabrics_type_connect:=1): 'nvme_fabrics_type_connect', (nvme_fabrics_type_property_get:=4): 'nvme_fabrics_type_property_get', (nvme_fabrics_type_auth_send:=5): 'nvme_fabrics_type_auth_send', (nvme_fabrics_type_auth_receive:=6): 'nvme_fabrics_type_auth_receive'}
@c.record
class struct_nvmf_common_command(c.Struct):
  SIZE = 64
  opcode: int
  resv1: int
  command_id: int
  fctype: int
  resv2: c.Array[ctypes.c_ubyte, Literal[35]]
  ts: c.Array[ctypes.c_ubyte, Literal[24]]
struct_nvmf_common_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('resv1', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('fctype', ctypes.c_ubyte, 4), ('resv2', c.Array[ctypes.c_ubyte, Literal[35]], 5), ('ts', c.Array[ctypes.c_ubyte, Literal[24]], 40)])
_anonenum48: dict[int, str] = {(NVME_DISC_EFLAGS_EPCSD:=2): 'NVME_DISC_EFLAGS_EPCSD', (NVME_DISC_EFLAGS_DUPRETINFO:=1): 'NVME_DISC_EFLAGS_DUPRETINFO'}
@c.record
class struct_nvmf_disc_rsp_page_entry(c.Struct):
  SIZE = 1024
  trtype: int
  adrfam: int
  subtype: int
  treq: int
  portid: int
  cntlid: int
  asqsz: int
  eflags: int
  resv10: c.Array[ctypes.c_ubyte, Literal[20]]
  trsvcid: c.Array[ctypes.c_char, Literal[32]]
  resv64: c.Array[ctypes.c_ubyte, Literal[192]]
  subnqn: c.Array[ctypes.c_char, Literal[256]]
  traddr: c.Array[ctypes.c_char, Literal[256]]
  tsas: union_tsas
@c.record
class union_tsas(c.Struct):
  SIZE = 256
  common: c.Array[ctypes.c_char, Literal[256]]
  rdma: struct_rdma
  tcp: struct_tcp
@c.record
class struct_rdma(c.Struct):
  SIZE = 256
  qptype: int
  prtype: int
  cms: int
  resv3: c.Array[ctypes.c_ubyte, Literal[5]]
  pkey: int
  resv10: c.Array[ctypes.c_ubyte, Literal[246]]
struct_rdma.register_fields([('qptype', ctypes.c_ubyte, 0), ('prtype', ctypes.c_ubyte, 1), ('cms', ctypes.c_ubyte, 2), ('resv3', c.Array[ctypes.c_ubyte, Literal[5]], 3), ('pkey', ctypes.c_uint16, 8), ('resv10', c.Array[ctypes.c_ubyte, Literal[246]], 10)])
@c.record
class struct_tcp(c.Struct):
  SIZE = 1
  sectype: int
struct_tcp.register_fields([('sectype', ctypes.c_ubyte, 0)])
union_tsas.register_fields([('common', c.Array[ctypes.c_char, Literal[256]], 0), ('rdma', struct_rdma, 0), ('tcp', struct_tcp, 0)])
struct_nvmf_disc_rsp_page_entry.register_fields([('trtype', ctypes.c_ubyte, 0), ('adrfam', ctypes.c_ubyte, 1), ('subtype', ctypes.c_ubyte, 2), ('treq', ctypes.c_ubyte, 3), ('portid', ctypes.c_uint16, 4), ('cntlid', ctypes.c_uint16, 6), ('asqsz', ctypes.c_uint16, 8), ('eflags', ctypes.c_uint16, 10), ('resv10', c.Array[ctypes.c_ubyte, Literal[20]], 12), ('trsvcid', c.Array[ctypes.c_char, Literal[32]], 32), ('resv64', c.Array[ctypes.c_ubyte, Literal[192]], 64), ('subnqn', c.Array[ctypes.c_char, Literal[256]], 256), ('traddr', c.Array[ctypes.c_char, Literal[256]], 512), ('tsas', union_tsas, 768)])
@c.record
class struct_nvmf_disc_rsp_page_hdr(c.Struct):
  SIZE = 1024
  genctr: int
  numrec: int
  recfmt: int
  resv14: c.Array[ctypes.c_ubyte, Literal[1006]]
  entries: c.Array[struct_nvmf_disc_rsp_page_entry, Literal[0]]
struct_nvmf_disc_rsp_page_hdr.register_fields([('genctr', ctypes.c_uint64, 0), ('numrec', ctypes.c_uint64, 8), ('recfmt', ctypes.c_uint16, 16), ('resv14', c.Array[ctypes.c_ubyte, Literal[1006]], 18), ('entries', c.Array[struct_nvmf_disc_rsp_page_entry, Literal[0]], 1024)])
_anonenum49: dict[int, str] = {(NVME_CONNECT_DISABLE_SQFLOW:=4): 'NVME_CONNECT_DISABLE_SQFLOW'}
@c.record
class struct_nvmf_connect_command(c.Struct):
  SIZE = 64
  opcode: int
  resv1: int
  command_id: int
  fctype: int
  resv2: c.Array[ctypes.c_ubyte, Literal[19]]
  dptr: union_nvme_data_ptr
  recfmt: int
  qid: int
  sqsize: int
  cattr: int
  resv3: int
  kato: int
  resv4: c.Array[ctypes.c_ubyte, Literal[12]]
struct_nvmf_connect_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('resv1', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('fctype', ctypes.c_ubyte, 4), ('resv2', c.Array[ctypes.c_ubyte, Literal[19]], 5), ('dptr', union_nvme_data_ptr, 24), ('recfmt', ctypes.c_uint16, 40), ('qid', ctypes.c_uint16, 42), ('sqsize', ctypes.c_uint16, 44), ('cattr', ctypes.c_ubyte, 46), ('resv3', ctypes.c_ubyte, 47), ('kato', ctypes.c_uint32, 48), ('resv4', c.Array[ctypes.c_ubyte, Literal[12]], 52)])
_anonenum50: dict[int, str] = {(NVME_CONNECT_AUTHREQ_ASCR:=262144): 'NVME_CONNECT_AUTHREQ_ASCR', (NVME_CONNECT_AUTHREQ_ATR:=131072): 'NVME_CONNECT_AUTHREQ_ATR'}
@c.record
class struct_nvmf_connect_data(c.Struct):
  SIZE = 1024
  hostid: uuid_t
  cntlid: int
  resv4: c.Array[ctypes.c_char, Literal[238]]
  subsysnqn: c.Array[ctypes.c_char, Literal[256]]
  hostnqn: c.Array[ctypes.c_char, Literal[256]]
  resv5: c.Array[ctypes.c_char, Literal[256]]
@c.record
class uuid_t(c.Struct):
  SIZE = 16
  b: c.Array[ctypes.c_ubyte, Literal[16]]
uuid_t.register_fields([('b', c.Array[ctypes.c_ubyte, Literal[16]], 0)])
struct_nvmf_connect_data.register_fields([('hostid', uuid_t, 0), ('cntlid', ctypes.c_uint16, 16), ('resv4', c.Array[ctypes.c_char, Literal[238]], 18), ('subsysnqn', c.Array[ctypes.c_char, Literal[256]], 256), ('hostnqn', c.Array[ctypes.c_char, Literal[256]], 512), ('resv5', c.Array[ctypes.c_char, Literal[256]], 768)])
@c.record
class struct_nvmf_property_set_command(c.Struct):
  SIZE = 64
  opcode: int
  resv1: int
  command_id: int
  fctype: int
  resv2: c.Array[ctypes.c_ubyte, Literal[35]]
  attrib: int
  resv3: c.Array[ctypes.c_ubyte, Literal[3]]
  offset: int
  value: int
  resv4: c.Array[ctypes.c_ubyte, Literal[8]]
struct_nvmf_property_set_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('resv1', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('fctype', ctypes.c_ubyte, 4), ('resv2', c.Array[ctypes.c_ubyte, Literal[35]], 5), ('attrib', ctypes.c_ubyte, 40), ('resv3', c.Array[ctypes.c_ubyte, Literal[3]], 41), ('offset', ctypes.c_uint32, 44), ('value', ctypes.c_uint64, 48), ('resv4', c.Array[ctypes.c_ubyte, Literal[8]], 56)])
@c.record
class struct_nvmf_property_get_command(c.Struct):
  SIZE = 64
  opcode: int
  resv1: int
  command_id: int
  fctype: int
  resv2: c.Array[ctypes.c_ubyte, Literal[35]]
  attrib: int
  resv3: c.Array[ctypes.c_ubyte, Literal[3]]
  offset: int
  resv4: c.Array[ctypes.c_ubyte, Literal[16]]
struct_nvmf_property_get_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('resv1', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('fctype', ctypes.c_ubyte, 4), ('resv2', c.Array[ctypes.c_ubyte, Literal[35]], 5), ('attrib', ctypes.c_ubyte, 40), ('resv3', c.Array[ctypes.c_ubyte, Literal[3]], 41), ('offset', ctypes.c_uint32, 44), ('resv4', c.Array[ctypes.c_ubyte, Literal[16]], 48)])
@c.record
class struct_nvmf_auth_common_command(c.Struct):
  SIZE = 64
  opcode: int
  resv1: int
  command_id: int
  fctype: int
  resv2: c.Array[ctypes.c_ubyte, Literal[19]]
  dptr: union_nvme_data_ptr
  resv3: int
  spsp0: int
  spsp1: int
  secp: int
  al_tl: int
  resv4: c.Array[ctypes.c_ubyte, Literal[16]]
struct_nvmf_auth_common_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('resv1', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('fctype', ctypes.c_ubyte, 4), ('resv2', c.Array[ctypes.c_ubyte, Literal[19]], 5), ('dptr', union_nvme_data_ptr, 24), ('resv3', ctypes.c_ubyte, 40), ('spsp0', ctypes.c_ubyte, 41), ('spsp1', ctypes.c_ubyte, 42), ('secp', ctypes.c_ubyte, 43), ('al_tl', ctypes.c_uint32, 44), ('resv4', c.Array[ctypes.c_ubyte, Literal[16]], 48)])
@c.record
class struct_nvmf_auth_send_command(c.Struct):
  SIZE = 64
  opcode: int
  resv1: int
  command_id: int
  fctype: int
  resv2: c.Array[ctypes.c_ubyte, Literal[19]]
  dptr: union_nvme_data_ptr
  resv3: int
  spsp0: int
  spsp1: int
  secp: int
  tl: int
  resv4: c.Array[ctypes.c_ubyte, Literal[16]]
struct_nvmf_auth_send_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('resv1', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('fctype', ctypes.c_ubyte, 4), ('resv2', c.Array[ctypes.c_ubyte, Literal[19]], 5), ('dptr', union_nvme_data_ptr, 24), ('resv3', ctypes.c_ubyte, 40), ('spsp0', ctypes.c_ubyte, 41), ('spsp1', ctypes.c_ubyte, 42), ('secp', ctypes.c_ubyte, 43), ('tl', ctypes.c_uint32, 44), ('resv4', c.Array[ctypes.c_ubyte, Literal[16]], 48)])
@c.record
class struct_nvmf_auth_receive_command(c.Struct):
  SIZE = 64
  opcode: int
  resv1: int
  command_id: int
  fctype: int
  resv2: c.Array[ctypes.c_ubyte, Literal[19]]
  dptr: union_nvme_data_ptr
  resv3: int
  spsp0: int
  spsp1: int
  secp: int
  al: int
  resv4: c.Array[ctypes.c_ubyte, Literal[16]]
struct_nvmf_auth_receive_command.register_fields([('opcode', ctypes.c_ubyte, 0), ('resv1', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('fctype', ctypes.c_ubyte, 4), ('resv2', c.Array[ctypes.c_ubyte, Literal[19]], 5), ('dptr', union_nvme_data_ptr, 24), ('resv3', ctypes.c_ubyte, 40), ('spsp0', ctypes.c_ubyte, 41), ('spsp1', ctypes.c_ubyte, 42), ('secp', ctypes.c_ubyte, 43), ('al', ctypes.c_uint32, 44), ('resv4', c.Array[ctypes.c_ubyte, Literal[16]], 48)])
_anonenum51: dict[int, str] = {(NVME_AUTH_DHCHAP_PROTOCOL_IDENTIFIER:=233): 'NVME_AUTH_DHCHAP_PROTOCOL_IDENTIFIER'}
_anonenum52: dict[int, str] = {(NVME_AUTH_COMMON_MESSAGES:=0): 'NVME_AUTH_COMMON_MESSAGES', (NVME_AUTH_DHCHAP_MESSAGES:=1): 'NVME_AUTH_DHCHAP_MESSAGES'}
_anonenum53: dict[int, str] = {(NVME_AUTH_DHCHAP_MESSAGE_NEGOTIATE:=0): 'NVME_AUTH_DHCHAP_MESSAGE_NEGOTIATE', (NVME_AUTH_DHCHAP_MESSAGE_CHALLENGE:=1): 'NVME_AUTH_DHCHAP_MESSAGE_CHALLENGE', (NVME_AUTH_DHCHAP_MESSAGE_REPLY:=2): 'NVME_AUTH_DHCHAP_MESSAGE_REPLY', (NVME_AUTH_DHCHAP_MESSAGE_SUCCESS1:=3): 'NVME_AUTH_DHCHAP_MESSAGE_SUCCESS1', (NVME_AUTH_DHCHAP_MESSAGE_SUCCESS2:=4): 'NVME_AUTH_DHCHAP_MESSAGE_SUCCESS2', (NVME_AUTH_DHCHAP_MESSAGE_FAILURE2:=240): 'NVME_AUTH_DHCHAP_MESSAGE_FAILURE2', (NVME_AUTH_DHCHAP_MESSAGE_FAILURE1:=241): 'NVME_AUTH_DHCHAP_MESSAGE_FAILURE1'}
@c.record
class struct_nvmf_auth_dhchap_protocol_descriptor(c.Struct):
  SIZE = 64
  authid: int
  rsvd: int
  halen: int
  dhlen: int
  idlist: c.Array[ctypes.c_ubyte, Literal[60]]
struct_nvmf_auth_dhchap_protocol_descriptor.register_fields([('authid', ctypes.c_ubyte, 0), ('rsvd', ctypes.c_ubyte, 1), ('halen', ctypes.c_ubyte, 2), ('dhlen', ctypes.c_ubyte, 3), ('idlist', c.Array[ctypes.c_ubyte, Literal[60]], 4)])
_anonenum54: dict[int, str] = {(NVME_AUTH_DHCHAP_AUTH_ID:=1): 'NVME_AUTH_DHCHAP_AUTH_ID'}
_anonenum55: dict[int, str] = {(NVME_AUTH_HASH_SHA256:=1): 'NVME_AUTH_HASH_SHA256', (NVME_AUTH_HASH_SHA384:=2): 'NVME_AUTH_HASH_SHA384', (NVME_AUTH_HASH_SHA512:=3): 'NVME_AUTH_HASH_SHA512', (NVME_AUTH_HASH_INVALID:=255): 'NVME_AUTH_HASH_INVALID'}
_anonenum56: dict[int, str] = {(NVME_AUTH_DHGROUP_NULL:=0): 'NVME_AUTH_DHGROUP_NULL', (NVME_AUTH_DHGROUP_2048:=1): 'NVME_AUTH_DHGROUP_2048', (NVME_AUTH_DHGROUP_3072:=2): 'NVME_AUTH_DHGROUP_3072', (NVME_AUTH_DHGROUP_4096:=3): 'NVME_AUTH_DHGROUP_4096', (NVME_AUTH_DHGROUP_6144:=4): 'NVME_AUTH_DHGROUP_6144', (NVME_AUTH_DHGROUP_8192:=5): 'NVME_AUTH_DHGROUP_8192', (NVME_AUTH_DHGROUP_INVALID:=255): 'NVME_AUTH_DHGROUP_INVALID'}
_anonenum57: dict[int, str] = {(NVME_AUTH_SECP_NOSC:=0): 'NVME_AUTH_SECP_NOSC', (NVME_AUTH_SECP_SC:=1): 'NVME_AUTH_SECP_SC', (NVME_AUTH_SECP_NEWTLSPSK:=2): 'NVME_AUTH_SECP_NEWTLSPSK', (NVME_AUTH_SECP_REPLACETLSPSK:=3): 'NVME_AUTH_SECP_REPLACETLSPSK'}
@c.record
class union_nvmf_auth_protocol(c.Struct):
  SIZE = 64
  dhchap: struct_nvmf_auth_dhchap_protocol_descriptor
union_nvmf_auth_protocol.register_fields([('dhchap', struct_nvmf_auth_dhchap_protocol_descriptor, 0)])
@c.record
class struct_nvmf_auth_dhchap_negotiate_data(c.Struct):
  SIZE = 8
  auth_type: int
  auth_id: int
  rsvd: int
  t_id: int
  sc_c: int
  napd: int
  auth_protocol: c.Array[union_nvmf_auth_protocol, Literal[0]]
struct_nvmf_auth_dhchap_negotiate_data.register_fields([('auth_type', ctypes.c_ubyte, 0), ('auth_id', ctypes.c_ubyte, 1), ('rsvd', ctypes.c_uint16, 2), ('t_id', ctypes.c_uint16, 4), ('sc_c', ctypes.c_ubyte, 6), ('napd', ctypes.c_ubyte, 7), ('auth_protocol', c.Array[union_nvmf_auth_protocol, Literal[0]], 8)])
@c.record
class struct_nvmf_auth_dhchap_challenge_data(c.Struct):
  SIZE = 16
  auth_type: int
  auth_id: int
  rsvd1: int
  t_id: int
  hl: int
  rsvd2: int
  hashid: int
  dhgid: int
  dhvlen: int
  seqnum: int
  cval: c.Array[ctypes.c_ubyte, Literal[0]]
struct_nvmf_auth_dhchap_challenge_data.register_fields([('auth_type', ctypes.c_ubyte, 0), ('auth_id', ctypes.c_ubyte, 1), ('rsvd1', ctypes.c_uint16, 2), ('t_id', ctypes.c_uint16, 4), ('hl', ctypes.c_ubyte, 6), ('rsvd2', ctypes.c_ubyte, 7), ('hashid', ctypes.c_ubyte, 8), ('dhgid', ctypes.c_ubyte, 9), ('dhvlen', ctypes.c_uint16, 10), ('seqnum', ctypes.c_uint32, 12), ('cval', c.Array[ctypes.c_ubyte, Literal[0]], 16)])
@c.record
class struct_nvmf_auth_dhchap_reply_data(c.Struct):
  SIZE = 16
  auth_type: int
  auth_id: int
  rsvd1: int
  t_id: int
  hl: int
  rsvd2: int
  cvalid: int
  rsvd3: int
  dhvlen: int
  seqnum: int
  rval: c.Array[ctypes.c_ubyte, Literal[0]]
struct_nvmf_auth_dhchap_reply_data.register_fields([('auth_type', ctypes.c_ubyte, 0), ('auth_id', ctypes.c_ubyte, 1), ('rsvd1', ctypes.c_uint16, 2), ('t_id', ctypes.c_uint16, 4), ('hl', ctypes.c_ubyte, 6), ('rsvd2', ctypes.c_ubyte, 7), ('cvalid', ctypes.c_ubyte, 8), ('rsvd3', ctypes.c_ubyte, 9), ('dhvlen', ctypes.c_uint16, 10), ('seqnum', ctypes.c_uint32, 12), ('rval', c.Array[ctypes.c_ubyte, Literal[0]], 16)])
_anonenum58: dict[int, str] = {(NVME_AUTH_DHCHAP_RESPONSE_VALID:=1): 'NVME_AUTH_DHCHAP_RESPONSE_VALID'}
@c.record
class struct_nvmf_auth_dhchap_success1_data(c.Struct):
  SIZE = 16
  auth_type: int
  auth_id: int
  rsvd1: int
  t_id: int
  hl: int
  rsvd2: int
  rvalid: int
  rsvd3: c.Array[ctypes.c_ubyte, Literal[7]]
  rval: c.Array[ctypes.c_ubyte, Literal[0]]
struct_nvmf_auth_dhchap_success1_data.register_fields([('auth_type', ctypes.c_ubyte, 0), ('auth_id', ctypes.c_ubyte, 1), ('rsvd1', ctypes.c_uint16, 2), ('t_id', ctypes.c_uint16, 4), ('hl', ctypes.c_ubyte, 6), ('rsvd2', ctypes.c_ubyte, 7), ('rvalid', ctypes.c_ubyte, 8), ('rsvd3', c.Array[ctypes.c_ubyte, Literal[7]], 9), ('rval', c.Array[ctypes.c_ubyte, Literal[0]], 16)])
@c.record
class struct_nvmf_auth_dhchap_success2_data(c.Struct):
  SIZE = 16
  auth_type: int
  auth_id: int
  rsvd1: int
  t_id: int
  rsvd2: c.Array[ctypes.c_ubyte, Literal[10]]
struct_nvmf_auth_dhchap_success2_data.register_fields([('auth_type', ctypes.c_ubyte, 0), ('auth_id', ctypes.c_ubyte, 1), ('rsvd1', ctypes.c_uint16, 2), ('t_id', ctypes.c_uint16, 4), ('rsvd2', c.Array[ctypes.c_ubyte, Literal[10]], 6)])
@c.record
class struct_nvmf_auth_dhchap_failure_data(c.Struct):
  SIZE = 8
  auth_type: int
  auth_id: int
  rsvd1: int
  t_id: int
  rescode: int
  rescode_exp: int
struct_nvmf_auth_dhchap_failure_data.register_fields([('auth_type', ctypes.c_ubyte, 0), ('auth_id', ctypes.c_ubyte, 1), ('rsvd1', ctypes.c_uint16, 2), ('t_id', ctypes.c_uint16, 4), ('rescode', ctypes.c_ubyte, 6), ('rescode_exp', ctypes.c_ubyte, 7)])
_anonenum59: dict[int, str] = {(NVME_AUTH_DHCHAP_FAILURE_REASON_FAILED:=1): 'NVME_AUTH_DHCHAP_FAILURE_REASON_FAILED'}
_anonenum60: dict[int, str] = {(NVME_AUTH_DHCHAP_FAILURE_FAILED:=1): 'NVME_AUTH_DHCHAP_FAILURE_FAILED', (NVME_AUTH_DHCHAP_FAILURE_NOT_USABLE:=2): 'NVME_AUTH_DHCHAP_FAILURE_NOT_USABLE', (NVME_AUTH_DHCHAP_FAILURE_CONCAT_MISMATCH:=3): 'NVME_AUTH_DHCHAP_FAILURE_CONCAT_MISMATCH', (NVME_AUTH_DHCHAP_FAILURE_HASH_UNUSABLE:=4): 'NVME_AUTH_DHCHAP_FAILURE_HASH_UNUSABLE', (NVME_AUTH_DHCHAP_FAILURE_DHGROUP_UNUSABLE:=5): 'NVME_AUTH_DHCHAP_FAILURE_DHGROUP_UNUSABLE', (NVME_AUTH_DHCHAP_FAILURE_INCORRECT_PAYLOAD:=6): 'NVME_AUTH_DHCHAP_FAILURE_INCORRECT_PAYLOAD', (NVME_AUTH_DHCHAP_FAILURE_INCORRECT_MESSAGE:=7): 'NVME_AUTH_DHCHAP_FAILURE_INCORRECT_MESSAGE'}
@c.record
class struct_nvme_dbbuf(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  command_id: int
  rsvd1: c.Array[ctypes.c_uint32, Literal[5]]
  prp1: int
  prp2: int
  rsvd12: c.Array[ctypes.c_uint32, Literal[6]]
struct_nvme_dbbuf.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('command_id', ctypes.c_uint16, 2), ('rsvd1', c.Array[ctypes.c_uint32, Literal[5]], 4), ('prp1', ctypes.c_uint64, 24), ('prp2', ctypes.c_uint64, 32), ('rsvd12', c.Array[ctypes.c_uint32, Literal[6]], 40)])
@c.record
class struct_streams_directive_params(c.Struct):
  SIZE = 32
  msl: int
  nssa: int
  nsso: int
  rsvd: c.Array[ctypes.c_ubyte, Literal[10]]
  sws: int
  sgs: int
  nsa: int
  nso: int
  rsvd2: c.Array[ctypes.c_ubyte, Literal[6]]
struct_streams_directive_params.register_fields([('msl', ctypes.c_uint16, 0), ('nssa', ctypes.c_uint16, 2), ('nsso', ctypes.c_uint16, 4), ('rsvd', c.Array[ctypes.c_ubyte, Literal[10]], 6), ('sws', ctypes.c_uint32, 16), ('sgs', ctypes.c_uint16, 20), ('nsa', ctypes.c_uint16, 22), ('nso', ctypes.c_uint16, 24), ('rsvd2', c.Array[ctypes.c_ubyte, Literal[6]], 26)])
@c.record
class struct_nvme_command(c.Struct):
  SIZE = 64
  common: struct_nvme_common_command
  rw: struct_nvme_rw_command
  identify: struct_nvme_identify
  features: struct_nvme_features
  create_cq: struct_nvme_create_cq
  create_sq: struct_nvme_create_sq
  delete_queue: struct_nvme_delete_queue
  dlfw: struct_nvme_download_firmware
  format: struct_nvme_format_cmd
  dsm: struct_nvme_dsm_cmd
  write_zeroes: struct_nvme_write_zeroes_cmd
  zms: struct_nvme_zone_mgmt_send_cmd
  zmr: struct_nvme_zone_mgmt_recv_cmd
  abort: struct_nvme_abort_cmd
  get_log_page: struct_nvme_get_log_page_command
  fabrics: struct_nvmf_common_command
  connect: struct_nvmf_connect_command
  prop_set: struct_nvmf_property_set_command
  prop_get: struct_nvmf_property_get_command
  auth_common: struct_nvmf_auth_common_command
  auth_send: struct_nvmf_auth_send_command
  auth_receive: struct_nvmf_auth_receive_command
  dbbuf: struct_nvme_dbbuf
  directive: struct_nvme_directive_cmd
  imr: struct_nvme_io_mgmt_recv_cmd
struct_nvme_command.register_fields([('common', struct_nvme_common_command, 0), ('rw', struct_nvme_rw_command, 0), ('identify', struct_nvme_identify, 0), ('features', struct_nvme_features, 0), ('create_cq', struct_nvme_create_cq, 0), ('create_sq', struct_nvme_create_sq, 0), ('delete_queue', struct_nvme_delete_queue, 0), ('dlfw', struct_nvme_download_firmware, 0), ('format', struct_nvme_format_cmd, 0), ('dsm', struct_nvme_dsm_cmd, 0), ('write_zeroes', struct_nvme_write_zeroes_cmd, 0), ('zms', struct_nvme_zone_mgmt_send_cmd, 0), ('zmr', struct_nvme_zone_mgmt_recv_cmd, 0), ('abort', struct_nvme_abort_cmd, 0), ('get_log_page', struct_nvme_get_log_page_command, 0), ('fabrics', struct_nvmf_common_command, 0), ('connect', struct_nvmf_connect_command, 0), ('prop_set', struct_nvmf_property_set_command, 0), ('prop_get', struct_nvmf_property_get_command, 0), ('auth_common', struct_nvmf_auth_common_command, 0), ('auth_send', struct_nvmf_auth_send_command, 0), ('auth_receive', struct_nvmf_auth_receive_command, 0), ('dbbuf', struct_nvme_dbbuf, 0), ('directive', struct_nvme_directive_cmd, 0), ('imr', struct_nvme_io_mgmt_recv_cmd, 0)])
@c.record
class struct_nvme_error_slot(c.Struct):
  SIZE = 64
  error_count: int
  sqid: int
  cmdid: int
  status_field: int
  param_error_location: int
  lba: int
  nsid: int
  vs: int
  resv: c.Array[ctypes.c_ubyte, Literal[3]]
  cs: int
  resv2: c.Array[ctypes.c_ubyte, Literal[24]]
struct_nvme_error_slot.register_fields([('error_count', ctypes.c_uint64, 0), ('sqid', ctypes.c_uint16, 8), ('cmdid', ctypes.c_uint16, 10), ('status_field', ctypes.c_uint16, 12), ('param_error_location', ctypes.c_uint16, 14), ('lba', ctypes.c_uint64, 16), ('nsid', ctypes.c_uint32, 24), ('vs', ctypes.c_ubyte, 28), ('resv', c.Array[ctypes.c_ubyte, Literal[3]], 29), ('cs', ctypes.c_uint64, 32), ('resv2', c.Array[ctypes.c_ubyte, Literal[24]], 40)])
_anonenum61: dict[int, str] = {(NVME_SCT_GENERIC:=0): 'NVME_SCT_GENERIC', (NVME_SC_SUCCESS:=0): 'NVME_SC_SUCCESS', (NVME_SC_INVALID_OPCODE:=1): 'NVME_SC_INVALID_OPCODE', (NVME_SC_INVALID_FIELD:=2): 'NVME_SC_INVALID_FIELD', (NVME_SC_CMDID_CONFLICT:=3): 'NVME_SC_CMDID_CONFLICT', (NVME_SC_DATA_XFER_ERROR:=4): 'NVME_SC_DATA_XFER_ERROR', (NVME_SC_POWER_LOSS:=5): 'NVME_SC_POWER_LOSS', (NVME_SC_INTERNAL:=6): 'NVME_SC_INTERNAL', (NVME_SC_ABORT_REQ:=7): 'NVME_SC_ABORT_REQ', (NVME_SC_ABORT_QUEUE:=8): 'NVME_SC_ABORT_QUEUE', (NVME_SC_FUSED_FAIL:=9): 'NVME_SC_FUSED_FAIL', (NVME_SC_FUSED_MISSING:=10): 'NVME_SC_FUSED_MISSING', (NVME_SC_INVALID_NS:=11): 'NVME_SC_INVALID_NS', (NVME_SC_CMD_SEQ_ERROR:=12): 'NVME_SC_CMD_SEQ_ERROR', (NVME_SC_SGL_INVALID_LAST:=13): 'NVME_SC_SGL_INVALID_LAST', (NVME_SC_SGL_INVALID_COUNT:=14): 'NVME_SC_SGL_INVALID_COUNT', (NVME_SC_SGL_INVALID_DATA:=15): 'NVME_SC_SGL_INVALID_DATA', (NVME_SC_SGL_INVALID_METADATA:=16): 'NVME_SC_SGL_INVALID_METADATA', (NVME_SC_SGL_INVALID_TYPE:=17): 'NVME_SC_SGL_INVALID_TYPE', (NVME_SC_CMB_INVALID_USE:=18): 'NVME_SC_CMB_INVALID_USE', (NVME_SC_PRP_INVALID_OFFSET:=19): 'NVME_SC_PRP_INVALID_OFFSET', (NVME_SC_ATOMIC_WU_EXCEEDED:=20): 'NVME_SC_ATOMIC_WU_EXCEEDED', (NVME_SC_OP_DENIED:=21): 'NVME_SC_OP_DENIED', (NVME_SC_SGL_INVALID_OFFSET:=22): 'NVME_SC_SGL_INVALID_OFFSET', (NVME_SC_RESERVED:=23): 'NVME_SC_RESERVED', (NVME_SC_HOST_ID_INCONSIST:=24): 'NVME_SC_HOST_ID_INCONSIST', (NVME_SC_KA_TIMEOUT_EXPIRED:=25): 'NVME_SC_KA_TIMEOUT_EXPIRED', (NVME_SC_KA_TIMEOUT_INVALID:=26): 'NVME_SC_KA_TIMEOUT_INVALID', (NVME_SC_ABORTED_PREEMPT_ABORT:=27): 'NVME_SC_ABORTED_PREEMPT_ABORT', (NVME_SC_SANITIZE_FAILED:=28): 'NVME_SC_SANITIZE_FAILED', (NVME_SC_SANITIZE_IN_PROGRESS:=29): 'NVME_SC_SANITIZE_IN_PROGRESS', (NVME_SC_SGL_INVALID_GRANULARITY:=30): 'NVME_SC_SGL_INVALID_GRANULARITY', (NVME_SC_CMD_NOT_SUP_CMB_QUEUE:=31): 'NVME_SC_CMD_NOT_SUP_CMB_QUEUE', (NVME_SC_NS_WRITE_PROTECTED:=32): 'NVME_SC_NS_WRITE_PROTECTED', (NVME_SC_CMD_INTERRUPTED:=33): 'NVME_SC_CMD_INTERRUPTED', (NVME_SC_TRANSIENT_TR_ERR:=34): 'NVME_SC_TRANSIENT_TR_ERR', (NVME_SC_ADMIN_COMMAND_MEDIA_NOT_READY:=36): 'NVME_SC_ADMIN_COMMAND_MEDIA_NOT_READY', (NVME_SC_INVALID_IO_CMD_SET:=44): 'NVME_SC_INVALID_IO_CMD_SET', (NVME_SC_LBA_RANGE:=128): 'NVME_SC_LBA_RANGE', (NVME_SC_CAP_EXCEEDED:=129): 'NVME_SC_CAP_EXCEEDED', (NVME_SC_NS_NOT_READY:=130): 'NVME_SC_NS_NOT_READY', (NVME_SC_RESERVATION_CONFLICT:=131): 'NVME_SC_RESERVATION_CONFLICT', (NVME_SC_FORMAT_IN_PROGRESS:=132): 'NVME_SC_FORMAT_IN_PROGRESS', (NVME_SCT_COMMAND_SPECIFIC:=256): 'NVME_SCT_COMMAND_SPECIFIC', (NVME_SC_CQ_INVALID:=256): 'NVME_SC_CQ_INVALID', (NVME_SC_QID_INVALID:=257): 'NVME_SC_QID_INVALID', (NVME_SC_QUEUE_SIZE:=258): 'NVME_SC_QUEUE_SIZE', (NVME_SC_ABORT_LIMIT:=259): 'NVME_SC_ABORT_LIMIT', (NVME_SC_ABORT_MISSING:=260): 'NVME_SC_ABORT_MISSING', (NVME_SC_ASYNC_LIMIT:=261): 'NVME_SC_ASYNC_LIMIT', (NVME_SC_FIRMWARE_SLOT:=262): 'NVME_SC_FIRMWARE_SLOT', (NVME_SC_FIRMWARE_IMAGE:=263): 'NVME_SC_FIRMWARE_IMAGE', (NVME_SC_INVALID_VECTOR:=264): 'NVME_SC_INVALID_VECTOR', (NVME_SC_INVALID_LOG_PAGE:=265): 'NVME_SC_INVALID_LOG_PAGE', (NVME_SC_INVALID_FORMAT:=266): 'NVME_SC_INVALID_FORMAT', (NVME_SC_FW_NEEDS_CONV_RESET:=267): 'NVME_SC_FW_NEEDS_CONV_RESET', (NVME_SC_INVALID_QUEUE:=268): 'NVME_SC_INVALID_QUEUE', (NVME_SC_FEATURE_NOT_SAVEABLE:=269): 'NVME_SC_FEATURE_NOT_SAVEABLE', (NVME_SC_FEATURE_NOT_CHANGEABLE:=270): 'NVME_SC_FEATURE_NOT_CHANGEABLE', (NVME_SC_FEATURE_NOT_PER_NS:=271): 'NVME_SC_FEATURE_NOT_PER_NS', (NVME_SC_FW_NEEDS_SUBSYS_RESET:=272): 'NVME_SC_FW_NEEDS_SUBSYS_RESET', (NVME_SC_FW_NEEDS_RESET:=273): 'NVME_SC_FW_NEEDS_RESET', (NVME_SC_FW_NEEDS_MAX_TIME:=274): 'NVME_SC_FW_NEEDS_MAX_TIME', (NVME_SC_FW_ACTIVATE_PROHIBITED:=275): 'NVME_SC_FW_ACTIVATE_PROHIBITED', (NVME_SC_OVERLAPPING_RANGE:=276): 'NVME_SC_OVERLAPPING_RANGE', (NVME_SC_NS_INSUFFICIENT_CAP:=277): 'NVME_SC_NS_INSUFFICIENT_CAP', (NVME_SC_NS_ID_UNAVAILABLE:=278): 'NVME_SC_NS_ID_UNAVAILABLE', (NVME_SC_NS_ALREADY_ATTACHED:=280): 'NVME_SC_NS_ALREADY_ATTACHED', (NVME_SC_NS_IS_PRIVATE:=281): 'NVME_SC_NS_IS_PRIVATE', (NVME_SC_NS_NOT_ATTACHED:=282): 'NVME_SC_NS_NOT_ATTACHED', (NVME_SC_THIN_PROV_NOT_SUPP:=283): 'NVME_SC_THIN_PROV_NOT_SUPP', (NVME_SC_CTRL_LIST_INVALID:=284): 'NVME_SC_CTRL_LIST_INVALID', (NVME_SC_SELF_TEST_IN_PROGRESS:=285): 'NVME_SC_SELF_TEST_IN_PROGRESS', (NVME_SC_BP_WRITE_PROHIBITED:=286): 'NVME_SC_BP_WRITE_PROHIBITED', (NVME_SC_CTRL_ID_INVALID:=287): 'NVME_SC_CTRL_ID_INVALID', (NVME_SC_SEC_CTRL_STATE_INVALID:=288): 'NVME_SC_SEC_CTRL_STATE_INVALID', (NVME_SC_CTRL_RES_NUM_INVALID:=289): 'NVME_SC_CTRL_RES_NUM_INVALID', (NVME_SC_RES_ID_INVALID:=290): 'NVME_SC_RES_ID_INVALID', (NVME_SC_PMR_SAN_PROHIBITED:=291): 'NVME_SC_PMR_SAN_PROHIBITED', (NVME_SC_ANA_GROUP_ID_INVALID:=292): 'NVME_SC_ANA_GROUP_ID_INVALID', (NVME_SC_ANA_ATTACH_FAILED:=293): 'NVME_SC_ANA_ATTACH_FAILED', (NVME_SC_BAD_ATTRIBUTES:=384): 'NVME_SC_BAD_ATTRIBUTES', (NVME_SC_INVALID_PI:=385): 'NVME_SC_INVALID_PI', (NVME_SC_READ_ONLY:=386): 'NVME_SC_READ_ONLY', (NVME_SC_CMD_SIZE_LIM_EXCEEDED:=387): 'NVME_SC_CMD_SIZE_LIM_EXCEEDED', (NVME_SC_CONNECT_FORMAT:=384): 'NVME_SC_CONNECT_FORMAT', (NVME_SC_CONNECT_CTRL_BUSY:=385): 'NVME_SC_CONNECT_CTRL_BUSY', (NVME_SC_CONNECT_INVALID_PARAM:=386): 'NVME_SC_CONNECT_INVALID_PARAM', (NVME_SC_CONNECT_RESTART_DISC:=387): 'NVME_SC_CONNECT_RESTART_DISC', (NVME_SC_CONNECT_INVALID_HOST:=388): 'NVME_SC_CONNECT_INVALID_HOST', (NVME_SC_DISCOVERY_RESTART:=400): 'NVME_SC_DISCOVERY_RESTART', (NVME_SC_AUTH_REQUIRED:=401): 'NVME_SC_AUTH_REQUIRED', (NVME_SC_ZONE_BOUNDARY_ERROR:=440): 'NVME_SC_ZONE_BOUNDARY_ERROR', (NVME_SC_ZONE_FULL:=441): 'NVME_SC_ZONE_FULL', (NVME_SC_ZONE_READ_ONLY:=442): 'NVME_SC_ZONE_READ_ONLY', (NVME_SC_ZONE_OFFLINE:=443): 'NVME_SC_ZONE_OFFLINE', (NVME_SC_ZONE_INVALID_WRITE:=444): 'NVME_SC_ZONE_INVALID_WRITE', (NVME_SC_ZONE_TOO_MANY_ACTIVE:=445): 'NVME_SC_ZONE_TOO_MANY_ACTIVE', (NVME_SC_ZONE_TOO_MANY_OPEN:=446): 'NVME_SC_ZONE_TOO_MANY_OPEN', (NVME_SC_ZONE_INVALID_TRANSITION:=447): 'NVME_SC_ZONE_INVALID_TRANSITION', (NVME_SCT_MEDIA_ERROR:=512): 'NVME_SCT_MEDIA_ERROR', (NVME_SC_WRITE_FAULT:=640): 'NVME_SC_WRITE_FAULT', (NVME_SC_READ_ERROR:=641): 'NVME_SC_READ_ERROR', (NVME_SC_GUARD_CHECK:=642): 'NVME_SC_GUARD_CHECK', (NVME_SC_APPTAG_CHECK:=643): 'NVME_SC_APPTAG_CHECK', (NVME_SC_REFTAG_CHECK:=644): 'NVME_SC_REFTAG_CHECK', (NVME_SC_COMPARE_FAILED:=645): 'NVME_SC_COMPARE_FAILED', (NVME_SC_ACCESS_DENIED:=646): 'NVME_SC_ACCESS_DENIED', (NVME_SC_UNWRITTEN_BLOCK:=647): 'NVME_SC_UNWRITTEN_BLOCK', (NVME_SCT_PATH:=768): 'NVME_SCT_PATH', (NVME_SC_INTERNAL_PATH_ERROR:=768): 'NVME_SC_INTERNAL_PATH_ERROR', (NVME_SC_ANA_PERSISTENT_LOSS:=769): 'NVME_SC_ANA_PERSISTENT_LOSS', (NVME_SC_ANA_INACCESSIBLE:=770): 'NVME_SC_ANA_INACCESSIBLE', (NVME_SC_ANA_TRANSITION:=771): 'NVME_SC_ANA_TRANSITION', (NVME_SC_CTRL_PATH_ERROR:=864): 'NVME_SC_CTRL_PATH_ERROR', (NVME_SC_HOST_PATH_ERROR:=880): 'NVME_SC_HOST_PATH_ERROR', (NVME_SC_HOST_ABORTED_CMD:=881): 'NVME_SC_HOST_ABORTED_CMD', (NVME_SC_MASK:=255): 'NVME_SC_MASK', (NVME_SCT_MASK:=1792): 'NVME_SCT_MASK', (NVME_SCT_SC_MASK:=2047): 'NVME_SCT_SC_MASK', (NVME_STATUS_CRD:=6144): 'NVME_STATUS_CRD', (NVME_STATUS_MORE:=8192): 'NVME_STATUS_MORE', (NVME_STATUS_DNR:=16384): 'NVME_STATUS_DNR'}
@c.record
class struct_nvme_completion(c.Struct):
  SIZE = 16
  result: union_nvme_result
  sq_head: int
  sq_id: int
  command_id: int
  status: int
@c.record
class union_nvme_result(c.Struct):
  SIZE = 8
  u16: int
  u32: int
  u64: int
union_nvme_result.register_fields([('u16', ctypes.c_uint16, 0), ('u32', ctypes.c_uint32, 0), ('u64', ctypes.c_uint64, 0)])
struct_nvme_completion.register_fields([('result', union_nvme_result, 0), ('sq_head', ctypes.c_uint16, 8), ('sq_id', ctypes.c_uint16, 10), ('command_id', ctypes.c_uint16, 12), ('status', ctypes.c_uint16, 14)])
_anonenum62: dict[int, str] = {(NVME_AEN_RESV_LOG_PAGE_AVALIABLE:=0): 'NVME_AEN_RESV_LOG_PAGE_AVALIABLE'}
_anonenum63: dict[int, str] = {(NVME_PR_LOG_EMPTY_LOG_PAGE:=0): 'NVME_PR_LOG_EMPTY_LOG_PAGE', (NVME_PR_LOG_REGISTRATION_PREEMPTED:=1): 'NVME_PR_LOG_REGISTRATION_PREEMPTED', (NVME_PR_LOG_RESERVATION_RELEASED:=2): 'NVME_PR_LOG_RESERVATION_RELEASED', (NVME_PR_LOG_RESERVATOIN_PREEMPTED:=3): 'NVME_PR_LOG_RESERVATOIN_PREEMPTED'}
_anonenum64: dict[int, str] = {(NVME_PR_NOTIFY_BIT_REG_PREEMPTED:=1): 'NVME_PR_NOTIFY_BIT_REG_PREEMPTED', (NVME_PR_NOTIFY_BIT_RESV_RELEASED:=2): 'NVME_PR_NOTIFY_BIT_RESV_RELEASED', (NVME_PR_NOTIFY_BIT_RESV_PREEMPTED:=3): 'NVME_PR_NOTIFY_BIT_RESV_PREEMPTED'}
@c.record
class struct_nvme_pr_log(c.Struct):
  SIZE = 64
  count: int
  type: int
  nr_pages: int
  rsvd1: c.Array[ctypes.c_ubyte, Literal[2]]
  nsid: int
  rsvd2: c.Array[ctypes.c_ubyte, Literal[48]]
struct_nvme_pr_log.register_fields([('count', ctypes.c_uint64, 0), ('type', ctypes.c_ubyte, 8), ('nr_pages', ctypes.c_ubyte, 9), ('rsvd1', c.Array[ctypes.c_ubyte, Literal[2]], 10), ('nsid', ctypes.c_uint32, 12), ('rsvd2', c.Array[ctypes.c_ubyte, Literal[48]], 16)])
@c.record
class struct_nvmet_pr_register_data(c.Struct):
  SIZE = 16
  crkey: int
  nrkey: int
struct_nvmet_pr_register_data.register_fields([('crkey', ctypes.c_uint64, 0), ('nrkey', ctypes.c_uint64, 8)])
@c.record
class struct_nvmet_pr_acquire_data(c.Struct):
  SIZE = 16
  crkey: int
  prkey: int
struct_nvmet_pr_acquire_data.register_fields([('crkey', ctypes.c_uint64, 0), ('prkey', ctypes.c_uint64, 8)])
@c.record
class struct_nvmet_pr_release_data(c.Struct):
  SIZE = 8
  crkey: int
struct_nvmet_pr_release_data.register_fields([('crkey', ctypes.c_uint64, 0)])
enum_nvme_pr_capabilities: dict[int, str] = {(NVME_PR_SUPPORT_PTPL:=1): 'NVME_PR_SUPPORT_PTPL', (NVME_PR_SUPPORT_WRITE_EXCLUSIVE:=2): 'NVME_PR_SUPPORT_WRITE_EXCLUSIVE', (NVME_PR_SUPPORT_EXCLUSIVE_ACCESS:=4): 'NVME_PR_SUPPORT_EXCLUSIVE_ACCESS', (NVME_PR_SUPPORT_WRITE_EXCLUSIVE_REG_ONLY:=8): 'NVME_PR_SUPPORT_WRITE_EXCLUSIVE_REG_ONLY', (NVME_PR_SUPPORT_EXCLUSIVE_ACCESS_REG_ONLY:=16): 'NVME_PR_SUPPORT_EXCLUSIVE_ACCESS_REG_ONLY', (NVME_PR_SUPPORT_WRITE_EXCLUSIVE_ALL_REGS:=32): 'NVME_PR_SUPPORT_WRITE_EXCLUSIVE_ALL_REGS', (NVME_PR_SUPPORT_EXCLUSIVE_ACCESS_ALL_REGS:=64): 'NVME_PR_SUPPORT_EXCLUSIVE_ACCESS_ALL_REGS', (NVME_PR_SUPPORT_IEKEY_VER_1_3_DEF:=128): 'NVME_PR_SUPPORT_IEKEY_VER_1_3_DEF'}
enum_nvme_pr_register_action: dict[int, str] = {(NVME_PR_REGISTER_ACT_REG:=0): 'NVME_PR_REGISTER_ACT_REG', (NVME_PR_REGISTER_ACT_UNREG:=1): 'NVME_PR_REGISTER_ACT_UNREG', (NVME_PR_REGISTER_ACT_REPLACE:=2): 'NVME_PR_REGISTER_ACT_REPLACE'}
enum_nvme_pr_acquire_action: dict[int, str] = {(NVME_PR_ACQUIRE_ACT_ACQUIRE:=0): 'NVME_PR_ACQUIRE_ACT_ACQUIRE', (NVME_PR_ACQUIRE_ACT_PREEMPT:=1): 'NVME_PR_ACQUIRE_ACT_PREEMPT', (NVME_PR_ACQUIRE_ACT_PREEMPT_AND_ABORT:=2): 'NVME_PR_ACQUIRE_ACT_PREEMPT_AND_ABORT'}
enum_nvme_pr_release_action: dict[int, str] = {(NVME_PR_RELEASE_ACT_RELEASE:=0): 'NVME_PR_RELEASE_ACT_RELEASE', (NVME_PR_RELEASE_ACT_CLEAR:=1): 'NVME_PR_RELEASE_ACT_CLEAR'}
enum_nvme_pr_change_ptpl: dict[int, str] = {(NVME_PR_CPTPL_NO_CHANGE:=0): 'NVME_PR_CPTPL_NO_CHANGE', (NVME_PR_CPTPL_RESV:=1073741824): 'NVME_PR_CPTPL_RESV', (NVME_PR_CPTPL_CLEARED:=-2147483648): 'NVME_PR_CPTPL_CLEARED', (NVME_PR_CPTPL_PERSIST:=-1073741824): 'NVME_PR_CPTPL_PERSIST'}
NVMF_NQN_FIELD_LEN = 256
NVMF_NQN_SIZE = 223
NVMF_TRSVCID_SIZE = 32
NVMF_TRADDR_SIZE = 256
NVMF_TSAS_SIZE = 256
NVME_DISC_SUBSYS_NAME = "nqn.2014-8.org.nvmexpress.discovery"
NVME_NSID_ALL = 0xffffffff
NVME_SUBSYS_RESET = 0x4E564D65
NVME_TREQ_SECURE_CHANNEL_MASK = (NVMF_TREQ_REQUIRED | NVMF_TREQ_NOT_REQUIRED)
NVME_AQ_DEPTH = 32
NVME_NR_AEN_COMMANDS = 1
NVME_AQ_BLK_MQ_DEPTH = (NVME_AQ_DEPTH - NVME_NR_AEN_COMMANDS)
NVME_AQ_MQ_TAG_DEPTH = (NVME_AQ_BLK_MQ_DEPTH - 1)
NVME_CAP_MQES = lambda cap: ((cap) & 0xffff) # type: ignore
NVME_CAP_TIMEOUT = lambda cap: (((cap) >> 24) & 0xff) # type: ignore
NVME_CAP_STRIDE = lambda cap: (((cap) >> 32) & 0xf) # type: ignore
NVME_CAP_NSSRC = lambda cap: (((cap) >> 36) & 0x1) # type: ignore
NVME_CAP_CSS = lambda cap: (((cap) >> 37) & 0xff) # type: ignore
NVME_CAP_MPSMIN = lambda cap: (((cap) >> 48) & 0xf) # type: ignore
NVME_CAP_MPSMAX = lambda cap: (((cap) >> 52) & 0xf) # type: ignore
NVME_CAP_CMBS = lambda cap: (((cap) >> 57) & 0x1) # type: ignore
NVME_CMB_BIR = lambda cmbloc: ((cmbloc) & 0x7) # type: ignore
NVME_CMB_OFST = lambda cmbloc: (((cmbloc) >> 12) & 0xfffff) # type: ignore
NVME_CRTO_CRIMT = lambda crto: ((crto) >> 16) # type: ignore
NVME_CRTO_CRWMT = lambda crto: ((crto) & 0xffff) # type: ignore
NVME_ADM_SQES = 6
NVME_NVM_IOSQES = 6
NVME_NVM_IOCQES = 4
NVME_NIDT_EUI64_LEN = 8
NVME_NIDT_NGUID_LEN = 16
NVME_NIDT_UUID_LEN = 16
NVME_NIDT_CSI_LEN = 1
FDPCFG_FDPE = (1 << 0)
NVME_ANA_LOG_RGO = (1 << 0)
show_fabrics_type_name = lambda type: __print_symbolic(type, nvme_fabrics_type_name(nvme_fabrics_type_property_set), nvme_fabrics_type_name(nvme_fabrics_type_connect), nvme_fabrics_type_name(nvme_fabrics_type_property_get), nvme_fabrics_type_name(nvme_fabrics_type_auth_send), nvme_fabrics_type_name(nvme_fabrics_type_auth_receive)) # type: ignore
NVME_CNTLID_MIN = 1
NVME_CNTLID_MAX = 0xffef
NVME_CNTLID_DYNAMIC = 0xffff
MAX_DISC_LOGS = 255
NVME_SCT = lambda status: ((status) >> 8 & 7) # type: ignore
NVME_VS = lambda major,minor,tertiary: (((major) << 16) | ((minor) << 8) | (tertiary)) # type: ignore
NVME_MAJOR = lambda ver: ((ver) >> 16) # type: ignore
NVME_MINOR = lambda ver: (((ver) >> 8) & 0xff) # type: ignore
NVME_TERTIARY = lambda ver: ((ver) & 0xff) # type: ignore
NVME_PR_IGNORE_KEY = (1 << 3)