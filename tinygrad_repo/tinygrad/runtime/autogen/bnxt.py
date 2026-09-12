# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_hwrm_cmd_hdr(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_cmd_hdr.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_resp_hdr(c.Struct):
  SIZE = 8
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
struct_hwrm_resp_hdr.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6)])
@c.record
class struct_hwrm_ver_get_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  hwrm_intf_maj: int
  hwrm_intf_min: int
  hwrm_intf_upd: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
struct_hwrm_ver_get_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('hwrm_intf_maj', ctypes.c_ubyte, 16), ('hwrm_intf_min', ctypes.c_ubyte, 17), ('hwrm_intf_upd', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 19)])
@c.record
class struct_hwrm_ver_get_output(c.Struct):
  SIZE = 176
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  hwrm_intf_maj_8b: int
  hwrm_intf_min_8b: int
  hwrm_intf_upd_8b: int
  hwrm_intf_rsvd_8b: int
  hwrm_fw_maj_8b: int
  hwrm_fw_min_8b: int
  hwrm_fw_bld_8b: int
  hwrm_fw_rsvd_8b: int
  mgmt_fw_maj_8b: int
  mgmt_fw_min_8b: int
  mgmt_fw_bld_8b: int
  mgmt_fw_rsvd_8b: int
  netctrl_fw_maj_8b: int
  netctrl_fw_min_8b: int
  netctrl_fw_bld_8b: int
  netctrl_fw_rsvd_8b: int
  dev_caps_cfg: int
  roce_fw_maj_8b: int
  roce_fw_min_8b: int
  roce_fw_bld_8b: int
  roce_fw_rsvd_8b: int
  hwrm_fw_name: c.Array[ctypes.c_char, Literal[16]]
  mgmt_fw_name: c.Array[ctypes.c_char, Literal[16]]
  netctrl_fw_name: c.Array[ctypes.c_char, Literal[16]]
  active_pkg_name: c.Array[ctypes.c_char, Literal[16]]
  roce_fw_name: c.Array[ctypes.c_char, Literal[16]]
  chip_num: int
  chip_rev: int
  chip_metal: int
  chip_bond_id: int
  chip_platform_type: int
  max_req_win_len: int
  max_resp_len: int
  def_req_timeout: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  always_1: int
  hwrm_intf_major: int
  hwrm_intf_minor: int
  hwrm_intf_build: int
  hwrm_intf_patch: int
  hwrm_fw_major: int
  hwrm_fw_minor: int
  hwrm_fw_build: int
  hwrm_fw_patch: int
  mgmt_fw_major: int
  mgmt_fw_minor: int
  mgmt_fw_build: int
  mgmt_fw_patch: int
  netctrl_fw_major: int
  netctrl_fw_minor: int
  netctrl_fw_build: int
  netctrl_fw_patch: int
  roce_fw_major: int
  roce_fw_minor: int
  roce_fw_build: int
  roce_fw_patch: int
  max_ext_req_len: int
  max_req_timeout: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_ver_get_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('hwrm_intf_maj_8b', ctypes.c_ubyte, 8), ('hwrm_intf_min_8b', ctypes.c_ubyte, 9), ('hwrm_intf_upd_8b', ctypes.c_ubyte, 10), ('hwrm_intf_rsvd_8b', ctypes.c_ubyte, 11), ('hwrm_fw_maj_8b', ctypes.c_ubyte, 12), ('hwrm_fw_min_8b', ctypes.c_ubyte, 13), ('hwrm_fw_bld_8b', ctypes.c_ubyte, 14), ('hwrm_fw_rsvd_8b', ctypes.c_ubyte, 15), ('mgmt_fw_maj_8b', ctypes.c_ubyte, 16), ('mgmt_fw_min_8b', ctypes.c_ubyte, 17), ('mgmt_fw_bld_8b', ctypes.c_ubyte, 18), ('mgmt_fw_rsvd_8b', ctypes.c_ubyte, 19), ('netctrl_fw_maj_8b', ctypes.c_ubyte, 20), ('netctrl_fw_min_8b', ctypes.c_ubyte, 21), ('netctrl_fw_bld_8b', ctypes.c_ubyte, 22), ('netctrl_fw_rsvd_8b', ctypes.c_ubyte, 23), ('dev_caps_cfg', ctypes.c_uint32, 24), ('roce_fw_maj_8b', ctypes.c_ubyte, 28), ('roce_fw_min_8b', ctypes.c_ubyte, 29), ('roce_fw_bld_8b', ctypes.c_ubyte, 30), ('roce_fw_rsvd_8b', ctypes.c_ubyte, 31), ('hwrm_fw_name', c.Array[ctypes.c_char, Literal[16]], 32), ('mgmt_fw_name', c.Array[ctypes.c_char, Literal[16]], 48), ('netctrl_fw_name', c.Array[ctypes.c_char, Literal[16]], 64), ('active_pkg_name', c.Array[ctypes.c_char, Literal[16]], 80), ('roce_fw_name', c.Array[ctypes.c_char, Literal[16]], 96), ('chip_num', ctypes.c_uint16, 112), ('chip_rev', ctypes.c_ubyte, 114), ('chip_metal', ctypes.c_ubyte, 115), ('chip_bond_id', ctypes.c_ubyte, 116), ('chip_platform_type', ctypes.c_ubyte, 117), ('max_req_win_len', ctypes.c_uint16, 118), ('max_resp_len', ctypes.c_uint16, 120), ('def_req_timeout', ctypes.c_uint16, 122), ('flags', ctypes.c_ubyte, 124), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 125), ('always_1', ctypes.c_ubyte, 127), ('hwrm_intf_major', ctypes.c_uint16, 128), ('hwrm_intf_minor', ctypes.c_uint16, 130), ('hwrm_intf_build', ctypes.c_uint16, 132), ('hwrm_intf_patch', ctypes.c_uint16, 134), ('hwrm_fw_major', ctypes.c_uint16, 136), ('hwrm_fw_minor', ctypes.c_uint16, 138), ('hwrm_fw_build', ctypes.c_uint16, 140), ('hwrm_fw_patch', ctypes.c_uint16, 142), ('mgmt_fw_major', ctypes.c_uint16, 144), ('mgmt_fw_minor', ctypes.c_uint16, 146), ('mgmt_fw_build', ctypes.c_uint16, 148), ('mgmt_fw_patch', ctypes.c_uint16, 150), ('netctrl_fw_major', ctypes.c_uint16, 152), ('netctrl_fw_minor', ctypes.c_uint16, 154), ('netctrl_fw_build', ctypes.c_uint16, 156), ('netctrl_fw_patch', ctypes.c_uint16, 158), ('roce_fw_major', ctypes.c_uint16, 160), ('roce_fw_minor', ctypes.c_uint16, 162), ('roce_fw_build', ctypes.c_uint16, 164), ('roce_fw_patch', ctypes.c_uint16, 166), ('max_ext_req_len', ctypes.c_uint16, 168), ('max_req_timeout', ctypes.c_uint16, 170), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 172), ('valid', ctypes.c_ubyte, 175)])
@c.record
class struct_hwrm_func_reset_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  vf_id: int
  func_reset_level: int
  unused_0: int
struct_hwrm_func_reset_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('vf_id', ctypes.c_uint16, 20), ('func_reset_level', ctypes.c_ubyte, 22), ('unused_0', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_func_reset_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_reset_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_func_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_func_qcaps_output(c.Struct):
  SIZE = 144
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  fid: int
  port_id: int
  flags: int
  mac_address: c.Array[ctypes.c_ubyte, Literal[6]]
  max_rsscos_ctx: int
  max_cmpl_rings: int
  max_tx_rings: int
  max_rx_rings: int
  max_l2_ctxs: int
  max_vnics: int
  first_vf_id: int
  max_vfs: int
  max_stat_ctx: int
  max_encap_records: int
  max_decap_records: int
  max_tx_em_flows: int
  max_tx_wm_flows: int
  max_rx_em_flows: int
  max_rx_wm_flows: int
  max_mcast_filters: int
  max_flow_id: int
  max_hw_ring_grps: int
  max_sp_tx_rings: int
  max_msix_vfs: int
  flags_ext: int
  max_schqs: int
  mpc_chnls_cap: int
  max_key_ctxs_alloc: int
  flags_ext2: int
  tunnel_disable_flag: int
  xid_partition_cap: int
  device_serial_number: c.Array[ctypes.c_ubyte, Literal[8]]
  ctxs_per_partition: int
  max_tso_segs: int
  roce_vf_max_av: int
  roce_vf_max_cq: int
  roce_vf_max_mrw: int
  roce_vf_max_qp: int
  roce_vf_max_srq: int
  roce_vf_max_gid: int
  flags_ext3: int
  max_roce_vfs: int
  max_crypto_rx_flow_filters: int
  unused_3: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_func_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('fid', ctypes.c_uint16, 8), ('port_id', ctypes.c_uint16, 10), ('flags', ctypes.c_uint32, 12), ('mac_address', c.Array[ctypes.c_ubyte, Literal[6]], 16), ('max_rsscos_ctx', ctypes.c_uint16, 22), ('max_cmpl_rings', ctypes.c_uint16, 24), ('max_tx_rings', ctypes.c_uint16, 26), ('max_rx_rings', ctypes.c_uint16, 28), ('max_l2_ctxs', ctypes.c_uint16, 30), ('max_vnics', ctypes.c_uint16, 32), ('first_vf_id', ctypes.c_uint16, 34), ('max_vfs', ctypes.c_uint16, 36), ('max_stat_ctx', ctypes.c_uint16, 38), ('max_encap_records', ctypes.c_uint32, 40), ('max_decap_records', ctypes.c_uint32, 44), ('max_tx_em_flows', ctypes.c_uint32, 48), ('max_tx_wm_flows', ctypes.c_uint32, 52), ('max_rx_em_flows', ctypes.c_uint32, 56), ('max_rx_wm_flows', ctypes.c_uint32, 60), ('max_mcast_filters', ctypes.c_uint32, 64), ('max_flow_id', ctypes.c_uint32, 68), ('max_hw_ring_grps', ctypes.c_uint32, 72), ('max_sp_tx_rings', ctypes.c_uint16, 76), ('max_msix_vfs', ctypes.c_uint16, 78), ('flags_ext', ctypes.c_uint32, 80), ('max_schqs', ctypes.c_ubyte, 84), ('mpc_chnls_cap', ctypes.c_ubyte, 85), ('max_key_ctxs_alloc', ctypes.c_uint16, 86), ('flags_ext2', ctypes.c_uint32, 88), ('tunnel_disable_flag', ctypes.c_uint16, 92), ('xid_partition_cap', ctypes.c_uint16, 94), ('device_serial_number', c.Array[ctypes.c_ubyte, Literal[8]], 96), ('ctxs_per_partition', ctypes.c_uint16, 104), ('max_tso_segs', ctypes.c_uint16, 106), ('roce_vf_max_av', ctypes.c_uint32, 108), ('roce_vf_max_cq', ctypes.c_uint32, 112), ('roce_vf_max_mrw', ctypes.c_uint32, 116), ('roce_vf_max_qp', ctypes.c_uint32, 120), ('roce_vf_max_srq', ctypes.c_uint32, 124), ('roce_vf_max_gid', ctypes.c_uint32, 128), ('flags_ext3', ctypes.c_uint32, 132), ('max_roce_vfs', ctypes.c_uint16, 136), ('max_crypto_rx_flow_filters', ctypes.c_uint16, 138), ('unused_3', c.Array[ctypes.c_ubyte, Literal[3]], 140), ('valid', ctypes.c_ubyte, 143)])
@c.record
class struct_hwrm_func_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_func_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_func_qcfg_output(c.Struct):
  SIZE = 176
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  fid: int
  port_id: int
  vlan: int
  flags: int
  mac_address: c.Array[ctypes.c_ubyte, Literal[6]]
  pci_id: int
  alloc_rsscos_ctx: int
  alloc_cmpl_rings: int
  alloc_tx_rings: int
  alloc_rx_rings: int
  alloc_l2_ctx: int
  alloc_vnics: int
  admin_mtu: int
  mru: int
  stat_ctx_id: int
  port_partition_type: int
  port_pf_cnt: int
  dflt_vnic_id: int
  max_mtu_configured: int
  min_bw: int
  max_bw: int
  evb_mode: int
  options: int
  alloc_vfs: int
  alloc_mcast_filters: int
  alloc_hw_ring_grps: int
  alloc_sp_tx_rings: int
  alloc_stat_ctx: int
  alloc_msix: int
  registered_vfs: int
  l2_doorbell_bar_size_kb: int
  active_endpoints: int
  always_1: int
  reset_addr_poll: int
  legacy_l2_db_size_kb: int
  svif_info: int
  mpc_chnls: int
  db_page_size: int
  roce_vnic_id: int
  partition_min_bw: int
  partition_max_bw: int
  host_mtu: int
  flags2: int
  stag_vid: int
  port_kdnet_mode: int
  kdnet_pcie_function: int
  port_kdnet_fid: int
  unused_5: int
  roce_bidi_opt_mode: int
  num_ktls_tx_key_ctxs: int
  num_ktls_rx_key_ctxs: int
  lag_id: int
  parif: int
  fw_lag_id: int
  unused_6: int
  num_quic_tx_key_ctxs: int
  num_quic_rx_key_ctxs: int
  roce_max_av_per_vf: int
  roce_max_cq_per_vf: int
  roce_max_mrw_per_vf: int
  roce_max_qp_per_vf: int
  roce_max_srq_per_vf: int
  roce_max_gid_per_vf: int
  xid_partition_cfg: int
  mirror_vnic_id: int
  max_link_width: int
  max_link_speed: int
  negotiated_link_width: int
  negotiated_link_speed: int
  unused_7: c.Array[ctypes.c_ubyte, Literal[2]]
  pcie_compliance: int
  unused_8: int
  l2_db_multi_page_size_kb: int
  unused_9: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_func_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('fid', ctypes.c_uint16, 8), ('port_id', ctypes.c_uint16, 10), ('vlan', ctypes.c_uint16, 12), ('flags', ctypes.c_uint16, 14), ('mac_address', c.Array[ctypes.c_ubyte, Literal[6]], 16), ('pci_id', ctypes.c_uint16, 22), ('alloc_rsscos_ctx', ctypes.c_uint16, 24), ('alloc_cmpl_rings', ctypes.c_uint16, 26), ('alloc_tx_rings', ctypes.c_uint16, 28), ('alloc_rx_rings', ctypes.c_uint16, 30), ('alloc_l2_ctx', ctypes.c_uint16, 32), ('alloc_vnics', ctypes.c_uint16, 34), ('admin_mtu', ctypes.c_uint16, 36), ('mru', ctypes.c_uint16, 38), ('stat_ctx_id', ctypes.c_uint16, 40), ('port_partition_type', ctypes.c_ubyte, 42), ('port_pf_cnt', ctypes.c_ubyte, 43), ('dflt_vnic_id', ctypes.c_uint16, 44), ('max_mtu_configured', ctypes.c_uint16, 46), ('min_bw', ctypes.c_uint32, 48), ('max_bw', ctypes.c_uint32, 52), ('evb_mode', ctypes.c_ubyte, 56), ('options', ctypes.c_ubyte, 57), ('alloc_vfs', ctypes.c_uint16, 58), ('alloc_mcast_filters', ctypes.c_uint32, 60), ('alloc_hw_ring_grps', ctypes.c_uint32, 64), ('alloc_sp_tx_rings', ctypes.c_uint16, 68), ('alloc_stat_ctx', ctypes.c_uint16, 70), ('alloc_msix', ctypes.c_uint16, 72), ('registered_vfs', ctypes.c_uint16, 74), ('l2_doorbell_bar_size_kb', ctypes.c_uint16, 76), ('active_endpoints', ctypes.c_ubyte, 78), ('always_1', ctypes.c_ubyte, 79), ('reset_addr_poll', ctypes.c_uint32, 80), ('legacy_l2_db_size_kb', ctypes.c_uint16, 84), ('svif_info', ctypes.c_uint16, 86), ('mpc_chnls', ctypes.c_ubyte, 88), ('db_page_size', ctypes.c_ubyte, 89), ('roce_vnic_id', ctypes.c_uint16, 90), ('partition_min_bw', ctypes.c_uint32, 92), ('partition_max_bw', ctypes.c_uint32, 96), ('host_mtu', ctypes.c_uint16, 100), ('flags2', ctypes.c_uint16, 102), ('stag_vid', ctypes.c_uint16, 104), ('port_kdnet_mode', ctypes.c_ubyte, 106), ('kdnet_pcie_function', ctypes.c_ubyte, 107), ('port_kdnet_fid', ctypes.c_uint16, 108), ('unused_5', ctypes.c_ubyte, 110), ('roce_bidi_opt_mode', ctypes.c_ubyte, 111), ('num_ktls_tx_key_ctxs', ctypes.c_uint32, 112), ('num_ktls_rx_key_ctxs', ctypes.c_uint32, 116), ('lag_id', ctypes.c_ubyte, 120), ('parif', ctypes.c_ubyte, 121), ('fw_lag_id', ctypes.c_ubyte, 122), ('unused_6', ctypes.c_ubyte, 123), ('num_quic_tx_key_ctxs', ctypes.c_uint32, 124), ('num_quic_rx_key_ctxs', ctypes.c_uint32, 128), ('roce_max_av_per_vf', ctypes.c_uint32, 132), ('roce_max_cq_per_vf', ctypes.c_uint32, 136), ('roce_max_mrw_per_vf', ctypes.c_uint32, 140), ('roce_max_qp_per_vf', ctypes.c_uint32, 144), ('roce_max_srq_per_vf', ctypes.c_uint32, 148), ('roce_max_gid_per_vf', ctypes.c_uint32, 152), ('xid_partition_cfg', ctypes.c_uint16, 156), ('mirror_vnic_id', ctypes.c_uint16, 158), ('max_link_width', ctypes.c_ubyte, 160), ('max_link_speed', ctypes.c_ubyte, 161), ('negotiated_link_width', ctypes.c_ubyte, 162), ('negotiated_link_speed', ctypes.c_ubyte, 163), ('unused_7', c.Array[ctypes.c_ubyte, Literal[2]], 164), ('pcie_compliance', ctypes.c_ubyte, 166), ('unused_8', ctypes.c_ubyte, 167), ('l2_db_multi_page_size_kb', ctypes.c_uint16, 168), ('unused_9', c.Array[ctypes.c_ubyte, Literal[5]], 170), ('valid', ctypes.c_ubyte, 175)])
@c.record
class struct_hwrm_func_drv_rgtr_input(c.Struct):
  SIZE = 112
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  os_type: int
  ver_maj_8b: int
  ver_min_8b: int
  ver_upd_8b: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  timestamp: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
  vf_req_fwd: c.Array[ctypes.c_uint32, Literal[8]]
  async_event_fwd: c.Array[ctypes.c_uint32, Literal[8]]
  ver_maj: int
  ver_min: int
  ver_upd: int
  ver_patch: int
struct_hwrm_func_drv_rgtr_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('os_type', ctypes.c_uint16, 24), ('ver_maj_8b', ctypes.c_ubyte, 26), ('ver_min_8b', ctypes.c_ubyte, 27), ('ver_upd_8b', ctypes.c_ubyte, 28), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 29), ('timestamp', ctypes.c_uint32, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 36), ('vf_req_fwd', c.Array[ctypes.c_uint32, Literal[8]], 40), ('async_event_fwd', c.Array[ctypes.c_uint32, Literal[8]], 72), ('ver_maj', ctypes.c_uint16, 104), ('ver_min', ctypes.c_uint16, 106), ('ver_upd', ctypes.c_uint16, 108), ('ver_patch', ctypes.c_uint16, 110)])
@c.record
class struct_hwrm_func_drv_rgtr_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_func_drv_rgtr_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_backing_store_cfg_v2_input(c.Struct):
  SIZE = 64
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  type: int
  instance: int
  flags: int
  page_dir: int
  num_entries: int
  entry_size: int
  page_size_pbl_level: int
  subtype_valid_cnt: int
  split_entry_0: int
  split_entry_1: int
  split_entry_2: int
  split_entry_3: int
  enables: int
  next_bs_offset: int
struct_hwrm_func_backing_store_cfg_v2_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('type', ctypes.c_uint16, 16), ('instance', ctypes.c_uint16, 18), ('flags', ctypes.c_uint32, 20), ('page_dir', ctypes.c_uint64, 24), ('num_entries', ctypes.c_uint32, 32), ('entry_size', ctypes.c_uint16, 36), ('page_size_pbl_level', ctypes.c_ubyte, 38), ('subtype_valid_cnt', ctypes.c_ubyte, 39), ('split_entry_0', ctypes.c_uint32, 40), ('split_entry_1', ctypes.c_uint32, 44), ('split_entry_2', ctypes.c_uint32, 48), ('split_entry_3', ctypes.c_uint32, 52), ('enables', ctypes.c_uint32, 56), ('next_bs_offset', ctypes.c_uint32, 60)])
@c.record
class struct_hwrm_func_backing_store_cfg_v2_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  rsvd0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_backing_store_cfg_v2_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('rsvd0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_backing_store_qcaps_v2_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  type: int
  rsvd: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_func_backing_store_qcaps_v2_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('type', ctypes.c_uint16, 16), ('rsvd', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_func_backing_store_qcaps_v2_output(c.Struct):
  SIZE = 56
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  type: int
  entry_size: int
  flags: int
  instance_bit_map: int
  ctx_init_value: int
  ctx_init_offset: int
  entry_multiple: int
  rsvd: int
  max_num_entries: int
  min_num_entries: int
  next_valid_type: int
  subtype_valid_cnt: int
  exact_cnt_bit_map: int
  split_entry_0: int
  split_entry_1: int
  split_entry_2: int
  split_entry_3: int
  max_instance_count: int
  rsvd3: int
  valid: int
struct_hwrm_func_backing_store_qcaps_v2_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('type', ctypes.c_uint16, 8), ('entry_size', ctypes.c_uint16, 10), ('flags', ctypes.c_uint32, 12), ('instance_bit_map', ctypes.c_uint32, 16), ('ctx_init_value', ctypes.c_ubyte, 20), ('ctx_init_offset', ctypes.c_ubyte, 21), ('entry_multiple', ctypes.c_ubyte, 22), ('rsvd', ctypes.c_ubyte, 23), ('max_num_entries', ctypes.c_uint32, 24), ('min_num_entries', ctypes.c_uint32, 28), ('next_valid_type', ctypes.c_uint16, 32), ('subtype_valid_cnt', ctypes.c_ubyte, 34), ('exact_cnt_bit_map', ctypes.c_ubyte, 35), ('split_entry_0', ctypes.c_uint32, 36), ('split_entry_1', ctypes.c_uint32, 40), ('split_entry_2', ctypes.c_uint32, 44), ('split_entry_3', ctypes.c_uint32, 48), ('max_instance_count', ctypes.c_uint16, 52), ('rsvd3', ctypes.c_ubyte, 54), ('valid', ctypes.c_ubyte, 55)])
@c.record
class struct_hwrm_port_phy_cfg_input(c.Struct):
  SIZE = 64
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  port_id: int
  force_link_speed: int
  auto_mode: int
  auto_duplex: int
  auto_pause: int
  mgmt_flag: int
  auto_link_speed: int
  auto_link_speed_mask: int
  wirespeed: int
  lpbk: int
  force_pause: int
  unused_1: int
  preemphasis: int
  eee_link_speed_mask: int
  force_pam4_link_speed: int
  tx_lpi_timer: int
  auto_link_pam4_speed_mask: int
  force_link_speeds2: int
  auto_link_speeds2_mask: int
  unused_2: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_phy_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('port_id', ctypes.c_uint16, 24), ('force_link_speed', ctypes.c_uint16, 26), ('auto_mode', ctypes.c_ubyte, 28), ('auto_duplex', ctypes.c_ubyte, 29), ('auto_pause', ctypes.c_ubyte, 30), ('mgmt_flag', ctypes.c_ubyte, 31), ('auto_link_speed', ctypes.c_uint16, 32), ('auto_link_speed_mask', ctypes.c_uint16, 34), ('wirespeed', ctypes.c_ubyte, 36), ('lpbk', ctypes.c_ubyte, 37), ('force_pause', ctypes.c_ubyte, 38), ('unused_1', ctypes.c_ubyte, 39), ('preemphasis', ctypes.c_uint32, 40), ('eee_link_speed_mask', ctypes.c_uint16, 44), ('force_pam4_link_speed', ctypes.c_uint16, 46), ('tx_lpi_timer', ctypes.c_uint32, 48), ('auto_link_pam4_speed_mask', ctypes.c_uint16, 52), ('force_link_speeds2', ctypes.c_uint16, 54), ('auto_link_speeds2_mask', ctypes.c_uint16, 56), ('unused_2', c.Array[ctypes.c_ubyte, Literal[6]], 58)])
@c.record
class struct_hwrm_port_phy_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_port_phy_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_alloc_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  virtio_net_fid: int
  vnic_id: int
struct_hwrm_vnic_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('virtio_net_fid', ctypes.c_uint16, 20), ('vnic_id', ctypes.c_uint16, 22)])
@c.record
class struct_hwrm_vnic_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  vnic_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_vnic_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('vnic_id', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_cfg_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  vnic_id: int
  dflt_ring_grp: int
  rss_rule: int
  cos_rule: int
  lb_rule: int
  mru: int
  default_rx_ring_id: int
  default_cmpl_ring_id: int
  queue_id: int
  rx_csum_v2_mode: int
  l2_cqe_mode: int
  raw_qp_id: int
struct_hwrm_vnic_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('vnic_id', ctypes.c_uint16, 24), ('dflt_ring_grp', ctypes.c_uint16, 26), ('rss_rule', ctypes.c_uint16, 28), ('cos_rule', ctypes.c_uint16, 30), ('lb_rule', ctypes.c_uint16, 32), ('mru', ctypes.c_uint16, 34), ('default_rx_ring_id', ctypes.c_uint16, 36), ('default_cmpl_ring_id', ctypes.c_uint16, 38), ('queue_id', ctypes.c_uint16, 40), ('rx_csum_v2_mode', ctypes.c_ubyte, 42), ('l2_cqe_mode', ctypes.c_ubyte, 43), ('raw_qp_id', ctypes.c_uint32, 44)])
@c.record
class struct_hwrm_vnic_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_vnic_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_ring_alloc_input(c.Struct):
  SIZE = 96
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  ring_type: int
  cmpl_coal_cnt: int
  flags: int
  page_tbl_addr: int
  fbo: int
  page_size: int
  page_tbl_depth: int
  schq_id: int
  length: int
  logical_id: int
  cmpl_ring_id: int
  queue_id: int
  rx_buf_size: int
  rx_ring_id: int
  nq_ring_id: int
  ring_arb_cfg: int
  steering_tag: int
  reserved3: int
  stat_ctx_id: int
  reserved4: int
  max_bw: int
  int_mode: int
  mpc_chnls_type: int
  rx_rate_profile_sel: int
  unused_4: int
  cq_handle: int
  dpi: int
  unused_5: c.Array[ctypes.c_uint16, Literal[3]]
struct_hwrm_ring_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('ring_type', ctypes.c_ubyte, 20), ('cmpl_coal_cnt', ctypes.c_ubyte, 21), ('flags', ctypes.c_uint16, 22), ('page_tbl_addr', ctypes.c_uint64, 24), ('fbo', ctypes.c_uint32, 32), ('page_size', ctypes.c_ubyte, 36), ('page_tbl_depth', ctypes.c_ubyte, 37), ('schq_id', ctypes.c_uint16, 38), ('length', ctypes.c_uint32, 40), ('logical_id', ctypes.c_uint16, 44), ('cmpl_ring_id', ctypes.c_uint16, 46), ('queue_id', ctypes.c_uint16, 48), ('rx_buf_size', ctypes.c_uint16, 50), ('rx_ring_id', ctypes.c_uint16, 52), ('nq_ring_id', ctypes.c_uint16, 54), ('ring_arb_cfg', ctypes.c_uint16, 56), ('steering_tag', ctypes.c_uint16, 58), ('reserved3', ctypes.c_uint32, 60), ('stat_ctx_id', ctypes.c_uint32, 64), ('reserved4', ctypes.c_uint32, 68), ('max_bw', ctypes.c_uint32, 72), ('int_mode', ctypes.c_ubyte, 76), ('mpc_chnls_type', ctypes.c_ubyte, 77), ('rx_rate_profile_sel', ctypes.c_ubyte, 78), ('unused_4', ctypes.c_ubyte, 79), ('cq_handle', ctypes.c_uint64, 80), ('dpi', ctypes.c_uint16, 88), ('unused_5', c.Array[ctypes.c_uint16, Literal[3]], 90)])
@c.record
class struct_hwrm_ring_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  ring_id: int
  logical_ring_id: int
  push_buffer_index: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  valid: int
struct_hwrm_ring_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('ring_id', ctypes.c_uint16, 8), ('logical_ring_id', ctypes.c_uint16, 10), ('push_buffer_index', ctypes.c_ubyte, 12), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 13), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_l2_filter_alloc_input(c.Struct):
  SIZE = 96
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  l2_addr: c.Array[ctypes.c_ubyte, Literal[6]]
  num_vlans: int
  t_num_vlans: int
  l2_addr_mask: c.Array[ctypes.c_ubyte, Literal[6]]
  l2_ovlan: int
  l2_ovlan_mask: int
  l2_ivlan: int
  l2_ivlan_mask: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[2]]
  t_l2_addr: c.Array[ctypes.c_ubyte, Literal[6]]
  unused_2: c.Array[ctypes.c_ubyte, Literal[2]]
  t_l2_addr_mask: c.Array[ctypes.c_ubyte, Literal[6]]
  t_l2_ovlan: int
  t_l2_ovlan_mask: int
  t_l2_ivlan: int
  t_l2_ivlan_mask: int
  src_type: int
  unused_3: int
  src_id: int
  tunnel_type: int
  unused_4: int
  dst_id: int
  mirror_vnic_id: int
  pri_hint: int
  unused_5: int
  unused_6: int
  l2_filter_id_hint: int
struct_hwrm_cfa_l2_filter_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('l2_addr', c.Array[ctypes.c_ubyte, Literal[6]], 24), ('num_vlans', ctypes.c_ubyte, 30), ('t_num_vlans', ctypes.c_ubyte, 31), ('l2_addr_mask', c.Array[ctypes.c_ubyte, Literal[6]], 32), ('l2_ovlan', ctypes.c_uint16, 38), ('l2_ovlan_mask', ctypes.c_uint16, 40), ('l2_ivlan', ctypes.c_uint16, 42), ('l2_ivlan_mask', ctypes.c_uint16, 44), ('unused_1', c.Array[ctypes.c_ubyte, Literal[2]], 46), ('t_l2_addr', c.Array[ctypes.c_ubyte, Literal[6]], 48), ('unused_2', c.Array[ctypes.c_ubyte, Literal[2]], 54), ('t_l2_addr_mask', c.Array[ctypes.c_ubyte, Literal[6]], 56), ('t_l2_ovlan', ctypes.c_uint16, 62), ('t_l2_ovlan_mask', ctypes.c_uint16, 64), ('t_l2_ivlan', ctypes.c_uint16, 66), ('t_l2_ivlan_mask', ctypes.c_uint16, 68), ('src_type', ctypes.c_ubyte, 70), ('unused_3', ctypes.c_ubyte, 71), ('src_id', ctypes.c_uint32, 72), ('tunnel_type', ctypes.c_ubyte, 76), ('unused_4', ctypes.c_ubyte, 77), ('dst_id', ctypes.c_uint16, 78), ('mirror_vnic_id', ctypes.c_uint16, 80), ('pri_hint', ctypes.c_ubyte, 82), ('unused_5', ctypes.c_ubyte, 83), ('unused_6', ctypes.c_uint32, 84), ('l2_filter_id_hint', ctypes.c_uint64, 88)])
@c.record
class struct_hwrm_cfa_l2_filter_alloc_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  l2_filter_id: int
  flow_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_cfa_l2_filter_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('l2_filter_id', ctypes.c_uint64, 8), ('flow_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 20), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_stat_ctx_alloc_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  stats_dma_addr: int
  update_period_ms: int
  stat_ctx_flags: int
  unused_0: int
  stats_dma_length: int
  flags: int
  steering_tag: int
  stat_ctx_id: int
  alloc_seq_id: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_stat_ctx_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('stats_dma_addr', ctypes.c_uint64, 16), ('update_period_ms', ctypes.c_uint32, 24), ('stat_ctx_flags', ctypes.c_ubyte, 28), ('unused_0', ctypes.c_ubyte, 29), ('stats_dma_length', ctypes.c_uint16, 30), ('flags', ctypes.c_uint16, 32), ('steering_tag', ctypes.c_uint16, 34), ('stat_ctx_id', ctypes.c_uint32, 36), ('alloc_seq_id', ctypes.c_uint16, 40), ('unused_1', c.Array[ctypes.c_ubyte, Literal[6]], 42)])
@c.record
class struct_hwrm_stat_ctx_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  stat_ctx_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_stat_ctx_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('stat_ctx_id', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_cmdq_init(c.Struct):
  SIZE = 16
  cmdq_pbl: int
  cmdq_size_cmdq_lvl: int
  creq_ring_id: int
  prod_idx: int
struct_cmdq_init.register_fields([('cmdq_pbl', ctypes.c_uint64, 0), ('cmdq_size_cmdq_lvl', ctypes.c_uint16, 8), ('creq_ring_id', ctypes.c_uint16, 10), ('prod_idx', ctypes.c_uint32, 12)])
@c.record
class struct_cmdq_base(c.Struct):
  SIZE = 16
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
struct_cmdq_base.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_creq_base(c.Struct):
  SIZE = 16
  type: int
  reserved56: c.Array[ctypes.c_ubyte, Literal[7]]
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_base.register_fields([('type', ctypes.c_ubyte, 0), ('reserved56', c.Array[ctypes.c_ubyte, Literal[7]], 1), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_query_version(c.Struct):
  SIZE = 16
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
struct_cmdq_query_version.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_creq_query_version_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  fw_maj: int
  fw_minor: int
  fw_bld: int
  fw_rsvd: int
  v: int
  event: int
  reserved16: int
  intf_maj: int
  intf_minor: int
  intf_bld: int
  intf_rsvd: int
struct_creq_query_version_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('fw_maj', ctypes.c_ubyte, 4), ('fw_minor', ctypes.c_ubyte, 5), ('fw_bld', ctypes.c_ubyte, 6), ('fw_rsvd', ctypes.c_ubyte, 7), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved16', ctypes.c_uint16, 10), ('intf_maj', ctypes.c_ubyte, 12), ('intf_minor', ctypes.c_ubyte, 13), ('intf_bld', ctypes.c_ubyte, 14), ('intf_rsvd', ctypes.c_ubyte, 15)])
@c.record
class struct_cmdq_initialize_fw(c.Struct):
  SIZE = 112
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  qpc_pg_size_qpc_lvl: int
  mrw_pg_size_mrw_lvl: int
  srq_pg_size_srq_lvl: int
  cq_pg_size_cq_lvl: int
  tqm_pg_size_tqm_lvl: int
  tim_pg_size_tim_lvl: int
  log2_dbr_pg_size: int
  qpc_page_dir: int
  mrw_page_dir: int
  srq_page_dir: int
  cq_page_dir: int
  tqm_page_dir: int
  tim_page_dir: int
  number_of_qp: int
  number_of_mrw: int
  number_of_srq: int
  number_of_cq: int
  max_qp_per_vf: int
  max_mrw_per_vf: int
  max_srq_per_vf: int
  max_cq_per_vf: int
  max_gid_per_vf: int
  stat_ctx_id: int
struct_cmdq_initialize_fw.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('qpc_pg_size_qpc_lvl', ctypes.c_ubyte, 16), ('mrw_pg_size_mrw_lvl', ctypes.c_ubyte, 17), ('srq_pg_size_srq_lvl', ctypes.c_ubyte, 18), ('cq_pg_size_cq_lvl', ctypes.c_ubyte, 19), ('tqm_pg_size_tqm_lvl', ctypes.c_ubyte, 20), ('tim_pg_size_tim_lvl', ctypes.c_ubyte, 21), ('log2_dbr_pg_size', ctypes.c_uint16, 22), ('qpc_page_dir', ctypes.c_uint64, 24), ('mrw_page_dir', ctypes.c_uint64, 32), ('srq_page_dir', ctypes.c_uint64, 40), ('cq_page_dir', ctypes.c_uint64, 48), ('tqm_page_dir', ctypes.c_uint64, 56), ('tim_page_dir', ctypes.c_uint64, 64), ('number_of_qp', ctypes.c_uint32, 72), ('number_of_mrw', ctypes.c_uint32, 76), ('number_of_srq', ctypes.c_uint32, 80), ('number_of_cq', ctypes.c_uint32, 84), ('max_qp_per_vf', ctypes.c_uint32, 88), ('max_mrw_per_vf', ctypes.c_uint32, 92), ('max_srq_per_vf', ctypes.c_uint32, 96), ('max_cq_per_vf', ctypes.c_uint32, 100), ('max_gid_per_vf', ctypes.c_uint32, 104), ('stat_ctx_id', ctypes.c_uint32, 108)])
@c.record
class struct_creq_initialize_fw_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_initialize_fw_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_create_qp(c.Struct):
  SIZE = 104
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  qp_handle: int
  qp_flags: int
  type: int
  sq_pg_size_sq_lvl: int
  rq_pg_size_rq_lvl: int
  unused_0: int
  dpi: int
  sq_size: int
  rq_size: int
  sq_fwo_sq_sge: int
  rq_fwo_rq_sge: int
  scq_cid: int
  rcq_cid: int
  srq_cid: int
  pd_id: int
  sq_pbl: int
  rq_pbl: int
  irrq_addr: int
  orrq_addr: int
  request_xid: int
  steering_tag: int
  reserved16: int
struct_cmdq_create_qp.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('qp_handle', ctypes.c_uint64, 16), ('qp_flags', ctypes.c_uint32, 24), ('type', ctypes.c_ubyte, 28), ('sq_pg_size_sq_lvl', ctypes.c_ubyte, 29), ('rq_pg_size_rq_lvl', ctypes.c_ubyte, 30), ('unused_0', ctypes.c_ubyte, 31), ('dpi', ctypes.c_uint32, 32), ('sq_size', ctypes.c_uint32, 36), ('rq_size', ctypes.c_uint32, 40), ('sq_fwo_sq_sge', ctypes.c_uint16, 44), ('rq_fwo_rq_sge', ctypes.c_uint16, 46), ('scq_cid', ctypes.c_uint32, 48), ('rcq_cid', ctypes.c_uint32, 52), ('srq_cid', ctypes.c_uint32, 56), ('pd_id', ctypes.c_uint32, 60), ('sq_pbl', ctypes.c_uint64, 64), ('rq_pbl', ctypes.c_uint64, 72), ('irrq_addr', ctypes.c_uint64, 80), ('orrq_addr', ctypes.c_uint64, 88), ('request_xid', ctypes.c_uint32, 96), ('steering_tag', ctypes.c_uint16, 100), ('reserved16', ctypes.c_uint16, 102)])
@c.record
class struct_creq_create_qp_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  optimized_transmit_enabled: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[5]]
struct_creq_create_qp_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('optimized_transmit_enabled', ctypes.c_ubyte, 10), ('reserved48', c.Array[ctypes.c_ubyte, Literal[5]], 11)])
@c.record
class struct_cmdq_modify_qp(c.Struct):
  SIZE = 144
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  qp_type: int
  resp_addr: int
  modify_mask: int
  qp_cid: int
  network_type_en_sqd_async_notify_new_state: int
  access: int
  pkey: int
  qkey: int
  dgid: c.Array[ctypes.c_uint32, Literal[4]]
  flow_label: int
  sgid_index: int
  hop_limit: int
  traffic_class: int
  dest_mac: c.Array[ctypes.c_uint16, Literal[3]]
  tos_dscp_tos_ecn: int
  path_mtu_pingpong_push_enable: int
  timeout: int
  retry_cnt: int
  rnr_retry: int
  min_rnr_timer: int
  rq_psn: int
  sq_psn: int
  max_rd_atomic: int
  max_dest_rd_atomic: int
  enable_cc: int
  sq_size: int
  rq_size: int
  sq_sge: int
  rq_sge: int
  max_inline_data: int
  dest_qp_id: int
  pingpong_push_dpi: int
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  vlan_pcp_vlan_dei_vlan_id: int
  irrq_addr: int
  orrq_addr: int
  ext_modify_mask: int
  ext_stats_ctx_id: int
  schq_id: int
  unused_0: int
  reserved32: int
struct_cmdq_modify_qp.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('qp_type', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('modify_mask', ctypes.c_uint32, 16), ('qp_cid', ctypes.c_uint32, 20), ('network_type_en_sqd_async_notify_new_state', ctypes.c_ubyte, 24), ('access', ctypes.c_ubyte, 25), ('pkey', ctypes.c_uint16, 26), ('qkey', ctypes.c_uint32, 28), ('dgid', c.Array[ctypes.c_uint32, Literal[4]], 32), ('flow_label', ctypes.c_uint32, 48), ('sgid_index', ctypes.c_uint16, 52), ('hop_limit', ctypes.c_ubyte, 54), ('traffic_class', ctypes.c_ubyte, 55), ('dest_mac', c.Array[ctypes.c_uint16, Literal[3]], 56), ('tos_dscp_tos_ecn', ctypes.c_ubyte, 62), ('path_mtu_pingpong_push_enable', ctypes.c_ubyte, 63), ('timeout', ctypes.c_ubyte, 64), ('retry_cnt', ctypes.c_ubyte, 65), ('rnr_retry', ctypes.c_ubyte, 66), ('min_rnr_timer', ctypes.c_ubyte, 67), ('rq_psn', ctypes.c_uint32, 68), ('sq_psn', ctypes.c_uint32, 72), ('max_rd_atomic', ctypes.c_ubyte, 76), ('max_dest_rd_atomic', ctypes.c_ubyte, 77), ('enable_cc', ctypes.c_uint16, 78), ('sq_size', ctypes.c_uint32, 80), ('rq_size', ctypes.c_uint32, 84), ('sq_sge', ctypes.c_uint16, 88), ('rq_sge', ctypes.c_uint16, 90), ('max_inline_data', ctypes.c_uint32, 92), ('dest_qp_id', ctypes.c_uint32, 96), ('pingpong_push_dpi', ctypes.c_uint32, 100), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 104), ('vlan_pcp_vlan_dei_vlan_id', ctypes.c_uint16, 110), ('irrq_addr', ctypes.c_uint64, 112), ('orrq_addr', ctypes.c_uint64, 120), ('ext_modify_mask', ctypes.c_uint32, 128), ('ext_stats_ctx_id', ctypes.c_uint32, 132), ('schq_id', ctypes.c_uint16, 136), ('unused_0', ctypes.c_uint16, 138), ('reserved32', ctypes.c_uint32, 140)])
@c.record
class struct_creq_modify_qp_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  pingpong_push_state_index_enabled: int
  reserved8: int
  lag_src_mac: int
struct_creq_modify_qp_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('pingpong_push_state_index_enabled', ctypes.c_ubyte, 10), ('reserved8', ctypes.c_ubyte, 11), ('lag_src_mac', ctypes.c_uint32, 12)])
@c.record
class struct_cmdq_create_cq(c.Struct):
  SIZE = 64
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  cq_handle: int
  pg_size_lvl: int
  cq_fco_cnq_id: int
  dpi: int
  cq_size: int
  pbl: int
  steering_tag: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[2]]
  coalescing: int
  reserved64: int
struct_cmdq_create_cq.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('cq_handle', ctypes.c_uint64, 16), ('pg_size_lvl', ctypes.c_uint32, 24), ('cq_fco_cnq_id', ctypes.c_uint32, 28), ('dpi', ctypes.c_uint32, 32), ('cq_size', ctypes.c_uint32, 36), ('pbl', ctypes.c_uint64, 40), ('steering_tag', ctypes.c_uint16, 48), ('reserved48', c.Array[ctypes.c_ubyte, Literal[2]], 50), ('coalescing', ctypes.c_uint32, 52), ('reserved64', ctypes.c_uint64, 56)])
@c.record
class struct_creq_create_cq_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_create_cq_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_register_mr(c.Struct):
  SIZE = 56
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  log2_pg_size_lvl: int
  access: int
  log2_pbl_pg_size: int
  key: int
  pbl: int
  va: int
  mr_size: int
  steering_tag: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_cmdq_register_mr.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('log2_pg_size_lvl', ctypes.c_ubyte, 16), ('access', ctypes.c_ubyte, 17), ('log2_pbl_pg_size', ctypes.c_uint16, 18), ('key', ctypes.c_uint32, 20), ('pbl', ctypes.c_uint64, 24), ('va', ctypes.c_uint64, 32), ('mr_size', ctypes.c_uint64, 40), ('steering_tag', ctypes.c_uint16, 48), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 50)])
@c.record
class struct_creq_register_mr_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_register_mr_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_add_gid(c.Struct):
  SIZE = 48
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  gid: c.Array[ctypes.c_uint32, Literal[4]]
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  vlan: int
  ipid: int
  stats_ctx: int
  unused_0: int
struct_cmdq_add_gid.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('gid', c.Array[ctypes.c_uint32, Literal[4]], 16), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 32), ('vlan', ctypes.c_uint16, 38), ('ipid', ctypes.c_uint16, 40), ('stats_ctx', ctypes.c_uint16, 42), ('unused_0', ctypes.c_uint32, 44)])
@c.record
class struct_creq_add_gid_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_add_gid_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_sq_sge(c.Struct):
  SIZE = 16
  va_or_pa: int
  l_key: int
  size: int
struct_sq_sge.register_fields([('va_or_pa', ctypes.c_uint64, 0), ('l_key', ctypes.c_uint32, 8), ('size', ctypes.c_uint32, 12)])
@c.record
class struct_sq_rdma_hdr(c.Struct):
  SIZE = 32
  wqe_type: int
  flags: int
  wqe_size: int
  reserved8: int
  imm_data: int
  length: int
  reserved32_1: int
  remote_va: int
  remote_key: int
  timestamp: int
struct_sq_rdma_hdr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('wqe_size', ctypes.c_ubyte, 2), ('reserved8', ctypes.c_ubyte, 3), ('imm_data', ctypes.c_uint32, 4), ('length', ctypes.c_uint32, 8), ('reserved32_1', ctypes.c_uint32, 12), ('remote_va', ctypes.c_uint64, 16), ('remote_key', ctypes.c_uint32, 24), ('timestamp', ctypes.c_uint32, 28)])
@c.record
class struct_cq_base(c.Struct):
  SIZE = 32
  reserved64_1: int
  reserved64_2: int
  reserved64_3: int
  cqe_type_toggle: int
  status: int
  reserved16: int
  opaque: int
struct_cq_base.register_fields([('reserved64_1', ctypes.c_uint64, 0), ('reserved64_2', ctypes.c_uint64, 8), ('reserved64_3', ctypes.c_uint64, 16), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('reserved16', ctypes.c_uint16, 26), ('opaque', ctypes.c_uint32, 28)])
@c.record
class struct_cq_req(c.Struct):
  SIZE = 32
  qp_handle: int
  sq_cons_idx: int
  reserved16_1: int
  reserved32_2: int
  reserved64: int
  cqe_type_toggle: int
  status: int
  reserved16_2: int
  reserved32_1: int
struct_cq_req.register_fields([('qp_handle', ctypes.c_uint64, 0), ('sq_cons_idx', ctypes.c_uint16, 8), ('reserved16_1', ctypes.c_uint16, 10), ('reserved32_2', ctypes.c_uint32, 12), ('reserved64', ctypes.c_uint64, 16), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('reserved16_2', ctypes.c_uint16, 26), ('reserved32_1', ctypes.c_uint32, 28)])
HWRM_VER_GET = 0x0
HWRM_FUNC_ECHO_RESPONSE = 0xb
HWRM_ERROR_RECOVERY_QCFG = 0xc
HWRM_FUNC_DRV_IF_CHANGE = 0xd
HWRM_FUNC_BUF_UNRGTR = 0xe
HWRM_FUNC_VF_CFG = 0xf
HWRM_RESERVED1 = 0x10
HWRM_FUNC_RESET = 0x11
HWRM_FUNC_GETFID = 0x12
HWRM_FUNC_VF_ALLOC = 0x13
HWRM_FUNC_VF_FREE = 0x14
HWRM_FUNC_QCAPS = 0x15
HWRM_FUNC_QCFG = 0x16
HWRM_FUNC_CFG = 0x17
HWRM_FUNC_QSTATS = 0x18
HWRM_FUNC_CLR_STATS = 0x19
HWRM_FUNC_DRV_UNRGTR = 0x1a
HWRM_FUNC_VF_RESC_FREE = 0x1b
HWRM_FUNC_VF_VNIC_IDS_QUERY = 0x1c
HWRM_FUNC_DRV_RGTR = 0x1d
HWRM_FUNC_DRV_QVER = 0x1e
HWRM_FUNC_BUF_RGTR = 0x1f
HWRM_PORT_PHY_CFG = 0x20
HWRM_PORT_MAC_CFG = 0x21
HWRM_PORT_TS_QUERY = 0x22
HWRM_PORT_QSTATS = 0x23
HWRM_PORT_LPBK_QSTATS = 0x24
HWRM_PORT_CLR_STATS = 0x25
HWRM_PORT_LPBK_CLR_STATS = 0x26
HWRM_PORT_PHY_QCFG = 0x27
HWRM_PORT_MAC_QCFG = 0x28
HWRM_PORT_MAC_PTP_QCFG = 0x29
HWRM_PORT_PHY_QCAPS = 0x2a
HWRM_PORT_PHY_I2C_WRITE = 0x2b
HWRM_PORT_PHY_I2C_READ = 0x2c
HWRM_PORT_LED_CFG = 0x2d
HWRM_PORT_LED_QCFG = 0x2e
HWRM_PORT_LED_QCAPS = 0x2f
HWRM_QUEUE_QPORTCFG = 0x30
HWRM_QUEUE_QCFG = 0x31
HWRM_QUEUE_CFG = 0x32
HWRM_FUNC_VLAN_CFG = 0x33
HWRM_FUNC_VLAN_QCFG = 0x34
HWRM_QUEUE_PFCENABLE_QCFG = 0x35
HWRM_QUEUE_PFCENABLE_CFG = 0x36
HWRM_QUEUE_PRI2COS_QCFG = 0x37
HWRM_QUEUE_PRI2COS_CFG = 0x38
HWRM_QUEUE_COS2BW_QCFG = 0x39
HWRM_QUEUE_COS2BW_CFG = 0x3a
HWRM_QUEUE_DSCP_QCAPS = 0x3b
HWRM_QUEUE_DSCP2PRI_QCFG = 0x3c
HWRM_QUEUE_DSCP2PRI_CFG = 0x3d
HWRM_VNIC_ALLOC = 0x40
HWRM_VNIC_FREE = 0x41
HWRM_VNIC_CFG = 0x42
HWRM_VNIC_QCFG = 0x43
HWRM_VNIC_TPA_CFG = 0x44
HWRM_VNIC_TPA_QCFG = 0x45
HWRM_VNIC_RSS_CFG = 0x46
HWRM_VNIC_RSS_QCFG = 0x47
HWRM_VNIC_PLCMODES_CFG = 0x48
HWRM_VNIC_PLCMODES_QCFG = 0x49
HWRM_VNIC_QCAPS = 0x4a
HWRM_VNIC_UPDATE = 0x4b
HWRM_RING_ALLOC = 0x50
HWRM_RING_FREE = 0x51
HWRM_RING_CMPL_RING_QAGGINT_PARAMS = 0x52
HWRM_RING_CMPL_RING_CFG_AGGINT_PARAMS = 0x53
HWRM_RING_AGGINT_QCAPS = 0x54
HWRM_RING_SCHQ_ALLOC = 0x55
HWRM_RING_SCHQ_CFG = 0x56
HWRM_RING_SCHQ_FREE = 0x57
HWRM_RING_RESET = 0x5e
HWRM_RING_GRP_ALLOC = 0x60
HWRM_RING_GRP_FREE = 0x61
HWRM_RING_CFG = 0x62
HWRM_RING_QCFG = 0x63
HWRM_RESERVED5 = 0x64
HWRM_RESERVED6 = 0x65
HWRM_VNIC_RSS_COS_LB_CTX_ALLOC = 0x70
HWRM_VNIC_RSS_COS_LB_CTX_FREE = 0x71
HWRM_QUEUE_MPLS_QCAPS = 0x80
HWRM_QUEUE_MPLSTC2PRI_QCFG = 0x81
HWRM_QUEUE_MPLSTC2PRI_CFG = 0x82
HWRM_QUEUE_VLANPRI_QCAPS = 0x83
HWRM_QUEUE_VLANPRI2PRI_QCFG = 0x84
HWRM_QUEUE_VLANPRI2PRI_CFG = 0x85
HWRM_QUEUE_GLOBAL_CFG = 0x86
HWRM_QUEUE_GLOBAL_QCFG = 0x87
HWRM_QUEUE_ADPTV_QOS_RX_FEATURE_QCFG = 0x88
HWRM_QUEUE_ADPTV_QOS_RX_FEATURE_CFG = 0x89
HWRM_QUEUE_ADPTV_QOS_TX_FEATURE_QCFG = 0x8a
HWRM_QUEUE_ADPTV_QOS_TX_FEATURE_CFG = 0x8b
HWRM_QUEUE_QCAPS = 0x8c
HWRM_QUEUE_ADPTV_QOS_RX_TUNING_QCFG = 0x8d
HWRM_QUEUE_ADPTV_QOS_RX_TUNING_CFG = 0x8e
HWRM_QUEUE_ADPTV_QOS_TX_TUNING_QCFG = 0x8f
HWRM_CFA_L2_FILTER_ALLOC = 0x90
HWRM_CFA_L2_FILTER_FREE = 0x91
HWRM_CFA_L2_FILTER_CFG = 0x92
HWRM_CFA_L2_SET_RX_MASK = 0x93
HWRM_CFA_VLAN_ANTISPOOF_CFG = 0x94
HWRM_CFA_TUNNEL_FILTER_ALLOC = 0x95
HWRM_CFA_TUNNEL_FILTER_FREE = 0x96
HWRM_CFA_ENCAP_RECORD_ALLOC = 0x97
HWRM_CFA_ENCAP_RECORD_FREE = 0x98
HWRM_CFA_NTUPLE_FILTER_ALLOC = 0x99
HWRM_CFA_NTUPLE_FILTER_FREE = 0x9a
HWRM_CFA_NTUPLE_FILTER_CFG = 0x9b
HWRM_CFA_EM_FLOW_ALLOC = 0x9c
HWRM_CFA_EM_FLOW_FREE = 0x9d
HWRM_CFA_EM_FLOW_CFG = 0x9e
HWRM_TUNNEL_DST_PORT_QUERY = 0xa0
HWRM_TUNNEL_DST_PORT_ALLOC = 0xa1
HWRM_TUNNEL_DST_PORT_FREE = 0xa2
HWRM_QUEUE_ADPTV_QOS_TX_TUNING_CFG = 0xa3
HWRM_STAT_CTX_ENG_QUERY = 0xaf
HWRM_STAT_CTX_ALLOC = 0xb0
HWRM_STAT_CTX_FREE = 0xb1
HWRM_STAT_CTX_QUERY = 0xb2
HWRM_STAT_CTX_CLR_STATS = 0xb3
HWRM_PORT_QSTATS_EXT = 0xb4
HWRM_PORT_PHY_MDIO_WRITE = 0xb5
HWRM_PORT_PHY_MDIO_READ = 0xb6
HWRM_PORT_PHY_MDIO_BUS_ACQUIRE = 0xb7
HWRM_PORT_PHY_MDIO_BUS_RELEASE = 0xb8
HWRM_PORT_QSTATS_EXT_PFC_WD = 0xb9
HWRM_RESERVED7 = 0xba
HWRM_PORT_TX_FIR_CFG = 0xbb
HWRM_PORT_TX_FIR_QCFG = 0xbc
HWRM_PORT_ECN_QSTATS = 0xbd
HWRM_FW_LIVEPATCH_QUERY = 0xbe
HWRM_FW_LIVEPATCH = 0xbf
HWRM_FW_RESET = 0xc0
HWRM_FW_QSTATUS = 0xc1
HWRM_FW_HEALTH_CHECK = 0xc2
HWRM_FW_SYNC = 0xc3
HWRM_FW_STATE_QCAPS = 0xc4
HWRM_FW_STATE_QUIESCE = 0xc5
HWRM_FW_STATE_BACKUP = 0xc6
HWRM_FW_STATE_RESTORE = 0xc7
HWRM_FW_SET_TIME = 0xc8
HWRM_FW_GET_TIME = 0xc9
HWRM_FW_SET_STRUCTURED_DATA = 0xca
HWRM_FW_GET_STRUCTURED_DATA = 0xcb
HWRM_FW_IPC_MAILBOX = 0xcc
HWRM_FW_ECN_CFG = 0xcd
HWRM_FW_ECN_QCFG = 0xce
HWRM_FW_SECURE_CFG = 0xcf
HWRM_EXEC_FWD_RESP = 0xd0
HWRM_REJECT_FWD_RESP = 0xd1
HWRM_FWD_RESP = 0xd2
HWRM_FWD_ASYNC_EVENT_CMPL = 0xd3
HWRM_OEM_CMD = 0xd4
HWRM_PORT_PRBS_TEST = 0xd5
HWRM_PORT_SFP_SIDEBAND_CFG = 0xd6
HWRM_PORT_SFP_SIDEBAND_QCFG = 0xd7
HWRM_FW_STATE_UNQUIESCE = 0xd8
HWRM_PORT_DSC_DUMP = 0xd9
HWRM_PORT_EP_TX_QCFG = 0xda
HWRM_PORT_EP_TX_CFG = 0xdb
HWRM_PORT_CFG = 0xdc
HWRM_PORT_QCFG = 0xdd
HWRM_PORT_MAC_QCAPS = 0xdf
HWRM_TEMP_MONITOR_QUERY = 0xe0
HWRM_REG_POWER_QUERY = 0xe1
HWRM_CORE_FREQUENCY_QUERY = 0xe2
HWRM_REG_POWER_HISTOGRAM = 0xe3
HWRM_MONITOR_PAX_HISTOGRAM_START = 0xe4
HWRM_MONITOR_PAX_HISTOGRAM_COLLECT = 0xe5
HWRM_STAT_QUERY_ROCE_STATS = 0xe6
HWRM_STAT_QUERY_ROCE_STATS_EXT = 0xe7
HWRM_WOL_FILTER_ALLOC = 0xf0
HWRM_WOL_FILTER_FREE = 0xf1
HWRM_WOL_FILTER_QCFG = 0xf2
HWRM_WOL_REASON_QCFG = 0xf3
HWRM_CFA_METER_QCAPS = 0xf4
HWRM_CFA_METER_PROFILE_ALLOC = 0xf5
HWRM_CFA_METER_PROFILE_FREE = 0xf6
HWRM_CFA_METER_PROFILE_CFG = 0xf7
HWRM_CFA_METER_INSTANCE_ALLOC = 0xf8
HWRM_CFA_METER_INSTANCE_FREE = 0xf9
HWRM_CFA_METER_INSTANCE_CFG = 0xfa
HWRM_CFA_VFR_ALLOC = 0xfd
HWRM_CFA_VFR_FREE = 0xfe
HWRM_CFA_VF_PAIR_ALLOC = 0x100
HWRM_CFA_VF_PAIR_FREE = 0x101
HWRM_CFA_VF_PAIR_INFO = 0x102
HWRM_CFA_FLOW_ALLOC = 0x103
HWRM_CFA_FLOW_FREE = 0x104
HWRM_CFA_FLOW_FLUSH = 0x105
HWRM_CFA_FLOW_STATS = 0x106
HWRM_CFA_FLOW_INFO = 0x107
HWRM_CFA_DECAP_FILTER_ALLOC = 0x108
HWRM_CFA_DECAP_FILTER_FREE = 0x109
HWRM_CFA_VLAN_ANTISPOOF_QCFG = 0x10a
HWRM_CFA_REDIRECT_TUNNEL_TYPE_ALLOC = 0x10b
HWRM_CFA_REDIRECT_TUNNEL_TYPE_FREE = 0x10c
HWRM_CFA_PAIR_ALLOC = 0x10d
HWRM_CFA_PAIR_FREE = 0x10e
HWRM_CFA_PAIR_INFO = 0x10f
HWRM_FW_IPC_MSG = 0x110
HWRM_CFA_REDIRECT_TUNNEL_TYPE_INFO = 0x111
HWRM_CFA_REDIRECT_QUERY_TUNNEL_TYPE = 0x112
HWRM_CFA_FLOW_AGING_TIMER_RESET = 0x113
HWRM_CFA_FLOW_AGING_CFG = 0x114
HWRM_CFA_FLOW_AGING_QCFG = 0x115
HWRM_CFA_FLOW_AGING_QCAPS = 0x116
HWRM_CFA_CTX_MEM_RGTR = 0x117
HWRM_CFA_CTX_MEM_UNRGTR = 0x118
HWRM_CFA_CTX_MEM_QCTX = 0x119
HWRM_CFA_CTX_MEM_QCAPS = 0x11a
HWRM_CFA_COUNTER_QCAPS = 0x11b
HWRM_CFA_COUNTER_CFG = 0x11c
HWRM_CFA_COUNTER_QCFG = 0x11d
HWRM_CFA_COUNTER_QSTATS = 0x11e
HWRM_CFA_TCP_FLAG_PROCESS_QCFG = 0x11f
HWRM_CFA_EEM_QCAPS = 0x120
HWRM_CFA_EEM_CFG = 0x121
HWRM_CFA_EEM_QCFG = 0x122
HWRM_CFA_EEM_OP = 0x123
HWRM_CFA_ADV_FLOW_MGNT_QCAPS = 0x124
HWRM_CFA_TFLIB = 0x125
HWRM_CFA_LAG_GROUP_MEMBER_RGTR = 0x126
HWRM_CFA_LAG_GROUP_MEMBER_UNRGTR = 0x127
HWRM_CFA_TLS_FILTER_ALLOC = 0x128
HWRM_CFA_TLS_FILTER_FREE = 0x129
HWRM_CFA_RELEASE_AFM_FUNC = 0x12a
HWRM_ENGINE_CKV_STATUS = 0x12e
HWRM_ENGINE_CKV_CKEK_ADD = 0x12f
HWRM_ENGINE_CKV_CKEK_DELETE = 0x130
HWRM_ENGINE_CKV_KEY_ADD = 0x131
HWRM_ENGINE_CKV_KEY_DELETE = 0x132
HWRM_ENGINE_CKV_FLUSH = 0x133
HWRM_ENGINE_CKV_RNG_GET = 0x134
HWRM_ENGINE_CKV_KEY_GEN = 0x135
HWRM_ENGINE_CKV_KEY_LABEL_CFG = 0x136
HWRM_ENGINE_CKV_KEY_LABEL_QCFG = 0x137
HWRM_ENGINE_QG_CONFIG_QUERY = 0x13c
HWRM_ENGINE_QG_QUERY = 0x13d
HWRM_ENGINE_QG_METER_PROFILE_CONFIG_QUERY = 0x13e
HWRM_ENGINE_QG_METER_PROFILE_QUERY = 0x13f
HWRM_ENGINE_QG_METER_PROFILE_ALLOC = 0x140
HWRM_ENGINE_QG_METER_PROFILE_FREE = 0x141
HWRM_ENGINE_QG_METER_QUERY = 0x142
HWRM_ENGINE_QG_METER_BIND = 0x143
HWRM_ENGINE_QG_METER_UNBIND = 0x144
HWRM_ENGINE_QG_FUNC_BIND = 0x145
HWRM_ENGINE_SG_CONFIG_QUERY = 0x146
HWRM_ENGINE_SG_QUERY = 0x147
HWRM_ENGINE_SG_METER_QUERY = 0x148
HWRM_ENGINE_SG_METER_CONFIG = 0x149
HWRM_ENGINE_SG_QG_BIND = 0x14a
HWRM_ENGINE_QG_SG_UNBIND = 0x14b
HWRM_ENGINE_CONFIG_QUERY = 0x154
HWRM_ENGINE_STATS_CONFIG = 0x155
HWRM_ENGINE_STATS_CLEAR = 0x156
HWRM_ENGINE_STATS_QUERY = 0x157
HWRM_ENGINE_STATS_QUERY_CONTINUOUS_ERROR = 0x158
HWRM_ENGINE_RQ_ALLOC = 0x15e
HWRM_ENGINE_RQ_FREE = 0x15f
HWRM_ENGINE_CQ_ALLOC = 0x160
HWRM_ENGINE_CQ_FREE = 0x161
HWRM_ENGINE_NQ_ALLOC = 0x162
HWRM_ENGINE_NQ_FREE = 0x163
HWRM_ENGINE_ON_DIE_RQE_CREDITS = 0x164
HWRM_ENGINE_FUNC_QCFG = 0x165
HWRM_FUNC_RESOURCE_QCAPS = 0x190
HWRM_FUNC_VF_RESOURCE_CFG = 0x191
HWRM_FUNC_BACKING_STORE_QCAPS = 0x192
HWRM_FUNC_BACKING_STORE_CFG = 0x193
HWRM_FUNC_BACKING_STORE_QCFG = 0x194
HWRM_FUNC_VF_BW_CFG = 0x195
HWRM_FUNC_VF_BW_QCFG = 0x196
HWRM_FUNC_HOST_PF_IDS_QUERY = 0x197
HWRM_FUNC_QSTATS_EXT = 0x198
HWRM_STAT_EXT_CTX_QUERY = 0x199
HWRM_FUNC_SPD_CFG = 0x19a
HWRM_FUNC_SPD_QCFG = 0x19b
HWRM_FUNC_PTP_PIN_QCFG = 0x19c
HWRM_FUNC_PTP_PIN_CFG = 0x19d
HWRM_FUNC_PTP_CFG = 0x19e
HWRM_FUNC_PTP_TS_QUERY = 0x19f
HWRM_FUNC_PTP_EXT_CFG = 0x1a0
HWRM_FUNC_PTP_EXT_QCFG = 0x1a1
HWRM_FUNC_KEY_CTX_ALLOC = 0x1a2
HWRM_FUNC_BACKING_STORE_CFG_V2 = 0x1a3
HWRM_FUNC_BACKING_STORE_QCFG_V2 = 0x1a4
HWRM_FUNC_DBR_PACING_CFG = 0x1a5
HWRM_FUNC_DBR_PACING_QCFG = 0x1a6
HWRM_FUNC_DBR_PACING_BROADCAST_EVENT = 0x1a7
HWRM_FUNC_BACKING_STORE_QCAPS_V2 = 0x1a8
HWRM_FUNC_DBR_PACING_NQLIST_QUERY = 0x1a9
HWRM_FUNC_DBR_RECOVERY_COMPLETED = 0x1aa
HWRM_FUNC_SYNCE_CFG = 0x1ab
HWRM_FUNC_SYNCE_QCFG = 0x1ac
HWRM_FUNC_KEY_CTX_FREE = 0x1ad
HWRM_FUNC_LAG_MODE_CFG = 0x1ae
HWRM_FUNC_LAG_MODE_QCFG = 0x1af
HWRM_FUNC_LAG_CREATE = 0x1b0
HWRM_FUNC_LAG_UPDATE = 0x1b1
HWRM_FUNC_LAG_FREE = 0x1b2
HWRM_FUNC_LAG_QCFG = 0x1b3
HWRM_FUNC_TTX_PACING_RATE_PROF_QUERY = 0x1c3
HWRM_FUNC_TTX_PACING_RATE_QUERY = 0x1c4
HWRM_SELFTEST_QLIST = 0x200
HWRM_SELFTEST_EXEC = 0x201
HWRM_SELFTEST_IRQ = 0x202
HWRM_SELFTEST_RETRIEVE_SERDES_DATA = 0x203
HWRM_PCIE_QSTATS = 0x204
HWRM_MFG_FRU_WRITE_CONTROL = 0x205
HWRM_MFG_TIMERS_QUERY = 0x206
HWRM_MFG_OTP_CFG = 0x207
HWRM_MFG_OTP_QCFG = 0x208
HWRM_MFG_HDMA_TEST = 0x209
HWRM_MFG_FRU_EEPROM_WRITE = 0x20a
HWRM_MFG_FRU_EEPROM_READ = 0x20b
HWRM_MFG_SOC_IMAGE = 0x20c
HWRM_MFG_SOC_QSTATUS = 0x20d
HWRM_MFG_PARAM_CRITICAL_DATA_FINALIZE = 0x20e
HWRM_MFG_PARAM_CRITICAL_DATA_READ = 0x20f
HWRM_MFG_PARAM_CRITICAL_DATA_HEALTH = 0x210
HWRM_MFG_PRVSN_EXPORT_CSR = 0x211
HWRM_MFG_PRVSN_IMPORT_CERT = 0x212
HWRM_MFG_PRVSN_GET_STATE = 0x213
HWRM_MFG_GET_NVM_MEASUREMENT = 0x214
HWRM_MFG_PSOC_QSTATUS = 0x215
HWRM_MFG_SELFTEST_QLIST = 0x216
HWRM_MFG_SELFTEST_EXEC = 0x217
HWRM_STAT_GENERIC_QSTATS = 0x218
HWRM_MFG_PRVSN_EXPORT_CERT = 0x219
HWRM_STAT_DB_ERROR_QSTATS = 0x21a
HWRM_MFG_TESTS = 0x21b
HWRM_MFG_WRITE_CERT_NVM = 0x21c
HWRM_PORT_POE_CFG = 0x230
HWRM_PORT_POE_QCFG = 0x231
HWRM_PORT_PHY_FDRSTAT = 0x232
HWRM_UDCC_QCAPS = 0x258
HWRM_UDCC_CFG = 0x259
HWRM_UDCC_QCFG = 0x25a
HWRM_UDCC_SESSION_CFG = 0x25b
HWRM_UDCC_SESSION_QCFG = 0x25c
HWRM_UDCC_SESSION_QUERY = 0x25d
HWRM_UDCC_COMP_CFG = 0x25e
HWRM_UDCC_COMP_QCFG = 0x25f
HWRM_UDCC_COMP_QUERY = 0x260
HWRM_QUEUE_PFCWD_TIMEOUT_QCAPS = 0x261
HWRM_QUEUE_PFCWD_TIMEOUT_CFG = 0x262
HWRM_QUEUE_PFCWD_TIMEOUT_QCFG = 0x263
HWRM_QUEUE_ADPTV_QOS_RX_QCFG = 0x264
HWRM_QUEUE_ADPTV_QOS_TX_QCFG = 0x265
HWRM_TF = 0x2bc
HWRM_TF_VERSION_GET = 0x2bd
HWRM_TF_SESSION_OPEN = 0x2c6
HWRM_TF_SESSION_REGISTER = 0x2c8
HWRM_TF_SESSION_UNREGISTER = 0x2c9
HWRM_TF_SESSION_CLOSE = 0x2ca
HWRM_TF_SESSION_QCFG = 0x2cb
HWRM_TF_SESSION_RESC_QCAPS = 0x2cc
HWRM_TF_SESSION_RESC_ALLOC = 0x2cd
HWRM_TF_SESSION_RESC_FREE = 0x2ce
HWRM_TF_SESSION_RESC_FLUSH = 0x2cf
HWRM_TF_SESSION_RESC_INFO = 0x2d0
HWRM_TF_SESSION_HOTUP_STATE_SET = 0x2d1
HWRM_TF_SESSION_HOTUP_STATE_GET = 0x2d2
HWRM_TF_TBL_TYPE_GET = 0x2da
HWRM_TF_TBL_TYPE_SET = 0x2db
HWRM_TF_TBL_TYPE_BULK_GET = 0x2dc
HWRM_TF_EM_INSERT = 0x2ea
HWRM_TF_EM_DELETE = 0x2eb
HWRM_TF_EM_HASH_INSERT = 0x2ec
HWRM_TF_EM_MOVE = 0x2ed
HWRM_TF_TCAM_SET = 0x2f8
HWRM_TF_TCAM_GET = 0x2f9
HWRM_TF_TCAM_MOVE = 0x2fa
HWRM_TF_TCAM_FREE = 0x2fb
HWRM_TF_GLOBAL_CFG_SET = 0x2fc
HWRM_TF_GLOBAL_CFG_GET = 0x2fd
HWRM_TF_IF_TBL_SET = 0x2fe
HWRM_TF_IF_TBL_GET = 0x2ff
HWRM_TF_RESC_USAGE_SET = 0x300
HWRM_TF_RESC_USAGE_QUERY = 0x301
HWRM_TF_TBL_TYPE_ALLOC = 0x302
HWRM_TF_TBL_TYPE_FREE = 0x303
HWRM_TFC_TBL_SCOPE_QCAPS = 0x380
HWRM_TFC_TBL_SCOPE_ID_ALLOC = 0x381
HWRM_TFC_TBL_SCOPE_CONFIG = 0x382
HWRM_TFC_TBL_SCOPE_DECONFIG = 0x383
HWRM_TFC_TBL_SCOPE_FID_ADD = 0x384
HWRM_TFC_TBL_SCOPE_FID_REM = 0x385
HWRM_TFC_TBL_SCOPE_POOL_ALLOC = 0x386
HWRM_TFC_TBL_SCOPE_POOL_FREE = 0x387
HWRM_TFC_SESSION_ID_ALLOC = 0x388
HWRM_TFC_SESSION_FID_ADD = 0x389
HWRM_TFC_SESSION_FID_REM = 0x38a
HWRM_TFC_IDENT_ALLOC = 0x38b
HWRM_TFC_IDENT_FREE = 0x38c
HWRM_TFC_IDX_TBL_ALLOC = 0x38d
HWRM_TFC_IDX_TBL_ALLOC_SET = 0x38e
HWRM_TFC_IDX_TBL_SET = 0x38f
HWRM_TFC_IDX_TBL_GET = 0x390
HWRM_TFC_IDX_TBL_FREE = 0x391
HWRM_TFC_GLOBAL_ID_ALLOC = 0x392
HWRM_TFC_TCAM_SET = 0x393
HWRM_TFC_TCAM_GET = 0x394
HWRM_TFC_TCAM_ALLOC = 0x395
HWRM_TFC_TCAM_ALLOC_SET = 0x396
HWRM_TFC_TCAM_FREE = 0x397
HWRM_TFC_IF_TBL_SET = 0x398
HWRM_TFC_IF_TBL_GET = 0x399
HWRM_TFC_TBL_SCOPE_CONFIG_GET = 0x39a
HWRM_TFC_RESC_USAGE_QUERY = 0x39b
HWRM_TFC_GLOBAL_ID_FREE = 0x39c
HWRM_TFC_TCAM_PRI_UPDATE = 0x39d
HWRM_TFC_HOT_UPGRADE_PROCESS = 0x3a0
HWRM_SV = 0x400
HWRM_DBG_SERDES_TEST = 0xff0e
HWRM_DBG_LOG_BUFFER_FLUSH = 0xff0f
HWRM_DBG_READ_DIRECT = 0xff10
HWRM_DBG_READ_INDIRECT = 0xff11
HWRM_DBG_WRITE_DIRECT = 0xff12
HWRM_DBG_WRITE_INDIRECT = 0xff13
HWRM_DBG_DUMP = 0xff14
HWRM_DBG_ERASE_NVM = 0xff15
HWRM_DBG_CFG = 0xff16
HWRM_DBG_COREDUMP_LIST = 0xff17
HWRM_DBG_COREDUMP_INITIATE = 0xff18
HWRM_DBG_COREDUMP_RETRIEVE = 0xff19
HWRM_DBG_FW_CLI = 0xff1a
HWRM_DBG_I2C_CMD = 0xff1b
HWRM_DBG_RING_INFO_GET = 0xff1c
HWRM_DBG_CRASHDUMP_HEADER = 0xff1d
HWRM_DBG_CRASHDUMP_ERASE = 0xff1e
HWRM_DBG_DRV_TRACE = 0xff1f
HWRM_DBG_QCAPS = 0xff20
HWRM_DBG_QCFG = 0xff21
HWRM_DBG_CRASHDUMP_MEDIUM_CFG = 0xff22
HWRM_DBG_USEQ_ALLOC = 0xff23
HWRM_DBG_USEQ_FREE = 0xff24
HWRM_DBG_USEQ_FLUSH = 0xff25
HWRM_DBG_USEQ_QCAPS = 0xff26
HWRM_DBG_USEQ_CW_CFG = 0xff27
HWRM_DBG_USEQ_SCHED_CFG = 0xff28
HWRM_DBG_USEQ_RUN = 0xff29
HWRM_DBG_USEQ_DELIVERY_REQ = 0xff2a
HWRM_DBG_USEQ_RESP_HDR = 0xff2b
HWRM_DBG_COREDUMP_CAPTURE = 0xff2c
HWRM_DBG_PTRACE = 0xff2d
HWRM_DBG_SIM_CABLE_STATE = 0xff2e
HWRM_DBG_TOKEN_QUERY_AUTH_IDS = 0xff2f
HWRM_DBG_TOKEN_CFG = 0xff30
HWRM_NVM_GET_VPD_FIELD_INFO = 0xffea
HWRM_NVM_SET_VPD_FIELD_INFO = 0xffeb
HWRM_NVM_DEFRAG = 0xffec
HWRM_NVM_REQ_ARBITRATION = 0xffed
HWRM_NVM_FACTORY_DEFAULTS = 0xffee
HWRM_NVM_VALIDATE_OPTION = 0xffef
HWRM_NVM_FLUSH = 0xfff0
HWRM_NVM_GET_VARIABLE = 0xfff1
HWRM_NVM_SET_VARIABLE = 0xfff2
HWRM_NVM_INSTALL_UPDATE = 0xfff3
HWRM_NVM_MODIFY = 0xfff4
HWRM_NVM_VERIFY_UPDATE = 0xfff5
HWRM_NVM_GET_DEV_INFO = 0xfff6
HWRM_NVM_ERASE_DIR_ENTRY = 0xfff7
HWRM_NVM_MOD_DIR_ENTRY = 0xfff8
HWRM_NVM_FIND_DIR_ENTRY = 0xfff9
HWRM_NVM_GET_DIR_ENTRIES = 0xfffa
HWRM_NVM_GET_DIR_INFO = 0xfffb
HWRM_NVM_RAW_DUMP = 0xfffc
HWRM_NVM_READ = 0xfffd
HWRM_NVM_WRITE = 0xfffe
HWRM_NVM_RAW_WRITE_BLK = 0xffff
HWRM_LAST = HWRM_NVM_RAW_WRITE_BLK
HWRM_ERR_CODE_SUCCESS = 0x0
HWRM_ERR_CODE_FAIL = 0x1
HWRM_ERR_CODE_INVALID_PARAMS = 0x2
HWRM_ERR_CODE_RESOURCE_ACCESS_DENIED = 0x3
HWRM_ERR_CODE_RESOURCE_ALLOC_ERROR = 0x4
HWRM_ERR_CODE_INVALID_FLAGS = 0x5
HWRM_ERR_CODE_INVALID_ENABLES = 0x6
HWRM_ERR_CODE_UNSUPPORTED_TLV = 0x7
HWRM_ERR_CODE_NO_BUFFER = 0x8
HWRM_ERR_CODE_UNSUPPORTED_OPTION_ERR = 0x9
HWRM_ERR_CODE_HOT_RESET_PROGRESS = 0xa
HWRM_ERR_CODE_HOT_RESET_FAIL = 0xb
HWRM_ERR_CODE_NO_FLOW_COUNTER_DURING_ALLOC = 0xc
HWRM_ERR_CODE_KEY_HASH_COLLISION = 0xd
HWRM_ERR_CODE_KEY_ALREADY_EXISTS = 0xe
HWRM_ERR_CODE_HWRM_ERROR = 0xf
HWRM_ERR_CODE_BUSY = 0x10
HWRM_ERR_CODE_RESOURCE_LOCKED = 0x11
HWRM_ERR_CODE_PF_UNAVAILABLE = 0x12
HWRM_ERR_CODE_ENTITY_NOT_PRESENT = 0x13
HWRM_ERR_CODE_SECURE_SOC_ERROR = 0x14
HWRM_ERR_CODE_TLV_ENCAPSULATED_RESPONSE = 0x8000
HWRM_ERR_CODE_UNKNOWN_ERR = 0xfffe
HWRM_ERR_CODE_CMD_NOT_SUPPORTED = 0xffff
HWRM_ERR_CODE_LAST = HWRM_ERR_CODE_CMD_NOT_SUPPORTED
HWRM_MAX_REQ_LEN = 128
HWRM_MAX_RESP_LEN = 704
HWRM_RESP_VALID_KEY = 1
HWRM_TARGET_ID_BONO = 0xFFF8
HWRM_TARGET_ID_KONG = 0xFFF9
HWRM_TARGET_ID_APE = 0xFFFA
HWRM_TARGET_ID_TOOLS = 0xFFFD
HWRM_VERSION_MAJOR = 1
HWRM_VERSION_MINOR = 10
HWRM_VERSION_UPDATE = 3
HWRM_VERSION_RSVD = 133
HWRM_VERSION_STR = "1.10.3.133"
FUNC_RESET_REQ_ENABLES_VF_ID_VALID = 0x1
FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETALL = 0x0
FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETME = 0x1
FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETCHILDREN = 0x2
FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETVF = 0x3
FUNC_RESET_REQ_FUNC_RESET_LEVEL_LAST = FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETVF
FUNC_QCAPS_RESP_FLAGS_PUSH_MODE_SUPPORTED = 0x1
FUNC_QCAPS_RESP_FLAGS_GLOBAL_MSIX_AUTOMASKING = 0x2
FUNC_QCAPS_RESP_FLAGS_PTP_SUPPORTED = 0x4
FUNC_QCAPS_RESP_FLAGS_ROCE_V1_SUPPORTED = 0x8
FUNC_QCAPS_RESP_FLAGS_ROCE_V2_SUPPORTED = 0x10
FUNC_QCAPS_RESP_FLAGS_WOL_MAGICPKT_SUPPORTED = 0x20
FUNC_QCAPS_RESP_FLAGS_WOL_BMP_SUPPORTED = 0x40
FUNC_QCAPS_RESP_FLAGS_TX_RING_RL_SUPPORTED = 0x80
FUNC_QCAPS_RESP_FLAGS_TX_BW_CFG_SUPPORTED = 0x100
FUNC_QCAPS_RESP_FLAGS_VF_TX_RING_RL_SUPPORTED = 0x200
FUNC_QCAPS_RESP_FLAGS_VF_BW_CFG_SUPPORTED = 0x400
FUNC_QCAPS_RESP_FLAGS_STD_TX_RING_MODE_SUPPORTED = 0x800
FUNC_QCAPS_RESP_FLAGS_GENEVE_TUN_FLAGS_SUPPORTED = 0x1000
FUNC_QCAPS_RESP_FLAGS_NVGRE_TUN_FLAGS_SUPPORTED = 0x2000
FUNC_QCAPS_RESP_FLAGS_GRE_TUN_FLAGS_SUPPORTED = 0x4000
FUNC_QCAPS_RESP_FLAGS_MPLS_TUN_FLAGS_SUPPORTED = 0x8000
FUNC_QCAPS_RESP_FLAGS_PCIE_STATS_SUPPORTED = 0x10000
FUNC_QCAPS_RESP_FLAGS_ADOPTED_PF_SUPPORTED = 0x20000
FUNC_QCAPS_RESP_FLAGS_ADMIN_PF_SUPPORTED = 0x40000
FUNC_QCAPS_RESP_FLAGS_LINK_ADMIN_STATUS_SUPPORTED = 0x80000
FUNC_QCAPS_RESP_FLAGS_WCB_PUSH_MODE = 0x100000
FUNC_QCAPS_RESP_FLAGS_DYNAMIC_TX_RING_ALLOC = 0x200000
FUNC_QCAPS_RESP_FLAGS_HOT_RESET_CAPABLE = 0x400000
FUNC_QCAPS_RESP_FLAGS_ERROR_RECOVERY_CAPABLE = 0x800000
FUNC_QCAPS_RESP_FLAGS_EXT_STATS_SUPPORTED = 0x1000000
FUNC_QCAPS_RESP_FLAGS_ERR_RECOVER_RELOAD = 0x2000000
FUNC_QCAPS_RESP_FLAGS_NOTIFY_VF_DEF_VNIC_CHNG_SUPPORTED = 0x4000000
FUNC_QCAPS_RESP_FLAGS_VLAN_ACCELERATION_TX_DISABLED = 0x8000000
FUNC_QCAPS_RESP_FLAGS_COREDUMP_CMD_SUPPORTED = 0x10000000
FUNC_QCAPS_RESP_FLAGS_CRASHDUMP_CMD_SUPPORTED = 0x20000000
FUNC_QCAPS_RESP_FLAGS_PFC_WD_STATS_SUPPORTED = 0x40000000
FUNC_QCAPS_RESP_FLAGS_DBG_QCAPS_CMD_SUPPORTED = 0x80000000
FUNC_QCAPS_RESP_FLAGS_EXT_ECN_MARK_SUPPORTED = 0x1
FUNC_QCAPS_RESP_FLAGS_EXT_ECN_STATS_SUPPORTED = 0x2
FUNC_QCAPS_RESP_FLAGS_EXT_EXT_HW_STATS_SUPPORTED = 0x4
FUNC_QCAPS_RESP_FLAGS_EXT_HOT_RESET_IF_SUPPORT = 0x8
FUNC_QCAPS_RESP_FLAGS_EXT_PROXY_MODE_SUPPORT = 0x10
FUNC_QCAPS_RESP_FLAGS_EXT_TX_PROXY_SRC_INTF_OVERRIDE_SUPPORT = 0x20
FUNC_QCAPS_RESP_FLAGS_EXT_SCHQ_SUPPORTED = 0x40
FUNC_QCAPS_RESP_FLAGS_EXT_PPP_PUSH_MODE_SUPPORTED = 0x80
FUNC_QCAPS_RESP_FLAGS_EXT_EVB_MODE_CFG_NOT_SUPPORTED = 0x100
FUNC_QCAPS_RESP_FLAGS_EXT_SOC_SPD_SUPPORTED = 0x200
FUNC_QCAPS_RESP_FLAGS_EXT_FW_LIVEPATCH_SUPPORTED = 0x400
FUNC_QCAPS_RESP_FLAGS_EXT_FAST_RESET_CAPABLE = 0x800
FUNC_QCAPS_RESP_FLAGS_EXT_TX_METADATA_CFG_CAPABLE = 0x1000
FUNC_QCAPS_RESP_FLAGS_EXT_NVM_OPTION_ACTION_SUPPORTED = 0x2000
FUNC_QCAPS_RESP_FLAGS_EXT_BD_METADATA_SUPPORTED = 0x4000
FUNC_QCAPS_RESP_FLAGS_EXT_ECHO_REQUEST_SUPPORTED = 0x8000
FUNC_QCAPS_RESP_FLAGS_EXT_NPAR_1_2_SUPPORTED = 0x10000
FUNC_QCAPS_RESP_FLAGS_EXT_PTP_PTM_SUPPORTED = 0x20000
FUNC_QCAPS_RESP_FLAGS_EXT_PTP_PPS_SUPPORTED = 0x40000
FUNC_QCAPS_RESP_FLAGS_EXT_VF_CFG_ASYNC_FOR_PF_SUPPORTED = 0x80000
FUNC_QCAPS_RESP_FLAGS_EXT_PARTITION_BW_SUPPORTED = 0x100000
FUNC_QCAPS_RESP_FLAGS_EXT_DFLT_VLAN_TPID_PCP_SUPPORTED = 0x200000
FUNC_QCAPS_RESP_FLAGS_EXT_KTLS_SUPPORTED = 0x400000
FUNC_QCAPS_RESP_FLAGS_EXT_EP_RATE_CONTROL = 0x800000
FUNC_QCAPS_RESP_FLAGS_EXT_MIN_BW_SUPPORTED = 0x1000000
FUNC_QCAPS_RESP_FLAGS_EXT_TX_COAL_CMPL_CAP = 0x2000000
FUNC_QCAPS_RESP_FLAGS_EXT_BS_V2_SUPPORTED = 0x4000000
FUNC_QCAPS_RESP_FLAGS_EXT_BS_V2_REQUIRED = 0x8000000
FUNC_QCAPS_RESP_FLAGS_EXT_PTP_64BIT_RTC_SUPPORTED = 0x10000000
FUNC_QCAPS_RESP_FLAGS_EXT_DBR_PACING_SUPPORTED = 0x20000000
FUNC_QCAPS_RESP_FLAGS_EXT_HW_DBR_DROP_RECOV_SUPPORTED = 0x40000000
FUNC_QCAPS_RESP_FLAGS_EXT_DISABLE_CQ_OVERFLOW_DETECTION_SUPPORTED = 0x80000000
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_TCE = 0x1
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_RCE = 0x2
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_TE_CFA = 0x4
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_RE_CFA = 0x8
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_PRIMATE = 0x10
FUNC_QCAPS_RESP_FLAGS_EXT2_RX_ALL_PKTS_TIMESTAMPS_SUPPORTED = 0x1
FUNC_QCAPS_RESP_FLAGS_EXT2_QUIC_SUPPORTED = 0x2
FUNC_QCAPS_RESP_FLAGS_EXT2_KDNET_SUPPORTED = 0x4
FUNC_QCAPS_RESP_FLAGS_EXT2_DBR_PACING_EXT_SUPPORTED = 0x8
FUNC_QCAPS_RESP_FLAGS_EXT2_SW_DBR_DROP_RECOVERY_SUPPORTED = 0x10
FUNC_QCAPS_RESP_FLAGS_EXT2_GENERIC_STATS_SUPPORTED = 0x20
FUNC_QCAPS_RESP_FLAGS_EXT2_UDP_GSO_SUPPORTED = 0x40
FUNC_QCAPS_RESP_FLAGS_EXT2_SYNCE_SUPPORTED = 0x80
FUNC_QCAPS_RESP_FLAGS_EXT2_DBR_PACING_V0_SUPPORTED = 0x100
FUNC_QCAPS_RESP_FLAGS_EXT2_TX_PKT_TS_CMPL_SUPPORTED = 0x200
FUNC_QCAPS_RESP_FLAGS_EXT2_HW_LAG_SUPPORTED = 0x400
FUNC_QCAPS_RESP_FLAGS_EXT2_ON_CHIP_CTX_SUPPORTED = 0x800
FUNC_QCAPS_RESP_FLAGS_EXT2_STEERING_TAG_SUPPORTED = 0x1000
FUNC_QCAPS_RESP_FLAGS_EXT2_ENHANCED_VF_SCALE_SUPPORTED = 0x2000
FUNC_QCAPS_RESP_FLAGS_EXT2_KEY_XID_PARTITION_SUPPORTED = 0x4000
FUNC_QCAPS_RESP_FLAGS_EXT2_CONCURRENT_KTLS_QUIC_SUPPORTED = 0x8000
FUNC_QCAPS_RESP_FLAGS_EXT2_SCHQ_CROSS_TC_CAP_SUPPORTED = 0x10000
FUNC_QCAPS_RESP_FLAGS_EXT2_SCHQ_PER_TC_CAP_SUPPORTED = 0x20000
FUNC_QCAPS_RESP_FLAGS_EXT2_SCHQ_PER_TC_RESERVATION_SUPPORTED = 0x40000
FUNC_QCAPS_RESP_FLAGS_EXT2_DB_ERROR_STATS_SUPPORTED = 0x80000
FUNC_QCAPS_RESP_FLAGS_EXT2_ROCE_VF_RESOURCE_MGMT_SUPPORTED = 0x100000
FUNC_QCAPS_RESP_FLAGS_EXT2_UDCC_SUPPORTED = 0x200000
FUNC_QCAPS_RESP_FLAGS_EXT2_TIMED_TX_SO_TXTIME_SUPPORTED = 0x400000
FUNC_QCAPS_RESP_FLAGS_EXT2_SW_MAX_RESOURCE_LIMITS_SUPPORTED = 0x800000
FUNC_QCAPS_RESP_FLAGS_EXT2_TF_INGRESS_NIC_FLOW_SUPPORTED = 0x1000000
FUNC_QCAPS_RESP_FLAGS_EXT2_LPBK_STATS_SUPPORTED = 0x2000000
FUNC_QCAPS_RESP_FLAGS_EXT2_TF_EGRESS_NIC_FLOW_SUPPORTED = 0x4000000
FUNC_QCAPS_RESP_FLAGS_EXT2_MULTI_LOSSLESS_QUEUES_SUPPORTED = 0x8000000
FUNC_QCAPS_RESP_FLAGS_EXT2_PEER_MMAP_SUPPORTED = 0x10000000
FUNC_QCAPS_RESP_FLAGS_EXT2_TIMED_TX_PACING_SUPPORTED = 0x20000000
FUNC_QCAPS_RESP_FLAGS_EXT2_VF_STAT_EJECTION_SUPPORTED = 0x40000000
FUNC_QCAPS_RESP_FLAGS_EXT2_HOST_COREDUMP_SUPPORTED = 0x80000000
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_VXLAN = 0x1
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_NGE = 0x2
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_NVGRE = 0x4
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_L2GRE = 0x8
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_GRE = 0x10
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_IPINIP = 0x20
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_MPLS = 0x40
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_PPPOE = 0x80
FUNC_QCAPS_RESP_XID_PARTITION_CAP_TX_CK = 0x1
FUNC_QCAPS_RESP_XID_PARTITION_CAP_RX_CK = 0x2
FUNC_QCAPS_RESP_FLAGS_EXT3_RM_RSV_WHILE_ALLOC_CAP = 0x1
FUNC_QCAPS_RESP_FLAGS_EXT3_REQUIRE_L2_FILTER = 0x2
FUNC_QCAPS_RESP_FLAGS_EXT3_MAX_ROCE_VFS_SUPPORTED = 0x4
FUNC_QCAPS_RESP_FLAGS_EXT3_RX_RATE_PROFILE_SEL_SUPPORTED = 0x8
FUNC_QCAPS_RESP_FLAGS_EXT3_BIDI_OPT_SUPPORTED = 0x10
FUNC_QCAPS_RESP_FLAGS_EXT3_MIRROR_ON_ROCE_SUPPORTED = 0x20
FUNC_QCAPS_RESP_FLAGS_EXT3_ROCE_VF_DYN_ALLOC_SUPPORT = 0x40
FUNC_QCAPS_RESP_FLAGS_EXT3_CHANGE_UDP_SRCPORT_SUPPORT = 0x80
FUNC_QCAPS_RESP_FLAGS_EXT3_PCIE_COMPLIANCE_SUPPORTED = 0x100
FUNC_QCAPS_RESP_FLAGS_EXT3_MULTI_L2_DB_SUPPORTED = 0x200
FUNC_QCAPS_RESP_FLAGS_EXT3_PCIE_SECURE_ATS_SUPPORTED = 0x400
FUNC_QCAPS_RESP_FLAGS_EXT3_MBUF_STATS_SUPPORTED = 0x800
FUNC_QCFG_RESP_FLAGS_OOB_WOL_MAGICPKT_ENABLED = 0x1
FUNC_QCFG_RESP_FLAGS_OOB_WOL_BMP_ENABLED = 0x2
FUNC_QCFG_RESP_FLAGS_FW_DCBX_AGENT_ENABLED = 0x4
FUNC_QCFG_RESP_FLAGS_STD_TX_RING_MODE_ENABLED = 0x8
FUNC_QCFG_RESP_FLAGS_FW_LLDP_AGENT_ENABLED = 0x10
FUNC_QCFG_RESP_FLAGS_MULTI_HOST = 0x20
FUNC_QCFG_RESP_FLAGS_TRUSTED_VF = 0x40
FUNC_QCFG_RESP_FLAGS_SECURE_MODE_ENABLED = 0x80
FUNC_QCFG_RESP_FLAGS_PREBOOT_LEGACY_L2_RINGS = 0x100
FUNC_QCFG_RESP_FLAGS_HOT_RESET_ALLOWED = 0x200
FUNC_QCFG_RESP_FLAGS_PPP_PUSH_MODE_ENABLED = 0x400
FUNC_QCFG_RESP_FLAGS_RING_MONITOR_ENABLED = 0x800
FUNC_QCFG_RESP_FLAGS_FAST_RESET_ALLOWED = 0x1000
FUNC_QCFG_RESP_FLAGS_MULTI_ROOT = 0x2000
FUNC_QCFG_RESP_FLAGS_ENABLE_RDMA_SRIOV = 0x4000
FUNC_QCFG_RESP_FLAGS_ROCE_VNIC_ID_VALID = 0x8000
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_SPF = 0x0
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_MPFS = 0x1
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_NPAR1_0 = 0x2
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_NPAR1_5 = 0x3
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_NPAR2_0 = 0x4
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_NPAR1_2 = 0x5
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_UNKNOWN = 0xff
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_LAST = FUNC_QCFG_RESP_PORT_PARTITION_TYPE_UNKNOWN
FUNC_QCFG_RESP_PORT_PF_CNT_UNAVAIL = 0x0
FUNC_QCFG_RESP_PORT_PF_CNT_LAST = FUNC_QCFG_RESP_PORT_PF_CNT_UNAVAIL
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_MASK = 0xfffffff
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_SFT = 0
FUNC_QCFG_RESP_MIN_BW_SCALE = 0x10000000
FUNC_QCFG_RESP_MIN_BW_SCALE_BITS = (0x0 << 28)
FUNC_QCFG_RESP_MIN_BW_SCALE_BYTES = (0x1 << 28)
FUNC_QCFG_RESP_MIN_BW_SCALE_LAST = FUNC_QCFG_RESP_MIN_BW_SCALE_BYTES
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_SFT = 29
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_LAST = FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_INVALID
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_MASK = 0xfffffff
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_SFT = 0
FUNC_QCFG_RESP_MAX_BW_SCALE = 0x10000000
FUNC_QCFG_RESP_MAX_BW_SCALE_BITS = (0x0 << 28)
FUNC_QCFG_RESP_MAX_BW_SCALE_BYTES = (0x1 << 28)
FUNC_QCFG_RESP_MAX_BW_SCALE_LAST = FUNC_QCFG_RESP_MAX_BW_SCALE_BYTES
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_SFT = 29
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_LAST = FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_INVALID
FUNC_QCFG_RESP_EVB_MODE_NO_EVB = 0x0
FUNC_QCFG_RESP_EVB_MODE_VEB = 0x1
FUNC_QCFG_RESP_EVB_MODE_VEPA = 0x2
FUNC_QCFG_RESP_EVB_MODE_LAST = FUNC_QCFG_RESP_EVB_MODE_VEPA
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_MASK = 0x3
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_SFT = 0
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_SIZE_64 = 0x0
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_SIZE_128 = 0x1
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_LAST = FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_SIZE_128
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_MASK = 0xc
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_SFT = 2
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_FORCED_DOWN = (0x0 << 2)
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_FORCED_UP = (0x1 << 2)
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_AUTO = (0x2 << 2)
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_LAST = FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_AUTO
FUNC_QCFG_RESP_OPTIONS_RSVD_MASK = 0xf0
FUNC_QCFG_RESP_OPTIONS_RSVD_SFT = 4
FUNC_QCFG_RESP_SVIF_INFO_SVIF_MASK = 0x7fff
FUNC_QCFG_RESP_SVIF_INFO_SVIF_SFT = 0
FUNC_QCFG_RESP_SVIF_INFO_SVIF_VALID = 0x8000
FUNC_QCFG_RESP_MPC_CHNLS_TCE_ENABLED = 0x1
FUNC_QCFG_RESP_MPC_CHNLS_RCE_ENABLED = 0x2
FUNC_QCFG_RESP_MPC_CHNLS_TE_CFA_ENABLED = 0x4
FUNC_QCFG_RESP_MPC_CHNLS_RE_CFA_ENABLED = 0x8
FUNC_QCFG_RESP_MPC_CHNLS_PRIMATE_ENABLED = 0x10
FUNC_QCFG_RESP_DB_PAGE_SIZE_4KB = 0x0
FUNC_QCFG_RESP_DB_PAGE_SIZE_8KB = 0x1
FUNC_QCFG_RESP_DB_PAGE_SIZE_16KB = 0x2
FUNC_QCFG_RESP_DB_PAGE_SIZE_32KB = 0x3
FUNC_QCFG_RESP_DB_PAGE_SIZE_64KB = 0x4
FUNC_QCFG_RESP_DB_PAGE_SIZE_128KB = 0x5
FUNC_QCFG_RESP_DB_PAGE_SIZE_256KB = 0x6
FUNC_QCFG_RESP_DB_PAGE_SIZE_512KB = 0x7
FUNC_QCFG_RESP_DB_PAGE_SIZE_1MB = 0x8
FUNC_QCFG_RESP_DB_PAGE_SIZE_2MB = 0x9
FUNC_QCFG_RESP_DB_PAGE_SIZE_4MB = 0xa
FUNC_QCFG_RESP_DB_PAGE_SIZE_LAST = FUNC_QCFG_RESP_DB_PAGE_SIZE_4MB
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_MASK = 0xfffffff
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_SFT = 0
FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE = 0x10000000
FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE_BITS = (0x0 << 28)
FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE_BYTES = (0x1 << 28)
FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE_LAST = FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE_BYTES
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_SFT = 29
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_LAST = FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_PERCENT1_100
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_MASK = 0xfffffff
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_SFT = 0
FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE = 0x10000000
FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE_BITS = (0x0 << 28)
FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE_BYTES = (0x1 << 28)
FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE_LAST = FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE_BYTES
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_SFT = 29
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_LAST = FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_PERCENT1_100
FUNC_QCFG_RESP_FLAGS2_SRIOV_DSCP_INSERT_ENABLED = 0x1
FUNC_QCFG_RESP_PORT_KDNET_MODE_DISABLED = 0x0
FUNC_QCFG_RESP_PORT_KDNET_MODE_ENABLED = 0x1
FUNC_QCFG_RESP_PORT_KDNET_MODE_LAST = FUNC_QCFG_RESP_PORT_KDNET_MODE_ENABLED
FUNC_QCFG_RESP_ROCE_BIDI_OPT_MODE_DISABLED = 0x1
FUNC_QCFG_RESP_ROCE_BIDI_OPT_MODE_DEDICATED = 0x2
FUNC_QCFG_RESP_ROCE_BIDI_OPT_MODE_SHARED = 0x4
FUNC_QCFG_RESP_XID_PARTITION_CFG_TX_CK = 0x1
FUNC_QCFG_RESP_XID_PARTITION_CFG_RX_CK = 0x2
FUNC_QCFG_RESP_MAX_LINK_WIDTH_UNKNOWN = 0x0
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X1 = 0x1
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X2 = 0x2
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X4 = 0x4
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X8 = 0x8
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X16 = 0x10
FUNC_QCFG_RESP_MAX_LINK_WIDTH_LAST = FUNC_QCFG_RESP_MAX_LINK_WIDTH_X16
FUNC_QCFG_RESP_MAX_LINK_SPEED_UNKNOWN = 0x0
FUNC_QCFG_RESP_MAX_LINK_SPEED_G1 = 0x1
FUNC_QCFG_RESP_MAX_LINK_SPEED_G2 = 0x2
FUNC_QCFG_RESP_MAX_LINK_SPEED_G3 = 0x3
FUNC_QCFG_RESP_MAX_LINK_SPEED_G4 = 0x4
FUNC_QCFG_RESP_MAX_LINK_SPEED_G5 = 0x5
FUNC_QCFG_RESP_MAX_LINK_SPEED_LAST = FUNC_QCFG_RESP_MAX_LINK_SPEED_G5
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_UNKNOWN = 0x0
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X1 = 0x1
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X2 = 0x2
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X4 = 0x4
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X8 = 0x8
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X16 = 0x10
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_LAST = FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X16
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_UNKNOWN = 0x0
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G1 = 0x1
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G2 = 0x2
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G3 = 0x3
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G4 = 0x4
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G5 = 0x5
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_LAST = FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G5
FUNC_DRV_RGTR_REQ_FLAGS_FWD_ALL_MODE = 0x1
FUNC_DRV_RGTR_REQ_FLAGS_FWD_NONE_MODE = 0x2
FUNC_DRV_RGTR_REQ_FLAGS_16BIT_VER_MODE = 0x4
FUNC_DRV_RGTR_REQ_FLAGS_FLOW_HANDLE_64BIT_MODE = 0x8
FUNC_DRV_RGTR_REQ_FLAGS_HOT_RESET_SUPPORT = 0x10
FUNC_DRV_RGTR_REQ_FLAGS_ERROR_RECOVERY_SUPPORT = 0x20
FUNC_DRV_RGTR_REQ_FLAGS_MASTER_SUPPORT = 0x40
FUNC_DRV_RGTR_REQ_FLAGS_FAST_RESET_SUPPORT = 0x80
FUNC_DRV_RGTR_REQ_FLAGS_RSS_STRICT_HASH_TYPE_SUPPORT = 0x100
FUNC_DRV_RGTR_REQ_FLAGS_NPAR_1_2_SUPPORT = 0x200
FUNC_DRV_RGTR_REQ_FLAGS_ASYM_QUEUE_CFG_SUPPORT = 0x400
FUNC_DRV_RGTR_REQ_FLAGS_TF_INGRESS_NIC_FLOW_MODE = 0x800
FUNC_DRV_RGTR_REQ_FLAGS_TF_EGRESS_NIC_FLOW_MODE = 0x1000
FUNC_DRV_RGTR_REQ_ENABLES_OS_TYPE = 0x1
FUNC_DRV_RGTR_REQ_ENABLES_VER = 0x2
FUNC_DRV_RGTR_REQ_ENABLES_TIMESTAMP = 0x4
FUNC_DRV_RGTR_REQ_ENABLES_VF_REQ_FWD = 0x8
FUNC_DRV_RGTR_REQ_ENABLES_ASYNC_EVENT_FWD = 0x10
FUNC_DRV_RGTR_REQ_OS_TYPE_UNKNOWN = 0x0
FUNC_DRV_RGTR_REQ_OS_TYPE_OTHER = 0x1
FUNC_DRV_RGTR_REQ_OS_TYPE_MSDOS = 0xe
FUNC_DRV_RGTR_REQ_OS_TYPE_WINDOWS = 0x12
FUNC_DRV_RGTR_REQ_OS_TYPE_SOLARIS = 0x1d
FUNC_DRV_RGTR_REQ_OS_TYPE_LINUX = 0x24
FUNC_DRV_RGTR_REQ_OS_TYPE_FREEBSD = 0x2a
FUNC_DRV_RGTR_REQ_OS_TYPE_ESXI = 0x68
FUNC_DRV_RGTR_REQ_OS_TYPE_WIN864 = 0x73
FUNC_DRV_RGTR_REQ_OS_TYPE_WIN2012R2 = 0x74
FUNC_DRV_RGTR_REQ_OS_TYPE_UEFI = 0x8000
FUNC_DRV_RGTR_REQ_OS_TYPE_LAST = FUNC_DRV_RGTR_REQ_OS_TYPE_UEFI
FUNC_DRV_RGTR_RESP_FLAGS_IF_CHANGE_SUPPORTED = 0x1
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_QP = 0x1
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_SRQ = 0x2
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_CQ = 0x4
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_VNIC = 0x8
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_STAT = 0x10
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_MRAV = 0x20
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_TKC = 0x40
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_RKC = 0x80
FUNC_BACKING_STORE_CFG_REQ_FLAGS_PREBOOT_MODE = 0x1
FUNC_BACKING_STORE_CFG_REQ_FLAGS_MRAV_RESERVATION_SPLIT = 0x2
FUNC_BACKING_STORE_CFG_REQ_ENABLES_QP = 0x1
FUNC_BACKING_STORE_CFG_REQ_ENABLES_SRQ = 0x2
FUNC_BACKING_STORE_CFG_REQ_ENABLES_CQ = 0x4
FUNC_BACKING_STORE_CFG_REQ_ENABLES_VNIC = 0x8
FUNC_BACKING_STORE_CFG_REQ_ENABLES_STAT = 0x10
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_SP = 0x20
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING0 = 0x40
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING1 = 0x80
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING2 = 0x100
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING3 = 0x200
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING4 = 0x400
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING5 = 0x800
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING6 = 0x1000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING7 = 0x2000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_MRAV = 0x4000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TIM = 0x8000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING8 = 0x10000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING9 = 0x20000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING10 = 0x40000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TKC = 0x80000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_RKC = 0x100000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_QP_FAST_QPMD = 0x200000
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_QP = 0x0
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CQ = 0x2
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_STAT = 0x4
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_TIM = 0xf
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SQ_DB_SHADOW = 0x16
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_RQ_DB_SHADOW = 0x17
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SRQ_DB_SHADOW = 0x18
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CQ_DB_SHADOW = 0x19
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_XID_PARTITION = 0x1d
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_AFM_KONG_HWRM_TRACE = 0x2a
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_ERR_QPC_TRACE = 0x2b
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_LAST = FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_INVALID
FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_PREBOOT_MODE = 0x1
FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_BS_CFG_ALL_DONE = 0x2
FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_BS_EXTEND = 0x4
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_MASK = 0xf
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_SFT = 0
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LAST = FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LVL_2
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_LAST = FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_V2_REQ_ENABLES_NEXT_BS_OFFSET = 0x1
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_QP = 0x0
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CQ = 0x2
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_STAT = 0x4
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_TIM = 0xf
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SQ_DB_SHADOW = 0x16
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_RQ_DB_SHADOW = 0x17
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SRQ_DB_SHADOW = 0x18
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CQ_DB_SHADOW = 0x19
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_XID_PARTITION_TABLE = 0x1d
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_AFM_KONG_HWRM_TRACE = 0x2a
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_ERR_QPC_TRACE = 0x2b
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_LAST = FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_INVALID
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_QP = 0x0
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CQ = 0x2
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_STAT = 0x4
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_TIM = 0xf
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_XID_PARTITION = 0x1d
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_ERR_QPC_TRACE = 0x2a
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_LAST = FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_INVALID
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_MASK = 0xf
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_SFT = 0
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LVL_0 = 0x0
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LVL_1 = 0x1
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LVL_2 = 0x2
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LAST = FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LVL_2
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_SFT = 4
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_LAST = FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_1G
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_QP = 0x0
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CQ = 0x2
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_STAT = 0x4
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_TIM = 0xf
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SQ_DB_SHADOW = 0x16
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_RQ_DB_SHADOW = 0x17
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SRQ_DB_SHADOW = 0x18
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CQ_DB_SHADOW = 0x19
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_XID_PARTITION = 0x1d
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_AFM_KONG_HWRM_TRACE = 0x2a
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_ERR_QPC_TRACE = 0x2b
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_LAST = FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_INVALID
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_QP = 0x0
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CQ = 0x2
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_STAT = 0x4
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_TIM = 0xf
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SQ_DB_SHADOW = 0x16
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_RQ_DB_SHADOW = 0x17
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SRQ_DB_SHADOW = 0x18
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CQ_DB_SHADOW = 0x19
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_XID_PARTITION = 0x1d
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_AFM_KONG_HWRM_TRACE = 0x2a
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_ERR_QPC_TRACE = 0x2b
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_LAST = FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_INVALID
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_ENABLE_CTX_KIND_INIT = 0x1
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_TYPE_VALID = 0x2
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_DRIVER_MANAGED_MEMORY = 0x4
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_ROCE_QP_PSEUDO_STATIC_ALLOC = 0x8
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_FW_DBG_TRACE = 0x10
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_FW_BIN_DBG_TRACE = 0x20
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_NEXT_BS_OFFSET = 0x40
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_SPLIT_ENTRY_0_EXACT = 0x1
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_SPLIT_ENTRY_1_EXACT = 0x2
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_SPLIT_ENTRY_2_EXACT = 0x4
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_SPLIT_ENTRY_3_EXACT = 0x8
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_UNUSED_MASK = 0xf0
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_UNUSED_SFT = 4
PORT_PHY_CFG_REQ_FLAGS_RESET_PHY = 0x1
PORT_PHY_CFG_REQ_FLAGS_DEPRECATED = 0x2
PORT_PHY_CFG_REQ_FLAGS_FORCE = 0x4
PORT_PHY_CFG_REQ_FLAGS_RESTART_AUTONEG = 0x8
PORT_PHY_CFG_REQ_FLAGS_EEE_ENABLE = 0x10
PORT_PHY_CFG_REQ_FLAGS_EEE_DISABLE = 0x20
PORT_PHY_CFG_REQ_FLAGS_EEE_TX_LPI_ENABLE = 0x40
PORT_PHY_CFG_REQ_FLAGS_EEE_TX_LPI_DISABLE = 0x80
PORT_PHY_CFG_REQ_FLAGS_FEC_AUTONEG_ENABLE = 0x100
PORT_PHY_CFG_REQ_FLAGS_FEC_AUTONEG_DISABLE = 0x200
PORT_PHY_CFG_REQ_FLAGS_FEC_CLAUSE74_ENABLE = 0x400
PORT_PHY_CFG_REQ_FLAGS_FEC_CLAUSE74_DISABLE = 0x800
PORT_PHY_CFG_REQ_FLAGS_FEC_CLAUSE91_ENABLE = 0x1000
PORT_PHY_CFG_REQ_FLAGS_FEC_CLAUSE91_DISABLE = 0x2000
PORT_PHY_CFG_REQ_FLAGS_FORCE_LINK_DWN = 0x4000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS544_1XN_ENABLE = 0x8000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS544_1XN_DISABLE = 0x10000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS544_IEEE_ENABLE = 0x20000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS544_IEEE_DISABLE = 0x40000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS272_1XN_ENABLE = 0x80000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS272_1XN_DISABLE = 0x100000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS272_IEEE_ENABLE = 0x200000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS272_IEEE_DISABLE = 0x400000
PORT_PHY_CFG_REQ_FLAGS_LINK_TRAINING_ENABLE = 0x800000
PORT_PHY_CFG_REQ_FLAGS_LINK_TRAINING_DISABLE = 0x1000000
PORT_PHY_CFG_REQ_FLAGS_PRECODING_ENABLE = 0x2000000
PORT_PHY_CFG_REQ_FLAGS_PRECODING_DISABLE = 0x4000000
PORT_PHY_CFG_REQ_ENABLES_AUTO_MODE = 0x1
PORT_PHY_CFG_REQ_ENABLES_AUTO_DUPLEX = 0x2
PORT_PHY_CFG_REQ_ENABLES_AUTO_PAUSE = 0x4
PORT_PHY_CFG_REQ_ENABLES_AUTO_LINK_SPEED = 0x8
PORT_PHY_CFG_REQ_ENABLES_AUTO_LINK_SPEED_MASK = 0x10
PORT_PHY_CFG_REQ_ENABLES_WIRESPEED = 0x20
PORT_PHY_CFG_REQ_ENABLES_LPBK = 0x40
PORT_PHY_CFG_REQ_ENABLES_PREEMPHASIS = 0x80
PORT_PHY_CFG_REQ_ENABLES_FORCE_PAUSE = 0x100
PORT_PHY_CFG_REQ_ENABLES_EEE_LINK_SPEED_MASK = 0x200
PORT_PHY_CFG_REQ_ENABLES_TX_LPI_TIMER = 0x400
PORT_PHY_CFG_REQ_ENABLES_FORCE_PAM4_LINK_SPEED = 0x800
PORT_PHY_CFG_REQ_ENABLES_AUTO_PAM4_LINK_SPEED_MASK = 0x1000
PORT_PHY_CFG_REQ_ENABLES_FORCE_LINK_SPEEDS2 = 0x2000
PORT_PHY_CFG_REQ_ENABLES_AUTO_LINK_SPEEDS2_MASK = 0x4000
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_100MB = 0x1
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_1GB = 0xa
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_2GB = 0x14
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_2_5GB = 0x19
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_10GB = 0x64
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_20GB = 0xc8
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_25GB = 0xfa
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_40GB = 0x190
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_50GB = 0x1f4
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_100GB = 0x3e8
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_10MB = 0xffff
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_LAST = PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_10MB
PORT_PHY_CFG_REQ_AUTO_MODE_NONE = 0x0
PORT_PHY_CFG_REQ_AUTO_MODE_ALL_SPEEDS = 0x1
PORT_PHY_CFG_REQ_AUTO_MODE_ONE_SPEED = 0x2
PORT_PHY_CFG_REQ_AUTO_MODE_ONE_OR_BELOW = 0x3
PORT_PHY_CFG_REQ_AUTO_MODE_SPEED_MASK = 0x4
PORT_PHY_CFG_REQ_AUTO_MODE_LAST = PORT_PHY_CFG_REQ_AUTO_MODE_SPEED_MASK
PORT_PHY_CFG_REQ_AUTO_DUPLEX_HALF = 0x0
PORT_PHY_CFG_REQ_AUTO_DUPLEX_FULL = 0x1
PORT_PHY_CFG_REQ_AUTO_DUPLEX_BOTH = 0x2
PORT_PHY_CFG_REQ_AUTO_DUPLEX_LAST = PORT_PHY_CFG_REQ_AUTO_DUPLEX_BOTH
PORT_PHY_CFG_REQ_AUTO_PAUSE_TX = 0x1
PORT_PHY_CFG_REQ_AUTO_PAUSE_RX = 0x2
PORT_PHY_CFG_REQ_AUTO_PAUSE_AUTONEG_PAUSE = 0x4
PORT_PHY_CFG_REQ_MGMT_FLAG_LINK_RELEASE = 0x1
PORT_PHY_CFG_REQ_MGMT_FLAG_MGMT_VALID = 0x80
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_100MB = 0x1
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_1GB = 0xa
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_2GB = 0x14
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_2_5GB = 0x19
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_10GB = 0x64
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_20GB = 0xc8
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_25GB = 0xfa
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_40GB = 0x190
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_50GB = 0x1f4
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_100GB = 0x3e8
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_10MB = 0xffff
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_LAST = PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_10MB
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_100MBHD = 0x1
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_100MB = 0x2
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_1GBHD = 0x4
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_1GB = 0x8
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_2GB = 0x10
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_2_5GB = 0x20
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_10GB = 0x40
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_20GB = 0x80
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_25GB = 0x100
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_40GB = 0x200
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_50GB = 0x400
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_100GB = 0x800
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_10MBHD = 0x1000
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_10MB = 0x2000
PORT_PHY_CFG_REQ_WIRESPEED_OFF = 0x0
PORT_PHY_CFG_REQ_WIRESPEED_ON = 0x1
PORT_PHY_CFG_REQ_WIRESPEED_LAST = PORT_PHY_CFG_REQ_WIRESPEED_ON
PORT_PHY_CFG_REQ_LPBK_NONE = 0x0
PORT_PHY_CFG_REQ_LPBK_LOCAL = 0x1
PORT_PHY_CFG_REQ_LPBK_REMOTE = 0x2
PORT_PHY_CFG_REQ_LPBK_EXTERNAL = 0x3
PORT_PHY_CFG_REQ_LPBK_LAST = PORT_PHY_CFG_REQ_LPBK_EXTERNAL
PORT_PHY_CFG_REQ_FORCE_PAUSE_TX = 0x1
PORT_PHY_CFG_REQ_FORCE_PAUSE_RX = 0x2
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_RSVD1 = 0x1
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_100MB = 0x2
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_RSVD2 = 0x4
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_1GB = 0x8
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_RSVD3 = 0x10
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_RSVD4 = 0x20
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_10GB = 0x40
PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_50GB = 0x1f4
PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_100GB = 0x3e8
PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_200GB = 0x7d0
PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_LAST = PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_200GB
PORT_PHY_CFG_REQ_TX_LPI_TIMER_MASK = 0xffffff
PORT_PHY_CFG_REQ_TX_LPI_TIMER_SFT = 0
PORT_PHY_CFG_REQ_AUTO_LINK_PAM4_SPEED_MASK_50G = 0x1
PORT_PHY_CFG_REQ_AUTO_LINK_PAM4_SPEED_MASK_100G = 0x2
PORT_PHY_CFG_REQ_AUTO_LINK_PAM4_SPEED_MASK_200G = 0x4
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_1GB = 0xa
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_10GB = 0x64
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_25GB = 0xfa
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_40GB = 0x190
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_50GB = 0x1f4
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_100GB = 0x3e8
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_50GB_PAM4_56 = 0x1f5
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_100GB_PAM4_56 = 0x3e9
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_200GB_PAM4_56 = 0x7d1
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_400GB_PAM4_56 = 0xfa1
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_100GB_PAM4_112 = 0x3ea
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_200GB_PAM4_112 = 0x7d2
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_400GB_PAM4_112 = 0xfa2
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_800GB_PAM4_112 = 0x1f42
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_LAST = PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_800GB_PAM4_112
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_1GB = 0x1
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_10GB = 0x2
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_25GB = 0x4
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_40GB = 0x8
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_50GB = 0x10
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_100GB = 0x20
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_50GB_PAM4_56 = 0x40
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_100GB_PAM4_56 = 0x80
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_200GB_PAM4_56 = 0x100
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_400GB_PAM4_56 = 0x200
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_100GB_PAM4_112 = 0x400
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_200GB_PAM4_112 = 0x800
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_400GB_PAM4_112 = 0x1000
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_800GB_PAM4_112 = 0x2000
PORT_PHY_CFG_CMD_ERR_CODE_UNKNOWN = 0x0
PORT_PHY_CFG_CMD_ERR_CODE_ILLEGAL_SPEED = 0x1
PORT_PHY_CFG_CMD_ERR_CODE_RETRY = 0x2
PORT_PHY_CFG_CMD_ERR_CODE_LAST = PORT_PHY_CFG_CMD_ERR_CODE_RETRY
VNIC_ALLOC_REQ_FLAGS_DEFAULT = 0x1
VNIC_ALLOC_REQ_FLAGS_VIRTIO_NET_FID_VALID = 0x2
VNIC_ALLOC_REQ_FLAGS_VNIC_ID_VALID = 0x4
VNIC_UPDATE_REQ_ENABLES_VNIC_STATE_VALID = 0x1
VNIC_UPDATE_REQ_ENABLES_MRU_VALID = 0x2
VNIC_UPDATE_REQ_ENABLES_METADATA_FORMAT_TYPE_VALID = 0x4
VNIC_UPDATE_REQ_VNIC_STATE_NORMAL = 0x0
VNIC_UPDATE_REQ_VNIC_STATE_DROP = 0x1
VNIC_UPDATE_REQ_VNIC_STATE_LAST = VNIC_UPDATE_REQ_VNIC_STATE_DROP
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_0 = 0x0
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_1 = 0x1
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_2 = 0x2
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_3 = 0x3
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_4 = 0x4
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_LAST = VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_4
VNIC_CFG_REQ_FLAGS_DEFAULT = 0x1
VNIC_CFG_REQ_FLAGS_VLAN_STRIP_MODE = 0x2
VNIC_CFG_REQ_FLAGS_BD_STALL_MODE = 0x4
VNIC_CFG_REQ_FLAGS_ROCE_DUAL_VNIC_MODE = 0x8
VNIC_CFG_REQ_FLAGS_ROCE_ONLY_VNIC_MODE = 0x10
VNIC_CFG_REQ_FLAGS_RSS_DFLT_CR_MODE = 0x20
VNIC_CFG_REQ_FLAGS_ROCE_MIRRORING_CAPABLE_VNIC_MODE = 0x40
VNIC_CFG_REQ_FLAGS_PORTCOS_MAPPING_MODE = 0x80
VNIC_CFG_REQ_ENABLES_DFLT_RING_GRP = 0x1
VNIC_CFG_REQ_ENABLES_RSS_RULE = 0x2
VNIC_CFG_REQ_ENABLES_COS_RULE = 0x4
VNIC_CFG_REQ_ENABLES_LB_RULE = 0x8
VNIC_CFG_REQ_ENABLES_MRU = 0x10
VNIC_CFG_REQ_ENABLES_DEFAULT_RX_RING_ID = 0x20
VNIC_CFG_REQ_ENABLES_DEFAULT_CMPL_RING_ID = 0x40
VNIC_CFG_REQ_ENABLES_QUEUE_ID = 0x80
VNIC_CFG_REQ_ENABLES_RX_CSUM_V2_MODE = 0x100
VNIC_CFG_REQ_ENABLES_L2_CQE_MODE = 0x200
VNIC_CFG_REQ_ENABLES_RAW_QP_ID = 0x400
VNIC_CFG_REQ_RX_CSUM_V2_MODE_DEFAULT = 0x0
VNIC_CFG_REQ_RX_CSUM_V2_MODE_ALL_OK = 0x1
VNIC_CFG_REQ_RX_CSUM_V2_MODE_MAX = 0x2
VNIC_CFG_REQ_RX_CSUM_V2_MODE_LAST = VNIC_CFG_REQ_RX_CSUM_V2_MODE_MAX
VNIC_CFG_REQ_L2_CQE_MODE_DEFAULT = 0x0
VNIC_CFG_REQ_L2_CQE_MODE_COMPRESSED = 0x1
VNIC_CFG_REQ_L2_CQE_MODE_MIXED = 0x2
VNIC_CFG_REQ_L2_CQE_MODE_LAST = VNIC_CFG_REQ_L2_CQE_MODE_MIXED
VNIC_QCAPS_RESP_FLAGS_UNUSED = 0x1
VNIC_QCAPS_RESP_FLAGS_VLAN_STRIP_CAP = 0x2
VNIC_QCAPS_RESP_FLAGS_BD_STALL_CAP = 0x4
VNIC_QCAPS_RESP_FLAGS_ROCE_DUAL_VNIC_CAP = 0x8
VNIC_QCAPS_RESP_FLAGS_ROCE_ONLY_VNIC_CAP = 0x10
VNIC_QCAPS_RESP_FLAGS_RSS_DFLT_CR_CAP = 0x20
VNIC_QCAPS_RESP_FLAGS_ROCE_MIRRORING_CAPABLE_VNIC_CAP = 0x40
VNIC_QCAPS_RESP_FLAGS_OUTERMOST_RSS_CAP = 0x80
VNIC_QCAPS_RESP_FLAGS_COS_ASSIGNMENT_CAP = 0x100
VNIC_QCAPS_RESP_FLAGS_RX_CMPL_V2_CAP = 0x200
VNIC_QCAPS_RESP_FLAGS_VNIC_STATE_CAP = 0x400
VNIC_QCAPS_RESP_FLAGS_VIRTIO_NET_VNIC_ALLOC_CAP = 0x800
VNIC_QCAPS_RESP_FLAGS_METADATA_FORMAT_CAP = 0x1000
VNIC_QCAPS_RESP_FLAGS_RSS_STRICT_HASH_TYPE_CAP = 0x2000
VNIC_QCAPS_RESP_FLAGS_RSS_HASH_TYPE_DELTA_CAP = 0x4000
VNIC_QCAPS_RESP_FLAGS_RING_SELECT_MODE_TOEPLITZ_CAP = 0x8000
VNIC_QCAPS_RESP_FLAGS_RING_SELECT_MODE_XOR_CAP = 0x10000
VNIC_QCAPS_RESP_FLAGS_RING_SELECT_MODE_TOEPLITZ_CHKSM_CAP = 0x20000
VNIC_QCAPS_RESP_FLAGS_RSS_IPV6_FLOW_LABEL_CAP = 0x40000
VNIC_QCAPS_RESP_FLAGS_RX_CMPL_V3_CAP = 0x80000
VNIC_QCAPS_RESP_FLAGS_L2_CQE_MODE_CAP = 0x100000
VNIC_QCAPS_RESP_FLAGS_RSS_IPSEC_AH_SPI_IPV4_CAP = 0x200000
VNIC_QCAPS_RESP_FLAGS_RSS_IPSEC_ESP_SPI_IPV4_CAP = 0x400000
VNIC_QCAPS_RESP_FLAGS_RSS_IPSEC_AH_SPI_IPV6_CAP = 0x800000
VNIC_QCAPS_RESP_FLAGS_RSS_IPSEC_ESP_SPI_IPV6_CAP = 0x1000000
VNIC_QCAPS_RESP_FLAGS_OUTERMOST_RSS_TRUSTED_VF_CAP = 0x2000000
VNIC_QCAPS_RESP_FLAGS_PORTCOS_MAPPING_MODE = 0x4000000
VNIC_QCAPS_RESP_FLAGS_RSS_PROF_TCAM_MODE_ENABLED = 0x8000000
VNIC_QCAPS_RESP_FLAGS_VNIC_RSS_HASH_MODE_CAP = 0x10000000
VNIC_QCAPS_RESP_FLAGS_HW_TUNNEL_TPA_CAP = 0x20000000
VNIC_QCAPS_RESP_FLAGS_RE_FLUSH_CAP = 0x40000000
VNIC_TPA_CFG_REQ_FLAGS_TPA = 0x1
VNIC_TPA_CFG_REQ_FLAGS_ENCAP_TPA = 0x2
VNIC_TPA_CFG_REQ_FLAGS_RSC_WND_UPDATE = 0x4
VNIC_TPA_CFG_REQ_FLAGS_GRO = 0x8
VNIC_TPA_CFG_REQ_FLAGS_AGG_WITH_ECN = 0x10
VNIC_TPA_CFG_REQ_FLAGS_AGG_WITH_SAME_GRE_SEQ = 0x20
VNIC_TPA_CFG_REQ_FLAGS_GRO_IPID_CHECK = 0x40
VNIC_TPA_CFG_REQ_FLAGS_GRO_TTL_CHECK = 0x80
VNIC_TPA_CFG_REQ_FLAGS_AGG_PACK_AS_GRO = 0x100
VNIC_TPA_CFG_REQ_ENABLES_MAX_AGG_SEGS = 0x1
VNIC_TPA_CFG_REQ_ENABLES_MAX_AGGS = 0x2
VNIC_TPA_CFG_REQ_ENABLES_MAX_AGG_TIMER = 0x4
VNIC_TPA_CFG_REQ_ENABLES_MIN_AGG_LEN = 0x8
VNIC_TPA_CFG_REQ_ENABLES_TNL_TPA_EN = 0x10
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_1 = 0x0
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_2 = 0x1
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_4 = 0x2
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_8 = 0x3
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_MAX = 0x1f
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_LAST = VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_MAX
VNIC_TPA_CFG_REQ_MAX_AGGS_1 = 0x0
VNIC_TPA_CFG_REQ_MAX_AGGS_2 = 0x1
VNIC_TPA_CFG_REQ_MAX_AGGS_4 = 0x2
VNIC_TPA_CFG_REQ_MAX_AGGS_8 = 0x3
VNIC_TPA_CFG_REQ_MAX_AGGS_16 = 0x4
VNIC_TPA_CFG_REQ_MAX_AGGS_MAX = 0x7
VNIC_TPA_CFG_REQ_MAX_AGGS_LAST = VNIC_TPA_CFG_REQ_MAX_AGGS_MAX
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_VXLAN = 0x1
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_GENEVE = 0x2
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_NVGRE = 0x4
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_GRE = 0x8
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_IPV4 = 0x10
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_IPV6 = 0x20
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_VXLAN_GPE = 0x40
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_VXLAN_CUST1 = 0x80
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_GRE_CUST1 = 0x100
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR1 = 0x200
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR2 = 0x400
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR3 = 0x800
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR4 = 0x1000
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR5 = 0x2000
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR6 = 0x4000
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR7 = 0x8000
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR8 = 0x10000
VNIC_TPA_QCFG_RESP_FLAGS_TPA = 0x1
VNIC_TPA_QCFG_RESP_FLAGS_ENCAP_TPA = 0x2
VNIC_TPA_QCFG_RESP_FLAGS_RSC_WND_UPDATE = 0x4
VNIC_TPA_QCFG_RESP_FLAGS_GRO = 0x8
VNIC_TPA_QCFG_RESP_FLAGS_AGG_WITH_ECN = 0x10
VNIC_TPA_QCFG_RESP_FLAGS_AGG_WITH_SAME_GRE_SEQ = 0x20
VNIC_TPA_QCFG_RESP_FLAGS_GRO_IPID_CHECK = 0x40
VNIC_TPA_QCFG_RESP_FLAGS_GRO_TTL_CHECK = 0x80
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_1 = 0x0
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_2 = 0x1
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_4 = 0x2
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_8 = 0x3
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_MAX = 0x1f
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_LAST = VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_MAX
VNIC_TPA_QCFG_RESP_MAX_AGGS_1 = 0x0
VNIC_TPA_QCFG_RESP_MAX_AGGS_2 = 0x1
VNIC_TPA_QCFG_RESP_MAX_AGGS_4 = 0x2
VNIC_TPA_QCFG_RESP_MAX_AGGS_8 = 0x3
VNIC_TPA_QCFG_RESP_MAX_AGGS_16 = 0x4
VNIC_TPA_QCFG_RESP_MAX_AGGS_MAX = 0x7
VNIC_TPA_QCFG_RESP_MAX_AGGS_LAST = VNIC_TPA_QCFG_RESP_MAX_AGGS_MAX
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_VXLAN = 0x1
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_GENEVE = 0x2
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_NVGRE = 0x4
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_GRE = 0x8
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_IPV4 = 0x10
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_IPV6 = 0x20
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_VXLAN_GPE = 0x40
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_VXLAN_CUST1 = 0x80
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_GRE_CUST1 = 0x100
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR1 = 0x200
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR2 = 0x400
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR3 = 0x800
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR4 = 0x1000
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR5 = 0x2000
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR6 = 0x4000
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR7 = 0x8000
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR8 = 0x10000
VNIC_RSS_CFG_REQ_HASH_TYPE_IPV4 = 0x1
VNIC_RSS_CFG_REQ_HASH_TYPE_TCP_IPV4 = 0x2
VNIC_RSS_CFG_REQ_HASH_TYPE_UDP_IPV4 = 0x4
VNIC_RSS_CFG_REQ_HASH_TYPE_IPV6 = 0x8
VNIC_RSS_CFG_REQ_HASH_TYPE_TCP_IPV6 = 0x10
VNIC_RSS_CFG_REQ_HASH_TYPE_UDP_IPV6 = 0x20
VNIC_RSS_CFG_REQ_HASH_TYPE_IPV6_FLOW_LABEL = 0x40
VNIC_RSS_CFG_REQ_HASH_TYPE_AH_SPI_IPV4 = 0x80
VNIC_RSS_CFG_REQ_HASH_TYPE_ESP_SPI_IPV4 = 0x100
VNIC_RSS_CFG_REQ_HASH_TYPE_AH_SPI_IPV6 = 0x200
VNIC_RSS_CFG_REQ_HASH_TYPE_ESP_SPI_IPV6 = 0x400
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_DEFAULT = 0x1
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_INNERMOST_4 = 0x2
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_INNERMOST_2 = 0x4
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_OUTERMOST_4 = 0x8
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_OUTERMOST_2 = 0x10
VNIC_RSS_CFG_REQ_FLAGS_HASH_TYPE_INCLUDE = 0x1
VNIC_RSS_CFG_REQ_FLAGS_HASH_TYPE_EXCLUDE = 0x2
VNIC_RSS_CFG_REQ_FLAGS_IPSEC_HASH_TYPE_CFG_SUPPORT = 0x4
VNIC_RSS_CFG_REQ_RING_SELECT_MODE_TOEPLITZ = 0x0
VNIC_RSS_CFG_REQ_RING_SELECT_MODE_XOR = 0x1
VNIC_RSS_CFG_REQ_RING_SELECT_MODE_TOEPLITZ_CHECKSUM = 0x2
VNIC_RSS_CFG_REQ_RING_SELECT_MODE_LAST = VNIC_RSS_CFG_REQ_RING_SELECT_MODE_TOEPLITZ_CHECKSUM
VNIC_RSS_CFG_CMD_ERR_CODE_UNKNOWN = 0x0
VNIC_RSS_CFG_CMD_ERR_CODE_INTERFACE_NOT_READY = 0x1
VNIC_RSS_CFG_CMD_ERR_CODE_UNABLE_TO_GET_RSS_CFG = 0x2
VNIC_RSS_CFG_CMD_ERR_CODE_HASH_TYPE_UNSUPPORTED = 0x3
VNIC_RSS_CFG_CMD_ERR_CODE_HASH_TYPE_ERR = 0x4
VNIC_RSS_CFG_CMD_ERR_CODE_HASH_MODE_FAIL = 0x5
VNIC_RSS_CFG_CMD_ERR_CODE_RING_GRP_TABLE_ALLOC_ERR = 0x6
VNIC_RSS_CFG_CMD_ERR_CODE_HASH_KEY_ALLOC_ERR = 0x7
VNIC_RSS_CFG_CMD_ERR_CODE_DMA_FAILED = 0x8
VNIC_RSS_CFG_CMD_ERR_CODE_RX_RING_ALLOC_ERR = 0x9
VNIC_RSS_CFG_CMD_ERR_CODE_CMPL_RING_ALLOC_ERR = 0xa
VNIC_RSS_CFG_CMD_ERR_CODE_HW_SET_RSS_FAILED = 0xb
VNIC_RSS_CFG_CMD_ERR_CODE_CTX_INVALID = 0xc
VNIC_RSS_CFG_CMD_ERR_CODE_VNIC_INVALID = 0xd
VNIC_RSS_CFG_CMD_ERR_CODE_VNIC_RING_TABLE_PAIR_INVALID = 0xe
VNIC_RSS_CFG_CMD_ERR_CODE_LAST = VNIC_RSS_CFG_CMD_ERR_CODE_VNIC_RING_TABLE_PAIR_INVALID
VNIC_RSS_QCFG_RESP_HASH_TYPE_IPV4 = 0x1
VNIC_RSS_QCFG_RESP_HASH_TYPE_TCP_IPV4 = 0x2
VNIC_RSS_QCFG_RESP_HASH_TYPE_UDP_IPV4 = 0x4
VNIC_RSS_QCFG_RESP_HASH_TYPE_IPV6 = 0x8
VNIC_RSS_QCFG_RESP_HASH_TYPE_TCP_IPV6 = 0x10
VNIC_RSS_QCFG_RESP_HASH_TYPE_UDP_IPV6 = 0x20
VNIC_RSS_QCFG_RESP_HASH_TYPE_IPV6_FLOW_LABEL = 0x40
VNIC_RSS_QCFG_RESP_HASH_TYPE_AH_SPI_IPV4 = 0x80
VNIC_RSS_QCFG_RESP_HASH_TYPE_ESP_SPI_IPV4 = 0x100
VNIC_RSS_QCFG_RESP_HASH_TYPE_AH_SPI_IPV6 = 0x200
VNIC_RSS_QCFG_RESP_HASH_TYPE_ESP_SPI_IPV6 = 0x400
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_DEFAULT = 0x1
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_INNERMOST_4 = 0x2
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_INNERMOST_2 = 0x4
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_OUTERMOST_4 = 0x8
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_OUTERMOST_2 = 0x10
VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_TOEPLITZ = 0x0
VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_XOR = 0x1
VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_TOEPLITZ_CHECKSUM = 0x2
VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_LAST = VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_TOEPLITZ_CHECKSUM
VNIC_PLCMODES_CFG_REQ_FLAGS_REGULAR_PLACEMENT = 0x1
VNIC_PLCMODES_CFG_REQ_FLAGS_JUMBO_PLACEMENT = 0x2
VNIC_PLCMODES_CFG_REQ_FLAGS_HDS_IPV4 = 0x4
VNIC_PLCMODES_CFG_REQ_FLAGS_HDS_IPV6 = 0x8
VNIC_PLCMODES_CFG_REQ_FLAGS_HDS_FCOE = 0x10
VNIC_PLCMODES_CFG_REQ_FLAGS_HDS_ROCE = 0x20
VNIC_PLCMODES_CFG_REQ_FLAGS_VIRTIO_PLACEMENT = 0x40
VNIC_PLCMODES_CFG_REQ_ENABLES_JUMBO_THRESH_VALID = 0x1
VNIC_PLCMODES_CFG_REQ_ENABLES_HDS_OFFSET_VALID = 0x2
VNIC_PLCMODES_CFG_REQ_ENABLES_HDS_THRESHOLD_VALID = 0x4
VNIC_PLCMODES_CFG_REQ_ENABLES_MAX_BDS_VALID = 0x8
VNIC_PLCMODES_CFG_CMD_ERR_CODE_UNKNOWN = 0x0
VNIC_PLCMODES_CFG_CMD_ERR_CODE_INVALID_HDS_THRESHOLD = 0x1
VNIC_PLCMODES_CFG_CMD_ERR_CODE_LAST = VNIC_PLCMODES_CFG_CMD_ERR_CODE_INVALID_HDS_THRESHOLD
RING_ALLOC_REQ_ENABLES_RING_ARB_CFG = 0x2
RING_ALLOC_REQ_ENABLES_STAT_CTX_ID_VALID = 0x8
RING_ALLOC_REQ_ENABLES_MAX_BW_VALID = 0x20
RING_ALLOC_REQ_ENABLES_RX_RING_ID_VALID = 0x40
RING_ALLOC_REQ_ENABLES_NQ_RING_ID_VALID = 0x80
RING_ALLOC_REQ_ENABLES_RX_BUF_SIZE_VALID = 0x100
RING_ALLOC_REQ_ENABLES_SCHQ_ID = 0x200
RING_ALLOC_REQ_ENABLES_MPC_CHNLS_TYPE = 0x400
RING_ALLOC_REQ_ENABLES_STEERING_TAG_VALID = 0x800
RING_ALLOC_REQ_ENABLES_RX_RATE_PROFILE_VALID = 0x1000
RING_ALLOC_REQ_ENABLES_DPI_VALID = 0x2000
RING_ALLOC_REQ_RING_TYPE_L2_CMPL = 0x0
RING_ALLOC_REQ_RING_TYPE_TX = 0x1
RING_ALLOC_REQ_RING_TYPE_RX = 0x2
RING_ALLOC_REQ_RING_TYPE_ROCE_CMPL = 0x3
RING_ALLOC_REQ_RING_TYPE_RX_AGG = 0x4
RING_ALLOC_REQ_RING_TYPE_NQ = 0x5
RING_ALLOC_REQ_RING_TYPE_LAST = RING_ALLOC_REQ_RING_TYPE_NQ
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_OFF = 0x0
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_4 = 0x1
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_8 = 0x2
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_12 = 0x3
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_16 = 0x4
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_24 = 0x5
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_32 = 0x6
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_48 = 0x7
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_64 = 0x8
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_96 = 0x9
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_128 = 0xa
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_192 = 0xb
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_256 = 0xc
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_320 = 0xd
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_384 = 0xe
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_MAX = 0xf
RING_ALLOC_REQ_CMPL_COAL_CNT_LAST = RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_MAX
RING_ALLOC_REQ_FLAGS_RX_SOP_PAD = 0x1
RING_ALLOC_REQ_FLAGS_DISABLE_CQ_OVERFLOW_DETECTION = 0x2
RING_ALLOC_REQ_FLAGS_NQ_DBR_PACING = 0x4
RING_ALLOC_REQ_FLAGS_TX_PKT_TS_CMPL_ENABLE = 0x8
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_MASK = 0xf
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_SFT = 0
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_SP = 0x1
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_WFQ = 0x2
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_LAST = RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_WFQ
RING_ALLOC_REQ_RING_ARB_CFG_RSVD_MASK = 0xf0
RING_ALLOC_REQ_RING_ARB_CFG_RSVD_SFT = 4
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_PARAM_MASK = 0xff00
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_PARAM_SFT = 8
RING_ALLOC_REQ_MAX_BW_BW_VALUE_MASK = 0xfffffff
RING_ALLOC_REQ_MAX_BW_BW_VALUE_SFT = 0
RING_ALLOC_REQ_MAX_BW_SCALE = 0x10000000
RING_ALLOC_REQ_MAX_BW_SCALE_BITS = (0x0 << 28)
RING_ALLOC_REQ_MAX_BW_SCALE_BYTES = (0x1 << 28)
RING_ALLOC_REQ_MAX_BW_SCALE_LAST = RING_ALLOC_REQ_MAX_BW_SCALE_BYTES
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_SFT = 29
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_LAST = RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_INVALID
RING_ALLOC_REQ_INT_MODE_LEGACY = 0x0
RING_ALLOC_REQ_INT_MODE_RSVD = 0x1
RING_ALLOC_REQ_INT_MODE_MSIX = 0x2
RING_ALLOC_REQ_INT_MODE_POLL = 0x3
RING_ALLOC_REQ_INT_MODE_LAST = RING_ALLOC_REQ_INT_MODE_POLL
RING_ALLOC_REQ_MPC_CHNLS_TYPE_TCE = 0x0
RING_ALLOC_REQ_MPC_CHNLS_TYPE_RCE = 0x1
RING_ALLOC_REQ_MPC_CHNLS_TYPE_TE_CFA = 0x2
RING_ALLOC_REQ_MPC_CHNLS_TYPE_RE_CFA = 0x3
RING_ALLOC_REQ_MPC_CHNLS_TYPE_PRIMATE = 0x4
RING_ALLOC_REQ_MPC_CHNLS_TYPE_LAST = RING_ALLOC_REQ_MPC_CHNLS_TYPE_PRIMATE
RING_ALLOC_REQ_RX_RATE_PROFILE_SEL_DEFAULT = 0x0
RING_ALLOC_REQ_RX_RATE_PROFILE_SEL_POLL_MODE = 0x1
RING_ALLOC_REQ_RX_RATE_PROFILE_SEL_LAST = RING_ALLOC_REQ_RX_RATE_PROFILE_SEL_POLL_MODE
RING_ALLOC_RESP_PUSH_BUFFER_INDEX_PING_BUFFER = 0x0
RING_ALLOC_RESP_PUSH_BUFFER_INDEX_PONG_BUFFER = 0x1
RING_ALLOC_RESP_PUSH_BUFFER_INDEX_LAST = RING_ALLOC_RESP_PUSH_BUFFER_INDEX_PONG_BUFFER
CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH = 0x1
CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_TX = 0x0
CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_RX = 0x1
CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_LAST = CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_RX
CFA_L2_FILTER_ALLOC_REQ_FLAGS_LOOPBACK = 0x2
CFA_L2_FILTER_ALLOC_REQ_FLAGS_DROP = 0x4
CFA_L2_FILTER_ALLOC_REQ_FLAGS_OUTERMOST = 0x8
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_MASK = 0x30
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_SFT = 4
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_NO_ROCE_L2 = (0x0 << 4)
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_L2 = (0x1 << 4)
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_ROCE = (0x2 << 4)
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_LAST = CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_ROCE
CFA_L2_FILTER_ALLOC_REQ_FLAGS_XDP_DISABLE = 0x40
CFA_L2_FILTER_ALLOC_REQ_FLAGS_SOURCE_VALID = 0x80
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR = 0x1
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR_MASK = 0x2
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_OVLAN = 0x4
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_OVLAN_MASK = 0x8
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_IVLAN = 0x10
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_IVLAN_MASK = 0x20
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_ADDR = 0x40
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_ADDR_MASK = 0x80
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_OVLAN = 0x100
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_OVLAN_MASK = 0x200
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_IVLAN = 0x400
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_IVLAN_MASK = 0x800
CFA_L2_FILTER_ALLOC_REQ_ENABLES_SRC_TYPE = 0x1000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_SRC_ID = 0x2000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_TUNNEL_TYPE = 0x4000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_DST_ID = 0x8000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_MIRROR_VNIC_ID = 0x10000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_NUM_VLANS = 0x20000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_NUM_VLANS = 0x40000
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_NPORT = 0x0
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_PF = 0x1
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_VF = 0x2
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_VNIC = 0x3
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_KONG = 0x4
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_APE = 0x5
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_BONO = 0x6
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_TANG = 0x7
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_LAST = CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_TANG
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_NONTUNNEL = 0x0
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN = 0x1
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_NVGRE = 0x2
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2GRE = 0x3
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPIP = 0x4
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_GENEVE = 0x5
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_MPLS = 0x6
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_STT = 0x7
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE = 0x8
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_V4 = 0x9
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE_V1 = 0xa
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2_ETYPE = 0xb
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE_V6 = 0xc
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE = 0x10
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL = 0xff
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_LAST = CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_NO_PREFER = 0x0
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_ABOVE_FILTER = 0x1
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_BELOW_FILTER = 0x2
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_MAX = 0x3
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_MIN = 0x4
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_LAST = CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_MIN
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_VALUE_MASK = 0x3fffffff
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_VALUE_SFT = 0
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE = 0x40000000
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE_INT = (0x0 << 30)
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE_EXT = (0x1 << 30)
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE_LAST = CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE_EXT
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR = 0x80000000
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR_RX = (0x0 << 31)
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR_TX = (0x1 << 31)
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR_LAST = CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR_TX
CFA_L2_FILTER_CFG_REQ_FLAGS_PATH = 0x1
CFA_L2_FILTER_CFG_REQ_FLAGS_PATH_TX = 0x0
CFA_L2_FILTER_CFG_REQ_FLAGS_PATH_RX = 0x1
CFA_L2_FILTER_CFG_REQ_FLAGS_PATH_LAST = CFA_L2_FILTER_CFG_REQ_FLAGS_PATH_RX
CFA_L2_FILTER_CFG_REQ_FLAGS_DROP = 0x2
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_MASK = 0xc
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_SFT = 2
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_NO_ROCE_L2 = (0x0 << 2)
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_L2 = (0x1 << 2)
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_ROCE = (0x2 << 2)
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_LAST = CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_ROCE
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_MASK = 0x30
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_SFT = 4
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_NO_UPDATE = (0x0 << 4)
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_BYPASS_LKUP = (0x1 << 4)
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_ENABLE_LKUP = (0x2 << 4)
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_RESTORE_FW_OP = (0x3 << 4)
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_LAST = CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_RESTORE_FW_OP
CFA_L2_FILTER_CFG_REQ_ENABLES_DST_ID = 0x1
CFA_L2_FILTER_CFG_REQ_ENABLES_NEW_MIRROR_VNIC_ID = 0x2
CFA_L2_FILTER_CFG_REQ_ENABLES_PROF_FUNC = 0x4
CFA_L2_FILTER_CFG_REQ_ENABLES_L2_CONTEXT_ID = 0x8
STAT_CTX_ALLOC_REQ_STAT_CTX_FLAGS_ROCE = 0x1
STAT_CTX_ALLOC_REQ_STAT_CTX_FLAGS_DUP_HOST_BUF = 0x2
STAT_CTX_ALLOC_REQ_FLAGS_STEERING_TAG_VALID = 0x1
STAT_CTX_QUERY_REQ_FLAGS_COUNTER_MASK = 0x1
DBC_DBC_INDEX_MASK = 0xffffff
DBC_DBC_INDEX_SFT = 0
DBC_DBC_EPOCH = 0x1000000
DBC_DBC_TOGGLE_MASK = 0x6000000
DBC_DBC_TOGGLE_SFT = 25
DBC_DBC_XID_MASK = 0xfffff
DBC_DBC_XID_SFT = 0
DBC_DBC_PATH_MASK = 0x3000000
DBC_DBC_PATH_SFT = 24
DBC_DBC_PATH_ROCE = (0x0 << 24)
DBC_DBC_PATH_L2 = (0x1 << 24)
DBC_DBC_PATH_ENGINE = (0x2 << 24)
DBC_DBC_PATH_LAST = DBC_DBC_PATH_ENGINE
DBC_DBC_VALID = 0x4000000
DBC_DBC_DEBUG_TRACE = 0x8000000
DBC_DBC_TYPE_MASK = 0xf0000000
DBC_DBC_TYPE_SFT = 28
DBC_DBC_TYPE_SQ = (0x0 << 28)
DBC_DBC_TYPE_RQ = (0x1 << 28)
DBC_DBC_TYPE_SRQ = (0x2 << 28)
DBC_DBC_TYPE_SRQ_ARM = (0x3 << 28)
DBC_DBC_TYPE_CQ = (0x4 << 28)
DBC_DBC_TYPE_CQ_ARMSE = (0x5 << 28)
DBC_DBC_TYPE_CQ_ARMALL = (0x6 << 28)
DBC_DBC_TYPE_CQ_ARMENA = (0x7 << 28)
DBC_DBC_TYPE_SRQ_ARMENA = (0x8 << 28)
DBC_DBC_TYPE_CQ_CUTOFF_ACK = (0x9 << 28)
DBC_DBC_TYPE_NQ = (0xa << 28)
DBC_DBC_TYPE_NQ_ARM = (0xb << 28)
DBC_DBC_TYPE_NQ_MASK = (0xe << 28)
DBC_DBC_TYPE_NULL = (0xf << 28)
DBC_DBC_TYPE_LAST = DBC_DBC_TYPE_NULL
CMDQ_INIT_CMDQ_LVL_MASK = 0x3
CMDQ_INIT_CMDQ_LVL_SFT = 0
CMDQ_INIT_CMDQ_SIZE_MASK = 0xfffc
CMDQ_INIT_CMDQ_SIZE_SFT = 2
CMDQ_BASE_OPCODE_CREATE_QP = 0x1
CMDQ_BASE_OPCODE_DESTROY_QP = 0x2
CMDQ_BASE_OPCODE_MODIFY_QP = 0x3
CMDQ_BASE_OPCODE_QUERY_QP = 0x4
CMDQ_BASE_OPCODE_CREATE_SRQ = 0x5
CMDQ_BASE_OPCODE_DESTROY_SRQ = 0x6
CMDQ_BASE_OPCODE_QUERY_SRQ = 0x8
CMDQ_BASE_OPCODE_CREATE_CQ = 0x9
CMDQ_BASE_OPCODE_DESTROY_CQ = 0xa
CMDQ_BASE_OPCODE_RESIZE_CQ = 0xc
CMDQ_BASE_OPCODE_ALLOCATE_MRW = 0xd
CMDQ_BASE_OPCODE_DEALLOCATE_KEY = 0xe
CMDQ_BASE_OPCODE_REGISTER_MR = 0xf
CMDQ_BASE_OPCODE_DEREGISTER_MR = 0x10
CMDQ_BASE_OPCODE_ADD_GID = 0x11
CMDQ_BASE_OPCODE_DELETE_GID = 0x12
CMDQ_BASE_OPCODE_MODIFY_GID = 0x17
CMDQ_BASE_OPCODE_QUERY_GID = 0x18
CMDQ_BASE_OPCODE_CREATE_QP1 = 0x13
CMDQ_BASE_OPCODE_DESTROY_QP1 = 0x14
CMDQ_BASE_OPCODE_CREATE_AH = 0x15
CMDQ_BASE_OPCODE_DESTROY_AH = 0x16
CMDQ_BASE_OPCODE_INITIALIZE_FW = 0x80
CMDQ_BASE_OPCODE_DEINITIALIZE_FW = 0x81
CMDQ_BASE_OPCODE_STOP_FUNC = 0x82
CMDQ_BASE_OPCODE_QUERY_FUNC = 0x83
CMDQ_BASE_OPCODE_SET_FUNC_RESOURCES = 0x84
CMDQ_BASE_OPCODE_READ_CONTEXT = 0x85
CMDQ_BASE_OPCODE_VF_BACKCHANNEL_REQUEST = 0x86
CMDQ_BASE_OPCODE_READ_VF_MEMORY = 0x87
CMDQ_BASE_OPCODE_COMPLETE_VF_REQUEST = 0x88
CMDQ_BASE_OPCODE_EXTEND_CONTEXT_ARRRAY = 0x89
CMDQ_BASE_OPCODE_MAP_TC_TO_COS = 0x8a
CMDQ_BASE_OPCODE_QUERY_VERSION = 0x8b
CMDQ_BASE_OPCODE_MODIFY_ROCE_CC = 0x8c
CMDQ_BASE_OPCODE_QUERY_ROCE_CC = 0x8d
CMDQ_BASE_OPCODE_QUERY_ROCE_STATS = 0x8e
CMDQ_BASE_OPCODE_SET_LINK_AGGR_MODE = 0x8f
CMDQ_BASE_OPCODE_MODIFY_CQ = 0x90
CMDQ_BASE_OPCODE_QUERY_QP_EXTEND = 0x91
CMDQ_BASE_OPCODE_QUERY_ROCE_STATS_EXT = 0x92
CMDQ_BASE_OPCODE_ROCE_MIRROR_CFG = 0x99
CMDQ_BASE_OPCODE_LAST = CMDQ_BASE_OPCODE_ROCE_MIRROR_CFG
CREQ_BASE_TYPE_MASK = 0x3f
CREQ_BASE_TYPE_SFT = 0
CREQ_BASE_TYPE_QP_EVENT = 0x38
CREQ_BASE_TYPE_FUNC_EVENT = 0x3a
CREQ_BASE_TYPE_LAST = CREQ_BASE_TYPE_FUNC_EVENT
CREQ_BASE_V = 0x1
CMDQ_QUERY_VERSION_OPCODE_QUERY_VERSION = 0x8b
CMDQ_QUERY_VERSION_OPCODE_LAST = CMDQ_QUERY_VERSION_OPCODE_QUERY_VERSION
CREQ_QUERY_VERSION_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_VERSION_RESP_TYPE_SFT = 0
CREQ_QUERY_VERSION_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_VERSION_RESP_TYPE_LAST = CREQ_QUERY_VERSION_RESP_TYPE_QP_EVENT
CREQ_QUERY_VERSION_RESP_V = 0x1
CREQ_QUERY_VERSION_RESP_EVENT_QUERY_VERSION = 0x8b
CREQ_QUERY_VERSION_RESP_EVENT_LAST = CREQ_QUERY_VERSION_RESP_EVENT_QUERY_VERSION
CMDQ_INITIALIZE_FW_OPCODE_INITIALIZE_FW = 0x80
CMDQ_INITIALIZE_FW_OPCODE_LAST = CMDQ_INITIALIZE_FW_OPCODE_INITIALIZE_FW
CMDQ_INITIALIZE_FW_FLAGS_MRAV_RESERVATION_SPLIT = 0x1
CMDQ_INITIALIZE_FW_FLAGS_HW_REQUESTER_RETX_SUPPORTED = 0x2
CMDQ_INITIALIZE_FW_FLAGS_OPTIMIZE_MODIFY_QP_SUPPORTED = 0x8
CMDQ_INITIALIZE_FW_FLAGS_L2_VF_RESOURCE_MGMT = 0x10
CMDQ_INITIALIZE_FW_FLAGS_MIRROR_ON_ROCE_SUPPORTED = 0x80
CMDQ_INITIALIZE_FW_QPC_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_QPC_LVL_SFT = 0
CMDQ_INITIALIZE_FW_QPC_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_QPC_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_QPC_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_QPC_LVL_LAST = CMDQ_INITIALIZE_FW_QPC_LVL_LVL_2
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_MRW_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_MRW_LVL_SFT = 0
CMDQ_INITIALIZE_FW_MRW_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_MRW_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_MRW_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_MRW_LVL_LAST = CMDQ_INITIALIZE_FW_MRW_LVL_LVL_2
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_SRQ_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_SRQ_LVL_SFT = 0
CMDQ_INITIALIZE_FW_SRQ_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_SRQ_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_SRQ_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_SRQ_LVL_LAST = CMDQ_INITIALIZE_FW_SRQ_LVL_LVL_2
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_CQ_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_CQ_LVL_SFT = 0
CMDQ_INITIALIZE_FW_CQ_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_CQ_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_CQ_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_CQ_LVL_LAST = CMDQ_INITIALIZE_FW_CQ_LVL_LVL_2
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_TQM_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_TQM_LVL_SFT = 0
CMDQ_INITIALIZE_FW_TQM_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_TQM_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_TQM_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_TQM_LVL_LAST = CMDQ_INITIALIZE_FW_TQM_LVL_LVL_2
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_TIM_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_TIM_LVL_SFT = 0
CMDQ_INITIALIZE_FW_TIM_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_TIM_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_TIM_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_TIM_LVL_LAST = CMDQ_INITIALIZE_FW_TIM_LVL_LVL_2
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_MASK = 0xf
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_SFT = 0
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_4K = 0x0
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_8K = 0x1
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_16K = 0x2
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_32K = 0x3
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_64K = 0x4
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_128K = 0x5
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_256K = 0x6
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_512K = 0x7
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_1M = 0x8
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_2M = 0x9
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_4M = 0xa
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_8M = 0xb
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_16M = 0xc
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_32M = 0xd
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_64M = 0xe
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_128M = 0xf
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_128M
CMDQ_INITIALIZE_FW_RSVD_MASK = 0xfff0
CMDQ_INITIALIZE_FW_RSVD_SFT = 4
CREQ_INITIALIZE_FW_RESP_TYPE_MASK = 0x3f
CREQ_INITIALIZE_FW_RESP_TYPE_SFT = 0
CREQ_INITIALIZE_FW_RESP_TYPE_QP_EVENT = 0x38
CREQ_INITIALIZE_FW_RESP_TYPE_LAST = CREQ_INITIALIZE_FW_RESP_TYPE_QP_EVENT
CREQ_INITIALIZE_FW_RESP_V = 0x1
CREQ_INITIALIZE_FW_RESP_EVENT_INITIALIZE_FW = 0x80
CREQ_INITIALIZE_FW_RESP_EVENT_LAST = CREQ_INITIALIZE_FW_RESP_EVENT_INITIALIZE_FW
CMDQ_DEINITIALIZE_FW_OPCODE_DEINITIALIZE_FW = 0x81
CMDQ_DEINITIALIZE_FW_OPCODE_LAST = CMDQ_DEINITIALIZE_FW_OPCODE_DEINITIALIZE_FW
CREQ_DEINITIALIZE_FW_RESP_TYPE_MASK = 0x3f
CREQ_DEINITIALIZE_FW_RESP_TYPE_SFT = 0
CREQ_DEINITIALIZE_FW_RESP_TYPE_QP_EVENT = 0x38
CREQ_DEINITIALIZE_FW_RESP_TYPE_LAST = CREQ_DEINITIALIZE_FW_RESP_TYPE_QP_EVENT
CREQ_DEINITIALIZE_FW_RESP_V = 0x1
CREQ_DEINITIALIZE_FW_RESP_EVENT_DEINITIALIZE_FW = 0x81
CREQ_DEINITIALIZE_FW_RESP_EVENT_LAST = CREQ_DEINITIALIZE_FW_RESP_EVENT_DEINITIALIZE_FW
CMDQ_CREATE_QP_OPCODE_CREATE_QP = 0x1
CMDQ_CREATE_QP_OPCODE_LAST = CMDQ_CREATE_QP_OPCODE_CREATE_QP
CMDQ_CREATE_QP_QP_FLAGS_SRQ_USED = 0x1
CMDQ_CREATE_QP_QP_FLAGS_FORCE_COMPLETION = 0x2
CMDQ_CREATE_QP_QP_FLAGS_RESERVED_LKEY_ENABLE = 0x4
CMDQ_CREATE_QP_QP_FLAGS_FR_PMR_ENABLED = 0x8
CMDQ_CREATE_QP_QP_FLAGS_VARIABLE_SIZED_WQE_ENABLED = 0x10
CMDQ_CREATE_QP_QP_FLAGS_OPTIMIZED_TRANSMIT_ENABLED = 0x20
CMDQ_CREATE_QP_QP_FLAGS_RESPONDER_UD_CQE_WITH_CFA = 0x40
CMDQ_CREATE_QP_QP_FLAGS_EXT_STATS_ENABLED = 0x80
CMDQ_CREATE_QP_QP_FLAGS_EXPRESS_MODE_ENABLED = 0x100
CMDQ_CREATE_QP_QP_FLAGS_STEERING_TAG_VALID = 0x200
CMDQ_CREATE_QP_QP_FLAGS_RDMA_READ_OR_ATOMICS_USED = 0x400
CMDQ_CREATE_QP_QP_FLAGS_LAST = CMDQ_CREATE_QP_QP_FLAGS_RDMA_READ_OR_ATOMICS_USED
CMDQ_CREATE_QP_TYPE_RC = 0x2
CMDQ_CREATE_QP_TYPE_UD = 0x4
CMDQ_CREATE_QP_TYPE_RAW_ETHERTYPE = 0x6
CMDQ_CREATE_QP_TYPE_GSI = 0x7
CMDQ_CREATE_QP_TYPE_LAST = CMDQ_CREATE_QP_TYPE_GSI
CMDQ_CREATE_QP_SQ_LVL_MASK = 0xf
CMDQ_CREATE_QP_SQ_LVL_SFT = 0
CMDQ_CREATE_QP_SQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_QP_SQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_QP_SQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_QP_SQ_LVL_LAST = CMDQ_CREATE_QP_SQ_LVL_LVL_2
CMDQ_CREATE_QP_SQ_PG_SIZE_MASK = 0xf0
CMDQ_CREATE_QP_SQ_PG_SIZE_SFT = 4
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_LAST = CMDQ_CREATE_QP_SQ_PG_SIZE_PG_1G
CMDQ_CREATE_QP_RQ_LVL_MASK = 0xf
CMDQ_CREATE_QP_RQ_LVL_SFT = 0
CMDQ_CREATE_QP_RQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_QP_RQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_QP_RQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_QP_RQ_LVL_LAST = CMDQ_CREATE_QP_RQ_LVL_LVL_2
CMDQ_CREATE_QP_RQ_PG_SIZE_MASK = 0xf0
CMDQ_CREATE_QP_RQ_PG_SIZE_SFT = 4
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_LAST = CMDQ_CREATE_QP_RQ_PG_SIZE_PG_1G
CMDQ_CREATE_QP_SQ_SGE_MASK = 0xf
CMDQ_CREATE_QP_SQ_SGE_SFT = 0
CMDQ_CREATE_QP_SQ_FWO_MASK = 0xfff0
CMDQ_CREATE_QP_SQ_FWO_SFT = 4
CMDQ_CREATE_QP_RQ_SGE_MASK = 0xf
CMDQ_CREATE_QP_RQ_SGE_SFT = 0
CMDQ_CREATE_QP_RQ_FWO_MASK = 0xfff0
CMDQ_CREATE_QP_RQ_FWO_SFT = 4
CREQ_CREATE_QP_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_QP_RESP_TYPE_SFT = 0
CREQ_CREATE_QP_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_QP_RESP_TYPE_LAST = CREQ_CREATE_QP_RESP_TYPE_QP_EVENT
CREQ_CREATE_QP_RESP_V = 0x1
CREQ_CREATE_QP_RESP_EVENT_CREATE_QP = 0x1
CREQ_CREATE_QP_RESP_EVENT_LAST = CREQ_CREATE_QP_RESP_EVENT_CREATE_QP
CMDQ_DESTROY_QP_OPCODE_DESTROY_QP = 0x2
CMDQ_DESTROY_QP_OPCODE_LAST = CMDQ_DESTROY_QP_OPCODE_DESTROY_QP
CREQ_DESTROY_QP_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_QP_RESP_TYPE_SFT = 0
CREQ_DESTROY_QP_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_QP_RESP_TYPE_LAST = CREQ_DESTROY_QP_RESP_TYPE_QP_EVENT
CREQ_DESTROY_QP_RESP_V = 0x1
CREQ_DESTROY_QP_RESP_EVENT_DESTROY_QP = 0x2
CREQ_DESTROY_QP_RESP_EVENT_LAST = CREQ_DESTROY_QP_RESP_EVENT_DESTROY_QP
CMDQ_MODIFY_QP_OPCODE_MODIFY_QP = 0x3
CMDQ_MODIFY_QP_OPCODE_LAST = CMDQ_MODIFY_QP_OPCODE_MODIFY_QP
CMDQ_MODIFY_QP_FLAGS_SRQ_USED = 0x1
CMDQ_MODIFY_QP_QP_TYPE_RC = 0x2
CMDQ_MODIFY_QP_QP_TYPE_UD = 0x4
CMDQ_MODIFY_QP_QP_TYPE_RAW_ETHERTYPE = 0x6
CMDQ_MODIFY_QP_QP_TYPE_GSI = 0x7
CMDQ_MODIFY_QP_QP_TYPE_LAST = CMDQ_MODIFY_QP_QP_TYPE_GSI
CMDQ_MODIFY_QP_MODIFY_MASK_STATE = 0x1
CMDQ_MODIFY_QP_MODIFY_MASK_EN_SQD_ASYNC_NOTIFY = 0x2
CMDQ_MODIFY_QP_MODIFY_MASK_ACCESS = 0x4
CMDQ_MODIFY_QP_MODIFY_MASK_PKEY = 0x8
CMDQ_MODIFY_QP_MODIFY_MASK_QKEY = 0x10
CMDQ_MODIFY_QP_MODIFY_MASK_DGID = 0x20
CMDQ_MODIFY_QP_MODIFY_MASK_FLOW_LABEL = 0x40
CMDQ_MODIFY_QP_MODIFY_MASK_SGID_INDEX = 0x80
CMDQ_MODIFY_QP_MODIFY_MASK_HOP_LIMIT = 0x100
CMDQ_MODIFY_QP_MODIFY_MASK_TRAFFIC_CLASS = 0x200
CMDQ_MODIFY_QP_MODIFY_MASK_DEST_MAC = 0x400
CMDQ_MODIFY_QP_MODIFY_MASK_PINGPONG_PUSH_MODE = 0x800
CMDQ_MODIFY_QP_MODIFY_MASK_PATH_MTU = 0x1000
CMDQ_MODIFY_QP_MODIFY_MASK_TIMEOUT = 0x2000
CMDQ_MODIFY_QP_MODIFY_MASK_RETRY_CNT = 0x4000
CMDQ_MODIFY_QP_MODIFY_MASK_RNR_RETRY = 0x8000
CMDQ_MODIFY_QP_MODIFY_MASK_RQ_PSN = 0x10000
CMDQ_MODIFY_QP_MODIFY_MASK_MAX_RD_ATOMIC = 0x20000
CMDQ_MODIFY_QP_MODIFY_MASK_MIN_RNR_TIMER = 0x40000
CMDQ_MODIFY_QP_MODIFY_MASK_SQ_PSN = 0x80000
CMDQ_MODIFY_QP_MODIFY_MASK_MAX_DEST_RD_ATOMIC = 0x100000
CMDQ_MODIFY_QP_MODIFY_MASK_SQ_SIZE = 0x200000
CMDQ_MODIFY_QP_MODIFY_MASK_RQ_SIZE = 0x400000
CMDQ_MODIFY_QP_MODIFY_MASK_SQ_SGE = 0x800000
CMDQ_MODIFY_QP_MODIFY_MASK_RQ_SGE = 0x1000000
CMDQ_MODIFY_QP_MODIFY_MASK_MAX_INLINE_DATA = 0x2000000
CMDQ_MODIFY_QP_MODIFY_MASK_DEST_QP_ID = 0x4000000
CMDQ_MODIFY_QP_MODIFY_MASK_SRC_MAC = 0x8000000
CMDQ_MODIFY_QP_MODIFY_MASK_VLAN_ID = 0x10000000
CMDQ_MODIFY_QP_MODIFY_MASK_ENABLE_CC = 0x20000000
CMDQ_MODIFY_QP_MODIFY_MASK_TOS_ECN = 0x40000000
CMDQ_MODIFY_QP_MODIFY_MASK_TOS_DSCP = 0x80000000
CMDQ_MODIFY_QP_NEW_STATE_MASK = 0xf
CMDQ_MODIFY_QP_NEW_STATE_SFT = 0
CMDQ_MODIFY_QP_NEW_STATE_RESET = 0x0
CMDQ_MODIFY_QP_NEW_STATE_INIT = 0x1
CMDQ_MODIFY_QP_NEW_STATE_RTR = 0x2
CMDQ_MODIFY_QP_NEW_STATE_RTS = 0x3
CMDQ_MODIFY_QP_NEW_STATE_SQD = 0x4
CMDQ_MODIFY_QP_NEW_STATE_SQE = 0x5
CMDQ_MODIFY_QP_NEW_STATE_ERR = 0x6
CMDQ_MODIFY_QP_NEW_STATE_LAST = CMDQ_MODIFY_QP_NEW_STATE_ERR
CMDQ_MODIFY_QP_EN_SQD_ASYNC_NOTIFY = 0x10
CMDQ_MODIFY_QP_UNUSED1 = 0x20
CMDQ_MODIFY_QP_NETWORK_TYPE_MASK = 0xc0
CMDQ_MODIFY_QP_NETWORK_TYPE_SFT = 6
CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV1 = (0x0 << 6)
CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV4 = (0x2 << 6)
CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV6 = (0x3 << 6)
CMDQ_MODIFY_QP_NETWORK_TYPE_LAST = CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV6
CMDQ_MODIFY_QP_ACCESS_REMOTE_ATOMIC_REMOTE_READ_REMOTE_WRITE_LOCAL_WRITE_MASK = 0xff
CMDQ_MODIFY_QP_ACCESS_REMOTE_ATOMIC_REMOTE_READ_REMOTE_WRITE_LOCAL_WRITE_SFT = 0
CMDQ_MODIFY_QP_ACCESS_LOCAL_WRITE = 0x1
CMDQ_MODIFY_QP_ACCESS_REMOTE_WRITE = 0x2
CMDQ_MODIFY_QP_ACCESS_REMOTE_READ = 0x4
CMDQ_MODIFY_QP_ACCESS_REMOTE_ATOMIC = 0x8
CMDQ_MODIFY_QP_TOS_ECN_MASK = 0x3
CMDQ_MODIFY_QP_TOS_ECN_SFT = 0
CMDQ_MODIFY_QP_TOS_DSCP_MASK = 0xfc
CMDQ_MODIFY_QP_TOS_DSCP_SFT = 2
CMDQ_MODIFY_QP_PINGPONG_PUSH_ENABLE = 0x1
CMDQ_MODIFY_QP_UNUSED3_MASK = 0xe
CMDQ_MODIFY_QP_UNUSED3_SFT = 1
CMDQ_MODIFY_QP_PATH_MTU_MASK = 0xf0
CMDQ_MODIFY_QP_PATH_MTU_SFT = 4
CMDQ_MODIFY_QP_PATH_MTU_MTU_256 = (0x0 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_512 = (0x1 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_1024 = (0x2 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_2048 = (0x3 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_4096 = (0x4 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_8192 = (0x5 << 4)
CMDQ_MODIFY_QP_PATH_MTU_LAST = CMDQ_MODIFY_QP_PATH_MTU_MTU_8192
CMDQ_MODIFY_QP_ENABLE_CC = 0x1
CMDQ_MODIFY_QP_UNUSED15_MASK = 0xfffe
CMDQ_MODIFY_QP_UNUSED15_SFT = 1
CMDQ_MODIFY_QP_VLAN_ID_MASK = 0xfff
CMDQ_MODIFY_QP_VLAN_ID_SFT = 0
CMDQ_MODIFY_QP_VLAN_DEI = 0x1000
CMDQ_MODIFY_QP_VLAN_PCP_MASK = 0xe000
CMDQ_MODIFY_QP_VLAN_PCP_SFT = 13
CMDQ_MODIFY_QP_EXT_MODIFY_MASK_EXT_STATS_CTX = 0x1
CMDQ_MODIFY_QP_EXT_MODIFY_MASK_SCHQ_ID_VALID = 0x2
CREQ_MODIFY_QP_RESP_TYPE_MASK = 0x3f
CREQ_MODIFY_QP_RESP_TYPE_SFT = 0
CREQ_MODIFY_QP_RESP_TYPE_QP_EVENT = 0x38
CREQ_MODIFY_QP_RESP_TYPE_LAST = CREQ_MODIFY_QP_RESP_TYPE_QP_EVENT
CREQ_MODIFY_QP_RESP_V = 0x1
CREQ_MODIFY_QP_RESP_EVENT_MODIFY_QP = 0x3
CREQ_MODIFY_QP_RESP_EVENT_LAST = CREQ_MODIFY_QP_RESP_EVENT_MODIFY_QP
CREQ_MODIFY_QP_RESP_PINGPONG_PUSH_ENABLED = 0x1
CREQ_MODIFY_QP_RESP_PINGPONG_PUSH_INDEX_MASK = 0xe
CREQ_MODIFY_QP_RESP_PINGPONG_PUSH_INDEX_SFT = 1
CREQ_MODIFY_QP_RESP_PINGPONG_PUSH_STATE = 0x10
CMDQ_QUERY_QP_OPCODE_QUERY_QP = 0x4
CMDQ_QUERY_QP_OPCODE_LAST = CMDQ_QUERY_QP_OPCODE_QUERY_QP
CREQ_QUERY_QP_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_QP_RESP_TYPE_SFT = 0
CREQ_QUERY_QP_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_QP_RESP_TYPE_LAST = CREQ_QUERY_QP_RESP_TYPE_QP_EVENT
CREQ_QUERY_QP_RESP_V = 0x1
CREQ_QUERY_QP_RESP_EVENT_QUERY_QP = 0x4
CREQ_QUERY_QP_RESP_EVENT_LAST = CREQ_QUERY_QP_RESP_EVENT_QUERY_QP
CREQ_QUERY_QP_RESP_SB_OPCODE_QUERY_QP = 0x4
CREQ_QUERY_QP_RESP_SB_OPCODE_LAST = CREQ_QUERY_QP_RESP_SB_OPCODE_QUERY_QP
CREQ_QUERY_QP_RESP_SB_STATE_MASK = 0xf
CREQ_QUERY_QP_RESP_SB_STATE_SFT = 0
CREQ_QUERY_QP_RESP_SB_STATE_RESET = 0x0
CREQ_QUERY_QP_RESP_SB_STATE_INIT = 0x1
CREQ_QUERY_QP_RESP_SB_STATE_RTR = 0x2
CREQ_QUERY_QP_RESP_SB_STATE_RTS = 0x3
CREQ_QUERY_QP_RESP_SB_STATE_SQD = 0x4
CREQ_QUERY_QP_RESP_SB_STATE_SQE = 0x5
CREQ_QUERY_QP_RESP_SB_STATE_ERR = 0x6
CREQ_QUERY_QP_RESP_SB_STATE_LAST = CREQ_QUERY_QP_RESP_SB_STATE_ERR
CREQ_QUERY_QP_RESP_SB_EN_SQD_ASYNC_NOTIFY = 0x10
CREQ_QUERY_QP_RESP_SB_UNUSED3_MASK = 0xe0
CREQ_QUERY_QP_RESP_SB_UNUSED3_SFT = 5
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_ATOMIC_REMOTE_READ_REMOTE_WRITE_LOCAL_WRITE_MASK = 0xff
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_ATOMIC_REMOTE_READ_REMOTE_WRITE_LOCAL_WRITE_SFT = 0
CREQ_QUERY_QP_RESP_SB_ACCESS_LOCAL_WRITE = 0x1
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_WRITE = 0x2
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_READ = 0x4
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_ATOMIC = 0x8
CREQ_QUERY_QP_RESP_SB_DEST_VLAN_ID_MASK = 0xfff
CREQ_QUERY_QP_RESP_SB_DEST_VLAN_ID_SFT = 0
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MASK = 0xf000
CREQ_QUERY_QP_RESP_SB_PATH_MTU_SFT = 12
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_256 = (0x0 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_512 = (0x1 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_1024 = (0x2 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_2048 = (0x3 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_4096 = (0x4 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_8192 = (0x5 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_LAST = CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_8192
CREQ_QUERY_QP_RESP_SB_TOS_ECN_MASK = 0x3
CREQ_QUERY_QP_RESP_SB_TOS_ECN_SFT = 0
CREQ_QUERY_QP_RESP_SB_TOS_DSCP_MASK = 0xfc
CREQ_QUERY_QP_RESP_SB_TOS_DSCP_SFT = 2
CREQ_QUERY_QP_RESP_SB_ENABLE_CC = 0x1
CREQ_QUERY_QP_RESP_SB_VLAN_ID_MASK = 0xfff
CREQ_QUERY_QP_RESP_SB_VLAN_ID_SFT = 0
CREQ_QUERY_QP_RESP_SB_VLAN_DEI = 0x1000
CREQ_QUERY_QP_RESP_SB_VLAN_PCP_MASK = 0xe000
CREQ_QUERY_QP_RESP_SB_VLAN_PCP_SFT = 13
CMDQ_QUERY_QP_EXTEND_OPCODE_QUERY_QP_EXTEND = 0x91
CMDQ_QUERY_QP_EXTEND_OPCODE_LAST = CMDQ_QUERY_QP_EXTEND_OPCODE_QUERY_QP_EXTEND
CMDQ_QUERY_QP_EXTEND_PF_NUM_MASK = 0xff
CMDQ_QUERY_QP_EXTEND_PF_NUM_SFT = 0
CMDQ_QUERY_QP_EXTEND_VF_NUM_MASK = 0xffff00
CMDQ_QUERY_QP_EXTEND_VF_NUM_SFT = 8
CMDQ_QUERY_QP_EXTEND_VF_VALID = 0x1000000
CREQ_QUERY_QP_EXTEND_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_QP_EXTEND_RESP_TYPE_SFT = 0
CREQ_QUERY_QP_EXTEND_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_QP_EXTEND_RESP_TYPE_LAST = CREQ_QUERY_QP_EXTEND_RESP_TYPE_QP_EVENT
CREQ_QUERY_QP_EXTEND_RESP_V = 0x1
CREQ_QUERY_QP_EXTEND_RESP_EVENT_QUERY_QP_EXTEND = 0x91
CREQ_QUERY_QP_EXTEND_RESP_EVENT_LAST = CREQ_QUERY_QP_EXTEND_RESP_EVENT_QUERY_QP_EXTEND
CREQ_QUERY_QP_EXTEND_RESP_SB_OPCODE_QUERY_QP_EXTEND = 0x91
CREQ_QUERY_QP_EXTEND_RESP_SB_OPCODE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_OPCODE_QUERY_QP_EXTEND
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_MASK = 0xf
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_SFT = 0
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_RESET = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_INIT = 0x1
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_RTR = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_RTS = 0x3
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_SQD = 0x4
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_SQE = 0x5
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_ERR = 0x6
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_ERR
CREQ_QUERY_QP_EXTEND_RESP_SB_UNUSED4_MASK = 0xf0
CREQ_QUERY_QP_EXTEND_RESP_SB_UNUSED4_SFT = 4
CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_ROCEV1 = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_ROCEV2_IPV4 = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_ROCEV2_IPV6 = 0x3
CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_ROCEV2_IPV6
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_MORE = 0x1
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_MORE_LAST = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_OPCODE_QUERY_QP_EXTEND = 0x91
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_OPCODE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_OPCODE_QUERY_QP_EXTEND
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_MASK = 0xf
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_SFT = 0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_RESET = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_INIT = 0x1
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_RTR = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_RTS = 0x3
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_SQD = 0x4
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_SQE = 0x5
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_ERR = 0x6
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_ERR
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_UNUSED4_MASK = 0xf0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_UNUSED4_SFT = 4
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_ROCEV1 = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_ROCEV2_IPV4 = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_ROCEV2_IPV6 = 0x3
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_ROCEV2_IPV6
CMDQ_CREATE_SRQ_OPCODE_CREATE_SRQ = 0x5
CMDQ_CREATE_SRQ_OPCODE_LAST = CMDQ_CREATE_SRQ_OPCODE_CREATE_SRQ
CMDQ_CREATE_SRQ_FLAGS_STEERING_TAG_VALID = 0x1
CMDQ_CREATE_SRQ_LVL_MASK = 0x3
CMDQ_CREATE_SRQ_LVL_SFT = 0
CMDQ_CREATE_SRQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_SRQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_SRQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_SRQ_LVL_LAST = CMDQ_CREATE_SRQ_LVL_LVL_2
CMDQ_CREATE_SRQ_PG_SIZE_MASK = 0x1c
CMDQ_CREATE_SRQ_PG_SIZE_SFT = 2
CMDQ_CREATE_SRQ_PG_SIZE_PG_4K = (0x0 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_8K = (0x1 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_64K = (0x2 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_2M = (0x3 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_8M = (0x4 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_1G = (0x5 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_LAST = CMDQ_CREATE_SRQ_PG_SIZE_PG_1G
CMDQ_CREATE_SRQ_UNUSED11_MASK = 0xffe0
CMDQ_CREATE_SRQ_UNUSED11_SFT = 5
CMDQ_CREATE_SRQ_EVENTQ_ID_MASK = 0xfff
CMDQ_CREATE_SRQ_EVENTQ_ID_SFT = 0
CMDQ_CREATE_SRQ_UNUSED4_MASK = 0xf000
CMDQ_CREATE_SRQ_UNUSED4_SFT = 12
CREQ_CREATE_SRQ_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_SRQ_RESP_TYPE_SFT = 0
CREQ_CREATE_SRQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_SRQ_RESP_TYPE_LAST = CREQ_CREATE_SRQ_RESP_TYPE_QP_EVENT
CREQ_CREATE_SRQ_RESP_V = 0x1
CREQ_CREATE_SRQ_RESP_EVENT_CREATE_SRQ = 0x5
CREQ_CREATE_SRQ_RESP_EVENT_LAST = CREQ_CREATE_SRQ_RESP_EVENT_CREATE_SRQ
CMDQ_DESTROY_SRQ_OPCODE_DESTROY_SRQ = 0x6
CMDQ_DESTROY_SRQ_OPCODE_LAST = CMDQ_DESTROY_SRQ_OPCODE_DESTROY_SRQ
CREQ_DESTROY_SRQ_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_SRQ_RESP_TYPE_SFT = 0
CREQ_DESTROY_SRQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_SRQ_RESP_TYPE_LAST = CREQ_DESTROY_SRQ_RESP_TYPE_QP_EVENT
CREQ_DESTROY_SRQ_RESP_V = 0x1
CREQ_DESTROY_SRQ_RESP_EVENT_DESTROY_SRQ = 0x6
CREQ_DESTROY_SRQ_RESP_EVENT_LAST = CREQ_DESTROY_SRQ_RESP_EVENT_DESTROY_SRQ
CREQ_DESTROY_SRQ_RESP_UNUSED0_MASK = 0xffff
CREQ_DESTROY_SRQ_RESP_UNUSED0_SFT = 0
CREQ_DESTROY_SRQ_RESP_ENABLE_FOR_ARM_MASK = 0x30000
CREQ_DESTROY_SRQ_RESP_ENABLE_FOR_ARM_SFT = 16
CMDQ_QUERY_SRQ_OPCODE_QUERY_SRQ = 0x8
CMDQ_QUERY_SRQ_OPCODE_LAST = CMDQ_QUERY_SRQ_OPCODE_QUERY_SRQ
CREQ_QUERY_SRQ_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_SRQ_RESP_TYPE_SFT = 0
CREQ_QUERY_SRQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_SRQ_RESP_TYPE_LAST = CREQ_QUERY_SRQ_RESP_TYPE_QP_EVENT
CREQ_QUERY_SRQ_RESP_V = 0x1
CREQ_QUERY_SRQ_RESP_EVENT_QUERY_SRQ = 0x8
CREQ_QUERY_SRQ_RESP_EVENT_LAST = CREQ_QUERY_SRQ_RESP_EVENT_QUERY_SRQ
CREQ_QUERY_SRQ_RESP_SB_OPCODE_QUERY_SRQ = 0x8
CREQ_QUERY_SRQ_RESP_SB_OPCODE_LAST = CREQ_QUERY_SRQ_RESP_SB_OPCODE_QUERY_SRQ
CMDQ_CREATE_CQ_OPCODE_CREATE_CQ = 0x9
CMDQ_CREATE_CQ_OPCODE_LAST = CMDQ_CREATE_CQ_OPCODE_CREATE_CQ
CMDQ_CREATE_CQ_FLAGS_DISABLE_CQ_OVERFLOW_DETECTION = 0x1
CMDQ_CREATE_CQ_FLAGS_STEERING_TAG_VALID = 0x2
CMDQ_CREATE_CQ_FLAGS_INFINITE_CQ_MODE = 0x4
CMDQ_CREATE_CQ_FLAGS_COALESCING_VALID = 0x8
CMDQ_CREATE_CQ_LVL_MASK = 0x3
CMDQ_CREATE_CQ_LVL_SFT = 0
CMDQ_CREATE_CQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_CQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_CQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_CQ_LVL_LAST = CMDQ_CREATE_CQ_LVL_LVL_2
CMDQ_CREATE_CQ_PG_SIZE_MASK = 0x1c
CMDQ_CREATE_CQ_PG_SIZE_SFT = 2
CMDQ_CREATE_CQ_PG_SIZE_PG_4K = (0x0 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_8K = (0x1 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_64K = (0x2 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_2M = (0x3 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_8M = (0x4 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_1G = (0x5 << 2)
CMDQ_CREATE_CQ_PG_SIZE_LAST = CMDQ_CREATE_CQ_PG_SIZE_PG_1G
CMDQ_CREATE_CQ_UNUSED27_MASK = 0xffffffe0
CMDQ_CREATE_CQ_UNUSED27_SFT = 5
CMDQ_CREATE_CQ_CNQ_ID_MASK = 0xfff
CMDQ_CREATE_CQ_CNQ_ID_SFT = 0
CMDQ_CREATE_CQ_CQ_FCO_MASK = 0xfffff000
CMDQ_CREATE_CQ_CQ_FCO_SFT = 12
CMDQ_CREATE_CQ_BUF_MAXTIME_MASK = 0x1ff
CMDQ_CREATE_CQ_BUF_MAXTIME_SFT = 0
CMDQ_CREATE_CQ_NORMAL_MAXBUF_MASK = 0x3e00
CMDQ_CREATE_CQ_NORMAL_MAXBUF_SFT = 9
CMDQ_CREATE_CQ_DURING_MAXBUF_MASK = 0x7c000
CMDQ_CREATE_CQ_DURING_MAXBUF_SFT = 14
CMDQ_CREATE_CQ_ENABLE_RING_IDLE_MODE = 0x80000
CMDQ_CREATE_CQ_UNUSED12_MASK = 0xfff00000
CMDQ_CREATE_CQ_UNUSED12_SFT = 20
CREQ_CREATE_CQ_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_CQ_RESP_TYPE_SFT = 0
CREQ_CREATE_CQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_CQ_RESP_TYPE_LAST = CREQ_CREATE_CQ_RESP_TYPE_QP_EVENT
CREQ_CREATE_CQ_RESP_V = 0x1
CREQ_CREATE_CQ_RESP_EVENT_CREATE_CQ = 0x9
CREQ_CREATE_CQ_RESP_EVENT_LAST = CREQ_CREATE_CQ_RESP_EVENT_CREATE_CQ
CMDQ_DESTROY_CQ_OPCODE_DESTROY_CQ = 0xa
CMDQ_DESTROY_CQ_OPCODE_LAST = CMDQ_DESTROY_CQ_OPCODE_DESTROY_CQ
CREQ_DESTROY_CQ_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_CQ_RESP_TYPE_SFT = 0
CREQ_DESTROY_CQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_CQ_RESP_TYPE_LAST = CREQ_DESTROY_CQ_RESP_TYPE_QP_EVENT
CREQ_DESTROY_CQ_RESP_V = 0x1
CREQ_DESTROY_CQ_RESP_EVENT_DESTROY_CQ = 0xa
CREQ_DESTROY_CQ_RESP_EVENT_LAST = CREQ_DESTROY_CQ_RESP_EVENT_DESTROY_CQ
CREQ_DESTROY_CQ_RESP_CQ_ARM_LVL_MASK = 0x3
CREQ_DESTROY_CQ_RESP_CQ_ARM_LVL_SFT = 0
CMDQ_RESIZE_CQ_OPCODE_RESIZE_CQ = 0xc
CMDQ_RESIZE_CQ_OPCODE_LAST = CMDQ_RESIZE_CQ_OPCODE_RESIZE_CQ
CMDQ_RESIZE_CQ_LVL_MASK = 0x3
CMDQ_RESIZE_CQ_LVL_SFT = 0
CMDQ_RESIZE_CQ_LVL_LVL_0 = 0x0
CMDQ_RESIZE_CQ_LVL_LVL_1 = 0x1
CMDQ_RESIZE_CQ_LVL_LVL_2 = 0x2
CMDQ_RESIZE_CQ_LVL_LAST = CMDQ_RESIZE_CQ_LVL_LVL_2
CMDQ_RESIZE_CQ_PG_SIZE_MASK = 0x1c
CMDQ_RESIZE_CQ_PG_SIZE_SFT = 2
CMDQ_RESIZE_CQ_PG_SIZE_PG_4K = (0x0 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_8K = (0x1 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_64K = (0x2 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_2M = (0x3 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_8M = (0x4 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_1G = (0x5 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_LAST = CMDQ_RESIZE_CQ_PG_SIZE_PG_1G
CMDQ_RESIZE_CQ_NEW_CQ_SIZE_MASK = 0x1fffffe0
CMDQ_RESIZE_CQ_NEW_CQ_SIZE_SFT = 5
CREQ_RESIZE_CQ_RESP_TYPE_MASK = 0x3f
CREQ_RESIZE_CQ_RESP_TYPE_SFT = 0
CREQ_RESIZE_CQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_RESIZE_CQ_RESP_TYPE_LAST = CREQ_RESIZE_CQ_RESP_TYPE_QP_EVENT
CREQ_RESIZE_CQ_RESP_V = 0x1
CREQ_RESIZE_CQ_RESP_EVENT_RESIZE_CQ = 0xc
CREQ_RESIZE_CQ_RESP_EVENT_LAST = CREQ_RESIZE_CQ_RESP_EVENT_RESIZE_CQ
CMDQ_ALLOCATE_MRW_OPCODE_ALLOCATE_MRW = 0xd
CMDQ_ALLOCATE_MRW_OPCODE_LAST = CMDQ_ALLOCATE_MRW_OPCODE_ALLOCATE_MRW
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MASK = 0xf
CMDQ_ALLOCATE_MRW_MRW_FLAGS_SFT = 0
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MR = 0x0
CMDQ_ALLOCATE_MRW_MRW_FLAGS_PMR = 0x1
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MW_TYPE1 = 0x2
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MW_TYPE2A = 0x3
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MW_TYPE2B = 0x4
CMDQ_ALLOCATE_MRW_MRW_FLAGS_LAST = CMDQ_ALLOCATE_MRW_MRW_FLAGS_MW_TYPE2B
CMDQ_ALLOCATE_MRW_STEERING_TAG_VALID = 0x10
CMDQ_ALLOCATE_MRW_UNUSED4_MASK = 0xe0
CMDQ_ALLOCATE_MRW_UNUSED4_SFT = 5
CMDQ_ALLOCATE_MRW_ACCESS_CONSUMER_OWNED_KEY = 0x20
CREQ_ALLOCATE_MRW_RESP_TYPE_MASK = 0x3f
CREQ_ALLOCATE_MRW_RESP_TYPE_SFT = 0
CREQ_ALLOCATE_MRW_RESP_TYPE_QP_EVENT = 0x38
CREQ_ALLOCATE_MRW_RESP_TYPE_LAST = CREQ_ALLOCATE_MRW_RESP_TYPE_QP_EVENT
CREQ_ALLOCATE_MRW_RESP_V = 0x1
CREQ_ALLOCATE_MRW_RESP_EVENT_ALLOCATE_MRW = 0xd
CREQ_ALLOCATE_MRW_RESP_EVENT_LAST = CREQ_ALLOCATE_MRW_RESP_EVENT_ALLOCATE_MRW
CMDQ_DEALLOCATE_KEY_OPCODE_DEALLOCATE_KEY = 0xe
CMDQ_DEALLOCATE_KEY_OPCODE_LAST = CMDQ_DEALLOCATE_KEY_OPCODE_DEALLOCATE_KEY
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MASK = 0xf
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_SFT = 0
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MR = 0x0
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_PMR = 0x1
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MW_TYPE1 = 0x2
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MW_TYPE2A = 0x3
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MW_TYPE2B = 0x4
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_LAST = CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MW_TYPE2B
CMDQ_DEALLOCATE_KEY_UNUSED4_MASK = 0xf0
CMDQ_DEALLOCATE_KEY_UNUSED4_SFT = 4
CREQ_DEALLOCATE_KEY_RESP_TYPE_MASK = 0x3f
CREQ_DEALLOCATE_KEY_RESP_TYPE_SFT = 0
CREQ_DEALLOCATE_KEY_RESP_TYPE_QP_EVENT = 0x38
CREQ_DEALLOCATE_KEY_RESP_TYPE_LAST = CREQ_DEALLOCATE_KEY_RESP_TYPE_QP_EVENT
CREQ_DEALLOCATE_KEY_RESP_V = 0x1
CREQ_DEALLOCATE_KEY_RESP_EVENT_DEALLOCATE_KEY = 0xe
CREQ_DEALLOCATE_KEY_RESP_EVENT_LAST = CREQ_DEALLOCATE_KEY_RESP_EVENT_DEALLOCATE_KEY
CMDQ_REGISTER_MR_OPCODE_REGISTER_MR = 0xf
CMDQ_REGISTER_MR_OPCODE_LAST = CMDQ_REGISTER_MR_OPCODE_REGISTER_MR
CMDQ_REGISTER_MR_FLAGS_ALLOC_MR = 0x1
CMDQ_REGISTER_MR_FLAGS_STEERING_TAG_VALID = 0x2
CMDQ_REGISTER_MR_FLAGS_ENABLE_RO = 0x4
CMDQ_REGISTER_MR_LVL_MASK = 0x3
CMDQ_REGISTER_MR_LVL_SFT = 0
CMDQ_REGISTER_MR_LVL_LVL_0 = 0x0
CMDQ_REGISTER_MR_LVL_LVL_1 = 0x1
CMDQ_REGISTER_MR_LVL_LVL_2 = 0x2
CMDQ_REGISTER_MR_LVL_LAST = CMDQ_REGISTER_MR_LVL_LVL_2
CMDQ_REGISTER_MR_LOG2_PG_SIZE_MASK = 0x7c
CMDQ_REGISTER_MR_LOG2_PG_SIZE_SFT = 2
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_4K = (0xc << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_8K = (0xd << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_64K = (0x10 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_256K = (0x12 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_1M = (0x14 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_2M = (0x15 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_4M = (0x16 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_1G = (0x1e << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_LAST = CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_1G
CMDQ_REGISTER_MR_UNUSED1 = 0x80
CMDQ_REGISTER_MR_ACCESS_LOCAL_WRITE = 0x1
CMDQ_REGISTER_MR_ACCESS_REMOTE_READ = 0x2
CMDQ_REGISTER_MR_ACCESS_REMOTE_WRITE = 0x4
CMDQ_REGISTER_MR_ACCESS_REMOTE_ATOMIC = 0x8
CMDQ_REGISTER_MR_ACCESS_MW_BIND = 0x10
CMDQ_REGISTER_MR_ACCESS_ZERO_BASED = 0x20
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_MASK = 0x1f
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_SFT = 0
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_4K = 0xc
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_8K = 0xd
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_64K = 0x10
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_256K = 0x12
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_1M = 0x14
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_2M = 0x15
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_4M = 0x16
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_1G = 0x1e
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_LAST = CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_1G
CMDQ_REGISTER_MR_UNUSED11_MASK = 0xffe0
CMDQ_REGISTER_MR_UNUSED11_SFT = 5
CREQ_REGISTER_MR_RESP_TYPE_MASK = 0x3f
CREQ_REGISTER_MR_RESP_TYPE_SFT = 0
CREQ_REGISTER_MR_RESP_TYPE_QP_EVENT = 0x38
CREQ_REGISTER_MR_RESP_TYPE_LAST = CREQ_REGISTER_MR_RESP_TYPE_QP_EVENT
CREQ_REGISTER_MR_RESP_V = 0x1
CREQ_REGISTER_MR_RESP_EVENT_REGISTER_MR = 0xf
CREQ_REGISTER_MR_RESP_EVENT_LAST = CREQ_REGISTER_MR_RESP_EVENT_REGISTER_MR
CMDQ_DEREGISTER_MR_OPCODE_DEREGISTER_MR = 0x10
CMDQ_DEREGISTER_MR_OPCODE_LAST = CMDQ_DEREGISTER_MR_OPCODE_DEREGISTER_MR
CREQ_DEREGISTER_MR_RESP_TYPE_MASK = 0x3f
CREQ_DEREGISTER_MR_RESP_TYPE_SFT = 0
CREQ_DEREGISTER_MR_RESP_TYPE_QP_EVENT = 0x38
CREQ_DEREGISTER_MR_RESP_TYPE_LAST = CREQ_DEREGISTER_MR_RESP_TYPE_QP_EVENT
CREQ_DEREGISTER_MR_RESP_V = 0x1
CREQ_DEREGISTER_MR_RESP_EVENT_DEREGISTER_MR = 0x10
CREQ_DEREGISTER_MR_RESP_EVENT_LAST = CREQ_DEREGISTER_MR_RESP_EVENT_DEREGISTER_MR
CMDQ_ADD_GID_OPCODE_ADD_GID = 0x11
CMDQ_ADD_GID_OPCODE_LAST = CMDQ_ADD_GID_OPCODE_ADD_GID
CMDQ_ADD_GID_VLAN_VLAN_EN_TPID_VLAN_ID_MASK = 0xffff
CMDQ_ADD_GID_VLAN_VLAN_EN_TPID_VLAN_ID_SFT = 0
CMDQ_ADD_GID_VLAN_VLAN_ID_MASK = 0xfff
CMDQ_ADD_GID_VLAN_VLAN_ID_SFT = 0
CMDQ_ADD_GID_VLAN_TPID_MASK = 0x7000
CMDQ_ADD_GID_VLAN_TPID_SFT = 12
CMDQ_ADD_GID_VLAN_TPID_TPID_88A8 = (0x0 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_8100 = (0x1 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_9100 = (0x2 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_9200 = (0x3 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_9300 = (0x4 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_CFG1 = (0x5 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_CFG2 = (0x6 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_CFG3 = (0x7 << 12)
CMDQ_ADD_GID_VLAN_TPID_LAST = CMDQ_ADD_GID_VLAN_TPID_TPID_CFG3
CMDQ_ADD_GID_VLAN_VLAN_EN = 0x8000
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_VALID_STATS_CTX_ID_MASK = 0xffff
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_VALID_STATS_CTX_ID_SFT = 0
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_ID_MASK = 0x7fff
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_ID_SFT = 0
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_VALID = 0x8000
CREQ_ADD_GID_RESP_TYPE_MASK = 0x3f
CREQ_ADD_GID_RESP_TYPE_SFT = 0
CREQ_ADD_GID_RESP_TYPE_QP_EVENT = 0x38
CREQ_ADD_GID_RESP_TYPE_LAST = CREQ_ADD_GID_RESP_TYPE_QP_EVENT
CREQ_ADD_GID_RESP_V = 0x1
CREQ_ADD_GID_RESP_EVENT_ADD_GID = 0x11
CREQ_ADD_GID_RESP_EVENT_LAST = CREQ_ADD_GID_RESP_EVENT_ADD_GID
CMDQ_DELETE_GID_OPCODE_DELETE_GID = 0x12
CMDQ_DELETE_GID_OPCODE_LAST = CMDQ_DELETE_GID_OPCODE_DELETE_GID
CREQ_DELETE_GID_RESP_TYPE_MASK = 0x3f
CREQ_DELETE_GID_RESP_TYPE_SFT = 0
CREQ_DELETE_GID_RESP_TYPE_QP_EVENT = 0x38
CREQ_DELETE_GID_RESP_TYPE_LAST = CREQ_DELETE_GID_RESP_TYPE_QP_EVENT
CREQ_DELETE_GID_RESP_V = 0x1
CREQ_DELETE_GID_RESP_EVENT_DELETE_GID = 0x12
CREQ_DELETE_GID_RESP_EVENT_LAST = CREQ_DELETE_GID_RESP_EVENT_DELETE_GID
CMDQ_MODIFY_GID_OPCODE_MODIFY_GID = 0x17
CMDQ_MODIFY_GID_OPCODE_LAST = CMDQ_MODIFY_GID_OPCODE_MODIFY_GID
CMDQ_MODIFY_GID_VLAN_VLAN_ID_MASK = 0xfff
CMDQ_MODIFY_GID_VLAN_VLAN_ID_SFT = 0
CMDQ_MODIFY_GID_VLAN_TPID_MASK = 0x7000
CMDQ_MODIFY_GID_VLAN_TPID_SFT = 12
CMDQ_MODIFY_GID_VLAN_TPID_TPID_88A8 = (0x0 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_8100 = (0x1 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_9100 = (0x2 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_9200 = (0x3 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_9300 = (0x4 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_CFG1 = (0x5 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_CFG2 = (0x6 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_CFG3 = (0x7 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_LAST = CMDQ_MODIFY_GID_VLAN_TPID_TPID_CFG3
CMDQ_MODIFY_GID_VLAN_VLAN_EN = 0x8000
CMDQ_MODIFY_GID_STATS_CTX_STATS_CTX_ID_MASK = 0x7fff
CMDQ_MODIFY_GID_STATS_CTX_STATS_CTX_ID_SFT = 0
CMDQ_MODIFY_GID_STATS_CTX_STATS_CTX_VALID = 0x8000
CREQ_MODIFY_GID_RESP_TYPE_MASK = 0x3f
CREQ_MODIFY_GID_RESP_TYPE_SFT = 0
CREQ_MODIFY_GID_RESP_TYPE_QP_EVENT = 0x38
CREQ_MODIFY_GID_RESP_TYPE_LAST = CREQ_MODIFY_GID_RESP_TYPE_QP_EVENT
CREQ_MODIFY_GID_RESP_V = 0x1
CREQ_MODIFY_GID_RESP_EVENT_ADD_GID = 0x11
CREQ_MODIFY_GID_RESP_EVENT_LAST = CREQ_MODIFY_GID_RESP_EVENT_ADD_GID
CMDQ_QUERY_GID_OPCODE_QUERY_GID = 0x18
CMDQ_QUERY_GID_OPCODE_LAST = CMDQ_QUERY_GID_OPCODE_QUERY_GID
CREQ_QUERY_GID_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_GID_RESP_TYPE_SFT = 0
CREQ_QUERY_GID_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_GID_RESP_TYPE_LAST = CREQ_QUERY_GID_RESP_TYPE_QP_EVENT
CREQ_QUERY_GID_RESP_V = 0x1
CREQ_QUERY_GID_RESP_EVENT_QUERY_GID = 0x18
CREQ_QUERY_GID_RESP_EVENT_LAST = CREQ_QUERY_GID_RESP_EVENT_QUERY_GID
CREQ_QUERY_GID_RESP_SB_OPCODE_QUERY_GID = 0x18
CREQ_QUERY_GID_RESP_SB_OPCODE_LAST = CREQ_QUERY_GID_RESP_SB_OPCODE_QUERY_GID
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_EN_TPID_VLAN_ID_MASK = 0xffff
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_EN_TPID_VLAN_ID_SFT = 0
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_ID_MASK = 0xfff
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_ID_SFT = 0
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_MASK = 0x7000
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_SFT = 12
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_88A8 = (0x0 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_8100 = (0x1 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_9100 = (0x2 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_9200 = (0x3 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_9300 = (0x4 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_CFG1 = (0x5 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_CFG2 = (0x6 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_CFG3 = (0x7 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_LAST = CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_CFG3
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_EN = 0x8000
CMDQ_CREATE_QP1_OPCODE_CREATE_QP1 = 0x13
CMDQ_CREATE_QP1_OPCODE_LAST = CMDQ_CREATE_QP1_OPCODE_CREATE_QP1
CMDQ_CREATE_QP1_QP_FLAGS_SRQ_USED = 0x1
CMDQ_CREATE_QP1_QP_FLAGS_FORCE_COMPLETION = 0x2
CMDQ_CREATE_QP1_QP_FLAGS_RESERVED_LKEY_ENABLE = 0x4
CMDQ_CREATE_QP1_QP_FLAGS_LAST = CMDQ_CREATE_QP1_QP_FLAGS_RESERVED_LKEY_ENABLE
CMDQ_CREATE_QP1_TYPE_GSI = 0x1
CMDQ_CREATE_QP1_TYPE_LAST = CMDQ_CREATE_QP1_TYPE_GSI
CMDQ_CREATE_QP1_SQ_LVL_MASK = 0xf
CMDQ_CREATE_QP1_SQ_LVL_SFT = 0
CMDQ_CREATE_QP1_SQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_QP1_SQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_QP1_SQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_QP1_SQ_LVL_LAST = CMDQ_CREATE_QP1_SQ_LVL_LVL_2
CMDQ_CREATE_QP1_SQ_PG_SIZE_MASK = 0xf0
CMDQ_CREATE_QP1_SQ_PG_SIZE_SFT = 4
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_LAST = CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_1G
CMDQ_CREATE_QP1_RQ_LVL_MASK = 0xf
CMDQ_CREATE_QP1_RQ_LVL_SFT = 0
CMDQ_CREATE_QP1_RQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_QP1_RQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_QP1_RQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_QP1_RQ_LVL_LAST = CMDQ_CREATE_QP1_RQ_LVL_LVL_2
CMDQ_CREATE_QP1_RQ_PG_SIZE_MASK = 0xf0
CMDQ_CREATE_QP1_RQ_PG_SIZE_SFT = 4
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_LAST = CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_1G
CMDQ_CREATE_QP1_SQ_SGE_MASK = 0xf
CMDQ_CREATE_QP1_SQ_SGE_SFT = 0
CMDQ_CREATE_QP1_SQ_FWO_MASK = 0xfff0
CMDQ_CREATE_QP1_SQ_FWO_SFT = 4
CMDQ_CREATE_QP1_RQ_SGE_MASK = 0xf
CMDQ_CREATE_QP1_RQ_SGE_SFT = 0
CMDQ_CREATE_QP1_RQ_FWO_MASK = 0xfff0
CMDQ_CREATE_QP1_RQ_FWO_SFT = 4
CREQ_CREATE_QP1_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_QP1_RESP_TYPE_SFT = 0
CREQ_CREATE_QP1_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_QP1_RESP_TYPE_LAST = CREQ_CREATE_QP1_RESP_TYPE_QP_EVENT
CREQ_CREATE_QP1_RESP_V = 0x1
CREQ_CREATE_QP1_RESP_EVENT_CREATE_QP1 = 0x13
CREQ_CREATE_QP1_RESP_EVENT_LAST = CREQ_CREATE_QP1_RESP_EVENT_CREATE_QP1
CMDQ_DESTROY_QP1_OPCODE_DESTROY_QP1 = 0x14
CMDQ_DESTROY_QP1_OPCODE_LAST = CMDQ_DESTROY_QP1_OPCODE_DESTROY_QP1
CREQ_DESTROY_QP1_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_QP1_RESP_TYPE_SFT = 0
CREQ_DESTROY_QP1_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_QP1_RESP_TYPE_LAST = CREQ_DESTROY_QP1_RESP_TYPE_QP_EVENT
CREQ_DESTROY_QP1_RESP_V = 0x1
CREQ_DESTROY_QP1_RESP_EVENT_DESTROY_QP1 = 0x14
CREQ_DESTROY_QP1_RESP_EVENT_LAST = CREQ_DESTROY_QP1_RESP_EVENT_DESTROY_QP1
CMDQ_CREATE_AH_OPCODE_CREATE_AH = 0x15
CMDQ_CREATE_AH_OPCODE_LAST = CMDQ_CREATE_AH_OPCODE_CREATE_AH
CMDQ_CREATE_AH_TYPE_V1 = 0x0
CMDQ_CREATE_AH_TYPE_V2IPV4 = 0x2
CMDQ_CREATE_AH_TYPE_V2IPV6 = 0x3
CMDQ_CREATE_AH_TYPE_LAST = CMDQ_CREATE_AH_TYPE_V2IPV6
CMDQ_CREATE_AH_FLOW_LABEL_MASK = 0xfffff
CMDQ_CREATE_AH_FLOW_LABEL_SFT = 0
CMDQ_CREATE_AH_DEST_VLAN_ID_MASK = 0xfff00000
CMDQ_CREATE_AH_DEST_VLAN_ID_SFT = 20
CMDQ_CREATE_AH_ENABLE_CC = 0x1
CREQ_CREATE_AH_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_AH_RESP_TYPE_SFT = 0
CREQ_CREATE_AH_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_AH_RESP_TYPE_LAST = CREQ_CREATE_AH_RESP_TYPE_QP_EVENT
CREQ_CREATE_AH_RESP_V = 0x1
CREQ_CREATE_AH_RESP_EVENT_CREATE_AH = 0x15
CREQ_CREATE_AH_RESP_EVENT_LAST = CREQ_CREATE_AH_RESP_EVENT_CREATE_AH
CMDQ_DESTROY_AH_OPCODE_DESTROY_AH = 0x16
CMDQ_DESTROY_AH_OPCODE_LAST = CMDQ_DESTROY_AH_OPCODE_DESTROY_AH
CREQ_DESTROY_AH_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_AH_RESP_TYPE_SFT = 0
CREQ_DESTROY_AH_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_AH_RESP_TYPE_LAST = CREQ_DESTROY_AH_RESP_TYPE_QP_EVENT
CREQ_DESTROY_AH_RESP_V = 0x1
CREQ_DESTROY_AH_RESP_EVENT_DESTROY_AH = 0x16
CREQ_DESTROY_AH_RESP_EVENT_LAST = CREQ_DESTROY_AH_RESP_EVENT_DESTROY_AH
CMDQ_QUERY_ROCE_STATS_OPCODE_QUERY_ROCE_STATS = 0x8e
CMDQ_QUERY_ROCE_STATS_OPCODE_LAST = CMDQ_QUERY_ROCE_STATS_OPCODE_QUERY_ROCE_STATS
CMDQ_QUERY_ROCE_STATS_FLAGS_COLLECTION_ID = 0x1
CMDQ_QUERY_ROCE_STATS_FLAGS_FUNCTION_ID = 0x2
CMDQ_QUERY_ROCE_STATS_PF_NUM_MASK = 0xff
CMDQ_QUERY_ROCE_STATS_PF_NUM_SFT = 0
CMDQ_QUERY_ROCE_STATS_VF_NUM_MASK = 0xffff00
CMDQ_QUERY_ROCE_STATS_VF_NUM_SFT = 8
CMDQ_QUERY_ROCE_STATS_VF_VALID = 0x1000000
CREQ_QUERY_ROCE_STATS_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_ROCE_STATS_RESP_TYPE_SFT = 0
CREQ_QUERY_ROCE_STATS_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_ROCE_STATS_RESP_TYPE_LAST = CREQ_QUERY_ROCE_STATS_RESP_TYPE_QP_EVENT
CREQ_QUERY_ROCE_STATS_RESP_V = 0x1
CREQ_QUERY_ROCE_STATS_RESP_EVENT_QUERY_ROCE_STATS = 0x8e
CREQ_QUERY_ROCE_STATS_RESP_EVENT_LAST = CREQ_QUERY_ROCE_STATS_RESP_EVENT_QUERY_ROCE_STATS
CREQ_QUERY_ROCE_STATS_RESP_SB_OPCODE_QUERY_ROCE_STATS = 0x8e
CREQ_QUERY_ROCE_STATS_RESP_SB_OPCODE_LAST = CREQ_QUERY_ROCE_STATS_RESP_SB_OPCODE_QUERY_ROCE_STATS
CMDQ_QUERY_ROCE_STATS_EXT_OPCODE_QUERY_ROCE_STATS = 0x92
CMDQ_QUERY_ROCE_STATS_EXT_OPCODE_LAST = CMDQ_QUERY_ROCE_STATS_EXT_OPCODE_QUERY_ROCE_STATS
CMDQ_QUERY_ROCE_STATS_EXT_FLAGS_COLLECTION_ID = 0x1
CMDQ_QUERY_ROCE_STATS_EXT_FLAGS_FUNCTION_ID = 0x2
CMDQ_QUERY_ROCE_STATS_EXT_PF_NUM_MASK = 0xff
CMDQ_QUERY_ROCE_STATS_EXT_PF_NUM_SFT = 0
CMDQ_QUERY_ROCE_STATS_EXT_VF_NUM_MASK = 0xffff00
CMDQ_QUERY_ROCE_STATS_EXT_VF_NUM_SFT = 8
CMDQ_QUERY_ROCE_STATS_EXT_VF_VALID = 0x1000000
CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_SFT = 0
CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_LAST = CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_QP_EVENT
CREQ_QUERY_ROCE_STATS_EXT_RESP_V = 0x1
CREQ_QUERY_ROCE_STATS_EXT_RESP_EVENT_QUERY_ROCE_STATS_EXT = 0x92
CREQ_QUERY_ROCE_STATS_EXT_RESP_EVENT_LAST = CREQ_QUERY_ROCE_STATS_EXT_RESP_EVENT_QUERY_ROCE_STATS_EXT
CREQ_QUERY_ROCE_STATS_EXT_RESP_SB_OPCODE_QUERY_ROCE_STATS_EXT = 0x92
CREQ_QUERY_ROCE_STATS_EXT_RESP_SB_OPCODE_LAST = CREQ_QUERY_ROCE_STATS_EXT_RESP_SB_OPCODE_QUERY_ROCE_STATS_EXT
CMDQ_ROCE_MIRROR_CFG_OPCODE_ROCE_MIRROR_CFG = 0x99
CMDQ_ROCE_MIRROR_CFG_OPCODE_LAST = CMDQ_ROCE_MIRROR_CFG_OPCODE_ROCE_MIRROR_CFG
CMDQ_ROCE_MIRROR_CFG_MIRROR_ENABLE = 0x1
CREQ_ROCE_MIRROR_CFG_RESP_TYPE_MASK = 0x3f
CREQ_ROCE_MIRROR_CFG_RESP_TYPE_SFT = 0
CREQ_ROCE_MIRROR_CFG_RESP_TYPE_QP_EVENT = 0x38
CREQ_ROCE_MIRROR_CFG_RESP_TYPE_LAST = CREQ_ROCE_MIRROR_CFG_RESP_TYPE_QP_EVENT
CREQ_ROCE_MIRROR_CFG_RESP_V = 0x1
CREQ_ROCE_MIRROR_CFG_RESP_EVENT_ROCE_MIRROR_CFG = 0x99
CREQ_ROCE_MIRROR_CFG_RESP_EVENT_LAST = CREQ_ROCE_MIRROR_CFG_RESP_EVENT_ROCE_MIRROR_CFG
CMDQ_QUERY_FUNC_OPCODE_QUERY_FUNC = 0x83
CMDQ_QUERY_FUNC_OPCODE_LAST = CMDQ_QUERY_FUNC_OPCODE_QUERY_FUNC
CREQ_QUERY_FUNC_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_FUNC_RESP_TYPE_SFT = 0
CREQ_QUERY_FUNC_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_FUNC_RESP_TYPE_LAST = CREQ_QUERY_FUNC_RESP_TYPE_QP_EVENT
CREQ_QUERY_FUNC_RESP_V = 0x1
CREQ_QUERY_FUNC_RESP_EVENT_QUERY_FUNC = 0x83
CREQ_QUERY_FUNC_RESP_EVENT_LAST = CREQ_QUERY_FUNC_RESP_EVENT_QUERY_FUNC
CREQ_QUERY_FUNC_RESP_SB_OPCODE_QUERY_FUNC = 0x83
CREQ_QUERY_FUNC_RESP_SB_OPCODE_LAST = CREQ_QUERY_FUNC_RESP_SB_OPCODE_QUERY_FUNC
CREQ_QUERY_FUNC_RESP_SB_RESIZE_QP = 0x1
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_MASK = 0xe
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_SFT = 1
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_CC_GEN0 = (0x0 << 1)
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_CC_GEN1 = (0x1 << 1)
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_CC_GEN1_EXT = (0x2 << 1)
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_LAST = CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_CC_GEN1_EXT
CREQ_QUERY_FUNC_RESP_SB_EXT_STATS = 0x10
CREQ_QUERY_FUNC_RESP_SB_MR_REGISTER_ALLOC = 0x20
CREQ_QUERY_FUNC_RESP_SB_OPTIMIZED_TRANSMIT_ENABLED = 0x40
CREQ_QUERY_FUNC_RESP_SB_CQE_V2 = 0x80
CREQ_QUERY_FUNC_RESP_SB_PINGPONG_PUSH_MODE = 0x100
CREQ_QUERY_FUNC_RESP_SB_HW_REQUESTER_RETX_ENABLED = 0x200
CREQ_QUERY_FUNC_RESP_SB_HW_RESPONDER_RETX_ENABLED = 0x400
CREQ_QUERY_FUNC_RESP_SB_ATOMIC_OPS_NOT_SUPPORTED = 0x1
CREQ_QUERY_FUNC_RESP_SB_DRV_VERSION_RGTR_SUPPORTED = 0x2
CREQ_QUERY_FUNC_RESP_SB_CREATE_QP_BATCH_SUPPORTED = 0x4
CREQ_QUERY_FUNC_RESP_SB_DESTROY_QP_BATCH_SUPPORTED = 0x8
CREQ_QUERY_FUNC_RESP_SB_ROCE_STATS_EXT_CTX_SUPPORTED = 0x10
CREQ_QUERY_FUNC_RESP_SB_CREATE_SRQ_SGE_SUPPORTED = 0x20
CREQ_QUERY_FUNC_RESP_SB_FIXED_SIZE_WQE_DISABLED = 0x40
CREQ_QUERY_FUNC_RESP_SB_DCN_SUPPORTED = 0x80
CREQ_QUERY_FUNC_RESP_SB_OPTIMIZE_MODIFY_QP_SUPPORTED = 0x1
CREQ_QUERY_FUNC_RESP_SB_CHANGE_UDP_SRC_PORT_WQE_SUPPORTED = 0x2
CREQ_QUERY_FUNC_RESP_SB_CQ_COALESCING_SUPPORTED = 0x4
CREQ_QUERY_FUNC_RESP_SB_MEMORY_REGION_RO_SUPPORTED = 0x8
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_MASK = 0x30
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_SFT = 4
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_HOST_PSN_TABLE = (0x0 << 4)
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_HOST_MSN_TABLE = (0x1 << 4)
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_IQM_MSN_TABLE = (0x2 << 4)
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_LAST = CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_IQM_MSN_TABLE
CREQ_QUERY_FUNC_RESP_SB_MAX_SRQ_EXTENDED = 0x40
CREQ_QUERY_FUNC_RESP_SB_MIN_RNR_RTR_RTS_OPT_SUPPORTED = 0x1000
CMDQ_SET_FUNC_RESOURCES_OPCODE_SET_FUNC_RESOURCES = 0x84
CMDQ_SET_FUNC_RESOURCES_OPCODE_LAST = CMDQ_SET_FUNC_RESOURCES_OPCODE_SET_FUNC_RESOURCES
CMDQ_SET_FUNC_RESOURCES_FLAGS_MRAV_RESERVATION_SPLIT = 0x1
CREQ_SET_FUNC_RESOURCES_RESP_TYPE_MASK = 0x3f
CREQ_SET_FUNC_RESOURCES_RESP_TYPE_SFT = 0
CREQ_SET_FUNC_RESOURCES_RESP_TYPE_QP_EVENT = 0x38
CREQ_SET_FUNC_RESOURCES_RESP_TYPE_LAST = CREQ_SET_FUNC_RESOURCES_RESP_TYPE_QP_EVENT
CREQ_SET_FUNC_RESOURCES_RESP_V = 0x1
CREQ_SET_FUNC_RESOURCES_RESP_EVENT_SET_FUNC_RESOURCES = 0x84
CREQ_SET_FUNC_RESOURCES_RESP_EVENT_LAST = CREQ_SET_FUNC_RESOURCES_RESP_EVENT_SET_FUNC_RESOURCES
CMDQ_READ_CONTEXT_OPCODE_READ_CONTEXT = 0x85
CMDQ_READ_CONTEXT_OPCODE_LAST = CMDQ_READ_CONTEXT_OPCODE_READ_CONTEXT
CMDQ_READ_CONTEXT_TYPE_QPC = 0x0
CMDQ_READ_CONTEXT_TYPE_CQ = 0x1
CMDQ_READ_CONTEXT_TYPE_MRW = 0x2
CMDQ_READ_CONTEXT_TYPE_SRQ = 0x3
CMDQ_READ_CONTEXT_TYPE_LAST = CMDQ_READ_CONTEXT_TYPE_SRQ
CREQ_READ_CONTEXT_TYPE_MASK = 0x3f
CREQ_READ_CONTEXT_TYPE_SFT = 0
CREQ_READ_CONTEXT_TYPE_QP_EVENT = 0x38
CREQ_READ_CONTEXT_TYPE_LAST = CREQ_READ_CONTEXT_TYPE_QP_EVENT
CREQ_READ_CONTEXT_V = 0x1
CREQ_READ_CONTEXT_EVENT_READ_CONTEXT = 0x85
CREQ_READ_CONTEXT_EVENT_LAST = CREQ_READ_CONTEXT_EVENT_READ_CONTEXT
CMDQ_MAP_TC_TO_COS_OPCODE_MAP_TC_TO_COS = 0x8a
CMDQ_MAP_TC_TO_COS_OPCODE_LAST = CMDQ_MAP_TC_TO_COS_OPCODE_MAP_TC_TO_COS
CMDQ_MAP_TC_TO_COS_COS0_NO_CHANGE = 0xffff
CMDQ_MAP_TC_TO_COS_COS0_LAST = CMDQ_MAP_TC_TO_COS_COS0_NO_CHANGE
CMDQ_MAP_TC_TO_COS_COS1_DISABLE = 0x8000
CMDQ_MAP_TC_TO_COS_COS1_NO_CHANGE = 0xffff
CMDQ_MAP_TC_TO_COS_COS1_LAST = CMDQ_MAP_TC_TO_COS_COS1_NO_CHANGE
CREQ_MAP_TC_TO_COS_RESP_TYPE_MASK = 0x3f
CREQ_MAP_TC_TO_COS_RESP_TYPE_SFT = 0
CREQ_MAP_TC_TO_COS_RESP_TYPE_QP_EVENT = 0x38
CREQ_MAP_TC_TO_COS_RESP_TYPE_LAST = CREQ_MAP_TC_TO_COS_RESP_TYPE_QP_EVENT
CREQ_MAP_TC_TO_COS_RESP_V = 0x1
CREQ_MAP_TC_TO_COS_RESP_EVENT_MAP_TC_TO_COS = 0x8a
CREQ_MAP_TC_TO_COS_RESP_EVENT_LAST = CREQ_MAP_TC_TO_COS_RESP_EVENT_MAP_TC_TO_COS
CMDQ_QUERY_ROCE_CC_OPCODE_QUERY_ROCE_CC = 0x8d
CMDQ_QUERY_ROCE_CC_OPCODE_LAST = CMDQ_QUERY_ROCE_CC_OPCODE_QUERY_ROCE_CC
CREQ_QUERY_ROCE_CC_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_ROCE_CC_RESP_TYPE_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_ROCE_CC_RESP_TYPE_LAST = CREQ_QUERY_ROCE_CC_RESP_TYPE_QP_EVENT
CREQ_QUERY_ROCE_CC_RESP_V = 0x1
CREQ_QUERY_ROCE_CC_RESP_EVENT_QUERY_ROCE_CC = 0x8d
CREQ_QUERY_ROCE_CC_RESP_EVENT_LAST = CREQ_QUERY_ROCE_CC_RESP_EVENT_QUERY_ROCE_CC
CREQ_QUERY_ROCE_CC_RESP_SB_OPCODE_QUERY_ROCE_CC = 0x8d
CREQ_QUERY_ROCE_CC_RESP_SB_OPCODE_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_OPCODE_QUERY_ROCE_CC
CREQ_QUERY_ROCE_CC_RESP_SB_ENABLE_CC = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_UNUSED7_MASK = 0xfe
CREQ_QUERY_ROCE_CC_RESP_SB_UNUSED7_SFT = 1
CREQ_QUERY_ROCE_CC_RESP_SB_TOS_ECN_MASK = 0x3
CREQ_QUERY_ROCE_CC_RESP_SB_TOS_ECN_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TOS_DSCP_MASK = 0xfc
CREQ_QUERY_ROCE_CC_RESP_SB_TOS_DSCP_SFT = 2
CREQ_QUERY_ROCE_CC_RESP_SB_ALT_VLAN_PCP_MASK = 0x7
CREQ_QUERY_ROCE_CC_RESP_SB_ALT_VLAN_PCP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD1_MASK = 0xf8
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD1_SFT = 3
CREQ_QUERY_ROCE_CC_RESP_SB_ALT_TOS_DSCP_MASK = 0x3f
CREQ_QUERY_ROCE_CC_RESP_SB_ALT_TOS_DSCP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD4_MASK = 0xc0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD4_SFT = 6
CREQ_QUERY_ROCE_CC_RESP_SB_CC_MODE_DCTCP = 0x0
CREQ_QUERY_ROCE_CC_RESP_SB_CC_MODE_PROBABILISTIC = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_CC_MODE_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_CC_MODE_PROBABILISTIC
CREQ_QUERY_ROCE_CC_RESP_SB_RTT_MASK = 0x3fff
CREQ_QUERY_ROCE_CC_RESP_SB_RTT_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD5_MASK = 0xc000
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD5_SFT = 14
CREQ_QUERY_ROCE_CC_RESP_SB_TCP_CP_MASK = 0x3ff
CREQ_QUERY_ROCE_CC_RESP_SB_TCP_CP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD6_MASK = 0xfc00
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD6_SFT = 10
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_MORE = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_MORE_LAST = 0x0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED = 0x2
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_OPCODE_QUERY_ROCE_CC = 0x8d
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_OPCODE_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_TLV_OPCODE_QUERY_ROCE_CC
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ENABLE_CC = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_UNUSED7_MASK = 0xfe
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_UNUSED7_SFT = 1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TOS_ECN_MASK = 0x3
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TOS_ECN_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TOS_DSCP_MASK = 0xfc
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TOS_DSCP_SFT = 2
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ALT_VLAN_PCP_MASK = 0x7
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ALT_VLAN_PCP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD1_MASK = 0xf8
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD1_SFT = 3
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ALT_TOS_DSCP_MASK = 0x3f
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ALT_TOS_DSCP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD4_MASK = 0xc0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD4_SFT = 6
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_CC_MODE_DCTCP = 0x0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_CC_MODE_PROBABILISTIC = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_CC_MODE_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_TLV_CC_MODE_PROBABILISTIC
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RTT_MASK = 0x3fff
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RTT_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD5_MASK = 0xc000
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD5_SFT = 14
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TCP_CP_MASK = 0x3ff
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TCP_CP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD6_MASK = 0xfc00
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD6_SFT = 10
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_MORE = 0x1
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_MORE_LAST = 0x0
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED = 0x2
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED_LAST = CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_NOT_ECT = 0x0
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_ECT_1 = 0x1
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_ECT_0 = 0x2
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_LAST = CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_ECT_0
CMDQ_MODIFY_ROCE_CC_OPCODE_MODIFY_ROCE_CC = 0x8c
CMDQ_MODIFY_ROCE_CC_OPCODE_LAST = CMDQ_MODIFY_ROCE_CC_OPCODE_MODIFY_ROCE_CC
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_ENABLE_CC = 0x1
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_G = 0x2
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_NUMPHASEPERSTATE = 0x4
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_INIT_CR = 0x8
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_INIT_TR = 0x10
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TOS_ECN = 0x20
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TOS_DSCP = 0x40
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_ALT_VLAN_PCP = 0x80
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_ALT_TOS_DSCP = 0x100
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_RTT = 0x200
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_CC_MODE = 0x400
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TCP_CP = 0x800
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TX_QUEUE = 0x1000
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_INACTIVITY_CP = 0x2000
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TIME_PER_PHASE = 0x4000
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_PKTS_PER_PHASE = 0x8000
CMDQ_MODIFY_ROCE_CC_ENABLE_CC = 0x1
CMDQ_MODIFY_ROCE_CC_RSVD1_MASK = 0xfe
CMDQ_MODIFY_ROCE_CC_RSVD1_SFT = 1
CMDQ_MODIFY_ROCE_CC_TOS_ECN_MASK = 0x3
CMDQ_MODIFY_ROCE_CC_TOS_ECN_SFT = 0
CMDQ_MODIFY_ROCE_CC_TOS_DSCP_MASK = 0xfc
CMDQ_MODIFY_ROCE_CC_TOS_DSCP_SFT = 2
CMDQ_MODIFY_ROCE_CC_ALT_VLAN_PCP_MASK = 0x7
CMDQ_MODIFY_ROCE_CC_ALT_VLAN_PCP_SFT = 0
CMDQ_MODIFY_ROCE_CC_RSVD3_MASK = 0xf8
CMDQ_MODIFY_ROCE_CC_RSVD3_SFT = 3
CMDQ_MODIFY_ROCE_CC_ALT_TOS_DSCP_MASK = 0x3f
CMDQ_MODIFY_ROCE_CC_ALT_TOS_DSCP_SFT = 0
CMDQ_MODIFY_ROCE_CC_RSVD4_MASK = 0xffc0
CMDQ_MODIFY_ROCE_CC_RSVD4_SFT = 6
CMDQ_MODIFY_ROCE_CC_RTT_MASK = 0x3fff
CMDQ_MODIFY_ROCE_CC_RTT_SFT = 0
CMDQ_MODIFY_ROCE_CC_RSVD5_MASK = 0xc000
CMDQ_MODIFY_ROCE_CC_RSVD5_SFT = 14
CMDQ_MODIFY_ROCE_CC_TCP_CP_MASK = 0x3ff
CMDQ_MODIFY_ROCE_CC_TCP_CP_SFT = 0
CMDQ_MODIFY_ROCE_CC_RSVD6_MASK = 0xfc00
CMDQ_MODIFY_ROCE_CC_RSVD6_SFT = 10
CMDQ_MODIFY_ROCE_CC_CC_MODE_DCTCP_CC_MODE = 0x0
CMDQ_MODIFY_ROCE_CC_CC_MODE_PROBABILISTIC_CC_MODE = 0x1
CMDQ_MODIFY_ROCE_CC_CC_MODE_LAST = CMDQ_MODIFY_ROCE_CC_CC_MODE_PROBABILISTIC_CC_MODE
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_MORE = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_MORE_LAST = 0x0
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED = 0x2
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED_LAST = CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED_YES
CMDQ_MODIFY_ROCE_CC_TLV_OPCODE_MODIFY_ROCE_CC = 0x8c
CMDQ_MODIFY_ROCE_CC_TLV_OPCODE_LAST = CMDQ_MODIFY_ROCE_CC_TLV_OPCODE_MODIFY_ROCE_CC
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_ENABLE_CC = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_G = 0x2
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_NUMPHASEPERSTATE = 0x4
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_INIT_CR = 0x8
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_INIT_TR = 0x10
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TOS_ECN = 0x20
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TOS_DSCP = 0x40
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_ALT_VLAN_PCP = 0x80
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_ALT_TOS_DSCP = 0x100
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_RTT = 0x200
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_CC_MODE = 0x400
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TCP_CP = 0x800
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TX_QUEUE = 0x1000
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_INACTIVITY_CP = 0x2000
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TIME_PER_PHASE = 0x4000
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_PKTS_PER_PHASE = 0x8000
CMDQ_MODIFY_ROCE_CC_TLV_ENABLE_CC = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_RSVD1_MASK = 0xfe
CMDQ_MODIFY_ROCE_CC_TLV_RSVD1_SFT = 1
CMDQ_MODIFY_ROCE_CC_TLV_TOS_ECN_MASK = 0x3
CMDQ_MODIFY_ROCE_CC_TLV_TOS_ECN_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_TOS_DSCP_MASK = 0xfc
CMDQ_MODIFY_ROCE_CC_TLV_TOS_DSCP_SFT = 2
CMDQ_MODIFY_ROCE_CC_TLV_ALT_VLAN_PCP_MASK = 0x7
CMDQ_MODIFY_ROCE_CC_TLV_ALT_VLAN_PCP_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD3_MASK = 0xf8
CMDQ_MODIFY_ROCE_CC_TLV_RSVD3_SFT = 3
CMDQ_MODIFY_ROCE_CC_TLV_ALT_TOS_DSCP_MASK = 0x3f
CMDQ_MODIFY_ROCE_CC_TLV_ALT_TOS_DSCP_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD4_MASK = 0xffc0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD4_SFT = 6
CMDQ_MODIFY_ROCE_CC_TLV_RTT_MASK = 0x3fff
CMDQ_MODIFY_ROCE_CC_TLV_RTT_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD5_MASK = 0xc000
CMDQ_MODIFY_ROCE_CC_TLV_RSVD5_SFT = 14
CMDQ_MODIFY_ROCE_CC_TLV_TCP_CP_MASK = 0x3ff
CMDQ_MODIFY_ROCE_CC_TLV_TCP_CP_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD6_MASK = 0xfc00
CMDQ_MODIFY_ROCE_CC_TLV_RSVD6_SFT = 10
CMDQ_MODIFY_ROCE_CC_TLV_CC_MODE_DCTCP_CC_MODE = 0x0
CMDQ_MODIFY_ROCE_CC_TLV_CC_MODE_PROBABILISTIC_CC_MODE = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_CC_MODE_LAST = CMDQ_MODIFY_ROCE_CC_TLV_CC_MODE_PROBABILISTIC_CC_MODE
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_MORE = 0x1
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_MORE_LAST = 0x0
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED = 0x2
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED_LAST = CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED_YES
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_MIN_TIME_BETWEEN_CNPS = 0x1
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_INIT_CP = 0x2
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_TR_UPDATE_MODE = 0x4
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_TR_UPDATE_CYCLES = 0x8
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_FR_NUM_RTTS = 0x10
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_AI_RATE_INCREASE = 0x20
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_REDUCTION_RELAX_RTTS_TH = 0x40
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_ADDITIONAL_RELAX_CR_TH = 0x80
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CR_MIN_TH = 0x100
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_BW_AVG_WEIGHT = 0x200
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_ACTUAL_CR_FACTOR = 0x400
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_MAX_CP_CR_TH = 0x800
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CP_BIAS_EN = 0x1000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CP_BIAS = 0x2000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CNP_ECN = 0x4000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_RTT_JITTER_EN = 0x8000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_LINK_BYTES_PER_USEC = 0x10000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_RESET_CC_CR_TH = 0x20000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CR_WIDTH = 0x40000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_QUOTA_PERIOD_MIN = 0x80000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_QUOTA_PERIOD_MAX = 0x100000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_QUOTA_PERIOD_ABS_MAX = 0x200000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_TR_LOWER_BOUND = 0x400000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CR_PROB_FACTOR = 0x800000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_TR_PROB_FACTOR = 0x1000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_FAIRNESS_CR_TH = 0x2000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_RED_DIV = 0x4000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CNP_RATIO_TH = 0x8000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_EXP_AI_RTTS = 0x10000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_EXP_AI_CR_CP_RATIO = 0x20000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CP_EXP_UPDATE_TH = 0x40000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_HIGH_EXP_AI_RTTS_TH1 = 0x80000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_HIGH_EXP_AI_RTTS_TH2 = 0x100000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_USE_RATE_TABLE = 0x200000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_LINK64B_PER_RTT = 0x400000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_ACTUAL_CR_CONG_FREE_RTTS_TH = 0x800000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_SEVERE_CONG_CR_TH1 = 0x1000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_SEVERE_CONG_CR_TH2 = 0x2000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CC_ACK_BYTES = 0x4000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_REDUCE_INIT_EN = 0x8000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_REDUCE_INIT_CONG_FREE_RTTS_TH = 0x10000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_RANDOM_NO_RED_EN = 0x20000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_ACTUAL_CR_SHIFT_CORRECTION_EN = 0x40000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_QUOTA_PERIOD_ADJUST_EN = 0x80000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_NOT_ECT = 0x0
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_ECT_1 = 0x1
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_ECT_0 = 0x2
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_LAST = CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_ECT_0
CREQ_MODIFY_ROCE_CC_RESP_TYPE_MASK = 0x3f
CREQ_MODIFY_ROCE_CC_RESP_TYPE_SFT = 0
CREQ_MODIFY_ROCE_CC_RESP_TYPE_QP_EVENT = 0x38
CREQ_MODIFY_ROCE_CC_RESP_TYPE_LAST = CREQ_MODIFY_ROCE_CC_RESP_TYPE_QP_EVENT
CREQ_MODIFY_ROCE_CC_RESP_V = 0x1
CREQ_MODIFY_ROCE_CC_RESP_EVENT_MODIFY_ROCE_CC = 0x8c
CREQ_MODIFY_ROCE_CC_RESP_EVENT_LAST = CREQ_MODIFY_ROCE_CC_RESP_EVENT_MODIFY_ROCE_CC
CMDQ_SET_LINK_AGGR_MODE_OPCODE_SET_LINK_AGGR_MODE = 0x8f
CMDQ_SET_LINK_AGGR_MODE_OPCODE_LAST = CMDQ_SET_LINK_AGGR_MODE_OPCODE_SET_LINK_AGGR_MODE
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_AGGR_EN = 0x1
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_ACTIVE_PORT_MAP = 0x2
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_MEMBER_PORT_MAP = 0x4
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_AGGR_MODE = 0x8
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_STAT_CTX_ID = 0x10
CMDQ_SET_LINK_AGGR_MODE_AGGR_ENABLE = 0x1
CMDQ_SET_LINK_AGGR_MODE_RSVD1_MASK = 0xfe
CMDQ_SET_LINK_AGGR_MODE_RSVD1_SFT = 1
CMDQ_SET_LINK_AGGR_MODE_ACTIVE_PORT_MAP_MASK = 0xf
CMDQ_SET_LINK_AGGR_MODE_ACTIVE_PORT_MAP_SFT = 0
CMDQ_SET_LINK_AGGR_MODE_RSVD2_MASK = 0xf0
CMDQ_SET_LINK_AGGR_MODE_RSVD2_SFT = 4
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_ACTIVE_ACTIVE = 0x1
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_ACTIVE_BACKUP = 0x2
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_BALANCE_XOR = 0x3
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_802_3_AD = 0x4
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_LAST = CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_802_3_AD
CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_MASK = 0x3f
CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_SFT = 0
CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_QP_EVENT = 0x38
CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_LAST = CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_QP_EVENT
CREQ_SET_LINK_AGGR_MODE_RESP_V = 0x1
CREQ_SET_LINK_AGGR_MODE_RESP_EVENT_SET_LINK_AGGR_MODE = 0x8f
CREQ_SET_LINK_AGGR_MODE_RESP_EVENT_LAST = CREQ_SET_LINK_AGGR_MODE_RESP_EVENT_SET_LINK_AGGR_MODE
CREQ_FUNC_EVENT_TYPE_MASK = 0x3f
CREQ_FUNC_EVENT_TYPE_SFT = 0
CREQ_FUNC_EVENT_TYPE_FUNC_EVENT = 0x3a
CREQ_FUNC_EVENT_TYPE_LAST = CREQ_FUNC_EVENT_TYPE_FUNC_EVENT
CREQ_FUNC_EVENT_V = 0x1
CREQ_FUNC_EVENT_EVENT_TX_WQE_ERROR = 0x1
CREQ_FUNC_EVENT_EVENT_TX_DATA_ERROR = 0x2
CREQ_FUNC_EVENT_EVENT_RX_WQE_ERROR = 0x3
CREQ_FUNC_EVENT_EVENT_RX_DATA_ERROR = 0x4
CREQ_FUNC_EVENT_EVENT_CQ_ERROR = 0x5
CREQ_FUNC_EVENT_EVENT_TQM_ERROR = 0x6
CREQ_FUNC_EVENT_EVENT_CFCQ_ERROR = 0x7
CREQ_FUNC_EVENT_EVENT_CFCS_ERROR = 0x8
CREQ_FUNC_EVENT_EVENT_CFCC_ERROR = 0x9
CREQ_FUNC_EVENT_EVENT_CFCM_ERROR = 0xa
CREQ_FUNC_EVENT_EVENT_TIM_ERROR = 0xb
CREQ_FUNC_EVENT_EVENT_VF_COMM_REQUEST = 0x80
CREQ_FUNC_EVENT_EVENT_RESOURCE_EXHAUSTED = 0x81
CREQ_FUNC_EVENT_EVENT_LAST = CREQ_FUNC_EVENT_EVENT_RESOURCE_EXHAUSTED
CREQ_QP_EVENT_TYPE_MASK = 0x3f
CREQ_QP_EVENT_TYPE_SFT = 0
CREQ_QP_EVENT_TYPE_QP_EVENT = 0x38
CREQ_QP_EVENT_TYPE_LAST = CREQ_QP_EVENT_TYPE_QP_EVENT
CREQ_QP_EVENT_STATUS_SUCCESS = 0x0
CREQ_QP_EVENT_STATUS_FAIL = 0x1
CREQ_QP_EVENT_STATUS_RESOURCES = 0x2
CREQ_QP_EVENT_STATUS_INVALID_CMD = 0x3
CREQ_QP_EVENT_STATUS_NOT_IMPLEMENTED = 0x4
CREQ_QP_EVENT_STATUS_INVALID_PARAMETER = 0x5
CREQ_QP_EVENT_STATUS_HARDWARE_ERROR = 0x6
CREQ_QP_EVENT_STATUS_INTERNAL_ERROR = 0x7
CREQ_QP_EVENT_STATUS_LAST = CREQ_QP_EVENT_STATUS_INTERNAL_ERROR
CREQ_QP_EVENT_V = 0x1
CREQ_QP_EVENT_EVENT_CREATE_QP = 0x1
CREQ_QP_EVENT_EVENT_DESTROY_QP = 0x2
CREQ_QP_EVENT_EVENT_MODIFY_QP = 0x3
CREQ_QP_EVENT_EVENT_QUERY_QP = 0x4
CREQ_QP_EVENT_EVENT_CREATE_SRQ = 0x5
CREQ_QP_EVENT_EVENT_DESTROY_SRQ = 0x6
CREQ_QP_EVENT_EVENT_QUERY_SRQ = 0x8
CREQ_QP_EVENT_EVENT_CREATE_CQ = 0x9
CREQ_QP_EVENT_EVENT_DESTROY_CQ = 0xa
CREQ_QP_EVENT_EVENT_RESIZE_CQ = 0xc
CREQ_QP_EVENT_EVENT_ALLOCATE_MRW = 0xd
CREQ_QP_EVENT_EVENT_DEALLOCATE_KEY = 0xe
CREQ_QP_EVENT_EVENT_REGISTER_MR = 0xf
CREQ_QP_EVENT_EVENT_DEREGISTER_MR = 0x10
CREQ_QP_EVENT_EVENT_ADD_GID = 0x11
CREQ_QP_EVENT_EVENT_DELETE_GID = 0x12
CREQ_QP_EVENT_EVENT_MODIFY_GID = 0x17
CREQ_QP_EVENT_EVENT_QUERY_GID = 0x18
CREQ_QP_EVENT_EVENT_CREATE_QP1 = 0x13
CREQ_QP_EVENT_EVENT_DESTROY_QP1 = 0x14
CREQ_QP_EVENT_EVENT_CREATE_AH = 0x15
CREQ_QP_EVENT_EVENT_DESTROY_AH = 0x16
CREQ_QP_EVENT_EVENT_INITIALIZE_FW = 0x80
CREQ_QP_EVENT_EVENT_DEINITIALIZE_FW = 0x81
CREQ_QP_EVENT_EVENT_STOP_FUNC = 0x82
CREQ_QP_EVENT_EVENT_QUERY_FUNC = 0x83
CREQ_QP_EVENT_EVENT_SET_FUNC_RESOURCES = 0x84
CREQ_QP_EVENT_EVENT_READ_CONTEXT = 0x85
CREQ_QP_EVENT_EVENT_MAP_TC_TO_COS = 0x8a
CREQ_QP_EVENT_EVENT_QUERY_VERSION = 0x8b
CREQ_QP_EVENT_EVENT_MODIFY_CC = 0x8c
CREQ_QP_EVENT_EVENT_QUERY_CC = 0x8d
CREQ_QP_EVENT_EVENT_QUERY_ROCE_STATS = 0x8e
CREQ_QP_EVENT_EVENT_SET_LINK_AGGR_MODE = 0x8f
CREQ_QP_EVENT_EVENT_QUERY_QP_EXTEND = 0x91
CREQ_QP_EVENT_EVENT_QP_ERROR_NOTIFICATION = 0xc0
CREQ_QP_EVENT_EVENT_CQ_ERROR_NOTIFICATION = 0xc1
CREQ_QP_EVENT_EVENT_LAST = CREQ_QP_EVENT_EVENT_CQ_ERROR_NOTIFICATION
CREQ_QP_ERROR_NOTIFICATION_TYPE_MASK = 0x3f
CREQ_QP_ERROR_NOTIFICATION_TYPE_SFT = 0
CREQ_QP_ERROR_NOTIFICATION_TYPE_QP_EVENT = 0x38
CREQ_QP_ERROR_NOTIFICATION_TYPE_LAST = CREQ_QP_ERROR_NOTIFICATION_TYPE_QP_EVENT
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_NO_ERROR = 0X0
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_OPCODE_ERROR = 0X1
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_TIMEOUT_RETRY_LIMIT = 0X2
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RNR_TIMEOUT_RETRY_LIMIT = 0X3
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_NAK_ARRIVAL_1 = 0X4
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_NAK_ARRIVAL_2 = 0X5
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_NAK_ARRIVAL_3 = 0X6
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_NAK_ARRIVAL_4 = 0X7
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RX_MEMORY_ERROR = 0X8
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_TX_MEMORY_ERROR = 0X9
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_READ_RESP_LENGTH = 0XA
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_INVALID_READ_RESP = 0XB
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_ILLEGAL_BIND = 0XC
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_ILLEGAL_FAST_REG = 0XD
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_ILLEGAL_INVALIDATE = 0XE
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_CMP_ERROR = 0XF
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RETRAN_LOCAL_ERROR = 0X10
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_WQE_FORMAT_ERROR = 0X11
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_ORRQ_FORMAT_ERROR = 0X12
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_INVALID_AVID_ERROR = 0X13
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_AV_DOMAIN_ERROR = 0X14
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_CQ_LOAD_ERROR = 0X15
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_SERV_TYPE_ERROR = 0X16
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_INVALID_OP_ERROR = 0X17
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_TX_PCI_ERROR = 0X18
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RX_PCI_ERROR = 0X19
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_PROD_WQE_MSMTCH_ERROR = 0X1A
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_PSN_RANGE_CHECK_ERROR = 0X1B
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RETX_SETUP_ERROR = 0X1C
CREQ_QP_ERROR_NOTIFICATION_V = 0x1
CREQ_QP_ERROR_NOTIFICATION_EVENT_QP_ERROR_NOTIFICATION = 0xc0
CREQ_QP_ERROR_NOTIFICATION_EVENT_LAST = CREQ_QP_ERROR_NOTIFICATION_EVENT_QP_ERROR_NOTIFICATION
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_NO_ERROR = 0x0
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_EXCEED_MAX = 0x1
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_PAYLOAD_LENGTH_MISMATCH = 0x2
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_EXCEEDS_WQE = 0x3
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_OPCODE_ERROR = 0x4
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_PSN_SEQ_ERROR_RETRY_LIMIT = 0x5
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_INVALID_R_KEY = 0x6
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_DOMAIN_ERROR = 0x7
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_NO_PERMISSION = 0x8
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_RANGE_ERROR = 0x9
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_INVALID_R_KEY = 0xa
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_DOMAIN_ERROR = 0xb
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_NO_PERMISSION = 0xc
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_RANGE_ERROR = 0xd
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_IRRQ_OFLOW = 0xe
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_UNSUPPORTED_OPCODE = 0xf
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_UNALIGN_ATOMIC = 0x10
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_REM_INVALIDATE = 0x11
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_MEMORY_ERROR = 0x12
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_SRQ_ERROR = 0x13
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_CMP_ERROR = 0x14
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_INVALID_DUP_RKEY = 0x15
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_WQE_FORMAT_ERROR = 0x16
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_IRRQ_FORMAT_ERROR = 0x17
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_CQ_LOAD_ERROR = 0x18
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_SRQ_LOAD_ERROR = 0x19
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_PCI_ERROR = 0x1b
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_PCI_ERROR = 0x1c
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_PSN_NOT_FOUND = 0x1d
CREQ_CQ_ERROR_NOTIFICATION_TYPE_MASK = 0x3f
CREQ_CQ_ERROR_NOTIFICATION_TYPE_SFT = 0
CREQ_CQ_ERROR_NOTIFICATION_TYPE_CQ_EVENT = 0x38
CREQ_CQ_ERROR_NOTIFICATION_TYPE_LAST = CREQ_CQ_ERROR_NOTIFICATION_TYPE_CQ_EVENT
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_REQ_CQ_INVALID_ERROR = 0x1
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_REQ_CQ_OVERFLOW_ERROR = 0x2
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_REQ_CQ_LOAD_ERROR = 0x3
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_RES_CQ_INVALID_ERROR = 0x4
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_RES_CQ_OVERFLOW_ERROR = 0x5
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_RES_CQ_LOAD_ERROR = 0x6
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_LAST = CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_RES_CQ_LOAD_ERROR
CREQ_CQ_ERROR_NOTIFICATION_V = 0x1
CREQ_CQ_ERROR_NOTIFICATION_EVENT_CQ_ERROR_NOTIFICATION = 0xc1
CREQ_CQ_ERROR_NOTIFICATION_EVENT_LAST = CREQ_CQ_ERROR_NOTIFICATION_EVENT_CQ_ERROR_NOTIFICATION
SQ_BASE_WQE_TYPE_SEND = 0x0
SQ_BASE_WQE_TYPE_SEND_W_IMMEAD = 0x1
SQ_BASE_WQE_TYPE_SEND_W_INVALID = 0x2
SQ_BASE_WQE_TYPE_WRITE_WQE = 0x4
SQ_BASE_WQE_TYPE_WRITE_W_IMMEAD = 0x5
SQ_BASE_WQE_TYPE_READ_WQE = 0x6
SQ_BASE_WQE_TYPE_ATOMIC_CS = 0x8
SQ_BASE_WQE_TYPE_ATOMIC_FA = 0xb
SQ_BASE_WQE_TYPE_LOCAL_INVALID = 0xc
SQ_BASE_WQE_TYPE_FR_PMR = 0xd
SQ_BASE_WQE_TYPE_BIND = 0xe
SQ_BASE_WQE_TYPE_FR_PPMR = 0xf
SQ_BASE_WQE_TYPE_LAST = SQ_BASE_WQE_TYPE_FR_PPMR
SQ_PSN_SEARCH_START_PSN_MASK = 0xffffff
SQ_PSN_SEARCH_START_PSN_SFT = 0
SQ_PSN_SEARCH_OPCODE_MASK = 0xff000000
SQ_PSN_SEARCH_OPCODE_SFT = 24
SQ_PSN_SEARCH_NEXT_PSN_MASK = 0xffffff
SQ_PSN_SEARCH_NEXT_PSN_SFT = 0
SQ_PSN_SEARCH_FLAGS_MASK = 0xff000000
SQ_PSN_SEARCH_FLAGS_SFT = 24
SQ_PSN_SEARCH_EXT_START_PSN_MASK = 0xffffff
SQ_PSN_SEARCH_EXT_START_PSN_SFT = 0
SQ_PSN_SEARCH_EXT_OPCODE_MASK = 0xff000000
SQ_PSN_SEARCH_EXT_OPCODE_SFT = 24
SQ_PSN_SEARCH_EXT_NEXT_PSN_MASK = 0xffffff
SQ_PSN_SEARCH_EXT_NEXT_PSN_SFT = 0
SQ_PSN_SEARCH_EXT_FLAGS_MASK = 0xff000000
SQ_PSN_SEARCH_EXT_FLAGS_SFT = 24
SQ_MSN_SEARCH_START_PSN_MASK = 0xffffff
SQ_MSN_SEARCH_START_PSN_SFT = 0
SQ_MSN_SEARCH_NEXT_PSN_MASK = 0xffffff000000
SQ_MSN_SEARCH_NEXT_PSN_SFT = 24
SQ_MSN_SEARCH_START_IDX_MASK = 0xffff000000000000
SQ_MSN_SEARCH_START_IDX_SFT = 48
SQ_SEND_WQE_TYPE_SEND = 0x0
SQ_SEND_WQE_TYPE_SEND_W_IMMEAD = 0x1
SQ_SEND_WQE_TYPE_SEND_W_INVALID = 0x2
SQ_SEND_WQE_TYPE_LAST = SQ_SEND_WQE_TYPE_SEND_W_INVALID
SQ_SEND_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_SEND_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_SEND_FLAGS_SIGNAL_COMP = 0x1
SQ_SEND_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_SEND_FLAGS_UC_FENCE = 0x4
SQ_SEND_FLAGS_SE = 0x8
SQ_SEND_FLAGS_INLINE = 0x10
SQ_SEND_FLAGS_WQE_TS_EN = 0x20
SQ_SEND_FLAGS_DEBUG_TRACE = 0x40
SQ_SEND_DST_QP_MASK = 0xffffff
SQ_SEND_DST_QP_SFT = 0
SQ_SEND_AVID_MASK = 0xfffff
SQ_SEND_AVID_SFT = 0
SQ_SEND_TIMESTAMP_MASK = 0xffffff
SQ_SEND_TIMESTAMP_SFT = 0
SQ_SEND_HDR_WQE_TYPE_SEND = 0x0
SQ_SEND_HDR_WQE_TYPE_SEND_W_IMMEAD = 0x1
SQ_SEND_HDR_WQE_TYPE_SEND_W_INVALID = 0x2
SQ_SEND_HDR_WQE_TYPE_LAST = SQ_SEND_HDR_WQE_TYPE_SEND_W_INVALID
SQ_SEND_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_SEND_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_SEND_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_SEND_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_SEND_HDR_FLAGS_UC_FENCE = 0x4
SQ_SEND_HDR_FLAGS_SE = 0x8
SQ_SEND_HDR_FLAGS_INLINE = 0x10
SQ_SEND_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_SEND_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_SEND_HDR_DST_QP_MASK = 0xffffff
SQ_SEND_HDR_DST_QP_SFT = 0
SQ_SEND_HDR_AVID_MASK = 0xfffff
SQ_SEND_HDR_AVID_SFT = 0
SQ_SEND_HDR_TIMESTAMP_MASK = 0xffffff
SQ_SEND_HDR_TIMESTAMP_SFT = 0
SQ_SEND_RAWETH_QP1_WQE_TYPE_SEND = 0x0
SQ_SEND_RAWETH_QP1_WQE_TYPE_LAST = SQ_SEND_RAWETH_QP1_WQE_TYPE_SEND
SQ_SEND_RAWETH_QP1_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_SEND_RAWETH_QP1_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_SEND_RAWETH_QP1_FLAGS_SIGNAL_COMP = 0x1
SQ_SEND_RAWETH_QP1_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_SEND_RAWETH_QP1_FLAGS_UC_FENCE = 0x4
SQ_SEND_RAWETH_QP1_FLAGS_SE = 0x8
SQ_SEND_RAWETH_QP1_FLAGS_INLINE = 0x10
SQ_SEND_RAWETH_QP1_FLAGS_WQE_TS_EN = 0x20
SQ_SEND_RAWETH_QP1_FLAGS_DEBUG_TRACE = 0x40
SQ_SEND_RAWETH_QP1_LFLAGS_TCP_UDP_CHKSUM = 0x1
SQ_SEND_RAWETH_QP1_LFLAGS_IP_CHKSUM = 0x2
SQ_SEND_RAWETH_QP1_LFLAGS_NOCRC = 0x4
SQ_SEND_RAWETH_QP1_LFLAGS_STAMP = 0x8
SQ_SEND_RAWETH_QP1_LFLAGS_T_IP_CHKSUM = 0x10
SQ_SEND_RAWETH_QP1_LFLAGS_ROCE_CRC = 0x100
SQ_SEND_RAWETH_QP1_LFLAGS_FCOE_CRC = 0x200
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_VID_MASK = 0xfff
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_VID_SFT = 0
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_DE = 0x1000
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_PRI_MASK = 0xe000
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_PRI_SFT = 13
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_MASK = 0x70000
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_SFT = 16
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID88A8 = (0x0 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID8100 = (0x1 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID9100 = (0x2 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID9200 = (0x3 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID9300 = (0x4 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPIDCFG = (0x5 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_LAST = SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPIDCFG
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_RESERVED_MASK = 0xff80000
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_RESERVED_SFT = 19
SQ_SEND_RAWETH_QP1_CFA_META_KEY_MASK = 0xf0000000
SQ_SEND_RAWETH_QP1_CFA_META_KEY_SFT = 28
SQ_SEND_RAWETH_QP1_CFA_META_KEY_NONE = (0x0 << 28)
SQ_SEND_RAWETH_QP1_CFA_META_KEY_VLAN_TAG = (0x1 << 28)
SQ_SEND_RAWETH_QP1_CFA_META_KEY_LAST = SQ_SEND_RAWETH_QP1_CFA_META_KEY_VLAN_TAG
SQ_SEND_RAWETH_QP1_TIMESTAMP_MASK = 0xffffff
SQ_SEND_RAWETH_QP1_TIMESTAMP_SFT = 0
SQ_SEND_RAWETH_QP1_HDR_WQE_TYPE_SEND = 0x0
SQ_SEND_RAWETH_QP1_HDR_WQE_TYPE_LAST = SQ_SEND_RAWETH_QP1_HDR_WQE_TYPE_SEND
SQ_SEND_RAWETH_QP1_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_SEND_RAWETH_QP1_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_SEND_RAWETH_QP1_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_SEND_RAWETH_QP1_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_SEND_RAWETH_QP1_HDR_FLAGS_UC_FENCE = 0x4
SQ_SEND_RAWETH_QP1_HDR_FLAGS_SE = 0x8
SQ_SEND_RAWETH_QP1_HDR_FLAGS_INLINE = 0x10
SQ_SEND_RAWETH_QP1_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_SEND_RAWETH_QP1_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_TCP_UDP_CHKSUM = 0x1
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_IP_CHKSUM = 0x2
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_NOCRC = 0x4
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_STAMP = 0x8
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_T_IP_CHKSUM = 0x10
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_ROCE_CRC = 0x100
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_FCOE_CRC = 0x200
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_VID_MASK = 0xfff
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_VID_SFT = 0
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_DE = 0x1000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_PRI_MASK = 0xe000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_PRI_SFT = 13
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_MASK = 0x70000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_SFT = 16
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID88A8 = (0x0 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID8100 = (0x1 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID9100 = (0x2 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID9200 = (0x3 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID9300 = (0x4 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPIDCFG = (0x5 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_LAST = SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPIDCFG
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_RESERVED_MASK = 0xff80000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_RESERVED_SFT = 19
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_MASK = 0xf0000000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_SFT = 28
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_NONE = (0x0 << 28)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_VLAN_TAG = (0x1 << 28)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_LAST = SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_VLAN_TAG
SQ_SEND_RAWETH_QP1_HDR_TIMESTAMP_MASK = 0xffffff
SQ_SEND_RAWETH_QP1_HDR_TIMESTAMP_SFT = 0
SQ_RDMA_WQE_TYPE_WRITE_WQE = 0x4
SQ_RDMA_WQE_TYPE_WRITE_W_IMMEAD = 0x5
SQ_RDMA_WQE_TYPE_READ_WQE = 0x6
SQ_RDMA_WQE_TYPE_LAST = SQ_RDMA_WQE_TYPE_READ_WQE
SQ_RDMA_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_RDMA_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_RDMA_FLAGS_SIGNAL_COMP = 0x1
SQ_RDMA_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_RDMA_FLAGS_UC_FENCE = 0x4
SQ_RDMA_FLAGS_SE = 0x8
SQ_RDMA_FLAGS_INLINE = 0x10
SQ_RDMA_FLAGS_WQE_TS_EN = 0x20
SQ_RDMA_FLAGS_DEBUG_TRACE = 0x40
SQ_RDMA_TIMESTAMP_MASK = 0xffffff
SQ_RDMA_TIMESTAMP_SFT = 0
SQ_RDMA_HDR_WQE_TYPE_WRITE_WQE = 0x4
SQ_RDMA_HDR_WQE_TYPE_WRITE_W_IMMEAD = 0x5
SQ_RDMA_HDR_WQE_TYPE_READ_WQE = 0x6
SQ_RDMA_HDR_WQE_TYPE_LAST = SQ_RDMA_HDR_WQE_TYPE_READ_WQE
SQ_RDMA_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_RDMA_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_RDMA_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_RDMA_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_RDMA_HDR_FLAGS_UC_FENCE = 0x4
SQ_RDMA_HDR_FLAGS_SE = 0x8
SQ_RDMA_HDR_FLAGS_INLINE = 0x10
SQ_RDMA_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_RDMA_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_RDMA_HDR_TIMESTAMP_MASK = 0xffffff
SQ_RDMA_HDR_TIMESTAMP_SFT = 0
SQ_ATOMIC_WQE_TYPE_ATOMIC_CS = 0x8
SQ_ATOMIC_WQE_TYPE_ATOMIC_FA = 0xb
SQ_ATOMIC_WQE_TYPE_LAST = SQ_ATOMIC_WQE_TYPE_ATOMIC_FA
SQ_ATOMIC_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_ATOMIC_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_ATOMIC_FLAGS_SIGNAL_COMP = 0x1
SQ_ATOMIC_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_ATOMIC_FLAGS_UC_FENCE = 0x4
SQ_ATOMIC_FLAGS_SE = 0x8
SQ_ATOMIC_FLAGS_INLINE = 0x10
SQ_ATOMIC_FLAGS_WQE_TS_EN = 0x20
SQ_ATOMIC_FLAGS_DEBUG_TRACE = 0x40
SQ_ATOMIC_HDR_WQE_TYPE_ATOMIC_CS = 0x8
SQ_ATOMIC_HDR_WQE_TYPE_ATOMIC_FA = 0xb
SQ_ATOMIC_HDR_WQE_TYPE_LAST = SQ_ATOMIC_HDR_WQE_TYPE_ATOMIC_FA
SQ_ATOMIC_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_ATOMIC_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_ATOMIC_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_ATOMIC_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_ATOMIC_HDR_FLAGS_UC_FENCE = 0x4
SQ_ATOMIC_HDR_FLAGS_SE = 0x8
SQ_ATOMIC_HDR_FLAGS_INLINE = 0x10
SQ_ATOMIC_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_ATOMIC_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_LOCALINVALIDATE_WQE_TYPE_LOCAL_INVALID = 0xc
SQ_LOCALINVALIDATE_WQE_TYPE_LAST = SQ_LOCALINVALIDATE_WQE_TYPE_LOCAL_INVALID
SQ_LOCALINVALIDATE_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_LOCALINVALIDATE_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_LOCALINVALIDATE_FLAGS_SIGNAL_COMP = 0x1
SQ_LOCALINVALIDATE_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_LOCALINVALIDATE_FLAGS_UC_FENCE = 0x4
SQ_LOCALINVALIDATE_FLAGS_SE = 0x8
SQ_LOCALINVALIDATE_FLAGS_INLINE = 0x10
SQ_LOCALINVALIDATE_FLAGS_WQE_TS_EN = 0x20
SQ_LOCALINVALIDATE_FLAGS_DEBUG_TRACE = 0x40
SQ_LOCALINVALIDATE_HDR_WQE_TYPE_LOCAL_INVALID = 0xc
SQ_LOCALINVALIDATE_HDR_WQE_TYPE_LAST = SQ_LOCALINVALIDATE_HDR_WQE_TYPE_LOCAL_INVALID
SQ_LOCALINVALIDATE_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_LOCALINVALIDATE_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_LOCALINVALIDATE_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_LOCALINVALIDATE_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_LOCALINVALIDATE_HDR_FLAGS_UC_FENCE = 0x4
SQ_LOCALINVALIDATE_HDR_FLAGS_SE = 0x8
SQ_LOCALINVALIDATE_HDR_FLAGS_INLINE = 0x10
SQ_LOCALINVALIDATE_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_LOCALINVALIDATE_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_FR_PMR_WQE_TYPE_FR_PMR = 0xd
SQ_FR_PMR_WQE_TYPE_LAST = SQ_FR_PMR_WQE_TYPE_FR_PMR
SQ_FR_PMR_FLAGS_SIGNAL_COMP = 0x1
SQ_FR_PMR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_FR_PMR_FLAGS_UC_FENCE = 0x4
SQ_FR_PMR_FLAGS_SE = 0x8
SQ_FR_PMR_FLAGS_INLINE = 0x10
SQ_FR_PMR_FLAGS_WQE_TS_EN = 0x20
SQ_FR_PMR_FLAGS_DEBUG_TRACE = 0x40
SQ_FR_PMR_ACCESS_CNTL_LOCAL_WRITE = 0x1
SQ_FR_PMR_ACCESS_CNTL_REMOTE_READ = 0x2
SQ_FR_PMR_ACCESS_CNTL_REMOTE_WRITE = 0x4
SQ_FR_PMR_ACCESS_CNTL_REMOTE_ATOMIC = 0x8
SQ_FR_PMR_ACCESS_CNTL_WINDOW_BIND = 0x10
SQ_FR_PMR_PAGE_SIZE_LOG_MASK = 0x1f
SQ_FR_PMR_PAGE_SIZE_LOG_SFT = 0
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_4K = 0x0
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8K = 0x1
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_16K = 0x2
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_32K = 0x3
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_64K = 0x4
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_128K = 0x5
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_256K = 0x6
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_512K = 0x7
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_1M = 0x8
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_2M = 0x9
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_4M = 0xa
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8M = 0xb
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_16M = 0xc
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_32M = 0xd
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_64M = 0xe
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_128M = 0xf
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_256M = 0x10
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_512M = 0x11
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_1G = 0x12
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_2G = 0x13
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_4G = 0x14
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8G = 0x15
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_16G = 0x16
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_32G = 0x17
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_64G = 0x18
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_128G = 0x19
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_256G = 0x1a
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_512G = 0x1b
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_1T = 0x1c
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_2T = 0x1d
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_4T = 0x1e
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8T = 0x1f
SQ_FR_PMR_PAGE_SIZE_LOG_LAST = SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8T
SQ_FR_PMR_ZERO_BASED = 0x20
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_MASK = 0x1f
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_SFT = 0
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_4K = 0x0
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8K = 0x1
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_16K = 0x2
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_32K = 0x3
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_64K = 0x4
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_128K = 0x5
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_256K = 0x6
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_512K = 0x7
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_1M = 0x8
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_2M = 0x9
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_4M = 0xa
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8M = 0xb
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_16M = 0xc
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_32M = 0xd
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_64M = 0xe
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_128M = 0xf
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_256M = 0x10
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_512M = 0x11
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_1G = 0x12
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_2G = 0x13
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_4G = 0x14
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8G = 0x15
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_16G = 0x16
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_32G = 0x17
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_64G = 0x18
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_128G = 0x19
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_256G = 0x1a
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_512G = 0x1b
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_1T = 0x1c
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_2T = 0x1d
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_4T = 0x1e
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8T = 0x1f
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_LAST = SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8T
SQ_FR_PMR_NUMLEVELS_MASK = 0xc0
SQ_FR_PMR_NUMLEVELS_SFT = 6
SQ_FR_PMR_NUMLEVELS_PHYSICAL = (0x0 << 6)
SQ_FR_PMR_NUMLEVELS_LAYER1 = (0x1 << 6)
SQ_FR_PMR_NUMLEVELS_LAYER2 = (0x2 << 6)
SQ_FR_PMR_NUMLEVELS_LAST = SQ_FR_PMR_NUMLEVELS_LAYER2
SQ_FR_PMR_HDR_WQE_TYPE_FR_PMR = 0xd
SQ_FR_PMR_HDR_WQE_TYPE_LAST = SQ_FR_PMR_HDR_WQE_TYPE_FR_PMR
SQ_FR_PMR_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_FR_PMR_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_FR_PMR_HDR_FLAGS_UC_FENCE = 0x4
SQ_FR_PMR_HDR_FLAGS_SE = 0x8
SQ_FR_PMR_HDR_FLAGS_INLINE = 0x10
SQ_FR_PMR_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_FR_PMR_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_FR_PMR_HDR_ACCESS_CNTL_LOCAL_WRITE = 0x1
SQ_FR_PMR_HDR_ACCESS_CNTL_REMOTE_READ = 0x2
SQ_FR_PMR_HDR_ACCESS_CNTL_REMOTE_WRITE = 0x4
SQ_FR_PMR_HDR_ACCESS_CNTL_REMOTE_ATOMIC = 0x8
SQ_FR_PMR_HDR_ACCESS_CNTL_WINDOW_BIND = 0x10
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_MASK = 0x1f
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_SFT = 0
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_4K = 0x0
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8K = 0x1
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_16K = 0x2
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_32K = 0x3
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_64K = 0x4
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_128K = 0x5
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_256K = 0x6
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_512K = 0x7
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_1M = 0x8
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_2M = 0x9
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_4M = 0xa
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8M = 0xb
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_16M = 0xc
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_32M = 0xd
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_64M = 0xe
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_128M = 0xf
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_256M = 0x10
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_512M = 0x11
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_1G = 0x12
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_2G = 0x13
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_4G = 0x14
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8G = 0x15
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_16G = 0x16
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_32G = 0x17
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_64G = 0x18
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_128G = 0x19
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_256G = 0x1a
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_512G = 0x1b
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_1T = 0x1c
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_2T = 0x1d
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_4T = 0x1e
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8T = 0x1f
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_LAST = SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8T
SQ_FR_PMR_HDR_ZERO_BASED = 0x20
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_MASK = 0x1f
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_SFT = 0
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_4K = 0x0
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8K = 0x1
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_16K = 0x2
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_32K = 0x3
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_64K = 0x4
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_128K = 0x5
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_256K = 0x6
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_512K = 0x7
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_1M = 0x8
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_2M = 0x9
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_4M = 0xa
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8M = 0xb
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_16M = 0xc
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_32M = 0xd
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_64M = 0xe
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_128M = 0xf
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_256M = 0x10
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_512M = 0x11
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_1G = 0x12
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_2G = 0x13
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_4G = 0x14
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8G = 0x15
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_16G = 0x16
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_32G = 0x17
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_64G = 0x18
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_128G = 0x19
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_256G = 0x1a
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_512G = 0x1b
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_1T = 0x1c
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_2T = 0x1d
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_4T = 0x1e
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8T = 0x1f
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_LAST = SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8T
SQ_FR_PMR_HDR_NUMLEVELS_MASK = 0xc0
SQ_FR_PMR_HDR_NUMLEVELS_SFT = 6
SQ_FR_PMR_HDR_NUMLEVELS_PHYSICAL = (0x0 << 6)
SQ_FR_PMR_HDR_NUMLEVELS_LAYER1 = (0x1 << 6)
SQ_FR_PMR_HDR_NUMLEVELS_LAYER2 = (0x2 << 6)
SQ_FR_PMR_HDR_NUMLEVELS_LAST = SQ_FR_PMR_HDR_NUMLEVELS_LAYER2
SQ_BIND_WQE_TYPE_BIND = 0xe
SQ_BIND_WQE_TYPE_LAST = SQ_BIND_WQE_TYPE_BIND
SQ_BIND_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_BIND_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_BIND_FLAGS_SIGNAL_COMP = 0x1
SQ_BIND_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_BIND_FLAGS_UC_FENCE = 0x4
SQ_BIND_FLAGS_SE = 0x8
SQ_BIND_FLAGS_INLINE = 0x10
SQ_BIND_FLAGS_WQE_TS_EN = 0x20
SQ_BIND_FLAGS_DEBUG_TRACE = 0x40
SQ_BIND_ACCESS_CNTL_WINDOW_BIND_REMOTE_ATOMIC_REMOTE_WRITE_REMOTE_READ_LOCAL_WRITE_MASK = 0xff
SQ_BIND_ACCESS_CNTL_WINDOW_BIND_REMOTE_ATOMIC_REMOTE_WRITE_REMOTE_READ_LOCAL_WRITE_SFT = 0
SQ_BIND_ACCESS_CNTL_LOCAL_WRITE = 0x1
SQ_BIND_ACCESS_CNTL_REMOTE_READ = 0x2
SQ_BIND_ACCESS_CNTL_REMOTE_WRITE = 0x4
SQ_BIND_ACCESS_CNTL_REMOTE_ATOMIC = 0x8
SQ_BIND_ACCESS_CNTL_WINDOW_BIND = 0x10
SQ_BIND_ZERO_BASED = 0x1
SQ_BIND_MW_TYPE = 0x2
SQ_BIND_MW_TYPE_TYPE1 = (0x0 << 1)
SQ_BIND_MW_TYPE_TYPE2 = (0x1 << 1)
SQ_BIND_MW_TYPE_LAST = SQ_BIND_MW_TYPE_TYPE2
SQ_BIND_HDR_WQE_TYPE_BIND = 0xe
SQ_BIND_HDR_WQE_TYPE_LAST = SQ_BIND_HDR_WQE_TYPE_BIND
SQ_BIND_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_BIND_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_BIND_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_BIND_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_BIND_HDR_FLAGS_UC_FENCE = 0x4
SQ_BIND_HDR_FLAGS_SE = 0x8
SQ_BIND_HDR_FLAGS_INLINE = 0x10
SQ_BIND_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_BIND_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_BIND_HDR_ACCESS_CNTL_WINDOW_BIND_REMOTE_ATOMIC_REMOTE_WRITE_REMOTE_READ_LOCAL_WRITE_MASK = 0xff
SQ_BIND_HDR_ACCESS_CNTL_WINDOW_BIND_REMOTE_ATOMIC_REMOTE_WRITE_REMOTE_READ_LOCAL_WRITE_SFT = 0
SQ_BIND_HDR_ACCESS_CNTL_LOCAL_WRITE = 0x1
SQ_BIND_HDR_ACCESS_CNTL_REMOTE_READ = 0x2
SQ_BIND_HDR_ACCESS_CNTL_REMOTE_WRITE = 0x4
SQ_BIND_HDR_ACCESS_CNTL_REMOTE_ATOMIC = 0x8
SQ_BIND_HDR_ACCESS_CNTL_WINDOW_BIND = 0x10
SQ_BIND_HDR_ZERO_BASED = 0x1
SQ_BIND_HDR_MW_TYPE = 0x2
SQ_BIND_HDR_MW_TYPE_TYPE1 = (0x0 << 1)
SQ_BIND_HDR_MW_TYPE_TYPE2 = (0x1 << 1)
SQ_BIND_HDR_MW_TYPE_LAST = SQ_BIND_HDR_MW_TYPE_TYPE2
CQ_BASE_TOGGLE = 0x1
CQ_BASE_CQE_TYPE_MASK = 0x1e
CQ_BASE_CQE_TYPE_SFT = 1
CQ_BASE_CQE_TYPE_REQ = (0x0 << 1)
CQ_BASE_CQE_TYPE_RES_RC = (0x1 << 1)
CQ_BASE_CQE_TYPE_RES_UD = (0x2 << 1)
CQ_BASE_CQE_TYPE_RES_RAWETH_QP1 = (0x3 << 1)
CQ_BASE_CQE_TYPE_RES_UD_CFA = (0x4 << 1)
CQ_BASE_CQE_TYPE_REQ_V3 = (0x8 << 1)
CQ_BASE_CQE_TYPE_RES_RC_V3 = (0x9 << 1)
CQ_BASE_CQE_TYPE_RES_UD_V3 = (0xa << 1)
CQ_BASE_CQE_TYPE_RES_RAWETH_QP1_V3 = (0xb << 1)
CQ_BASE_CQE_TYPE_RES_UD_CFA_V3 = (0xc << 1)
CQ_BASE_CQE_TYPE_NO_OP = (0xd << 1)
CQ_BASE_CQE_TYPE_TERMINAL = (0xe << 1)
CQ_BASE_CQE_TYPE_CUT_OFF = (0xf << 1)
CQ_BASE_CQE_TYPE_LAST = CQ_BASE_CQE_TYPE_CUT_OFF
CQ_BASE_STATUS_OK = 0x0
CQ_BASE_STATUS_BAD_RESPONSE_ERR = 0x1
CQ_BASE_STATUS_LOCAL_LENGTH_ERR = 0x2
CQ_BASE_STATUS_HW_LOCAL_LENGTH_ERR = 0x3
CQ_BASE_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_BASE_STATUS_LOCAL_PROTECTION_ERR = 0x5
CQ_BASE_STATUS_LOCAL_ACCESS_ERROR = 0x6
CQ_BASE_STATUS_MEMORY_MGT_OPERATION_ERR = 0x7
CQ_BASE_STATUS_REMOTE_INVALID_REQUEST_ERR = 0x8
CQ_BASE_STATUS_REMOTE_ACCESS_ERR = 0x9
CQ_BASE_STATUS_REMOTE_OPERATION_ERR = 0xa
CQ_BASE_STATUS_RNR_NAK_RETRY_CNT_ERR = 0xb
CQ_BASE_STATUS_TRANSPORT_RETRY_CNT_ERR = 0xc
CQ_BASE_STATUS_WORK_REQUEST_FLUSHED_ERR = 0xd
CQ_BASE_STATUS_HW_FLUSH_ERR = 0xe
CQ_BASE_STATUS_OVERFLOW_ERR = 0xf
CQ_BASE_STATUS_LAST = CQ_BASE_STATUS_OVERFLOW_ERR
CQ_REQ_TOGGLE = 0x1
CQ_REQ_CQE_TYPE_MASK = 0x1e
CQ_REQ_CQE_TYPE_SFT = 1
CQ_REQ_CQE_TYPE_REQ = (0x0 << 1)
CQ_REQ_CQE_TYPE_LAST = CQ_REQ_CQE_TYPE_REQ
CQ_REQ_PUSH = 0x20
CQ_REQ_STATUS_OK = 0x0
CQ_REQ_STATUS_BAD_RESPONSE_ERR = 0x1
CQ_REQ_STATUS_LOCAL_LENGTH_ERR = 0x2
CQ_REQ_STATUS_LOCAL_QP_OPERATION_ERR = 0x3
CQ_REQ_STATUS_LOCAL_PROTECTION_ERR = 0x4
CQ_REQ_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_REQ_STATUS_REMOTE_INVALID_REQUEST_ERR = 0x6
CQ_REQ_STATUS_REMOTE_ACCESS_ERR = 0x7
CQ_REQ_STATUS_REMOTE_OPERATION_ERR = 0x8
CQ_REQ_STATUS_RNR_NAK_RETRY_CNT_ERR = 0x9
CQ_REQ_STATUS_TRANSPORT_RETRY_CNT_ERR = 0xa
CQ_REQ_STATUS_WORK_REQUEST_FLUSHED_ERR = 0xb
CQ_REQ_STATUS_LAST = CQ_REQ_STATUS_WORK_REQUEST_FLUSHED_ERR
CQ_RES_RC_TOGGLE = 0x1
CQ_RES_RC_CQE_TYPE_MASK = 0x1e
CQ_RES_RC_CQE_TYPE_SFT = 1
CQ_RES_RC_CQE_TYPE_RES_RC = (0x1 << 1)
CQ_RES_RC_CQE_TYPE_LAST = CQ_RES_RC_CQE_TYPE_RES_RC
CQ_RES_RC_STATUS_OK = 0x0
CQ_RES_RC_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_RC_STATUS_LOCAL_LENGTH_ERR = 0x2
CQ_RES_RC_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_RC_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_RC_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_RC_STATUS_REMOTE_INVALID_REQUEST_ERR = 0x6
CQ_RES_RC_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_RC_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_RC_STATUS_LAST = CQ_RES_RC_STATUS_HW_FLUSH_ERR
CQ_RES_RC_FLAGS_SRQ = 0x1
CQ_RES_RC_FLAGS_SRQ_RQ = 0x0
CQ_RES_RC_FLAGS_SRQ_SRQ = 0x1
CQ_RES_RC_FLAGS_SRQ_LAST = CQ_RES_RC_FLAGS_SRQ_SRQ
CQ_RES_RC_FLAGS_IMM = 0x2
CQ_RES_RC_FLAGS_INV = 0x4
CQ_RES_RC_FLAGS_RDMA = 0x8
CQ_RES_RC_FLAGS_RDMA_SEND = (0x0 << 3)
CQ_RES_RC_FLAGS_RDMA_RDMA_WRITE = (0x1 << 3)
CQ_RES_RC_FLAGS_RDMA_LAST = CQ_RES_RC_FLAGS_RDMA_RDMA_WRITE
CQ_RES_RC_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_RC_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_LENGTH_MASK = 0x3fff
CQ_RES_UD_LENGTH_SFT = 0
CQ_RES_UD_CFA_METADATA_VID_MASK = 0xfff
CQ_RES_UD_CFA_METADATA_VID_SFT = 0
CQ_RES_UD_CFA_METADATA_DE = 0x1000
CQ_RES_UD_CFA_METADATA_PRI_MASK = 0xe000
CQ_RES_UD_CFA_METADATA_PRI_SFT = 13
CQ_RES_UD_TOGGLE = 0x1
CQ_RES_UD_CQE_TYPE_MASK = 0x1e
CQ_RES_UD_CQE_TYPE_SFT = 1
CQ_RES_UD_CQE_TYPE_RES_UD = (0x2 << 1)
CQ_RES_UD_CQE_TYPE_LAST = CQ_RES_UD_CQE_TYPE_RES_UD
CQ_RES_UD_STATUS_OK = 0x0
CQ_RES_UD_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_UD_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_UD_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_UD_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_UD_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_UD_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_UD_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_UD_STATUS_LAST = CQ_RES_UD_STATUS_HW_FLUSH_ERR
CQ_RES_UD_FLAGS_SRQ = 0x1
CQ_RES_UD_FLAGS_SRQ_RQ = 0x0
CQ_RES_UD_FLAGS_SRQ_SRQ = 0x1
CQ_RES_UD_FLAGS_SRQ_LAST = CQ_RES_UD_FLAGS_SRQ_SRQ
CQ_RES_UD_FLAGS_IMM = 0x2
CQ_RES_UD_FLAGS_UNUSED_MASK = 0xc
CQ_RES_UD_FLAGS_UNUSED_SFT = 2
CQ_RES_UD_FLAGS_ROCE_IP_VER_MASK = 0x30
CQ_RES_UD_FLAGS_ROCE_IP_VER_SFT = 4
CQ_RES_UD_FLAGS_ROCE_IP_VER_V1 = (0x0 << 4)
CQ_RES_UD_FLAGS_ROCE_IP_VER_V2IPV4 = (0x2 << 4)
CQ_RES_UD_FLAGS_ROCE_IP_VER_V2IPV6 = (0x3 << 4)
CQ_RES_UD_FLAGS_ROCE_IP_VER_LAST = CQ_RES_UD_FLAGS_ROCE_IP_VER_V2IPV6
CQ_RES_UD_FLAGS_META_FORMAT_MASK = 0x3c0
CQ_RES_UD_FLAGS_META_FORMAT_SFT = 6
CQ_RES_UD_FLAGS_META_FORMAT_NONE = (0x0 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_VLAN = (0x1 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_TUNNEL_ID = (0x2 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_CHDR_DATA = (0x3 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_HDR_OFFSET = (0x4 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_LAST = CQ_RES_UD_FLAGS_META_FORMAT_HDR_OFFSET
CQ_RES_UD_FLAGS_EXT_META_FORMAT_MASK = 0xc00
CQ_RES_UD_FLAGS_EXT_META_FORMAT_SFT = 10
CQ_RES_UD_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_UD_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_SRC_QP_HIGH_MASK = 0xff000000
CQ_RES_UD_SRC_QP_HIGH_SFT = 24
CQ_RES_UD_V2_LENGTH_MASK = 0x3fff
CQ_RES_UD_V2_LENGTH_SFT = 0
CQ_RES_UD_V2_CFA_METADATA0_VID_MASK = 0xfff
CQ_RES_UD_V2_CFA_METADATA0_VID_SFT = 0
CQ_RES_UD_V2_CFA_METADATA0_DE = 0x1000
CQ_RES_UD_V2_CFA_METADATA0_PRI_MASK = 0xe000
CQ_RES_UD_V2_CFA_METADATA0_PRI_SFT = 13
CQ_RES_UD_V2_TOGGLE = 0x1
CQ_RES_UD_V2_CQE_TYPE_MASK = 0x1e
CQ_RES_UD_V2_CQE_TYPE_SFT = 1
CQ_RES_UD_V2_CQE_TYPE_RES_UD = (0x2 << 1)
CQ_RES_UD_V2_CQE_TYPE_LAST = CQ_RES_UD_V2_CQE_TYPE_RES_UD
CQ_RES_UD_V2_STATUS_OK = 0x0
CQ_RES_UD_V2_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_UD_V2_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_UD_V2_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_UD_V2_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_UD_V2_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_UD_V2_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_UD_V2_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_UD_V2_STATUS_LAST = CQ_RES_UD_V2_STATUS_HW_FLUSH_ERR
CQ_RES_UD_V2_FLAGS_SRQ = 0x1
CQ_RES_UD_V2_FLAGS_SRQ_RQ = 0x0
CQ_RES_UD_V2_FLAGS_SRQ_SRQ = 0x1
CQ_RES_UD_V2_FLAGS_SRQ_LAST = CQ_RES_UD_V2_FLAGS_SRQ_SRQ
CQ_RES_UD_V2_FLAGS_IMM = 0x2
CQ_RES_UD_V2_FLAGS_UNUSED_MASK = 0xc
CQ_RES_UD_V2_FLAGS_UNUSED_SFT = 2
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_MASK = 0x30
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_SFT = 4
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_V1 = (0x0 << 4)
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_V2IPV4 = (0x2 << 4)
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_V2IPV6 = (0x3 << 4)
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_LAST = CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_V2IPV6
CQ_RES_UD_V2_FLAGS_META_FORMAT_MASK = 0x3c0
CQ_RES_UD_V2_FLAGS_META_FORMAT_SFT = 6
CQ_RES_UD_V2_FLAGS_META_FORMAT_NONE = (0x0 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_ACT_REC_PTR = (0x1 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_TUNNEL_ID = (0x2 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_CHDR_DATA = (0x3 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_HDR_OFFSET = (0x4 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_LAST = CQ_RES_UD_V2_FLAGS_META_FORMAT_HDR_OFFSET
CQ_RES_UD_V2_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_UD_V2_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_V2_CFA_METADATA1_MASK = 0xf00000
CQ_RES_UD_V2_CFA_METADATA1_SFT = 20
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_MASK = 0x700000
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_SFT = 20
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID88A8 = (0x0 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID8100 = (0x1 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID9100 = (0x2 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID9200 = (0x3 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID9300 = (0x4 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPIDCFG = (0x5 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_LAST = CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPIDCFG
CQ_RES_UD_V2_CFA_METADATA1_VALID = 0x800000
CQ_RES_UD_V2_SRC_QP_HIGH_MASK = 0xff000000
CQ_RES_UD_V2_SRC_QP_HIGH_SFT = 24
CQ_RES_UD_CFA_LENGTH_MASK = 0x3fff
CQ_RES_UD_CFA_LENGTH_SFT = 0
CQ_RES_UD_CFA_QID_MASK = 0xfffff
CQ_RES_UD_CFA_QID_SFT = 0
CQ_RES_UD_CFA_CFA_METADATA_VID_MASK = 0xfff
CQ_RES_UD_CFA_CFA_METADATA_VID_SFT = 0
CQ_RES_UD_CFA_CFA_METADATA_DE = 0x1000
CQ_RES_UD_CFA_CFA_METADATA_PRI_MASK = 0xe000
CQ_RES_UD_CFA_CFA_METADATA_PRI_SFT = 13
CQ_RES_UD_CFA_CFA_METADATA_TPID_MASK = 0xffff0000
CQ_RES_UD_CFA_CFA_METADATA_TPID_SFT = 16
CQ_RES_UD_CFA_TOGGLE = 0x1
CQ_RES_UD_CFA_CQE_TYPE_MASK = 0x1e
CQ_RES_UD_CFA_CQE_TYPE_SFT = 1
CQ_RES_UD_CFA_CQE_TYPE_RES_UD_CFA = (0x4 << 1)
CQ_RES_UD_CFA_CQE_TYPE_LAST = CQ_RES_UD_CFA_CQE_TYPE_RES_UD_CFA
CQ_RES_UD_CFA_STATUS_OK = 0x0
CQ_RES_UD_CFA_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_UD_CFA_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_UD_CFA_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_UD_CFA_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_UD_CFA_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_UD_CFA_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_UD_CFA_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_UD_CFA_STATUS_LAST = CQ_RES_UD_CFA_STATUS_HW_FLUSH_ERR
CQ_RES_UD_CFA_FLAGS_SRQ = 0x1
CQ_RES_UD_CFA_FLAGS_SRQ_RQ = 0x0
CQ_RES_UD_CFA_FLAGS_SRQ_SRQ = 0x1
CQ_RES_UD_CFA_FLAGS_SRQ_LAST = CQ_RES_UD_CFA_FLAGS_SRQ_SRQ
CQ_RES_UD_CFA_FLAGS_IMM = 0x2
CQ_RES_UD_CFA_FLAGS_UNUSED_MASK = 0xc
CQ_RES_UD_CFA_FLAGS_UNUSED_SFT = 2
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_MASK = 0x30
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_SFT = 4
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_V1 = (0x0 << 4)
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_V2IPV4 = (0x2 << 4)
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_V2IPV6 = (0x3 << 4)
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_LAST = CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_V2IPV6
CQ_RES_UD_CFA_FLAGS_META_FORMAT_MASK = 0x3c0
CQ_RES_UD_CFA_FLAGS_META_FORMAT_SFT = 6
CQ_RES_UD_CFA_FLAGS_META_FORMAT_NONE = (0x0 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_VLAN = (0x1 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_TUNNEL_ID = (0x2 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_CHDR_DATA = (0x3 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_HDR_OFFSET = (0x4 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_LAST = CQ_RES_UD_CFA_FLAGS_META_FORMAT_HDR_OFFSET
CQ_RES_UD_CFA_FLAGS_EXT_META_FORMAT_MASK = 0xc00
CQ_RES_UD_CFA_FLAGS_EXT_META_FORMAT_SFT = 10
CQ_RES_UD_CFA_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_UD_CFA_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_CFA_SRC_QP_HIGH_MASK = 0xff000000
CQ_RES_UD_CFA_SRC_QP_HIGH_SFT = 24
CQ_RES_UD_CFA_V2_LENGTH_MASK = 0x3fff
CQ_RES_UD_CFA_V2_LENGTH_SFT = 0
CQ_RES_UD_CFA_V2_CFA_METADATA0_VID_MASK = 0xfff
CQ_RES_UD_CFA_V2_CFA_METADATA0_VID_SFT = 0
CQ_RES_UD_CFA_V2_CFA_METADATA0_DE = 0x1000
CQ_RES_UD_CFA_V2_CFA_METADATA0_PRI_MASK = 0xe000
CQ_RES_UD_CFA_V2_CFA_METADATA0_PRI_SFT = 13
CQ_RES_UD_CFA_V2_QID_MASK = 0xfffff
CQ_RES_UD_CFA_V2_QID_SFT = 0
CQ_RES_UD_CFA_V2_TOGGLE = 0x1
CQ_RES_UD_CFA_V2_CQE_TYPE_MASK = 0x1e
CQ_RES_UD_CFA_V2_CQE_TYPE_SFT = 1
CQ_RES_UD_CFA_V2_CQE_TYPE_RES_UD_CFA = (0x4 << 1)
CQ_RES_UD_CFA_V2_CQE_TYPE_LAST = CQ_RES_UD_CFA_V2_CQE_TYPE_RES_UD_CFA
CQ_RES_UD_CFA_V2_STATUS_OK = 0x0
CQ_RES_UD_CFA_V2_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_UD_CFA_V2_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_UD_CFA_V2_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_UD_CFA_V2_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_UD_CFA_V2_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_UD_CFA_V2_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_UD_CFA_V2_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_UD_CFA_V2_STATUS_LAST = CQ_RES_UD_CFA_V2_STATUS_HW_FLUSH_ERR
CQ_RES_UD_CFA_V2_FLAGS_SRQ = 0x1
CQ_RES_UD_CFA_V2_FLAGS_SRQ_RQ = 0x0
CQ_RES_UD_CFA_V2_FLAGS_SRQ_SRQ = 0x1
CQ_RES_UD_CFA_V2_FLAGS_SRQ_LAST = CQ_RES_UD_CFA_V2_FLAGS_SRQ_SRQ
CQ_RES_UD_CFA_V2_FLAGS_IMM = 0x2
CQ_RES_UD_CFA_V2_FLAGS_UNUSED_MASK = 0xc
CQ_RES_UD_CFA_V2_FLAGS_UNUSED_SFT = 2
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_MASK = 0x30
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_SFT = 4
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_V1 = (0x0 << 4)
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_V2IPV4 = (0x2 << 4)
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_V2IPV6 = (0x3 << 4)
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_LAST = CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_V2IPV6
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_MASK = 0x3c0
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_SFT = 6
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_NONE = (0x0 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_ACT_REC_PTR = (0x1 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_TUNNEL_ID = (0x2 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_CHDR_DATA = (0x3 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_HDR_OFFSET = (0x4 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_LAST = CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_HDR_OFFSET
CQ_RES_UD_CFA_V2_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_UD_CFA_V2_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_CFA_V2_CFA_METADATA1_MASK = 0xf00000
CQ_RES_UD_CFA_V2_CFA_METADATA1_SFT = 20
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_MASK = 0x700000
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_SFT = 20
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID88A8 = (0x0 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID8100 = (0x1 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID9100 = (0x2 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID9200 = (0x3 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID9300 = (0x4 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPIDCFG = (0x5 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_LAST = CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPIDCFG
CQ_RES_UD_CFA_V2_CFA_METADATA1_VALID = 0x800000
CQ_RES_UD_CFA_V2_SRC_QP_HIGH_MASK = 0xff000000
CQ_RES_UD_CFA_V2_SRC_QP_HIGH_SFT = 24
CQ_RES_RAWETH_QP1_LENGTH_MASK = 0x3fff
CQ_RES_RAWETH_QP1_LENGTH_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_MASK = 0x3ff
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ERROR = 0x1
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_MASK = 0x3c0
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_SFT = 6
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_NOT_KNOWN = (0x0 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_IP = (0x1 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_TCP = (0x2 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_UDP = (0x3 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_FCOE = (0x4 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_ROCE = (0x5 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_ICMP = (0x7 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_PTP_WO_TIMESTAMP = (0x8 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_PTP_W_TIMESTAMP = (0x9 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_LAST = CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_PTP_W_TIMESTAMP
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_IP_CS_ERROR = 0x10
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_L4_CS_ERROR = 0x20
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_IP_CS_ERROR = 0x40
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_L4_CS_ERROR = 0x80
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_CRC_ERROR = 0x100
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_MASK = 0xe00
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_SFT = 9
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_NO_ERROR = (0x0 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_VERSION = (0x1 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_HDR_LEN = (0x2 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_TUNNEL_TOTAL_ERROR = (0x3 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_IP_TOTAL_ERROR = (0x4 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_UDP_TOTAL_ERROR = (0x5 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_TTL = (0x6 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_LAST = CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_TTL
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_MASK = 0xf000
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_SFT = 12
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_NO_ERROR = (0x0 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_VERSION = (0x1 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_HDR_LEN = (0x2 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_TTL = (0x3 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_IP_TOTAL_ERROR = (0x4 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_UDP_TOTAL_ERROR = (0x5 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_HDR_LEN = (0x6 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_HDR_LEN_TOO_SMALL = (0x7 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_OPT_LEN = (0x8 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_LAST = CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_OPT_LEN
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_IP_CS_CALC = 0x1
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_L4_CS_CALC = 0x2
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_T_IP_CS_CALC = 0x4
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_T_L4_CS_CALC = 0x8
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_MASK = 0xf0
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_SFT = 4
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_NONE = (0x0 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_VLAN = (0x1 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_TUNNEL_ID = (0x2 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_CHDR_DATA = (0x3 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_HDR_OFFSET = (0x4 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_LAST = CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_HDR_OFFSET
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_IP_TYPE = 0x100
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_CALC = 0x200
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_EXT_META_FORMAT_MASK = 0xc00
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_EXT_META_FORMAT_SFT = 10
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_MASK = 0xffff0000
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_SFT = 16
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_PRI_DE_VID_MASK = 0xffff
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_PRI_DE_VID_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_VID_MASK = 0xfff
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_VID_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_DE = 0x1000
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_PRI_MASK = 0xe000
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_PRI_SFT = 13
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_TPID_MASK = 0xffff0000
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_TPID_SFT = 16
CQ_RES_RAWETH_QP1_TOGGLE = 0x1
CQ_RES_RAWETH_QP1_CQE_TYPE_MASK = 0x1e
CQ_RES_RAWETH_QP1_CQE_TYPE_SFT = 1
CQ_RES_RAWETH_QP1_CQE_TYPE_RES_RAWETH_QP1 = (0x3 << 1)
CQ_RES_RAWETH_QP1_CQE_TYPE_LAST = CQ_RES_RAWETH_QP1_CQE_TYPE_RES_RAWETH_QP1
CQ_RES_RAWETH_QP1_STATUS_OK = 0x0
CQ_RES_RAWETH_QP1_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_RAWETH_QP1_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_RAWETH_QP1_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_RAWETH_QP1_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_RAWETH_QP1_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_RAWETH_QP1_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_RAWETH_QP1_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_RAWETH_QP1_STATUS_LAST = CQ_RES_RAWETH_QP1_STATUS_HW_FLUSH_ERR
CQ_RES_RAWETH_QP1_FLAGS_SRQ = 0x1
CQ_RES_RAWETH_QP1_FLAGS_SRQ_RQ = 0x0
CQ_RES_RAWETH_QP1_FLAGS_SRQ_SRQ = 0x1
CQ_RES_RAWETH_QP1_FLAGS_SRQ_LAST = CQ_RES_RAWETH_QP1_FLAGS_SRQ_SRQ
CQ_RES_RAWETH_QP1_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_RAWETH_QP1_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_PAYLOAD_OFFSET_MASK = 0xff000000
CQ_RES_RAWETH_QP1_RAWETH_QP1_PAYLOAD_OFFSET_SFT = 24
CQ_RES_RAWETH_QP1_V2_LENGTH_MASK = 0x3fff
CQ_RES_RAWETH_QP1_V2_LENGTH_SFT = 0
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_MASK = 0x3ff
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_SFT = 0
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ERROR = 0x1
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_MASK = 0x3c0
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_SFT = 6
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_NOT_KNOWN = (0x0 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_IP = (0x1 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_TCP = (0x2 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_UDP = (0x3 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_FCOE = (0x4 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_ROCE = (0x5 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_ICMP = (0x7 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_PTP_WO_TIMESTAMP = (0x8 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_PTP_W_TIMESTAMP = (0x9 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_LAST = CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_PTP_W_TIMESTAMP
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_IP_CS_ERROR = 0x10
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_L4_CS_ERROR = 0x20
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_IP_CS_ERROR = 0x40
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_L4_CS_ERROR = 0x80
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_CRC_ERROR = 0x100
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_MASK = 0xe00
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_SFT = 9
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_NO_ERROR = (0x0 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_VERSION = (0x1 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_HDR_LEN = (0x2 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_TUNNEL_TOTAL_ERROR = (0x3 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_IP_TOTAL_ERROR = (0x4 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_UDP_TOTAL_ERROR = (0x5 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_TTL = (0x6 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_LAST = CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_TTL
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_MASK = 0xf000
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_SFT = 12
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_NO_ERROR = (0x0 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_VERSION = (0x1 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_HDR_LEN = (0x2 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_TTL = (0x3 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_IP_TOTAL_ERROR = (0x4 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_UDP_TOTAL_ERROR = (0x5 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_HDR_LEN = (0x6 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_HDR_LEN_TOO_SMALL = (0x7 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_OPT_LEN = (0x8 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_LAST = CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_OPT_LEN
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_VID_MASK = 0xfff
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_VID_SFT = 0
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_DE = 0x1000
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_PRI_MASK = 0xe000
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_PRI_SFT = 13
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_CS_ALL_OK_MODE = 0x8
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_MASK = 0xf0
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_SFT = 4
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_NONE = (0x0 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_ACT_REC_PTR = (0x1 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_TUNNEL_ID = (0x2 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_CHDR_DATA = (0x3 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_HDR_OFFSET = (0x4 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_LAST = CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_HDR_OFFSET
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_IP_TYPE = 0x100
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_CALC = 0x200
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_CS_OK_MASK = 0xfc00
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_CS_OK_SFT = 10
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_MASK = 0xffff0000
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_SFT = 16
CQ_RES_RAWETH_QP1_V2_TOGGLE = 0x1
CQ_RES_RAWETH_QP1_V2_CQE_TYPE_MASK = 0x1e
CQ_RES_RAWETH_QP1_V2_CQE_TYPE_SFT = 1
CQ_RES_RAWETH_QP1_V2_CQE_TYPE_RES_RAWETH_QP1 = (0x3 << 1)
CQ_RES_RAWETH_QP1_V2_CQE_TYPE_LAST = CQ_RES_RAWETH_QP1_V2_CQE_TYPE_RES_RAWETH_QP1
CQ_RES_RAWETH_QP1_V2_STATUS_OK = 0x0
CQ_RES_RAWETH_QP1_V2_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_RAWETH_QP1_V2_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_RAWETH_QP1_V2_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_RAWETH_QP1_V2_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_RAWETH_QP1_V2_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_RAWETH_QP1_V2_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_RAWETH_QP1_V2_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_RAWETH_QP1_V2_STATUS_LAST = CQ_RES_RAWETH_QP1_V2_STATUS_HW_FLUSH_ERR
CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ = 0x1
CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ_RQ = 0x0
CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ_SRQ = 0x1
CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ_LAST = CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ_SRQ
CQ_RES_RAWETH_QP1_V2_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_RAWETH_QP1_V2_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_MASK = 0xf00000
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_SFT = 20
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_MASK = 0x700000
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_SFT = 20
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID88A8 = (0x0 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID8100 = (0x1 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID9100 = (0x2 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID9200 = (0x3 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID9300 = (0x4 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPIDCFG = (0x5 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_LAST = CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPIDCFG
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_VALID = 0x800000
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_PAYLOAD_OFFSET_MASK = 0xff000000
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_PAYLOAD_OFFSET_SFT = 24
CQ_TERMINAL_TOGGLE = 0x1
CQ_TERMINAL_CQE_TYPE_MASK = 0x1e
CQ_TERMINAL_CQE_TYPE_SFT = 1
CQ_TERMINAL_CQE_TYPE_TERMINAL = (0xe << 1)
CQ_TERMINAL_CQE_TYPE_LAST = CQ_TERMINAL_CQE_TYPE_TERMINAL
CQ_TERMINAL_STATUS_OK = 0x0
CQ_TERMINAL_STATUS_LAST = CQ_TERMINAL_STATUS_OK
CQ_CUTOFF_TOGGLE = 0x1
CQ_CUTOFF_CQE_TYPE_MASK = 0x1e
CQ_CUTOFF_CQE_TYPE_SFT = 1
CQ_CUTOFF_CQE_TYPE_CUT_OFF = (0xf << 1)
CQ_CUTOFF_CQE_TYPE_LAST = CQ_CUTOFF_CQE_TYPE_CUT_OFF
CQ_CUTOFF_RESIZE_TOGGLE_MASK = 0x60
CQ_CUTOFF_RESIZE_TOGGLE_SFT = 5
CQ_CUTOFF_STATUS_OK = 0x0
CQ_CUTOFF_STATUS_LAST = CQ_CUTOFF_STATUS_OK
PTU_PTE_VALID = 0x1
PTU_PTE_LAST = 0x2
PTU_PTE_NEXT_TO_LAST = 0x4
PTU_PTE_UNUSED_MASK = 0xff8
PTU_PTE_UNUSED_SFT = 3
PTU_PTE_PAGE_MASK = 0xfffff000
PTU_PTE_PAGE_SFT = 12
PTU_PDE_VALID = 0x1
PTU_PDE_UNUSED_MASK = 0xffe
PTU_PDE_UNUSED_SFT = 1
PTU_PDE_PAGE_MASK = 0xfffff000
PTU_PDE_PAGE_SFT = 12
RCFW_CMDQ_TRIG_VAL = 1
RCFW_COMM_PCI_BAR_REGION = 0
RCFW_COMM_CONS_PCI_BAR_REGION = 2
RCFW_COMM_BASE_OFFSET = 0x600
RCFW_PF_VF_COMM_PROD_OFFSET = 0xc
RCFW_COMM_TRIG_OFFSET = 0x100
RCFW_COMM_SIZE = 0x104
RCFW_DBR_PCI_BAR_REGION = 2
RCFW_DBR_BASE_PAGE_SHIFT = 12
RCFW_FW_STALL_MAX_TIMEOUT = 40
RCFW_CMD_NON_BLOCKING_SHADOW_QD = 64
RCFW_CMD_WAIT_TIME_MS = 20000
BNXT_QPLIB_CMDQE_MAX_CNT = 8192
BNXT_QPLIB_CMDQE_BYTES = lambda depth: ((depth) * BNXT_QPLIB_CMDQE_UNITS) # type: ignore
RCFW_MAX_COOKIE_VALUE = (BNXT_QPLIB_CMDQE_MAX_CNT - 1)
RCFW_CMD_IS_BLOCKING = 0x8000
HWRM_VERSION_DEV_ATTR_MAX_DPI = 0x1000A0000000D
HWRM_VERSION_READ_CTX = 0x1000A00030012
BNXT_QPLIB_CREQE_MAX_CNT = (64 * 1024)
BNXT_QPLIB_CREQE_UNITS = 16
CREQ_ENTRY_POLL_BUDGET = 0x100
BNXT_QPLIB_OOS_COUNT_MASK = 0xFFFFFFFF
FIRMWARE_FIRST_FLAG = (31)
BNXT_RE_MAX_QPC_COUNT = (64 * 1024)
BNXT_RE_MAX_MRW_COUNT = (64 * 1024)
BNXT_RE_MAX_SRQC_COUNT = (64 * 1024)
BNXT_RE_MAX_CQ_COUNT = (64 * 1024)
BNXT_RE_MAX_MRW_COUNT_64K = (64 * 1024)
BNXT_RE_MAX_MRW_COUNT_256K = (256 * 1024)
BNXT_QPLIB_DBR_VALID = (0x1 << 26)
BNXT_QPLIB_DBR_EPOCH_SHIFT = 24
BNXT_QPLIB_DBR_TOGGLE_SHIFT = 25
BNXT_QPLIB_DBR_PF_DB_OFFSET = 0x10000
BNXT_QPLIB_DBR_VF_DB_OFFSET = 0x4000
BNXT_QPLIB_MAX_QP_CTX_ENTRY_SIZE = 448
BNXT_QPLIB_MAX_SRQ_CTX_ENTRY_SIZE = 64
BNXT_QPLIB_MAX_CQ_CTX_ENTRY_SIZE = 64
BNXT_QPLIB_MAX_MRW_CTX_ENTRY_SIZE = 128
BNXT_QPLIB_INIT_DBHDR = lambda xid,type,indx,toggle: (((u64)(((xid) & DBC_DBC_XID_MASK) | DBC_DBC_PATH_ROCE | (type) | BNXT_QPLIB_DBR_VALID) << 32) | (indx) | (((u32)(toggle)) << (BNXT_QPLIB_DBR_TOGGLE_SHIFT))) # type: ignore
BNXT_RE_HW_RETX = lambda a: _is_hw_retx_supported((a)) # type: ignore
HWRM_CMD_MAX_TIMEOUT = 60000
BNXT_HWRM_TARGET = 0xffff
BNXT_HWRM_NO_CMPL_RING = -1
BNXT_HWRM_REQ_MAX_SIZE = 128
BNXT_HWRM_DMA_ALIGN = 16
BNXT_HWRM_SENTINEL = 0xb6e1f68a12e9a7eb
HWRM_SHORT_MIN_TIMEOUT = 3
HWRM_SHORT_MAX_TIMEOUT = 10
HWRM_SHORT_TIMEOUT_COUNTER = 5
HWRM_MIN_TIMEOUT = 25
HWRM_MAX_TIMEOUT = 40
HWRM_VALID_BIT_DELAY_USEC = 50000