from __future__ import annotations
import struct, random, socket, ctypes, functools, itertools
from tinygrad.helpers import getenv, wait_cond, round_up, next_power2, ceildiv, DEBUG, hi32, lo32, to_be32, to_be64
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.autogen import mlx5, pci

MLX_DEBUG = getenv("MLX_DEBUG", 0)

MLX5_CMD_STRUCTS = {v: (getattr(mlx5, f"struct_mlx5_ifc_{n[12:].lower()}_in_bits", None),
  getattr(mlx5, f"struct_mlx5_ifc_{n[12:].lower()}_out_bits", None)) for n, v in mlx5.__dict__.items() if n.startswith("MLX5_CMD_OP_")}
MLX5_CMD_STRUCTS[mlx5.MLX5_CMD_OP_ACCESS_REG] = (mlx5.struct_mlx5_ifc_access_register_in_bits, mlx5.struct_mlx5_ifc_access_register_out_bits)

def ipv4_to_gid(ip): return bytes(10) + b'\xff\xff' + socket.inet_aton(ip)

def udp_sport(lqpn, rqpn):
  v = (lqpn * rqpn ^ ((lqpn * rqpn) >> 20) ^ ((lqpn * rqpn) >> 40)) & 0xFFFFF
  return ((v & 0x3FFF) ^ ((v & 0xFC000) >> 14)) | 0xC000

def ifc_get(buf, bit_off, width):
  byte_off, bit_in, n = bit_off // 8, bit_off % 8, (bit_off % 8 + width + 7) // 8
  return (int.from_bytes(buf[byte_off:byte_off + n], 'big') >> (n * 8 - bit_in - width)) & ((1 << width) - 1)

def ifc_set(buf, bit_off, width, value):
  byte_off, bit_in, n = bit_off // 8, bit_off % 8, (bit_off % 8 + width + 7) // 8
  shift, val = n * 8 - bit_in - width, int.from_bytes(buf[byte_off:byte_off + n], 'big')
  buf[byte_off:byte_off + n] = ((val & ~(((1 << width) - 1) << shift)) | ((value & ((1 << width) - 1)) << shift)).to_bytes(n, 'big')

@functools.cache
def ifc_fields(ifc_struct): return {name: (off, ctypes.sizeof(typ)) for name, typ, off in ifc_struct._real_fields_ if not name.startswith('reserved')}

def ifc_subfield(ifc_struct, field_name):
  for name, typ, off in ifc_struct._real_fields_:
    if name == field_name: return typ, ifc_fields(ifc_struct)[field_name][0]
  raise KeyError(f"no field '{field_name}' in {ifc_struct}")

def fill_ifc(buf, ifc_struct, base=0, **kw):
  fields = ifc_fields(ifc_struct)
  for name, val in kw.items():
    if isinstance(val, dict):
      sub_struct, sub_off = ifc_subfield(ifc_struct, name)
      fill_ifc(buf, sub_struct, base=base + sub_off, **val)
    else: ifc_set(buf, base + fields[name][0], fields[name][1], val)

def ifc_decode(buf, ifc_struct, base=0):return {name: ifc_get(buf, base + off, width) for name, (off, width) in ifc_fields(ifc_struct).items()}

class MLXCmdQueue:
  def __init__(self, dev):
    self.dev, self._tok = dev, itertools.count(1)

    cmd_l = dev.iseg_r('cmdq_addr_l_sz') & 0xFF
    self.log_stride, self.max_reg_cmds = cmd_l & 0xF, (1 << ((cmd_l >> 4) & 0xF)) - 1

    stride = next_power2(ctypes.sizeof(mlx5.struct_mlx5_cmd_prot_block))
    self.queue, self.queue_paddrs = dev.pci_dev.alloc_sysmem(0x1000 + 1024 * stride)
    self.mboxes = [(off:=0x1000 + i * stride, self.queue_paddrs[1 + (i * stride) // 0x1000] + (off % 0x1000)) for i in range(1024)]

    dev.iseg_w('cmdq_addr_h', hi32(self.queue_paddrs[0]))
    dev.iseg_w('cmdq_addr_l_sz', lo32(self.queue_paddrs[0]) | cmd_l)

  def create_mbox_chain(self, base, tok, data):
    n = ceildiv(len(data), chunk_sz:=mlx5.MLX5_CMD_DATA_BLOCK_SIZE)
    for i in range(n):
      off, _ = self.mboxes[base + i]
      blk = mlx5.struct_mlx5_cmd_prot_block(data=(ctypes.c_ubyte*chunk_sz).from_buffer_copy(data[i*chunk_sz:(i+1)*chunk_sz].ljust(chunk_sz, b'\x00')),
        next=to_be64(self.mboxes[base+i+1][1]) if i < n-1 else 0, block_num=to_be32(i), token=tok)
      self.queue[off:off + ctypes.sizeof(mlx5.struct_mlx5_cmd_prot_block)] = bytes(blk)
    return (self.mboxes[base][0], self.mboxes[base][1], n)

  def exec(self, opcode, op_mod=0, payload=b'', raw=False, **kw):
    in_struct, out_struct = MLX5_CMD_STRUCTS[opcode]
    out_sz = max(0, ctypes.sizeof(out_struct) - 16) if out_struct else 0
    tok, slot = (next(self._tok) % 255) + 1, self.max_reg_cmds if opcode == mlx5.MLX5_CMD_OP_MANAGE_PAGES else 0

    # serialize input
    inp_sz = max(16, ceildiv(max((off + w for off, w in ifc_fields(in_struct).values()), default=0), 8))
    fill_ifc(inp:=bytearray(inp_sz + len(payload)), in_struct, opcode=opcode, op_mod=op_mod, **kw)
    if payload: inp[inp_sz:] = payload

    # prepare mailboxes and build command layout
    _, in_ptr, n_in = self.create_mbox_chain(0, tok, inp[16:])
    _, out_ptr, n_out = self.create_mbox_chain(n_in, tok, bytes(out_sz))
    cmd = mlx5.struct_mlx5_cmd_layout(type=mlx5.MLX5_PCI_CMD_XPORT, inlen=to_be32(len(inp)), in_ptr=to_be64(in_ptr),
      _in=(ctypes.c_uint32*4)(*(int.from_bytes(inp[i:i+4], 'little') for i in range(0, 16, 4))),
      out_ptr=to_be64(out_ptr), outlen=to_be32(16 + out_sz), token=tok, status_own=mlx5.CMD_OWNER_HW)
    cmd_bytes = bytearray(bytes(cmd))
    cmd_bytes[mlx5.struct_mlx5_cmd_layout.sig.offset] = (~functools.reduce(lambda a, b: a ^ b, cmd_bytes)) & 0xFF  # type: ignore[attr-defined]

    # submit and wait for completion
    slot_view = self.queue.view(slot << self.log_stride, len(cmd_bytes))
    slot_view[:] = cmd_bytes
    self.dev.iseg_w('cmd_dbell', 1 << slot)
    wait_cond(lambda: slot_view[mlx5.struct_mlx5_cmd_layout.status_own.offset] & mlx5.CMD_OWNER_HW, value=0,  # type: ignore[attr-defined]
              msg=f"cmd 0x{opcode:04x}")

    # check status and read output
    assert slot_view[mlx5.struct_mlx5_cmd_layout.status_own.offset] >> 1 == 0, f"cmd 0x{opcode:04x} delivery error"  # type: ignore[attr-defined]

    out_view = slot_view.view(mlx5.struct_mlx5_cmd_layout.out.offset, 16 + out_sz)  # type: ignore[attr-defined]
    status, syndrome = struct.unpack('>I', out_view[0:4])[0] >> 24, struct.unpack('>I', out_view[4:8])[0]
    assert status == 0, f"cmd 0x{opcode:04x} failed status=0x{status:x} syn=0x{syndrome:08x}"

    ret = bytearray(out_view[:16])
    ret += b''.join(bytes(self.queue[va:va + mlx5.MLX5_CMD_DATA_BLOCK_SIZE]) for va, _ in self.mboxes[n_in:n_in+n_out])[:out_sz]
    return ret if raw else (ifc_decode(ret, out_struct) if out_struct else ret)

class MLXDev:
  def __init__(self, pci_dev:PCIDevice, ip:str=getenv("MLX_IP", "10.0.0.1")):
    self.pci_dev, self.devfmt, self.bar = pci_dev, pci_dev.pcibus, pci_dev.map_bar(0, fmt='I')

    fw_rev, cmdif_sub = self.iseg_r('fw_rev'), self.iseg_r('cmdif_rev_fw_sub')
    if DEBUG >= 2: print(f"mlx5 {self.devfmt}: firmware {fw_rev >> 16}.{fw_rev & 0xFFFF}.{cmdif_sub & 0xFFFF}")
    assert (cmdif_sub >> 16) == 5, f"unsupported mlx version: {cmdif_sub >> 16}"

    self.init_hw(ip)

  def rreg(self, off): return to_be32(self.bar[off // 4])
  def wreg(self, off, val): self.bar[off // 4] = to_be32(val)
  def iseg_r(self, field): return self.rreg(getattr(mlx5.struct_mlx5_init_seg, field).offset)
  def iseg_w(self, field, val): self.wreg(getattr(mlx5.struct_mlx5_init_seg, field).offset, val)

  def init_hw(self, ip):
    wait_cond(lambda: self.iseg_r('initializing') & 0x80000000, value=0, msg="FW init timeout")
    self.pci_dev.write_config(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.cmd = MLXCmdQueue(self)
    wait_cond(lambda: self.iseg_r('initializing') & 0x80000000, value=0, msg="FW init timeout")

    self.cmd.exec(mlx5.MLX5_CMD_OP_ENABLE_HCA)
    if self.cmd.exec(mlx5.MLX5_CMD_OP_QUERY_ISSI)['supported_issi_dw0'] & 2:
      self.cmd.exec(mlx5.MLX5_CMD_OP_SET_ISSI, current_issi=1)

    self.provide_pages(mlx5.MLX5_BOOT_PAGES)
    self.cmd.exec(mlx5.MLX5_CMD_OP_ACCESS_REG, register_id=mlx5.MLX5_REG_HOST_ENDIANNESS, payload=bytearray(16))

    self.init_hca()

    self.uar = self.cmd.exec(mlx5.MLX5_CMD_OP_ALLOC_UAR)['uar']
    self.uar_view = self.pci_dev.map_bar(0, off=self.uar * 0x1000, size=0x1000, fmt='Q')

    vport = self.cmd.exec(mlx5.MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT, raw=True)
    nvc_struct, nvc_off = ifc_subfield(mlx5.struct_mlx5_ifc_query_nic_vport_context_out_bits, 'nic_vport_context')
    mac_struct, mac_off = ifc_subfield(nvc_struct, 'permanent_address')
    self.mac = ifc_get(vport, nvc_off + mac_off + 16, 48)

    # enable roce
    self.cmd.exec(mlx5.MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT, field_select=dict(roce_en=1), nic_vport_context=dict(roce_en=1))

    dbr_mem, self.dbr_paddrs = self.pci_dev.alloc_sysmem(0x1000)
    self.dbr = dbr_mem.view(fmt='I')
    self.dbr_alloc = BumpAllocator(0x1000, wrap=False)

    self.pd = self.cmd.exec(mlx5.MLX5_CMD_OP_ALLOC_PD)['pd']
    res = self.cmd.exec(mlx5.MLX5_CMD_OP_CREATE_MKEY, memory_key_mkey_entry=dict(access_mode_1_0=0, rw=1, rr=1, lw=1, lr=1, qpn=0xFFFFFF,
                                                                                 mkey_7_0=(key_lo:=0x22), length64=1, pd=self.pd))
    self.mkey = (res['mkey_index'] << 8) | key_lo

    self.local_gid = ipv4_to_gid(ip)
    self.cmd.exec(mlx5.MLX5_CMD_OP_SET_ROCE_ADDRESS, roce_address=dict(roce_version=2, source_l3_address=int.from_bytes(self.local_gid, 'big'),
                  roce_l3_type=0, source_mac_47_32=hi32(self.mac), source_mac_31_0=lo32(self.mac)), roce_address_index=0, vhca_port_num=1)

    if DEBUG >= 2: print(f"mlx5 {self.devfmt}: booted mac={self.mac.to_bytes(6,'big').hex(':')} mkey=0x{self.mkey:x}")

  def register_mem(self, paddrs:list[int], size:int, log_page_size:int=12) -> int:
    n = len(paddrs)
    mtt = struct.pack(f'>{round_up(n, 2)}Q', *paddrs, *([0] * (round_up(n, 2) - n)))
    if MLX_DEBUG >= 1: print(f"mlx5 {self.devfmt}: register_mem pages={n} page_sz={1 << log_page_size} mtt_bytes={len(mtt)}")
    self.provide_pages(mlx5.MLX5_INIT_PAGES)
    res = self.cmd.exec(mlx5.MLX5_CMD_OP_CREATE_MKEY, translations_octword_actual_size=ceildiv(n, 2), payload=mtt,
      memory_key_mkey_entry=dict(access_mode_1_0=1, lr=1, lw=1, rr=1, rw=1, pd=self.pd, qpn=0xFFFFFF, mkey_7_0=(key_lo:=0x33),
                                 start_addr=paddrs[0], len=size, log_page_size=log_page_size, translations_octword_size=ceildiv(n, 2)))
    return (res['mkey_index'] << 8) | key_lo

  def unregister_mem(self, mkey:int): self.cmd.exec(mlx5.MLX5_CMD_OP_DESTROY_MKEY, mkey_index=mkey >> 8)

  def provide_pages(self, mode):
    if (npages:=self.cmd.exec(mlx5.MLX5_CMD_OP_QUERY_PAGES, op_mod=mode)['num_pages']) <= 0: return
    if MLX_DEBUG >= 1: print(f"mlx5 {self.devfmt}: provide_pages mode={mode}, {npages} pages")
    mem, paddrs = self.pci_dev.alloc_sysmem(npages * 0x1000)
    self.cmd.exec(mlx5.MLX5_CMD_OP_MANAGE_PAGES, op_mod=mlx5.MLX5_PAGES_GIVE, input_num_entries=npages, payload=struct.pack(f'>{npages}Q', *paddrs))

  def hca_query_cap(self, cap_type, cap_struct, mode):
    raw = bytearray(self.cmd.exec(mlx5.MLX5_CMD_OP_QUERY_HCA_CAP, op_mod=(cap_type << 1) | mode, raw=True)[16:16+4096])
    return raw, ifc_decode(raw, cap_struct)

  def hca_set_cap(self, cap_type, cap_struct, raw, **kwargs):
    fill_ifc(cap:=bytearray(raw), cap_struct, **kwargs)
    self.cmd.exec(mlx5.MLX5_CMD_OP_SET_HCA_CAP, op_mod=cap_type << 1, capability=int.from_bytes(cap[:4096].ljust(4096, b'\x00'), 'big'))

  def init_hca(self):
    gen_caps, gen_cur = self.hca_query_cap(mlx5.MLX5_CAP_GENERAL, mlx5.struct_mlx5_ifc_cmd_hca_cap_bits, mode=1)
    self.hca_set_cap(mlx5.MLX5_CAP_GENERAL, mlx5.struct_mlx5_ifc_cmd_hca_cap_bits, gen_caps,
      pkey_table_size=0, cmdif_checksum=0, log_uar_page_sz=0, log_max_qp=18, roce=1)

    roce_cur_raw, roce_cur = self.hca_query_cap(mlx5.MLX5_CAP_ROCE, mlx5.struct_mlx5_ifc_roce_cap_bits, mode=1)
    self.hca_set_cap(mlx5.MLX5_CAP_ROCE, mlx5.struct_mlx5_ifc_roce_cap_bits, roce_cur_raw, sw_r_roce_src_udp_port=1)

    self.provide_pages(mlx5.MLX5_INIT_PAGES)
    self.cmd.exec(mlx5.MLX5_CMD_OP_INIT_HCA, sw_owner_id=random.getrandbits(128))

    _, self.caps = self.hca_query_cap(mlx5.MLX5_CAP_GENERAL, mlx5.struct_mlx5_ifc_cmd_hca_cap_bits, 1)

    if MLX_DEBUG >= 4: print(f"mlx5 {self.devfmt}: HCA initialized with gen_caps={gen_cur} roce_caps={roce_cur}")

class MLXQP:
  def __init__(self, dev:MLXDev, log_sq_size=4, log_rq_size=4, log_eq_size=7, log_cq_size=7):
    self.dev, self.cq_size, self.log_sq_size, self.log_rq_size, self.head = dev, 1 << log_cq_size, log_sq_size, log_rq_size, 0

    self.cq_dbr, self.qp_dbr = dev.dbr_alloc.alloc(8, alignment=8), dev.dbr_alloc.alloc(8, alignment=8)

    # create EQ, CQ
    self.eq_mem, self.eq_paddrs, self.eq_info = self.create_queue(mlx5.MLX5_CMD_OP_CREATE_EQ, log_eq_size, entry_sz=64, owner_off=31,
      eq_context_entry=dict(log_eq_size=log_eq_size, uar_page=dev.uar, log_page_size=0))

    self.cq_mem, self.cq_paddrs, self.cq_info = self.create_queue(mlx5.MLX5_CMD_OP_CREATE_CQ, log_cq_size, entry_sz=64, owner_off=63,
      cq_context=dict(log_cq_size=log_cq_size, uar_page=dev.uar, c_eqn_or_apu_element=self.eq_info['eq_number'],
                      dbr_addr=dev.dbr_paddrs[0] + self.cq_dbr, log_page_size=0))

    # create QP, buffer is RQ (16B stride) + SQ (64B stride)
    self.sq_offset = (1 << log_rq_size) << 4
    self.qp_buf, self.qp_paddrs, self.qp_info = self.create_queue(mlx5.MLX5_CMD_OP_CREATE_QP, log_sq_size, entry_sz=64,
      owner_off=0, extra_sz=self.sq_offset,
      qpc=dict(st=0, pm_state=3, pd=dev.pd, cqn_snd=self.cq_info['cqn'], cqn_rcv=self.cq_info['cqn'], log_msg_max=30, log_rq_size=log_rq_size,
               log_rq_stride=0, log_sq_size=log_sq_size, rlky=1, uar_page=dev.uar, log_page_size=0, dbr_addr=dev.dbr_paddrs[0] + self.qp_dbr))

    # transition to INIT
    self.qp_op(mlx5.MLX5_CMD_OP_RST2INIT_QP, qpc_args=dict(log_ack_req_freq=8), addr_args=dict(pkey_index=0, vhca_port_num=1))

    for i in range(self.cq_size): self.cq_mem[i * 64 + 63] = 0x01  # init owner bits so poll_cq waits for real CQEs
    if MLX_DEBUG >= 1: print(f"mlx5: QP 0x{self.qp_info['qpn']:x} (EQ={self.eq_info['eq_number']} CQ=0x{self.cq_info['cqn']:x})")

  def create_queue(self, opcode, log_size, entry_sz, owner_off, extra_sz=0, **ctx_kw):
    mem, paddrs = self.dev.pci_dev.alloc_sysmem((n := ceildiv((1 << log_size) * entry_sz + extra_sz, 0x1000)) * 0x1000)
    return mem, paddrs, self.dev.cmd.exec(opcode, payload=struct.pack(f'>{n}Q', *paddrs), **ctx_kw)

  def qp_op(self, opcode, qpc_args=None, addr_args=None, **kwargs):
    qpc_args = dict(st=0, pm_state=3, pd=self.dev.pd, cqn_snd=self.cq_info['cqn'], cqn_rcv=self.cq_info['cqn'], **(qpc_args or {}))
    self.dev.cmd.exec(opcode, qpn=self.qp_info['qpn'], qpc=(qpc_args or {}) | {'primary_address_path': addr_args or {}}, **kwargs)

  def connect(self, remote:MLXQP):
    self.qp_op(mlx5.MLX5_CMD_OP_INIT2RTR_QP, opt_param_mask=0x1A,
      qpc_args=dict(mtu=5, log_msg_max=self.dev.caps['log_max_msg'], remote_qpn=remote.qp_info['qpn'], log_ack_req_freq=8,
                    log_rra_max=3, rre=1, rwe=1, min_rnr_nak=1, next_rcv_psn=0),
      addr_args=dict(pkey_index=0, src_addr_index=0, hop_limit=64, udp_sport=udp_sport(self.qp_info['qpn'], remote.qp_info['qpn']), vhca_port_num=1,
                     rmac_47_32=hi32(remote.dev.mac), rmac_31_0=lo32(remote.dev.mac), rgid_rip=int.from_bytes(remote.dev.local_gid, 'big')))
    self.qp_op(mlx5.MLX5_CMD_OP_RTR2RTS_QP, qpc_args=dict(log_ack_req_freq=8, next_send_psn=0, log_sra_max=3, retry_count=7, rnr_retry=7),
      addr_args=dict(ack_timeout=14, vhca_port_num=1))

    if MLX_DEBUG >= 1: print(f"mlx5: QP 0x{self.qp_info['qpn']:x} connected (remote=0x{remote.qp_info['qpn']:x})")
