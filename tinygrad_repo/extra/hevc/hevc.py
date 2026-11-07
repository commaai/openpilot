import dataclasses, enum, argparse, os, itertools, time, ctypes
from typing import Any
from tinygrad import Tensor, dtypes, Device, TinyJit
from tinygrad.helpers import DEBUG, round_up, ceildiv, Timing, prod
from tinygrad.runtime.autogen import avcodec, nv_570 as nv_gpu

class BitReader:
  def __init__(self, data:bytes): self.reader, self.current_bits, self.bits, self.read_bits, self.total = iter(data), 0, 0, 0, len(data) * 8
  def empty(self): return self.read_bits == self.total and self.current_bits == 0
  def peak_bits(self, n):
    while self.current_bits < n:
      self.bits = (self.bits << 8) | next(self.reader)
      self.current_bits += 8
      self.read_bits += 8
    return (self.bits >> (self.current_bits - n)) & ((1 << n) - 1)
  def _next_bits(self, n):
    val = self.peak_bits(n)
    self.bits &= (1 << (self.current_bits - n)) - 1
    self.current_bits -= n
    return val

  def u(self, n): return self._next_bits(n)

  # 9.2 Parsing process for 0-th order Exp-Golomb codes
  def ue_v(self):
    leading_zero_bits = -1
    while True:
      bit = self.u(1)
      leading_zero_bits += 1
      if bit == 1: break

    part = self.u(leading_zero_bits)

    if leading_zero_bits == 0: return 0
    return (1 << leading_zero_bits) - 1 + part

  # 9.2.2 Mapping process for signed Exp-Golomb codes
  def se_v(self):
    k = self.ue_v()
    return (-1 ** (k + 1)) * (k // 2)

# 7.3.1.1 General NAL unit syntax
def _hevc_get_rbsp(dat:bytes, off=0) -> bytes:
  rbsp = bytes()
  while off < len(dat):
    if off + 2 < len(dat) and dat[off:off+3] == b'\x00\x00\x03':
      rbsp += bytes([0, 0])
      off += 3
    else:
      rbsp += bytes([dat[off]])
      off += 1
  return rbsp

class HevcSlice:
  # 7.3.3 Profile, tier and level syntax
  def profile_tier_level(self, r:BitReader, enable:bool, max_sub_layers:int):
    assert enable and max_sub_layers == 0, "no sublayers supported"
    self._notimpl_profile_tier_level = r.u(88)
    self.general_level_idc = r.u(8)

  # 7.3.7 Short-term reference picture set syntax
  def st_ref_pic_set(self, r:BitReader, stRpsIdx:int, num_short_term_ref_pic_sets:int=0, sps=None):
    inter_ref_pic_set_prediction_flag = r.u(1) if stRpsIdx != 0 else 0

    if inter_ref_pic_set_prediction_flag:
      if stRpsIdx == num_short_term_ref_pic_sets:
        delta_idx_minus1 = r.ue_v()
      delta_rps_sign = r.u(1)
      abs_delta_rps_minus1 = r.ue_v()

      NumDeltaPocs = sps.num_negative_pics + sps.num_positive_pics
      for i in range(NumDeltaPocs + 1):
        used_by_curr_pic_flag = r.u(1)
        if not used_by_curr_pic_flag:
          use_delta_flag = r.u(1)
    else:
      self.num_negative_pics = r.ue_v()
      self.num_positive_pics = r.ue_v()
      for i in range(self.num_negative_pics):
        delta_poc_s0_minus1 = r.ue_v()
        used_by_curr_pic_s0_flag = r.u(1)
      for i in range(self.num_positive_pics):
        delta_poc_s1_minus1 = r.ue_v()
        used_by_curr_pic_s1_flag = r.u(1)

# 7.3.2.2 Sequence parameter set RBSP syntax
class SPS(HevcSlice):
  def __init__(self, r:BitReader):
    self.sps_video_parameter_set_id = r.u(4)
    self.sps_max_sub_layers_minus1 = r.u(3)
    self.sps_temporal_id_nesting_flag = r.u(1)

    self.profile_tier_level(r, True, self.sps_max_sub_layers_minus1)

    self.sps_seq_parameter_set_id = r.ue_v()
    self.chroma_format_idc = r.ue_v()
    self.separate_colour_plane_flag = r.u(1) if self.chroma_format_idc == 3 else 0
    self.pic_width_in_luma_samples = r.ue_v()
    self.pic_height_in_luma_samples = r.ue_v()
    self.conformance_window_flag = r.u(1)

    if self.conformance_window_flag:
      self.conf_win_left_offset = r.ue_v()
      self.conf_win_right_offset = r.ue_v()
      self.conf_win_top_offset = r.ue_v()
      self.conf_win_bottom_offset = r.ue_v()
    else: self.conf_win_left_offset = self.conf_win_right_offset = self.conf_win_top_offset = self.conf_win_bottom_offset = 0

    self.bit_depth_luma = r.ue_v() + 8
    self.bit_depth_chroma = r.ue_v() + 8
    self.log2_max_pic_order_cnt_lsb_minus4 = r.ue_v()
    self.sps_sub_layer_ordering_info_present_flag = r.u(1)
    self.sps_max_dec_pic_buffering, self.sps_max_num_reorder_pics, self.sps_max_latency_increase_plus1 = [], [], []
    for i in range((0 if self.sps_sub_layer_ordering_info_present_flag else self.sps_max_sub_layers_minus1), self.sps_max_sub_layers_minus1 + 1):
      self.sps_max_dec_pic_buffering.append(r.ue_v() + 1)
      self.sps_max_num_reorder_pics.append(r.ue_v())
      self.sps_max_latency_increase_plus1.append(r.ue_v())
    self.log2_min_luma_coding_block_size = r.ue_v() + 3
    self.log2_max_luma_coding_block_size = self.log2_min_luma_coding_block_size + r.ue_v()
    self.log2_min_transform_block_size = r.ue_v() + 2
    self.log2_max_transform_block_size = self.log2_min_transform_block_size + r.ue_v()
    self.max_transform_hierarchy_depth_inter = r.ue_v()
    self.max_transform_hierarchy_depth_intra = r.ue_v()
    if scaling_list_enabled_flag := r.u(1):
      if sps_scaling_list_data_present_flag := r.u(1): assert False, "scaling_list_data parsing not implemented"
    self.amp_enabled_flag = r.u(1)
    self.sample_adaptive_offset_enabled_flag = r.u(1)
    self.pcm_enabled_flag = r.u(1)
    assert self.pcm_enabled_flag == 0, "pcm not implemented"
    self.num_short_term_ref_pic_sets = r.ue_v()
    for i in range(self.num_short_term_ref_pic_sets):
      self.st_ref_pic_set(r, i, self.num_short_term_ref_pic_sets)
    self.long_term_ref_pics_present_flag = r.u(1)
    if self.long_term_ref_pics_present_flag: assert False, "long_term_ref_pics parsing not implemented"
    self.sps_temporal_mvp_enabled_flag = r.u(1)
    self.strong_intra_smoothing_enabled_flag = r.u(1)

# 7.3.2.3 Picture parameter set RBSP syntax
class PPS(HevcSlice):
  def __init__(self, r:BitReader):
    self.pps_pic_parameter_set_id = r.ue_v()
    self.pps_seq_parameter_set_id = r.ue_v()
    self.dependent_slice_segments_enabled_flag = r.u(1)
    self.output_flag_present_flag = r.u(1)
    self.num_extra_slice_header_bits = r.u(3)
    self.sign_data_hiding_enabled_flag = r.u(1)
    self.cabac_init_present_flag = r.u(1)
    self.num_ref_idx_l0_default_active = r.ue_v() + 1
    self.num_ref_idx_l1_default_active = r.ue_v() + 1
    self.init_qp = r.se_v() + 26
    self.constrained_intra_pred_flag = r.u(1)
    self.transform_skip_enabled_flag = r.u(1)
    self.cu_qp_delta_enabled_flag = r.u(1)
    if self.cu_qp_delta_enabled_flag: self.diff_cu_qp_delta_depth = r.ue_v()

    self.pps_cb_qp_offset = r.se_v()
    self.pps_cr_qp_offset = r.se_v()
    self.pps_slice_chroma_qp_offsets_present_flag = r.u(1)
    self.weighted_pred_flag = r.u(1)
    self.weighted_bipred_flag = r.u(1)
    self.transquant_bypass_enabled_flag = r.u(1)
    self.tiles_enabled_flag = r.u(1)
    self.entropy_coding_sync_enabled_flag = r.u(1)
    if self.tiles_enabled_flag:
      self.num_tile_columns_minus1 = r.ue_v()
      self.num_tile_rows_minus1 = r.ue_v()
      self.uniform_spacing_flag = r.u(1)
      self.column_width_minus1, self.row_height_minus1 = [], []
      if not self.uniform_spacing_flag:
        for i in range(self.num_tile_columns_minus1): self.column_width_minus1.append(r.ue_v())
        for i in range(self.num_tile_rows_minus1): self.row_height_minus1.append(r.ue_v())
      self.loop_filter_across_tiles_enabled_flag = r.u(1)
    self.loop_filter_across_slices_enabled_flag = r.u(1)
    self.deblocking_filter_control_present_flag = r.u(1)
    if self.deblocking_filter_control_present_flag: assert False, "deblocking_filter parsing not implemented"
    self.scaling_list_data_present_flag = r.u(1)
    if self.scaling_list_data_present_flag: assert False, "scaling_list_data parsing not implemented"
    self.lists_modification_present_flag = r.u(1)
    self.log2_parallel_merge_level = r.ue_v() + 2

# 7.3.6 Slice segment header syntax
class SliceSegment(HevcSlice):
  def __init__(self, r:BitReader, nal_unit_type:int, sps:SPS, pps:PPS):
    self.first_slice_segment_in_pic_flag = r.u(1)
    if nal_unit_type >= avcodec.HEVC_NAL_BLA_W_LP and nal_unit_type <= avcodec.HEVC_NAL_RSV_IRAP_VCL23:
      self.no_output_of_prior_pics_flag = r.u(1)
    self.slice_pic_parameter_set_id = r.ue_v()
    if not self.first_slice_segment_in_pic_flag:
      if pps.dependent_slice_segments_enabled_flag:
        self.dependent_slice_segment_flag = r.u(1)
      self.slice_segment_address = r.ue_v()
    self.dependent_slice_segment_flag = 0
    if not self.dependent_slice_segment_flag:
      r.u(pps.num_extra_slice_header_bits) # extra bits ignored
      self.slice_type = r.ue_v()

      self.sw_skip_start = r.read_bits - r.current_bits
      self.pic_output_flag = r.u(1) if pps.output_flag_present_flag else 0
      self.colour_plane_id = r.u(2) if sps.separate_colour_plane_flag else 0

      if nal_unit_type != avcodec.HEVC_NAL_IDR_W_RADL and nal_unit_type != avcodec.HEVC_NAL_IDR_N_LP:
        self.slice_pic_order_cnt_lsb = r.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4)

        self.short_term_ref_pic_set_sps_flag = r.u(1)
        if not self.short_term_ref_pic_set_sps_flag:
          self.short_term_ref_pics_in_slice_start = r.read_bits - r.current_bits
          self.st_ref_pic_set(r, sps.num_short_term_ref_pic_sets, sps=sps)
          self.short_term_ref_pics_in_slice_end = r.read_bits - r.current_bits
        elif sps.num_short_term_ref_pic_sets > 1: assert False, "short_term_ref_pic_set parsing not implemented"

        if sps.long_term_ref_pics_present_flag: assert False, "long_term_ref_pics parsing not implemented"

        self.sw_skip_end = r.read_bits - r.current_bits
        self.slice_temporal_mvp_enabled_flag = r.u(1) if sps.sps_temporal_mvp_enabled_flag else 0
      else: self.slice_pic_order_cnt_lsb, self.sw_skip_end = 0, self.sw_skip_start

      if sps.sample_adaptive_offset_enabled_flag:
        slice_sao_luma_flag = r.u(1)
        ChromaArrayType = sps.chroma_format_idc if sps.separate_colour_plane_flag == 0 else 0
        slice_sao_chroma_flag = r.u(1) if ChromaArrayType != 0 else 0

      if self.slice_type in {avcodec.HEVC_SLICE_B, avcodec.HEVC_SLICE_B}:
        if num_ref_idx_active_override_flag := r.u(1):
          num_ref_idx_l0_active_minus1 = r.ue_v()
          num_ref_idx_l1_active_minus1 = r.ue_v() if self.slice_type == avcodec.HEVC_SLICE_B else 0

def fill_sps_into_dev_context(device_ctx, sps:SPS):
  device_ctx.chroma_format_idc = sps.chroma_format_idc
  device_ctx.pic_width_in_luma_samples = sps.pic_width_in_luma_samples
  device_ctx.pic_height_in_luma_samples = sps.pic_height_in_luma_samples
  device_ctx.bit_depth_luma = sps.bit_depth_luma
  device_ctx.bit_depth_chroma = sps.bit_depth_chroma
  device_ctx.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4
  device_ctx.log2_min_luma_coding_block_size = sps.log2_min_luma_coding_block_size
  device_ctx.log2_max_luma_coding_block_size = sps.log2_max_luma_coding_block_size
  device_ctx.log2_min_transform_block_size = sps.log2_min_transform_block_size
  device_ctx.log2_max_transform_block_size = sps.log2_max_transform_block_size
  device_ctx.amp_enabled_flag = sps.amp_enabled_flag
  device_ctx.pcm_enabled_flag = sps.pcm_enabled_flag
  device_ctx.sample_adaptive_offset_enabled_flag = sps.sample_adaptive_offset_enabled_flag
  device_ctx.sps_temporal_mvp_enabled_flag = sps.sps_temporal_mvp_enabled_flag
  device_ctx.strong_intra_smoothing_enabled_flag = sps.strong_intra_smoothing_enabled_flag

def fill_pps_into_dev_context(device_ctx, pps:PPS):
  device_ctx.sign_data_hiding_enabled_flag = pps.sign_data_hiding_enabled_flag
  device_ctx.cabac_init_present_flag = pps.cabac_init_present_flag
  device_ctx.num_ref_idx_l0_default_active = pps.num_ref_idx_l0_default_active
  device_ctx.num_ref_idx_l1_default_active = pps.num_ref_idx_l1_default_active
  device_ctx.init_qp = pps.init_qp
  device_ctx.cu_qp_delta_enabled_flag = pps.cu_qp_delta_enabled_flag
  device_ctx.diff_cu_qp_delta_depth = getattr(pps, 'diff_cu_qp_delta_depth', 0)
  device_ctx.pps_cb_qp_offset = pps.pps_cb_qp_offset
  device_ctx.pps_cr_qp_offset = pps.pps_cr_qp_offset
  device_ctx.pps_slice_chroma_qp_offsets_present_flag = pps.pps_slice_chroma_qp_offsets_present_flag
  device_ctx.weighted_pred_flag = pps.weighted_pred_flag
  device_ctx.weighted_bipred_flag = pps.weighted_bipred_flag
  device_ctx.transquant_bypass_enabled_flag = pps.transquant_bypass_enabled_flag
  device_ctx.tiles_enabled_flag = pps.tiles_enabled_flag
  device_ctx.entropy_coding_sync_enabled_flag = pps.entropy_coding_sync_enabled_flag
  device_ctx.loop_filter_across_slices_enabled_flag = pps.loop_filter_across_slices_enabled_flag
  device_ctx.deblocking_filter_control_present_flag = pps.deblocking_filter_control_present_flag
  device_ctx.scaling_list_data_present_flag = pps.scaling_list_data_present_flag
  device_ctx.lists_modification_present_flag = pps.lists_modification_present_flag
  device_ctx.log2_parallel_merge_level = pps.log2_parallel_merge_level
  device_ctx.loop_filter_across_tiles_enabled_flag = getattr(pps, 'loop_filter_across_tiles_enabled_flag', 0)

def parse_hevc_file_headers(dat:bytes, device="NV"):
  res = []
  nal_unit_start = 1
  history:list[tuple[int, int, int]] = []
  device_ctx = nv_gpu.nvdec_hevc_pic_s(gptimer_timeout_value=92720000, tileformat=1, sw_start_code_e=1, pattern_id=2)
  nal_infos = []
  ctx_bytes = bytes()
  align_ctx_bytes_size = 0x300

  def _flush_picture():
    nonlocal res, history, device_ctx, nal_infos, ctx_bytes, align_ctx_bytes_size

    if not len(nal_infos): return

    hdr, nal_unit_type = nal_infos[0][0]
    assert all(nal_unit_type == x[0][1] for x in nal_infos), "all NAL units in a picture must be of the same type"

    device_ctx.curr_pic_idx = next(i for i in range(16) if all(d[0] != i for d in history))

    if nal_unit_type in {avcodec.HEVC_NAL_IDR_W_RADL, avcodec.HEVC_NAL_IDR_N_LP}:
      history = []

    device_ctx.num_ref_frames = len(history)
    device_ctx.IDR_picture_flag = int(nal_unit_type in {avcodec.HEVC_NAL_IDR_W_RADL, avcodec.HEVC_NAL_IDR_N_LP})
    device_ctx.RAP_picture_flag = int(nal_unit_type >= avcodec.HEVC_NAL_BLA_W_LP and nal_unit_type <= avcodec.HEVC_NAL_RSV_IRAP_VCL23)
    device_ctx.RefDiffPicOrderCnts=(ctypes.c_int16 * 16)()
    device_ctx.colMvBuffersize = (round_up(sps.pic_width_in_luma_samples, 64) * round_up(sps.pic_height_in_luma_samples, 64) // 16) // 256
    device_ctx.framestride=(ctypes.c_uint32 * 2)(round_up(sps.pic_width_in_luma_samples, 64), round_up(sps.pic_width_in_luma_samples, 64))
    device_ctx.sw_hdr_skip_length = hdr.sw_skip_end - hdr.sw_skip_start
    device_ctx.num_bits_short_term_ref_pics_in_slice = max(0, device_ctx.sw_hdr_skip_length - 9)
    device_ctx.stream_len = sum(x[2] for x in nal_infos)

    if pps.tiles_enabled_flag:
      device_ctx.num_tile_columns = pps.num_tile_columns_minus1 + 1
      device_ctx.num_tile_rows = pps.num_tile_rows_minus1 + 1

    device_ctx.num_short_term_ref_pic_sets = sps.num_short_term_ref_pic_sets

    luma_h_rounded = round_up(sps.pic_height_in_luma_samples, 64)
    device_ctx.HevcSaoBufferOffset = (608 * luma_h_rounded) >> 8
    device_ctx.HevcBsdCtrlOffset = ((device_ctx.HevcSaoBufferOffset<<8) + 4864 * luma_h_rounded) >> 8

    device_ctx.v1.hevc_main10_444_ext.HevcFltAboveOffset = ((device_ctx.HevcBsdCtrlOffset<<8) + 152 * luma_h_rounded) >> 8
    device_ctx.v1.hevc_main10_444_ext.HevcSaoAboveOffset = ((device_ctx.v1.hevc_main10_444_ext.HevcFltAboveOffset<<8) + 2000 * luma_h_rounded) >> 8
    device_ctx.v3.HevcSliceEdgeOffset = device_ctx.v1.hevc_main10_444_ext.HevcSaoAboveOffset

    before_list, after_list = [], []
    for pic_idx, poc, _ in history:
      device_ctx.RefDiffPicOrderCnts[pic_idx] = hdr.slice_pic_order_cnt_lsb - poc
      if hdr.slice_pic_order_cnt_lsb < poc: after_list.append((poc - hdr.slice_pic_order_cnt_lsb, pic_idx))
      else: before_list.append((hdr.slice_pic_order_cnt_lsb - poc, pic_idx))
    before_list.sort()
    after_list.sort()

    device_ctx.initreflistidxl0 = (ctypes.c_uint8 * 16)(*[idx for _,idx in before_list + after_list])
    if hdr.slice_type == avcodec.HEVC_SLICE_B: device_ctx.initreflistidxl1 = (ctypes.c_uint8 * 16)(*[idx for _,idx in after_list + before_list])

    locl_ctx_bytes = bytes(device_ctx)
    locl_ctx_bytes += b'\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00' # blackwell extension
    locl_ctx_bytes += bytes(0x200 - len(locl_ctx_bytes)) # pad to 512 bytes

    pic_width_in_ctbs = ceildiv(sps.pic_width_in_luma_samples, (1 << sps.log2_max_luma_coding_block_size))
    pic_height_in_ctbs = ceildiv(sps.pic_height_in_luma_samples, (1 << sps.log2_max_luma_coding_block_size))
    # append tile sizes 0x200
    if pps.tiles_enabled_flag and pps.uniform_spacing_flag:
      assert device_ctx.num_tile_columns == 1 and device_ctx.num_tile_rows == 1, "not implemented: uniform spacing with multiple tiles"
      locl_ctx_bytes += pic_width_in_ctbs.to_bytes(2, "little") + pic_height_in_ctbs.to_bytes(2, "little")
    else:
      if pps.tiles_enabled_flag and not getattr(pps, 'uniform_spacing_flag', 0):
        column_width = [cw_minus1 + 1 for cw_minus1 in pps.column_width_minus1[0:pps.num_tile_columns_minus1]]
        row_height = [rh_minus1 + 1 for rh_minus1 in pps.row_height_minus1[0:pps.num_tile_rows_minus1]]
      else:
        column_width = []
        row_height = []

      column_width.append(pic_width_in_ctbs - sum(column_width))
      row_height.append(pic_height_in_ctbs - sum(row_height))

      for c in column_width:
        for r in row_height: locl_ctx_bytes += c.to_bytes(2, "little") + r.to_bytes(2, "little")

    luma_size = round_up(sps.pic_width_in_luma_samples, 64) * round_up(sps.pic_height_in_luma_samples, 64)
    chroma_size = round_up(sps.pic_width_in_luma_samples, 64) * round_up((sps.pic_height_in_luma_samples + 1) // 2, 64)
    is_hist = nal_unit_type in {avcodec.HEVC_NAL_TRAIL_R, avcodec.HEVC_NAL_IDR_N_LP, avcodec.HEVC_NAL_IDR_W_RADL}

    res.append((nal_infos[0][1], device_ctx.stream_len, device_ctx.curr_pic_idx, len(history), is_hist))

    locl_ctx_bytes += (align_ctx_bytes_size - len(locl_ctx_bytes)) * b'\x00'
    ctx_bytes += locl_ctx_bytes

    if nal_unit_type in {avcodec.HEVC_NAL_TRAIL_R, avcodec.HEVC_NAL_IDR_N_LP, avcodec.HEVC_NAL_IDR_W_RADL}:
      history.append((device_ctx.curr_pic_idx, hdr.slice_pic_order_cnt_lsb, None))

    if len(history) >= sps.sps_max_dec_pic_buffering[0]:
      # remove the oldest poc
      history.pop(0)

    nal_infos = []

  cnt = 0
  while nal_unit_start < len(dat):
    assert dat[nal_unit_start:nal_unit_start+3] == b"\x00\x00\x01", "NAL unit start code not found"

    pos = dat.find(b"\x00\x00\x01", nal_unit_start + 3)
    nal_unit_len = (pos if pos != -1 else len(dat)) - nal_unit_start

    # 7.3.1.1 General NAL unit syntax
    nal_unit_type = (dat[nal_unit_start+3] >> 1) & 0x3F
    slice_dat = dat[nal_unit_start+5:nal_unit_start+nal_unit_len]

    if nal_unit_type == avcodec.HEVC_NAL_SPS:
      sps = SPS(BitReader(_hevc_get_rbsp(slice_dat)))
      fill_sps_into_dev_context(device_ctx, sps)
    elif nal_unit_type == avcodec.HEVC_NAL_PPS:
      pps = PPS(BitReader(_hevc_get_rbsp(slice_dat)))
      fill_pps_into_dev_context(device_ctx, pps)
    elif nal_unit_type in {avcodec.HEVC_NAL_IDR_N_LP, avcodec.HEVC_NAL_IDR_W_RADL, avcodec.HEVC_NAL_TRAIL_R, avcodec.HEVC_NAL_TRAIL_N}:
      hdr = SliceSegment(BitReader(slice_dat), nal_unit_type, sps, pps)

      if hdr.first_slice_segment_in_pic_flag == 1: _flush_picture()
      nal_infos.append(((hdr, nal_unit_type), nal_unit_start, nal_unit_len))

    nal_unit_start += nal_unit_len
  _flush_picture()

  w = sps.pic_width_in_luma_samples - 2 * (sps.conf_win_left_offset + sps.conf_win_right_offset)
  h = sps.pic_height_in_luma_samples - 2 * (sps.conf_win_top_offset  + sps.conf_win_bottom_offset)
  chroma_off = round_up(sps.pic_width_in_luma_samples, 64) * round_up(sps.pic_height_in_luma_samples, 64)
  opaque = Tensor(ctx_bytes, device=device).reshape(len(res), align_ctx_bytes_size)
  return opaque, res, w, h, sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples, chroma_off

def _addr_table(h, w, w_aligned):
  GOB_W, GOB_H = 64, 8
  GOB_SIZE = GOB_W * GOB_H
  BLOCK_H_GOBS = 2

  xs = Tensor.arange(w, dtype=dtypes.uint32).reshape(1, w)
  ys = Tensor.arange(h, dtype=dtypes.uint32).reshape(h, 1)

  gob_x = xs // GOB_W
  gob_y = ys // GOB_H
  super_block_y = gob_y // BLOCK_H_GOBS
  gob_y_in_block = gob_y  % BLOCK_H_GOBS
  stride_gobs = w_aligned // GOB_W

  base = ((super_block_y * stride_gobs + gob_x) * BLOCK_H_GOBS + gob_y_in_block) * GOB_SIZE

  lx, ly = xs % GOB_W, ys % GOB_H
  swiz = (lx & 0x0F) | ((ly & 0x03) << 4) | ((lx & 0x10) << 2) | ((ly & 0x04) << 5) | ((lx & 0x20) << 3)
  return (base + swiz).reshape(-1)

def nv12_to_bgr_from_planes(luma: Tensor, chroma: Tensor, h: int, w: int) -> Tensor:
  Y = luma.reshape(h, w).cast(dtypes.float32)

  uv = chroma.reshape(h // 2, w // 2, 2).cast(dtypes.float32)
  U_small = uv[..., 0]
  V_small = uv[..., 1]

  U = U_small.reshape(h // 2, 1, w // 2, 1).expand(h // 2, 2, w // 2, 2).reshape(h, w)
  V = V_small.reshape(h // 2, 1, w // 2, 1).expand(h // 2, 2, w // 2, 2).reshape(h, w)

  C = Y - 16.0
  D = U - 128.0
  E = V - 128.0

  R = 1.1643835616438356 * C + 1.5960267857142858 * E
  G = 1.1643835616438356 * C - 0.39176229009491365 * D - 0.8129676472377708 * E
  B = 1.1643835616438356 * C + 2.017232142857143  * D

  R = R.maximum(0.0).minimum(255.0)
  G = G.maximum(0.0).minimum(255.0)
  B = B.maximum(0.0).minimum(255.0)

  return Tensor.stack([B, G, R], dim=2).cast(dtypes.uint8)

def untile_nv12(src:Tensor, h:int, w:int, luma_w:int, chroma_off:int) -> Tensor:
  luma = src.reshape(-1)[_addr_table(h, w, round_up(luma_w, 64))]
  chroma = src.reshape(-1)[chroma_off:][_addr_table((h + 1) // 2, w, round_up(luma_w, 64))]
  return luma.cat(chroma).realize()

def to_bgr(tensor:Tensor, h:int, w:int, luma_w:int, chroma_off:int) -> Tensor:
  luma = tensor.reshape(-1)[_addr_table(h, w, round_up(luma_w, 64))]
  chroma = tensor.reshape(-1)[chroma_off:][_addr_table((h + 1) // 2, w, round_up(luma_w, 64))]
  return nv12_to_bgr_from_planes(luma, chroma, h, w).realize()
