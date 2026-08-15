# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_v11_gfx_mqd(c.Struct):
  SIZE = 2048
  shadow_base_lo: int
  shadow_base_hi: int
  gds_bkup_base_lo: int
  gds_bkup_base_hi: int
  fw_work_area_base_lo: int
  fw_work_area_base_hi: int
  shadow_initialized: int
  ib_vmid: int
  reserved_8: int
  reserved_9: int
  reserved_10: int
  reserved_11: int
  reserved_12: int
  reserved_13: int
  reserved_14: int
  reserved_15: int
  reserved_16: int
  reserved_17: int
  reserved_18: int
  reserved_19: int
  reserved_20: int
  reserved_21: int
  reserved_22: int
  reserved_23: int
  reserved_24: int
  reserved_25: int
  reserved_26: int
  reserved_27: int
  reserved_28: int
  reserved_29: int
  reserved_30: int
  reserved_31: int
  reserved_32: int
  reserved_33: int
  reserved_34: int
  reserved_35: int
  reserved_36: int
  reserved_37: int
  reserved_38: int
  reserved_39: int
  reserved_40: int
  reserved_41: int
  reserved_42: int
  reserved_43: int
  reserved_44: int
  reserved_45: int
  reserved_46: int
  reserved_47: int
  reserved_48: int
  reserved_49: int
  reserved_50: int
  reserved_51: int
  reserved_52: int
  reserved_53: int
  reserved_54: int
  reserved_55: int
  reserved_56: int
  reserved_57: int
  reserved_58: int
  reserved_59: int
  reserved_60: int
  reserved_61: int
  reserved_62: int
  reserved_63: int
  reserved_64: int
  reserved_65: int
  reserved_66: int
  reserved_67: int
  reserved_68: int
  reserved_69: int
  reserved_70: int
  reserved_71: int
  reserved_72: int
  reserved_73: int
  reserved_74: int
  reserved_75: int
  reserved_76: int
  reserved_77: int
  reserved_78: int
  reserved_79: int
  reserved_80: int
  reserved_81: int
  reserved_82: int
  reserved_83: int
  checksum_lo: int
  checksum_hi: int
  cp_mqd_query_time_lo: int
  cp_mqd_query_time_hi: int
  reserved_88: int
  reserved_89: int
  reserved_90: int
  reserved_91: int
  cp_mqd_query_wave_count: int
  cp_mqd_query_gfx_hqd_rptr: int
  cp_mqd_query_gfx_hqd_wptr: int
  cp_mqd_query_gfx_hqd_offset: int
  reserved_96: int
  reserved_97: int
  reserved_98: int
  reserved_99: int
  reserved_100: int
  reserved_101: int
  reserved_102: int
  reserved_103: int
  control_buf_addr_lo: int
  control_buf_addr_hi: int
  disable_queue: int
  reserved_107: int
  reserved_108: int
  reserved_109: int
  reserved_110: int
  reserved_111: int
  reserved_112: int
  reserved_113: int
  reserved_114: int
  reserved_115: int
  reserved_116: int
  reserved_117: int
  reserved_118: int
  reserved_119: int
  reserved_120: int
  reserved_121: int
  reserved_122: int
  reserved_123: int
  reserved_124: int
  reserved_125: int
  reserved_126: int
  reserved_127: int
  cp_mqd_base_addr: int
  cp_mqd_base_addr_hi: int
  cp_gfx_hqd_active: int
  cp_gfx_hqd_vmid: int
  reserved_131: int
  reserved_132: int
  cp_gfx_hqd_queue_priority: int
  cp_gfx_hqd_quantum: int
  cp_gfx_hqd_base: int
  cp_gfx_hqd_base_hi: int
  cp_gfx_hqd_rptr: int
  cp_gfx_hqd_rptr_addr: int
  cp_gfx_hqd_rptr_addr_hi: int
  cp_rb_wptr_poll_addr_lo: int
  cp_rb_wptr_poll_addr_hi: int
  cp_rb_doorbell_control: int
  cp_gfx_hqd_offset: int
  cp_gfx_hqd_cntl: int
  reserved_146: int
  reserved_147: int
  cp_gfx_hqd_csmd_rptr: int
  cp_gfx_hqd_wptr: int
  cp_gfx_hqd_wptr_hi: int
  reserved_151: int
  reserved_152: int
  reserved_153: int
  reserved_154: int
  reserved_155: int
  cp_gfx_hqd_mapped: int
  cp_gfx_hqd_que_mgr_control: int
  reserved_158: int
  reserved_159: int
  cp_gfx_hqd_hq_status0: int
  cp_gfx_hqd_hq_control0: int
  cp_gfx_mqd_control: int
  reserved_163: int
  reserved_164: int
  reserved_165: int
  reserved_166: int
  reserved_167: int
  reserved_168: int
  reserved_169: int
  cp_num_prim_needed_count0_lo: int
  cp_num_prim_needed_count0_hi: int
  cp_num_prim_needed_count1_lo: int
  cp_num_prim_needed_count1_hi: int
  cp_num_prim_needed_count2_lo: int
  cp_num_prim_needed_count2_hi: int
  cp_num_prim_needed_count3_lo: int
  cp_num_prim_needed_count3_hi: int
  cp_num_prim_written_count0_lo: int
  cp_num_prim_written_count0_hi: int
  cp_num_prim_written_count1_lo: int
  cp_num_prim_written_count1_hi: int
  cp_num_prim_written_count2_lo: int
  cp_num_prim_written_count2_hi: int
  cp_num_prim_written_count3_lo: int
  cp_num_prim_written_count3_hi: int
  reserved_186: int
  reserved_187: int
  reserved_188: int
  reserved_189: int
  mp1_smn_fps_cnt: int
  sq_thread_trace_buf0_base: int
  sq_thread_trace_buf0_size: int
  sq_thread_trace_buf1_base: int
  sq_thread_trace_buf1_size: int
  sq_thread_trace_wptr: int
  sq_thread_trace_mask: int
  sq_thread_trace_token_mask: int
  sq_thread_trace_ctrl: int
  sq_thread_trace_status: int
  sq_thread_trace_dropped_cntr: int
  sq_thread_trace_finish_done_debug: int
  sq_thread_trace_gfx_draw_cntr: int
  sq_thread_trace_gfx_marker_cntr: int
  sq_thread_trace_hp3d_draw_cntr: int
  sq_thread_trace_hp3d_marker_cntr: int
  reserved_206: int
  reserved_207: int
  cp_sc_psinvoc_count0_lo: int
  cp_sc_psinvoc_count0_hi: int
  cp_pa_cprim_count_lo: int
  cp_pa_cprim_count_hi: int
  cp_pa_cinvoc_count_lo: int
  cp_pa_cinvoc_count_hi: int
  cp_vgt_vsinvoc_count_lo: int
  cp_vgt_vsinvoc_count_hi: int
  cp_vgt_gsinvoc_count_lo: int
  cp_vgt_gsinvoc_count_hi: int
  cp_vgt_gsprim_count_lo: int
  cp_vgt_gsprim_count_hi: int
  cp_vgt_iaprim_count_lo: int
  cp_vgt_iaprim_count_hi: int
  cp_vgt_iavert_count_lo: int
  cp_vgt_iavert_count_hi: int
  cp_vgt_hsinvoc_count_lo: int
  cp_vgt_hsinvoc_count_hi: int
  cp_vgt_dsinvoc_count_lo: int
  cp_vgt_dsinvoc_count_hi: int
  cp_vgt_csinvoc_count_lo: int
  cp_vgt_csinvoc_count_hi: int
  reserved_230: int
  reserved_231: int
  reserved_232: int
  reserved_233: int
  reserved_234: int
  reserved_235: int
  reserved_236: int
  reserved_237: int
  reserved_238: int
  reserved_239: int
  reserved_240: int
  reserved_241: int
  reserved_242: int
  reserved_243: int
  reserved_244: int
  reserved_245: int
  reserved_246: int
  reserved_247: int
  reserved_248: int
  reserved_249: int
  reserved_250: int
  reserved_251: int
  reserved_252: int
  reserved_253: int
  reserved_254: int
  reserved_255: int
  reserved_256: int
  reserved_257: int
  reserved_258: int
  reserved_259: int
  reserved_260: int
  reserved_261: int
  reserved_262: int
  reserved_263: int
  reserved_264: int
  reserved_265: int
  reserved_266: int
  reserved_267: int
  vgt_strmout_buffer_filled_size_0: int
  vgt_strmout_buffer_filled_size_1: int
  vgt_strmout_buffer_filled_size_2: int
  vgt_strmout_buffer_filled_size_3: int
  reserved_272: int
  reserved_273: int
  reserved_274: int
  reserved_275: int
  vgt_dma_max_size: int
  vgt_dma_num_instances: int
  reserved_278: int
  reserved_279: int
  reserved_280: int
  reserved_281: int
  reserved_282: int
  reserved_283: int
  reserved_284: int
  reserved_285: int
  reserved_286: int
  reserved_287: int
  it_set_base_ib_addr_lo: int
  it_set_base_ib_addr_hi: int
  reserved_290: int
  reserved_291: int
  reserved_292: int
  reserved_293: int
  reserved_294: int
  reserved_295: int
  reserved_296: int
  reserved_297: int
  reserved_298: int
  reserved_299: int
  reserved_300: int
  reserved_301: int
  reserved_302: int
  reserved_303: int
  reserved_304: int
  reserved_305: int
  reserved_306: int
  reserved_307: int
  reserved_308: int
  reserved_309: int
  reserved_310: int
  reserved_311: int
  reserved_312: int
  reserved_313: int
  reserved_314: int
  reserved_315: int
  reserved_316: int
  reserved_317: int
  reserved_318: int
  reserved_319: int
  reserved_320: int
  reserved_321: int
  reserved_322: int
  reserved_323: int
  reserved_324: int
  reserved_325: int
  reserved_326: int
  reserved_327: int
  reserved_328: int
  reserved_329: int
  reserved_330: int
  reserved_331: int
  reserved_332: int
  reserved_333: int
  reserved_334: int
  reserved_335: int
  reserved_336: int
  reserved_337: int
  reserved_338: int
  reserved_339: int
  reserved_340: int
  reserved_341: int
  reserved_342: int
  reserved_343: int
  reserved_344: int
  reserved_345: int
  reserved_346: int
  reserved_347: int
  reserved_348: int
  reserved_349: int
  reserved_350: int
  reserved_351: int
  reserved_352: int
  reserved_353: int
  reserved_354: int
  reserved_355: int
  spi_shader_pgm_rsrc3_ps: int
  spi_shader_pgm_rsrc3_vs: int
  spi_shader_pgm_rsrc3_gs: int
  spi_shader_pgm_rsrc3_hs: int
  spi_shader_pgm_rsrc4_ps: int
  spi_shader_pgm_rsrc4_vs: int
  spi_shader_pgm_rsrc4_gs: int
  spi_shader_pgm_rsrc4_hs: int
  db_occlusion_count0_low_00: int
  db_occlusion_count0_hi_00: int
  db_occlusion_count1_low_00: int
  db_occlusion_count1_hi_00: int
  db_occlusion_count2_low_00: int
  db_occlusion_count2_hi_00: int
  db_occlusion_count3_low_00: int
  db_occlusion_count3_hi_00: int
  db_occlusion_count0_low_01: int
  db_occlusion_count0_hi_01: int
  db_occlusion_count1_low_01: int
  db_occlusion_count1_hi_01: int
  db_occlusion_count2_low_01: int
  db_occlusion_count2_hi_01: int
  db_occlusion_count3_low_01: int
  db_occlusion_count3_hi_01: int
  db_occlusion_count0_low_02: int
  db_occlusion_count0_hi_02: int
  db_occlusion_count1_low_02: int
  db_occlusion_count1_hi_02: int
  db_occlusion_count2_low_02: int
  db_occlusion_count2_hi_02: int
  db_occlusion_count3_low_02: int
  db_occlusion_count3_hi_02: int
  db_occlusion_count0_low_03: int
  db_occlusion_count0_hi_03: int
  db_occlusion_count1_low_03: int
  db_occlusion_count1_hi_03: int
  db_occlusion_count2_low_03: int
  db_occlusion_count2_hi_03: int
  db_occlusion_count3_low_03: int
  db_occlusion_count3_hi_03: int
  db_occlusion_count0_low_04: int
  db_occlusion_count0_hi_04: int
  db_occlusion_count1_low_04: int
  db_occlusion_count1_hi_04: int
  db_occlusion_count2_low_04: int
  db_occlusion_count2_hi_04: int
  db_occlusion_count3_low_04: int
  db_occlusion_count3_hi_04: int
  db_occlusion_count0_low_05: int
  db_occlusion_count0_hi_05: int
  db_occlusion_count1_low_05: int
  db_occlusion_count1_hi_05: int
  db_occlusion_count2_low_05: int
  db_occlusion_count2_hi_05: int
  db_occlusion_count3_low_05: int
  db_occlusion_count3_hi_05: int
  db_occlusion_count0_low_06: int
  db_occlusion_count0_hi_06: int
  db_occlusion_count1_low_06: int
  db_occlusion_count1_hi_06: int
  db_occlusion_count2_low_06: int
  db_occlusion_count2_hi_06: int
  db_occlusion_count3_low_06: int
  db_occlusion_count3_hi_06: int
  db_occlusion_count0_low_07: int
  db_occlusion_count0_hi_07: int
  db_occlusion_count1_low_07: int
  db_occlusion_count1_hi_07: int
  db_occlusion_count2_low_07: int
  db_occlusion_count2_hi_07: int
  db_occlusion_count3_low_07: int
  db_occlusion_count3_hi_07: int
  db_occlusion_count0_low_10: int
  db_occlusion_count0_hi_10: int
  db_occlusion_count1_low_10: int
  db_occlusion_count1_hi_10: int
  db_occlusion_count2_low_10: int
  db_occlusion_count2_hi_10: int
  db_occlusion_count3_low_10: int
  db_occlusion_count3_hi_10: int
  db_occlusion_count0_low_11: int
  db_occlusion_count0_hi_11: int
  db_occlusion_count1_low_11: int
  db_occlusion_count1_hi_11: int
  db_occlusion_count2_low_11: int
  db_occlusion_count2_hi_11: int
  db_occlusion_count3_low_11: int
  db_occlusion_count3_hi_11: int
  db_occlusion_count0_low_12: int
  db_occlusion_count0_hi_12: int
  db_occlusion_count1_low_12: int
  db_occlusion_count1_hi_12: int
  db_occlusion_count2_low_12: int
  db_occlusion_count2_hi_12: int
  db_occlusion_count3_low_12: int
  db_occlusion_count3_hi_12: int
  db_occlusion_count0_low_13: int
  db_occlusion_count0_hi_13: int
  db_occlusion_count1_low_13: int
  db_occlusion_count1_hi_13: int
  db_occlusion_count2_low_13: int
  db_occlusion_count2_hi_13: int
  db_occlusion_count3_low_13: int
  db_occlusion_count3_hi_13: int
  db_occlusion_count0_low_14: int
  db_occlusion_count0_hi_14: int
  db_occlusion_count1_low_14: int
  db_occlusion_count1_hi_14: int
  db_occlusion_count2_low_14: int
  db_occlusion_count2_hi_14: int
  db_occlusion_count3_low_14: int
  db_occlusion_count3_hi_14: int
  db_occlusion_count0_low_15: int
  db_occlusion_count0_hi_15: int
  db_occlusion_count1_low_15: int
  db_occlusion_count1_hi_15: int
  db_occlusion_count2_low_15: int
  db_occlusion_count2_hi_15: int
  db_occlusion_count3_low_15: int
  db_occlusion_count3_hi_15: int
  db_occlusion_count0_low_16: int
  db_occlusion_count0_hi_16: int
  db_occlusion_count1_low_16: int
  db_occlusion_count1_hi_16: int
  db_occlusion_count2_low_16: int
  db_occlusion_count2_hi_16: int
  db_occlusion_count3_low_16: int
  db_occlusion_count3_hi_16: int
  db_occlusion_count0_low_17: int
  db_occlusion_count0_hi_17: int
  db_occlusion_count1_low_17: int
  db_occlusion_count1_hi_17: int
  db_occlusion_count2_low_17: int
  db_occlusion_count2_hi_17: int
  db_occlusion_count3_low_17: int
  db_occlusion_count3_hi_17: int
  reserved_492: int
  reserved_493: int
  reserved_494: int
  reserved_495: int
  reserved_496: int
  reserved_497: int
  reserved_498: int
  reserved_499: int
  reserved_500: int
  reserved_501: int
  reserved_502: int
  reserved_503: int
  reserved_504: int
  reserved_505: int
  reserved_506: int
  reserved_507: int
  reserved_508: int
  reserved_509: int
  reserved_510: int
  reserved_511: int
struct_v11_gfx_mqd.register_fields([('shadow_base_lo', ctypes.c_uint32, 0), ('shadow_base_hi', ctypes.c_uint32, 4), ('gds_bkup_base_lo', ctypes.c_uint32, 8), ('gds_bkup_base_hi', ctypes.c_uint32, 12), ('fw_work_area_base_lo', ctypes.c_uint32, 16), ('fw_work_area_base_hi', ctypes.c_uint32, 20), ('shadow_initialized', ctypes.c_uint32, 24), ('ib_vmid', ctypes.c_uint32, 28), ('reserved_8', ctypes.c_uint32, 32), ('reserved_9', ctypes.c_uint32, 36), ('reserved_10', ctypes.c_uint32, 40), ('reserved_11', ctypes.c_uint32, 44), ('reserved_12', ctypes.c_uint32, 48), ('reserved_13', ctypes.c_uint32, 52), ('reserved_14', ctypes.c_uint32, 56), ('reserved_15', ctypes.c_uint32, 60), ('reserved_16', ctypes.c_uint32, 64), ('reserved_17', ctypes.c_uint32, 68), ('reserved_18', ctypes.c_uint32, 72), ('reserved_19', ctypes.c_uint32, 76), ('reserved_20', ctypes.c_uint32, 80), ('reserved_21', ctypes.c_uint32, 84), ('reserved_22', ctypes.c_uint32, 88), ('reserved_23', ctypes.c_uint32, 92), ('reserved_24', ctypes.c_uint32, 96), ('reserved_25', ctypes.c_uint32, 100), ('reserved_26', ctypes.c_uint32, 104), ('reserved_27', ctypes.c_uint32, 108), ('reserved_28', ctypes.c_uint32, 112), ('reserved_29', ctypes.c_uint32, 116), ('reserved_30', ctypes.c_uint32, 120), ('reserved_31', ctypes.c_uint32, 124), ('reserved_32', ctypes.c_uint32, 128), ('reserved_33', ctypes.c_uint32, 132), ('reserved_34', ctypes.c_uint32, 136), ('reserved_35', ctypes.c_uint32, 140), ('reserved_36', ctypes.c_uint32, 144), ('reserved_37', ctypes.c_uint32, 148), ('reserved_38', ctypes.c_uint32, 152), ('reserved_39', ctypes.c_uint32, 156), ('reserved_40', ctypes.c_uint32, 160), ('reserved_41', ctypes.c_uint32, 164), ('reserved_42', ctypes.c_uint32, 168), ('reserved_43', ctypes.c_uint32, 172), ('reserved_44', ctypes.c_uint32, 176), ('reserved_45', ctypes.c_uint32, 180), ('reserved_46', ctypes.c_uint32, 184), ('reserved_47', ctypes.c_uint32, 188), ('reserved_48', ctypes.c_uint32, 192), ('reserved_49', ctypes.c_uint32, 196), ('reserved_50', ctypes.c_uint32, 200), ('reserved_51', ctypes.c_uint32, 204), ('reserved_52', ctypes.c_uint32, 208), ('reserved_53', ctypes.c_uint32, 212), ('reserved_54', ctypes.c_uint32, 216), ('reserved_55', ctypes.c_uint32, 220), ('reserved_56', ctypes.c_uint32, 224), ('reserved_57', ctypes.c_uint32, 228), ('reserved_58', ctypes.c_uint32, 232), ('reserved_59', ctypes.c_uint32, 236), ('reserved_60', ctypes.c_uint32, 240), ('reserved_61', ctypes.c_uint32, 244), ('reserved_62', ctypes.c_uint32, 248), ('reserved_63', ctypes.c_uint32, 252), ('reserved_64', ctypes.c_uint32, 256), ('reserved_65', ctypes.c_uint32, 260), ('reserved_66', ctypes.c_uint32, 264), ('reserved_67', ctypes.c_uint32, 268), ('reserved_68', ctypes.c_uint32, 272), ('reserved_69', ctypes.c_uint32, 276), ('reserved_70', ctypes.c_uint32, 280), ('reserved_71', ctypes.c_uint32, 284), ('reserved_72', ctypes.c_uint32, 288), ('reserved_73', ctypes.c_uint32, 292), ('reserved_74', ctypes.c_uint32, 296), ('reserved_75', ctypes.c_uint32, 300), ('reserved_76', ctypes.c_uint32, 304), ('reserved_77', ctypes.c_uint32, 308), ('reserved_78', ctypes.c_uint32, 312), ('reserved_79', ctypes.c_uint32, 316), ('reserved_80', ctypes.c_uint32, 320), ('reserved_81', ctypes.c_uint32, 324), ('reserved_82', ctypes.c_uint32, 328), ('reserved_83', ctypes.c_uint32, 332), ('checksum_lo', ctypes.c_uint32, 336), ('checksum_hi', ctypes.c_uint32, 340), ('cp_mqd_query_time_lo', ctypes.c_uint32, 344), ('cp_mqd_query_time_hi', ctypes.c_uint32, 348), ('reserved_88', ctypes.c_uint32, 352), ('reserved_89', ctypes.c_uint32, 356), ('reserved_90', ctypes.c_uint32, 360), ('reserved_91', ctypes.c_uint32, 364), ('cp_mqd_query_wave_count', ctypes.c_uint32, 368), ('cp_mqd_query_gfx_hqd_rptr', ctypes.c_uint32, 372), ('cp_mqd_query_gfx_hqd_wptr', ctypes.c_uint32, 376), ('cp_mqd_query_gfx_hqd_offset', ctypes.c_uint32, 380), ('reserved_96', ctypes.c_uint32, 384), ('reserved_97', ctypes.c_uint32, 388), ('reserved_98', ctypes.c_uint32, 392), ('reserved_99', ctypes.c_uint32, 396), ('reserved_100', ctypes.c_uint32, 400), ('reserved_101', ctypes.c_uint32, 404), ('reserved_102', ctypes.c_uint32, 408), ('reserved_103', ctypes.c_uint32, 412), ('control_buf_addr_lo', ctypes.c_uint32, 416), ('control_buf_addr_hi', ctypes.c_uint32, 420), ('disable_queue', ctypes.c_uint32, 424), ('reserved_107', ctypes.c_uint32, 428), ('reserved_108', ctypes.c_uint32, 432), ('reserved_109', ctypes.c_uint32, 436), ('reserved_110', ctypes.c_uint32, 440), ('reserved_111', ctypes.c_uint32, 444), ('reserved_112', ctypes.c_uint32, 448), ('reserved_113', ctypes.c_uint32, 452), ('reserved_114', ctypes.c_uint32, 456), ('reserved_115', ctypes.c_uint32, 460), ('reserved_116', ctypes.c_uint32, 464), ('reserved_117', ctypes.c_uint32, 468), ('reserved_118', ctypes.c_uint32, 472), ('reserved_119', ctypes.c_uint32, 476), ('reserved_120', ctypes.c_uint32, 480), ('reserved_121', ctypes.c_uint32, 484), ('reserved_122', ctypes.c_uint32, 488), ('reserved_123', ctypes.c_uint32, 492), ('reserved_124', ctypes.c_uint32, 496), ('reserved_125', ctypes.c_uint32, 500), ('reserved_126', ctypes.c_uint32, 504), ('reserved_127', ctypes.c_uint32, 508), ('cp_mqd_base_addr', ctypes.c_uint32, 512), ('cp_mqd_base_addr_hi', ctypes.c_uint32, 516), ('cp_gfx_hqd_active', ctypes.c_uint32, 520), ('cp_gfx_hqd_vmid', ctypes.c_uint32, 524), ('reserved_131', ctypes.c_uint32, 528), ('reserved_132', ctypes.c_uint32, 532), ('cp_gfx_hqd_queue_priority', ctypes.c_uint32, 536), ('cp_gfx_hqd_quantum', ctypes.c_uint32, 540), ('cp_gfx_hqd_base', ctypes.c_uint32, 544), ('cp_gfx_hqd_base_hi', ctypes.c_uint32, 548), ('cp_gfx_hqd_rptr', ctypes.c_uint32, 552), ('cp_gfx_hqd_rptr_addr', ctypes.c_uint32, 556), ('cp_gfx_hqd_rptr_addr_hi', ctypes.c_uint32, 560), ('cp_rb_wptr_poll_addr_lo', ctypes.c_uint32, 564), ('cp_rb_wptr_poll_addr_hi', ctypes.c_uint32, 568), ('cp_rb_doorbell_control', ctypes.c_uint32, 572), ('cp_gfx_hqd_offset', ctypes.c_uint32, 576), ('cp_gfx_hqd_cntl', ctypes.c_uint32, 580), ('reserved_146', ctypes.c_uint32, 584), ('reserved_147', ctypes.c_uint32, 588), ('cp_gfx_hqd_csmd_rptr', ctypes.c_uint32, 592), ('cp_gfx_hqd_wptr', ctypes.c_uint32, 596), ('cp_gfx_hqd_wptr_hi', ctypes.c_uint32, 600), ('reserved_151', ctypes.c_uint32, 604), ('reserved_152', ctypes.c_uint32, 608), ('reserved_153', ctypes.c_uint32, 612), ('reserved_154', ctypes.c_uint32, 616), ('reserved_155', ctypes.c_uint32, 620), ('cp_gfx_hqd_mapped', ctypes.c_uint32, 624), ('cp_gfx_hqd_que_mgr_control', ctypes.c_uint32, 628), ('reserved_158', ctypes.c_uint32, 632), ('reserved_159', ctypes.c_uint32, 636), ('cp_gfx_hqd_hq_status0', ctypes.c_uint32, 640), ('cp_gfx_hqd_hq_control0', ctypes.c_uint32, 644), ('cp_gfx_mqd_control', ctypes.c_uint32, 648), ('reserved_163', ctypes.c_uint32, 652), ('reserved_164', ctypes.c_uint32, 656), ('reserved_165', ctypes.c_uint32, 660), ('reserved_166', ctypes.c_uint32, 664), ('reserved_167', ctypes.c_uint32, 668), ('reserved_168', ctypes.c_uint32, 672), ('reserved_169', ctypes.c_uint32, 676), ('cp_num_prim_needed_count0_lo', ctypes.c_uint32, 680), ('cp_num_prim_needed_count0_hi', ctypes.c_uint32, 684), ('cp_num_prim_needed_count1_lo', ctypes.c_uint32, 688), ('cp_num_prim_needed_count1_hi', ctypes.c_uint32, 692), ('cp_num_prim_needed_count2_lo', ctypes.c_uint32, 696), ('cp_num_prim_needed_count2_hi', ctypes.c_uint32, 700), ('cp_num_prim_needed_count3_lo', ctypes.c_uint32, 704), ('cp_num_prim_needed_count3_hi', ctypes.c_uint32, 708), ('cp_num_prim_written_count0_lo', ctypes.c_uint32, 712), ('cp_num_prim_written_count0_hi', ctypes.c_uint32, 716), ('cp_num_prim_written_count1_lo', ctypes.c_uint32, 720), ('cp_num_prim_written_count1_hi', ctypes.c_uint32, 724), ('cp_num_prim_written_count2_lo', ctypes.c_uint32, 728), ('cp_num_prim_written_count2_hi', ctypes.c_uint32, 732), ('cp_num_prim_written_count3_lo', ctypes.c_uint32, 736), ('cp_num_prim_written_count3_hi', ctypes.c_uint32, 740), ('reserved_186', ctypes.c_uint32, 744), ('reserved_187', ctypes.c_uint32, 748), ('reserved_188', ctypes.c_uint32, 752), ('reserved_189', ctypes.c_uint32, 756), ('mp1_smn_fps_cnt', ctypes.c_uint32, 760), ('sq_thread_trace_buf0_base', ctypes.c_uint32, 764), ('sq_thread_trace_buf0_size', ctypes.c_uint32, 768), ('sq_thread_trace_buf1_base', ctypes.c_uint32, 772), ('sq_thread_trace_buf1_size', ctypes.c_uint32, 776), ('sq_thread_trace_wptr', ctypes.c_uint32, 780), ('sq_thread_trace_mask', ctypes.c_uint32, 784), ('sq_thread_trace_token_mask', ctypes.c_uint32, 788), ('sq_thread_trace_ctrl', ctypes.c_uint32, 792), ('sq_thread_trace_status', ctypes.c_uint32, 796), ('sq_thread_trace_dropped_cntr', ctypes.c_uint32, 800), ('sq_thread_trace_finish_done_debug', ctypes.c_uint32, 804), ('sq_thread_trace_gfx_draw_cntr', ctypes.c_uint32, 808), ('sq_thread_trace_gfx_marker_cntr', ctypes.c_uint32, 812), ('sq_thread_trace_hp3d_draw_cntr', ctypes.c_uint32, 816), ('sq_thread_trace_hp3d_marker_cntr', ctypes.c_uint32, 820), ('reserved_206', ctypes.c_uint32, 824), ('reserved_207', ctypes.c_uint32, 828), ('cp_sc_psinvoc_count0_lo', ctypes.c_uint32, 832), ('cp_sc_psinvoc_count0_hi', ctypes.c_uint32, 836), ('cp_pa_cprim_count_lo', ctypes.c_uint32, 840), ('cp_pa_cprim_count_hi', ctypes.c_uint32, 844), ('cp_pa_cinvoc_count_lo', ctypes.c_uint32, 848), ('cp_pa_cinvoc_count_hi', ctypes.c_uint32, 852), ('cp_vgt_vsinvoc_count_lo', ctypes.c_uint32, 856), ('cp_vgt_vsinvoc_count_hi', ctypes.c_uint32, 860), ('cp_vgt_gsinvoc_count_lo', ctypes.c_uint32, 864), ('cp_vgt_gsinvoc_count_hi', ctypes.c_uint32, 868), ('cp_vgt_gsprim_count_lo', ctypes.c_uint32, 872), ('cp_vgt_gsprim_count_hi', ctypes.c_uint32, 876), ('cp_vgt_iaprim_count_lo', ctypes.c_uint32, 880), ('cp_vgt_iaprim_count_hi', ctypes.c_uint32, 884), ('cp_vgt_iavert_count_lo', ctypes.c_uint32, 888), ('cp_vgt_iavert_count_hi', ctypes.c_uint32, 892), ('cp_vgt_hsinvoc_count_lo', ctypes.c_uint32, 896), ('cp_vgt_hsinvoc_count_hi', ctypes.c_uint32, 900), ('cp_vgt_dsinvoc_count_lo', ctypes.c_uint32, 904), ('cp_vgt_dsinvoc_count_hi', ctypes.c_uint32, 908), ('cp_vgt_csinvoc_count_lo', ctypes.c_uint32, 912), ('cp_vgt_csinvoc_count_hi', ctypes.c_uint32, 916), ('reserved_230', ctypes.c_uint32, 920), ('reserved_231', ctypes.c_uint32, 924), ('reserved_232', ctypes.c_uint32, 928), ('reserved_233', ctypes.c_uint32, 932), ('reserved_234', ctypes.c_uint32, 936), ('reserved_235', ctypes.c_uint32, 940), ('reserved_236', ctypes.c_uint32, 944), ('reserved_237', ctypes.c_uint32, 948), ('reserved_238', ctypes.c_uint32, 952), ('reserved_239', ctypes.c_uint32, 956), ('reserved_240', ctypes.c_uint32, 960), ('reserved_241', ctypes.c_uint32, 964), ('reserved_242', ctypes.c_uint32, 968), ('reserved_243', ctypes.c_uint32, 972), ('reserved_244', ctypes.c_uint32, 976), ('reserved_245', ctypes.c_uint32, 980), ('reserved_246', ctypes.c_uint32, 984), ('reserved_247', ctypes.c_uint32, 988), ('reserved_248', ctypes.c_uint32, 992), ('reserved_249', ctypes.c_uint32, 996), ('reserved_250', ctypes.c_uint32, 1000), ('reserved_251', ctypes.c_uint32, 1004), ('reserved_252', ctypes.c_uint32, 1008), ('reserved_253', ctypes.c_uint32, 1012), ('reserved_254', ctypes.c_uint32, 1016), ('reserved_255', ctypes.c_uint32, 1020), ('reserved_256', ctypes.c_uint32, 1024), ('reserved_257', ctypes.c_uint32, 1028), ('reserved_258', ctypes.c_uint32, 1032), ('reserved_259', ctypes.c_uint32, 1036), ('reserved_260', ctypes.c_uint32, 1040), ('reserved_261', ctypes.c_uint32, 1044), ('reserved_262', ctypes.c_uint32, 1048), ('reserved_263', ctypes.c_uint32, 1052), ('reserved_264', ctypes.c_uint32, 1056), ('reserved_265', ctypes.c_uint32, 1060), ('reserved_266', ctypes.c_uint32, 1064), ('reserved_267', ctypes.c_uint32, 1068), ('vgt_strmout_buffer_filled_size_0', ctypes.c_uint32, 1072), ('vgt_strmout_buffer_filled_size_1', ctypes.c_uint32, 1076), ('vgt_strmout_buffer_filled_size_2', ctypes.c_uint32, 1080), ('vgt_strmout_buffer_filled_size_3', ctypes.c_uint32, 1084), ('reserved_272', ctypes.c_uint32, 1088), ('reserved_273', ctypes.c_uint32, 1092), ('reserved_274', ctypes.c_uint32, 1096), ('reserved_275', ctypes.c_uint32, 1100), ('vgt_dma_max_size', ctypes.c_uint32, 1104), ('vgt_dma_num_instances', ctypes.c_uint32, 1108), ('reserved_278', ctypes.c_uint32, 1112), ('reserved_279', ctypes.c_uint32, 1116), ('reserved_280', ctypes.c_uint32, 1120), ('reserved_281', ctypes.c_uint32, 1124), ('reserved_282', ctypes.c_uint32, 1128), ('reserved_283', ctypes.c_uint32, 1132), ('reserved_284', ctypes.c_uint32, 1136), ('reserved_285', ctypes.c_uint32, 1140), ('reserved_286', ctypes.c_uint32, 1144), ('reserved_287', ctypes.c_uint32, 1148), ('it_set_base_ib_addr_lo', ctypes.c_uint32, 1152), ('it_set_base_ib_addr_hi', ctypes.c_uint32, 1156), ('reserved_290', ctypes.c_uint32, 1160), ('reserved_291', ctypes.c_uint32, 1164), ('reserved_292', ctypes.c_uint32, 1168), ('reserved_293', ctypes.c_uint32, 1172), ('reserved_294', ctypes.c_uint32, 1176), ('reserved_295', ctypes.c_uint32, 1180), ('reserved_296', ctypes.c_uint32, 1184), ('reserved_297', ctypes.c_uint32, 1188), ('reserved_298', ctypes.c_uint32, 1192), ('reserved_299', ctypes.c_uint32, 1196), ('reserved_300', ctypes.c_uint32, 1200), ('reserved_301', ctypes.c_uint32, 1204), ('reserved_302', ctypes.c_uint32, 1208), ('reserved_303', ctypes.c_uint32, 1212), ('reserved_304', ctypes.c_uint32, 1216), ('reserved_305', ctypes.c_uint32, 1220), ('reserved_306', ctypes.c_uint32, 1224), ('reserved_307', ctypes.c_uint32, 1228), ('reserved_308', ctypes.c_uint32, 1232), ('reserved_309', ctypes.c_uint32, 1236), ('reserved_310', ctypes.c_uint32, 1240), ('reserved_311', ctypes.c_uint32, 1244), ('reserved_312', ctypes.c_uint32, 1248), ('reserved_313', ctypes.c_uint32, 1252), ('reserved_314', ctypes.c_uint32, 1256), ('reserved_315', ctypes.c_uint32, 1260), ('reserved_316', ctypes.c_uint32, 1264), ('reserved_317', ctypes.c_uint32, 1268), ('reserved_318', ctypes.c_uint32, 1272), ('reserved_319', ctypes.c_uint32, 1276), ('reserved_320', ctypes.c_uint32, 1280), ('reserved_321', ctypes.c_uint32, 1284), ('reserved_322', ctypes.c_uint32, 1288), ('reserved_323', ctypes.c_uint32, 1292), ('reserved_324', ctypes.c_uint32, 1296), ('reserved_325', ctypes.c_uint32, 1300), ('reserved_326', ctypes.c_uint32, 1304), ('reserved_327', ctypes.c_uint32, 1308), ('reserved_328', ctypes.c_uint32, 1312), ('reserved_329', ctypes.c_uint32, 1316), ('reserved_330', ctypes.c_uint32, 1320), ('reserved_331', ctypes.c_uint32, 1324), ('reserved_332', ctypes.c_uint32, 1328), ('reserved_333', ctypes.c_uint32, 1332), ('reserved_334', ctypes.c_uint32, 1336), ('reserved_335', ctypes.c_uint32, 1340), ('reserved_336', ctypes.c_uint32, 1344), ('reserved_337', ctypes.c_uint32, 1348), ('reserved_338', ctypes.c_uint32, 1352), ('reserved_339', ctypes.c_uint32, 1356), ('reserved_340', ctypes.c_uint32, 1360), ('reserved_341', ctypes.c_uint32, 1364), ('reserved_342', ctypes.c_uint32, 1368), ('reserved_343', ctypes.c_uint32, 1372), ('reserved_344', ctypes.c_uint32, 1376), ('reserved_345', ctypes.c_uint32, 1380), ('reserved_346', ctypes.c_uint32, 1384), ('reserved_347', ctypes.c_uint32, 1388), ('reserved_348', ctypes.c_uint32, 1392), ('reserved_349', ctypes.c_uint32, 1396), ('reserved_350', ctypes.c_uint32, 1400), ('reserved_351', ctypes.c_uint32, 1404), ('reserved_352', ctypes.c_uint32, 1408), ('reserved_353', ctypes.c_uint32, 1412), ('reserved_354', ctypes.c_uint32, 1416), ('reserved_355', ctypes.c_uint32, 1420), ('spi_shader_pgm_rsrc3_ps', ctypes.c_uint32, 1424), ('spi_shader_pgm_rsrc3_vs', ctypes.c_uint32, 1428), ('spi_shader_pgm_rsrc3_gs', ctypes.c_uint32, 1432), ('spi_shader_pgm_rsrc3_hs', ctypes.c_uint32, 1436), ('spi_shader_pgm_rsrc4_ps', ctypes.c_uint32, 1440), ('spi_shader_pgm_rsrc4_vs', ctypes.c_uint32, 1444), ('spi_shader_pgm_rsrc4_gs', ctypes.c_uint32, 1448), ('spi_shader_pgm_rsrc4_hs', ctypes.c_uint32, 1452), ('db_occlusion_count0_low_00', ctypes.c_uint32, 1456), ('db_occlusion_count0_hi_00', ctypes.c_uint32, 1460), ('db_occlusion_count1_low_00', ctypes.c_uint32, 1464), ('db_occlusion_count1_hi_00', ctypes.c_uint32, 1468), ('db_occlusion_count2_low_00', ctypes.c_uint32, 1472), ('db_occlusion_count2_hi_00', ctypes.c_uint32, 1476), ('db_occlusion_count3_low_00', ctypes.c_uint32, 1480), ('db_occlusion_count3_hi_00', ctypes.c_uint32, 1484), ('db_occlusion_count0_low_01', ctypes.c_uint32, 1488), ('db_occlusion_count0_hi_01', ctypes.c_uint32, 1492), ('db_occlusion_count1_low_01', ctypes.c_uint32, 1496), ('db_occlusion_count1_hi_01', ctypes.c_uint32, 1500), ('db_occlusion_count2_low_01', ctypes.c_uint32, 1504), ('db_occlusion_count2_hi_01', ctypes.c_uint32, 1508), ('db_occlusion_count3_low_01', ctypes.c_uint32, 1512), ('db_occlusion_count3_hi_01', ctypes.c_uint32, 1516), ('db_occlusion_count0_low_02', ctypes.c_uint32, 1520), ('db_occlusion_count0_hi_02', ctypes.c_uint32, 1524), ('db_occlusion_count1_low_02', ctypes.c_uint32, 1528), ('db_occlusion_count1_hi_02', ctypes.c_uint32, 1532), ('db_occlusion_count2_low_02', ctypes.c_uint32, 1536), ('db_occlusion_count2_hi_02', ctypes.c_uint32, 1540), ('db_occlusion_count3_low_02', ctypes.c_uint32, 1544), ('db_occlusion_count3_hi_02', ctypes.c_uint32, 1548), ('db_occlusion_count0_low_03', ctypes.c_uint32, 1552), ('db_occlusion_count0_hi_03', ctypes.c_uint32, 1556), ('db_occlusion_count1_low_03', ctypes.c_uint32, 1560), ('db_occlusion_count1_hi_03', ctypes.c_uint32, 1564), ('db_occlusion_count2_low_03', ctypes.c_uint32, 1568), ('db_occlusion_count2_hi_03', ctypes.c_uint32, 1572), ('db_occlusion_count3_low_03', ctypes.c_uint32, 1576), ('db_occlusion_count3_hi_03', ctypes.c_uint32, 1580), ('db_occlusion_count0_low_04', ctypes.c_uint32, 1584), ('db_occlusion_count0_hi_04', ctypes.c_uint32, 1588), ('db_occlusion_count1_low_04', ctypes.c_uint32, 1592), ('db_occlusion_count1_hi_04', ctypes.c_uint32, 1596), ('db_occlusion_count2_low_04', ctypes.c_uint32, 1600), ('db_occlusion_count2_hi_04', ctypes.c_uint32, 1604), ('db_occlusion_count3_low_04', ctypes.c_uint32, 1608), ('db_occlusion_count3_hi_04', ctypes.c_uint32, 1612), ('db_occlusion_count0_low_05', ctypes.c_uint32, 1616), ('db_occlusion_count0_hi_05', ctypes.c_uint32, 1620), ('db_occlusion_count1_low_05', ctypes.c_uint32, 1624), ('db_occlusion_count1_hi_05', ctypes.c_uint32, 1628), ('db_occlusion_count2_low_05', ctypes.c_uint32, 1632), ('db_occlusion_count2_hi_05', ctypes.c_uint32, 1636), ('db_occlusion_count3_low_05', ctypes.c_uint32, 1640), ('db_occlusion_count3_hi_05', ctypes.c_uint32, 1644), ('db_occlusion_count0_low_06', ctypes.c_uint32, 1648), ('db_occlusion_count0_hi_06', ctypes.c_uint32, 1652), ('db_occlusion_count1_low_06', ctypes.c_uint32, 1656), ('db_occlusion_count1_hi_06', ctypes.c_uint32, 1660), ('db_occlusion_count2_low_06', ctypes.c_uint32, 1664), ('db_occlusion_count2_hi_06', ctypes.c_uint32, 1668), ('db_occlusion_count3_low_06', ctypes.c_uint32, 1672), ('db_occlusion_count3_hi_06', ctypes.c_uint32, 1676), ('db_occlusion_count0_low_07', ctypes.c_uint32, 1680), ('db_occlusion_count0_hi_07', ctypes.c_uint32, 1684), ('db_occlusion_count1_low_07', ctypes.c_uint32, 1688), ('db_occlusion_count1_hi_07', ctypes.c_uint32, 1692), ('db_occlusion_count2_low_07', ctypes.c_uint32, 1696), ('db_occlusion_count2_hi_07', ctypes.c_uint32, 1700), ('db_occlusion_count3_low_07', ctypes.c_uint32, 1704), ('db_occlusion_count3_hi_07', ctypes.c_uint32, 1708), ('db_occlusion_count0_low_10', ctypes.c_uint32, 1712), ('db_occlusion_count0_hi_10', ctypes.c_uint32, 1716), ('db_occlusion_count1_low_10', ctypes.c_uint32, 1720), ('db_occlusion_count1_hi_10', ctypes.c_uint32, 1724), ('db_occlusion_count2_low_10', ctypes.c_uint32, 1728), ('db_occlusion_count2_hi_10', ctypes.c_uint32, 1732), ('db_occlusion_count3_low_10', ctypes.c_uint32, 1736), ('db_occlusion_count3_hi_10', ctypes.c_uint32, 1740), ('db_occlusion_count0_low_11', ctypes.c_uint32, 1744), ('db_occlusion_count0_hi_11', ctypes.c_uint32, 1748), ('db_occlusion_count1_low_11', ctypes.c_uint32, 1752), ('db_occlusion_count1_hi_11', ctypes.c_uint32, 1756), ('db_occlusion_count2_low_11', ctypes.c_uint32, 1760), ('db_occlusion_count2_hi_11', ctypes.c_uint32, 1764), ('db_occlusion_count3_low_11', ctypes.c_uint32, 1768), ('db_occlusion_count3_hi_11', ctypes.c_uint32, 1772), ('db_occlusion_count0_low_12', ctypes.c_uint32, 1776), ('db_occlusion_count0_hi_12', ctypes.c_uint32, 1780), ('db_occlusion_count1_low_12', ctypes.c_uint32, 1784), ('db_occlusion_count1_hi_12', ctypes.c_uint32, 1788), ('db_occlusion_count2_low_12', ctypes.c_uint32, 1792), ('db_occlusion_count2_hi_12', ctypes.c_uint32, 1796), ('db_occlusion_count3_low_12', ctypes.c_uint32, 1800), ('db_occlusion_count3_hi_12', ctypes.c_uint32, 1804), ('db_occlusion_count0_low_13', ctypes.c_uint32, 1808), ('db_occlusion_count0_hi_13', ctypes.c_uint32, 1812), ('db_occlusion_count1_low_13', ctypes.c_uint32, 1816), ('db_occlusion_count1_hi_13', ctypes.c_uint32, 1820), ('db_occlusion_count2_low_13', ctypes.c_uint32, 1824), ('db_occlusion_count2_hi_13', ctypes.c_uint32, 1828), ('db_occlusion_count3_low_13', ctypes.c_uint32, 1832), ('db_occlusion_count3_hi_13', ctypes.c_uint32, 1836), ('db_occlusion_count0_low_14', ctypes.c_uint32, 1840), ('db_occlusion_count0_hi_14', ctypes.c_uint32, 1844), ('db_occlusion_count1_low_14', ctypes.c_uint32, 1848), ('db_occlusion_count1_hi_14', ctypes.c_uint32, 1852), ('db_occlusion_count2_low_14', ctypes.c_uint32, 1856), ('db_occlusion_count2_hi_14', ctypes.c_uint32, 1860), ('db_occlusion_count3_low_14', ctypes.c_uint32, 1864), ('db_occlusion_count3_hi_14', ctypes.c_uint32, 1868), ('db_occlusion_count0_low_15', ctypes.c_uint32, 1872), ('db_occlusion_count0_hi_15', ctypes.c_uint32, 1876), ('db_occlusion_count1_low_15', ctypes.c_uint32, 1880), ('db_occlusion_count1_hi_15', ctypes.c_uint32, 1884), ('db_occlusion_count2_low_15', ctypes.c_uint32, 1888), ('db_occlusion_count2_hi_15', ctypes.c_uint32, 1892), ('db_occlusion_count3_low_15', ctypes.c_uint32, 1896), ('db_occlusion_count3_hi_15', ctypes.c_uint32, 1900), ('db_occlusion_count0_low_16', ctypes.c_uint32, 1904), ('db_occlusion_count0_hi_16', ctypes.c_uint32, 1908), ('db_occlusion_count1_low_16', ctypes.c_uint32, 1912), ('db_occlusion_count1_hi_16', ctypes.c_uint32, 1916), ('db_occlusion_count2_low_16', ctypes.c_uint32, 1920), ('db_occlusion_count2_hi_16', ctypes.c_uint32, 1924), ('db_occlusion_count3_low_16', ctypes.c_uint32, 1928), ('db_occlusion_count3_hi_16', ctypes.c_uint32, 1932), ('db_occlusion_count0_low_17', ctypes.c_uint32, 1936), ('db_occlusion_count0_hi_17', ctypes.c_uint32, 1940), ('db_occlusion_count1_low_17', ctypes.c_uint32, 1944), ('db_occlusion_count1_hi_17', ctypes.c_uint32, 1948), ('db_occlusion_count2_low_17', ctypes.c_uint32, 1952), ('db_occlusion_count2_hi_17', ctypes.c_uint32, 1956), ('db_occlusion_count3_low_17', ctypes.c_uint32, 1960), ('db_occlusion_count3_hi_17', ctypes.c_uint32, 1964), ('reserved_492', ctypes.c_uint32, 1968), ('reserved_493', ctypes.c_uint32, 1972), ('reserved_494', ctypes.c_uint32, 1976), ('reserved_495', ctypes.c_uint32, 1980), ('reserved_496', ctypes.c_uint32, 1984), ('reserved_497', ctypes.c_uint32, 1988), ('reserved_498', ctypes.c_uint32, 1992), ('reserved_499', ctypes.c_uint32, 1996), ('reserved_500', ctypes.c_uint32, 2000), ('reserved_501', ctypes.c_uint32, 2004), ('reserved_502', ctypes.c_uint32, 2008), ('reserved_503', ctypes.c_uint32, 2012), ('reserved_504', ctypes.c_uint32, 2016), ('reserved_505', ctypes.c_uint32, 2020), ('reserved_506', ctypes.c_uint32, 2024), ('reserved_507', ctypes.c_uint32, 2028), ('reserved_508', ctypes.c_uint32, 2032), ('reserved_509', ctypes.c_uint32, 2036), ('reserved_510', ctypes.c_uint32, 2040), ('reserved_511', ctypes.c_uint32, 2044)])
@c.record
class struct_v11_sdma_mqd(c.Struct):
  SIZE = 512
  sdmax_rlcx_rb_cntl: int
  sdmax_rlcx_rb_base: int
  sdmax_rlcx_rb_base_hi: int
  sdmax_rlcx_rb_rptr: int
  sdmax_rlcx_rb_rptr_hi: int
  sdmax_rlcx_rb_wptr: int
  sdmax_rlcx_rb_wptr_hi: int
  sdmax_rlcx_rb_rptr_addr_hi: int
  sdmax_rlcx_rb_rptr_addr_lo: int
  sdmax_rlcx_ib_cntl: int
  sdmax_rlcx_ib_rptr: int
  sdmax_rlcx_ib_offset: int
  sdmax_rlcx_ib_base_lo: int
  sdmax_rlcx_ib_base_hi: int
  sdmax_rlcx_ib_size: int
  sdmax_rlcx_skip_cntl: int
  sdmax_rlcx_context_status: int
  sdmax_rlcx_doorbell: int
  sdmax_rlcx_doorbell_log: int
  sdmax_rlcx_doorbell_offset: int
  sdmax_rlcx_csa_addr_lo: int
  sdmax_rlcx_csa_addr_hi: int
  sdmax_rlcx_sched_cntl: int
  sdmax_rlcx_ib_sub_remain: int
  sdmax_rlcx_preempt: int
  sdmax_rlcx_dummy_reg: int
  sdmax_rlcx_rb_wptr_poll_addr_hi: int
  sdmax_rlcx_rb_wptr_poll_addr_lo: int
  sdmax_rlcx_rb_aql_cntl: int
  sdmax_rlcx_minor_ptr_update: int
  sdmax_rlcx_rb_preempt: int
  sdmax_rlcx_midcmd_data0: int
  sdmax_rlcx_midcmd_data1: int
  sdmax_rlcx_midcmd_data2: int
  sdmax_rlcx_midcmd_data3: int
  sdmax_rlcx_midcmd_data4: int
  sdmax_rlcx_midcmd_data5: int
  sdmax_rlcx_midcmd_data6: int
  sdmax_rlcx_midcmd_data7: int
  sdmax_rlcx_midcmd_data8: int
  sdmax_rlcx_midcmd_data9: int
  sdmax_rlcx_midcmd_data10: int
  sdmax_rlcx_midcmd_cntl: int
  sdmax_rlcx_f32_dbg0: int
  sdmax_rlcx_f32_dbg1: int
  reserved_45: int
  reserved_46: int
  reserved_47: int
  reserved_48: int
  reserved_49: int
  reserved_50: int
  reserved_51: int
  reserved_52: int
  reserved_53: int
  reserved_54: int
  reserved_55: int
  reserved_56: int
  reserved_57: int
  reserved_58: int
  reserved_59: int
  reserved_60: int
  reserved_61: int
  reserved_62: int
  reserved_63: int
  reserved_64: int
  reserved_65: int
  reserved_66: int
  reserved_67: int
  reserved_68: int
  reserved_69: int
  reserved_70: int
  reserved_71: int
  reserved_72: int
  reserved_73: int
  reserved_74: int
  reserved_75: int
  reserved_76: int
  reserved_77: int
  reserved_78: int
  reserved_79: int
  reserved_80: int
  reserved_81: int
  reserved_82: int
  reserved_83: int
  reserved_84: int
  reserved_85: int
  reserved_86: int
  reserved_87: int
  reserved_88: int
  reserved_89: int
  reserved_90: int
  reserved_91: int
  reserved_92: int
  reserved_93: int
  reserved_94: int
  reserved_95: int
  reserved_96: int
  reserved_97: int
  reserved_98: int
  reserved_99: int
  reserved_100: int
  reserved_101: int
  reserved_102: int
  reserved_103: int
  reserved_104: int
  reserved_105: int
  reserved_106: int
  reserved_107: int
  reserved_108: int
  reserved_109: int
  reserved_110: int
  reserved_111: int
  reserved_112: int
  reserved_113: int
  reserved_114: int
  reserved_115: int
  reserved_116: int
  reserved_117: int
  reserved_118: int
  reserved_119: int
  reserved_120: int
  reserved_121: int
  reserved_122: int
  reserved_123: int
  reserved_124: int
  reserved_125: int
  sdma_engine_id: int
  sdma_queue_id: int
struct_v11_sdma_mqd.register_fields([('sdmax_rlcx_rb_cntl', ctypes.c_uint32, 0), ('sdmax_rlcx_rb_base', ctypes.c_uint32, 4), ('sdmax_rlcx_rb_base_hi', ctypes.c_uint32, 8), ('sdmax_rlcx_rb_rptr', ctypes.c_uint32, 12), ('sdmax_rlcx_rb_rptr_hi', ctypes.c_uint32, 16), ('sdmax_rlcx_rb_wptr', ctypes.c_uint32, 20), ('sdmax_rlcx_rb_wptr_hi', ctypes.c_uint32, 24), ('sdmax_rlcx_rb_rptr_addr_hi', ctypes.c_uint32, 28), ('sdmax_rlcx_rb_rptr_addr_lo', ctypes.c_uint32, 32), ('sdmax_rlcx_ib_cntl', ctypes.c_uint32, 36), ('sdmax_rlcx_ib_rptr', ctypes.c_uint32, 40), ('sdmax_rlcx_ib_offset', ctypes.c_uint32, 44), ('sdmax_rlcx_ib_base_lo', ctypes.c_uint32, 48), ('sdmax_rlcx_ib_base_hi', ctypes.c_uint32, 52), ('sdmax_rlcx_ib_size', ctypes.c_uint32, 56), ('sdmax_rlcx_skip_cntl', ctypes.c_uint32, 60), ('sdmax_rlcx_context_status', ctypes.c_uint32, 64), ('sdmax_rlcx_doorbell', ctypes.c_uint32, 68), ('sdmax_rlcx_doorbell_log', ctypes.c_uint32, 72), ('sdmax_rlcx_doorbell_offset', ctypes.c_uint32, 76), ('sdmax_rlcx_csa_addr_lo', ctypes.c_uint32, 80), ('sdmax_rlcx_csa_addr_hi', ctypes.c_uint32, 84), ('sdmax_rlcx_sched_cntl', ctypes.c_uint32, 88), ('sdmax_rlcx_ib_sub_remain', ctypes.c_uint32, 92), ('sdmax_rlcx_preempt', ctypes.c_uint32, 96), ('sdmax_rlcx_dummy_reg', ctypes.c_uint32, 100), ('sdmax_rlcx_rb_wptr_poll_addr_hi', ctypes.c_uint32, 104), ('sdmax_rlcx_rb_wptr_poll_addr_lo', ctypes.c_uint32, 108), ('sdmax_rlcx_rb_aql_cntl', ctypes.c_uint32, 112), ('sdmax_rlcx_minor_ptr_update', ctypes.c_uint32, 116), ('sdmax_rlcx_rb_preempt', ctypes.c_uint32, 120), ('sdmax_rlcx_midcmd_data0', ctypes.c_uint32, 124), ('sdmax_rlcx_midcmd_data1', ctypes.c_uint32, 128), ('sdmax_rlcx_midcmd_data2', ctypes.c_uint32, 132), ('sdmax_rlcx_midcmd_data3', ctypes.c_uint32, 136), ('sdmax_rlcx_midcmd_data4', ctypes.c_uint32, 140), ('sdmax_rlcx_midcmd_data5', ctypes.c_uint32, 144), ('sdmax_rlcx_midcmd_data6', ctypes.c_uint32, 148), ('sdmax_rlcx_midcmd_data7', ctypes.c_uint32, 152), ('sdmax_rlcx_midcmd_data8', ctypes.c_uint32, 156), ('sdmax_rlcx_midcmd_data9', ctypes.c_uint32, 160), ('sdmax_rlcx_midcmd_data10', ctypes.c_uint32, 164), ('sdmax_rlcx_midcmd_cntl', ctypes.c_uint32, 168), ('sdmax_rlcx_f32_dbg0', ctypes.c_uint32, 172), ('sdmax_rlcx_f32_dbg1', ctypes.c_uint32, 176), ('reserved_45', ctypes.c_uint32, 180), ('reserved_46', ctypes.c_uint32, 184), ('reserved_47', ctypes.c_uint32, 188), ('reserved_48', ctypes.c_uint32, 192), ('reserved_49', ctypes.c_uint32, 196), ('reserved_50', ctypes.c_uint32, 200), ('reserved_51', ctypes.c_uint32, 204), ('reserved_52', ctypes.c_uint32, 208), ('reserved_53', ctypes.c_uint32, 212), ('reserved_54', ctypes.c_uint32, 216), ('reserved_55', ctypes.c_uint32, 220), ('reserved_56', ctypes.c_uint32, 224), ('reserved_57', ctypes.c_uint32, 228), ('reserved_58', ctypes.c_uint32, 232), ('reserved_59', ctypes.c_uint32, 236), ('reserved_60', ctypes.c_uint32, 240), ('reserved_61', ctypes.c_uint32, 244), ('reserved_62', ctypes.c_uint32, 248), ('reserved_63', ctypes.c_uint32, 252), ('reserved_64', ctypes.c_uint32, 256), ('reserved_65', ctypes.c_uint32, 260), ('reserved_66', ctypes.c_uint32, 264), ('reserved_67', ctypes.c_uint32, 268), ('reserved_68', ctypes.c_uint32, 272), ('reserved_69', ctypes.c_uint32, 276), ('reserved_70', ctypes.c_uint32, 280), ('reserved_71', ctypes.c_uint32, 284), ('reserved_72', ctypes.c_uint32, 288), ('reserved_73', ctypes.c_uint32, 292), ('reserved_74', ctypes.c_uint32, 296), ('reserved_75', ctypes.c_uint32, 300), ('reserved_76', ctypes.c_uint32, 304), ('reserved_77', ctypes.c_uint32, 308), ('reserved_78', ctypes.c_uint32, 312), ('reserved_79', ctypes.c_uint32, 316), ('reserved_80', ctypes.c_uint32, 320), ('reserved_81', ctypes.c_uint32, 324), ('reserved_82', ctypes.c_uint32, 328), ('reserved_83', ctypes.c_uint32, 332), ('reserved_84', ctypes.c_uint32, 336), ('reserved_85', ctypes.c_uint32, 340), ('reserved_86', ctypes.c_uint32, 344), ('reserved_87', ctypes.c_uint32, 348), ('reserved_88', ctypes.c_uint32, 352), ('reserved_89', ctypes.c_uint32, 356), ('reserved_90', ctypes.c_uint32, 360), ('reserved_91', ctypes.c_uint32, 364), ('reserved_92', ctypes.c_uint32, 368), ('reserved_93', ctypes.c_uint32, 372), ('reserved_94', ctypes.c_uint32, 376), ('reserved_95', ctypes.c_uint32, 380), ('reserved_96', ctypes.c_uint32, 384), ('reserved_97', ctypes.c_uint32, 388), ('reserved_98', ctypes.c_uint32, 392), ('reserved_99', ctypes.c_uint32, 396), ('reserved_100', ctypes.c_uint32, 400), ('reserved_101', ctypes.c_uint32, 404), ('reserved_102', ctypes.c_uint32, 408), ('reserved_103', ctypes.c_uint32, 412), ('reserved_104', ctypes.c_uint32, 416), ('reserved_105', ctypes.c_uint32, 420), ('reserved_106', ctypes.c_uint32, 424), ('reserved_107', ctypes.c_uint32, 428), ('reserved_108', ctypes.c_uint32, 432), ('reserved_109', ctypes.c_uint32, 436), ('reserved_110', ctypes.c_uint32, 440), ('reserved_111', ctypes.c_uint32, 444), ('reserved_112', ctypes.c_uint32, 448), ('reserved_113', ctypes.c_uint32, 452), ('reserved_114', ctypes.c_uint32, 456), ('reserved_115', ctypes.c_uint32, 460), ('reserved_116', ctypes.c_uint32, 464), ('reserved_117', ctypes.c_uint32, 468), ('reserved_118', ctypes.c_uint32, 472), ('reserved_119', ctypes.c_uint32, 476), ('reserved_120', ctypes.c_uint32, 480), ('reserved_121', ctypes.c_uint32, 484), ('reserved_122', ctypes.c_uint32, 488), ('reserved_123', ctypes.c_uint32, 492), ('reserved_124', ctypes.c_uint32, 496), ('reserved_125', ctypes.c_uint32, 500), ('sdma_engine_id', ctypes.c_uint32, 504), ('sdma_queue_id', ctypes.c_uint32, 508)])
@c.record
class struct_v11_compute_mqd(c.Struct):
  SIZE = 2048
  header: int
  compute_dispatch_initiator: int
  compute_dim_x: int
  compute_dim_y: int
  compute_dim_z: int
  compute_start_x: int
  compute_start_y: int
  compute_start_z: int
  compute_num_thread_x: int
  compute_num_thread_y: int
  compute_num_thread_z: int
  compute_pipelinestat_enable: int
  compute_perfcount_enable: int
  compute_pgm_lo: int
  compute_pgm_hi: int
  compute_dispatch_pkt_addr_lo: int
  compute_dispatch_pkt_addr_hi: int
  compute_dispatch_scratch_base_lo: int
  compute_dispatch_scratch_base_hi: int
  compute_pgm_rsrc1: int
  compute_pgm_rsrc2: int
  compute_vmid: int
  compute_resource_limits: int
  compute_static_thread_mgmt_se0: int
  compute_static_thread_mgmt_se1: int
  compute_tmpring_size: int
  compute_static_thread_mgmt_se2: int
  compute_static_thread_mgmt_se3: int
  compute_restart_x: int
  compute_restart_y: int
  compute_restart_z: int
  compute_thread_trace_enable: int
  compute_misc_reserved: int
  compute_dispatch_id: int
  compute_threadgroup_id: int
  compute_req_ctrl: int
  reserved_36: int
  compute_user_accum_0: int
  compute_user_accum_1: int
  compute_user_accum_2: int
  compute_user_accum_3: int
  compute_pgm_rsrc3: int
  compute_ddid_index: int
  compute_shader_chksum: int
  compute_static_thread_mgmt_se4: int
  compute_static_thread_mgmt_se5: int
  compute_static_thread_mgmt_se6: int
  compute_static_thread_mgmt_se7: int
  compute_dispatch_interleave: int
  compute_relaunch: int
  compute_wave_restore_addr_lo: int
  compute_wave_restore_addr_hi: int
  compute_wave_restore_control: int
  reserved_53: int
  reserved_54: int
  reserved_55: int
  reserved_56: int
  reserved_57: int
  reserved_58: int
  reserved_59: int
  reserved_60: int
  reserved_61: int
  reserved_62: int
  reserved_63: int
  reserved_64: int
  compute_user_data_0: int
  compute_user_data_1: int
  compute_user_data_2: int
  compute_user_data_3: int
  compute_user_data_4: int
  compute_user_data_5: int
  compute_user_data_6: int
  compute_user_data_7: int
  compute_user_data_8: int
  compute_user_data_9: int
  compute_user_data_10: int
  compute_user_data_11: int
  compute_user_data_12: int
  compute_user_data_13: int
  compute_user_data_14: int
  compute_user_data_15: int
  cp_compute_csinvoc_count_lo: int
  cp_compute_csinvoc_count_hi: int
  reserved_83: int
  reserved_84: int
  reserved_85: int
  cp_mqd_query_time_lo: int
  cp_mqd_query_time_hi: int
  cp_mqd_connect_start_time_lo: int
  cp_mqd_connect_start_time_hi: int
  cp_mqd_connect_end_time_lo: int
  cp_mqd_connect_end_time_hi: int
  cp_mqd_connect_end_wf_count: int
  cp_mqd_connect_end_pq_rptr: int
  cp_mqd_connect_end_pq_wptr: int
  cp_mqd_connect_end_ib_rptr: int
  cp_mqd_readindex_lo: int
  cp_mqd_readindex_hi: int
  cp_mqd_save_start_time_lo: int
  cp_mqd_save_start_time_hi: int
  cp_mqd_save_end_time_lo: int
  cp_mqd_save_end_time_hi: int
  cp_mqd_restore_start_time_lo: int
  cp_mqd_restore_start_time_hi: int
  cp_mqd_restore_end_time_lo: int
  cp_mqd_restore_end_time_hi: int
  disable_queue: int
  reserved_107: int
  gds_cs_ctxsw_cnt0: int
  gds_cs_ctxsw_cnt1: int
  gds_cs_ctxsw_cnt2: int
  gds_cs_ctxsw_cnt3: int
  reserved_112: int
  reserved_113: int
  cp_pq_exe_status_lo: int
  cp_pq_exe_status_hi: int
  cp_packet_id_lo: int
  cp_packet_id_hi: int
  cp_packet_exe_status_lo: int
  cp_packet_exe_status_hi: int
  gds_save_base_addr_lo: int
  gds_save_base_addr_hi: int
  gds_save_mask_lo: int
  gds_save_mask_hi: int
  ctx_save_base_addr_lo: int
  ctx_save_base_addr_hi: int
  reserved_126: int
  reserved_127: int
  cp_mqd_base_addr_lo: int
  cp_mqd_base_addr_hi: int
  cp_hqd_active: int
  cp_hqd_vmid: int
  cp_hqd_persistent_state: int
  cp_hqd_pipe_priority: int
  cp_hqd_queue_priority: int
  cp_hqd_quantum: int
  cp_hqd_pq_base_lo: int
  cp_hqd_pq_base_hi: int
  cp_hqd_pq_rptr: int
  cp_hqd_pq_rptr_report_addr_lo: int
  cp_hqd_pq_rptr_report_addr_hi: int
  cp_hqd_pq_wptr_poll_addr_lo: int
  cp_hqd_pq_wptr_poll_addr_hi: int
  cp_hqd_pq_doorbell_control: int
  reserved_144: int
  cp_hqd_pq_control: int
  cp_hqd_ib_base_addr_lo: int
  cp_hqd_ib_base_addr_hi: int
  cp_hqd_ib_rptr: int
  cp_hqd_ib_control: int
  cp_hqd_iq_timer: int
  cp_hqd_iq_rptr: int
  cp_hqd_dequeue_request: int
  cp_hqd_dma_offload: int
  cp_hqd_sema_cmd: int
  cp_hqd_msg_type: int
  cp_hqd_atomic0_preop_lo: int
  cp_hqd_atomic0_preop_hi: int
  cp_hqd_atomic1_preop_lo: int
  cp_hqd_atomic1_preop_hi: int
  cp_hqd_hq_status0: int
  cp_hqd_hq_control0: int
  cp_mqd_control: int
  cp_hqd_hq_status1: int
  cp_hqd_hq_control1: int
  cp_hqd_eop_base_addr_lo: int
  cp_hqd_eop_base_addr_hi: int
  cp_hqd_eop_control: int
  cp_hqd_eop_rptr: int
  cp_hqd_eop_wptr: int
  cp_hqd_eop_done_events: int
  cp_hqd_ctx_save_base_addr_lo: int
  cp_hqd_ctx_save_base_addr_hi: int
  cp_hqd_ctx_save_control: int
  cp_hqd_cntl_stack_offset: int
  cp_hqd_cntl_stack_size: int
  cp_hqd_wg_state_offset: int
  cp_hqd_ctx_save_size: int
  cp_hqd_gds_resource_state: int
  cp_hqd_error: int
  cp_hqd_eop_wptr_mem: int
  cp_hqd_aql_control: int
  cp_hqd_pq_wptr_lo: int
  cp_hqd_pq_wptr_hi: int
  reserved_184: int
  reserved_185: int
  reserved_186: int
  reserved_187: int
  reserved_188: int
  reserved_189: int
  reserved_190: int
  reserved_191: int
  iqtimer_pkt_header: int
  iqtimer_pkt_dw0: int
  iqtimer_pkt_dw1: int
  iqtimer_pkt_dw2: int
  iqtimer_pkt_dw3: int
  iqtimer_pkt_dw4: int
  iqtimer_pkt_dw5: int
  iqtimer_pkt_dw6: int
  iqtimer_pkt_dw7: int
  iqtimer_pkt_dw8: int
  iqtimer_pkt_dw9: int
  iqtimer_pkt_dw10: int
  iqtimer_pkt_dw11: int
  iqtimer_pkt_dw12: int
  iqtimer_pkt_dw13: int
  iqtimer_pkt_dw14: int
  iqtimer_pkt_dw15: int
  iqtimer_pkt_dw16: int
  iqtimer_pkt_dw17: int
  iqtimer_pkt_dw18: int
  iqtimer_pkt_dw19: int
  iqtimer_pkt_dw20: int
  iqtimer_pkt_dw21: int
  iqtimer_pkt_dw22: int
  iqtimer_pkt_dw23: int
  iqtimer_pkt_dw24: int
  iqtimer_pkt_dw25: int
  iqtimer_pkt_dw26: int
  iqtimer_pkt_dw27: int
  iqtimer_pkt_dw28: int
  iqtimer_pkt_dw29: int
  iqtimer_pkt_dw30: int
  iqtimer_pkt_dw31: int
  reserved_225: int
  reserved_226: int
  reserved_227: int
  set_resources_header: int
  set_resources_dw1: int
  set_resources_dw2: int
  set_resources_dw3: int
  set_resources_dw4: int
  set_resources_dw5: int
  set_resources_dw6: int
  set_resources_dw7: int
  reserved_236: int
  reserved_237: int
  reserved_238: int
  reserved_239: int
  queue_doorbell_id0: int
  queue_doorbell_id1: int
  queue_doorbell_id2: int
  queue_doorbell_id3: int
  queue_doorbell_id4: int
  queue_doorbell_id5: int
  queue_doorbell_id6: int
  queue_doorbell_id7: int
  queue_doorbell_id8: int
  queue_doorbell_id9: int
  queue_doorbell_id10: int
  queue_doorbell_id11: int
  queue_doorbell_id12: int
  queue_doorbell_id13: int
  queue_doorbell_id14: int
  queue_doorbell_id15: int
  control_buf_addr_lo: int
  control_buf_addr_hi: int
  control_buf_wptr_lo: int
  control_buf_wptr_hi: int
  control_buf_dptr_lo: int
  control_buf_dptr_hi: int
  control_buf_num_entries: int
  draw_ring_addr_lo: int
  draw_ring_addr_hi: int
  reserved_265: int
  reserved_266: int
  reserved_267: int
  reserved_268: int
  reserved_269: int
  reserved_270: int
  reserved_271: int
  reserved_272: int
  reserved_273: int
  reserved_274: int
  reserved_275: int
  reserved_276: int
  reserved_277: int
  reserved_278: int
  reserved_279: int
  reserved_280: int
  reserved_281: int
  reserved_282: int
  reserved_283: int
  reserved_284: int
  reserved_285: int
  reserved_286: int
  reserved_287: int
  reserved_288: int
  reserved_289: int
  reserved_290: int
  reserved_291: int
  reserved_292: int
  reserved_293: int
  reserved_294: int
  reserved_295: int
  reserved_296: int
  reserved_297: int
  reserved_298: int
  reserved_299: int
  reserved_300: int
  reserved_301: int
  reserved_302: int
  reserved_303: int
  reserved_304: int
  reserved_305: int
  reserved_306: int
  reserved_307: int
  reserved_308: int
  reserved_309: int
  reserved_310: int
  reserved_311: int
  reserved_312: int
  reserved_313: int
  reserved_314: int
  reserved_315: int
  reserved_316: int
  reserved_317: int
  reserved_318: int
  reserved_319: int
  reserved_320: int
  reserved_321: int
  reserved_322: int
  reserved_323: int
  reserved_324: int
  reserved_325: int
  reserved_326: int
  reserved_327: int
  reserved_328: int
  reserved_329: int
  reserved_330: int
  reserved_331: int
  reserved_332: int
  reserved_333: int
  reserved_334: int
  reserved_335: int
  reserved_336: int
  reserved_337: int
  reserved_338: int
  reserved_339: int
  reserved_340: int
  reserved_341: int
  reserved_342: int
  reserved_343: int
  reserved_344: int
  reserved_345: int
  reserved_346: int
  reserved_347: int
  reserved_348: int
  reserved_349: int
  reserved_350: int
  reserved_351: int
  reserved_352: int
  reserved_353: int
  reserved_354: int
  reserved_355: int
  reserved_356: int
  reserved_357: int
  reserved_358: int
  reserved_359: int
  reserved_360: int
  reserved_361: int
  reserved_362: int
  reserved_363: int
  reserved_364: int
  reserved_365: int
  reserved_366: int
  reserved_367: int
  reserved_368: int
  reserved_369: int
  reserved_370: int
  reserved_371: int
  reserved_372: int
  reserved_373: int
  reserved_374: int
  reserved_375: int
  reserved_376: int
  reserved_377: int
  reserved_378: int
  reserved_379: int
  reserved_380: int
  reserved_381: int
  reserved_382: int
  reserved_383: int
  reserved_384: int
  reserved_385: int
  reserved_386: int
  reserved_387: int
  reserved_388: int
  reserved_389: int
  reserved_390: int
  reserved_391: int
  reserved_392: int
  reserved_393: int
  reserved_394: int
  reserved_395: int
  reserved_396: int
  reserved_397: int
  reserved_398: int
  reserved_399: int
  reserved_400: int
  reserved_401: int
  reserved_402: int
  reserved_403: int
  reserved_404: int
  reserved_405: int
  reserved_406: int
  reserved_407: int
  reserved_408: int
  reserved_409: int
  reserved_410: int
  reserved_411: int
  reserved_412: int
  reserved_413: int
  reserved_414: int
  reserved_415: int
  reserved_416: int
  reserved_417: int
  reserved_418: int
  reserved_419: int
  reserved_420: int
  reserved_421: int
  reserved_422: int
  reserved_423: int
  reserved_424: int
  reserved_425: int
  reserved_426: int
  reserved_427: int
  reserved_428: int
  reserved_429: int
  reserved_430: int
  reserved_431: int
  reserved_432: int
  reserved_433: int
  reserved_434: int
  reserved_435: int
  reserved_436: int
  reserved_437: int
  reserved_438: int
  reserved_439: int
  reserved_440: int
  reserved_441: int
  reserved_442: int
  reserved_443: int
  reserved_444: int
  reserved_445: int
  reserved_446: int
  reserved_447: int
  gws_0_val: int
  gws_1_val: int
  gws_2_val: int
  gws_3_val: int
  gws_4_val: int
  gws_5_val: int
  gws_6_val: int
  gws_7_val: int
  gws_8_val: int
  gws_9_val: int
  gws_10_val: int
  gws_11_val: int
  gws_12_val: int
  gws_13_val: int
  gws_14_val: int
  gws_15_val: int
  gws_16_val: int
  gws_17_val: int
  gws_18_val: int
  gws_19_val: int
  gws_20_val: int
  gws_21_val: int
  gws_22_val: int
  gws_23_val: int
  gws_24_val: int
  gws_25_val: int
  gws_26_val: int
  gws_27_val: int
  gws_28_val: int
  gws_29_val: int
  gws_30_val: int
  gws_31_val: int
  gws_32_val: int
  gws_33_val: int
  gws_34_val: int
  gws_35_val: int
  gws_36_val: int
  gws_37_val: int
  gws_38_val: int
  gws_39_val: int
  gws_40_val: int
  gws_41_val: int
  gws_42_val: int
  gws_43_val: int
  gws_44_val: int
  gws_45_val: int
  gws_46_val: int
  gws_47_val: int
  gws_48_val: int
  gws_49_val: int
  gws_50_val: int
  gws_51_val: int
  gws_52_val: int
  gws_53_val: int
  gws_54_val: int
  gws_55_val: int
  gws_56_val: int
  gws_57_val: int
  gws_58_val: int
  gws_59_val: int
  gws_60_val: int
  gws_61_val: int
  gws_62_val: int
  gws_63_val: int
struct_v11_compute_mqd.register_fields([('header', ctypes.c_uint32, 0), ('compute_dispatch_initiator', ctypes.c_uint32, 4), ('compute_dim_x', ctypes.c_uint32, 8), ('compute_dim_y', ctypes.c_uint32, 12), ('compute_dim_z', ctypes.c_uint32, 16), ('compute_start_x', ctypes.c_uint32, 20), ('compute_start_y', ctypes.c_uint32, 24), ('compute_start_z', ctypes.c_uint32, 28), ('compute_num_thread_x', ctypes.c_uint32, 32), ('compute_num_thread_y', ctypes.c_uint32, 36), ('compute_num_thread_z', ctypes.c_uint32, 40), ('compute_pipelinestat_enable', ctypes.c_uint32, 44), ('compute_perfcount_enable', ctypes.c_uint32, 48), ('compute_pgm_lo', ctypes.c_uint32, 52), ('compute_pgm_hi', ctypes.c_uint32, 56), ('compute_dispatch_pkt_addr_lo', ctypes.c_uint32, 60), ('compute_dispatch_pkt_addr_hi', ctypes.c_uint32, 64), ('compute_dispatch_scratch_base_lo', ctypes.c_uint32, 68), ('compute_dispatch_scratch_base_hi', ctypes.c_uint32, 72), ('compute_pgm_rsrc1', ctypes.c_uint32, 76), ('compute_pgm_rsrc2', ctypes.c_uint32, 80), ('compute_vmid', ctypes.c_uint32, 84), ('compute_resource_limits', ctypes.c_uint32, 88), ('compute_static_thread_mgmt_se0', ctypes.c_uint32, 92), ('compute_static_thread_mgmt_se1', ctypes.c_uint32, 96), ('compute_tmpring_size', ctypes.c_uint32, 100), ('compute_static_thread_mgmt_se2', ctypes.c_uint32, 104), ('compute_static_thread_mgmt_se3', ctypes.c_uint32, 108), ('compute_restart_x', ctypes.c_uint32, 112), ('compute_restart_y', ctypes.c_uint32, 116), ('compute_restart_z', ctypes.c_uint32, 120), ('compute_thread_trace_enable', ctypes.c_uint32, 124), ('compute_misc_reserved', ctypes.c_uint32, 128), ('compute_dispatch_id', ctypes.c_uint32, 132), ('compute_threadgroup_id', ctypes.c_uint32, 136), ('compute_req_ctrl', ctypes.c_uint32, 140), ('reserved_36', ctypes.c_uint32, 144), ('compute_user_accum_0', ctypes.c_uint32, 148), ('compute_user_accum_1', ctypes.c_uint32, 152), ('compute_user_accum_2', ctypes.c_uint32, 156), ('compute_user_accum_3', ctypes.c_uint32, 160), ('compute_pgm_rsrc3', ctypes.c_uint32, 164), ('compute_ddid_index', ctypes.c_uint32, 168), ('compute_shader_chksum', ctypes.c_uint32, 172), ('compute_static_thread_mgmt_se4', ctypes.c_uint32, 176), ('compute_static_thread_mgmt_se5', ctypes.c_uint32, 180), ('compute_static_thread_mgmt_se6', ctypes.c_uint32, 184), ('compute_static_thread_mgmt_se7', ctypes.c_uint32, 188), ('compute_dispatch_interleave', ctypes.c_uint32, 192), ('compute_relaunch', ctypes.c_uint32, 196), ('compute_wave_restore_addr_lo', ctypes.c_uint32, 200), ('compute_wave_restore_addr_hi', ctypes.c_uint32, 204), ('compute_wave_restore_control', ctypes.c_uint32, 208), ('reserved_53', ctypes.c_uint32, 212), ('reserved_54', ctypes.c_uint32, 216), ('reserved_55', ctypes.c_uint32, 220), ('reserved_56', ctypes.c_uint32, 224), ('reserved_57', ctypes.c_uint32, 228), ('reserved_58', ctypes.c_uint32, 232), ('reserved_59', ctypes.c_uint32, 236), ('reserved_60', ctypes.c_uint32, 240), ('reserved_61', ctypes.c_uint32, 244), ('reserved_62', ctypes.c_uint32, 248), ('reserved_63', ctypes.c_uint32, 252), ('reserved_64', ctypes.c_uint32, 256), ('compute_user_data_0', ctypes.c_uint32, 260), ('compute_user_data_1', ctypes.c_uint32, 264), ('compute_user_data_2', ctypes.c_uint32, 268), ('compute_user_data_3', ctypes.c_uint32, 272), ('compute_user_data_4', ctypes.c_uint32, 276), ('compute_user_data_5', ctypes.c_uint32, 280), ('compute_user_data_6', ctypes.c_uint32, 284), ('compute_user_data_7', ctypes.c_uint32, 288), ('compute_user_data_8', ctypes.c_uint32, 292), ('compute_user_data_9', ctypes.c_uint32, 296), ('compute_user_data_10', ctypes.c_uint32, 300), ('compute_user_data_11', ctypes.c_uint32, 304), ('compute_user_data_12', ctypes.c_uint32, 308), ('compute_user_data_13', ctypes.c_uint32, 312), ('compute_user_data_14', ctypes.c_uint32, 316), ('compute_user_data_15', ctypes.c_uint32, 320), ('cp_compute_csinvoc_count_lo', ctypes.c_uint32, 324), ('cp_compute_csinvoc_count_hi', ctypes.c_uint32, 328), ('reserved_83', ctypes.c_uint32, 332), ('reserved_84', ctypes.c_uint32, 336), ('reserved_85', ctypes.c_uint32, 340), ('cp_mqd_query_time_lo', ctypes.c_uint32, 344), ('cp_mqd_query_time_hi', ctypes.c_uint32, 348), ('cp_mqd_connect_start_time_lo', ctypes.c_uint32, 352), ('cp_mqd_connect_start_time_hi', ctypes.c_uint32, 356), ('cp_mqd_connect_end_time_lo', ctypes.c_uint32, 360), ('cp_mqd_connect_end_time_hi', ctypes.c_uint32, 364), ('cp_mqd_connect_end_wf_count', ctypes.c_uint32, 368), ('cp_mqd_connect_end_pq_rptr', ctypes.c_uint32, 372), ('cp_mqd_connect_end_pq_wptr', ctypes.c_uint32, 376), ('cp_mqd_connect_end_ib_rptr', ctypes.c_uint32, 380), ('cp_mqd_readindex_lo', ctypes.c_uint32, 384), ('cp_mqd_readindex_hi', ctypes.c_uint32, 388), ('cp_mqd_save_start_time_lo', ctypes.c_uint32, 392), ('cp_mqd_save_start_time_hi', ctypes.c_uint32, 396), ('cp_mqd_save_end_time_lo', ctypes.c_uint32, 400), ('cp_mqd_save_end_time_hi', ctypes.c_uint32, 404), ('cp_mqd_restore_start_time_lo', ctypes.c_uint32, 408), ('cp_mqd_restore_start_time_hi', ctypes.c_uint32, 412), ('cp_mqd_restore_end_time_lo', ctypes.c_uint32, 416), ('cp_mqd_restore_end_time_hi', ctypes.c_uint32, 420), ('disable_queue', ctypes.c_uint32, 424), ('reserved_107', ctypes.c_uint32, 428), ('gds_cs_ctxsw_cnt0', ctypes.c_uint32, 432), ('gds_cs_ctxsw_cnt1', ctypes.c_uint32, 436), ('gds_cs_ctxsw_cnt2', ctypes.c_uint32, 440), ('gds_cs_ctxsw_cnt3', ctypes.c_uint32, 444), ('reserved_112', ctypes.c_uint32, 448), ('reserved_113', ctypes.c_uint32, 452), ('cp_pq_exe_status_lo', ctypes.c_uint32, 456), ('cp_pq_exe_status_hi', ctypes.c_uint32, 460), ('cp_packet_id_lo', ctypes.c_uint32, 464), ('cp_packet_id_hi', ctypes.c_uint32, 468), ('cp_packet_exe_status_lo', ctypes.c_uint32, 472), ('cp_packet_exe_status_hi', ctypes.c_uint32, 476), ('gds_save_base_addr_lo', ctypes.c_uint32, 480), ('gds_save_base_addr_hi', ctypes.c_uint32, 484), ('gds_save_mask_lo', ctypes.c_uint32, 488), ('gds_save_mask_hi', ctypes.c_uint32, 492), ('ctx_save_base_addr_lo', ctypes.c_uint32, 496), ('ctx_save_base_addr_hi', ctypes.c_uint32, 500), ('reserved_126', ctypes.c_uint32, 504), ('reserved_127', ctypes.c_uint32, 508), ('cp_mqd_base_addr_lo', ctypes.c_uint32, 512), ('cp_mqd_base_addr_hi', ctypes.c_uint32, 516), ('cp_hqd_active', ctypes.c_uint32, 520), ('cp_hqd_vmid', ctypes.c_uint32, 524), ('cp_hqd_persistent_state', ctypes.c_uint32, 528), ('cp_hqd_pipe_priority', ctypes.c_uint32, 532), ('cp_hqd_queue_priority', ctypes.c_uint32, 536), ('cp_hqd_quantum', ctypes.c_uint32, 540), ('cp_hqd_pq_base_lo', ctypes.c_uint32, 544), ('cp_hqd_pq_base_hi', ctypes.c_uint32, 548), ('cp_hqd_pq_rptr', ctypes.c_uint32, 552), ('cp_hqd_pq_rptr_report_addr_lo', ctypes.c_uint32, 556), ('cp_hqd_pq_rptr_report_addr_hi', ctypes.c_uint32, 560), ('cp_hqd_pq_wptr_poll_addr_lo', ctypes.c_uint32, 564), ('cp_hqd_pq_wptr_poll_addr_hi', ctypes.c_uint32, 568), ('cp_hqd_pq_doorbell_control', ctypes.c_uint32, 572), ('reserved_144', ctypes.c_uint32, 576), ('cp_hqd_pq_control', ctypes.c_uint32, 580), ('cp_hqd_ib_base_addr_lo', ctypes.c_uint32, 584), ('cp_hqd_ib_base_addr_hi', ctypes.c_uint32, 588), ('cp_hqd_ib_rptr', ctypes.c_uint32, 592), ('cp_hqd_ib_control', ctypes.c_uint32, 596), ('cp_hqd_iq_timer', ctypes.c_uint32, 600), ('cp_hqd_iq_rptr', ctypes.c_uint32, 604), ('cp_hqd_dequeue_request', ctypes.c_uint32, 608), ('cp_hqd_dma_offload', ctypes.c_uint32, 612), ('cp_hqd_sema_cmd', ctypes.c_uint32, 616), ('cp_hqd_msg_type', ctypes.c_uint32, 620), ('cp_hqd_atomic0_preop_lo', ctypes.c_uint32, 624), ('cp_hqd_atomic0_preop_hi', ctypes.c_uint32, 628), ('cp_hqd_atomic1_preop_lo', ctypes.c_uint32, 632), ('cp_hqd_atomic1_preop_hi', ctypes.c_uint32, 636), ('cp_hqd_hq_status0', ctypes.c_uint32, 640), ('cp_hqd_hq_control0', ctypes.c_uint32, 644), ('cp_mqd_control', ctypes.c_uint32, 648), ('cp_hqd_hq_status1', ctypes.c_uint32, 652), ('cp_hqd_hq_control1', ctypes.c_uint32, 656), ('cp_hqd_eop_base_addr_lo', ctypes.c_uint32, 660), ('cp_hqd_eop_base_addr_hi', ctypes.c_uint32, 664), ('cp_hqd_eop_control', ctypes.c_uint32, 668), ('cp_hqd_eop_rptr', ctypes.c_uint32, 672), ('cp_hqd_eop_wptr', ctypes.c_uint32, 676), ('cp_hqd_eop_done_events', ctypes.c_uint32, 680), ('cp_hqd_ctx_save_base_addr_lo', ctypes.c_uint32, 684), ('cp_hqd_ctx_save_base_addr_hi', ctypes.c_uint32, 688), ('cp_hqd_ctx_save_control', ctypes.c_uint32, 692), ('cp_hqd_cntl_stack_offset', ctypes.c_uint32, 696), ('cp_hqd_cntl_stack_size', ctypes.c_uint32, 700), ('cp_hqd_wg_state_offset', ctypes.c_uint32, 704), ('cp_hqd_ctx_save_size', ctypes.c_uint32, 708), ('cp_hqd_gds_resource_state', ctypes.c_uint32, 712), ('cp_hqd_error', ctypes.c_uint32, 716), ('cp_hqd_eop_wptr_mem', ctypes.c_uint32, 720), ('cp_hqd_aql_control', ctypes.c_uint32, 724), ('cp_hqd_pq_wptr_lo', ctypes.c_uint32, 728), ('cp_hqd_pq_wptr_hi', ctypes.c_uint32, 732), ('reserved_184', ctypes.c_uint32, 736), ('reserved_185', ctypes.c_uint32, 740), ('reserved_186', ctypes.c_uint32, 744), ('reserved_187', ctypes.c_uint32, 748), ('reserved_188', ctypes.c_uint32, 752), ('reserved_189', ctypes.c_uint32, 756), ('reserved_190', ctypes.c_uint32, 760), ('reserved_191', ctypes.c_uint32, 764), ('iqtimer_pkt_header', ctypes.c_uint32, 768), ('iqtimer_pkt_dw0', ctypes.c_uint32, 772), ('iqtimer_pkt_dw1', ctypes.c_uint32, 776), ('iqtimer_pkt_dw2', ctypes.c_uint32, 780), ('iqtimer_pkt_dw3', ctypes.c_uint32, 784), ('iqtimer_pkt_dw4', ctypes.c_uint32, 788), ('iqtimer_pkt_dw5', ctypes.c_uint32, 792), ('iqtimer_pkt_dw6', ctypes.c_uint32, 796), ('iqtimer_pkt_dw7', ctypes.c_uint32, 800), ('iqtimer_pkt_dw8', ctypes.c_uint32, 804), ('iqtimer_pkt_dw9', ctypes.c_uint32, 808), ('iqtimer_pkt_dw10', ctypes.c_uint32, 812), ('iqtimer_pkt_dw11', ctypes.c_uint32, 816), ('iqtimer_pkt_dw12', ctypes.c_uint32, 820), ('iqtimer_pkt_dw13', ctypes.c_uint32, 824), ('iqtimer_pkt_dw14', ctypes.c_uint32, 828), ('iqtimer_pkt_dw15', ctypes.c_uint32, 832), ('iqtimer_pkt_dw16', ctypes.c_uint32, 836), ('iqtimer_pkt_dw17', ctypes.c_uint32, 840), ('iqtimer_pkt_dw18', ctypes.c_uint32, 844), ('iqtimer_pkt_dw19', ctypes.c_uint32, 848), ('iqtimer_pkt_dw20', ctypes.c_uint32, 852), ('iqtimer_pkt_dw21', ctypes.c_uint32, 856), ('iqtimer_pkt_dw22', ctypes.c_uint32, 860), ('iqtimer_pkt_dw23', ctypes.c_uint32, 864), ('iqtimer_pkt_dw24', ctypes.c_uint32, 868), ('iqtimer_pkt_dw25', ctypes.c_uint32, 872), ('iqtimer_pkt_dw26', ctypes.c_uint32, 876), ('iqtimer_pkt_dw27', ctypes.c_uint32, 880), ('iqtimer_pkt_dw28', ctypes.c_uint32, 884), ('iqtimer_pkt_dw29', ctypes.c_uint32, 888), ('iqtimer_pkt_dw30', ctypes.c_uint32, 892), ('iqtimer_pkt_dw31', ctypes.c_uint32, 896), ('reserved_225', ctypes.c_uint32, 900), ('reserved_226', ctypes.c_uint32, 904), ('reserved_227', ctypes.c_uint32, 908), ('set_resources_header', ctypes.c_uint32, 912), ('set_resources_dw1', ctypes.c_uint32, 916), ('set_resources_dw2', ctypes.c_uint32, 920), ('set_resources_dw3', ctypes.c_uint32, 924), ('set_resources_dw4', ctypes.c_uint32, 928), ('set_resources_dw5', ctypes.c_uint32, 932), ('set_resources_dw6', ctypes.c_uint32, 936), ('set_resources_dw7', ctypes.c_uint32, 940), ('reserved_236', ctypes.c_uint32, 944), ('reserved_237', ctypes.c_uint32, 948), ('reserved_238', ctypes.c_uint32, 952), ('reserved_239', ctypes.c_uint32, 956), ('queue_doorbell_id0', ctypes.c_uint32, 960), ('queue_doorbell_id1', ctypes.c_uint32, 964), ('queue_doorbell_id2', ctypes.c_uint32, 968), ('queue_doorbell_id3', ctypes.c_uint32, 972), ('queue_doorbell_id4', ctypes.c_uint32, 976), ('queue_doorbell_id5', ctypes.c_uint32, 980), ('queue_doorbell_id6', ctypes.c_uint32, 984), ('queue_doorbell_id7', ctypes.c_uint32, 988), ('queue_doorbell_id8', ctypes.c_uint32, 992), ('queue_doorbell_id9', ctypes.c_uint32, 996), ('queue_doorbell_id10', ctypes.c_uint32, 1000), ('queue_doorbell_id11', ctypes.c_uint32, 1004), ('queue_doorbell_id12', ctypes.c_uint32, 1008), ('queue_doorbell_id13', ctypes.c_uint32, 1012), ('queue_doorbell_id14', ctypes.c_uint32, 1016), ('queue_doorbell_id15', ctypes.c_uint32, 1020), ('control_buf_addr_lo', ctypes.c_uint32, 1024), ('control_buf_addr_hi', ctypes.c_uint32, 1028), ('control_buf_wptr_lo', ctypes.c_uint32, 1032), ('control_buf_wptr_hi', ctypes.c_uint32, 1036), ('control_buf_dptr_lo', ctypes.c_uint32, 1040), ('control_buf_dptr_hi', ctypes.c_uint32, 1044), ('control_buf_num_entries', ctypes.c_uint32, 1048), ('draw_ring_addr_lo', ctypes.c_uint32, 1052), ('draw_ring_addr_hi', ctypes.c_uint32, 1056), ('reserved_265', ctypes.c_uint32, 1060), ('reserved_266', ctypes.c_uint32, 1064), ('reserved_267', ctypes.c_uint32, 1068), ('reserved_268', ctypes.c_uint32, 1072), ('reserved_269', ctypes.c_uint32, 1076), ('reserved_270', ctypes.c_uint32, 1080), ('reserved_271', ctypes.c_uint32, 1084), ('reserved_272', ctypes.c_uint32, 1088), ('reserved_273', ctypes.c_uint32, 1092), ('reserved_274', ctypes.c_uint32, 1096), ('reserved_275', ctypes.c_uint32, 1100), ('reserved_276', ctypes.c_uint32, 1104), ('reserved_277', ctypes.c_uint32, 1108), ('reserved_278', ctypes.c_uint32, 1112), ('reserved_279', ctypes.c_uint32, 1116), ('reserved_280', ctypes.c_uint32, 1120), ('reserved_281', ctypes.c_uint32, 1124), ('reserved_282', ctypes.c_uint32, 1128), ('reserved_283', ctypes.c_uint32, 1132), ('reserved_284', ctypes.c_uint32, 1136), ('reserved_285', ctypes.c_uint32, 1140), ('reserved_286', ctypes.c_uint32, 1144), ('reserved_287', ctypes.c_uint32, 1148), ('reserved_288', ctypes.c_uint32, 1152), ('reserved_289', ctypes.c_uint32, 1156), ('reserved_290', ctypes.c_uint32, 1160), ('reserved_291', ctypes.c_uint32, 1164), ('reserved_292', ctypes.c_uint32, 1168), ('reserved_293', ctypes.c_uint32, 1172), ('reserved_294', ctypes.c_uint32, 1176), ('reserved_295', ctypes.c_uint32, 1180), ('reserved_296', ctypes.c_uint32, 1184), ('reserved_297', ctypes.c_uint32, 1188), ('reserved_298', ctypes.c_uint32, 1192), ('reserved_299', ctypes.c_uint32, 1196), ('reserved_300', ctypes.c_uint32, 1200), ('reserved_301', ctypes.c_uint32, 1204), ('reserved_302', ctypes.c_uint32, 1208), ('reserved_303', ctypes.c_uint32, 1212), ('reserved_304', ctypes.c_uint32, 1216), ('reserved_305', ctypes.c_uint32, 1220), ('reserved_306', ctypes.c_uint32, 1224), ('reserved_307', ctypes.c_uint32, 1228), ('reserved_308', ctypes.c_uint32, 1232), ('reserved_309', ctypes.c_uint32, 1236), ('reserved_310', ctypes.c_uint32, 1240), ('reserved_311', ctypes.c_uint32, 1244), ('reserved_312', ctypes.c_uint32, 1248), ('reserved_313', ctypes.c_uint32, 1252), ('reserved_314', ctypes.c_uint32, 1256), ('reserved_315', ctypes.c_uint32, 1260), ('reserved_316', ctypes.c_uint32, 1264), ('reserved_317', ctypes.c_uint32, 1268), ('reserved_318', ctypes.c_uint32, 1272), ('reserved_319', ctypes.c_uint32, 1276), ('reserved_320', ctypes.c_uint32, 1280), ('reserved_321', ctypes.c_uint32, 1284), ('reserved_322', ctypes.c_uint32, 1288), ('reserved_323', ctypes.c_uint32, 1292), ('reserved_324', ctypes.c_uint32, 1296), ('reserved_325', ctypes.c_uint32, 1300), ('reserved_326', ctypes.c_uint32, 1304), ('reserved_327', ctypes.c_uint32, 1308), ('reserved_328', ctypes.c_uint32, 1312), ('reserved_329', ctypes.c_uint32, 1316), ('reserved_330', ctypes.c_uint32, 1320), ('reserved_331', ctypes.c_uint32, 1324), ('reserved_332', ctypes.c_uint32, 1328), ('reserved_333', ctypes.c_uint32, 1332), ('reserved_334', ctypes.c_uint32, 1336), ('reserved_335', ctypes.c_uint32, 1340), ('reserved_336', ctypes.c_uint32, 1344), ('reserved_337', ctypes.c_uint32, 1348), ('reserved_338', ctypes.c_uint32, 1352), ('reserved_339', ctypes.c_uint32, 1356), ('reserved_340', ctypes.c_uint32, 1360), ('reserved_341', ctypes.c_uint32, 1364), ('reserved_342', ctypes.c_uint32, 1368), ('reserved_343', ctypes.c_uint32, 1372), ('reserved_344', ctypes.c_uint32, 1376), ('reserved_345', ctypes.c_uint32, 1380), ('reserved_346', ctypes.c_uint32, 1384), ('reserved_347', ctypes.c_uint32, 1388), ('reserved_348', ctypes.c_uint32, 1392), ('reserved_349', ctypes.c_uint32, 1396), ('reserved_350', ctypes.c_uint32, 1400), ('reserved_351', ctypes.c_uint32, 1404), ('reserved_352', ctypes.c_uint32, 1408), ('reserved_353', ctypes.c_uint32, 1412), ('reserved_354', ctypes.c_uint32, 1416), ('reserved_355', ctypes.c_uint32, 1420), ('reserved_356', ctypes.c_uint32, 1424), ('reserved_357', ctypes.c_uint32, 1428), ('reserved_358', ctypes.c_uint32, 1432), ('reserved_359', ctypes.c_uint32, 1436), ('reserved_360', ctypes.c_uint32, 1440), ('reserved_361', ctypes.c_uint32, 1444), ('reserved_362', ctypes.c_uint32, 1448), ('reserved_363', ctypes.c_uint32, 1452), ('reserved_364', ctypes.c_uint32, 1456), ('reserved_365', ctypes.c_uint32, 1460), ('reserved_366', ctypes.c_uint32, 1464), ('reserved_367', ctypes.c_uint32, 1468), ('reserved_368', ctypes.c_uint32, 1472), ('reserved_369', ctypes.c_uint32, 1476), ('reserved_370', ctypes.c_uint32, 1480), ('reserved_371', ctypes.c_uint32, 1484), ('reserved_372', ctypes.c_uint32, 1488), ('reserved_373', ctypes.c_uint32, 1492), ('reserved_374', ctypes.c_uint32, 1496), ('reserved_375', ctypes.c_uint32, 1500), ('reserved_376', ctypes.c_uint32, 1504), ('reserved_377', ctypes.c_uint32, 1508), ('reserved_378', ctypes.c_uint32, 1512), ('reserved_379', ctypes.c_uint32, 1516), ('reserved_380', ctypes.c_uint32, 1520), ('reserved_381', ctypes.c_uint32, 1524), ('reserved_382', ctypes.c_uint32, 1528), ('reserved_383', ctypes.c_uint32, 1532), ('reserved_384', ctypes.c_uint32, 1536), ('reserved_385', ctypes.c_uint32, 1540), ('reserved_386', ctypes.c_uint32, 1544), ('reserved_387', ctypes.c_uint32, 1548), ('reserved_388', ctypes.c_uint32, 1552), ('reserved_389', ctypes.c_uint32, 1556), ('reserved_390', ctypes.c_uint32, 1560), ('reserved_391', ctypes.c_uint32, 1564), ('reserved_392', ctypes.c_uint32, 1568), ('reserved_393', ctypes.c_uint32, 1572), ('reserved_394', ctypes.c_uint32, 1576), ('reserved_395', ctypes.c_uint32, 1580), ('reserved_396', ctypes.c_uint32, 1584), ('reserved_397', ctypes.c_uint32, 1588), ('reserved_398', ctypes.c_uint32, 1592), ('reserved_399', ctypes.c_uint32, 1596), ('reserved_400', ctypes.c_uint32, 1600), ('reserved_401', ctypes.c_uint32, 1604), ('reserved_402', ctypes.c_uint32, 1608), ('reserved_403', ctypes.c_uint32, 1612), ('reserved_404', ctypes.c_uint32, 1616), ('reserved_405', ctypes.c_uint32, 1620), ('reserved_406', ctypes.c_uint32, 1624), ('reserved_407', ctypes.c_uint32, 1628), ('reserved_408', ctypes.c_uint32, 1632), ('reserved_409', ctypes.c_uint32, 1636), ('reserved_410', ctypes.c_uint32, 1640), ('reserved_411', ctypes.c_uint32, 1644), ('reserved_412', ctypes.c_uint32, 1648), ('reserved_413', ctypes.c_uint32, 1652), ('reserved_414', ctypes.c_uint32, 1656), ('reserved_415', ctypes.c_uint32, 1660), ('reserved_416', ctypes.c_uint32, 1664), ('reserved_417', ctypes.c_uint32, 1668), ('reserved_418', ctypes.c_uint32, 1672), ('reserved_419', ctypes.c_uint32, 1676), ('reserved_420', ctypes.c_uint32, 1680), ('reserved_421', ctypes.c_uint32, 1684), ('reserved_422', ctypes.c_uint32, 1688), ('reserved_423', ctypes.c_uint32, 1692), ('reserved_424', ctypes.c_uint32, 1696), ('reserved_425', ctypes.c_uint32, 1700), ('reserved_426', ctypes.c_uint32, 1704), ('reserved_427', ctypes.c_uint32, 1708), ('reserved_428', ctypes.c_uint32, 1712), ('reserved_429', ctypes.c_uint32, 1716), ('reserved_430', ctypes.c_uint32, 1720), ('reserved_431', ctypes.c_uint32, 1724), ('reserved_432', ctypes.c_uint32, 1728), ('reserved_433', ctypes.c_uint32, 1732), ('reserved_434', ctypes.c_uint32, 1736), ('reserved_435', ctypes.c_uint32, 1740), ('reserved_436', ctypes.c_uint32, 1744), ('reserved_437', ctypes.c_uint32, 1748), ('reserved_438', ctypes.c_uint32, 1752), ('reserved_439', ctypes.c_uint32, 1756), ('reserved_440', ctypes.c_uint32, 1760), ('reserved_441', ctypes.c_uint32, 1764), ('reserved_442', ctypes.c_uint32, 1768), ('reserved_443', ctypes.c_uint32, 1772), ('reserved_444', ctypes.c_uint32, 1776), ('reserved_445', ctypes.c_uint32, 1780), ('reserved_446', ctypes.c_uint32, 1784), ('reserved_447', ctypes.c_uint32, 1788), ('gws_0_val', ctypes.c_uint32, 1792), ('gws_1_val', ctypes.c_uint32, 1796), ('gws_2_val', ctypes.c_uint32, 1800), ('gws_3_val', ctypes.c_uint32, 1804), ('gws_4_val', ctypes.c_uint32, 1808), ('gws_5_val', ctypes.c_uint32, 1812), ('gws_6_val', ctypes.c_uint32, 1816), ('gws_7_val', ctypes.c_uint32, 1820), ('gws_8_val', ctypes.c_uint32, 1824), ('gws_9_val', ctypes.c_uint32, 1828), ('gws_10_val', ctypes.c_uint32, 1832), ('gws_11_val', ctypes.c_uint32, 1836), ('gws_12_val', ctypes.c_uint32, 1840), ('gws_13_val', ctypes.c_uint32, 1844), ('gws_14_val', ctypes.c_uint32, 1848), ('gws_15_val', ctypes.c_uint32, 1852), ('gws_16_val', ctypes.c_uint32, 1856), ('gws_17_val', ctypes.c_uint32, 1860), ('gws_18_val', ctypes.c_uint32, 1864), ('gws_19_val', ctypes.c_uint32, 1868), ('gws_20_val', ctypes.c_uint32, 1872), ('gws_21_val', ctypes.c_uint32, 1876), ('gws_22_val', ctypes.c_uint32, 1880), ('gws_23_val', ctypes.c_uint32, 1884), ('gws_24_val', ctypes.c_uint32, 1888), ('gws_25_val', ctypes.c_uint32, 1892), ('gws_26_val', ctypes.c_uint32, 1896), ('gws_27_val', ctypes.c_uint32, 1900), ('gws_28_val', ctypes.c_uint32, 1904), ('gws_29_val', ctypes.c_uint32, 1908), ('gws_30_val', ctypes.c_uint32, 1912), ('gws_31_val', ctypes.c_uint32, 1916), ('gws_32_val', ctypes.c_uint32, 1920), ('gws_33_val', ctypes.c_uint32, 1924), ('gws_34_val', ctypes.c_uint32, 1928), ('gws_35_val', ctypes.c_uint32, 1932), ('gws_36_val', ctypes.c_uint32, 1936), ('gws_37_val', ctypes.c_uint32, 1940), ('gws_38_val', ctypes.c_uint32, 1944), ('gws_39_val', ctypes.c_uint32, 1948), ('gws_40_val', ctypes.c_uint32, 1952), ('gws_41_val', ctypes.c_uint32, 1956), ('gws_42_val', ctypes.c_uint32, 1960), ('gws_43_val', ctypes.c_uint32, 1964), ('gws_44_val', ctypes.c_uint32, 1968), ('gws_45_val', ctypes.c_uint32, 1972), ('gws_46_val', ctypes.c_uint32, 1976), ('gws_47_val', ctypes.c_uint32, 1980), ('gws_48_val', ctypes.c_uint32, 1984), ('gws_49_val', ctypes.c_uint32, 1988), ('gws_50_val', ctypes.c_uint32, 1992), ('gws_51_val', ctypes.c_uint32, 1996), ('gws_52_val', ctypes.c_uint32, 2000), ('gws_53_val', ctypes.c_uint32, 2004), ('gws_54_val', ctypes.c_uint32, 2008), ('gws_55_val', ctypes.c_uint32, 2012), ('gws_56_val', ctypes.c_uint32, 2016), ('gws_57_val', ctypes.c_uint32, 2020), ('gws_58_val', ctypes.c_uint32, 2024), ('gws_59_val', ctypes.c_uint32, 2028), ('gws_60_val', ctypes.c_uint32, 2032), ('gws_61_val', ctypes.c_uint32, 2036), ('gws_62_val', ctypes.c_uint32, 2040), ('gws_63_val', ctypes.c_uint32, 2044)])
@c.record
class struct_v12_gfx_mqd(c.Struct):
  SIZE = 2048
  shadow_base_lo: int
  shadow_base_hi: int
  reserved_2: int
  reserved_3: int
  fw_work_area_base_lo: int
  fw_work_area_base_hi: int
  shadow_initialized: int
  ib_vmid: int
  reserved_8: int
  reserved_9: int
  reserved_10: int
  reserved_11: int
  reserved_12: int
  reserved_13: int
  reserved_14: int
  reserved_15: int
  reserved_16: int
  reserved_17: int
  reserved_18: int
  reserved_19: int
  reserved_20: int
  reserved_21: int
  reserved_22: int
  reserved_23: int
  reserved_24: int
  reserved_25: int
  reserved_26: int
  reserved_27: int
  reserved_28: int
  reserved_29: int
  reserved_30: int
  reserved_31: int
  reserved_32: int
  reserved_33: int
  reserved_34: int
  reserved_35: int
  reserved_36: int
  reserved_37: int
  reserved_38: int
  reserved_39: int
  reserved_40: int
  reserved_41: int
  reserved_42: int
  reserved_43: int
  reserved_44: int
  reserved_45: int
  reserved_46: int
  reserved_47: int
  reserved_48: int
  reserved_49: int
  reserved_50: int
  reserved_51: int
  reserved_52: int
  reserved_53: int
  reserved_54: int
  reserved_55: int
  reserved_56: int
  reserved_57: int
  reserved_58: int
  reserved_59: int
  reserved_60: int
  reserved_61: int
  reserved_62: int
  reserved_63: int
  reserved_64: int
  reserved_65: int
  reserved_66: int
  reserved_67: int
  reserved_68: int
  reserved_69: int
  reserved_70: int
  reserved_71: int
  reserved_72: int
  reserved_73: int
  reserved_74: int
  reserved_75: int
  reserved_76: int
  reserved_77: int
  reserved_78: int
  reserved_79: int
  reserved_80: int
  reserved_81: int
  reserved_82: int
  reserved_83: int
  checksum_lo: int
  checksum_hi: int
  cp_mqd_query_time_lo: int
  cp_mqd_query_time_hi: int
  reserved_88: int
  reserved_89: int
  reserved_90: int
  reserved_91: int
  cp_mqd_query_wave_count: int
  cp_mqd_query_gfx_hqd_rptr: int
  cp_mqd_query_gfx_hqd_wptr: int
  cp_mqd_query_gfx_hqd_offset: int
  reserved_96: int
  reserved_97: int
  reserved_98: int
  reserved_99: int
  reserved_100: int
  reserved_101: int
  reserved_102: int
  reserved_103: int
  task_shader_control_buf_addr_lo: int
  task_shader_control_buf_addr_hi: int
  task_shader_read_rptr_lo: int
  task_shader_read_rptr_hi: int
  task_shader_num_entries: int
  task_shader_num_entries_bits: int
  task_shader_ring_buffer_addr_lo: int
  task_shader_ring_buffer_addr_hi: int
  reserved_112: int
  reserved_113: int
  reserved_114: int
  reserved_115: int
  reserved_116: int
  reserved_117: int
  reserved_118: int
  reserved_119: int
  reserved_120: int
  reserved_121: int
  reserved_122: int
  reserved_123: int
  reserved_124: int
  reserved_125: int
  reserved_126: int
  reserved_127: int
  cp_mqd_base_addr: int
  cp_mqd_base_addr_hi: int
  cp_gfx_hqd_active: int
  cp_gfx_hqd_vmid: int
  reserved_132: int
  reserved_133: int
  cp_gfx_hqd_queue_priority: int
  cp_gfx_hqd_quantum: int
  cp_gfx_hqd_base: int
  cp_gfx_hqd_base_hi: int
  cp_gfx_hqd_rptr: int
  cp_gfx_hqd_rptr_addr: int
  cp_gfx_hqd_rptr_addr_hi: int
  cp_rb_wptr_poll_addr_lo: int
  cp_rb_wptr_poll_addr_hi: int
  cp_rb_doorbell_control: int
  cp_gfx_hqd_offset: int
  cp_gfx_hqd_cntl: int
  reserved_146: int
  reserved_147: int
  cp_gfx_hqd_csmd_rptr: int
  cp_gfx_hqd_wptr: int
  cp_gfx_hqd_wptr_hi: int
  reserved_151: int
  reserved_152: int
  reserved_153: int
  reserved_154: int
  reserved_155: int
  cp_gfx_hqd_mapped: int
  cp_gfx_hqd_que_mgr_control: int
  reserved_158: int
  reserved_159: int
  cp_gfx_hqd_hq_status0: int
  cp_gfx_hqd_hq_control0: int
  cp_gfx_mqd_control: int
  reserved_163: int
  reserved_164: int
  reserved_165: int
  reserved_166: int
  reserved_167: int
  reserved_168: int
  reserved_169: int
  reserved_170: int
  reserved_171: int
  reserved_172: int
  reserved_173: int
  reserved_174: int
  reserved_175: int
  reserved_176: int
  reserved_177: int
  reserved_178: int
  reserved_179: int
  reserved_180: int
  reserved_181: int
  reserved_182: int
  reserved_183: int
  reserved_184: int
  reserved_185: int
  reserved_186: int
  reserved_187: int
  reserved_188: int
  reserved_189: int
  reserved_190: int
  reserved_191: int
  reserved_192: int
  reserved_193: int
  reserved_194: int
  reserved_195: int
  reserved_196: int
  reserved_197: int
  reserved_198: int
  reserved_199: int
  reserved_200: int
  reserved_201: int
  reserved_202: int
  reserved_203: int
  reserved_204: int
  reserved_205: int
  reserved_206: int
  reserved_207: int
  reserved_208: int
  reserved_209: int
  reserved_210: int
  reserved_211: int
  reserved_212: int
  reserved_213: int
  reserved_214: int
  reserved_215: int
  reserved_216: int
  reserved_217: int
  reserved_218: int
  reserved_219: int
  reserved_220: int
  reserved_221: int
  reserved_222: int
  reserved_223: int
  reserved_224: int
  reserved_225: int
  reserved_226: int
  reserved_227: int
  reserved_228: int
  reserved_229: int
  reserved_230: int
  reserved_231: int
  reserved_232: int
  reserved_233: int
  reserved_234: int
  reserved_235: int
  reserved_236: int
  reserved_237: int
  reserved_238: int
  reserved_239: int
  reserved_240: int
  reserved_241: int
  reserved_242: int
  reserved_243: int
  reserved_244: int
  reserved_245: int
  reserved_246: int
  reserved_247: int
  reserved_248: int
  reserved_249: int
  reserved_250: int
  reserved_251: int
  reserved_252: int
  reserved_253: int
  reserved_254: int
  reserved_255: int
  reserved_256: int
  reserved_257: int
  reserved_258: int
  reserved_259: int
  reserved_260: int
  reserved_261: int
  reserved_262: int
  reserved_263: int
  reserved_264: int
  reserved_265: int
  reserved_266: int
  reserved_267: int
  reserved_268: int
  reserved_269: int
  reserved_270: int
  reserved_271: int
  dfwx_flags: int
  dfwx_slot: int
  dfwx_client_data_addr_lo: int
  dfwx_client_data_addr_hi: int
  reserved_276: int
  reserved_277: int
  reserved_278: int
  reserved_279: int
  reserved_280: int
  reserved_281: int
  reserved_282: int
  reserved_283: int
  reserved_284: int
  reserved_285: int
  reserved_286: int
  reserved_287: int
  reserved_288: int
  reserved_289: int
  reserved_290: int
  reserved_291: int
  reserved_292: int
  reserved_293: int
  reserved_294: int
  reserved_295: int
  reserved_296: int
  reserved_297: int
  reserved_298: int
  reserved_299: int
  reserved_300: int
  reserved_301: int
  reserved_302: int
  reserved_303: int
  reserved_304: int
  reserved_305: int
  reserved_306: int
  reserved_307: int
  reserved_308: int
  reserved_309: int
  reserved_310: int
  reserved_311: int
  reserved_312: int
  reserved_313: int
  reserved_314: int
  reserved_315: int
  reserved_316: int
  reserved_317: int
  reserved_318: int
  reserved_319: int
  reserved_320: int
  reserved_321: int
  reserved_322: int
  reserved_323: int
  reserved_324: int
  reserved_325: int
  reserved_326: int
  reserved_327: int
  reserved_328: int
  reserved_329: int
  reserved_330: int
  reserved_331: int
  reserved_332: int
  reserved_333: int
  reserved_334: int
  reserved_335: int
  reserved_336: int
  reserved_337: int
  reserved_338: int
  reserved_339: int
  reserved_340: int
  reserved_341: int
  reserved_342: int
  reserved_343: int
  reserved_344: int
  reserved_345: int
  reserved_346: int
  reserved_347: int
  reserved_348: int
  reserved_349: int
  reserved_350: int
  reserved_351: int
  reserved_352: int
  reserved_353: int
  reserved_354: int
  reserved_355: int
  reserved_356: int
  reserved_357: int
  reserved_358: int
  reserved_359: int
  reserved_360: int
  reserved_361: int
  reserved_362: int
  reserved_363: int
  reserved_364: int
  reserved_365: int
  reserved_366: int
  reserved_367: int
  reserved_368: int
  reserved_369: int
  reserved_370: int
  reserved_371: int
  reserved_372: int
  reserved_373: int
  reserved_374: int
  reserved_375: int
  reserved_376: int
  reserved_377: int
  reserved_378: int
  reserved_379: int
  reserved_380: int
  reserved_381: int
  reserved_382: int
  reserved_383: int
  reserved_384: int
  reserved_385: int
  reserved_386: int
  reserved_387: int
  reserved_388: int
  reserved_389: int
  reserved_390: int
  reserved_391: int
  reserved_392: int
  reserved_393: int
  reserved_394: int
  reserved_395: int
  reserved_396: int
  reserved_397: int
  reserved_398: int
  reserved_399: int
  reserved_400: int
  reserved_401: int
  reserved_402: int
  reserved_403: int
  reserved_404: int
  reserved_405: int
  reserved_406: int
  reserved_407: int
  reserved_408: int
  reserved_409: int
  reserved_410: int
  reserved_411: int
  reserved_412: int
  reserved_413: int
  reserved_414: int
  reserved_415: int
  reserved_416: int
  reserved_417: int
  reserved_418: int
  reserved_419: int
  reserved_420: int
  reserved_421: int
  reserved_422: int
  reserved_423: int
  reserved_424: int
  reserved_425: int
  reserved_426: int
  reserved_427: int
  reserved_428: int
  reserved_429: int
  reserved_430: int
  reserved_431: int
  reserved_432: int
  reserved_433: int
  reserved_434: int
  reserved_435: int
  reserved_436: int
  reserved_437: int
  reserved_438: int
  reserved_439: int
  reserved_440: int
  reserved_441: int
  reserved_442: int
  reserved_443: int
  reserved_444: int
  reserved_445: int
  reserved_446: int
  reserved_447: int
  reserved_448: int
  reserved_449: int
  reserved_450: int
  reserved_451: int
  reserved_452: int
  reserved_453: int
  reserved_454: int
  reserved_455: int
  reserved_456: int
  reserved_457: int
  reserved_458: int
  reserved_459: int
  reserved_460: int
  reserved_461: int
  reserved_462: int
  reserved_463: int
  reserved_464: int
  reserved_465: int
  reserved_466: int
  reserved_467: int
  reserved_468: int
  reserved_469: int
  reserved_470: int
  reserved_471: int
  reserved_472: int
  reserved_473: int
  reserved_474: int
  reserved_475: int
  reserved_476: int
  reserved_477: int
  reserved_478: int
  reserved_479: int
  reserved_480: int
  reserved_481: int
  reserved_482: int
  reserved_483: int
  reserved_484: int
  reserved_485: int
  reserved_486: int
  reserved_487: int
  reserved_488: int
  reserved_489: int
  reserved_490: int
  reserved_491: int
  reserved_492: int
  reserved_493: int
  reserved_494: int
  reserved_495: int
  reserved_496: int
  reserved_497: int
  reserved_498: int
  reserved_499: int
  reserved_500: int
  reserved_501: int
  reserved_502: int
  reserved_503: int
  reserved_504: int
  reserved_505: int
  reserved_506: int
  reserved_507: int
  reserved_508: int
  reserved_509: int
  reserved_510: int
  reserved_511: int
uint32_t: TypeAlias = ctypes.c_uint32
struct_v12_gfx_mqd.register_fields([('shadow_base_lo', uint32_t, 0), ('shadow_base_hi', uint32_t, 4), ('reserved_2', uint32_t, 8), ('reserved_3', uint32_t, 12), ('fw_work_area_base_lo', uint32_t, 16), ('fw_work_area_base_hi', uint32_t, 20), ('shadow_initialized', uint32_t, 24), ('ib_vmid', uint32_t, 28), ('reserved_8', uint32_t, 32), ('reserved_9', uint32_t, 36), ('reserved_10', uint32_t, 40), ('reserved_11', uint32_t, 44), ('reserved_12', uint32_t, 48), ('reserved_13', uint32_t, 52), ('reserved_14', uint32_t, 56), ('reserved_15', uint32_t, 60), ('reserved_16', uint32_t, 64), ('reserved_17', uint32_t, 68), ('reserved_18', uint32_t, 72), ('reserved_19', uint32_t, 76), ('reserved_20', uint32_t, 80), ('reserved_21', uint32_t, 84), ('reserved_22', uint32_t, 88), ('reserved_23', uint32_t, 92), ('reserved_24', uint32_t, 96), ('reserved_25', uint32_t, 100), ('reserved_26', uint32_t, 104), ('reserved_27', uint32_t, 108), ('reserved_28', uint32_t, 112), ('reserved_29', uint32_t, 116), ('reserved_30', uint32_t, 120), ('reserved_31', uint32_t, 124), ('reserved_32', uint32_t, 128), ('reserved_33', uint32_t, 132), ('reserved_34', uint32_t, 136), ('reserved_35', uint32_t, 140), ('reserved_36', uint32_t, 144), ('reserved_37', uint32_t, 148), ('reserved_38', uint32_t, 152), ('reserved_39', uint32_t, 156), ('reserved_40', uint32_t, 160), ('reserved_41', uint32_t, 164), ('reserved_42', uint32_t, 168), ('reserved_43', uint32_t, 172), ('reserved_44', uint32_t, 176), ('reserved_45', uint32_t, 180), ('reserved_46', uint32_t, 184), ('reserved_47', uint32_t, 188), ('reserved_48', uint32_t, 192), ('reserved_49', uint32_t, 196), ('reserved_50', uint32_t, 200), ('reserved_51', uint32_t, 204), ('reserved_52', uint32_t, 208), ('reserved_53', uint32_t, 212), ('reserved_54', uint32_t, 216), ('reserved_55', uint32_t, 220), ('reserved_56', uint32_t, 224), ('reserved_57', uint32_t, 228), ('reserved_58', uint32_t, 232), ('reserved_59', uint32_t, 236), ('reserved_60', uint32_t, 240), ('reserved_61', uint32_t, 244), ('reserved_62', uint32_t, 248), ('reserved_63', uint32_t, 252), ('reserved_64', uint32_t, 256), ('reserved_65', uint32_t, 260), ('reserved_66', uint32_t, 264), ('reserved_67', uint32_t, 268), ('reserved_68', uint32_t, 272), ('reserved_69', uint32_t, 276), ('reserved_70', uint32_t, 280), ('reserved_71', uint32_t, 284), ('reserved_72', uint32_t, 288), ('reserved_73', uint32_t, 292), ('reserved_74', uint32_t, 296), ('reserved_75', uint32_t, 300), ('reserved_76', uint32_t, 304), ('reserved_77', uint32_t, 308), ('reserved_78', uint32_t, 312), ('reserved_79', uint32_t, 316), ('reserved_80', uint32_t, 320), ('reserved_81', uint32_t, 324), ('reserved_82', uint32_t, 328), ('reserved_83', uint32_t, 332), ('checksum_lo', uint32_t, 336), ('checksum_hi', uint32_t, 340), ('cp_mqd_query_time_lo', uint32_t, 344), ('cp_mqd_query_time_hi', uint32_t, 348), ('reserved_88', uint32_t, 352), ('reserved_89', uint32_t, 356), ('reserved_90', uint32_t, 360), ('reserved_91', uint32_t, 364), ('cp_mqd_query_wave_count', uint32_t, 368), ('cp_mqd_query_gfx_hqd_rptr', uint32_t, 372), ('cp_mqd_query_gfx_hqd_wptr', uint32_t, 376), ('cp_mqd_query_gfx_hqd_offset', uint32_t, 380), ('reserved_96', uint32_t, 384), ('reserved_97', uint32_t, 388), ('reserved_98', uint32_t, 392), ('reserved_99', uint32_t, 396), ('reserved_100', uint32_t, 400), ('reserved_101', uint32_t, 404), ('reserved_102', uint32_t, 408), ('reserved_103', uint32_t, 412), ('task_shader_control_buf_addr_lo', uint32_t, 416), ('task_shader_control_buf_addr_hi', uint32_t, 420), ('task_shader_read_rptr_lo', uint32_t, 424), ('task_shader_read_rptr_hi', uint32_t, 428), ('task_shader_num_entries', uint32_t, 432), ('task_shader_num_entries_bits', uint32_t, 436), ('task_shader_ring_buffer_addr_lo', uint32_t, 440), ('task_shader_ring_buffer_addr_hi', uint32_t, 444), ('reserved_112', uint32_t, 448), ('reserved_113', uint32_t, 452), ('reserved_114', uint32_t, 456), ('reserved_115', uint32_t, 460), ('reserved_116', uint32_t, 464), ('reserved_117', uint32_t, 468), ('reserved_118', uint32_t, 472), ('reserved_119', uint32_t, 476), ('reserved_120', uint32_t, 480), ('reserved_121', uint32_t, 484), ('reserved_122', uint32_t, 488), ('reserved_123', uint32_t, 492), ('reserved_124', uint32_t, 496), ('reserved_125', uint32_t, 500), ('reserved_126', uint32_t, 504), ('reserved_127', uint32_t, 508), ('cp_mqd_base_addr', uint32_t, 512), ('cp_mqd_base_addr_hi', uint32_t, 516), ('cp_gfx_hqd_active', uint32_t, 520), ('cp_gfx_hqd_vmid', uint32_t, 524), ('reserved_132', uint32_t, 528), ('reserved_133', uint32_t, 532), ('cp_gfx_hqd_queue_priority', uint32_t, 536), ('cp_gfx_hqd_quantum', uint32_t, 540), ('cp_gfx_hqd_base', uint32_t, 544), ('cp_gfx_hqd_base_hi', uint32_t, 548), ('cp_gfx_hqd_rptr', uint32_t, 552), ('cp_gfx_hqd_rptr_addr', uint32_t, 556), ('cp_gfx_hqd_rptr_addr_hi', uint32_t, 560), ('cp_rb_wptr_poll_addr_lo', uint32_t, 564), ('cp_rb_wptr_poll_addr_hi', uint32_t, 568), ('cp_rb_doorbell_control', uint32_t, 572), ('cp_gfx_hqd_offset', uint32_t, 576), ('cp_gfx_hqd_cntl', uint32_t, 580), ('reserved_146', uint32_t, 584), ('reserved_147', uint32_t, 588), ('cp_gfx_hqd_csmd_rptr', uint32_t, 592), ('cp_gfx_hqd_wptr', uint32_t, 596), ('cp_gfx_hqd_wptr_hi', uint32_t, 600), ('reserved_151', uint32_t, 604), ('reserved_152', uint32_t, 608), ('reserved_153', uint32_t, 612), ('reserved_154', uint32_t, 616), ('reserved_155', uint32_t, 620), ('cp_gfx_hqd_mapped', uint32_t, 624), ('cp_gfx_hqd_que_mgr_control', uint32_t, 628), ('reserved_158', uint32_t, 632), ('reserved_159', uint32_t, 636), ('cp_gfx_hqd_hq_status0', uint32_t, 640), ('cp_gfx_hqd_hq_control0', uint32_t, 644), ('cp_gfx_mqd_control', uint32_t, 648), ('reserved_163', uint32_t, 652), ('reserved_164', uint32_t, 656), ('reserved_165', uint32_t, 660), ('reserved_166', uint32_t, 664), ('reserved_167', uint32_t, 668), ('reserved_168', uint32_t, 672), ('reserved_169', uint32_t, 676), ('reserved_170', uint32_t, 680), ('reserved_171', uint32_t, 684), ('reserved_172', uint32_t, 688), ('reserved_173', uint32_t, 692), ('reserved_174', uint32_t, 696), ('reserved_175', uint32_t, 700), ('reserved_176', uint32_t, 704), ('reserved_177', uint32_t, 708), ('reserved_178', uint32_t, 712), ('reserved_179', uint32_t, 716), ('reserved_180', uint32_t, 720), ('reserved_181', uint32_t, 724), ('reserved_182', uint32_t, 728), ('reserved_183', uint32_t, 732), ('reserved_184', uint32_t, 736), ('reserved_185', uint32_t, 740), ('reserved_186', uint32_t, 744), ('reserved_187', uint32_t, 748), ('reserved_188', uint32_t, 752), ('reserved_189', uint32_t, 756), ('reserved_190', uint32_t, 760), ('reserved_191', uint32_t, 764), ('reserved_192', uint32_t, 768), ('reserved_193', uint32_t, 772), ('reserved_194', uint32_t, 776), ('reserved_195', uint32_t, 780), ('reserved_196', uint32_t, 784), ('reserved_197', uint32_t, 788), ('reserved_198', uint32_t, 792), ('reserved_199', uint32_t, 796), ('reserved_200', uint32_t, 800), ('reserved_201', uint32_t, 804), ('reserved_202', uint32_t, 808), ('reserved_203', uint32_t, 812), ('reserved_204', uint32_t, 816), ('reserved_205', uint32_t, 820), ('reserved_206', uint32_t, 824), ('reserved_207', uint32_t, 828), ('reserved_208', uint32_t, 832), ('reserved_209', uint32_t, 836), ('reserved_210', uint32_t, 840), ('reserved_211', uint32_t, 844), ('reserved_212', uint32_t, 848), ('reserved_213', uint32_t, 852), ('reserved_214', uint32_t, 856), ('reserved_215', uint32_t, 860), ('reserved_216', uint32_t, 864), ('reserved_217', uint32_t, 868), ('reserved_218', uint32_t, 872), ('reserved_219', uint32_t, 876), ('reserved_220', uint32_t, 880), ('reserved_221', uint32_t, 884), ('reserved_222', uint32_t, 888), ('reserved_223', uint32_t, 892), ('reserved_224', uint32_t, 896), ('reserved_225', uint32_t, 900), ('reserved_226', uint32_t, 904), ('reserved_227', uint32_t, 908), ('reserved_228', uint32_t, 912), ('reserved_229', uint32_t, 916), ('reserved_230', uint32_t, 920), ('reserved_231', uint32_t, 924), ('reserved_232', uint32_t, 928), ('reserved_233', uint32_t, 932), ('reserved_234', uint32_t, 936), ('reserved_235', uint32_t, 940), ('reserved_236', uint32_t, 944), ('reserved_237', uint32_t, 948), ('reserved_238', uint32_t, 952), ('reserved_239', uint32_t, 956), ('reserved_240', uint32_t, 960), ('reserved_241', uint32_t, 964), ('reserved_242', uint32_t, 968), ('reserved_243', uint32_t, 972), ('reserved_244', uint32_t, 976), ('reserved_245', uint32_t, 980), ('reserved_246', uint32_t, 984), ('reserved_247', uint32_t, 988), ('reserved_248', uint32_t, 992), ('reserved_249', uint32_t, 996), ('reserved_250', uint32_t, 1000), ('reserved_251', uint32_t, 1004), ('reserved_252', uint32_t, 1008), ('reserved_253', uint32_t, 1012), ('reserved_254', uint32_t, 1016), ('reserved_255', uint32_t, 1020), ('reserved_256', uint32_t, 1024), ('reserved_257', uint32_t, 1028), ('reserved_258', uint32_t, 1032), ('reserved_259', uint32_t, 1036), ('reserved_260', uint32_t, 1040), ('reserved_261', uint32_t, 1044), ('reserved_262', uint32_t, 1048), ('reserved_263', uint32_t, 1052), ('reserved_264', uint32_t, 1056), ('reserved_265', uint32_t, 1060), ('reserved_266', uint32_t, 1064), ('reserved_267', uint32_t, 1068), ('reserved_268', uint32_t, 1072), ('reserved_269', uint32_t, 1076), ('reserved_270', uint32_t, 1080), ('reserved_271', uint32_t, 1084), ('dfwx_flags', uint32_t, 1088), ('dfwx_slot', uint32_t, 1092), ('dfwx_client_data_addr_lo', uint32_t, 1096), ('dfwx_client_data_addr_hi', uint32_t, 1100), ('reserved_276', uint32_t, 1104), ('reserved_277', uint32_t, 1108), ('reserved_278', uint32_t, 1112), ('reserved_279', uint32_t, 1116), ('reserved_280', uint32_t, 1120), ('reserved_281', uint32_t, 1124), ('reserved_282', uint32_t, 1128), ('reserved_283', uint32_t, 1132), ('reserved_284', uint32_t, 1136), ('reserved_285', uint32_t, 1140), ('reserved_286', uint32_t, 1144), ('reserved_287', uint32_t, 1148), ('reserved_288', uint32_t, 1152), ('reserved_289', uint32_t, 1156), ('reserved_290', uint32_t, 1160), ('reserved_291', uint32_t, 1164), ('reserved_292', uint32_t, 1168), ('reserved_293', uint32_t, 1172), ('reserved_294', uint32_t, 1176), ('reserved_295', uint32_t, 1180), ('reserved_296', uint32_t, 1184), ('reserved_297', uint32_t, 1188), ('reserved_298', uint32_t, 1192), ('reserved_299', uint32_t, 1196), ('reserved_300', uint32_t, 1200), ('reserved_301', uint32_t, 1204), ('reserved_302', uint32_t, 1208), ('reserved_303', uint32_t, 1212), ('reserved_304', uint32_t, 1216), ('reserved_305', uint32_t, 1220), ('reserved_306', uint32_t, 1224), ('reserved_307', uint32_t, 1228), ('reserved_308', uint32_t, 1232), ('reserved_309', uint32_t, 1236), ('reserved_310', uint32_t, 1240), ('reserved_311', uint32_t, 1244), ('reserved_312', uint32_t, 1248), ('reserved_313', uint32_t, 1252), ('reserved_314', uint32_t, 1256), ('reserved_315', uint32_t, 1260), ('reserved_316', uint32_t, 1264), ('reserved_317', uint32_t, 1268), ('reserved_318', uint32_t, 1272), ('reserved_319', uint32_t, 1276), ('reserved_320', uint32_t, 1280), ('reserved_321', uint32_t, 1284), ('reserved_322', uint32_t, 1288), ('reserved_323', uint32_t, 1292), ('reserved_324', uint32_t, 1296), ('reserved_325', uint32_t, 1300), ('reserved_326', uint32_t, 1304), ('reserved_327', uint32_t, 1308), ('reserved_328', uint32_t, 1312), ('reserved_329', uint32_t, 1316), ('reserved_330', uint32_t, 1320), ('reserved_331', uint32_t, 1324), ('reserved_332', uint32_t, 1328), ('reserved_333', uint32_t, 1332), ('reserved_334', uint32_t, 1336), ('reserved_335', uint32_t, 1340), ('reserved_336', uint32_t, 1344), ('reserved_337', uint32_t, 1348), ('reserved_338', uint32_t, 1352), ('reserved_339', uint32_t, 1356), ('reserved_340', uint32_t, 1360), ('reserved_341', uint32_t, 1364), ('reserved_342', uint32_t, 1368), ('reserved_343', uint32_t, 1372), ('reserved_344', uint32_t, 1376), ('reserved_345', uint32_t, 1380), ('reserved_346', uint32_t, 1384), ('reserved_347', uint32_t, 1388), ('reserved_348', uint32_t, 1392), ('reserved_349', uint32_t, 1396), ('reserved_350', uint32_t, 1400), ('reserved_351', uint32_t, 1404), ('reserved_352', uint32_t, 1408), ('reserved_353', uint32_t, 1412), ('reserved_354', uint32_t, 1416), ('reserved_355', uint32_t, 1420), ('reserved_356', uint32_t, 1424), ('reserved_357', uint32_t, 1428), ('reserved_358', uint32_t, 1432), ('reserved_359', uint32_t, 1436), ('reserved_360', uint32_t, 1440), ('reserved_361', uint32_t, 1444), ('reserved_362', uint32_t, 1448), ('reserved_363', uint32_t, 1452), ('reserved_364', uint32_t, 1456), ('reserved_365', uint32_t, 1460), ('reserved_366', uint32_t, 1464), ('reserved_367', uint32_t, 1468), ('reserved_368', uint32_t, 1472), ('reserved_369', uint32_t, 1476), ('reserved_370', uint32_t, 1480), ('reserved_371', uint32_t, 1484), ('reserved_372', uint32_t, 1488), ('reserved_373', uint32_t, 1492), ('reserved_374', uint32_t, 1496), ('reserved_375', uint32_t, 1500), ('reserved_376', uint32_t, 1504), ('reserved_377', uint32_t, 1508), ('reserved_378', uint32_t, 1512), ('reserved_379', uint32_t, 1516), ('reserved_380', uint32_t, 1520), ('reserved_381', uint32_t, 1524), ('reserved_382', uint32_t, 1528), ('reserved_383', uint32_t, 1532), ('reserved_384', uint32_t, 1536), ('reserved_385', uint32_t, 1540), ('reserved_386', uint32_t, 1544), ('reserved_387', uint32_t, 1548), ('reserved_388', uint32_t, 1552), ('reserved_389', uint32_t, 1556), ('reserved_390', uint32_t, 1560), ('reserved_391', uint32_t, 1564), ('reserved_392', uint32_t, 1568), ('reserved_393', uint32_t, 1572), ('reserved_394', uint32_t, 1576), ('reserved_395', uint32_t, 1580), ('reserved_396', uint32_t, 1584), ('reserved_397', uint32_t, 1588), ('reserved_398', uint32_t, 1592), ('reserved_399', uint32_t, 1596), ('reserved_400', uint32_t, 1600), ('reserved_401', uint32_t, 1604), ('reserved_402', uint32_t, 1608), ('reserved_403', uint32_t, 1612), ('reserved_404', uint32_t, 1616), ('reserved_405', uint32_t, 1620), ('reserved_406', uint32_t, 1624), ('reserved_407', uint32_t, 1628), ('reserved_408', uint32_t, 1632), ('reserved_409', uint32_t, 1636), ('reserved_410', uint32_t, 1640), ('reserved_411', uint32_t, 1644), ('reserved_412', uint32_t, 1648), ('reserved_413', uint32_t, 1652), ('reserved_414', uint32_t, 1656), ('reserved_415', uint32_t, 1660), ('reserved_416', uint32_t, 1664), ('reserved_417', uint32_t, 1668), ('reserved_418', uint32_t, 1672), ('reserved_419', uint32_t, 1676), ('reserved_420', uint32_t, 1680), ('reserved_421', uint32_t, 1684), ('reserved_422', uint32_t, 1688), ('reserved_423', uint32_t, 1692), ('reserved_424', uint32_t, 1696), ('reserved_425', uint32_t, 1700), ('reserved_426', uint32_t, 1704), ('reserved_427', uint32_t, 1708), ('reserved_428', uint32_t, 1712), ('reserved_429', uint32_t, 1716), ('reserved_430', uint32_t, 1720), ('reserved_431', uint32_t, 1724), ('reserved_432', uint32_t, 1728), ('reserved_433', uint32_t, 1732), ('reserved_434', uint32_t, 1736), ('reserved_435', uint32_t, 1740), ('reserved_436', uint32_t, 1744), ('reserved_437', uint32_t, 1748), ('reserved_438', uint32_t, 1752), ('reserved_439', uint32_t, 1756), ('reserved_440', uint32_t, 1760), ('reserved_441', uint32_t, 1764), ('reserved_442', uint32_t, 1768), ('reserved_443', uint32_t, 1772), ('reserved_444', uint32_t, 1776), ('reserved_445', uint32_t, 1780), ('reserved_446', uint32_t, 1784), ('reserved_447', uint32_t, 1788), ('reserved_448', uint32_t, 1792), ('reserved_449', uint32_t, 1796), ('reserved_450', uint32_t, 1800), ('reserved_451', uint32_t, 1804), ('reserved_452', uint32_t, 1808), ('reserved_453', uint32_t, 1812), ('reserved_454', uint32_t, 1816), ('reserved_455', uint32_t, 1820), ('reserved_456', uint32_t, 1824), ('reserved_457', uint32_t, 1828), ('reserved_458', uint32_t, 1832), ('reserved_459', uint32_t, 1836), ('reserved_460', uint32_t, 1840), ('reserved_461', uint32_t, 1844), ('reserved_462', uint32_t, 1848), ('reserved_463', uint32_t, 1852), ('reserved_464', uint32_t, 1856), ('reserved_465', uint32_t, 1860), ('reserved_466', uint32_t, 1864), ('reserved_467', uint32_t, 1868), ('reserved_468', uint32_t, 1872), ('reserved_469', uint32_t, 1876), ('reserved_470', uint32_t, 1880), ('reserved_471', uint32_t, 1884), ('reserved_472', uint32_t, 1888), ('reserved_473', uint32_t, 1892), ('reserved_474', uint32_t, 1896), ('reserved_475', uint32_t, 1900), ('reserved_476', uint32_t, 1904), ('reserved_477', uint32_t, 1908), ('reserved_478', uint32_t, 1912), ('reserved_479', uint32_t, 1916), ('reserved_480', uint32_t, 1920), ('reserved_481', uint32_t, 1924), ('reserved_482', uint32_t, 1928), ('reserved_483', uint32_t, 1932), ('reserved_484', uint32_t, 1936), ('reserved_485', uint32_t, 1940), ('reserved_486', uint32_t, 1944), ('reserved_487', uint32_t, 1948), ('reserved_488', uint32_t, 1952), ('reserved_489', uint32_t, 1956), ('reserved_490', uint32_t, 1960), ('reserved_491', uint32_t, 1964), ('reserved_492', uint32_t, 1968), ('reserved_493', uint32_t, 1972), ('reserved_494', uint32_t, 1976), ('reserved_495', uint32_t, 1980), ('reserved_496', uint32_t, 1984), ('reserved_497', uint32_t, 1988), ('reserved_498', uint32_t, 1992), ('reserved_499', uint32_t, 1996), ('reserved_500', uint32_t, 2000), ('reserved_501', uint32_t, 2004), ('reserved_502', uint32_t, 2008), ('reserved_503', uint32_t, 2012), ('reserved_504', uint32_t, 2016), ('reserved_505', uint32_t, 2020), ('reserved_506', uint32_t, 2024), ('reserved_507', uint32_t, 2028), ('reserved_508', uint32_t, 2032), ('reserved_509', uint32_t, 2036), ('reserved_510', uint32_t, 2040), ('reserved_511', uint32_t, 2044)])
@c.record
class struct_v12_sdma_mqd(c.Struct):
  SIZE = 512
  sdmax_rlcx_rb_cntl: int
  sdmax_rlcx_rb_base: int
  sdmax_rlcx_rb_base_hi: int
  sdmax_rlcx_rb_rptr: int
  sdmax_rlcx_rb_rptr_hi: int
  sdmax_rlcx_rb_wptr: int
  sdmax_rlcx_rb_wptr_hi: int
  sdmax_rlcx_rb_rptr_addr_lo: int
  sdmax_rlcx_rb_rptr_addr_hi: int
  sdmax_rlcx_ib_cntl: int
  sdmax_rlcx_ib_rptr: int
  sdmax_rlcx_ib_offset: int
  sdmax_rlcx_ib_base_lo: int
  sdmax_rlcx_ib_base_hi: int
  sdmax_rlcx_ib_size: int
  sdmax_rlcx_doorbell: int
  sdmax_rlcx_doorbell_log: int
  sdmax_rlcx_doorbell_offset: int
  sdmax_rlcx_csa_addr_lo: int
  sdmax_rlcx_csa_addr_hi: int
  sdmax_rlcx_sched_cntl: int
  sdmax_rlcx_ib_sub_remain: int
  sdmax_rlcx_preempt: int
  sdmax_rlcx_dummy_reg: int
  sdmax_rlcx_rb_wptr_poll_addr_lo: int
  sdmax_rlcx_rb_wptr_poll_addr_hi: int
  sdmax_rlcx_rb_aql_cntl: int
  sdmax_rlcx_minor_ptr_update: int
  sdmax_rlcx_mcu_dbg0: int
  sdmax_rlcx_mcu_dbg1: int
  sdmax_rlcx_context_switch_status: int
  sdmax_rlcx_midcmd_cntl: int
  sdmax_rlcx_midcmd_data0: int
  sdmax_rlcx_midcmd_data1: int
  sdmax_rlcx_midcmd_data2: int
  sdmax_rlcx_midcmd_data3: int
  sdmax_rlcx_midcmd_data4: int
  sdmax_rlcx_midcmd_data5: int
  sdmax_rlcx_midcmd_data6: int
  sdmax_rlcx_midcmd_data7: int
  sdmax_rlcx_midcmd_data8: int
  sdmax_rlcx_midcmd_data9: int
  sdmax_rlcx_midcmd_data10: int
  sdmax_rlcx_wait_unsatisfied_thd: int
  sdmax_rlcx_mqd_base_addr_lo: int
  sdmax_rlcx_mqd_base_addr_hi: int
  sdmax_rlcx_mqd_control: int
  reserved_47: int
  reserved_48: int
  reserved_49: int
  reserved_50: int
  reserved_51: int
  reserved_52: int
  reserved_53: int
  reserved_54: int
  reserved_55: int
  reserved_56: int
  reserved_57: int
  reserved_58: int
  reserved_59: int
  reserved_60: int
  reserved_61: int
  reserved_62: int
  reserved_63: int
  reserved_64: int
  reserved_65: int
  reserved_66: int
  reserved_67: int
  reserved_68: int
  reserved_69: int
  reserved_70: int
  reserved_71: int
  reserved_72: int
  reserved_73: int
  reserved_74: int
  reserved_75: int
  reserved_76: int
  reserved_77: int
  reserved_78: int
  reserved_79: int
  reserved_80: int
  reserved_81: int
  reserved_82: int
  reserved_83: int
  reserved_84: int
  reserved_85: int
  reserved_86: int
  reserved_87: int
  reserved_88: int
  reserved_89: int
  reserved_90: int
  reserved_91: int
  reserved_92: int
  reserved_93: int
  reserved_94: int
  reserved_95: int
  reserved_96: int
  reserved_97: int
  reserved_98: int
  reserved_99: int
  reserved_100: int
  reserved_101: int
  reserved_102: int
  reserved_103: int
  reserved_104: int
  reserved_105: int
  reserved_106: int
  reserved_107: int
  reserved_108: int
  reserved_109: int
  reserved_110: int
  reserved_111: int
  reserved_112: int
  reserved_113: int
  reserved_114: int
  reserved_115: int
  reserved_116: int
  reserved_117: int
  reserved_118: int
  reserved_119: int
  reserved_120: int
  reserved_121: int
  reserved_122: int
  reserved_123: int
  reserved_124: int
  reserved_125: int
  sdma_engine_id: int
  sdma_queue_id: int
struct_v12_sdma_mqd.register_fields([('sdmax_rlcx_rb_cntl', uint32_t, 0), ('sdmax_rlcx_rb_base', uint32_t, 4), ('sdmax_rlcx_rb_base_hi', uint32_t, 8), ('sdmax_rlcx_rb_rptr', uint32_t, 12), ('sdmax_rlcx_rb_rptr_hi', uint32_t, 16), ('sdmax_rlcx_rb_wptr', uint32_t, 20), ('sdmax_rlcx_rb_wptr_hi', uint32_t, 24), ('sdmax_rlcx_rb_rptr_addr_lo', uint32_t, 28), ('sdmax_rlcx_rb_rptr_addr_hi', uint32_t, 32), ('sdmax_rlcx_ib_cntl', uint32_t, 36), ('sdmax_rlcx_ib_rptr', uint32_t, 40), ('sdmax_rlcx_ib_offset', uint32_t, 44), ('sdmax_rlcx_ib_base_lo', uint32_t, 48), ('sdmax_rlcx_ib_base_hi', uint32_t, 52), ('sdmax_rlcx_ib_size', uint32_t, 56), ('sdmax_rlcx_doorbell', uint32_t, 60), ('sdmax_rlcx_doorbell_log', uint32_t, 64), ('sdmax_rlcx_doorbell_offset', uint32_t, 68), ('sdmax_rlcx_csa_addr_lo', uint32_t, 72), ('sdmax_rlcx_csa_addr_hi', uint32_t, 76), ('sdmax_rlcx_sched_cntl', uint32_t, 80), ('sdmax_rlcx_ib_sub_remain', uint32_t, 84), ('sdmax_rlcx_preempt', uint32_t, 88), ('sdmax_rlcx_dummy_reg', uint32_t, 92), ('sdmax_rlcx_rb_wptr_poll_addr_lo', uint32_t, 96), ('sdmax_rlcx_rb_wptr_poll_addr_hi', uint32_t, 100), ('sdmax_rlcx_rb_aql_cntl', uint32_t, 104), ('sdmax_rlcx_minor_ptr_update', uint32_t, 108), ('sdmax_rlcx_mcu_dbg0', uint32_t, 112), ('sdmax_rlcx_mcu_dbg1', uint32_t, 116), ('sdmax_rlcx_context_switch_status', uint32_t, 120), ('sdmax_rlcx_midcmd_cntl', uint32_t, 124), ('sdmax_rlcx_midcmd_data0', uint32_t, 128), ('sdmax_rlcx_midcmd_data1', uint32_t, 132), ('sdmax_rlcx_midcmd_data2', uint32_t, 136), ('sdmax_rlcx_midcmd_data3', uint32_t, 140), ('sdmax_rlcx_midcmd_data4', uint32_t, 144), ('sdmax_rlcx_midcmd_data5', uint32_t, 148), ('sdmax_rlcx_midcmd_data6', uint32_t, 152), ('sdmax_rlcx_midcmd_data7', uint32_t, 156), ('sdmax_rlcx_midcmd_data8', uint32_t, 160), ('sdmax_rlcx_midcmd_data9', uint32_t, 164), ('sdmax_rlcx_midcmd_data10', uint32_t, 168), ('sdmax_rlcx_wait_unsatisfied_thd', uint32_t, 172), ('sdmax_rlcx_mqd_base_addr_lo', uint32_t, 176), ('sdmax_rlcx_mqd_base_addr_hi', uint32_t, 180), ('sdmax_rlcx_mqd_control', uint32_t, 184), ('reserved_47', uint32_t, 188), ('reserved_48', uint32_t, 192), ('reserved_49', uint32_t, 196), ('reserved_50', uint32_t, 200), ('reserved_51', uint32_t, 204), ('reserved_52', uint32_t, 208), ('reserved_53', uint32_t, 212), ('reserved_54', uint32_t, 216), ('reserved_55', uint32_t, 220), ('reserved_56', uint32_t, 224), ('reserved_57', uint32_t, 228), ('reserved_58', uint32_t, 232), ('reserved_59', uint32_t, 236), ('reserved_60', uint32_t, 240), ('reserved_61', uint32_t, 244), ('reserved_62', uint32_t, 248), ('reserved_63', uint32_t, 252), ('reserved_64', uint32_t, 256), ('reserved_65', uint32_t, 260), ('reserved_66', uint32_t, 264), ('reserved_67', uint32_t, 268), ('reserved_68', uint32_t, 272), ('reserved_69', uint32_t, 276), ('reserved_70', uint32_t, 280), ('reserved_71', uint32_t, 284), ('reserved_72', uint32_t, 288), ('reserved_73', uint32_t, 292), ('reserved_74', uint32_t, 296), ('reserved_75', uint32_t, 300), ('reserved_76', uint32_t, 304), ('reserved_77', uint32_t, 308), ('reserved_78', uint32_t, 312), ('reserved_79', uint32_t, 316), ('reserved_80', uint32_t, 320), ('reserved_81', uint32_t, 324), ('reserved_82', uint32_t, 328), ('reserved_83', uint32_t, 332), ('reserved_84', uint32_t, 336), ('reserved_85', uint32_t, 340), ('reserved_86', uint32_t, 344), ('reserved_87', uint32_t, 348), ('reserved_88', uint32_t, 352), ('reserved_89', uint32_t, 356), ('reserved_90', uint32_t, 360), ('reserved_91', uint32_t, 364), ('reserved_92', uint32_t, 368), ('reserved_93', uint32_t, 372), ('reserved_94', uint32_t, 376), ('reserved_95', uint32_t, 380), ('reserved_96', uint32_t, 384), ('reserved_97', uint32_t, 388), ('reserved_98', uint32_t, 392), ('reserved_99', uint32_t, 396), ('reserved_100', uint32_t, 400), ('reserved_101', uint32_t, 404), ('reserved_102', uint32_t, 408), ('reserved_103', uint32_t, 412), ('reserved_104', uint32_t, 416), ('reserved_105', uint32_t, 420), ('reserved_106', uint32_t, 424), ('reserved_107', uint32_t, 428), ('reserved_108', uint32_t, 432), ('reserved_109', uint32_t, 436), ('reserved_110', uint32_t, 440), ('reserved_111', uint32_t, 444), ('reserved_112', uint32_t, 448), ('reserved_113', uint32_t, 452), ('reserved_114', uint32_t, 456), ('reserved_115', uint32_t, 460), ('reserved_116', uint32_t, 464), ('reserved_117', uint32_t, 468), ('reserved_118', uint32_t, 472), ('reserved_119', uint32_t, 476), ('reserved_120', uint32_t, 480), ('reserved_121', uint32_t, 484), ('reserved_122', uint32_t, 488), ('reserved_123', uint32_t, 492), ('reserved_124', uint32_t, 496), ('reserved_125', uint32_t, 500), ('sdma_engine_id', uint32_t, 504), ('sdma_queue_id', uint32_t, 508)])
@c.record
class struct_v12_compute_mqd(c.Struct):
  SIZE = 2048
  header: int
  compute_dispatch_initiator: int
  compute_dim_x: int
  compute_dim_y: int
  compute_dim_z: int
  compute_start_x: int
  compute_start_y: int
  compute_start_z: int
  compute_num_thread_x: int
  compute_num_thread_y: int
  compute_num_thread_z: int
  compute_pipelinestat_enable: int
  compute_perfcount_enable: int
  compute_pgm_lo: int
  compute_pgm_hi: int
  compute_dispatch_pkt_addr_lo: int
  compute_dispatch_pkt_addr_hi: int
  compute_dispatch_scratch_base_lo: int
  compute_dispatch_scratch_base_hi: int
  compute_pgm_rsrc1: int
  compute_pgm_rsrc2: int
  compute_vmid: int
  compute_resource_limits: int
  compute_static_thread_mgmt_se0: int
  compute_static_thread_mgmt_se1: int
  compute_tmpring_size: int
  compute_static_thread_mgmt_se2: int
  compute_static_thread_mgmt_se3: int
  compute_restart_x: int
  compute_restart_y: int
  compute_restart_z: int
  compute_thread_trace_enable: int
  compute_misc_reserved: int
  compute_dispatch_id: int
  compute_threadgroup_id: int
  compute_req_ctrl: int
  reserved_36: int
  compute_user_accum_0: int
  compute_user_accum_1: int
  compute_user_accum_2: int
  compute_user_accum_3: int
  compute_pgm_rsrc3: int
  compute_ddid_index: int
  compute_shader_chksum: int
  compute_static_thread_mgmt_se4: int
  compute_static_thread_mgmt_se5: int
  compute_static_thread_mgmt_se6: int
  compute_static_thread_mgmt_se7: int
  compute_dispatch_interleave: int
  compute_relaunch: int
  compute_wave_restore_addr_lo: int
  compute_wave_restore_addr_hi: int
  compute_wave_restore_control: int
  reserved_53: int
  reserved_54: int
  reserved_55: int
  reserved_56: int
  reserved_57: int
  reserved_58: int
  compute_static_thread_mgmt_se8: int
  reserved_60: int
  reserved_61: int
  reserved_62: int
  reserved_63: int
  reserved_64: int
  compute_user_data_0: int
  compute_user_data_1: int
  compute_user_data_2: int
  compute_user_data_3: int
  compute_user_data_4: int
  compute_user_data_5: int
  compute_user_data_6: int
  compute_user_data_7: int
  compute_user_data_8: int
  compute_user_data_9: int
  compute_user_data_10: int
  compute_user_data_11: int
  compute_user_data_12: int
  compute_user_data_13: int
  compute_user_data_14: int
  compute_user_data_15: int
  cp_compute_csinvoc_count_lo: int
  cp_compute_csinvoc_count_hi: int
  reserved_83: int
  reserved_84: int
  reserved_85: int
  cp_mqd_query_time_lo: int
  cp_mqd_query_time_hi: int
  cp_mqd_connect_start_time_lo: int
  cp_mqd_connect_start_time_hi: int
  cp_mqd_connect_end_time_lo: int
  cp_mqd_connect_end_time_hi: int
  cp_mqd_connect_end_wf_count: int
  cp_mqd_connect_end_pq_rptr: int
  cp_mqd_connect_end_pq_wptr: int
  cp_mqd_connect_end_ib_rptr: int
  cp_mqd_readindex_lo: int
  cp_mqd_readindex_hi: int
  cp_mqd_save_start_time_lo: int
  cp_mqd_save_start_time_hi: int
  cp_mqd_save_end_time_lo: int
  cp_mqd_save_end_time_hi: int
  cp_mqd_restore_start_time_lo: int
  cp_mqd_restore_start_time_hi: int
  cp_mqd_restore_end_time_lo: int
  cp_mqd_restore_end_time_hi: int
  disable_queue: int
  reserved_107: int
  reserved_108: int
  reserved_109: int
  reserved_110: int
  reserved_111: int
  reserved_112: int
  reserved_113: int
  cp_pq_exe_status_lo: int
  cp_pq_exe_status_hi: int
  cp_packet_id_lo: int
  cp_packet_id_hi: int
  cp_packet_exe_status_lo: int
  cp_packet_exe_status_hi: int
  reserved_120: int
  reserved_121: int
  reserved_122: int
  reserved_123: int
  ctx_save_base_addr_lo: int
  ctx_save_base_addr_hi: int
  reserved_126: int
  reserved_127: int
  cp_mqd_base_addr_lo: int
  cp_mqd_base_addr_hi: int
  cp_hqd_active: int
  cp_hqd_vmid: int
  cp_hqd_persistent_state: int
  cp_hqd_pipe_priority: int
  cp_hqd_queue_priority: int
  cp_hqd_quantum: int
  cp_hqd_pq_base_lo: int
  cp_hqd_pq_base_hi: int
  cp_hqd_pq_rptr: int
  cp_hqd_pq_rptr_report_addr_lo: int
  cp_hqd_pq_rptr_report_addr_hi: int
  cp_hqd_pq_wptr_poll_addr_lo: int
  cp_hqd_pq_wptr_poll_addr_hi: int
  cp_hqd_pq_doorbell_control: int
  reserved_144: int
  cp_hqd_pq_control: int
  cp_hqd_ib_base_addr_lo: int
  cp_hqd_ib_base_addr_hi: int
  cp_hqd_ib_rptr: int
  cp_hqd_ib_control: int
  cp_hqd_iq_timer: int
  cp_hqd_iq_rptr: int
  cp_hqd_dequeue_request: int
  cp_hqd_dma_offload: int
  cp_hqd_sema_cmd: int
  cp_hqd_msg_type: int
  cp_hqd_atomic0_preop_lo: int
  cp_hqd_atomic0_preop_hi: int
  cp_hqd_atomic1_preop_lo: int
  cp_hqd_atomic1_preop_hi: int
  cp_hqd_hq_status0: int
  cp_hqd_hq_control0: int
  cp_mqd_control: int
  cp_hqd_hq_status1: int
  cp_hqd_hq_control1: int
  cp_hqd_eop_base_addr_lo: int
  cp_hqd_eop_base_addr_hi: int
  cp_hqd_eop_control: int
  cp_hqd_eop_rptr: int
  cp_hqd_eop_wptr: int
  cp_hqd_eop_done_events: int
  cp_hqd_ctx_save_base_addr_lo: int
  cp_hqd_ctx_save_base_addr_hi: int
  cp_hqd_ctx_save_control: int
  cp_hqd_cntl_stack_offset: int
  cp_hqd_cntl_stack_size: int
  cp_hqd_wg_state_offset: int
  cp_hqd_ctx_save_size: int
  reserved_178: int
  cp_hqd_error: int
  cp_hqd_eop_wptr_mem: int
  cp_hqd_aql_control: int
  cp_hqd_pq_wptr_lo: int
  cp_hqd_pq_wptr_hi: int
  reserved_184: int
  reserved_185: int
  reserved_186: int
  reserved_187: int
  reserved_188: int
  reserved_189: int
  reserved_190: int
  reserved_191: int
  iqtimer_pkt_header: int
  iqtimer_pkt_dw0: int
  iqtimer_pkt_dw1: int
  iqtimer_pkt_dw2: int
  iqtimer_pkt_dw3: int
  iqtimer_pkt_dw4: int
  iqtimer_pkt_dw5: int
  iqtimer_pkt_dw6: int
  iqtimer_pkt_dw7: int
  iqtimer_pkt_dw8: int
  iqtimer_pkt_dw9: int
  iqtimer_pkt_dw10: int
  iqtimer_pkt_dw11: int
  iqtimer_pkt_dw12: int
  iqtimer_pkt_dw13: int
  iqtimer_pkt_dw14: int
  iqtimer_pkt_dw15: int
  iqtimer_pkt_dw16: int
  iqtimer_pkt_dw17: int
  iqtimer_pkt_dw18: int
  iqtimer_pkt_dw19: int
  iqtimer_pkt_dw20: int
  iqtimer_pkt_dw21: int
  iqtimer_pkt_dw22: int
  iqtimer_pkt_dw23: int
  iqtimer_pkt_dw24: int
  iqtimer_pkt_dw25: int
  iqtimer_pkt_dw26: int
  iqtimer_pkt_dw27: int
  iqtimer_pkt_dw28: int
  iqtimer_pkt_dw29: int
  iqtimer_pkt_dw30: int
  iqtimer_pkt_dw31: int
  reserved_225: int
  reserved_226: int
  reserved_227: int
  set_resources_header: int
  set_resources_dw1: int
  set_resources_dw2: int
  set_resources_dw3: int
  set_resources_dw4: int
  set_resources_dw5: int
  set_resources_dw6: int
  set_resources_dw7: int
  reserved_236: int
  reserved_237: int
  reserved_238: int
  reserved_239: int
  queue_doorbell_id0: int
  queue_doorbell_id1: int
  queue_doorbell_id2: int
  queue_doorbell_id3: int
  queue_doorbell_id4: int
  queue_doorbell_id5: int
  queue_doorbell_id6: int
  queue_doorbell_id7: int
  queue_doorbell_id8: int
  queue_doorbell_id9: int
  queue_doorbell_id10: int
  queue_doorbell_id11: int
  queue_doorbell_id12: int
  queue_doorbell_id13: int
  queue_doorbell_id14: int
  queue_doorbell_id15: int
  control_buf_addr_lo: int
  control_buf_addr_hi: int
  control_buf_wptr_lo: int
  control_buf_wptr_hi: int
  control_buf_dptr_lo: int
  control_buf_dptr_hi: int
  control_buf_num_entries: int
  draw_ring_addr_lo: int
  draw_ring_addr_hi: int
  reserved_265: int
  reserved_266: int
  reserved_267: int
  reserved_268: int
  reserved_269: int
  reserved_270: int
  reserved_271: int
  dfwx_flags: int
  dfwx_slot: int
  dfwx_client_data_addr_lo: int
  dfwx_client_data_addr_hi: int
  reserved_276: int
  reserved_277: int
  reserved_278: int
  reserved_279: int
  reserved_280: int
  reserved_281: int
  reserved_282: int
  reserved_283: int
  reserved_284: int
  reserved_285: int
  reserved_286: int
  reserved_287: int
  reserved_288: int
  reserved_289: int
  reserved_290: int
  reserved_291: int
  reserved_292: int
  reserved_293: int
  reserved_294: int
  reserved_295: int
  reserved_296: int
  reserved_297: int
  reserved_298: int
  reserved_299: int
  reserved_300: int
  reserved_301: int
  reserved_302: int
  reserved_303: int
  reserved_304: int
  reserved_305: int
  reserved_306: int
  reserved_307: int
  reserved_308: int
  reserved_309: int
  reserved_310: int
  reserved_311: int
  reserved_312: int
  reserved_313: int
  reserved_314: int
  reserved_315: int
  reserved_316: int
  reserved_317: int
  reserved_318: int
  reserved_319: int
  reserved_320: int
  reserved_321: int
  reserved_322: int
  reserved_323: int
  reserved_324: int
  reserved_325: int
  reserved_326: int
  reserved_327: int
  reserved_328: int
  reserved_329: int
  reserved_330: int
  reserved_331: int
  reserved_332: int
  reserved_333: int
  reserved_334: int
  reserved_335: int
  reserved_336: int
  reserved_337: int
  reserved_338: int
  reserved_339: int
  reserved_340: int
  reserved_341: int
  reserved_342: int
  reserved_343: int
  reserved_344: int
  reserved_345: int
  reserved_346: int
  reserved_347: int
  reserved_348: int
  reserved_349: int
  reserved_350: int
  reserved_351: int
  reserved_352: int
  reserved_353: int
  reserved_354: int
  reserved_355: int
  reserved_356: int
  reserved_357: int
  reserved_358: int
  reserved_359: int
  reserved_360: int
  reserved_361: int
  reserved_362: int
  reserved_363: int
  reserved_364: int
  reserved_365: int
  reserved_366: int
  reserved_367: int
  reserved_368: int
  reserved_369: int
  reserved_370: int
  reserved_371: int
  reserved_372: int
  reserved_373: int
  reserved_374: int
  reserved_375: int
  reserved_376: int
  reserved_377: int
  reserved_378: int
  reserved_379: int
  reserved_380: int
  reserved_381: int
  reserved_382: int
  reserved_383: int
  reserved_384: int
  reserved_385: int
  reserved_386: int
  reserved_387: int
  reserved_388: int
  reserved_389: int
  reserved_390: int
  reserved_391: int
  reserved_392: int
  reserved_393: int
  reserved_394: int
  reserved_395: int
  reserved_396: int
  reserved_397: int
  reserved_398: int
  reserved_399: int
  reserved_400: int
  reserved_401: int
  reserved_402: int
  reserved_403: int
  reserved_404: int
  reserved_405: int
  reserved_406: int
  reserved_407: int
  reserved_408: int
  reserved_409: int
  reserved_410: int
  reserved_411: int
  reserved_412: int
  reserved_413: int
  reserved_414: int
  reserved_415: int
  reserved_416: int
  reserved_417: int
  reserved_418: int
  reserved_419: int
  reserved_420: int
  reserved_421: int
  reserved_422: int
  reserved_423: int
  reserved_424: int
  reserved_425: int
  reserved_426: int
  reserved_427: int
  reserved_428: int
  reserved_429: int
  reserved_430: int
  reserved_431: int
  reserved_432: int
  reserved_433: int
  reserved_434: int
  reserved_435: int
  reserved_436: int
  reserved_437: int
  reserved_438: int
  reserved_439: int
  reserved_440: int
  reserved_441: int
  reserved_442: int
  reserved_443: int
  reserved_444: int
  reserved_445: int
  reserved_446: int
  reserved_447: int
  gws_0_val: int
  gws_1_val: int
  gws_2_val: int
  gws_3_val: int
  gws_4_val: int
  gws_5_val: int
  gws_6_val: int
  gws_7_val: int
  gws_8_val: int
  gws_9_val: int
  gws_10_val: int
  gws_11_val: int
  gws_12_val: int
  gws_13_val: int
  gws_14_val: int
  gws_15_val: int
  gws_16_val: int
  gws_17_val: int
  gws_18_val: int
  gws_19_val: int
  gws_20_val: int
  gws_21_val: int
  gws_22_val: int
  gws_23_val: int
  gws_24_val: int
  gws_25_val: int
  gws_26_val: int
  gws_27_val: int
  gws_28_val: int
  gws_29_val: int
  gws_30_val: int
  gws_31_val: int
  gws_32_val: int
  gws_33_val: int
  gws_34_val: int
  gws_35_val: int
  gws_36_val: int
  gws_37_val: int
  gws_38_val: int
  gws_39_val: int
  gws_40_val: int
  gws_41_val: int
  gws_42_val: int
  gws_43_val: int
  gws_44_val: int
  gws_45_val: int
  gws_46_val: int
  gws_47_val: int
  gws_48_val: int
  gws_49_val: int
  gws_50_val: int
  gws_51_val: int
  gws_52_val: int
  gws_53_val: int
  gws_54_val: int
  gws_55_val: int
  gws_56_val: int
  gws_57_val: int
  gws_58_val: int
  gws_59_val: int
  gws_60_val: int
  gws_61_val: int
  gws_62_val: int
  gws_63_val: int
struct_v12_compute_mqd.register_fields([('header', uint32_t, 0), ('compute_dispatch_initiator', uint32_t, 4), ('compute_dim_x', uint32_t, 8), ('compute_dim_y', uint32_t, 12), ('compute_dim_z', uint32_t, 16), ('compute_start_x', uint32_t, 20), ('compute_start_y', uint32_t, 24), ('compute_start_z', uint32_t, 28), ('compute_num_thread_x', uint32_t, 32), ('compute_num_thread_y', uint32_t, 36), ('compute_num_thread_z', uint32_t, 40), ('compute_pipelinestat_enable', uint32_t, 44), ('compute_perfcount_enable', uint32_t, 48), ('compute_pgm_lo', uint32_t, 52), ('compute_pgm_hi', uint32_t, 56), ('compute_dispatch_pkt_addr_lo', uint32_t, 60), ('compute_dispatch_pkt_addr_hi', uint32_t, 64), ('compute_dispatch_scratch_base_lo', uint32_t, 68), ('compute_dispatch_scratch_base_hi', uint32_t, 72), ('compute_pgm_rsrc1', uint32_t, 76), ('compute_pgm_rsrc2', uint32_t, 80), ('compute_vmid', uint32_t, 84), ('compute_resource_limits', uint32_t, 88), ('compute_static_thread_mgmt_se0', uint32_t, 92), ('compute_static_thread_mgmt_se1', uint32_t, 96), ('compute_tmpring_size', uint32_t, 100), ('compute_static_thread_mgmt_se2', uint32_t, 104), ('compute_static_thread_mgmt_se3', uint32_t, 108), ('compute_restart_x', uint32_t, 112), ('compute_restart_y', uint32_t, 116), ('compute_restart_z', uint32_t, 120), ('compute_thread_trace_enable', uint32_t, 124), ('compute_misc_reserved', uint32_t, 128), ('compute_dispatch_id', uint32_t, 132), ('compute_threadgroup_id', uint32_t, 136), ('compute_req_ctrl', uint32_t, 140), ('reserved_36', uint32_t, 144), ('compute_user_accum_0', uint32_t, 148), ('compute_user_accum_1', uint32_t, 152), ('compute_user_accum_2', uint32_t, 156), ('compute_user_accum_3', uint32_t, 160), ('compute_pgm_rsrc3', uint32_t, 164), ('compute_ddid_index', uint32_t, 168), ('compute_shader_chksum', uint32_t, 172), ('compute_static_thread_mgmt_se4', uint32_t, 176), ('compute_static_thread_mgmt_se5', uint32_t, 180), ('compute_static_thread_mgmt_se6', uint32_t, 184), ('compute_static_thread_mgmt_se7', uint32_t, 188), ('compute_dispatch_interleave', uint32_t, 192), ('compute_relaunch', uint32_t, 196), ('compute_wave_restore_addr_lo', uint32_t, 200), ('compute_wave_restore_addr_hi', uint32_t, 204), ('compute_wave_restore_control', uint32_t, 208), ('reserved_53', uint32_t, 212), ('reserved_54', uint32_t, 216), ('reserved_55', uint32_t, 220), ('reserved_56', uint32_t, 224), ('reserved_57', uint32_t, 228), ('reserved_58', uint32_t, 232), ('compute_static_thread_mgmt_se8', uint32_t, 236), ('reserved_60', uint32_t, 240), ('reserved_61', uint32_t, 244), ('reserved_62', uint32_t, 248), ('reserved_63', uint32_t, 252), ('reserved_64', uint32_t, 256), ('compute_user_data_0', uint32_t, 260), ('compute_user_data_1', uint32_t, 264), ('compute_user_data_2', uint32_t, 268), ('compute_user_data_3', uint32_t, 272), ('compute_user_data_4', uint32_t, 276), ('compute_user_data_5', uint32_t, 280), ('compute_user_data_6', uint32_t, 284), ('compute_user_data_7', uint32_t, 288), ('compute_user_data_8', uint32_t, 292), ('compute_user_data_9', uint32_t, 296), ('compute_user_data_10', uint32_t, 300), ('compute_user_data_11', uint32_t, 304), ('compute_user_data_12', uint32_t, 308), ('compute_user_data_13', uint32_t, 312), ('compute_user_data_14', uint32_t, 316), ('compute_user_data_15', uint32_t, 320), ('cp_compute_csinvoc_count_lo', uint32_t, 324), ('cp_compute_csinvoc_count_hi', uint32_t, 328), ('reserved_83', uint32_t, 332), ('reserved_84', uint32_t, 336), ('reserved_85', uint32_t, 340), ('cp_mqd_query_time_lo', uint32_t, 344), ('cp_mqd_query_time_hi', uint32_t, 348), ('cp_mqd_connect_start_time_lo', uint32_t, 352), ('cp_mqd_connect_start_time_hi', uint32_t, 356), ('cp_mqd_connect_end_time_lo', uint32_t, 360), ('cp_mqd_connect_end_time_hi', uint32_t, 364), ('cp_mqd_connect_end_wf_count', uint32_t, 368), ('cp_mqd_connect_end_pq_rptr', uint32_t, 372), ('cp_mqd_connect_end_pq_wptr', uint32_t, 376), ('cp_mqd_connect_end_ib_rptr', uint32_t, 380), ('cp_mqd_readindex_lo', uint32_t, 384), ('cp_mqd_readindex_hi', uint32_t, 388), ('cp_mqd_save_start_time_lo', uint32_t, 392), ('cp_mqd_save_start_time_hi', uint32_t, 396), ('cp_mqd_save_end_time_lo', uint32_t, 400), ('cp_mqd_save_end_time_hi', uint32_t, 404), ('cp_mqd_restore_start_time_lo', uint32_t, 408), ('cp_mqd_restore_start_time_hi', uint32_t, 412), ('cp_mqd_restore_end_time_lo', uint32_t, 416), ('cp_mqd_restore_end_time_hi', uint32_t, 420), ('disable_queue', uint32_t, 424), ('reserved_107', uint32_t, 428), ('reserved_108', uint32_t, 432), ('reserved_109', uint32_t, 436), ('reserved_110', uint32_t, 440), ('reserved_111', uint32_t, 444), ('reserved_112', uint32_t, 448), ('reserved_113', uint32_t, 452), ('cp_pq_exe_status_lo', uint32_t, 456), ('cp_pq_exe_status_hi', uint32_t, 460), ('cp_packet_id_lo', uint32_t, 464), ('cp_packet_id_hi', uint32_t, 468), ('cp_packet_exe_status_lo', uint32_t, 472), ('cp_packet_exe_status_hi', uint32_t, 476), ('reserved_120', uint32_t, 480), ('reserved_121', uint32_t, 484), ('reserved_122', uint32_t, 488), ('reserved_123', uint32_t, 492), ('ctx_save_base_addr_lo', uint32_t, 496), ('ctx_save_base_addr_hi', uint32_t, 500), ('reserved_126', uint32_t, 504), ('reserved_127', uint32_t, 508), ('cp_mqd_base_addr_lo', uint32_t, 512), ('cp_mqd_base_addr_hi', uint32_t, 516), ('cp_hqd_active', uint32_t, 520), ('cp_hqd_vmid', uint32_t, 524), ('cp_hqd_persistent_state', uint32_t, 528), ('cp_hqd_pipe_priority', uint32_t, 532), ('cp_hqd_queue_priority', uint32_t, 536), ('cp_hqd_quantum', uint32_t, 540), ('cp_hqd_pq_base_lo', uint32_t, 544), ('cp_hqd_pq_base_hi', uint32_t, 548), ('cp_hqd_pq_rptr', uint32_t, 552), ('cp_hqd_pq_rptr_report_addr_lo', uint32_t, 556), ('cp_hqd_pq_rptr_report_addr_hi', uint32_t, 560), ('cp_hqd_pq_wptr_poll_addr_lo', uint32_t, 564), ('cp_hqd_pq_wptr_poll_addr_hi', uint32_t, 568), ('cp_hqd_pq_doorbell_control', uint32_t, 572), ('reserved_144', uint32_t, 576), ('cp_hqd_pq_control', uint32_t, 580), ('cp_hqd_ib_base_addr_lo', uint32_t, 584), ('cp_hqd_ib_base_addr_hi', uint32_t, 588), ('cp_hqd_ib_rptr', uint32_t, 592), ('cp_hqd_ib_control', uint32_t, 596), ('cp_hqd_iq_timer', uint32_t, 600), ('cp_hqd_iq_rptr', uint32_t, 604), ('cp_hqd_dequeue_request', uint32_t, 608), ('cp_hqd_dma_offload', uint32_t, 612), ('cp_hqd_sema_cmd', uint32_t, 616), ('cp_hqd_msg_type', uint32_t, 620), ('cp_hqd_atomic0_preop_lo', uint32_t, 624), ('cp_hqd_atomic0_preop_hi', uint32_t, 628), ('cp_hqd_atomic1_preop_lo', uint32_t, 632), ('cp_hqd_atomic1_preop_hi', uint32_t, 636), ('cp_hqd_hq_status0', uint32_t, 640), ('cp_hqd_hq_control0', uint32_t, 644), ('cp_mqd_control', uint32_t, 648), ('cp_hqd_hq_status1', uint32_t, 652), ('cp_hqd_hq_control1', uint32_t, 656), ('cp_hqd_eop_base_addr_lo', uint32_t, 660), ('cp_hqd_eop_base_addr_hi', uint32_t, 664), ('cp_hqd_eop_control', uint32_t, 668), ('cp_hqd_eop_rptr', uint32_t, 672), ('cp_hqd_eop_wptr', uint32_t, 676), ('cp_hqd_eop_done_events', uint32_t, 680), ('cp_hqd_ctx_save_base_addr_lo', uint32_t, 684), ('cp_hqd_ctx_save_base_addr_hi', uint32_t, 688), ('cp_hqd_ctx_save_control', uint32_t, 692), ('cp_hqd_cntl_stack_offset', uint32_t, 696), ('cp_hqd_cntl_stack_size', uint32_t, 700), ('cp_hqd_wg_state_offset', uint32_t, 704), ('cp_hqd_ctx_save_size', uint32_t, 708), ('reserved_178', uint32_t, 712), ('cp_hqd_error', uint32_t, 716), ('cp_hqd_eop_wptr_mem', uint32_t, 720), ('cp_hqd_aql_control', uint32_t, 724), ('cp_hqd_pq_wptr_lo', uint32_t, 728), ('cp_hqd_pq_wptr_hi', uint32_t, 732), ('reserved_184', uint32_t, 736), ('reserved_185', uint32_t, 740), ('reserved_186', uint32_t, 744), ('reserved_187', uint32_t, 748), ('reserved_188', uint32_t, 752), ('reserved_189', uint32_t, 756), ('reserved_190', uint32_t, 760), ('reserved_191', uint32_t, 764), ('iqtimer_pkt_header', uint32_t, 768), ('iqtimer_pkt_dw0', uint32_t, 772), ('iqtimer_pkt_dw1', uint32_t, 776), ('iqtimer_pkt_dw2', uint32_t, 780), ('iqtimer_pkt_dw3', uint32_t, 784), ('iqtimer_pkt_dw4', uint32_t, 788), ('iqtimer_pkt_dw5', uint32_t, 792), ('iqtimer_pkt_dw6', uint32_t, 796), ('iqtimer_pkt_dw7', uint32_t, 800), ('iqtimer_pkt_dw8', uint32_t, 804), ('iqtimer_pkt_dw9', uint32_t, 808), ('iqtimer_pkt_dw10', uint32_t, 812), ('iqtimer_pkt_dw11', uint32_t, 816), ('iqtimer_pkt_dw12', uint32_t, 820), ('iqtimer_pkt_dw13', uint32_t, 824), ('iqtimer_pkt_dw14', uint32_t, 828), ('iqtimer_pkt_dw15', uint32_t, 832), ('iqtimer_pkt_dw16', uint32_t, 836), ('iqtimer_pkt_dw17', uint32_t, 840), ('iqtimer_pkt_dw18', uint32_t, 844), ('iqtimer_pkt_dw19', uint32_t, 848), ('iqtimer_pkt_dw20', uint32_t, 852), ('iqtimer_pkt_dw21', uint32_t, 856), ('iqtimer_pkt_dw22', uint32_t, 860), ('iqtimer_pkt_dw23', uint32_t, 864), ('iqtimer_pkt_dw24', uint32_t, 868), ('iqtimer_pkt_dw25', uint32_t, 872), ('iqtimer_pkt_dw26', uint32_t, 876), ('iqtimer_pkt_dw27', uint32_t, 880), ('iqtimer_pkt_dw28', uint32_t, 884), ('iqtimer_pkt_dw29', uint32_t, 888), ('iqtimer_pkt_dw30', uint32_t, 892), ('iqtimer_pkt_dw31', uint32_t, 896), ('reserved_225', uint32_t, 900), ('reserved_226', uint32_t, 904), ('reserved_227', uint32_t, 908), ('set_resources_header', uint32_t, 912), ('set_resources_dw1', uint32_t, 916), ('set_resources_dw2', uint32_t, 920), ('set_resources_dw3', uint32_t, 924), ('set_resources_dw4', uint32_t, 928), ('set_resources_dw5', uint32_t, 932), ('set_resources_dw6', uint32_t, 936), ('set_resources_dw7', uint32_t, 940), ('reserved_236', uint32_t, 944), ('reserved_237', uint32_t, 948), ('reserved_238', uint32_t, 952), ('reserved_239', uint32_t, 956), ('queue_doorbell_id0', uint32_t, 960), ('queue_doorbell_id1', uint32_t, 964), ('queue_doorbell_id2', uint32_t, 968), ('queue_doorbell_id3', uint32_t, 972), ('queue_doorbell_id4', uint32_t, 976), ('queue_doorbell_id5', uint32_t, 980), ('queue_doorbell_id6', uint32_t, 984), ('queue_doorbell_id7', uint32_t, 988), ('queue_doorbell_id8', uint32_t, 992), ('queue_doorbell_id9', uint32_t, 996), ('queue_doorbell_id10', uint32_t, 1000), ('queue_doorbell_id11', uint32_t, 1004), ('queue_doorbell_id12', uint32_t, 1008), ('queue_doorbell_id13', uint32_t, 1012), ('queue_doorbell_id14', uint32_t, 1016), ('queue_doorbell_id15', uint32_t, 1020), ('control_buf_addr_lo', uint32_t, 1024), ('control_buf_addr_hi', uint32_t, 1028), ('control_buf_wptr_lo', uint32_t, 1032), ('control_buf_wptr_hi', uint32_t, 1036), ('control_buf_dptr_lo', uint32_t, 1040), ('control_buf_dptr_hi', uint32_t, 1044), ('control_buf_num_entries', uint32_t, 1048), ('draw_ring_addr_lo', uint32_t, 1052), ('draw_ring_addr_hi', uint32_t, 1056), ('reserved_265', uint32_t, 1060), ('reserved_266', uint32_t, 1064), ('reserved_267', uint32_t, 1068), ('reserved_268', uint32_t, 1072), ('reserved_269', uint32_t, 1076), ('reserved_270', uint32_t, 1080), ('reserved_271', uint32_t, 1084), ('dfwx_flags', uint32_t, 1088), ('dfwx_slot', uint32_t, 1092), ('dfwx_client_data_addr_lo', uint32_t, 1096), ('dfwx_client_data_addr_hi', uint32_t, 1100), ('reserved_276', uint32_t, 1104), ('reserved_277', uint32_t, 1108), ('reserved_278', uint32_t, 1112), ('reserved_279', uint32_t, 1116), ('reserved_280', uint32_t, 1120), ('reserved_281', uint32_t, 1124), ('reserved_282', uint32_t, 1128), ('reserved_283', uint32_t, 1132), ('reserved_284', uint32_t, 1136), ('reserved_285', uint32_t, 1140), ('reserved_286', uint32_t, 1144), ('reserved_287', uint32_t, 1148), ('reserved_288', uint32_t, 1152), ('reserved_289', uint32_t, 1156), ('reserved_290', uint32_t, 1160), ('reserved_291', uint32_t, 1164), ('reserved_292', uint32_t, 1168), ('reserved_293', uint32_t, 1172), ('reserved_294', uint32_t, 1176), ('reserved_295', uint32_t, 1180), ('reserved_296', uint32_t, 1184), ('reserved_297', uint32_t, 1188), ('reserved_298', uint32_t, 1192), ('reserved_299', uint32_t, 1196), ('reserved_300', uint32_t, 1200), ('reserved_301', uint32_t, 1204), ('reserved_302', uint32_t, 1208), ('reserved_303', uint32_t, 1212), ('reserved_304', uint32_t, 1216), ('reserved_305', uint32_t, 1220), ('reserved_306', uint32_t, 1224), ('reserved_307', uint32_t, 1228), ('reserved_308', uint32_t, 1232), ('reserved_309', uint32_t, 1236), ('reserved_310', uint32_t, 1240), ('reserved_311', uint32_t, 1244), ('reserved_312', uint32_t, 1248), ('reserved_313', uint32_t, 1252), ('reserved_314', uint32_t, 1256), ('reserved_315', uint32_t, 1260), ('reserved_316', uint32_t, 1264), ('reserved_317', uint32_t, 1268), ('reserved_318', uint32_t, 1272), ('reserved_319', uint32_t, 1276), ('reserved_320', uint32_t, 1280), ('reserved_321', uint32_t, 1284), ('reserved_322', uint32_t, 1288), ('reserved_323', uint32_t, 1292), ('reserved_324', uint32_t, 1296), ('reserved_325', uint32_t, 1300), ('reserved_326', uint32_t, 1304), ('reserved_327', uint32_t, 1308), ('reserved_328', uint32_t, 1312), ('reserved_329', uint32_t, 1316), ('reserved_330', uint32_t, 1320), ('reserved_331', uint32_t, 1324), ('reserved_332', uint32_t, 1328), ('reserved_333', uint32_t, 1332), ('reserved_334', uint32_t, 1336), ('reserved_335', uint32_t, 1340), ('reserved_336', uint32_t, 1344), ('reserved_337', uint32_t, 1348), ('reserved_338', uint32_t, 1352), ('reserved_339', uint32_t, 1356), ('reserved_340', uint32_t, 1360), ('reserved_341', uint32_t, 1364), ('reserved_342', uint32_t, 1368), ('reserved_343', uint32_t, 1372), ('reserved_344', uint32_t, 1376), ('reserved_345', uint32_t, 1380), ('reserved_346', uint32_t, 1384), ('reserved_347', uint32_t, 1388), ('reserved_348', uint32_t, 1392), ('reserved_349', uint32_t, 1396), ('reserved_350', uint32_t, 1400), ('reserved_351', uint32_t, 1404), ('reserved_352', uint32_t, 1408), ('reserved_353', uint32_t, 1412), ('reserved_354', uint32_t, 1416), ('reserved_355', uint32_t, 1420), ('reserved_356', uint32_t, 1424), ('reserved_357', uint32_t, 1428), ('reserved_358', uint32_t, 1432), ('reserved_359', uint32_t, 1436), ('reserved_360', uint32_t, 1440), ('reserved_361', uint32_t, 1444), ('reserved_362', uint32_t, 1448), ('reserved_363', uint32_t, 1452), ('reserved_364', uint32_t, 1456), ('reserved_365', uint32_t, 1460), ('reserved_366', uint32_t, 1464), ('reserved_367', uint32_t, 1468), ('reserved_368', uint32_t, 1472), ('reserved_369', uint32_t, 1476), ('reserved_370', uint32_t, 1480), ('reserved_371', uint32_t, 1484), ('reserved_372', uint32_t, 1488), ('reserved_373', uint32_t, 1492), ('reserved_374', uint32_t, 1496), ('reserved_375', uint32_t, 1500), ('reserved_376', uint32_t, 1504), ('reserved_377', uint32_t, 1508), ('reserved_378', uint32_t, 1512), ('reserved_379', uint32_t, 1516), ('reserved_380', uint32_t, 1520), ('reserved_381', uint32_t, 1524), ('reserved_382', uint32_t, 1528), ('reserved_383', uint32_t, 1532), ('reserved_384', uint32_t, 1536), ('reserved_385', uint32_t, 1540), ('reserved_386', uint32_t, 1544), ('reserved_387', uint32_t, 1548), ('reserved_388', uint32_t, 1552), ('reserved_389', uint32_t, 1556), ('reserved_390', uint32_t, 1560), ('reserved_391', uint32_t, 1564), ('reserved_392', uint32_t, 1568), ('reserved_393', uint32_t, 1572), ('reserved_394', uint32_t, 1576), ('reserved_395', uint32_t, 1580), ('reserved_396', uint32_t, 1584), ('reserved_397', uint32_t, 1588), ('reserved_398', uint32_t, 1592), ('reserved_399', uint32_t, 1596), ('reserved_400', uint32_t, 1600), ('reserved_401', uint32_t, 1604), ('reserved_402', uint32_t, 1608), ('reserved_403', uint32_t, 1612), ('reserved_404', uint32_t, 1616), ('reserved_405', uint32_t, 1620), ('reserved_406', uint32_t, 1624), ('reserved_407', uint32_t, 1628), ('reserved_408', uint32_t, 1632), ('reserved_409', uint32_t, 1636), ('reserved_410', uint32_t, 1640), ('reserved_411', uint32_t, 1644), ('reserved_412', uint32_t, 1648), ('reserved_413', uint32_t, 1652), ('reserved_414', uint32_t, 1656), ('reserved_415', uint32_t, 1660), ('reserved_416', uint32_t, 1664), ('reserved_417', uint32_t, 1668), ('reserved_418', uint32_t, 1672), ('reserved_419', uint32_t, 1676), ('reserved_420', uint32_t, 1680), ('reserved_421', uint32_t, 1684), ('reserved_422', uint32_t, 1688), ('reserved_423', uint32_t, 1692), ('reserved_424', uint32_t, 1696), ('reserved_425', uint32_t, 1700), ('reserved_426', uint32_t, 1704), ('reserved_427', uint32_t, 1708), ('reserved_428', uint32_t, 1712), ('reserved_429', uint32_t, 1716), ('reserved_430', uint32_t, 1720), ('reserved_431', uint32_t, 1724), ('reserved_432', uint32_t, 1728), ('reserved_433', uint32_t, 1732), ('reserved_434', uint32_t, 1736), ('reserved_435', uint32_t, 1740), ('reserved_436', uint32_t, 1744), ('reserved_437', uint32_t, 1748), ('reserved_438', uint32_t, 1752), ('reserved_439', uint32_t, 1756), ('reserved_440', uint32_t, 1760), ('reserved_441', uint32_t, 1764), ('reserved_442', uint32_t, 1768), ('reserved_443', uint32_t, 1772), ('reserved_444', uint32_t, 1776), ('reserved_445', uint32_t, 1780), ('reserved_446', uint32_t, 1784), ('reserved_447', uint32_t, 1788), ('gws_0_val', uint32_t, 1792), ('gws_1_val', uint32_t, 1796), ('gws_2_val', uint32_t, 1800), ('gws_3_val', uint32_t, 1804), ('gws_4_val', uint32_t, 1808), ('gws_5_val', uint32_t, 1812), ('gws_6_val', uint32_t, 1816), ('gws_7_val', uint32_t, 1820), ('gws_8_val', uint32_t, 1824), ('gws_9_val', uint32_t, 1828), ('gws_10_val', uint32_t, 1832), ('gws_11_val', uint32_t, 1836), ('gws_12_val', uint32_t, 1840), ('gws_13_val', uint32_t, 1844), ('gws_14_val', uint32_t, 1848), ('gws_15_val', uint32_t, 1852), ('gws_16_val', uint32_t, 1856), ('gws_17_val', uint32_t, 1860), ('gws_18_val', uint32_t, 1864), ('gws_19_val', uint32_t, 1868), ('gws_20_val', uint32_t, 1872), ('gws_21_val', uint32_t, 1876), ('gws_22_val', uint32_t, 1880), ('gws_23_val', uint32_t, 1884), ('gws_24_val', uint32_t, 1888), ('gws_25_val', uint32_t, 1892), ('gws_26_val', uint32_t, 1896), ('gws_27_val', uint32_t, 1900), ('gws_28_val', uint32_t, 1904), ('gws_29_val', uint32_t, 1908), ('gws_30_val', uint32_t, 1912), ('gws_31_val', uint32_t, 1916), ('gws_32_val', uint32_t, 1920), ('gws_33_val', uint32_t, 1924), ('gws_34_val', uint32_t, 1928), ('gws_35_val', uint32_t, 1932), ('gws_36_val', uint32_t, 1936), ('gws_37_val', uint32_t, 1940), ('gws_38_val', uint32_t, 1944), ('gws_39_val', uint32_t, 1948), ('gws_40_val', uint32_t, 1952), ('gws_41_val', uint32_t, 1956), ('gws_42_val', uint32_t, 1960), ('gws_43_val', uint32_t, 1964), ('gws_44_val', uint32_t, 1968), ('gws_45_val', uint32_t, 1972), ('gws_46_val', uint32_t, 1976), ('gws_47_val', uint32_t, 1980), ('gws_48_val', uint32_t, 1984), ('gws_49_val', uint32_t, 1988), ('gws_50_val', uint32_t, 1992), ('gws_51_val', uint32_t, 1996), ('gws_52_val', uint32_t, 2000), ('gws_53_val', uint32_t, 2004), ('gws_54_val', uint32_t, 2008), ('gws_55_val', uint32_t, 2012), ('gws_56_val', uint32_t, 2016), ('gws_57_val', uint32_t, 2020), ('gws_58_val', uint32_t, 2024), ('gws_59_val', uint32_t, 2028), ('gws_60_val', uint32_t, 2032), ('gws_61_val', uint32_t, 2036), ('gws_62_val', uint32_t, 2040), ('gws_63_val', uint32_t, 2044)])
enum_amdgpu_vm_level: dict[int, str] = {(AMDGPU_VM_PDB2:=0): 'AMDGPU_VM_PDB2', (AMDGPU_VM_PDB1:=1): 'AMDGPU_VM_PDB1', (AMDGPU_VM_PDB0:=2): 'AMDGPU_VM_PDB0', (AMDGPU_VM_PTB:=3): 'AMDGPU_VM_PTB'}
table: dict[int, str] = {(IP_DISCOVERY:=0): 'IP_DISCOVERY', (GC:=1): 'GC', (HARVEST_INFO:=2): 'HARVEST_INFO', (VCN_INFO:=3): 'VCN_INFO', (MALL_INFO:=4): 'MALL_INFO', (NPS_INFO:=5): 'NPS_INFO', (TOTAL_TABLES:=6): 'TOTAL_TABLES'}
@c.record
class struct_table_info(c.Struct):
  SIZE = 8
  offset: int
  checksum: int
  size: int
  padding: int
uint16_t: TypeAlias = ctypes.c_uint16
struct_table_info.register_fields([('offset', uint16_t, 0), ('checksum', uint16_t, 2), ('size', uint16_t, 4), ('padding', uint16_t, 6)])
table_info: TypeAlias = struct_table_info
@c.record
class struct_binary_header(c.Struct):
  SIZE = 60
  binary_signature: int
  version_major: int
  version_minor: int
  binary_checksum: int
  binary_size: int
  table_list: c.Array[struct_table_info, Literal[6]]
struct_binary_header.register_fields([('binary_signature', uint32_t, 0), ('version_major', uint16_t, 4), ('version_minor', uint16_t, 6), ('binary_checksum', uint16_t, 8), ('binary_size', uint16_t, 10), ('table_list', c.Array[table_info, Literal[6]], 12)])
binary_header: TypeAlias = struct_binary_header
@c.record
class struct_die_info(c.Struct):
  SIZE = 4
  die_id: int
  die_offset: int
struct_die_info.register_fields([('die_id', uint16_t, 0), ('die_offset', uint16_t, 2)])
die_info: TypeAlias = struct_die_info
@c.record
class struct_ip_discovery_header(c.Struct):
  SIZE = 80
  signature: int
  version: int
  size: int
  id: int
  num_dies: int
  die_info: c.Array[struct_die_info, Literal[16]]
  padding: c.Array[ctypes.c_uint16, Literal[1]]
  base_addr_64_bit: int
  reserved: int
  reserved2: int
uint8_t: TypeAlias = ctypes.c_ubyte
struct_ip_discovery_header.register_fields([('signature', uint32_t, 0), ('version', uint16_t, 4), ('size', uint16_t, 6), ('id', uint32_t, 8), ('num_dies', uint16_t, 12), ('die_info', c.Array[die_info, Literal[16]], 14), ('padding', c.Array[uint16_t, Literal[1]], 78), ('base_addr_64_bit', uint8_t, 78, 1, 0), ('reserved', uint8_t, 78, 7, 1), ('reserved2', uint8_t, 79)])
ip_discovery_header: TypeAlias = struct_ip_discovery_header
@c.record
class struct_ip(c.Struct):
  SIZE = 8
  hw_id: int
  number_instance: int
  num_base_address: int
  major: int
  minor: int
  revision: int
  harvest: int
  reserved: int
  base_address: c.Array[ctypes.c_uint32, Literal[0]]
struct_ip.register_fields([('hw_id', uint16_t, 0), ('number_instance', uint8_t, 2), ('num_base_address', uint8_t, 3), ('major', uint8_t, 4), ('minor', uint8_t, 5), ('revision', uint8_t, 6), ('harvest', uint8_t, 7, 4, 0), ('reserved', uint8_t, 7, 4, 4), ('base_address', c.Array[uint32_t, Literal[0]], 8)])
ip: TypeAlias = struct_ip
@c.record
class struct_ip_v3(c.Struct):
  SIZE = 8
  hw_id: int
  instance_number: int
  num_base_address: int
  major: int
  minor: int
  revision: int
  sub_revision: int
  variant: int
  base_address: c.Array[ctypes.c_uint32, Literal[0]]
struct_ip_v3.register_fields([('hw_id', uint16_t, 0), ('instance_number', uint8_t, 2), ('num_base_address', uint8_t, 3), ('major', uint8_t, 4), ('minor', uint8_t, 5), ('revision', uint8_t, 6), ('sub_revision', uint8_t, 7, 4, 0), ('variant', uint8_t, 7, 4, 4), ('base_address', c.Array[uint32_t, Literal[0]], 8)])
ip_v3: TypeAlias = struct_ip_v3
@c.record
class struct_ip_v4(c.Struct):
  SIZE = 7
  hw_id: int
  instance_number: int
  num_base_address: int
  major: int
  minor: int
  revision: int
struct_ip_v4.register_fields([('hw_id', uint16_t, 0), ('instance_number', uint8_t, 2), ('num_base_address', uint8_t, 3), ('major', uint8_t, 4), ('minor', uint8_t, 5), ('revision', uint8_t, 6)])
ip_v4: TypeAlias = struct_ip_v4
@c.record
class struct_die_header(c.Struct):
  SIZE = 4
  die_id: int
  num_ips: int
struct_die_header.register_fields([('die_id', uint16_t, 0), ('num_ips', uint16_t, 2)])
die_header: TypeAlias = struct_die_header
@c.record
class struct_ip_structure(c.Struct):
  SIZE = 24
  header: c.POINTER[struct_ip_discovery_header]
  die: struct_die
@c.record
class struct_die(c.Struct):
  SIZE = 16
  die_header: c.POINTER[struct_die_header]
  ip_list: c.POINTER[struct_ip]
  ip_v3_list: c.POINTER[struct_ip_v3]
  ip_v4_list: c.POINTER[struct_ip_v4]
struct_die.register_fields([('die_header', c.POINTER[die_header], 0), ('ip_list', c.POINTER[ip], 8), ('ip_v3_list', c.POINTER[ip_v3], 8), ('ip_v4_list', c.POINTER[ip_v4], 8)])
struct_ip_structure.register_fields([('header', c.POINTER[ip_discovery_header], 0), ('die', struct_die, 8)])
ip_structure: TypeAlias = struct_ip_structure
@c.record
class struct_gpu_info_header(c.Struct):
  SIZE = 12
  table_id: int
  version_major: int
  version_minor: int
  size: int
struct_gpu_info_header.register_fields([('table_id', uint32_t, 0), ('version_major', uint16_t, 4), ('version_minor', uint16_t, 6), ('size', uint32_t, 8)])
@c.record
class struct_gc_info_v1_0(c.Struct):
  SIZE = 88
  header: struct_gpu_info_header
  gc_num_se: int
  gc_num_wgp0_per_sa: int
  gc_num_wgp1_per_sa: int
  gc_num_rb_per_se: int
  gc_num_gl2c: int
  gc_num_gprs: int
  gc_num_max_gs_thds: int
  gc_gs_table_depth: int
  gc_gsprim_buff_depth: int
  gc_parameter_cache_depth: int
  gc_double_offchip_lds_buffer: int
  gc_wave_size: int
  gc_max_waves_per_simd: int
  gc_max_scratch_slots_per_cu: int
  gc_lds_size: int
  gc_num_sc_per_se: int
  gc_num_sa_per_se: int
  gc_num_packer_per_sc: int
  gc_num_gl2a: int
struct_gc_info_v1_0.register_fields([('header', struct_gpu_info_header, 0), ('gc_num_se', uint32_t, 12), ('gc_num_wgp0_per_sa', uint32_t, 16), ('gc_num_wgp1_per_sa', uint32_t, 20), ('gc_num_rb_per_se', uint32_t, 24), ('gc_num_gl2c', uint32_t, 28), ('gc_num_gprs', uint32_t, 32), ('gc_num_max_gs_thds', uint32_t, 36), ('gc_gs_table_depth', uint32_t, 40), ('gc_gsprim_buff_depth', uint32_t, 44), ('gc_parameter_cache_depth', uint32_t, 48), ('gc_double_offchip_lds_buffer', uint32_t, 52), ('gc_wave_size', uint32_t, 56), ('gc_max_waves_per_simd', uint32_t, 60), ('gc_max_scratch_slots_per_cu', uint32_t, 64), ('gc_lds_size', uint32_t, 68), ('gc_num_sc_per_se', uint32_t, 72), ('gc_num_sa_per_se', uint32_t, 76), ('gc_num_packer_per_sc', uint32_t, 80), ('gc_num_gl2a', uint32_t, 84)])
@c.record
class struct_gc_info_v1_1(c.Struct):
  SIZE = 100
  header: struct_gpu_info_header
  gc_num_se: int
  gc_num_wgp0_per_sa: int
  gc_num_wgp1_per_sa: int
  gc_num_rb_per_se: int
  gc_num_gl2c: int
  gc_num_gprs: int
  gc_num_max_gs_thds: int
  gc_gs_table_depth: int
  gc_gsprim_buff_depth: int
  gc_parameter_cache_depth: int
  gc_double_offchip_lds_buffer: int
  gc_wave_size: int
  gc_max_waves_per_simd: int
  gc_max_scratch_slots_per_cu: int
  gc_lds_size: int
  gc_num_sc_per_se: int
  gc_num_sa_per_se: int
  gc_num_packer_per_sc: int
  gc_num_gl2a: int
  gc_num_tcp_per_sa: int
  gc_num_sdp_interface: int
  gc_num_tcps: int
struct_gc_info_v1_1.register_fields([('header', struct_gpu_info_header, 0), ('gc_num_se', uint32_t, 12), ('gc_num_wgp0_per_sa', uint32_t, 16), ('gc_num_wgp1_per_sa', uint32_t, 20), ('gc_num_rb_per_se', uint32_t, 24), ('gc_num_gl2c', uint32_t, 28), ('gc_num_gprs', uint32_t, 32), ('gc_num_max_gs_thds', uint32_t, 36), ('gc_gs_table_depth', uint32_t, 40), ('gc_gsprim_buff_depth', uint32_t, 44), ('gc_parameter_cache_depth', uint32_t, 48), ('gc_double_offchip_lds_buffer', uint32_t, 52), ('gc_wave_size', uint32_t, 56), ('gc_max_waves_per_simd', uint32_t, 60), ('gc_max_scratch_slots_per_cu', uint32_t, 64), ('gc_lds_size', uint32_t, 68), ('gc_num_sc_per_se', uint32_t, 72), ('gc_num_sa_per_se', uint32_t, 76), ('gc_num_packer_per_sc', uint32_t, 80), ('gc_num_gl2a', uint32_t, 84), ('gc_num_tcp_per_sa', uint32_t, 88), ('gc_num_sdp_interface', uint32_t, 92), ('gc_num_tcps', uint32_t, 96)])
@c.record
class struct_gc_info_v1_2(c.Struct):
  SIZE = 132
  header: struct_gpu_info_header
  gc_num_se: int
  gc_num_wgp0_per_sa: int
  gc_num_wgp1_per_sa: int
  gc_num_rb_per_se: int
  gc_num_gl2c: int
  gc_num_gprs: int
  gc_num_max_gs_thds: int
  gc_gs_table_depth: int
  gc_gsprim_buff_depth: int
  gc_parameter_cache_depth: int
  gc_double_offchip_lds_buffer: int
  gc_wave_size: int
  gc_max_waves_per_simd: int
  gc_max_scratch_slots_per_cu: int
  gc_lds_size: int
  gc_num_sc_per_se: int
  gc_num_sa_per_se: int
  gc_num_packer_per_sc: int
  gc_num_gl2a: int
  gc_num_tcp_per_sa: int
  gc_num_sdp_interface: int
  gc_num_tcps: int
  gc_num_tcp_per_wpg: int
  gc_tcp_l1_size: int
  gc_num_sqc_per_wgp: int
  gc_l1_instruction_cache_size_per_sqc: int
  gc_l1_data_cache_size_per_sqc: int
  gc_gl1c_per_sa: int
  gc_gl1c_size_per_instance: int
  gc_gl2c_per_gpu: int
struct_gc_info_v1_2.register_fields([('header', struct_gpu_info_header, 0), ('gc_num_se', uint32_t, 12), ('gc_num_wgp0_per_sa', uint32_t, 16), ('gc_num_wgp1_per_sa', uint32_t, 20), ('gc_num_rb_per_se', uint32_t, 24), ('gc_num_gl2c', uint32_t, 28), ('gc_num_gprs', uint32_t, 32), ('gc_num_max_gs_thds', uint32_t, 36), ('gc_gs_table_depth', uint32_t, 40), ('gc_gsprim_buff_depth', uint32_t, 44), ('gc_parameter_cache_depth', uint32_t, 48), ('gc_double_offchip_lds_buffer', uint32_t, 52), ('gc_wave_size', uint32_t, 56), ('gc_max_waves_per_simd', uint32_t, 60), ('gc_max_scratch_slots_per_cu', uint32_t, 64), ('gc_lds_size', uint32_t, 68), ('gc_num_sc_per_se', uint32_t, 72), ('gc_num_sa_per_se', uint32_t, 76), ('gc_num_packer_per_sc', uint32_t, 80), ('gc_num_gl2a', uint32_t, 84), ('gc_num_tcp_per_sa', uint32_t, 88), ('gc_num_sdp_interface', uint32_t, 92), ('gc_num_tcps', uint32_t, 96), ('gc_num_tcp_per_wpg', uint32_t, 100), ('gc_tcp_l1_size', uint32_t, 104), ('gc_num_sqc_per_wgp', uint32_t, 108), ('gc_l1_instruction_cache_size_per_sqc', uint32_t, 112), ('gc_l1_data_cache_size_per_sqc', uint32_t, 116), ('gc_gl1c_per_sa', uint32_t, 120), ('gc_gl1c_size_per_instance', uint32_t, 124), ('gc_gl2c_per_gpu', uint32_t, 128)])
@c.record
class struct_gc_info_v1_3(c.Struct):
  SIZE = 164
  header: struct_gpu_info_header
  gc_num_se: int
  gc_num_wgp0_per_sa: int
  gc_num_wgp1_per_sa: int
  gc_num_rb_per_se: int
  gc_num_gl2c: int
  gc_num_gprs: int
  gc_num_max_gs_thds: int
  gc_gs_table_depth: int
  gc_gsprim_buff_depth: int
  gc_parameter_cache_depth: int
  gc_double_offchip_lds_buffer: int
  gc_wave_size: int
  gc_max_waves_per_simd: int
  gc_max_scratch_slots_per_cu: int
  gc_lds_size: int
  gc_num_sc_per_se: int
  gc_num_sa_per_se: int
  gc_num_packer_per_sc: int
  gc_num_gl2a: int
  gc_num_tcp_per_sa: int
  gc_num_sdp_interface: int
  gc_num_tcps: int
  gc_num_tcp_per_wpg: int
  gc_tcp_l1_size: int
  gc_num_sqc_per_wgp: int
  gc_l1_instruction_cache_size_per_sqc: int
  gc_l1_data_cache_size_per_sqc: int
  gc_gl1c_per_sa: int
  gc_gl1c_size_per_instance: int
  gc_gl2c_per_gpu: int
  gc_tcp_size_per_cu: int
  gc_tcp_cache_line_size: int
  gc_instruction_cache_size_per_sqc: int
  gc_instruction_cache_line_size: int
  gc_scalar_data_cache_size_per_sqc: int
  gc_scalar_data_cache_line_size: int
  gc_tcc_size: int
  gc_tcc_cache_line_size: int
struct_gc_info_v1_3.register_fields([('header', struct_gpu_info_header, 0), ('gc_num_se', uint32_t, 12), ('gc_num_wgp0_per_sa', uint32_t, 16), ('gc_num_wgp1_per_sa', uint32_t, 20), ('gc_num_rb_per_se', uint32_t, 24), ('gc_num_gl2c', uint32_t, 28), ('gc_num_gprs', uint32_t, 32), ('gc_num_max_gs_thds', uint32_t, 36), ('gc_gs_table_depth', uint32_t, 40), ('gc_gsprim_buff_depth', uint32_t, 44), ('gc_parameter_cache_depth', uint32_t, 48), ('gc_double_offchip_lds_buffer', uint32_t, 52), ('gc_wave_size', uint32_t, 56), ('gc_max_waves_per_simd', uint32_t, 60), ('gc_max_scratch_slots_per_cu', uint32_t, 64), ('gc_lds_size', uint32_t, 68), ('gc_num_sc_per_se', uint32_t, 72), ('gc_num_sa_per_se', uint32_t, 76), ('gc_num_packer_per_sc', uint32_t, 80), ('gc_num_gl2a', uint32_t, 84), ('gc_num_tcp_per_sa', uint32_t, 88), ('gc_num_sdp_interface', uint32_t, 92), ('gc_num_tcps', uint32_t, 96), ('gc_num_tcp_per_wpg', uint32_t, 100), ('gc_tcp_l1_size', uint32_t, 104), ('gc_num_sqc_per_wgp', uint32_t, 108), ('gc_l1_instruction_cache_size_per_sqc', uint32_t, 112), ('gc_l1_data_cache_size_per_sqc', uint32_t, 116), ('gc_gl1c_per_sa', uint32_t, 120), ('gc_gl1c_size_per_instance', uint32_t, 124), ('gc_gl2c_per_gpu', uint32_t, 128), ('gc_tcp_size_per_cu', uint32_t, 132), ('gc_tcp_cache_line_size', uint32_t, 136), ('gc_instruction_cache_size_per_sqc', uint32_t, 140), ('gc_instruction_cache_line_size', uint32_t, 144), ('gc_scalar_data_cache_size_per_sqc', uint32_t, 148), ('gc_scalar_data_cache_line_size', uint32_t, 152), ('gc_tcc_size', uint32_t, 156), ('gc_tcc_cache_line_size', uint32_t, 160)])
@c.record
class struct_gc_info_v2_0(c.Struct):
  SIZE = 80
  header: struct_gpu_info_header
  gc_num_se: int
  gc_num_cu_per_sh: int
  gc_num_sh_per_se: int
  gc_num_rb_per_se: int
  gc_num_tccs: int
  gc_num_gprs: int
  gc_num_max_gs_thds: int
  gc_gs_table_depth: int
  gc_gsprim_buff_depth: int
  gc_parameter_cache_depth: int
  gc_double_offchip_lds_buffer: int
  gc_wave_size: int
  gc_max_waves_per_simd: int
  gc_max_scratch_slots_per_cu: int
  gc_lds_size: int
  gc_num_sc_per_se: int
  gc_num_packer_per_sc: int
struct_gc_info_v2_0.register_fields([('header', struct_gpu_info_header, 0), ('gc_num_se', uint32_t, 12), ('gc_num_cu_per_sh', uint32_t, 16), ('gc_num_sh_per_se', uint32_t, 20), ('gc_num_rb_per_se', uint32_t, 24), ('gc_num_tccs', uint32_t, 28), ('gc_num_gprs', uint32_t, 32), ('gc_num_max_gs_thds', uint32_t, 36), ('gc_gs_table_depth', uint32_t, 40), ('gc_gsprim_buff_depth', uint32_t, 44), ('gc_parameter_cache_depth', uint32_t, 48), ('gc_double_offchip_lds_buffer', uint32_t, 52), ('gc_wave_size', uint32_t, 56), ('gc_max_waves_per_simd', uint32_t, 60), ('gc_max_scratch_slots_per_cu', uint32_t, 64), ('gc_lds_size', uint32_t, 68), ('gc_num_sc_per_se', uint32_t, 72), ('gc_num_packer_per_sc', uint32_t, 76)])
@c.record
class struct_gc_info_v2_1(c.Struct):
  SIZE = 108
  header: struct_gpu_info_header
  gc_num_se: int
  gc_num_cu_per_sh: int
  gc_num_sh_per_se: int
  gc_num_rb_per_se: int
  gc_num_tccs: int
  gc_num_gprs: int
  gc_num_max_gs_thds: int
  gc_gs_table_depth: int
  gc_gsprim_buff_depth: int
  gc_parameter_cache_depth: int
  gc_double_offchip_lds_buffer: int
  gc_wave_size: int
  gc_max_waves_per_simd: int
  gc_max_scratch_slots_per_cu: int
  gc_lds_size: int
  gc_num_sc_per_se: int
  gc_num_packer_per_sc: int
  gc_num_tcp_per_sh: int
  gc_tcp_size_per_cu: int
  gc_num_sdp_interface: int
  gc_num_cu_per_sqc: int
  gc_instruction_cache_size_per_sqc: int
  gc_scalar_data_cache_size_per_sqc: int
  gc_tcc_size: int
struct_gc_info_v2_1.register_fields([('header', struct_gpu_info_header, 0), ('gc_num_se', uint32_t, 12), ('gc_num_cu_per_sh', uint32_t, 16), ('gc_num_sh_per_se', uint32_t, 20), ('gc_num_rb_per_se', uint32_t, 24), ('gc_num_tccs', uint32_t, 28), ('gc_num_gprs', uint32_t, 32), ('gc_num_max_gs_thds', uint32_t, 36), ('gc_gs_table_depth', uint32_t, 40), ('gc_gsprim_buff_depth', uint32_t, 44), ('gc_parameter_cache_depth', uint32_t, 48), ('gc_double_offchip_lds_buffer', uint32_t, 52), ('gc_wave_size', uint32_t, 56), ('gc_max_waves_per_simd', uint32_t, 60), ('gc_max_scratch_slots_per_cu', uint32_t, 64), ('gc_lds_size', uint32_t, 68), ('gc_num_sc_per_se', uint32_t, 72), ('gc_num_packer_per_sc', uint32_t, 76), ('gc_num_tcp_per_sh', uint32_t, 80), ('gc_tcp_size_per_cu', uint32_t, 84), ('gc_num_sdp_interface', uint32_t, 88), ('gc_num_cu_per_sqc', uint32_t, 92), ('gc_instruction_cache_size_per_sqc', uint32_t, 96), ('gc_scalar_data_cache_size_per_sqc', uint32_t, 100), ('gc_tcc_size', uint32_t, 104)])
@c.record
class struct_harvest_info_header(c.Struct):
  SIZE = 8
  signature: int
  version: int
struct_harvest_info_header.register_fields([('signature', uint32_t, 0), ('version', uint32_t, 4)])
harvest_info_header: TypeAlias = struct_harvest_info_header
@c.record
class struct_harvest_info(c.Struct):
  SIZE = 4
  hw_id: int
  number_instance: int
  reserved: int
struct_harvest_info.register_fields([('hw_id', uint16_t, 0), ('number_instance', uint8_t, 2), ('reserved', uint8_t, 3)])
harvest_info: TypeAlias = struct_harvest_info
@c.record
class struct_harvest_table(c.Struct):
  SIZE = 136
  header: struct_harvest_info_header
  list: c.Array[struct_harvest_info, Literal[32]]
struct_harvest_table.register_fields([('header', harvest_info_header, 0), ('list', c.Array[harvest_info, Literal[32]], 8)])
harvest_table: TypeAlias = struct_harvest_table
@c.record
class struct_mall_info_header(c.Struct):
  SIZE = 12
  table_id: int
  version_major: int
  version_minor: int
  size_bytes: int
struct_mall_info_header.register_fields([('table_id', uint32_t, 0), ('version_major', uint16_t, 4), ('version_minor', uint16_t, 6), ('size_bytes', uint32_t, 8)])
@c.record
class struct_mall_info_v1_0(c.Struct):
  SIZE = 48
  header: struct_mall_info_header
  mall_size_per_m: int
  m_s_present: int
  m_half_use: int
  m_mall_config: int
  reserved: c.Array[ctypes.c_uint32, Literal[5]]
struct_mall_info_v1_0.register_fields([('header', struct_mall_info_header, 0), ('mall_size_per_m', uint32_t, 12), ('m_s_present', uint32_t, 16), ('m_half_use', uint32_t, 20), ('m_mall_config', uint32_t, 24), ('reserved', c.Array[uint32_t, Literal[5]], 28)])
@c.record
class struct_mall_info_v2_0(c.Struct):
  SIZE = 48
  header: struct_mall_info_header
  mall_size_per_umc: int
  reserved: c.Array[ctypes.c_uint32, Literal[8]]
struct_mall_info_v2_0.register_fields([('header', struct_mall_info_header, 0), ('mall_size_per_umc', uint32_t, 12), ('reserved', c.Array[uint32_t, Literal[8]], 16)])
@c.record
class struct_vcn_info_header(c.Struct):
  SIZE = 12
  table_id: int
  version_major: int
  version_minor: int
  size_bytes: int
struct_vcn_info_header.register_fields([('table_id', uint32_t, 0), ('version_major', uint16_t, 4), ('version_minor', uint16_t, 6), ('size_bytes', uint32_t, 8)])
@c.record
class struct_vcn_instance_info_v1_0(c.Struct):
  SIZE = 16
  instance_num: int
  fuse_data: union__fuse_data
  reserved: c.Array[ctypes.c_uint32, Literal[2]]
@c.record
class union__fuse_data(c.Struct):
  SIZE = 4
  bits: union__fuse_data_bits
  all_bits: int
@c.record
class union__fuse_data_bits(c.Struct):
  SIZE = 4
  av1_disabled: int
  vp9_disabled: int
  hevc_disabled: int
  h264_disabled: int
  reserved: int
union__fuse_data_bits.register_fields([('av1_disabled', uint32_t, 0, 1, 0), ('vp9_disabled', uint32_t, 0, 1, 1), ('hevc_disabled', uint32_t, 0, 1, 2), ('h264_disabled', uint32_t, 0, 1, 3), ('reserved', uint32_t, 0, 28, 4)])
union__fuse_data.register_fields([('bits', union__fuse_data_bits, 0), ('all_bits', uint32_t, 0)])
struct_vcn_instance_info_v1_0.register_fields([('instance_num', uint32_t, 0), ('fuse_data', union__fuse_data, 4), ('reserved', c.Array[uint32_t, Literal[2]], 8)])
@c.record
class struct_vcn_info_v1_0(c.Struct):
  SIZE = 96
  header: struct_vcn_info_header
  num_of_instances: int
  instance_info: c.Array[struct_vcn_instance_info_v1_0, Literal[4]]
  reserved: c.Array[ctypes.c_uint32, Literal[4]]
struct_vcn_info_v1_0.register_fields([('header', struct_vcn_info_header, 0), ('num_of_instances', uint32_t, 12), ('instance_info', c.Array[struct_vcn_instance_info_v1_0, Literal[4]], 16), ('reserved', c.Array[uint32_t, Literal[4]], 80)])
@c.record
class struct_nps_info_header(c.Struct):
  SIZE = 12
  table_id: int
  version_major: int
  version_minor: int
  size_bytes: int
struct_nps_info_header.register_fields([('table_id', uint32_t, 0), ('version_major', uint16_t, 4), ('version_minor', uint16_t, 6), ('size_bytes', uint32_t, 8)])
@c.record
class struct_nps_instance_info_v1_0(c.Struct):
  SIZE = 16
  base_address: int
  limit_address: int
uint64_t: TypeAlias = ctypes.c_uint64
struct_nps_instance_info_v1_0.register_fields([('base_address', uint64_t, 0), ('limit_address', uint64_t, 8)])
@c.record
class struct_nps_info_v1_0(c.Struct):
  SIZE = 212
  header: struct_nps_info_header
  nps_type: int
  count: int
  instance_info: c.Array[struct_nps_instance_info_v1_0, Literal[12]]
struct_nps_info_v1_0.register_fields([('header', struct_nps_info_header, 0), ('nps_type', uint32_t, 12), ('count', uint32_t, 16), ('instance_info', c.Array[struct_nps_instance_info_v1_0, Literal[12]], 20)])
enum_amd_hw_ip_block_type: dict[int, str] = {(GC_HWIP:=1): 'GC_HWIP', (HDP_HWIP:=2): 'HDP_HWIP', (SDMA0_HWIP:=3): 'SDMA0_HWIP', (SDMA1_HWIP:=4): 'SDMA1_HWIP', (SDMA2_HWIP:=5): 'SDMA2_HWIP', (SDMA3_HWIP:=6): 'SDMA3_HWIP', (SDMA4_HWIP:=7): 'SDMA4_HWIP', (SDMA5_HWIP:=8): 'SDMA5_HWIP', (SDMA6_HWIP:=9): 'SDMA6_HWIP', (SDMA7_HWIP:=10): 'SDMA7_HWIP', (LSDMA_HWIP:=11): 'LSDMA_HWIP', (MMHUB_HWIP:=12): 'MMHUB_HWIP', (ATHUB_HWIP:=13): 'ATHUB_HWIP', (NBIO_HWIP:=14): 'NBIO_HWIP', (MP0_HWIP:=15): 'MP0_HWIP', (MP1_HWIP:=16): 'MP1_HWIP', (UVD_HWIP:=17): 'UVD_HWIP', (VCN_HWIP:=17): 'VCN_HWIP', (JPEG_HWIP:=17): 'JPEG_HWIP', (VCN1_HWIP:=18): 'VCN1_HWIP', (VCE_HWIP:=19): 'VCE_HWIP', (VPE_HWIP:=20): 'VPE_HWIP', (DF_HWIP:=21): 'DF_HWIP', (DCE_HWIP:=22): 'DCE_HWIP', (OSSSYS_HWIP:=23): 'OSSSYS_HWIP', (SMUIO_HWIP:=24): 'SMUIO_HWIP', (PWR_HWIP:=25): 'PWR_HWIP', (NBIF_HWIP:=26): 'NBIF_HWIP', (THM_HWIP:=27): 'THM_HWIP', (CLK_HWIP:=28): 'CLK_HWIP', (UMC_HWIP:=29): 'UMC_HWIP', (RSMU_HWIP:=30): 'RSMU_HWIP', (XGMI_HWIP:=31): 'XGMI_HWIP', (DCI_HWIP:=32): 'DCI_HWIP', (PCIE_HWIP:=33): 'PCIE_HWIP', (ISP_HWIP:=34): 'ISP_HWIP', (MAX_HWIP:=35): 'MAX_HWIP'}
@c.record
class struct_common_firmware_header(c.Struct):
  SIZE = 32
  size_bytes: int
  header_size_bytes: int
  header_version_major: int
  header_version_minor: int
  ip_version_major: int
  ip_version_minor: int
  ucode_version: int
  ucode_size_bytes: int
  ucode_array_offset_bytes: int
  crc32: int
struct_common_firmware_header.register_fields([('size_bytes', uint32_t, 0), ('header_size_bytes', uint32_t, 4), ('header_version_major', uint16_t, 8), ('header_version_minor', uint16_t, 10), ('ip_version_major', uint16_t, 12), ('ip_version_minor', uint16_t, 14), ('ucode_version', uint32_t, 16), ('ucode_size_bytes', uint32_t, 20), ('ucode_array_offset_bytes', uint32_t, 24), ('crc32', uint32_t, 28)])
@c.record
class struct_mc_firmware_header_v1_0(c.Struct):
  SIZE = 40
  header: struct_common_firmware_header
  io_debug_size_bytes: int
  io_debug_array_offset_bytes: int
struct_mc_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('io_debug_size_bytes', uint32_t, 32), ('io_debug_array_offset_bytes', uint32_t, 36)])
@c.record
class struct_smc_firmware_header_v1_0(c.Struct):
  SIZE = 36
  header: struct_common_firmware_header
  ucode_start_addr: int
struct_smc_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('ucode_start_addr', uint32_t, 32)])
@c.record
class struct_smc_firmware_header_v2_0(c.Struct):
  SIZE = 44
  v1_0: struct_smc_firmware_header_v1_0
  ppt_offset_bytes: int
  ppt_size_bytes: int
struct_smc_firmware_header_v2_0.register_fields([('v1_0', struct_smc_firmware_header_v1_0, 0), ('ppt_offset_bytes', uint32_t, 36), ('ppt_size_bytes', uint32_t, 40)])
@c.record
class struct_smc_soft_pptable_entry(c.Struct):
  SIZE = 12
  id: int
  ppt_offset_bytes: int
  ppt_size_bytes: int
struct_smc_soft_pptable_entry.register_fields([('id', uint32_t, 0), ('ppt_offset_bytes', uint32_t, 4), ('ppt_size_bytes', uint32_t, 8)])
@c.record
class struct_smc_firmware_header_v2_1(c.Struct):
  SIZE = 44
  v1_0: struct_smc_firmware_header_v1_0
  pptable_count: int
  pptable_entry_offset: int
struct_smc_firmware_header_v2_1.register_fields([('v1_0', struct_smc_firmware_header_v1_0, 0), ('pptable_count', uint32_t, 36), ('pptable_entry_offset', uint32_t, 40)])
@c.record
class struct_psp_fw_legacy_bin_desc(c.Struct):
  SIZE = 12
  fw_version: int
  offset_bytes: int
  size_bytes: int
struct_psp_fw_legacy_bin_desc.register_fields([('fw_version', uint32_t, 0), ('offset_bytes', uint32_t, 4), ('size_bytes', uint32_t, 8)])
@c.record
class struct_psp_firmware_header_v1_0(c.Struct):
  SIZE = 44
  header: struct_common_firmware_header
  sos: struct_psp_fw_legacy_bin_desc
struct_psp_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('sos', struct_psp_fw_legacy_bin_desc, 32)])
@c.record
class struct_psp_firmware_header_v1_1(c.Struct):
  SIZE = 68
  v1_0: struct_psp_firmware_header_v1_0
  toc: struct_psp_fw_legacy_bin_desc
  kdb: struct_psp_fw_legacy_bin_desc
struct_psp_firmware_header_v1_1.register_fields([('v1_0', struct_psp_firmware_header_v1_0, 0), ('toc', struct_psp_fw_legacy_bin_desc, 44), ('kdb', struct_psp_fw_legacy_bin_desc, 56)])
@c.record
class struct_psp_firmware_header_v1_2(c.Struct):
  SIZE = 68
  v1_0: struct_psp_firmware_header_v1_0
  res: struct_psp_fw_legacy_bin_desc
  kdb: struct_psp_fw_legacy_bin_desc
struct_psp_firmware_header_v1_2.register_fields([('v1_0', struct_psp_firmware_header_v1_0, 0), ('res', struct_psp_fw_legacy_bin_desc, 44), ('kdb', struct_psp_fw_legacy_bin_desc, 56)])
@c.record
class struct_psp_firmware_header_v1_3(c.Struct):
  SIZE = 116
  v1_1: struct_psp_firmware_header_v1_1
  spl: struct_psp_fw_legacy_bin_desc
  rl: struct_psp_fw_legacy_bin_desc
  sys_drv_aux: struct_psp_fw_legacy_bin_desc
  sos_aux: struct_psp_fw_legacy_bin_desc
struct_psp_firmware_header_v1_3.register_fields([('v1_1', struct_psp_firmware_header_v1_1, 0), ('spl', struct_psp_fw_legacy_bin_desc, 68), ('rl', struct_psp_fw_legacy_bin_desc, 80), ('sys_drv_aux', struct_psp_fw_legacy_bin_desc, 92), ('sos_aux', struct_psp_fw_legacy_bin_desc, 104)])
@c.record
class struct_psp_fw_bin_desc(c.Struct):
  SIZE = 16
  fw_type: int
  fw_version: int
  offset_bytes: int
  size_bytes: int
struct_psp_fw_bin_desc.register_fields([('fw_type', uint32_t, 0), ('fw_version', uint32_t, 4), ('offset_bytes', uint32_t, 8), ('size_bytes', uint32_t, 12)])
enum_psp_fw_type: dict[int, str] = {(PSP_FW_TYPE_UNKOWN:=0): 'PSP_FW_TYPE_UNKOWN', (PSP_FW_TYPE_PSP_SOS:=1): 'PSP_FW_TYPE_PSP_SOS', (PSP_FW_TYPE_PSP_SYS_DRV:=2): 'PSP_FW_TYPE_PSP_SYS_DRV', (PSP_FW_TYPE_PSP_KDB:=3): 'PSP_FW_TYPE_PSP_KDB', (PSP_FW_TYPE_PSP_TOC:=4): 'PSP_FW_TYPE_PSP_TOC', (PSP_FW_TYPE_PSP_SPL:=5): 'PSP_FW_TYPE_PSP_SPL', (PSP_FW_TYPE_PSP_RL:=6): 'PSP_FW_TYPE_PSP_RL', (PSP_FW_TYPE_PSP_SOC_DRV:=7): 'PSP_FW_TYPE_PSP_SOC_DRV', (PSP_FW_TYPE_PSP_INTF_DRV:=8): 'PSP_FW_TYPE_PSP_INTF_DRV', (PSP_FW_TYPE_PSP_DBG_DRV:=9): 'PSP_FW_TYPE_PSP_DBG_DRV', (PSP_FW_TYPE_PSP_RAS_DRV:=10): 'PSP_FW_TYPE_PSP_RAS_DRV', (PSP_FW_TYPE_PSP_IPKEYMGR_DRV:=11): 'PSP_FW_TYPE_PSP_IPKEYMGR_DRV', (PSP_FW_TYPE_MAX_INDEX:=12): 'PSP_FW_TYPE_MAX_INDEX'}
@c.record
class struct_psp_firmware_header_v2_0(c.Struct):
  SIZE = 52
  header: struct_common_firmware_header
  psp_fw_bin_count: int
  psp_fw_bin: c.Array[struct_psp_fw_bin_desc, Literal[1]]
struct_psp_firmware_header_v2_0.register_fields([('header', struct_common_firmware_header, 0), ('psp_fw_bin_count', uint32_t, 32), ('psp_fw_bin', c.Array[struct_psp_fw_bin_desc, Literal[1]], 36)])
@c.record
class struct_psp_firmware_header_v2_1(c.Struct):
  SIZE = 56
  header: struct_common_firmware_header
  psp_fw_bin_count: int
  psp_aux_fw_bin_index: int
  psp_fw_bin: c.Array[struct_psp_fw_bin_desc, Literal[1]]
struct_psp_firmware_header_v2_1.register_fields([('header', struct_common_firmware_header, 0), ('psp_fw_bin_count', uint32_t, 32), ('psp_aux_fw_bin_index', uint32_t, 36), ('psp_fw_bin', c.Array[struct_psp_fw_bin_desc, Literal[1]], 40)])
@c.record
class struct_ta_firmware_header_v1_0(c.Struct):
  SIZE = 92
  header: struct_common_firmware_header
  xgmi: struct_psp_fw_legacy_bin_desc
  ras: struct_psp_fw_legacy_bin_desc
  hdcp: struct_psp_fw_legacy_bin_desc
  dtm: struct_psp_fw_legacy_bin_desc
  securedisplay: struct_psp_fw_legacy_bin_desc
struct_ta_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('xgmi', struct_psp_fw_legacy_bin_desc, 32), ('ras', struct_psp_fw_legacy_bin_desc, 44), ('hdcp', struct_psp_fw_legacy_bin_desc, 56), ('dtm', struct_psp_fw_legacy_bin_desc, 68), ('securedisplay', struct_psp_fw_legacy_bin_desc, 80)])
enum_ta_fw_type: dict[int, str] = {(TA_FW_TYPE_UNKOWN:=0): 'TA_FW_TYPE_UNKOWN', (TA_FW_TYPE_PSP_ASD:=1): 'TA_FW_TYPE_PSP_ASD', (TA_FW_TYPE_PSP_XGMI:=2): 'TA_FW_TYPE_PSP_XGMI', (TA_FW_TYPE_PSP_RAS:=3): 'TA_FW_TYPE_PSP_RAS', (TA_FW_TYPE_PSP_HDCP:=4): 'TA_FW_TYPE_PSP_HDCP', (TA_FW_TYPE_PSP_DTM:=5): 'TA_FW_TYPE_PSP_DTM', (TA_FW_TYPE_PSP_RAP:=6): 'TA_FW_TYPE_PSP_RAP', (TA_FW_TYPE_PSP_SECUREDISPLAY:=7): 'TA_FW_TYPE_PSP_SECUREDISPLAY', (TA_FW_TYPE_MAX_INDEX:=8): 'TA_FW_TYPE_MAX_INDEX'}
@c.record
class struct_ta_firmware_header_v2_0(c.Struct):
  SIZE = 52
  header: struct_common_firmware_header
  ta_fw_bin_count: int
  ta_fw_bin: c.Array[struct_psp_fw_bin_desc, Literal[1]]
struct_ta_firmware_header_v2_0.register_fields([('header', struct_common_firmware_header, 0), ('ta_fw_bin_count', uint32_t, 32), ('ta_fw_bin', c.Array[struct_psp_fw_bin_desc, Literal[1]], 36)])
@c.record
class struct_gfx_firmware_header_v1_0(c.Struct):
  SIZE = 44
  header: struct_common_firmware_header
  ucode_feature_version: int
  jt_offset: int
  jt_size: int
struct_gfx_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('ucode_feature_version', uint32_t, 32), ('jt_offset', uint32_t, 36), ('jt_size', uint32_t, 40)])
@c.record
class struct_gfx_firmware_header_v2_0(c.Struct):
  SIZE = 60
  header: struct_common_firmware_header
  ucode_feature_version: int
  ucode_size_bytes: int
  ucode_offset_bytes: int
  data_size_bytes: int
  data_offset_bytes: int
  ucode_start_addr_lo: int
  ucode_start_addr_hi: int
struct_gfx_firmware_header_v2_0.register_fields([('header', struct_common_firmware_header, 0), ('ucode_feature_version', uint32_t, 32), ('ucode_size_bytes', uint32_t, 36), ('ucode_offset_bytes', uint32_t, 40), ('data_size_bytes', uint32_t, 44), ('data_offset_bytes', uint32_t, 48), ('ucode_start_addr_lo', uint32_t, 52), ('ucode_start_addr_hi', uint32_t, 56)])
@c.record
class struct_mes_firmware_header_v1_0(c.Struct):
  SIZE = 72
  header: struct_common_firmware_header
  mes_ucode_version: int
  mes_ucode_size_bytes: int
  mes_ucode_offset_bytes: int
  mes_ucode_data_version: int
  mes_ucode_data_size_bytes: int
  mes_ucode_data_offset_bytes: int
  mes_uc_start_addr_lo: int
  mes_uc_start_addr_hi: int
  mes_data_start_addr_lo: int
  mes_data_start_addr_hi: int
struct_mes_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('mes_ucode_version', uint32_t, 32), ('mes_ucode_size_bytes', uint32_t, 36), ('mes_ucode_offset_bytes', uint32_t, 40), ('mes_ucode_data_version', uint32_t, 44), ('mes_ucode_data_size_bytes', uint32_t, 48), ('mes_ucode_data_offset_bytes', uint32_t, 52), ('mes_uc_start_addr_lo', uint32_t, 56), ('mes_uc_start_addr_hi', uint32_t, 60), ('mes_data_start_addr_lo', uint32_t, 64), ('mes_data_start_addr_hi', uint32_t, 68)])
@c.record
class struct_rlc_firmware_header_v1_0(c.Struct):
  SIZE = 52
  header: struct_common_firmware_header
  ucode_feature_version: int
  save_and_restore_offset: int
  clear_state_descriptor_offset: int
  avail_scratch_ram_locations: int
  master_pkt_description_offset: int
struct_rlc_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('ucode_feature_version', uint32_t, 32), ('save_and_restore_offset', uint32_t, 36), ('clear_state_descriptor_offset', uint32_t, 40), ('avail_scratch_ram_locations', uint32_t, 44), ('master_pkt_description_offset', uint32_t, 48)])
@c.record
class struct_rlc_firmware_header_v2_0(c.Struct):
  SIZE = 104
  header: struct_common_firmware_header
  ucode_feature_version: int
  jt_offset: int
  jt_size: int
  save_and_restore_offset: int
  clear_state_descriptor_offset: int
  avail_scratch_ram_locations: int
  reg_restore_list_size: int
  reg_list_format_start: int
  reg_list_format_separate_start: int
  starting_offsets_start: int
  reg_list_format_size_bytes: int
  reg_list_format_array_offset_bytes: int
  reg_list_size_bytes: int
  reg_list_array_offset_bytes: int
  reg_list_format_separate_size_bytes: int
  reg_list_format_separate_array_offset_bytes: int
  reg_list_separate_size_bytes: int
  reg_list_separate_array_offset_bytes: int
struct_rlc_firmware_header_v2_0.register_fields([('header', struct_common_firmware_header, 0), ('ucode_feature_version', uint32_t, 32), ('jt_offset', uint32_t, 36), ('jt_size', uint32_t, 40), ('save_and_restore_offset', uint32_t, 44), ('clear_state_descriptor_offset', uint32_t, 48), ('avail_scratch_ram_locations', uint32_t, 52), ('reg_restore_list_size', uint32_t, 56), ('reg_list_format_start', uint32_t, 60), ('reg_list_format_separate_start', uint32_t, 64), ('starting_offsets_start', uint32_t, 68), ('reg_list_format_size_bytes', uint32_t, 72), ('reg_list_format_array_offset_bytes', uint32_t, 76), ('reg_list_size_bytes', uint32_t, 80), ('reg_list_array_offset_bytes', uint32_t, 84), ('reg_list_format_separate_size_bytes', uint32_t, 88), ('reg_list_format_separate_array_offset_bytes', uint32_t, 92), ('reg_list_separate_size_bytes', uint32_t, 96), ('reg_list_separate_array_offset_bytes', uint32_t, 100)])
@c.record
class struct_rlc_firmware_header_v2_1(c.Struct):
  SIZE = 156
  v2_0: struct_rlc_firmware_header_v2_0
  reg_list_format_direct_reg_list_length: int
  save_restore_list_cntl_ucode_ver: int
  save_restore_list_cntl_feature_ver: int
  save_restore_list_cntl_size_bytes: int
  save_restore_list_cntl_offset_bytes: int
  save_restore_list_gpm_ucode_ver: int
  save_restore_list_gpm_feature_ver: int
  save_restore_list_gpm_size_bytes: int
  save_restore_list_gpm_offset_bytes: int
  save_restore_list_srm_ucode_ver: int
  save_restore_list_srm_feature_ver: int
  save_restore_list_srm_size_bytes: int
  save_restore_list_srm_offset_bytes: int
struct_rlc_firmware_header_v2_1.register_fields([('v2_0', struct_rlc_firmware_header_v2_0, 0), ('reg_list_format_direct_reg_list_length', uint32_t, 104), ('save_restore_list_cntl_ucode_ver', uint32_t, 108), ('save_restore_list_cntl_feature_ver', uint32_t, 112), ('save_restore_list_cntl_size_bytes', uint32_t, 116), ('save_restore_list_cntl_offset_bytes', uint32_t, 120), ('save_restore_list_gpm_ucode_ver', uint32_t, 124), ('save_restore_list_gpm_feature_ver', uint32_t, 128), ('save_restore_list_gpm_size_bytes', uint32_t, 132), ('save_restore_list_gpm_offset_bytes', uint32_t, 136), ('save_restore_list_srm_ucode_ver', uint32_t, 140), ('save_restore_list_srm_feature_ver', uint32_t, 144), ('save_restore_list_srm_size_bytes', uint32_t, 148), ('save_restore_list_srm_offset_bytes', uint32_t, 152)])
@c.record
class struct_rlc_firmware_header_v2_2(c.Struct):
  SIZE = 172
  v2_1: struct_rlc_firmware_header_v2_1
  rlc_iram_ucode_size_bytes: int
  rlc_iram_ucode_offset_bytes: int
  rlc_dram_ucode_size_bytes: int
  rlc_dram_ucode_offset_bytes: int
struct_rlc_firmware_header_v2_2.register_fields([('v2_1', struct_rlc_firmware_header_v2_1, 0), ('rlc_iram_ucode_size_bytes', uint32_t, 156), ('rlc_iram_ucode_offset_bytes', uint32_t, 160), ('rlc_dram_ucode_size_bytes', uint32_t, 164), ('rlc_dram_ucode_offset_bytes', uint32_t, 168)])
@c.record
class struct_rlc_firmware_header_v2_3(c.Struct):
  SIZE = 204
  v2_2: struct_rlc_firmware_header_v2_2
  rlcp_ucode_version: int
  rlcp_ucode_feature_version: int
  rlcp_ucode_size_bytes: int
  rlcp_ucode_offset_bytes: int
  rlcv_ucode_version: int
  rlcv_ucode_feature_version: int
  rlcv_ucode_size_bytes: int
  rlcv_ucode_offset_bytes: int
struct_rlc_firmware_header_v2_3.register_fields([('v2_2', struct_rlc_firmware_header_v2_2, 0), ('rlcp_ucode_version', uint32_t, 172), ('rlcp_ucode_feature_version', uint32_t, 176), ('rlcp_ucode_size_bytes', uint32_t, 180), ('rlcp_ucode_offset_bytes', uint32_t, 184), ('rlcv_ucode_version', uint32_t, 188), ('rlcv_ucode_feature_version', uint32_t, 192), ('rlcv_ucode_size_bytes', uint32_t, 196), ('rlcv_ucode_offset_bytes', uint32_t, 200)])
@c.record
class struct_rlc_firmware_header_v2_4(c.Struct):
  SIZE = 244
  v2_3: struct_rlc_firmware_header_v2_3
  global_tap_delays_ucode_size_bytes: int
  global_tap_delays_ucode_offset_bytes: int
  se0_tap_delays_ucode_size_bytes: int
  se0_tap_delays_ucode_offset_bytes: int
  se1_tap_delays_ucode_size_bytes: int
  se1_tap_delays_ucode_offset_bytes: int
  se2_tap_delays_ucode_size_bytes: int
  se2_tap_delays_ucode_offset_bytes: int
  se3_tap_delays_ucode_size_bytes: int
  se3_tap_delays_ucode_offset_bytes: int
struct_rlc_firmware_header_v2_4.register_fields([('v2_3', struct_rlc_firmware_header_v2_3, 0), ('global_tap_delays_ucode_size_bytes', uint32_t, 204), ('global_tap_delays_ucode_offset_bytes', uint32_t, 208), ('se0_tap_delays_ucode_size_bytes', uint32_t, 212), ('se0_tap_delays_ucode_offset_bytes', uint32_t, 216), ('se1_tap_delays_ucode_size_bytes', uint32_t, 220), ('se1_tap_delays_ucode_offset_bytes', uint32_t, 224), ('se2_tap_delays_ucode_size_bytes', uint32_t, 228), ('se2_tap_delays_ucode_offset_bytes', uint32_t, 232), ('se3_tap_delays_ucode_size_bytes', uint32_t, 236), ('se3_tap_delays_ucode_offset_bytes', uint32_t, 240)])
@c.record
class struct_sdma_firmware_header_v1_0(c.Struct):
  SIZE = 48
  header: struct_common_firmware_header
  ucode_feature_version: int
  ucode_change_version: int
  jt_offset: int
  jt_size: int
struct_sdma_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('ucode_feature_version', uint32_t, 32), ('ucode_change_version', uint32_t, 36), ('jt_offset', uint32_t, 40), ('jt_size', uint32_t, 44)])
@c.record
class struct_sdma_firmware_header_v1_1(c.Struct):
  SIZE = 52
  v1_0: struct_sdma_firmware_header_v1_0
  digest_size: int
struct_sdma_firmware_header_v1_1.register_fields([('v1_0', struct_sdma_firmware_header_v1_0, 0), ('digest_size', uint32_t, 48)])
@c.record
class struct_sdma_firmware_header_v2_0(c.Struct):
  SIZE = 64
  header: struct_common_firmware_header
  ucode_feature_version: int
  ctx_ucode_size_bytes: int
  ctx_jt_offset: int
  ctx_jt_size: int
  ctl_ucode_offset: int
  ctl_ucode_size_bytes: int
  ctl_jt_offset: int
  ctl_jt_size: int
struct_sdma_firmware_header_v2_0.register_fields([('header', struct_common_firmware_header, 0), ('ucode_feature_version', uint32_t, 32), ('ctx_ucode_size_bytes', uint32_t, 36), ('ctx_jt_offset', uint32_t, 40), ('ctx_jt_size', uint32_t, 44), ('ctl_ucode_offset', uint32_t, 48), ('ctl_ucode_size_bytes', uint32_t, 52), ('ctl_jt_offset', uint32_t, 56), ('ctl_jt_size', uint32_t, 60)])
@c.record
class struct_vpe_firmware_header_v1_0(c.Struct):
  SIZE = 64
  header: struct_common_firmware_header
  ucode_feature_version: int
  ctx_ucode_size_bytes: int
  ctx_jt_offset: int
  ctx_jt_size: int
  ctl_ucode_offset: int
  ctl_ucode_size_bytes: int
  ctl_jt_offset: int
  ctl_jt_size: int
struct_vpe_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('ucode_feature_version', uint32_t, 32), ('ctx_ucode_size_bytes', uint32_t, 36), ('ctx_jt_offset', uint32_t, 40), ('ctx_jt_size', uint32_t, 44), ('ctl_ucode_offset', uint32_t, 48), ('ctl_ucode_size_bytes', uint32_t, 52), ('ctl_jt_offset', uint32_t, 56), ('ctl_jt_size', uint32_t, 60)])
@c.record
class struct_umsch_mm_firmware_header_v1_0(c.Struct):
  SIZE = 80
  header: struct_common_firmware_header
  umsch_mm_ucode_version: int
  umsch_mm_ucode_size_bytes: int
  umsch_mm_ucode_offset_bytes: int
  umsch_mm_ucode_data_version: int
  umsch_mm_ucode_data_size_bytes: int
  umsch_mm_ucode_data_offset_bytes: int
  umsch_mm_irq_start_addr_lo: int
  umsch_mm_irq_start_addr_hi: int
  umsch_mm_uc_start_addr_lo: int
  umsch_mm_uc_start_addr_hi: int
  umsch_mm_data_start_addr_lo: int
  umsch_mm_data_start_addr_hi: int
struct_umsch_mm_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('umsch_mm_ucode_version', uint32_t, 32), ('umsch_mm_ucode_size_bytes', uint32_t, 36), ('umsch_mm_ucode_offset_bytes', uint32_t, 40), ('umsch_mm_ucode_data_version', uint32_t, 44), ('umsch_mm_ucode_data_size_bytes', uint32_t, 48), ('umsch_mm_ucode_data_offset_bytes', uint32_t, 52), ('umsch_mm_irq_start_addr_lo', uint32_t, 56), ('umsch_mm_irq_start_addr_hi', uint32_t, 60), ('umsch_mm_uc_start_addr_lo', uint32_t, 64), ('umsch_mm_uc_start_addr_hi', uint32_t, 68), ('umsch_mm_data_start_addr_lo', uint32_t, 72), ('umsch_mm_data_start_addr_hi', uint32_t, 76)])
@c.record
class struct_sdma_firmware_header_v3_0(c.Struct):
  SIZE = 44
  header: struct_common_firmware_header
  ucode_feature_version: int
  ucode_offset_bytes: int
  ucode_size_bytes: int
struct_sdma_firmware_header_v3_0.register_fields([('header', struct_common_firmware_header, 0), ('ucode_feature_version', uint32_t, 32), ('ucode_offset_bytes', uint32_t, 36), ('ucode_size_bytes', uint32_t, 40)])
@c.record
class struct_gpu_info_firmware_v1_0(c.Struct):
  SIZE = 60
  gc_num_se: int
  gc_num_cu_per_sh: int
  gc_num_sh_per_se: int
  gc_num_rb_per_se: int
  gc_num_tccs: int
  gc_num_gprs: int
  gc_num_max_gs_thds: int
  gc_gs_table_depth: int
  gc_gsprim_buff_depth: int
  gc_parameter_cache_depth: int
  gc_double_offchip_lds_buffer: int
  gc_wave_size: int
  gc_max_waves_per_simd: int
  gc_max_scratch_slots_per_cu: int
  gc_lds_size: int
struct_gpu_info_firmware_v1_0.register_fields([('gc_num_se', uint32_t, 0), ('gc_num_cu_per_sh', uint32_t, 4), ('gc_num_sh_per_se', uint32_t, 8), ('gc_num_rb_per_se', uint32_t, 12), ('gc_num_tccs', uint32_t, 16), ('gc_num_gprs', uint32_t, 20), ('gc_num_max_gs_thds', uint32_t, 24), ('gc_gs_table_depth', uint32_t, 28), ('gc_gsprim_buff_depth', uint32_t, 32), ('gc_parameter_cache_depth', uint32_t, 36), ('gc_double_offchip_lds_buffer', uint32_t, 40), ('gc_wave_size', uint32_t, 44), ('gc_max_waves_per_simd', uint32_t, 48), ('gc_max_scratch_slots_per_cu', uint32_t, 52), ('gc_lds_size', uint32_t, 56)])
@c.record
class struct_gpu_info_firmware_v1_1(c.Struct):
  SIZE = 68
  v1_0: struct_gpu_info_firmware_v1_0
  num_sc_per_sh: int
  num_packer_per_sc: int
struct_gpu_info_firmware_v1_1.register_fields([('v1_0', struct_gpu_info_firmware_v1_0, 0), ('num_sc_per_sh', uint32_t, 60), ('num_packer_per_sc', uint32_t, 64)])
@c.record
class struct_gpu_info_firmware_header_v1_0(c.Struct):
  SIZE = 36
  header: struct_common_firmware_header
  version_major: int
  version_minor: int
struct_gpu_info_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('version_major', uint16_t, 32), ('version_minor', uint16_t, 34)])
@c.record
class struct_dmcu_firmware_header_v1_0(c.Struct):
  SIZE = 40
  header: struct_common_firmware_header
  intv_offset_bytes: int
  intv_size_bytes: int
struct_dmcu_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('intv_offset_bytes', uint32_t, 32), ('intv_size_bytes', uint32_t, 36)])
@c.record
class struct_dmcub_firmware_header_v1_0(c.Struct):
  SIZE = 40
  header: struct_common_firmware_header
  inst_const_bytes: int
  bss_data_bytes: int
struct_dmcub_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('inst_const_bytes', uint32_t, 32), ('bss_data_bytes', uint32_t, 36)])
@c.record
class struct_imu_firmware_header_v1_0(c.Struct):
  SIZE = 48
  header: struct_common_firmware_header
  imu_iram_ucode_size_bytes: int
  imu_iram_ucode_offset_bytes: int
  imu_dram_ucode_size_bytes: int
  imu_dram_ucode_offset_bytes: int
struct_imu_firmware_header_v1_0.register_fields([('header', struct_common_firmware_header, 0), ('imu_iram_ucode_size_bytes', uint32_t, 32), ('imu_iram_ucode_offset_bytes', uint32_t, 36), ('imu_dram_ucode_size_bytes', uint32_t, 40), ('imu_dram_ucode_offset_bytes', uint32_t, 44)])
@c.record
class union_amdgpu_firmware_header(c.Struct):
  SIZE = 256
  common: struct_common_firmware_header
  mc: struct_mc_firmware_header_v1_0
  smc: struct_smc_firmware_header_v1_0
  smc_v2_0: struct_smc_firmware_header_v2_0
  psp: struct_psp_firmware_header_v1_0
  psp_v1_1: struct_psp_firmware_header_v1_1
  psp_v1_3: struct_psp_firmware_header_v1_3
  psp_v2_0: struct_psp_firmware_header_v2_0
  psp_v2_1: struct_psp_firmware_header_v2_0
  ta: struct_ta_firmware_header_v1_0
  ta_v2_0: struct_ta_firmware_header_v2_0
  gfx: struct_gfx_firmware_header_v1_0
  gfx_v2_0: struct_gfx_firmware_header_v2_0
  rlc: struct_rlc_firmware_header_v1_0
  rlc_v2_0: struct_rlc_firmware_header_v2_0
  rlc_v2_1: struct_rlc_firmware_header_v2_1
  rlc_v2_2: struct_rlc_firmware_header_v2_2
  rlc_v2_3: struct_rlc_firmware_header_v2_3
  rlc_v2_4: struct_rlc_firmware_header_v2_4
  sdma: struct_sdma_firmware_header_v1_0
  sdma_v1_1: struct_sdma_firmware_header_v1_1
  sdma_v2_0: struct_sdma_firmware_header_v2_0
  sdma_v3_0: struct_sdma_firmware_header_v3_0
  gpu_info: struct_gpu_info_firmware_header_v1_0
  dmcu: struct_dmcu_firmware_header_v1_0
  dmcub: struct_dmcub_firmware_header_v1_0
  imu: struct_imu_firmware_header_v1_0
  raw: c.Array[ctypes.c_ubyte, Literal[256]]
union_amdgpu_firmware_header.register_fields([('common', struct_common_firmware_header, 0), ('mc', struct_mc_firmware_header_v1_0, 0), ('smc', struct_smc_firmware_header_v1_0, 0), ('smc_v2_0', struct_smc_firmware_header_v2_0, 0), ('psp', struct_psp_firmware_header_v1_0, 0), ('psp_v1_1', struct_psp_firmware_header_v1_1, 0), ('psp_v1_3', struct_psp_firmware_header_v1_3, 0), ('psp_v2_0', struct_psp_firmware_header_v2_0, 0), ('psp_v2_1', struct_psp_firmware_header_v2_0, 0), ('ta', struct_ta_firmware_header_v1_0, 0), ('ta_v2_0', struct_ta_firmware_header_v2_0, 0), ('gfx', struct_gfx_firmware_header_v1_0, 0), ('gfx_v2_0', struct_gfx_firmware_header_v2_0, 0), ('rlc', struct_rlc_firmware_header_v1_0, 0), ('rlc_v2_0', struct_rlc_firmware_header_v2_0, 0), ('rlc_v2_1', struct_rlc_firmware_header_v2_1, 0), ('rlc_v2_2', struct_rlc_firmware_header_v2_2, 0), ('rlc_v2_3', struct_rlc_firmware_header_v2_3, 0), ('rlc_v2_4', struct_rlc_firmware_header_v2_4, 0), ('sdma', struct_sdma_firmware_header_v1_0, 0), ('sdma_v1_1', struct_sdma_firmware_header_v1_1, 0), ('sdma_v2_0', struct_sdma_firmware_header_v2_0, 0), ('sdma_v3_0', struct_sdma_firmware_header_v3_0, 0), ('gpu_info', struct_gpu_info_firmware_header_v1_0, 0), ('dmcu', struct_dmcu_firmware_header_v1_0, 0), ('dmcub', struct_dmcub_firmware_header_v1_0, 0), ('imu', struct_imu_firmware_header_v1_0, 0), ('raw', c.Array[uint8_t, Literal[256]], 0)])
enum_AMDGPU_UCODE_ID: dict[int, str] = {(AMDGPU_UCODE_ID_CAP:=0): 'AMDGPU_UCODE_ID_CAP', (AMDGPU_UCODE_ID_SDMA0:=1): 'AMDGPU_UCODE_ID_SDMA0', (AMDGPU_UCODE_ID_SDMA1:=2): 'AMDGPU_UCODE_ID_SDMA1', (AMDGPU_UCODE_ID_SDMA2:=3): 'AMDGPU_UCODE_ID_SDMA2', (AMDGPU_UCODE_ID_SDMA3:=4): 'AMDGPU_UCODE_ID_SDMA3', (AMDGPU_UCODE_ID_SDMA4:=5): 'AMDGPU_UCODE_ID_SDMA4', (AMDGPU_UCODE_ID_SDMA5:=6): 'AMDGPU_UCODE_ID_SDMA5', (AMDGPU_UCODE_ID_SDMA6:=7): 'AMDGPU_UCODE_ID_SDMA6', (AMDGPU_UCODE_ID_SDMA7:=8): 'AMDGPU_UCODE_ID_SDMA7', (AMDGPU_UCODE_ID_SDMA_UCODE_TH0:=9): 'AMDGPU_UCODE_ID_SDMA_UCODE_TH0', (AMDGPU_UCODE_ID_SDMA_UCODE_TH1:=10): 'AMDGPU_UCODE_ID_SDMA_UCODE_TH1', (AMDGPU_UCODE_ID_SDMA_RS64:=11): 'AMDGPU_UCODE_ID_SDMA_RS64', (AMDGPU_UCODE_ID_CP_CE:=12): 'AMDGPU_UCODE_ID_CP_CE', (AMDGPU_UCODE_ID_CP_PFP:=13): 'AMDGPU_UCODE_ID_CP_PFP', (AMDGPU_UCODE_ID_CP_ME:=14): 'AMDGPU_UCODE_ID_CP_ME', (AMDGPU_UCODE_ID_CP_RS64_PFP:=15): 'AMDGPU_UCODE_ID_CP_RS64_PFP', (AMDGPU_UCODE_ID_CP_RS64_ME:=16): 'AMDGPU_UCODE_ID_CP_RS64_ME', (AMDGPU_UCODE_ID_CP_RS64_MEC:=17): 'AMDGPU_UCODE_ID_CP_RS64_MEC', (AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK:=18): 'AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK', (AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK:=19): 'AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK', (AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK:=20): 'AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK', (AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK:=21): 'AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK', (AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK:=22): 'AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK', (AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK:=23): 'AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK', (AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK:=24): 'AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK', (AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK:=25): 'AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK', (AMDGPU_UCODE_ID_CP_MEC1:=26): 'AMDGPU_UCODE_ID_CP_MEC1', (AMDGPU_UCODE_ID_CP_MEC1_JT:=27): 'AMDGPU_UCODE_ID_CP_MEC1_JT', (AMDGPU_UCODE_ID_CP_MEC2:=28): 'AMDGPU_UCODE_ID_CP_MEC2', (AMDGPU_UCODE_ID_CP_MEC2_JT:=29): 'AMDGPU_UCODE_ID_CP_MEC2_JT', (AMDGPU_UCODE_ID_CP_MES:=30): 'AMDGPU_UCODE_ID_CP_MES', (AMDGPU_UCODE_ID_CP_MES_DATA:=31): 'AMDGPU_UCODE_ID_CP_MES_DATA', (AMDGPU_UCODE_ID_CP_MES1:=32): 'AMDGPU_UCODE_ID_CP_MES1', (AMDGPU_UCODE_ID_CP_MES1_DATA:=33): 'AMDGPU_UCODE_ID_CP_MES1_DATA', (AMDGPU_UCODE_ID_IMU_I:=34): 'AMDGPU_UCODE_ID_IMU_I', (AMDGPU_UCODE_ID_IMU_D:=35): 'AMDGPU_UCODE_ID_IMU_D', (AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS:=36): 'AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS', (AMDGPU_UCODE_ID_SE0_TAP_DELAYS:=37): 'AMDGPU_UCODE_ID_SE0_TAP_DELAYS', (AMDGPU_UCODE_ID_SE1_TAP_DELAYS:=38): 'AMDGPU_UCODE_ID_SE1_TAP_DELAYS', (AMDGPU_UCODE_ID_SE2_TAP_DELAYS:=39): 'AMDGPU_UCODE_ID_SE2_TAP_DELAYS', (AMDGPU_UCODE_ID_SE3_TAP_DELAYS:=40): 'AMDGPU_UCODE_ID_SE3_TAP_DELAYS', (AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL:=41): 'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL', (AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM:=42): 'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM', (AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM:=43): 'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM', (AMDGPU_UCODE_ID_RLC_IRAM:=44): 'AMDGPU_UCODE_ID_RLC_IRAM', (AMDGPU_UCODE_ID_RLC_DRAM:=45): 'AMDGPU_UCODE_ID_RLC_DRAM', (AMDGPU_UCODE_ID_RLC_P:=46): 'AMDGPU_UCODE_ID_RLC_P', (AMDGPU_UCODE_ID_RLC_V:=47): 'AMDGPU_UCODE_ID_RLC_V', (AMDGPU_UCODE_ID_RLC_G:=48): 'AMDGPU_UCODE_ID_RLC_G', (AMDGPU_UCODE_ID_STORAGE:=49): 'AMDGPU_UCODE_ID_STORAGE', (AMDGPU_UCODE_ID_SMC:=50): 'AMDGPU_UCODE_ID_SMC', (AMDGPU_UCODE_ID_PPTABLE:=51): 'AMDGPU_UCODE_ID_PPTABLE', (AMDGPU_UCODE_ID_UVD:=52): 'AMDGPU_UCODE_ID_UVD', (AMDGPU_UCODE_ID_UVD1:=53): 'AMDGPU_UCODE_ID_UVD1', (AMDGPU_UCODE_ID_VCE:=54): 'AMDGPU_UCODE_ID_VCE', (AMDGPU_UCODE_ID_VCN:=55): 'AMDGPU_UCODE_ID_VCN', (AMDGPU_UCODE_ID_VCN1:=56): 'AMDGPU_UCODE_ID_VCN1', (AMDGPU_UCODE_ID_DMCU_ERAM:=57): 'AMDGPU_UCODE_ID_DMCU_ERAM', (AMDGPU_UCODE_ID_DMCU_INTV:=58): 'AMDGPU_UCODE_ID_DMCU_INTV', (AMDGPU_UCODE_ID_VCN0_RAM:=59): 'AMDGPU_UCODE_ID_VCN0_RAM', (AMDGPU_UCODE_ID_VCN1_RAM:=60): 'AMDGPU_UCODE_ID_VCN1_RAM', (AMDGPU_UCODE_ID_DMCUB:=61): 'AMDGPU_UCODE_ID_DMCUB', (AMDGPU_UCODE_ID_VPE_CTX:=62): 'AMDGPU_UCODE_ID_VPE_CTX', (AMDGPU_UCODE_ID_VPE_CTL:=63): 'AMDGPU_UCODE_ID_VPE_CTL', (AMDGPU_UCODE_ID_VPE:=64): 'AMDGPU_UCODE_ID_VPE', (AMDGPU_UCODE_ID_UMSCH_MM_UCODE:=65): 'AMDGPU_UCODE_ID_UMSCH_MM_UCODE', (AMDGPU_UCODE_ID_UMSCH_MM_DATA:=66): 'AMDGPU_UCODE_ID_UMSCH_MM_DATA', (AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER:=67): 'AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER', (AMDGPU_UCODE_ID_P2S_TABLE:=68): 'AMDGPU_UCODE_ID_P2S_TABLE', (AMDGPU_UCODE_ID_JPEG_RAM:=69): 'AMDGPU_UCODE_ID_JPEG_RAM', (AMDGPU_UCODE_ID_ISP:=70): 'AMDGPU_UCODE_ID_ISP', (AMDGPU_UCODE_ID_MAXIMUM:=71): 'AMDGPU_UCODE_ID_MAXIMUM'}
enum_AMDGPU_UCODE_STATUS: dict[int, str] = {(AMDGPU_UCODE_STATUS_INVALID:=0): 'AMDGPU_UCODE_STATUS_INVALID', (AMDGPU_UCODE_STATUS_NOT_LOADED:=1): 'AMDGPU_UCODE_STATUS_NOT_LOADED', (AMDGPU_UCODE_STATUS_LOADED:=2): 'AMDGPU_UCODE_STATUS_LOADED'}
enum_amdgpu_firmware_load_type: dict[int, str] = {(AMDGPU_FW_LOAD_DIRECT:=0): 'AMDGPU_FW_LOAD_DIRECT', (AMDGPU_FW_LOAD_PSP:=1): 'AMDGPU_FW_LOAD_PSP', (AMDGPU_FW_LOAD_SMU:=2): 'AMDGPU_FW_LOAD_SMU', (AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO:=3): 'AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO'}
@c.record
class struct_amdgpu_firmware_info(c.Struct):
  SIZE = 48
  ucode_id: int
  fw: c.POINTER[struct_firmware]
  mc_addr: int
  kaddr: ctypes.c_void_p
  ucode_size: int
  tmr_mc_addr_lo: int
  tmr_mc_addr_hi: int
class struct_firmware(c.Struct): pass
struct_amdgpu_firmware_info.register_fields([('ucode_id', ctypes.c_uint32, 0), ('fw', c.POINTER[struct_firmware], 8), ('mc_addr', uint64_t, 16), ('kaddr', ctypes.c_void_p, 24), ('ucode_size', uint32_t, 32), ('tmr_mc_addr_lo', uint32_t, 36), ('tmr_mc_addr_hi', uint32_t, 40)])
enum_psp_gfx_crtl_cmd_id: dict[int, str] = {(GFX_CTRL_CMD_ID_INIT_RBI_RING:=65536): 'GFX_CTRL_CMD_ID_INIT_RBI_RING', (GFX_CTRL_CMD_ID_INIT_GPCOM_RING:=131072): 'GFX_CTRL_CMD_ID_INIT_GPCOM_RING', (GFX_CTRL_CMD_ID_DESTROY_RINGS:=196608): 'GFX_CTRL_CMD_ID_DESTROY_RINGS', (GFX_CTRL_CMD_ID_CAN_INIT_RINGS:=262144): 'GFX_CTRL_CMD_ID_CAN_INIT_RINGS', (GFX_CTRL_CMD_ID_ENABLE_INT:=327680): 'GFX_CTRL_CMD_ID_ENABLE_INT', (GFX_CTRL_CMD_ID_DISABLE_INT:=393216): 'GFX_CTRL_CMD_ID_DISABLE_INT', (GFX_CTRL_CMD_ID_MODE1_RST:=458752): 'GFX_CTRL_CMD_ID_MODE1_RST', (GFX_CTRL_CMD_ID_GBR_IH_SET:=524288): 'GFX_CTRL_CMD_ID_GBR_IH_SET', (GFX_CTRL_CMD_ID_CONSUME_CMD:=589824): 'GFX_CTRL_CMD_ID_CONSUME_CMD', (GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING:=786432): 'GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING', (GFX_CTRL_CMD_ID_MAX:=983040): 'GFX_CTRL_CMD_ID_MAX'}
@c.record
class struct_psp_gfx_ctrl(c.Struct):
  SIZE = 32
  cmd_resp: int
  rbi_wptr: int
  rbi_rptr: int
  gpcom_wptr: int
  gpcom_rptr: int
  ring_addr_lo: int
  ring_addr_hi: int
  ring_buf_size: int
struct_psp_gfx_ctrl.register_fields([('cmd_resp', ctypes.c_uint32, 0), ('rbi_wptr', ctypes.c_uint32, 4), ('rbi_rptr', ctypes.c_uint32, 8), ('gpcom_wptr', ctypes.c_uint32, 12), ('gpcom_rptr', ctypes.c_uint32, 16), ('ring_addr_lo', ctypes.c_uint32, 20), ('ring_addr_hi', ctypes.c_uint32, 24), ('ring_buf_size', ctypes.c_uint32, 28)])
enum_psp_gfx_cmd_id: dict[int, str] = {(GFX_CMD_ID_LOAD_TA:=1): 'GFX_CMD_ID_LOAD_TA', (GFX_CMD_ID_UNLOAD_TA:=2): 'GFX_CMD_ID_UNLOAD_TA', (GFX_CMD_ID_INVOKE_CMD:=3): 'GFX_CMD_ID_INVOKE_CMD', (GFX_CMD_ID_LOAD_ASD:=4): 'GFX_CMD_ID_LOAD_ASD', (GFX_CMD_ID_SETUP_TMR:=5): 'GFX_CMD_ID_SETUP_TMR', (GFX_CMD_ID_LOAD_IP_FW:=6): 'GFX_CMD_ID_LOAD_IP_FW', (GFX_CMD_ID_DESTROY_TMR:=7): 'GFX_CMD_ID_DESTROY_TMR', (GFX_CMD_ID_SAVE_RESTORE:=8): 'GFX_CMD_ID_SAVE_RESTORE', (GFX_CMD_ID_SETUP_VMR:=9): 'GFX_CMD_ID_SETUP_VMR', (GFX_CMD_ID_DESTROY_VMR:=10): 'GFX_CMD_ID_DESTROY_VMR', (GFX_CMD_ID_PROG_REG:=11): 'GFX_CMD_ID_PROG_REG', (GFX_CMD_ID_GET_FW_ATTESTATION:=15): 'GFX_CMD_ID_GET_FW_ATTESTATION', (GFX_CMD_ID_LOAD_TOC:=32): 'GFX_CMD_ID_LOAD_TOC', (GFX_CMD_ID_AUTOLOAD_RLC:=33): 'GFX_CMD_ID_AUTOLOAD_RLC', (GFX_CMD_ID_BOOT_CFG:=34): 'GFX_CMD_ID_BOOT_CFG', (GFX_CMD_ID_SRIOV_SPATIAL_PART:=39): 'GFX_CMD_ID_SRIOV_SPATIAL_PART'}
enum_psp_gfx_boot_config_cmd: dict[int, str] = {(BOOTCFG_CMD_SET:=1): 'BOOTCFG_CMD_SET', (BOOTCFG_CMD_GET:=2): 'BOOTCFG_CMD_GET', (BOOTCFG_CMD_INVALIDATE:=3): 'BOOTCFG_CMD_INVALIDATE'}
enum_psp_gfx_boot_config: dict[int, str] = {(BOOT_CONFIG_GECC:=1): 'BOOT_CONFIG_GECC'}
@c.record
class struct_psp_gfx_cmd_load_ta(c.Struct):
  SIZE = 24
  app_phy_addr_lo: int
  app_phy_addr_hi: int
  app_len: int
  cmd_buf_phy_addr_lo: int
  cmd_buf_phy_addr_hi: int
  cmd_buf_len: int
struct_psp_gfx_cmd_load_ta.register_fields([('app_phy_addr_lo', ctypes.c_uint32, 0), ('app_phy_addr_hi', ctypes.c_uint32, 4), ('app_len', ctypes.c_uint32, 8), ('cmd_buf_phy_addr_lo', ctypes.c_uint32, 12), ('cmd_buf_phy_addr_hi', ctypes.c_uint32, 16), ('cmd_buf_len', ctypes.c_uint32, 20)])
@c.record
class struct_psp_gfx_cmd_unload_ta(c.Struct):
  SIZE = 4
  session_id: int
struct_psp_gfx_cmd_unload_ta.register_fields([('session_id', ctypes.c_uint32, 0)])
@c.record
class struct_psp_gfx_buf_desc(c.Struct):
  SIZE = 12
  buf_phy_addr_lo: int
  buf_phy_addr_hi: int
  buf_size: int
struct_psp_gfx_buf_desc.register_fields([('buf_phy_addr_lo', ctypes.c_uint32, 0), ('buf_phy_addr_hi', ctypes.c_uint32, 4), ('buf_size', ctypes.c_uint32, 8)])
@c.record
class struct_psp_gfx_buf_list(c.Struct):
  SIZE = 776
  num_desc: int
  total_size: int
  buf_desc: c.Array[struct_psp_gfx_buf_desc, Literal[64]]
struct_psp_gfx_buf_list.register_fields([('num_desc', ctypes.c_uint32, 0), ('total_size', ctypes.c_uint32, 4), ('buf_desc', c.Array[struct_psp_gfx_buf_desc, Literal[64]], 8)])
@c.record
class struct_psp_gfx_cmd_invoke_cmd(c.Struct):
  SIZE = 784
  session_id: int
  ta_cmd_id: int
  buf: struct_psp_gfx_buf_list
struct_psp_gfx_cmd_invoke_cmd.register_fields([('session_id', ctypes.c_uint32, 0), ('ta_cmd_id', ctypes.c_uint32, 4), ('buf', struct_psp_gfx_buf_list, 8)])
@c.record
class struct_psp_gfx_cmd_setup_tmr(c.Struct):
  SIZE = 24
  buf_phy_addr_lo: int
  buf_phy_addr_hi: int
  buf_size: int
  bitfield: struct_psp_gfx_cmd_setup_tmr_bitfield
  tmr_flags: int
  system_phy_addr_lo: int
  system_phy_addr_hi: int
@c.record
class struct_psp_gfx_cmd_setup_tmr_bitfield(c.Struct):
  SIZE = 4
  sriov_enabled: int
  virt_phy_addr: int
  reserved: int
struct_psp_gfx_cmd_setup_tmr_bitfield.register_fields([('sriov_enabled', ctypes.c_uint32, 0, 1, 0), ('virt_phy_addr', ctypes.c_uint32, 0, 1, 1), ('reserved', ctypes.c_uint32, 0, 30, 2)])
struct_psp_gfx_cmd_setup_tmr.register_fields([('buf_phy_addr_lo', ctypes.c_uint32, 0), ('buf_phy_addr_hi', ctypes.c_uint32, 4), ('buf_size', ctypes.c_uint32, 8), ('bitfield', struct_psp_gfx_cmd_setup_tmr_bitfield, 12), ('tmr_flags', ctypes.c_uint32, 12), ('system_phy_addr_lo', ctypes.c_uint32, 16), ('system_phy_addr_hi', ctypes.c_uint32, 20)])
enum_psp_gfx_fw_type: dict[int, str] = {(GFX_FW_TYPE_NONE:=0): 'GFX_FW_TYPE_NONE', (GFX_FW_TYPE_CP_ME:=1): 'GFX_FW_TYPE_CP_ME', (GFX_FW_TYPE_CP_PFP:=2): 'GFX_FW_TYPE_CP_PFP', (GFX_FW_TYPE_CP_CE:=3): 'GFX_FW_TYPE_CP_CE', (GFX_FW_TYPE_CP_MEC:=4): 'GFX_FW_TYPE_CP_MEC', (GFX_FW_TYPE_CP_MEC_ME1:=5): 'GFX_FW_TYPE_CP_MEC_ME1', (GFX_FW_TYPE_CP_MEC_ME2:=6): 'GFX_FW_TYPE_CP_MEC_ME2', (GFX_FW_TYPE_RLC_V:=7): 'GFX_FW_TYPE_RLC_V', (GFX_FW_TYPE_RLC_G:=8): 'GFX_FW_TYPE_RLC_G', (GFX_FW_TYPE_SDMA0:=9): 'GFX_FW_TYPE_SDMA0', (GFX_FW_TYPE_SDMA1:=10): 'GFX_FW_TYPE_SDMA1', (GFX_FW_TYPE_DMCU_ERAM:=11): 'GFX_FW_TYPE_DMCU_ERAM', (GFX_FW_TYPE_DMCU_ISR:=12): 'GFX_FW_TYPE_DMCU_ISR', (GFX_FW_TYPE_VCN:=13): 'GFX_FW_TYPE_VCN', (GFX_FW_TYPE_UVD:=14): 'GFX_FW_TYPE_UVD', (GFX_FW_TYPE_VCE:=15): 'GFX_FW_TYPE_VCE', (GFX_FW_TYPE_ISP:=16): 'GFX_FW_TYPE_ISP', (GFX_FW_TYPE_ACP:=17): 'GFX_FW_TYPE_ACP', (GFX_FW_TYPE_SMU:=18): 'GFX_FW_TYPE_SMU', (GFX_FW_TYPE_MMSCH:=19): 'GFX_FW_TYPE_MMSCH', (GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM:=20): 'GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM', (GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM:=21): 'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM', (GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL:=22): 'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL', (GFX_FW_TYPE_UVD1:=23): 'GFX_FW_TYPE_UVD1', (GFX_FW_TYPE_TOC:=24): 'GFX_FW_TYPE_TOC', (GFX_FW_TYPE_RLC_P:=25): 'GFX_FW_TYPE_RLC_P', (GFX_FW_TYPE_RLC_IRAM:=26): 'GFX_FW_TYPE_RLC_IRAM', (GFX_FW_TYPE_GLOBAL_TAP_DELAYS:=27): 'GFX_FW_TYPE_GLOBAL_TAP_DELAYS', (GFX_FW_TYPE_SE0_TAP_DELAYS:=28): 'GFX_FW_TYPE_SE0_TAP_DELAYS', (GFX_FW_TYPE_SE1_TAP_DELAYS:=29): 'GFX_FW_TYPE_SE1_TAP_DELAYS', (GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS:=30): 'GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS', (GFX_FW_TYPE_SDMA0_JT:=31): 'GFX_FW_TYPE_SDMA0_JT', (GFX_FW_TYPE_SDMA1_JT:=32): 'GFX_FW_TYPE_SDMA1_JT', (GFX_FW_TYPE_CP_MES:=33): 'GFX_FW_TYPE_CP_MES', (GFX_FW_TYPE_MES_STACK:=34): 'GFX_FW_TYPE_MES_STACK', (GFX_FW_TYPE_RLC_SRM_DRAM_SR:=35): 'GFX_FW_TYPE_RLC_SRM_DRAM_SR', (GFX_FW_TYPE_RLCG_SCRATCH_SR:=36): 'GFX_FW_TYPE_RLCG_SCRATCH_SR', (GFX_FW_TYPE_RLCP_SCRATCH_SR:=37): 'GFX_FW_TYPE_RLCP_SCRATCH_SR', (GFX_FW_TYPE_RLCV_SCRATCH_SR:=38): 'GFX_FW_TYPE_RLCV_SCRATCH_SR', (GFX_FW_TYPE_RLX6_DRAM_SR:=39): 'GFX_FW_TYPE_RLX6_DRAM_SR', (GFX_FW_TYPE_SDMA0_PG_CONTEXT:=40): 'GFX_FW_TYPE_SDMA0_PG_CONTEXT', (GFX_FW_TYPE_SDMA1_PG_CONTEXT:=41): 'GFX_FW_TYPE_SDMA1_PG_CONTEXT', (GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM:=42): 'GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM', (GFX_FW_TYPE_SE0_MUX_SELECT_RAM:=43): 'GFX_FW_TYPE_SE0_MUX_SELECT_RAM', (GFX_FW_TYPE_SE1_MUX_SELECT_RAM:=44): 'GFX_FW_TYPE_SE1_MUX_SELECT_RAM', (GFX_FW_TYPE_ACCUM_CTRL_RAM:=45): 'GFX_FW_TYPE_ACCUM_CTRL_RAM', (GFX_FW_TYPE_RLCP_CAM:=46): 'GFX_FW_TYPE_RLCP_CAM', (GFX_FW_TYPE_RLC_SPP_CAM_EXT:=47): 'GFX_FW_TYPE_RLC_SPP_CAM_EXT', (GFX_FW_TYPE_RLC_DRAM_BOOT:=48): 'GFX_FW_TYPE_RLC_DRAM_BOOT', (GFX_FW_TYPE_VCN0_RAM:=49): 'GFX_FW_TYPE_VCN0_RAM', (GFX_FW_TYPE_VCN1_RAM:=50): 'GFX_FW_TYPE_VCN1_RAM', (GFX_FW_TYPE_DMUB:=51): 'GFX_FW_TYPE_DMUB', (GFX_FW_TYPE_SDMA2:=52): 'GFX_FW_TYPE_SDMA2', (GFX_FW_TYPE_SDMA3:=53): 'GFX_FW_TYPE_SDMA3', (GFX_FW_TYPE_SDMA4:=54): 'GFX_FW_TYPE_SDMA4', (GFX_FW_TYPE_SDMA5:=55): 'GFX_FW_TYPE_SDMA5', (GFX_FW_TYPE_SDMA6:=56): 'GFX_FW_TYPE_SDMA6', (GFX_FW_TYPE_SDMA7:=57): 'GFX_FW_TYPE_SDMA7', (GFX_FW_TYPE_VCN1:=58): 'GFX_FW_TYPE_VCN1', (GFX_FW_TYPE_CAP:=62): 'GFX_FW_TYPE_CAP', (GFX_FW_TYPE_SE2_TAP_DELAYS:=65): 'GFX_FW_TYPE_SE2_TAP_DELAYS', (GFX_FW_TYPE_SE3_TAP_DELAYS:=66): 'GFX_FW_TYPE_SE3_TAP_DELAYS', (GFX_FW_TYPE_REG_LIST:=67): 'GFX_FW_TYPE_REG_LIST', (GFX_FW_TYPE_IMU_I:=68): 'GFX_FW_TYPE_IMU_I', (GFX_FW_TYPE_IMU_D:=69): 'GFX_FW_TYPE_IMU_D', (GFX_FW_TYPE_LSDMA:=70): 'GFX_FW_TYPE_LSDMA', (GFX_FW_TYPE_SDMA_UCODE_TH0:=71): 'GFX_FW_TYPE_SDMA_UCODE_TH0', (GFX_FW_TYPE_SDMA_UCODE_TH1:=72): 'GFX_FW_TYPE_SDMA_UCODE_TH1', (GFX_FW_TYPE_PPTABLE:=73): 'GFX_FW_TYPE_PPTABLE', (GFX_FW_TYPE_DISCRETE_USB4:=74): 'GFX_FW_TYPE_DISCRETE_USB4', (GFX_FW_TYPE_TA:=75): 'GFX_FW_TYPE_TA', (GFX_FW_TYPE_RS64_MES:=76): 'GFX_FW_TYPE_RS64_MES', (GFX_FW_TYPE_RS64_MES_STACK:=77): 'GFX_FW_TYPE_RS64_MES_STACK', (GFX_FW_TYPE_RS64_KIQ:=78): 'GFX_FW_TYPE_RS64_KIQ', (GFX_FW_TYPE_RS64_KIQ_STACK:=79): 'GFX_FW_TYPE_RS64_KIQ_STACK', (GFX_FW_TYPE_ISP_DATA:=80): 'GFX_FW_TYPE_ISP_DATA', (GFX_FW_TYPE_CP_MES_KIQ:=81): 'GFX_FW_TYPE_CP_MES_KIQ', (GFX_FW_TYPE_MES_KIQ_STACK:=82): 'GFX_FW_TYPE_MES_KIQ_STACK', (GFX_FW_TYPE_UMSCH_DATA:=83): 'GFX_FW_TYPE_UMSCH_DATA', (GFX_FW_TYPE_UMSCH_UCODE:=84): 'GFX_FW_TYPE_UMSCH_UCODE', (GFX_FW_TYPE_UMSCH_CMD_BUFFER:=85): 'GFX_FW_TYPE_UMSCH_CMD_BUFFER', (GFX_FW_TYPE_USB_DP_COMBO_PHY:=86): 'GFX_FW_TYPE_USB_DP_COMBO_PHY', (GFX_FW_TYPE_RS64_PFP:=87): 'GFX_FW_TYPE_RS64_PFP', (GFX_FW_TYPE_RS64_ME:=88): 'GFX_FW_TYPE_RS64_ME', (GFX_FW_TYPE_RS64_MEC:=89): 'GFX_FW_TYPE_RS64_MEC', (GFX_FW_TYPE_RS64_PFP_P0_STACK:=90): 'GFX_FW_TYPE_RS64_PFP_P0_STACK', (GFX_FW_TYPE_RS64_PFP_P1_STACK:=91): 'GFX_FW_TYPE_RS64_PFP_P1_STACK', (GFX_FW_TYPE_RS64_ME_P0_STACK:=92): 'GFX_FW_TYPE_RS64_ME_P0_STACK', (GFX_FW_TYPE_RS64_ME_P1_STACK:=93): 'GFX_FW_TYPE_RS64_ME_P1_STACK', (GFX_FW_TYPE_RS64_MEC_P0_STACK:=94): 'GFX_FW_TYPE_RS64_MEC_P0_STACK', (GFX_FW_TYPE_RS64_MEC_P1_STACK:=95): 'GFX_FW_TYPE_RS64_MEC_P1_STACK', (GFX_FW_TYPE_RS64_MEC_P2_STACK:=96): 'GFX_FW_TYPE_RS64_MEC_P2_STACK', (GFX_FW_TYPE_RS64_MEC_P3_STACK:=97): 'GFX_FW_TYPE_RS64_MEC_P3_STACK', (GFX_FW_TYPE_VPEC_FW1:=100): 'GFX_FW_TYPE_VPEC_FW1', (GFX_FW_TYPE_VPEC_FW2:=101): 'GFX_FW_TYPE_VPEC_FW2', (GFX_FW_TYPE_VPE:=102): 'GFX_FW_TYPE_VPE', (GFX_FW_TYPE_JPEG_RAM:=128): 'GFX_FW_TYPE_JPEG_RAM', (GFX_FW_TYPE_P2S_TABLE:=129): 'GFX_FW_TYPE_P2S_TABLE', (GFX_FW_TYPE_MAX:=130): 'GFX_FW_TYPE_MAX'}
@c.record
class struct_psp_gfx_cmd_load_ip_fw(c.Struct):
  SIZE = 16
  fw_phy_addr_lo: int
  fw_phy_addr_hi: int
  fw_size: int
  fw_type: int
struct_psp_gfx_cmd_load_ip_fw.register_fields([('fw_phy_addr_lo', ctypes.c_uint32, 0), ('fw_phy_addr_hi', ctypes.c_uint32, 4), ('fw_size', ctypes.c_uint32, 8), ('fw_type', ctypes.c_uint32, 12)])
@c.record
class struct_psp_gfx_cmd_save_restore_ip_fw(c.Struct):
  SIZE = 20
  save_fw: int
  save_restore_addr_lo: int
  save_restore_addr_hi: int
  buf_size: int
  fw_type: int
struct_psp_gfx_cmd_save_restore_ip_fw.register_fields([('save_fw', ctypes.c_uint32, 0), ('save_restore_addr_lo', ctypes.c_uint32, 4), ('save_restore_addr_hi', ctypes.c_uint32, 8), ('buf_size', ctypes.c_uint32, 12), ('fw_type', ctypes.c_uint32, 16)])
@c.record
class struct_psp_gfx_cmd_reg_prog(c.Struct):
  SIZE = 8
  reg_value: int
  reg_id: int
struct_psp_gfx_cmd_reg_prog.register_fields([('reg_value', ctypes.c_uint32, 0), ('reg_id', ctypes.c_uint32, 4)])
@c.record
class struct_psp_gfx_cmd_load_toc(c.Struct):
  SIZE = 12
  toc_phy_addr_lo: int
  toc_phy_addr_hi: int
  toc_size: int
struct_psp_gfx_cmd_load_toc.register_fields([('toc_phy_addr_lo', ctypes.c_uint32, 0), ('toc_phy_addr_hi', ctypes.c_uint32, 4), ('toc_size', ctypes.c_uint32, 8)])
@c.record
class struct_psp_gfx_cmd_boot_cfg(c.Struct):
  SIZE = 16
  timestamp: int
  sub_cmd: int
  boot_config: int
  boot_config_valid: int
struct_psp_gfx_cmd_boot_cfg.register_fields([('timestamp', ctypes.c_uint32, 0), ('sub_cmd', ctypes.c_uint32, 4), ('boot_config', ctypes.c_uint32, 8), ('boot_config_valid', ctypes.c_uint32, 12)])
@c.record
class struct_psp_gfx_cmd_sriov_spatial_part(c.Struct):
  SIZE = 16
  mode: int
  override_ips: int
  override_xcds_avail: int
  override_this_aid: int
struct_psp_gfx_cmd_sriov_spatial_part.register_fields([('mode', ctypes.c_uint32, 0), ('override_ips', ctypes.c_uint32, 4), ('override_xcds_avail', ctypes.c_uint32, 8), ('override_this_aid', ctypes.c_uint32, 12)])
@c.record
class union_psp_gfx_commands(c.Struct):
  SIZE = 784
  cmd_load_ta: struct_psp_gfx_cmd_load_ta
  cmd_unload_ta: struct_psp_gfx_cmd_unload_ta
  cmd_invoke_cmd: struct_psp_gfx_cmd_invoke_cmd
  cmd_setup_tmr: struct_psp_gfx_cmd_setup_tmr
  cmd_load_ip_fw: struct_psp_gfx_cmd_load_ip_fw
  cmd_save_restore_ip_fw: struct_psp_gfx_cmd_save_restore_ip_fw
  cmd_setup_reg_prog: struct_psp_gfx_cmd_reg_prog
  cmd_setup_vmr: struct_psp_gfx_cmd_setup_tmr
  cmd_load_toc: struct_psp_gfx_cmd_load_toc
  boot_cfg: struct_psp_gfx_cmd_boot_cfg
  cmd_spatial_part: struct_psp_gfx_cmd_sriov_spatial_part
union_psp_gfx_commands.register_fields([('cmd_load_ta', struct_psp_gfx_cmd_load_ta, 0), ('cmd_unload_ta', struct_psp_gfx_cmd_unload_ta, 0), ('cmd_invoke_cmd', struct_psp_gfx_cmd_invoke_cmd, 0), ('cmd_setup_tmr', struct_psp_gfx_cmd_setup_tmr, 0), ('cmd_load_ip_fw', struct_psp_gfx_cmd_load_ip_fw, 0), ('cmd_save_restore_ip_fw', struct_psp_gfx_cmd_save_restore_ip_fw, 0), ('cmd_setup_reg_prog', struct_psp_gfx_cmd_reg_prog, 0), ('cmd_setup_vmr', struct_psp_gfx_cmd_setup_tmr, 0), ('cmd_load_toc', struct_psp_gfx_cmd_load_toc, 0), ('boot_cfg', struct_psp_gfx_cmd_boot_cfg, 0), ('cmd_spatial_part', struct_psp_gfx_cmd_sriov_spatial_part, 0)])
@c.record
class struct_psp_gfx_uresp_reserved(c.Struct):
  SIZE = 32
  reserved: c.Array[ctypes.c_uint32, Literal[8]]
struct_psp_gfx_uresp_reserved.register_fields([('reserved', c.Array[ctypes.c_uint32, Literal[8]], 0)])
@c.record
class struct_psp_gfx_uresp_fwar_db_info(c.Struct):
  SIZE = 8
  fwar_db_addr_lo: int
  fwar_db_addr_hi: int
struct_psp_gfx_uresp_fwar_db_info.register_fields([('fwar_db_addr_lo', ctypes.c_uint32, 0), ('fwar_db_addr_hi', ctypes.c_uint32, 4)])
@c.record
class struct_psp_gfx_uresp_bootcfg(c.Struct):
  SIZE = 4
  boot_cfg: int
struct_psp_gfx_uresp_bootcfg.register_fields([('boot_cfg', ctypes.c_uint32, 0)])
@c.record
class union_psp_gfx_uresp(c.Struct):
  SIZE = 32
  reserved: struct_psp_gfx_uresp_reserved
  boot_cfg: struct_psp_gfx_uresp_bootcfg
  fwar_db_info: struct_psp_gfx_uresp_fwar_db_info
union_psp_gfx_uresp.register_fields([('reserved', struct_psp_gfx_uresp_reserved, 0), ('boot_cfg', struct_psp_gfx_uresp_bootcfg, 0), ('fwar_db_info', struct_psp_gfx_uresp_fwar_db_info, 0)])
@c.record
class struct_psp_gfx_resp(c.Struct):
  SIZE = 96
  status: int
  session_id: int
  fw_addr_lo: int
  fw_addr_hi: int
  tmr_size: int
  reserved: c.Array[ctypes.c_uint32, Literal[11]]
  uresp: union_psp_gfx_uresp
struct_psp_gfx_resp.register_fields([('status', ctypes.c_uint32, 0), ('session_id', ctypes.c_uint32, 4), ('fw_addr_lo', ctypes.c_uint32, 8), ('fw_addr_hi', ctypes.c_uint32, 12), ('tmr_size', ctypes.c_uint32, 16), ('reserved', c.Array[ctypes.c_uint32, Literal[11]], 20), ('uresp', union_psp_gfx_uresp, 64)])
@c.record
class struct_psp_gfx_cmd_resp(c.Struct):
  SIZE = 1024
  buf_size: int
  buf_version: int
  cmd_id: int
  resp_buf_addr_lo: int
  resp_buf_addr_hi: int
  resp_offset: int
  resp_buf_size: int
  cmd: union_psp_gfx_commands
  reserved_1: c.Array[ctypes.c_ubyte, Literal[52]]
  resp: struct_psp_gfx_resp
  reserved_2: c.Array[ctypes.c_ubyte, Literal[64]]
struct_psp_gfx_cmd_resp.register_fields([('buf_size', ctypes.c_uint32, 0), ('buf_version', ctypes.c_uint32, 4), ('cmd_id', ctypes.c_uint32, 8), ('resp_buf_addr_lo', ctypes.c_uint32, 12), ('resp_buf_addr_hi', ctypes.c_uint32, 16), ('resp_offset', ctypes.c_uint32, 20), ('resp_buf_size', ctypes.c_uint32, 24), ('cmd', union_psp_gfx_commands, 28), ('reserved_1', c.Array[ctypes.c_ubyte, Literal[52]], 812), ('resp', struct_psp_gfx_resp, 864), ('reserved_2', c.Array[ctypes.c_ubyte, Literal[64]], 960)])
@c.record
class struct_psp_gfx_rb_frame(c.Struct):
  SIZE = 64
  cmd_buf_addr_lo: int
  cmd_buf_addr_hi: int
  cmd_buf_size: int
  fence_addr_lo: int
  fence_addr_hi: int
  fence_value: int
  sid_lo: int
  sid_hi: int
  vmid: int
  frame_type: int
  reserved1: c.Array[ctypes.c_ubyte, Literal[2]]
  reserved2: c.Array[ctypes.c_uint32, Literal[7]]
struct_psp_gfx_rb_frame.register_fields([('cmd_buf_addr_lo', ctypes.c_uint32, 0), ('cmd_buf_addr_hi', ctypes.c_uint32, 4), ('cmd_buf_size', ctypes.c_uint32, 8), ('fence_addr_lo', ctypes.c_uint32, 12), ('fence_addr_hi', ctypes.c_uint32, 16), ('fence_value', ctypes.c_uint32, 20), ('sid_lo', ctypes.c_uint32, 24), ('sid_hi', ctypes.c_uint32, 28), ('vmid', ctypes.c_ubyte, 32), ('frame_type', ctypes.c_ubyte, 33), ('reserved1', c.Array[ctypes.c_ubyte, Literal[2]], 34), ('reserved2', c.Array[ctypes.c_uint32, Literal[7]], 36)])
enum_tee_error_code: dict[int, str] = {(TEE_SUCCESS:=0): 'TEE_SUCCESS', (TEE_ERROR_NOT_SUPPORTED:=4294901770): 'TEE_ERROR_NOT_SUPPORTED'}
enum_psp_shared_mem_size: dict[int, str] = {(PSP_ASD_SHARED_MEM_SIZE:=0): 'PSP_ASD_SHARED_MEM_SIZE', (PSP_XGMI_SHARED_MEM_SIZE:=16384): 'PSP_XGMI_SHARED_MEM_SIZE', (PSP_RAS_SHARED_MEM_SIZE:=16384): 'PSP_RAS_SHARED_MEM_SIZE', (PSP_HDCP_SHARED_MEM_SIZE:=16384): 'PSP_HDCP_SHARED_MEM_SIZE', (PSP_DTM_SHARED_MEM_SIZE:=16384): 'PSP_DTM_SHARED_MEM_SIZE', (PSP_RAP_SHARED_MEM_SIZE:=16384): 'PSP_RAP_SHARED_MEM_SIZE', (PSP_SECUREDISPLAY_SHARED_MEM_SIZE:=16384): 'PSP_SECUREDISPLAY_SHARED_MEM_SIZE'}
enum_ta_type_id: dict[int, str] = {(TA_TYPE_XGMI:=1): 'TA_TYPE_XGMI', (TA_TYPE_RAS:=2): 'TA_TYPE_RAS', (TA_TYPE_HDCP:=3): 'TA_TYPE_HDCP', (TA_TYPE_DTM:=4): 'TA_TYPE_DTM', (TA_TYPE_RAP:=5): 'TA_TYPE_RAP', (TA_TYPE_SECUREDISPLAY:=6): 'TA_TYPE_SECUREDISPLAY', (TA_TYPE_MAX_INDEX:=7): 'TA_TYPE_MAX_INDEX'}
class struct_psp_context(c.Struct): pass
class struct_psp_xgmi_node_info(c.Struct): pass
class struct_psp_xgmi_topology_info(c.Struct): pass
class struct_psp_bin_desc(c.Struct): pass
enum_psp_bootloader_cmd: dict[int, str] = {(PSP_BL__LOAD_SYSDRV:=65536): 'PSP_BL__LOAD_SYSDRV', (PSP_BL__LOAD_SOSDRV:=131072): 'PSP_BL__LOAD_SOSDRV', (PSP_BL__LOAD_KEY_DATABASE:=524288): 'PSP_BL__LOAD_KEY_DATABASE', (PSP_BL__LOAD_SOCDRV:=720896): 'PSP_BL__LOAD_SOCDRV', (PSP_BL__LOAD_DBGDRV:=786432): 'PSP_BL__LOAD_DBGDRV', (PSP_BL__LOAD_HADDRV:=786432): 'PSP_BL__LOAD_HADDRV', (PSP_BL__LOAD_INTFDRV:=851968): 'PSP_BL__LOAD_INTFDRV', (PSP_BL__LOAD_RASDRV:=917504): 'PSP_BL__LOAD_RASDRV', (PSP_BL__LOAD_IPKEYMGRDRV:=983040): 'PSP_BL__LOAD_IPKEYMGRDRV', (PSP_BL__DRAM_LONG_TRAIN:=1048576): 'PSP_BL__DRAM_LONG_TRAIN', (PSP_BL__DRAM_SHORT_TRAIN:=2097152): 'PSP_BL__DRAM_SHORT_TRAIN', (PSP_BL__LOAD_TOS_SPL_TABLE:=268435456): 'PSP_BL__LOAD_TOS_SPL_TABLE'}
enum_psp_ring_type: dict[int, str] = {(PSP_RING_TYPE__INVALID:=0): 'PSP_RING_TYPE__INVALID', (PSP_RING_TYPE__UM:=1): 'PSP_RING_TYPE__UM', (PSP_RING_TYPE__KM:=2): 'PSP_RING_TYPE__KM'}
enum_psp_reg_prog_id: dict[int, str] = {(PSP_REG_IH_RB_CNTL:=0): 'PSP_REG_IH_RB_CNTL', (PSP_REG_IH_RB_CNTL_RING1:=1): 'PSP_REG_IH_RB_CNTL_RING1', (PSP_REG_IH_RB_CNTL_RING2:=2): 'PSP_REG_IH_RB_CNTL_RING2', (PSP_REG_LAST:=3): 'PSP_REG_LAST'}
enum_psp_memory_training_init_flag: dict[int, str] = {(PSP_MEM_TRAIN_NOT_SUPPORT:=0): 'PSP_MEM_TRAIN_NOT_SUPPORT', (PSP_MEM_TRAIN_SUPPORT:=1): 'PSP_MEM_TRAIN_SUPPORT', (PSP_MEM_TRAIN_INIT_FAILED:=2): 'PSP_MEM_TRAIN_INIT_FAILED', (PSP_MEM_TRAIN_RESERVE_SUCCESS:=4): 'PSP_MEM_TRAIN_RESERVE_SUCCESS', (PSP_MEM_TRAIN_INIT_SUCCESS:=8): 'PSP_MEM_TRAIN_INIT_SUCCESS'}
enum_psp_memory_training_ops: dict[int, str] = {(PSP_MEM_TRAIN_SEND_LONG_MSG:=1): 'PSP_MEM_TRAIN_SEND_LONG_MSG', (PSP_MEM_TRAIN_SAVE:=2): 'PSP_MEM_TRAIN_SAVE', (PSP_MEM_TRAIN_RESTORE:=4): 'PSP_MEM_TRAIN_RESTORE', (PSP_MEM_TRAIN_SEND_SHORT_MSG:=8): 'PSP_MEM_TRAIN_SEND_SHORT_MSG', (PSP_MEM_TRAIN_COLD_BOOT:=1): 'PSP_MEM_TRAIN_COLD_BOOT', (PSP_MEM_TRAIN_RESUME:=8): 'PSP_MEM_TRAIN_RESUME'}
enum_psp_runtime_entry_type: dict[int, str] = {(PSP_RUNTIME_ENTRY_TYPE_INVALID:=0): 'PSP_RUNTIME_ENTRY_TYPE_INVALID', (PSP_RUNTIME_ENTRY_TYPE_TEST:=1): 'PSP_RUNTIME_ENTRY_TYPE_TEST', (PSP_RUNTIME_ENTRY_TYPE_MGPU_COMMON:=2): 'PSP_RUNTIME_ENTRY_TYPE_MGPU_COMMON', (PSP_RUNTIME_ENTRY_TYPE_MGPU_WAFL:=3): 'PSP_RUNTIME_ENTRY_TYPE_MGPU_WAFL', (PSP_RUNTIME_ENTRY_TYPE_MGPU_XGMI:=4): 'PSP_RUNTIME_ENTRY_TYPE_MGPU_XGMI', (PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG:=5): 'PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG', (PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS:=6): 'PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS'}
enum_psp_runtime_boot_cfg_feature: dict[int, str] = {(BOOT_CFG_FEATURE_GECC:=1): 'BOOT_CFG_FEATURE_GECC', (BOOT_CFG_FEATURE_TWO_STAGE_DRAM_TRAINING:=2): 'BOOT_CFG_FEATURE_TWO_STAGE_DRAM_TRAINING'}
enum_psp_runtime_scpm_authentication: dict[int, str] = {(SCPM_DISABLE:=0): 'SCPM_DISABLE', (SCPM_ENABLE:=1): 'SCPM_ENABLE', (SCPM_ENABLE_WITH_SCPM_ERR:=2): 'SCPM_ENABLE_WITH_SCPM_ERR'}
class struct_amdgpu_device(c.Struct): pass
enum_amdgpu_interrupt_state: dict[int, str] = {(AMDGPU_IRQ_STATE_DISABLE:=0): 'AMDGPU_IRQ_STATE_DISABLE', (AMDGPU_IRQ_STATE_ENABLE:=1): 'AMDGPU_IRQ_STATE_ENABLE'}
@c.record
class struct_amdgpu_iv_entry(c.Struct):
  SIZE = 72
  client_id: int
  src_id: int
  ring_id: int
  vmid: int
  vmid_src: int
  timestamp: int
  timestamp_src: int
  pasid: int
  node_id: int
  src_data: c.Array[ctypes.c_uint32, Literal[4]]
  iv_entry: c.POINTER[ctypes.c_uint32]
struct_amdgpu_iv_entry.register_fields([('client_id', ctypes.c_uint32, 0), ('src_id', ctypes.c_uint32, 4), ('ring_id', ctypes.c_uint32, 8), ('vmid', ctypes.c_uint32, 12), ('vmid_src', ctypes.c_uint32, 16), ('timestamp', uint64_t, 24), ('timestamp_src', ctypes.c_uint32, 32), ('pasid', ctypes.c_uint32, 36), ('node_id', ctypes.c_uint32, 40), ('src_data', c.Array[ctypes.c_uint32, Literal[4]], 44), ('iv_entry', c.POINTER[uint32_t], 64)])
enum_interrupt_node_id_per_aid: dict[int, str] = {(AID0_NODEID:=0): 'AID0_NODEID', (XCD0_NODEID:=1): 'XCD0_NODEID', (XCD1_NODEID:=2): 'XCD1_NODEID', (AID1_NODEID:=4): 'AID1_NODEID', (XCD2_NODEID:=5): 'XCD2_NODEID', (XCD3_NODEID:=6): 'XCD3_NODEID', (AID2_NODEID:=8): 'AID2_NODEID', (XCD4_NODEID:=9): 'XCD4_NODEID', (XCD5_NODEID:=10): 'XCD5_NODEID', (AID3_NODEID:=12): 'AID3_NODEID', (XCD6_NODEID:=13): 'XCD6_NODEID', (XCD7_NODEID:=14): 'XCD7_NODEID', (NODEID_MAX:=15): 'NODEID_MAX'}
enum_AMDGPU_DOORBELL_ASSIGNMENT: dict[int, str] = {(AMDGPU_DOORBELL_KIQ:=0): 'AMDGPU_DOORBELL_KIQ', (AMDGPU_DOORBELL_HIQ:=1): 'AMDGPU_DOORBELL_HIQ', (AMDGPU_DOORBELL_DIQ:=2): 'AMDGPU_DOORBELL_DIQ', (AMDGPU_DOORBELL_MEC_RING0:=16): 'AMDGPU_DOORBELL_MEC_RING0', (AMDGPU_DOORBELL_MEC_RING1:=17): 'AMDGPU_DOORBELL_MEC_RING1', (AMDGPU_DOORBELL_MEC_RING2:=18): 'AMDGPU_DOORBELL_MEC_RING2', (AMDGPU_DOORBELL_MEC_RING3:=19): 'AMDGPU_DOORBELL_MEC_RING3', (AMDGPU_DOORBELL_MEC_RING4:=20): 'AMDGPU_DOORBELL_MEC_RING4', (AMDGPU_DOORBELL_MEC_RING5:=21): 'AMDGPU_DOORBELL_MEC_RING5', (AMDGPU_DOORBELL_MEC_RING6:=22): 'AMDGPU_DOORBELL_MEC_RING6', (AMDGPU_DOORBELL_MEC_RING7:=23): 'AMDGPU_DOORBELL_MEC_RING7', (AMDGPU_DOORBELL_GFX_RING0:=32): 'AMDGPU_DOORBELL_GFX_RING0', (AMDGPU_DOORBELL_sDMA_ENGINE0:=480): 'AMDGPU_DOORBELL_sDMA_ENGINE0', (AMDGPU_DOORBELL_sDMA_ENGINE1:=481): 'AMDGPU_DOORBELL_sDMA_ENGINE1', (AMDGPU_DOORBELL_IH:=488): 'AMDGPU_DOORBELL_IH', (AMDGPU_DOORBELL_MAX_ASSIGNMENT:=1023): 'AMDGPU_DOORBELL_MAX_ASSIGNMENT', (AMDGPU_DOORBELL_INVALID:=65535): 'AMDGPU_DOORBELL_INVALID'}
enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT: dict[int, str] = {(AMDGPU_VEGA20_DOORBELL_KIQ:=0): 'AMDGPU_VEGA20_DOORBELL_KIQ', (AMDGPU_VEGA20_DOORBELL_HIQ:=1): 'AMDGPU_VEGA20_DOORBELL_HIQ', (AMDGPU_VEGA20_DOORBELL_DIQ:=2): 'AMDGPU_VEGA20_DOORBELL_DIQ', (AMDGPU_VEGA20_DOORBELL_MEC_RING0:=3): 'AMDGPU_VEGA20_DOORBELL_MEC_RING0', (AMDGPU_VEGA20_DOORBELL_MEC_RING1:=4): 'AMDGPU_VEGA20_DOORBELL_MEC_RING1', (AMDGPU_VEGA20_DOORBELL_MEC_RING2:=5): 'AMDGPU_VEGA20_DOORBELL_MEC_RING2', (AMDGPU_VEGA20_DOORBELL_MEC_RING3:=6): 'AMDGPU_VEGA20_DOORBELL_MEC_RING3', (AMDGPU_VEGA20_DOORBELL_MEC_RING4:=7): 'AMDGPU_VEGA20_DOORBELL_MEC_RING4', (AMDGPU_VEGA20_DOORBELL_MEC_RING5:=8): 'AMDGPU_VEGA20_DOORBELL_MEC_RING5', (AMDGPU_VEGA20_DOORBELL_MEC_RING6:=9): 'AMDGPU_VEGA20_DOORBELL_MEC_RING6', (AMDGPU_VEGA20_DOORBELL_MEC_RING7:=10): 'AMDGPU_VEGA20_DOORBELL_MEC_RING7', (AMDGPU_VEGA20_DOORBELL_USERQUEUE_START:=11): 'AMDGPU_VEGA20_DOORBELL_USERQUEUE_START', (AMDGPU_VEGA20_DOORBELL_USERQUEUE_END:=138): 'AMDGPU_VEGA20_DOORBELL_USERQUEUE_END', (AMDGPU_VEGA20_DOORBELL_GFX_RING0:=139): 'AMDGPU_VEGA20_DOORBELL_GFX_RING0', (AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE0:=256): 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE0', (AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE1:=266): 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE1', (AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE2:=276): 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE2', (AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE3:=286): 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE3', (AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE4:=296): 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE4', (AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE5:=306): 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE5', (AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE6:=316): 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE6', (AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE7:=326): 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE7', (AMDGPU_VEGA20_DOORBELL_IH:=376): 'AMDGPU_VEGA20_DOORBELL_IH', (AMDGPU_VEGA20_DOORBELL64_VCN0_1:=392): 'AMDGPU_VEGA20_DOORBELL64_VCN0_1', (AMDGPU_VEGA20_DOORBELL64_VCN2_3:=393): 'AMDGPU_VEGA20_DOORBELL64_VCN2_3', (AMDGPU_VEGA20_DOORBELL64_VCN4_5:=394): 'AMDGPU_VEGA20_DOORBELL64_VCN4_5', (AMDGPU_VEGA20_DOORBELL64_VCN6_7:=395): 'AMDGPU_VEGA20_DOORBELL64_VCN6_7', (AMDGPU_VEGA20_DOORBELL64_VCN8_9:=396): 'AMDGPU_VEGA20_DOORBELL64_VCN8_9', (AMDGPU_VEGA20_DOORBELL64_VCNa_b:=397): 'AMDGPU_VEGA20_DOORBELL64_VCNa_b', (AMDGPU_VEGA20_DOORBELL64_VCNc_d:=398): 'AMDGPU_VEGA20_DOORBELL64_VCNc_d', (AMDGPU_VEGA20_DOORBELL64_VCNe_f:=399): 'AMDGPU_VEGA20_DOORBELL64_VCNe_f', (AMDGPU_VEGA20_DOORBELL64_UVD_RING0_1:=392): 'AMDGPU_VEGA20_DOORBELL64_UVD_RING0_1', (AMDGPU_VEGA20_DOORBELL64_UVD_RING2_3:=393): 'AMDGPU_VEGA20_DOORBELL64_UVD_RING2_3', (AMDGPU_VEGA20_DOORBELL64_UVD_RING4_5:=394): 'AMDGPU_VEGA20_DOORBELL64_UVD_RING4_5', (AMDGPU_VEGA20_DOORBELL64_UVD_RING6_7:=395): 'AMDGPU_VEGA20_DOORBELL64_UVD_RING6_7', (AMDGPU_VEGA20_DOORBELL64_VCE_RING0_1:=396): 'AMDGPU_VEGA20_DOORBELL64_VCE_RING0_1', (AMDGPU_VEGA20_DOORBELL64_VCE_RING2_3:=397): 'AMDGPU_VEGA20_DOORBELL64_VCE_RING2_3', (AMDGPU_VEGA20_DOORBELL64_VCE_RING4_5:=398): 'AMDGPU_VEGA20_DOORBELL64_VCE_RING4_5', (AMDGPU_VEGA20_DOORBELL64_VCE_RING6_7:=399): 'AMDGPU_VEGA20_DOORBELL64_VCE_RING6_7', (AMDGPU_VEGA20_DOORBELL64_FIRST_NON_CP:=256): 'AMDGPU_VEGA20_DOORBELL64_FIRST_NON_CP', (AMDGPU_VEGA20_DOORBELL64_LAST_NON_CP:=399): 'AMDGPU_VEGA20_DOORBELL64_LAST_NON_CP', (AMDGPU_VEGA20_DOORBELL_XCC1_KIQ_START:=400): 'AMDGPU_VEGA20_DOORBELL_XCC1_KIQ_START', (AMDGPU_VEGA20_DOORBELL_XCC1_MEC_RING0_START:=407): 'AMDGPU_VEGA20_DOORBELL_XCC1_MEC_RING0_START', (AMDGPU_VEGA20_DOORBELL_AID1_sDMA_START:=464): 'AMDGPU_VEGA20_DOORBELL_AID1_sDMA_START', (AMDGPU_VEGA20_DOORBELL_MAX_ASSIGNMENT:=503): 'AMDGPU_VEGA20_DOORBELL_MAX_ASSIGNMENT', (AMDGPU_VEGA20_DOORBELL_INVALID:=65535): 'AMDGPU_VEGA20_DOORBELL_INVALID'}
enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT: dict[int, str] = {(AMDGPU_NAVI10_DOORBELL_KIQ:=0): 'AMDGPU_NAVI10_DOORBELL_KIQ', (AMDGPU_NAVI10_DOORBELL_HIQ:=1): 'AMDGPU_NAVI10_DOORBELL_HIQ', (AMDGPU_NAVI10_DOORBELL_DIQ:=2): 'AMDGPU_NAVI10_DOORBELL_DIQ', (AMDGPU_NAVI10_DOORBELL_MEC_RING0:=3): 'AMDGPU_NAVI10_DOORBELL_MEC_RING0', (AMDGPU_NAVI10_DOORBELL_MEC_RING1:=4): 'AMDGPU_NAVI10_DOORBELL_MEC_RING1', (AMDGPU_NAVI10_DOORBELL_MEC_RING2:=5): 'AMDGPU_NAVI10_DOORBELL_MEC_RING2', (AMDGPU_NAVI10_DOORBELL_MEC_RING3:=6): 'AMDGPU_NAVI10_DOORBELL_MEC_RING3', (AMDGPU_NAVI10_DOORBELL_MEC_RING4:=7): 'AMDGPU_NAVI10_DOORBELL_MEC_RING4', (AMDGPU_NAVI10_DOORBELL_MEC_RING5:=8): 'AMDGPU_NAVI10_DOORBELL_MEC_RING5', (AMDGPU_NAVI10_DOORBELL_MEC_RING6:=9): 'AMDGPU_NAVI10_DOORBELL_MEC_RING6', (AMDGPU_NAVI10_DOORBELL_MEC_RING7:=10): 'AMDGPU_NAVI10_DOORBELL_MEC_RING7', (AMDGPU_NAVI10_DOORBELL_MES_RING0:=11): 'AMDGPU_NAVI10_DOORBELL_MES_RING0', (AMDGPU_NAVI10_DOORBELL_MES_RING1:=12): 'AMDGPU_NAVI10_DOORBELL_MES_RING1', (AMDGPU_NAVI10_DOORBELL_USERQUEUE_START:=13): 'AMDGPU_NAVI10_DOORBELL_USERQUEUE_START', (AMDGPU_NAVI10_DOORBELL_USERQUEUE_END:=138): 'AMDGPU_NAVI10_DOORBELL_USERQUEUE_END', (AMDGPU_NAVI10_DOORBELL_GFX_RING0:=139): 'AMDGPU_NAVI10_DOORBELL_GFX_RING0', (AMDGPU_NAVI10_DOORBELL_GFX_RING1:=140): 'AMDGPU_NAVI10_DOORBELL_GFX_RING1', (AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_START:=141): 'AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_START', (AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_END:=255): 'AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_END', (AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0:=256): 'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0', (AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE1:=266): 'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE1', (AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE2:=276): 'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE2', (AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE3:=286): 'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE3', (AMDGPU_NAVI10_DOORBELL_IH:=376): 'AMDGPU_NAVI10_DOORBELL_IH', (AMDGPU_NAVI10_DOORBELL64_VCN0_1:=392): 'AMDGPU_NAVI10_DOORBELL64_VCN0_1', (AMDGPU_NAVI10_DOORBELL64_VCN2_3:=393): 'AMDGPU_NAVI10_DOORBELL64_VCN2_3', (AMDGPU_NAVI10_DOORBELL64_VCN4_5:=394): 'AMDGPU_NAVI10_DOORBELL64_VCN4_5', (AMDGPU_NAVI10_DOORBELL64_VCN6_7:=395): 'AMDGPU_NAVI10_DOORBELL64_VCN6_7', (AMDGPU_NAVI10_DOORBELL64_VCN8_9:=396): 'AMDGPU_NAVI10_DOORBELL64_VCN8_9', (AMDGPU_NAVI10_DOORBELL64_VCNa_b:=397): 'AMDGPU_NAVI10_DOORBELL64_VCNa_b', (AMDGPU_NAVI10_DOORBELL64_VCNc_d:=398): 'AMDGPU_NAVI10_DOORBELL64_VCNc_d', (AMDGPU_NAVI10_DOORBELL64_VCNe_f:=399): 'AMDGPU_NAVI10_DOORBELL64_VCNe_f', (AMDGPU_NAVI10_DOORBELL64_VPE:=400): 'AMDGPU_NAVI10_DOORBELL64_VPE', (AMDGPU_NAVI10_DOORBELL64_FIRST_NON_CP:=256): 'AMDGPU_NAVI10_DOORBELL64_FIRST_NON_CP', (AMDGPU_NAVI10_DOORBELL64_LAST_NON_CP:=400): 'AMDGPU_NAVI10_DOORBELL64_LAST_NON_CP', (AMDGPU_NAVI10_DOORBELL_MAX_ASSIGNMENT:=400): 'AMDGPU_NAVI10_DOORBELL_MAX_ASSIGNMENT', (AMDGPU_NAVI10_DOORBELL_INVALID:=65535): 'AMDGPU_NAVI10_DOORBELL_INVALID'}
enum_AMDGPU_DOORBELL64_ASSIGNMENT: dict[int, str] = {(AMDGPU_DOORBELL64_KIQ:=0): 'AMDGPU_DOORBELL64_KIQ', (AMDGPU_DOORBELL64_HIQ:=1): 'AMDGPU_DOORBELL64_HIQ', (AMDGPU_DOORBELL64_DIQ:=2): 'AMDGPU_DOORBELL64_DIQ', (AMDGPU_DOORBELL64_MEC_RING0:=3): 'AMDGPU_DOORBELL64_MEC_RING0', (AMDGPU_DOORBELL64_MEC_RING1:=4): 'AMDGPU_DOORBELL64_MEC_RING1', (AMDGPU_DOORBELL64_MEC_RING2:=5): 'AMDGPU_DOORBELL64_MEC_RING2', (AMDGPU_DOORBELL64_MEC_RING3:=6): 'AMDGPU_DOORBELL64_MEC_RING3', (AMDGPU_DOORBELL64_MEC_RING4:=7): 'AMDGPU_DOORBELL64_MEC_RING4', (AMDGPU_DOORBELL64_MEC_RING5:=8): 'AMDGPU_DOORBELL64_MEC_RING5', (AMDGPU_DOORBELL64_MEC_RING6:=9): 'AMDGPU_DOORBELL64_MEC_RING6', (AMDGPU_DOORBELL64_MEC_RING7:=10): 'AMDGPU_DOORBELL64_MEC_RING7', (AMDGPU_DOORBELL64_USERQUEUE_START:=11): 'AMDGPU_DOORBELL64_USERQUEUE_START', (AMDGPU_DOORBELL64_USERQUEUE_END:=138): 'AMDGPU_DOORBELL64_USERQUEUE_END', (AMDGPU_DOORBELL64_GFX_RING0:=139): 'AMDGPU_DOORBELL64_GFX_RING0', (AMDGPU_DOORBELL64_sDMA_ENGINE0:=240): 'AMDGPU_DOORBELL64_sDMA_ENGINE0', (AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE0:=241): 'AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE0', (AMDGPU_DOORBELL64_sDMA_ENGINE1:=242): 'AMDGPU_DOORBELL64_sDMA_ENGINE1', (AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE1:=243): 'AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE1', (AMDGPU_DOORBELL64_IH:=244): 'AMDGPU_DOORBELL64_IH', (AMDGPU_DOORBELL64_IH_RING1:=245): 'AMDGPU_DOORBELL64_IH_RING1', (AMDGPU_DOORBELL64_IH_RING2:=246): 'AMDGPU_DOORBELL64_IH_RING2', (AMDGPU_DOORBELL64_VCN0_1:=248): 'AMDGPU_DOORBELL64_VCN0_1', (AMDGPU_DOORBELL64_VCN2_3:=249): 'AMDGPU_DOORBELL64_VCN2_3', (AMDGPU_DOORBELL64_VCN4_5:=250): 'AMDGPU_DOORBELL64_VCN4_5', (AMDGPU_DOORBELL64_VCN6_7:=251): 'AMDGPU_DOORBELL64_VCN6_7', (AMDGPU_DOORBELL64_UVD_RING0_1:=248): 'AMDGPU_DOORBELL64_UVD_RING0_1', (AMDGPU_DOORBELL64_UVD_RING2_3:=249): 'AMDGPU_DOORBELL64_UVD_RING2_3', (AMDGPU_DOORBELL64_UVD_RING4_5:=250): 'AMDGPU_DOORBELL64_UVD_RING4_5', (AMDGPU_DOORBELL64_UVD_RING6_7:=251): 'AMDGPU_DOORBELL64_UVD_RING6_7', (AMDGPU_DOORBELL64_VCE_RING0_1:=252): 'AMDGPU_DOORBELL64_VCE_RING0_1', (AMDGPU_DOORBELL64_VCE_RING2_3:=253): 'AMDGPU_DOORBELL64_VCE_RING2_3', (AMDGPU_DOORBELL64_VCE_RING4_5:=254): 'AMDGPU_DOORBELL64_VCE_RING4_5', (AMDGPU_DOORBELL64_VCE_RING6_7:=255): 'AMDGPU_DOORBELL64_VCE_RING6_7', (AMDGPU_DOORBELL64_FIRST_NON_CP:=240): 'AMDGPU_DOORBELL64_FIRST_NON_CP', (AMDGPU_DOORBELL64_LAST_NON_CP:=255): 'AMDGPU_DOORBELL64_LAST_NON_CP', (AMDGPU_DOORBELL64_MAX_ASSIGNMENT:=255): 'AMDGPU_DOORBELL64_MAX_ASSIGNMENT', (AMDGPU_DOORBELL64_INVALID:=65535): 'AMDGPU_DOORBELL64_INVALID'}
enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1: dict[int, str] = {(AMDGPU_DOORBELL_LAYOUT1_KIQ_START:=0): 'AMDGPU_DOORBELL_LAYOUT1_KIQ_START', (AMDGPU_DOORBELL_LAYOUT1_HIQ:=1): 'AMDGPU_DOORBELL_LAYOUT1_HIQ', (AMDGPU_DOORBELL_LAYOUT1_DIQ:=2): 'AMDGPU_DOORBELL_LAYOUT1_DIQ', (AMDGPU_DOORBELL_LAYOUT1_MEC_RING_START:=8): 'AMDGPU_DOORBELL_LAYOUT1_MEC_RING_START', (AMDGPU_DOORBELL_LAYOUT1_MEC_RING_END:=15): 'AMDGPU_DOORBELL_LAYOUT1_MEC_RING_END', (AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_START:=16): 'AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_START', (AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_END:=31): 'AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_END', (AMDGPU_DOORBELL_LAYOUT1_XCC_RANGE:=32): 'AMDGPU_DOORBELL_LAYOUT1_XCC_RANGE', (AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_START:=256): 'AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_START', (AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_END:=415): 'AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_END', (AMDGPU_DOORBELL_LAYOUT1_IH:=416): 'AMDGPU_DOORBELL_LAYOUT1_IH', (AMDGPU_DOORBELL_LAYOUT1_VCN_START:=432): 'AMDGPU_DOORBELL_LAYOUT1_VCN_START', (AMDGPU_DOORBELL_LAYOUT1_VCN_END:=488): 'AMDGPU_DOORBELL_LAYOUT1_VCN_END', (AMDGPU_DOORBELL_LAYOUT1_FIRST_NON_CP:=256): 'AMDGPU_DOORBELL_LAYOUT1_FIRST_NON_CP', (AMDGPU_DOORBELL_LAYOUT1_LAST_NON_CP:=488): 'AMDGPU_DOORBELL_LAYOUT1_LAST_NON_CP', (AMDGPU_DOORBELL_LAYOUT1_MAX_ASSIGNMENT:=488): 'AMDGPU_DOORBELL_LAYOUT1_MAX_ASSIGNMENT', (AMDGPU_DOORBELL_LAYOUT1_INVALID:=65535): 'AMDGPU_DOORBELL_LAYOUT1_INVALID'}
@c.record
class struct_v9_sdma_mqd(c.Struct):
  SIZE = 512
  sdmax_rlcx_rb_cntl: int
  sdmax_rlcx_rb_base: int
  sdmax_rlcx_rb_base_hi: int
  sdmax_rlcx_rb_rptr: int
  sdmax_rlcx_rb_rptr_hi: int
  sdmax_rlcx_rb_wptr: int
  sdmax_rlcx_rb_wptr_hi: int
  sdmax_rlcx_rb_wptr_poll_cntl: int
  sdmax_rlcx_rb_rptr_addr_hi: int
  sdmax_rlcx_rb_rptr_addr_lo: int
  sdmax_rlcx_ib_cntl: int
  sdmax_rlcx_ib_rptr: int
  sdmax_rlcx_ib_offset: int
  sdmax_rlcx_ib_base_lo: int
  sdmax_rlcx_ib_base_hi: int
  sdmax_rlcx_ib_size: int
  sdmax_rlcx_skip_cntl: int
  sdmax_rlcx_context_status: int
  sdmax_rlcx_doorbell: int
  sdmax_rlcx_status: int
  sdmax_rlcx_doorbell_log: int
  sdmax_rlcx_watermark: int
  sdmax_rlcx_doorbell_offset: int
  sdmax_rlcx_csa_addr_lo: int
  sdmax_rlcx_csa_addr_hi: int
  sdmax_rlcx_ib_sub_remain: int
  sdmax_rlcx_preempt: int
  sdmax_rlcx_dummy_reg: int
  sdmax_rlcx_rb_wptr_poll_addr_hi: int
  sdmax_rlcx_rb_wptr_poll_addr_lo: int
  sdmax_rlcx_rb_aql_cntl: int
  sdmax_rlcx_minor_ptr_update: int
  sdmax_rlcx_midcmd_data0: int
  sdmax_rlcx_midcmd_data1: int
  sdmax_rlcx_midcmd_data2: int
  sdmax_rlcx_midcmd_data3: int
  sdmax_rlcx_midcmd_data4: int
  sdmax_rlcx_midcmd_data5: int
  sdmax_rlcx_midcmd_data6: int
  sdmax_rlcx_midcmd_data7: int
  sdmax_rlcx_midcmd_data8: int
  sdmax_rlcx_midcmd_cntl: int
  reserved_42: int
  reserved_43: int
  reserved_44: int
  reserved_45: int
  reserved_46: int
  reserved_47: int
  reserved_48: int
  reserved_49: int
  reserved_50: int
  reserved_51: int
  reserved_52: int
  reserved_53: int
  reserved_54: int
  reserved_55: int
  reserved_56: int
  reserved_57: int
  reserved_58: int
  reserved_59: int
  reserved_60: int
  reserved_61: int
  reserved_62: int
  reserved_63: int
  reserved_64: int
  reserved_65: int
  reserved_66: int
  reserved_67: int
  reserved_68: int
  reserved_69: int
  reserved_70: int
  reserved_71: int
  reserved_72: int
  reserved_73: int
  reserved_74: int
  reserved_75: int
  reserved_76: int
  reserved_77: int
  reserved_78: int
  reserved_79: int
  reserved_80: int
  reserved_81: int
  reserved_82: int
  reserved_83: int
  reserved_84: int
  reserved_85: int
  reserved_86: int
  reserved_87: int
  reserved_88: int
  reserved_89: int
  reserved_90: int
  reserved_91: int
  reserved_92: int
  reserved_93: int
  reserved_94: int
  reserved_95: int
  reserved_96: int
  reserved_97: int
  reserved_98: int
  reserved_99: int
  reserved_100: int
  reserved_101: int
  reserved_102: int
  reserved_103: int
  reserved_104: int
  reserved_105: int
  reserved_106: int
  reserved_107: int
  reserved_108: int
  reserved_109: int
  reserved_110: int
  reserved_111: int
  reserved_112: int
  reserved_113: int
  reserved_114: int
  reserved_115: int
  reserved_116: int
  reserved_117: int
  reserved_118: int
  reserved_119: int
  reserved_120: int
  reserved_121: int
  reserved_122: int
  reserved_123: int
  reserved_124: int
  reserved_125: int
  sdma_engine_id: int
  sdma_queue_id: int
struct_v9_sdma_mqd.register_fields([('sdmax_rlcx_rb_cntl', uint32_t, 0), ('sdmax_rlcx_rb_base', uint32_t, 4), ('sdmax_rlcx_rb_base_hi', uint32_t, 8), ('sdmax_rlcx_rb_rptr', uint32_t, 12), ('sdmax_rlcx_rb_rptr_hi', uint32_t, 16), ('sdmax_rlcx_rb_wptr', uint32_t, 20), ('sdmax_rlcx_rb_wptr_hi', uint32_t, 24), ('sdmax_rlcx_rb_wptr_poll_cntl', uint32_t, 28), ('sdmax_rlcx_rb_rptr_addr_hi', uint32_t, 32), ('sdmax_rlcx_rb_rptr_addr_lo', uint32_t, 36), ('sdmax_rlcx_ib_cntl', uint32_t, 40), ('sdmax_rlcx_ib_rptr', uint32_t, 44), ('sdmax_rlcx_ib_offset', uint32_t, 48), ('sdmax_rlcx_ib_base_lo', uint32_t, 52), ('sdmax_rlcx_ib_base_hi', uint32_t, 56), ('sdmax_rlcx_ib_size', uint32_t, 60), ('sdmax_rlcx_skip_cntl', uint32_t, 64), ('sdmax_rlcx_context_status', uint32_t, 68), ('sdmax_rlcx_doorbell', uint32_t, 72), ('sdmax_rlcx_status', uint32_t, 76), ('sdmax_rlcx_doorbell_log', uint32_t, 80), ('sdmax_rlcx_watermark', uint32_t, 84), ('sdmax_rlcx_doorbell_offset', uint32_t, 88), ('sdmax_rlcx_csa_addr_lo', uint32_t, 92), ('sdmax_rlcx_csa_addr_hi', uint32_t, 96), ('sdmax_rlcx_ib_sub_remain', uint32_t, 100), ('sdmax_rlcx_preempt', uint32_t, 104), ('sdmax_rlcx_dummy_reg', uint32_t, 108), ('sdmax_rlcx_rb_wptr_poll_addr_hi', uint32_t, 112), ('sdmax_rlcx_rb_wptr_poll_addr_lo', uint32_t, 116), ('sdmax_rlcx_rb_aql_cntl', uint32_t, 120), ('sdmax_rlcx_minor_ptr_update', uint32_t, 124), ('sdmax_rlcx_midcmd_data0', uint32_t, 128), ('sdmax_rlcx_midcmd_data1', uint32_t, 132), ('sdmax_rlcx_midcmd_data2', uint32_t, 136), ('sdmax_rlcx_midcmd_data3', uint32_t, 140), ('sdmax_rlcx_midcmd_data4', uint32_t, 144), ('sdmax_rlcx_midcmd_data5', uint32_t, 148), ('sdmax_rlcx_midcmd_data6', uint32_t, 152), ('sdmax_rlcx_midcmd_data7', uint32_t, 156), ('sdmax_rlcx_midcmd_data8', uint32_t, 160), ('sdmax_rlcx_midcmd_cntl', uint32_t, 164), ('reserved_42', uint32_t, 168), ('reserved_43', uint32_t, 172), ('reserved_44', uint32_t, 176), ('reserved_45', uint32_t, 180), ('reserved_46', uint32_t, 184), ('reserved_47', uint32_t, 188), ('reserved_48', uint32_t, 192), ('reserved_49', uint32_t, 196), ('reserved_50', uint32_t, 200), ('reserved_51', uint32_t, 204), ('reserved_52', uint32_t, 208), ('reserved_53', uint32_t, 212), ('reserved_54', uint32_t, 216), ('reserved_55', uint32_t, 220), ('reserved_56', uint32_t, 224), ('reserved_57', uint32_t, 228), ('reserved_58', uint32_t, 232), ('reserved_59', uint32_t, 236), ('reserved_60', uint32_t, 240), ('reserved_61', uint32_t, 244), ('reserved_62', uint32_t, 248), ('reserved_63', uint32_t, 252), ('reserved_64', uint32_t, 256), ('reserved_65', uint32_t, 260), ('reserved_66', uint32_t, 264), ('reserved_67', uint32_t, 268), ('reserved_68', uint32_t, 272), ('reserved_69', uint32_t, 276), ('reserved_70', uint32_t, 280), ('reserved_71', uint32_t, 284), ('reserved_72', uint32_t, 288), ('reserved_73', uint32_t, 292), ('reserved_74', uint32_t, 296), ('reserved_75', uint32_t, 300), ('reserved_76', uint32_t, 304), ('reserved_77', uint32_t, 308), ('reserved_78', uint32_t, 312), ('reserved_79', uint32_t, 316), ('reserved_80', uint32_t, 320), ('reserved_81', uint32_t, 324), ('reserved_82', uint32_t, 328), ('reserved_83', uint32_t, 332), ('reserved_84', uint32_t, 336), ('reserved_85', uint32_t, 340), ('reserved_86', uint32_t, 344), ('reserved_87', uint32_t, 348), ('reserved_88', uint32_t, 352), ('reserved_89', uint32_t, 356), ('reserved_90', uint32_t, 360), ('reserved_91', uint32_t, 364), ('reserved_92', uint32_t, 368), ('reserved_93', uint32_t, 372), ('reserved_94', uint32_t, 376), ('reserved_95', uint32_t, 380), ('reserved_96', uint32_t, 384), ('reserved_97', uint32_t, 388), ('reserved_98', uint32_t, 392), ('reserved_99', uint32_t, 396), ('reserved_100', uint32_t, 400), ('reserved_101', uint32_t, 404), ('reserved_102', uint32_t, 408), ('reserved_103', uint32_t, 412), ('reserved_104', uint32_t, 416), ('reserved_105', uint32_t, 420), ('reserved_106', uint32_t, 424), ('reserved_107', uint32_t, 428), ('reserved_108', uint32_t, 432), ('reserved_109', uint32_t, 436), ('reserved_110', uint32_t, 440), ('reserved_111', uint32_t, 444), ('reserved_112', uint32_t, 448), ('reserved_113', uint32_t, 452), ('reserved_114', uint32_t, 456), ('reserved_115', uint32_t, 460), ('reserved_116', uint32_t, 464), ('reserved_117', uint32_t, 468), ('reserved_118', uint32_t, 472), ('reserved_119', uint32_t, 476), ('reserved_120', uint32_t, 480), ('reserved_121', uint32_t, 484), ('reserved_122', uint32_t, 488), ('reserved_123', uint32_t, 492), ('reserved_124', uint32_t, 496), ('reserved_125', uint32_t, 500), ('sdma_engine_id', uint32_t, 504), ('sdma_queue_id', uint32_t, 508)])
@c.record
class struct_v9_mqd(c.Struct):
  SIZE = 2048
  header: int
  compute_dispatch_initiator: int
  compute_dim_x: int
  compute_dim_y: int
  compute_dim_z: int
  compute_start_x: int
  compute_start_y: int
  compute_start_z: int
  compute_num_thread_x: int
  compute_num_thread_y: int
  compute_num_thread_z: int
  compute_pipelinestat_enable: int
  compute_perfcount_enable: int
  compute_pgm_lo: int
  compute_pgm_hi: int
  compute_tba_lo: int
  compute_tba_hi: int
  compute_tma_lo: int
  compute_tma_hi: int
  compute_pgm_rsrc1: int
  compute_pgm_rsrc2: int
  compute_vmid: int
  compute_resource_limits: int
  compute_static_thread_mgmt_se0: int
  compute_static_thread_mgmt_se1: int
  compute_tmpring_size: int
  compute_static_thread_mgmt_se2: int
  compute_static_thread_mgmt_se3: int
  compute_restart_x: int
  compute_restart_y: int
  compute_restart_z: int
  compute_thread_trace_enable: int
  compute_misc_reserved: int
  compute_dispatch_id: int
  compute_threadgroup_id: int
  compute_relaunch: int
  compute_wave_restore_addr_lo: int
  compute_wave_restore_addr_hi: int
  compute_wave_restore_control: int
  compute_static_thread_mgmt_se4: int
  compute_static_thread_mgmt_se5: int
  compute_static_thread_mgmt_se6: int
  compute_static_thread_mgmt_se7: int
  compute_current_logic_xcc_id: int
  compute_restart_cg_tg_id: int
  compute_tg_chunk_size: int
  compute_restore_tg_chunk_size: int
  reserved_43: int
  reserved_44: int
  reserved_45: int
  reserved_46: int
  reserved_47: int
  reserved_48: int
  reserved_49: int
  reserved_50: int
  reserved_51: int
  reserved_52: int
  reserved_53: int
  reserved_54: int
  reserved_55: int
  reserved_56: int
  reserved_57: int
  reserved_58: int
  reserved_59: int
  reserved_60: int
  reserved_61: int
  reserved_62: int
  reserved_63: int
  reserved_64: int
  compute_user_data_0: int
  compute_user_data_1: int
  compute_user_data_2: int
  compute_user_data_3: int
  compute_user_data_4: int
  compute_user_data_5: int
  compute_user_data_6: int
  compute_user_data_7: int
  compute_user_data_8: int
  compute_user_data_9: int
  compute_user_data_10: int
  compute_user_data_11: int
  compute_user_data_12: int
  compute_user_data_13: int
  compute_user_data_14: int
  compute_user_data_15: int
  cp_compute_csinvoc_count_lo: int
  cp_compute_csinvoc_count_hi: int
  reserved_83: int
  reserved_84: int
  reserved_85: int
  cp_mqd_query_time_lo: int
  cp_mqd_query_time_hi: int
  cp_mqd_connect_start_time_lo: int
  cp_mqd_connect_start_time_hi: int
  cp_mqd_connect_end_time_lo: int
  cp_mqd_connect_end_time_hi: int
  cp_mqd_connect_end_wf_count: int
  cp_mqd_connect_end_pq_rptr: int
  cp_mqd_connect_end_pq_wptr: int
  cp_mqd_connect_end_ib_rptr: int
  cp_mqd_readindex_lo: int
  cp_mqd_readindex_hi: int
  cp_mqd_save_start_time_lo: int
  cp_mqd_save_start_time_hi: int
  cp_mqd_save_end_time_lo: int
  cp_mqd_save_end_time_hi: int
  cp_mqd_restore_start_time_lo: int
  cp_mqd_restore_start_time_hi: int
  cp_mqd_restore_end_time_lo: int
  cp_mqd_restore_end_time_hi: int
  disable_queue: int
  reserved_107: int
  gds_cs_ctxsw_cnt0: int
  gds_cs_ctxsw_cnt1: int
  gds_cs_ctxsw_cnt2: int
  gds_cs_ctxsw_cnt3: int
  reserved_112: int
  reserved_113: int
  cp_pq_exe_status_lo: int
  cp_pq_exe_status_hi: int
  cp_packet_id_lo: int
  cp_packet_id_hi: int
  cp_packet_exe_status_lo: int
  cp_packet_exe_status_hi: int
  gds_save_base_addr_lo: int
  gds_save_base_addr_hi: int
  gds_save_mask_lo: int
  gds_save_mask_hi: int
  ctx_save_base_addr_lo: int
  ctx_save_base_addr_hi: int
  dynamic_cu_mask_addr_lo: int
  dynamic_cu_mask_addr_hi: int
  cp_mqd_base_addr_lo: int
  cp_mqd_base_addr_hi: int
  cp_hqd_active: int
  cp_hqd_vmid: int
  cp_hqd_persistent_state: int
  cp_hqd_pipe_priority: int
  cp_hqd_queue_priority: int
  cp_hqd_quantum: int
  cp_hqd_pq_base_lo: int
  cp_hqd_pq_base_hi: int
  cp_hqd_pq_rptr: int
  cp_hqd_pq_rptr_report_addr_lo: int
  cp_hqd_pq_rptr_report_addr_hi: int
  cp_hqd_pq_wptr_poll_addr_lo: int
  cp_hqd_pq_wptr_poll_addr_hi: int
  cp_hqd_pq_doorbell_control: int
  reserved_144: int
  cp_hqd_pq_control: int
  cp_hqd_ib_base_addr_lo: int
  cp_hqd_ib_base_addr_hi: int
  cp_hqd_ib_rptr: int
  cp_hqd_ib_control: int
  cp_hqd_iq_timer: int
  cp_hqd_iq_rptr: int
  cp_hqd_dequeue_request: int
  cp_hqd_dma_offload: int
  cp_hqd_sema_cmd: int
  cp_hqd_msg_type: int
  cp_hqd_atomic0_preop_lo: int
  cp_hqd_atomic0_preop_hi: int
  cp_hqd_atomic1_preop_lo: int
  cp_hqd_atomic1_preop_hi: int
  cp_hqd_hq_status0: int
  cp_hqd_hq_control0: int
  cp_mqd_control: int
  cp_hqd_hq_status1: int
  cp_hqd_hq_control1: int
  cp_hqd_eop_base_addr_lo: int
  cp_hqd_eop_base_addr_hi: int
  cp_hqd_eop_control: int
  cp_hqd_eop_rptr: int
  cp_hqd_eop_wptr: int
  cp_hqd_eop_done_events: int
  cp_hqd_ctx_save_base_addr_lo: int
  cp_hqd_ctx_save_base_addr_hi: int
  cp_hqd_ctx_save_control: int
  cp_hqd_cntl_stack_offset: int
  cp_hqd_cntl_stack_size: int
  cp_hqd_wg_state_offset: int
  cp_hqd_ctx_save_size: int
  cp_hqd_gds_resource_state: int
  cp_hqd_error: int
  cp_hqd_eop_wptr_mem: int
  cp_hqd_aql_control: int
  cp_hqd_pq_wptr_lo: int
  cp_hqd_pq_wptr_hi: int
  reserved_184: int
  reserved_185: int
  reserved_186: int
  reserved_187: int
  reserved_188: int
  reserved_189: int
  reserved_190: int
  reserved_191: int
  iqtimer_pkt_header: int
  iqtimer_pkt_dw0: int
  iqtimer_pkt_dw1: int
  iqtimer_pkt_dw2: int
  iqtimer_pkt_dw3: int
  iqtimer_pkt_dw4: int
  iqtimer_pkt_dw5: int
  iqtimer_pkt_dw6: int
  iqtimer_pkt_dw7: int
  iqtimer_pkt_dw8: int
  iqtimer_pkt_dw9: int
  iqtimer_pkt_dw10: int
  iqtimer_pkt_dw11: int
  iqtimer_pkt_dw12: int
  iqtimer_pkt_dw13: int
  iqtimer_pkt_dw14: int
  iqtimer_pkt_dw15: int
  iqtimer_pkt_dw16: int
  iqtimer_pkt_dw17: int
  iqtimer_pkt_dw18: int
  iqtimer_pkt_dw19: int
  iqtimer_pkt_dw20: int
  iqtimer_pkt_dw21: int
  iqtimer_pkt_dw22: int
  iqtimer_pkt_dw23: int
  iqtimer_pkt_dw24: int
  iqtimer_pkt_dw25: int
  iqtimer_pkt_dw26: int
  iqtimer_pkt_dw27: int
  iqtimer_pkt_dw28: int
  iqtimer_pkt_dw29: int
  iqtimer_pkt_dw30: int
  iqtimer_pkt_dw31: int
  reserved_225: int
  reserved_226: int
  pm4_target_xcc_in_xcp: int
  cp_mqd_stride_size: int
  reserved_227: int
  set_resources_header: int
  set_resources_dw1: int
  set_resources_dw2: int
  set_resources_dw3: int
  set_resources_dw4: int
  set_resources_dw5: int
  set_resources_dw6: int
  set_resources_dw7: int
  reserved_236: int
  reserved_237: int
  reserved_238: int
  reserved_239: int
  queue_doorbell_id0: int
  queue_doorbell_id1: int
  queue_doorbell_id2: int
  queue_doorbell_id3: int
  queue_doorbell_id4: int
  queue_doorbell_id5: int
  queue_doorbell_id6: int
  queue_doorbell_id7: int
  queue_doorbell_id8: int
  queue_doorbell_id9: int
  queue_doorbell_id10: int
  queue_doorbell_id11: int
  queue_doorbell_id12: int
  queue_doorbell_id13: int
  queue_doorbell_id14: int
  queue_doorbell_id15: int
  reserved_256: int
  reserved_257: int
  reserved_258: int
  reserved_259: int
  reserved_260: int
  reserved_261: int
  reserved_262: int
  reserved_263: int
  reserved_264: int
  reserved_265: int
  reserved_266: int
  reserved_267: int
  reserved_268: int
  reserved_269: int
  reserved_270: int
  reserved_271: int
  reserved_272: int
  reserved_273: int
  reserved_274: int
  reserved_275: int
  reserved_276: int
  reserved_277: int
  reserved_278: int
  reserved_279: int
  reserved_280: int
  reserved_281: int
  reserved_282: int
  reserved_283: int
  reserved_284: int
  reserved_285: int
  reserved_286: int
  reserved_287: int
  reserved_288: int
  reserved_289: int
  reserved_290: int
  reserved_291: int
  reserved_292: int
  reserved_293: int
  reserved_294: int
  reserved_295: int
  reserved_296: int
  reserved_297: int
  reserved_298: int
  reserved_299: int
  reserved_300: int
  reserved_301: int
  reserved_302: int
  reserved_303: int
  reserved_304: int
  reserved_305: int
  reserved_306: int
  reserved_307: int
  reserved_308: int
  reserved_309: int
  reserved_310: int
  reserved_311: int
  reserved_312: int
  reserved_313: int
  reserved_314: int
  reserved_315: int
  reserved_316: int
  reserved_317: int
  reserved_318: int
  reserved_319: int
  reserved_320: int
  reserved_321: int
  reserved_322: int
  reserved_323: int
  reserved_324: int
  reserved_325: int
  reserved_326: int
  reserved_327: int
  reserved_328: int
  reserved_329: int
  reserved_330: int
  reserved_331: int
  reserved_332: int
  reserved_333: int
  reserved_334: int
  reserved_335: int
  reserved_336: int
  reserved_337: int
  reserved_338: int
  reserved_339: int
  reserved_340: int
  reserved_341: int
  reserved_342: int
  reserved_343: int
  reserved_344: int
  reserved_345: int
  reserved_346: int
  reserved_347: int
  reserved_348: int
  reserved_349: int
  reserved_350: int
  reserved_351: int
  reserved_352: int
  reserved_353: int
  reserved_354: int
  reserved_355: int
  reserved_356: int
  reserved_357: int
  reserved_358: int
  reserved_359: int
  reserved_360: int
  reserved_361: int
  reserved_362: int
  reserved_363: int
  reserved_364: int
  reserved_365: int
  reserved_366: int
  reserved_367: int
  reserved_368: int
  reserved_369: int
  reserved_370: int
  reserved_371: int
  reserved_372: int
  reserved_373: int
  reserved_374: int
  reserved_375: int
  reserved_376: int
  reserved_377: int
  reserved_378: int
  reserved_379: int
  reserved_380: int
  reserved_381: int
  reserved_382: int
  reserved_383: int
  reserved_384: int
  reserved_385: int
  reserved_386: int
  reserved_387: int
  reserved_388: int
  reserved_389: int
  reserved_390: int
  reserved_391: int
  reserved_392: int
  reserved_393: int
  reserved_394: int
  reserved_395: int
  reserved_396: int
  reserved_397: int
  reserved_398: int
  reserved_399: int
  reserved_400: int
  reserved_401: int
  reserved_402: int
  reserved_403: int
  reserved_404: int
  reserved_405: int
  reserved_406: int
  reserved_407: int
  reserved_408: int
  reserved_409: int
  reserved_410: int
  reserved_411: int
  reserved_412: int
  reserved_413: int
  reserved_414: int
  reserved_415: int
  reserved_416: int
  reserved_417: int
  reserved_418: int
  reserved_419: int
  reserved_420: int
  reserved_421: int
  reserved_422: int
  reserved_423: int
  reserved_424: int
  reserved_425: int
  reserved_426: int
  reserved_427: int
  reserved_428: int
  reserved_429: int
  reserved_430: int
  reserved_431: int
  reserved_432: int
  reserved_433: int
  reserved_434: int
  reserved_435: int
  reserved_436: int
  reserved_437: int
  reserved_438: int
  reserved_439: int
  reserved_440: int
  reserved_441: int
  reserved_442: int
  reserved_443: int
  reserved_444: int
  reserved_445: int
  reserved_446: int
  reserved_447: int
  reserved_448: int
  reserved_449: int
  reserved_450: int
  reserved_451: int
  reserved_452: int
  reserved_453: int
  reserved_454: int
  reserved_455: int
  reserved_456: int
  reserved_457: int
  reserved_458: int
  reserved_459: int
  reserved_460: int
  reserved_461: int
  reserved_462: int
  reserved_463: int
  reserved_464: int
  reserved_465: int
  reserved_466: int
  reserved_467: int
  reserved_468: int
  reserved_469: int
  reserved_470: int
  reserved_471: int
  reserved_472: int
  reserved_473: int
  reserved_474: int
  reserved_475: int
  reserved_476: int
  reserved_477: int
  reserved_478: int
  reserved_479: int
  reserved_480: int
  reserved_481: int
  reserved_482: int
  reserved_483: int
  reserved_484: int
  reserved_485: int
  reserved_486: int
  reserved_487: int
  reserved_488: int
  reserved_489: int
  reserved_490: int
  reserved_491: int
  reserved_492: int
  reserved_493: int
  reserved_494: int
  reserved_495: int
  reserved_496: int
  reserved_497: int
  reserved_498: int
  reserved_499: int
  reserved_500: int
  reserved_501: int
  reserved_502: int
  reserved_503: int
  reserved_504: int
  reserved_505: int
  reserved_506: int
  reserved_507: int
  reserved_508: int
  reserved_509: int
  reserved_510: int
  reserved_511: int
struct_v9_mqd.register_fields([('header', uint32_t, 0), ('compute_dispatch_initiator', uint32_t, 4), ('compute_dim_x', uint32_t, 8), ('compute_dim_y', uint32_t, 12), ('compute_dim_z', uint32_t, 16), ('compute_start_x', uint32_t, 20), ('compute_start_y', uint32_t, 24), ('compute_start_z', uint32_t, 28), ('compute_num_thread_x', uint32_t, 32), ('compute_num_thread_y', uint32_t, 36), ('compute_num_thread_z', uint32_t, 40), ('compute_pipelinestat_enable', uint32_t, 44), ('compute_perfcount_enable', uint32_t, 48), ('compute_pgm_lo', uint32_t, 52), ('compute_pgm_hi', uint32_t, 56), ('compute_tba_lo', uint32_t, 60), ('compute_tba_hi', uint32_t, 64), ('compute_tma_lo', uint32_t, 68), ('compute_tma_hi', uint32_t, 72), ('compute_pgm_rsrc1', uint32_t, 76), ('compute_pgm_rsrc2', uint32_t, 80), ('compute_vmid', uint32_t, 84), ('compute_resource_limits', uint32_t, 88), ('compute_static_thread_mgmt_se0', uint32_t, 92), ('compute_static_thread_mgmt_se1', uint32_t, 96), ('compute_tmpring_size', uint32_t, 100), ('compute_static_thread_mgmt_se2', uint32_t, 104), ('compute_static_thread_mgmt_se3', uint32_t, 108), ('compute_restart_x', uint32_t, 112), ('compute_restart_y', uint32_t, 116), ('compute_restart_z', uint32_t, 120), ('compute_thread_trace_enable', uint32_t, 124), ('compute_misc_reserved', uint32_t, 128), ('compute_dispatch_id', uint32_t, 132), ('compute_threadgroup_id', uint32_t, 136), ('compute_relaunch', uint32_t, 140), ('compute_wave_restore_addr_lo', uint32_t, 144), ('compute_wave_restore_addr_hi', uint32_t, 148), ('compute_wave_restore_control', uint32_t, 152), ('compute_static_thread_mgmt_se4', uint32_t, 156), ('compute_static_thread_mgmt_se5', uint32_t, 160), ('compute_static_thread_mgmt_se6', uint32_t, 164), ('compute_static_thread_mgmt_se7', uint32_t, 168), ('compute_current_logic_xcc_id', uint32_t, 156), ('compute_restart_cg_tg_id', uint32_t, 160), ('compute_tg_chunk_size', uint32_t, 164), ('compute_restore_tg_chunk_size', uint32_t, 168), ('reserved_43', uint32_t, 172), ('reserved_44', uint32_t, 176), ('reserved_45', uint32_t, 180), ('reserved_46', uint32_t, 184), ('reserved_47', uint32_t, 188), ('reserved_48', uint32_t, 192), ('reserved_49', uint32_t, 196), ('reserved_50', uint32_t, 200), ('reserved_51', uint32_t, 204), ('reserved_52', uint32_t, 208), ('reserved_53', uint32_t, 212), ('reserved_54', uint32_t, 216), ('reserved_55', uint32_t, 220), ('reserved_56', uint32_t, 224), ('reserved_57', uint32_t, 228), ('reserved_58', uint32_t, 232), ('reserved_59', uint32_t, 236), ('reserved_60', uint32_t, 240), ('reserved_61', uint32_t, 244), ('reserved_62', uint32_t, 248), ('reserved_63', uint32_t, 252), ('reserved_64', uint32_t, 256), ('compute_user_data_0', uint32_t, 260), ('compute_user_data_1', uint32_t, 264), ('compute_user_data_2', uint32_t, 268), ('compute_user_data_3', uint32_t, 272), ('compute_user_data_4', uint32_t, 276), ('compute_user_data_5', uint32_t, 280), ('compute_user_data_6', uint32_t, 284), ('compute_user_data_7', uint32_t, 288), ('compute_user_data_8', uint32_t, 292), ('compute_user_data_9', uint32_t, 296), ('compute_user_data_10', uint32_t, 300), ('compute_user_data_11', uint32_t, 304), ('compute_user_data_12', uint32_t, 308), ('compute_user_data_13', uint32_t, 312), ('compute_user_data_14', uint32_t, 316), ('compute_user_data_15', uint32_t, 320), ('cp_compute_csinvoc_count_lo', uint32_t, 324), ('cp_compute_csinvoc_count_hi', uint32_t, 328), ('reserved_83', uint32_t, 332), ('reserved_84', uint32_t, 336), ('reserved_85', uint32_t, 340), ('cp_mqd_query_time_lo', uint32_t, 344), ('cp_mqd_query_time_hi', uint32_t, 348), ('cp_mqd_connect_start_time_lo', uint32_t, 352), ('cp_mqd_connect_start_time_hi', uint32_t, 356), ('cp_mqd_connect_end_time_lo', uint32_t, 360), ('cp_mqd_connect_end_time_hi', uint32_t, 364), ('cp_mqd_connect_end_wf_count', uint32_t, 368), ('cp_mqd_connect_end_pq_rptr', uint32_t, 372), ('cp_mqd_connect_end_pq_wptr', uint32_t, 376), ('cp_mqd_connect_end_ib_rptr', uint32_t, 380), ('cp_mqd_readindex_lo', uint32_t, 384), ('cp_mqd_readindex_hi', uint32_t, 388), ('cp_mqd_save_start_time_lo', uint32_t, 392), ('cp_mqd_save_start_time_hi', uint32_t, 396), ('cp_mqd_save_end_time_lo', uint32_t, 400), ('cp_mqd_save_end_time_hi', uint32_t, 404), ('cp_mqd_restore_start_time_lo', uint32_t, 408), ('cp_mqd_restore_start_time_hi', uint32_t, 412), ('cp_mqd_restore_end_time_lo', uint32_t, 416), ('cp_mqd_restore_end_time_hi', uint32_t, 420), ('disable_queue', uint32_t, 424), ('reserved_107', uint32_t, 428), ('gds_cs_ctxsw_cnt0', uint32_t, 432), ('gds_cs_ctxsw_cnt1', uint32_t, 436), ('gds_cs_ctxsw_cnt2', uint32_t, 440), ('gds_cs_ctxsw_cnt3', uint32_t, 444), ('reserved_112', uint32_t, 448), ('reserved_113', uint32_t, 452), ('cp_pq_exe_status_lo', uint32_t, 456), ('cp_pq_exe_status_hi', uint32_t, 460), ('cp_packet_id_lo', uint32_t, 464), ('cp_packet_id_hi', uint32_t, 468), ('cp_packet_exe_status_lo', uint32_t, 472), ('cp_packet_exe_status_hi', uint32_t, 476), ('gds_save_base_addr_lo', uint32_t, 480), ('gds_save_base_addr_hi', uint32_t, 484), ('gds_save_mask_lo', uint32_t, 488), ('gds_save_mask_hi', uint32_t, 492), ('ctx_save_base_addr_lo', uint32_t, 496), ('ctx_save_base_addr_hi', uint32_t, 500), ('dynamic_cu_mask_addr_lo', uint32_t, 504), ('dynamic_cu_mask_addr_hi', uint32_t, 508), ('cp_mqd_base_addr_lo', uint32_t, 512), ('cp_mqd_base_addr_hi', uint32_t, 516), ('cp_hqd_active', uint32_t, 520), ('cp_hqd_vmid', uint32_t, 524), ('cp_hqd_persistent_state', uint32_t, 528), ('cp_hqd_pipe_priority', uint32_t, 532), ('cp_hqd_queue_priority', uint32_t, 536), ('cp_hqd_quantum', uint32_t, 540), ('cp_hqd_pq_base_lo', uint32_t, 544), ('cp_hqd_pq_base_hi', uint32_t, 548), ('cp_hqd_pq_rptr', uint32_t, 552), ('cp_hqd_pq_rptr_report_addr_lo', uint32_t, 556), ('cp_hqd_pq_rptr_report_addr_hi', uint32_t, 560), ('cp_hqd_pq_wptr_poll_addr_lo', uint32_t, 564), ('cp_hqd_pq_wptr_poll_addr_hi', uint32_t, 568), ('cp_hqd_pq_doorbell_control', uint32_t, 572), ('reserved_144', uint32_t, 576), ('cp_hqd_pq_control', uint32_t, 580), ('cp_hqd_ib_base_addr_lo', uint32_t, 584), ('cp_hqd_ib_base_addr_hi', uint32_t, 588), ('cp_hqd_ib_rptr', uint32_t, 592), ('cp_hqd_ib_control', uint32_t, 596), ('cp_hqd_iq_timer', uint32_t, 600), ('cp_hqd_iq_rptr', uint32_t, 604), ('cp_hqd_dequeue_request', uint32_t, 608), ('cp_hqd_dma_offload', uint32_t, 612), ('cp_hqd_sema_cmd', uint32_t, 616), ('cp_hqd_msg_type', uint32_t, 620), ('cp_hqd_atomic0_preop_lo', uint32_t, 624), ('cp_hqd_atomic0_preop_hi', uint32_t, 628), ('cp_hqd_atomic1_preop_lo', uint32_t, 632), ('cp_hqd_atomic1_preop_hi', uint32_t, 636), ('cp_hqd_hq_status0', uint32_t, 640), ('cp_hqd_hq_control0', uint32_t, 644), ('cp_mqd_control', uint32_t, 648), ('cp_hqd_hq_status1', uint32_t, 652), ('cp_hqd_hq_control1', uint32_t, 656), ('cp_hqd_eop_base_addr_lo', uint32_t, 660), ('cp_hqd_eop_base_addr_hi', uint32_t, 664), ('cp_hqd_eop_control', uint32_t, 668), ('cp_hqd_eop_rptr', uint32_t, 672), ('cp_hqd_eop_wptr', uint32_t, 676), ('cp_hqd_eop_done_events', uint32_t, 680), ('cp_hqd_ctx_save_base_addr_lo', uint32_t, 684), ('cp_hqd_ctx_save_base_addr_hi', uint32_t, 688), ('cp_hqd_ctx_save_control', uint32_t, 692), ('cp_hqd_cntl_stack_offset', uint32_t, 696), ('cp_hqd_cntl_stack_size', uint32_t, 700), ('cp_hqd_wg_state_offset', uint32_t, 704), ('cp_hqd_ctx_save_size', uint32_t, 708), ('cp_hqd_gds_resource_state', uint32_t, 712), ('cp_hqd_error', uint32_t, 716), ('cp_hqd_eop_wptr_mem', uint32_t, 720), ('cp_hqd_aql_control', uint32_t, 724), ('cp_hqd_pq_wptr_lo', uint32_t, 728), ('cp_hqd_pq_wptr_hi', uint32_t, 732), ('reserved_184', uint32_t, 736), ('reserved_185', uint32_t, 740), ('reserved_186', uint32_t, 744), ('reserved_187', uint32_t, 748), ('reserved_188', uint32_t, 752), ('reserved_189', uint32_t, 756), ('reserved_190', uint32_t, 760), ('reserved_191', uint32_t, 764), ('iqtimer_pkt_header', uint32_t, 768), ('iqtimer_pkt_dw0', uint32_t, 772), ('iqtimer_pkt_dw1', uint32_t, 776), ('iqtimer_pkt_dw2', uint32_t, 780), ('iqtimer_pkt_dw3', uint32_t, 784), ('iqtimer_pkt_dw4', uint32_t, 788), ('iqtimer_pkt_dw5', uint32_t, 792), ('iqtimer_pkt_dw6', uint32_t, 796), ('iqtimer_pkt_dw7', uint32_t, 800), ('iqtimer_pkt_dw8', uint32_t, 804), ('iqtimer_pkt_dw9', uint32_t, 808), ('iqtimer_pkt_dw10', uint32_t, 812), ('iqtimer_pkt_dw11', uint32_t, 816), ('iqtimer_pkt_dw12', uint32_t, 820), ('iqtimer_pkt_dw13', uint32_t, 824), ('iqtimer_pkt_dw14', uint32_t, 828), ('iqtimer_pkt_dw15', uint32_t, 832), ('iqtimer_pkt_dw16', uint32_t, 836), ('iqtimer_pkt_dw17', uint32_t, 840), ('iqtimer_pkt_dw18', uint32_t, 844), ('iqtimer_pkt_dw19', uint32_t, 848), ('iqtimer_pkt_dw20', uint32_t, 852), ('iqtimer_pkt_dw21', uint32_t, 856), ('iqtimer_pkt_dw22', uint32_t, 860), ('iqtimer_pkt_dw23', uint32_t, 864), ('iqtimer_pkt_dw24', uint32_t, 868), ('iqtimer_pkt_dw25', uint32_t, 872), ('iqtimer_pkt_dw26', uint32_t, 876), ('iqtimer_pkt_dw27', uint32_t, 880), ('iqtimer_pkt_dw28', uint32_t, 884), ('iqtimer_pkt_dw29', uint32_t, 888), ('iqtimer_pkt_dw30', uint32_t, 892), ('iqtimer_pkt_dw31', uint32_t, 896), ('reserved_225', uint32_t, 900), ('reserved_226', uint32_t, 904), ('pm4_target_xcc_in_xcp', uint32_t, 900), ('cp_mqd_stride_size', uint32_t, 904), ('reserved_227', uint32_t, 908), ('set_resources_header', uint32_t, 912), ('set_resources_dw1', uint32_t, 916), ('set_resources_dw2', uint32_t, 920), ('set_resources_dw3', uint32_t, 924), ('set_resources_dw4', uint32_t, 928), ('set_resources_dw5', uint32_t, 932), ('set_resources_dw6', uint32_t, 936), ('set_resources_dw7', uint32_t, 940), ('reserved_236', uint32_t, 944), ('reserved_237', uint32_t, 948), ('reserved_238', uint32_t, 952), ('reserved_239', uint32_t, 956), ('queue_doorbell_id0', uint32_t, 960), ('queue_doorbell_id1', uint32_t, 964), ('queue_doorbell_id2', uint32_t, 968), ('queue_doorbell_id3', uint32_t, 972), ('queue_doorbell_id4', uint32_t, 976), ('queue_doorbell_id5', uint32_t, 980), ('queue_doorbell_id6', uint32_t, 984), ('queue_doorbell_id7', uint32_t, 988), ('queue_doorbell_id8', uint32_t, 992), ('queue_doorbell_id9', uint32_t, 996), ('queue_doorbell_id10', uint32_t, 1000), ('queue_doorbell_id11', uint32_t, 1004), ('queue_doorbell_id12', uint32_t, 1008), ('queue_doorbell_id13', uint32_t, 1012), ('queue_doorbell_id14', uint32_t, 1016), ('queue_doorbell_id15', uint32_t, 1020), ('reserved_256', uint32_t, 1024), ('reserved_257', uint32_t, 1028), ('reserved_258', uint32_t, 1032), ('reserved_259', uint32_t, 1036), ('reserved_260', uint32_t, 1040), ('reserved_261', uint32_t, 1044), ('reserved_262', uint32_t, 1048), ('reserved_263', uint32_t, 1052), ('reserved_264', uint32_t, 1056), ('reserved_265', uint32_t, 1060), ('reserved_266', uint32_t, 1064), ('reserved_267', uint32_t, 1068), ('reserved_268', uint32_t, 1072), ('reserved_269', uint32_t, 1076), ('reserved_270', uint32_t, 1080), ('reserved_271', uint32_t, 1084), ('reserved_272', uint32_t, 1088), ('reserved_273', uint32_t, 1092), ('reserved_274', uint32_t, 1096), ('reserved_275', uint32_t, 1100), ('reserved_276', uint32_t, 1104), ('reserved_277', uint32_t, 1108), ('reserved_278', uint32_t, 1112), ('reserved_279', uint32_t, 1116), ('reserved_280', uint32_t, 1120), ('reserved_281', uint32_t, 1124), ('reserved_282', uint32_t, 1128), ('reserved_283', uint32_t, 1132), ('reserved_284', uint32_t, 1136), ('reserved_285', uint32_t, 1140), ('reserved_286', uint32_t, 1144), ('reserved_287', uint32_t, 1148), ('reserved_288', uint32_t, 1152), ('reserved_289', uint32_t, 1156), ('reserved_290', uint32_t, 1160), ('reserved_291', uint32_t, 1164), ('reserved_292', uint32_t, 1168), ('reserved_293', uint32_t, 1172), ('reserved_294', uint32_t, 1176), ('reserved_295', uint32_t, 1180), ('reserved_296', uint32_t, 1184), ('reserved_297', uint32_t, 1188), ('reserved_298', uint32_t, 1192), ('reserved_299', uint32_t, 1196), ('reserved_300', uint32_t, 1200), ('reserved_301', uint32_t, 1204), ('reserved_302', uint32_t, 1208), ('reserved_303', uint32_t, 1212), ('reserved_304', uint32_t, 1216), ('reserved_305', uint32_t, 1220), ('reserved_306', uint32_t, 1224), ('reserved_307', uint32_t, 1228), ('reserved_308', uint32_t, 1232), ('reserved_309', uint32_t, 1236), ('reserved_310', uint32_t, 1240), ('reserved_311', uint32_t, 1244), ('reserved_312', uint32_t, 1248), ('reserved_313', uint32_t, 1252), ('reserved_314', uint32_t, 1256), ('reserved_315', uint32_t, 1260), ('reserved_316', uint32_t, 1264), ('reserved_317', uint32_t, 1268), ('reserved_318', uint32_t, 1272), ('reserved_319', uint32_t, 1276), ('reserved_320', uint32_t, 1280), ('reserved_321', uint32_t, 1284), ('reserved_322', uint32_t, 1288), ('reserved_323', uint32_t, 1292), ('reserved_324', uint32_t, 1296), ('reserved_325', uint32_t, 1300), ('reserved_326', uint32_t, 1304), ('reserved_327', uint32_t, 1308), ('reserved_328', uint32_t, 1312), ('reserved_329', uint32_t, 1316), ('reserved_330', uint32_t, 1320), ('reserved_331', uint32_t, 1324), ('reserved_332', uint32_t, 1328), ('reserved_333', uint32_t, 1332), ('reserved_334', uint32_t, 1336), ('reserved_335', uint32_t, 1340), ('reserved_336', uint32_t, 1344), ('reserved_337', uint32_t, 1348), ('reserved_338', uint32_t, 1352), ('reserved_339', uint32_t, 1356), ('reserved_340', uint32_t, 1360), ('reserved_341', uint32_t, 1364), ('reserved_342', uint32_t, 1368), ('reserved_343', uint32_t, 1372), ('reserved_344', uint32_t, 1376), ('reserved_345', uint32_t, 1380), ('reserved_346', uint32_t, 1384), ('reserved_347', uint32_t, 1388), ('reserved_348', uint32_t, 1392), ('reserved_349', uint32_t, 1396), ('reserved_350', uint32_t, 1400), ('reserved_351', uint32_t, 1404), ('reserved_352', uint32_t, 1408), ('reserved_353', uint32_t, 1412), ('reserved_354', uint32_t, 1416), ('reserved_355', uint32_t, 1420), ('reserved_356', uint32_t, 1424), ('reserved_357', uint32_t, 1428), ('reserved_358', uint32_t, 1432), ('reserved_359', uint32_t, 1436), ('reserved_360', uint32_t, 1440), ('reserved_361', uint32_t, 1444), ('reserved_362', uint32_t, 1448), ('reserved_363', uint32_t, 1452), ('reserved_364', uint32_t, 1456), ('reserved_365', uint32_t, 1460), ('reserved_366', uint32_t, 1464), ('reserved_367', uint32_t, 1468), ('reserved_368', uint32_t, 1472), ('reserved_369', uint32_t, 1476), ('reserved_370', uint32_t, 1480), ('reserved_371', uint32_t, 1484), ('reserved_372', uint32_t, 1488), ('reserved_373', uint32_t, 1492), ('reserved_374', uint32_t, 1496), ('reserved_375', uint32_t, 1500), ('reserved_376', uint32_t, 1504), ('reserved_377', uint32_t, 1508), ('reserved_378', uint32_t, 1512), ('reserved_379', uint32_t, 1516), ('reserved_380', uint32_t, 1520), ('reserved_381', uint32_t, 1524), ('reserved_382', uint32_t, 1528), ('reserved_383', uint32_t, 1532), ('reserved_384', uint32_t, 1536), ('reserved_385', uint32_t, 1540), ('reserved_386', uint32_t, 1544), ('reserved_387', uint32_t, 1548), ('reserved_388', uint32_t, 1552), ('reserved_389', uint32_t, 1556), ('reserved_390', uint32_t, 1560), ('reserved_391', uint32_t, 1564), ('reserved_392', uint32_t, 1568), ('reserved_393', uint32_t, 1572), ('reserved_394', uint32_t, 1576), ('reserved_395', uint32_t, 1580), ('reserved_396', uint32_t, 1584), ('reserved_397', uint32_t, 1588), ('reserved_398', uint32_t, 1592), ('reserved_399', uint32_t, 1596), ('reserved_400', uint32_t, 1600), ('reserved_401', uint32_t, 1604), ('reserved_402', uint32_t, 1608), ('reserved_403', uint32_t, 1612), ('reserved_404', uint32_t, 1616), ('reserved_405', uint32_t, 1620), ('reserved_406', uint32_t, 1624), ('reserved_407', uint32_t, 1628), ('reserved_408', uint32_t, 1632), ('reserved_409', uint32_t, 1636), ('reserved_410', uint32_t, 1640), ('reserved_411', uint32_t, 1644), ('reserved_412', uint32_t, 1648), ('reserved_413', uint32_t, 1652), ('reserved_414', uint32_t, 1656), ('reserved_415', uint32_t, 1660), ('reserved_416', uint32_t, 1664), ('reserved_417', uint32_t, 1668), ('reserved_418', uint32_t, 1672), ('reserved_419', uint32_t, 1676), ('reserved_420', uint32_t, 1680), ('reserved_421', uint32_t, 1684), ('reserved_422', uint32_t, 1688), ('reserved_423', uint32_t, 1692), ('reserved_424', uint32_t, 1696), ('reserved_425', uint32_t, 1700), ('reserved_426', uint32_t, 1704), ('reserved_427', uint32_t, 1708), ('reserved_428', uint32_t, 1712), ('reserved_429', uint32_t, 1716), ('reserved_430', uint32_t, 1720), ('reserved_431', uint32_t, 1724), ('reserved_432', uint32_t, 1728), ('reserved_433', uint32_t, 1732), ('reserved_434', uint32_t, 1736), ('reserved_435', uint32_t, 1740), ('reserved_436', uint32_t, 1744), ('reserved_437', uint32_t, 1748), ('reserved_438', uint32_t, 1752), ('reserved_439', uint32_t, 1756), ('reserved_440', uint32_t, 1760), ('reserved_441', uint32_t, 1764), ('reserved_442', uint32_t, 1768), ('reserved_443', uint32_t, 1772), ('reserved_444', uint32_t, 1776), ('reserved_445', uint32_t, 1780), ('reserved_446', uint32_t, 1784), ('reserved_447', uint32_t, 1788), ('reserved_448', uint32_t, 1792), ('reserved_449', uint32_t, 1796), ('reserved_450', uint32_t, 1800), ('reserved_451', uint32_t, 1804), ('reserved_452', uint32_t, 1808), ('reserved_453', uint32_t, 1812), ('reserved_454', uint32_t, 1816), ('reserved_455', uint32_t, 1820), ('reserved_456', uint32_t, 1824), ('reserved_457', uint32_t, 1828), ('reserved_458', uint32_t, 1832), ('reserved_459', uint32_t, 1836), ('reserved_460', uint32_t, 1840), ('reserved_461', uint32_t, 1844), ('reserved_462', uint32_t, 1848), ('reserved_463', uint32_t, 1852), ('reserved_464', uint32_t, 1856), ('reserved_465', uint32_t, 1860), ('reserved_466', uint32_t, 1864), ('reserved_467', uint32_t, 1868), ('reserved_468', uint32_t, 1872), ('reserved_469', uint32_t, 1876), ('reserved_470', uint32_t, 1880), ('reserved_471', uint32_t, 1884), ('reserved_472', uint32_t, 1888), ('reserved_473', uint32_t, 1892), ('reserved_474', uint32_t, 1896), ('reserved_475', uint32_t, 1900), ('reserved_476', uint32_t, 1904), ('reserved_477', uint32_t, 1908), ('reserved_478', uint32_t, 1912), ('reserved_479', uint32_t, 1916), ('reserved_480', uint32_t, 1920), ('reserved_481', uint32_t, 1924), ('reserved_482', uint32_t, 1928), ('reserved_483', uint32_t, 1932), ('reserved_484', uint32_t, 1936), ('reserved_485', uint32_t, 1940), ('reserved_486', uint32_t, 1944), ('reserved_487', uint32_t, 1948), ('reserved_488', uint32_t, 1952), ('reserved_489', uint32_t, 1956), ('reserved_490', uint32_t, 1960), ('reserved_491', uint32_t, 1964), ('reserved_492', uint32_t, 1968), ('reserved_493', uint32_t, 1972), ('reserved_494', uint32_t, 1976), ('reserved_495', uint32_t, 1980), ('reserved_496', uint32_t, 1984), ('reserved_497', uint32_t, 1988), ('reserved_498', uint32_t, 1992), ('reserved_499', uint32_t, 1996), ('reserved_500', uint32_t, 2000), ('reserved_501', uint32_t, 2004), ('reserved_502', uint32_t, 2008), ('reserved_503', uint32_t, 2012), ('reserved_504', uint32_t, 2016), ('reserved_505', uint32_t, 2020), ('reserved_506', uint32_t, 2024), ('reserved_507', uint32_t, 2028), ('reserved_508', uint32_t, 2032), ('reserved_509', uint32_t, 2036), ('reserved_510', uint32_t, 2040), ('reserved_511', uint32_t, 2044)])
@c.record
class struct_v9_mqd_allocation(c.Struct):
  SIZE = 2064
  mqd: struct_v9_mqd
  wptr_poll_mem: int
  rptr_report_mem: int
  dynamic_cu_mask: int
  dynamic_rb_mask: int
struct_v9_mqd_allocation.register_fields([('mqd', struct_v9_mqd, 0), ('wptr_poll_mem', uint32_t, 2048), ('rptr_report_mem', uint32_t, 2052), ('dynamic_cu_mask', uint32_t, 2056), ('dynamic_rb_mask', uint32_t, 2060)])
@c.record
class struct_v9_ce_ib_state(c.Struct):
  SIZE = 40
  ce_ib_completion_status: int
  ce_constegnine_count: int
  ce_ibOffset_ib1: int
  ce_ibOffset_ib2: int
  ce_chainib_addrlo_ib1: int
  ce_chainib_addrlo_ib2: int
  ce_chainib_addrhi_ib1: int
  ce_chainib_addrhi_ib2: int
  ce_chainib_size_ib1: int
  ce_chainib_size_ib2: int
struct_v9_ce_ib_state.register_fields([('ce_ib_completion_status', uint32_t, 0), ('ce_constegnine_count', uint32_t, 4), ('ce_ibOffset_ib1', uint32_t, 8), ('ce_ibOffset_ib2', uint32_t, 12), ('ce_chainib_addrlo_ib1', uint32_t, 16), ('ce_chainib_addrlo_ib2', uint32_t, 20), ('ce_chainib_addrhi_ib1', uint32_t, 24), ('ce_chainib_addrhi_ib2', uint32_t, 28), ('ce_chainib_size_ib1', uint32_t, 32), ('ce_chainib_size_ib2', uint32_t, 36)])
@c.record
class struct_v9_de_ib_state(c.Struct):
  SIZE = 108
  ib_completion_status: int
  de_constEngine_count: int
  ib_offset_ib1: int
  ib_offset_ib2: int
  chain_ib_addrlo_ib1: int
  chain_ib_addrlo_ib2: int
  chain_ib_addrhi_ib1: int
  chain_ib_addrhi_ib2: int
  chain_ib_size_ib1: int
  chain_ib_size_ib2: int
  preamble_begin_ib1: int
  preamble_begin_ib2: int
  preamble_end_ib1: int
  preamble_end_ib2: int
  chain_ib_pream_addrlo_ib1: int
  chain_ib_pream_addrlo_ib2: int
  chain_ib_pream_addrhi_ib1: int
  chain_ib_pream_addrhi_ib2: int
  draw_indirect_baseLo: int
  draw_indirect_baseHi: int
  disp_indirect_baseLo: int
  disp_indirect_baseHi: int
  gds_backup_addrlo: int
  gds_backup_addrhi: int
  index_base_addrlo: int
  index_base_addrhi: int
  sample_cntl: int
struct_v9_de_ib_state.register_fields([('ib_completion_status', uint32_t, 0), ('de_constEngine_count', uint32_t, 4), ('ib_offset_ib1', uint32_t, 8), ('ib_offset_ib2', uint32_t, 12), ('chain_ib_addrlo_ib1', uint32_t, 16), ('chain_ib_addrlo_ib2', uint32_t, 20), ('chain_ib_addrhi_ib1', uint32_t, 24), ('chain_ib_addrhi_ib2', uint32_t, 28), ('chain_ib_size_ib1', uint32_t, 32), ('chain_ib_size_ib2', uint32_t, 36), ('preamble_begin_ib1', uint32_t, 40), ('preamble_begin_ib2', uint32_t, 44), ('preamble_end_ib1', uint32_t, 48), ('preamble_end_ib2', uint32_t, 52), ('chain_ib_pream_addrlo_ib1', uint32_t, 56), ('chain_ib_pream_addrlo_ib2', uint32_t, 60), ('chain_ib_pream_addrhi_ib1', uint32_t, 64), ('chain_ib_pream_addrhi_ib2', uint32_t, 68), ('draw_indirect_baseLo', uint32_t, 72), ('draw_indirect_baseHi', uint32_t, 76), ('disp_indirect_baseLo', uint32_t, 80), ('disp_indirect_baseHi', uint32_t, 84), ('gds_backup_addrlo', uint32_t, 88), ('gds_backup_addrhi', uint32_t, 92), ('index_base_addrlo', uint32_t, 96), ('index_base_addrhi', uint32_t, 100), ('sample_cntl', uint32_t, 104)])
@c.record
class struct_v9_gfx_meta_data(c.Struct):
  SIZE = 4096
  ce_payload: struct_v9_ce_ib_state
  reserved1: c.Array[ctypes.c_uint32, Literal[54]]
  de_payload: struct_v9_de_ib_state
  DeIbBaseAddrLo: int
  DeIbBaseAddrHi: int
  reserved2: c.Array[ctypes.c_uint32, Literal[931]]
struct_v9_gfx_meta_data.register_fields([('ce_payload', struct_v9_ce_ib_state, 0), ('reserved1', c.Array[uint32_t, Literal[54]], 40), ('de_payload', struct_v9_de_ib_state, 256), ('DeIbBaseAddrLo', uint32_t, 364), ('DeIbBaseAddrHi', uint32_t, 368), ('reserved2', c.Array[uint32_t, Literal[931]], 372)])
enum_soc15_ih_clientid: dict[int, str] = {(SOC15_IH_CLIENTID_IH:=0): 'SOC15_IH_CLIENTID_IH', (SOC15_IH_CLIENTID_ACP:=1): 'SOC15_IH_CLIENTID_ACP', (SOC15_IH_CLIENTID_ATHUB:=2): 'SOC15_IH_CLIENTID_ATHUB', (SOC15_IH_CLIENTID_BIF:=3): 'SOC15_IH_CLIENTID_BIF', (SOC15_IH_CLIENTID_DCE:=4): 'SOC15_IH_CLIENTID_DCE', (SOC15_IH_CLIENTID_ISP:=5): 'SOC15_IH_CLIENTID_ISP', (SOC15_IH_CLIENTID_PCIE0:=6): 'SOC15_IH_CLIENTID_PCIE0', (SOC15_IH_CLIENTID_RLC:=7): 'SOC15_IH_CLIENTID_RLC', (SOC15_IH_CLIENTID_SDMA0:=8): 'SOC15_IH_CLIENTID_SDMA0', (SOC15_IH_CLIENTID_SDMA1:=9): 'SOC15_IH_CLIENTID_SDMA1', (SOC15_IH_CLIENTID_SE0SH:=10): 'SOC15_IH_CLIENTID_SE0SH', (SOC15_IH_CLIENTID_SE1SH:=11): 'SOC15_IH_CLIENTID_SE1SH', (SOC15_IH_CLIENTID_SE2SH:=12): 'SOC15_IH_CLIENTID_SE2SH', (SOC15_IH_CLIENTID_SE3SH:=13): 'SOC15_IH_CLIENTID_SE3SH', (SOC15_IH_CLIENTID_UVD1:=14): 'SOC15_IH_CLIENTID_UVD1', (SOC15_IH_CLIENTID_THM:=15): 'SOC15_IH_CLIENTID_THM', (SOC15_IH_CLIENTID_UVD:=16): 'SOC15_IH_CLIENTID_UVD', (SOC15_IH_CLIENTID_VCE0:=17): 'SOC15_IH_CLIENTID_VCE0', (SOC15_IH_CLIENTID_VMC:=18): 'SOC15_IH_CLIENTID_VMC', (SOC15_IH_CLIENTID_XDMA:=19): 'SOC15_IH_CLIENTID_XDMA', (SOC15_IH_CLIENTID_GRBM_CP:=20): 'SOC15_IH_CLIENTID_GRBM_CP', (SOC15_IH_CLIENTID_ATS:=21): 'SOC15_IH_CLIENTID_ATS', (SOC15_IH_CLIENTID_ROM_SMUIO:=22): 'SOC15_IH_CLIENTID_ROM_SMUIO', (SOC15_IH_CLIENTID_DF:=23): 'SOC15_IH_CLIENTID_DF', (SOC15_IH_CLIENTID_VCE1:=24): 'SOC15_IH_CLIENTID_VCE1', (SOC15_IH_CLIENTID_PWR:=25): 'SOC15_IH_CLIENTID_PWR', (SOC15_IH_CLIENTID_RESERVED:=26): 'SOC15_IH_CLIENTID_RESERVED', (SOC15_IH_CLIENTID_UTCL2:=27): 'SOC15_IH_CLIENTID_UTCL2', (SOC15_IH_CLIENTID_EA:=28): 'SOC15_IH_CLIENTID_EA', (SOC15_IH_CLIENTID_UTCL2LOG:=29): 'SOC15_IH_CLIENTID_UTCL2LOG', (SOC15_IH_CLIENTID_MP0:=30): 'SOC15_IH_CLIENTID_MP0', (SOC15_IH_CLIENTID_MP1:=31): 'SOC15_IH_CLIENTID_MP1', (SOC15_IH_CLIENTID_MAX:=32): 'SOC15_IH_CLIENTID_MAX', (SOC15_IH_CLIENTID_VCN:=16): 'SOC15_IH_CLIENTID_VCN', (SOC15_IH_CLIENTID_VCN1:=14): 'SOC15_IH_CLIENTID_VCN1', (SOC15_IH_CLIENTID_SDMA2:=1): 'SOC15_IH_CLIENTID_SDMA2', (SOC15_IH_CLIENTID_SDMA3:=4): 'SOC15_IH_CLIENTID_SDMA3', (SOC15_IH_CLIENTID_SDMA3_Sienna_Cichlid:=5): 'SOC15_IH_CLIENTID_SDMA3_Sienna_Cichlid', (SOC15_IH_CLIENTID_SDMA4:=5): 'SOC15_IH_CLIENTID_SDMA4', (SOC15_IH_CLIENTID_SDMA5:=17): 'SOC15_IH_CLIENTID_SDMA5', (SOC15_IH_CLIENTID_SDMA6:=19): 'SOC15_IH_CLIENTID_SDMA6', (SOC15_IH_CLIENTID_SDMA7:=24): 'SOC15_IH_CLIENTID_SDMA7', (SOC15_IH_CLIENTID_VMC1:=6): 'SOC15_IH_CLIENTID_VMC1'}
enum_soc21_ih_clientid: dict[int, str] = {(SOC21_IH_CLIENTID_IH:=0): 'SOC21_IH_CLIENTID_IH', (SOC21_IH_CLIENTID_ATHUB:=2): 'SOC21_IH_CLIENTID_ATHUB', (SOC21_IH_CLIENTID_BIF:=3): 'SOC21_IH_CLIENTID_BIF', (SOC21_IH_CLIENTID_DCN:=4): 'SOC21_IH_CLIENTID_DCN', (SOC21_IH_CLIENTID_ISP:=5): 'SOC21_IH_CLIENTID_ISP', (SOC21_IH_CLIENTID_MP3:=6): 'SOC21_IH_CLIENTID_MP3', (SOC21_IH_CLIENTID_RLC:=7): 'SOC21_IH_CLIENTID_RLC', (SOC21_IH_CLIENTID_GFX:=10): 'SOC21_IH_CLIENTID_GFX', (SOC21_IH_CLIENTID_IMU:=11): 'SOC21_IH_CLIENTID_IMU', (SOC21_IH_CLIENTID_VCN1:=14): 'SOC21_IH_CLIENTID_VCN1', (SOC21_IH_CLIENTID_THM:=15): 'SOC21_IH_CLIENTID_THM', (SOC21_IH_CLIENTID_VCN:=16): 'SOC21_IH_CLIENTID_VCN', (SOC21_IH_CLIENTID_VPE1:=17): 'SOC21_IH_CLIENTID_VPE1', (SOC21_IH_CLIENTID_VMC:=18): 'SOC21_IH_CLIENTID_VMC', (SOC21_IH_CLIENTID_GRBM_CP:=20): 'SOC21_IH_CLIENTID_GRBM_CP', (SOC21_IH_CLIENTID_ROM_SMUIO:=22): 'SOC21_IH_CLIENTID_ROM_SMUIO', (SOC21_IH_CLIENTID_DF:=23): 'SOC21_IH_CLIENTID_DF', (SOC21_IH_CLIENTID_VPE:=24): 'SOC21_IH_CLIENTID_VPE', (SOC21_IH_CLIENTID_PWR:=25): 'SOC21_IH_CLIENTID_PWR', (SOC21_IH_CLIENTID_LSDMA:=26): 'SOC21_IH_CLIENTID_LSDMA', (SOC21_IH_CLIENTID_MP0:=30): 'SOC21_IH_CLIENTID_MP0', (SOC21_IH_CLIENTID_MP1:=31): 'SOC21_IH_CLIENTID_MP1', (SOC21_IH_CLIENTID_MAX:=32): 'SOC21_IH_CLIENTID_MAX'}
AMDGPU_VM_MAX_UPDATE_SIZE = 0x3FFFF
AMDGPU_PTE_VALID = (1 << 0)
AMDGPU_PTE_SYSTEM = (1 << 1)
AMDGPU_PTE_SNOOPED = (1 << 2)
AMDGPU_PTE_TMZ = (1 << 3)
AMDGPU_PTE_EXECUTABLE = (1 << 4)
AMDGPU_PTE_READABLE = (1 << 5)
AMDGPU_PTE_WRITEABLE = (1 << 6)
AMDGPU_PTE_FRAG = lambda x: ((x & 0x1f) << 7) # type: ignore
AMDGPU_PTE_PRT = (1 << 51)
AMDGPU_PDE_PTE = (1 << 54)
AMDGPU_PTE_LOG = (1 << 55)
AMDGPU_PTE_TF = (1 << 56)
AMDGPU_PTE_NOALLOC = (1 << 58)
AMDGPU_PDE_BFS = lambda a: (a << 59) # type: ignore
AMDGPU_VM_NORETRY_FLAGS = (AMDGPU_PTE_EXECUTABLE | AMDGPU_PDE_PTE | AMDGPU_PTE_TF)
AMDGPU_VM_NORETRY_FLAGS_TF = (AMDGPU_PTE_VALID | AMDGPU_PTE_SYSTEM | AMDGPU_PTE_PRT)
AMDGPU_PTE_MTYPE_VG10_SHIFT = lambda mtype: ((mtype) << 57) # type: ignore
AMDGPU_PTE_MTYPE_VG10_MASK = AMDGPU_PTE_MTYPE_VG10_SHIFT(3)
AMDGPU_PTE_MTYPE_VG10 = lambda flags,mtype: (((flags) & (~AMDGPU_PTE_MTYPE_VG10_MASK)) | AMDGPU_PTE_MTYPE_VG10_SHIFT(mtype)) # type: ignore
AMDGPU_MTYPE_NC = 0
AMDGPU_MTYPE_CC = 2
AMDGPU_PTE_MTYPE_NV10_SHIFT = lambda mtype: ((mtype) << 48) # type: ignore
AMDGPU_PTE_MTYPE_NV10_MASK = AMDGPU_PTE_MTYPE_NV10_SHIFT(7)
AMDGPU_PTE_MTYPE_NV10 = lambda flags,mtype: (((flags) & (~AMDGPU_PTE_MTYPE_NV10_MASK)) | AMDGPU_PTE_MTYPE_NV10_SHIFT(mtype)) # type: ignore
AMDGPU_PTE_PRT_GFX12 = (1 << 56)
AMDGPU_PTE_MTYPE_GFX12_SHIFT = lambda mtype: ((mtype) << 54) # type: ignore
AMDGPU_PTE_MTYPE_GFX12_MASK = AMDGPU_PTE_MTYPE_GFX12_SHIFT(3)
AMDGPU_PTE_MTYPE_GFX12 = lambda flags,mtype: (((flags) & (~AMDGPU_PTE_MTYPE_GFX12_MASK)) | AMDGPU_PTE_MTYPE_GFX12_SHIFT(mtype)) # type: ignore
AMDGPU_PTE_IS_PTE = (1 << 63)
AMDGPU_PDE_BFS_GFX12 = lambda a: (((a) & 0x1f) << 58) # type: ignore
AMDGPU_PDE_PTE_GFX12 = (1 << 63)
AMDGPU_VM_FAULT_STOP_NEVER = 0
AMDGPU_VM_FAULT_STOP_FIRST = 1
AMDGPU_VM_FAULT_STOP_ALWAYS = 2
AMDGPU_VM_RESERVED_VRAM = (8 << 20)
AMDGPU_MAX_VMHUBS = 13
AMDGPU_GFXHUB_START = 0
AMDGPU_MMHUB0_START = 8
AMDGPU_MMHUB1_START = 12
AMDGPU_GFXHUB = lambda x: (AMDGPU_GFXHUB_START + (x)) # type: ignore
AMDGPU_MMHUB0 = lambda x: (AMDGPU_MMHUB0_START + (x)) # type: ignore
AMDGPU_MMHUB1 = lambda x: (AMDGPU_MMHUB1_START + (x)) # type: ignore
AMDGPU_IS_GFXHUB = lambda x: ((x) >= AMDGPU_GFXHUB_START and (x) < AMDGPU_MMHUB0_START) # type: ignore
AMDGPU_IS_MMHUB0 = lambda x: ((x) >= AMDGPU_MMHUB0_START and (x) < AMDGPU_MMHUB1_START) # type: ignore
AMDGPU_IS_MMHUB1 = lambda x: ((x) >= AMDGPU_MMHUB1_START and (x) < AMDGPU_MAX_VMHUBS) # type: ignore
AMDGPU_VA_RESERVED_CSA_SIZE = (2 << 20)
AMDGPU_VA_RESERVED_SEQ64_SIZE = (2 << 20)
AMDGPU_VA_RESERVED_SEQ64_START = lambda adev: (AMDGPU_VA_RESERVED_CSA_START(adev) - AMDGPU_VA_RESERVED_SEQ64_SIZE) # type: ignore
AMDGPU_VA_RESERVED_TRAP_SIZE = (2 << 12)
AMDGPU_VA_RESERVED_TRAP_START = lambda adev: (AMDGPU_VA_RESERVED_SEQ64_START(adev) - AMDGPU_VA_RESERVED_TRAP_SIZE) # type: ignore
AMDGPU_VA_RESERVED_BOTTOM = (1 << 16)
AMDGPU_VA_RESERVED_TOP = (AMDGPU_VA_RESERVED_TRAP_SIZE + AMDGPU_VA_RESERVED_SEQ64_SIZE + AMDGPU_VA_RESERVED_CSA_SIZE)
AMDGPU_VM_USE_CPU_FOR_GFX = (1 << 0)
AMDGPU_VM_USE_CPU_FOR_COMPUTE = (1 << 1)
PSP_HEADER_SIZE = 256
BINARY_SIGNATURE = 0x28211407
DISCOVERY_TABLE_SIGNATURE = 0x53445049
GC_TABLE_ID = 0x4347
HARVEST_TABLE_SIGNATURE = 0x56524148
VCN_INFO_TABLE_ID = 0x004E4356
MALL_INFO_TABLE_ID = 0x4C4C414D
NPS_INFO_TABLE_ID = 0x0053504E
VCN_INFO_TABLE_MAX_NUM_INSTANCES = 4
NPS_INFO_TABLE_MAX_NUM_INSTANCES = 12
HWIP_MAX_INSTANCE = 44
HW_ID_MAX = 300
MP1_HWID = 1
MP2_HWID = 2
THM_HWID = 3
SMUIO_HWID = 4
FUSE_HWID = 5
CLKA_HWID = 6
PWR_HWID = 10
GC_HWID = 11
UVD_HWID = 12
VCN_HWID = UVD_HWID
AUDIO_AZ_HWID = 13
ACP_HWID = 14
DCI_HWID = 15
DMU_HWID = 271
DCO_HWID = 16
DIO_HWID = 272
XDMA_HWID = 17
DCEAZ_HWID = 18
DAZ_HWID = 274
SDPMUX_HWID = 19
NTB_HWID = 20
VPE_HWID = 21
IOHC_HWID = 24
L2IMU_HWID = 28
VCE_HWID = 32
MMHUB_HWID = 34
ATHUB_HWID = 35
DBGU_NBIO_HWID = 36
DFX_HWID = 37
DBGU0_HWID = 38
DBGU1_HWID = 39
OSSSYS_HWID = 40
HDP_HWID = 41
SDMA0_HWID = 42
SDMA1_HWID = 43
ISP_HWID = 44
DBGU_IO_HWID = 45
DF_HWID = 46
CLKB_HWID = 47
FCH_HWID = 48
DFX_DAP_HWID = 49
L1IMU_PCIE_HWID = 50
L1IMU_NBIF_HWID = 51
L1IMU_IOAGR_HWID = 52
L1IMU3_HWID = 53
L1IMU4_HWID = 54
L1IMU5_HWID = 55
L1IMU6_HWID = 56
L1IMU7_HWID = 57
L1IMU8_HWID = 58
L1IMU9_HWID = 59
L1IMU10_HWID = 60
L1IMU11_HWID = 61
L1IMU12_HWID = 62
L1IMU13_HWID = 63
L1IMU14_HWID = 64
L1IMU15_HWID = 65
WAFLC_HWID = 66
FCH_USB_PD_HWID = 67
SDMA2_HWID = 68
SDMA3_HWID = 69
PCIE_HWID = 70
PCS_HWID = 80
DDCL_HWID = 89
SST_HWID = 90
LSDMA_HWID = 91
IOAGR_HWID = 100
NBIF_HWID = 108
IOAPIC_HWID = 124
SYSTEMHUB_HWID = 128
NTBCCP_HWID = 144
UMC_HWID = 150
SATA_HWID = 168
USB_HWID = 170
CCXSEC_HWID = 176
XGMI_HWID = 200
XGBE_HWID = 216
MP0_HWID = 255
hw_id_map = {GC_HWIP:GC_HWID,HDP_HWIP:HDP_HWID,SDMA0_HWIP:SDMA0_HWID,SDMA1_HWIP:SDMA1_HWID,SDMA2_HWIP:SDMA2_HWID,SDMA3_HWIP:SDMA3_HWID,LSDMA_HWIP:LSDMA_HWID,MMHUB_HWIP:MMHUB_HWID,ATHUB_HWIP:ATHUB_HWID,NBIO_HWIP:NBIF_HWID,MP0_HWIP:MP0_HWID,MP1_HWIP:MP1_HWID,UVD_HWIP:UVD_HWID,VCE_HWIP:VCE_HWID,DF_HWIP:DF_HWID,DCE_HWIP:DMU_HWID,OSSSYS_HWIP:OSSSYS_HWID,SMUIO_HWIP:SMUIO_HWID,PWR_HWIP:PWR_HWID,NBIF_HWIP:NBIF_HWID,THM_HWIP:THM_HWID,CLK_HWIP:CLKA_HWID,UMC_HWIP:UMC_HWID,XGMI_HWIP:XGMI_HWID,DCI_HWIP:DCI_HWID,PCIE_HWIP:PCIE_HWID,VPE_HWIP:VPE_HWID,ISP_HWIP:ISP_HWID}
AMDGPU_SDMA0_UCODE_LOADED = 0x00000001
AMDGPU_SDMA1_UCODE_LOADED = 0x00000002
AMDGPU_CPCE_UCODE_LOADED = 0x00000004
AMDGPU_CPPFP_UCODE_LOADED = 0x00000008
AMDGPU_CPME_UCODE_LOADED = 0x00000010
AMDGPU_CPMEC1_UCODE_LOADED = 0x00000020
AMDGPU_CPMEC2_UCODE_LOADED = 0x00000040
AMDGPU_CPRLC_UCODE_LOADED = 0x00000100
PSP_GFX_CMD_BUF_VERSION = 0x00000001
GFX_CMD_STATUS_MASK = 0x0000FFFF
GFX_CMD_ID_MASK = 0x000F0000
GFX_CMD_RESERVED_MASK = 0x7FF00000
GFX_CMD_RESPONSE_MASK = 0x80000000
C2PMSG_CMD_GFX_USB_PD_FW_VER = 0x2000000
GFX_FLAG_RESPONSE = 0x80000000
GFX_BUF_MAX_DESC = 64
FRAME_TYPE_DESTROY = 1
PSP_ERR_UNKNOWN_COMMAND = 0x00000100
PSP_FENCE_BUFFER_SIZE = 0x1000
PSP_CMD_BUFFER_SIZE = 0x1000
PSP_1_MEG = 0x100000
PSP_TMR_ALIGNMENT = 0x100000
PSP_FW_NAME_LEN = 0x24
AMDGPU_XGMI_MAX_CONNECTED_NODES = 64
MEM_TRAIN_SYSTEM_SIGNATURE = 0x54534942
GDDR6_MEM_TRAINING_DATA_SIZE_IN_BYTES = 0x1000
GDDR6_MEM_TRAINING_OFFSET = 0x8000
BIST_MEM_TRAINING_ENCROACHED_SIZE = 0x2000000
PSP_RUNTIME_DB_SIZE_IN_BYTES = 0x10000
PSP_RUNTIME_DB_OFFSET = 0x100000
PSP_RUNTIME_DB_COOKIE_ID = 0x0ed5
PSP_RUNTIME_DB_VER_1 = 0x0100
PSP_RUNTIME_DB_DIAG_ENTRY_MAX_COUNT = 0x40
AMDGPU_MAX_IRQ_SRC_ID = 0x100
AMDGPU_MAX_IRQ_CLIENT_ID = 0x100
AMDGPU_IRQ_CLIENTID_LEGACY = 0
AMDGPU_IRQ_CLIENTID_MAX = SOC15_IH_CLIENTID_MAX
AMDGPU_IRQ_SRC_DATA_MAX_SIZE_DW = 4
SOC15_INTSRC_CP_END_OF_PIPE = 181
SOC15_INTSRC_CP_BAD_OPCODE = 183
SOC15_INTSRC_SQ_INTERRUPT_MSG = 239
SOC15_INTSRC_VMC_FAULT = 0
SOC15_INTSRC_VMC_UTCL2_POISON = 1
SOC15_INTSRC_SDMA_TRAP = 224
SOC15_INTSRC_SDMA_ECC = 220
SOC21_INTSRC_SDMA_TRAP = 49
SOC21_INTSRC_SDMA_ECC = 62
SOC15_CLIENT_ID_FROM_IH_ENTRY = lambda entry: ((entry[0]) & 0xff) # type: ignore
SOC15_SOURCE_ID_FROM_IH_ENTRY = lambda entry: ((entry[0]) >> 8 & 0xff) # type: ignore
SOC15_RING_ID_FROM_IH_ENTRY = lambda entry: ((entry[0]) >> 16 & 0xff) # type: ignore
SOC15_VMID_FROM_IH_ENTRY = lambda entry: ((entry[0]) >> 24 & 0xf) # type: ignore
SOC15_VMID_TYPE_FROM_IH_ENTRY = lambda entry: ((entry[0]) >> 31 & 0x1) # type: ignore
SOC15_PASID_FROM_IH_ENTRY = lambda entry: ((entry[3]) & 0xffff) # type: ignore
SOC15_NODEID_FROM_IH_ENTRY = lambda entry: ((entry[3]) >> 16 & 0xff) # type: ignore
SOC15_CONTEXT_ID0_FROM_IH_ENTRY = lambda entry: ((entry[4])) # type: ignore
SOC15_CONTEXT_ID1_FROM_IH_ENTRY = lambda entry: ((entry[5])) # type: ignore
SOC15_CONTEXT_ID2_FROM_IH_ENTRY = lambda entry: ((entry[6])) # type: ignore
SOC15_CONTEXT_ID3_FROM_IH_ENTRY = lambda entry: ((entry[7])) # type: ignore
GFX_9_0__SRCID__CP_RB_INTERRUPT_PKT = 176
GFX_9_0__SRCID__CP_IB1_INTERRUPT_PKT = 177
GFX_9_0__SRCID__CP_IB2_INTERRUPT_PKT = 178
GFX_9_0__SRCID__CP_PM4_PKT_RSVD_BIT_ERROR = 180
GFX_9_0__SRCID__CP_EOP_INTERRUPT = 181
GFX_9_0__SRCID__CP_BAD_OPCODE_ERROR = 183
GFX_9_0__SRCID__CP_PRIV_REG_FAULT = 184
GFX_9_0__SRCID__CP_PRIV_INSTR_FAULT = 185
GFX_9_0__SRCID__CP_WAIT_MEM_SEM_FAULT = 186
GFX_9_0__SRCID__CP_CTX_EMPTY_INTERRUPT = 187
GFX_9_0__SRCID__CP_CTX_BUSY_INTERRUPT = 188
GFX_9_0__SRCID__CP_ME_WAIT_REG_MEM_POLL_TIMEOUT = 192
GFX_9_0__SRCID__CP_SIG_INCOMPLETE = 193
GFX_9_0__SRCID__CP_PREEMPT_ACK = 194
GFX_9_0__SRCID__CP_GPF = 195
GFX_9_0__SRCID__CP_GDS_ALLOC_ERROR = 196
GFX_9_0__SRCID__CP_ECC_ERROR = 197
GFX_9_0__SRCID__CP_COMPUTE_QUERY_STATUS = 199
GFX_9_0__SRCID__CP_VM_DOORBELL = 200
GFX_9_0__SRCID__CP_FUE_ERROR = 201
GFX_9_0__SRCID__RLC_STRM_PERF_MONITOR_INTERRUPT = 202
GFX_9_0__SRCID__GRBM_RD_TIMEOUT_ERROR = 232
GFX_9_0__SRCID__GRBM_REG_GUI_IDLE = 233
GFX_9_0__SRCID__SQ_INTERRUPT_ID = 239
GFX_11_0_0__SRCID__UTCL2_FAULT = 0
GFX_11_0_0__SRCID__UTCL2_DATA_POISONING = 1
GFX_11_0_0__SRCID__MEM_ACCES_MON = 10
GFX_11_0_0__SRCID__SDMA_ATOMIC_RTN_DONE = 48
GFX_11_0_0__SRCID__SDMA_TRAP = 49
GFX_11_0_0__SRCID__SDMA_SRBMWRITE = 50
GFX_11_0_0__SRCID__SDMA_CTXEMPTY = 51
GFX_11_0_0__SRCID__SDMA_PREEMPT = 52
GFX_11_0_0__SRCID__SDMA_IB_PREEMPT = 53
GFX_11_0_0__SRCID__SDMA_DOORBELL_INVALID = 54
GFX_11_0_0__SRCID__SDMA_QUEUE_HANG = 55
GFX_11_0_0__SRCID__SDMA_ATOMIC_TIMEOUT = 56
GFX_11_0_0__SRCID__SDMA_POLL_TIMEOUT = 57
GFX_11_0_0__SRCID__SDMA_PAGE_TIMEOUT = 58
GFX_11_0_0__SRCID__SDMA_PAGE_NULL = 59
GFX_11_0_0__SRCID__SDMA_PAGE_FAULT = 60
GFX_11_0_0__SRCID__SDMA_VM_HOLE = 61
GFX_11_0_0__SRCID__SDMA_ECC = 62
GFX_11_0_0__SRCID__SDMA_FROZEN = 63
GFX_11_0_0__SRCID__SDMA_SRAM_ECC = 64
GFX_11_0_0__SRCID__SDMA_SEM_INCOMPLETE_TIMEOUT = 65
GFX_11_0_0__SRCID__SDMA_SEM_WAIT_FAIL_TIMEOUT = 66
GFX_11_0_0__SRCID__SDMA_FENCE = 67
GFX_11_0_0__SRCID__RLC_GC_FED_INTERRUPT = 128
GFX_11_0_0__SRCID__CP_GENERIC_INT = 177
GFX_11_0_0__SRCID__CP_PM4_PKT_RSVD_BIT_ERROR = 180
GFX_11_0_0__SRCID__CP_EOP_INTERRUPT = 181
GFX_11_0_0__SRCID__CP_BAD_OPCODE_ERROR = 183
GFX_11_0_0__SRCID__CP_PRIV_REG_FAULT = 184
GFX_11_0_0__SRCID__CP_PRIV_INSTR_FAULT = 185
GFX_11_0_0__SRCID__CP_WAIT_MEM_SEM_FAULT = 186
GFX_11_0_0__SRCID__CP_CTX_EMPTY_INTERRUPT = 187
GFX_11_0_0__SRCID__CP_CTX_BUSY_INTERRUPT = 188
GFX_11_0_0__SRCID__CP_ME_WAIT_REG_MEM_POLL_TIMEOUT = 192
GFX_11_0_0__SRCID__CP_SIG_INCOMPLETE = 193
GFX_11_0_0__SRCID__CP_PREEMPT_ACK = 194
GFX_11_0_0__SRCID__CP_GPF = 195
GFX_11_0_0__SRCID__CP_GDS_ALLOC_ERROR = 196
GFX_11_0_0__SRCID__CP_ECC_ERROR = 197
GFX_11_0_0__SRCID__CP_COMPUTE_QUERY_STATUS = 199
GFX_11_0_0__SRCID__CP_VM_DOORBELL = 200
GFX_11_0_0__SRCID__CP_FUE_ERROR = 201
GFX_11_0_0__SRCID__RLC_STRM_PERF_MONITOR_INTERRUPT = 202
GFX_11_0_0__SRCID__GRBM_RD_TIMEOUT_ERROR = 232
GFX_11_0_0__SRCID__GRBM_REG_GUI_IDLE = 233
GFX_11_0_0__SRCID__SQ_INTERRUPT_ID = 239
GFX_12_0_0__SRCID__UTCL2_FAULT = 0
GFX_12_0_0__SRCID__UTCL2_DATA_POISONING = 1
GFX_12_0_0__SRCID__MEM_ACCES_MON = 10
GFX_12_0_0__SRCID__SDMA_ATOMIC_RTN_DONE = 48
GFX_12_0_0__SRCID__SDMA_TRAP = 49
GFX_12_0_0__SRCID__SDMA_SRBMWRITE = 50
GFX_12_0_0__SRCID__SDMA_CTXEMPTY = 51
GFX_12_0_0__SRCID__SDMA_PREEMPT = 52
GFX_12_0_0__SRCID__SDMA_IB_PREEMPT = 53
GFX_12_0_0__SRCID__SDMA_DOORBELL_INVALID = 54
GFX_12_0_0__SRCID__SDMA_QUEUE_HANG = 55
GFX_12_0_0__SRCID__SDMA_ATOMIC_TIMEOUT = 56
GFX_12_0_0__SRCID__SDMA_POLL_TIMEOUT = 57
GFX_12_0_0__SRCID__SDMA_PAGE_TIMEOUT = 58
GFX_12_0_0__SRCID__SDMA_PAGE_NULL = 59
GFX_12_0_0__SRCID__SDMA_PAGE_FAULT = 60
GFX_12_0_0__SRCID__SDMA_VM_HOLE = 61
GFX_12_0_0__SRCID__SDMA_ECC = 62
GFX_12_0_0__SRCID__SDMA_FROZEN = 63
GFX_12_0_0__SRCID__SDMA_SRAM_ECC = 64
GFX_12_0_0__SRCID__SDMA_SEM_INCOMPLETE_TIMEOUT = 65
GFX_12_0_0__SRCID__SDMA_SEM_WAIT_FAIL_TIMEOUT = 66
GFX_12_0_0__SRCID__SDMA_FENCE = 70
GFX_12_0_0__SRCID__RLC_GC_FED_INTERRUPT = 128
GFX_12_0_0__SRCID__CP_GENERIC_INT = 177
GFX_12_0_0__SRCID__CP_PM4_PKT_RSVD_BIT_ERROR = 180
GFX_12_0_0__SRCID__CP_EOP_INTERRUPT = 181
GFX_12_0_0__SRCID__CP_BAD_OPCODE_ERROR = 183
GFX_12_0_0__SRCID__CP_PRIV_REG_FAULT = 184
GFX_12_0_0__SRCID__CP_PRIV_INSTR_FAULT = 185
GFX_12_0_0__SRCID__CP_WAIT_MEM_SEM_FAULT = 186
GFX_12_0_0__SRCID__CP_CTX_EMPTY_INTERRUPT = 187
GFX_12_0_0__SRCID__CP_CTX_BUSY_INTERRUPT = 188
GFX_12_0_0__SRCID__CP_ME_WAIT_REG_MEM_POLL_TIMEOUT = 192
GFX_12_0_0__SRCID__CP_SIG_INCOMPLETE = 193
GFX_12_0_0__SRCID__CP_PREEMPT_ACK = 194
GFX_12_0_0__SRCID__CP_GPF = 195
GFX_12_0_0__SRCID__CP_GDS_ALLOC_ERROR = 196
GFX_12_0_0__SRCID__CP_ECC_ERROR = 197
GFX_12_0_0__SRCID__CP_COMPUTE_QUERY_STATUS = 199
GFX_12_0_0__SRCID__CP_VM_DOORBELL = 200
GFX_12_0_0__SRCID__CP_FUE_ERROR = 201
GFX_12_0_0__SRCID__RLC_STRM_PERF_MONITOR_INTERRUPT = 202
GFX_12_0_0__SRCID__GRBM_RD_TIMEOUT_ERROR = 232
GFX_12_0_0__SRCID__GRBM_REG_GUI_IDLE = 233
GFX_12_0_0__SRCID__SQ_INTERRUPT_ID = 239
SDMA0_4_0__SRCID__SDMA_ATOMIC_RTN_DONE = 217
SDMA0_4_0__SRCID__SDMA_ATOMIC_TIMEOUT = 218
SDMA0_4_0__SRCID__SDMA_IB_PREEMPT = 219
SDMA0_4_0__SRCID__SDMA_ECC = 220
SDMA0_4_0__SRCID__SDMA_PAGE_FAULT = 221
SDMA0_4_0__SRCID__SDMA_PAGE_NULL = 222
SDMA0_4_0__SRCID__SDMA_XNACK = 223
SDMA0_4_0__SRCID__SDMA_TRAP = 224
SDMA0_4_0__SRCID__SDMA_SEM_INCOMPLETE_TIMEOUT = 225
SDMA0_4_0__SRCID__SDMA_SEM_WAIT_FAIL_TIMEOUT = 226
SDMA0_4_0__SRCID__SDMA_SRAM_ECC = 228
SDMA0_4_0__SRCID__SDMA_PREEMPT = 240
SDMA0_4_0__SRCID__SDMA_VM_HOLE = 242
SDMA0_4_0__SRCID__SDMA_CTXEMPTY = 243
SDMA0_4_0__SRCID__SDMA_DOORBELL_INVALID = 244
SDMA0_4_0__SRCID__SDMA_FROZEN = 245
SDMA0_4_0__SRCID__SDMA_POLL_TIMEOUT = 246
SDMA0_4_0__SRCID__SDMA_SRBMWRITE = 247
SDMA0_5_0__SRCID__SDMA_ATOMIC_RTN_DONE = 217
SDMA0_5_0__SRCID__SDMA_ATOMIC_TIMEOUT = 218
SDMA0_5_0__SRCID__SDMA_IB_PREEMPT = 219
SDMA0_5_0__SRCID__SDMA_ECC = 220
SDMA0_5_0__SRCID__SDMA_PAGE_FAULT = 221
SDMA0_5_0__SRCID__SDMA_PAGE_NULL = 222
SDMA0_5_0__SRCID__SDMA_XNACK = 223
SDMA0_5_0__SRCID__SDMA_TRAP = 224
SDMA0_5_0__SRCID__SDMA_SEM_INCOMPLETE_TIMEOUT = 225
SDMA0_5_0__SRCID__SDMA_SEM_WAIT_FAIL_TIMEOUT = 226
SDMA0_5_0__SRCID__SDMA_SRAM_ECC = 228
SDMA0_5_0__SRCID__SDMA_PREEMPT = 240
SDMA0_5_0__SRCID__SDMA_VM_HOLE = 242
SDMA0_5_0__SRCID__SDMA_CTXEMPTY = 243
SDMA0_5_0__SRCID__SDMA_DOORBELL_INVALID = 244
SDMA0_5_0__SRCID__SDMA_FROZEN = 245
SDMA0_5_0__SRCID__SDMA_POLL_TIMEOUT = 246
SDMA0_5_0__SRCID__SDMA_SRBMWRITE = 247