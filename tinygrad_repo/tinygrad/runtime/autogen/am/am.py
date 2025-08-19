# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-include', 'stdint.h']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





V11_STRUCTS_H_ = True # macro
uint32_t = True # macro
uint8_t = True # macro
uint16_t = True # macro
uint64_t = True # macro
class struct_v11_gfx_mqd(Structure):
    pass

struct_v11_gfx_mqd._pack_ = 1 # source:False
struct_v11_gfx_mqd._fields_ = [
    ('shadow_base_lo', ctypes.c_uint32),
    ('shadow_base_hi', ctypes.c_uint32),
    ('gds_bkup_base_lo', ctypes.c_uint32),
    ('gds_bkup_base_hi', ctypes.c_uint32),
    ('fw_work_area_base_lo', ctypes.c_uint32),
    ('fw_work_area_base_hi', ctypes.c_uint32),
    ('shadow_initialized', ctypes.c_uint32),
    ('ib_vmid', ctypes.c_uint32),
    ('reserved_8', ctypes.c_uint32),
    ('reserved_9', ctypes.c_uint32),
    ('reserved_10', ctypes.c_uint32),
    ('reserved_11', ctypes.c_uint32),
    ('reserved_12', ctypes.c_uint32),
    ('reserved_13', ctypes.c_uint32),
    ('reserved_14', ctypes.c_uint32),
    ('reserved_15', ctypes.c_uint32),
    ('reserved_16', ctypes.c_uint32),
    ('reserved_17', ctypes.c_uint32),
    ('reserved_18', ctypes.c_uint32),
    ('reserved_19', ctypes.c_uint32),
    ('reserved_20', ctypes.c_uint32),
    ('reserved_21', ctypes.c_uint32),
    ('reserved_22', ctypes.c_uint32),
    ('reserved_23', ctypes.c_uint32),
    ('reserved_24', ctypes.c_uint32),
    ('reserved_25', ctypes.c_uint32),
    ('reserved_26', ctypes.c_uint32),
    ('reserved_27', ctypes.c_uint32),
    ('reserved_28', ctypes.c_uint32),
    ('reserved_29', ctypes.c_uint32),
    ('reserved_30', ctypes.c_uint32),
    ('reserved_31', ctypes.c_uint32),
    ('reserved_32', ctypes.c_uint32),
    ('reserved_33', ctypes.c_uint32),
    ('reserved_34', ctypes.c_uint32),
    ('reserved_35', ctypes.c_uint32),
    ('reserved_36', ctypes.c_uint32),
    ('reserved_37', ctypes.c_uint32),
    ('reserved_38', ctypes.c_uint32),
    ('reserved_39', ctypes.c_uint32),
    ('reserved_40', ctypes.c_uint32),
    ('reserved_41', ctypes.c_uint32),
    ('reserved_42', ctypes.c_uint32),
    ('reserved_43', ctypes.c_uint32),
    ('reserved_44', ctypes.c_uint32),
    ('reserved_45', ctypes.c_uint32),
    ('reserved_46', ctypes.c_uint32),
    ('reserved_47', ctypes.c_uint32),
    ('reserved_48', ctypes.c_uint32),
    ('reserved_49', ctypes.c_uint32),
    ('reserved_50', ctypes.c_uint32),
    ('reserved_51', ctypes.c_uint32),
    ('reserved_52', ctypes.c_uint32),
    ('reserved_53', ctypes.c_uint32),
    ('reserved_54', ctypes.c_uint32),
    ('reserved_55', ctypes.c_uint32),
    ('reserved_56', ctypes.c_uint32),
    ('reserved_57', ctypes.c_uint32),
    ('reserved_58', ctypes.c_uint32),
    ('reserved_59', ctypes.c_uint32),
    ('reserved_60', ctypes.c_uint32),
    ('reserved_61', ctypes.c_uint32),
    ('reserved_62', ctypes.c_uint32),
    ('reserved_63', ctypes.c_uint32),
    ('reserved_64', ctypes.c_uint32),
    ('reserved_65', ctypes.c_uint32),
    ('reserved_66', ctypes.c_uint32),
    ('reserved_67', ctypes.c_uint32),
    ('reserved_68', ctypes.c_uint32),
    ('reserved_69', ctypes.c_uint32),
    ('reserved_70', ctypes.c_uint32),
    ('reserved_71', ctypes.c_uint32),
    ('reserved_72', ctypes.c_uint32),
    ('reserved_73', ctypes.c_uint32),
    ('reserved_74', ctypes.c_uint32),
    ('reserved_75', ctypes.c_uint32),
    ('reserved_76', ctypes.c_uint32),
    ('reserved_77', ctypes.c_uint32),
    ('reserved_78', ctypes.c_uint32),
    ('reserved_79', ctypes.c_uint32),
    ('reserved_80', ctypes.c_uint32),
    ('reserved_81', ctypes.c_uint32),
    ('reserved_82', ctypes.c_uint32),
    ('reserved_83', ctypes.c_uint32),
    ('checksum_lo', ctypes.c_uint32),
    ('checksum_hi', ctypes.c_uint32),
    ('cp_mqd_query_time_lo', ctypes.c_uint32),
    ('cp_mqd_query_time_hi', ctypes.c_uint32),
    ('reserved_88', ctypes.c_uint32),
    ('reserved_89', ctypes.c_uint32),
    ('reserved_90', ctypes.c_uint32),
    ('reserved_91', ctypes.c_uint32),
    ('cp_mqd_query_wave_count', ctypes.c_uint32),
    ('cp_mqd_query_gfx_hqd_rptr', ctypes.c_uint32),
    ('cp_mqd_query_gfx_hqd_wptr', ctypes.c_uint32),
    ('cp_mqd_query_gfx_hqd_offset', ctypes.c_uint32),
    ('reserved_96', ctypes.c_uint32),
    ('reserved_97', ctypes.c_uint32),
    ('reserved_98', ctypes.c_uint32),
    ('reserved_99', ctypes.c_uint32),
    ('reserved_100', ctypes.c_uint32),
    ('reserved_101', ctypes.c_uint32),
    ('reserved_102', ctypes.c_uint32),
    ('reserved_103', ctypes.c_uint32),
    ('control_buf_addr_lo', ctypes.c_uint32),
    ('control_buf_addr_hi', ctypes.c_uint32),
    ('disable_queue', ctypes.c_uint32),
    ('reserved_107', ctypes.c_uint32),
    ('reserved_108', ctypes.c_uint32),
    ('reserved_109', ctypes.c_uint32),
    ('reserved_110', ctypes.c_uint32),
    ('reserved_111', ctypes.c_uint32),
    ('reserved_112', ctypes.c_uint32),
    ('reserved_113', ctypes.c_uint32),
    ('reserved_114', ctypes.c_uint32),
    ('reserved_115', ctypes.c_uint32),
    ('reserved_116', ctypes.c_uint32),
    ('reserved_117', ctypes.c_uint32),
    ('reserved_118', ctypes.c_uint32),
    ('reserved_119', ctypes.c_uint32),
    ('reserved_120', ctypes.c_uint32),
    ('reserved_121', ctypes.c_uint32),
    ('reserved_122', ctypes.c_uint32),
    ('reserved_123', ctypes.c_uint32),
    ('reserved_124', ctypes.c_uint32),
    ('reserved_125', ctypes.c_uint32),
    ('reserved_126', ctypes.c_uint32),
    ('reserved_127', ctypes.c_uint32),
    ('cp_mqd_base_addr', ctypes.c_uint32),
    ('cp_mqd_base_addr_hi', ctypes.c_uint32),
    ('cp_gfx_hqd_active', ctypes.c_uint32),
    ('cp_gfx_hqd_vmid', ctypes.c_uint32),
    ('reserved_131', ctypes.c_uint32),
    ('reserved_132', ctypes.c_uint32),
    ('cp_gfx_hqd_queue_priority', ctypes.c_uint32),
    ('cp_gfx_hqd_quantum', ctypes.c_uint32),
    ('cp_gfx_hqd_base', ctypes.c_uint32),
    ('cp_gfx_hqd_base_hi', ctypes.c_uint32),
    ('cp_gfx_hqd_rptr', ctypes.c_uint32),
    ('cp_gfx_hqd_rptr_addr', ctypes.c_uint32),
    ('cp_gfx_hqd_rptr_addr_hi', ctypes.c_uint32),
    ('cp_rb_wptr_poll_addr_lo', ctypes.c_uint32),
    ('cp_rb_wptr_poll_addr_hi', ctypes.c_uint32),
    ('cp_rb_doorbell_control', ctypes.c_uint32),
    ('cp_gfx_hqd_offset', ctypes.c_uint32),
    ('cp_gfx_hqd_cntl', ctypes.c_uint32),
    ('reserved_146', ctypes.c_uint32),
    ('reserved_147', ctypes.c_uint32),
    ('cp_gfx_hqd_csmd_rptr', ctypes.c_uint32),
    ('cp_gfx_hqd_wptr', ctypes.c_uint32),
    ('cp_gfx_hqd_wptr_hi', ctypes.c_uint32),
    ('reserved_151', ctypes.c_uint32),
    ('reserved_152', ctypes.c_uint32),
    ('reserved_153', ctypes.c_uint32),
    ('reserved_154', ctypes.c_uint32),
    ('reserved_155', ctypes.c_uint32),
    ('cp_gfx_hqd_mapped', ctypes.c_uint32),
    ('cp_gfx_hqd_que_mgr_control', ctypes.c_uint32),
    ('reserved_158', ctypes.c_uint32),
    ('reserved_159', ctypes.c_uint32),
    ('cp_gfx_hqd_hq_status0', ctypes.c_uint32),
    ('cp_gfx_hqd_hq_control0', ctypes.c_uint32),
    ('cp_gfx_mqd_control', ctypes.c_uint32),
    ('reserved_163', ctypes.c_uint32),
    ('reserved_164', ctypes.c_uint32),
    ('reserved_165', ctypes.c_uint32),
    ('reserved_166', ctypes.c_uint32),
    ('reserved_167', ctypes.c_uint32),
    ('reserved_168', ctypes.c_uint32),
    ('reserved_169', ctypes.c_uint32),
    ('cp_num_prim_needed_count0_lo', ctypes.c_uint32),
    ('cp_num_prim_needed_count0_hi', ctypes.c_uint32),
    ('cp_num_prim_needed_count1_lo', ctypes.c_uint32),
    ('cp_num_prim_needed_count1_hi', ctypes.c_uint32),
    ('cp_num_prim_needed_count2_lo', ctypes.c_uint32),
    ('cp_num_prim_needed_count2_hi', ctypes.c_uint32),
    ('cp_num_prim_needed_count3_lo', ctypes.c_uint32),
    ('cp_num_prim_needed_count3_hi', ctypes.c_uint32),
    ('cp_num_prim_written_count0_lo', ctypes.c_uint32),
    ('cp_num_prim_written_count0_hi', ctypes.c_uint32),
    ('cp_num_prim_written_count1_lo', ctypes.c_uint32),
    ('cp_num_prim_written_count1_hi', ctypes.c_uint32),
    ('cp_num_prim_written_count2_lo', ctypes.c_uint32),
    ('cp_num_prim_written_count2_hi', ctypes.c_uint32),
    ('cp_num_prim_written_count3_lo', ctypes.c_uint32),
    ('cp_num_prim_written_count3_hi', ctypes.c_uint32),
    ('reserved_186', ctypes.c_uint32),
    ('reserved_187', ctypes.c_uint32),
    ('reserved_188', ctypes.c_uint32),
    ('reserved_189', ctypes.c_uint32),
    ('mp1_smn_fps_cnt', ctypes.c_uint32),
    ('sq_thread_trace_buf0_base', ctypes.c_uint32),
    ('sq_thread_trace_buf0_size', ctypes.c_uint32),
    ('sq_thread_trace_buf1_base', ctypes.c_uint32),
    ('sq_thread_trace_buf1_size', ctypes.c_uint32),
    ('sq_thread_trace_wptr', ctypes.c_uint32),
    ('sq_thread_trace_mask', ctypes.c_uint32),
    ('sq_thread_trace_token_mask', ctypes.c_uint32),
    ('sq_thread_trace_ctrl', ctypes.c_uint32),
    ('sq_thread_trace_status', ctypes.c_uint32),
    ('sq_thread_trace_dropped_cntr', ctypes.c_uint32),
    ('sq_thread_trace_finish_done_debug', ctypes.c_uint32),
    ('sq_thread_trace_gfx_draw_cntr', ctypes.c_uint32),
    ('sq_thread_trace_gfx_marker_cntr', ctypes.c_uint32),
    ('sq_thread_trace_hp3d_draw_cntr', ctypes.c_uint32),
    ('sq_thread_trace_hp3d_marker_cntr', ctypes.c_uint32),
    ('reserved_206', ctypes.c_uint32),
    ('reserved_207', ctypes.c_uint32),
    ('cp_sc_psinvoc_count0_lo', ctypes.c_uint32),
    ('cp_sc_psinvoc_count0_hi', ctypes.c_uint32),
    ('cp_pa_cprim_count_lo', ctypes.c_uint32),
    ('cp_pa_cprim_count_hi', ctypes.c_uint32),
    ('cp_pa_cinvoc_count_lo', ctypes.c_uint32),
    ('cp_pa_cinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_vsinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_vsinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_gsinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_gsinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_gsprim_count_lo', ctypes.c_uint32),
    ('cp_vgt_gsprim_count_hi', ctypes.c_uint32),
    ('cp_vgt_iaprim_count_lo', ctypes.c_uint32),
    ('cp_vgt_iaprim_count_hi', ctypes.c_uint32),
    ('cp_vgt_iavert_count_lo', ctypes.c_uint32),
    ('cp_vgt_iavert_count_hi', ctypes.c_uint32),
    ('cp_vgt_hsinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_hsinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_dsinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_dsinvoc_count_hi', ctypes.c_uint32),
    ('cp_vgt_csinvoc_count_lo', ctypes.c_uint32),
    ('cp_vgt_csinvoc_count_hi', ctypes.c_uint32),
    ('reserved_230', ctypes.c_uint32),
    ('reserved_231', ctypes.c_uint32),
    ('reserved_232', ctypes.c_uint32),
    ('reserved_233', ctypes.c_uint32),
    ('reserved_234', ctypes.c_uint32),
    ('reserved_235', ctypes.c_uint32),
    ('reserved_236', ctypes.c_uint32),
    ('reserved_237', ctypes.c_uint32),
    ('reserved_238', ctypes.c_uint32),
    ('reserved_239', ctypes.c_uint32),
    ('reserved_240', ctypes.c_uint32),
    ('reserved_241', ctypes.c_uint32),
    ('reserved_242', ctypes.c_uint32),
    ('reserved_243', ctypes.c_uint32),
    ('reserved_244', ctypes.c_uint32),
    ('reserved_245', ctypes.c_uint32),
    ('reserved_246', ctypes.c_uint32),
    ('reserved_247', ctypes.c_uint32),
    ('reserved_248', ctypes.c_uint32),
    ('reserved_249', ctypes.c_uint32),
    ('reserved_250', ctypes.c_uint32),
    ('reserved_251', ctypes.c_uint32),
    ('reserved_252', ctypes.c_uint32),
    ('reserved_253', ctypes.c_uint32),
    ('reserved_254', ctypes.c_uint32),
    ('reserved_255', ctypes.c_uint32),
    ('reserved_256', ctypes.c_uint32),
    ('reserved_257', ctypes.c_uint32),
    ('reserved_258', ctypes.c_uint32),
    ('reserved_259', ctypes.c_uint32),
    ('reserved_260', ctypes.c_uint32),
    ('reserved_261', ctypes.c_uint32),
    ('reserved_262', ctypes.c_uint32),
    ('reserved_263', ctypes.c_uint32),
    ('reserved_264', ctypes.c_uint32),
    ('reserved_265', ctypes.c_uint32),
    ('reserved_266', ctypes.c_uint32),
    ('reserved_267', ctypes.c_uint32),
    ('vgt_strmout_buffer_filled_size_0', ctypes.c_uint32),
    ('vgt_strmout_buffer_filled_size_1', ctypes.c_uint32),
    ('vgt_strmout_buffer_filled_size_2', ctypes.c_uint32),
    ('vgt_strmout_buffer_filled_size_3', ctypes.c_uint32),
    ('reserved_272', ctypes.c_uint32),
    ('reserved_273', ctypes.c_uint32),
    ('reserved_274', ctypes.c_uint32),
    ('reserved_275', ctypes.c_uint32),
    ('vgt_dma_max_size', ctypes.c_uint32),
    ('vgt_dma_num_instances', ctypes.c_uint32),
    ('reserved_278', ctypes.c_uint32),
    ('reserved_279', ctypes.c_uint32),
    ('reserved_280', ctypes.c_uint32),
    ('reserved_281', ctypes.c_uint32),
    ('reserved_282', ctypes.c_uint32),
    ('reserved_283', ctypes.c_uint32),
    ('reserved_284', ctypes.c_uint32),
    ('reserved_285', ctypes.c_uint32),
    ('reserved_286', ctypes.c_uint32),
    ('reserved_287', ctypes.c_uint32),
    ('it_set_base_ib_addr_lo', ctypes.c_uint32),
    ('it_set_base_ib_addr_hi', ctypes.c_uint32),
    ('reserved_290', ctypes.c_uint32),
    ('reserved_291', ctypes.c_uint32),
    ('reserved_292', ctypes.c_uint32),
    ('reserved_293', ctypes.c_uint32),
    ('reserved_294', ctypes.c_uint32),
    ('reserved_295', ctypes.c_uint32),
    ('reserved_296', ctypes.c_uint32),
    ('reserved_297', ctypes.c_uint32),
    ('reserved_298', ctypes.c_uint32),
    ('reserved_299', ctypes.c_uint32),
    ('reserved_300', ctypes.c_uint32),
    ('reserved_301', ctypes.c_uint32),
    ('reserved_302', ctypes.c_uint32),
    ('reserved_303', ctypes.c_uint32),
    ('reserved_304', ctypes.c_uint32),
    ('reserved_305', ctypes.c_uint32),
    ('reserved_306', ctypes.c_uint32),
    ('reserved_307', ctypes.c_uint32),
    ('reserved_308', ctypes.c_uint32),
    ('reserved_309', ctypes.c_uint32),
    ('reserved_310', ctypes.c_uint32),
    ('reserved_311', ctypes.c_uint32),
    ('reserved_312', ctypes.c_uint32),
    ('reserved_313', ctypes.c_uint32),
    ('reserved_314', ctypes.c_uint32),
    ('reserved_315', ctypes.c_uint32),
    ('reserved_316', ctypes.c_uint32),
    ('reserved_317', ctypes.c_uint32),
    ('reserved_318', ctypes.c_uint32),
    ('reserved_319', ctypes.c_uint32),
    ('reserved_320', ctypes.c_uint32),
    ('reserved_321', ctypes.c_uint32),
    ('reserved_322', ctypes.c_uint32),
    ('reserved_323', ctypes.c_uint32),
    ('reserved_324', ctypes.c_uint32),
    ('reserved_325', ctypes.c_uint32),
    ('reserved_326', ctypes.c_uint32),
    ('reserved_327', ctypes.c_uint32),
    ('reserved_328', ctypes.c_uint32),
    ('reserved_329', ctypes.c_uint32),
    ('reserved_330', ctypes.c_uint32),
    ('reserved_331', ctypes.c_uint32),
    ('reserved_332', ctypes.c_uint32),
    ('reserved_333', ctypes.c_uint32),
    ('reserved_334', ctypes.c_uint32),
    ('reserved_335', ctypes.c_uint32),
    ('reserved_336', ctypes.c_uint32),
    ('reserved_337', ctypes.c_uint32),
    ('reserved_338', ctypes.c_uint32),
    ('reserved_339', ctypes.c_uint32),
    ('reserved_340', ctypes.c_uint32),
    ('reserved_341', ctypes.c_uint32),
    ('reserved_342', ctypes.c_uint32),
    ('reserved_343', ctypes.c_uint32),
    ('reserved_344', ctypes.c_uint32),
    ('reserved_345', ctypes.c_uint32),
    ('reserved_346', ctypes.c_uint32),
    ('reserved_347', ctypes.c_uint32),
    ('reserved_348', ctypes.c_uint32),
    ('reserved_349', ctypes.c_uint32),
    ('reserved_350', ctypes.c_uint32),
    ('reserved_351', ctypes.c_uint32),
    ('reserved_352', ctypes.c_uint32),
    ('reserved_353', ctypes.c_uint32),
    ('reserved_354', ctypes.c_uint32),
    ('reserved_355', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc3_ps', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc3_vs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc3_gs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc3_hs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc4_ps', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc4_vs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc4_gs', ctypes.c_uint32),
    ('spi_shader_pgm_rsrc4_hs', ctypes.c_uint32),
    ('db_occlusion_count0_low_00', ctypes.c_uint32),
    ('db_occlusion_count0_hi_00', ctypes.c_uint32),
    ('db_occlusion_count1_low_00', ctypes.c_uint32),
    ('db_occlusion_count1_hi_00', ctypes.c_uint32),
    ('db_occlusion_count2_low_00', ctypes.c_uint32),
    ('db_occlusion_count2_hi_00', ctypes.c_uint32),
    ('db_occlusion_count3_low_00', ctypes.c_uint32),
    ('db_occlusion_count3_hi_00', ctypes.c_uint32),
    ('db_occlusion_count0_low_01', ctypes.c_uint32),
    ('db_occlusion_count0_hi_01', ctypes.c_uint32),
    ('db_occlusion_count1_low_01', ctypes.c_uint32),
    ('db_occlusion_count1_hi_01', ctypes.c_uint32),
    ('db_occlusion_count2_low_01', ctypes.c_uint32),
    ('db_occlusion_count2_hi_01', ctypes.c_uint32),
    ('db_occlusion_count3_low_01', ctypes.c_uint32),
    ('db_occlusion_count3_hi_01', ctypes.c_uint32),
    ('db_occlusion_count0_low_02', ctypes.c_uint32),
    ('db_occlusion_count0_hi_02', ctypes.c_uint32),
    ('db_occlusion_count1_low_02', ctypes.c_uint32),
    ('db_occlusion_count1_hi_02', ctypes.c_uint32),
    ('db_occlusion_count2_low_02', ctypes.c_uint32),
    ('db_occlusion_count2_hi_02', ctypes.c_uint32),
    ('db_occlusion_count3_low_02', ctypes.c_uint32),
    ('db_occlusion_count3_hi_02', ctypes.c_uint32),
    ('db_occlusion_count0_low_03', ctypes.c_uint32),
    ('db_occlusion_count0_hi_03', ctypes.c_uint32),
    ('db_occlusion_count1_low_03', ctypes.c_uint32),
    ('db_occlusion_count1_hi_03', ctypes.c_uint32),
    ('db_occlusion_count2_low_03', ctypes.c_uint32),
    ('db_occlusion_count2_hi_03', ctypes.c_uint32),
    ('db_occlusion_count3_low_03', ctypes.c_uint32),
    ('db_occlusion_count3_hi_03', ctypes.c_uint32),
    ('db_occlusion_count0_low_04', ctypes.c_uint32),
    ('db_occlusion_count0_hi_04', ctypes.c_uint32),
    ('db_occlusion_count1_low_04', ctypes.c_uint32),
    ('db_occlusion_count1_hi_04', ctypes.c_uint32),
    ('db_occlusion_count2_low_04', ctypes.c_uint32),
    ('db_occlusion_count2_hi_04', ctypes.c_uint32),
    ('db_occlusion_count3_low_04', ctypes.c_uint32),
    ('db_occlusion_count3_hi_04', ctypes.c_uint32),
    ('db_occlusion_count0_low_05', ctypes.c_uint32),
    ('db_occlusion_count0_hi_05', ctypes.c_uint32),
    ('db_occlusion_count1_low_05', ctypes.c_uint32),
    ('db_occlusion_count1_hi_05', ctypes.c_uint32),
    ('db_occlusion_count2_low_05', ctypes.c_uint32),
    ('db_occlusion_count2_hi_05', ctypes.c_uint32),
    ('db_occlusion_count3_low_05', ctypes.c_uint32),
    ('db_occlusion_count3_hi_05', ctypes.c_uint32),
    ('db_occlusion_count0_low_06', ctypes.c_uint32),
    ('db_occlusion_count0_hi_06', ctypes.c_uint32),
    ('db_occlusion_count1_low_06', ctypes.c_uint32),
    ('db_occlusion_count1_hi_06', ctypes.c_uint32),
    ('db_occlusion_count2_low_06', ctypes.c_uint32),
    ('db_occlusion_count2_hi_06', ctypes.c_uint32),
    ('db_occlusion_count3_low_06', ctypes.c_uint32),
    ('db_occlusion_count3_hi_06', ctypes.c_uint32),
    ('db_occlusion_count0_low_07', ctypes.c_uint32),
    ('db_occlusion_count0_hi_07', ctypes.c_uint32),
    ('db_occlusion_count1_low_07', ctypes.c_uint32),
    ('db_occlusion_count1_hi_07', ctypes.c_uint32),
    ('db_occlusion_count2_low_07', ctypes.c_uint32),
    ('db_occlusion_count2_hi_07', ctypes.c_uint32),
    ('db_occlusion_count3_low_07', ctypes.c_uint32),
    ('db_occlusion_count3_hi_07', ctypes.c_uint32),
    ('db_occlusion_count0_low_10', ctypes.c_uint32),
    ('db_occlusion_count0_hi_10', ctypes.c_uint32),
    ('db_occlusion_count1_low_10', ctypes.c_uint32),
    ('db_occlusion_count1_hi_10', ctypes.c_uint32),
    ('db_occlusion_count2_low_10', ctypes.c_uint32),
    ('db_occlusion_count2_hi_10', ctypes.c_uint32),
    ('db_occlusion_count3_low_10', ctypes.c_uint32),
    ('db_occlusion_count3_hi_10', ctypes.c_uint32),
    ('db_occlusion_count0_low_11', ctypes.c_uint32),
    ('db_occlusion_count0_hi_11', ctypes.c_uint32),
    ('db_occlusion_count1_low_11', ctypes.c_uint32),
    ('db_occlusion_count1_hi_11', ctypes.c_uint32),
    ('db_occlusion_count2_low_11', ctypes.c_uint32),
    ('db_occlusion_count2_hi_11', ctypes.c_uint32),
    ('db_occlusion_count3_low_11', ctypes.c_uint32),
    ('db_occlusion_count3_hi_11', ctypes.c_uint32),
    ('db_occlusion_count0_low_12', ctypes.c_uint32),
    ('db_occlusion_count0_hi_12', ctypes.c_uint32),
    ('db_occlusion_count1_low_12', ctypes.c_uint32),
    ('db_occlusion_count1_hi_12', ctypes.c_uint32),
    ('db_occlusion_count2_low_12', ctypes.c_uint32),
    ('db_occlusion_count2_hi_12', ctypes.c_uint32),
    ('db_occlusion_count3_low_12', ctypes.c_uint32),
    ('db_occlusion_count3_hi_12', ctypes.c_uint32),
    ('db_occlusion_count0_low_13', ctypes.c_uint32),
    ('db_occlusion_count0_hi_13', ctypes.c_uint32),
    ('db_occlusion_count1_low_13', ctypes.c_uint32),
    ('db_occlusion_count1_hi_13', ctypes.c_uint32),
    ('db_occlusion_count2_low_13', ctypes.c_uint32),
    ('db_occlusion_count2_hi_13', ctypes.c_uint32),
    ('db_occlusion_count3_low_13', ctypes.c_uint32),
    ('db_occlusion_count3_hi_13', ctypes.c_uint32),
    ('db_occlusion_count0_low_14', ctypes.c_uint32),
    ('db_occlusion_count0_hi_14', ctypes.c_uint32),
    ('db_occlusion_count1_low_14', ctypes.c_uint32),
    ('db_occlusion_count1_hi_14', ctypes.c_uint32),
    ('db_occlusion_count2_low_14', ctypes.c_uint32),
    ('db_occlusion_count2_hi_14', ctypes.c_uint32),
    ('db_occlusion_count3_low_14', ctypes.c_uint32),
    ('db_occlusion_count3_hi_14', ctypes.c_uint32),
    ('db_occlusion_count0_low_15', ctypes.c_uint32),
    ('db_occlusion_count0_hi_15', ctypes.c_uint32),
    ('db_occlusion_count1_low_15', ctypes.c_uint32),
    ('db_occlusion_count1_hi_15', ctypes.c_uint32),
    ('db_occlusion_count2_low_15', ctypes.c_uint32),
    ('db_occlusion_count2_hi_15', ctypes.c_uint32),
    ('db_occlusion_count3_low_15', ctypes.c_uint32),
    ('db_occlusion_count3_hi_15', ctypes.c_uint32),
    ('db_occlusion_count0_low_16', ctypes.c_uint32),
    ('db_occlusion_count0_hi_16', ctypes.c_uint32),
    ('db_occlusion_count1_low_16', ctypes.c_uint32),
    ('db_occlusion_count1_hi_16', ctypes.c_uint32),
    ('db_occlusion_count2_low_16', ctypes.c_uint32),
    ('db_occlusion_count2_hi_16', ctypes.c_uint32),
    ('db_occlusion_count3_low_16', ctypes.c_uint32),
    ('db_occlusion_count3_hi_16', ctypes.c_uint32),
    ('db_occlusion_count0_low_17', ctypes.c_uint32),
    ('db_occlusion_count0_hi_17', ctypes.c_uint32),
    ('db_occlusion_count1_low_17', ctypes.c_uint32),
    ('db_occlusion_count1_hi_17', ctypes.c_uint32),
    ('db_occlusion_count2_low_17', ctypes.c_uint32),
    ('db_occlusion_count2_hi_17', ctypes.c_uint32),
    ('db_occlusion_count3_low_17', ctypes.c_uint32),
    ('db_occlusion_count3_hi_17', ctypes.c_uint32),
    ('reserved_492', ctypes.c_uint32),
    ('reserved_493', ctypes.c_uint32),
    ('reserved_494', ctypes.c_uint32),
    ('reserved_495', ctypes.c_uint32),
    ('reserved_496', ctypes.c_uint32),
    ('reserved_497', ctypes.c_uint32),
    ('reserved_498', ctypes.c_uint32),
    ('reserved_499', ctypes.c_uint32),
    ('reserved_500', ctypes.c_uint32),
    ('reserved_501', ctypes.c_uint32),
    ('reserved_502', ctypes.c_uint32),
    ('reserved_503', ctypes.c_uint32),
    ('reserved_504', ctypes.c_uint32),
    ('reserved_505', ctypes.c_uint32),
    ('reserved_506', ctypes.c_uint32),
    ('reserved_507', ctypes.c_uint32),
    ('reserved_508', ctypes.c_uint32),
    ('reserved_509', ctypes.c_uint32),
    ('reserved_510', ctypes.c_uint32),
    ('reserved_511', ctypes.c_uint32),
]

class struct_v11_sdma_mqd(Structure):
    pass

struct_v11_sdma_mqd._pack_ = 1 # source:False
struct_v11_sdma_mqd._fields_ = [
    ('sdmax_rlcx_rb_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_rb_base', ctypes.c_uint32),
    ('sdmax_rlcx_rb_base_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_ib_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_ib_rptr', ctypes.c_uint32),
    ('sdmax_rlcx_ib_offset', ctypes.c_uint32),
    ('sdmax_rlcx_ib_base_lo', ctypes.c_uint32),
    ('sdmax_rlcx_ib_base_hi', ctypes.c_uint32),
    ('sdmax_rlcx_ib_size', ctypes.c_uint32),
    ('sdmax_rlcx_skip_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_context_status', ctypes.c_uint32),
    ('sdmax_rlcx_doorbell', ctypes.c_uint32),
    ('sdmax_rlcx_doorbell_log', ctypes.c_uint32),
    ('sdmax_rlcx_doorbell_offset', ctypes.c_uint32),
    ('sdmax_rlcx_csa_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_csa_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_sched_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_ib_sub_remain', ctypes.c_uint32),
    ('sdmax_rlcx_preempt', ctypes.c_uint32),
    ('sdmax_rlcx_dummy_reg', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr_poll_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr_poll_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_rb_aql_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_minor_ptr_update', ctypes.c_uint32),
    ('sdmax_rlcx_rb_preempt', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data0', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data1', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data2', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data3', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data4', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data5', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data6', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data7', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data8', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data9', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data10', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_f32_dbg0', ctypes.c_uint32),
    ('sdmax_rlcx_f32_dbg1', ctypes.c_uint32),
    ('reserved_45', ctypes.c_uint32),
    ('reserved_46', ctypes.c_uint32),
    ('reserved_47', ctypes.c_uint32),
    ('reserved_48', ctypes.c_uint32),
    ('reserved_49', ctypes.c_uint32),
    ('reserved_50', ctypes.c_uint32),
    ('reserved_51', ctypes.c_uint32),
    ('reserved_52', ctypes.c_uint32),
    ('reserved_53', ctypes.c_uint32),
    ('reserved_54', ctypes.c_uint32),
    ('reserved_55', ctypes.c_uint32),
    ('reserved_56', ctypes.c_uint32),
    ('reserved_57', ctypes.c_uint32),
    ('reserved_58', ctypes.c_uint32),
    ('reserved_59', ctypes.c_uint32),
    ('reserved_60', ctypes.c_uint32),
    ('reserved_61', ctypes.c_uint32),
    ('reserved_62', ctypes.c_uint32),
    ('reserved_63', ctypes.c_uint32),
    ('reserved_64', ctypes.c_uint32),
    ('reserved_65', ctypes.c_uint32),
    ('reserved_66', ctypes.c_uint32),
    ('reserved_67', ctypes.c_uint32),
    ('reserved_68', ctypes.c_uint32),
    ('reserved_69', ctypes.c_uint32),
    ('reserved_70', ctypes.c_uint32),
    ('reserved_71', ctypes.c_uint32),
    ('reserved_72', ctypes.c_uint32),
    ('reserved_73', ctypes.c_uint32),
    ('reserved_74', ctypes.c_uint32),
    ('reserved_75', ctypes.c_uint32),
    ('reserved_76', ctypes.c_uint32),
    ('reserved_77', ctypes.c_uint32),
    ('reserved_78', ctypes.c_uint32),
    ('reserved_79', ctypes.c_uint32),
    ('reserved_80', ctypes.c_uint32),
    ('reserved_81', ctypes.c_uint32),
    ('reserved_82', ctypes.c_uint32),
    ('reserved_83', ctypes.c_uint32),
    ('reserved_84', ctypes.c_uint32),
    ('reserved_85', ctypes.c_uint32),
    ('reserved_86', ctypes.c_uint32),
    ('reserved_87', ctypes.c_uint32),
    ('reserved_88', ctypes.c_uint32),
    ('reserved_89', ctypes.c_uint32),
    ('reserved_90', ctypes.c_uint32),
    ('reserved_91', ctypes.c_uint32),
    ('reserved_92', ctypes.c_uint32),
    ('reserved_93', ctypes.c_uint32),
    ('reserved_94', ctypes.c_uint32),
    ('reserved_95', ctypes.c_uint32),
    ('reserved_96', ctypes.c_uint32),
    ('reserved_97', ctypes.c_uint32),
    ('reserved_98', ctypes.c_uint32),
    ('reserved_99', ctypes.c_uint32),
    ('reserved_100', ctypes.c_uint32),
    ('reserved_101', ctypes.c_uint32),
    ('reserved_102', ctypes.c_uint32),
    ('reserved_103', ctypes.c_uint32),
    ('reserved_104', ctypes.c_uint32),
    ('reserved_105', ctypes.c_uint32),
    ('reserved_106', ctypes.c_uint32),
    ('reserved_107', ctypes.c_uint32),
    ('reserved_108', ctypes.c_uint32),
    ('reserved_109', ctypes.c_uint32),
    ('reserved_110', ctypes.c_uint32),
    ('reserved_111', ctypes.c_uint32),
    ('reserved_112', ctypes.c_uint32),
    ('reserved_113', ctypes.c_uint32),
    ('reserved_114', ctypes.c_uint32),
    ('reserved_115', ctypes.c_uint32),
    ('reserved_116', ctypes.c_uint32),
    ('reserved_117', ctypes.c_uint32),
    ('reserved_118', ctypes.c_uint32),
    ('reserved_119', ctypes.c_uint32),
    ('reserved_120', ctypes.c_uint32),
    ('reserved_121', ctypes.c_uint32),
    ('reserved_122', ctypes.c_uint32),
    ('reserved_123', ctypes.c_uint32),
    ('reserved_124', ctypes.c_uint32),
    ('reserved_125', ctypes.c_uint32),
    ('sdma_engine_id', ctypes.c_uint32),
    ('sdma_queue_id', ctypes.c_uint32),
]

class struct_v11_compute_mqd(Structure):
    pass

struct_v11_compute_mqd._pack_ = 1 # source:False
struct_v11_compute_mqd._fields_ = [
    ('header', ctypes.c_uint32),
    ('compute_dispatch_initiator', ctypes.c_uint32),
    ('compute_dim_x', ctypes.c_uint32),
    ('compute_dim_y', ctypes.c_uint32),
    ('compute_dim_z', ctypes.c_uint32),
    ('compute_start_x', ctypes.c_uint32),
    ('compute_start_y', ctypes.c_uint32),
    ('compute_start_z', ctypes.c_uint32),
    ('compute_num_thread_x', ctypes.c_uint32),
    ('compute_num_thread_y', ctypes.c_uint32),
    ('compute_num_thread_z', ctypes.c_uint32),
    ('compute_pipelinestat_enable', ctypes.c_uint32),
    ('compute_perfcount_enable', ctypes.c_uint32),
    ('compute_pgm_lo', ctypes.c_uint32),
    ('compute_pgm_hi', ctypes.c_uint32),
    ('compute_dispatch_pkt_addr_lo', ctypes.c_uint32),
    ('compute_dispatch_pkt_addr_hi', ctypes.c_uint32),
    ('compute_dispatch_scratch_base_lo', ctypes.c_uint32),
    ('compute_dispatch_scratch_base_hi', ctypes.c_uint32),
    ('compute_pgm_rsrc1', ctypes.c_uint32),
    ('compute_pgm_rsrc2', ctypes.c_uint32),
    ('compute_vmid', ctypes.c_uint32),
    ('compute_resource_limits', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se0', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se1', ctypes.c_uint32),
    ('compute_tmpring_size', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se2', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se3', ctypes.c_uint32),
    ('compute_restart_x', ctypes.c_uint32),
    ('compute_restart_y', ctypes.c_uint32),
    ('compute_restart_z', ctypes.c_uint32),
    ('compute_thread_trace_enable', ctypes.c_uint32),
    ('compute_misc_reserved', ctypes.c_uint32),
    ('compute_dispatch_id', ctypes.c_uint32),
    ('compute_threadgroup_id', ctypes.c_uint32),
    ('compute_req_ctrl', ctypes.c_uint32),
    ('reserved_36', ctypes.c_uint32),
    ('compute_user_accum_0', ctypes.c_uint32),
    ('compute_user_accum_1', ctypes.c_uint32),
    ('compute_user_accum_2', ctypes.c_uint32),
    ('compute_user_accum_3', ctypes.c_uint32),
    ('compute_pgm_rsrc3', ctypes.c_uint32),
    ('compute_ddid_index', ctypes.c_uint32),
    ('compute_shader_chksum', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se4', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se5', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se6', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se7', ctypes.c_uint32),
    ('compute_dispatch_interleave', ctypes.c_uint32),
    ('compute_relaunch', ctypes.c_uint32),
    ('compute_wave_restore_addr_lo', ctypes.c_uint32),
    ('compute_wave_restore_addr_hi', ctypes.c_uint32),
    ('compute_wave_restore_control', ctypes.c_uint32),
    ('reserved_53', ctypes.c_uint32),
    ('reserved_54', ctypes.c_uint32),
    ('reserved_55', ctypes.c_uint32),
    ('reserved_56', ctypes.c_uint32),
    ('reserved_57', ctypes.c_uint32),
    ('reserved_58', ctypes.c_uint32),
    ('reserved_59', ctypes.c_uint32),
    ('reserved_60', ctypes.c_uint32),
    ('reserved_61', ctypes.c_uint32),
    ('reserved_62', ctypes.c_uint32),
    ('reserved_63', ctypes.c_uint32),
    ('reserved_64', ctypes.c_uint32),
    ('compute_user_data_0', ctypes.c_uint32),
    ('compute_user_data_1', ctypes.c_uint32),
    ('compute_user_data_2', ctypes.c_uint32),
    ('compute_user_data_3', ctypes.c_uint32),
    ('compute_user_data_4', ctypes.c_uint32),
    ('compute_user_data_5', ctypes.c_uint32),
    ('compute_user_data_6', ctypes.c_uint32),
    ('compute_user_data_7', ctypes.c_uint32),
    ('compute_user_data_8', ctypes.c_uint32),
    ('compute_user_data_9', ctypes.c_uint32),
    ('compute_user_data_10', ctypes.c_uint32),
    ('compute_user_data_11', ctypes.c_uint32),
    ('compute_user_data_12', ctypes.c_uint32),
    ('compute_user_data_13', ctypes.c_uint32),
    ('compute_user_data_14', ctypes.c_uint32),
    ('compute_user_data_15', ctypes.c_uint32),
    ('cp_compute_csinvoc_count_lo', ctypes.c_uint32),
    ('cp_compute_csinvoc_count_hi', ctypes.c_uint32),
    ('reserved_83', ctypes.c_uint32),
    ('reserved_84', ctypes.c_uint32),
    ('reserved_85', ctypes.c_uint32),
    ('cp_mqd_query_time_lo', ctypes.c_uint32),
    ('cp_mqd_query_time_hi', ctypes.c_uint32),
    ('cp_mqd_connect_start_time_lo', ctypes.c_uint32),
    ('cp_mqd_connect_start_time_hi', ctypes.c_uint32),
    ('cp_mqd_connect_end_time_lo', ctypes.c_uint32),
    ('cp_mqd_connect_end_time_hi', ctypes.c_uint32),
    ('cp_mqd_connect_end_wf_count', ctypes.c_uint32),
    ('cp_mqd_connect_end_pq_rptr', ctypes.c_uint32),
    ('cp_mqd_connect_end_pq_wptr', ctypes.c_uint32),
    ('cp_mqd_connect_end_ib_rptr', ctypes.c_uint32),
    ('cp_mqd_readindex_lo', ctypes.c_uint32),
    ('cp_mqd_readindex_hi', ctypes.c_uint32),
    ('cp_mqd_save_start_time_lo', ctypes.c_uint32),
    ('cp_mqd_save_start_time_hi', ctypes.c_uint32),
    ('cp_mqd_save_end_time_lo', ctypes.c_uint32),
    ('cp_mqd_save_end_time_hi', ctypes.c_uint32),
    ('cp_mqd_restore_start_time_lo', ctypes.c_uint32),
    ('cp_mqd_restore_start_time_hi', ctypes.c_uint32),
    ('cp_mqd_restore_end_time_lo', ctypes.c_uint32),
    ('cp_mqd_restore_end_time_hi', ctypes.c_uint32),
    ('disable_queue', ctypes.c_uint32),
    ('reserved_107', ctypes.c_uint32),
    ('gds_cs_ctxsw_cnt0', ctypes.c_uint32),
    ('gds_cs_ctxsw_cnt1', ctypes.c_uint32),
    ('gds_cs_ctxsw_cnt2', ctypes.c_uint32),
    ('gds_cs_ctxsw_cnt3', ctypes.c_uint32),
    ('reserved_112', ctypes.c_uint32),
    ('reserved_113', ctypes.c_uint32),
    ('cp_pq_exe_status_lo', ctypes.c_uint32),
    ('cp_pq_exe_status_hi', ctypes.c_uint32),
    ('cp_packet_id_lo', ctypes.c_uint32),
    ('cp_packet_id_hi', ctypes.c_uint32),
    ('cp_packet_exe_status_lo', ctypes.c_uint32),
    ('cp_packet_exe_status_hi', ctypes.c_uint32),
    ('gds_save_base_addr_lo', ctypes.c_uint32),
    ('gds_save_base_addr_hi', ctypes.c_uint32),
    ('gds_save_mask_lo', ctypes.c_uint32),
    ('gds_save_mask_hi', ctypes.c_uint32),
    ('ctx_save_base_addr_lo', ctypes.c_uint32),
    ('ctx_save_base_addr_hi', ctypes.c_uint32),
    ('reserved_126', ctypes.c_uint32),
    ('reserved_127', ctypes.c_uint32),
    ('cp_mqd_base_addr_lo', ctypes.c_uint32),
    ('cp_mqd_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_active', ctypes.c_uint32),
    ('cp_hqd_vmid', ctypes.c_uint32),
    ('cp_hqd_persistent_state', ctypes.c_uint32),
    ('cp_hqd_pipe_priority', ctypes.c_uint32),
    ('cp_hqd_queue_priority', ctypes.c_uint32),
    ('cp_hqd_quantum', ctypes.c_uint32),
    ('cp_hqd_pq_base_lo', ctypes.c_uint32),
    ('cp_hqd_pq_base_hi', ctypes.c_uint32),
    ('cp_hqd_pq_rptr', ctypes.c_uint32),
    ('cp_hqd_pq_rptr_report_addr_lo', ctypes.c_uint32),
    ('cp_hqd_pq_rptr_report_addr_hi', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_poll_addr_lo', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_poll_addr_hi', ctypes.c_uint32),
    ('cp_hqd_pq_doorbell_control', ctypes.c_uint32),
    ('reserved_144', ctypes.c_uint32),
    ('cp_hqd_pq_control', ctypes.c_uint32),
    ('cp_hqd_ib_base_addr_lo', ctypes.c_uint32),
    ('cp_hqd_ib_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_ib_rptr', ctypes.c_uint32),
    ('cp_hqd_ib_control', ctypes.c_uint32),
    ('cp_hqd_iq_timer', ctypes.c_uint32),
    ('cp_hqd_iq_rptr', ctypes.c_uint32),
    ('cp_hqd_dequeue_request', ctypes.c_uint32),
    ('cp_hqd_dma_offload', ctypes.c_uint32),
    ('cp_hqd_sema_cmd', ctypes.c_uint32),
    ('cp_hqd_msg_type', ctypes.c_uint32),
    ('cp_hqd_atomic0_preop_lo', ctypes.c_uint32),
    ('cp_hqd_atomic0_preop_hi', ctypes.c_uint32),
    ('cp_hqd_atomic1_preop_lo', ctypes.c_uint32),
    ('cp_hqd_atomic1_preop_hi', ctypes.c_uint32),
    ('cp_hqd_hq_status0', ctypes.c_uint32),
    ('cp_hqd_hq_control0', ctypes.c_uint32),
    ('cp_mqd_control', ctypes.c_uint32),
    ('cp_hqd_hq_status1', ctypes.c_uint32),
    ('cp_hqd_hq_control1', ctypes.c_uint32),
    ('cp_hqd_eop_base_addr_lo', ctypes.c_uint32),
    ('cp_hqd_eop_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_eop_control', ctypes.c_uint32),
    ('cp_hqd_eop_rptr', ctypes.c_uint32),
    ('cp_hqd_eop_wptr', ctypes.c_uint32),
    ('cp_hqd_eop_done_events', ctypes.c_uint32),
    ('cp_hqd_ctx_save_base_addr_lo', ctypes.c_uint32),
    ('cp_hqd_ctx_save_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_ctx_save_control', ctypes.c_uint32),
    ('cp_hqd_cntl_stack_offset', ctypes.c_uint32),
    ('cp_hqd_cntl_stack_size', ctypes.c_uint32),
    ('cp_hqd_wg_state_offset', ctypes.c_uint32),
    ('cp_hqd_ctx_save_size', ctypes.c_uint32),
    ('cp_hqd_gds_resource_state', ctypes.c_uint32),
    ('cp_hqd_error', ctypes.c_uint32),
    ('cp_hqd_eop_wptr_mem', ctypes.c_uint32),
    ('cp_hqd_aql_control', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_lo', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_hi', ctypes.c_uint32),
    ('reserved_184', ctypes.c_uint32),
    ('reserved_185', ctypes.c_uint32),
    ('reserved_186', ctypes.c_uint32),
    ('reserved_187', ctypes.c_uint32),
    ('reserved_188', ctypes.c_uint32),
    ('reserved_189', ctypes.c_uint32),
    ('reserved_190', ctypes.c_uint32),
    ('reserved_191', ctypes.c_uint32),
    ('iqtimer_pkt_header', ctypes.c_uint32),
    ('iqtimer_pkt_dw0', ctypes.c_uint32),
    ('iqtimer_pkt_dw1', ctypes.c_uint32),
    ('iqtimer_pkt_dw2', ctypes.c_uint32),
    ('iqtimer_pkt_dw3', ctypes.c_uint32),
    ('iqtimer_pkt_dw4', ctypes.c_uint32),
    ('iqtimer_pkt_dw5', ctypes.c_uint32),
    ('iqtimer_pkt_dw6', ctypes.c_uint32),
    ('iqtimer_pkt_dw7', ctypes.c_uint32),
    ('iqtimer_pkt_dw8', ctypes.c_uint32),
    ('iqtimer_pkt_dw9', ctypes.c_uint32),
    ('iqtimer_pkt_dw10', ctypes.c_uint32),
    ('iqtimer_pkt_dw11', ctypes.c_uint32),
    ('iqtimer_pkt_dw12', ctypes.c_uint32),
    ('iqtimer_pkt_dw13', ctypes.c_uint32),
    ('iqtimer_pkt_dw14', ctypes.c_uint32),
    ('iqtimer_pkt_dw15', ctypes.c_uint32),
    ('iqtimer_pkt_dw16', ctypes.c_uint32),
    ('iqtimer_pkt_dw17', ctypes.c_uint32),
    ('iqtimer_pkt_dw18', ctypes.c_uint32),
    ('iqtimer_pkt_dw19', ctypes.c_uint32),
    ('iqtimer_pkt_dw20', ctypes.c_uint32),
    ('iqtimer_pkt_dw21', ctypes.c_uint32),
    ('iqtimer_pkt_dw22', ctypes.c_uint32),
    ('iqtimer_pkt_dw23', ctypes.c_uint32),
    ('iqtimer_pkt_dw24', ctypes.c_uint32),
    ('iqtimer_pkt_dw25', ctypes.c_uint32),
    ('iqtimer_pkt_dw26', ctypes.c_uint32),
    ('iqtimer_pkt_dw27', ctypes.c_uint32),
    ('iqtimer_pkt_dw28', ctypes.c_uint32),
    ('iqtimer_pkt_dw29', ctypes.c_uint32),
    ('iqtimer_pkt_dw30', ctypes.c_uint32),
    ('iqtimer_pkt_dw31', ctypes.c_uint32),
    ('reserved_225', ctypes.c_uint32),
    ('reserved_226', ctypes.c_uint32),
    ('reserved_227', ctypes.c_uint32),
    ('set_resources_header', ctypes.c_uint32),
    ('set_resources_dw1', ctypes.c_uint32),
    ('set_resources_dw2', ctypes.c_uint32),
    ('set_resources_dw3', ctypes.c_uint32),
    ('set_resources_dw4', ctypes.c_uint32),
    ('set_resources_dw5', ctypes.c_uint32),
    ('set_resources_dw6', ctypes.c_uint32),
    ('set_resources_dw7', ctypes.c_uint32),
    ('reserved_236', ctypes.c_uint32),
    ('reserved_237', ctypes.c_uint32),
    ('reserved_238', ctypes.c_uint32),
    ('reserved_239', ctypes.c_uint32),
    ('queue_doorbell_id0', ctypes.c_uint32),
    ('queue_doorbell_id1', ctypes.c_uint32),
    ('queue_doorbell_id2', ctypes.c_uint32),
    ('queue_doorbell_id3', ctypes.c_uint32),
    ('queue_doorbell_id4', ctypes.c_uint32),
    ('queue_doorbell_id5', ctypes.c_uint32),
    ('queue_doorbell_id6', ctypes.c_uint32),
    ('queue_doorbell_id7', ctypes.c_uint32),
    ('queue_doorbell_id8', ctypes.c_uint32),
    ('queue_doorbell_id9', ctypes.c_uint32),
    ('queue_doorbell_id10', ctypes.c_uint32),
    ('queue_doorbell_id11', ctypes.c_uint32),
    ('queue_doorbell_id12', ctypes.c_uint32),
    ('queue_doorbell_id13', ctypes.c_uint32),
    ('queue_doorbell_id14', ctypes.c_uint32),
    ('queue_doorbell_id15', ctypes.c_uint32),
    ('control_buf_addr_lo', ctypes.c_uint32),
    ('control_buf_addr_hi', ctypes.c_uint32),
    ('control_buf_wptr_lo', ctypes.c_uint32),
    ('control_buf_wptr_hi', ctypes.c_uint32),
    ('control_buf_dptr_lo', ctypes.c_uint32),
    ('control_buf_dptr_hi', ctypes.c_uint32),
    ('control_buf_num_entries', ctypes.c_uint32),
    ('draw_ring_addr_lo', ctypes.c_uint32),
    ('draw_ring_addr_hi', ctypes.c_uint32),
    ('reserved_265', ctypes.c_uint32),
    ('reserved_266', ctypes.c_uint32),
    ('reserved_267', ctypes.c_uint32),
    ('reserved_268', ctypes.c_uint32),
    ('reserved_269', ctypes.c_uint32),
    ('reserved_270', ctypes.c_uint32),
    ('reserved_271', ctypes.c_uint32),
    ('reserved_272', ctypes.c_uint32),
    ('reserved_273', ctypes.c_uint32),
    ('reserved_274', ctypes.c_uint32),
    ('reserved_275', ctypes.c_uint32),
    ('reserved_276', ctypes.c_uint32),
    ('reserved_277', ctypes.c_uint32),
    ('reserved_278', ctypes.c_uint32),
    ('reserved_279', ctypes.c_uint32),
    ('reserved_280', ctypes.c_uint32),
    ('reserved_281', ctypes.c_uint32),
    ('reserved_282', ctypes.c_uint32),
    ('reserved_283', ctypes.c_uint32),
    ('reserved_284', ctypes.c_uint32),
    ('reserved_285', ctypes.c_uint32),
    ('reserved_286', ctypes.c_uint32),
    ('reserved_287', ctypes.c_uint32),
    ('reserved_288', ctypes.c_uint32),
    ('reserved_289', ctypes.c_uint32),
    ('reserved_290', ctypes.c_uint32),
    ('reserved_291', ctypes.c_uint32),
    ('reserved_292', ctypes.c_uint32),
    ('reserved_293', ctypes.c_uint32),
    ('reserved_294', ctypes.c_uint32),
    ('reserved_295', ctypes.c_uint32),
    ('reserved_296', ctypes.c_uint32),
    ('reserved_297', ctypes.c_uint32),
    ('reserved_298', ctypes.c_uint32),
    ('reserved_299', ctypes.c_uint32),
    ('reserved_300', ctypes.c_uint32),
    ('reserved_301', ctypes.c_uint32),
    ('reserved_302', ctypes.c_uint32),
    ('reserved_303', ctypes.c_uint32),
    ('reserved_304', ctypes.c_uint32),
    ('reserved_305', ctypes.c_uint32),
    ('reserved_306', ctypes.c_uint32),
    ('reserved_307', ctypes.c_uint32),
    ('reserved_308', ctypes.c_uint32),
    ('reserved_309', ctypes.c_uint32),
    ('reserved_310', ctypes.c_uint32),
    ('reserved_311', ctypes.c_uint32),
    ('reserved_312', ctypes.c_uint32),
    ('reserved_313', ctypes.c_uint32),
    ('reserved_314', ctypes.c_uint32),
    ('reserved_315', ctypes.c_uint32),
    ('reserved_316', ctypes.c_uint32),
    ('reserved_317', ctypes.c_uint32),
    ('reserved_318', ctypes.c_uint32),
    ('reserved_319', ctypes.c_uint32),
    ('reserved_320', ctypes.c_uint32),
    ('reserved_321', ctypes.c_uint32),
    ('reserved_322', ctypes.c_uint32),
    ('reserved_323', ctypes.c_uint32),
    ('reserved_324', ctypes.c_uint32),
    ('reserved_325', ctypes.c_uint32),
    ('reserved_326', ctypes.c_uint32),
    ('reserved_327', ctypes.c_uint32),
    ('reserved_328', ctypes.c_uint32),
    ('reserved_329', ctypes.c_uint32),
    ('reserved_330', ctypes.c_uint32),
    ('reserved_331', ctypes.c_uint32),
    ('reserved_332', ctypes.c_uint32),
    ('reserved_333', ctypes.c_uint32),
    ('reserved_334', ctypes.c_uint32),
    ('reserved_335', ctypes.c_uint32),
    ('reserved_336', ctypes.c_uint32),
    ('reserved_337', ctypes.c_uint32),
    ('reserved_338', ctypes.c_uint32),
    ('reserved_339', ctypes.c_uint32),
    ('reserved_340', ctypes.c_uint32),
    ('reserved_341', ctypes.c_uint32),
    ('reserved_342', ctypes.c_uint32),
    ('reserved_343', ctypes.c_uint32),
    ('reserved_344', ctypes.c_uint32),
    ('reserved_345', ctypes.c_uint32),
    ('reserved_346', ctypes.c_uint32),
    ('reserved_347', ctypes.c_uint32),
    ('reserved_348', ctypes.c_uint32),
    ('reserved_349', ctypes.c_uint32),
    ('reserved_350', ctypes.c_uint32),
    ('reserved_351', ctypes.c_uint32),
    ('reserved_352', ctypes.c_uint32),
    ('reserved_353', ctypes.c_uint32),
    ('reserved_354', ctypes.c_uint32),
    ('reserved_355', ctypes.c_uint32),
    ('reserved_356', ctypes.c_uint32),
    ('reserved_357', ctypes.c_uint32),
    ('reserved_358', ctypes.c_uint32),
    ('reserved_359', ctypes.c_uint32),
    ('reserved_360', ctypes.c_uint32),
    ('reserved_361', ctypes.c_uint32),
    ('reserved_362', ctypes.c_uint32),
    ('reserved_363', ctypes.c_uint32),
    ('reserved_364', ctypes.c_uint32),
    ('reserved_365', ctypes.c_uint32),
    ('reserved_366', ctypes.c_uint32),
    ('reserved_367', ctypes.c_uint32),
    ('reserved_368', ctypes.c_uint32),
    ('reserved_369', ctypes.c_uint32),
    ('reserved_370', ctypes.c_uint32),
    ('reserved_371', ctypes.c_uint32),
    ('reserved_372', ctypes.c_uint32),
    ('reserved_373', ctypes.c_uint32),
    ('reserved_374', ctypes.c_uint32),
    ('reserved_375', ctypes.c_uint32),
    ('reserved_376', ctypes.c_uint32),
    ('reserved_377', ctypes.c_uint32),
    ('reserved_378', ctypes.c_uint32),
    ('reserved_379', ctypes.c_uint32),
    ('reserved_380', ctypes.c_uint32),
    ('reserved_381', ctypes.c_uint32),
    ('reserved_382', ctypes.c_uint32),
    ('reserved_383', ctypes.c_uint32),
    ('reserved_384', ctypes.c_uint32),
    ('reserved_385', ctypes.c_uint32),
    ('reserved_386', ctypes.c_uint32),
    ('reserved_387', ctypes.c_uint32),
    ('reserved_388', ctypes.c_uint32),
    ('reserved_389', ctypes.c_uint32),
    ('reserved_390', ctypes.c_uint32),
    ('reserved_391', ctypes.c_uint32),
    ('reserved_392', ctypes.c_uint32),
    ('reserved_393', ctypes.c_uint32),
    ('reserved_394', ctypes.c_uint32),
    ('reserved_395', ctypes.c_uint32),
    ('reserved_396', ctypes.c_uint32),
    ('reserved_397', ctypes.c_uint32),
    ('reserved_398', ctypes.c_uint32),
    ('reserved_399', ctypes.c_uint32),
    ('reserved_400', ctypes.c_uint32),
    ('reserved_401', ctypes.c_uint32),
    ('reserved_402', ctypes.c_uint32),
    ('reserved_403', ctypes.c_uint32),
    ('reserved_404', ctypes.c_uint32),
    ('reserved_405', ctypes.c_uint32),
    ('reserved_406', ctypes.c_uint32),
    ('reserved_407', ctypes.c_uint32),
    ('reserved_408', ctypes.c_uint32),
    ('reserved_409', ctypes.c_uint32),
    ('reserved_410', ctypes.c_uint32),
    ('reserved_411', ctypes.c_uint32),
    ('reserved_412', ctypes.c_uint32),
    ('reserved_413', ctypes.c_uint32),
    ('reserved_414', ctypes.c_uint32),
    ('reserved_415', ctypes.c_uint32),
    ('reserved_416', ctypes.c_uint32),
    ('reserved_417', ctypes.c_uint32),
    ('reserved_418', ctypes.c_uint32),
    ('reserved_419', ctypes.c_uint32),
    ('reserved_420', ctypes.c_uint32),
    ('reserved_421', ctypes.c_uint32),
    ('reserved_422', ctypes.c_uint32),
    ('reserved_423', ctypes.c_uint32),
    ('reserved_424', ctypes.c_uint32),
    ('reserved_425', ctypes.c_uint32),
    ('reserved_426', ctypes.c_uint32),
    ('reserved_427', ctypes.c_uint32),
    ('reserved_428', ctypes.c_uint32),
    ('reserved_429', ctypes.c_uint32),
    ('reserved_430', ctypes.c_uint32),
    ('reserved_431', ctypes.c_uint32),
    ('reserved_432', ctypes.c_uint32),
    ('reserved_433', ctypes.c_uint32),
    ('reserved_434', ctypes.c_uint32),
    ('reserved_435', ctypes.c_uint32),
    ('reserved_436', ctypes.c_uint32),
    ('reserved_437', ctypes.c_uint32),
    ('reserved_438', ctypes.c_uint32),
    ('reserved_439', ctypes.c_uint32),
    ('reserved_440', ctypes.c_uint32),
    ('reserved_441', ctypes.c_uint32),
    ('reserved_442', ctypes.c_uint32),
    ('reserved_443', ctypes.c_uint32),
    ('reserved_444', ctypes.c_uint32),
    ('reserved_445', ctypes.c_uint32),
    ('reserved_446', ctypes.c_uint32),
    ('reserved_447', ctypes.c_uint32),
    ('gws_0_val', ctypes.c_uint32),
    ('gws_1_val', ctypes.c_uint32),
    ('gws_2_val', ctypes.c_uint32),
    ('gws_3_val', ctypes.c_uint32),
    ('gws_4_val', ctypes.c_uint32),
    ('gws_5_val', ctypes.c_uint32),
    ('gws_6_val', ctypes.c_uint32),
    ('gws_7_val', ctypes.c_uint32),
    ('gws_8_val', ctypes.c_uint32),
    ('gws_9_val', ctypes.c_uint32),
    ('gws_10_val', ctypes.c_uint32),
    ('gws_11_val', ctypes.c_uint32),
    ('gws_12_val', ctypes.c_uint32),
    ('gws_13_val', ctypes.c_uint32),
    ('gws_14_val', ctypes.c_uint32),
    ('gws_15_val', ctypes.c_uint32),
    ('gws_16_val', ctypes.c_uint32),
    ('gws_17_val', ctypes.c_uint32),
    ('gws_18_val', ctypes.c_uint32),
    ('gws_19_val', ctypes.c_uint32),
    ('gws_20_val', ctypes.c_uint32),
    ('gws_21_val', ctypes.c_uint32),
    ('gws_22_val', ctypes.c_uint32),
    ('gws_23_val', ctypes.c_uint32),
    ('gws_24_val', ctypes.c_uint32),
    ('gws_25_val', ctypes.c_uint32),
    ('gws_26_val', ctypes.c_uint32),
    ('gws_27_val', ctypes.c_uint32),
    ('gws_28_val', ctypes.c_uint32),
    ('gws_29_val', ctypes.c_uint32),
    ('gws_30_val', ctypes.c_uint32),
    ('gws_31_val', ctypes.c_uint32),
    ('gws_32_val', ctypes.c_uint32),
    ('gws_33_val', ctypes.c_uint32),
    ('gws_34_val', ctypes.c_uint32),
    ('gws_35_val', ctypes.c_uint32),
    ('gws_36_val', ctypes.c_uint32),
    ('gws_37_val', ctypes.c_uint32),
    ('gws_38_val', ctypes.c_uint32),
    ('gws_39_val', ctypes.c_uint32),
    ('gws_40_val', ctypes.c_uint32),
    ('gws_41_val', ctypes.c_uint32),
    ('gws_42_val', ctypes.c_uint32),
    ('gws_43_val', ctypes.c_uint32),
    ('gws_44_val', ctypes.c_uint32),
    ('gws_45_val', ctypes.c_uint32),
    ('gws_46_val', ctypes.c_uint32),
    ('gws_47_val', ctypes.c_uint32),
    ('gws_48_val', ctypes.c_uint32),
    ('gws_49_val', ctypes.c_uint32),
    ('gws_50_val', ctypes.c_uint32),
    ('gws_51_val', ctypes.c_uint32),
    ('gws_52_val', ctypes.c_uint32),
    ('gws_53_val', ctypes.c_uint32),
    ('gws_54_val', ctypes.c_uint32),
    ('gws_55_val', ctypes.c_uint32),
    ('gws_56_val', ctypes.c_uint32),
    ('gws_57_val', ctypes.c_uint32),
    ('gws_58_val', ctypes.c_uint32),
    ('gws_59_val', ctypes.c_uint32),
    ('gws_60_val', ctypes.c_uint32),
    ('gws_61_val', ctypes.c_uint32),
    ('gws_62_val', ctypes.c_uint32),
    ('gws_63_val', ctypes.c_uint32),
]

V12_STRUCTS_H_ = True # macro
class struct_v12_gfx_mqd(Structure):
    pass

struct_v12_gfx_mqd._pack_ = 1 # source:False
struct_v12_gfx_mqd._fields_ = [
    ('shadow_base_lo', ctypes.c_uint32),
    ('shadow_base_hi', ctypes.c_uint32),
    ('reserved_2', ctypes.c_uint32),
    ('reserved_3', ctypes.c_uint32),
    ('fw_work_area_base_lo', ctypes.c_uint32),
    ('fw_work_area_base_hi', ctypes.c_uint32),
    ('shadow_initialized', ctypes.c_uint32),
    ('ib_vmid', ctypes.c_uint32),
    ('reserved_8', ctypes.c_uint32),
    ('reserved_9', ctypes.c_uint32),
    ('reserved_10', ctypes.c_uint32),
    ('reserved_11', ctypes.c_uint32),
    ('reserved_12', ctypes.c_uint32),
    ('reserved_13', ctypes.c_uint32),
    ('reserved_14', ctypes.c_uint32),
    ('reserved_15', ctypes.c_uint32),
    ('reserved_16', ctypes.c_uint32),
    ('reserved_17', ctypes.c_uint32),
    ('reserved_18', ctypes.c_uint32),
    ('reserved_19', ctypes.c_uint32),
    ('reserved_20', ctypes.c_uint32),
    ('reserved_21', ctypes.c_uint32),
    ('reserved_22', ctypes.c_uint32),
    ('reserved_23', ctypes.c_uint32),
    ('reserved_24', ctypes.c_uint32),
    ('reserved_25', ctypes.c_uint32),
    ('reserved_26', ctypes.c_uint32),
    ('reserved_27', ctypes.c_uint32),
    ('reserved_28', ctypes.c_uint32),
    ('reserved_29', ctypes.c_uint32),
    ('reserved_30', ctypes.c_uint32),
    ('reserved_31', ctypes.c_uint32),
    ('reserved_32', ctypes.c_uint32),
    ('reserved_33', ctypes.c_uint32),
    ('reserved_34', ctypes.c_uint32),
    ('reserved_35', ctypes.c_uint32),
    ('reserved_36', ctypes.c_uint32),
    ('reserved_37', ctypes.c_uint32),
    ('reserved_38', ctypes.c_uint32),
    ('reserved_39', ctypes.c_uint32),
    ('reserved_40', ctypes.c_uint32),
    ('reserved_41', ctypes.c_uint32),
    ('reserved_42', ctypes.c_uint32),
    ('reserved_43', ctypes.c_uint32),
    ('reserved_44', ctypes.c_uint32),
    ('reserved_45', ctypes.c_uint32),
    ('reserved_46', ctypes.c_uint32),
    ('reserved_47', ctypes.c_uint32),
    ('reserved_48', ctypes.c_uint32),
    ('reserved_49', ctypes.c_uint32),
    ('reserved_50', ctypes.c_uint32),
    ('reserved_51', ctypes.c_uint32),
    ('reserved_52', ctypes.c_uint32),
    ('reserved_53', ctypes.c_uint32),
    ('reserved_54', ctypes.c_uint32),
    ('reserved_55', ctypes.c_uint32),
    ('reserved_56', ctypes.c_uint32),
    ('reserved_57', ctypes.c_uint32),
    ('reserved_58', ctypes.c_uint32),
    ('reserved_59', ctypes.c_uint32),
    ('reserved_60', ctypes.c_uint32),
    ('reserved_61', ctypes.c_uint32),
    ('reserved_62', ctypes.c_uint32),
    ('reserved_63', ctypes.c_uint32),
    ('reserved_64', ctypes.c_uint32),
    ('reserved_65', ctypes.c_uint32),
    ('reserved_66', ctypes.c_uint32),
    ('reserved_67', ctypes.c_uint32),
    ('reserved_68', ctypes.c_uint32),
    ('reserved_69', ctypes.c_uint32),
    ('reserved_70', ctypes.c_uint32),
    ('reserved_71', ctypes.c_uint32),
    ('reserved_72', ctypes.c_uint32),
    ('reserved_73', ctypes.c_uint32),
    ('reserved_74', ctypes.c_uint32),
    ('reserved_75', ctypes.c_uint32),
    ('reserved_76', ctypes.c_uint32),
    ('reserved_77', ctypes.c_uint32),
    ('reserved_78', ctypes.c_uint32),
    ('reserved_79', ctypes.c_uint32),
    ('reserved_80', ctypes.c_uint32),
    ('reserved_81', ctypes.c_uint32),
    ('reserved_82', ctypes.c_uint32),
    ('reserved_83', ctypes.c_uint32),
    ('checksum_lo', ctypes.c_uint32),
    ('checksum_hi', ctypes.c_uint32),
    ('cp_mqd_query_time_lo', ctypes.c_uint32),
    ('cp_mqd_query_time_hi', ctypes.c_uint32),
    ('reserved_88', ctypes.c_uint32),
    ('reserved_89', ctypes.c_uint32),
    ('reserved_90', ctypes.c_uint32),
    ('reserved_91', ctypes.c_uint32),
    ('cp_mqd_query_wave_count', ctypes.c_uint32),
    ('cp_mqd_query_gfx_hqd_rptr', ctypes.c_uint32),
    ('cp_mqd_query_gfx_hqd_wptr', ctypes.c_uint32),
    ('cp_mqd_query_gfx_hqd_offset', ctypes.c_uint32),
    ('reserved_96', ctypes.c_uint32),
    ('reserved_97', ctypes.c_uint32),
    ('reserved_98', ctypes.c_uint32),
    ('reserved_99', ctypes.c_uint32),
    ('reserved_100', ctypes.c_uint32),
    ('reserved_101', ctypes.c_uint32),
    ('reserved_102', ctypes.c_uint32),
    ('reserved_103', ctypes.c_uint32),
    ('task_shader_control_buf_addr_lo', ctypes.c_uint32),
    ('task_shader_control_buf_addr_hi', ctypes.c_uint32),
    ('task_shader_read_rptr_lo', ctypes.c_uint32),
    ('task_shader_read_rptr_hi', ctypes.c_uint32),
    ('task_shader_num_entries', ctypes.c_uint32),
    ('task_shader_num_entries_bits', ctypes.c_uint32),
    ('task_shader_ring_buffer_addr_lo', ctypes.c_uint32),
    ('task_shader_ring_buffer_addr_hi', ctypes.c_uint32),
    ('reserved_112', ctypes.c_uint32),
    ('reserved_113', ctypes.c_uint32),
    ('reserved_114', ctypes.c_uint32),
    ('reserved_115', ctypes.c_uint32),
    ('reserved_116', ctypes.c_uint32),
    ('reserved_117', ctypes.c_uint32),
    ('reserved_118', ctypes.c_uint32),
    ('reserved_119', ctypes.c_uint32),
    ('reserved_120', ctypes.c_uint32),
    ('reserved_121', ctypes.c_uint32),
    ('reserved_122', ctypes.c_uint32),
    ('reserved_123', ctypes.c_uint32),
    ('reserved_124', ctypes.c_uint32),
    ('reserved_125', ctypes.c_uint32),
    ('reserved_126', ctypes.c_uint32),
    ('reserved_127', ctypes.c_uint32),
    ('cp_mqd_base_addr', ctypes.c_uint32),
    ('cp_mqd_base_addr_hi', ctypes.c_uint32),
    ('cp_gfx_hqd_active', ctypes.c_uint32),
    ('cp_gfx_hqd_vmid', ctypes.c_uint32),
    ('reserved_132', ctypes.c_uint32),
    ('reserved_133', ctypes.c_uint32),
    ('cp_gfx_hqd_queue_priority', ctypes.c_uint32),
    ('cp_gfx_hqd_quantum', ctypes.c_uint32),
    ('cp_gfx_hqd_base', ctypes.c_uint32),
    ('cp_gfx_hqd_base_hi', ctypes.c_uint32),
    ('cp_gfx_hqd_rptr', ctypes.c_uint32),
    ('cp_gfx_hqd_rptr_addr', ctypes.c_uint32),
    ('cp_gfx_hqd_rptr_addr_hi', ctypes.c_uint32),
    ('cp_rb_wptr_poll_addr_lo', ctypes.c_uint32),
    ('cp_rb_wptr_poll_addr_hi', ctypes.c_uint32),
    ('cp_rb_doorbell_control', ctypes.c_uint32),
    ('cp_gfx_hqd_offset', ctypes.c_uint32),
    ('cp_gfx_hqd_cntl', ctypes.c_uint32),
    ('reserved_146', ctypes.c_uint32),
    ('reserved_147', ctypes.c_uint32),
    ('cp_gfx_hqd_csmd_rptr', ctypes.c_uint32),
    ('cp_gfx_hqd_wptr', ctypes.c_uint32),
    ('cp_gfx_hqd_wptr_hi', ctypes.c_uint32),
    ('reserved_151', ctypes.c_uint32),
    ('reserved_152', ctypes.c_uint32),
    ('reserved_153', ctypes.c_uint32),
    ('reserved_154', ctypes.c_uint32),
    ('reserved_155', ctypes.c_uint32),
    ('cp_gfx_hqd_mapped', ctypes.c_uint32),
    ('cp_gfx_hqd_que_mgr_control', ctypes.c_uint32),
    ('reserved_158', ctypes.c_uint32),
    ('reserved_159', ctypes.c_uint32),
    ('cp_gfx_hqd_hq_status0', ctypes.c_uint32),
    ('cp_gfx_hqd_hq_control0', ctypes.c_uint32),
    ('cp_gfx_mqd_control', ctypes.c_uint32),
    ('reserved_163', ctypes.c_uint32),
    ('reserved_164', ctypes.c_uint32),
    ('reserved_165', ctypes.c_uint32),
    ('reserved_166', ctypes.c_uint32),
    ('reserved_167', ctypes.c_uint32),
    ('reserved_168', ctypes.c_uint32),
    ('reserved_169', ctypes.c_uint32),
    ('reserved_170', ctypes.c_uint32),
    ('reserved_171', ctypes.c_uint32),
    ('reserved_172', ctypes.c_uint32),
    ('reserved_173', ctypes.c_uint32),
    ('reserved_174', ctypes.c_uint32),
    ('reserved_175', ctypes.c_uint32),
    ('reserved_176', ctypes.c_uint32),
    ('reserved_177', ctypes.c_uint32),
    ('reserved_178', ctypes.c_uint32),
    ('reserved_179', ctypes.c_uint32),
    ('reserved_180', ctypes.c_uint32),
    ('reserved_181', ctypes.c_uint32),
    ('reserved_182', ctypes.c_uint32),
    ('reserved_183', ctypes.c_uint32),
    ('reserved_184', ctypes.c_uint32),
    ('reserved_185', ctypes.c_uint32),
    ('reserved_186', ctypes.c_uint32),
    ('reserved_187', ctypes.c_uint32),
    ('reserved_188', ctypes.c_uint32),
    ('reserved_189', ctypes.c_uint32),
    ('reserved_190', ctypes.c_uint32),
    ('reserved_191', ctypes.c_uint32),
    ('reserved_192', ctypes.c_uint32),
    ('reserved_193', ctypes.c_uint32),
    ('reserved_194', ctypes.c_uint32),
    ('reserved_195', ctypes.c_uint32),
    ('reserved_196', ctypes.c_uint32),
    ('reserved_197', ctypes.c_uint32),
    ('reserved_198', ctypes.c_uint32),
    ('reserved_199', ctypes.c_uint32),
    ('reserved_200', ctypes.c_uint32),
    ('reserved_201', ctypes.c_uint32),
    ('reserved_202', ctypes.c_uint32),
    ('reserved_203', ctypes.c_uint32),
    ('reserved_204', ctypes.c_uint32),
    ('reserved_205', ctypes.c_uint32),
    ('reserved_206', ctypes.c_uint32),
    ('reserved_207', ctypes.c_uint32),
    ('reserved_208', ctypes.c_uint32),
    ('reserved_209', ctypes.c_uint32),
    ('reserved_210', ctypes.c_uint32),
    ('reserved_211', ctypes.c_uint32),
    ('reserved_212', ctypes.c_uint32),
    ('reserved_213', ctypes.c_uint32),
    ('reserved_214', ctypes.c_uint32),
    ('reserved_215', ctypes.c_uint32),
    ('reserved_216', ctypes.c_uint32),
    ('reserved_217', ctypes.c_uint32),
    ('reserved_218', ctypes.c_uint32),
    ('reserved_219', ctypes.c_uint32),
    ('reserved_220', ctypes.c_uint32),
    ('reserved_221', ctypes.c_uint32),
    ('reserved_222', ctypes.c_uint32),
    ('reserved_223', ctypes.c_uint32),
    ('reserved_224', ctypes.c_uint32),
    ('reserved_225', ctypes.c_uint32),
    ('reserved_226', ctypes.c_uint32),
    ('reserved_227', ctypes.c_uint32),
    ('reserved_228', ctypes.c_uint32),
    ('reserved_229', ctypes.c_uint32),
    ('reserved_230', ctypes.c_uint32),
    ('reserved_231', ctypes.c_uint32),
    ('reserved_232', ctypes.c_uint32),
    ('reserved_233', ctypes.c_uint32),
    ('reserved_234', ctypes.c_uint32),
    ('reserved_235', ctypes.c_uint32),
    ('reserved_236', ctypes.c_uint32),
    ('reserved_237', ctypes.c_uint32),
    ('reserved_238', ctypes.c_uint32),
    ('reserved_239', ctypes.c_uint32),
    ('reserved_240', ctypes.c_uint32),
    ('reserved_241', ctypes.c_uint32),
    ('reserved_242', ctypes.c_uint32),
    ('reserved_243', ctypes.c_uint32),
    ('reserved_244', ctypes.c_uint32),
    ('reserved_245', ctypes.c_uint32),
    ('reserved_246', ctypes.c_uint32),
    ('reserved_247', ctypes.c_uint32),
    ('reserved_248', ctypes.c_uint32),
    ('reserved_249', ctypes.c_uint32),
    ('reserved_250', ctypes.c_uint32),
    ('reserved_251', ctypes.c_uint32),
    ('reserved_252', ctypes.c_uint32),
    ('reserved_253', ctypes.c_uint32),
    ('reserved_254', ctypes.c_uint32),
    ('reserved_255', ctypes.c_uint32),
    ('reserved_256', ctypes.c_uint32),
    ('reserved_257', ctypes.c_uint32),
    ('reserved_258', ctypes.c_uint32),
    ('reserved_259', ctypes.c_uint32),
    ('reserved_260', ctypes.c_uint32),
    ('reserved_261', ctypes.c_uint32),
    ('reserved_262', ctypes.c_uint32),
    ('reserved_263', ctypes.c_uint32),
    ('reserved_264', ctypes.c_uint32),
    ('reserved_265', ctypes.c_uint32),
    ('reserved_266', ctypes.c_uint32),
    ('reserved_267', ctypes.c_uint32),
    ('reserved_268', ctypes.c_uint32),
    ('reserved_269', ctypes.c_uint32),
    ('reserved_270', ctypes.c_uint32),
    ('reserved_271', ctypes.c_uint32),
    ('dfwx_flags', ctypes.c_uint32),
    ('dfwx_slot', ctypes.c_uint32),
    ('dfwx_client_data_addr_lo', ctypes.c_uint32),
    ('dfwx_client_data_addr_hi', ctypes.c_uint32),
    ('reserved_276', ctypes.c_uint32),
    ('reserved_277', ctypes.c_uint32),
    ('reserved_278', ctypes.c_uint32),
    ('reserved_279', ctypes.c_uint32),
    ('reserved_280', ctypes.c_uint32),
    ('reserved_281', ctypes.c_uint32),
    ('reserved_282', ctypes.c_uint32),
    ('reserved_283', ctypes.c_uint32),
    ('reserved_284', ctypes.c_uint32),
    ('reserved_285', ctypes.c_uint32),
    ('reserved_286', ctypes.c_uint32),
    ('reserved_287', ctypes.c_uint32),
    ('reserved_288', ctypes.c_uint32),
    ('reserved_289', ctypes.c_uint32),
    ('reserved_290', ctypes.c_uint32),
    ('reserved_291', ctypes.c_uint32),
    ('reserved_292', ctypes.c_uint32),
    ('reserved_293', ctypes.c_uint32),
    ('reserved_294', ctypes.c_uint32),
    ('reserved_295', ctypes.c_uint32),
    ('reserved_296', ctypes.c_uint32),
    ('reserved_297', ctypes.c_uint32),
    ('reserved_298', ctypes.c_uint32),
    ('reserved_299', ctypes.c_uint32),
    ('reserved_300', ctypes.c_uint32),
    ('reserved_301', ctypes.c_uint32),
    ('reserved_302', ctypes.c_uint32),
    ('reserved_303', ctypes.c_uint32),
    ('reserved_304', ctypes.c_uint32),
    ('reserved_305', ctypes.c_uint32),
    ('reserved_306', ctypes.c_uint32),
    ('reserved_307', ctypes.c_uint32),
    ('reserved_308', ctypes.c_uint32),
    ('reserved_309', ctypes.c_uint32),
    ('reserved_310', ctypes.c_uint32),
    ('reserved_311', ctypes.c_uint32),
    ('reserved_312', ctypes.c_uint32),
    ('reserved_313', ctypes.c_uint32),
    ('reserved_314', ctypes.c_uint32),
    ('reserved_315', ctypes.c_uint32),
    ('reserved_316', ctypes.c_uint32),
    ('reserved_317', ctypes.c_uint32),
    ('reserved_318', ctypes.c_uint32),
    ('reserved_319', ctypes.c_uint32),
    ('reserved_320', ctypes.c_uint32),
    ('reserved_321', ctypes.c_uint32),
    ('reserved_322', ctypes.c_uint32),
    ('reserved_323', ctypes.c_uint32),
    ('reserved_324', ctypes.c_uint32),
    ('reserved_325', ctypes.c_uint32),
    ('reserved_326', ctypes.c_uint32),
    ('reserved_327', ctypes.c_uint32),
    ('reserved_328', ctypes.c_uint32),
    ('reserved_329', ctypes.c_uint32),
    ('reserved_330', ctypes.c_uint32),
    ('reserved_331', ctypes.c_uint32),
    ('reserved_332', ctypes.c_uint32),
    ('reserved_333', ctypes.c_uint32),
    ('reserved_334', ctypes.c_uint32),
    ('reserved_335', ctypes.c_uint32),
    ('reserved_336', ctypes.c_uint32),
    ('reserved_337', ctypes.c_uint32),
    ('reserved_338', ctypes.c_uint32),
    ('reserved_339', ctypes.c_uint32),
    ('reserved_340', ctypes.c_uint32),
    ('reserved_341', ctypes.c_uint32),
    ('reserved_342', ctypes.c_uint32),
    ('reserved_343', ctypes.c_uint32),
    ('reserved_344', ctypes.c_uint32),
    ('reserved_345', ctypes.c_uint32),
    ('reserved_346', ctypes.c_uint32),
    ('reserved_347', ctypes.c_uint32),
    ('reserved_348', ctypes.c_uint32),
    ('reserved_349', ctypes.c_uint32),
    ('reserved_350', ctypes.c_uint32),
    ('reserved_351', ctypes.c_uint32),
    ('reserved_352', ctypes.c_uint32),
    ('reserved_353', ctypes.c_uint32),
    ('reserved_354', ctypes.c_uint32),
    ('reserved_355', ctypes.c_uint32),
    ('reserved_356', ctypes.c_uint32),
    ('reserved_357', ctypes.c_uint32),
    ('reserved_358', ctypes.c_uint32),
    ('reserved_359', ctypes.c_uint32),
    ('reserved_360', ctypes.c_uint32),
    ('reserved_361', ctypes.c_uint32),
    ('reserved_362', ctypes.c_uint32),
    ('reserved_363', ctypes.c_uint32),
    ('reserved_364', ctypes.c_uint32),
    ('reserved_365', ctypes.c_uint32),
    ('reserved_366', ctypes.c_uint32),
    ('reserved_367', ctypes.c_uint32),
    ('reserved_368', ctypes.c_uint32),
    ('reserved_369', ctypes.c_uint32),
    ('reserved_370', ctypes.c_uint32),
    ('reserved_371', ctypes.c_uint32),
    ('reserved_372', ctypes.c_uint32),
    ('reserved_373', ctypes.c_uint32),
    ('reserved_374', ctypes.c_uint32),
    ('reserved_375', ctypes.c_uint32),
    ('reserved_376', ctypes.c_uint32),
    ('reserved_377', ctypes.c_uint32),
    ('reserved_378', ctypes.c_uint32),
    ('reserved_379', ctypes.c_uint32),
    ('reserved_380', ctypes.c_uint32),
    ('reserved_381', ctypes.c_uint32),
    ('reserved_382', ctypes.c_uint32),
    ('reserved_383', ctypes.c_uint32),
    ('reserved_384', ctypes.c_uint32),
    ('reserved_385', ctypes.c_uint32),
    ('reserved_386', ctypes.c_uint32),
    ('reserved_387', ctypes.c_uint32),
    ('reserved_388', ctypes.c_uint32),
    ('reserved_389', ctypes.c_uint32),
    ('reserved_390', ctypes.c_uint32),
    ('reserved_391', ctypes.c_uint32),
    ('reserved_392', ctypes.c_uint32),
    ('reserved_393', ctypes.c_uint32),
    ('reserved_394', ctypes.c_uint32),
    ('reserved_395', ctypes.c_uint32),
    ('reserved_396', ctypes.c_uint32),
    ('reserved_397', ctypes.c_uint32),
    ('reserved_398', ctypes.c_uint32),
    ('reserved_399', ctypes.c_uint32),
    ('reserved_400', ctypes.c_uint32),
    ('reserved_401', ctypes.c_uint32),
    ('reserved_402', ctypes.c_uint32),
    ('reserved_403', ctypes.c_uint32),
    ('reserved_404', ctypes.c_uint32),
    ('reserved_405', ctypes.c_uint32),
    ('reserved_406', ctypes.c_uint32),
    ('reserved_407', ctypes.c_uint32),
    ('reserved_408', ctypes.c_uint32),
    ('reserved_409', ctypes.c_uint32),
    ('reserved_410', ctypes.c_uint32),
    ('reserved_411', ctypes.c_uint32),
    ('reserved_412', ctypes.c_uint32),
    ('reserved_413', ctypes.c_uint32),
    ('reserved_414', ctypes.c_uint32),
    ('reserved_415', ctypes.c_uint32),
    ('reserved_416', ctypes.c_uint32),
    ('reserved_417', ctypes.c_uint32),
    ('reserved_418', ctypes.c_uint32),
    ('reserved_419', ctypes.c_uint32),
    ('reserved_420', ctypes.c_uint32),
    ('reserved_421', ctypes.c_uint32),
    ('reserved_422', ctypes.c_uint32),
    ('reserved_423', ctypes.c_uint32),
    ('reserved_424', ctypes.c_uint32),
    ('reserved_425', ctypes.c_uint32),
    ('reserved_426', ctypes.c_uint32),
    ('reserved_427', ctypes.c_uint32),
    ('reserved_428', ctypes.c_uint32),
    ('reserved_429', ctypes.c_uint32),
    ('reserved_430', ctypes.c_uint32),
    ('reserved_431', ctypes.c_uint32),
    ('reserved_432', ctypes.c_uint32),
    ('reserved_433', ctypes.c_uint32),
    ('reserved_434', ctypes.c_uint32),
    ('reserved_435', ctypes.c_uint32),
    ('reserved_436', ctypes.c_uint32),
    ('reserved_437', ctypes.c_uint32),
    ('reserved_438', ctypes.c_uint32),
    ('reserved_439', ctypes.c_uint32),
    ('reserved_440', ctypes.c_uint32),
    ('reserved_441', ctypes.c_uint32),
    ('reserved_442', ctypes.c_uint32),
    ('reserved_443', ctypes.c_uint32),
    ('reserved_444', ctypes.c_uint32),
    ('reserved_445', ctypes.c_uint32),
    ('reserved_446', ctypes.c_uint32),
    ('reserved_447', ctypes.c_uint32),
    ('reserved_448', ctypes.c_uint32),
    ('reserved_449', ctypes.c_uint32),
    ('reserved_450', ctypes.c_uint32),
    ('reserved_451', ctypes.c_uint32),
    ('reserved_452', ctypes.c_uint32),
    ('reserved_453', ctypes.c_uint32),
    ('reserved_454', ctypes.c_uint32),
    ('reserved_455', ctypes.c_uint32),
    ('reserved_456', ctypes.c_uint32),
    ('reserved_457', ctypes.c_uint32),
    ('reserved_458', ctypes.c_uint32),
    ('reserved_459', ctypes.c_uint32),
    ('reserved_460', ctypes.c_uint32),
    ('reserved_461', ctypes.c_uint32),
    ('reserved_462', ctypes.c_uint32),
    ('reserved_463', ctypes.c_uint32),
    ('reserved_464', ctypes.c_uint32),
    ('reserved_465', ctypes.c_uint32),
    ('reserved_466', ctypes.c_uint32),
    ('reserved_467', ctypes.c_uint32),
    ('reserved_468', ctypes.c_uint32),
    ('reserved_469', ctypes.c_uint32),
    ('reserved_470', ctypes.c_uint32),
    ('reserved_471', ctypes.c_uint32),
    ('reserved_472', ctypes.c_uint32),
    ('reserved_473', ctypes.c_uint32),
    ('reserved_474', ctypes.c_uint32),
    ('reserved_475', ctypes.c_uint32),
    ('reserved_476', ctypes.c_uint32),
    ('reserved_477', ctypes.c_uint32),
    ('reserved_478', ctypes.c_uint32),
    ('reserved_479', ctypes.c_uint32),
    ('reserved_480', ctypes.c_uint32),
    ('reserved_481', ctypes.c_uint32),
    ('reserved_482', ctypes.c_uint32),
    ('reserved_483', ctypes.c_uint32),
    ('reserved_484', ctypes.c_uint32),
    ('reserved_485', ctypes.c_uint32),
    ('reserved_486', ctypes.c_uint32),
    ('reserved_487', ctypes.c_uint32),
    ('reserved_488', ctypes.c_uint32),
    ('reserved_489', ctypes.c_uint32),
    ('reserved_490', ctypes.c_uint32),
    ('reserved_491', ctypes.c_uint32),
    ('reserved_492', ctypes.c_uint32),
    ('reserved_493', ctypes.c_uint32),
    ('reserved_494', ctypes.c_uint32),
    ('reserved_495', ctypes.c_uint32),
    ('reserved_496', ctypes.c_uint32),
    ('reserved_497', ctypes.c_uint32),
    ('reserved_498', ctypes.c_uint32),
    ('reserved_499', ctypes.c_uint32),
    ('reserved_500', ctypes.c_uint32),
    ('reserved_501', ctypes.c_uint32),
    ('reserved_502', ctypes.c_uint32),
    ('reserved_503', ctypes.c_uint32),
    ('reserved_504', ctypes.c_uint32),
    ('reserved_505', ctypes.c_uint32),
    ('reserved_506', ctypes.c_uint32),
    ('reserved_507', ctypes.c_uint32),
    ('reserved_508', ctypes.c_uint32),
    ('reserved_509', ctypes.c_uint32),
    ('reserved_510', ctypes.c_uint32),
    ('reserved_511', ctypes.c_uint32),
]

class struct_v12_sdma_mqd(Structure):
    pass

struct_v12_sdma_mqd._pack_ = 1 # source:False
struct_v12_sdma_mqd._fields_ = [
    ('sdmax_rlcx_rb_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_rb_base', ctypes.c_uint32),
    ('sdmax_rlcx_rb_base_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_rb_rptr_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_ib_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_ib_rptr', ctypes.c_uint32),
    ('sdmax_rlcx_ib_offset', ctypes.c_uint32),
    ('sdmax_rlcx_ib_base_lo', ctypes.c_uint32),
    ('sdmax_rlcx_ib_base_hi', ctypes.c_uint32),
    ('sdmax_rlcx_ib_size', ctypes.c_uint32),
    ('sdmax_rlcx_doorbell', ctypes.c_uint32),
    ('sdmax_rlcx_doorbell_log', ctypes.c_uint32),
    ('sdmax_rlcx_doorbell_offset', ctypes.c_uint32),
    ('sdmax_rlcx_csa_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_csa_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_sched_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_ib_sub_remain', ctypes.c_uint32),
    ('sdmax_rlcx_preempt', ctypes.c_uint32),
    ('sdmax_rlcx_dummy_reg', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr_poll_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_rb_wptr_poll_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_rb_aql_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_minor_ptr_update', ctypes.c_uint32),
    ('sdmax_rlcx_mcu_dbg0', ctypes.c_uint32),
    ('sdmax_rlcx_mcu_dbg1', ctypes.c_uint32),
    ('sdmax_rlcx_context_switch_status', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_cntl', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data0', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data1', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data2', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data3', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data4', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data5', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data6', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data7', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data8', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data9', ctypes.c_uint32),
    ('sdmax_rlcx_midcmd_data10', ctypes.c_uint32),
    ('sdmax_rlcx_wait_unsatisfied_thd', ctypes.c_uint32),
    ('sdmax_rlcx_mqd_base_addr_lo', ctypes.c_uint32),
    ('sdmax_rlcx_mqd_base_addr_hi', ctypes.c_uint32),
    ('sdmax_rlcx_mqd_control', ctypes.c_uint32),
    ('reserved_47', ctypes.c_uint32),
    ('reserved_48', ctypes.c_uint32),
    ('reserved_49', ctypes.c_uint32),
    ('reserved_50', ctypes.c_uint32),
    ('reserved_51', ctypes.c_uint32),
    ('reserved_52', ctypes.c_uint32),
    ('reserved_53', ctypes.c_uint32),
    ('reserved_54', ctypes.c_uint32),
    ('reserved_55', ctypes.c_uint32),
    ('reserved_56', ctypes.c_uint32),
    ('reserved_57', ctypes.c_uint32),
    ('reserved_58', ctypes.c_uint32),
    ('reserved_59', ctypes.c_uint32),
    ('reserved_60', ctypes.c_uint32),
    ('reserved_61', ctypes.c_uint32),
    ('reserved_62', ctypes.c_uint32),
    ('reserved_63', ctypes.c_uint32),
    ('reserved_64', ctypes.c_uint32),
    ('reserved_65', ctypes.c_uint32),
    ('reserved_66', ctypes.c_uint32),
    ('reserved_67', ctypes.c_uint32),
    ('reserved_68', ctypes.c_uint32),
    ('reserved_69', ctypes.c_uint32),
    ('reserved_70', ctypes.c_uint32),
    ('reserved_71', ctypes.c_uint32),
    ('reserved_72', ctypes.c_uint32),
    ('reserved_73', ctypes.c_uint32),
    ('reserved_74', ctypes.c_uint32),
    ('reserved_75', ctypes.c_uint32),
    ('reserved_76', ctypes.c_uint32),
    ('reserved_77', ctypes.c_uint32),
    ('reserved_78', ctypes.c_uint32),
    ('reserved_79', ctypes.c_uint32),
    ('reserved_80', ctypes.c_uint32),
    ('reserved_81', ctypes.c_uint32),
    ('reserved_82', ctypes.c_uint32),
    ('reserved_83', ctypes.c_uint32),
    ('reserved_84', ctypes.c_uint32),
    ('reserved_85', ctypes.c_uint32),
    ('reserved_86', ctypes.c_uint32),
    ('reserved_87', ctypes.c_uint32),
    ('reserved_88', ctypes.c_uint32),
    ('reserved_89', ctypes.c_uint32),
    ('reserved_90', ctypes.c_uint32),
    ('reserved_91', ctypes.c_uint32),
    ('reserved_92', ctypes.c_uint32),
    ('reserved_93', ctypes.c_uint32),
    ('reserved_94', ctypes.c_uint32),
    ('reserved_95', ctypes.c_uint32),
    ('reserved_96', ctypes.c_uint32),
    ('reserved_97', ctypes.c_uint32),
    ('reserved_98', ctypes.c_uint32),
    ('reserved_99', ctypes.c_uint32),
    ('reserved_100', ctypes.c_uint32),
    ('reserved_101', ctypes.c_uint32),
    ('reserved_102', ctypes.c_uint32),
    ('reserved_103', ctypes.c_uint32),
    ('reserved_104', ctypes.c_uint32),
    ('reserved_105', ctypes.c_uint32),
    ('reserved_106', ctypes.c_uint32),
    ('reserved_107', ctypes.c_uint32),
    ('reserved_108', ctypes.c_uint32),
    ('reserved_109', ctypes.c_uint32),
    ('reserved_110', ctypes.c_uint32),
    ('reserved_111', ctypes.c_uint32),
    ('reserved_112', ctypes.c_uint32),
    ('reserved_113', ctypes.c_uint32),
    ('reserved_114', ctypes.c_uint32),
    ('reserved_115', ctypes.c_uint32),
    ('reserved_116', ctypes.c_uint32),
    ('reserved_117', ctypes.c_uint32),
    ('reserved_118', ctypes.c_uint32),
    ('reserved_119', ctypes.c_uint32),
    ('reserved_120', ctypes.c_uint32),
    ('reserved_121', ctypes.c_uint32),
    ('reserved_122', ctypes.c_uint32),
    ('reserved_123', ctypes.c_uint32),
    ('reserved_124', ctypes.c_uint32),
    ('reserved_125', ctypes.c_uint32),
    ('sdma_engine_id', ctypes.c_uint32),
    ('sdma_queue_id', ctypes.c_uint32),
]

class struct_v12_compute_mqd(Structure):
    pass

struct_v12_compute_mqd._pack_ = 1 # source:False
struct_v12_compute_mqd._fields_ = [
    ('header', ctypes.c_uint32),
    ('compute_dispatch_initiator', ctypes.c_uint32),
    ('compute_dim_x', ctypes.c_uint32),
    ('compute_dim_y', ctypes.c_uint32),
    ('compute_dim_z', ctypes.c_uint32),
    ('compute_start_x', ctypes.c_uint32),
    ('compute_start_y', ctypes.c_uint32),
    ('compute_start_z', ctypes.c_uint32),
    ('compute_num_thread_x', ctypes.c_uint32),
    ('compute_num_thread_y', ctypes.c_uint32),
    ('compute_num_thread_z', ctypes.c_uint32),
    ('compute_pipelinestat_enable', ctypes.c_uint32),
    ('compute_perfcount_enable', ctypes.c_uint32),
    ('compute_pgm_lo', ctypes.c_uint32),
    ('compute_pgm_hi', ctypes.c_uint32),
    ('compute_dispatch_pkt_addr_lo', ctypes.c_uint32),
    ('compute_dispatch_pkt_addr_hi', ctypes.c_uint32),
    ('compute_dispatch_scratch_base_lo', ctypes.c_uint32),
    ('compute_dispatch_scratch_base_hi', ctypes.c_uint32),
    ('compute_pgm_rsrc1', ctypes.c_uint32),
    ('compute_pgm_rsrc2', ctypes.c_uint32),
    ('compute_vmid', ctypes.c_uint32),
    ('compute_resource_limits', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se0', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se1', ctypes.c_uint32),
    ('compute_tmpring_size', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se2', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se3', ctypes.c_uint32),
    ('compute_restart_x', ctypes.c_uint32),
    ('compute_restart_y', ctypes.c_uint32),
    ('compute_restart_z', ctypes.c_uint32),
    ('compute_thread_trace_enable', ctypes.c_uint32),
    ('compute_misc_reserved', ctypes.c_uint32),
    ('compute_dispatch_id', ctypes.c_uint32),
    ('compute_threadgroup_id', ctypes.c_uint32),
    ('compute_req_ctrl', ctypes.c_uint32),
    ('reserved_36', ctypes.c_uint32),
    ('compute_user_accum_0', ctypes.c_uint32),
    ('compute_user_accum_1', ctypes.c_uint32),
    ('compute_user_accum_2', ctypes.c_uint32),
    ('compute_user_accum_3', ctypes.c_uint32),
    ('compute_pgm_rsrc3', ctypes.c_uint32),
    ('compute_ddid_index', ctypes.c_uint32),
    ('compute_shader_chksum', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se4', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se5', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se6', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se7', ctypes.c_uint32),
    ('compute_dispatch_interleave', ctypes.c_uint32),
    ('compute_relaunch', ctypes.c_uint32),
    ('compute_wave_restore_addr_lo', ctypes.c_uint32),
    ('compute_wave_restore_addr_hi', ctypes.c_uint32),
    ('compute_wave_restore_control', ctypes.c_uint32),
    ('reserved_53', ctypes.c_uint32),
    ('reserved_54', ctypes.c_uint32),
    ('reserved_55', ctypes.c_uint32),
    ('reserved_56', ctypes.c_uint32),
    ('reserved_57', ctypes.c_uint32),
    ('reserved_58', ctypes.c_uint32),
    ('compute_static_thread_mgmt_se8', ctypes.c_uint32),
    ('reserved_60', ctypes.c_uint32),
    ('reserved_61', ctypes.c_uint32),
    ('reserved_62', ctypes.c_uint32),
    ('reserved_63', ctypes.c_uint32),
    ('reserved_64', ctypes.c_uint32),
    ('compute_user_data_0', ctypes.c_uint32),
    ('compute_user_data_1', ctypes.c_uint32),
    ('compute_user_data_2', ctypes.c_uint32),
    ('compute_user_data_3', ctypes.c_uint32),
    ('compute_user_data_4', ctypes.c_uint32),
    ('compute_user_data_5', ctypes.c_uint32),
    ('compute_user_data_6', ctypes.c_uint32),
    ('compute_user_data_7', ctypes.c_uint32),
    ('compute_user_data_8', ctypes.c_uint32),
    ('compute_user_data_9', ctypes.c_uint32),
    ('compute_user_data_10', ctypes.c_uint32),
    ('compute_user_data_11', ctypes.c_uint32),
    ('compute_user_data_12', ctypes.c_uint32),
    ('compute_user_data_13', ctypes.c_uint32),
    ('compute_user_data_14', ctypes.c_uint32),
    ('compute_user_data_15', ctypes.c_uint32),
    ('cp_compute_csinvoc_count_lo', ctypes.c_uint32),
    ('cp_compute_csinvoc_count_hi', ctypes.c_uint32),
    ('reserved_83', ctypes.c_uint32),
    ('reserved_84', ctypes.c_uint32),
    ('reserved_85', ctypes.c_uint32),
    ('cp_mqd_query_time_lo', ctypes.c_uint32),
    ('cp_mqd_query_time_hi', ctypes.c_uint32),
    ('cp_mqd_connect_start_time_lo', ctypes.c_uint32),
    ('cp_mqd_connect_start_time_hi', ctypes.c_uint32),
    ('cp_mqd_connect_end_time_lo', ctypes.c_uint32),
    ('cp_mqd_connect_end_time_hi', ctypes.c_uint32),
    ('cp_mqd_connect_end_wf_count', ctypes.c_uint32),
    ('cp_mqd_connect_end_pq_rptr', ctypes.c_uint32),
    ('cp_mqd_connect_end_pq_wptr', ctypes.c_uint32),
    ('cp_mqd_connect_end_ib_rptr', ctypes.c_uint32),
    ('cp_mqd_readindex_lo', ctypes.c_uint32),
    ('cp_mqd_readindex_hi', ctypes.c_uint32),
    ('cp_mqd_save_start_time_lo', ctypes.c_uint32),
    ('cp_mqd_save_start_time_hi', ctypes.c_uint32),
    ('cp_mqd_save_end_time_lo', ctypes.c_uint32),
    ('cp_mqd_save_end_time_hi', ctypes.c_uint32),
    ('cp_mqd_restore_start_time_lo', ctypes.c_uint32),
    ('cp_mqd_restore_start_time_hi', ctypes.c_uint32),
    ('cp_mqd_restore_end_time_lo', ctypes.c_uint32),
    ('cp_mqd_restore_end_time_hi', ctypes.c_uint32),
    ('disable_queue', ctypes.c_uint32),
    ('reserved_107', ctypes.c_uint32),
    ('reserved_108', ctypes.c_uint32),
    ('reserved_109', ctypes.c_uint32),
    ('reserved_110', ctypes.c_uint32),
    ('reserved_111', ctypes.c_uint32),
    ('reserved_112', ctypes.c_uint32),
    ('reserved_113', ctypes.c_uint32),
    ('cp_pq_exe_status_lo', ctypes.c_uint32),
    ('cp_pq_exe_status_hi', ctypes.c_uint32),
    ('cp_packet_id_lo', ctypes.c_uint32),
    ('cp_packet_id_hi', ctypes.c_uint32),
    ('cp_packet_exe_status_lo', ctypes.c_uint32),
    ('cp_packet_exe_status_hi', ctypes.c_uint32),
    ('reserved_120', ctypes.c_uint32),
    ('reserved_121', ctypes.c_uint32),
    ('reserved_122', ctypes.c_uint32),
    ('reserved_123', ctypes.c_uint32),
    ('ctx_save_base_addr_lo', ctypes.c_uint32),
    ('ctx_save_base_addr_hi', ctypes.c_uint32),
    ('reserved_126', ctypes.c_uint32),
    ('reserved_127', ctypes.c_uint32),
    ('cp_mqd_base_addr_lo', ctypes.c_uint32),
    ('cp_mqd_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_active', ctypes.c_uint32),
    ('cp_hqd_vmid', ctypes.c_uint32),
    ('cp_hqd_persistent_state', ctypes.c_uint32),
    ('cp_hqd_pipe_priority', ctypes.c_uint32),
    ('cp_hqd_queue_priority', ctypes.c_uint32),
    ('cp_hqd_quantum', ctypes.c_uint32),
    ('cp_hqd_pq_base_lo', ctypes.c_uint32),
    ('cp_hqd_pq_base_hi', ctypes.c_uint32),
    ('cp_hqd_pq_rptr', ctypes.c_uint32),
    ('cp_hqd_pq_rptr_report_addr_lo', ctypes.c_uint32),
    ('cp_hqd_pq_rptr_report_addr_hi', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_poll_addr_lo', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_poll_addr_hi', ctypes.c_uint32),
    ('cp_hqd_pq_doorbell_control', ctypes.c_uint32),
    ('reserved_144', ctypes.c_uint32),
    ('cp_hqd_pq_control', ctypes.c_uint32),
    ('cp_hqd_ib_base_addr_lo', ctypes.c_uint32),
    ('cp_hqd_ib_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_ib_rptr', ctypes.c_uint32),
    ('cp_hqd_ib_control', ctypes.c_uint32),
    ('cp_hqd_iq_timer', ctypes.c_uint32),
    ('cp_hqd_iq_rptr', ctypes.c_uint32),
    ('cp_hqd_dequeue_request', ctypes.c_uint32),
    ('cp_hqd_dma_offload', ctypes.c_uint32),
    ('cp_hqd_sema_cmd', ctypes.c_uint32),
    ('cp_hqd_msg_type', ctypes.c_uint32),
    ('cp_hqd_atomic0_preop_lo', ctypes.c_uint32),
    ('cp_hqd_atomic0_preop_hi', ctypes.c_uint32),
    ('cp_hqd_atomic1_preop_lo', ctypes.c_uint32),
    ('cp_hqd_atomic1_preop_hi', ctypes.c_uint32),
    ('cp_hqd_hq_status0', ctypes.c_uint32),
    ('cp_hqd_hq_control0', ctypes.c_uint32),
    ('cp_mqd_control', ctypes.c_uint32),
    ('cp_hqd_hq_status1', ctypes.c_uint32),
    ('cp_hqd_hq_control1', ctypes.c_uint32),
    ('cp_hqd_eop_base_addr_lo', ctypes.c_uint32),
    ('cp_hqd_eop_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_eop_control', ctypes.c_uint32),
    ('cp_hqd_eop_rptr', ctypes.c_uint32),
    ('cp_hqd_eop_wptr', ctypes.c_uint32),
    ('cp_hqd_eop_done_events', ctypes.c_uint32),
    ('cp_hqd_ctx_save_base_addr_lo', ctypes.c_uint32),
    ('cp_hqd_ctx_save_base_addr_hi', ctypes.c_uint32),
    ('cp_hqd_ctx_save_control', ctypes.c_uint32),
    ('cp_hqd_cntl_stack_offset', ctypes.c_uint32),
    ('cp_hqd_cntl_stack_size', ctypes.c_uint32),
    ('cp_hqd_wg_state_offset', ctypes.c_uint32),
    ('cp_hqd_ctx_save_size', ctypes.c_uint32),
    ('reserved_178', ctypes.c_uint32),
    ('cp_hqd_error', ctypes.c_uint32),
    ('cp_hqd_eop_wptr_mem', ctypes.c_uint32),
    ('cp_hqd_aql_control', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_lo', ctypes.c_uint32),
    ('cp_hqd_pq_wptr_hi', ctypes.c_uint32),
    ('reserved_184', ctypes.c_uint32),
    ('reserved_185', ctypes.c_uint32),
    ('reserved_186', ctypes.c_uint32),
    ('reserved_187', ctypes.c_uint32),
    ('reserved_188', ctypes.c_uint32),
    ('reserved_189', ctypes.c_uint32),
    ('reserved_190', ctypes.c_uint32),
    ('reserved_191', ctypes.c_uint32),
    ('iqtimer_pkt_header', ctypes.c_uint32),
    ('iqtimer_pkt_dw0', ctypes.c_uint32),
    ('iqtimer_pkt_dw1', ctypes.c_uint32),
    ('iqtimer_pkt_dw2', ctypes.c_uint32),
    ('iqtimer_pkt_dw3', ctypes.c_uint32),
    ('iqtimer_pkt_dw4', ctypes.c_uint32),
    ('iqtimer_pkt_dw5', ctypes.c_uint32),
    ('iqtimer_pkt_dw6', ctypes.c_uint32),
    ('iqtimer_pkt_dw7', ctypes.c_uint32),
    ('iqtimer_pkt_dw8', ctypes.c_uint32),
    ('iqtimer_pkt_dw9', ctypes.c_uint32),
    ('iqtimer_pkt_dw10', ctypes.c_uint32),
    ('iqtimer_pkt_dw11', ctypes.c_uint32),
    ('iqtimer_pkt_dw12', ctypes.c_uint32),
    ('iqtimer_pkt_dw13', ctypes.c_uint32),
    ('iqtimer_pkt_dw14', ctypes.c_uint32),
    ('iqtimer_pkt_dw15', ctypes.c_uint32),
    ('iqtimer_pkt_dw16', ctypes.c_uint32),
    ('iqtimer_pkt_dw17', ctypes.c_uint32),
    ('iqtimer_pkt_dw18', ctypes.c_uint32),
    ('iqtimer_pkt_dw19', ctypes.c_uint32),
    ('iqtimer_pkt_dw20', ctypes.c_uint32),
    ('iqtimer_pkt_dw21', ctypes.c_uint32),
    ('iqtimer_pkt_dw22', ctypes.c_uint32),
    ('iqtimer_pkt_dw23', ctypes.c_uint32),
    ('iqtimer_pkt_dw24', ctypes.c_uint32),
    ('iqtimer_pkt_dw25', ctypes.c_uint32),
    ('iqtimer_pkt_dw26', ctypes.c_uint32),
    ('iqtimer_pkt_dw27', ctypes.c_uint32),
    ('iqtimer_pkt_dw28', ctypes.c_uint32),
    ('iqtimer_pkt_dw29', ctypes.c_uint32),
    ('iqtimer_pkt_dw30', ctypes.c_uint32),
    ('iqtimer_pkt_dw31', ctypes.c_uint32),
    ('reserved_225', ctypes.c_uint32),
    ('reserved_226', ctypes.c_uint32),
    ('reserved_227', ctypes.c_uint32),
    ('set_resources_header', ctypes.c_uint32),
    ('set_resources_dw1', ctypes.c_uint32),
    ('set_resources_dw2', ctypes.c_uint32),
    ('set_resources_dw3', ctypes.c_uint32),
    ('set_resources_dw4', ctypes.c_uint32),
    ('set_resources_dw5', ctypes.c_uint32),
    ('set_resources_dw6', ctypes.c_uint32),
    ('set_resources_dw7', ctypes.c_uint32),
    ('reserved_236', ctypes.c_uint32),
    ('reserved_237', ctypes.c_uint32),
    ('reserved_238', ctypes.c_uint32),
    ('reserved_239', ctypes.c_uint32),
    ('queue_doorbell_id0', ctypes.c_uint32),
    ('queue_doorbell_id1', ctypes.c_uint32),
    ('queue_doorbell_id2', ctypes.c_uint32),
    ('queue_doorbell_id3', ctypes.c_uint32),
    ('queue_doorbell_id4', ctypes.c_uint32),
    ('queue_doorbell_id5', ctypes.c_uint32),
    ('queue_doorbell_id6', ctypes.c_uint32),
    ('queue_doorbell_id7', ctypes.c_uint32),
    ('queue_doorbell_id8', ctypes.c_uint32),
    ('queue_doorbell_id9', ctypes.c_uint32),
    ('queue_doorbell_id10', ctypes.c_uint32),
    ('queue_doorbell_id11', ctypes.c_uint32),
    ('queue_doorbell_id12', ctypes.c_uint32),
    ('queue_doorbell_id13', ctypes.c_uint32),
    ('queue_doorbell_id14', ctypes.c_uint32),
    ('queue_doorbell_id15', ctypes.c_uint32),
    ('control_buf_addr_lo', ctypes.c_uint32),
    ('control_buf_addr_hi', ctypes.c_uint32),
    ('control_buf_wptr_lo', ctypes.c_uint32),
    ('control_buf_wptr_hi', ctypes.c_uint32),
    ('control_buf_dptr_lo', ctypes.c_uint32),
    ('control_buf_dptr_hi', ctypes.c_uint32),
    ('control_buf_num_entries', ctypes.c_uint32),
    ('draw_ring_addr_lo', ctypes.c_uint32),
    ('draw_ring_addr_hi', ctypes.c_uint32),
    ('reserved_265', ctypes.c_uint32),
    ('reserved_266', ctypes.c_uint32),
    ('reserved_267', ctypes.c_uint32),
    ('reserved_268', ctypes.c_uint32),
    ('reserved_269', ctypes.c_uint32),
    ('reserved_270', ctypes.c_uint32),
    ('reserved_271', ctypes.c_uint32),
    ('dfwx_flags', ctypes.c_uint32),
    ('dfwx_slot', ctypes.c_uint32),
    ('dfwx_client_data_addr_lo', ctypes.c_uint32),
    ('dfwx_client_data_addr_hi', ctypes.c_uint32),
    ('reserved_276', ctypes.c_uint32),
    ('reserved_277', ctypes.c_uint32),
    ('reserved_278', ctypes.c_uint32),
    ('reserved_279', ctypes.c_uint32),
    ('reserved_280', ctypes.c_uint32),
    ('reserved_281', ctypes.c_uint32),
    ('reserved_282', ctypes.c_uint32),
    ('reserved_283', ctypes.c_uint32),
    ('reserved_284', ctypes.c_uint32),
    ('reserved_285', ctypes.c_uint32),
    ('reserved_286', ctypes.c_uint32),
    ('reserved_287', ctypes.c_uint32),
    ('reserved_288', ctypes.c_uint32),
    ('reserved_289', ctypes.c_uint32),
    ('reserved_290', ctypes.c_uint32),
    ('reserved_291', ctypes.c_uint32),
    ('reserved_292', ctypes.c_uint32),
    ('reserved_293', ctypes.c_uint32),
    ('reserved_294', ctypes.c_uint32),
    ('reserved_295', ctypes.c_uint32),
    ('reserved_296', ctypes.c_uint32),
    ('reserved_297', ctypes.c_uint32),
    ('reserved_298', ctypes.c_uint32),
    ('reserved_299', ctypes.c_uint32),
    ('reserved_300', ctypes.c_uint32),
    ('reserved_301', ctypes.c_uint32),
    ('reserved_302', ctypes.c_uint32),
    ('reserved_303', ctypes.c_uint32),
    ('reserved_304', ctypes.c_uint32),
    ('reserved_305', ctypes.c_uint32),
    ('reserved_306', ctypes.c_uint32),
    ('reserved_307', ctypes.c_uint32),
    ('reserved_308', ctypes.c_uint32),
    ('reserved_309', ctypes.c_uint32),
    ('reserved_310', ctypes.c_uint32),
    ('reserved_311', ctypes.c_uint32),
    ('reserved_312', ctypes.c_uint32),
    ('reserved_313', ctypes.c_uint32),
    ('reserved_314', ctypes.c_uint32),
    ('reserved_315', ctypes.c_uint32),
    ('reserved_316', ctypes.c_uint32),
    ('reserved_317', ctypes.c_uint32),
    ('reserved_318', ctypes.c_uint32),
    ('reserved_319', ctypes.c_uint32),
    ('reserved_320', ctypes.c_uint32),
    ('reserved_321', ctypes.c_uint32),
    ('reserved_322', ctypes.c_uint32),
    ('reserved_323', ctypes.c_uint32),
    ('reserved_324', ctypes.c_uint32),
    ('reserved_325', ctypes.c_uint32),
    ('reserved_326', ctypes.c_uint32),
    ('reserved_327', ctypes.c_uint32),
    ('reserved_328', ctypes.c_uint32),
    ('reserved_329', ctypes.c_uint32),
    ('reserved_330', ctypes.c_uint32),
    ('reserved_331', ctypes.c_uint32),
    ('reserved_332', ctypes.c_uint32),
    ('reserved_333', ctypes.c_uint32),
    ('reserved_334', ctypes.c_uint32),
    ('reserved_335', ctypes.c_uint32),
    ('reserved_336', ctypes.c_uint32),
    ('reserved_337', ctypes.c_uint32),
    ('reserved_338', ctypes.c_uint32),
    ('reserved_339', ctypes.c_uint32),
    ('reserved_340', ctypes.c_uint32),
    ('reserved_341', ctypes.c_uint32),
    ('reserved_342', ctypes.c_uint32),
    ('reserved_343', ctypes.c_uint32),
    ('reserved_344', ctypes.c_uint32),
    ('reserved_345', ctypes.c_uint32),
    ('reserved_346', ctypes.c_uint32),
    ('reserved_347', ctypes.c_uint32),
    ('reserved_348', ctypes.c_uint32),
    ('reserved_349', ctypes.c_uint32),
    ('reserved_350', ctypes.c_uint32),
    ('reserved_351', ctypes.c_uint32),
    ('reserved_352', ctypes.c_uint32),
    ('reserved_353', ctypes.c_uint32),
    ('reserved_354', ctypes.c_uint32),
    ('reserved_355', ctypes.c_uint32),
    ('reserved_356', ctypes.c_uint32),
    ('reserved_357', ctypes.c_uint32),
    ('reserved_358', ctypes.c_uint32),
    ('reserved_359', ctypes.c_uint32),
    ('reserved_360', ctypes.c_uint32),
    ('reserved_361', ctypes.c_uint32),
    ('reserved_362', ctypes.c_uint32),
    ('reserved_363', ctypes.c_uint32),
    ('reserved_364', ctypes.c_uint32),
    ('reserved_365', ctypes.c_uint32),
    ('reserved_366', ctypes.c_uint32),
    ('reserved_367', ctypes.c_uint32),
    ('reserved_368', ctypes.c_uint32),
    ('reserved_369', ctypes.c_uint32),
    ('reserved_370', ctypes.c_uint32),
    ('reserved_371', ctypes.c_uint32),
    ('reserved_372', ctypes.c_uint32),
    ('reserved_373', ctypes.c_uint32),
    ('reserved_374', ctypes.c_uint32),
    ('reserved_375', ctypes.c_uint32),
    ('reserved_376', ctypes.c_uint32),
    ('reserved_377', ctypes.c_uint32),
    ('reserved_378', ctypes.c_uint32),
    ('reserved_379', ctypes.c_uint32),
    ('reserved_380', ctypes.c_uint32),
    ('reserved_381', ctypes.c_uint32),
    ('reserved_382', ctypes.c_uint32),
    ('reserved_383', ctypes.c_uint32),
    ('reserved_384', ctypes.c_uint32),
    ('reserved_385', ctypes.c_uint32),
    ('reserved_386', ctypes.c_uint32),
    ('reserved_387', ctypes.c_uint32),
    ('reserved_388', ctypes.c_uint32),
    ('reserved_389', ctypes.c_uint32),
    ('reserved_390', ctypes.c_uint32),
    ('reserved_391', ctypes.c_uint32),
    ('reserved_392', ctypes.c_uint32),
    ('reserved_393', ctypes.c_uint32),
    ('reserved_394', ctypes.c_uint32),
    ('reserved_395', ctypes.c_uint32),
    ('reserved_396', ctypes.c_uint32),
    ('reserved_397', ctypes.c_uint32),
    ('reserved_398', ctypes.c_uint32),
    ('reserved_399', ctypes.c_uint32),
    ('reserved_400', ctypes.c_uint32),
    ('reserved_401', ctypes.c_uint32),
    ('reserved_402', ctypes.c_uint32),
    ('reserved_403', ctypes.c_uint32),
    ('reserved_404', ctypes.c_uint32),
    ('reserved_405', ctypes.c_uint32),
    ('reserved_406', ctypes.c_uint32),
    ('reserved_407', ctypes.c_uint32),
    ('reserved_408', ctypes.c_uint32),
    ('reserved_409', ctypes.c_uint32),
    ('reserved_410', ctypes.c_uint32),
    ('reserved_411', ctypes.c_uint32),
    ('reserved_412', ctypes.c_uint32),
    ('reserved_413', ctypes.c_uint32),
    ('reserved_414', ctypes.c_uint32),
    ('reserved_415', ctypes.c_uint32),
    ('reserved_416', ctypes.c_uint32),
    ('reserved_417', ctypes.c_uint32),
    ('reserved_418', ctypes.c_uint32),
    ('reserved_419', ctypes.c_uint32),
    ('reserved_420', ctypes.c_uint32),
    ('reserved_421', ctypes.c_uint32),
    ('reserved_422', ctypes.c_uint32),
    ('reserved_423', ctypes.c_uint32),
    ('reserved_424', ctypes.c_uint32),
    ('reserved_425', ctypes.c_uint32),
    ('reserved_426', ctypes.c_uint32),
    ('reserved_427', ctypes.c_uint32),
    ('reserved_428', ctypes.c_uint32),
    ('reserved_429', ctypes.c_uint32),
    ('reserved_430', ctypes.c_uint32),
    ('reserved_431', ctypes.c_uint32),
    ('reserved_432', ctypes.c_uint32),
    ('reserved_433', ctypes.c_uint32),
    ('reserved_434', ctypes.c_uint32),
    ('reserved_435', ctypes.c_uint32),
    ('reserved_436', ctypes.c_uint32),
    ('reserved_437', ctypes.c_uint32),
    ('reserved_438', ctypes.c_uint32),
    ('reserved_439', ctypes.c_uint32),
    ('reserved_440', ctypes.c_uint32),
    ('reserved_441', ctypes.c_uint32),
    ('reserved_442', ctypes.c_uint32),
    ('reserved_443', ctypes.c_uint32),
    ('reserved_444', ctypes.c_uint32),
    ('reserved_445', ctypes.c_uint32),
    ('reserved_446', ctypes.c_uint32),
    ('reserved_447', ctypes.c_uint32),
    ('gws_0_val', ctypes.c_uint32),
    ('gws_1_val', ctypes.c_uint32),
    ('gws_2_val', ctypes.c_uint32),
    ('gws_3_val', ctypes.c_uint32),
    ('gws_4_val', ctypes.c_uint32),
    ('gws_5_val', ctypes.c_uint32),
    ('gws_6_val', ctypes.c_uint32),
    ('gws_7_val', ctypes.c_uint32),
    ('gws_8_val', ctypes.c_uint32),
    ('gws_9_val', ctypes.c_uint32),
    ('gws_10_val', ctypes.c_uint32),
    ('gws_11_val', ctypes.c_uint32),
    ('gws_12_val', ctypes.c_uint32),
    ('gws_13_val', ctypes.c_uint32),
    ('gws_14_val', ctypes.c_uint32),
    ('gws_15_val', ctypes.c_uint32),
    ('gws_16_val', ctypes.c_uint32),
    ('gws_17_val', ctypes.c_uint32),
    ('gws_18_val', ctypes.c_uint32),
    ('gws_19_val', ctypes.c_uint32),
    ('gws_20_val', ctypes.c_uint32),
    ('gws_21_val', ctypes.c_uint32),
    ('gws_22_val', ctypes.c_uint32),
    ('gws_23_val', ctypes.c_uint32),
    ('gws_24_val', ctypes.c_uint32),
    ('gws_25_val', ctypes.c_uint32),
    ('gws_26_val', ctypes.c_uint32),
    ('gws_27_val', ctypes.c_uint32),
    ('gws_28_val', ctypes.c_uint32),
    ('gws_29_val', ctypes.c_uint32),
    ('gws_30_val', ctypes.c_uint32),
    ('gws_31_val', ctypes.c_uint32),
    ('gws_32_val', ctypes.c_uint32),
    ('gws_33_val', ctypes.c_uint32),
    ('gws_34_val', ctypes.c_uint32),
    ('gws_35_val', ctypes.c_uint32),
    ('gws_36_val', ctypes.c_uint32),
    ('gws_37_val', ctypes.c_uint32),
    ('gws_38_val', ctypes.c_uint32),
    ('gws_39_val', ctypes.c_uint32),
    ('gws_40_val', ctypes.c_uint32),
    ('gws_41_val', ctypes.c_uint32),
    ('gws_42_val', ctypes.c_uint32),
    ('gws_43_val', ctypes.c_uint32),
    ('gws_44_val', ctypes.c_uint32),
    ('gws_45_val', ctypes.c_uint32),
    ('gws_46_val', ctypes.c_uint32),
    ('gws_47_val', ctypes.c_uint32),
    ('gws_48_val', ctypes.c_uint32),
    ('gws_49_val', ctypes.c_uint32),
    ('gws_50_val', ctypes.c_uint32),
    ('gws_51_val', ctypes.c_uint32),
    ('gws_52_val', ctypes.c_uint32),
    ('gws_53_val', ctypes.c_uint32),
    ('gws_54_val', ctypes.c_uint32),
    ('gws_55_val', ctypes.c_uint32),
    ('gws_56_val', ctypes.c_uint32),
    ('gws_57_val', ctypes.c_uint32),
    ('gws_58_val', ctypes.c_uint32),
    ('gws_59_val', ctypes.c_uint32),
    ('gws_60_val', ctypes.c_uint32),
    ('gws_61_val', ctypes.c_uint32),
    ('gws_62_val', ctypes.c_uint32),
    ('gws_63_val', ctypes.c_uint32),
]

__AMDGPU_VM_H__ = True # macro
AMDGPU_VM_MAX_UPDATE_SIZE = 0x3FFFF # macro
# def AMDGPU_VM_PTE_COUNT(adev):  # macro
#    return (1<<(adev)->vm_manager.block_size)
AMDGPU_PTE_VALID = (1<<0) # macro
AMDGPU_PTE_SYSTEM = (1<<1) # macro
AMDGPU_PTE_SNOOPED = (1<<2) # macro
AMDGPU_PTE_TMZ = (1<<3) # macro
AMDGPU_PTE_EXECUTABLE = (1<<4) # macro
AMDGPU_PTE_READABLE = (1<<5) # macro
AMDGPU_PTE_WRITEABLE = (1<<6) # macro
def AMDGPU_PTE_FRAG(x):  # macro
   return ((x&0x1f)<<7)
AMDGPU_PTE_PRT = (1<<51) # macro
AMDGPU_PDE_PTE = (1<<54) # macro
AMDGPU_PTE_LOG = (1<<55) # macro
AMDGPU_PTE_TF = (1<<56) # macro
AMDGPU_PTE_NOALLOC = (1<<58) # macro
def AMDGPU_PDE_BFS(a):  # macro
   return ( a<<59)
AMDGPU_VM_NORETRY_FLAGS = ((1<<4)|(1<<54)|(1<<56)) # macro
AMDGPU_VM_NORETRY_FLAGS_TF = ((1<<0)|(1<<1)|(1<<51)) # macro
def AMDGPU_PTE_MTYPE_VG10_SHIFT(mtype):  # macro
   return ( (mtype)<<57)
AMDGPU_PTE_MTYPE_VG10_MASK = AMDGPU_PTE_MTYPE_VG10_SHIFT ( 3 ) # macro
def AMDGPU_PTE_MTYPE_VG10(flags, mtype):  # macro
   return (( (flags)&(~AMDGPU_PTE_MTYPE_VG10_SHIFT(3)))|AMDGPU_PTE_MTYPE_VG10_SHIFT(mtype))
AMDGPU_MTYPE_NC = 0 # macro
AMDGPU_MTYPE_CC = 2 # macro
AMDGPU_PTE_DEFAULT_ATC = ((1<<1)|(1<<2)|(1<<4)|(1<<5)|(1<<6)|AMDGPU_PTE_MTYPE_VG10(0, 2)) # macro
def AMDGPU_PTE_MTYPE_NV10_SHIFT(mtype):  # macro
   return ( (mtype)<<48)
AMDGPU_PTE_MTYPE_NV10_MASK = AMDGPU_PTE_MTYPE_NV10_SHIFT ( 7 ) # macro
def AMDGPU_PTE_MTYPE_NV10(flags, mtype):  # macro
   return (( (flags)&(~AMDGPU_PTE_MTYPE_NV10_SHIFT(7)))|AMDGPU_PTE_MTYPE_NV10_SHIFT(mtype))
AMDGPU_PTE_PRT_GFX12 = (1<<56) # macro
def AMDGPU_PTE_MTYPE_GFX12_SHIFT(mtype):  # macro
   return ( (mtype)<<54)
AMDGPU_PTE_MTYPE_GFX12_MASK = AMDGPU_PTE_MTYPE_GFX12_SHIFT ( 3 ) # macro
def AMDGPU_PTE_MTYPE_GFX12(flags, mtype):  # macro
   return (( (flags)&(~AMDGPU_PTE_MTYPE_GFX12_SHIFT(3)))|AMDGPU_PTE_MTYPE_GFX12_SHIFT(mtype))
AMDGPU_PTE_IS_PTE = (1<<63) # macro
def AMDGPU_PDE_BFS_GFX12(a):  # macro
   return ( ((a)&0x1f)<<58)
AMDGPU_PDE_PTE_GFX12 = (1<<63) # macro
AMDGPU_VM_FAULT_STOP_NEVER = 0 # macro
AMDGPU_VM_FAULT_STOP_FIRST = 1 # macro
AMDGPU_VM_FAULT_STOP_ALWAYS = 2 # macro
AMDGPU_VM_RESERVED_VRAM = (8<<20) # macro
AMDGPU_MAX_VMHUBS = 13 # macro
AMDGPU_GFXHUB_START = 0 # macro
AMDGPU_MMHUB0_START = 8 # macro
AMDGPU_MMHUB1_START = 12 # macro
def AMDGPU_GFXHUB(x):  # macro
   return (0+(x))
def AMDGPU_MMHUB0(x):  # macro
   return (8+(x))
def AMDGPU_MMHUB1(x):  # macro
   return (12+(x))
def AMDGPU_IS_GFXHUB(x):  # macro
   return ((x)>=0 and (x)<8)
def AMDGPU_IS_MMHUB0(x):  # macro
   return ((x)>=8 and (x)<12)
def AMDGPU_IS_MMHUB1(x):  # macro
   return ((x)>=12 and (x)<13)
AMDGPU_VA_RESERVED_CSA_SIZE = (2<<20) # macro
# def AMDGPU_VA_RESERVED_CSA_START(adev):  # macro
#    return (((adev)->vm_manager.max_pfn<<AMDGPU_GPU_PAGE_SHIFT)-(2<<20))
AMDGPU_VA_RESERVED_SEQ64_SIZE = (2<<20) # macro
def AMDGPU_VA_RESERVED_SEQ64_START(adev):  # macro
   return (AMDGPU_VA_RESERVED_CSA_START(adev)-(2<<20))
AMDGPU_VA_RESERVED_TRAP_SIZE = (2<<12) # macro
def AMDGPU_VA_RESERVED_TRAP_START(adev):  # macro
   return (AMDGPU_VA_RESERVED_SEQ64_START(adev)-(2<<12))
AMDGPU_VA_RESERVED_BOTTOM = (1<<16) # macro
AMDGPU_VA_RESERVED_TOP = ((2<<12)+(2<<20)+(2<<20)) # macro
AMDGPU_VM_USE_CPU_FOR_GFX = (1<<0) # macro
AMDGPU_VM_USE_CPU_FOR_COMPUTE = (1<<1) # macro

# values for enumeration 'amdgpu_vm_level'
amdgpu_vm_level__enumvalues = {
    0: 'AMDGPU_VM_PDB2',
    1: 'AMDGPU_VM_PDB1',
    2: 'AMDGPU_VM_PDB0',
    3: 'AMDGPU_VM_PTB',
}
AMDGPU_VM_PDB2 = 0
AMDGPU_VM_PDB1 = 1
AMDGPU_VM_PDB0 = 2
AMDGPU_VM_PTB = 3
amdgpu_vm_level = ctypes.c_uint32 # enum
_DISCOVERY_H_ = True # macro
PSP_HEADER_SIZE = 256 # macro
BINARY_SIGNATURE = 0x28211407 # macro
DISCOVERY_TABLE_SIGNATURE = 0x53445049 # macro
GC_TABLE_ID = 0x4347 # macro
HARVEST_TABLE_SIGNATURE = 0x56524148 # macro
VCN_INFO_TABLE_ID = 0x004E4356 # macro
MALL_INFO_TABLE_ID = 0x4C4C414D # macro
NPS_INFO_TABLE_ID = 0x0053504E # macro
VCN_INFO_TABLE_MAX_NUM_INSTANCES = 4 # macro
NPS_INFO_TABLE_MAX_NUM_INSTANCES = 12 # macro
HWIP_MAX_INSTANCE = 44 # macro
HW_ID_MAX = 300 # macro
MP1_HWID = 1 # macro
MP2_HWID = 2 # macro
THM_HWID = 3 # macro
SMUIO_HWID = 4 # macro
FUSE_HWID = 5 # macro
CLKA_HWID = 6 # macro
PWR_HWID = 10 # macro
GC_HWID = 11 # macro
UVD_HWID = 12 # macro
VCN_HWID = 12 # macro
AUDIO_AZ_HWID = 13 # macro
ACP_HWID = 14 # macro
DCI_HWID = 15 # macro
DMU_HWID = 271 # macro
DCO_HWID = 16 # macro
DIO_HWID = 272 # macro
XDMA_HWID = 17 # macro
DCEAZ_HWID = 18 # macro
DAZ_HWID = 274 # macro
SDPMUX_HWID = 19 # macro
NTB_HWID = 20 # macro
VPE_HWID = 21 # macro
IOHC_HWID = 24 # macro
L2IMU_HWID = 28 # macro
VCE_HWID = 32 # macro
MMHUB_HWID = 34 # macro
ATHUB_HWID = 35 # macro
DBGU_NBIO_HWID = 36 # macro
DFX_HWID = 37 # macro
DBGU0_HWID = 38 # macro
DBGU1_HWID = 39 # macro
OSSSYS_HWID = 40 # macro
HDP_HWID = 41 # macro
SDMA0_HWID = 42 # macro
SDMA1_HWID = 43 # macro
ISP_HWID = 44 # macro
DBGU_IO_HWID = 45 # macro
DF_HWID = 46 # macro
CLKB_HWID = 47 # macro
FCH_HWID = 48 # macro
DFX_DAP_HWID = 49 # macro
L1IMU_PCIE_HWID = 50 # macro
L1IMU_NBIF_HWID = 51 # macro
L1IMU_IOAGR_HWID = 52 # macro
L1IMU3_HWID = 53 # macro
L1IMU4_HWID = 54 # macro
L1IMU5_HWID = 55 # macro
L1IMU6_HWID = 56 # macro
L1IMU7_HWID = 57 # macro
L1IMU8_HWID = 58 # macro
L1IMU9_HWID = 59 # macro
L1IMU10_HWID = 60 # macro
L1IMU11_HWID = 61 # macro
L1IMU12_HWID = 62 # macro
L1IMU13_HWID = 63 # macro
L1IMU14_HWID = 64 # macro
L1IMU15_HWID = 65 # macro
WAFLC_HWID = 66 # macro
FCH_USB_PD_HWID = 67 # macro
SDMA2_HWID = 68 # macro
SDMA3_HWID = 69 # macro
PCIE_HWID = 70 # macro
PCS_HWID = 80 # macro
DDCL_HWID = 89 # macro
SST_HWID = 90 # macro
LSDMA_HWID = 91 # macro
IOAGR_HWID = 100 # macro
NBIF_HWID = 108 # macro
IOAPIC_HWID = 124 # macro
SYSTEMHUB_HWID = 128 # macro
NTBCCP_HWID = 144 # macro
UMC_HWID = 150 # macro
SATA_HWID = 168 # macro
USB_HWID = 170 # macro
CCXSEC_HWID = 176 # macro
XGMI_HWID = 200 # macro
XGBE_HWID = 216 # macro
MP0_HWID = 255 # macro

# values for enumeration 'c__EA_table'
c__EA_table__enumvalues = {
    0: 'IP_DISCOVERY',
    1: 'GC',
    2: 'HARVEST_INFO',
    3: 'VCN_INFO',
    4: 'MALL_INFO',
    5: 'NPS_INFO',
    6: 'TOTAL_TABLES',
}
IP_DISCOVERY = 0
GC = 1
HARVEST_INFO = 2
VCN_INFO = 3
MALL_INFO = 4
NPS_INFO = 5
TOTAL_TABLES = 6
c__EA_table = ctypes.c_uint32 # enum
table = c__EA_table
table__enumvalues = c__EA_table__enumvalues
class struct_table_info(Structure):
    pass

struct_table_info._pack_ = 1 # source:False
struct_table_info._fields_ = [
    ('offset', ctypes.c_uint16),
    ('checksum', ctypes.c_uint16),
    ('size', ctypes.c_uint16),
    ('padding', ctypes.c_uint16),
]

table_info = struct_table_info
class struct_binary_header(Structure):
    pass

struct_binary_header._pack_ = 1 # source:False
struct_binary_header._fields_ = [
    ('binary_signature', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('binary_checksum', ctypes.c_uint16),
    ('binary_size', ctypes.c_uint16),
    ('table_list', struct_table_info * 6),
]

binary_header = struct_binary_header
class struct_die_info(Structure):
    pass

struct_die_info._pack_ = 1 # source:False
struct_die_info._fields_ = [
    ('die_id', ctypes.c_uint16),
    ('die_offset', ctypes.c_uint16),
]

die_info = struct_die_info
class struct_ip_discovery_header(Structure):
    pass

class union_ip_discovery_header_0(Union):
    pass

class struct_ip_discovery_header_0_0(Structure):
    pass

struct_ip_discovery_header_0_0._pack_ = 1 # source:False
struct_ip_discovery_header_0_0._fields_ = [
    ('base_addr_64_bit', ctypes.c_ubyte, 1),
    ('reserved', ctypes.c_ubyte, 7),
    ('reserved2', ctypes.c_ubyte, 8),
]

union_ip_discovery_header_0._pack_ = 1 # source:False
union_ip_discovery_header_0._anonymous_ = ('_0',)
union_ip_discovery_header_0._fields_ = [
    ('padding', ctypes.c_uint16 * 1),
    ('_0', struct_ip_discovery_header_0_0),
]

struct_ip_discovery_header._pack_ = 1 # source:False
struct_ip_discovery_header._anonymous_ = ('_0',)
struct_ip_discovery_header._fields_ = [
    ('signature', ctypes.c_uint32),
    ('version', ctypes.c_uint16),
    ('size', ctypes.c_uint16),
    ('id', ctypes.c_uint32),
    ('num_dies', ctypes.c_uint16),
    ('die_info', struct_die_info * 16),
    ('_0', union_ip_discovery_header_0),
]

ip_discovery_header = struct_ip_discovery_header
class struct_ip(Structure):
    pass

struct_ip._pack_ = 1 # source:False
struct_ip._fields_ = [
    ('hw_id', ctypes.c_uint16),
    ('number_instance', ctypes.c_ubyte),
    ('num_base_address', ctypes.c_ubyte),
    ('major', ctypes.c_ubyte),
    ('minor', ctypes.c_ubyte),
    ('revision', ctypes.c_ubyte),
    ('harvest', ctypes.c_ubyte, 4),
    ('reserved', ctypes.c_ubyte, 4),
    ('base_address', ctypes.c_uint32 * 0),
]

ip = struct_ip
class struct_ip_v3(Structure):
    pass

struct_ip_v3._pack_ = 1 # source:False
struct_ip_v3._fields_ = [
    ('hw_id', ctypes.c_uint16),
    ('instance_number', ctypes.c_ubyte),
    ('num_base_address', ctypes.c_ubyte),
    ('major', ctypes.c_ubyte),
    ('minor', ctypes.c_ubyte),
    ('revision', ctypes.c_ubyte),
    ('sub_revision', ctypes.c_ubyte, 4),
    ('variant', ctypes.c_ubyte, 4),
    ('base_address', ctypes.c_uint32 * 0),
]

ip_v3 = struct_ip_v3
class struct_ip_v4(Structure):
    pass

struct_ip_v4._pack_ = 1 # source:False
struct_ip_v4._fields_ = [
    ('hw_id', ctypes.c_uint16),
    ('instance_number', ctypes.c_ubyte),
    ('num_base_address', ctypes.c_ubyte),
    ('major', ctypes.c_ubyte),
    ('minor', ctypes.c_ubyte),
    ('revision', ctypes.c_ubyte),
]

ip_v4 = struct_ip_v4
class struct_die_header(Structure):
    pass

struct_die_header._pack_ = 1 # source:False
struct_die_header._fields_ = [
    ('die_id', ctypes.c_uint16),
    ('num_ips', ctypes.c_uint16),
]

die_header = struct_die_header
class struct_ip_structure(Structure):
    pass

class struct_die(Structure):
    pass

class union_die_0(Union):
    pass

union_die_0._pack_ = 1 # source:False
union_die_0._fields_ = [
    ('ip_list', ctypes.POINTER(struct_ip)),
    ('ip_v3_list', ctypes.POINTER(struct_ip_v3)),
    ('ip_v4_list', ctypes.POINTER(struct_ip_v4)),
]

struct_die._pack_ = 1 # source:False
struct_die._anonymous_ = ('_0',)
struct_die._fields_ = [
    ('die_header', ctypes.POINTER(struct_die_header)),
    ('_0', union_die_0),
]

struct_ip_structure._pack_ = 1 # source:False
struct_ip_structure._fields_ = [
    ('header', ctypes.POINTER(struct_ip_discovery_header)),
    ('die', struct_die),
]

ip_structure = struct_ip_structure
class struct_gpu_info_header(Structure):
    pass

struct_gpu_info_header._pack_ = 1 # source:False
struct_gpu_info_header._fields_ = [
    ('table_id', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('size', ctypes.c_uint32),
]

class struct_gc_info_v1_0(Structure):
    pass

struct_gc_info_v1_0._pack_ = 1 # source:False
struct_gc_info_v1_0._fields_ = [
    ('header', struct_gpu_info_header),
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_wgp0_per_sa', ctypes.c_uint32),
    ('gc_num_wgp1_per_sa', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_gl2c', ctypes.c_uint32),
    ('gc_num_gprs', ctypes.c_uint32),
    ('gc_num_max_gs_thds', ctypes.c_uint32),
    ('gc_gs_table_depth', ctypes.c_uint32),
    ('gc_gsprim_buff_depth', ctypes.c_uint32),
    ('gc_parameter_cache_depth', ctypes.c_uint32),
    ('gc_double_offchip_lds_buffer', ctypes.c_uint32),
    ('gc_wave_size', ctypes.c_uint32),
    ('gc_max_waves_per_simd', ctypes.c_uint32),
    ('gc_max_scratch_slots_per_cu', ctypes.c_uint32),
    ('gc_lds_size', ctypes.c_uint32),
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_sa_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
    ('gc_num_gl2a', ctypes.c_uint32),
]

class struct_gc_info_v1_1(Structure):
    pass

struct_gc_info_v1_1._pack_ = 1 # source:False
struct_gc_info_v1_1._fields_ = [
    ('header', struct_gpu_info_header),
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_wgp0_per_sa', ctypes.c_uint32),
    ('gc_num_wgp1_per_sa', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_gl2c', ctypes.c_uint32),
    ('gc_num_gprs', ctypes.c_uint32),
    ('gc_num_max_gs_thds', ctypes.c_uint32),
    ('gc_gs_table_depth', ctypes.c_uint32),
    ('gc_gsprim_buff_depth', ctypes.c_uint32),
    ('gc_parameter_cache_depth', ctypes.c_uint32),
    ('gc_double_offchip_lds_buffer', ctypes.c_uint32),
    ('gc_wave_size', ctypes.c_uint32),
    ('gc_max_waves_per_simd', ctypes.c_uint32),
    ('gc_max_scratch_slots_per_cu', ctypes.c_uint32),
    ('gc_lds_size', ctypes.c_uint32),
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_sa_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
    ('gc_num_gl2a', ctypes.c_uint32),
    ('gc_num_tcp_per_sa', ctypes.c_uint32),
    ('gc_num_sdp_interface', ctypes.c_uint32),
    ('gc_num_tcps', ctypes.c_uint32),
]

class struct_gc_info_v1_2(Structure):
    pass

struct_gc_info_v1_2._pack_ = 1 # source:False
struct_gc_info_v1_2._fields_ = [
    ('header', struct_gpu_info_header),
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_wgp0_per_sa', ctypes.c_uint32),
    ('gc_num_wgp1_per_sa', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_gl2c', ctypes.c_uint32),
    ('gc_num_gprs', ctypes.c_uint32),
    ('gc_num_max_gs_thds', ctypes.c_uint32),
    ('gc_gs_table_depth', ctypes.c_uint32),
    ('gc_gsprim_buff_depth', ctypes.c_uint32),
    ('gc_parameter_cache_depth', ctypes.c_uint32),
    ('gc_double_offchip_lds_buffer', ctypes.c_uint32),
    ('gc_wave_size', ctypes.c_uint32),
    ('gc_max_waves_per_simd', ctypes.c_uint32),
    ('gc_max_scratch_slots_per_cu', ctypes.c_uint32),
    ('gc_lds_size', ctypes.c_uint32),
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_sa_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
    ('gc_num_gl2a', ctypes.c_uint32),
    ('gc_num_tcp_per_sa', ctypes.c_uint32),
    ('gc_num_sdp_interface', ctypes.c_uint32),
    ('gc_num_tcps', ctypes.c_uint32),
    ('gc_num_tcp_per_wpg', ctypes.c_uint32),
    ('gc_tcp_l1_size', ctypes.c_uint32),
    ('gc_num_sqc_per_wgp', ctypes.c_uint32),
    ('gc_l1_instruction_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_l1_data_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_gl1c_per_sa', ctypes.c_uint32),
    ('gc_gl1c_size_per_instance', ctypes.c_uint32),
    ('gc_gl2c_per_gpu', ctypes.c_uint32),
]

class struct_gc_info_v1_3(Structure):
    pass

struct_gc_info_v1_3._pack_ = 1 # source:False
struct_gc_info_v1_3._fields_ = [
    ('header', struct_gpu_info_header),
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_wgp0_per_sa', ctypes.c_uint32),
    ('gc_num_wgp1_per_sa', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_gl2c', ctypes.c_uint32),
    ('gc_num_gprs', ctypes.c_uint32),
    ('gc_num_max_gs_thds', ctypes.c_uint32),
    ('gc_gs_table_depth', ctypes.c_uint32),
    ('gc_gsprim_buff_depth', ctypes.c_uint32),
    ('gc_parameter_cache_depth', ctypes.c_uint32),
    ('gc_double_offchip_lds_buffer', ctypes.c_uint32),
    ('gc_wave_size', ctypes.c_uint32),
    ('gc_max_waves_per_simd', ctypes.c_uint32),
    ('gc_max_scratch_slots_per_cu', ctypes.c_uint32),
    ('gc_lds_size', ctypes.c_uint32),
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_sa_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
    ('gc_num_gl2a', ctypes.c_uint32),
    ('gc_num_tcp_per_sa', ctypes.c_uint32),
    ('gc_num_sdp_interface', ctypes.c_uint32),
    ('gc_num_tcps', ctypes.c_uint32),
    ('gc_num_tcp_per_wpg', ctypes.c_uint32),
    ('gc_tcp_l1_size', ctypes.c_uint32),
    ('gc_num_sqc_per_wgp', ctypes.c_uint32),
    ('gc_l1_instruction_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_l1_data_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_gl1c_per_sa', ctypes.c_uint32),
    ('gc_gl1c_size_per_instance', ctypes.c_uint32),
    ('gc_gl2c_per_gpu', ctypes.c_uint32),
    ('gc_tcp_size_per_cu', ctypes.c_uint32),
    ('gc_tcp_cache_line_size', ctypes.c_uint32),
    ('gc_instruction_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_instruction_cache_line_size', ctypes.c_uint32),
    ('gc_scalar_data_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_scalar_data_cache_line_size', ctypes.c_uint32),
    ('gc_tcc_size', ctypes.c_uint32),
    ('gc_tcc_cache_line_size', ctypes.c_uint32),
]

class struct_gc_info_v2_0(Structure):
    pass

struct_gc_info_v2_0._pack_ = 1 # source:False
struct_gc_info_v2_0._fields_ = [
    ('header', struct_gpu_info_header),
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_cu_per_sh', ctypes.c_uint32),
    ('gc_num_sh_per_se', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_tccs', ctypes.c_uint32),
    ('gc_num_gprs', ctypes.c_uint32),
    ('gc_num_max_gs_thds', ctypes.c_uint32),
    ('gc_gs_table_depth', ctypes.c_uint32),
    ('gc_gsprim_buff_depth', ctypes.c_uint32),
    ('gc_parameter_cache_depth', ctypes.c_uint32),
    ('gc_double_offchip_lds_buffer', ctypes.c_uint32),
    ('gc_wave_size', ctypes.c_uint32),
    ('gc_max_waves_per_simd', ctypes.c_uint32),
    ('gc_max_scratch_slots_per_cu', ctypes.c_uint32),
    ('gc_lds_size', ctypes.c_uint32),
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
]

class struct_gc_info_v2_1(Structure):
    pass

struct_gc_info_v2_1._pack_ = 1 # source:False
struct_gc_info_v2_1._fields_ = [
    ('header', struct_gpu_info_header),
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_cu_per_sh', ctypes.c_uint32),
    ('gc_num_sh_per_se', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_tccs', ctypes.c_uint32),
    ('gc_num_gprs', ctypes.c_uint32),
    ('gc_num_max_gs_thds', ctypes.c_uint32),
    ('gc_gs_table_depth', ctypes.c_uint32),
    ('gc_gsprim_buff_depth', ctypes.c_uint32),
    ('gc_parameter_cache_depth', ctypes.c_uint32),
    ('gc_double_offchip_lds_buffer', ctypes.c_uint32),
    ('gc_wave_size', ctypes.c_uint32),
    ('gc_max_waves_per_simd', ctypes.c_uint32),
    ('gc_max_scratch_slots_per_cu', ctypes.c_uint32),
    ('gc_lds_size', ctypes.c_uint32),
    ('gc_num_sc_per_se', ctypes.c_uint32),
    ('gc_num_packer_per_sc', ctypes.c_uint32),
    ('gc_num_tcp_per_sh', ctypes.c_uint32),
    ('gc_tcp_size_per_cu', ctypes.c_uint32),
    ('gc_num_sdp_interface', ctypes.c_uint32),
    ('gc_num_cu_per_sqc', ctypes.c_uint32),
    ('gc_instruction_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_scalar_data_cache_size_per_sqc', ctypes.c_uint32),
    ('gc_tcc_size', ctypes.c_uint32),
]

class struct_harvest_info_header(Structure):
    pass

struct_harvest_info_header._pack_ = 1 # source:False
struct_harvest_info_header._fields_ = [
    ('signature', ctypes.c_uint32),
    ('version', ctypes.c_uint32),
]

harvest_info_header = struct_harvest_info_header
class struct_harvest_info(Structure):
    pass

struct_harvest_info._pack_ = 1 # source:False
struct_harvest_info._fields_ = [
    ('hw_id', ctypes.c_uint16),
    ('number_instance', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

harvest_info = struct_harvest_info
class struct_harvest_table(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('header', harvest_info_header),
    ('list', struct_harvest_info * 32),
     ]

harvest_table = struct_harvest_table
class struct_mall_info_header(Structure):
    pass

struct_mall_info_header._pack_ = 1 # source:False
struct_mall_info_header._fields_ = [
    ('table_id', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('size_bytes', ctypes.c_uint32),
]

class struct_mall_info_v1_0(Structure):
    pass

struct_mall_info_v1_0._pack_ = 1 # source:False
struct_mall_info_v1_0._fields_ = [
    ('header', struct_mall_info_header),
    ('mall_size_per_m', ctypes.c_uint32),
    ('m_s_present', ctypes.c_uint32),
    ('m_half_use', ctypes.c_uint32),
    ('m_mall_config', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 5),
]

class struct_mall_info_v2_0(Structure):
    pass

struct_mall_info_v2_0._pack_ = 1 # source:False
struct_mall_info_v2_0._fields_ = [
    ('header', struct_mall_info_header),
    ('mall_size_per_umc', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 8),
]

class struct_vcn_info_header(Structure):
    pass

struct_vcn_info_header._pack_ = 1 # source:False
struct_vcn_info_header._fields_ = [
    ('table_id', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('size_bytes', ctypes.c_uint32),
]

class struct_vcn_instance_info_v1_0(Structure):
    pass

class union__fuse_data(Union):
    pass

class struct__fuse_data_bits(Structure):
    pass

struct__fuse_data_bits._pack_ = 1 # source:False
struct__fuse_data_bits._fields_ = [
    ('av1_disabled', ctypes.c_uint32, 1),
    ('vp9_disabled', ctypes.c_uint32, 1),
    ('hevc_disabled', ctypes.c_uint32, 1),
    ('h264_disabled', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 28),
]

union__fuse_data._pack_ = 1 # source:False
union__fuse_data._fields_ = [
    ('bits', struct__fuse_data_bits),
    ('all_bits', ctypes.c_uint32),
]

struct_vcn_instance_info_v1_0._pack_ = 1 # source:False
struct_vcn_instance_info_v1_0._fields_ = [
    ('instance_num', ctypes.c_uint32),
    ('fuse_data', union__fuse_data),
    ('reserved', ctypes.c_uint32 * 2),
]

class struct_vcn_info_v1_0(Structure):
    pass

struct_vcn_info_v1_0._pack_ = 1 # source:False
struct_vcn_info_v1_0._fields_ = [
    ('header', struct_vcn_info_header),
    ('num_of_instances', ctypes.c_uint32),
    ('instance_info', struct_vcn_instance_info_v1_0 * 4),
    ('reserved', ctypes.c_uint32 * 4),
]

class struct_nps_info_header(Structure):
    pass

struct_nps_info_header._pack_ = 1 # source:False
struct_nps_info_header._fields_ = [
    ('table_id', ctypes.c_uint32),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
    ('size_bytes', ctypes.c_uint32),
]

class struct_nps_instance_info_v1_0(Structure):
    pass

struct_nps_instance_info_v1_0._pack_ = 1 # source:False
struct_nps_instance_info_v1_0._fields_ = [
    ('base_address', ctypes.c_uint64),
    ('limit_address', ctypes.c_uint64),
]

class struct_nps_info_v1_0(Structure):
    pass

struct_nps_info_v1_0._pack_ = 1 # source:False
struct_nps_info_v1_0._fields_ = [
    ('header', struct_nps_info_header),
    ('nps_type', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
    ('instance_info', struct_nps_instance_info_v1_0 * 12),
]


# values for enumeration 'amd_hw_ip_block_type'
amd_hw_ip_block_type__enumvalues = {
    1: 'GC_HWIP',
    2: 'HDP_HWIP',
    3: 'SDMA0_HWIP',
    4: 'SDMA1_HWIP',
    5: 'SDMA2_HWIP',
    6: 'SDMA3_HWIP',
    7: 'SDMA4_HWIP',
    8: 'SDMA5_HWIP',
    9: 'SDMA6_HWIP',
    10: 'SDMA7_HWIP',
    11: 'LSDMA_HWIP',
    12: 'MMHUB_HWIP',
    13: 'ATHUB_HWIP',
    14: 'NBIO_HWIP',
    15: 'MP0_HWIP',
    16: 'MP1_HWIP',
    17: 'UVD_HWIP',
    17: 'VCN_HWIP',
    17: 'JPEG_HWIP',
    18: 'VCN1_HWIP',
    19: 'VCE_HWIP',
    20: 'VPE_HWIP',
    21: 'DF_HWIP',
    22: 'DCE_HWIP',
    23: 'OSSSYS_HWIP',
    24: 'SMUIO_HWIP',
    25: 'PWR_HWIP',
    26: 'NBIF_HWIP',
    27: 'THM_HWIP',
    28: 'CLK_HWIP',
    29: 'UMC_HWIP',
    30: 'RSMU_HWIP',
    31: 'XGMI_HWIP',
    32: 'DCI_HWIP',
    33: 'PCIE_HWIP',
    34: 'ISP_HWIP',
    35: 'MAX_HWIP',
}
GC_HWIP = 1
HDP_HWIP = 2
SDMA0_HWIP = 3
SDMA1_HWIP = 4
SDMA2_HWIP = 5
SDMA3_HWIP = 6
SDMA4_HWIP = 7
SDMA5_HWIP = 8
SDMA6_HWIP = 9
SDMA7_HWIP = 10
LSDMA_HWIP = 11
MMHUB_HWIP = 12
ATHUB_HWIP = 13
NBIO_HWIP = 14
MP0_HWIP = 15
MP1_HWIP = 16
UVD_HWIP = 17
VCN_HWIP = 17
JPEG_HWIP = 17
VCN1_HWIP = 18
VCE_HWIP = 19
VPE_HWIP = 20
DF_HWIP = 21
DCE_HWIP = 22
OSSSYS_HWIP = 23
SMUIO_HWIP = 24
PWR_HWIP = 25
NBIF_HWIP = 26
THM_HWIP = 27
CLK_HWIP = 28
UMC_HWIP = 29
RSMU_HWIP = 30
XGMI_HWIP = 31
DCI_HWIP = 32
PCIE_HWIP = 33
ISP_HWIP = 34
MAX_HWIP = 35
amd_hw_ip_block_type = ctypes.c_uint32 # enum
# def AMDGPU_PTE_PRT_FLAG(adev):  # macro
#    return ((amdgpu_ip_version((adev),GC_HWIP,0)>=IP_VERSION(12,0,0))?(1<<56):(1<<51))
# def AMDGPU_PDE_BFS_FLAG(adev, a):  # macro
#    return ((amdgpu_ip_version((adev),GC_HWIP,0)>=IP_VERSION(12,0,0))?AMDGPU_PDE_BFS_GFX12(a):AMDGPU_PDE_BFS(a))
# def AMDGPU_PDE_PTE_FLAG(adev):  # macro
#    return ((amdgpu_ip_version((adev),GC_HWIP,0)>=IP_VERSION(12,0,0))?(1<<63):(1<<54))
hw_id_map = [['GC_HWIP', '11'],['HDP_HWIP', '41'],['SDMA0_HWIP', '42'],['SDMA1_HWIP', '43'],['SDMA2_HWIP', '68'],['SDMA3_HWIP', '69'],['LSDMA_HWIP', '91'],['MMHUB_HWIP', '34'],['ATHUB_HWIP', '35'],['NBIO_HWIP', '108'],['MP0_HWIP', '255'],['MP1_HWIP', '1'],['UVD_HWIP', '12'],['VCE_HWIP', '32'],['DF_HWIP', '46'],['DCE_HWIP', '271'],['OSSSYS_HWIP', '40'],['SMUIO_HWIP', '4'],['PWR_HWIP', '10'],['NBIF_HWIP', '108'],['THM_HWIP', '3'],['CLK_HWIP', '6'],['UMC_HWIP', '150'],['XGMI_HWIP', '200'],['DCI_HWIP', '15'],['PCIE_HWIP', '70'],['VPE_HWIP', '21'],['ISP_HWIP', '44']] # Variable ctypes.c_int32 * 35
__AMDGPU_UCODE_H__ = True # macro
int32_t = True # macro
int8_t = True # macro
int16_t = True # macro
bool = True # macro
u32 = True # macro
AMDGPU_SDMA0_UCODE_LOADED = 0x00000001 # macro
AMDGPU_SDMA1_UCODE_LOADED = 0x00000002 # macro
AMDGPU_CPCE_UCODE_LOADED = 0x00000004 # macro
AMDGPU_CPPFP_UCODE_LOADED = 0x00000008 # macro
AMDGPU_CPME_UCODE_LOADED = 0x00000010 # macro
AMDGPU_CPMEC1_UCODE_LOADED = 0x00000020 # macro
AMDGPU_CPMEC2_UCODE_LOADED = 0x00000040 # macro
AMDGPU_CPRLC_UCODE_LOADED = 0x00000100 # macro
class struct_common_firmware_header(Structure):
    pass

struct_common_firmware_header._pack_ = 1 # source:False
struct_common_firmware_header._fields_ = [
    ('size_bytes', ctypes.c_uint32),
    ('header_size_bytes', ctypes.c_uint32),
    ('header_version_major', ctypes.c_uint16),
    ('header_version_minor', ctypes.c_uint16),
    ('ip_version_major', ctypes.c_uint16),
    ('ip_version_minor', ctypes.c_uint16),
    ('ucode_version', ctypes.c_uint32),
    ('ucode_size_bytes', ctypes.c_uint32),
    ('ucode_array_offset_bytes', ctypes.c_uint32),
    ('crc32', ctypes.c_uint32),
]

class struct_mc_firmware_header_v1_0(Structure):
    pass

struct_mc_firmware_header_v1_0._pack_ = 1 # source:False
struct_mc_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('io_debug_size_bytes', ctypes.c_uint32),
    ('io_debug_array_offset_bytes', ctypes.c_uint32),
]

class struct_smc_firmware_header_v1_0(Structure):
    pass

struct_smc_firmware_header_v1_0._pack_ = 1 # source:False
struct_smc_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_start_addr', ctypes.c_uint32),
]

class struct_smc_firmware_header_v2_0(Structure):
    pass

struct_smc_firmware_header_v2_0._pack_ = 1 # source:False
struct_smc_firmware_header_v2_0._fields_ = [
    ('v1_0', struct_smc_firmware_header_v1_0),
    ('ppt_offset_bytes', ctypes.c_uint32),
    ('ppt_size_bytes', ctypes.c_uint32),
]

class struct_smc_soft_pptable_entry(Structure):
    pass

struct_smc_soft_pptable_entry._pack_ = 1 # source:False
struct_smc_soft_pptable_entry._fields_ = [
    ('id', ctypes.c_uint32),
    ('ppt_offset_bytes', ctypes.c_uint32),
    ('ppt_size_bytes', ctypes.c_uint32),
]

class struct_smc_firmware_header_v2_1(Structure):
    pass

struct_smc_firmware_header_v2_1._pack_ = 1 # source:False
struct_smc_firmware_header_v2_1._fields_ = [
    ('v1_0', struct_smc_firmware_header_v1_0),
    ('pptable_count', ctypes.c_uint32),
    ('pptable_entry_offset', ctypes.c_uint32),
]

class struct_psp_fw_legacy_bin_desc(Structure):
    pass

struct_psp_fw_legacy_bin_desc._pack_ = 1 # source:False
struct_psp_fw_legacy_bin_desc._fields_ = [
    ('fw_version', ctypes.c_uint32),
    ('offset_bytes', ctypes.c_uint32),
    ('size_bytes', ctypes.c_uint32),
]

class struct_psp_firmware_header_v1_0(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('header', struct_common_firmware_header),
    ('sos', struct_psp_fw_legacy_bin_desc),
     ]

class struct_psp_firmware_header_v1_1(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('v1_0', struct_psp_firmware_header_v1_0),
    ('toc', struct_psp_fw_legacy_bin_desc),
    ('kdb', struct_psp_fw_legacy_bin_desc),
     ]

class struct_psp_firmware_header_v1_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('v1_0', struct_psp_firmware_header_v1_0),
    ('res', struct_psp_fw_legacy_bin_desc),
    ('kdb', struct_psp_fw_legacy_bin_desc),
     ]

class struct_psp_firmware_header_v1_3(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('v1_1', struct_psp_firmware_header_v1_1),
    ('spl', struct_psp_fw_legacy_bin_desc),
    ('rl', struct_psp_fw_legacy_bin_desc),
    ('sys_drv_aux', struct_psp_fw_legacy_bin_desc),
    ('sos_aux', struct_psp_fw_legacy_bin_desc),
     ]

class struct_psp_fw_bin_desc(Structure):
    pass

struct_psp_fw_bin_desc._pack_ = 1 # source:False
struct_psp_fw_bin_desc._fields_ = [
    ('fw_type', ctypes.c_uint32),
    ('fw_version', ctypes.c_uint32),
    ('offset_bytes', ctypes.c_uint32),
    ('size_bytes', ctypes.c_uint32),
]

# UCODE_MAX_PSP_PACKAGING = (((ctypes.sizeof(amdgpu_firmware_header)-ctypes.sizeof(struct_common_firmware_header)-4)/ctypes.sizeof(struct_psp_fw_bin_desc))*2) # macro

# values for enumeration 'psp_fw_type'
psp_fw_type__enumvalues = {
    0: 'PSP_FW_TYPE_UNKOWN',
    1: 'PSP_FW_TYPE_PSP_SOS',
    2: 'PSP_FW_TYPE_PSP_SYS_DRV',
    3: 'PSP_FW_TYPE_PSP_KDB',
    4: 'PSP_FW_TYPE_PSP_TOC',
    5: 'PSP_FW_TYPE_PSP_SPL',
    6: 'PSP_FW_TYPE_PSP_RL',
    7: 'PSP_FW_TYPE_PSP_SOC_DRV',
    8: 'PSP_FW_TYPE_PSP_INTF_DRV',
    9: 'PSP_FW_TYPE_PSP_DBG_DRV',
    10: 'PSP_FW_TYPE_PSP_RAS_DRV',
    11: 'PSP_FW_TYPE_PSP_IPKEYMGR_DRV',
    12: 'PSP_FW_TYPE_MAX_INDEX',
}
PSP_FW_TYPE_UNKOWN = 0
PSP_FW_TYPE_PSP_SOS = 1
PSP_FW_TYPE_PSP_SYS_DRV = 2
PSP_FW_TYPE_PSP_KDB = 3
PSP_FW_TYPE_PSP_TOC = 4
PSP_FW_TYPE_PSP_SPL = 5
PSP_FW_TYPE_PSP_RL = 6
PSP_FW_TYPE_PSP_SOC_DRV = 7
PSP_FW_TYPE_PSP_INTF_DRV = 8
PSP_FW_TYPE_PSP_DBG_DRV = 9
PSP_FW_TYPE_PSP_RAS_DRV = 10
PSP_FW_TYPE_PSP_IPKEYMGR_DRV = 11
PSP_FW_TYPE_MAX_INDEX = 12
psp_fw_type = ctypes.c_uint32 # enum
class struct_psp_firmware_header_v2_0(Structure):
    pass

struct_psp_firmware_header_v2_0._pack_ = 1 # source:False
struct_psp_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('psp_fw_bin_count', ctypes.c_uint32),
    ('psp_fw_bin', struct_psp_fw_bin_desc * 1),
]

class struct_psp_firmware_header_v2_1(Structure):
    pass

struct_psp_firmware_header_v2_1._pack_ = 1 # source:False
struct_psp_firmware_header_v2_1._fields_ = [
    ('header', struct_common_firmware_header),
    ('psp_fw_bin_count', ctypes.c_uint32),
    ('psp_aux_fw_bin_index', ctypes.c_uint32),
    ('psp_fw_bin', struct_psp_fw_bin_desc * 1),
]

class struct_ta_firmware_header_v1_0(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('header', struct_common_firmware_header),
    ('xgmi', struct_psp_fw_legacy_bin_desc),
    ('ras', struct_psp_fw_legacy_bin_desc),
    ('hdcp', struct_psp_fw_legacy_bin_desc),
    ('dtm', struct_psp_fw_legacy_bin_desc),
    ('securedisplay', struct_psp_fw_legacy_bin_desc),
     ]


# values for enumeration 'ta_fw_type'
ta_fw_type__enumvalues = {
    0: 'TA_FW_TYPE_UNKOWN',
    1: 'TA_FW_TYPE_PSP_ASD',
    2: 'TA_FW_TYPE_PSP_XGMI',
    3: 'TA_FW_TYPE_PSP_RAS',
    4: 'TA_FW_TYPE_PSP_HDCP',
    5: 'TA_FW_TYPE_PSP_DTM',
    6: 'TA_FW_TYPE_PSP_RAP',
    7: 'TA_FW_TYPE_PSP_SECUREDISPLAY',
    8: 'TA_FW_TYPE_MAX_INDEX',
}
TA_FW_TYPE_UNKOWN = 0
TA_FW_TYPE_PSP_ASD = 1
TA_FW_TYPE_PSP_XGMI = 2
TA_FW_TYPE_PSP_RAS = 3
TA_FW_TYPE_PSP_HDCP = 4
TA_FW_TYPE_PSP_DTM = 5
TA_FW_TYPE_PSP_RAP = 6
TA_FW_TYPE_PSP_SECUREDISPLAY = 7
TA_FW_TYPE_MAX_INDEX = 8
ta_fw_type = ctypes.c_uint32 # enum
class struct_ta_firmware_header_v2_0(Structure):
    pass

struct_ta_firmware_header_v2_0._pack_ = 1 # source:False
struct_ta_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ta_fw_bin_count', ctypes.c_uint32),
    ('ta_fw_bin', struct_psp_fw_bin_desc * 1),
]

class struct_gfx_firmware_header_v1_0(Structure):
    pass

struct_gfx_firmware_header_v1_0._pack_ = 1 # source:False
struct_gfx_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('jt_offset', ctypes.c_uint32),
    ('jt_size', ctypes.c_uint32),
]

class struct_gfx_firmware_header_v2_0(Structure):
    pass

struct_gfx_firmware_header_v2_0._pack_ = 1 # source:False
struct_gfx_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ucode_size_bytes', ctypes.c_uint32),
    ('ucode_offset_bytes', ctypes.c_uint32),
    ('data_size_bytes', ctypes.c_uint32),
    ('data_offset_bytes', ctypes.c_uint32),
    ('ucode_start_addr_lo', ctypes.c_uint32),
    ('ucode_start_addr_hi', ctypes.c_uint32),
]

class struct_mes_firmware_header_v1_0(Structure):
    pass

struct_mes_firmware_header_v1_0._pack_ = 1 # source:False
struct_mes_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('mes_ucode_version', ctypes.c_uint32),
    ('mes_ucode_size_bytes', ctypes.c_uint32),
    ('mes_ucode_offset_bytes', ctypes.c_uint32),
    ('mes_ucode_data_version', ctypes.c_uint32),
    ('mes_ucode_data_size_bytes', ctypes.c_uint32),
    ('mes_ucode_data_offset_bytes', ctypes.c_uint32),
    ('mes_uc_start_addr_lo', ctypes.c_uint32),
    ('mes_uc_start_addr_hi', ctypes.c_uint32),
    ('mes_data_start_addr_lo', ctypes.c_uint32),
    ('mes_data_start_addr_hi', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v1_0(Structure):
    pass

struct_rlc_firmware_header_v1_0._pack_ = 1 # source:False
struct_rlc_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('save_and_restore_offset', ctypes.c_uint32),
    ('clear_state_descriptor_offset', ctypes.c_uint32),
    ('avail_scratch_ram_locations', ctypes.c_uint32),
    ('master_pkt_description_offset', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_0(Structure):
    pass

struct_rlc_firmware_header_v2_0._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('jt_offset', ctypes.c_uint32),
    ('jt_size', ctypes.c_uint32),
    ('save_and_restore_offset', ctypes.c_uint32),
    ('clear_state_descriptor_offset', ctypes.c_uint32),
    ('avail_scratch_ram_locations', ctypes.c_uint32),
    ('reg_restore_list_size', ctypes.c_uint32),
    ('reg_list_format_start', ctypes.c_uint32),
    ('reg_list_format_separate_start', ctypes.c_uint32),
    ('starting_offsets_start', ctypes.c_uint32),
    ('reg_list_format_size_bytes', ctypes.c_uint32),
    ('reg_list_format_array_offset_bytes', ctypes.c_uint32),
    ('reg_list_size_bytes', ctypes.c_uint32),
    ('reg_list_array_offset_bytes', ctypes.c_uint32),
    ('reg_list_format_separate_size_bytes', ctypes.c_uint32),
    ('reg_list_format_separate_array_offset_bytes', ctypes.c_uint32),
    ('reg_list_separate_size_bytes', ctypes.c_uint32),
    ('reg_list_separate_array_offset_bytes', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_1(Structure):
    pass

struct_rlc_firmware_header_v2_1._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_1._fields_ = [
    ('v2_0', struct_rlc_firmware_header_v2_0),
    ('reg_list_format_direct_reg_list_length', ctypes.c_uint32),
    ('save_restore_list_cntl_ucode_ver', ctypes.c_uint32),
    ('save_restore_list_cntl_feature_ver', ctypes.c_uint32),
    ('save_restore_list_cntl_size_bytes', ctypes.c_uint32),
    ('save_restore_list_cntl_offset_bytes', ctypes.c_uint32),
    ('save_restore_list_gpm_ucode_ver', ctypes.c_uint32),
    ('save_restore_list_gpm_feature_ver', ctypes.c_uint32),
    ('save_restore_list_gpm_size_bytes', ctypes.c_uint32),
    ('save_restore_list_gpm_offset_bytes', ctypes.c_uint32),
    ('save_restore_list_srm_ucode_ver', ctypes.c_uint32),
    ('save_restore_list_srm_feature_ver', ctypes.c_uint32),
    ('save_restore_list_srm_size_bytes', ctypes.c_uint32),
    ('save_restore_list_srm_offset_bytes', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_2(Structure):
    pass

struct_rlc_firmware_header_v2_2._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_2._fields_ = [
    ('v2_1', struct_rlc_firmware_header_v2_1),
    ('rlc_iram_ucode_size_bytes', ctypes.c_uint32),
    ('rlc_iram_ucode_offset_bytes', ctypes.c_uint32),
    ('rlc_dram_ucode_size_bytes', ctypes.c_uint32),
    ('rlc_dram_ucode_offset_bytes', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_3(Structure):
    pass

struct_rlc_firmware_header_v2_3._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_3._fields_ = [
    ('v2_2', struct_rlc_firmware_header_v2_2),
    ('rlcp_ucode_version', ctypes.c_uint32),
    ('rlcp_ucode_feature_version', ctypes.c_uint32),
    ('rlcp_ucode_size_bytes', ctypes.c_uint32),
    ('rlcp_ucode_offset_bytes', ctypes.c_uint32),
    ('rlcv_ucode_version', ctypes.c_uint32),
    ('rlcv_ucode_feature_version', ctypes.c_uint32),
    ('rlcv_ucode_size_bytes', ctypes.c_uint32),
    ('rlcv_ucode_offset_bytes', ctypes.c_uint32),
]

class struct_rlc_firmware_header_v2_4(Structure):
    pass

struct_rlc_firmware_header_v2_4._pack_ = 1 # source:False
struct_rlc_firmware_header_v2_4._fields_ = [
    ('v2_3', struct_rlc_firmware_header_v2_3),
    ('global_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('global_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
    ('se0_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('se0_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
    ('se1_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('se1_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
    ('se2_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('se2_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
    ('se3_tap_delays_ucode_size_bytes', ctypes.c_uint32),
    ('se3_tap_delays_ucode_offset_bytes', ctypes.c_uint32),
]

class struct_sdma_firmware_header_v1_0(Structure):
    pass

struct_sdma_firmware_header_v1_0._pack_ = 1 # source:False
struct_sdma_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ucode_change_version', ctypes.c_uint32),
    ('jt_offset', ctypes.c_uint32),
    ('jt_size', ctypes.c_uint32),
]

class struct_sdma_firmware_header_v1_1(Structure):
    pass

struct_sdma_firmware_header_v1_1._pack_ = 1 # source:False
struct_sdma_firmware_header_v1_1._fields_ = [
    ('v1_0', struct_sdma_firmware_header_v1_0),
    ('digest_size', ctypes.c_uint32),
]

class struct_sdma_firmware_header_v2_0(Structure):
    pass

struct_sdma_firmware_header_v2_0._pack_ = 1 # source:False
struct_sdma_firmware_header_v2_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ctx_ucode_size_bytes', ctypes.c_uint32),
    ('ctx_jt_offset', ctypes.c_uint32),
    ('ctx_jt_size', ctypes.c_uint32),
    ('ctl_ucode_offset', ctypes.c_uint32),
    ('ctl_ucode_size_bytes', ctypes.c_uint32),
    ('ctl_jt_offset', ctypes.c_uint32),
    ('ctl_jt_size', ctypes.c_uint32),
]

class struct_vpe_firmware_header_v1_0(Structure):
    pass

struct_vpe_firmware_header_v1_0._pack_ = 1 # source:False
struct_vpe_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ctx_ucode_size_bytes', ctypes.c_uint32),
    ('ctx_jt_offset', ctypes.c_uint32),
    ('ctx_jt_size', ctypes.c_uint32),
    ('ctl_ucode_offset', ctypes.c_uint32),
    ('ctl_ucode_size_bytes', ctypes.c_uint32),
    ('ctl_jt_offset', ctypes.c_uint32),
    ('ctl_jt_size', ctypes.c_uint32),
]

class struct_umsch_mm_firmware_header_v1_0(Structure):
    pass

struct_umsch_mm_firmware_header_v1_0._pack_ = 1 # source:False
struct_umsch_mm_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('umsch_mm_ucode_version', ctypes.c_uint32),
    ('umsch_mm_ucode_size_bytes', ctypes.c_uint32),
    ('umsch_mm_ucode_offset_bytes', ctypes.c_uint32),
    ('umsch_mm_ucode_data_version', ctypes.c_uint32),
    ('umsch_mm_ucode_data_size_bytes', ctypes.c_uint32),
    ('umsch_mm_ucode_data_offset_bytes', ctypes.c_uint32),
    ('umsch_mm_irq_start_addr_lo', ctypes.c_uint32),
    ('umsch_mm_irq_start_addr_hi', ctypes.c_uint32),
    ('umsch_mm_uc_start_addr_lo', ctypes.c_uint32),
    ('umsch_mm_uc_start_addr_hi', ctypes.c_uint32),
    ('umsch_mm_data_start_addr_lo', ctypes.c_uint32),
    ('umsch_mm_data_start_addr_hi', ctypes.c_uint32),
]

class struct_sdma_firmware_header_v3_0(Structure):
    pass

struct_sdma_firmware_header_v3_0._pack_ = 1 # source:False
struct_sdma_firmware_header_v3_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('ucode_feature_version', ctypes.c_uint32),
    ('ucode_offset_bytes', ctypes.c_uint32),
    ('ucode_size_bytes', ctypes.c_uint32),
]

class struct_gpu_info_firmware_v1_0(Structure):
    pass

struct_gpu_info_firmware_v1_0._pack_ = 1 # source:False
struct_gpu_info_firmware_v1_0._fields_ = [
    ('gc_num_se', ctypes.c_uint32),
    ('gc_num_cu_per_sh', ctypes.c_uint32),
    ('gc_num_sh_per_se', ctypes.c_uint32),
    ('gc_num_rb_per_se', ctypes.c_uint32),
    ('gc_num_tccs', ctypes.c_uint32),
    ('gc_num_gprs', ctypes.c_uint32),
    ('gc_num_max_gs_thds', ctypes.c_uint32),
    ('gc_gs_table_depth', ctypes.c_uint32),
    ('gc_gsprim_buff_depth', ctypes.c_uint32),
    ('gc_parameter_cache_depth', ctypes.c_uint32),
    ('gc_double_offchip_lds_buffer', ctypes.c_uint32),
    ('gc_wave_size', ctypes.c_uint32),
    ('gc_max_waves_per_simd', ctypes.c_uint32),
    ('gc_max_scratch_slots_per_cu', ctypes.c_uint32),
    ('gc_lds_size', ctypes.c_uint32),
]

class struct_gpu_info_firmware_v1_1(Structure):
    pass

struct_gpu_info_firmware_v1_1._pack_ = 1 # source:False
struct_gpu_info_firmware_v1_1._fields_ = [
    ('v1_0', struct_gpu_info_firmware_v1_0),
    ('num_sc_per_sh', ctypes.c_uint32),
    ('num_packer_per_sc', ctypes.c_uint32),
]

class struct_gpu_info_firmware_header_v1_0(Structure):
    pass

struct_gpu_info_firmware_header_v1_0._pack_ = 1 # source:False
struct_gpu_info_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('version_major', ctypes.c_uint16),
    ('version_minor', ctypes.c_uint16),
]

class struct_dmcu_firmware_header_v1_0(Structure):
    pass

struct_dmcu_firmware_header_v1_0._pack_ = 1 # source:False
struct_dmcu_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('intv_offset_bytes', ctypes.c_uint32),
    ('intv_size_bytes', ctypes.c_uint32),
]

class struct_dmcub_firmware_header_v1_0(Structure):
    pass

struct_dmcub_firmware_header_v1_0._pack_ = 1 # source:False
struct_dmcub_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('inst_const_bytes', ctypes.c_uint32),
    ('bss_data_bytes', ctypes.c_uint32),
]

class struct_imu_firmware_header_v1_0(Structure):
    pass

struct_imu_firmware_header_v1_0._pack_ = 1 # source:False
struct_imu_firmware_header_v1_0._fields_ = [
    ('header', struct_common_firmware_header),
    ('imu_iram_ucode_size_bytes', ctypes.c_uint32),
    ('imu_iram_ucode_offset_bytes', ctypes.c_uint32),
    ('imu_dram_ucode_size_bytes', ctypes.c_uint32),
    ('imu_dram_ucode_offset_bytes', ctypes.c_uint32),
]

class union_amdgpu_firmware_header(Union):
    pass

union_amdgpu_firmware_header._pack_ = 1 # source:False
union_amdgpu_firmware_header._fields_ = [
    ('common', struct_common_firmware_header),
    ('mc', struct_mc_firmware_header_v1_0),
    ('smc', struct_smc_firmware_header_v1_0),
    ('smc_v2_0', struct_smc_firmware_header_v2_0),
    ('psp', struct_psp_firmware_header_v1_0),
    ('psp_v1_1', struct_psp_firmware_header_v1_1),
    ('psp_v1_3', struct_psp_firmware_header_v1_3),
    ('psp_v2_0', struct_psp_firmware_header_v2_0),
    ('psp_v2_1', struct_psp_firmware_header_v2_0),
    ('ta', struct_ta_firmware_header_v1_0),
    ('ta_v2_0', struct_ta_firmware_header_v2_0),
    ('gfx', struct_gfx_firmware_header_v1_0),
    ('gfx_v2_0', struct_gfx_firmware_header_v2_0),
    ('rlc', struct_rlc_firmware_header_v1_0),
    ('rlc_v2_0', struct_rlc_firmware_header_v2_0),
    ('rlc_v2_1', struct_rlc_firmware_header_v2_1),
    ('rlc_v2_2', struct_rlc_firmware_header_v2_2),
    ('rlc_v2_3', struct_rlc_firmware_header_v2_3),
    ('rlc_v2_4', struct_rlc_firmware_header_v2_4),
    ('sdma', struct_sdma_firmware_header_v1_0),
    ('sdma_v1_1', struct_sdma_firmware_header_v1_1),
    ('sdma_v2_0', struct_sdma_firmware_header_v2_0),
    ('sdma_v3_0', struct_sdma_firmware_header_v3_0),
    ('gpu_info', struct_gpu_info_firmware_header_v1_0),
    ('dmcu', struct_dmcu_firmware_header_v1_0),
    ('dmcub', struct_dmcub_firmware_header_v1_0),
    ('imu', struct_imu_firmware_header_v1_0),
    ('raw', ctypes.c_ubyte * 256),
]


# values for enumeration 'AMDGPU_UCODE_ID'
AMDGPU_UCODE_ID__enumvalues = {
    0: 'AMDGPU_UCODE_ID_CAP',
    1: 'AMDGPU_UCODE_ID_SDMA0',
    2: 'AMDGPU_UCODE_ID_SDMA1',
    3: 'AMDGPU_UCODE_ID_SDMA2',
    4: 'AMDGPU_UCODE_ID_SDMA3',
    5: 'AMDGPU_UCODE_ID_SDMA4',
    6: 'AMDGPU_UCODE_ID_SDMA5',
    7: 'AMDGPU_UCODE_ID_SDMA6',
    8: 'AMDGPU_UCODE_ID_SDMA7',
    9: 'AMDGPU_UCODE_ID_SDMA_UCODE_TH0',
    10: 'AMDGPU_UCODE_ID_SDMA_UCODE_TH1',
    11: 'AMDGPU_UCODE_ID_SDMA_RS64',
    12: 'AMDGPU_UCODE_ID_CP_CE',
    13: 'AMDGPU_UCODE_ID_CP_PFP',
    14: 'AMDGPU_UCODE_ID_CP_ME',
    15: 'AMDGPU_UCODE_ID_CP_RS64_PFP',
    16: 'AMDGPU_UCODE_ID_CP_RS64_ME',
    17: 'AMDGPU_UCODE_ID_CP_RS64_MEC',
    18: 'AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK',
    19: 'AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK',
    20: 'AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK',
    21: 'AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK',
    22: 'AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK',
    23: 'AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK',
    24: 'AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK',
    25: 'AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK',
    26: 'AMDGPU_UCODE_ID_CP_MEC1',
    27: 'AMDGPU_UCODE_ID_CP_MEC1_JT',
    28: 'AMDGPU_UCODE_ID_CP_MEC2',
    29: 'AMDGPU_UCODE_ID_CP_MEC2_JT',
    30: 'AMDGPU_UCODE_ID_CP_MES',
    31: 'AMDGPU_UCODE_ID_CP_MES_DATA',
    32: 'AMDGPU_UCODE_ID_CP_MES1',
    33: 'AMDGPU_UCODE_ID_CP_MES1_DATA',
    34: 'AMDGPU_UCODE_ID_IMU_I',
    35: 'AMDGPU_UCODE_ID_IMU_D',
    36: 'AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS',
    37: 'AMDGPU_UCODE_ID_SE0_TAP_DELAYS',
    38: 'AMDGPU_UCODE_ID_SE1_TAP_DELAYS',
    39: 'AMDGPU_UCODE_ID_SE2_TAP_DELAYS',
    40: 'AMDGPU_UCODE_ID_SE3_TAP_DELAYS',
    41: 'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL',
    42: 'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM',
    43: 'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM',
    44: 'AMDGPU_UCODE_ID_RLC_IRAM',
    45: 'AMDGPU_UCODE_ID_RLC_DRAM',
    46: 'AMDGPU_UCODE_ID_RLC_P',
    47: 'AMDGPU_UCODE_ID_RLC_V',
    48: 'AMDGPU_UCODE_ID_RLC_G',
    49: 'AMDGPU_UCODE_ID_STORAGE',
    50: 'AMDGPU_UCODE_ID_SMC',
    51: 'AMDGPU_UCODE_ID_PPTABLE',
    52: 'AMDGPU_UCODE_ID_UVD',
    53: 'AMDGPU_UCODE_ID_UVD1',
    54: 'AMDGPU_UCODE_ID_VCE',
    55: 'AMDGPU_UCODE_ID_VCN',
    56: 'AMDGPU_UCODE_ID_VCN1',
    57: 'AMDGPU_UCODE_ID_DMCU_ERAM',
    58: 'AMDGPU_UCODE_ID_DMCU_INTV',
    59: 'AMDGPU_UCODE_ID_VCN0_RAM',
    60: 'AMDGPU_UCODE_ID_VCN1_RAM',
    61: 'AMDGPU_UCODE_ID_DMCUB',
    62: 'AMDGPU_UCODE_ID_VPE_CTX',
    63: 'AMDGPU_UCODE_ID_VPE_CTL',
    64: 'AMDGPU_UCODE_ID_VPE',
    65: 'AMDGPU_UCODE_ID_UMSCH_MM_UCODE',
    66: 'AMDGPU_UCODE_ID_UMSCH_MM_DATA',
    67: 'AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER',
    68: 'AMDGPU_UCODE_ID_P2S_TABLE',
    69: 'AMDGPU_UCODE_ID_JPEG_RAM',
    70: 'AMDGPU_UCODE_ID_ISP',
    71: 'AMDGPU_UCODE_ID_MAXIMUM',
}
AMDGPU_UCODE_ID_CAP = 0
AMDGPU_UCODE_ID_SDMA0 = 1
AMDGPU_UCODE_ID_SDMA1 = 2
AMDGPU_UCODE_ID_SDMA2 = 3
AMDGPU_UCODE_ID_SDMA3 = 4
AMDGPU_UCODE_ID_SDMA4 = 5
AMDGPU_UCODE_ID_SDMA5 = 6
AMDGPU_UCODE_ID_SDMA6 = 7
AMDGPU_UCODE_ID_SDMA7 = 8
AMDGPU_UCODE_ID_SDMA_UCODE_TH0 = 9
AMDGPU_UCODE_ID_SDMA_UCODE_TH1 = 10
AMDGPU_UCODE_ID_SDMA_RS64 = 11
AMDGPU_UCODE_ID_CP_CE = 12
AMDGPU_UCODE_ID_CP_PFP = 13
AMDGPU_UCODE_ID_CP_ME = 14
AMDGPU_UCODE_ID_CP_RS64_PFP = 15
AMDGPU_UCODE_ID_CP_RS64_ME = 16
AMDGPU_UCODE_ID_CP_RS64_MEC = 17
AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK = 18
AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK = 19
AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK = 20
AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK = 21
AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK = 22
AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK = 23
AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK = 24
AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK = 25
AMDGPU_UCODE_ID_CP_MEC1 = 26
AMDGPU_UCODE_ID_CP_MEC1_JT = 27
AMDGPU_UCODE_ID_CP_MEC2 = 28
AMDGPU_UCODE_ID_CP_MEC2_JT = 29
AMDGPU_UCODE_ID_CP_MES = 30
AMDGPU_UCODE_ID_CP_MES_DATA = 31
AMDGPU_UCODE_ID_CP_MES1 = 32
AMDGPU_UCODE_ID_CP_MES1_DATA = 33
AMDGPU_UCODE_ID_IMU_I = 34
AMDGPU_UCODE_ID_IMU_D = 35
AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS = 36
AMDGPU_UCODE_ID_SE0_TAP_DELAYS = 37
AMDGPU_UCODE_ID_SE1_TAP_DELAYS = 38
AMDGPU_UCODE_ID_SE2_TAP_DELAYS = 39
AMDGPU_UCODE_ID_SE3_TAP_DELAYS = 40
AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL = 41
AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM = 42
AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM = 43
AMDGPU_UCODE_ID_RLC_IRAM = 44
AMDGPU_UCODE_ID_RLC_DRAM = 45
AMDGPU_UCODE_ID_RLC_P = 46
AMDGPU_UCODE_ID_RLC_V = 47
AMDGPU_UCODE_ID_RLC_G = 48
AMDGPU_UCODE_ID_STORAGE = 49
AMDGPU_UCODE_ID_SMC = 50
AMDGPU_UCODE_ID_PPTABLE = 51
AMDGPU_UCODE_ID_UVD = 52
AMDGPU_UCODE_ID_UVD1 = 53
AMDGPU_UCODE_ID_VCE = 54
AMDGPU_UCODE_ID_VCN = 55
AMDGPU_UCODE_ID_VCN1 = 56
AMDGPU_UCODE_ID_DMCU_ERAM = 57
AMDGPU_UCODE_ID_DMCU_INTV = 58
AMDGPU_UCODE_ID_VCN0_RAM = 59
AMDGPU_UCODE_ID_VCN1_RAM = 60
AMDGPU_UCODE_ID_DMCUB = 61
AMDGPU_UCODE_ID_VPE_CTX = 62
AMDGPU_UCODE_ID_VPE_CTL = 63
AMDGPU_UCODE_ID_VPE = 64
AMDGPU_UCODE_ID_UMSCH_MM_UCODE = 65
AMDGPU_UCODE_ID_UMSCH_MM_DATA = 66
AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER = 67
AMDGPU_UCODE_ID_P2S_TABLE = 68
AMDGPU_UCODE_ID_JPEG_RAM = 69
AMDGPU_UCODE_ID_ISP = 70
AMDGPU_UCODE_ID_MAXIMUM = 71
AMDGPU_UCODE_ID = ctypes.c_uint32 # enum

# values for enumeration 'AMDGPU_UCODE_STATUS'
AMDGPU_UCODE_STATUS__enumvalues = {
    0: 'AMDGPU_UCODE_STATUS_INVALID',
    1: 'AMDGPU_UCODE_STATUS_NOT_LOADED',
    2: 'AMDGPU_UCODE_STATUS_LOADED',
}
AMDGPU_UCODE_STATUS_INVALID = 0
AMDGPU_UCODE_STATUS_NOT_LOADED = 1
AMDGPU_UCODE_STATUS_LOADED = 2
AMDGPU_UCODE_STATUS = ctypes.c_uint32 # enum

# values for enumeration 'amdgpu_firmware_load_type'
amdgpu_firmware_load_type__enumvalues = {
    0: 'AMDGPU_FW_LOAD_DIRECT',
    1: 'AMDGPU_FW_LOAD_PSP',
    2: 'AMDGPU_FW_LOAD_SMU',
    3: 'AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO',
}
AMDGPU_FW_LOAD_DIRECT = 0
AMDGPU_FW_LOAD_PSP = 1
AMDGPU_FW_LOAD_SMU = 2
AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO = 3
amdgpu_firmware_load_type = ctypes.c_uint32 # enum
class struct_amdgpu_firmware_info(Structure):
    pass

class struct_firmware(Structure):
    pass

struct_amdgpu_firmware_info._pack_ = 1 # source:False
struct_amdgpu_firmware_info._fields_ = [
    ('ucode_id', AMDGPU_UCODE_ID),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('fw', ctypes.POINTER(struct_firmware)),
    ('mc_addr', ctypes.c_uint64),
    ('kaddr', ctypes.POINTER(None)),
    ('ucode_size', ctypes.c_uint32),
    ('tmr_mc_addr_lo', ctypes.c_uint32),
    ('tmr_mc_addr_hi', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

_PSP_TEE_GFX_IF_H_ = True # macro
PSP_GFX_CMD_BUF_VERSION = 0x00000001 # macro
GFX_CMD_STATUS_MASK = 0x0000FFFF # macro
GFX_CMD_ID_MASK = 0x000F0000 # macro
GFX_CMD_RESERVED_MASK = 0x7FF00000 # macro
GFX_CMD_RESPONSE_MASK = 0x80000000 # macro
C2PMSG_CMD_GFX_USB_PD_FW_VER = 0x2000000 # macro
GFX_FLAG_RESPONSE = 0x80000000 # macro
GFX_BUF_MAX_DESC = 64 # macro
FRAME_TYPE_DESTROY = 1 # macro
PSP_ERR_UNKNOWN_COMMAND = 0x00000100 # macro

# values for enumeration 'psp_gfx_crtl_cmd_id'
psp_gfx_crtl_cmd_id__enumvalues = {
    65536: 'GFX_CTRL_CMD_ID_INIT_RBI_RING',
    131072: 'GFX_CTRL_CMD_ID_INIT_GPCOM_RING',
    196608: 'GFX_CTRL_CMD_ID_DESTROY_RINGS',
    262144: 'GFX_CTRL_CMD_ID_CAN_INIT_RINGS',
    327680: 'GFX_CTRL_CMD_ID_ENABLE_INT',
    393216: 'GFX_CTRL_CMD_ID_DISABLE_INT',
    458752: 'GFX_CTRL_CMD_ID_MODE1_RST',
    524288: 'GFX_CTRL_CMD_ID_GBR_IH_SET',
    589824: 'GFX_CTRL_CMD_ID_CONSUME_CMD',
    786432: 'GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING',
    983040: 'GFX_CTRL_CMD_ID_MAX',
}
GFX_CTRL_CMD_ID_INIT_RBI_RING = 65536
GFX_CTRL_CMD_ID_INIT_GPCOM_RING = 131072
GFX_CTRL_CMD_ID_DESTROY_RINGS = 196608
GFX_CTRL_CMD_ID_CAN_INIT_RINGS = 262144
GFX_CTRL_CMD_ID_ENABLE_INT = 327680
GFX_CTRL_CMD_ID_DISABLE_INT = 393216
GFX_CTRL_CMD_ID_MODE1_RST = 458752
GFX_CTRL_CMD_ID_GBR_IH_SET = 524288
GFX_CTRL_CMD_ID_CONSUME_CMD = 589824
GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING = 786432
GFX_CTRL_CMD_ID_MAX = 983040
psp_gfx_crtl_cmd_id = ctypes.c_uint32 # enum
class struct_psp_gfx_ctrl(Structure):
    pass

struct_psp_gfx_ctrl._pack_ = 1 # source:False
struct_psp_gfx_ctrl._fields_ = [
    ('cmd_resp', ctypes.c_uint32),
    ('rbi_wptr', ctypes.c_uint32),
    ('rbi_rptr', ctypes.c_uint32),
    ('gpcom_wptr', ctypes.c_uint32),
    ('gpcom_rptr', ctypes.c_uint32),
    ('ring_addr_lo', ctypes.c_uint32),
    ('ring_addr_hi', ctypes.c_uint32),
    ('ring_buf_size', ctypes.c_uint32),
]


# values for enumeration 'psp_gfx_cmd_id'
psp_gfx_cmd_id__enumvalues = {
    1: 'GFX_CMD_ID_LOAD_TA',
    2: 'GFX_CMD_ID_UNLOAD_TA',
    3: 'GFX_CMD_ID_INVOKE_CMD',
    4: 'GFX_CMD_ID_LOAD_ASD',
    5: 'GFX_CMD_ID_SETUP_TMR',
    6: 'GFX_CMD_ID_LOAD_IP_FW',
    7: 'GFX_CMD_ID_DESTROY_TMR',
    8: 'GFX_CMD_ID_SAVE_RESTORE',
    9: 'GFX_CMD_ID_SETUP_VMR',
    10: 'GFX_CMD_ID_DESTROY_VMR',
    11: 'GFX_CMD_ID_PROG_REG',
    15: 'GFX_CMD_ID_GET_FW_ATTESTATION',
    32: 'GFX_CMD_ID_LOAD_TOC',
    33: 'GFX_CMD_ID_AUTOLOAD_RLC',
    34: 'GFX_CMD_ID_BOOT_CFG',
    39: 'GFX_CMD_ID_SRIOV_SPATIAL_PART',
}
GFX_CMD_ID_LOAD_TA = 1
GFX_CMD_ID_UNLOAD_TA = 2
GFX_CMD_ID_INVOKE_CMD = 3
GFX_CMD_ID_LOAD_ASD = 4
GFX_CMD_ID_SETUP_TMR = 5
GFX_CMD_ID_LOAD_IP_FW = 6
GFX_CMD_ID_DESTROY_TMR = 7
GFX_CMD_ID_SAVE_RESTORE = 8
GFX_CMD_ID_SETUP_VMR = 9
GFX_CMD_ID_DESTROY_VMR = 10
GFX_CMD_ID_PROG_REG = 11
GFX_CMD_ID_GET_FW_ATTESTATION = 15
GFX_CMD_ID_LOAD_TOC = 32
GFX_CMD_ID_AUTOLOAD_RLC = 33
GFX_CMD_ID_BOOT_CFG = 34
GFX_CMD_ID_SRIOV_SPATIAL_PART = 39
psp_gfx_cmd_id = ctypes.c_uint32 # enum

# values for enumeration 'psp_gfx_boot_config_cmd'
psp_gfx_boot_config_cmd__enumvalues = {
    1: 'BOOTCFG_CMD_SET',
    2: 'BOOTCFG_CMD_GET',
    3: 'BOOTCFG_CMD_INVALIDATE',
}
BOOTCFG_CMD_SET = 1
BOOTCFG_CMD_GET = 2
BOOTCFG_CMD_INVALIDATE = 3
psp_gfx_boot_config_cmd = ctypes.c_uint32 # enum

# values for enumeration 'psp_gfx_boot_config'
psp_gfx_boot_config__enumvalues = {
    1: 'BOOT_CONFIG_GECC',
}
BOOT_CONFIG_GECC = 1
psp_gfx_boot_config = ctypes.c_uint32 # enum
class struct_psp_gfx_cmd_load_ta(Structure):
    pass

struct_psp_gfx_cmd_load_ta._pack_ = 1 # source:False
struct_psp_gfx_cmd_load_ta._fields_ = [
    ('app_phy_addr_lo', ctypes.c_uint32),
    ('app_phy_addr_hi', ctypes.c_uint32),
    ('app_len', ctypes.c_uint32),
    ('cmd_buf_phy_addr_lo', ctypes.c_uint32),
    ('cmd_buf_phy_addr_hi', ctypes.c_uint32),
    ('cmd_buf_len', ctypes.c_uint32),
]

class struct_psp_gfx_cmd_unload_ta(Structure):
    pass

struct_psp_gfx_cmd_unload_ta._pack_ = 1 # source:False
struct_psp_gfx_cmd_unload_ta._fields_ = [
    ('session_id', ctypes.c_uint32),
]

class struct_psp_gfx_buf_desc(Structure):
    pass

struct_psp_gfx_buf_desc._pack_ = 1 # source:False
struct_psp_gfx_buf_desc._fields_ = [
    ('buf_phy_addr_lo', ctypes.c_uint32),
    ('buf_phy_addr_hi', ctypes.c_uint32),
    ('buf_size', ctypes.c_uint32),
]

class struct_psp_gfx_buf_list(Structure):
    pass

struct_psp_gfx_buf_list._pack_ = 1 # source:False
struct_psp_gfx_buf_list._fields_ = [
    ('num_desc', ctypes.c_uint32),
    ('total_size', ctypes.c_uint32),
    ('buf_desc', struct_psp_gfx_buf_desc * 64),
]

class struct_psp_gfx_cmd_invoke_cmd(Structure):
    pass

struct_psp_gfx_cmd_invoke_cmd._pack_ = 1 # source:False
struct_psp_gfx_cmd_invoke_cmd._fields_ = [
    ('session_id', ctypes.c_uint32),
    ('ta_cmd_id', ctypes.c_uint32),
    ('buf', struct_psp_gfx_buf_list),
]

class struct_psp_gfx_cmd_setup_tmr(Structure):
    pass

class union_psp_gfx_cmd_setup_tmr_0(Union):
    pass

class struct_psp_gfx_cmd_setup_tmr_0_bitfield(Structure):
    pass

struct_psp_gfx_cmd_setup_tmr_0_bitfield._pack_ = 1 # source:False
struct_psp_gfx_cmd_setup_tmr_0_bitfield._fields_ = [
    ('sriov_enabled', ctypes.c_uint32, 1),
    ('virt_phy_addr', ctypes.c_uint32, 1),
    ('reserved', ctypes.c_uint32, 30),
]

union_psp_gfx_cmd_setup_tmr_0._pack_ = 1 # source:False
union_psp_gfx_cmd_setup_tmr_0._fields_ = [
    ('bitfield', struct_psp_gfx_cmd_setup_tmr_0_bitfield),
    ('tmr_flags', ctypes.c_uint32),
]

struct_psp_gfx_cmd_setup_tmr._pack_ = 1 # source:False
struct_psp_gfx_cmd_setup_tmr._anonymous_ = ('_0',)
struct_psp_gfx_cmd_setup_tmr._fields_ = [
    ('buf_phy_addr_lo', ctypes.c_uint32),
    ('buf_phy_addr_hi', ctypes.c_uint32),
    ('buf_size', ctypes.c_uint32),
    ('_0', union_psp_gfx_cmd_setup_tmr_0),
    ('system_phy_addr_lo', ctypes.c_uint32),
    ('system_phy_addr_hi', ctypes.c_uint32),
]


# values for enumeration 'psp_gfx_fw_type'
psp_gfx_fw_type__enumvalues = {
    0: 'GFX_FW_TYPE_NONE',
    1: 'GFX_FW_TYPE_CP_ME',
    2: 'GFX_FW_TYPE_CP_PFP',
    3: 'GFX_FW_TYPE_CP_CE',
    4: 'GFX_FW_TYPE_CP_MEC',
    5: 'GFX_FW_TYPE_CP_MEC_ME1',
    6: 'GFX_FW_TYPE_CP_MEC_ME2',
    7: 'GFX_FW_TYPE_RLC_V',
    8: 'GFX_FW_TYPE_RLC_G',
    9: 'GFX_FW_TYPE_SDMA0',
    10: 'GFX_FW_TYPE_SDMA1',
    11: 'GFX_FW_TYPE_DMCU_ERAM',
    12: 'GFX_FW_TYPE_DMCU_ISR',
    13: 'GFX_FW_TYPE_VCN',
    14: 'GFX_FW_TYPE_UVD',
    15: 'GFX_FW_TYPE_VCE',
    16: 'GFX_FW_TYPE_ISP',
    17: 'GFX_FW_TYPE_ACP',
    18: 'GFX_FW_TYPE_SMU',
    19: 'GFX_FW_TYPE_MMSCH',
    20: 'GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM',
    21: 'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM',
    22: 'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL',
    23: 'GFX_FW_TYPE_UVD1',
    24: 'GFX_FW_TYPE_TOC',
    25: 'GFX_FW_TYPE_RLC_P',
    26: 'GFX_FW_TYPE_RLC_IRAM',
    27: 'GFX_FW_TYPE_GLOBAL_TAP_DELAYS',
    28: 'GFX_FW_TYPE_SE0_TAP_DELAYS',
    29: 'GFX_FW_TYPE_SE1_TAP_DELAYS',
    30: 'GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS',
    31: 'GFX_FW_TYPE_SDMA0_JT',
    32: 'GFX_FW_TYPE_SDMA1_JT',
    33: 'GFX_FW_TYPE_CP_MES',
    34: 'GFX_FW_TYPE_MES_STACK',
    35: 'GFX_FW_TYPE_RLC_SRM_DRAM_SR',
    36: 'GFX_FW_TYPE_RLCG_SCRATCH_SR',
    37: 'GFX_FW_TYPE_RLCP_SCRATCH_SR',
    38: 'GFX_FW_TYPE_RLCV_SCRATCH_SR',
    39: 'GFX_FW_TYPE_RLX6_DRAM_SR',
    40: 'GFX_FW_TYPE_SDMA0_PG_CONTEXT',
    41: 'GFX_FW_TYPE_SDMA1_PG_CONTEXT',
    42: 'GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM',
    43: 'GFX_FW_TYPE_SE0_MUX_SELECT_RAM',
    44: 'GFX_FW_TYPE_SE1_MUX_SELECT_RAM',
    45: 'GFX_FW_TYPE_ACCUM_CTRL_RAM',
    46: 'GFX_FW_TYPE_RLCP_CAM',
    47: 'GFX_FW_TYPE_RLC_SPP_CAM_EXT',
    48: 'GFX_FW_TYPE_RLC_DRAM_BOOT',
    49: 'GFX_FW_TYPE_VCN0_RAM',
    50: 'GFX_FW_TYPE_VCN1_RAM',
    51: 'GFX_FW_TYPE_DMUB',
    52: 'GFX_FW_TYPE_SDMA2',
    53: 'GFX_FW_TYPE_SDMA3',
    54: 'GFX_FW_TYPE_SDMA4',
    55: 'GFX_FW_TYPE_SDMA5',
    56: 'GFX_FW_TYPE_SDMA6',
    57: 'GFX_FW_TYPE_SDMA7',
    58: 'GFX_FW_TYPE_VCN1',
    62: 'GFX_FW_TYPE_CAP',
    65: 'GFX_FW_TYPE_SE2_TAP_DELAYS',
    66: 'GFX_FW_TYPE_SE3_TAP_DELAYS',
    67: 'GFX_FW_TYPE_REG_LIST',
    68: 'GFX_FW_TYPE_IMU_I',
    69: 'GFX_FW_TYPE_IMU_D',
    70: 'GFX_FW_TYPE_LSDMA',
    71: 'GFX_FW_TYPE_SDMA_UCODE_TH0',
    72: 'GFX_FW_TYPE_SDMA_UCODE_TH1',
    73: 'GFX_FW_TYPE_PPTABLE',
    74: 'GFX_FW_TYPE_DISCRETE_USB4',
    75: 'GFX_FW_TYPE_TA',
    76: 'GFX_FW_TYPE_RS64_MES',
    77: 'GFX_FW_TYPE_RS64_MES_STACK',
    78: 'GFX_FW_TYPE_RS64_KIQ',
    79: 'GFX_FW_TYPE_RS64_KIQ_STACK',
    80: 'GFX_FW_TYPE_ISP_DATA',
    81: 'GFX_FW_TYPE_CP_MES_KIQ',
    82: 'GFX_FW_TYPE_MES_KIQ_STACK',
    83: 'GFX_FW_TYPE_UMSCH_DATA',
    84: 'GFX_FW_TYPE_UMSCH_UCODE',
    85: 'GFX_FW_TYPE_UMSCH_CMD_BUFFER',
    86: 'GFX_FW_TYPE_USB_DP_COMBO_PHY',
    87: 'GFX_FW_TYPE_RS64_PFP',
    88: 'GFX_FW_TYPE_RS64_ME',
    89: 'GFX_FW_TYPE_RS64_MEC',
    90: 'GFX_FW_TYPE_RS64_PFP_P0_STACK',
    91: 'GFX_FW_TYPE_RS64_PFP_P1_STACK',
    92: 'GFX_FW_TYPE_RS64_ME_P0_STACK',
    93: 'GFX_FW_TYPE_RS64_ME_P1_STACK',
    94: 'GFX_FW_TYPE_RS64_MEC_P0_STACK',
    95: 'GFX_FW_TYPE_RS64_MEC_P1_STACK',
    96: 'GFX_FW_TYPE_RS64_MEC_P2_STACK',
    97: 'GFX_FW_TYPE_RS64_MEC_P3_STACK',
    100: 'GFX_FW_TYPE_VPEC_FW1',
    101: 'GFX_FW_TYPE_VPEC_FW2',
    102: 'GFX_FW_TYPE_VPE',
    128: 'GFX_FW_TYPE_JPEG_RAM',
    129: 'GFX_FW_TYPE_P2S_TABLE',
    130: 'GFX_FW_TYPE_MAX',
}
GFX_FW_TYPE_NONE = 0
GFX_FW_TYPE_CP_ME = 1
GFX_FW_TYPE_CP_PFP = 2
GFX_FW_TYPE_CP_CE = 3
GFX_FW_TYPE_CP_MEC = 4
GFX_FW_TYPE_CP_MEC_ME1 = 5
GFX_FW_TYPE_CP_MEC_ME2 = 6
GFX_FW_TYPE_RLC_V = 7
GFX_FW_TYPE_RLC_G = 8
GFX_FW_TYPE_SDMA0 = 9
GFX_FW_TYPE_SDMA1 = 10
GFX_FW_TYPE_DMCU_ERAM = 11
GFX_FW_TYPE_DMCU_ISR = 12
GFX_FW_TYPE_VCN = 13
GFX_FW_TYPE_UVD = 14
GFX_FW_TYPE_VCE = 15
GFX_FW_TYPE_ISP = 16
GFX_FW_TYPE_ACP = 17
GFX_FW_TYPE_SMU = 18
GFX_FW_TYPE_MMSCH = 19
GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM = 20
GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM = 21
GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL = 22
GFX_FW_TYPE_UVD1 = 23
GFX_FW_TYPE_TOC = 24
GFX_FW_TYPE_RLC_P = 25
GFX_FW_TYPE_RLC_IRAM = 26
GFX_FW_TYPE_GLOBAL_TAP_DELAYS = 27
GFX_FW_TYPE_SE0_TAP_DELAYS = 28
GFX_FW_TYPE_SE1_TAP_DELAYS = 29
GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS = 30
GFX_FW_TYPE_SDMA0_JT = 31
GFX_FW_TYPE_SDMA1_JT = 32
GFX_FW_TYPE_CP_MES = 33
GFX_FW_TYPE_MES_STACK = 34
GFX_FW_TYPE_RLC_SRM_DRAM_SR = 35
GFX_FW_TYPE_RLCG_SCRATCH_SR = 36
GFX_FW_TYPE_RLCP_SCRATCH_SR = 37
GFX_FW_TYPE_RLCV_SCRATCH_SR = 38
GFX_FW_TYPE_RLX6_DRAM_SR = 39
GFX_FW_TYPE_SDMA0_PG_CONTEXT = 40
GFX_FW_TYPE_SDMA1_PG_CONTEXT = 41
GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM = 42
GFX_FW_TYPE_SE0_MUX_SELECT_RAM = 43
GFX_FW_TYPE_SE1_MUX_SELECT_RAM = 44
GFX_FW_TYPE_ACCUM_CTRL_RAM = 45
GFX_FW_TYPE_RLCP_CAM = 46
GFX_FW_TYPE_RLC_SPP_CAM_EXT = 47
GFX_FW_TYPE_RLC_DRAM_BOOT = 48
GFX_FW_TYPE_VCN0_RAM = 49
GFX_FW_TYPE_VCN1_RAM = 50
GFX_FW_TYPE_DMUB = 51
GFX_FW_TYPE_SDMA2 = 52
GFX_FW_TYPE_SDMA3 = 53
GFX_FW_TYPE_SDMA4 = 54
GFX_FW_TYPE_SDMA5 = 55
GFX_FW_TYPE_SDMA6 = 56
GFX_FW_TYPE_SDMA7 = 57
GFX_FW_TYPE_VCN1 = 58
GFX_FW_TYPE_CAP = 62
GFX_FW_TYPE_SE2_TAP_DELAYS = 65
GFX_FW_TYPE_SE3_TAP_DELAYS = 66
GFX_FW_TYPE_REG_LIST = 67
GFX_FW_TYPE_IMU_I = 68
GFX_FW_TYPE_IMU_D = 69
GFX_FW_TYPE_LSDMA = 70
GFX_FW_TYPE_SDMA_UCODE_TH0 = 71
GFX_FW_TYPE_SDMA_UCODE_TH1 = 72
GFX_FW_TYPE_PPTABLE = 73
GFX_FW_TYPE_DISCRETE_USB4 = 74
GFX_FW_TYPE_TA = 75
GFX_FW_TYPE_RS64_MES = 76
GFX_FW_TYPE_RS64_MES_STACK = 77
GFX_FW_TYPE_RS64_KIQ = 78
GFX_FW_TYPE_RS64_KIQ_STACK = 79
GFX_FW_TYPE_ISP_DATA = 80
GFX_FW_TYPE_CP_MES_KIQ = 81
GFX_FW_TYPE_MES_KIQ_STACK = 82
GFX_FW_TYPE_UMSCH_DATA = 83
GFX_FW_TYPE_UMSCH_UCODE = 84
GFX_FW_TYPE_UMSCH_CMD_BUFFER = 85
GFX_FW_TYPE_USB_DP_COMBO_PHY = 86
GFX_FW_TYPE_RS64_PFP = 87
GFX_FW_TYPE_RS64_ME = 88
GFX_FW_TYPE_RS64_MEC = 89
GFX_FW_TYPE_RS64_PFP_P0_STACK = 90
GFX_FW_TYPE_RS64_PFP_P1_STACK = 91
GFX_FW_TYPE_RS64_ME_P0_STACK = 92
GFX_FW_TYPE_RS64_ME_P1_STACK = 93
GFX_FW_TYPE_RS64_MEC_P0_STACK = 94
GFX_FW_TYPE_RS64_MEC_P1_STACK = 95
GFX_FW_TYPE_RS64_MEC_P2_STACK = 96
GFX_FW_TYPE_RS64_MEC_P3_STACK = 97
GFX_FW_TYPE_VPEC_FW1 = 100
GFX_FW_TYPE_VPEC_FW2 = 101
GFX_FW_TYPE_VPE = 102
GFX_FW_TYPE_JPEG_RAM = 128
GFX_FW_TYPE_P2S_TABLE = 129
GFX_FW_TYPE_MAX = 130
psp_gfx_fw_type = ctypes.c_uint32 # enum
class struct_psp_gfx_cmd_load_ip_fw(Structure):
    pass

struct_psp_gfx_cmd_load_ip_fw._pack_ = 1 # source:False
struct_psp_gfx_cmd_load_ip_fw._fields_ = [
    ('fw_phy_addr_lo', ctypes.c_uint32),
    ('fw_phy_addr_hi', ctypes.c_uint32),
    ('fw_size', ctypes.c_uint32),
    ('fw_type', psp_gfx_fw_type),
]

class struct_psp_gfx_cmd_save_restore_ip_fw(Structure):
    pass

struct_psp_gfx_cmd_save_restore_ip_fw._pack_ = 1 # source:False
struct_psp_gfx_cmd_save_restore_ip_fw._fields_ = [
    ('save_fw', ctypes.c_uint32),
    ('save_restore_addr_lo', ctypes.c_uint32),
    ('save_restore_addr_hi', ctypes.c_uint32),
    ('buf_size', ctypes.c_uint32),
    ('fw_type', psp_gfx_fw_type),
]

class struct_psp_gfx_cmd_reg_prog(Structure):
    pass

struct_psp_gfx_cmd_reg_prog._pack_ = 1 # source:False
struct_psp_gfx_cmd_reg_prog._fields_ = [
    ('reg_value', ctypes.c_uint32),
    ('reg_id', ctypes.c_uint32),
]

class struct_psp_gfx_cmd_load_toc(Structure):
    pass

struct_psp_gfx_cmd_load_toc._pack_ = 1 # source:False
struct_psp_gfx_cmd_load_toc._fields_ = [
    ('toc_phy_addr_lo', ctypes.c_uint32),
    ('toc_phy_addr_hi', ctypes.c_uint32),
    ('toc_size', ctypes.c_uint32),
]

class struct_psp_gfx_cmd_boot_cfg(Structure):
    pass

struct_psp_gfx_cmd_boot_cfg._pack_ = 1 # source:False
struct_psp_gfx_cmd_boot_cfg._fields_ = [
    ('timestamp', ctypes.c_uint32),
    ('sub_cmd', psp_gfx_boot_config_cmd),
    ('boot_config', ctypes.c_uint32),
    ('boot_config_valid', ctypes.c_uint32),
]

class struct_psp_gfx_cmd_sriov_spatial_part(Structure):
    pass

struct_psp_gfx_cmd_sriov_spatial_part._pack_ = 1 # source:False
struct_psp_gfx_cmd_sriov_spatial_part._fields_ = [
    ('mode', ctypes.c_uint32),
    ('override_ips', ctypes.c_uint32),
    ('override_xcds_avail', ctypes.c_uint32),
    ('override_this_aid', ctypes.c_uint32),
]

class union_psp_gfx_commands(Union):
    pass

union_psp_gfx_commands._pack_ = 1 # source:False
union_psp_gfx_commands._fields_ = [
    ('cmd_load_ta', struct_psp_gfx_cmd_load_ta),
    ('cmd_unload_ta', struct_psp_gfx_cmd_unload_ta),
    ('cmd_invoke_cmd', struct_psp_gfx_cmd_invoke_cmd),
    ('cmd_setup_tmr', struct_psp_gfx_cmd_setup_tmr),
    ('cmd_load_ip_fw', struct_psp_gfx_cmd_load_ip_fw),
    ('cmd_save_restore_ip_fw', struct_psp_gfx_cmd_save_restore_ip_fw),
    ('cmd_setup_reg_prog', struct_psp_gfx_cmd_reg_prog),
    ('cmd_setup_vmr', struct_psp_gfx_cmd_setup_tmr),
    ('cmd_load_toc', struct_psp_gfx_cmd_load_toc),
    ('boot_cfg', struct_psp_gfx_cmd_boot_cfg),
    ('cmd_spatial_part', struct_psp_gfx_cmd_sriov_spatial_part),
    ('PADDING_0', ctypes.c_ubyte * 768),
]

class struct_psp_gfx_uresp_reserved(Structure):
    pass

struct_psp_gfx_uresp_reserved._pack_ = 1 # source:False
struct_psp_gfx_uresp_reserved._fields_ = [
    ('reserved', ctypes.c_uint32 * 8),
]

class struct_psp_gfx_uresp_fwar_db_info(Structure):
    pass

struct_psp_gfx_uresp_fwar_db_info._pack_ = 1 # source:False
struct_psp_gfx_uresp_fwar_db_info._fields_ = [
    ('fwar_db_addr_lo', ctypes.c_uint32),
    ('fwar_db_addr_hi', ctypes.c_uint32),
]

class struct_psp_gfx_uresp_bootcfg(Structure):
    pass

struct_psp_gfx_uresp_bootcfg._pack_ = 1 # source:False
struct_psp_gfx_uresp_bootcfg._fields_ = [
    ('boot_cfg', ctypes.c_uint32),
]

class union_psp_gfx_uresp(Union):
    pass

union_psp_gfx_uresp._pack_ = 1 # source:False
union_psp_gfx_uresp._fields_ = [
    ('reserved', struct_psp_gfx_uresp_reserved),
    ('boot_cfg', struct_psp_gfx_uresp_bootcfg),
    ('fwar_db_info', struct_psp_gfx_uresp_fwar_db_info),
    ('PADDING_0', ctypes.c_ubyte * 24),
]

class struct_psp_gfx_resp(Structure):
    pass

struct_psp_gfx_resp._pack_ = 1 # source:False
struct_psp_gfx_resp._fields_ = [
    ('status', ctypes.c_uint32),
    ('session_id', ctypes.c_uint32),
    ('fw_addr_lo', ctypes.c_uint32),
    ('fw_addr_hi', ctypes.c_uint32),
    ('tmr_size', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 11),
    ('uresp', union_psp_gfx_uresp),
]

class struct_psp_gfx_cmd_resp(Structure):
    pass

struct_psp_gfx_cmd_resp._pack_ = 1 # source:False
struct_psp_gfx_cmd_resp._fields_ = [
    ('buf_size', ctypes.c_uint32),
    ('buf_version', ctypes.c_uint32),
    ('cmd_id', ctypes.c_uint32),
    ('resp_buf_addr_lo', ctypes.c_uint32),
    ('resp_buf_addr_hi', ctypes.c_uint32),
    ('resp_offset', ctypes.c_uint32),
    ('resp_buf_size', ctypes.c_uint32),
    ('cmd', union_psp_gfx_commands),
    ('reserved_1', ctypes.c_ubyte * 52),
    ('resp', struct_psp_gfx_resp),
    ('reserved_2', ctypes.c_ubyte * 64),
]

class struct_psp_gfx_rb_frame(Structure):
    pass

struct_psp_gfx_rb_frame._pack_ = 1 # source:False
struct_psp_gfx_rb_frame._fields_ = [
    ('cmd_buf_addr_lo', ctypes.c_uint32),
    ('cmd_buf_addr_hi', ctypes.c_uint32),
    ('cmd_buf_size', ctypes.c_uint32),
    ('fence_addr_lo', ctypes.c_uint32),
    ('fence_addr_hi', ctypes.c_uint32),
    ('fence_value', ctypes.c_uint32),
    ('sid_lo', ctypes.c_uint32),
    ('sid_hi', ctypes.c_uint32),
    ('vmid', ctypes.c_ubyte),
    ('frame_type', ctypes.c_ubyte),
    ('reserved1', ctypes.c_ubyte * 2),
    ('reserved2', ctypes.c_uint32 * 7),
]


# values for enumeration 'tee_error_code'
tee_error_code__enumvalues = {
    0: 'TEE_SUCCESS',
    4294901770: 'TEE_ERROR_NOT_SUPPORTED',
}
TEE_SUCCESS = 0
TEE_ERROR_NOT_SUPPORTED = 4294901770
tee_error_code = ctypes.c_uint32 # enum
__AMDGPU_PSP_H__ = True # macro
PSP_FENCE_BUFFER_SIZE = 0x1000 # macro
PSP_CMD_BUFFER_SIZE = 0x1000 # macro
PSP_1_MEG = 0x100000 # macro
# def PSP_TMR_SIZE(adev):  # macro
#    return ((adev)->asic_type==CHIP_ALDEBARAN?0x800000:0x400000)
PSP_TMR_ALIGNMENT = 0x100000 # macro
PSP_FW_NAME_LEN = 0x24 # macro
AMDGPU_XGMI_MAX_CONNECTED_NODES = 64 # macro
MEM_TRAIN_SYSTEM_SIGNATURE = 0x54534942 # macro
GDDR6_MEM_TRAINING_DATA_SIZE_IN_BYTES = 0x1000 # macro
GDDR6_MEM_TRAINING_OFFSET = 0x8000 # macro
BIST_MEM_TRAINING_ENCROACHED_SIZE = 0x2000000 # macro
PSP_RUNTIME_DB_SIZE_IN_BYTES = 0x10000 # macro
PSP_RUNTIME_DB_OFFSET = 0x100000 # macro
PSP_RUNTIME_DB_COOKIE_ID = 0x0ed5 # macro
PSP_RUNTIME_DB_VER_1 = 0x0100 # macro
PSP_RUNTIME_DB_DIAG_ENTRY_MAX_COUNT = 0x40 # macro

# values for enumeration 'psp_shared_mem_size'
psp_shared_mem_size__enumvalues = {
    0: 'PSP_ASD_SHARED_MEM_SIZE',
    16384: 'PSP_XGMI_SHARED_MEM_SIZE',
    16384: 'PSP_RAS_SHARED_MEM_SIZE',
    16384: 'PSP_HDCP_SHARED_MEM_SIZE',
    16384: 'PSP_DTM_SHARED_MEM_SIZE',
    16384: 'PSP_RAP_SHARED_MEM_SIZE',
    16384: 'PSP_SECUREDISPLAY_SHARED_MEM_SIZE',
}
PSP_ASD_SHARED_MEM_SIZE = 0
PSP_XGMI_SHARED_MEM_SIZE = 16384
PSP_RAS_SHARED_MEM_SIZE = 16384
PSP_HDCP_SHARED_MEM_SIZE = 16384
PSP_DTM_SHARED_MEM_SIZE = 16384
PSP_RAP_SHARED_MEM_SIZE = 16384
PSP_SECUREDISPLAY_SHARED_MEM_SIZE = 16384
psp_shared_mem_size = ctypes.c_uint32 # enum

# values for enumeration 'ta_type_id'
ta_type_id__enumvalues = {
    1: 'TA_TYPE_XGMI',
    2: 'TA_TYPE_RAS',
    3: 'TA_TYPE_HDCP',
    4: 'TA_TYPE_DTM',
    5: 'TA_TYPE_RAP',
    6: 'TA_TYPE_SECUREDISPLAY',
    7: 'TA_TYPE_MAX_INDEX',
}
TA_TYPE_XGMI = 1
TA_TYPE_RAS = 2
TA_TYPE_HDCP = 3
TA_TYPE_DTM = 4
TA_TYPE_RAP = 5
TA_TYPE_SECUREDISPLAY = 6
TA_TYPE_MAX_INDEX = 7
ta_type_id = ctypes.c_uint32 # enum
class struct_psp_context(Structure):
    pass

class struct_psp_xgmi_node_info(Structure):
    pass

class struct_psp_xgmi_topology_info(Structure):
    pass

class struct_psp_bin_desc(Structure):
    pass


# values for enumeration 'psp_bootloader_cmd'
psp_bootloader_cmd__enumvalues = {
    65536: 'PSP_BL__LOAD_SYSDRV',
    131072: 'PSP_BL__LOAD_SOSDRV',
    524288: 'PSP_BL__LOAD_KEY_DATABASE',
    720896: 'PSP_BL__LOAD_SOCDRV',
    786432: 'PSP_BL__LOAD_DBGDRV',
    786432: 'PSP_BL__LOAD_HADDRV',
    851968: 'PSP_BL__LOAD_INTFDRV',
    917504: 'PSP_BL__LOAD_RASDRV',
    983040: 'PSP_BL__LOAD_IPKEYMGRDRV',
    1048576: 'PSP_BL__DRAM_LONG_TRAIN',
    2097152: 'PSP_BL__DRAM_SHORT_TRAIN',
    268435456: 'PSP_BL__LOAD_TOS_SPL_TABLE',
}
PSP_BL__LOAD_SYSDRV = 65536
PSP_BL__LOAD_SOSDRV = 131072
PSP_BL__LOAD_KEY_DATABASE = 524288
PSP_BL__LOAD_SOCDRV = 720896
PSP_BL__LOAD_DBGDRV = 786432
PSP_BL__LOAD_HADDRV = 786432
PSP_BL__LOAD_INTFDRV = 851968
PSP_BL__LOAD_RASDRV = 917504
PSP_BL__LOAD_IPKEYMGRDRV = 983040
PSP_BL__DRAM_LONG_TRAIN = 1048576
PSP_BL__DRAM_SHORT_TRAIN = 2097152
PSP_BL__LOAD_TOS_SPL_TABLE = 268435456
psp_bootloader_cmd = ctypes.c_uint32 # enum

# values for enumeration 'psp_ring_type'
psp_ring_type__enumvalues = {
    0: 'PSP_RING_TYPE__INVALID',
    1: 'PSP_RING_TYPE__UM',
    2: 'PSP_RING_TYPE__KM',
}
PSP_RING_TYPE__INVALID = 0
PSP_RING_TYPE__UM = 1
PSP_RING_TYPE__KM = 2
psp_ring_type = ctypes.c_uint32 # enum

# values for enumeration 'psp_reg_prog_id'
psp_reg_prog_id__enumvalues = {
    0: 'PSP_REG_IH_RB_CNTL',
    1: 'PSP_REG_IH_RB_CNTL_RING1',
    2: 'PSP_REG_IH_RB_CNTL_RING2',
    3: 'PSP_REG_LAST',
}
PSP_REG_IH_RB_CNTL = 0
PSP_REG_IH_RB_CNTL_RING1 = 1
PSP_REG_IH_RB_CNTL_RING2 = 2
PSP_REG_LAST = 3
psp_reg_prog_id = ctypes.c_uint32 # enum

# values for enumeration 'psp_memory_training_init_flag'
psp_memory_training_init_flag__enumvalues = {
    0: 'PSP_MEM_TRAIN_NOT_SUPPORT',
    1: 'PSP_MEM_TRAIN_SUPPORT',
    2: 'PSP_MEM_TRAIN_INIT_FAILED',
    4: 'PSP_MEM_TRAIN_RESERVE_SUCCESS',
    8: 'PSP_MEM_TRAIN_INIT_SUCCESS',
}
PSP_MEM_TRAIN_NOT_SUPPORT = 0
PSP_MEM_TRAIN_SUPPORT = 1
PSP_MEM_TRAIN_INIT_FAILED = 2
PSP_MEM_TRAIN_RESERVE_SUCCESS = 4
PSP_MEM_TRAIN_INIT_SUCCESS = 8
psp_memory_training_init_flag = ctypes.c_uint32 # enum

# values for enumeration 'psp_memory_training_ops'
psp_memory_training_ops__enumvalues = {
    1: 'PSP_MEM_TRAIN_SEND_LONG_MSG',
    2: 'PSP_MEM_TRAIN_SAVE',
    4: 'PSP_MEM_TRAIN_RESTORE',
    8: 'PSP_MEM_TRAIN_SEND_SHORT_MSG',
    1: 'PSP_MEM_TRAIN_COLD_BOOT',
    8: 'PSP_MEM_TRAIN_RESUME',
}
PSP_MEM_TRAIN_SEND_LONG_MSG = 1
PSP_MEM_TRAIN_SAVE = 2
PSP_MEM_TRAIN_RESTORE = 4
PSP_MEM_TRAIN_SEND_SHORT_MSG = 8
PSP_MEM_TRAIN_COLD_BOOT = 1
PSP_MEM_TRAIN_RESUME = 8
psp_memory_training_ops = ctypes.c_uint32 # enum

# values for enumeration 'psp_runtime_entry_type'
psp_runtime_entry_type__enumvalues = {
    0: 'PSP_RUNTIME_ENTRY_TYPE_INVALID',
    1: 'PSP_RUNTIME_ENTRY_TYPE_TEST',
    2: 'PSP_RUNTIME_ENTRY_TYPE_MGPU_COMMON',
    3: 'PSP_RUNTIME_ENTRY_TYPE_MGPU_WAFL',
    4: 'PSP_RUNTIME_ENTRY_TYPE_MGPU_XGMI',
    5: 'PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG',
    6: 'PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS',
}
PSP_RUNTIME_ENTRY_TYPE_INVALID = 0
PSP_RUNTIME_ENTRY_TYPE_TEST = 1
PSP_RUNTIME_ENTRY_TYPE_MGPU_COMMON = 2
PSP_RUNTIME_ENTRY_TYPE_MGPU_WAFL = 3
PSP_RUNTIME_ENTRY_TYPE_MGPU_XGMI = 4
PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG = 5
PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS = 6
psp_runtime_entry_type = ctypes.c_uint32 # enum

# values for enumeration 'psp_runtime_boot_cfg_feature'
psp_runtime_boot_cfg_feature__enumvalues = {
    1: 'BOOT_CFG_FEATURE_GECC',
    2: 'BOOT_CFG_FEATURE_TWO_STAGE_DRAM_TRAINING',
}
BOOT_CFG_FEATURE_GECC = 1
BOOT_CFG_FEATURE_TWO_STAGE_DRAM_TRAINING = 2
psp_runtime_boot_cfg_feature = ctypes.c_uint32 # enum

# values for enumeration 'psp_runtime_scpm_authentication'
psp_runtime_scpm_authentication__enumvalues = {
    0: 'SCPM_DISABLE',
    1: 'SCPM_ENABLE',
    2: 'SCPM_ENABLE_WITH_SCPM_ERR',
}
SCPM_DISABLE = 0
SCPM_ENABLE = 1
SCPM_ENABLE_WITH_SCPM_ERR = 2
psp_runtime_scpm_authentication = ctypes.c_uint32 # enum
__AMDGPU_IRQ_H__ = True # macro
AMDGPU_MAX_IRQ_SRC_ID = 0x100 # macro
AMDGPU_MAX_IRQ_CLIENT_ID = 0x100 # macro
AMDGPU_IRQ_CLIENTID_LEGACY = 0 # macro
AMDGPU_IRQ_SRC_DATA_MAX_SIZE_DW = 4 # macro
class struct_amdgpu_device(Structure):
    pass


# values for enumeration 'amdgpu_interrupt_state'
amdgpu_interrupt_state__enumvalues = {
    0: 'AMDGPU_IRQ_STATE_DISABLE',
    1: 'AMDGPU_IRQ_STATE_ENABLE',
}
AMDGPU_IRQ_STATE_DISABLE = 0
AMDGPU_IRQ_STATE_ENABLE = 1
amdgpu_interrupt_state = ctypes.c_uint32 # enum
class struct_amdgpu_iv_entry(Structure):
    pass

struct_amdgpu_iv_entry._pack_ = 1 # source:False
struct_amdgpu_iv_entry._fields_ = [
    ('client_id', ctypes.c_uint32),
    ('src_id', ctypes.c_uint32),
    ('ring_id', ctypes.c_uint32),
    ('vmid', ctypes.c_uint32),
    ('vmid_src', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('timestamp', ctypes.c_uint64),
    ('timestamp_src', ctypes.c_uint32),
    ('pasid', ctypes.c_uint32),
    ('node_id', ctypes.c_uint32),
    ('src_data', ctypes.c_uint32 * 4),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('iv_entry', ctypes.POINTER(ctypes.c_uint32)),
]


# values for enumeration 'interrupt_node_id_per_aid'
interrupt_node_id_per_aid__enumvalues = {
    0: 'AID0_NODEID',
    1: 'XCD0_NODEID',
    2: 'XCD1_NODEID',
    4: 'AID1_NODEID',
    5: 'XCD2_NODEID',
    6: 'XCD3_NODEID',
    8: 'AID2_NODEID',
    9: 'XCD4_NODEID',
    10: 'XCD5_NODEID',
    12: 'AID3_NODEID',
    13: 'XCD6_NODEID',
    14: 'XCD7_NODEID',
    15: 'NODEID_MAX',
}
AID0_NODEID = 0
XCD0_NODEID = 1
XCD1_NODEID = 2
AID1_NODEID = 4
XCD2_NODEID = 5
XCD3_NODEID = 6
AID2_NODEID = 8
XCD4_NODEID = 9
XCD5_NODEID = 10
AID3_NODEID = 12
XCD6_NODEID = 13
XCD7_NODEID = 14
NODEID_MAX = 15
interrupt_node_id_per_aid = ctypes.c_uint32 # enum
AMDGPU_DOORBELL_H = True # macro

# values for enumeration 'AMDGPU_DOORBELL_ASSIGNMENT'
AMDGPU_DOORBELL_ASSIGNMENT__enumvalues = {
    0: 'AMDGPU_DOORBELL_KIQ',
    1: 'AMDGPU_DOORBELL_HIQ',
    2: 'AMDGPU_DOORBELL_DIQ',
    16: 'AMDGPU_DOORBELL_MEC_RING0',
    17: 'AMDGPU_DOORBELL_MEC_RING1',
    18: 'AMDGPU_DOORBELL_MEC_RING2',
    19: 'AMDGPU_DOORBELL_MEC_RING3',
    20: 'AMDGPU_DOORBELL_MEC_RING4',
    21: 'AMDGPU_DOORBELL_MEC_RING5',
    22: 'AMDGPU_DOORBELL_MEC_RING6',
    23: 'AMDGPU_DOORBELL_MEC_RING7',
    32: 'AMDGPU_DOORBELL_GFX_RING0',
    480: 'AMDGPU_DOORBELL_sDMA_ENGINE0',
    481: 'AMDGPU_DOORBELL_sDMA_ENGINE1',
    488: 'AMDGPU_DOORBELL_IH',
    1023: 'AMDGPU_DOORBELL_MAX_ASSIGNMENT',
    65535: 'AMDGPU_DOORBELL_INVALID',
}
AMDGPU_DOORBELL_KIQ = 0
AMDGPU_DOORBELL_HIQ = 1
AMDGPU_DOORBELL_DIQ = 2
AMDGPU_DOORBELL_MEC_RING0 = 16
AMDGPU_DOORBELL_MEC_RING1 = 17
AMDGPU_DOORBELL_MEC_RING2 = 18
AMDGPU_DOORBELL_MEC_RING3 = 19
AMDGPU_DOORBELL_MEC_RING4 = 20
AMDGPU_DOORBELL_MEC_RING5 = 21
AMDGPU_DOORBELL_MEC_RING6 = 22
AMDGPU_DOORBELL_MEC_RING7 = 23
AMDGPU_DOORBELL_GFX_RING0 = 32
AMDGPU_DOORBELL_sDMA_ENGINE0 = 480
AMDGPU_DOORBELL_sDMA_ENGINE1 = 481
AMDGPU_DOORBELL_IH = 488
AMDGPU_DOORBELL_MAX_ASSIGNMENT = 1023
AMDGPU_DOORBELL_INVALID = 65535
AMDGPU_DOORBELL_ASSIGNMENT = ctypes.c_uint32 # enum

# values for enumeration 'AMDGPU_VEGA20_DOORBELL_ASSIGNMENT'
AMDGPU_VEGA20_DOORBELL_ASSIGNMENT__enumvalues = {
    0: 'AMDGPU_VEGA20_DOORBELL_KIQ',
    1: 'AMDGPU_VEGA20_DOORBELL_HIQ',
    2: 'AMDGPU_VEGA20_DOORBELL_DIQ',
    3: 'AMDGPU_VEGA20_DOORBELL_MEC_RING0',
    4: 'AMDGPU_VEGA20_DOORBELL_MEC_RING1',
    5: 'AMDGPU_VEGA20_DOORBELL_MEC_RING2',
    6: 'AMDGPU_VEGA20_DOORBELL_MEC_RING3',
    7: 'AMDGPU_VEGA20_DOORBELL_MEC_RING4',
    8: 'AMDGPU_VEGA20_DOORBELL_MEC_RING5',
    9: 'AMDGPU_VEGA20_DOORBELL_MEC_RING6',
    10: 'AMDGPU_VEGA20_DOORBELL_MEC_RING7',
    11: 'AMDGPU_VEGA20_DOORBELL_USERQUEUE_START',
    138: 'AMDGPU_VEGA20_DOORBELL_USERQUEUE_END',
    139: 'AMDGPU_VEGA20_DOORBELL_GFX_RING0',
    256: 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE0',
    266: 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE1',
    276: 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE2',
    286: 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE3',
    296: 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE4',
    306: 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE5',
    316: 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE6',
    326: 'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE7',
    376: 'AMDGPU_VEGA20_DOORBELL_IH',
    392: 'AMDGPU_VEGA20_DOORBELL64_VCN0_1',
    393: 'AMDGPU_VEGA20_DOORBELL64_VCN2_3',
    394: 'AMDGPU_VEGA20_DOORBELL64_VCN4_5',
    395: 'AMDGPU_VEGA20_DOORBELL64_VCN6_7',
    396: 'AMDGPU_VEGA20_DOORBELL64_VCN8_9',
    397: 'AMDGPU_VEGA20_DOORBELL64_VCNa_b',
    398: 'AMDGPU_VEGA20_DOORBELL64_VCNc_d',
    399: 'AMDGPU_VEGA20_DOORBELL64_VCNe_f',
    392: 'AMDGPU_VEGA20_DOORBELL64_UVD_RING0_1',
    393: 'AMDGPU_VEGA20_DOORBELL64_UVD_RING2_3',
    394: 'AMDGPU_VEGA20_DOORBELL64_UVD_RING4_5',
    395: 'AMDGPU_VEGA20_DOORBELL64_UVD_RING6_7',
    396: 'AMDGPU_VEGA20_DOORBELL64_VCE_RING0_1',
    397: 'AMDGPU_VEGA20_DOORBELL64_VCE_RING2_3',
    398: 'AMDGPU_VEGA20_DOORBELL64_VCE_RING4_5',
    399: 'AMDGPU_VEGA20_DOORBELL64_VCE_RING6_7',
    256: 'AMDGPU_VEGA20_DOORBELL64_FIRST_NON_CP',
    399: 'AMDGPU_VEGA20_DOORBELL64_LAST_NON_CP',
    400: 'AMDGPU_VEGA20_DOORBELL_XCC1_KIQ_START',
    407: 'AMDGPU_VEGA20_DOORBELL_XCC1_MEC_RING0_START',
    464: 'AMDGPU_VEGA20_DOORBELL_AID1_sDMA_START',
    503: 'AMDGPU_VEGA20_DOORBELL_MAX_ASSIGNMENT',
    65535: 'AMDGPU_VEGA20_DOORBELL_INVALID',
}
AMDGPU_VEGA20_DOORBELL_KIQ = 0
AMDGPU_VEGA20_DOORBELL_HIQ = 1
AMDGPU_VEGA20_DOORBELL_DIQ = 2
AMDGPU_VEGA20_DOORBELL_MEC_RING0 = 3
AMDGPU_VEGA20_DOORBELL_MEC_RING1 = 4
AMDGPU_VEGA20_DOORBELL_MEC_RING2 = 5
AMDGPU_VEGA20_DOORBELL_MEC_RING3 = 6
AMDGPU_VEGA20_DOORBELL_MEC_RING4 = 7
AMDGPU_VEGA20_DOORBELL_MEC_RING5 = 8
AMDGPU_VEGA20_DOORBELL_MEC_RING6 = 9
AMDGPU_VEGA20_DOORBELL_MEC_RING7 = 10
AMDGPU_VEGA20_DOORBELL_USERQUEUE_START = 11
AMDGPU_VEGA20_DOORBELL_USERQUEUE_END = 138
AMDGPU_VEGA20_DOORBELL_GFX_RING0 = 139
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE0 = 256
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE1 = 266
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE2 = 276
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE3 = 286
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE4 = 296
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE5 = 306
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE6 = 316
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE7 = 326
AMDGPU_VEGA20_DOORBELL_IH = 376
AMDGPU_VEGA20_DOORBELL64_VCN0_1 = 392
AMDGPU_VEGA20_DOORBELL64_VCN2_3 = 393
AMDGPU_VEGA20_DOORBELL64_VCN4_5 = 394
AMDGPU_VEGA20_DOORBELL64_VCN6_7 = 395
AMDGPU_VEGA20_DOORBELL64_VCN8_9 = 396
AMDGPU_VEGA20_DOORBELL64_VCNa_b = 397
AMDGPU_VEGA20_DOORBELL64_VCNc_d = 398
AMDGPU_VEGA20_DOORBELL64_VCNe_f = 399
AMDGPU_VEGA20_DOORBELL64_UVD_RING0_1 = 392
AMDGPU_VEGA20_DOORBELL64_UVD_RING2_3 = 393
AMDGPU_VEGA20_DOORBELL64_UVD_RING4_5 = 394
AMDGPU_VEGA20_DOORBELL64_UVD_RING6_7 = 395
AMDGPU_VEGA20_DOORBELL64_VCE_RING0_1 = 396
AMDGPU_VEGA20_DOORBELL64_VCE_RING2_3 = 397
AMDGPU_VEGA20_DOORBELL64_VCE_RING4_5 = 398
AMDGPU_VEGA20_DOORBELL64_VCE_RING6_7 = 399
AMDGPU_VEGA20_DOORBELL64_FIRST_NON_CP = 256
AMDGPU_VEGA20_DOORBELL64_LAST_NON_CP = 399
AMDGPU_VEGA20_DOORBELL_XCC1_KIQ_START = 400
AMDGPU_VEGA20_DOORBELL_XCC1_MEC_RING0_START = 407
AMDGPU_VEGA20_DOORBELL_AID1_sDMA_START = 464
AMDGPU_VEGA20_DOORBELL_MAX_ASSIGNMENT = 503
AMDGPU_VEGA20_DOORBELL_INVALID = 65535
AMDGPU_VEGA20_DOORBELL_ASSIGNMENT = ctypes.c_uint32 # enum

# values for enumeration 'AMDGPU_NAVI10_DOORBELL_ASSIGNMENT'
AMDGPU_NAVI10_DOORBELL_ASSIGNMENT__enumvalues = {
    0: 'AMDGPU_NAVI10_DOORBELL_KIQ',
    1: 'AMDGPU_NAVI10_DOORBELL_HIQ',
    2: 'AMDGPU_NAVI10_DOORBELL_DIQ',
    3: 'AMDGPU_NAVI10_DOORBELL_MEC_RING0',
    4: 'AMDGPU_NAVI10_DOORBELL_MEC_RING1',
    5: 'AMDGPU_NAVI10_DOORBELL_MEC_RING2',
    6: 'AMDGPU_NAVI10_DOORBELL_MEC_RING3',
    7: 'AMDGPU_NAVI10_DOORBELL_MEC_RING4',
    8: 'AMDGPU_NAVI10_DOORBELL_MEC_RING5',
    9: 'AMDGPU_NAVI10_DOORBELL_MEC_RING6',
    10: 'AMDGPU_NAVI10_DOORBELL_MEC_RING7',
    11: 'AMDGPU_NAVI10_DOORBELL_MES_RING0',
    12: 'AMDGPU_NAVI10_DOORBELL_MES_RING1',
    13: 'AMDGPU_NAVI10_DOORBELL_USERQUEUE_START',
    138: 'AMDGPU_NAVI10_DOORBELL_USERQUEUE_END',
    139: 'AMDGPU_NAVI10_DOORBELL_GFX_RING0',
    140: 'AMDGPU_NAVI10_DOORBELL_GFX_RING1',
    141: 'AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_START',
    255: 'AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_END',
    256: 'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0',
    266: 'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE1',
    276: 'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE2',
    286: 'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE3',
    376: 'AMDGPU_NAVI10_DOORBELL_IH',
    392: 'AMDGPU_NAVI10_DOORBELL64_VCN0_1',
    393: 'AMDGPU_NAVI10_DOORBELL64_VCN2_3',
    394: 'AMDGPU_NAVI10_DOORBELL64_VCN4_5',
    395: 'AMDGPU_NAVI10_DOORBELL64_VCN6_7',
    396: 'AMDGPU_NAVI10_DOORBELL64_VCN8_9',
    397: 'AMDGPU_NAVI10_DOORBELL64_VCNa_b',
    398: 'AMDGPU_NAVI10_DOORBELL64_VCNc_d',
    399: 'AMDGPU_NAVI10_DOORBELL64_VCNe_f',
    400: 'AMDGPU_NAVI10_DOORBELL64_VPE',
    256: 'AMDGPU_NAVI10_DOORBELL64_FIRST_NON_CP',
    400: 'AMDGPU_NAVI10_DOORBELL64_LAST_NON_CP',
    400: 'AMDGPU_NAVI10_DOORBELL_MAX_ASSIGNMENT',
    65535: 'AMDGPU_NAVI10_DOORBELL_INVALID',
}
AMDGPU_NAVI10_DOORBELL_KIQ = 0
AMDGPU_NAVI10_DOORBELL_HIQ = 1
AMDGPU_NAVI10_DOORBELL_DIQ = 2
AMDGPU_NAVI10_DOORBELL_MEC_RING0 = 3
AMDGPU_NAVI10_DOORBELL_MEC_RING1 = 4
AMDGPU_NAVI10_DOORBELL_MEC_RING2 = 5
AMDGPU_NAVI10_DOORBELL_MEC_RING3 = 6
AMDGPU_NAVI10_DOORBELL_MEC_RING4 = 7
AMDGPU_NAVI10_DOORBELL_MEC_RING5 = 8
AMDGPU_NAVI10_DOORBELL_MEC_RING6 = 9
AMDGPU_NAVI10_DOORBELL_MEC_RING7 = 10
AMDGPU_NAVI10_DOORBELL_MES_RING0 = 11
AMDGPU_NAVI10_DOORBELL_MES_RING1 = 12
AMDGPU_NAVI10_DOORBELL_USERQUEUE_START = 13
AMDGPU_NAVI10_DOORBELL_USERQUEUE_END = 138
AMDGPU_NAVI10_DOORBELL_GFX_RING0 = 139
AMDGPU_NAVI10_DOORBELL_GFX_RING1 = 140
AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_START = 141
AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_END = 255
AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 = 256
AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE1 = 266
AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE2 = 276
AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE3 = 286
AMDGPU_NAVI10_DOORBELL_IH = 376
AMDGPU_NAVI10_DOORBELL64_VCN0_1 = 392
AMDGPU_NAVI10_DOORBELL64_VCN2_3 = 393
AMDGPU_NAVI10_DOORBELL64_VCN4_5 = 394
AMDGPU_NAVI10_DOORBELL64_VCN6_7 = 395
AMDGPU_NAVI10_DOORBELL64_VCN8_9 = 396
AMDGPU_NAVI10_DOORBELL64_VCNa_b = 397
AMDGPU_NAVI10_DOORBELL64_VCNc_d = 398
AMDGPU_NAVI10_DOORBELL64_VCNe_f = 399
AMDGPU_NAVI10_DOORBELL64_VPE = 400
AMDGPU_NAVI10_DOORBELL64_FIRST_NON_CP = 256
AMDGPU_NAVI10_DOORBELL64_LAST_NON_CP = 400
AMDGPU_NAVI10_DOORBELL_MAX_ASSIGNMENT = 400
AMDGPU_NAVI10_DOORBELL_INVALID = 65535
AMDGPU_NAVI10_DOORBELL_ASSIGNMENT = ctypes.c_uint32 # enum

# values for enumeration 'AMDGPU_DOORBELL64_ASSIGNMENT'
AMDGPU_DOORBELL64_ASSIGNMENT__enumvalues = {
    0: 'AMDGPU_DOORBELL64_KIQ',
    1: 'AMDGPU_DOORBELL64_HIQ',
    2: 'AMDGPU_DOORBELL64_DIQ',
    3: 'AMDGPU_DOORBELL64_MEC_RING0',
    4: 'AMDGPU_DOORBELL64_MEC_RING1',
    5: 'AMDGPU_DOORBELL64_MEC_RING2',
    6: 'AMDGPU_DOORBELL64_MEC_RING3',
    7: 'AMDGPU_DOORBELL64_MEC_RING4',
    8: 'AMDGPU_DOORBELL64_MEC_RING5',
    9: 'AMDGPU_DOORBELL64_MEC_RING6',
    10: 'AMDGPU_DOORBELL64_MEC_RING7',
    11: 'AMDGPU_DOORBELL64_USERQUEUE_START',
    138: 'AMDGPU_DOORBELL64_USERQUEUE_END',
    139: 'AMDGPU_DOORBELL64_GFX_RING0',
    240: 'AMDGPU_DOORBELL64_sDMA_ENGINE0',
    241: 'AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE0',
    242: 'AMDGPU_DOORBELL64_sDMA_ENGINE1',
    243: 'AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE1',
    244: 'AMDGPU_DOORBELL64_IH',
    245: 'AMDGPU_DOORBELL64_IH_RING1',
    246: 'AMDGPU_DOORBELL64_IH_RING2',
    248: 'AMDGPU_DOORBELL64_VCN0_1',
    249: 'AMDGPU_DOORBELL64_VCN2_3',
    250: 'AMDGPU_DOORBELL64_VCN4_5',
    251: 'AMDGPU_DOORBELL64_VCN6_7',
    248: 'AMDGPU_DOORBELL64_UVD_RING0_1',
    249: 'AMDGPU_DOORBELL64_UVD_RING2_3',
    250: 'AMDGPU_DOORBELL64_UVD_RING4_5',
    251: 'AMDGPU_DOORBELL64_UVD_RING6_7',
    252: 'AMDGPU_DOORBELL64_VCE_RING0_1',
    253: 'AMDGPU_DOORBELL64_VCE_RING2_3',
    254: 'AMDGPU_DOORBELL64_VCE_RING4_5',
    255: 'AMDGPU_DOORBELL64_VCE_RING6_7',
    240: 'AMDGPU_DOORBELL64_FIRST_NON_CP',
    255: 'AMDGPU_DOORBELL64_LAST_NON_CP',
    255: 'AMDGPU_DOORBELL64_MAX_ASSIGNMENT',
    65535: 'AMDGPU_DOORBELL64_INVALID',
}
AMDGPU_DOORBELL64_KIQ = 0
AMDGPU_DOORBELL64_HIQ = 1
AMDGPU_DOORBELL64_DIQ = 2
AMDGPU_DOORBELL64_MEC_RING0 = 3
AMDGPU_DOORBELL64_MEC_RING1 = 4
AMDGPU_DOORBELL64_MEC_RING2 = 5
AMDGPU_DOORBELL64_MEC_RING3 = 6
AMDGPU_DOORBELL64_MEC_RING4 = 7
AMDGPU_DOORBELL64_MEC_RING5 = 8
AMDGPU_DOORBELL64_MEC_RING6 = 9
AMDGPU_DOORBELL64_MEC_RING7 = 10
AMDGPU_DOORBELL64_USERQUEUE_START = 11
AMDGPU_DOORBELL64_USERQUEUE_END = 138
AMDGPU_DOORBELL64_GFX_RING0 = 139
AMDGPU_DOORBELL64_sDMA_ENGINE0 = 240
AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE0 = 241
AMDGPU_DOORBELL64_sDMA_ENGINE1 = 242
AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE1 = 243
AMDGPU_DOORBELL64_IH = 244
AMDGPU_DOORBELL64_IH_RING1 = 245
AMDGPU_DOORBELL64_IH_RING2 = 246
AMDGPU_DOORBELL64_VCN0_1 = 248
AMDGPU_DOORBELL64_VCN2_3 = 249
AMDGPU_DOORBELL64_VCN4_5 = 250
AMDGPU_DOORBELL64_VCN6_7 = 251
AMDGPU_DOORBELL64_UVD_RING0_1 = 248
AMDGPU_DOORBELL64_UVD_RING2_3 = 249
AMDGPU_DOORBELL64_UVD_RING4_5 = 250
AMDGPU_DOORBELL64_UVD_RING6_7 = 251
AMDGPU_DOORBELL64_VCE_RING0_1 = 252
AMDGPU_DOORBELL64_VCE_RING2_3 = 253
AMDGPU_DOORBELL64_VCE_RING4_5 = 254
AMDGPU_DOORBELL64_VCE_RING6_7 = 255
AMDGPU_DOORBELL64_FIRST_NON_CP = 240
AMDGPU_DOORBELL64_LAST_NON_CP = 255
AMDGPU_DOORBELL64_MAX_ASSIGNMENT = 255
AMDGPU_DOORBELL64_INVALID = 65535
AMDGPU_DOORBELL64_ASSIGNMENT = ctypes.c_uint32 # enum

# values for enumeration 'AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1'
AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1__enumvalues = {
    0: 'AMDGPU_DOORBELL_LAYOUT1_KIQ_START',
    1: 'AMDGPU_DOORBELL_LAYOUT1_HIQ',
    2: 'AMDGPU_DOORBELL_LAYOUT1_DIQ',
    8: 'AMDGPU_DOORBELL_LAYOUT1_MEC_RING_START',
    15: 'AMDGPU_DOORBELL_LAYOUT1_MEC_RING_END',
    16: 'AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_START',
    31: 'AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_END',
    32: 'AMDGPU_DOORBELL_LAYOUT1_XCC_RANGE',
    256: 'AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_START',
    415: 'AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_END',
    416: 'AMDGPU_DOORBELL_LAYOUT1_IH',
    432: 'AMDGPU_DOORBELL_LAYOUT1_VCN_START',
    488: 'AMDGPU_DOORBELL_LAYOUT1_VCN_END',
    256: 'AMDGPU_DOORBELL_LAYOUT1_FIRST_NON_CP',
    488: 'AMDGPU_DOORBELL_LAYOUT1_LAST_NON_CP',
    488: 'AMDGPU_DOORBELL_LAYOUT1_MAX_ASSIGNMENT',
    65535: 'AMDGPU_DOORBELL_LAYOUT1_INVALID',
}
AMDGPU_DOORBELL_LAYOUT1_KIQ_START = 0
AMDGPU_DOORBELL_LAYOUT1_HIQ = 1
AMDGPU_DOORBELL_LAYOUT1_DIQ = 2
AMDGPU_DOORBELL_LAYOUT1_MEC_RING_START = 8
AMDGPU_DOORBELL_LAYOUT1_MEC_RING_END = 15
AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_START = 16
AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_END = 31
AMDGPU_DOORBELL_LAYOUT1_XCC_RANGE = 32
AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_START = 256
AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_END = 415
AMDGPU_DOORBELL_LAYOUT1_IH = 416
AMDGPU_DOORBELL_LAYOUT1_VCN_START = 432
AMDGPU_DOORBELL_LAYOUT1_VCN_END = 488
AMDGPU_DOORBELL_LAYOUT1_FIRST_NON_CP = 256
AMDGPU_DOORBELL_LAYOUT1_LAST_NON_CP = 488
AMDGPU_DOORBELL_LAYOUT1_MAX_ASSIGNMENT = 488
AMDGPU_DOORBELL_LAYOUT1_INVALID = 65535
AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1 = ctypes.c_uint32 # enum
__SOC15_IH_CLIENTID_H__ = True # macro

# values for enumeration 'soc15_ih_clientid'
soc15_ih_clientid__enumvalues = {
    0: 'SOC15_IH_CLIENTID_IH',
    1: 'SOC15_IH_CLIENTID_ACP',
    2: 'SOC15_IH_CLIENTID_ATHUB',
    3: 'SOC15_IH_CLIENTID_BIF',
    4: 'SOC15_IH_CLIENTID_DCE',
    5: 'SOC15_IH_CLIENTID_ISP',
    6: 'SOC15_IH_CLIENTID_PCIE0',
    7: 'SOC15_IH_CLIENTID_RLC',
    8: 'SOC15_IH_CLIENTID_SDMA0',
    9: 'SOC15_IH_CLIENTID_SDMA1',
    10: 'SOC15_IH_CLIENTID_SE0SH',
    11: 'SOC15_IH_CLIENTID_SE1SH',
    12: 'SOC15_IH_CLIENTID_SE2SH',
    13: 'SOC15_IH_CLIENTID_SE3SH',
    14: 'SOC15_IH_CLIENTID_UVD1',
    15: 'SOC15_IH_CLIENTID_THM',
    16: 'SOC15_IH_CLIENTID_UVD',
    17: 'SOC15_IH_CLIENTID_VCE0',
    18: 'SOC15_IH_CLIENTID_VMC',
    19: 'SOC15_IH_CLIENTID_XDMA',
    20: 'SOC15_IH_CLIENTID_GRBM_CP',
    21: 'SOC15_IH_CLIENTID_ATS',
    22: 'SOC15_IH_CLIENTID_ROM_SMUIO',
    23: 'SOC15_IH_CLIENTID_DF',
    24: 'SOC15_IH_CLIENTID_VCE1',
    25: 'SOC15_IH_CLIENTID_PWR',
    26: 'SOC15_IH_CLIENTID_RESERVED',
    27: 'SOC15_IH_CLIENTID_UTCL2',
    28: 'SOC15_IH_CLIENTID_EA',
    29: 'SOC15_IH_CLIENTID_UTCL2LOG',
    30: 'SOC15_IH_CLIENTID_MP0',
    31: 'SOC15_IH_CLIENTID_MP1',
    32: 'SOC15_IH_CLIENTID_MAX',
    16: 'SOC15_IH_CLIENTID_VCN',
    14: 'SOC15_IH_CLIENTID_VCN1',
    1: 'SOC15_IH_CLIENTID_SDMA2',
    4: 'SOC15_IH_CLIENTID_SDMA3',
    5: 'SOC15_IH_CLIENTID_SDMA3_Sienna_Cichlid',
    5: 'SOC15_IH_CLIENTID_SDMA4',
    17: 'SOC15_IH_CLIENTID_SDMA5',
    19: 'SOC15_IH_CLIENTID_SDMA6',
    24: 'SOC15_IH_CLIENTID_SDMA7',
    6: 'SOC15_IH_CLIENTID_VMC1',
}
SOC15_IH_CLIENTID_IH = 0
SOC15_IH_CLIENTID_ACP = 1
SOC15_IH_CLIENTID_ATHUB = 2
SOC15_IH_CLIENTID_BIF = 3
SOC15_IH_CLIENTID_DCE = 4
SOC15_IH_CLIENTID_ISP = 5
SOC15_IH_CLIENTID_PCIE0 = 6
SOC15_IH_CLIENTID_RLC = 7
SOC15_IH_CLIENTID_SDMA0 = 8
SOC15_IH_CLIENTID_SDMA1 = 9
SOC15_IH_CLIENTID_SE0SH = 10
SOC15_IH_CLIENTID_SE1SH = 11
SOC15_IH_CLIENTID_SE2SH = 12
SOC15_IH_CLIENTID_SE3SH = 13
SOC15_IH_CLIENTID_UVD1 = 14
SOC15_IH_CLIENTID_THM = 15
SOC15_IH_CLIENTID_UVD = 16
SOC15_IH_CLIENTID_VCE0 = 17
SOC15_IH_CLIENTID_VMC = 18
SOC15_IH_CLIENTID_XDMA = 19
SOC15_IH_CLIENTID_GRBM_CP = 20
SOC15_IH_CLIENTID_ATS = 21
SOC15_IH_CLIENTID_ROM_SMUIO = 22
SOC15_IH_CLIENTID_DF = 23
SOC15_IH_CLIENTID_VCE1 = 24
SOC15_IH_CLIENTID_PWR = 25
SOC15_IH_CLIENTID_RESERVED = 26
SOC15_IH_CLIENTID_UTCL2 = 27
SOC15_IH_CLIENTID_EA = 28
SOC15_IH_CLIENTID_UTCL2LOG = 29
SOC15_IH_CLIENTID_MP0 = 30
SOC15_IH_CLIENTID_MP1 = 31
SOC15_IH_CLIENTID_MAX = 32
SOC15_IH_CLIENTID_VCN = 16
SOC15_IH_CLIENTID_VCN1 = 14
SOC15_IH_CLIENTID_SDMA2 = 1
SOC15_IH_CLIENTID_SDMA3 = 4
SOC15_IH_CLIENTID_SDMA3_Sienna_Cichlid = 5
SOC15_IH_CLIENTID_SDMA4 = 5
SOC15_IH_CLIENTID_SDMA5 = 17
SOC15_IH_CLIENTID_SDMA6 = 19
SOC15_IH_CLIENTID_SDMA7 = 24
SOC15_IH_CLIENTID_VMC1 = 6
soc15_ih_clientid = ctypes.c_uint32 # enum
AMDGPU_IRQ_CLIENTID_MAX = SOC15_IH_CLIENTID_MAX # macro
soc15_ih_clientid_name = [] # Variable ctypes.POINTER(ctypes.c_char) * 0

# values for enumeration 'soc21_ih_clientid'
soc21_ih_clientid__enumvalues = {
    0: 'SOC21_IH_CLIENTID_IH',
    2: 'SOC21_IH_CLIENTID_ATHUB',
    3: 'SOC21_IH_CLIENTID_BIF',
    4: 'SOC21_IH_CLIENTID_DCN',
    5: 'SOC21_IH_CLIENTID_ISP',
    6: 'SOC21_IH_CLIENTID_MP3',
    7: 'SOC21_IH_CLIENTID_RLC',
    10: 'SOC21_IH_CLIENTID_GFX',
    11: 'SOC21_IH_CLIENTID_IMU',
    14: 'SOC21_IH_CLIENTID_VCN1',
    15: 'SOC21_IH_CLIENTID_THM',
    16: 'SOC21_IH_CLIENTID_VCN',
    17: 'SOC21_IH_CLIENTID_VPE1',
    18: 'SOC21_IH_CLIENTID_VMC',
    20: 'SOC21_IH_CLIENTID_GRBM_CP',
    22: 'SOC21_IH_CLIENTID_ROM_SMUIO',
    23: 'SOC21_IH_CLIENTID_DF',
    24: 'SOC21_IH_CLIENTID_VPE',
    25: 'SOC21_IH_CLIENTID_PWR',
    26: 'SOC21_IH_CLIENTID_LSDMA',
    30: 'SOC21_IH_CLIENTID_MP0',
    31: 'SOC21_IH_CLIENTID_MP1',
    32: 'SOC21_IH_CLIENTID_MAX',
}
SOC21_IH_CLIENTID_IH = 0
SOC21_IH_CLIENTID_ATHUB = 2
SOC21_IH_CLIENTID_BIF = 3
SOC21_IH_CLIENTID_DCN = 4
SOC21_IH_CLIENTID_ISP = 5
SOC21_IH_CLIENTID_MP3 = 6
SOC21_IH_CLIENTID_RLC = 7
SOC21_IH_CLIENTID_GFX = 10
SOC21_IH_CLIENTID_IMU = 11
SOC21_IH_CLIENTID_VCN1 = 14
SOC21_IH_CLIENTID_THM = 15
SOC21_IH_CLIENTID_VCN = 16
SOC21_IH_CLIENTID_VPE1 = 17
SOC21_IH_CLIENTID_VMC = 18
SOC21_IH_CLIENTID_GRBM_CP = 20
SOC21_IH_CLIENTID_ROM_SMUIO = 22
SOC21_IH_CLIENTID_DF = 23
SOC21_IH_CLIENTID_VPE = 24
SOC21_IH_CLIENTID_PWR = 25
SOC21_IH_CLIENTID_LSDMA = 26
SOC21_IH_CLIENTID_MP0 = 30
SOC21_IH_CLIENTID_MP1 = 31
SOC21_IH_CLIENTID_MAX = 32
soc21_ih_clientid = ctypes.c_uint32 # enum
__all__ = \
    ['ACP_HWID', 'AID0_NODEID', 'AID1_NODEID', 'AID2_NODEID',
    'AID3_NODEID', 'AMDGPU_CPCE_UCODE_LOADED',
    'AMDGPU_CPMEC1_UCODE_LOADED', 'AMDGPU_CPMEC2_UCODE_LOADED',
    'AMDGPU_CPME_UCODE_LOADED', 'AMDGPU_CPPFP_UCODE_LOADED',
    'AMDGPU_CPRLC_UCODE_LOADED', 'AMDGPU_DOORBELL64_ASSIGNMENT',
    'AMDGPU_DOORBELL64_DIQ', 'AMDGPU_DOORBELL64_FIRST_NON_CP',
    'AMDGPU_DOORBELL64_GFX_RING0', 'AMDGPU_DOORBELL64_HIQ',
    'AMDGPU_DOORBELL64_IH', 'AMDGPU_DOORBELL64_IH_RING1',
    'AMDGPU_DOORBELL64_IH_RING2', 'AMDGPU_DOORBELL64_INVALID',
    'AMDGPU_DOORBELL64_KIQ', 'AMDGPU_DOORBELL64_LAST_NON_CP',
    'AMDGPU_DOORBELL64_MAX_ASSIGNMENT', 'AMDGPU_DOORBELL64_MEC_RING0',
    'AMDGPU_DOORBELL64_MEC_RING1', 'AMDGPU_DOORBELL64_MEC_RING2',
    'AMDGPU_DOORBELL64_MEC_RING3', 'AMDGPU_DOORBELL64_MEC_RING4',
    'AMDGPU_DOORBELL64_MEC_RING5', 'AMDGPU_DOORBELL64_MEC_RING6',
    'AMDGPU_DOORBELL64_MEC_RING7', 'AMDGPU_DOORBELL64_USERQUEUE_END',
    'AMDGPU_DOORBELL64_USERQUEUE_START',
    'AMDGPU_DOORBELL64_UVD_RING0_1', 'AMDGPU_DOORBELL64_UVD_RING2_3',
    'AMDGPU_DOORBELL64_UVD_RING4_5', 'AMDGPU_DOORBELL64_UVD_RING6_7',
    'AMDGPU_DOORBELL64_VCE_RING0_1', 'AMDGPU_DOORBELL64_VCE_RING2_3',
    'AMDGPU_DOORBELL64_VCE_RING4_5', 'AMDGPU_DOORBELL64_VCE_RING6_7',
    'AMDGPU_DOORBELL64_VCN0_1', 'AMDGPU_DOORBELL64_VCN2_3',
    'AMDGPU_DOORBELL64_VCN4_5', 'AMDGPU_DOORBELL64_VCN6_7',
    'AMDGPU_DOORBELL64_sDMA_ENGINE0',
    'AMDGPU_DOORBELL64_sDMA_ENGINE1',
    'AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE0',
    'AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE1',
    'AMDGPU_DOORBELL_ASSIGNMENT',
    'AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1', 'AMDGPU_DOORBELL_DIQ',
    'AMDGPU_DOORBELL_GFX_RING0', 'AMDGPU_DOORBELL_H',
    'AMDGPU_DOORBELL_HIQ', 'AMDGPU_DOORBELL_IH',
    'AMDGPU_DOORBELL_INVALID', 'AMDGPU_DOORBELL_KIQ',
    'AMDGPU_DOORBELL_LAYOUT1_DIQ',
    'AMDGPU_DOORBELL_LAYOUT1_FIRST_NON_CP',
    'AMDGPU_DOORBELL_LAYOUT1_HIQ', 'AMDGPU_DOORBELL_LAYOUT1_IH',
    'AMDGPU_DOORBELL_LAYOUT1_INVALID',
    'AMDGPU_DOORBELL_LAYOUT1_KIQ_START',
    'AMDGPU_DOORBELL_LAYOUT1_LAST_NON_CP',
    'AMDGPU_DOORBELL_LAYOUT1_MAX_ASSIGNMENT',
    'AMDGPU_DOORBELL_LAYOUT1_MEC_RING_END',
    'AMDGPU_DOORBELL_LAYOUT1_MEC_RING_START',
    'AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_END',
    'AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_START',
    'AMDGPU_DOORBELL_LAYOUT1_VCN_END',
    'AMDGPU_DOORBELL_LAYOUT1_VCN_START',
    'AMDGPU_DOORBELL_LAYOUT1_XCC_RANGE',
    'AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_END',
    'AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_START',
    'AMDGPU_DOORBELL_MAX_ASSIGNMENT', 'AMDGPU_DOORBELL_MEC_RING0',
    'AMDGPU_DOORBELL_MEC_RING1', 'AMDGPU_DOORBELL_MEC_RING2',
    'AMDGPU_DOORBELL_MEC_RING3', 'AMDGPU_DOORBELL_MEC_RING4',
    'AMDGPU_DOORBELL_MEC_RING5', 'AMDGPU_DOORBELL_MEC_RING6',
    'AMDGPU_DOORBELL_MEC_RING7', 'AMDGPU_DOORBELL_sDMA_ENGINE0',
    'AMDGPU_DOORBELL_sDMA_ENGINE1', 'AMDGPU_FW_LOAD_DIRECT',
    'AMDGPU_FW_LOAD_PSP', 'AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO',
    'AMDGPU_FW_LOAD_SMU', 'AMDGPU_GFXHUB_START',
    'AMDGPU_IRQ_CLIENTID_LEGACY', 'AMDGPU_IRQ_CLIENTID_MAX',
    'AMDGPU_IRQ_SRC_DATA_MAX_SIZE_DW', 'AMDGPU_IRQ_STATE_DISABLE',
    'AMDGPU_IRQ_STATE_ENABLE', 'AMDGPU_MAX_IRQ_CLIENT_ID',
    'AMDGPU_MAX_IRQ_SRC_ID', 'AMDGPU_MAX_VMHUBS',
    'AMDGPU_MMHUB0_START', 'AMDGPU_MMHUB1_START', 'AMDGPU_MTYPE_CC',
    'AMDGPU_MTYPE_NC', 'AMDGPU_NAVI10_DOORBELL64_FIRST_NON_CP',
    'AMDGPU_NAVI10_DOORBELL64_LAST_NON_CP',
    'AMDGPU_NAVI10_DOORBELL64_VCN0_1',
    'AMDGPU_NAVI10_DOORBELL64_VCN2_3',
    'AMDGPU_NAVI10_DOORBELL64_VCN4_5',
    'AMDGPU_NAVI10_DOORBELL64_VCN6_7',
    'AMDGPU_NAVI10_DOORBELL64_VCN8_9',
    'AMDGPU_NAVI10_DOORBELL64_VCNa_b',
    'AMDGPU_NAVI10_DOORBELL64_VCNc_d',
    'AMDGPU_NAVI10_DOORBELL64_VCNe_f', 'AMDGPU_NAVI10_DOORBELL64_VPE',
    'AMDGPU_NAVI10_DOORBELL_ASSIGNMENT', 'AMDGPU_NAVI10_DOORBELL_DIQ',
    'AMDGPU_NAVI10_DOORBELL_GFX_RING0',
    'AMDGPU_NAVI10_DOORBELL_GFX_RING1',
    'AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_END',
    'AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_START',
    'AMDGPU_NAVI10_DOORBELL_HIQ', 'AMDGPU_NAVI10_DOORBELL_IH',
    'AMDGPU_NAVI10_DOORBELL_INVALID', 'AMDGPU_NAVI10_DOORBELL_KIQ',
    'AMDGPU_NAVI10_DOORBELL_MAX_ASSIGNMENT',
    'AMDGPU_NAVI10_DOORBELL_MEC_RING0',
    'AMDGPU_NAVI10_DOORBELL_MEC_RING1',
    'AMDGPU_NAVI10_DOORBELL_MEC_RING2',
    'AMDGPU_NAVI10_DOORBELL_MEC_RING3',
    'AMDGPU_NAVI10_DOORBELL_MEC_RING4',
    'AMDGPU_NAVI10_DOORBELL_MEC_RING5',
    'AMDGPU_NAVI10_DOORBELL_MEC_RING6',
    'AMDGPU_NAVI10_DOORBELL_MEC_RING7',
    'AMDGPU_NAVI10_DOORBELL_MES_RING0',
    'AMDGPU_NAVI10_DOORBELL_MES_RING1',
    'AMDGPU_NAVI10_DOORBELL_USERQUEUE_END',
    'AMDGPU_NAVI10_DOORBELL_USERQUEUE_START',
    'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0',
    'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE1',
    'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE2',
    'AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE3', 'AMDGPU_PDE_PTE',
    'AMDGPU_PDE_PTE_GFX12', 'AMDGPU_PTE_DEFAULT_ATC',
    'AMDGPU_PTE_EXECUTABLE', 'AMDGPU_PTE_IS_PTE', 'AMDGPU_PTE_LOG',
    'AMDGPU_PTE_MTYPE_GFX12_MASK', 'AMDGPU_PTE_MTYPE_NV10_MASK',
    'AMDGPU_PTE_MTYPE_VG10_MASK', 'AMDGPU_PTE_NOALLOC',
    'AMDGPU_PTE_PRT', 'AMDGPU_PTE_PRT_GFX12', 'AMDGPU_PTE_READABLE',
    'AMDGPU_PTE_SNOOPED', 'AMDGPU_PTE_SYSTEM', 'AMDGPU_PTE_TF',
    'AMDGPU_PTE_TMZ', 'AMDGPU_PTE_VALID', 'AMDGPU_PTE_WRITEABLE',
    'AMDGPU_SDMA0_UCODE_LOADED', 'AMDGPU_SDMA1_UCODE_LOADED',
    'AMDGPU_UCODE_ID', 'AMDGPU_UCODE_ID_CAP', 'AMDGPU_UCODE_ID_CP_CE',
    'AMDGPU_UCODE_ID_CP_ME', 'AMDGPU_UCODE_ID_CP_MEC1',
    'AMDGPU_UCODE_ID_CP_MEC1_JT', 'AMDGPU_UCODE_ID_CP_MEC2',
    'AMDGPU_UCODE_ID_CP_MEC2_JT', 'AMDGPU_UCODE_ID_CP_MES',
    'AMDGPU_UCODE_ID_CP_MES1', 'AMDGPU_UCODE_ID_CP_MES1_DATA',
    'AMDGPU_UCODE_ID_CP_MES_DATA', 'AMDGPU_UCODE_ID_CP_PFP',
    'AMDGPU_UCODE_ID_CP_RS64_ME', 'AMDGPU_UCODE_ID_CP_RS64_MEC',
    'AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_PFP',
    'AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK',
    'AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK', 'AMDGPU_UCODE_ID_DMCUB',
    'AMDGPU_UCODE_ID_DMCU_ERAM', 'AMDGPU_UCODE_ID_DMCU_INTV',
    'AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS', 'AMDGPU_UCODE_ID_IMU_D',
    'AMDGPU_UCODE_ID_IMU_I', 'AMDGPU_UCODE_ID_ISP',
    'AMDGPU_UCODE_ID_JPEG_RAM', 'AMDGPU_UCODE_ID_MAXIMUM',
    'AMDGPU_UCODE_ID_P2S_TABLE', 'AMDGPU_UCODE_ID_PPTABLE',
    'AMDGPU_UCODE_ID_RLC_DRAM', 'AMDGPU_UCODE_ID_RLC_G',
    'AMDGPU_UCODE_ID_RLC_IRAM', 'AMDGPU_UCODE_ID_RLC_P',
    'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL',
    'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM',
    'AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM',
    'AMDGPU_UCODE_ID_RLC_V', 'AMDGPU_UCODE_ID_SDMA0',
    'AMDGPU_UCODE_ID_SDMA1', 'AMDGPU_UCODE_ID_SDMA2',
    'AMDGPU_UCODE_ID_SDMA3', 'AMDGPU_UCODE_ID_SDMA4',
    'AMDGPU_UCODE_ID_SDMA5', 'AMDGPU_UCODE_ID_SDMA6',
    'AMDGPU_UCODE_ID_SDMA7', 'AMDGPU_UCODE_ID_SDMA_RS64',
    'AMDGPU_UCODE_ID_SDMA_UCODE_TH0',
    'AMDGPU_UCODE_ID_SDMA_UCODE_TH1',
    'AMDGPU_UCODE_ID_SE0_TAP_DELAYS',
    'AMDGPU_UCODE_ID_SE1_TAP_DELAYS',
    'AMDGPU_UCODE_ID_SE2_TAP_DELAYS',
    'AMDGPU_UCODE_ID_SE3_TAP_DELAYS', 'AMDGPU_UCODE_ID_SMC',
    'AMDGPU_UCODE_ID_STORAGE', 'AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER',
    'AMDGPU_UCODE_ID_UMSCH_MM_DATA', 'AMDGPU_UCODE_ID_UMSCH_MM_UCODE',
    'AMDGPU_UCODE_ID_UVD', 'AMDGPU_UCODE_ID_UVD1',
    'AMDGPU_UCODE_ID_VCE', 'AMDGPU_UCODE_ID_VCN',
    'AMDGPU_UCODE_ID_VCN0_RAM', 'AMDGPU_UCODE_ID_VCN1',
    'AMDGPU_UCODE_ID_VCN1_RAM', 'AMDGPU_UCODE_ID_VPE',
    'AMDGPU_UCODE_ID_VPE_CTL', 'AMDGPU_UCODE_ID_VPE_CTX',
    'AMDGPU_UCODE_STATUS', 'AMDGPU_UCODE_STATUS_INVALID',
    'AMDGPU_UCODE_STATUS_LOADED', 'AMDGPU_UCODE_STATUS_NOT_LOADED',
    'AMDGPU_VA_RESERVED_BOTTOM', 'AMDGPU_VA_RESERVED_CSA_SIZE',
    'AMDGPU_VA_RESERVED_SEQ64_SIZE', 'AMDGPU_VA_RESERVED_TOP',
    'AMDGPU_VA_RESERVED_TRAP_SIZE',
    'AMDGPU_VEGA20_DOORBELL64_FIRST_NON_CP',
    'AMDGPU_VEGA20_DOORBELL64_LAST_NON_CP',
    'AMDGPU_VEGA20_DOORBELL64_UVD_RING0_1',
    'AMDGPU_VEGA20_DOORBELL64_UVD_RING2_3',
    'AMDGPU_VEGA20_DOORBELL64_UVD_RING4_5',
    'AMDGPU_VEGA20_DOORBELL64_UVD_RING6_7',
    'AMDGPU_VEGA20_DOORBELL64_VCE_RING0_1',
    'AMDGPU_VEGA20_DOORBELL64_VCE_RING2_3',
    'AMDGPU_VEGA20_DOORBELL64_VCE_RING4_5',
    'AMDGPU_VEGA20_DOORBELL64_VCE_RING6_7',
    'AMDGPU_VEGA20_DOORBELL64_VCN0_1',
    'AMDGPU_VEGA20_DOORBELL64_VCN2_3',
    'AMDGPU_VEGA20_DOORBELL64_VCN4_5',
    'AMDGPU_VEGA20_DOORBELL64_VCN6_7',
    'AMDGPU_VEGA20_DOORBELL64_VCN8_9',
    'AMDGPU_VEGA20_DOORBELL64_VCNa_b',
    'AMDGPU_VEGA20_DOORBELL64_VCNc_d',
    'AMDGPU_VEGA20_DOORBELL64_VCNe_f',
    'AMDGPU_VEGA20_DOORBELL_AID1_sDMA_START',
    'AMDGPU_VEGA20_DOORBELL_ASSIGNMENT', 'AMDGPU_VEGA20_DOORBELL_DIQ',
    'AMDGPU_VEGA20_DOORBELL_GFX_RING0', 'AMDGPU_VEGA20_DOORBELL_HIQ',
    'AMDGPU_VEGA20_DOORBELL_IH', 'AMDGPU_VEGA20_DOORBELL_INVALID',
    'AMDGPU_VEGA20_DOORBELL_KIQ',
    'AMDGPU_VEGA20_DOORBELL_MAX_ASSIGNMENT',
    'AMDGPU_VEGA20_DOORBELL_MEC_RING0',
    'AMDGPU_VEGA20_DOORBELL_MEC_RING1',
    'AMDGPU_VEGA20_DOORBELL_MEC_RING2',
    'AMDGPU_VEGA20_DOORBELL_MEC_RING3',
    'AMDGPU_VEGA20_DOORBELL_MEC_RING4',
    'AMDGPU_VEGA20_DOORBELL_MEC_RING5',
    'AMDGPU_VEGA20_DOORBELL_MEC_RING6',
    'AMDGPU_VEGA20_DOORBELL_MEC_RING7',
    'AMDGPU_VEGA20_DOORBELL_USERQUEUE_END',
    'AMDGPU_VEGA20_DOORBELL_USERQUEUE_START',
    'AMDGPU_VEGA20_DOORBELL_XCC1_KIQ_START',
    'AMDGPU_VEGA20_DOORBELL_XCC1_MEC_RING0_START',
    'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE0',
    'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE1',
    'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE2',
    'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE3',
    'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE4',
    'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE5',
    'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE6',
    'AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE7',
    'AMDGPU_VM_FAULT_STOP_ALWAYS', 'AMDGPU_VM_FAULT_STOP_FIRST',
    'AMDGPU_VM_FAULT_STOP_NEVER', 'AMDGPU_VM_MAX_UPDATE_SIZE',
    'AMDGPU_VM_NORETRY_FLAGS', 'AMDGPU_VM_NORETRY_FLAGS_TF',
    'AMDGPU_VM_PDB0', 'AMDGPU_VM_PDB1', 'AMDGPU_VM_PDB2',
    'AMDGPU_VM_PTB', 'AMDGPU_VM_RESERVED_VRAM',
    'AMDGPU_VM_USE_CPU_FOR_COMPUTE', 'AMDGPU_VM_USE_CPU_FOR_GFX',
    'AMDGPU_XGMI_MAX_CONNECTED_NODES', 'ATHUB_HWID', 'ATHUB_HWIP',
    'AUDIO_AZ_HWID', 'BINARY_SIGNATURE',
    'BIST_MEM_TRAINING_ENCROACHED_SIZE', 'BOOTCFG_CMD_GET',
    'BOOTCFG_CMD_INVALIDATE', 'BOOTCFG_CMD_SET',
    'BOOT_CFG_FEATURE_GECC',
    'BOOT_CFG_FEATURE_TWO_STAGE_DRAM_TRAINING', 'BOOT_CONFIG_GECC',
    'C2PMSG_CMD_GFX_USB_PD_FW_VER', 'CCXSEC_HWID', 'CLKA_HWID',
    'CLKB_HWID', 'CLK_HWIP', 'DAZ_HWID', 'DBGU0_HWID', 'DBGU1_HWID',
    'DBGU_IO_HWID', 'DBGU_NBIO_HWID', 'DCEAZ_HWID', 'DCE_HWIP',
    'DCI_HWID', 'DCI_HWIP', 'DCO_HWID', 'DDCL_HWID', 'DFX_DAP_HWID',
    'DFX_HWID', 'DF_HWID', 'DF_HWIP', 'DIO_HWID',
    'DISCOVERY_TABLE_SIGNATURE', 'DMU_HWID', 'FCH_HWID',
    'FCH_USB_PD_HWID', 'FRAME_TYPE_DESTROY', 'FUSE_HWID', 'GC',
    'GC_HWID', 'GC_HWIP', 'GC_TABLE_ID',
    'GDDR6_MEM_TRAINING_DATA_SIZE_IN_BYTES',
    'GDDR6_MEM_TRAINING_OFFSET', 'GFX_BUF_MAX_DESC',
    'GFX_CMD_ID_AUTOLOAD_RLC', 'GFX_CMD_ID_BOOT_CFG',
    'GFX_CMD_ID_DESTROY_TMR', 'GFX_CMD_ID_DESTROY_VMR',
    'GFX_CMD_ID_GET_FW_ATTESTATION', 'GFX_CMD_ID_INVOKE_CMD',
    'GFX_CMD_ID_LOAD_ASD', 'GFX_CMD_ID_LOAD_IP_FW',
    'GFX_CMD_ID_LOAD_TA', 'GFX_CMD_ID_LOAD_TOC', 'GFX_CMD_ID_MASK',
    'GFX_CMD_ID_PROG_REG', 'GFX_CMD_ID_SAVE_RESTORE',
    'GFX_CMD_ID_SETUP_TMR', 'GFX_CMD_ID_SETUP_VMR',
    'GFX_CMD_ID_SRIOV_SPATIAL_PART', 'GFX_CMD_ID_UNLOAD_TA',
    'GFX_CMD_RESERVED_MASK', 'GFX_CMD_RESPONSE_MASK',
    'GFX_CMD_STATUS_MASK', 'GFX_CTRL_CMD_ID_CAN_INIT_RINGS',
    'GFX_CTRL_CMD_ID_CONSUME_CMD',
    'GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING',
    'GFX_CTRL_CMD_ID_DESTROY_RINGS', 'GFX_CTRL_CMD_ID_DISABLE_INT',
    'GFX_CTRL_CMD_ID_ENABLE_INT', 'GFX_CTRL_CMD_ID_GBR_IH_SET',
    'GFX_CTRL_CMD_ID_INIT_GPCOM_RING',
    'GFX_CTRL_CMD_ID_INIT_RBI_RING', 'GFX_CTRL_CMD_ID_MAX',
    'GFX_CTRL_CMD_ID_MODE1_RST', 'GFX_FLAG_RESPONSE',
    'GFX_FW_TYPE_ACCUM_CTRL_RAM', 'GFX_FW_TYPE_ACP',
    'GFX_FW_TYPE_CAP', 'GFX_FW_TYPE_CP_CE', 'GFX_FW_TYPE_CP_ME',
    'GFX_FW_TYPE_CP_MEC', 'GFX_FW_TYPE_CP_MEC_ME1',
    'GFX_FW_TYPE_CP_MEC_ME2', 'GFX_FW_TYPE_CP_MES',
    'GFX_FW_TYPE_CP_MES_KIQ', 'GFX_FW_TYPE_CP_PFP',
    'GFX_FW_TYPE_DISCRETE_USB4', 'GFX_FW_TYPE_DMCU_ERAM',
    'GFX_FW_TYPE_DMCU_ISR', 'GFX_FW_TYPE_DMUB',
    'GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM',
    'GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS',
    'GFX_FW_TYPE_GLOBAL_TAP_DELAYS', 'GFX_FW_TYPE_IMU_D',
    'GFX_FW_TYPE_IMU_I', 'GFX_FW_TYPE_ISP', 'GFX_FW_TYPE_ISP_DATA',
    'GFX_FW_TYPE_JPEG_RAM', 'GFX_FW_TYPE_LSDMA', 'GFX_FW_TYPE_MAX',
    'GFX_FW_TYPE_MES_KIQ_STACK', 'GFX_FW_TYPE_MES_STACK',
    'GFX_FW_TYPE_MMSCH', 'GFX_FW_TYPE_NONE', 'GFX_FW_TYPE_P2S_TABLE',
    'GFX_FW_TYPE_PPTABLE', 'GFX_FW_TYPE_REG_LIST',
    'GFX_FW_TYPE_RLCG_SCRATCH_SR', 'GFX_FW_TYPE_RLCP_CAM',
    'GFX_FW_TYPE_RLCP_SCRATCH_SR', 'GFX_FW_TYPE_RLCV_SCRATCH_SR',
    'GFX_FW_TYPE_RLC_DRAM_BOOT', 'GFX_FW_TYPE_RLC_G',
    'GFX_FW_TYPE_RLC_IRAM', 'GFX_FW_TYPE_RLC_P',
    'GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM',
    'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL',
    'GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM',
    'GFX_FW_TYPE_RLC_SPP_CAM_EXT', 'GFX_FW_TYPE_RLC_SRM_DRAM_SR',
    'GFX_FW_TYPE_RLC_V', 'GFX_FW_TYPE_RLX6_DRAM_SR',
    'GFX_FW_TYPE_RS64_KIQ', 'GFX_FW_TYPE_RS64_KIQ_STACK',
    'GFX_FW_TYPE_RS64_ME', 'GFX_FW_TYPE_RS64_MEC',
    'GFX_FW_TYPE_RS64_MEC_P0_STACK', 'GFX_FW_TYPE_RS64_MEC_P1_STACK',
    'GFX_FW_TYPE_RS64_MEC_P2_STACK', 'GFX_FW_TYPE_RS64_MEC_P3_STACK',
    'GFX_FW_TYPE_RS64_MES', 'GFX_FW_TYPE_RS64_MES_STACK',
    'GFX_FW_TYPE_RS64_ME_P0_STACK', 'GFX_FW_TYPE_RS64_ME_P1_STACK',
    'GFX_FW_TYPE_RS64_PFP', 'GFX_FW_TYPE_RS64_PFP_P0_STACK',
    'GFX_FW_TYPE_RS64_PFP_P1_STACK', 'GFX_FW_TYPE_SDMA0',
    'GFX_FW_TYPE_SDMA0_JT', 'GFX_FW_TYPE_SDMA0_PG_CONTEXT',
    'GFX_FW_TYPE_SDMA1', 'GFX_FW_TYPE_SDMA1_JT',
    'GFX_FW_TYPE_SDMA1_PG_CONTEXT', 'GFX_FW_TYPE_SDMA2',
    'GFX_FW_TYPE_SDMA3', 'GFX_FW_TYPE_SDMA4', 'GFX_FW_TYPE_SDMA5',
    'GFX_FW_TYPE_SDMA6', 'GFX_FW_TYPE_SDMA7',
    'GFX_FW_TYPE_SDMA_UCODE_TH0', 'GFX_FW_TYPE_SDMA_UCODE_TH1',
    'GFX_FW_TYPE_SE0_MUX_SELECT_RAM', 'GFX_FW_TYPE_SE0_TAP_DELAYS',
    'GFX_FW_TYPE_SE1_MUX_SELECT_RAM', 'GFX_FW_TYPE_SE1_TAP_DELAYS',
    'GFX_FW_TYPE_SE2_TAP_DELAYS', 'GFX_FW_TYPE_SE3_TAP_DELAYS',
    'GFX_FW_TYPE_SMU', 'GFX_FW_TYPE_TA', 'GFX_FW_TYPE_TOC',
    'GFX_FW_TYPE_UMSCH_CMD_BUFFER', 'GFX_FW_TYPE_UMSCH_DATA',
    'GFX_FW_TYPE_UMSCH_UCODE', 'GFX_FW_TYPE_USB_DP_COMBO_PHY',
    'GFX_FW_TYPE_UVD', 'GFX_FW_TYPE_UVD1', 'GFX_FW_TYPE_VCE',
    'GFX_FW_TYPE_VCN', 'GFX_FW_TYPE_VCN0_RAM', 'GFX_FW_TYPE_VCN1',
    'GFX_FW_TYPE_VCN1_RAM', 'GFX_FW_TYPE_VPE', 'GFX_FW_TYPE_VPEC_FW1',
    'GFX_FW_TYPE_VPEC_FW2', 'HARVEST_INFO', 'HARVEST_TABLE_SIGNATURE',
    'HDP_HWID', 'HDP_HWIP', 'HWIP_MAX_INSTANCE', 'HW_ID_MAX',
    'IOAGR_HWID', 'IOAPIC_HWID', 'IOHC_HWID', 'IP_DISCOVERY',
    'ISP_HWID', 'ISP_HWIP', 'JPEG_HWIP', 'L1IMU10_HWID',
    'L1IMU11_HWID', 'L1IMU12_HWID', 'L1IMU13_HWID', 'L1IMU14_HWID',
    'L1IMU15_HWID', 'L1IMU3_HWID', 'L1IMU4_HWID', 'L1IMU5_HWID',
    'L1IMU6_HWID', 'L1IMU7_HWID', 'L1IMU8_HWID', 'L1IMU9_HWID',
    'L1IMU_IOAGR_HWID', 'L1IMU_NBIF_HWID', 'L1IMU_PCIE_HWID',
    'L2IMU_HWID', 'LSDMA_HWID', 'LSDMA_HWIP', 'MALL_INFO',
    'MALL_INFO_TABLE_ID', 'MAX_HWIP', 'MEM_TRAIN_SYSTEM_SIGNATURE',
    'MMHUB_HWID', 'MMHUB_HWIP', 'MP0_HWID', 'MP0_HWIP', 'MP1_HWID',
    'MP1_HWIP', 'MP2_HWID', 'NBIF_HWID', 'NBIF_HWIP', 'NBIO_HWIP',
    'NODEID_MAX', 'NPS_INFO', 'NPS_INFO_TABLE_ID',
    'NPS_INFO_TABLE_MAX_NUM_INSTANCES', 'NTBCCP_HWID', 'NTB_HWID',
    'OSSSYS_HWID', 'OSSSYS_HWIP', 'PCIE_HWID', 'PCIE_HWIP',
    'PCS_HWID', 'PSP_1_MEG', 'PSP_ASD_SHARED_MEM_SIZE',
    'PSP_BL__DRAM_LONG_TRAIN', 'PSP_BL__DRAM_SHORT_TRAIN',
    'PSP_BL__LOAD_DBGDRV', 'PSP_BL__LOAD_HADDRV',
    'PSP_BL__LOAD_INTFDRV', 'PSP_BL__LOAD_IPKEYMGRDRV',
    'PSP_BL__LOAD_KEY_DATABASE', 'PSP_BL__LOAD_RASDRV',
    'PSP_BL__LOAD_SOCDRV', 'PSP_BL__LOAD_SOSDRV',
    'PSP_BL__LOAD_SYSDRV', 'PSP_BL__LOAD_TOS_SPL_TABLE',
    'PSP_CMD_BUFFER_SIZE', 'PSP_DTM_SHARED_MEM_SIZE',
    'PSP_ERR_UNKNOWN_COMMAND', 'PSP_FENCE_BUFFER_SIZE',
    'PSP_FW_NAME_LEN', 'PSP_FW_TYPE_MAX_INDEX',
    'PSP_FW_TYPE_PSP_DBG_DRV', 'PSP_FW_TYPE_PSP_INTF_DRV',
    'PSP_FW_TYPE_PSP_IPKEYMGR_DRV', 'PSP_FW_TYPE_PSP_KDB',
    'PSP_FW_TYPE_PSP_RAS_DRV', 'PSP_FW_TYPE_PSP_RL',
    'PSP_FW_TYPE_PSP_SOC_DRV', 'PSP_FW_TYPE_PSP_SOS',
    'PSP_FW_TYPE_PSP_SPL', 'PSP_FW_TYPE_PSP_SYS_DRV',
    'PSP_FW_TYPE_PSP_TOC', 'PSP_FW_TYPE_UNKOWN',
    'PSP_GFX_CMD_BUF_VERSION', 'PSP_HDCP_SHARED_MEM_SIZE',
    'PSP_HEADER_SIZE', 'PSP_MEM_TRAIN_COLD_BOOT',
    'PSP_MEM_TRAIN_INIT_FAILED', 'PSP_MEM_TRAIN_INIT_SUCCESS',
    'PSP_MEM_TRAIN_NOT_SUPPORT', 'PSP_MEM_TRAIN_RESERVE_SUCCESS',
    'PSP_MEM_TRAIN_RESTORE', 'PSP_MEM_TRAIN_RESUME',
    'PSP_MEM_TRAIN_SAVE', 'PSP_MEM_TRAIN_SEND_LONG_MSG',
    'PSP_MEM_TRAIN_SEND_SHORT_MSG', 'PSP_MEM_TRAIN_SUPPORT',
    'PSP_RAP_SHARED_MEM_SIZE', 'PSP_RAS_SHARED_MEM_SIZE',
    'PSP_REG_IH_RB_CNTL', 'PSP_REG_IH_RB_CNTL_RING1',
    'PSP_REG_IH_RB_CNTL_RING2', 'PSP_REG_LAST',
    'PSP_RING_TYPE__INVALID', 'PSP_RING_TYPE__KM',
    'PSP_RING_TYPE__UM', 'PSP_RUNTIME_DB_COOKIE_ID',
    'PSP_RUNTIME_DB_DIAG_ENTRY_MAX_COUNT', 'PSP_RUNTIME_DB_OFFSET',
    'PSP_RUNTIME_DB_SIZE_IN_BYTES', 'PSP_RUNTIME_DB_VER_1',
    'PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG',
    'PSP_RUNTIME_ENTRY_TYPE_INVALID',
    'PSP_RUNTIME_ENTRY_TYPE_MGPU_COMMON',
    'PSP_RUNTIME_ENTRY_TYPE_MGPU_WAFL',
    'PSP_RUNTIME_ENTRY_TYPE_MGPU_XGMI',
    'PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS',
    'PSP_RUNTIME_ENTRY_TYPE_TEST',
    'PSP_SECUREDISPLAY_SHARED_MEM_SIZE', 'PSP_TMR_ALIGNMENT',
    'PSP_XGMI_SHARED_MEM_SIZE', 'PWR_HWID', 'PWR_HWIP', 'RSMU_HWIP',
    'SATA_HWID', 'SCPM_DISABLE', 'SCPM_ENABLE',
    'SCPM_ENABLE_WITH_SCPM_ERR', 'SDMA0_HWID', 'SDMA0_HWIP',
    'SDMA1_HWID', 'SDMA1_HWIP', 'SDMA2_HWID', 'SDMA2_HWIP',
    'SDMA3_HWID', 'SDMA3_HWIP', 'SDMA4_HWIP', 'SDMA5_HWIP',
    'SDMA6_HWIP', 'SDMA7_HWIP', 'SDPMUX_HWID', 'SMUIO_HWID',
    'SMUIO_HWIP', 'SOC15_IH_CLIENTID_ACP', 'SOC15_IH_CLIENTID_ATHUB',
    'SOC15_IH_CLIENTID_ATS', 'SOC15_IH_CLIENTID_BIF',
    'SOC15_IH_CLIENTID_DCE', 'SOC15_IH_CLIENTID_DF',
    'SOC15_IH_CLIENTID_EA', 'SOC15_IH_CLIENTID_GRBM_CP',
    'SOC15_IH_CLIENTID_IH', 'SOC15_IH_CLIENTID_ISP',
    'SOC15_IH_CLIENTID_MAX', 'SOC15_IH_CLIENTID_MP0',
    'SOC15_IH_CLIENTID_MP1', 'SOC15_IH_CLIENTID_PCIE0',
    'SOC15_IH_CLIENTID_PWR', 'SOC15_IH_CLIENTID_RESERVED',
    'SOC15_IH_CLIENTID_RLC', 'SOC15_IH_CLIENTID_ROM_SMUIO',
    'SOC15_IH_CLIENTID_SDMA0', 'SOC15_IH_CLIENTID_SDMA1',
    'SOC15_IH_CLIENTID_SDMA2', 'SOC15_IH_CLIENTID_SDMA3',
    'SOC15_IH_CLIENTID_SDMA3_Sienna_Cichlid',
    'SOC15_IH_CLIENTID_SDMA4', 'SOC15_IH_CLIENTID_SDMA5',
    'SOC15_IH_CLIENTID_SDMA6', 'SOC15_IH_CLIENTID_SDMA7',
    'SOC15_IH_CLIENTID_SE0SH', 'SOC15_IH_CLIENTID_SE1SH',
    'SOC15_IH_CLIENTID_SE2SH', 'SOC15_IH_CLIENTID_SE3SH',
    'SOC15_IH_CLIENTID_THM', 'SOC15_IH_CLIENTID_UTCL2',
    'SOC15_IH_CLIENTID_UTCL2LOG', 'SOC15_IH_CLIENTID_UVD',
    'SOC15_IH_CLIENTID_UVD1', 'SOC15_IH_CLIENTID_VCE0',
    'SOC15_IH_CLIENTID_VCE1', 'SOC15_IH_CLIENTID_VCN',
    'SOC15_IH_CLIENTID_VCN1', 'SOC15_IH_CLIENTID_VMC',
    'SOC15_IH_CLIENTID_VMC1', 'SOC15_IH_CLIENTID_XDMA',
    'SOC21_IH_CLIENTID_ATHUB', 'SOC21_IH_CLIENTID_BIF',
    'SOC21_IH_CLIENTID_DCN', 'SOC21_IH_CLIENTID_DF',
    'SOC21_IH_CLIENTID_GFX', 'SOC21_IH_CLIENTID_GRBM_CP',
    'SOC21_IH_CLIENTID_IH', 'SOC21_IH_CLIENTID_IMU',
    'SOC21_IH_CLIENTID_ISP', 'SOC21_IH_CLIENTID_LSDMA',
    'SOC21_IH_CLIENTID_MAX', 'SOC21_IH_CLIENTID_MP0',
    'SOC21_IH_CLIENTID_MP1', 'SOC21_IH_CLIENTID_MP3',
    'SOC21_IH_CLIENTID_PWR', 'SOC21_IH_CLIENTID_RLC',
    'SOC21_IH_CLIENTID_ROM_SMUIO', 'SOC21_IH_CLIENTID_THM',
    'SOC21_IH_CLIENTID_VCN', 'SOC21_IH_CLIENTID_VCN1',
    'SOC21_IH_CLIENTID_VMC', 'SOC21_IH_CLIENTID_VPE',
    'SOC21_IH_CLIENTID_VPE1', 'SST_HWID', 'SYSTEMHUB_HWID',
    'TA_FW_TYPE_MAX_INDEX', 'TA_FW_TYPE_PSP_ASD',
    'TA_FW_TYPE_PSP_DTM', 'TA_FW_TYPE_PSP_HDCP', 'TA_FW_TYPE_PSP_RAP',
    'TA_FW_TYPE_PSP_RAS', 'TA_FW_TYPE_PSP_SECUREDISPLAY',
    'TA_FW_TYPE_PSP_XGMI', 'TA_FW_TYPE_UNKOWN', 'TA_TYPE_DTM',
    'TA_TYPE_HDCP', 'TA_TYPE_MAX_INDEX', 'TA_TYPE_RAP', 'TA_TYPE_RAS',
    'TA_TYPE_SECUREDISPLAY', 'TA_TYPE_XGMI',
    'TEE_ERROR_NOT_SUPPORTED', 'TEE_SUCCESS', 'THM_HWID', 'THM_HWIP',
    'TOTAL_TABLES', 'UMC_HWID', 'UMC_HWIP', 'USB_HWID', 'UVD_HWID',
    'UVD_HWIP', 'V11_STRUCTS_H_', 'V12_STRUCTS_H_', 'VCE_HWID',
    'VCE_HWIP', 'VCN1_HWIP', 'VCN_HWID', 'VCN_HWIP', 'VCN_INFO',
    'VCN_INFO_TABLE_ID', 'VCN_INFO_TABLE_MAX_NUM_INSTANCES',
    'VPE_HWID', 'VPE_HWIP', 'WAFLC_HWID', 'XCD0_NODEID',
    'XCD1_NODEID', 'XCD2_NODEID', 'XCD3_NODEID', 'XCD4_NODEID',
    'XCD5_NODEID', 'XCD6_NODEID', 'XCD7_NODEID', 'XDMA_HWID',
    'XGBE_HWID', 'XGMI_HWID', 'XGMI_HWIP', '_DISCOVERY_H_',
    '_PSP_TEE_GFX_IF_H_', '__AMDGPU_IRQ_H__', '__AMDGPU_PSP_H__',
    '__AMDGPU_UCODE_H__', '__AMDGPU_VM_H__',
    '__SOC15_IH_CLIENTID_H__', 'amd_hw_ip_block_type',
    'amdgpu_firmware_load_type', 'amdgpu_interrupt_state',
    'amdgpu_vm_level', 'binary_header', 'bool', 'c__EA_table',
    'die_header', 'die_info', 'harvest_info', 'harvest_info_header',
    'harvest_table', 'hw_id_map', 'int16_t', 'int32_t', 'int8_t',
    'interrupt_node_id_per_aid', 'ip', 'ip_discovery_header',
    'ip_structure', 'ip_v3', 'ip_v4', 'psp_bootloader_cmd',
    'psp_fw_type', 'psp_gfx_boot_config', 'psp_gfx_boot_config_cmd',
    'psp_gfx_cmd_id', 'psp_gfx_crtl_cmd_id', 'psp_gfx_fw_type',
    'psp_memory_training_init_flag', 'psp_memory_training_ops',
    'psp_reg_prog_id', 'psp_ring_type',
    'psp_runtime_boot_cfg_feature', 'psp_runtime_entry_type',
    'psp_runtime_scpm_authentication', 'psp_shared_mem_size',
    'soc15_ih_clientid', 'soc15_ih_clientid_name',
    'soc21_ih_clientid', 'struct__fuse_data_bits',
    'struct_amdgpu_device', 'struct_amdgpu_firmware_info',
    'struct_amdgpu_iv_entry', 'struct_binary_header',
    'struct_common_firmware_header', 'struct_die',
    'struct_die_header', 'struct_die_info',
    'struct_dmcu_firmware_header_v1_0',
    'struct_dmcub_firmware_header_v1_0', 'struct_firmware',
    'struct_gc_info_v1_0', 'struct_gc_info_v1_1',
    'struct_gc_info_v1_2', 'struct_gc_info_v1_3',
    'struct_gc_info_v2_0', 'struct_gc_info_v2_1',
    'struct_gfx_firmware_header_v1_0',
    'struct_gfx_firmware_header_v2_0',
    'struct_gpu_info_firmware_header_v1_0',
    'struct_gpu_info_firmware_v1_0', 'struct_gpu_info_firmware_v1_1',
    'struct_gpu_info_header', 'struct_harvest_info',
    'struct_harvest_info_header', 'struct_harvest_table',
    'struct_imu_firmware_header_v1_0', 'struct_ip',
    'struct_ip_discovery_header', 'struct_ip_discovery_header_0_0',
    'struct_ip_structure', 'struct_ip_v3', 'struct_ip_v4',
    'struct_mall_info_header', 'struct_mall_info_v1_0',
    'struct_mall_info_v2_0', 'struct_mc_firmware_header_v1_0',
    'struct_mes_firmware_header_v1_0', 'struct_nps_info_header',
    'struct_nps_info_v1_0', 'struct_nps_instance_info_v1_0',
    'struct_psp_bin_desc', 'struct_psp_context',
    'struct_psp_firmware_header_v1_0',
    'struct_psp_firmware_header_v1_1',
    'struct_psp_firmware_header_v1_2',
    'struct_psp_firmware_header_v1_3',
    'struct_psp_firmware_header_v2_0',
    'struct_psp_firmware_header_v2_1', 'struct_psp_fw_bin_desc',
    'struct_psp_fw_legacy_bin_desc', 'struct_psp_gfx_buf_desc',
    'struct_psp_gfx_buf_list', 'struct_psp_gfx_cmd_boot_cfg',
    'struct_psp_gfx_cmd_invoke_cmd', 'struct_psp_gfx_cmd_load_ip_fw',
    'struct_psp_gfx_cmd_load_ta', 'struct_psp_gfx_cmd_load_toc',
    'struct_psp_gfx_cmd_reg_prog', 'struct_psp_gfx_cmd_resp',
    'struct_psp_gfx_cmd_save_restore_ip_fw',
    'struct_psp_gfx_cmd_setup_tmr',
    'struct_psp_gfx_cmd_setup_tmr_0_bitfield',
    'struct_psp_gfx_cmd_sriov_spatial_part',
    'struct_psp_gfx_cmd_unload_ta', 'struct_psp_gfx_ctrl',
    'struct_psp_gfx_rb_frame', 'struct_psp_gfx_resp',
    'struct_psp_gfx_uresp_bootcfg',
    'struct_psp_gfx_uresp_fwar_db_info',
    'struct_psp_gfx_uresp_reserved', 'struct_psp_xgmi_node_info',
    'struct_psp_xgmi_topology_info',
    'struct_rlc_firmware_header_v1_0',
    'struct_rlc_firmware_header_v2_0',
    'struct_rlc_firmware_header_v2_1',
    'struct_rlc_firmware_header_v2_2',
    'struct_rlc_firmware_header_v2_3',
    'struct_rlc_firmware_header_v2_4',
    'struct_sdma_firmware_header_v1_0',
    'struct_sdma_firmware_header_v1_1',
    'struct_sdma_firmware_header_v2_0',
    'struct_sdma_firmware_header_v3_0',
    'struct_smc_firmware_header_v1_0',
    'struct_smc_firmware_header_v2_0',
    'struct_smc_firmware_header_v2_1',
    'struct_smc_soft_pptable_entry', 'struct_ta_firmware_header_v1_0',
    'struct_ta_firmware_header_v2_0', 'struct_table_info',
    'struct_umsch_mm_firmware_header_v1_0', 'struct_v11_compute_mqd',
    'struct_v11_gfx_mqd', 'struct_v11_sdma_mqd',
    'struct_v12_compute_mqd', 'struct_v12_gfx_mqd',
    'struct_v12_sdma_mqd', 'struct_vcn_info_header',
    'struct_vcn_info_v1_0', 'struct_vcn_instance_info_v1_0',
    'struct_vpe_firmware_header_v1_0', 'ta_fw_type', 'ta_type_id',
    'table', 'table__enumvalues', 'table_info', 'tee_error_code',
    'u32', 'uint16_t', 'uint32_t', 'uint64_t', 'uint8_t',
    'union__fuse_data', 'union_amdgpu_firmware_header', 'union_die_0',
    'union_ip_discovery_header_0', 'union_psp_gfx_cmd_setup_tmr_0',
    'union_psp_gfx_commands', 'union_psp_gfx_uresp']
