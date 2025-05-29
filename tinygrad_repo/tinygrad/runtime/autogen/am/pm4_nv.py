# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
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





F32_MES_PM4_PACKETS_H = True # macro
uint32_t = True # macro
int32_t = True # macro
PM4_MES_HEADER_DEFINED = True # macro
PM4_MEC_RELEASE_MEM_DEFINED = True # macro
PM4_MEC_WRITE_DATA_DEFINED = True # macro
class union_PM4_MES_TYPE_3_HEADER(Union):
    pass

class struct_PM4_MES_TYPE_3_HEADER_0(Structure):
    pass

struct_PM4_MES_TYPE_3_HEADER_0._pack_ = 1 # source:False
struct_PM4_MES_TYPE_3_HEADER_0._fields_ = [
    ('reserved1', ctypes.c_uint32, 8),
    ('opcode', ctypes.c_uint32, 8),
    ('count', ctypes.c_uint32, 14),
    ('type', ctypes.c_uint32, 2),
]

union_PM4_MES_TYPE_3_HEADER._pack_ = 1 # source:False
union_PM4_MES_TYPE_3_HEADER._anonymous_ = ('_0',)
union_PM4_MES_TYPE_3_HEADER._fields_ = [
    ('_0', struct_PM4_MES_TYPE_3_HEADER_0),
    ('u32All', ctypes.c_uint32),
]


# values for enumeration 'c_uint32'
c_uint32__enumvalues = {
    5: 'event_index__mec_release_mem__end_of_pipe',
    6: 'event_index__mec_release_mem__shader_done',
}
event_index__mec_release_mem__end_of_pipe = 5
event_index__mec_release_mem__shader_done = 6
c_uint32 = ctypes.c_uint32 # enum

# values for enumeration 'c_uint32'
c_uint32__enumvalues = {
    0: 'cache_policy__mec_release_mem__lru',
    1: 'cache_policy__mec_release_mem__stream',
}
cache_policy__mec_release_mem__lru = 0
cache_policy__mec_release_mem__stream = 1
c_uint32 = ctypes.c_uint32 # enum

# values for enumeration 'c_uint32'
c_uint32__enumvalues = {
    0: 'pq_exe_status__mec_release_mem__default',
    1: 'pq_exe_status__mec_release_mem__phase_update',
}
pq_exe_status__mec_release_mem__default = 0
pq_exe_status__mec_release_mem__phase_update = 1
c_uint32 = ctypes.c_uint32 # enum

# values for enumeration 'c_uint32'
c_uint32__enumvalues = {
    0: 'dst_sel__mec_release_mem__memory_controller',
    1: 'dst_sel__mec_release_mem__tc_l2',
    2: 'dst_sel__mec_release_mem__queue_write_pointer_register',
    3: 'dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit',
}
dst_sel__mec_release_mem__memory_controller = 0
dst_sel__mec_release_mem__tc_l2 = 1
dst_sel__mec_release_mem__queue_write_pointer_register = 2
dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit = 3
c_uint32 = ctypes.c_uint32 # enum

# values for enumeration 'c_uint32'
c_uint32__enumvalues = {
    0: 'int_sel__mec_release_mem__none',
    1: 'int_sel__mec_release_mem__send_interrupt_only',
    2: 'int_sel__mec_release_mem__send_interrupt_after_write_confirm',
    3: 'int_sel__mec_release_mem__send_data_after_write_confirm',
    4: 'int_sel__mec_release_mem__unconditionally_send_int_ctxid',
    5: 'int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare',
    6: 'int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare',
}
int_sel__mec_release_mem__none = 0
int_sel__mec_release_mem__send_interrupt_only = 1
int_sel__mec_release_mem__send_interrupt_after_write_confirm = 2
int_sel__mec_release_mem__send_data_after_write_confirm = 3
int_sel__mec_release_mem__unconditionally_send_int_ctxid = 4
int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare = 5
int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare = 6
c_uint32 = ctypes.c_uint32 # enum

# values for enumeration 'c_uint32'
c_uint32__enumvalues = {
    0: 'data_sel__mec_release_mem__none',
    1: 'data_sel__mec_release_mem__send_32_bit_low',
    2: 'data_sel__mec_release_mem__send_64_bit_data',
    3: 'data_sel__mec_release_mem__send_gpu_clock_counter',
    4: 'data_sel__mec_release_mem__send_cp_perfcounter_hi_lo',
    5: 'data_sel__mec_release_mem__store_gds_data_to_memory',
}
data_sel__mec_release_mem__none = 0
data_sel__mec_release_mem__send_32_bit_low = 1
data_sel__mec_release_mem__send_64_bit_data = 2
data_sel__mec_release_mem__send_gpu_clock_counter = 3
data_sel__mec_release_mem__send_cp_perfcounter_hi_lo = 4
data_sel__mec_release_mem__store_gds_data_to_memory = 5
c_uint32 = ctypes.c_uint32 # enum
class struct_pm4_mec_release_mem(Structure):
    pass

class union_pm4_mec_release_mem_0(Union):
    pass

union_pm4_mec_release_mem_0._pack_ = 1 # source:False
union_pm4_mec_release_mem_0._fields_ = [
    ('header', union_PM4_MES_TYPE_3_HEADER),
    ('ordinal1', ctypes.c_uint32),
]

class union_pm4_mec_release_mem_1(Union):
    pass

class struct_pm4_mec_release_mem_1_bitfields2(Structure):
    pass

struct_pm4_mec_release_mem_1_bitfields2._pack_ = 1 # source:False
struct_pm4_mec_release_mem_1_bitfields2._fields_ = [
    ('event_type', ctypes.c_uint32, 6),
    ('reserved1', ctypes.c_uint32, 2),
    ('event_index', c_uint32, 4),
    ('tcl1_vol_action_ena', ctypes.c_uint32, 1),
    ('tc_vol_action_ena', ctypes.c_uint32, 1),
    ('reserved2', ctypes.c_uint32, 1),
    ('tc_wb_action_ena', ctypes.c_uint32, 1),
    ('tcl1_action_ena', ctypes.c_uint32, 1),
    ('tc_action_ena', ctypes.c_uint32, 1),
    ('reserved3', ctypes.c_uint32, 1),
    ('tc_nc_action_ena', ctypes.c_uint32, 1),
    ('tc_wc_action_ena', ctypes.c_uint32, 1),
    ('tc_md_action_ena', ctypes.c_uint32, 1),
    ('reserved4', ctypes.c_uint32, 3),
    ('cache_policy', c_uint32, 2),
    ('reserved5', ctypes.c_uint32, 2),
    ('pq_exe_status', c_uint32, 1),
    ('reserved6', ctypes.c_uint32, 2),
]

union_pm4_mec_release_mem_1._pack_ = 1 # source:False
union_pm4_mec_release_mem_1._fields_ = [
    ('bitfields2', struct_pm4_mec_release_mem_1_bitfields2),
    ('ordinal2', ctypes.c_uint32),
]

class union_pm4_mec_release_mem_2(Union):
    pass

class struct_pm4_mec_release_mem_2_bitfields3(Structure):
    pass

struct_pm4_mec_release_mem_2_bitfields3._pack_ = 1 # source:False
struct_pm4_mec_release_mem_2_bitfields3._fields_ = [
    ('reserved7', ctypes.c_uint32, 16),
    ('dst_sel', c_uint32, 2),
    ('reserved8', ctypes.c_uint32, 6),
    ('int_sel', c_uint32, 3),
    ('reserved9', ctypes.c_uint32, 2),
    ('data_sel', c_uint32, 3),
]

union_pm4_mec_release_mem_2._pack_ = 1 # source:False
union_pm4_mec_release_mem_2._fields_ = [
    ('bitfields3', struct_pm4_mec_release_mem_2_bitfields3),
    ('ordinal3', ctypes.c_uint32),
]

class union_pm4_mec_release_mem_3(Union):
    pass

class struct_pm4_mec_release_mem_3_bitfields4(Structure):
    pass

struct_pm4_mec_release_mem_3_bitfields4._pack_ = 1 # source:False
struct_pm4_mec_release_mem_3_bitfields4._fields_ = [
    ('reserved10', ctypes.c_uint32, 2),
    ('address_lo_32b', ctypes.c_uint32, 30),
]

class struct_pm4_mec_release_mem_3_bitfields4b(Structure):
    pass

struct_pm4_mec_release_mem_3_bitfields4b._pack_ = 1 # source:False
struct_pm4_mec_release_mem_3_bitfields4b._fields_ = [
    ('reserved11', ctypes.c_uint32, 3),
    ('address_lo_64b', ctypes.c_uint32, 29),
]

union_pm4_mec_release_mem_3._pack_ = 1 # source:False
union_pm4_mec_release_mem_3._fields_ = [
    ('bitfields4', struct_pm4_mec_release_mem_3_bitfields4),
    ('bitfields4b', struct_pm4_mec_release_mem_3_bitfields4b),
    ('reserved12', ctypes.c_uint32),
    ('ordinal4', ctypes.c_uint32),
]

class union_pm4_mec_release_mem_4(Union):
    pass

union_pm4_mec_release_mem_4._pack_ = 1 # source:False
union_pm4_mec_release_mem_4._fields_ = [
    ('address_hi', ctypes.c_uint32),
    ('reserved13', ctypes.c_uint32),
    ('ordinal5', ctypes.c_uint32),
]

class union_pm4_mec_release_mem_5(Union):
    pass

class struct_pm4_mec_release_mem_5_bitfields6c(Structure):
    pass

struct_pm4_mec_release_mem_5_bitfields6c._pack_ = 1 # source:False
struct_pm4_mec_release_mem_5_bitfields6c._fields_ = [
    ('dw_offset', ctypes.c_uint32, 16),
    ('num_dwords', ctypes.c_uint32, 16),
]

union_pm4_mec_release_mem_5._pack_ = 1 # source:False
union_pm4_mec_release_mem_5._fields_ = [
    ('data_lo', ctypes.c_uint32),
    ('cmp_data_lo', ctypes.c_uint32),
    ('bitfields6c', struct_pm4_mec_release_mem_5_bitfields6c),
    ('reserved14', ctypes.c_uint32),
    ('ordinal6', ctypes.c_uint32),
]

class union_pm4_mec_release_mem_6(Union):
    pass

union_pm4_mec_release_mem_6._pack_ = 1 # source:False
union_pm4_mec_release_mem_6._fields_ = [
    ('data_hi', ctypes.c_uint32),
    ('cmp_data_hi', ctypes.c_uint32),
    ('reserved15', ctypes.c_uint32),
    ('reserved16', ctypes.c_uint32),
    ('ordinal7', ctypes.c_uint32),
]

struct_pm4_mec_release_mem._pack_ = 1 # source:False
struct_pm4_mec_release_mem._anonymous_ = ('_0', '_1', '_2', '_3', '_4', '_5', '_6',)
struct_pm4_mec_release_mem._fields_ = [
    ('_0', union_pm4_mec_release_mem_0),
    ('_1', union_pm4_mec_release_mem_1),
    ('_2', union_pm4_mec_release_mem_2),
    ('_3', union_pm4_mec_release_mem_3),
    ('_4', union_pm4_mec_release_mem_4),
    ('_5', union_pm4_mec_release_mem_5),
    ('_6', union_pm4_mec_release_mem_6),
    ('int_ctxid', ctypes.c_uint32),
]


# values for enumeration 'WRITE_DATA_dst_sel_enum'
WRITE_DATA_dst_sel_enum__enumvalues = {
    0: 'dst_sel___write_data__mem_mapped_register',
    2: 'dst_sel___write_data__tc_l2',
    3: 'dst_sel___write_data__gds',
    5: 'dst_sel___write_data__memory',
    6: 'dst_sel___write_data__memory_mapped_adc_persistent_state',
}
dst_sel___write_data__mem_mapped_register = 0
dst_sel___write_data__tc_l2 = 2
dst_sel___write_data__gds = 3
dst_sel___write_data__memory = 5
dst_sel___write_data__memory_mapped_adc_persistent_state = 6
WRITE_DATA_dst_sel_enum = ctypes.c_uint32 # enum

# values for enumeration 'WRITE_DATA_addr_incr_enum'
WRITE_DATA_addr_incr_enum__enumvalues = {
    0: 'addr_incr___write_data__increment_address',
    1: 'addr_incr___write_data__do_not_increment_address',
}
addr_incr___write_data__increment_address = 0
addr_incr___write_data__do_not_increment_address = 1
WRITE_DATA_addr_incr_enum = ctypes.c_uint32 # enum

# values for enumeration 'WRITE_DATA_wr_confirm_enum'
WRITE_DATA_wr_confirm_enum__enumvalues = {
    0: 'wr_confirm___write_data__do_not_wait_for_write_confirmation',
    1: 'wr_confirm___write_data__wait_for_write_confirmation',
}
wr_confirm___write_data__do_not_wait_for_write_confirmation = 0
wr_confirm___write_data__wait_for_write_confirmation = 1
WRITE_DATA_wr_confirm_enum = ctypes.c_uint32 # enum

# values for enumeration 'WRITE_DATA_cache_policy_enum'
WRITE_DATA_cache_policy_enum__enumvalues = {
    0: 'cache_policy___write_data__lru',
    1: 'cache_policy___write_data__stream',
}
cache_policy___write_data__lru = 0
cache_policy___write_data__stream = 1
WRITE_DATA_cache_policy_enum = ctypes.c_uint32 # enum
class struct_pm4_mec_write_data_mmio(Structure):
    pass

class union_pm4_mec_write_data_mmio_0(Union):
    pass

union_pm4_mec_write_data_mmio_0._pack_ = 1 # source:False
union_pm4_mec_write_data_mmio_0._fields_ = [
    ('header', union_PM4_MES_TYPE_3_HEADER),
    ('ordinal1', ctypes.c_uint32),
]

class union_pm4_mec_write_data_mmio_1(Union):
    pass

class struct_pm4_mec_write_data_mmio_1_bitfields2(Structure):
    pass

struct_pm4_mec_write_data_mmio_1_bitfields2._pack_ = 1 # source:False
struct_pm4_mec_write_data_mmio_1_bitfields2._fields_ = [
    ('reserved1', ctypes.c_uint32, 8),
    ('dst_sel', ctypes.c_uint32, 4),
    ('reserved2', ctypes.c_uint32, 4),
    ('addr_incr', ctypes.c_uint32, 1),
    ('reserved3', ctypes.c_uint32, 2),
    ('resume_vf', ctypes.c_uint32, 1),
    ('wr_confirm', ctypes.c_uint32, 1),
    ('reserved4', ctypes.c_uint32, 4),
    ('cache_policy', ctypes.c_uint32, 2),
    ('reserved5', ctypes.c_uint32, 5),
]

union_pm4_mec_write_data_mmio_1._pack_ = 1 # source:False
union_pm4_mec_write_data_mmio_1._fields_ = [
    ('bitfields2', struct_pm4_mec_write_data_mmio_1_bitfields2),
    ('ordinal2', ctypes.c_uint32),
]

class union_pm4_mec_write_data_mmio_2(Union):
    pass

class struct_pm4_mec_write_data_mmio_2_bitfields3(Structure):
    pass

struct_pm4_mec_write_data_mmio_2_bitfields3._pack_ = 1 # source:False
struct_pm4_mec_write_data_mmio_2_bitfields3._fields_ = [
    ('dst_mmreg_addr', ctypes.c_uint32, 18),
    ('reserved6', ctypes.c_uint32, 14),
]

union_pm4_mec_write_data_mmio_2._pack_ = 1 # source:False
union_pm4_mec_write_data_mmio_2._fields_ = [
    ('bitfields3', struct_pm4_mec_write_data_mmio_2_bitfields3),
    ('ordinal3', ctypes.c_uint32),
]

struct_pm4_mec_write_data_mmio._pack_ = 1 # source:False
struct_pm4_mec_write_data_mmio._anonymous_ = ('_0', '_1', '_2',)
struct_pm4_mec_write_data_mmio._fields_ = [
    ('_0', union_pm4_mec_write_data_mmio_0),
    ('_1', union_pm4_mec_write_data_mmio_1),
    ('_2', union_pm4_mec_write_data_mmio_2),
    ('reserved7', ctypes.c_uint32),
    ('data', ctypes.c_uint32),
]


# values for enumeration 'c__Ea_CACHE_FLUSH_AND_INV_TS_EVENT'
c__Ea_CACHE_FLUSH_AND_INV_TS_EVENT__enumvalues = {
    20: 'CACHE_FLUSH_AND_INV_TS_EVENT',
}
CACHE_FLUSH_AND_INV_TS_EVENT = 20
c__Ea_CACHE_FLUSH_AND_INV_TS_EVENT = ctypes.c_uint32 # enum
NVD_H = True # macro
PACKET_TYPE0 = 0 # macro
PACKET_TYPE1 = 1 # macro
PACKET_TYPE2 = 2 # macro
PACKET_TYPE3 = 3 # macro
def CP_PACKET_GET_TYPE(h):  # macro
   return (((h)>>30)&3)
def CP_PACKET_GET_COUNT(h):  # macro
   return (((h)>>16)&0x3FFF)
def CP_PACKET0_GET_REG(h):  # macro
   return ((h)&0xFFFF)
def CP_PACKET3_GET_OPCODE(h):  # macro
   return (((h)>>8)&0xFF)
def PACKET0(reg, n):  # macro
   return ((0<<30)|((reg)&0xFFFF)|((n)&0x3FFF)<<16)
CP_PACKET2 = 0x80000000 # macro
PACKET2_PAD_SHIFT = 0 # macro
PACKET2_PAD_MASK = (0x3fffffff<<0) # macro
# def PACKET2(v):  # macro
#    return (0x80000000|REG_SET(PACKET2_PAD,(v)))
def PACKET3(op, n):  # macro
   return ((3<<30)|(((op)&0xFF)<<8)|((n)&0x3FFF)<<16)
def PACKET3_COMPUTE(op, n):  # macro
   return (PACKET3(op,n)|1<<1)
PACKET3_NOP = 0x10 # macro
PACKET3_SET_BASE = 0x11 # macro
def PACKET3_BASE_INDEX(x):  # macro
   return ((x)<<0)
CE_PARTITION_BASE = 3 # macro
PACKET3_CLEAR_STATE = 0x12 # macro
PACKET3_INDEX_BUFFER_SIZE = 0x13 # macro
PACKET3_DISPATCH_DIRECT = 0x15 # macro
PACKET3_DISPATCH_INDIRECT = 0x16 # macro
PACKET3_INDIRECT_BUFFER_END = 0x17 # macro
PACKET3_INDIRECT_BUFFER_CNST_END = 0x19 # macro
PACKET3_ATOMIC_GDS = 0x1D # macro
PACKET3_ATOMIC_MEM = 0x1E # macro
PACKET3_OCCLUSION_QUERY = 0x1F # macro
PACKET3_SET_PREDICATION = 0x20 # macro
PACKET3_REG_RMW = 0x21 # macro
PACKET3_COND_EXEC = 0x22 # macro
PACKET3_PRED_EXEC = 0x23 # macro
PACKET3_DRAW_INDIRECT = 0x24 # macro
PACKET3_DRAW_INDEX_INDIRECT = 0x25 # macro
PACKET3_INDEX_BASE = 0x26 # macro
PACKET3_DRAW_INDEX_2 = 0x27 # macro
PACKET3_CONTEXT_CONTROL = 0x28 # macro
PACKET3_INDEX_TYPE = 0x2A # macro
PACKET3_DRAW_INDIRECT_MULTI = 0x2C # macro
PACKET3_DRAW_INDEX_AUTO = 0x2D # macro
PACKET3_NUM_INSTANCES = 0x2F # macro
PACKET3_DRAW_INDEX_MULTI_AUTO = 0x30 # macro
PACKET3_INDIRECT_BUFFER_PRIV = 0x32 # macro
PACKET3_INDIRECT_BUFFER_CNST = 0x33 # macro
PACKET3_COND_INDIRECT_BUFFER_CNST = 0x33 # macro
PACKET3_STRMOUT_BUFFER_UPDATE = 0x34 # macro
PACKET3_DRAW_INDEX_OFFSET_2 = 0x35 # macro
PACKET3_DRAW_PREAMBLE = 0x36 # macro
PACKET3_WRITE_DATA = 0x37 # macro
def WRITE_DATA_DST_SEL(x):  # macro
   return ((x)<<8)
WR_ONE_ADDR = (1<<16) # macro
WR_CONFIRM = (1<<20) # macro
def WRITE_DATA_CACHE_POLICY(x):  # macro
   return ((x)<<25)
def WRITE_DATA_ENGINE_SEL(x):  # macro
   return ((x)<<30)
PACKET3_DRAW_INDEX_INDIRECT_MULTI = 0x38 # macro
PACKET3_MEM_SEMAPHORE = 0x39 # macro
PACKET3_SEM_USE_MAILBOX = (0x1<<16) # macro
PACKET3_SEM_SEL_SIGNAL_TYPE = (0x1<<20) # macro
PACKET3_SEM_SEL_SIGNAL = (0x6<<29) # macro
PACKET3_SEM_SEL_WAIT = (0x7<<29) # macro
PACKET3_DRAW_INDEX_MULTI_INST = 0x3A # macro
PACKET3_COPY_DW = 0x3B # macro
PACKET3_WAIT_REG_MEM = 0x3C # macro
def WAIT_REG_MEM_FUNCTION(x):  # macro
   return ((x)<<0)
def WAIT_REG_MEM_MEM_SPACE(x):  # macro
   return ((x)<<4)
def WAIT_REG_MEM_OPERATION(x):  # macro
   return ((x)<<6)
def WAIT_REG_MEM_ENGINE(x):  # macro
   return ((x)<<8)
PACKET3_INDIRECT_BUFFER = 0x3F # macro
INDIRECT_BUFFER_VALID = (1<<23) # macro
def INDIRECT_BUFFER_CACHE_POLICY(x):  # macro
   return ((x)<<28)
def INDIRECT_BUFFER_PRE_ENB(x):  # macro
   return ((x)<<21)
def INDIRECT_BUFFER_PRE_RESUME(x):  # macro
   return ((x)<<30)
PACKET3_COND_INDIRECT_BUFFER = 0x3F # macro
PACKET3_COPY_DATA = 0x40 # macro
PACKET3_CP_DMA = 0x41 # macro
PACKET3_PFP_SYNC_ME = 0x42 # macro
PACKET3_SURFACE_SYNC = 0x43 # macro
PACKET3_ME_INITIALIZE = 0x44 # macro
PACKET3_COND_WRITE = 0x45 # macro
PACKET3_EVENT_WRITE = 0x46 # macro
def EVENT_TYPE(x):  # macro
   return ((x)<<0)
def EVENT_INDEX(x):  # macro
   return ((x)<<8)
PACKET3_EVENT_WRITE_EOP = 0x47 # macro
PACKET3_EVENT_WRITE_EOS = 0x48 # macro
PACKET3_RELEASE_MEM = 0x49 # macro
def PACKET3_RELEASE_MEM_EVENT_TYPE(x):  # macro
   return ((x)<<0)
def PACKET3_RELEASE_MEM_EVENT_INDEX(x):  # macro
   return ((x)<<8)
PACKET3_RELEASE_MEM_GCR_GLM_WB = (1<<12) # macro
PACKET3_RELEASE_MEM_GCR_GLM_INV = (1<<13) # macro
PACKET3_RELEASE_MEM_GCR_GLV_INV = (1<<14) # macro
PACKET3_RELEASE_MEM_GCR_GL1_INV = (1<<15) # macro
PACKET3_RELEASE_MEM_GCR_GL2_US = (1<<16) # macro
PACKET3_RELEASE_MEM_GCR_GL2_RANGE = (1<<17) # macro
PACKET3_RELEASE_MEM_GCR_GL2_DISCARD = (1<<19) # macro
PACKET3_RELEASE_MEM_GCR_GL2_INV = (1<<20) # macro
PACKET3_RELEASE_MEM_GCR_GL2_WB = (1<<21) # macro
PACKET3_RELEASE_MEM_GCR_SEQ = (1<<22) # macro
def PACKET3_RELEASE_MEM_CACHE_POLICY(x):  # macro
   return ((x)<<25)
PACKET3_RELEASE_MEM_EXECUTE = (1<<28) # macro
def PACKET3_RELEASE_MEM_DATA_SEL(x):  # macro
   return ((x)<<29)
def PACKET3_RELEASE_MEM_INT_SEL(x):  # macro
   return ((x)<<24)
def PACKET3_RELEASE_MEM_DST_SEL(x):  # macro
   return ((x)<<16)
PACKET3_PREAMBLE_CNTL = 0x4A # macro
PACKET3_PREAMBLE_BEGIN_CLEAR_STATE = (2<<28) # macro
PACKET3_PREAMBLE_END_CLEAR_STATE = (3<<28) # macro
PACKET3_DMA_DATA = 0x50 # macro
def PACKET3_DMA_DATA_ENGINE(x):  # macro
   return ((x)<<0)
def PACKET3_DMA_DATA_SRC_CACHE_POLICY(x):  # macro
   return ((x)<<13)
def PACKET3_DMA_DATA_DST_SEL(x):  # macro
   return ((x)<<20)
def PACKET3_DMA_DATA_DST_CACHE_POLICY(x):  # macro
   return ((x)<<25)
def PACKET3_DMA_DATA_SRC_SEL(x):  # macro
   return ((x)<<29)
PACKET3_DMA_DATA_CP_SYNC = (1<<31) # macro
PACKET3_DMA_DATA_CMD_SAS = (1<<26) # macro
PACKET3_DMA_DATA_CMD_DAS = (1<<27) # macro
PACKET3_DMA_DATA_CMD_SAIC = (1<<28) # macro
PACKET3_DMA_DATA_CMD_DAIC = (1<<29) # macro
PACKET3_DMA_DATA_CMD_RAW_WAIT = (1<<30) # macro
PACKET3_CONTEXT_REG_RMW = 0x51 # macro
PACKET3_GFX_CNTX_UPDATE = 0x52 # macro
PACKET3_BLK_CNTX_UPDATE = 0x53 # macro
PACKET3_INCR_UPDT_STATE = 0x55 # macro
PACKET3_ACQUIRE_MEM = 0x58 # macro
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(x):  # macro
   return ((x)<<0)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_RANGE(x):  # macro
   return ((x)<<2)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(x):  # macro
   return ((x)<<4)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(x):  # macro
   return ((x)<<5)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(x):  # macro
   return ((x)<<6)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(x):  # macro
   return ((x)<<7)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(x):  # macro
   return ((x)<<8)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(x):  # macro
   return ((x)<<9)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_US(x):  # macro
   return ((x)<<10)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_RANGE(x):  # macro
   return ((x)<<11)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_DISCARD(x):  # macro
   return ((x)<<13)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(x):  # macro
   return ((x)<<14)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(x):  # macro
   return ((x)<<15)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_SEQ(x):  # macro
   return ((x)<<16)
PACKET3_ACQUIRE_MEM_GCR_RANGE_IS_PA = (1<<18) # macro
PACKET3_REWIND = 0x59 # macro
PACKET3_INTERRUPT = 0x5A # macro
PACKET3_GEN_PDEPTE = 0x5B # macro
PACKET3_INDIRECT_BUFFER_PASID = 0x5C # macro
PACKET3_PRIME_UTCL2 = 0x5D # macro
PACKET3_LOAD_UCONFIG_REG = 0x5E # macro
PACKET3_LOAD_SH_REG = 0x5F # macro
PACKET3_LOAD_CONFIG_REG = 0x60 # macro
PACKET3_LOAD_CONTEXT_REG = 0x61 # macro
PACKET3_LOAD_COMPUTE_STATE = 0x62 # macro
PACKET3_LOAD_SH_REG_INDEX = 0x63 # macro
PACKET3_SET_CONFIG_REG = 0x68 # macro
PACKET3_SET_CONFIG_REG_START = 0x00002000 # macro
PACKET3_SET_CONFIG_REG_END = 0x00002c00 # macro
PACKET3_SET_CONTEXT_REG = 0x69 # macro
PACKET3_SET_CONTEXT_REG_START = 0x0000a000 # macro
PACKET3_SET_CONTEXT_REG_END = 0x0000a400 # macro
PACKET3_SET_CONTEXT_REG_INDEX = 0x6A # macro
PACKET3_SET_VGPR_REG_DI_MULTI = 0x71 # macro
PACKET3_SET_SH_REG_DI = 0x72 # macro
PACKET3_SET_CONTEXT_REG_INDIRECT = 0x73 # macro
PACKET3_SET_SH_REG_DI_MULTI = 0x74 # macro
PACKET3_GFX_PIPE_LOCK = 0x75 # macro
PACKET3_SET_SH_REG = 0x76 # macro
PACKET3_SET_SH_REG_START = 0x00002c00 # macro
PACKET3_SET_SH_REG_END = 0x00003000 # macro
PACKET3_SET_SH_REG_OFFSET = 0x77 # macro
PACKET3_SET_QUEUE_REG = 0x78 # macro
PACKET3_SET_UCONFIG_REG = 0x79 # macro
PACKET3_SET_UCONFIG_REG_START = 0x0000c000 # macro
PACKET3_SET_UCONFIG_REG_END = 0x0000c400 # macro
PACKET3_SET_UCONFIG_REG_INDEX = 0x7A # macro
PACKET3_FORWARD_HEADER = 0x7C # macro
PACKET3_SCRATCH_RAM_WRITE = 0x7D # macro
PACKET3_SCRATCH_RAM_READ = 0x7E # macro
PACKET3_LOAD_CONST_RAM = 0x80 # macro
PACKET3_WRITE_CONST_RAM = 0x81 # macro
PACKET3_DUMP_CONST_RAM = 0x83 # macro
PACKET3_INCREMENT_CE_COUNTER = 0x84 # macro
PACKET3_INCREMENT_DE_COUNTER = 0x85 # macro
PACKET3_WAIT_ON_CE_COUNTER = 0x86 # macro
PACKET3_WAIT_ON_DE_COUNTER_DIFF = 0x88 # macro
PACKET3_SWITCH_BUFFER = 0x8B # macro
PACKET3_DISPATCH_DRAW_PREAMBLE = 0x8C # macro
PACKET3_DISPATCH_DRAW_PREAMBLE_ACE = 0x8C # macro
PACKET3_DISPATCH_DRAW = 0x8D # macro
PACKET3_DISPATCH_DRAW_ACE = 0x8D # macro
PACKET3_GET_LOD_STATS = 0x8E # macro
PACKET3_DRAW_MULTI_PREAMBLE = 0x8F # macro
PACKET3_FRAME_CONTROL = 0x90 # macro
FRAME_TMZ = (1<<0) # macro
def FRAME_CMD(x):  # macro
   return ((x)<<28)
PACKET3_INDEX_ATTRIBUTES_INDIRECT = 0x91 # macro
PACKET3_WAIT_REG_MEM64 = 0x93 # macro
PACKET3_COND_PREEMPT = 0x94 # macro
PACKET3_HDP_FLUSH = 0x95 # macro
PACKET3_COPY_DATA_RB = 0x96 # macro
PACKET3_INVALIDATE_TLBS = 0x98 # macro
def PACKET3_INVALIDATE_TLBS_DST_SEL(x):  # macro
   return ((x)<<0)
def PACKET3_INVALIDATE_TLBS_ALL_HUB(x):  # macro
   return ((x)<<4)
def PACKET3_INVALIDATE_TLBS_PASID(x):  # macro
   return ((x)<<5)
PACKET3_AQL_PACKET = 0x99 # macro
PACKET3_DMA_DATA_FILL_MULTI = 0x9A # macro
PACKET3_SET_SH_REG_INDEX = 0x9B # macro
PACKET3_DRAW_INDIRECT_COUNT_MULTI = 0x9C # macro
PACKET3_DRAW_INDEX_INDIRECT_COUNT_MULTI = 0x9D # macro
PACKET3_DUMP_CONST_RAM_OFFSET = 0x9E # macro
PACKET3_LOAD_CONTEXT_REG_INDEX = 0x9F # macro
PACKET3_SET_RESOURCES = 0xA0 # macro
def PACKET3_SET_RESOURCES_VMID_MASK(x):  # macro
   return ((x)<<0)
def PACKET3_SET_RESOURCES_UNMAP_LATENTY(x):  # macro
   return ((x)<<16)
def PACKET3_SET_RESOURCES_QUEUE_TYPE(x):  # macro
   return ((x)<<29)
PACKET3_MAP_PROCESS = 0xA1 # macro
PACKET3_MAP_QUEUES = 0xA2 # macro
def PACKET3_MAP_QUEUES_QUEUE_SEL(x):  # macro
   return ((x)<<4)
def PACKET3_MAP_QUEUES_VMID(x):  # macro
   return ((x)<<8)
def PACKET3_MAP_QUEUES_QUEUE(x):  # macro
   return ((x)<<13)
def PACKET3_MAP_QUEUES_PIPE(x):  # macro
   return ((x)<<16)
def PACKET3_MAP_QUEUES_ME(x):  # macro
   return ((x)<<18)
def PACKET3_MAP_QUEUES_QUEUE_TYPE(x):  # macro
   return ((x)<<21)
def PACKET3_MAP_QUEUES_ALLOC_FORMAT(x):  # macro
   return ((x)<<24)
def PACKET3_MAP_QUEUES_ENGINE_SEL(x):  # macro
   return ((x)<<26)
def PACKET3_MAP_QUEUES_NUM_QUEUES(x):  # macro
   return ((x)<<29)
def PACKET3_MAP_QUEUES_CHECK_DISABLE(x):  # macro
   return ((x)<<1)
def PACKET3_MAP_QUEUES_DOORBELL_OFFSET(x):  # macro
   return ((x)<<2)
PACKET3_UNMAP_QUEUES = 0xA3 # macro
def PACKET3_UNMAP_QUEUES_ACTION(x):  # macro
   return ((x)<<0)
def PACKET3_UNMAP_QUEUES_QUEUE_SEL(x):  # macro
   return ((x)<<4)
def PACKET3_UNMAP_QUEUES_ENGINE_SEL(x):  # macro
   return ((x)<<26)
def PACKET3_UNMAP_QUEUES_NUM_QUEUES(x):  # macro
   return ((x)<<29)
def PACKET3_UNMAP_QUEUES_PASID(x):  # macro
   return ((x)<<0)
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET0(x):  # macro
   return ((x)<<2)
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET1(x):  # macro
   return ((x)<<2)
def PACKET3_UNMAP_QUEUES_RB_WPTR(x):  # macro
   return ((x)<<0)
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET2(x):  # macro
   return ((x)<<2)
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET3(x):  # macro
   return ((x)<<2)
PACKET3_QUERY_STATUS = 0xA4 # macro
def PACKET3_QUERY_STATUS_CONTEXT_ID(x):  # macro
   return ((x)<<0)
def PACKET3_QUERY_STATUS_INTERRUPT_SEL(x):  # macro
   return ((x)<<28)
def PACKET3_QUERY_STATUS_COMMAND(x):  # macro
   return ((x)<<30)
def PACKET3_QUERY_STATUS_PASID(x):  # macro
   return ((x)<<0)
def PACKET3_QUERY_STATUS_DOORBELL_OFFSET(x):  # macro
   return ((x)<<2)
def PACKET3_QUERY_STATUS_ENG_SEL(x):  # macro
   return ((x)<<25)
PACKET3_RUN_LIST = 0xA5 # macro
PACKET3_MAP_PROCESS_VM = 0xA6 # macro
PACKET3_SET_Q_PREEMPTION_MODE = 0xF0 # macro
def PACKET3_SET_Q_PREEMPTION_MODE_IB_VMID(x):  # macro
   return ((x)<<0)
PACKET3_SET_Q_PREEMPTION_MODE_INIT_SHADOW_MEM = (1<<0) # macro
__all__ = \
    ['CACHE_FLUSH_AND_INV_TS_EVENT', 'CE_PARTITION_BASE',
    'CP_PACKET2', 'F32_MES_PM4_PACKETS_H', 'FRAME_TMZ',
    'INDIRECT_BUFFER_VALID', 'NVD_H', 'PACKET2_PAD_MASK',
    'PACKET2_PAD_SHIFT', 'PACKET3_ACQUIRE_MEM',
    'PACKET3_ACQUIRE_MEM_GCR_RANGE_IS_PA', 'PACKET3_AQL_PACKET',
    'PACKET3_ATOMIC_GDS', 'PACKET3_ATOMIC_MEM',
    'PACKET3_BLK_CNTX_UPDATE', 'PACKET3_CLEAR_STATE',
    'PACKET3_COND_EXEC', 'PACKET3_COND_INDIRECT_BUFFER',
    'PACKET3_COND_INDIRECT_BUFFER_CNST', 'PACKET3_COND_PREEMPT',
    'PACKET3_COND_WRITE', 'PACKET3_CONTEXT_CONTROL',
    'PACKET3_CONTEXT_REG_RMW', 'PACKET3_COPY_DATA',
    'PACKET3_COPY_DATA_RB', 'PACKET3_COPY_DW', 'PACKET3_CP_DMA',
    'PACKET3_DISPATCH_DIRECT', 'PACKET3_DISPATCH_DRAW',
    'PACKET3_DISPATCH_DRAW_ACE', 'PACKET3_DISPATCH_DRAW_PREAMBLE',
    'PACKET3_DISPATCH_DRAW_PREAMBLE_ACE', 'PACKET3_DISPATCH_INDIRECT',
    'PACKET3_DMA_DATA', 'PACKET3_DMA_DATA_CMD_DAIC',
    'PACKET3_DMA_DATA_CMD_DAS', 'PACKET3_DMA_DATA_CMD_RAW_WAIT',
    'PACKET3_DMA_DATA_CMD_SAIC', 'PACKET3_DMA_DATA_CMD_SAS',
    'PACKET3_DMA_DATA_CP_SYNC', 'PACKET3_DMA_DATA_FILL_MULTI',
    'PACKET3_DRAW_INDEX_2', 'PACKET3_DRAW_INDEX_AUTO',
    'PACKET3_DRAW_INDEX_INDIRECT',
    'PACKET3_DRAW_INDEX_INDIRECT_COUNT_MULTI',
    'PACKET3_DRAW_INDEX_INDIRECT_MULTI',
    'PACKET3_DRAW_INDEX_MULTI_AUTO', 'PACKET3_DRAW_INDEX_MULTI_INST',
    'PACKET3_DRAW_INDEX_OFFSET_2', 'PACKET3_DRAW_INDIRECT',
    'PACKET3_DRAW_INDIRECT_COUNT_MULTI',
    'PACKET3_DRAW_INDIRECT_MULTI', 'PACKET3_DRAW_MULTI_PREAMBLE',
    'PACKET3_DRAW_PREAMBLE', 'PACKET3_DUMP_CONST_RAM',
    'PACKET3_DUMP_CONST_RAM_OFFSET', 'PACKET3_EVENT_WRITE',
    'PACKET3_EVENT_WRITE_EOP', 'PACKET3_EVENT_WRITE_EOS',
    'PACKET3_FORWARD_HEADER', 'PACKET3_FRAME_CONTROL',
    'PACKET3_GEN_PDEPTE', 'PACKET3_GET_LOD_STATS',
    'PACKET3_GFX_CNTX_UPDATE', 'PACKET3_GFX_PIPE_LOCK',
    'PACKET3_HDP_FLUSH', 'PACKET3_INCREMENT_CE_COUNTER',
    'PACKET3_INCREMENT_DE_COUNTER', 'PACKET3_INCR_UPDT_STATE',
    'PACKET3_INDEX_ATTRIBUTES_INDIRECT', 'PACKET3_INDEX_BASE',
    'PACKET3_INDEX_BUFFER_SIZE', 'PACKET3_INDEX_TYPE',
    'PACKET3_INDIRECT_BUFFER', 'PACKET3_INDIRECT_BUFFER_CNST',
    'PACKET3_INDIRECT_BUFFER_CNST_END', 'PACKET3_INDIRECT_BUFFER_END',
    'PACKET3_INDIRECT_BUFFER_PASID', 'PACKET3_INDIRECT_BUFFER_PRIV',
    'PACKET3_INTERRUPT', 'PACKET3_INVALIDATE_TLBS',
    'PACKET3_LOAD_COMPUTE_STATE', 'PACKET3_LOAD_CONFIG_REG',
    'PACKET3_LOAD_CONST_RAM', 'PACKET3_LOAD_CONTEXT_REG',
    'PACKET3_LOAD_CONTEXT_REG_INDEX', 'PACKET3_LOAD_SH_REG',
    'PACKET3_LOAD_SH_REG_INDEX', 'PACKET3_LOAD_UCONFIG_REG',
    'PACKET3_MAP_PROCESS', 'PACKET3_MAP_PROCESS_VM',
    'PACKET3_MAP_QUEUES', 'PACKET3_MEM_SEMAPHORE',
    'PACKET3_ME_INITIALIZE', 'PACKET3_NOP', 'PACKET3_NUM_INSTANCES',
    'PACKET3_OCCLUSION_QUERY', 'PACKET3_PFP_SYNC_ME',
    'PACKET3_PREAMBLE_BEGIN_CLEAR_STATE', 'PACKET3_PREAMBLE_CNTL',
    'PACKET3_PREAMBLE_END_CLEAR_STATE', 'PACKET3_PRED_EXEC',
    'PACKET3_PRIME_UTCL2', 'PACKET3_QUERY_STATUS', 'PACKET3_REG_RMW',
    'PACKET3_RELEASE_MEM', 'PACKET3_RELEASE_MEM_EXECUTE',
    'PACKET3_RELEASE_MEM_GCR_GL1_INV',
    'PACKET3_RELEASE_MEM_GCR_GL2_DISCARD',
    'PACKET3_RELEASE_MEM_GCR_GL2_INV',
    'PACKET3_RELEASE_MEM_GCR_GL2_RANGE',
    'PACKET3_RELEASE_MEM_GCR_GL2_US',
    'PACKET3_RELEASE_MEM_GCR_GL2_WB',
    'PACKET3_RELEASE_MEM_GCR_GLM_INV',
    'PACKET3_RELEASE_MEM_GCR_GLM_WB',
    'PACKET3_RELEASE_MEM_GCR_GLV_INV', 'PACKET3_RELEASE_MEM_GCR_SEQ',
    'PACKET3_REWIND', 'PACKET3_RUN_LIST', 'PACKET3_SCRATCH_RAM_READ',
    'PACKET3_SCRATCH_RAM_WRITE', 'PACKET3_SEM_SEL_SIGNAL',
    'PACKET3_SEM_SEL_SIGNAL_TYPE', 'PACKET3_SEM_SEL_WAIT',
    'PACKET3_SEM_USE_MAILBOX', 'PACKET3_SET_BASE',
    'PACKET3_SET_CONFIG_REG', 'PACKET3_SET_CONFIG_REG_END',
    'PACKET3_SET_CONFIG_REG_START', 'PACKET3_SET_CONTEXT_REG',
    'PACKET3_SET_CONTEXT_REG_END', 'PACKET3_SET_CONTEXT_REG_INDEX',
    'PACKET3_SET_CONTEXT_REG_INDIRECT',
    'PACKET3_SET_CONTEXT_REG_START', 'PACKET3_SET_PREDICATION',
    'PACKET3_SET_QUEUE_REG', 'PACKET3_SET_Q_PREEMPTION_MODE',
    'PACKET3_SET_Q_PREEMPTION_MODE_INIT_SHADOW_MEM',
    'PACKET3_SET_RESOURCES', 'PACKET3_SET_SH_REG',
    'PACKET3_SET_SH_REG_DI', 'PACKET3_SET_SH_REG_DI_MULTI',
    'PACKET3_SET_SH_REG_END', 'PACKET3_SET_SH_REG_INDEX',
    'PACKET3_SET_SH_REG_OFFSET', 'PACKET3_SET_SH_REG_START',
    'PACKET3_SET_UCONFIG_REG', 'PACKET3_SET_UCONFIG_REG_END',
    'PACKET3_SET_UCONFIG_REG_INDEX', 'PACKET3_SET_UCONFIG_REG_START',
    'PACKET3_SET_VGPR_REG_DI_MULTI', 'PACKET3_STRMOUT_BUFFER_UPDATE',
    'PACKET3_SURFACE_SYNC', 'PACKET3_SWITCH_BUFFER',
    'PACKET3_UNMAP_QUEUES', 'PACKET3_WAIT_ON_CE_COUNTER',
    'PACKET3_WAIT_ON_DE_COUNTER_DIFF', 'PACKET3_WAIT_REG_MEM',
    'PACKET3_WAIT_REG_MEM64', 'PACKET3_WRITE_CONST_RAM',
    'PACKET3_WRITE_DATA', 'PACKET_TYPE0', 'PACKET_TYPE1',
    'PACKET_TYPE2', 'PACKET_TYPE3', 'PM4_MEC_RELEASE_MEM_DEFINED',
    'PM4_MEC_WRITE_DATA_DEFINED', 'PM4_MES_HEADER_DEFINED',
    'WRITE_DATA_addr_incr_enum', 'WRITE_DATA_cache_policy_enum',
    'WRITE_DATA_dst_sel_enum', 'WRITE_DATA_wr_confirm_enum',
    'WR_CONFIRM', 'WR_ONE_ADDR',
    'addr_incr___write_data__do_not_increment_address',
    'addr_incr___write_data__increment_address',
    'c__Ea_CACHE_FLUSH_AND_INV_TS_EVENT', 'c_uint32', 'c_uint32',
    'c_uint32', 'c_uint32', 'c_uint32', 'c_uint32',
    'cache_policy___write_data__lru',
    'cache_policy___write_data__stream',
    'cache_policy__mec_release_mem__lru',
    'cache_policy__mec_release_mem__stream',
    'data_sel__mec_release_mem__none',
    'data_sel__mec_release_mem__send_32_bit_low',
    'data_sel__mec_release_mem__send_64_bit_data',
    'data_sel__mec_release_mem__send_cp_perfcounter_hi_lo',
    'data_sel__mec_release_mem__send_gpu_clock_counter',
    'data_sel__mec_release_mem__store_gds_data_to_memory',
    'dst_sel___write_data__gds',
    'dst_sel___write_data__mem_mapped_register',
    'dst_sel___write_data__memory',
    'dst_sel___write_data__memory_mapped_adc_persistent_state',
    'dst_sel___write_data__tc_l2',
    'dst_sel__mec_release_mem__memory_controller',
    'dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit',
    'dst_sel__mec_release_mem__queue_write_pointer_register',
    'dst_sel__mec_release_mem__tc_l2',
    'event_index__mec_release_mem__end_of_pipe',
    'event_index__mec_release_mem__shader_done', 'int32_t',
    'int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare',
    'int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare',
    'int_sel__mec_release_mem__none',
    'int_sel__mec_release_mem__send_data_after_write_confirm',
    'int_sel__mec_release_mem__send_interrupt_after_write_confirm',
    'int_sel__mec_release_mem__send_interrupt_only',
    'int_sel__mec_release_mem__unconditionally_send_int_ctxid',
    'pq_exe_status__mec_release_mem__default',
    'pq_exe_status__mec_release_mem__phase_update',
    'struct_PM4_MES_TYPE_3_HEADER_0', 'struct_pm4_mec_release_mem',
    'struct_pm4_mec_release_mem_1_bitfields2',
    'struct_pm4_mec_release_mem_2_bitfields3',
    'struct_pm4_mec_release_mem_3_bitfields4',
    'struct_pm4_mec_release_mem_3_bitfields4b',
    'struct_pm4_mec_release_mem_5_bitfields6c',
    'struct_pm4_mec_write_data_mmio',
    'struct_pm4_mec_write_data_mmio_1_bitfields2',
    'struct_pm4_mec_write_data_mmio_2_bitfields3', 'uint32_t',
    'union_PM4_MES_TYPE_3_HEADER', 'union_pm4_mec_release_mem_0',
    'union_pm4_mec_release_mem_1', 'union_pm4_mec_release_mem_2',
    'union_pm4_mec_release_mem_3', 'union_pm4_mec_release_mem_4',
    'union_pm4_mec_release_mem_5', 'union_pm4_mec_release_mem_6',
    'union_pm4_mec_write_data_mmio_0',
    'union_pm4_mec_write_data_mmio_1',
    'union_pm4_mec_write_data_mmio_2',
    'wr_confirm___write_data__do_not_wait_for_write_confirmation',
    'wr_confirm___write_data__wait_for_write_confirmation']
