# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util


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

class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['libibverbs'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['libibverbs'] = ctypes.CDLL(ctypes.util.find_library('ibverbs'), use_errno=True) #  ctypes.CDLL('libibverbs')
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





INFINIBAND_VERBS_H = True # macro
VERBS_API_H = True # macro
# def RDMA_UAPI_PTR(_type, _name):  # macro
#    return {_type_name;__aligned_u64_name##_data_u64;}
IB_USER_IOCTL_VERBS_H = True # macro
IB_USER_VERBS_H = True # macro
IB_USER_VERBS_ABI_VERSION = 6 # macro
IB_USER_VERBS_CMD_THRESHOLD = 50 # macro
IB_USER_VERBS_CMD_COMMAND_MASK = 0xff # macro
IB_USER_VERBS_CMD_FLAG_EXTENDED = 0x80000000 # macro
IB_USER_VERBS_MAX_LOG_IND_TBL_SIZE = 0x0d # macro
IB_DEVICE_NAME_MAX = 64 # macro
IB_UVERBS_ACCESS_OPTIONAL_FIRST = (1<<20) # macro
IB_UVERBS_ACCESS_OPTIONAL_LAST = (1<<29) # macro
# ibv_flow_action_esp_keymat_aes_gcm = ib_uverbs_flow_action_esp_keymat_aes_gcm # macro
# ibv_flow_action_esp_replay_bmp = ib_uverbs_flow_action_esp_replay_bmp # macro
# ibv_flow_action_esp_encap = ib_uverbs_flow_action_esp_encap # macro
# ibv_flow_action_esp = ib_uverbs_flow_action_esp # macro
IBV_ACCESS_OPTIONAL_FIRST = (1<<20) # macro
__attribute_const = True # macro
# def vext_field_avail(type, fld, sz):  # macro
#    return (offsetof(type,fld)<(sz))
# __VERBS_ABI_IS_EXTENDED = ((void*)UINTPTR_MAX) # macro
IBV_DEVICE_RAW_SCATTER_FCS = (1<<34) # macro
IBV_DEVICE_PCI_WRITE_END_PADDING = (1<<36) # macro
# IBV_ALLOCATOR_USE_DEFAULT = ((void*)-1) # macro
ETHERNET_LL_SIZE = 6 # macro
IB_ROCE_UDP_ENCAP_VALID_PORT_MIN = (0xC000) # macro
IB_ROCE_UDP_ENCAP_VALID_PORT_MAX = (0xFFFF) # macro
IB_GRH_FLOWLABEL_MASK = (0x000FFFFF) # macro

# values for enumeration 'ib_uverbs_write_cmds'
ib_uverbs_write_cmds__enumvalues = {
    0: 'IB_USER_VERBS_CMD_GET_CONTEXT',
    1: 'IB_USER_VERBS_CMD_QUERY_DEVICE',
    2: 'IB_USER_VERBS_CMD_QUERY_PORT',
    3: 'IB_USER_VERBS_CMD_ALLOC_PD',
    4: 'IB_USER_VERBS_CMD_DEALLOC_PD',
    5: 'IB_USER_VERBS_CMD_CREATE_AH',
    6: 'IB_USER_VERBS_CMD_MODIFY_AH',
    7: 'IB_USER_VERBS_CMD_QUERY_AH',
    8: 'IB_USER_VERBS_CMD_DESTROY_AH',
    9: 'IB_USER_VERBS_CMD_REG_MR',
    10: 'IB_USER_VERBS_CMD_REG_SMR',
    11: 'IB_USER_VERBS_CMD_REREG_MR',
    12: 'IB_USER_VERBS_CMD_QUERY_MR',
    13: 'IB_USER_VERBS_CMD_DEREG_MR',
    14: 'IB_USER_VERBS_CMD_ALLOC_MW',
    15: 'IB_USER_VERBS_CMD_BIND_MW',
    16: 'IB_USER_VERBS_CMD_DEALLOC_MW',
    17: 'IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL',
    18: 'IB_USER_VERBS_CMD_CREATE_CQ',
    19: 'IB_USER_VERBS_CMD_RESIZE_CQ',
    20: 'IB_USER_VERBS_CMD_DESTROY_CQ',
    21: 'IB_USER_VERBS_CMD_POLL_CQ',
    22: 'IB_USER_VERBS_CMD_PEEK_CQ',
    23: 'IB_USER_VERBS_CMD_REQ_NOTIFY_CQ',
    24: 'IB_USER_VERBS_CMD_CREATE_QP',
    25: 'IB_USER_VERBS_CMD_QUERY_QP',
    26: 'IB_USER_VERBS_CMD_MODIFY_QP',
    27: 'IB_USER_VERBS_CMD_DESTROY_QP',
    28: 'IB_USER_VERBS_CMD_POST_SEND',
    29: 'IB_USER_VERBS_CMD_POST_RECV',
    30: 'IB_USER_VERBS_CMD_ATTACH_MCAST',
    31: 'IB_USER_VERBS_CMD_DETACH_MCAST',
    32: 'IB_USER_VERBS_CMD_CREATE_SRQ',
    33: 'IB_USER_VERBS_CMD_MODIFY_SRQ',
    34: 'IB_USER_VERBS_CMD_QUERY_SRQ',
    35: 'IB_USER_VERBS_CMD_DESTROY_SRQ',
    36: 'IB_USER_VERBS_CMD_POST_SRQ_RECV',
    37: 'IB_USER_VERBS_CMD_OPEN_XRCD',
    38: 'IB_USER_VERBS_CMD_CLOSE_XRCD',
    39: 'IB_USER_VERBS_CMD_CREATE_XSRQ',
    40: 'IB_USER_VERBS_CMD_OPEN_QP',
}
IB_USER_VERBS_CMD_GET_CONTEXT = 0
IB_USER_VERBS_CMD_QUERY_DEVICE = 1
IB_USER_VERBS_CMD_QUERY_PORT = 2
IB_USER_VERBS_CMD_ALLOC_PD = 3
IB_USER_VERBS_CMD_DEALLOC_PD = 4
IB_USER_VERBS_CMD_CREATE_AH = 5
IB_USER_VERBS_CMD_MODIFY_AH = 6
IB_USER_VERBS_CMD_QUERY_AH = 7
IB_USER_VERBS_CMD_DESTROY_AH = 8
IB_USER_VERBS_CMD_REG_MR = 9
IB_USER_VERBS_CMD_REG_SMR = 10
IB_USER_VERBS_CMD_REREG_MR = 11
IB_USER_VERBS_CMD_QUERY_MR = 12
IB_USER_VERBS_CMD_DEREG_MR = 13
IB_USER_VERBS_CMD_ALLOC_MW = 14
IB_USER_VERBS_CMD_BIND_MW = 15
IB_USER_VERBS_CMD_DEALLOC_MW = 16
IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL = 17
IB_USER_VERBS_CMD_CREATE_CQ = 18
IB_USER_VERBS_CMD_RESIZE_CQ = 19
IB_USER_VERBS_CMD_DESTROY_CQ = 20
IB_USER_VERBS_CMD_POLL_CQ = 21
IB_USER_VERBS_CMD_PEEK_CQ = 22
IB_USER_VERBS_CMD_REQ_NOTIFY_CQ = 23
IB_USER_VERBS_CMD_CREATE_QP = 24
IB_USER_VERBS_CMD_QUERY_QP = 25
IB_USER_VERBS_CMD_MODIFY_QP = 26
IB_USER_VERBS_CMD_DESTROY_QP = 27
IB_USER_VERBS_CMD_POST_SEND = 28
IB_USER_VERBS_CMD_POST_RECV = 29
IB_USER_VERBS_CMD_ATTACH_MCAST = 30
IB_USER_VERBS_CMD_DETACH_MCAST = 31
IB_USER_VERBS_CMD_CREATE_SRQ = 32
IB_USER_VERBS_CMD_MODIFY_SRQ = 33
IB_USER_VERBS_CMD_QUERY_SRQ = 34
IB_USER_VERBS_CMD_DESTROY_SRQ = 35
IB_USER_VERBS_CMD_POST_SRQ_RECV = 36
IB_USER_VERBS_CMD_OPEN_XRCD = 37
IB_USER_VERBS_CMD_CLOSE_XRCD = 38
IB_USER_VERBS_CMD_CREATE_XSRQ = 39
IB_USER_VERBS_CMD_OPEN_QP = 40
ib_uverbs_write_cmds = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_IB_USER_VERBS_EX_CMD_QUERY_DEVICE'
c__Ea_IB_USER_VERBS_EX_CMD_QUERY_DEVICE__enumvalues = {
    1: 'IB_USER_VERBS_EX_CMD_QUERY_DEVICE',
    18: 'IB_USER_VERBS_EX_CMD_CREATE_CQ',
    24: 'IB_USER_VERBS_EX_CMD_CREATE_QP',
    26: 'IB_USER_VERBS_EX_CMD_MODIFY_QP',
    50: 'IB_USER_VERBS_EX_CMD_CREATE_FLOW',
    51: 'IB_USER_VERBS_EX_CMD_DESTROY_FLOW',
    52: 'IB_USER_VERBS_EX_CMD_CREATE_WQ',
    53: 'IB_USER_VERBS_EX_CMD_MODIFY_WQ',
    54: 'IB_USER_VERBS_EX_CMD_DESTROY_WQ',
    55: 'IB_USER_VERBS_EX_CMD_CREATE_RWQ_IND_TBL',
    56: 'IB_USER_VERBS_EX_CMD_DESTROY_RWQ_IND_TBL',
    57: 'IB_USER_VERBS_EX_CMD_MODIFY_CQ',
}
IB_USER_VERBS_EX_CMD_QUERY_DEVICE = 1
IB_USER_VERBS_EX_CMD_CREATE_CQ = 18
IB_USER_VERBS_EX_CMD_CREATE_QP = 24
IB_USER_VERBS_EX_CMD_MODIFY_QP = 26
IB_USER_VERBS_EX_CMD_CREATE_FLOW = 50
IB_USER_VERBS_EX_CMD_DESTROY_FLOW = 51
IB_USER_VERBS_EX_CMD_CREATE_WQ = 52
IB_USER_VERBS_EX_CMD_MODIFY_WQ = 53
IB_USER_VERBS_EX_CMD_DESTROY_WQ = 54
IB_USER_VERBS_EX_CMD_CREATE_RWQ_IND_TBL = 55
IB_USER_VERBS_EX_CMD_DESTROY_RWQ_IND_TBL = 56
IB_USER_VERBS_EX_CMD_MODIFY_CQ = 57
c__Ea_IB_USER_VERBS_EX_CMD_QUERY_DEVICE = ctypes.c_uint32 # enum

# values for enumeration 'ib_placement_type'
ib_placement_type__enumvalues = {
    1: 'IB_FLUSH_GLOBAL',
    2: 'IB_FLUSH_PERSISTENT',
}
IB_FLUSH_GLOBAL = 1
IB_FLUSH_PERSISTENT = 2
ib_placement_type = ctypes.c_uint32 # enum

# values for enumeration 'ib_selectivity_level'
ib_selectivity_level__enumvalues = {
    0: 'IB_FLUSH_RANGE',
    1: 'IB_FLUSH_MR',
}
IB_FLUSH_RANGE = 0
IB_FLUSH_MR = 1
ib_selectivity_level = ctypes.c_uint32 # enum
class struct_ib_uverbs_async_event_desc(Structure):
    pass

struct_ib_uverbs_async_event_desc._pack_ = 1 # source:False
struct_ib_uverbs_async_event_desc._fields_ = [
    ('element', ctypes.c_uint64),
    ('event_type', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_comp_event_desc(Structure):
    pass

struct_ib_uverbs_comp_event_desc._pack_ = 1 # source:False
struct_ib_uverbs_comp_event_desc._fields_ = [
    ('cq_handle', ctypes.c_uint64),
]

class struct_ib_uverbs_cq_moderation_caps(Structure):
    pass

struct_ib_uverbs_cq_moderation_caps._pack_ = 1 # source:False
struct_ib_uverbs_cq_moderation_caps._fields_ = [
    ('max_cq_moderation_count', ctypes.c_uint16),
    ('max_cq_moderation_period', ctypes.c_uint16),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_cmd_hdr(Structure):
    pass

struct_ib_uverbs_cmd_hdr._pack_ = 1 # source:False
struct_ib_uverbs_cmd_hdr._fields_ = [
    ('command', ctypes.c_uint32),
    ('in_words', ctypes.c_uint16),
    ('out_words', ctypes.c_uint16),
]

class struct_ib_uverbs_ex_cmd_hdr(Structure):
    pass

struct_ib_uverbs_ex_cmd_hdr._pack_ = 1 # source:False
struct_ib_uverbs_ex_cmd_hdr._fields_ = [
    ('response', ctypes.c_uint64),
    ('provider_in_words', ctypes.c_uint16),
    ('provider_out_words', ctypes.c_uint16),
    ('cmd_hdr_reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_get_context(Structure):
    pass

struct_ib_uverbs_get_context._pack_ = 1 # source:False
struct_ib_uverbs_get_context._fields_ = [
    ('response', ctypes.c_uint64),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_get_context_resp(Structure):
    pass

struct_ib_uverbs_get_context_resp._pack_ = 1 # source:False
struct_ib_uverbs_get_context_resp._fields_ = [
    ('async_fd', ctypes.c_uint32),
    ('num_comp_vectors', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_query_device(Structure):
    pass

struct_ib_uverbs_query_device._pack_ = 1 # source:False
struct_ib_uverbs_query_device._fields_ = [
    ('response', ctypes.c_uint64),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_query_device_resp(Structure):
    pass

struct_ib_uverbs_query_device_resp._pack_ = 1 # source:False
struct_ib_uverbs_query_device_resp._fields_ = [
    ('fw_ver', ctypes.c_uint64),
    ('node_guid', ctypes.c_uint64),
    ('sys_image_guid', ctypes.c_uint64),
    ('max_mr_size', ctypes.c_uint64),
    ('page_size_cap', ctypes.c_uint64),
    ('vendor_id', ctypes.c_uint32),
    ('vendor_part_id', ctypes.c_uint32),
    ('hw_ver', ctypes.c_uint32),
    ('max_qp', ctypes.c_uint32),
    ('max_qp_wr', ctypes.c_uint32),
    ('device_cap_flags', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('max_sge_rd', ctypes.c_uint32),
    ('max_cq', ctypes.c_uint32),
    ('max_cqe', ctypes.c_uint32),
    ('max_mr', ctypes.c_uint32),
    ('max_pd', ctypes.c_uint32),
    ('max_qp_rd_atom', ctypes.c_uint32),
    ('max_ee_rd_atom', ctypes.c_uint32),
    ('max_res_rd_atom', ctypes.c_uint32),
    ('max_qp_init_rd_atom', ctypes.c_uint32),
    ('max_ee_init_rd_atom', ctypes.c_uint32),
    ('atomic_cap', ctypes.c_uint32),
    ('max_ee', ctypes.c_uint32),
    ('max_rdd', ctypes.c_uint32),
    ('max_mw', ctypes.c_uint32),
    ('max_raw_ipv6_qp', ctypes.c_uint32),
    ('max_raw_ethy_qp', ctypes.c_uint32),
    ('max_mcast_grp', ctypes.c_uint32),
    ('max_mcast_qp_attach', ctypes.c_uint32),
    ('max_total_mcast_qp_attach', ctypes.c_uint32),
    ('max_ah', ctypes.c_uint32),
    ('max_fmr', ctypes.c_uint32),
    ('max_map_per_fmr', ctypes.c_uint32),
    ('max_srq', ctypes.c_uint32),
    ('max_srq_wr', ctypes.c_uint32),
    ('max_srq_sge', ctypes.c_uint32),
    ('max_pkeys', ctypes.c_uint16),
    ('local_ca_ack_delay', ctypes.c_ubyte),
    ('phys_port_cnt', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte * 4),
]

class struct_ib_uverbs_ex_query_device(Structure):
    pass

struct_ib_uverbs_ex_query_device._pack_ = 1 # source:False
struct_ib_uverbs_ex_query_device._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_odp_caps(Structure):
    pass

class struct_ib_uverbs_odp_caps_per_transport_caps(Structure):
    pass

struct_ib_uverbs_odp_caps_per_transport_caps._pack_ = 1 # source:False
struct_ib_uverbs_odp_caps_per_transport_caps._fields_ = [
    ('rc_odp_caps', ctypes.c_uint32),
    ('uc_odp_caps', ctypes.c_uint32),
    ('ud_odp_caps', ctypes.c_uint32),
]

struct_ib_uverbs_odp_caps._pack_ = 1 # source:False
struct_ib_uverbs_odp_caps._fields_ = [
    ('general_caps', ctypes.c_uint64),
    ('per_transport_caps', struct_ib_uverbs_odp_caps_per_transport_caps),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_rss_caps(Structure):
    pass

struct_ib_uverbs_rss_caps._pack_ = 1 # source:False
struct_ib_uverbs_rss_caps._fields_ = [
    ('supported_qpts', ctypes.c_uint32),
    ('max_rwq_indirection_tables', ctypes.c_uint32),
    ('max_rwq_indirection_table_size', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_tm_caps(Structure):
    pass

struct_ib_uverbs_tm_caps._pack_ = 1 # source:False
struct_ib_uverbs_tm_caps._fields_ = [
    ('max_rndv_hdr_size', ctypes.c_uint32),
    ('max_num_tags', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('max_ops', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_ex_query_device_resp(Structure):
    pass

struct_ib_uverbs_ex_query_device_resp._pack_ = 1 # source:False
struct_ib_uverbs_ex_query_device_resp._fields_ = [
    ('base', struct_ib_uverbs_query_device_resp),
    ('comp_mask', ctypes.c_uint32),
    ('response_length', ctypes.c_uint32),
    ('odp_caps', struct_ib_uverbs_odp_caps),
    ('timestamp_mask', ctypes.c_uint64),
    ('hca_core_clock', ctypes.c_uint64),
    ('device_cap_flags_ex', ctypes.c_uint64),
    ('rss_caps', struct_ib_uverbs_rss_caps),
    ('max_wq_type_rq', ctypes.c_uint32),
    ('raw_packet_caps', ctypes.c_uint32),
    ('tm_caps', struct_ib_uverbs_tm_caps),
    ('cq_moderation_caps', struct_ib_uverbs_cq_moderation_caps),
    ('max_dm_size', ctypes.c_uint64),
    ('xrc_odp_caps', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_query_port(Structure):
    pass

struct_ib_uverbs_query_port._pack_ = 1 # source:False
struct_ib_uverbs_query_port._fields_ = [
    ('response', ctypes.c_uint64),
    ('port_num', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte * 7),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_query_port_resp(Structure):
    pass

struct_ib_uverbs_query_port_resp._pack_ = 1 # source:False
struct_ib_uverbs_query_port_resp._fields_ = [
    ('port_cap_flags', ctypes.c_uint32),
    ('max_msg_sz', ctypes.c_uint32),
    ('bad_pkey_cntr', ctypes.c_uint32),
    ('qkey_viol_cntr', ctypes.c_uint32),
    ('gid_tbl_len', ctypes.c_uint32),
    ('pkey_tbl_len', ctypes.c_uint16),
    ('lid', ctypes.c_uint16),
    ('sm_lid', ctypes.c_uint16),
    ('state', ctypes.c_ubyte),
    ('max_mtu', ctypes.c_ubyte),
    ('active_mtu', ctypes.c_ubyte),
    ('lmc', ctypes.c_ubyte),
    ('max_vl_num', ctypes.c_ubyte),
    ('sm_sl', ctypes.c_ubyte),
    ('subnet_timeout', ctypes.c_ubyte),
    ('init_type_reply', ctypes.c_ubyte),
    ('active_width', ctypes.c_ubyte),
    ('active_speed', ctypes.c_ubyte),
    ('phys_state', ctypes.c_ubyte),
    ('link_layer', ctypes.c_ubyte),
    ('flags', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

class struct_ib_uverbs_alloc_pd(Structure):
    pass

struct_ib_uverbs_alloc_pd._pack_ = 1 # source:False
struct_ib_uverbs_alloc_pd._fields_ = [
    ('response', ctypes.c_uint64),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_alloc_pd_resp(Structure):
    pass

struct_ib_uverbs_alloc_pd_resp._pack_ = 1 # source:False
struct_ib_uverbs_alloc_pd_resp._fields_ = [
    ('pd_handle', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint32 * 0),
]

class struct_ib_uverbs_dealloc_pd(Structure):
    pass

struct_ib_uverbs_dealloc_pd._pack_ = 1 # source:False
struct_ib_uverbs_dealloc_pd._fields_ = [
    ('pd_handle', ctypes.c_uint32),
]

class struct_ib_uverbs_open_xrcd(Structure):
    pass

struct_ib_uverbs_open_xrcd._pack_ = 1 # source:False
struct_ib_uverbs_open_xrcd._fields_ = [
    ('response', ctypes.c_uint64),
    ('fd', ctypes.c_uint32),
    ('oflags', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_open_xrcd_resp(Structure):
    pass

struct_ib_uverbs_open_xrcd_resp._pack_ = 1 # source:False
struct_ib_uverbs_open_xrcd_resp._fields_ = [
    ('xrcd_handle', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint32 * 0),
]

class struct_ib_uverbs_close_xrcd(Structure):
    pass

struct_ib_uverbs_close_xrcd._pack_ = 1 # source:False
struct_ib_uverbs_close_xrcd._fields_ = [
    ('xrcd_handle', ctypes.c_uint32),
]

class struct_ib_uverbs_reg_mr(Structure):
    pass

struct_ib_uverbs_reg_mr._pack_ = 1 # source:False
struct_ib_uverbs_reg_mr._fields_ = [
    ('response', ctypes.c_uint64),
    ('start', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('hca_va', ctypes.c_uint64),
    ('pd_handle', ctypes.c_uint32),
    ('access_flags', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_reg_mr_resp(Structure):
    pass

struct_ib_uverbs_reg_mr_resp._pack_ = 1 # source:False
struct_ib_uverbs_reg_mr_resp._fields_ = [
    ('mr_handle', ctypes.c_uint32),
    ('lkey', ctypes.c_uint32),
    ('rkey', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint32 * 0),
]

class struct_ib_uverbs_rereg_mr(Structure):
    pass

struct_ib_uverbs_rereg_mr._pack_ = 1 # source:False
struct_ib_uverbs_rereg_mr._fields_ = [
    ('response', ctypes.c_uint64),
    ('mr_handle', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('start', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('hca_va', ctypes.c_uint64),
    ('pd_handle', ctypes.c_uint32),
    ('access_flags', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_rereg_mr_resp(Structure):
    pass

struct_ib_uverbs_rereg_mr_resp._pack_ = 1 # source:False
struct_ib_uverbs_rereg_mr_resp._fields_ = [
    ('lkey', ctypes.c_uint32),
    ('rkey', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_dereg_mr(Structure):
    pass

struct_ib_uverbs_dereg_mr._pack_ = 1 # source:False
struct_ib_uverbs_dereg_mr._fields_ = [
    ('mr_handle', ctypes.c_uint32),
]

class struct_ib_uverbs_alloc_mw(Structure):
    pass

struct_ib_uverbs_alloc_mw._pack_ = 1 # source:False
struct_ib_uverbs_alloc_mw._fields_ = [
    ('response', ctypes.c_uint64),
    ('pd_handle', ctypes.c_uint32),
    ('mw_type', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte * 3),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_alloc_mw_resp(Structure):
    pass

struct_ib_uverbs_alloc_mw_resp._pack_ = 1 # source:False
struct_ib_uverbs_alloc_mw_resp._fields_ = [
    ('mw_handle', ctypes.c_uint32),
    ('rkey', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_dealloc_mw(Structure):
    pass

struct_ib_uverbs_dealloc_mw._pack_ = 1 # source:False
struct_ib_uverbs_dealloc_mw._fields_ = [
    ('mw_handle', ctypes.c_uint32),
]

class struct_ib_uverbs_create_comp_channel(Structure):
    pass

struct_ib_uverbs_create_comp_channel._pack_ = 1 # source:False
struct_ib_uverbs_create_comp_channel._fields_ = [
    ('response', ctypes.c_uint64),
]

class struct_ib_uverbs_create_comp_channel_resp(Structure):
    pass

struct_ib_uverbs_create_comp_channel_resp._pack_ = 1 # source:False
struct_ib_uverbs_create_comp_channel_resp._fields_ = [
    ('fd', ctypes.c_uint32),
]

class struct_ib_uverbs_create_cq(Structure):
    pass

struct_ib_uverbs_create_cq._pack_ = 1 # source:False
struct_ib_uverbs_create_cq._fields_ = [
    ('response', ctypes.c_uint64),
    ('user_handle', ctypes.c_uint64),
    ('cqe', ctypes.c_uint32),
    ('comp_vector', ctypes.c_uint32),
    ('comp_channel', ctypes.c_int32),
    ('reserved', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]


# values for enumeration 'ib_uverbs_ex_create_cq_flags'
ib_uverbs_ex_create_cq_flags__enumvalues = {
    1: 'IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION',
    2: 'IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN',
}
IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION = 1
IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN = 2
ib_uverbs_ex_create_cq_flags = ctypes.c_uint32 # enum
class struct_ib_uverbs_ex_create_cq(Structure):
    pass

struct_ib_uverbs_ex_create_cq._pack_ = 1 # source:False
struct_ib_uverbs_ex_create_cq._fields_ = [
    ('user_handle', ctypes.c_uint64),
    ('cqe', ctypes.c_uint32),
    ('comp_vector', ctypes.c_uint32),
    ('comp_channel', ctypes.c_int32),
    ('comp_mask', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_create_cq_resp(Structure):
    pass

struct_ib_uverbs_create_cq_resp._pack_ = 1 # source:False
struct_ib_uverbs_create_cq_resp._fields_ = [
    ('cq_handle', ctypes.c_uint32),
    ('cqe', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_ex_create_cq_resp(Structure):
    pass

struct_ib_uverbs_ex_create_cq_resp._pack_ = 1 # source:False
struct_ib_uverbs_ex_create_cq_resp._fields_ = [
    ('base', struct_ib_uverbs_create_cq_resp),
    ('comp_mask', ctypes.c_uint32),
    ('response_length', ctypes.c_uint32),
]

class struct_ib_uverbs_resize_cq(Structure):
    pass

struct_ib_uverbs_resize_cq._pack_ = 1 # source:False
struct_ib_uverbs_resize_cq._fields_ = [
    ('response', ctypes.c_uint64),
    ('cq_handle', ctypes.c_uint32),
    ('cqe', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_resize_cq_resp(Structure):
    pass

struct_ib_uverbs_resize_cq_resp._pack_ = 1 # source:False
struct_ib_uverbs_resize_cq_resp._fields_ = [
    ('cqe', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_poll_cq(Structure):
    pass

struct_ib_uverbs_poll_cq._pack_ = 1 # source:False
struct_ib_uverbs_poll_cq._fields_ = [
    ('response', ctypes.c_uint64),
    ('cq_handle', ctypes.c_uint32),
    ('ne', ctypes.c_uint32),
]


# values for enumeration 'ib_uverbs_wc_opcode'
ib_uverbs_wc_opcode__enumvalues = {
    0: 'IB_UVERBS_WC_SEND',
    1: 'IB_UVERBS_WC_RDMA_WRITE',
    2: 'IB_UVERBS_WC_RDMA_READ',
    3: 'IB_UVERBS_WC_COMP_SWAP',
    4: 'IB_UVERBS_WC_FETCH_ADD',
    5: 'IB_UVERBS_WC_BIND_MW',
    6: 'IB_UVERBS_WC_LOCAL_INV',
    7: 'IB_UVERBS_WC_TSO',
    8: 'IB_UVERBS_WC_FLUSH',
    9: 'IB_UVERBS_WC_ATOMIC_WRITE',
}
IB_UVERBS_WC_SEND = 0
IB_UVERBS_WC_RDMA_WRITE = 1
IB_UVERBS_WC_RDMA_READ = 2
IB_UVERBS_WC_COMP_SWAP = 3
IB_UVERBS_WC_FETCH_ADD = 4
IB_UVERBS_WC_BIND_MW = 5
IB_UVERBS_WC_LOCAL_INV = 6
IB_UVERBS_WC_TSO = 7
IB_UVERBS_WC_FLUSH = 8
IB_UVERBS_WC_ATOMIC_WRITE = 9
ib_uverbs_wc_opcode = ctypes.c_uint32 # enum
class struct_ib_uverbs_wc(Structure):
    pass

class union_ib_uverbs_wc_ex(Union):
    pass

union_ib_uverbs_wc_ex._pack_ = 1 # source:False
union_ib_uverbs_wc_ex._fields_ = [
    ('imm_data', ctypes.c_uint32),
    ('invalidate_rkey', ctypes.c_uint32),
]

struct_ib_uverbs_wc._pack_ = 1 # source:False
struct_ib_uverbs_wc._fields_ = [
    ('wr_id', ctypes.c_uint64),
    ('status', ctypes.c_uint32),
    ('opcode', ctypes.c_uint32),
    ('vendor_err', ctypes.c_uint32),
    ('byte_len', ctypes.c_uint32),
    ('ex', union_ib_uverbs_wc_ex),
    ('qp_num', ctypes.c_uint32),
    ('src_qp', ctypes.c_uint32),
    ('wc_flags', ctypes.c_uint32),
    ('pkey_index', ctypes.c_uint16),
    ('slid', ctypes.c_uint16),
    ('sl', ctypes.c_ubyte),
    ('dlid_path_bits', ctypes.c_ubyte),
    ('port_num', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

class struct_ib_uverbs_poll_cq_resp(Structure):
    pass

struct_ib_uverbs_poll_cq_resp._pack_ = 1 # source:False
struct_ib_uverbs_poll_cq_resp._fields_ = [
    ('count', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('wc', struct_ib_uverbs_wc * 0),
]

class struct_ib_uverbs_req_notify_cq(Structure):
    pass

struct_ib_uverbs_req_notify_cq._pack_ = 1 # source:False
struct_ib_uverbs_req_notify_cq._fields_ = [
    ('cq_handle', ctypes.c_uint32),
    ('solicited_only', ctypes.c_uint32),
]

class struct_ib_uverbs_destroy_cq(Structure):
    pass

struct_ib_uverbs_destroy_cq._pack_ = 1 # source:False
struct_ib_uverbs_destroy_cq._fields_ = [
    ('response', ctypes.c_uint64),
    ('cq_handle', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_destroy_cq_resp(Structure):
    pass

struct_ib_uverbs_destroy_cq_resp._pack_ = 1 # source:False
struct_ib_uverbs_destroy_cq_resp._fields_ = [
    ('comp_events_reported', ctypes.c_uint32),
    ('async_events_reported', ctypes.c_uint32),
]

class struct_ib_uverbs_global_route(Structure):
    pass

struct_ib_uverbs_global_route._pack_ = 1 # source:False
struct_ib_uverbs_global_route._fields_ = [
    ('dgid', ctypes.c_ubyte * 16),
    ('flow_label', ctypes.c_uint32),
    ('sgid_index', ctypes.c_ubyte),
    ('hop_limit', ctypes.c_ubyte),
    ('traffic_class', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

class struct_ib_uverbs_ah_attr(Structure):
    pass

struct_ib_uverbs_ah_attr._pack_ = 1 # source:False
struct_ib_uverbs_ah_attr._fields_ = [
    ('grh', struct_ib_uverbs_global_route),
    ('dlid', ctypes.c_uint16),
    ('sl', ctypes.c_ubyte),
    ('src_path_bits', ctypes.c_ubyte),
    ('static_rate', ctypes.c_ubyte),
    ('is_global', ctypes.c_ubyte),
    ('port_num', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

class struct_ib_uverbs_qp_attr(Structure):
    pass

struct_ib_uverbs_qp_attr._pack_ = 1 # source:False
struct_ib_uverbs_qp_attr._fields_ = [
    ('qp_attr_mask', ctypes.c_uint32),
    ('qp_state', ctypes.c_uint32),
    ('cur_qp_state', ctypes.c_uint32),
    ('path_mtu', ctypes.c_uint32),
    ('path_mig_state', ctypes.c_uint32),
    ('qkey', ctypes.c_uint32),
    ('rq_psn', ctypes.c_uint32),
    ('sq_psn', ctypes.c_uint32),
    ('dest_qp_num', ctypes.c_uint32),
    ('qp_access_flags', ctypes.c_uint32),
    ('ah_attr', struct_ib_uverbs_ah_attr),
    ('alt_ah_attr', struct_ib_uverbs_ah_attr),
    ('max_send_wr', ctypes.c_uint32),
    ('max_recv_wr', ctypes.c_uint32),
    ('max_send_sge', ctypes.c_uint32),
    ('max_recv_sge', ctypes.c_uint32),
    ('max_inline_data', ctypes.c_uint32),
    ('pkey_index', ctypes.c_uint16),
    ('alt_pkey_index', ctypes.c_uint16),
    ('en_sqd_async_notify', ctypes.c_ubyte),
    ('sq_draining', ctypes.c_ubyte),
    ('max_rd_atomic', ctypes.c_ubyte),
    ('max_dest_rd_atomic', ctypes.c_ubyte),
    ('min_rnr_timer', ctypes.c_ubyte),
    ('port_num', ctypes.c_ubyte),
    ('timeout', ctypes.c_ubyte),
    ('retry_cnt', ctypes.c_ubyte),
    ('rnr_retry', ctypes.c_ubyte),
    ('alt_port_num', ctypes.c_ubyte),
    ('alt_timeout', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte * 5),
]

class struct_ib_uverbs_create_qp(Structure):
    pass

struct_ib_uverbs_create_qp._pack_ = 1 # source:False
struct_ib_uverbs_create_qp._fields_ = [
    ('response', ctypes.c_uint64),
    ('user_handle', ctypes.c_uint64),
    ('pd_handle', ctypes.c_uint32),
    ('send_cq_handle', ctypes.c_uint32),
    ('recv_cq_handle', ctypes.c_uint32),
    ('srq_handle', ctypes.c_uint32),
    ('max_send_wr', ctypes.c_uint32),
    ('max_recv_wr', ctypes.c_uint32),
    ('max_send_sge', ctypes.c_uint32),
    ('max_recv_sge', ctypes.c_uint32),
    ('max_inline_data', ctypes.c_uint32),
    ('sq_sig_all', ctypes.c_ubyte),
    ('qp_type', ctypes.c_ubyte),
    ('is_srq', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
    ('driver_data', ctypes.c_uint64 * 0),
]


# values for enumeration 'ib_uverbs_create_qp_mask'
ib_uverbs_create_qp_mask__enumvalues = {
    1: 'IB_UVERBS_CREATE_QP_MASK_IND_TABLE',
}
IB_UVERBS_CREATE_QP_MASK_IND_TABLE = 1
ib_uverbs_create_qp_mask = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_IB_UVERBS_CREATE_QP_SUP_COMP_MASK'
c__Ea_IB_UVERBS_CREATE_QP_SUP_COMP_MASK__enumvalues = {
    1: 'IB_UVERBS_CREATE_QP_SUP_COMP_MASK',
}
IB_UVERBS_CREATE_QP_SUP_COMP_MASK = 1
c__Ea_IB_UVERBS_CREATE_QP_SUP_COMP_MASK = ctypes.c_uint32 # enum
class struct_ib_uverbs_ex_create_qp(Structure):
    pass

struct_ib_uverbs_ex_create_qp._pack_ = 1 # source:False
struct_ib_uverbs_ex_create_qp._fields_ = [
    ('user_handle', ctypes.c_uint64),
    ('pd_handle', ctypes.c_uint32),
    ('send_cq_handle', ctypes.c_uint32),
    ('recv_cq_handle', ctypes.c_uint32),
    ('srq_handle', ctypes.c_uint32),
    ('max_send_wr', ctypes.c_uint32),
    ('max_recv_wr', ctypes.c_uint32),
    ('max_send_sge', ctypes.c_uint32),
    ('max_recv_sge', ctypes.c_uint32),
    ('max_inline_data', ctypes.c_uint32),
    ('sq_sig_all', ctypes.c_ubyte),
    ('qp_type', ctypes.c_ubyte),
    ('is_srq', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
    ('comp_mask', ctypes.c_uint32),
    ('create_flags', ctypes.c_uint32),
    ('rwq_ind_tbl_handle', ctypes.c_uint32),
    ('source_qpn', ctypes.c_uint32),
]

class struct_ib_uverbs_open_qp(Structure):
    pass

struct_ib_uverbs_open_qp._pack_ = 1 # source:False
struct_ib_uverbs_open_qp._fields_ = [
    ('response', ctypes.c_uint64),
    ('user_handle', ctypes.c_uint64),
    ('pd_handle', ctypes.c_uint32),
    ('qpn', ctypes.c_uint32),
    ('qp_type', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte * 7),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_create_qp_resp(Structure):
    pass

struct_ib_uverbs_create_qp_resp._pack_ = 1 # source:False
struct_ib_uverbs_create_qp_resp._fields_ = [
    ('qp_handle', ctypes.c_uint32),
    ('qpn', ctypes.c_uint32),
    ('max_send_wr', ctypes.c_uint32),
    ('max_recv_wr', ctypes.c_uint32),
    ('max_send_sge', ctypes.c_uint32),
    ('max_recv_sge', ctypes.c_uint32),
    ('max_inline_data', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint32 * 0),
]

class struct_ib_uverbs_ex_create_qp_resp(Structure):
    pass

struct_ib_uverbs_ex_create_qp_resp._pack_ = 1 # source:False
struct_ib_uverbs_ex_create_qp_resp._fields_ = [
    ('base', struct_ib_uverbs_create_qp_resp),
    ('comp_mask', ctypes.c_uint32),
    ('response_length', ctypes.c_uint32),
]

class struct_ib_uverbs_qp_dest(Structure):
    pass

struct_ib_uverbs_qp_dest._pack_ = 1 # source:False
struct_ib_uverbs_qp_dest._fields_ = [
    ('dgid', ctypes.c_ubyte * 16),
    ('flow_label', ctypes.c_uint32),
    ('dlid', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
    ('sgid_index', ctypes.c_ubyte),
    ('hop_limit', ctypes.c_ubyte),
    ('traffic_class', ctypes.c_ubyte),
    ('sl', ctypes.c_ubyte),
    ('src_path_bits', ctypes.c_ubyte),
    ('static_rate', ctypes.c_ubyte),
    ('is_global', ctypes.c_ubyte),
    ('port_num', ctypes.c_ubyte),
]

class struct_ib_uverbs_query_qp(Structure):
    pass

struct_ib_uverbs_query_qp._pack_ = 1 # source:False
struct_ib_uverbs_query_qp._fields_ = [
    ('response', ctypes.c_uint64),
    ('qp_handle', ctypes.c_uint32),
    ('attr_mask', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_query_qp_resp(Structure):
    pass

struct_ib_uverbs_query_qp_resp._pack_ = 1 # source:False
struct_ib_uverbs_query_qp_resp._fields_ = [
    ('dest', struct_ib_uverbs_qp_dest),
    ('alt_dest', struct_ib_uverbs_qp_dest),
    ('max_send_wr', ctypes.c_uint32),
    ('max_recv_wr', ctypes.c_uint32),
    ('max_send_sge', ctypes.c_uint32),
    ('max_recv_sge', ctypes.c_uint32),
    ('max_inline_data', ctypes.c_uint32),
    ('qkey', ctypes.c_uint32),
    ('rq_psn', ctypes.c_uint32),
    ('sq_psn', ctypes.c_uint32),
    ('dest_qp_num', ctypes.c_uint32),
    ('qp_access_flags', ctypes.c_uint32),
    ('pkey_index', ctypes.c_uint16),
    ('alt_pkey_index', ctypes.c_uint16),
    ('qp_state', ctypes.c_ubyte),
    ('cur_qp_state', ctypes.c_ubyte),
    ('path_mtu', ctypes.c_ubyte),
    ('path_mig_state', ctypes.c_ubyte),
    ('sq_draining', ctypes.c_ubyte),
    ('max_rd_atomic', ctypes.c_ubyte),
    ('max_dest_rd_atomic', ctypes.c_ubyte),
    ('min_rnr_timer', ctypes.c_ubyte),
    ('port_num', ctypes.c_ubyte),
    ('timeout', ctypes.c_ubyte),
    ('retry_cnt', ctypes.c_ubyte),
    ('rnr_retry', ctypes.c_ubyte),
    ('alt_port_num', ctypes.c_ubyte),
    ('alt_timeout', ctypes.c_ubyte),
    ('sq_sig_all', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte * 5),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_modify_qp(Structure):
    pass

struct_ib_uverbs_modify_qp._pack_ = 1 # source:False
struct_ib_uverbs_modify_qp._fields_ = [
    ('dest', struct_ib_uverbs_qp_dest),
    ('alt_dest', struct_ib_uverbs_qp_dest),
    ('qp_handle', ctypes.c_uint32),
    ('attr_mask', ctypes.c_uint32),
    ('qkey', ctypes.c_uint32),
    ('rq_psn', ctypes.c_uint32),
    ('sq_psn', ctypes.c_uint32),
    ('dest_qp_num', ctypes.c_uint32),
    ('qp_access_flags', ctypes.c_uint32),
    ('pkey_index', ctypes.c_uint16),
    ('alt_pkey_index', ctypes.c_uint16),
    ('qp_state', ctypes.c_ubyte),
    ('cur_qp_state', ctypes.c_ubyte),
    ('path_mtu', ctypes.c_ubyte),
    ('path_mig_state', ctypes.c_ubyte),
    ('en_sqd_async_notify', ctypes.c_ubyte),
    ('max_rd_atomic', ctypes.c_ubyte),
    ('max_dest_rd_atomic', ctypes.c_ubyte),
    ('min_rnr_timer', ctypes.c_ubyte),
    ('port_num', ctypes.c_ubyte),
    ('timeout', ctypes.c_ubyte),
    ('retry_cnt', ctypes.c_ubyte),
    ('rnr_retry', ctypes.c_ubyte),
    ('alt_port_num', ctypes.c_ubyte),
    ('alt_timeout', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte * 2),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_ex_modify_qp(Structure):
    pass

struct_ib_uverbs_ex_modify_qp._pack_ = 1 # source:False
struct_ib_uverbs_ex_modify_qp._fields_ = [
    ('base', struct_ib_uverbs_modify_qp),
    ('rate_limit', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_ex_modify_qp_resp(Structure):
    pass

struct_ib_uverbs_ex_modify_qp_resp._pack_ = 1 # source:False
struct_ib_uverbs_ex_modify_qp_resp._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('response_length', ctypes.c_uint32),
]

class struct_ib_uverbs_destroy_qp(Structure):
    pass

struct_ib_uverbs_destroy_qp._pack_ = 1 # source:False
struct_ib_uverbs_destroy_qp._fields_ = [
    ('response', ctypes.c_uint64),
    ('qp_handle', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_destroy_qp_resp(Structure):
    pass

struct_ib_uverbs_destroy_qp_resp._pack_ = 1 # source:False
struct_ib_uverbs_destroy_qp_resp._fields_ = [
    ('events_reported', ctypes.c_uint32),
]

class struct_ib_uverbs_sge(Structure):
    pass

struct_ib_uverbs_sge._pack_ = 1 # source:False
struct_ib_uverbs_sge._fields_ = [
    ('addr', ctypes.c_uint64),
    ('length', ctypes.c_uint32),
    ('lkey', ctypes.c_uint32),
]


# values for enumeration 'ib_uverbs_wr_opcode'
ib_uverbs_wr_opcode__enumvalues = {
    0: 'IB_UVERBS_WR_RDMA_WRITE',
    1: 'IB_UVERBS_WR_RDMA_WRITE_WITH_IMM',
    2: 'IB_UVERBS_WR_SEND',
    3: 'IB_UVERBS_WR_SEND_WITH_IMM',
    4: 'IB_UVERBS_WR_RDMA_READ',
    5: 'IB_UVERBS_WR_ATOMIC_CMP_AND_SWP',
    6: 'IB_UVERBS_WR_ATOMIC_FETCH_AND_ADD',
    7: 'IB_UVERBS_WR_LOCAL_INV',
    8: 'IB_UVERBS_WR_BIND_MW',
    9: 'IB_UVERBS_WR_SEND_WITH_INV',
    10: 'IB_UVERBS_WR_TSO',
    11: 'IB_UVERBS_WR_RDMA_READ_WITH_INV',
    12: 'IB_UVERBS_WR_MASKED_ATOMIC_CMP_AND_SWP',
    13: 'IB_UVERBS_WR_MASKED_ATOMIC_FETCH_AND_ADD',
    14: 'IB_UVERBS_WR_FLUSH',
    15: 'IB_UVERBS_WR_ATOMIC_WRITE',
}
IB_UVERBS_WR_RDMA_WRITE = 0
IB_UVERBS_WR_RDMA_WRITE_WITH_IMM = 1
IB_UVERBS_WR_SEND = 2
IB_UVERBS_WR_SEND_WITH_IMM = 3
IB_UVERBS_WR_RDMA_READ = 4
IB_UVERBS_WR_ATOMIC_CMP_AND_SWP = 5
IB_UVERBS_WR_ATOMIC_FETCH_AND_ADD = 6
IB_UVERBS_WR_LOCAL_INV = 7
IB_UVERBS_WR_BIND_MW = 8
IB_UVERBS_WR_SEND_WITH_INV = 9
IB_UVERBS_WR_TSO = 10
IB_UVERBS_WR_RDMA_READ_WITH_INV = 11
IB_UVERBS_WR_MASKED_ATOMIC_CMP_AND_SWP = 12
IB_UVERBS_WR_MASKED_ATOMIC_FETCH_AND_ADD = 13
IB_UVERBS_WR_FLUSH = 14
IB_UVERBS_WR_ATOMIC_WRITE = 15
ib_uverbs_wr_opcode = ctypes.c_uint32 # enum
class struct_ib_uverbs_send_wr(Structure):
    pass

class union_ib_uverbs_send_wr_ex(Union):
    pass

union_ib_uverbs_send_wr_ex._pack_ = 1 # source:False
union_ib_uverbs_send_wr_ex._fields_ = [
    ('imm_data', ctypes.c_uint32),
    ('invalidate_rkey', ctypes.c_uint32),
]

class union_ib_uverbs_send_wr_wr(Union):
    pass

class struct_ib_uverbs_send_wr_1_rdma(Structure):
    pass

struct_ib_uverbs_send_wr_1_rdma._pack_ = 1 # source:False
struct_ib_uverbs_send_wr_1_rdma._fields_ = [
    ('remote_addr', ctypes.c_uint64),
    ('rkey', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_send_wr_1_atomic(Structure):
    pass

struct_ib_uverbs_send_wr_1_atomic._pack_ = 1 # source:False
struct_ib_uverbs_send_wr_1_atomic._fields_ = [
    ('remote_addr', ctypes.c_uint64),
    ('compare_add', ctypes.c_uint64),
    ('swap', ctypes.c_uint64),
    ('rkey', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_send_wr_1_ud(Structure):
    pass

struct_ib_uverbs_send_wr_1_ud._pack_ = 1 # source:False
struct_ib_uverbs_send_wr_1_ud._fields_ = [
    ('ah', ctypes.c_uint32),
    ('remote_qpn', ctypes.c_uint32),
    ('remote_qkey', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

union_ib_uverbs_send_wr_wr._pack_ = 1 # source:False
union_ib_uverbs_send_wr_wr._fields_ = [
    ('rdma', struct_ib_uverbs_send_wr_1_rdma),
    ('atomic', struct_ib_uverbs_send_wr_1_atomic),
    ('ud', struct_ib_uverbs_send_wr_1_ud),
    ('PADDING_0', ctypes.c_ubyte * 16),
]

struct_ib_uverbs_send_wr._pack_ = 1 # source:False
struct_ib_uverbs_send_wr._fields_ = [
    ('wr_id', ctypes.c_uint64),
    ('num_sge', ctypes.c_uint32),
    ('opcode', ctypes.c_uint32),
    ('send_flags', ctypes.c_uint32),
    ('ex', union_ib_uverbs_send_wr_ex),
    ('wr', union_ib_uverbs_send_wr_wr),
]

class struct_ib_uverbs_post_send(Structure):
    pass

struct_ib_uverbs_post_send._pack_ = 1 # source:False
struct_ib_uverbs_post_send._fields_ = [
    ('response', ctypes.c_uint64),
    ('qp_handle', ctypes.c_uint32),
    ('wr_count', ctypes.c_uint32),
    ('sge_count', ctypes.c_uint32),
    ('wqe_size', ctypes.c_uint32),
    ('send_wr', struct_ib_uverbs_send_wr * 0),
]

class struct_ib_uverbs_post_send_resp(Structure):
    pass

struct_ib_uverbs_post_send_resp._pack_ = 1 # source:False
struct_ib_uverbs_post_send_resp._fields_ = [
    ('bad_wr', ctypes.c_uint32),
]

class struct_ib_uverbs_recv_wr(Structure):
    pass

struct_ib_uverbs_recv_wr._pack_ = 1 # source:False
struct_ib_uverbs_recv_wr._fields_ = [
    ('wr_id', ctypes.c_uint64),
    ('num_sge', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_post_recv(Structure):
    pass

struct_ib_uverbs_post_recv._pack_ = 1 # source:False
struct_ib_uverbs_post_recv._fields_ = [
    ('response', ctypes.c_uint64),
    ('qp_handle', ctypes.c_uint32),
    ('wr_count', ctypes.c_uint32),
    ('sge_count', ctypes.c_uint32),
    ('wqe_size', ctypes.c_uint32),
    ('recv_wr', struct_ib_uverbs_recv_wr * 0),
]

class struct_ib_uverbs_post_recv_resp(Structure):
    pass

struct_ib_uverbs_post_recv_resp._pack_ = 1 # source:False
struct_ib_uverbs_post_recv_resp._fields_ = [
    ('bad_wr', ctypes.c_uint32),
]

class struct_ib_uverbs_post_srq_recv(Structure):
    pass

struct_ib_uverbs_post_srq_recv._pack_ = 1 # source:False
struct_ib_uverbs_post_srq_recv._fields_ = [
    ('response', ctypes.c_uint64),
    ('srq_handle', ctypes.c_uint32),
    ('wr_count', ctypes.c_uint32),
    ('sge_count', ctypes.c_uint32),
    ('wqe_size', ctypes.c_uint32),
    ('recv', struct_ib_uverbs_recv_wr * 0),
]

class struct_ib_uverbs_post_srq_recv_resp(Structure):
    pass

struct_ib_uverbs_post_srq_recv_resp._pack_ = 1 # source:False
struct_ib_uverbs_post_srq_recv_resp._fields_ = [
    ('bad_wr', ctypes.c_uint32),
]

class struct_ib_uverbs_create_ah(Structure):
    pass

struct_ib_uverbs_create_ah._pack_ = 1 # source:False
struct_ib_uverbs_create_ah._fields_ = [
    ('response', ctypes.c_uint64),
    ('user_handle', ctypes.c_uint64),
    ('pd_handle', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('attr', struct_ib_uverbs_ah_attr),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_create_ah_resp(Structure):
    pass

struct_ib_uverbs_create_ah_resp._pack_ = 1 # source:False
struct_ib_uverbs_create_ah_resp._fields_ = [
    ('ah_handle', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint32 * 0),
]

class struct_ib_uverbs_destroy_ah(Structure):
    pass

struct_ib_uverbs_destroy_ah._pack_ = 1 # source:False
struct_ib_uverbs_destroy_ah._fields_ = [
    ('ah_handle', ctypes.c_uint32),
]

class struct_ib_uverbs_attach_mcast(Structure):
    pass

struct_ib_uverbs_attach_mcast._pack_ = 1 # source:False
struct_ib_uverbs_attach_mcast._fields_ = [
    ('gid', ctypes.c_ubyte * 16),
    ('qp_handle', ctypes.c_uint32),
    ('mlid', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_detach_mcast(Structure):
    pass

struct_ib_uverbs_detach_mcast._pack_ = 1 # source:False
struct_ib_uverbs_detach_mcast._fields_ = [
    ('gid', ctypes.c_ubyte * 16),
    ('qp_handle', ctypes.c_uint32),
    ('mlid', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_flow_spec_hdr(Structure):
    pass

struct_ib_uverbs_flow_spec_hdr._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_hdr._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
    ('flow_spec_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_flow_eth_filter(Structure):
    pass

struct_ib_uverbs_flow_eth_filter._pack_ = 1 # source:False
struct_ib_uverbs_flow_eth_filter._fields_ = [
    ('dst_mac', ctypes.c_ubyte * 6),
    ('src_mac', ctypes.c_ubyte * 6),
    ('ether_type', ctypes.c_uint16),
    ('vlan_tag', ctypes.c_uint16),
]

class struct_ib_uverbs_flow_spec_eth(Structure):
    pass

class union_ib_uverbs_flow_spec_eth_0(Union):
    pass

class struct_ib_uverbs_flow_spec_eth_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_eth_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_eth_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_eth_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_eth_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_eth_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_eth_0_0),
]

struct_ib_uverbs_flow_spec_eth._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_eth._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_eth._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_eth_0),
    ('val', struct_ib_uverbs_flow_eth_filter),
    ('mask', struct_ib_uverbs_flow_eth_filter),
]

class struct_ib_uverbs_flow_ipv4_filter(Structure):
    pass

struct_ib_uverbs_flow_ipv4_filter._pack_ = 1 # source:False
struct_ib_uverbs_flow_ipv4_filter._fields_ = [
    ('src_ip', ctypes.c_uint32),
    ('dst_ip', ctypes.c_uint32),
    ('proto', ctypes.c_ubyte),
    ('tos', ctypes.c_ubyte),
    ('ttl', ctypes.c_ubyte),
    ('flags', ctypes.c_ubyte),
]

class struct_ib_uverbs_flow_spec_ipv4(Structure):
    pass

class union_ib_uverbs_flow_spec_ipv4_0(Union):
    pass

class struct_ib_uverbs_flow_spec_ipv4_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_ipv4_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_ipv4_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_ipv4_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_ipv4_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_ipv4_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_ipv4_0_0),
]

struct_ib_uverbs_flow_spec_ipv4._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_ipv4._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_ipv4._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_ipv4_0),
    ('val', struct_ib_uverbs_flow_ipv4_filter),
    ('mask', struct_ib_uverbs_flow_ipv4_filter),
]

class struct_ib_uverbs_flow_tcp_udp_filter(Structure):
    pass

struct_ib_uverbs_flow_tcp_udp_filter._pack_ = 1 # source:False
struct_ib_uverbs_flow_tcp_udp_filter._fields_ = [
    ('dst_port', ctypes.c_uint16),
    ('src_port', ctypes.c_uint16),
]

class struct_ib_uverbs_flow_spec_tcp_udp(Structure):
    pass

class union_ib_uverbs_flow_spec_tcp_udp_0(Union):
    pass

class struct_ib_uverbs_flow_spec_tcp_udp_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_tcp_udp_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_tcp_udp_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_tcp_udp_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_tcp_udp_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_tcp_udp_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_tcp_udp_0_0),
]

struct_ib_uverbs_flow_spec_tcp_udp._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_tcp_udp._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_tcp_udp._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_tcp_udp_0),
    ('val', struct_ib_uverbs_flow_tcp_udp_filter),
    ('mask', struct_ib_uverbs_flow_tcp_udp_filter),
]

class struct_ib_uverbs_flow_ipv6_filter(Structure):
    pass

struct_ib_uverbs_flow_ipv6_filter._pack_ = 1 # source:False
struct_ib_uverbs_flow_ipv6_filter._fields_ = [
    ('src_ip', ctypes.c_ubyte * 16),
    ('dst_ip', ctypes.c_ubyte * 16),
    ('flow_label', ctypes.c_uint32),
    ('next_hdr', ctypes.c_ubyte),
    ('traffic_class', ctypes.c_ubyte),
    ('hop_limit', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

class struct_ib_uverbs_flow_spec_ipv6(Structure):
    pass

class union_ib_uverbs_flow_spec_ipv6_0(Union):
    pass

class struct_ib_uverbs_flow_spec_ipv6_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_ipv6_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_ipv6_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_ipv6_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_ipv6_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_ipv6_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_ipv6_0_0),
]

struct_ib_uverbs_flow_spec_ipv6._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_ipv6._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_ipv6._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_ipv6_0),
    ('val', struct_ib_uverbs_flow_ipv6_filter),
    ('mask', struct_ib_uverbs_flow_ipv6_filter),
]

class struct_ib_uverbs_flow_spec_action_tag(Structure):
    pass

class union_ib_uverbs_flow_spec_action_tag_0(Union):
    pass

class struct_ib_uverbs_flow_spec_action_tag_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_action_tag_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_action_tag_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_action_tag_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_action_tag_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_action_tag_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_action_tag_0_0),
]

struct_ib_uverbs_flow_spec_action_tag._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_action_tag._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_action_tag._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_action_tag_0),
    ('tag_id', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
]

class struct_ib_uverbs_flow_spec_action_drop(Structure):
    pass

class union_ib_uverbs_flow_spec_action_drop_0(Union):
    pass

class struct_ib_uverbs_flow_spec_action_drop_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_action_drop_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_action_drop_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_action_drop_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_action_drop_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_action_drop_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_action_drop_0_0),
]

struct_ib_uverbs_flow_spec_action_drop._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_action_drop._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_action_drop._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_action_drop_0),
]

class struct_ib_uverbs_flow_spec_action_handle(Structure):
    pass

class union_ib_uverbs_flow_spec_action_handle_0(Union):
    pass

class struct_ib_uverbs_flow_spec_action_handle_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_action_handle_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_action_handle_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_action_handle_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_action_handle_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_action_handle_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_action_handle_0_0),
]

struct_ib_uverbs_flow_spec_action_handle._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_action_handle._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_action_handle._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_action_handle_0),
    ('handle', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
]

class struct_ib_uverbs_flow_spec_action_count(Structure):
    pass

class union_ib_uverbs_flow_spec_action_count_0(Union):
    pass

class struct_ib_uverbs_flow_spec_action_count_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_action_count_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_action_count_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_action_count_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_action_count_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_action_count_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_action_count_0_0),
]

struct_ib_uverbs_flow_spec_action_count._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_action_count._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_action_count._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_action_count_0),
    ('handle', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
]

class struct_ib_uverbs_flow_tunnel_filter(Structure):
    pass

struct_ib_uverbs_flow_tunnel_filter._pack_ = 1 # source:False
struct_ib_uverbs_flow_tunnel_filter._fields_ = [
    ('tunnel_id', ctypes.c_uint32),
]

class struct_ib_uverbs_flow_spec_tunnel(Structure):
    pass

class union_ib_uverbs_flow_spec_tunnel_0(Union):
    pass

class struct_ib_uverbs_flow_spec_tunnel_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_tunnel_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_tunnel_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_tunnel_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_tunnel_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_tunnel_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_tunnel_0_0),
]

struct_ib_uverbs_flow_spec_tunnel._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_tunnel._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_tunnel._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_tunnel_0),
    ('val', struct_ib_uverbs_flow_tunnel_filter),
    ('mask', struct_ib_uverbs_flow_tunnel_filter),
]

class struct_ib_uverbs_flow_spec_esp_filter(Structure):
    pass

struct_ib_uverbs_flow_spec_esp_filter._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_esp_filter._fields_ = [
    ('spi', ctypes.c_uint32),
    ('seq', ctypes.c_uint32),
]

class struct_ib_uverbs_flow_spec_esp(Structure):
    pass

class union_ib_uverbs_flow_spec_esp_0(Union):
    pass

class struct_ib_uverbs_flow_spec_esp_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_esp_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_esp_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_esp_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_esp_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_esp_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_esp_0_0),
]

struct_ib_uverbs_flow_spec_esp._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_esp._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_esp._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_esp_0),
    ('val', struct_ib_uverbs_flow_spec_esp_filter),
    ('mask', struct_ib_uverbs_flow_spec_esp_filter),
]

class struct_ib_uverbs_flow_gre_filter(Structure):
    pass

struct_ib_uverbs_flow_gre_filter._pack_ = 1 # source:False
struct_ib_uverbs_flow_gre_filter._fields_ = [
    ('c_ks_res0_ver', ctypes.c_uint16),
    ('protocol', ctypes.c_uint16),
    ('key', ctypes.c_uint32),
]

class struct_ib_uverbs_flow_spec_gre(Structure):
    pass

class union_ib_uverbs_flow_spec_gre_0(Union):
    pass

class struct_ib_uverbs_flow_spec_gre_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_gre_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_gre_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_gre_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_gre_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_gre_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_gre_0_0),
]

struct_ib_uverbs_flow_spec_gre._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_gre._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_gre._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_gre_0),
    ('val', struct_ib_uverbs_flow_gre_filter),
    ('mask', struct_ib_uverbs_flow_gre_filter),
]

class struct_ib_uverbs_flow_mpls_filter(Structure):
    pass

struct_ib_uverbs_flow_mpls_filter._pack_ = 1 # source:False
struct_ib_uverbs_flow_mpls_filter._fields_ = [
    ('label', ctypes.c_uint32),
]

class struct_ib_uverbs_flow_spec_mpls(Structure):
    pass

class union_ib_uverbs_flow_spec_mpls_0(Union):
    pass

class struct_ib_uverbs_flow_spec_mpls_0_0(Structure):
    pass

struct_ib_uverbs_flow_spec_mpls_0_0._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_mpls_0_0._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

union_ib_uverbs_flow_spec_mpls_0._pack_ = 1 # source:False
union_ib_uverbs_flow_spec_mpls_0._anonymous_ = ('_0',)
union_ib_uverbs_flow_spec_mpls_0._fields_ = [
    ('hdr', struct_ib_uverbs_flow_spec_hdr),
    ('_0', struct_ib_uverbs_flow_spec_mpls_0_0),
]

struct_ib_uverbs_flow_spec_mpls._pack_ = 1 # source:False
struct_ib_uverbs_flow_spec_mpls._anonymous_ = ('_0',)
struct_ib_uverbs_flow_spec_mpls._fields_ = [
    ('_0', union_ib_uverbs_flow_spec_mpls_0),
    ('val', struct_ib_uverbs_flow_mpls_filter),
    ('mask', struct_ib_uverbs_flow_mpls_filter),
]

class struct_ib_uverbs_flow_attr(Structure):
    pass

struct_ib_uverbs_flow_attr._pack_ = 1 # source:False
struct_ib_uverbs_flow_attr._fields_ = [
    ('type', ctypes.c_uint32),
    ('size', ctypes.c_uint16),
    ('priority', ctypes.c_uint16),
    ('num_of_specs', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte * 2),
    ('port', ctypes.c_ubyte),
    ('flags', ctypes.c_uint32),
    ('flow_specs', struct_ib_uverbs_flow_spec_hdr * 0),
]

class struct_ib_uverbs_create_flow(Structure):
    pass

struct_ib_uverbs_create_flow._pack_ = 1 # source:False
struct_ib_uverbs_create_flow._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('qp_handle', ctypes.c_uint32),
    ('flow_attr', struct_ib_uverbs_flow_attr),
]

class struct_ib_uverbs_create_flow_resp(Structure):
    pass

struct_ib_uverbs_create_flow_resp._pack_ = 1 # source:False
struct_ib_uverbs_create_flow_resp._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('flow_handle', ctypes.c_uint32),
]

class struct_ib_uverbs_destroy_flow(Structure):
    pass

struct_ib_uverbs_destroy_flow._pack_ = 1 # source:False
struct_ib_uverbs_destroy_flow._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('flow_handle', ctypes.c_uint32),
]

class struct_ib_uverbs_create_srq(Structure):
    pass

struct_ib_uverbs_create_srq._pack_ = 1 # source:False
struct_ib_uverbs_create_srq._fields_ = [
    ('response', ctypes.c_uint64),
    ('user_handle', ctypes.c_uint64),
    ('pd_handle', ctypes.c_uint32),
    ('max_wr', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('srq_limit', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_create_xsrq(Structure):
    pass

struct_ib_uverbs_create_xsrq._pack_ = 1 # source:False
struct_ib_uverbs_create_xsrq._fields_ = [
    ('response', ctypes.c_uint64),
    ('user_handle', ctypes.c_uint64),
    ('srq_type', ctypes.c_uint32),
    ('pd_handle', ctypes.c_uint32),
    ('max_wr', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('srq_limit', ctypes.c_uint32),
    ('max_num_tags', ctypes.c_uint32),
    ('xrcd_handle', ctypes.c_uint32),
    ('cq_handle', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_create_srq_resp(Structure):
    pass

struct_ib_uverbs_create_srq_resp._pack_ = 1 # source:False
struct_ib_uverbs_create_srq_resp._fields_ = [
    ('srq_handle', ctypes.c_uint32),
    ('max_wr', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('srqn', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint32 * 0),
]

class struct_ib_uverbs_modify_srq(Structure):
    pass

struct_ib_uverbs_modify_srq._pack_ = 1 # source:False
struct_ib_uverbs_modify_srq._fields_ = [
    ('srq_handle', ctypes.c_uint32),
    ('attr_mask', ctypes.c_uint32),
    ('max_wr', ctypes.c_uint32),
    ('srq_limit', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_query_srq(Structure):
    pass

struct_ib_uverbs_query_srq._pack_ = 1 # source:False
struct_ib_uverbs_query_srq._fields_ = [
    ('response', ctypes.c_uint64),
    ('srq_handle', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('driver_data', ctypes.c_uint64 * 0),
]

class struct_ib_uverbs_query_srq_resp(Structure):
    pass

struct_ib_uverbs_query_srq_resp._pack_ = 1 # source:False
struct_ib_uverbs_query_srq_resp._fields_ = [
    ('max_wr', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('srq_limit', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_destroy_srq(Structure):
    pass

struct_ib_uverbs_destroy_srq._pack_ = 1 # source:False
struct_ib_uverbs_destroy_srq._fields_ = [
    ('response', ctypes.c_uint64),
    ('srq_handle', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_destroy_srq_resp(Structure):
    pass

struct_ib_uverbs_destroy_srq_resp._pack_ = 1 # source:False
struct_ib_uverbs_destroy_srq_resp._fields_ = [
    ('events_reported', ctypes.c_uint32),
]

class struct_ib_uverbs_ex_create_wq(Structure):
    pass

struct_ib_uverbs_ex_create_wq._pack_ = 1 # source:False
struct_ib_uverbs_ex_create_wq._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('wq_type', ctypes.c_uint32),
    ('user_handle', ctypes.c_uint64),
    ('pd_handle', ctypes.c_uint32),
    ('cq_handle', ctypes.c_uint32),
    ('max_wr', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('create_flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_ex_create_wq_resp(Structure):
    pass

struct_ib_uverbs_ex_create_wq_resp._pack_ = 1 # source:False
struct_ib_uverbs_ex_create_wq_resp._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('response_length', ctypes.c_uint32),
    ('wq_handle', ctypes.c_uint32),
    ('max_wr', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('wqn', ctypes.c_uint32),
]

class struct_ib_uverbs_ex_destroy_wq(Structure):
    pass

struct_ib_uverbs_ex_destroy_wq._pack_ = 1 # source:False
struct_ib_uverbs_ex_destroy_wq._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('wq_handle', ctypes.c_uint32),
]

class struct_ib_uverbs_ex_destroy_wq_resp(Structure):
    pass

struct_ib_uverbs_ex_destroy_wq_resp._pack_ = 1 # source:False
struct_ib_uverbs_ex_destroy_wq_resp._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('response_length', ctypes.c_uint32),
    ('events_reported', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
]

class struct_ib_uverbs_ex_modify_wq(Structure):
    pass

struct_ib_uverbs_ex_modify_wq._pack_ = 1 # source:False
struct_ib_uverbs_ex_modify_wq._fields_ = [
    ('attr_mask', ctypes.c_uint32),
    ('wq_handle', ctypes.c_uint32),
    ('wq_state', ctypes.c_uint32),
    ('curr_wq_state', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('flags_mask', ctypes.c_uint32),
]

class struct_ib_uverbs_ex_create_rwq_ind_table(Structure):
    pass

struct_ib_uverbs_ex_create_rwq_ind_table._pack_ = 1 # source:False
struct_ib_uverbs_ex_create_rwq_ind_table._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('log_ind_tbl_size', ctypes.c_uint32),
    ('wq_handles', ctypes.c_uint32 * 0),
]

class struct_ib_uverbs_ex_create_rwq_ind_table_resp(Structure):
    pass

struct_ib_uverbs_ex_create_rwq_ind_table_resp._pack_ = 1 # source:False
struct_ib_uverbs_ex_create_rwq_ind_table_resp._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('response_length', ctypes.c_uint32),
    ('ind_tbl_handle', ctypes.c_uint32),
    ('ind_tbl_num', ctypes.c_uint32),
]

class struct_ib_uverbs_ex_destroy_rwq_ind_table(Structure):
    pass

struct_ib_uverbs_ex_destroy_rwq_ind_table._pack_ = 1 # source:False
struct_ib_uverbs_ex_destroy_rwq_ind_table._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('ind_tbl_handle', ctypes.c_uint32),
]

class struct_ib_uverbs_cq_moderation(Structure):
    pass

struct_ib_uverbs_cq_moderation._pack_ = 1 # source:False
struct_ib_uverbs_cq_moderation._fields_ = [
    ('cq_count', ctypes.c_uint16),
    ('cq_period', ctypes.c_uint16),
]

class struct_ib_uverbs_ex_modify_cq(Structure):
    pass

struct_ib_uverbs_ex_modify_cq._pack_ = 1 # source:False
struct_ib_uverbs_ex_modify_cq._fields_ = [
    ('cq_handle', ctypes.c_uint32),
    ('attr_mask', ctypes.c_uint32),
    ('attr', struct_ib_uverbs_cq_moderation),
    ('reserved', ctypes.c_uint32),
]


# values for enumeration 'ib_uverbs_device_cap_flags'
ib_uverbs_device_cap_flags__enumvalues = {
    1: 'IB_UVERBS_DEVICE_RESIZE_MAX_WR',
    2: 'IB_UVERBS_DEVICE_BAD_PKEY_CNTR',
    4: 'IB_UVERBS_DEVICE_BAD_QKEY_CNTR',
    8: 'IB_UVERBS_DEVICE_RAW_MULTI',
    16: 'IB_UVERBS_DEVICE_AUTO_PATH_MIG',
    32: 'IB_UVERBS_DEVICE_CHANGE_PHY_PORT',
    64: 'IB_UVERBS_DEVICE_UD_AV_PORT_ENFORCE',
    128: 'IB_UVERBS_DEVICE_CURR_QP_STATE_MOD',
    256: 'IB_UVERBS_DEVICE_SHUTDOWN_PORT',
    1024: 'IB_UVERBS_DEVICE_PORT_ACTIVE_EVENT',
    2048: 'IB_UVERBS_DEVICE_SYS_IMAGE_GUID',
    4096: 'IB_UVERBS_DEVICE_RC_RNR_NAK_GEN',
    8192: 'IB_UVERBS_DEVICE_SRQ_RESIZE',
    16384: 'IB_UVERBS_DEVICE_N_NOTIFY_CQ',
    131072: 'IB_UVERBS_DEVICE_MEM_WINDOW',
    262144: 'IB_UVERBS_DEVICE_UD_IP_CSUM',
    1048576: 'IB_UVERBS_DEVICE_XRC',
    2097152: 'IB_UVERBS_DEVICE_MEM_MGT_EXTENSIONS',
    8388608: 'IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2A',
    16777216: 'IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2B',
    33554432: 'IB_UVERBS_DEVICE_RC_IP_CSUM',
    67108864: 'IB_UVERBS_DEVICE_RAW_IP_CSUM',
    536870912: 'IB_UVERBS_DEVICE_MANAGED_FLOW_STEERING',
    17179869184: 'IB_UVERBS_DEVICE_RAW_SCATTER_FCS',
    68719476736: 'IB_UVERBS_DEVICE_PCI_WRITE_END_PADDING',
    274877906944: 'IB_UVERBS_DEVICE_FLUSH_GLOBAL',
    549755813888: 'IB_UVERBS_DEVICE_FLUSH_PERSISTENT',
    1099511627776: 'IB_UVERBS_DEVICE_ATOMIC_WRITE',
}
IB_UVERBS_DEVICE_RESIZE_MAX_WR = 1
IB_UVERBS_DEVICE_BAD_PKEY_CNTR = 2
IB_UVERBS_DEVICE_BAD_QKEY_CNTR = 4
IB_UVERBS_DEVICE_RAW_MULTI = 8
IB_UVERBS_DEVICE_AUTO_PATH_MIG = 16
IB_UVERBS_DEVICE_CHANGE_PHY_PORT = 32
IB_UVERBS_DEVICE_UD_AV_PORT_ENFORCE = 64
IB_UVERBS_DEVICE_CURR_QP_STATE_MOD = 128
IB_UVERBS_DEVICE_SHUTDOWN_PORT = 256
IB_UVERBS_DEVICE_PORT_ACTIVE_EVENT = 1024
IB_UVERBS_DEVICE_SYS_IMAGE_GUID = 2048
IB_UVERBS_DEVICE_RC_RNR_NAK_GEN = 4096
IB_UVERBS_DEVICE_SRQ_RESIZE = 8192
IB_UVERBS_DEVICE_N_NOTIFY_CQ = 16384
IB_UVERBS_DEVICE_MEM_WINDOW = 131072
IB_UVERBS_DEVICE_UD_IP_CSUM = 262144
IB_UVERBS_DEVICE_XRC = 1048576
IB_UVERBS_DEVICE_MEM_MGT_EXTENSIONS = 2097152
IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2A = 8388608
IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2B = 16777216
IB_UVERBS_DEVICE_RC_IP_CSUM = 33554432
IB_UVERBS_DEVICE_RAW_IP_CSUM = 67108864
IB_UVERBS_DEVICE_MANAGED_FLOW_STEERING = 536870912
IB_UVERBS_DEVICE_RAW_SCATTER_FCS = 17179869184
IB_UVERBS_DEVICE_PCI_WRITE_END_PADDING = 68719476736
IB_UVERBS_DEVICE_FLUSH_GLOBAL = 274877906944
IB_UVERBS_DEVICE_FLUSH_PERSISTENT = 549755813888
IB_UVERBS_DEVICE_ATOMIC_WRITE = 1099511627776
ib_uverbs_device_cap_flags = ctypes.c_uint64 # enum

# values for enumeration 'ib_uverbs_raw_packet_caps'
ib_uverbs_raw_packet_caps__enumvalues = {
    1: 'IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING',
    2: 'IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS',
    4: 'IB_UVERBS_RAW_PACKET_CAP_IP_CSUM',
    8: 'IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP',
}
IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING = 1
IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS = 2
IB_UVERBS_RAW_PACKET_CAP_IP_CSUM = 4
IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP = 8
ib_uverbs_raw_packet_caps = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_core_support'
ib_uverbs_core_support__enumvalues = {
    1: 'IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS',
}
IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS = 1
ib_uverbs_core_support = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_access_flags'
ib_uverbs_access_flags__enumvalues = {
    1: 'IB_UVERBS_ACCESS_LOCAL_WRITE',
    2: 'IB_UVERBS_ACCESS_REMOTE_WRITE',
    4: 'IB_UVERBS_ACCESS_REMOTE_READ',
    8: 'IB_UVERBS_ACCESS_REMOTE_ATOMIC',
    16: 'IB_UVERBS_ACCESS_MW_BIND',
    32: 'IB_UVERBS_ACCESS_ZERO_BASED',
    64: 'IB_UVERBS_ACCESS_ON_DEMAND',
    128: 'IB_UVERBS_ACCESS_HUGETLB',
    256: 'IB_UVERBS_ACCESS_FLUSH_GLOBAL',
    512: 'IB_UVERBS_ACCESS_FLUSH_PERSISTENT',
    1048576: 'IB_UVERBS_ACCESS_RELAXED_ORDERING',
    1072693248: 'IB_UVERBS_ACCESS_OPTIONAL_RANGE',
}
IB_UVERBS_ACCESS_LOCAL_WRITE = 1
IB_UVERBS_ACCESS_REMOTE_WRITE = 2
IB_UVERBS_ACCESS_REMOTE_READ = 4
IB_UVERBS_ACCESS_REMOTE_ATOMIC = 8
IB_UVERBS_ACCESS_MW_BIND = 16
IB_UVERBS_ACCESS_ZERO_BASED = 32
IB_UVERBS_ACCESS_ON_DEMAND = 64
IB_UVERBS_ACCESS_HUGETLB = 128
IB_UVERBS_ACCESS_FLUSH_GLOBAL = 256
IB_UVERBS_ACCESS_FLUSH_PERSISTENT = 512
IB_UVERBS_ACCESS_RELAXED_ORDERING = 1048576
IB_UVERBS_ACCESS_OPTIONAL_RANGE = 1072693248
ib_uverbs_access_flags = ctypes.c_uint32 # enum
IBV_ACCESS_OPTIONAL_RANGE = IB_UVERBS_ACCESS_OPTIONAL_RANGE # macro

# values for enumeration 'ib_uverbs_srq_type'
ib_uverbs_srq_type__enumvalues = {
    0: 'IB_UVERBS_SRQT_BASIC',
    1: 'IB_UVERBS_SRQT_XRC',
    2: 'IB_UVERBS_SRQT_TM',
}
IB_UVERBS_SRQT_BASIC = 0
IB_UVERBS_SRQT_XRC = 1
IB_UVERBS_SRQT_TM = 2
ib_uverbs_srq_type = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_wq_type'
ib_uverbs_wq_type__enumvalues = {
    0: 'IB_UVERBS_WQT_RQ',
}
IB_UVERBS_WQT_RQ = 0
ib_uverbs_wq_type = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_wq_flags'
ib_uverbs_wq_flags__enumvalues = {
    1: 'IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING',
    2: 'IB_UVERBS_WQ_FLAGS_SCATTER_FCS',
    4: 'IB_UVERBS_WQ_FLAGS_DELAY_DROP',
    8: 'IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING',
}
IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING = 1
IB_UVERBS_WQ_FLAGS_SCATTER_FCS = 2
IB_UVERBS_WQ_FLAGS_DELAY_DROP = 4
IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING = 8
ib_uverbs_wq_flags = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_qp_type'
ib_uverbs_qp_type__enumvalues = {
    2: 'IB_UVERBS_QPT_RC',
    3: 'IB_UVERBS_QPT_UC',
    4: 'IB_UVERBS_QPT_UD',
    8: 'IB_UVERBS_QPT_RAW_PACKET',
    9: 'IB_UVERBS_QPT_XRC_INI',
    10: 'IB_UVERBS_QPT_XRC_TGT',
    255: 'IB_UVERBS_QPT_DRIVER',
}
IB_UVERBS_QPT_RC = 2
IB_UVERBS_QPT_UC = 3
IB_UVERBS_QPT_UD = 4
IB_UVERBS_QPT_RAW_PACKET = 8
IB_UVERBS_QPT_XRC_INI = 9
IB_UVERBS_QPT_XRC_TGT = 10
IB_UVERBS_QPT_DRIVER = 255
ib_uverbs_qp_type = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_qp_create_flags'
ib_uverbs_qp_create_flags__enumvalues = {
    2: 'IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK',
    256: 'IB_UVERBS_QP_CREATE_SCATTER_FCS',
    512: 'IB_UVERBS_QP_CREATE_CVLAN_STRIPPING',
    2048: 'IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING',
    4096: 'IB_UVERBS_QP_CREATE_SQ_SIG_ALL',
}
IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK = 2
IB_UVERBS_QP_CREATE_SCATTER_FCS = 256
IB_UVERBS_QP_CREATE_CVLAN_STRIPPING = 512
IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING = 2048
IB_UVERBS_QP_CREATE_SQ_SIG_ALL = 4096
ib_uverbs_qp_create_flags = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_query_port_cap_flags'
ib_uverbs_query_port_cap_flags__enumvalues = {
    2: 'IB_UVERBS_PCF_SM',
    4: 'IB_UVERBS_PCF_NOTICE_SUP',
    8: 'IB_UVERBS_PCF_TRAP_SUP',
    16: 'IB_UVERBS_PCF_OPT_IPD_SUP',
    32: 'IB_UVERBS_PCF_AUTO_MIGR_SUP',
    64: 'IB_UVERBS_PCF_SL_MAP_SUP',
    128: 'IB_UVERBS_PCF_MKEY_NVRAM',
    256: 'IB_UVERBS_PCF_PKEY_NVRAM',
    512: 'IB_UVERBS_PCF_LED_INFO_SUP',
    1024: 'IB_UVERBS_PCF_SM_DISABLED',
    2048: 'IB_UVERBS_PCF_SYS_IMAGE_GUID_SUP',
    4096: 'IB_UVERBS_PCF_PKEY_SW_EXT_PORT_TRAP_SUP',
    16384: 'IB_UVERBS_PCF_EXTENDED_SPEEDS_SUP',
    65536: 'IB_UVERBS_PCF_CM_SUP',
    131072: 'IB_UVERBS_PCF_SNMP_TUNNEL_SUP',
    262144: 'IB_UVERBS_PCF_REINIT_SUP',
    524288: 'IB_UVERBS_PCF_DEVICE_MGMT_SUP',
    1048576: 'IB_UVERBS_PCF_VENDOR_CLASS_SUP',
    2097152: 'IB_UVERBS_PCF_DR_NOTICE_SUP',
    4194304: 'IB_UVERBS_PCF_CAP_MASK_NOTICE_SUP',
    8388608: 'IB_UVERBS_PCF_BOOT_MGMT_SUP',
    16777216: 'IB_UVERBS_PCF_LINK_LATENCY_SUP',
    33554432: 'IB_UVERBS_PCF_CLIENT_REG_SUP',
    134217728: 'IB_UVERBS_PCF_LINK_SPEED_WIDTH_TABLE_SUP',
    268435456: 'IB_UVERBS_PCF_VENDOR_SPECIFIC_MADS_TABLE_SUP',
    536870912: 'IB_UVERBS_PCF_MCAST_PKEY_TRAP_SUPPRESSION_SUP',
    1073741824: 'IB_UVERBS_PCF_MCAST_FDB_TOP_SUP',
    2147483648: 'IB_UVERBS_PCF_HIERARCHY_INFO_SUP',
    67108864: 'IB_UVERBS_PCF_IP_BASED_GIDS',
}
IB_UVERBS_PCF_SM = 2
IB_UVERBS_PCF_NOTICE_SUP = 4
IB_UVERBS_PCF_TRAP_SUP = 8
IB_UVERBS_PCF_OPT_IPD_SUP = 16
IB_UVERBS_PCF_AUTO_MIGR_SUP = 32
IB_UVERBS_PCF_SL_MAP_SUP = 64
IB_UVERBS_PCF_MKEY_NVRAM = 128
IB_UVERBS_PCF_PKEY_NVRAM = 256
IB_UVERBS_PCF_LED_INFO_SUP = 512
IB_UVERBS_PCF_SM_DISABLED = 1024
IB_UVERBS_PCF_SYS_IMAGE_GUID_SUP = 2048
IB_UVERBS_PCF_PKEY_SW_EXT_PORT_TRAP_SUP = 4096
IB_UVERBS_PCF_EXTENDED_SPEEDS_SUP = 16384
IB_UVERBS_PCF_CM_SUP = 65536
IB_UVERBS_PCF_SNMP_TUNNEL_SUP = 131072
IB_UVERBS_PCF_REINIT_SUP = 262144
IB_UVERBS_PCF_DEVICE_MGMT_SUP = 524288
IB_UVERBS_PCF_VENDOR_CLASS_SUP = 1048576
IB_UVERBS_PCF_DR_NOTICE_SUP = 2097152
IB_UVERBS_PCF_CAP_MASK_NOTICE_SUP = 4194304
IB_UVERBS_PCF_BOOT_MGMT_SUP = 8388608
IB_UVERBS_PCF_LINK_LATENCY_SUP = 16777216
IB_UVERBS_PCF_CLIENT_REG_SUP = 33554432
IB_UVERBS_PCF_LINK_SPEED_WIDTH_TABLE_SUP = 134217728
IB_UVERBS_PCF_VENDOR_SPECIFIC_MADS_TABLE_SUP = 268435456
IB_UVERBS_PCF_MCAST_PKEY_TRAP_SUPPRESSION_SUP = 536870912
IB_UVERBS_PCF_MCAST_FDB_TOP_SUP = 1073741824
IB_UVERBS_PCF_HIERARCHY_INFO_SUP = 2147483648
IB_UVERBS_PCF_IP_BASED_GIDS = 67108864
ib_uverbs_query_port_cap_flags = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_query_port_flags'
ib_uverbs_query_port_flags__enumvalues = {
    1: 'IB_UVERBS_QPF_GRH_REQUIRED',
}
IB_UVERBS_QPF_GRH_REQUIRED = 1
ib_uverbs_query_port_flags = ctypes.c_uint32 # enum
IBV_QPF_GRH_REQUIRED = IB_UVERBS_QPF_GRH_REQUIRED # macro

# values for enumeration 'ib_uverbs_flow_action_esp_keymat'
ib_uverbs_flow_action_esp_keymat__enumvalues = {
    0: 'IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM',
}
IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM = 0
ib_uverbs_flow_action_esp_keymat = ctypes.c_uint32 # enum
ibv_flow_action_esp_keymat = ib_uverbs_flow_action_esp_keymat # macro
IBV_FLOW_ACTION_ESP_KEYMAT_AES_GCM = IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM # macro

# values for enumeration 'ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo'
ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo__enumvalues = {
    0: 'IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ',
}
IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ = 0
ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo = ctypes.c_uint32 # enum
ibv_flow_action_esp_keymat_aes_gcm_iv_algo = ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo # macro
IBV_FLOW_ACTION_IV_ALGO_SEQ = IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ # macro
class struct_ib_uverbs_flow_action_esp_keymat_aes_gcm(Structure):
    pass

struct_ib_uverbs_flow_action_esp_keymat_aes_gcm._pack_ = 1 # source:False
struct_ib_uverbs_flow_action_esp_keymat_aes_gcm._fields_ = [
    ('iv', ctypes.c_uint64),
    ('iv_algo', ctypes.c_uint32),
    ('salt', ctypes.c_uint32),
    ('icv_len', ctypes.c_uint32),
    ('key_len', ctypes.c_uint32),
    ('aes_key', ctypes.c_uint32 * 8),
]


# values for enumeration 'ib_uverbs_flow_action_esp_replay'
ib_uverbs_flow_action_esp_replay__enumvalues = {
    0: 'IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE',
    1: 'IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP',
}
IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE = 0
IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP = 1
ib_uverbs_flow_action_esp_replay = ctypes.c_uint32 # enum
ibv_flow_action_esp_replay = ib_uverbs_flow_action_esp_replay # macro
IBV_FLOW_ACTION_ESP_REPLAY_NONE = IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE # macro
IBV_FLOW_ACTION_ESP_REPLAY_BMP = IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP # macro
class struct_ib_uverbs_flow_action_esp_replay_bmp(Structure):
    pass

struct_ib_uverbs_flow_action_esp_replay_bmp._pack_ = 1 # source:False
struct_ib_uverbs_flow_action_esp_replay_bmp._fields_ = [
    ('size', ctypes.c_uint32),
]


# values for enumeration 'ib_uverbs_flow_action_esp_flags'
ib_uverbs_flow_action_esp_flags__enumvalues = {
    0: 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO',
    1: 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD',
    0: 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL',
    2: 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT',
    0: 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT',
    4: 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT',
    8: 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW',
}
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO = 0
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD = 1
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL = 0
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT = 2
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT = 0
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT = 4
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW = 8
ib_uverbs_flow_action_esp_flags = ctypes.c_uint32 # enum
ibv_flow_action_esp_flags = ib_uverbs_flow_action_esp_flags # macro
IBV_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO # macro
IBV_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD # macro
IBV_FLOW_ACTION_ESP_FLAGS_TUNNEL = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL # macro
IBV_FLOW_ACTION_ESP_FLAGS_TRANSPORT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT # macro
IBV_FLOW_ACTION_ESP_FLAGS_DECRYPT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT # macro
IBV_FLOW_ACTION_ESP_FLAGS_ENCRYPT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT # macro
IBV_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW # macro
class struct_ib_uverbs_flow_action_esp_encap(Structure):
    pass

class union_ib_uverbs_flow_action_esp_encap_0(Union):
    pass

union_ib_uverbs_flow_action_esp_encap_0._pack_ = 1 # source:False
union_ib_uverbs_flow_action_esp_encap_0._fields_ = [
    ('val_ptr', ctypes.POINTER(None)),
    ('val_ptr_data_u64', ctypes.c_uint64),
]

class union_ib_uverbs_flow_action_esp_encap_1(Union):
    pass

union_ib_uverbs_flow_action_esp_encap_1._pack_ = 1 # source:False
union_ib_uverbs_flow_action_esp_encap_1._fields_ = [
    ('next_ptr', ctypes.POINTER(struct_ib_uverbs_flow_action_esp_encap)),
    ('next_ptr_data_u64', ctypes.c_uint64),
]

struct_ib_uverbs_flow_action_esp_encap._pack_ = 1 # source:False
struct_ib_uverbs_flow_action_esp_encap._anonymous_ = ('_0', '_1',)
struct_ib_uverbs_flow_action_esp_encap._fields_ = [
    ('_0', union_ib_uverbs_flow_action_esp_encap_0),
    ('_1', union_ib_uverbs_flow_action_esp_encap_1),
    ('len', ctypes.c_uint16),
    ('type', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_ib_uverbs_flow_action_esp(Structure):
    pass

struct_ib_uverbs_flow_action_esp._pack_ = 1 # source:False
struct_ib_uverbs_flow_action_esp._fields_ = [
    ('spi', ctypes.c_uint32),
    ('seq', ctypes.c_uint32),
    ('tfc_pad', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('hard_limit_pkts', ctypes.c_uint64),
]


# values for enumeration 'ib_uverbs_read_counters_flags'
ib_uverbs_read_counters_flags__enumvalues = {
    1: 'IB_UVERBS_READ_COUNTERS_PREFER_CACHED',
}
IB_UVERBS_READ_COUNTERS_PREFER_CACHED = 1
ib_uverbs_read_counters_flags = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_advise_mr_advice'
ib_uverbs_advise_mr_advice__enumvalues = {
    0: 'IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH',
    1: 'IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE',
    2: 'IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT',
}
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH = 0
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE = 1
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT = 2
ib_uverbs_advise_mr_advice = ctypes.c_uint32 # enum
ibv_advise_mr_advice = ib_uverbs_advise_mr_advice # macro
IBV_ADVISE_MR_ADVICE_PREFETCH = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH # macro
IBV_ADVISE_MR_ADVICE_PREFETCH_WRITE = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE # macro
IBV_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT # macro

# values for enumeration 'ib_uverbs_advise_mr_flag'
ib_uverbs_advise_mr_flag__enumvalues = {
    1: 'IB_UVERBS_ADVISE_MR_FLAG_FLUSH',
}
IB_UVERBS_ADVISE_MR_FLAG_FLUSH = 1
ib_uverbs_advise_mr_flag = ctypes.c_uint32 # enum
IBV_ADVISE_MR_FLAG_FLUSH = IB_UVERBS_ADVISE_MR_FLAG_FLUSH # macro
class struct_ib_uverbs_query_port_resp_ex(Structure):
    pass

struct_ib_uverbs_query_port_resp_ex._pack_ = 1 # source:False
struct_ib_uverbs_query_port_resp_ex._fields_ = [
    ('legacy_resp', struct_ib_uverbs_query_port_resp),
    ('port_cap_flags2', ctypes.c_uint16),
    ('reserved', ctypes.c_ubyte * 2),
    ('active_speed_ex', ctypes.c_uint32),
]

class struct_ib_uverbs_qp_cap(Structure):
    pass

struct_ib_uverbs_qp_cap._pack_ = 1 # source:False
struct_ib_uverbs_qp_cap._fields_ = [
    ('max_send_wr', ctypes.c_uint32),
    ('max_recv_wr', ctypes.c_uint32),
    ('max_send_sge', ctypes.c_uint32),
    ('max_recv_sge', ctypes.c_uint32),
    ('max_inline_data', ctypes.c_uint32),
]


# values for enumeration 'rdma_driver_id'
rdma_driver_id__enumvalues = {
    0: 'RDMA_DRIVER_UNKNOWN',
    1: 'RDMA_DRIVER_MLX5',
    2: 'RDMA_DRIVER_MLX4',
    3: 'RDMA_DRIVER_CXGB3',
    4: 'RDMA_DRIVER_CXGB4',
    5: 'RDMA_DRIVER_MTHCA',
    6: 'RDMA_DRIVER_BNXT_RE',
    7: 'RDMA_DRIVER_OCRDMA',
    8: 'RDMA_DRIVER_NES',
    9: 'RDMA_DRIVER_I40IW',
    9: 'RDMA_DRIVER_IRDMA',
    10: 'RDMA_DRIVER_VMW_PVRDMA',
    11: 'RDMA_DRIVER_QEDR',
    12: 'RDMA_DRIVER_HNS',
    13: 'RDMA_DRIVER_USNIC',
    14: 'RDMA_DRIVER_RXE',
    15: 'RDMA_DRIVER_HFI1',
    16: 'RDMA_DRIVER_QIB',
    17: 'RDMA_DRIVER_EFA',
    18: 'RDMA_DRIVER_SIW',
    19: 'RDMA_DRIVER_ERDMA',
    20: 'RDMA_DRIVER_MANA',
}
RDMA_DRIVER_UNKNOWN = 0
RDMA_DRIVER_MLX5 = 1
RDMA_DRIVER_MLX4 = 2
RDMA_DRIVER_CXGB3 = 3
RDMA_DRIVER_CXGB4 = 4
RDMA_DRIVER_MTHCA = 5
RDMA_DRIVER_BNXT_RE = 6
RDMA_DRIVER_OCRDMA = 7
RDMA_DRIVER_NES = 8
RDMA_DRIVER_I40IW = 9
RDMA_DRIVER_IRDMA = 9
RDMA_DRIVER_VMW_PVRDMA = 10
RDMA_DRIVER_QEDR = 11
RDMA_DRIVER_HNS = 12
RDMA_DRIVER_USNIC = 13
RDMA_DRIVER_RXE = 14
RDMA_DRIVER_HFI1 = 15
RDMA_DRIVER_QIB = 16
RDMA_DRIVER_EFA = 17
RDMA_DRIVER_SIW = 18
RDMA_DRIVER_ERDMA = 19
RDMA_DRIVER_MANA = 20
rdma_driver_id = ctypes.c_uint32 # enum

# values for enumeration 'ib_uverbs_gid_type'
ib_uverbs_gid_type__enumvalues = {
    0: 'IB_UVERBS_GID_TYPE_IB',
    1: 'IB_UVERBS_GID_TYPE_ROCE_V1',
    2: 'IB_UVERBS_GID_TYPE_ROCE_V2',
}
IB_UVERBS_GID_TYPE_IB = 0
IB_UVERBS_GID_TYPE_ROCE_V1 = 1
IB_UVERBS_GID_TYPE_ROCE_V2 = 2
ib_uverbs_gid_type = ctypes.c_uint32 # enum
class struct_ib_uverbs_gid_entry(Structure):
    pass

struct_ib_uverbs_gid_entry._pack_ = 1 # source:False
struct_ib_uverbs_gid_entry._fields_ = [
    ('gid', ctypes.c_uint64 * 2),
    ('gid_index', ctypes.c_uint32),
    ('port_num', ctypes.c_uint32),
    ('gid_type', ctypes.c_uint32),
    ('netdev_ifindex', ctypes.c_uint32),
]

class union_ibv_gid(Union):
    pass

class struct_ibv_gid_global(Structure):
    pass

struct_ibv_gid_global._pack_ = 1 # source:False
struct_ibv_gid_global._fields_ = [
    ('subnet_prefix', ctypes.c_uint64),
    ('interface_id', ctypes.c_uint64),
]

union_ibv_gid._pack_ = 1 # source:False
union_ibv_gid._fields_ = [
    ('raw', ctypes.c_ubyte * 16),
    ('global', struct_ibv_gid_global),
]


# values for enumeration 'ibv_gid_type'
ibv_gid_type__enumvalues = {
    0: 'IBV_GID_TYPE_IB',
    1: 'IBV_GID_TYPE_ROCE_V1',
    2: 'IBV_GID_TYPE_ROCE_V2',
}
IBV_GID_TYPE_IB = 0
IBV_GID_TYPE_ROCE_V1 = 1
IBV_GID_TYPE_ROCE_V2 = 2
ibv_gid_type = ctypes.c_uint32 # enum
class struct_ibv_gid_entry(Structure):
    pass

struct_ibv_gid_entry._pack_ = 1 # source:False
struct_ibv_gid_entry._fields_ = [
    ('gid', union_ibv_gid),
    ('gid_index', ctypes.c_uint32),
    ('port_num', ctypes.c_uint32),
    ('gid_type', ctypes.c_uint32),
    ('ndev_ifindex', ctypes.c_uint32),
]


# values for enumeration 'ibv_node_type'
ibv_node_type__enumvalues = {
    -1: 'IBV_NODE_UNKNOWN',
    1: 'IBV_NODE_CA',
    2: 'IBV_NODE_SWITCH',
    3: 'IBV_NODE_ROUTER',
    4: 'IBV_NODE_RNIC',
    5: 'IBV_NODE_USNIC',
    6: 'IBV_NODE_USNIC_UDP',
    7: 'IBV_NODE_UNSPECIFIED',
}
IBV_NODE_UNKNOWN = -1
IBV_NODE_CA = 1
IBV_NODE_SWITCH = 2
IBV_NODE_ROUTER = 3
IBV_NODE_RNIC = 4
IBV_NODE_USNIC = 5
IBV_NODE_USNIC_UDP = 6
IBV_NODE_UNSPECIFIED = 7
ibv_node_type = ctypes.c_int32 # enum

# values for enumeration 'ibv_transport_type'
ibv_transport_type__enumvalues = {
    -1: 'IBV_TRANSPORT_UNKNOWN',
    0: 'IBV_TRANSPORT_IB',
    1: 'IBV_TRANSPORT_IWARP',
    2: 'IBV_TRANSPORT_USNIC',
    3: 'IBV_TRANSPORT_USNIC_UDP',
    4: 'IBV_TRANSPORT_UNSPECIFIED',
}
IBV_TRANSPORT_UNKNOWN = -1
IBV_TRANSPORT_IB = 0
IBV_TRANSPORT_IWARP = 1
IBV_TRANSPORT_USNIC = 2
IBV_TRANSPORT_USNIC_UDP = 3
IBV_TRANSPORT_UNSPECIFIED = 4
ibv_transport_type = ctypes.c_int32 # enum

# values for enumeration 'ibv_device_cap_flags'
ibv_device_cap_flags__enumvalues = {
    1: 'IBV_DEVICE_RESIZE_MAX_WR',
    2: 'IBV_DEVICE_BAD_PKEY_CNTR',
    4: 'IBV_DEVICE_BAD_QKEY_CNTR',
    8: 'IBV_DEVICE_RAW_MULTI',
    16: 'IBV_DEVICE_AUTO_PATH_MIG',
    32: 'IBV_DEVICE_CHANGE_PHY_PORT',
    64: 'IBV_DEVICE_UD_AV_PORT_ENFORCE',
    128: 'IBV_DEVICE_CURR_QP_STATE_MOD',
    256: 'IBV_DEVICE_SHUTDOWN_PORT',
    512: 'IBV_DEVICE_INIT_TYPE',
    1024: 'IBV_DEVICE_PORT_ACTIVE_EVENT',
    2048: 'IBV_DEVICE_SYS_IMAGE_GUID',
    4096: 'IBV_DEVICE_RC_RNR_NAK_GEN',
    8192: 'IBV_DEVICE_SRQ_RESIZE',
    16384: 'IBV_DEVICE_N_NOTIFY_CQ',
    131072: 'IBV_DEVICE_MEM_WINDOW',
    262144: 'IBV_DEVICE_UD_IP_CSUM',
    1048576: 'IBV_DEVICE_XRC',
    2097152: 'IBV_DEVICE_MEM_MGT_EXTENSIONS',
    8388608: 'IBV_DEVICE_MEM_WINDOW_TYPE_2A',
    16777216: 'IBV_DEVICE_MEM_WINDOW_TYPE_2B',
    33554432: 'IBV_DEVICE_RC_IP_CSUM',
    67108864: 'IBV_DEVICE_RAW_IP_CSUM',
    536870912: 'IBV_DEVICE_MANAGED_FLOW_STEERING',
}
IBV_DEVICE_RESIZE_MAX_WR = 1
IBV_DEVICE_BAD_PKEY_CNTR = 2
IBV_DEVICE_BAD_QKEY_CNTR = 4
IBV_DEVICE_RAW_MULTI = 8
IBV_DEVICE_AUTO_PATH_MIG = 16
IBV_DEVICE_CHANGE_PHY_PORT = 32
IBV_DEVICE_UD_AV_PORT_ENFORCE = 64
IBV_DEVICE_CURR_QP_STATE_MOD = 128
IBV_DEVICE_SHUTDOWN_PORT = 256
IBV_DEVICE_INIT_TYPE = 512
IBV_DEVICE_PORT_ACTIVE_EVENT = 1024
IBV_DEVICE_SYS_IMAGE_GUID = 2048
IBV_DEVICE_RC_RNR_NAK_GEN = 4096
IBV_DEVICE_SRQ_RESIZE = 8192
IBV_DEVICE_N_NOTIFY_CQ = 16384
IBV_DEVICE_MEM_WINDOW = 131072
IBV_DEVICE_UD_IP_CSUM = 262144
IBV_DEVICE_XRC = 1048576
IBV_DEVICE_MEM_MGT_EXTENSIONS = 2097152
IBV_DEVICE_MEM_WINDOW_TYPE_2A = 8388608
IBV_DEVICE_MEM_WINDOW_TYPE_2B = 16777216
IBV_DEVICE_RC_IP_CSUM = 33554432
IBV_DEVICE_RAW_IP_CSUM = 67108864
IBV_DEVICE_MANAGED_FLOW_STEERING = 536870912
ibv_device_cap_flags = ctypes.c_uint32 # enum

# values for enumeration 'ibv_fork_status'
ibv_fork_status__enumvalues = {
    0: 'IBV_FORK_DISABLED',
    1: 'IBV_FORK_ENABLED',
    2: 'IBV_FORK_UNNEEDED',
}
IBV_FORK_DISABLED = 0
IBV_FORK_ENABLED = 1
IBV_FORK_UNNEEDED = 2
ibv_fork_status = ctypes.c_uint32 # enum

# values for enumeration 'ibv_atomic_cap'
ibv_atomic_cap__enumvalues = {
    0: 'IBV_ATOMIC_NONE',
    1: 'IBV_ATOMIC_HCA',
    2: 'IBV_ATOMIC_GLOB',
}
IBV_ATOMIC_NONE = 0
IBV_ATOMIC_HCA = 1
IBV_ATOMIC_GLOB = 2
ibv_atomic_cap = ctypes.c_uint32 # enum
class struct_ibv_alloc_dm_attr(Structure):
    pass

struct_ibv_alloc_dm_attr._pack_ = 1 # source:False
struct_ibv_alloc_dm_attr._fields_ = [
    ('length', ctypes.c_uint64),
    ('log_align_req', ctypes.c_uint32),
    ('comp_mask', ctypes.c_uint32),
]


# values for enumeration 'ibv_dm_mask'
ibv_dm_mask__enumvalues = {
    1: 'IBV_DM_MASK_HANDLE',
}
IBV_DM_MASK_HANDLE = 1
ibv_dm_mask = ctypes.c_uint32 # enum
class struct_ibv_dm(Structure):
    pass

class struct_ibv_context(Structure):
    pass

struct_ibv_dm._pack_ = 1 # source:False
struct_ibv_dm._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('memcpy_to_dm', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_dm), ctypes.c_uint64, ctypes.POINTER(None), ctypes.c_uint64)),
    ('memcpy_from_dm', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct_ibv_dm), ctypes.c_uint64, ctypes.c_uint64)),
    ('comp_mask', ctypes.c_uint32),
    ('handle', ctypes.c_uint32),
]

class struct_ibv_device(Structure):
    pass

class struct_ibv_context_ops(Structure):
    pass

class struct_ibv_device_attr(Structure):
    pass

class struct__compat_ibv_port_attr(Structure):
    pass

class struct_ibv_mw(Structure):
    pass

class struct_ibv_pd(Structure):
    pass


# values for enumeration 'ibv_mw_type'
ibv_mw_type__enumvalues = {
    1: 'IBV_MW_TYPE_1',
    2: 'IBV_MW_TYPE_2',
}
IBV_MW_TYPE_1 = 1
IBV_MW_TYPE_2 = 2
ibv_mw_type = ctypes.c_uint32 # enum
class struct_ibv_qp(Structure):
    pass

class struct_ibv_mw_bind(Structure):
    pass

class struct_ibv_cq(Structure):
    pass

class struct_ibv_wc(Structure):
    pass

class struct_ibv_srq(Structure):
    pass

class struct_ibv_recv_wr(Structure):
    pass

class struct_ibv_send_wr(Structure):
    pass

struct_ibv_context_ops._pack_ = 1 # source:False
struct_ibv_context_ops._fields_ = [
    ('_compat_query_device', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_device_attr))),
    ('_compat_query_port', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), ctypes.c_ubyte, ctypes.POINTER(struct__compat_ibv_port_attr))),
    ('_compat_alloc_pd', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_dealloc_pd', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_reg_mr', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_rereg_mr', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_dereg_mr', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('alloc_mw', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_mw), ctypes.POINTER(struct_ibv_pd), ibv_mw_type)),
    ('bind_mw', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_mw), ctypes.POINTER(struct_ibv_mw_bind))),
    ('dealloc_mw', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_mw))),
    ('_compat_create_cq', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('poll_cq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq), ctypes.c_int32, ctypes.POINTER(struct_ibv_wc))),
    ('req_notify_cq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq), ctypes.c_int32)),
    ('_compat_cq_event', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_resize_cq', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_destroy_cq', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_create_srq', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_modify_srq', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_query_srq', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_destroy_srq', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('post_srq_recv', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_recv_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_recv_wr)))),
    ('_compat_create_qp', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_query_qp', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_modify_qp', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_destroy_qp', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('post_send', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_send_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_send_wr)))),
    ('post_recv', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_recv_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_recv_wr)))),
    ('_compat_create_ah', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_destroy_ah', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_attach_mcast', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_detach_mcast', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
    ('_compat_async_event', ctypes.CFUNCTYPE(ctypes.POINTER(None))),
]

class union_c__UA_pthread_mutex_t(Union):
    pass

class struct___pthread_mutex_s(Structure):
    pass

class struct___pthread_internal_list(Structure):
    pass

struct___pthread_internal_list._pack_ = 1 # source:False
struct___pthread_internal_list._fields_ = [
    ('__prev', ctypes.POINTER(struct___pthread_internal_list)),
    ('__next', ctypes.POINTER(struct___pthread_internal_list)),
]

struct___pthread_mutex_s._pack_ = 1 # source:False
struct___pthread_mutex_s._fields_ = [
    ('__lock', ctypes.c_int32),
    ('__count', ctypes.c_uint32),
    ('__owner', ctypes.c_int32),
    ('__nusers', ctypes.c_uint32),
    ('__kind', ctypes.c_int32),
    ('__spins', ctypes.c_int16),
    ('__elision', ctypes.c_int16),
    ('__list', struct___pthread_internal_list),
]

union_c__UA_pthread_mutex_t._pack_ = 1 # source:False
union_c__UA_pthread_mutex_t._fields_ = [
    ('__data', struct___pthread_mutex_s),
    ('__size', ctypes.c_char * 40),
    ('__align', ctypes.c_int64),
    ('PADDING_0', ctypes.c_ubyte * 32),
]

struct_ibv_context._pack_ = 1 # source:False
struct_ibv_context._fields_ = [
    ('device', ctypes.POINTER(struct_ibv_device)),
    ('ops', struct_ibv_context_ops),
    ('cmd_fd', ctypes.c_int32),
    ('async_fd', ctypes.c_int32),
    ('num_comp_vectors', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('mutex', union_c__UA_pthread_mutex_t),
    ('abi_compat', ctypes.POINTER(None)),
]

class struct__ibv_device_ops(Structure):
    pass

struct__ibv_device_ops._pack_ = 1 # source:False
struct__ibv_device_ops._fields_ = [
    ('_dummy1', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_device), ctypes.c_int32)),
    ('_dummy2', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_context))),
]

struct_ibv_device._pack_ = 1 # source:False
struct_ibv_device._fields_ = [
    ('_ops', struct__ibv_device_ops),
    ('node_type', ibv_node_type),
    ('transport_type', ibv_transport_type),
    ('name', ctypes.c_char * 64),
    ('dev_name', ctypes.c_char * 64),
    ('dev_path', ctypes.c_char * 256),
    ('ibdev_path', ctypes.c_char * 256),
]

struct_ibv_device_attr._pack_ = 1 # source:False
struct_ibv_device_attr._fields_ = [
    ('fw_ver', ctypes.c_char * 64),
    ('node_guid', ctypes.c_uint64),
    ('sys_image_guid', ctypes.c_uint64),
    ('max_mr_size', ctypes.c_uint64),
    ('page_size_cap', ctypes.c_uint64),
    ('vendor_id', ctypes.c_uint32),
    ('vendor_part_id', ctypes.c_uint32),
    ('hw_ver', ctypes.c_uint32),
    ('max_qp', ctypes.c_int32),
    ('max_qp_wr', ctypes.c_int32),
    ('device_cap_flags', ctypes.c_uint32),
    ('max_sge', ctypes.c_int32),
    ('max_sge_rd', ctypes.c_int32),
    ('max_cq', ctypes.c_int32),
    ('max_cqe', ctypes.c_int32),
    ('max_mr', ctypes.c_int32),
    ('max_pd', ctypes.c_int32),
    ('max_qp_rd_atom', ctypes.c_int32),
    ('max_ee_rd_atom', ctypes.c_int32),
    ('max_res_rd_atom', ctypes.c_int32),
    ('max_qp_init_rd_atom', ctypes.c_int32),
    ('max_ee_init_rd_atom', ctypes.c_int32),
    ('atomic_cap', ibv_atomic_cap),
    ('max_ee', ctypes.c_int32),
    ('max_rdd', ctypes.c_int32),
    ('max_mw', ctypes.c_int32),
    ('max_raw_ipv6_qp', ctypes.c_int32),
    ('max_raw_ethy_qp', ctypes.c_int32),
    ('max_mcast_grp', ctypes.c_int32),
    ('max_mcast_qp_attach', ctypes.c_int32),
    ('max_total_mcast_qp_attach', ctypes.c_int32),
    ('max_ah', ctypes.c_int32),
    ('max_fmr', ctypes.c_int32),
    ('max_map_per_fmr', ctypes.c_int32),
    ('max_srq', ctypes.c_int32),
    ('max_srq_wr', ctypes.c_int32),
    ('max_srq_sge', ctypes.c_int32),
    ('max_pkeys', ctypes.c_uint16),
    ('local_ca_ack_delay', ctypes.c_ubyte),
    ('phys_port_cnt', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_ibv_mw._pack_ = 1 # source:False
struct_ibv_mw._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('rkey', ctypes.c_uint32),
    ('handle', ctypes.c_uint32),
    ('type', ibv_mw_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_ibv_pd._pack_ = 1 # source:False
struct_ibv_pd._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('handle', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]


# values for enumeration 'ibv_qp_state'
ibv_qp_state__enumvalues = {
    0: 'IBV_QPS_RESET',
    1: 'IBV_QPS_INIT',
    2: 'IBV_QPS_RTR',
    3: 'IBV_QPS_RTS',
    4: 'IBV_QPS_SQD',
    5: 'IBV_QPS_SQE',
    6: 'IBV_QPS_ERR',
    7: 'IBV_QPS_UNKNOWN',
}
IBV_QPS_RESET = 0
IBV_QPS_INIT = 1
IBV_QPS_RTR = 2
IBV_QPS_RTS = 3
IBV_QPS_SQD = 4
IBV_QPS_SQE = 5
IBV_QPS_ERR = 6
IBV_QPS_UNKNOWN = 7
ibv_qp_state = ctypes.c_uint32 # enum

# values for enumeration 'ibv_qp_type'
ibv_qp_type__enumvalues = {
    2: 'IBV_QPT_RC',
    3: 'IBV_QPT_UC',
    4: 'IBV_QPT_UD',
    8: 'IBV_QPT_RAW_PACKET',
    9: 'IBV_QPT_XRC_SEND',
    10: 'IBV_QPT_XRC_RECV',
    255: 'IBV_QPT_DRIVER',
}
IBV_QPT_RC = 2
IBV_QPT_UC = 3
IBV_QPT_UD = 4
IBV_QPT_RAW_PACKET = 8
IBV_QPT_XRC_SEND = 9
IBV_QPT_XRC_RECV = 10
IBV_QPT_DRIVER = 255
ibv_qp_type = ctypes.c_uint32 # enum
class union_c__UA_pthread_cond_t(Union):
    pass

class struct___pthread_cond_s(Structure):
    pass

class union_c__UA___atomic_wide_counter(Union):
    pass

class struct_c__UA___atomic_wide_counter___value32(Structure):
    pass

struct_c__UA___atomic_wide_counter___value32._pack_ = 1 # source:False
struct_c__UA___atomic_wide_counter___value32._fields_ = [
    ('__low', ctypes.c_uint32),
    ('__high', ctypes.c_uint32),
]

union_c__UA___atomic_wide_counter._pack_ = 1 # source:False
union_c__UA___atomic_wide_counter._fields_ = [
    ('__value64', ctypes.c_uint64),
    ('__value32', struct_c__UA___atomic_wide_counter___value32),
]

struct___pthread_cond_s._pack_ = 1 # source:False
struct___pthread_cond_s._fields_ = [
    ('__wseq', union_c__UA___atomic_wide_counter),
    ('__g1_start', union_c__UA___atomic_wide_counter),
    ('__g_refs', ctypes.c_uint32 * 2),
    ('__g_size', ctypes.c_uint32 * 2),
    ('__g1_orig_size', ctypes.c_uint32),
    ('__wrefs', ctypes.c_uint32),
    ('__g_signals', ctypes.c_uint32 * 2),
]

union_c__UA_pthread_cond_t._pack_ = 1 # source:False
union_c__UA_pthread_cond_t._fields_ = [
    ('__data', struct___pthread_cond_s),
    ('__size', ctypes.c_char * 48),
    ('__align', ctypes.c_int64),
    ('PADDING_0', ctypes.c_ubyte * 40),
]

struct_ibv_qp._pack_ = 1 # source:False
struct_ibv_qp._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('qp_context', ctypes.POINTER(None)),
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('send_cq', ctypes.POINTER(struct_ibv_cq)),
    ('recv_cq', ctypes.POINTER(struct_ibv_cq)),
    ('srq', ctypes.POINTER(struct_ibv_srq)),
    ('handle', ctypes.c_uint32),
    ('qp_num', ctypes.c_uint32),
    ('state', ibv_qp_state),
    ('qp_type', ibv_qp_type),
    ('mutex', union_c__UA_pthread_mutex_t),
    ('cond', union_c__UA_pthread_cond_t),
    ('events_completed', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_ibv_comp_channel(Structure):
    pass

struct_ibv_cq._pack_ = 1 # source:False
struct_ibv_cq._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('channel', ctypes.POINTER(struct_ibv_comp_channel)),
    ('cq_context', ctypes.POINTER(None)),
    ('handle', ctypes.c_uint32),
    ('cqe', ctypes.c_int32),
    ('mutex', union_c__UA_pthread_mutex_t),
    ('cond', union_c__UA_pthread_cond_t),
    ('comp_events_completed', ctypes.c_uint32),
    ('async_events_completed', ctypes.c_uint32),
]

struct_ibv_comp_channel._pack_ = 1 # source:False
struct_ibv_comp_channel._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('fd', ctypes.c_int32),
    ('refcnt', ctypes.c_int32),
]

struct_ibv_srq._pack_ = 1 # source:False
struct_ibv_srq._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('srq_context', ctypes.POINTER(None)),
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('handle', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('mutex', union_c__UA_pthread_mutex_t),
    ('cond', union_c__UA_pthread_cond_t),
    ('events_completed', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_ibv_mw_bind_info(Structure):
    pass

class struct_ibv_mr(Structure):
    pass

struct_ibv_mw_bind_info._pack_ = 1 # source:False
struct_ibv_mw_bind_info._fields_ = [
    ('mr', ctypes.POINTER(struct_ibv_mr)),
    ('addr', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('mw_access_flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_ibv_mw_bind._pack_ = 1 # source:False
struct_ibv_mw_bind._fields_ = [
    ('wr_id', ctypes.c_uint64),
    ('send_flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('bind_info', struct_ibv_mw_bind_info),
]

struct_ibv_mr._pack_ = 1 # source:False
struct_ibv_mr._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('addr', ctypes.POINTER(None)),
    ('length', ctypes.c_uint64),
    ('handle', ctypes.c_uint32),
    ('lkey', ctypes.c_uint32),
    ('rkey', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]


# values for enumeration 'ibv_wc_status'
ibv_wc_status__enumvalues = {
    0: 'IBV_WC_SUCCESS',
    1: 'IBV_WC_LOC_LEN_ERR',
    2: 'IBV_WC_LOC_QP_OP_ERR',
    3: 'IBV_WC_LOC_EEC_OP_ERR',
    4: 'IBV_WC_LOC_PROT_ERR',
    5: 'IBV_WC_WR_FLUSH_ERR',
    6: 'IBV_WC_MW_BIND_ERR',
    7: 'IBV_WC_BAD_RESP_ERR',
    8: 'IBV_WC_LOC_ACCESS_ERR',
    9: 'IBV_WC_REM_INV_REQ_ERR',
    10: 'IBV_WC_REM_ACCESS_ERR',
    11: 'IBV_WC_REM_OP_ERR',
    12: 'IBV_WC_RETRY_EXC_ERR',
    13: 'IBV_WC_RNR_RETRY_EXC_ERR',
    14: 'IBV_WC_LOC_RDD_VIOL_ERR',
    15: 'IBV_WC_REM_INV_RD_REQ_ERR',
    16: 'IBV_WC_REM_ABORT_ERR',
    17: 'IBV_WC_INV_EECN_ERR',
    18: 'IBV_WC_INV_EEC_STATE_ERR',
    19: 'IBV_WC_FATAL_ERR',
    20: 'IBV_WC_RESP_TIMEOUT_ERR',
    21: 'IBV_WC_GENERAL_ERR',
    22: 'IBV_WC_TM_ERR',
    23: 'IBV_WC_TM_RNDV_INCOMPLETE',
}
IBV_WC_SUCCESS = 0
IBV_WC_LOC_LEN_ERR = 1
IBV_WC_LOC_QP_OP_ERR = 2
IBV_WC_LOC_EEC_OP_ERR = 3
IBV_WC_LOC_PROT_ERR = 4
IBV_WC_WR_FLUSH_ERR = 5
IBV_WC_MW_BIND_ERR = 6
IBV_WC_BAD_RESP_ERR = 7
IBV_WC_LOC_ACCESS_ERR = 8
IBV_WC_REM_INV_REQ_ERR = 9
IBV_WC_REM_ACCESS_ERR = 10
IBV_WC_REM_OP_ERR = 11
IBV_WC_RETRY_EXC_ERR = 12
IBV_WC_RNR_RETRY_EXC_ERR = 13
IBV_WC_LOC_RDD_VIOL_ERR = 14
IBV_WC_REM_INV_RD_REQ_ERR = 15
IBV_WC_REM_ABORT_ERR = 16
IBV_WC_INV_EECN_ERR = 17
IBV_WC_INV_EEC_STATE_ERR = 18
IBV_WC_FATAL_ERR = 19
IBV_WC_RESP_TIMEOUT_ERR = 20
IBV_WC_GENERAL_ERR = 21
IBV_WC_TM_ERR = 22
IBV_WC_TM_RNDV_INCOMPLETE = 23
ibv_wc_status = ctypes.c_uint32 # enum

# values for enumeration 'ibv_wc_opcode'
ibv_wc_opcode__enumvalues = {
    0: 'IBV_WC_SEND',
    1: 'IBV_WC_RDMA_WRITE',
    2: 'IBV_WC_RDMA_READ',
    3: 'IBV_WC_COMP_SWAP',
    4: 'IBV_WC_FETCH_ADD',
    5: 'IBV_WC_BIND_MW',
    6: 'IBV_WC_LOCAL_INV',
    7: 'IBV_WC_TSO',
    8: 'IBV_WC_FLUSH',
    9: 'IBV_WC_ATOMIC_WRITE',
    128: 'IBV_WC_RECV',
    129: 'IBV_WC_RECV_RDMA_WITH_IMM',
    130: 'IBV_WC_TM_ADD',
    131: 'IBV_WC_TM_DEL',
    132: 'IBV_WC_TM_SYNC',
    133: 'IBV_WC_TM_RECV',
    134: 'IBV_WC_TM_NO_TAG',
    135: 'IBV_WC_DRIVER1',
    136: 'IBV_WC_DRIVER2',
    137: 'IBV_WC_DRIVER3',
}
IBV_WC_SEND = 0
IBV_WC_RDMA_WRITE = 1
IBV_WC_RDMA_READ = 2
IBV_WC_COMP_SWAP = 3
IBV_WC_FETCH_ADD = 4
IBV_WC_BIND_MW = 5
IBV_WC_LOCAL_INV = 6
IBV_WC_TSO = 7
IBV_WC_FLUSH = 8
IBV_WC_ATOMIC_WRITE = 9
IBV_WC_RECV = 128
IBV_WC_RECV_RDMA_WITH_IMM = 129
IBV_WC_TM_ADD = 130
IBV_WC_TM_DEL = 131
IBV_WC_TM_SYNC = 132
IBV_WC_TM_RECV = 133
IBV_WC_TM_NO_TAG = 134
IBV_WC_DRIVER1 = 135
IBV_WC_DRIVER2 = 136
IBV_WC_DRIVER3 = 137
ibv_wc_opcode = ctypes.c_uint32 # enum
class union_ibv_wc_0(Union):
    pass

union_ibv_wc_0._pack_ = 1 # source:False
union_ibv_wc_0._fields_ = [
    ('imm_data', ctypes.c_uint32),
    ('invalidated_rkey', ctypes.c_uint32),
]

struct_ibv_wc._pack_ = 1 # source:False
struct_ibv_wc._anonymous_ = ('_0',)
struct_ibv_wc._fields_ = [
    ('wr_id', ctypes.c_uint64),
    ('status', ibv_wc_status),
    ('opcode', ibv_wc_opcode),
    ('vendor_err', ctypes.c_uint32),
    ('byte_len', ctypes.c_uint32),
    ('_0', union_ibv_wc_0),
    ('qp_num', ctypes.c_uint32),
    ('src_qp', ctypes.c_uint32),
    ('wc_flags', ctypes.c_uint32),
    ('pkey_index', ctypes.c_uint16),
    ('slid', ctypes.c_uint16),
    ('sl', ctypes.c_ubyte),
    ('dlid_path_bits', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

class struct_ibv_sge(Structure):
    pass

struct_ibv_recv_wr._pack_ = 1 # source:False
struct_ibv_recv_wr._fields_ = [
    ('wr_id', ctypes.c_uint64),
    ('next', ctypes.POINTER(struct_ibv_recv_wr)),
    ('sg_list', ctypes.POINTER(struct_ibv_sge)),
    ('num_sge', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_ibv_sge._pack_ = 1 # source:False
struct_ibv_sge._fields_ = [
    ('addr', ctypes.c_uint64),
    ('length', ctypes.c_uint32),
    ('lkey', ctypes.c_uint32),
]


# values for enumeration 'ibv_wr_opcode'
ibv_wr_opcode__enumvalues = {
    0: 'IBV_WR_RDMA_WRITE',
    1: 'IBV_WR_RDMA_WRITE_WITH_IMM',
    2: 'IBV_WR_SEND',
    3: 'IBV_WR_SEND_WITH_IMM',
    4: 'IBV_WR_RDMA_READ',
    5: 'IBV_WR_ATOMIC_CMP_AND_SWP',
    6: 'IBV_WR_ATOMIC_FETCH_AND_ADD',
    7: 'IBV_WR_LOCAL_INV',
    8: 'IBV_WR_BIND_MW',
    9: 'IBV_WR_SEND_WITH_INV',
    10: 'IBV_WR_TSO',
    11: 'IBV_WR_DRIVER1',
    14: 'IBV_WR_FLUSH',
    15: 'IBV_WR_ATOMIC_WRITE',
}
IBV_WR_RDMA_WRITE = 0
IBV_WR_RDMA_WRITE_WITH_IMM = 1
IBV_WR_SEND = 2
IBV_WR_SEND_WITH_IMM = 3
IBV_WR_RDMA_READ = 4
IBV_WR_ATOMIC_CMP_AND_SWP = 5
IBV_WR_ATOMIC_FETCH_AND_ADD = 6
IBV_WR_LOCAL_INV = 7
IBV_WR_BIND_MW = 8
IBV_WR_SEND_WITH_INV = 9
IBV_WR_TSO = 10
IBV_WR_DRIVER1 = 11
IBV_WR_FLUSH = 14
IBV_WR_ATOMIC_WRITE = 15
ibv_wr_opcode = ctypes.c_uint32 # enum
class union_ibv_send_wr_0(Union):
    pass

union_ibv_send_wr_0._pack_ = 1 # source:False
union_ibv_send_wr_0._fields_ = [
    ('imm_data', ctypes.c_uint32),
    ('invalidate_rkey', ctypes.c_uint32),
]

class union_ibv_send_wr_wr(Union):
    pass

class struct_ibv_send_wr_1_rdma(Structure):
    pass

struct_ibv_send_wr_1_rdma._pack_ = 1 # source:False
struct_ibv_send_wr_1_rdma._fields_ = [
    ('remote_addr', ctypes.c_uint64),
    ('rkey', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_ibv_send_wr_1_atomic(Structure):
    pass

struct_ibv_send_wr_1_atomic._pack_ = 1 # source:False
struct_ibv_send_wr_1_atomic._fields_ = [
    ('remote_addr', ctypes.c_uint64),
    ('compare_add', ctypes.c_uint64),
    ('swap', ctypes.c_uint64),
    ('rkey', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_ibv_send_wr_1_ud(Structure):
    pass

class struct_ibv_ah(Structure):
    pass

struct_ibv_send_wr_1_ud._pack_ = 1 # source:False
struct_ibv_send_wr_1_ud._fields_ = [
    ('ah', ctypes.POINTER(struct_ibv_ah)),
    ('remote_qpn', ctypes.c_uint32),
    ('remote_qkey', ctypes.c_uint32),
]

union_ibv_send_wr_wr._pack_ = 1 # source:False
union_ibv_send_wr_wr._fields_ = [
    ('rdma', struct_ibv_send_wr_1_rdma),
    ('atomic', struct_ibv_send_wr_1_atomic),
    ('ud', struct_ibv_send_wr_1_ud),
    ('PADDING_0', ctypes.c_ubyte * 16),
]

class union_ibv_send_wr_qp_type(Union):
    pass

class struct_ibv_send_wr_2_xrc(Structure):
    pass

struct_ibv_send_wr_2_xrc._pack_ = 1 # source:False
struct_ibv_send_wr_2_xrc._fields_ = [
    ('remote_srqn', ctypes.c_uint32),
]

union_ibv_send_wr_qp_type._pack_ = 1 # source:False
union_ibv_send_wr_qp_type._fields_ = [
    ('xrc', struct_ibv_send_wr_2_xrc),
]

class union_ibv_send_wr_3(Union):
    pass

class struct_ibv_send_wr_3_bind_mw(Structure):
    pass

struct_ibv_send_wr_3_bind_mw._pack_ = 1 # source:False
struct_ibv_send_wr_3_bind_mw._fields_ = [
    ('mw', ctypes.POINTER(struct_ibv_mw)),
    ('rkey', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('bind_info', struct_ibv_mw_bind_info),
]

class struct_ibv_send_wr_3_tso(Structure):
    pass

struct_ibv_send_wr_3_tso._pack_ = 1 # source:False
struct_ibv_send_wr_3_tso._fields_ = [
    ('hdr', ctypes.POINTER(None)),
    ('hdr_sz', ctypes.c_uint16),
    ('mss', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

union_ibv_send_wr_3._pack_ = 1 # source:False
union_ibv_send_wr_3._fields_ = [
    ('bind_mw', struct_ibv_send_wr_3_bind_mw),
    ('tso', struct_ibv_send_wr_3_tso),
    ('PADDING_0', ctypes.c_ubyte * 32),
]

struct_ibv_send_wr._pack_ = 1 # source:False
struct_ibv_send_wr._anonymous_ = ('_0', '_1',)
struct_ibv_send_wr._fields_ = [
    ('wr_id', ctypes.c_uint64),
    ('next', ctypes.POINTER(struct_ibv_send_wr)),
    ('sg_list', ctypes.POINTER(struct_ibv_sge)),
    ('num_sge', ctypes.c_int32),
    ('opcode', ibv_wr_opcode),
    ('send_flags', ctypes.c_uint32),
    ('_0', union_ibv_send_wr_0),
    ('wr', union_ibv_send_wr_wr),
    ('qp_type', union_ibv_send_wr_qp_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_1', union_ibv_send_wr_3),
]

struct_ibv_ah._pack_ = 1 # source:False
struct_ibv_ah._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('handle', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_ibv_query_device_ex_input(Structure):
    pass

struct_ibv_query_device_ex_input._pack_ = 1 # source:False
struct_ibv_query_device_ex_input._fields_ = [
    ('comp_mask', ctypes.c_uint32),
]


# values for enumeration 'ibv_odp_transport_cap_bits'
ibv_odp_transport_cap_bits__enumvalues = {
    1: 'IBV_ODP_SUPPORT_SEND',
    2: 'IBV_ODP_SUPPORT_RECV',
    4: 'IBV_ODP_SUPPORT_WRITE',
    8: 'IBV_ODP_SUPPORT_READ',
    16: 'IBV_ODP_SUPPORT_ATOMIC',
    32: 'IBV_ODP_SUPPORT_SRQ_RECV',
}
IBV_ODP_SUPPORT_SEND = 1
IBV_ODP_SUPPORT_RECV = 2
IBV_ODP_SUPPORT_WRITE = 4
IBV_ODP_SUPPORT_READ = 8
IBV_ODP_SUPPORT_ATOMIC = 16
IBV_ODP_SUPPORT_SRQ_RECV = 32
ibv_odp_transport_cap_bits = ctypes.c_uint32 # enum
class struct_ibv_odp_caps(Structure):
    pass

class struct_ibv_odp_caps_per_transport_caps(Structure):
    pass

struct_ibv_odp_caps_per_transport_caps._pack_ = 1 # source:False
struct_ibv_odp_caps_per_transport_caps._fields_ = [
    ('rc_odp_caps', ctypes.c_uint32),
    ('uc_odp_caps', ctypes.c_uint32),
    ('ud_odp_caps', ctypes.c_uint32),
]

struct_ibv_odp_caps._pack_ = 1 # source:False
struct_ibv_odp_caps._fields_ = [
    ('general_caps', ctypes.c_uint64),
    ('per_transport_caps', struct_ibv_odp_caps_per_transport_caps),
    ('PADDING_0', ctypes.c_ubyte * 4),
]


# values for enumeration 'ibv_odp_general_caps'
ibv_odp_general_caps__enumvalues = {
    1: 'IBV_ODP_SUPPORT',
    2: 'IBV_ODP_SUPPORT_IMPLICIT',
}
IBV_ODP_SUPPORT = 1
IBV_ODP_SUPPORT_IMPLICIT = 2
ibv_odp_general_caps = ctypes.c_uint32 # enum
class struct_ibv_tso_caps(Structure):
    pass

struct_ibv_tso_caps._pack_ = 1 # source:False
struct_ibv_tso_caps._fields_ = [
    ('max_tso', ctypes.c_uint32),
    ('supported_qpts', ctypes.c_uint32),
]


# values for enumeration 'ibv_rx_hash_function_flags'
ibv_rx_hash_function_flags__enumvalues = {
    1: 'IBV_RX_HASH_FUNC_TOEPLITZ',
}
IBV_RX_HASH_FUNC_TOEPLITZ = 1
ibv_rx_hash_function_flags = ctypes.c_uint32 # enum

# values for enumeration 'ibv_rx_hash_fields'
ibv_rx_hash_fields__enumvalues = {
    1: 'IBV_RX_HASH_SRC_IPV4',
    2: 'IBV_RX_HASH_DST_IPV4',
    4: 'IBV_RX_HASH_SRC_IPV6',
    8: 'IBV_RX_HASH_DST_IPV6',
    16: 'IBV_RX_HASH_SRC_PORT_TCP',
    32: 'IBV_RX_HASH_DST_PORT_TCP',
    64: 'IBV_RX_HASH_SRC_PORT_UDP',
    128: 'IBV_RX_HASH_DST_PORT_UDP',
    256: 'IBV_RX_HASH_IPSEC_SPI',
    2147483648: 'IBV_RX_HASH_INNER',
}
IBV_RX_HASH_SRC_IPV4 = 1
IBV_RX_HASH_DST_IPV4 = 2
IBV_RX_HASH_SRC_IPV6 = 4
IBV_RX_HASH_DST_IPV6 = 8
IBV_RX_HASH_SRC_PORT_TCP = 16
IBV_RX_HASH_DST_PORT_TCP = 32
IBV_RX_HASH_SRC_PORT_UDP = 64
IBV_RX_HASH_DST_PORT_UDP = 128
IBV_RX_HASH_IPSEC_SPI = 256
IBV_RX_HASH_INNER = 2147483648
ibv_rx_hash_fields = ctypes.c_uint32 # enum
class struct_ibv_rss_caps(Structure):
    pass

struct_ibv_rss_caps._pack_ = 1 # source:False
struct_ibv_rss_caps._fields_ = [
    ('supported_qpts', ctypes.c_uint32),
    ('max_rwq_indirection_tables', ctypes.c_uint32),
    ('max_rwq_indirection_table_size', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('rx_hash_fields_mask', ctypes.c_uint64),
    ('rx_hash_function', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

class struct_ibv_packet_pacing_caps(Structure):
    pass

struct_ibv_packet_pacing_caps._pack_ = 1 # source:False
struct_ibv_packet_pacing_caps._fields_ = [
    ('qp_rate_limit_min', ctypes.c_uint32),
    ('qp_rate_limit_max', ctypes.c_uint32),
    ('supported_qpts', ctypes.c_uint32),
]


# values for enumeration 'ibv_raw_packet_caps'
ibv_raw_packet_caps__enumvalues = {
    1: 'IBV_RAW_PACKET_CAP_CVLAN_STRIPPING',
    2: 'IBV_RAW_PACKET_CAP_SCATTER_FCS',
    4: 'IBV_RAW_PACKET_CAP_IP_CSUM',
    8: 'IBV_RAW_PACKET_CAP_DELAY_DROP',
}
IBV_RAW_PACKET_CAP_CVLAN_STRIPPING = 1
IBV_RAW_PACKET_CAP_SCATTER_FCS = 2
IBV_RAW_PACKET_CAP_IP_CSUM = 4
IBV_RAW_PACKET_CAP_DELAY_DROP = 8
ibv_raw_packet_caps = ctypes.c_uint32 # enum

# values for enumeration 'ibv_tm_cap_flags'
ibv_tm_cap_flags__enumvalues = {
    1: 'IBV_TM_CAP_RC',
}
IBV_TM_CAP_RC = 1
ibv_tm_cap_flags = ctypes.c_uint32 # enum
class struct_ibv_tm_caps(Structure):
    pass

struct_ibv_tm_caps._pack_ = 1 # source:False
struct_ibv_tm_caps._fields_ = [
    ('max_rndv_hdr_size', ctypes.c_uint32),
    ('max_num_tags', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('max_ops', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
]

class struct_ibv_cq_moderation_caps(Structure):
    pass

struct_ibv_cq_moderation_caps._pack_ = 1 # source:False
struct_ibv_cq_moderation_caps._fields_ = [
    ('max_cq_count', ctypes.c_uint16),
    ('max_cq_period', ctypes.c_uint16),
]


# values for enumeration 'ibv_pci_atomic_op_size'
ibv_pci_atomic_op_size__enumvalues = {
    1: 'IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP',
    2: 'IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP',
    4: 'IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP',
}
IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP = 1
IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP = 2
IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP = 4
ibv_pci_atomic_op_size = ctypes.c_uint32 # enum
class struct_ibv_pci_atomic_caps(Structure):
    pass

struct_ibv_pci_atomic_caps._pack_ = 1 # source:False
struct_ibv_pci_atomic_caps._fields_ = [
    ('fetch_add', ctypes.c_uint16),
    ('swap', ctypes.c_uint16),
    ('compare_swap', ctypes.c_uint16),
]

class struct_ibv_device_attr_ex(Structure):
    pass

struct_ibv_device_attr_ex._pack_ = 1 # source:False
struct_ibv_device_attr_ex._fields_ = [
    ('orig_attr', struct_ibv_device_attr),
    ('comp_mask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('odp_caps', struct_ibv_odp_caps),
    ('completion_timestamp_mask', ctypes.c_uint64),
    ('hca_core_clock', ctypes.c_uint64),
    ('device_cap_flags_ex', ctypes.c_uint64),
    ('tso_caps', struct_ibv_tso_caps),
    ('rss_caps', struct_ibv_rss_caps),
    ('max_wq_type_rq', ctypes.c_uint32),
    ('packet_pacing_caps', struct_ibv_packet_pacing_caps),
    ('raw_packet_caps', ctypes.c_uint32),
    ('tm_caps', struct_ibv_tm_caps),
    ('cq_mod_caps', struct_ibv_cq_moderation_caps),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('max_dm_size', ctypes.c_uint64),
    ('pci_atomic_caps', struct_ibv_pci_atomic_caps),
    ('PADDING_2', ctypes.c_ubyte * 2),
    ('xrc_odp_caps', ctypes.c_uint32),
    ('phys_port_cnt_ex', ctypes.c_uint32),
]


# values for enumeration 'ibv_mtu'
ibv_mtu__enumvalues = {
    1: 'IBV_MTU_256',
    2: 'IBV_MTU_512',
    3: 'IBV_MTU_1024',
    4: 'IBV_MTU_2048',
    5: 'IBV_MTU_4096',
}
IBV_MTU_256 = 1
IBV_MTU_512 = 2
IBV_MTU_1024 = 3
IBV_MTU_2048 = 4
IBV_MTU_4096 = 5
ibv_mtu = ctypes.c_uint32 # enum

# values for enumeration 'ibv_port_state'
ibv_port_state__enumvalues = {
    0: 'IBV_PORT_NOP',
    1: 'IBV_PORT_DOWN',
    2: 'IBV_PORT_INIT',
    3: 'IBV_PORT_ARMED',
    4: 'IBV_PORT_ACTIVE',
    5: 'IBV_PORT_ACTIVE_DEFER',
}
IBV_PORT_NOP = 0
IBV_PORT_DOWN = 1
IBV_PORT_INIT = 2
IBV_PORT_ARMED = 3
IBV_PORT_ACTIVE = 4
IBV_PORT_ACTIVE_DEFER = 5
ibv_port_state = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_IBV_LINK_LAYER_UNSPECIFIED'
c__Ea_IBV_LINK_LAYER_UNSPECIFIED__enumvalues = {
    0: 'IBV_LINK_LAYER_UNSPECIFIED',
    1: 'IBV_LINK_LAYER_INFINIBAND',
    2: 'IBV_LINK_LAYER_ETHERNET',
}
IBV_LINK_LAYER_UNSPECIFIED = 0
IBV_LINK_LAYER_INFINIBAND = 1
IBV_LINK_LAYER_ETHERNET = 2
c__Ea_IBV_LINK_LAYER_UNSPECIFIED = ctypes.c_uint32 # enum

# values for enumeration 'ibv_port_cap_flags'
ibv_port_cap_flags__enumvalues = {
    2: 'IBV_PORT_SM',
    4: 'IBV_PORT_NOTICE_SUP',
    8: 'IBV_PORT_TRAP_SUP',
    16: 'IBV_PORT_OPT_IPD_SUP',
    32: 'IBV_PORT_AUTO_MIGR_SUP',
    64: 'IBV_PORT_SL_MAP_SUP',
    128: 'IBV_PORT_MKEY_NVRAM',
    256: 'IBV_PORT_PKEY_NVRAM',
    512: 'IBV_PORT_LED_INFO_SUP',
    2048: 'IBV_PORT_SYS_IMAGE_GUID_SUP',
    4096: 'IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP',
    16384: 'IBV_PORT_EXTENDED_SPEEDS_SUP',
    32768: 'IBV_PORT_CAP_MASK2_SUP',
    65536: 'IBV_PORT_CM_SUP',
    131072: 'IBV_PORT_SNMP_TUNNEL_SUP',
    262144: 'IBV_PORT_REINIT_SUP',
    524288: 'IBV_PORT_DEVICE_MGMT_SUP',
    1048576: 'IBV_PORT_VENDOR_CLASS_SUP',
    2097152: 'IBV_PORT_DR_NOTICE_SUP',
    4194304: 'IBV_PORT_CAP_MASK_NOTICE_SUP',
    8388608: 'IBV_PORT_BOOT_MGMT_SUP',
    16777216: 'IBV_PORT_LINK_LATENCY_SUP',
    33554432: 'IBV_PORT_CLIENT_REG_SUP',
    67108864: 'IBV_PORT_IP_BASED_GIDS',
}
IBV_PORT_SM = 2
IBV_PORT_NOTICE_SUP = 4
IBV_PORT_TRAP_SUP = 8
IBV_PORT_OPT_IPD_SUP = 16
IBV_PORT_AUTO_MIGR_SUP = 32
IBV_PORT_SL_MAP_SUP = 64
IBV_PORT_MKEY_NVRAM = 128
IBV_PORT_PKEY_NVRAM = 256
IBV_PORT_LED_INFO_SUP = 512
IBV_PORT_SYS_IMAGE_GUID_SUP = 2048
IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP = 4096
IBV_PORT_EXTENDED_SPEEDS_SUP = 16384
IBV_PORT_CAP_MASK2_SUP = 32768
IBV_PORT_CM_SUP = 65536
IBV_PORT_SNMP_TUNNEL_SUP = 131072
IBV_PORT_REINIT_SUP = 262144
IBV_PORT_DEVICE_MGMT_SUP = 524288
IBV_PORT_VENDOR_CLASS_SUP = 1048576
IBV_PORT_DR_NOTICE_SUP = 2097152
IBV_PORT_CAP_MASK_NOTICE_SUP = 4194304
IBV_PORT_BOOT_MGMT_SUP = 8388608
IBV_PORT_LINK_LATENCY_SUP = 16777216
IBV_PORT_CLIENT_REG_SUP = 33554432
IBV_PORT_IP_BASED_GIDS = 67108864
ibv_port_cap_flags = ctypes.c_uint32 # enum

# values for enumeration 'ibv_port_cap_flags2'
ibv_port_cap_flags2__enumvalues = {
    1: 'IBV_PORT_SET_NODE_DESC_SUP',
    2: 'IBV_PORT_INFO_EXT_SUP',
    4: 'IBV_PORT_VIRT_SUP',
    8: 'IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP',
    16: 'IBV_PORT_LINK_WIDTH_2X_SUP',
    32: 'IBV_PORT_LINK_SPEED_HDR_SUP',
    1024: 'IBV_PORT_LINK_SPEED_NDR_SUP',
    4096: 'IBV_PORT_LINK_SPEED_XDR_SUP',
}
IBV_PORT_SET_NODE_DESC_SUP = 1
IBV_PORT_INFO_EXT_SUP = 2
IBV_PORT_VIRT_SUP = 4
IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP = 8
IBV_PORT_LINK_WIDTH_2X_SUP = 16
IBV_PORT_LINK_SPEED_HDR_SUP = 32
IBV_PORT_LINK_SPEED_NDR_SUP = 1024
IBV_PORT_LINK_SPEED_XDR_SUP = 4096
ibv_port_cap_flags2 = ctypes.c_uint32 # enum
class struct_ibv_port_attr(Structure):
    pass

struct_ibv_port_attr._pack_ = 1 # source:False
struct_ibv_port_attr._fields_ = [
    ('state', ibv_port_state),
    ('max_mtu', ibv_mtu),
    ('active_mtu', ibv_mtu),
    ('gid_tbl_len', ctypes.c_int32),
    ('port_cap_flags', ctypes.c_uint32),
    ('max_msg_sz', ctypes.c_uint32),
    ('bad_pkey_cntr', ctypes.c_uint32),
    ('qkey_viol_cntr', ctypes.c_uint32),
    ('pkey_tbl_len', ctypes.c_uint16),
    ('lid', ctypes.c_uint16),
    ('sm_lid', ctypes.c_uint16),
    ('lmc', ctypes.c_ubyte),
    ('max_vl_num', ctypes.c_ubyte),
    ('sm_sl', ctypes.c_ubyte),
    ('subnet_timeout', ctypes.c_ubyte),
    ('init_type_reply', ctypes.c_ubyte),
    ('active_width', ctypes.c_ubyte),
    ('active_speed', ctypes.c_ubyte),
    ('phys_state', ctypes.c_ubyte),
    ('link_layer', ctypes.c_ubyte),
    ('flags', ctypes.c_ubyte),
    ('port_cap_flags2', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('active_speed_ex', ctypes.c_uint32),
]


# values for enumeration 'ibv_event_type'
ibv_event_type__enumvalues = {
    0: 'IBV_EVENT_CQ_ERR',
    1: 'IBV_EVENT_QP_FATAL',
    2: 'IBV_EVENT_QP_REQ_ERR',
    3: 'IBV_EVENT_QP_ACCESS_ERR',
    4: 'IBV_EVENT_COMM_EST',
    5: 'IBV_EVENT_SQ_DRAINED',
    6: 'IBV_EVENT_PATH_MIG',
    7: 'IBV_EVENT_PATH_MIG_ERR',
    8: 'IBV_EVENT_DEVICE_FATAL',
    9: 'IBV_EVENT_PORT_ACTIVE',
    10: 'IBV_EVENT_PORT_ERR',
    11: 'IBV_EVENT_LID_CHANGE',
    12: 'IBV_EVENT_PKEY_CHANGE',
    13: 'IBV_EVENT_SM_CHANGE',
    14: 'IBV_EVENT_SRQ_ERR',
    15: 'IBV_EVENT_SRQ_LIMIT_REACHED',
    16: 'IBV_EVENT_QP_LAST_WQE_REACHED',
    17: 'IBV_EVENT_CLIENT_REREGISTER',
    18: 'IBV_EVENT_GID_CHANGE',
    19: 'IBV_EVENT_WQ_FATAL',
}
IBV_EVENT_CQ_ERR = 0
IBV_EVENT_QP_FATAL = 1
IBV_EVENT_QP_REQ_ERR = 2
IBV_EVENT_QP_ACCESS_ERR = 3
IBV_EVENT_COMM_EST = 4
IBV_EVENT_SQ_DRAINED = 5
IBV_EVENT_PATH_MIG = 6
IBV_EVENT_PATH_MIG_ERR = 7
IBV_EVENT_DEVICE_FATAL = 8
IBV_EVENT_PORT_ACTIVE = 9
IBV_EVENT_PORT_ERR = 10
IBV_EVENT_LID_CHANGE = 11
IBV_EVENT_PKEY_CHANGE = 12
IBV_EVENT_SM_CHANGE = 13
IBV_EVENT_SRQ_ERR = 14
IBV_EVENT_SRQ_LIMIT_REACHED = 15
IBV_EVENT_QP_LAST_WQE_REACHED = 16
IBV_EVENT_CLIENT_REREGISTER = 17
IBV_EVENT_GID_CHANGE = 18
IBV_EVENT_WQ_FATAL = 19
ibv_event_type = ctypes.c_uint32 # enum
class struct_ibv_async_event(Structure):
    pass

class union_ibv_async_event_element(Union):
    pass

class struct_ibv_wq(Structure):
    pass

union_ibv_async_event_element._pack_ = 1 # source:False
union_ibv_async_event_element._fields_ = [
    ('cq', ctypes.POINTER(struct_ibv_cq)),
    ('qp', ctypes.POINTER(struct_ibv_qp)),
    ('srq', ctypes.POINTER(struct_ibv_srq)),
    ('wq', ctypes.POINTER(struct_ibv_wq)),
    ('port_num', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_ibv_async_event._pack_ = 1 # source:False
struct_ibv_async_event._fields_ = [
    ('element', union_ibv_async_event_element),
    ('event_type', ibv_event_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
]


# values for enumeration 'ibv_wq_state'
ibv_wq_state__enumvalues = {
    0: 'IBV_WQS_RESET',
    1: 'IBV_WQS_RDY',
    2: 'IBV_WQS_ERR',
    3: 'IBV_WQS_UNKNOWN',
}
IBV_WQS_RESET = 0
IBV_WQS_RDY = 1
IBV_WQS_ERR = 2
IBV_WQS_UNKNOWN = 3
ibv_wq_state = ctypes.c_uint32 # enum

# values for enumeration 'ibv_wq_type'
ibv_wq_type__enumvalues = {
    0: 'IBV_WQT_RQ',
}
IBV_WQT_RQ = 0
ibv_wq_type = ctypes.c_uint32 # enum
struct_ibv_wq._pack_ = 1 # source:False
struct_ibv_wq._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('wq_context', ctypes.POINTER(None)),
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('cq', ctypes.POINTER(struct_ibv_cq)),
    ('wq_num', ctypes.c_uint32),
    ('handle', ctypes.c_uint32),
    ('state', ibv_wq_state),
    ('wq_type', ibv_wq_type),
    ('post_recv', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_wq), ctypes.POINTER(struct_ibv_recv_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_recv_wr)))),
    ('mutex', union_c__UA_pthread_mutex_t),
    ('cond', union_c__UA_pthread_cond_t),
    ('events_completed', ctypes.c_uint32),
    ('comp_mask', ctypes.c_uint32),
]

try:
    ibv_wc_status_str = _libraries['libibverbs'].ibv_wc_status_str
    ibv_wc_status_str.restype = ctypes.POINTER(ctypes.c_char)
    ibv_wc_status_str.argtypes = [ibv_wc_status]
except AttributeError:
    pass

# values for enumeration 'c__Ea_IBV_WC_IP_CSUM_OK_SHIFT'
c__Ea_IBV_WC_IP_CSUM_OK_SHIFT__enumvalues = {
    2: 'IBV_WC_IP_CSUM_OK_SHIFT',
}
IBV_WC_IP_CSUM_OK_SHIFT = 2
c__Ea_IBV_WC_IP_CSUM_OK_SHIFT = ctypes.c_uint32 # enum

# values for enumeration 'ibv_create_cq_wc_flags'
ibv_create_cq_wc_flags__enumvalues = {
    1: 'IBV_WC_EX_WITH_BYTE_LEN',
    2: 'IBV_WC_EX_WITH_IMM',
    4: 'IBV_WC_EX_WITH_QP_NUM',
    8: 'IBV_WC_EX_WITH_SRC_QP',
    16: 'IBV_WC_EX_WITH_SLID',
    32: 'IBV_WC_EX_WITH_SL',
    64: 'IBV_WC_EX_WITH_DLID_PATH_BITS',
    128: 'IBV_WC_EX_WITH_COMPLETION_TIMESTAMP',
    256: 'IBV_WC_EX_WITH_CVLAN',
    512: 'IBV_WC_EX_WITH_FLOW_TAG',
    1024: 'IBV_WC_EX_WITH_TM_INFO',
    2048: 'IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK',
}
IBV_WC_EX_WITH_BYTE_LEN = 1
IBV_WC_EX_WITH_IMM = 2
IBV_WC_EX_WITH_QP_NUM = 4
IBV_WC_EX_WITH_SRC_QP = 8
IBV_WC_EX_WITH_SLID = 16
IBV_WC_EX_WITH_SL = 32
IBV_WC_EX_WITH_DLID_PATH_BITS = 64
IBV_WC_EX_WITH_COMPLETION_TIMESTAMP = 128
IBV_WC_EX_WITH_CVLAN = 256
IBV_WC_EX_WITH_FLOW_TAG = 512
IBV_WC_EX_WITH_TM_INFO = 1024
IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK = 2048
ibv_create_cq_wc_flags = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_IBV_WC_STANDARD_FLAGS'
c__Ea_IBV_WC_STANDARD_FLAGS__enumvalues = {
    127: 'IBV_WC_STANDARD_FLAGS',
}
IBV_WC_STANDARD_FLAGS = 127
c__Ea_IBV_WC_STANDARD_FLAGS = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_IBV_CREATE_CQ_SUP_WC_FLAGS'
c__Ea_IBV_CREATE_CQ_SUP_WC_FLAGS__enumvalues = {
    4095: 'IBV_CREATE_CQ_SUP_WC_FLAGS',
}
IBV_CREATE_CQ_SUP_WC_FLAGS = 4095
c__Ea_IBV_CREATE_CQ_SUP_WC_FLAGS = ctypes.c_uint32 # enum

# values for enumeration 'ibv_wc_flags'
ibv_wc_flags__enumvalues = {
    1: 'IBV_WC_GRH',
    2: 'IBV_WC_WITH_IMM',
    4: 'IBV_WC_IP_CSUM_OK',
    8: 'IBV_WC_WITH_INV',
    16: 'IBV_WC_TM_SYNC_REQ',
    32: 'IBV_WC_TM_MATCH',
    64: 'IBV_WC_TM_DATA_VALID',
}
IBV_WC_GRH = 1
IBV_WC_WITH_IMM = 2
IBV_WC_IP_CSUM_OK = 4
IBV_WC_WITH_INV = 8
IBV_WC_TM_SYNC_REQ = 16
IBV_WC_TM_MATCH = 32
IBV_WC_TM_DATA_VALID = 64
ibv_wc_flags = ctypes.c_uint32 # enum

# values for enumeration 'ibv_access_flags'
ibv_access_flags__enumvalues = {
    1: 'IBV_ACCESS_LOCAL_WRITE',
    2: 'IBV_ACCESS_REMOTE_WRITE',
    4: 'IBV_ACCESS_REMOTE_READ',
    8: 'IBV_ACCESS_REMOTE_ATOMIC',
    16: 'IBV_ACCESS_MW_BIND',
    32: 'IBV_ACCESS_ZERO_BASED',
    64: 'IBV_ACCESS_ON_DEMAND',
    128: 'IBV_ACCESS_HUGETLB',
    256: 'IBV_ACCESS_FLUSH_GLOBAL',
    512: 'IBV_ACCESS_FLUSH_PERSISTENT',
    1048576: 'IBV_ACCESS_RELAXED_ORDERING',
}
IBV_ACCESS_LOCAL_WRITE = 1
IBV_ACCESS_REMOTE_WRITE = 2
IBV_ACCESS_REMOTE_READ = 4
IBV_ACCESS_REMOTE_ATOMIC = 8
IBV_ACCESS_MW_BIND = 16
IBV_ACCESS_ZERO_BASED = 32
IBV_ACCESS_ON_DEMAND = 64
IBV_ACCESS_HUGETLB = 128
IBV_ACCESS_FLUSH_GLOBAL = 256
IBV_ACCESS_FLUSH_PERSISTENT = 512
IBV_ACCESS_RELAXED_ORDERING = 1048576
ibv_access_flags = ctypes.c_uint32 # enum
class struct_ibv_td_init_attr(Structure):
    pass

struct_ibv_td_init_attr._pack_ = 1 # source:False
struct_ibv_td_init_attr._fields_ = [
    ('comp_mask', ctypes.c_uint32),
]

class struct_ibv_td(Structure):
    pass

struct_ibv_td._pack_ = 1 # source:False
struct_ibv_td._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
]


# values for enumeration 'ibv_xrcd_init_attr_mask'
ibv_xrcd_init_attr_mask__enumvalues = {
    1: 'IBV_XRCD_INIT_ATTR_FD',
    2: 'IBV_XRCD_INIT_ATTR_OFLAGS',
    4: 'IBV_XRCD_INIT_ATTR_RESERVED',
}
IBV_XRCD_INIT_ATTR_FD = 1
IBV_XRCD_INIT_ATTR_OFLAGS = 2
IBV_XRCD_INIT_ATTR_RESERVED = 4
ibv_xrcd_init_attr_mask = ctypes.c_uint32 # enum
class struct_ibv_xrcd_init_attr(Structure):
    pass

struct_ibv_xrcd_init_attr._pack_ = 1 # source:False
struct_ibv_xrcd_init_attr._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('fd', ctypes.c_int32),
    ('oflags', ctypes.c_int32),
]

class struct_ibv_xrcd(Structure):
    pass

struct_ibv_xrcd._pack_ = 1 # source:False
struct_ibv_xrcd._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
]


# values for enumeration 'ibv_rereg_mr_flags'
ibv_rereg_mr_flags__enumvalues = {
    1: 'IBV_REREG_MR_CHANGE_TRANSLATION',
    2: 'IBV_REREG_MR_CHANGE_PD',
    4: 'IBV_REREG_MR_CHANGE_ACCESS',
    7: 'IBV_REREG_MR_FLAGS_SUPPORTED',
}
IBV_REREG_MR_CHANGE_TRANSLATION = 1
IBV_REREG_MR_CHANGE_PD = 2
IBV_REREG_MR_CHANGE_ACCESS = 4
IBV_REREG_MR_FLAGS_SUPPORTED = 7
ibv_rereg_mr_flags = ctypes.c_uint32 # enum
class struct_ibv_global_route(Structure):
    pass

struct_ibv_global_route._pack_ = 1 # source:False
struct_ibv_global_route._fields_ = [
    ('dgid', union_ibv_gid),
    ('flow_label', ctypes.c_uint32),
    ('sgid_index', ctypes.c_ubyte),
    ('hop_limit', ctypes.c_ubyte),
    ('traffic_class', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
]

class struct_ibv_grh(Structure):
    pass

struct_ibv_grh._pack_ = 1 # source:False
struct_ibv_grh._fields_ = [
    ('version_tclass_flow', ctypes.c_uint32),
    ('paylen', ctypes.c_uint16),
    ('next_hdr', ctypes.c_ubyte),
    ('hop_limit', ctypes.c_ubyte),
    ('sgid', union_ibv_gid),
    ('dgid', union_ibv_gid),
]


# values for enumeration 'ibv_rate'
ibv_rate__enumvalues = {
    0: 'IBV_RATE_MAX',
    2: 'IBV_RATE_2_5_GBPS',
    5: 'IBV_RATE_5_GBPS',
    3: 'IBV_RATE_10_GBPS',
    6: 'IBV_RATE_20_GBPS',
    4: 'IBV_RATE_30_GBPS',
    7: 'IBV_RATE_40_GBPS',
    8: 'IBV_RATE_60_GBPS',
    9: 'IBV_RATE_80_GBPS',
    10: 'IBV_RATE_120_GBPS',
    11: 'IBV_RATE_14_GBPS',
    12: 'IBV_RATE_56_GBPS',
    13: 'IBV_RATE_112_GBPS',
    14: 'IBV_RATE_168_GBPS',
    15: 'IBV_RATE_25_GBPS',
    16: 'IBV_RATE_100_GBPS',
    17: 'IBV_RATE_200_GBPS',
    18: 'IBV_RATE_300_GBPS',
    19: 'IBV_RATE_28_GBPS',
    20: 'IBV_RATE_50_GBPS',
    21: 'IBV_RATE_400_GBPS',
    22: 'IBV_RATE_600_GBPS',
    23: 'IBV_RATE_800_GBPS',
    24: 'IBV_RATE_1200_GBPS',
}
IBV_RATE_MAX = 0
IBV_RATE_2_5_GBPS = 2
IBV_RATE_5_GBPS = 5
IBV_RATE_10_GBPS = 3
IBV_RATE_20_GBPS = 6
IBV_RATE_30_GBPS = 4
IBV_RATE_40_GBPS = 7
IBV_RATE_60_GBPS = 8
IBV_RATE_80_GBPS = 9
IBV_RATE_120_GBPS = 10
IBV_RATE_14_GBPS = 11
IBV_RATE_56_GBPS = 12
IBV_RATE_112_GBPS = 13
IBV_RATE_168_GBPS = 14
IBV_RATE_25_GBPS = 15
IBV_RATE_100_GBPS = 16
IBV_RATE_200_GBPS = 17
IBV_RATE_300_GBPS = 18
IBV_RATE_28_GBPS = 19
IBV_RATE_50_GBPS = 20
IBV_RATE_400_GBPS = 21
IBV_RATE_600_GBPS = 22
IBV_RATE_800_GBPS = 23
IBV_RATE_1200_GBPS = 24
ibv_rate = ctypes.c_uint32 # enum
try:
    ibv_rate_to_mult = _libraries['libibverbs'].ibv_rate_to_mult
    ibv_rate_to_mult.restype = ctypes.c_int32
    ibv_rate_to_mult.argtypes = [ibv_rate]
except AttributeError:
    pass
try:
    mult_to_ibv_rate = _libraries['libibverbs'].mult_to_ibv_rate
    mult_to_ibv_rate.restype = ibv_rate
    mult_to_ibv_rate.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    ibv_rate_to_mbps = _libraries['libibverbs'].ibv_rate_to_mbps
    ibv_rate_to_mbps.restype = ctypes.c_int32
    ibv_rate_to_mbps.argtypes = [ibv_rate]
except AttributeError:
    pass
try:
    mbps_to_ibv_rate = _libraries['libibverbs'].mbps_to_ibv_rate
    mbps_to_ibv_rate.restype = ibv_rate
    mbps_to_ibv_rate.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
class struct_ibv_ah_attr(Structure):
    pass

struct_ibv_ah_attr._pack_ = 1 # source:False
struct_ibv_ah_attr._fields_ = [
    ('grh', struct_ibv_global_route),
    ('dlid', ctypes.c_uint16),
    ('sl', ctypes.c_ubyte),
    ('src_path_bits', ctypes.c_ubyte),
    ('static_rate', ctypes.c_ubyte),
    ('is_global', ctypes.c_ubyte),
    ('port_num', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
]


# values for enumeration 'ibv_srq_attr_mask'
ibv_srq_attr_mask__enumvalues = {
    1: 'IBV_SRQ_MAX_WR',
    2: 'IBV_SRQ_LIMIT',
}
IBV_SRQ_MAX_WR = 1
IBV_SRQ_LIMIT = 2
ibv_srq_attr_mask = ctypes.c_uint32 # enum
class struct_ibv_srq_attr(Structure):
    pass

struct_ibv_srq_attr._pack_ = 1 # source:False
struct_ibv_srq_attr._fields_ = [
    ('max_wr', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('srq_limit', ctypes.c_uint32),
]

class struct_ibv_srq_init_attr(Structure):
    pass

struct_ibv_srq_init_attr._pack_ = 1 # source:False
struct_ibv_srq_init_attr._fields_ = [
    ('srq_context', ctypes.POINTER(None)),
    ('attr', struct_ibv_srq_attr),
    ('PADDING_0', ctypes.c_ubyte * 4),
]


# values for enumeration 'ibv_srq_type'
ibv_srq_type__enumvalues = {
    0: 'IBV_SRQT_BASIC',
    1: 'IBV_SRQT_XRC',
    2: 'IBV_SRQT_TM',
}
IBV_SRQT_BASIC = 0
IBV_SRQT_XRC = 1
IBV_SRQT_TM = 2
ibv_srq_type = ctypes.c_uint32 # enum

# values for enumeration 'ibv_srq_init_attr_mask'
ibv_srq_init_attr_mask__enumvalues = {
    1: 'IBV_SRQ_INIT_ATTR_TYPE',
    2: 'IBV_SRQ_INIT_ATTR_PD',
    4: 'IBV_SRQ_INIT_ATTR_XRCD',
    8: 'IBV_SRQ_INIT_ATTR_CQ',
    16: 'IBV_SRQ_INIT_ATTR_TM',
    32: 'IBV_SRQ_INIT_ATTR_RESERVED',
}
IBV_SRQ_INIT_ATTR_TYPE = 1
IBV_SRQ_INIT_ATTR_PD = 2
IBV_SRQ_INIT_ATTR_XRCD = 4
IBV_SRQ_INIT_ATTR_CQ = 8
IBV_SRQ_INIT_ATTR_TM = 16
IBV_SRQ_INIT_ATTR_RESERVED = 32
ibv_srq_init_attr_mask = ctypes.c_uint32 # enum
class struct_ibv_tm_cap(Structure):
    pass

struct_ibv_tm_cap._pack_ = 1 # source:False
struct_ibv_tm_cap._fields_ = [
    ('max_num_tags', ctypes.c_uint32),
    ('max_ops', ctypes.c_uint32),
]

class struct_ibv_srq_init_attr_ex(Structure):
    pass

struct_ibv_srq_init_attr_ex._pack_ = 1 # source:False
struct_ibv_srq_init_attr_ex._fields_ = [
    ('srq_context', ctypes.POINTER(None)),
    ('attr', struct_ibv_srq_attr),
    ('comp_mask', ctypes.c_uint32),
    ('srq_type', ibv_srq_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('xrcd', ctypes.POINTER(struct_ibv_xrcd)),
    ('cq', ctypes.POINTER(struct_ibv_cq)),
    ('tm_cap', struct_ibv_tm_cap),
]


# values for enumeration 'ibv_wq_init_attr_mask'
ibv_wq_init_attr_mask__enumvalues = {
    1: 'IBV_WQ_INIT_ATTR_FLAGS',
    2: 'IBV_WQ_INIT_ATTR_RESERVED',
}
IBV_WQ_INIT_ATTR_FLAGS = 1
IBV_WQ_INIT_ATTR_RESERVED = 2
ibv_wq_init_attr_mask = ctypes.c_uint32 # enum

# values for enumeration 'ibv_wq_flags'
ibv_wq_flags__enumvalues = {
    1: 'IBV_WQ_FLAGS_CVLAN_STRIPPING',
    2: 'IBV_WQ_FLAGS_SCATTER_FCS',
    4: 'IBV_WQ_FLAGS_DELAY_DROP',
    8: 'IBV_WQ_FLAGS_PCI_WRITE_END_PADDING',
    16: 'IBV_WQ_FLAGS_RESERVED',
}
IBV_WQ_FLAGS_CVLAN_STRIPPING = 1
IBV_WQ_FLAGS_SCATTER_FCS = 2
IBV_WQ_FLAGS_DELAY_DROP = 4
IBV_WQ_FLAGS_PCI_WRITE_END_PADDING = 8
IBV_WQ_FLAGS_RESERVED = 16
ibv_wq_flags = ctypes.c_uint32 # enum
class struct_ibv_wq_init_attr(Structure):
    pass

struct_ibv_wq_init_attr._pack_ = 1 # source:False
struct_ibv_wq_init_attr._fields_ = [
    ('wq_context', ctypes.POINTER(None)),
    ('wq_type', ibv_wq_type),
    ('max_wr', ctypes.c_uint32),
    ('max_sge', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('cq', ctypes.POINTER(struct_ibv_cq)),
    ('comp_mask', ctypes.c_uint32),
    ('create_flags', ctypes.c_uint32),
]


# values for enumeration 'ibv_wq_attr_mask'
ibv_wq_attr_mask__enumvalues = {
    1: 'IBV_WQ_ATTR_STATE',
    2: 'IBV_WQ_ATTR_CURR_STATE',
    4: 'IBV_WQ_ATTR_FLAGS',
    8: 'IBV_WQ_ATTR_RESERVED',
}
IBV_WQ_ATTR_STATE = 1
IBV_WQ_ATTR_CURR_STATE = 2
IBV_WQ_ATTR_FLAGS = 4
IBV_WQ_ATTR_RESERVED = 8
ibv_wq_attr_mask = ctypes.c_uint32 # enum
class struct_ibv_wq_attr(Structure):
    pass

struct_ibv_wq_attr._pack_ = 1 # source:False
struct_ibv_wq_attr._fields_ = [
    ('attr_mask', ctypes.c_uint32),
    ('wq_state', ibv_wq_state),
    ('curr_wq_state', ibv_wq_state),
    ('flags', ctypes.c_uint32),
    ('flags_mask', ctypes.c_uint32),
]

class struct_ibv_rwq_ind_table(Structure):
    pass

struct_ibv_rwq_ind_table._pack_ = 1 # source:False
struct_ibv_rwq_ind_table._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('ind_tbl_handle', ctypes.c_int32),
    ('ind_tbl_num', ctypes.c_int32),
    ('comp_mask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]


# values for enumeration 'ibv_ind_table_init_attr_mask'
ibv_ind_table_init_attr_mask__enumvalues = {
    1: 'IBV_CREATE_IND_TABLE_RESERVED',
}
IBV_CREATE_IND_TABLE_RESERVED = 1
ibv_ind_table_init_attr_mask = ctypes.c_uint32 # enum
class struct_ibv_rwq_ind_table_init_attr(Structure):
    pass

struct_ibv_rwq_ind_table_init_attr._pack_ = 1 # source:False
struct_ibv_rwq_ind_table_init_attr._fields_ = [
    ('log_ind_tbl_size', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('ind_tbl', ctypes.POINTER(ctypes.POINTER(struct_ibv_wq))),
    ('comp_mask', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_ibv_qp_cap(Structure):
    pass

struct_ibv_qp_cap._pack_ = 1 # source:False
struct_ibv_qp_cap._fields_ = [
    ('max_send_wr', ctypes.c_uint32),
    ('max_recv_wr', ctypes.c_uint32),
    ('max_send_sge', ctypes.c_uint32),
    ('max_recv_sge', ctypes.c_uint32),
    ('max_inline_data', ctypes.c_uint32),
]

class struct_ibv_qp_init_attr(Structure):
    pass

struct_ibv_qp_init_attr._pack_ = 1 # source:False
struct_ibv_qp_init_attr._fields_ = [
    ('qp_context', ctypes.POINTER(None)),
    ('send_cq', ctypes.POINTER(struct_ibv_cq)),
    ('recv_cq', ctypes.POINTER(struct_ibv_cq)),
    ('srq', ctypes.POINTER(struct_ibv_srq)),
    ('cap', struct_ibv_qp_cap),
    ('qp_type', ibv_qp_type),
    ('sq_sig_all', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]


# values for enumeration 'ibv_qp_init_attr_mask'
ibv_qp_init_attr_mask__enumvalues = {
    1: 'IBV_QP_INIT_ATTR_PD',
    2: 'IBV_QP_INIT_ATTR_XRCD',
    4: 'IBV_QP_INIT_ATTR_CREATE_FLAGS',
    8: 'IBV_QP_INIT_ATTR_MAX_TSO_HEADER',
    16: 'IBV_QP_INIT_ATTR_IND_TABLE',
    32: 'IBV_QP_INIT_ATTR_RX_HASH',
    64: 'IBV_QP_INIT_ATTR_SEND_OPS_FLAGS',
}
IBV_QP_INIT_ATTR_PD = 1
IBV_QP_INIT_ATTR_XRCD = 2
IBV_QP_INIT_ATTR_CREATE_FLAGS = 4
IBV_QP_INIT_ATTR_MAX_TSO_HEADER = 8
IBV_QP_INIT_ATTR_IND_TABLE = 16
IBV_QP_INIT_ATTR_RX_HASH = 32
IBV_QP_INIT_ATTR_SEND_OPS_FLAGS = 64
ibv_qp_init_attr_mask = ctypes.c_uint32 # enum

# values for enumeration 'ibv_qp_create_flags'
ibv_qp_create_flags__enumvalues = {
    2: 'IBV_QP_CREATE_BLOCK_SELF_MCAST_LB',
    256: 'IBV_QP_CREATE_SCATTER_FCS',
    512: 'IBV_QP_CREATE_CVLAN_STRIPPING',
    1024: 'IBV_QP_CREATE_SOURCE_QPN',
    2048: 'IBV_QP_CREATE_PCI_WRITE_END_PADDING',
}
IBV_QP_CREATE_BLOCK_SELF_MCAST_LB = 2
IBV_QP_CREATE_SCATTER_FCS = 256
IBV_QP_CREATE_CVLAN_STRIPPING = 512
IBV_QP_CREATE_SOURCE_QPN = 1024
IBV_QP_CREATE_PCI_WRITE_END_PADDING = 2048
ibv_qp_create_flags = ctypes.c_uint32 # enum

# values for enumeration 'ibv_qp_create_send_ops_flags'
ibv_qp_create_send_ops_flags__enumvalues = {
    1: 'IBV_QP_EX_WITH_RDMA_WRITE',
    2: 'IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM',
    4: 'IBV_QP_EX_WITH_SEND',
    8: 'IBV_QP_EX_WITH_SEND_WITH_IMM',
    16: 'IBV_QP_EX_WITH_RDMA_READ',
    32: 'IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP',
    64: 'IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD',
    128: 'IBV_QP_EX_WITH_LOCAL_INV',
    256: 'IBV_QP_EX_WITH_BIND_MW',
    512: 'IBV_QP_EX_WITH_SEND_WITH_INV',
    1024: 'IBV_QP_EX_WITH_TSO',
    2048: 'IBV_QP_EX_WITH_FLUSH',
    4096: 'IBV_QP_EX_WITH_ATOMIC_WRITE',
}
IBV_QP_EX_WITH_RDMA_WRITE = 1
IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM = 2
IBV_QP_EX_WITH_SEND = 4
IBV_QP_EX_WITH_SEND_WITH_IMM = 8
IBV_QP_EX_WITH_RDMA_READ = 16
IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP = 32
IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD = 64
IBV_QP_EX_WITH_LOCAL_INV = 128
IBV_QP_EX_WITH_BIND_MW = 256
IBV_QP_EX_WITH_SEND_WITH_INV = 512
IBV_QP_EX_WITH_TSO = 1024
IBV_QP_EX_WITH_FLUSH = 2048
IBV_QP_EX_WITH_ATOMIC_WRITE = 4096
ibv_qp_create_send_ops_flags = ctypes.c_uint32 # enum
class struct_ibv_rx_hash_conf(Structure):
    pass

struct_ibv_rx_hash_conf._pack_ = 1 # source:False
struct_ibv_rx_hash_conf._fields_ = [
    ('rx_hash_function', ctypes.c_ubyte),
    ('rx_hash_key_len', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 6),
    ('rx_hash_key', ctypes.POINTER(ctypes.c_ubyte)),
    ('rx_hash_fields_mask', ctypes.c_uint64),
]

class struct_ibv_qp_init_attr_ex(Structure):
    pass

struct_ibv_qp_init_attr_ex._pack_ = 1 # source:False
struct_ibv_qp_init_attr_ex._fields_ = [
    ('qp_context', ctypes.POINTER(None)),
    ('send_cq', ctypes.POINTER(struct_ibv_cq)),
    ('recv_cq', ctypes.POINTER(struct_ibv_cq)),
    ('srq', ctypes.POINTER(struct_ibv_srq)),
    ('cap', struct_ibv_qp_cap),
    ('qp_type', ibv_qp_type),
    ('sq_sig_all', ctypes.c_int32),
    ('comp_mask', ctypes.c_uint32),
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('xrcd', ctypes.POINTER(struct_ibv_xrcd)),
    ('create_flags', ctypes.c_uint32),
    ('max_tso_header', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('rwq_ind_tbl', ctypes.POINTER(struct_ibv_rwq_ind_table)),
    ('rx_hash_conf', struct_ibv_rx_hash_conf),
    ('source_qpn', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('send_ops_flags', ctypes.c_uint64),
]


# values for enumeration 'ibv_qp_open_attr_mask'
ibv_qp_open_attr_mask__enumvalues = {
    1: 'IBV_QP_OPEN_ATTR_NUM',
    2: 'IBV_QP_OPEN_ATTR_XRCD',
    4: 'IBV_QP_OPEN_ATTR_CONTEXT',
    8: 'IBV_QP_OPEN_ATTR_TYPE',
    16: 'IBV_QP_OPEN_ATTR_RESERVED',
}
IBV_QP_OPEN_ATTR_NUM = 1
IBV_QP_OPEN_ATTR_XRCD = 2
IBV_QP_OPEN_ATTR_CONTEXT = 4
IBV_QP_OPEN_ATTR_TYPE = 8
IBV_QP_OPEN_ATTR_RESERVED = 16
ibv_qp_open_attr_mask = ctypes.c_uint32 # enum
class struct_ibv_qp_open_attr(Structure):
    pass

struct_ibv_qp_open_attr._pack_ = 1 # source:False
struct_ibv_qp_open_attr._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('qp_num', ctypes.c_uint32),
    ('xrcd', ctypes.POINTER(struct_ibv_xrcd)),
    ('qp_context', ctypes.POINTER(None)),
    ('qp_type', ibv_qp_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
]


# values for enumeration 'ibv_qp_attr_mask'
ibv_qp_attr_mask__enumvalues = {
    1: 'IBV_QP_STATE',
    2: 'IBV_QP_CUR_STATE',
    4: 'IBV_QP_EN_SQD_ASYNC_NOTIFY',
    8: 'IBV_QP_ACCESS_FLAGS',
    16: 'IBV_QP_PKEY_INDEX',
    32: 'IBV_QP_PORT',
    64: 'IBV_QP_QKEY',
    128: 'IBV_QP_AV',
    256: 'IBV_QP_PATH_MTU',
    512: 'IBV_QP_TIMEOUT',
    1024: 'IBV_QP_RETRY_CNT',
    2048: 'IBV_QP_RNR_RETRY',
    4096: 'IBV_QP_RQ_PSN',
    8192: 'IBV_QP_MAX_QP_RD_ATOMIC',
    16384: 'IBV_QP_ALT_PATH',
    32768: 'IBV_QP_MIN_RNR_TIMER',
    65536: 'IBV_QP_SQ_PSN',
    131072: 'IBV_QP_MAX_DEST_RD_ATOMIC',
    262144: 'IBV_QP_PATH_MIG_STATE',
    524288: 'IBV_QP_CAP',
    1048576: 'IBV_QP_DEST_QPN',
    33554432: 'IBV_QP_RATE_LIMIT',
}
IBV_QP_STATE = 1
IBV_QP_CUR_STATE = 2
IBV_QP_EN_SQD_ASYNC_NOTIFY = 4
IBV_QP_ACCESS_FLAGS = 8
IBV_QP_PKEY_INDEX = 16
IBV_QP_PORT = 32
IBV_QP_QKEY = 64
IBV_QP_AV = 128
IBV_QP_PATH_MTU = 256
IBV_QP_TIMEOUT = 512
IBV_QP_RETRY_CNT = 1024
IBV_QP_RNR_RETRY = 2048
IBV_QP_RQ_PSN = 4096
IBV_QP_MAX_QP_RD_ATOMIC = 8192
IBV_QP_ALT_PATH = 16384
IBV_QP_MIN_RNR_TIMER = 32768
IBV_QP_SQ_PSN = 65536
IBV_QP_MAX_DEST_RD_ATOMIC = 131072
IBV_QP_PATH_MIG_STATE = 262144
IBV_QP_CAP = 524288
IBV_QP_DEST_QPN = 1048576
IBV_QP_RATE_LIMIT = 33554432
ibv_qp_attr_mask = ctypes.c_uint32 # enum

# values for enumeration 'ibv_query_qp_data_in_order_flags'
ibv_query_qp_data_in_order_flags__enumvalues = {
    1: 'IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS',
}
IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS = 1
ibv_query_qp_data_in_order_flags = ctypes.c_uint32 # enum

# values for enumeration 'ibv_query_qp_data_in_order_caps'
ibv_query_qp_data_in_order_caps__enumvalues = {
    1: 'IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG',
    2: 'IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES',
}
IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG = 1
IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES = 2
ibv_query_qp_data_in_order_caps = ctypes.c_uint32 # enum

# values for enumeration 'ibv_mig_state'
ibv_mig_state__enumvalues = {
    0: 'IBV_MIG_MIGRATED',
    1: 'IBV_MIG_REARM',
    2: 'IBV_MIG_ARMED',
}
IBV_MIG_MIGRATED = 0
IBV_MIG_REARM = 1
IBV_MIG_ARMED = 2
ibv_mig_state = ctypes.c_uint32 # enum
class struct_ibv_qp_attr(Structure):
    pass

struct_ibv_qp_attr._pack_ = 1 # source:False
struct_ibv_qp_attr._fields_ = [
    ('qp_state', ibv_qp_state),
    ('cur_qp_state', ibv_qp_state),
    ('path_mtu', ibv_mtu),
    ('path_mig_state', ibv_mig_state),
    ('qkey', ctypes.c_uint32),
    ('rq_psn', ctypes.c_uint32),
    ('sq_psn', ctypes.c_uint32),
    ('dest_qp_num', ctypes.c_uint32),
    ('qp_access_flags', ctypes.c_uint32),
    ('cap', struct_ibv_qp_cap),
    ('ah_attr', struct_ibv_ah_attr),
    ('alt_ah_attr', struct_ibv_ah_attr),
    ('pkey_index', ctypes.c_uint16),
    ('alt_pkey_index', ctypes.c_uint16),
    ('en_sqd_async_notify', ctypes.c_ubyte),
    ('sq_draining', ctypes.c_ubyte),
    ('max_rd_atomic', ctypes.c_ubyte),
    ('max_dest_rd_atomic', ctypes.c_ubyte),
    ('min_rnr_timer', ctypes.c_ubyte),
    ('port_num', ctypes.c_ubyte),
    ('timeout', ctypes.c_ubyte),
    ('retry_cnt', ctypes.c_ubyte),
    ('rnr_retry', ctypes.c_ubyte),
    ('alt_port_num', ctypes.c_ubyte),
    ('alt_timeout', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('rate_limit', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_ibv_qp_rate_limit_attr(Structure):
    pass

struct_ibv_qp_rate_limit_attr._pack_ = 1 # source:False
struct_ibv_qp_rate_limit_attr._fields_ = [
    ('rate_limit', ctypes.c_uint32),
    ('max_burst_sz', ctypes.c_uint32),
    ('typical_pkt_sz', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('comp_mask', ctypes.c_uint32),
]

try:
    ibv_wr_opcode_str = _libraries['libibverbs'].ibv_wr_opcode_str
    ibv_wr_opcode_str.restype = ctypes.POINTER(ctypes.c_char)
    ibv_wr_opcode_str.argtypes = [ibv_wr_opcode]
except AttributeError:
    pass

# values for enumeration 'ibv_send_flags'
ibv_send_flags__enumvalues = {
    1: 'IBV_SEND_FENCE',
    2: 'IBV_SEND_SIGNALED',
    4: 'IBV_SEND_SOLICITED',
    8: 'IBV_SEND_INLINE',
    16: 'IBV_SEND_IP_CSUM',
}
IBV_SEND_FENCE = 1
IBV_SEND_SIGNALED = 2
IBV_SEND_SOLICITED = 4
IBV_SEND_INLINE = 8
IBV_SEND_IP_CSUM = 16
ibv_send_flags = ctypes.c_uint32 # enum

# values for enumeration 'ibv_placement_type'
ibv_placement_type__enumvalues = {
    1: 'IBV_FLUSH_GLOBAL',
    2: 'IBV_FLUSH_PERSISTENT',
}
IBV_FLUSH_GLOBAL = 1
IBV_FLUSH_PERSISTENT = 2
ibv_placement_type = ctypes.c_uint32 # enum

# values for enumeration 'ibv_selectivity_level'
ibv_selectivity_level__enumvalues = {
    0: 'IBV_FLUSH_RANGE',
    1: 'IBV_FLUSH_MR',
}
IBV_FLUSH_RANGE = 0
IBV_FLUSH_MR = 1
ibv_selectivity_level = ctypes.c_uint32 # enum
class struct_ibv_data_buf(Structure):
    pass

struct_ibv_data_buf._pack_ = 1 # source:False
struct_ibv_data_buf._fields_ = [
    ('addr', ctypes.POINTER(None)),
    ('length', ctypes.c_uint64),
]


# values for enumeration 'ibv_ops_wr_opcode'
ibv_ops_wr_opcode__enumvalues = {
    0: 'IBV_WR_TAG_ADD',
    1: 'IBV_WR_TAG_DEL',
    2: 'IBV_WR_TAG_SYNC',
}
IBV_WR_TAG_ADD = 0
IBV_WR_TAG_DEL = 1
IBV_WR_TAG_SYNC = 2
ibv_ops_wr_opcode = ctypes.c_uint32 # enum

# values for enumeration 'ibv_ops_flags'
ibv_ops_flags__enumvalues = {
    1: 'IBV_OPS_SIGNALED',
    2: 'IBV_OPS_TM_SYNC',
}
IBV_OPS_SIGNALED = 1
IBV_OPS_TM_SYNC = 2
ibv_ops_flags = ctypes.c_uint32 # enum
class struct_ibv_ops_wr(Structure):
    pass

class struct_ibv_ops_wr_tm(Structure):
    pass

class struct_ibv_ops_wr_0_add(Structure):
    pass

struct_ibv_ops_wr_0_add._pack_ = 1 # source:False
struct_ibv_ops_wr_0_add._fields_ = [
    ('recv_wr_id', ctypes.c_uint64),
    ('sg_list', ctypes.POINTER(struct_ibv_sge)),
    ('num_sge', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('tag', ctypes.c_uint64),
    ('mask', ctypes.c_uint64),
]

struct_ibv_ops_wr_tm._pack_ = 1 # source:False
struct_ibv_ops_wr_tm._fields_ = [
    ('unexpected_cnt', ctypes.c_uint32),
    ('handle', ctypes.c_uint32),
    ('add', struct_ibv_ops_wr_0_add),
]

struct_ibv_ops_wr._pack_ = 1 # source:False
struct_ibv_ops_wr._fields_ = [
    ('wr_id', ctypes.c_uint64),
    ('next', ctypes.POINTER(struct_ibv_ops_wr)),
    ('opcode', ibv_ops_wr_opcode),
    ('flags', ctypes.c_int32),
    ('tm', struct_ibv_ops_wr_tm),
]

class struct_ibv_qp_ex(Structure):
    pass

struct_ibv_qp_ex._pack_ = 1 # source:False
struct_ibv_qp_ex._fields_ = [
    ('qp_base', struct_ibv_qp),
    ('comp_mask', ctypes.c_uint64),
    ('wr_id', ctypes.c_uint64),
    ('wr_flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('wr_atomic_cmp_swp', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)),
    ('wr_atomic_fetch_add', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64)),
    ('wr_bind_mw', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(struct_ibv_mw), ctypes.c_uint32, ctypes.POINTER(struct_ibv_mw_bind_info))),
    ('wr_local_inv', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32)),
    ('wr_rdma_read', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32, ctypes.c_uint64)),
    ('wr_rdma_write', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32, ctypes.c_uint64)),
    ('wr_rdma_write_imm', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32)),
    ('wr_send', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex))),
    ('wr_send_imm', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32)),
    ('wr_send_inv', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32)),
    ('wr_send_tso', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(None), ctypes.c_uint16, ctypes.c_uint16)),
    ('wr_set_ud_addr', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(struct_ibv_ah), ctypes.c_uint32, ctypes.c_uint32)),
    ('wr_set_xrc_srqn', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32)),
    ('wr_set_inline_data', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(None), ctypes.c_uint64)),
    ('wr_set_inline_data_list', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint64, ctypes.POINTER(struct_ibv_data_buf))),
    ('wr_set_sge', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32)),
    ('wr_set_sge_list', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint64, ctypes.POINTER(struct_ibv_sge))),
    ('wr_start', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex))),
    ('wr_complete', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp_ex))),
    ('wr_abort', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex))),
    ('wr_atomic_write', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(None))),
    ('wr_flush', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_ubyte)),
]

try:
    ibv_qp_to_qp_ex = _libraries['libibverbs'].ibv_qp_to_qp_ex
    ibv_qp_to_qp_ex.restype = ctypes.POINTER(struct_ibv_qp_ex)
    ibv_qp_to_qp_ex.argtypes = [ctypes.POINTER(struct_ibv_qp)]
except AttributeError:
    pass
uint32_t = ctypes.c_uint32
uint64_t = ctypes.c_uint64
try:
    ibv_wr_atomic_cmp_swp = _libraries['libibverbs'].ibv_wr_atomic_cmp_swp
    ibv_wr_atomic_cmp_swp.restype = None
    ibv_wr_atomic_cmp_swp.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, uint64_t, uint64_t]
except AttributeError:
    pass
try:
    ibv_wr_atomic_fetch_add = _libraries['libibverbs'].ibv_wr_atomic_fetch_add
    ibv_wr_atomic_fetch_add.restype = None
    ibv_wr_atomic_fetch_add.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, uint64_t]
except AttributeError:
    pass
try:
    ibv_wr_bind_mw = _libraries['libibverbs'].ibv_wr_bind_mw
    ibv_wr_bind_mw.restype = None
    ibv_wr_bind_mw.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(struct_ibv_mw), uint32_t, ctypes.POINTER(struct_ibv_mw_bind_info)]
except AttributeError:
    pass
try:
    ibv_wr_local_inv = _libraries['libibverbs'].ibv_wr_local_inv
    ibv_wr_local_inv.restype = None
    ibv_wr_local_inv.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t]
except AttributeError:
    pass
try:
    ibv_wr_rdma_read = _libraries['libibverbs'].ibv_wr_rdma_read
    ibv_wr_rdma_read.restype = None
    ibv_wr_rdma_read.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t]
except AttributeError:
    pass
try:
    ibv_wr_rdma_write = _libraries['libibverbs'].ibv_wr_rdma_write
    ibv_wr_rdma_write.restype = None
    ibv_wr_rdma_write.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t]
except AttributeError:
    pass
size_t = ctypes.c_uint64
uint8_t = ctypes.c_uint8
try:
    ibv_wr_flush = _libraries['libibverbs'].ibv_wr_flush
    ibv_wr_flush.restype = None
    ibv_wr_flush.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, size_t, uint8_t, uint8_t]
except AttributeError:
    pass
__be32 = ctypes.c_uint32
try:
    ibv_wr_rdma_write_imm = _libraries['libibverbs'].ibv_wr_rdma_write_imm
    ibv_wr_rdma_write_imm.restype = None
    ibv_wr_rdma_write_imm.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, __be32]
except AttributeError:
    pass
try:
    ibv_wr_send = _libraries['libibverbs'].ibv_wr_send
    ibv_wr_send.restype = None
    ibv_wr_send.argtypes = [ctypes.POINTER(struct_ibv_qp_ex)]
except AttributeError:
    pass
try:
    ibv_wr_send_imm = _libraries['libibverbs'].ibv_wr_send_imm
    ibv_wr_send_imm.restype = None
    ibv_wr_send_imm.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), __be32]
except AttributeError:
    pass
try:
    ibv_wr_send_inv = _libraries['libibverbs'].ibv_wr_send_inv
    ibv_wr_send_inv.restype = None
    ibv_wr_send_inv.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t]
except AttributeError:
    pass
uint16_t = ctypes.c_uint16
try:
    ibv_wr_send_tso = _libraries['libibverbs'].ibv_wr_send_tso
    ibv_wr_send_tso.restype = None
    ibv_wr_send_tso.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(None), uint16_t, uint16_t]
except AttributeError:
    pass
try:
    ibv_wr_set_ud_addr = _libraries['libibverbs'].ibv_wr_set_ud_addr
    ibv_wr_set_ud_addr.restype = None
    ibv_wr_set_ud_addr.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(struct_ibv_ah), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    ibv_wr_set_xrc_srqn = _libraries['libibverbs'].ibv_wr_set_xrc_srqn
    ibv_wr_set_xrc_srqn.restype = None
    ibv_wr_set_xrc_srqn.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t]
except AttributeError:
    pass
try:
    ibv_wr_set_inline_data = _libraries['libibverbs'].ibv_wr_set_inline_data
    ibv_wr_set_inline_data.restype = None
    ibv_wr_set_inline_data.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    ibv_wr_set_inline_data_list = _libraries['libibverbs'].ibv_wr_set_inline_data_list
    ibv_wr_set_inline_data_list.restype = None
    ibv_wr_set_inline_data_list.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), size_t, ctypes.POINTER(struct_ibv_data_buf)]
except AttributeError:
    pass
try:
    ibv_wr_set_sge = _libraries['libibverbs'].ibv_wr_set_sge
    ibv_wr_set_sge.restype = None
    ibv_wr_set_sge.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, uint32_t]
except AttributeError:
    pass
try:
    ibv_wr_set_sge_list = _libraries['libibverbs'].ibv_wr_set_sge_list
    ibv_wr_set_sge_list.restype = None
    ibv_wr_set_sge_list.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), size_t, ctypes.POINTER(struct_ibv_sge)]
except AttributeError:
    pass
try:
    ibv_wr_start = _libraries['libibverbs'].ibv_wr_start
    ibv_wr_start.restype = None
    ibv_wr_start.argtypes = [ctypes.POINTER(struct_ibv_qp_ex)]
except AttributeError:
    pass
try:
    ibv_wr_complete = _libraries['libibverbs'].ibv_wr_complete
    ibv_wr_complete.restype = ctypes.c_int32
    ibv_wr_complete.argtypes = [ctypes.POINTER(struct_ibv_qp_ex)]
except AttributeError:
    pass
try:
    ibv_wr_abort = _libraries['libibverbs'].ibv_wr_abort
    ibv_wr_abort.restype = None
    ibv_wr_abort.argtypes = [ctypes.POINTER(struct_ibv_qp_ex)]
except AttributeError:
    pass
try:
    ibv_wr_atomic_write = _libraries['libibverbs'].ibv_wr_atomic_write
    ibv_wr_atomic_write.restype = None
    ibv_wr_atomic_write.argtypes = [ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_ibv_ece(Structure):
    pass

struct_ibv_ece._pack_ = 1 # source:False
struct_ibv_ece._fields_ = [
    ('vendor_id', ctypes.c_uint32),
    ('options', ctypes.c_uint32),
    ('comp_mask', ctypes.c_uint32),
]

class struct_ibv_poll_cq_attr(Structure):
    pass

struct_ibv_poll_cq_attr._pack_ = 1 # source:False
struct_ibv_poll_cq_attr._fields_ = [
    ('comp_mask', ctypes.c_uint32),
]

class struct_ibv_wc_tm_info(Structure):
    pass

struct_ibv_wc_tm_info._pack_ = 1 # source:False
struct_ibv_wc_tm_info._fields_ = [
    ('tag', ctypes.c_uint64),
    ('priv', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_ibv_cq_ex(Structure):
    pass

struct_ibv_cq_ex._pack_ = 1 # source:False
struct_ibv_cq_ex._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('channel', ctypes.POINTER(struct_ibv_comp_channel)),
    ('cq_context', ctypes.POINTER(None)),
    ('handle', ctypes.c_uint32),
    ('cqe', ctypes.c_int32),
    ('mutex', union_c__UA_pthread_mutex_t),
    ('cond', union_c__UA_pthread_cond_t),
    ('comp_events_completed', ctypes.c_uint32),
    ('async_events_completed', ctypes.c_uint32),
    ('comp_mask', ctypes.c_uint32),
    ('status', ibv_wc_status),
    ('wr_id', ctypes.c_uint64),
    ('start_poll', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq_ex), ctypes.POINTER(struct_ibv_poll_cq_attr))),
    ('next_poll', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq_ex))),
    ('end_poll', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_opcode', ctypes.CFUNCTYPE(ibv_wc_opcode, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_vendor_err', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_byte_len', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_imm_data', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_qp_num', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_src_qp', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_wc_flags', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_slid', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_sl', ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_dlid_path_bits', ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_completion_ts', ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_cvlan', ctypes.CFUNCTYPE(ctypes.c_uint16, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_flow_tag', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
    ('read_tm_info', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_cq_ex), ctypes.POINTER(struct_ibv_wc_tm_info))),
    ('read_completion_wallclock_ns', ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(struct_ibv_cq_ex))),
]

try:
    ibv_cq_ex_to_cq = _libraries['libibverbs'].ibv_cq_ex_to_cq
    ibv_cq_ex_to_cq.restype = ctypes.POINTER(struct_ibv_cq)
    ibv_cq_ex_to_cq.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass

# values for enumeration 'ibv_cq_attr_mask'
ibv_cq_attr_mask__enumvalues = {
    1: 'IBV_CQ_ATTR_MODERATE',
    2: 'IBV_CQ_ATTR_RESERVED',
}
IBV_CQ_ATTR_MODERATE = 1
IBV_CQ_ATTR_RESERVED = 2
ibv_cq_attr_mask = ctypes.c_uint32 # enum
class struct_ibv_moderate_cq(Structure):
    pass

struct_ibv_moderate_cq._pack_ = 1 # source:False
struct_ibv_moderate_cq._fields_ = [
    ('cq_count', ctypes.c_uint16),
    ('cq_period', ctypes.c_uint16),
]

class struct_ibv_modify_cq_attr(Structure):
    pass

struct_ibv_modify_cq_attr._pack_ = 1 # source:False
struct_ibv_modify_cq_attr._fields_ = [
    ('attr_mask', ctypes.c_uint32),
    ('moderate', struct_ibv_moderate_cq),
]

try:
    ibv_start_poll = _libraries['libibverbs'].ibv_start_poll
    ibv_start_poll.restype = ctypes.c_int32
    ibv_start_poll.argtypes = [ctypes.POINTER(struct_ibv_cq_ex), ctypes.POINTER(struct_ibv_poll_cq_attr)]
except AttributeError:
    pass
try:
    ibv_next_poll = _libraries['libibverbs'].ibv_next_poll
    ibv_next_poll.restype = ctypes.c_int32
    ibv_next_poll.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_end_poll = _libraries['libibverbs'].ibv_end_poll
    ibv_end_poll.restype = None
    ibv_end_poll.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_opcode = _libraries['libibverbs'].ibv_wc_read_opcode
    ibv_wc_read_opcode.restype = ibv_wc_opcode
    ibv_wc_read_opcode.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_vendor_err = _libraries['libibverbs'].ibv_wc_read_vendor_err
    ibv_wc_read_vendor_err.restype = uint32_t
    ibv_wc_read_vendor_err.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_byte_len = _libraries['libibverbs'].ibv_wc_read_byte_len
    ibv_wc_read_byte_len.restype = uint32_t
    ibv_wc_read_byte_len.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_imm_data = _libraries['libibverbs'].ibv_wc_read_imm_data
    ibv_wc_read_imm_data.restype = __be32
    ibv_wc_read_imm_data.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_invalidated_rkey = _libraries['libibverbs'].ibv_wc_read_invalidated_rkey
    ibv_wc_read_invalidated_rkey.restype = uint32_t
    ibv_wc_read_invalidated_rkey.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_qp_num = _libraries['libibverbs'].ibv_wc_read_qp_num
    ibv_wc_read_qp_num.restype = uint32_t
    ibv_wc_read_qp_num.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_src_qp = _libraries['libibverbs'].ibv_wc_read_src_qp
    ibv_wc_read_src_qp.restype = uint32_t
    ibv_wc_read_src_qp.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_wc_flags = _libraries['libibverbs'].ibv_wc_read_wc_flags
    ibv_wc_read_wc_flags.restype = ctypes.c_uint32
    ibv_wc_read_wc_flags.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_slid = _libraries['libibverbs'].ibv_wc_read_slid
    ibv_wc_read_slid.restype = uint32_t
    ibv_wc_read_slid.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_sl = _libraries['libibverbs'].ibv_wc_read_sl
    ibv_wc_read_sl.restype = uint8_t
    ibv_wc_read_sl.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_dlid_path_bits = _libraries['libibverbs'].ibv_wc_read_dlid_path_bits
    ibv_wc_read_dlid_path_bits.restype = uint8_t
    ibv_wc_read_dlid_path_bits.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_completion_ts = _libraries['libibverbs'].ibv_wc_read_completion_ts
    ibv_wc_read_completion_ts.restype = uint64_t
    ibv_wc_read_completion_ts.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_completion_wallclock_ns = _libraries['libibverbs'].ibv_wc_read_completion_wallclock_ns
    ibv_wc_read_completion_wallclock_ns.restype = uint64_t
    ibv_wc_read_completion_wallclock_ns.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_cvlan = _libraries['libibverbs'].ibv_wc_read_cvlan
    ibv_wc_read_cvlan.restype = uint16_t
    ibv_wc_read_cvlan.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_flow_tag = _libraries['libibverbs'].ibv_wc_read_flow_tag
    ibv_wc_read_flow_tag.restype = uint32_t
    ibv_wc_read_flow_tag.argtypes = [ctypes.POINTER(struct_ibv_cq_ex)]
except AttributeError:
    pass
try:
    ibv_wc_read_tm_info = _libraries['libibverbs'].ibv_wc_read_tm_info
    ibv_wc_read_tm_info.restype = None
    ibv_wc_read_tm_info.argtypes = [ctypes.POINTER(struct_ibv_cq_ex), ctypes.POINTER(struct_ibv_wc_tm_info)]
except AttributeError:
    pass
try:
    ibv_post_wq_recv = _libraries['libibverbs'].ibv_post_wq_recv
    ibv_post_wq_recv.restype = ctypes.c_int32
    ibv_post_wq_recv.argtypes = [ctypes.POINTER(struct_ibv_wq), ctypes.POINTER(struct_ibv_recv_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_recv_wr))]
except AttributeError:
    pass

# values for enumeration 'ibv_flow_flags'
ibv_flow_flags__enumvalues = {
    2: 'IBV_FLOW_ATTR_FLAGS_DONT_TRAP',
    4: 'IBV_FLOW_ATTR_FLAGS_EGRESS',
}
IBV_FLOW_ATTR_FLAGS_DONT_TRAP = 2
IBV_FLOW_ATTR_FLAGS_EGRESS = 4
ibv_flow_flags = ctypes.c_uint32 # enum

# values for enumeration 'ibv_flow_attr_type'
ibv_flow_attr_type__enumvalues = {
    0: 'IBV_FLOW_ATTR_NORMAL',
    1: 'IBV_FLOW_ATTR_ALL_DEFAULT',
    2: 'IBV_FLOW_ATTR_MC_DEFAULT',
    3: 'IBV_FLOW_ATTR_SNIFFER',
}
IBV_FLOW_ATTR_NORMAL = 0
IBV_FLOW_ATTR_ALL_DEFAULT = 1
IBV_FLOW_ATTR_MC_DEFAULT = 2
IBV_FLOW_ATTR_SNIFFER = 3
ibv_flow_attr_type = ctypes.c_uint32 # enum

# values for enumeration 'ibv_flow_spec_type'
ibv_flow_spec_type__enumvalues = {
    32: 'IBV_FLOW_SPEC_ETH',
    48: 'IBV_FLOW_SPEC_IPV4',
    49: 'IBV_FLOW_SPEC_IPV6',
    50: 'IBV_FLOW_SPEC_IPV4_EXT',
    52: 'IBV_FLOW_SPEC_ESP',
    64: 'IBV_FLOW_SPEC_TCP',
    65: 'IBV_FLOW_SPEC_UDP',
    80: 'IBV_FLOW_SPEC_VXLAN_TUNNEL',
    81: 'IBV_FLOW_SPEC_GRE',
    96: 'IBV_FLOW_SPEC_MPLS',
    256: 'IBV_FLOW_SPEC_INNER',
    4096: 'IBV_FLOW_SPEC_ACTION_TAG',
    4097: 'IBV_FLOW_SPEC_ACTION_DROP',
    4098: 'IBV_FLOW_SPEC_ACTION_HANDLE',
    4099: 'IBV_FLOW_SPEC_ACTION_COUNT',
}
IBV_FLOW_SPEC_ETH = 32
IBV_FLOW_SPEC_IPV4 = 48
IBV_FLOW_SPEC_IPV6 = 49
IBV_FLOW_SPEC_IPV4_EXT = 50
IBV_FLOW_SPEC_ESP = 52
IBV_FLOW_SPEC_TCP = 64
IBV_FLOW_SPEC_UDP = 65
IBV_FLOW_SPEC_VXLAN_TUNNEL = 80
IBV_FLOW_SPEC_GRE = 81
IBV_FLOW_SPEC_MPLS = 96
IBV_FLOW_SPEC_INNER = 256
IBV_FLOW_SPEC_ACTION_TAG = 4096
IBV_FLOW_SPEC_ACTION_DROP = 4097
IBV_FLOW_SPEC_ACTION_HANDLE = 4098
IBV_FLOW_SPEC_ACTION_COUNT = 4099
ibv_flow_spec_type = ctypes.c_uint32 # enum
class struct_ibv_flow_eth_filter(Structure):
    pass

struct_ibv_flow_eth_filter._pack_ = 1 # source:False
struct_ibv_flow_eth_filter._fields_ = [
    ('dst_mac', ctypes.c_ubyte * 6),
    ('src_mac', ctypes.c_ubyte * 6),
    ('ether_type', ctypes.c_uint16),
    ('vlan_tag', ctypes.c_uint16),
]

class struct_ibv_flow_spec_eth(Structure):
    pass

struct_ibv_flow_spec_eth._pack_ = 1 # source:False
struct_ibv_flow_spec_eth._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('val', struct_ibv_flow_eth_filter),
    ('mask', struct_ibv_flow_eth_filter),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

class struct_ibv_flow_ipv4_filter(Structure):
    pass

struct_ibv_flow_ipv4_filter._pack_ = 1 # source:False
struct_ibv_flow_ipv4_filter._fields_ = [
    ('src_ip', ctypes.c_uint32),
    ('dst_ip', ctypes.c_uint32),
]

class struct_ibv_flow_spec_ipv4(Structure):
    pass

struct_ibv_flow_spec_ipv4._pack_ = 1 # source:False
struct_ibv_flow_spec_ipv4._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('val', struct_ibv_flow_ipv4_filter),
    ('mask', struct_ibv_flow_ipv4_filter),
]

class struct_ibv_flow_ipv4_ext_filter(Structure):
    pass

struct_ibv_flow_ipv4_ext_filter._pack_ = 1 # source:False
struct_ibv_flow_ipv4_ext_filter._fields_ = [
    ('src_ip', ctypes.c_uint32),
    ('dst_ip', ctypes.c_uint32),
    ('proto', ctypes.c_ubyte),
    ('tos', ctypes.c_ubyte),
    ('ttl', ctypes.c_ubyte),
    ('flags', ctypes.c_ubyte),
]

class struct_ibv_flow_spec_ipv4_ext(Structure):
    pass

struct_ibv_flow_spec_ipv4_ext._pack_ = 1 # source:False
struct_ibv_flow_spec_ipv4_ext._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('val', struct_ibv_flow_ipv4_ext_filter),
    ('mask', struct_ibv_flow_ipv4_ext_filter),
]

class struct_ibv_flow_ipv6_filter(Structure):
    pass

struct_ibv_flow_ipv6_filter._pack_ = 1 # source:False
struct_ibv_flow_ipv6_filter._fields_ = [
    ('src_ip', ctypes.c_ubyte * 16),
    ('dst_ip', ctypes.c_ubyte * 16),
    ('flow_label', ctypes.c_uint32),
    ('next_hdr', ctypes.c_ubyte),
    ('traffic_class', ctypes.c_ubyte),
    ('hop_limit', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
]

class struct_ibv_flow_spec_ipv6(Structure):
    pass

struct_ibv_flow_spec_ipv6._pack_ = 1 # source:False
struct_ibv_flow_spec_ipv6._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('val', struct_ibv_flow_ipv6_filter),
    ('mask', struct_ibv_flow_ipv6_filter),
]

class struct_ibv_flow_esp_filter(Structure):
    pass

struct_ibv_flow_esp_filter._pack_ = 1 # source:False
struct_ibv_flow_esp_filter._fields_ = [
    ('spi', ctypes.c_uint32),
    ('seq', ctypes.c_uint32),
]

class struct_ibv_flow_spec_esp(Structure):
    pass

struct_ibv_flow_spec_esp._pack_ = 1 # source:False
struct_ibv_flow_spec_esp._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('val', struct_ibv_flow_esp_filter),
    ('mask', struct_ibv_flow_esp_filter),
]

class struct_ibv_flow_tcp_udp_filter(Structure):
    pass

struct_ibv_flow_tcp_udp_filter._pack_ = 1 # source:False
struct_ibv_flow_tcp_udp_filter._fields_ = [
    ('dst_port', ctypes.c_uint16),
    ('src_port', ctypes.c_uint16),
]

class struct_ibv_flow_spec_tcp_udp(Structure):
    pass

struct_ibv_flow_spec_tcp_udp._pack_ = 1 # source:False
struct_ibv_flow_spec_tcp_udp._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('val', struct_ibv_flow_tcp_udp_filter),
    ('mask', struct_ibv_flow_tcp_udp_filter),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

class struct_ibv_flow_gre_filter(Structure):
    pass

struct_ibv_flow_gre_filter._pack_ = 1 # source:False
struct_ibv_flow_gre_filter._fields_ = [
    ('c_ks_res0_ver', ctypes.c_uint16),
    ('protocol', ctypes.c_uint16),
    ('key', ctypes.c_uint32),
]

class struct_ibv_flow_spec_gre(Structure):
    pass

struct_ibv_flow_spec_gre._pack_ = 1 # source:False
struct_ibv_flow_spec_gre._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('val', struct_ibv_flow_gre_filter),
    ('mask', struct_ibv_flow_gre_filter),
]

class struct_ibv_flow_mpls_filter(Structure):
    pass

struct_ibv_flow_mpls_filter._pack_ = 1 # source:False
struct_ibv_flow_mpls_filter._fields_ = [
    ('label', ctypes.c_uint32),
]

class struct_ibv_flow_spec_mpls(Structure):
    pass

struct_ibv_flow_spec_mpls._pack_ = 1 # source:False
struct_ibv_flow_spec_mpls._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('val', struct_ibv_flow_mpls_filter),
    ('mask', struct_ibv_flow_mpls_filter),
]

class struct_ibv_flow_tunnel_filter(Structure):
    pass

struct_ibv_flow_tunnel_filter._pack_ = 1 # source:False
struct_ibv_flow_tunnel_filter._fields_ = [
    ('tunnel_id', ctypes.c_uint32),
]

class struct_ibv_flow_spec_tunnel(Structure):
    pass

struct_ibv_flow_spec_tunnel._pack_ = 1 # source:False
struct_ibv_flow_spec_tunnel._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('val', struct_ibv_flow_tunnel_filter),
    ('mask', struct_ibv_flow_tunnel_filter),
]

class struct_ibv_flow_spec_action_tag(Structure):
    pass

struct_ibv_flow_spec_action_tag._pack_ = 1 # source:False
struct_ibv_flow_spec_action_tag._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('tag_id', ctypes.c_uint32),
]

class struct_ibv_flow_spec_action_drop(Structure):
    pass

struct_ibv_flow_spec_action_drop._pack_ = 1 # source:False
struct_ibv_flow_spec_action_drop._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

class struct_ibv_flow_spec_action_handle(Structure):
    pass

class struct_ibv_flow_action(Structure):
    pass

struct_ibv_flow_spec_action_handle._pack_ = 1 # source:False
struct_ibv_flow_spec_action_handle._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('action', ctypes.POINTER(struct_ibv_flow_action)),
]

struct_ibv_flow_action._pack_ = 1 # source:False
struct_ibv_flow_action._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
]

class struct_ibv_flow_spec_counter_action(Structure):
    pass

class struct_ibv_counters(Structure):
    pass

struct_ibv_flow_spec_counter_action._pack_ = 1 # source:False
struct_ibv_flow_spec_counter_action._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('counters', ctypes.POINTER(struct_ibv_counters)),
]

struct_ibv_counters._pack_ = 1 # source:False
struct_ibv_counters._fields_ = [
    ('context', ctypes.POINTER(struct_ibv_context)),
]

class struct_ibv_flow_spec(Structure):
    pass

class union_ibv_flow_spec_0(Union):
    pass

class struct_ibv_flow_spec_0_hdr(Structure):
    pass

struct_ibv_flow_spec_0_hdr._pack_ = 1 # source:False
struct_ibv_flow_spec_0_hdr._fields_ = [
    ('type', ibv_flow_spec_type),
    ('size', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

union_ibv_flow_spec_0._pack_ = 1 # source:False
union_ibv_flow_spec_0._fields_ = [
    ('hdr', struct_ibv_flow_spec_0_hdr),
    ('eth', struct_ibv_flow_spec_eth),
    ('ipv4', struct_ibv_flow_spec_ipv4),
    ('tcp_udp', struct_ibv_flow_spec_tcp_udp),
    ('ipv4_ext', struct_ibv_flow_spec_ipv4_ext),
    ('ipv6', struct_ibv_flow_spec_ipv6),
    ('esp', struct_ibv_flow_spec_esp),
    ('tunnel', struct_ibv_flow_spec_tunnel),
    ('gre', struct_ibv_flow_spec_gre),
    ('mpls', struct_ibv_flow_spec_mpls),
    ('flow_tag', struct_ibv_flow_spec_action_tag),
    ('drop', struct_ibv_flow_spec_action_drop),
    ('handle', struct_ibv_flow_spec_action_handle),
    ('flow_count', struct_ibv_flow_spec_counter_action),
    ('PADDING_0', ctypes.c_ubyte * 72),
]

struct_ibv_flow_spec._pack_ = 1 # source:False
struct_ibv_flow_spec._anonymous_ = ('_0',)
struct_ibv_flow_spec._fields_ = [
    ('_0', union_ibv_flow_spec_0),
]

class struct_ibv_flow_attr(Structure):
    pass

struct_ibv_flow_attr._pack_ = 1 # source:False
struct_ibv_flow_attr._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('type', ibv_flow_attr_type),
    ('size', ctypes.c_uint16),
    ('priority', ctypes.c_uint16),
    ('num_of_specs', ctypes.c_ubyte),
    ('port', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('flags', ctypes.c_uint32),
]

class struct_ibv_flow(Structure):
    pass

struct_ibv_flow._pack_ = 1 # source:False
struct_ibv_flow._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('context', ctypes.POINTER(struct_ibv_context)),
    ('handle', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]


# values for enumeration 'ibv_flow_action_esp_mask'
ibv_flow_action_esp_mask__enumvalues = {
    1: 'IBV_FLOW_ACTION_ESP_MASK_ESN',
}
IBV_FLOW_ACTION_ESP_MASK_ESN = 1
ibv_flow_action_esp_mask = ctypes.c_uint32 # enum
class struct_ibv_flow_action_esp_attr(Structure):
    pass

struct_ibv_flow_action_esp_attr._pack_ = 1 # source:False
struct_ibv_flow_action_esp_attr._fields_ = [
    ('esp_attr', ctypes.POINTER(struct_ib_uverbs_flow_action_esp)),
    ('keymat_proto', ib_uverbs_flow_action_esp_keymat),
    ('keymat_len', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('keymat_ptr', ctypes.POINTER(None)),
    ('replay_proto', ib_uverbs_flow_action_esp_replay),
    ('replay_len', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('replay_ptr', ctypes.POINTER(None)),
    ('esp_encap', ctypes.POINTER(struct_ib_uverbs_flow_action_esp_encap)),
    ('comp_mask', ctypes.c_uint32),
    ('esn', ctypes.c_uint32),
]


# values for enumeration 'c__Ea_IBV_SYSFS_NAME_MAX'
c__Ea_IBV_SYSFS_NAME_MAX__enumvalues = {
    64: 'IBV_SYSFS_NAME_MAX',
    256: 'IBV_SYSFS_PATH_MAX',
}
IBV_SYSFS_NAME_MAX = 64
IBV_SYSFS_PATH_MAX = 256
c__Ea_IBV_SYSFS_NAME_MAX = ctypes.c_uint32 # enum

# values for enumeration 'ibv_cq_init_attr_mask'
ibv_cq_init_attr_mask__enumvalues = {
    1: 'IBV_CQ_INIT_ATTR_MASK_FLAGS',
    2: 'IBV_CQ_INIT_ATTR_MASK_PD',
}
IBV_CQ_INIT_ATTR_MASK_FLAGS = 1
IBV_CQ_INIT_ATTR_MASK_PD = 2
ibv_cq_init_attr_mask = ctypes.c_uint32 # enum

# values for enumeration 'ibv_create_cq_attr_flags'
ibv_create_cq_attr_flags__enumvalues = {
    1: 'IBV_CREATE_CQ_ATTR_SINGLE_THREADED',
    2: 'IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN',
}
IBV_CREATE_CQ_ATTR_SINGLE_THREADED = 1
IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN = 2
ibv_create_cq_attr_flags = ctypes.c_uint32 # enum
class struct_ibv_cq_init_attr_ex(Structure):
    pass

struct_ibv_cq_init_attr_ex._pack_ = 1 # source:False
struct_ibv_cq_init_attr_ex._fields_ = [
    ('cqe', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('cq_context', ctypes.POINTER(None)),
    ('channel', ctypes.POINTER(struct_ibv_comp_channel)),
    ('comp_vector', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('wc_flags', ctypes.c_uint64),
    ('comp_mask', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('parent_domain', ctypes.POINTER(struct_ibv_pd)),
]


# values for enumeration 'ibv_parent_domain_init_attr_mask'
ibv_parent_domain_init_attr_mask__enumvalues = {
    1: 'IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS',
    2: 'IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT',
}
IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS = 1
IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT = 2
ibv_parent_domain_init_attr_mask = ctypes.c_uint32 # enum
class struct_ibv_parent_domain_init_attr(Structure):
    pass

struct_ibv_parent_domain_init_attr._pack_ = 1 # source:False
struct_ibv_parent_domain_init_attr._fields_ = [
    ('pd', ctypes.POINTER(struct_ibv_pd)),
    ('td', ctypes.POINTER(struct_ibv_td)),
    ('comp_mask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('alloc', ctypes.CFUNCTYPE(ctypes.POINTER(None), ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)),
    ('free', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(None), ctypes.POINTER(None), ctypes.c_uint64)),
    ('pd_context', ctypes.POINTER(None)),
]

class struct_ibv_counters_init_attr(Structure):
    pass

struct_ibv_counters_init_attr._pack_ = 1 # source:False
struct_ibv_counters_init_attr._fields_ = [
    ('comp_mask', ctypes.c_uint32),
]


# values for enumeration 'ibv_counter_description'
ibv_counter_description__enumvalues = {
    0: 'IBV_COUNTER_PACKETS',
    1: 'IBV_COUNTER_BYTES',
}
IBV_COUNTER_PACKETS = 0
IBV_COUNTER_BYTES = 1
ibv_counter_description = ctypes.c_uint32 # enum
class struct_ibv_counter_attach_attr(Structure):
    pass

struct_ibv_counter_attach_attr._pack_ = 1 # source:False
struct_ibv_counter_attach_attr._fields_ = [
    ('counter_desc', ibv_counter_description),
    ('index', ctypes.c_uint32),
    ('comp_mask', ctypes.c_uint32),
]


# values for enumeration 'ibv_read_counters_flags'
ibv_read_counters_flags__enumvalues = {
    1: 'IBV_READ_COUNTERS_ATTR_PREFER_CACHED',
}
IBV_READ_COUNTERS_ATTR_PREFER_CACHED = 1
ibv_read_counters_flags = ctypes.c_uint32 # enum

# values for enumeration 'ibv_values_mask'
ibv_values_mask__enumvalues = {
    1: 'IBV_VALUES_MASK_RAW_CLOCK',
    2: 'IBV_VALUES_MASK_RESERVED',
}
IBV_VALUES_MASK_RAW_CLOCK = 1
IBV_VALUES_MASK_RESERVED = 2
ibv_values_mask = ctypes.c_uint32 # enum
class struct_ibv_values_ex(Structure):
    pass

class struct_timespec(Structure):
    pass

struct_timespec._pack_ = 1 # source:False
struct_timespec._fields_ = [
    ('tv_sec', ctypes.c_int64),
    ('tv_nsec', ctypes.c_int64),
]

struct_ibv_values_ex._pack_ = 1 # source:False
struct_ibv_values_ex._fields_ = [
    ('comp_mask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('raw_clock', struct_timespec),
]

class struct_verbs_context(Structure):
    pass

class struct_verbs_ex_private(Structure):
    pass

struct_verbs_context._pack_ = 1 # source:False
struct_verbs_context._fields_ = [
    ('query_port', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), ctypes.c_ubyte, ctypes.POINTER(struct_ibv_port_attr), ctypes.c_uint64)),
    ('advise_mr', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_pd), ib_uverbs_advise_mr_advice, ctypes.c_uint32, ctypes.POINTER(struct_ibv_sge), ctypes.c_uint32)),
    ('alloc_null_mr', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_mr), ctypes.POINTER(struct_ibv_pd))),
    ('read_counters', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_counters), ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_uint32)),
    ('attach_counters_point_flow', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_counters), ctypes.POINTER(struct_ibv_counter_attach_attr), ctypes.POINTER(struct_ibv_flow))),
    ('create_counters', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_counters), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_counters_init_attr))),
    ('destroy_counters', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_counters))),
    ('reg_dm_mr', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_mr), ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_dm), ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32)),
    ('alloc_dm', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_dm), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_alloc_dm_attr))),
    ('free_dm', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_dm))),
    ('modify_flow_action_esp', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_flow_action), ctypes.POINTER(struct_ibv_flow_action_esp_attr))),
    ('destroy_flow_action', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_flow_action))),
    ('create_flow_action_esp', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_flow_action), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_flow_action_esp_attr))),
    ('modify_qp_rate_limit', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_qp_rate_limit_attr))),
    ('alloc_parent_domain', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_parent_domain_init_attr))),
    ('dealloc_td', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_td))),
    ('alloc_td', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_td), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_td_init_attr))),
    ('modify_cq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq), ctypes.POINTER(struct_ibv_modify_cq_attr))),
    ('post_srq_ops', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_ops_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_ops_wr)))),
    ('destroy_rwq_ind_table', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_rwq_ind_table))),
    ('create_rwq_ind_table', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_rwq_ind_table), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_rwq_ind_table_init_attr))),
    ('destroy_wq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_wq))),
    ('modify_wq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_wq), ctypes.POINTER(struct_ibv_wq_attr))),
    ('create_wq', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_wq), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_wq_init_attr))),
    ('query_rt_values', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_values_ex))),
    ('create_cq_ex', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_cq_ex), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_cq_init_attr_ex))),
    ('priv', ctypes.POINTER(struct_verbs_ex_private)),
    ('query_device_ex', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_query_device_ex_input), ctypes.POINTER(struct_ibv_device_attr_ex), ctypes.c_uint64)),
    ('ibv_destroy_flow', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_flow))),
    ('ABI_placeholder2', ctypes.CFUNCTYPE(None)),
    ('ibv_create_flow', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_flow), ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_flow_attr))),
    ('ABI_placeholder1', ctypes.CFUNCTYPE(None)),
    ('open_qp', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_qp_open_attr))),
    ('create_qp_ex', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_qp_init_attr_ex))),
    ('get_srq_num', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(ctypes.c_uint32))),
    ('create_srq_ex', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_srq_init_attr_ex))),
    ('open_xrcd', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_xrcd), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_xrcd_init_attr))),
    ('close_xrcd', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_xrcd))),
    ('_ABI_placeholder3', ctypes.c_uint64),
    ('sz', ctypes.c_uint64),
    ('context', struct_ibv_context),
]

try:
    verbs_get_ctx = _libraries['libibverbs'].verbs_get_ctx
    verbs_get_ctx.restype = ctypes.POINTER(struct_verbs_context)
    verbs_get_ctx.argtypes = [ctypes.POINTER(struct_ibv_context)]
except AttributeError:
    pass
# def verbs_get_ctx_op(ctx, op):  # macro
#    return ({struct_verbs_context*__vctx=verbs_get_ctx(ctx);(!__vctx or (__vctx->sz<ctypes.sizeof(*__vctx)-offsetof(struct_verbs_context,op)) or !__vctx->op)?NULL:__vctx;})
try:
    ibv_get_device_list = _libraries['libibverbs'].ibv_get_device_list
    ibv_get_device_list.restype = ctypes.POINTER(ctypes.POINTER(struct_ibv_device))
    ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    ibv_free_device_list = _libraries['libibverbs'].ibv_free_device_list
    ibv_free_device_list.restype = None
    ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_ibv_device))]
except AttributeError:
    pass
try:
    ibv_get_device_name = _libraries['libibverbs'].ibv_get_device_name
    ibv_get_device_name.restype = ctypes.POINTER(ctypes.c_char)
    ibv_get_device_name.argtypes = [ctypes.POINTER(struct_ibv_device)]
except AttributeError:
    pass
try:
    ibv_get_device_index = _libraries['libibverbs'].ibv_get_device_index
    ibv_get_device_index.restype = ctypes.c_int32
    ibv_get_device_index.argtypes = [ctypes.POINTER(struct_ibv_device)]
except AttributeError:
    pass
__be64 = ctypes.c_uint64
try:
    ibv_get_device_guid = _libraries['libibverbs'].ibv_get_device_guid
    ibv_get_device_guid.restype = __be64
    ibv_get_device_guid.argtypes = [ctypes.POINTER(struct_ibv_device)]
except AttributeError:
    pass
try:
    ibv_open_device = _libraries['libibverbs'].ibv_open_device
    ibv_open_device.restype = ctypes.POINTER(struct_ibv_context)
    ibv_open_device.argtypes = [ctypes.POINTER(struct_ibv_device)]
except AttributeError:
    pass
try:
    ibv_close_device = _libraries['libibverbs'].ibv_close_device
    ibv_close_device.restype = ctypes.c_int32
    ibv_close_device.argtypes = [ctypes.POINTER(struct_ibv_context)]
except AttributeError:
    pass
try:
    ibv_import_device = _libraries['libibverbs'].ibv_import_device
    ibv_import_device.restype = ctypes.POINTER(struct_ibv_context)
    ibv_import_device.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    ibv_import_pd = _libraries['libibverbs'].ibv_import_pd
    ibv_import_pd.restype = ctypes.POINTER(struct_ibv_pd)
    ibv_import_pd.argtypes = [ctypes.POINTER(struct_ibv_context), uint32_t]
except AttributeError:
    pass
try:
    ibv_unimport_pd = _libraries['libibverbs'].ibv_unimport_pd
    ibv_unimport_pd.restype = None
    ibv_unimport_pd.argtypes = [ctypes.POINTER(struct_ibv_pd)]
except AttributeError:
    pass
try:
    ibv_import_mr = _libraries['libibverbs'].ibv_import_mr
    ibv_import_mr.restype = ctypes.POINTER(struct_ibv_mr)
    ibv_import_mr.argtypes = [ctypes.POINTER(struct_ibv_pd), uint32_t]
except AttributeError:
    pass
try:
    ibv_unimport_mr = _libraries['libibverbs'].ibv_unimport_mr
    ibv_unimport_mr.restype = None
    ibv_unimport_mr.argtypes = [ctypes.POINTER(struct_ibv_mr)]
except AttributeError:
    pass
try:
    ibv_import_dm = _libraries['libibverbs'].ibv_import_dm
    ibv_import_dm.restype = ctypes.POINTER(struct_ibv_dm)
    ibv_import_dm.argtypes = [ctypes.POINTER(struct_ibv_context), uint32_t]
except AttributeError:
    pass
try:
    ibv_unimport_dm = _libraries['libibverbs'].ibv_unimport_dm
    ibv_unimport_dm.restype = None
    ibv_unimport_dm.argtypes = [ctypes.POINTER(struct_ibv_dm)]
except AttributeError:
    pass
try:
    ibv_get_async_event = _libraries['libibverbs'].ibv_get_async_event
    ibv_get_async_event.restype = ctypes.c_int32
    ibv_get_async_event.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_async_event)]
except AttributeError:
    pass
try:
    ibv_ack_async_event = _libraries['libibverbs'].ibv_ack_async_event
    ibv_ack_async_event.restype = None
    ibv_ack_async_event.argtypes = [ctypes.POINTER(struct_ibv_async_event)]
except AttributeError:
    pass
try:
    ibv_query_device = _libraries['libibverbs'].ibv_query_device
    ibv_query_device.restype = ctypes.c_int32
    ibv_query_device.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_device_attr)]
except AttributeError:
    pass
try:
    ___ibv_query_port = _libraries['libibverbs'].___ibv_query_port
    ___ibv_query_port.restype = ctypes.c_int32
    ___ibv_query_port.argtypes = [ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.POINTER(struct_ibv_port_attr)]
except AttributeError:
    pass
def ibv_query_port(context, port_num, port_attr):  # macro
   return ___ibv_query_port(context,port_num,port_attr)
try:
    ibv_query_gid = _libraries['libibverbs'].ibv_query_gid
    ibv_query_gid.restype = ctypes.c_int32
    ibv_query_gid.argtypes = [ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.c_int32, ctypes.POINTER(union_ibv_gid)]
except AttributeError:
    pass
try:
    _ibv_query_gid_ex = _libraries['libibverbs']._ibv_query_gid_ex
    _ibv_query_gid_ex.restype = ctypes.c_int32
    _ibv_query_gid_ex.argtypes = [ctypes.POINTER(struct_ibv_context), uint32_t, uint32_t, ctypes.POINTER(struct_ibv_gid_entry), uint32_t, size_t]
except AttributeError:
    pass
try:
    ibv_query_gid_ex = _libraries['libibverbs'].ibv_query_gid_ex
    ibv_query_gid_ex.restype = ctypes.c_int32
    ibv_query_gid_ex.argtypes = [ctypes.POINTER(struct_ibv_context), uint32_t, uint32_t, ctypes.POINTER(struct_ibv_gid_entry), uint32_t]
except AttributeError:
    pass
ssize_t = ctypes.c_int64
try:
    _ibv_query_gid_table = _libraries['libibverbs']._ibv_query_gid_table
    _ibv_query_gid_table.restype = ssize_t
    _ibv_query_gid_table.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_gid_entry), size_t, uint32_t, size_t]
except AttributeError:
    pass
try:
    ibv_query_gid_table = _libraries['libibverbs'].ibv_query_gid_table
    ibv_query_gid_table.restype = ssize_t
    ibv_query_gid_table.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_gid_entry), size_t, uint32_t]
except AttributeError:
    pass
try:
    ibv_query_pkey = _libraries['libibverbs'].ibv_query_pkey
    ibv_query_pkey.restype = ctypes.c_int32
    ibv_query_pkey.argtypes = [ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint16)]
except AttributeError:
    pass
__be16 = ctypes.c_uint16
try:
    ibv_get_pkey_index = _libraries['libibverbs'].ibv_get_pkey_index
    ibv_get_pkey_index.restype = ctypes.c_int32
    ibv_get_pkey_index.argtypes = [ctypes.POINTER(struct_ibv_context), uint8_t, __be16]
except AttributeError:
    pass
try:
    ibv_alloc_pd = _libraries['libibverbs'].ibv_alloc_pd
    ibv_alloc_pd.restype = ctypes.POINTER(struct_ibv_pd)
    ibv_alloc_pd.argtypes = [ctypes.POINTER(struct_ibv_context)]
except AttributeError:
    pass
try:
    ibv_dealloc_pd = _libraries['libibverbs'].ibv_dealloc_pd
    ibv_dealloc_pd.restype = ctypes.c_int32
    ibv_dealloc_pd.argtypes = [ctypes.POINTER(struct_ibv_pd)]
except AttributeError:
    pass
try:
    ibv_create_flow = _libraries['libibverbs'].ibv_create_flow
    ibv_create_flow.restype = ctypes.POINTER(struct_ibv_flow)
    ibv_create_flow.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_flow_attr)]
except AttributeError:
    pass
try:
    ibv_destroy_flow = _libraries['libibverbs'].ibv_destroy_flow
    ibv_destroy_flow.restype = ctypes.c_int32
    ibv_destroy_flow.argtypes = [ctypes.POINTER(struct_ibv_flow)]
except AttributeError:
    pass
try:
    ibv_create_flow_action_esp = _libraries['libibverbs'].ibv_create_flow_action_esp
    ibv_create_flow_action_esp.restype = ctypes.POINTER(struct_ibv_flow_action)
    ibv_create_flow_action_esp.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_flow_action_esp_attr)]
except AttributeError:
    pass
try:
    ibv_modify_flow_action_esp = _libraries['libibverbs'].ibv_modify_flow_action_esp
    ibv_modify_flow_action_esp.restype = ctypes.c_int32
    ibv_modify_flow_action_esp.argtypes = [ctypes.POINTER(struct_ibv_flow_action), ctypes.POINTER(struct_ibv_flow_action_esp_attr)]
except AttributeError:
    pass
try:
    ibv_destroy_flow_action = _libraries['libibverbs'].ibv_destroy_flow_action
    ibv_destroy_flow_action.restype = ctypes.c_int32
    ibv_destroy_flow_action.argtypes = [ctypes.POINTER(struct_ibv_flow_action)]
except AttributeError:
    pass
try:
    ibv_open_xrcd = _libraries['libibverbs'].ibv_open_xrcd
    ibv_open_xrcd.restype = ctypes.POINTER(struct_ibv_xrcd)
    ibv_open_xrcd.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_xrcd_init_attr)]
except AttributeError:
    pass
try:
    ibv_close_xrcd = _libraries['libibverbs'].ibv_close_xrcd
    ibv_close_xrcd.restype = ctypes.c_int32
    ibv_close_xrcd.argtypes = [ctypes.POINTER(struct_ibv_xrcd)]
except AttributeError:
    pass
try:
    ibv_reg_mr_iova2 = _libraries['libibverbs'].ibv_reg_mr_iova2
    ibv_reg_mr_iova2.restype = ctypes.POINTER(struct_ibv_mr)
    ibv_reg_mr_iova2.argtypes = [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(None), size_t, uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    __ibv_reg_mr = _libraries['libibverbs'].__ibv_reg_mr
    __ibv_reg_mr.restype = ctypes.POINTER(struct_ibv_mr)
    __ibv_reg_mr.argtypes = [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(None), size_t, ctypes.c_uint32, ctypes.c_int32]
except AttributeError:
    pass
# def ibv_reg_mr(pd, addr, length, access):  # macro
#    return __ibv_reg_mr(pd,addr,length,access,__builtin_constant_p(((access)&IB_UVERBS_ACCESS_OPTIONAL_RANGE)==0))
try:
    __ibv_reg_mr_iova = _libraries['libibverbs'].__ibv_reg_mr_iova
    __ibv_reg_mr_iova.restype = ctypes.POINTER(struct_ibv_mr)
    __ibv_reg_mr_iova.argtypes = [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(None), size_t, uint64_t, ctypes.c_uint32, ctypes.c_int32]
except AttributeError:
    pass
# def ibv_reg_mr_iova(pd, addr, length, iova, access):  # macro
#    return __ibv_reg_mr_iova(pd,addr,length,iova,access,__builtin_constant_p(((access)&IB_UVERBS_ACCESS_OPTIONAL_RANGE)==0))
try:
    ibv_reg_dmabuf_mr = _libraries['libibverbs'].ibv_reg_dmabuf_mr
    ibv_reg_dmabuf_mr.restype = ctypes.POINTER(struct_ibv_mr)
    ibv_reg_dmabuf_mr.argtypes = [ctypes.POINTER(struct_ibv_pd), uint64_t, size_t, uint64_t, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass

# values for enumeration 'ibv_rereg_mr_err_code'
ibv_rereg_mr_err_code__enumvalues = {
    -1: 'IBV_REREG_MR_ERR_INPUT',
    -2: 'IBV_REREG_MR_ERR_DONT_FORK_NEW',
    -3: 'IBV_REREG_MR_ERR_DO_FORK_OLD',
    -4: 'IBV_REREG_MR_ERR_CMD',
    -5: 'IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW',
}
IBV_REREG_MR_ERR_INPUT = -1
IBV_REREG_MR_ERR_DONT_FORK_NEW = -2
IBV_REREG_MR_ERR_DO_FORK_OLD = -3
IBV_REREG_MR_ERR_CMD = -4
IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW = -5
ibv_rereg_mr_err_code = ctypes.c_int32 # enum
try:
    ibv_rereg_mr = _libraries['libibverbs'].ibv_rereg_mr
    ibv_rereg_mr.restype = ctypes.c_int32
    ibv_rereg_mr.argtypes = [ctypes.POINTER(struct_ibv_mr), ctypes.c_int32, ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    ibv_dereg_mr = _libraries['libibverbs'].ibv_dereg_mr
    ibv_dereg_mr.restype = ctypes.c_int32
    ibv_dereg_mr.argtypes = [ctypes.POINTER(struct_ibv_mr)]
except AttributeError:
    pass
try:
    ibv_alloc_mw = _libraries['libibverbs'].ibv_alloc_mw
    ibv_alloc_mw.restype = ctypes.POINTER(struct_ibv_mw)
    ibv_alloc_mw.argtypes = [ctypes.POINTER(struct_ibv_pd), ibv_mw_type]
except AttributeError:
    pass
try:
    ibv_dealloc_mw = _libraries['libibverbs'].ibv_dealloc_mw
    ibv_dealloc_mw.restype = ctypes.c_int32
    ibv_dealloc_mw.argtypes = [ctypes.POINTER(struct_ibv_mw)]
except AttributeError:
    pass
try:
    ibv_inc_rkey = _libraries['libibverbs'].ibv_inc_rkey
    ibv_inc_rkey.restype = uint32_t
    ibv_inc_rkey.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    ibv_bind_mw = _libraries['libibverbs'].ibv_bind_mw
    ibv_bind_mw.restype = ctypes.c_int32
    ibv_bind_mw.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_mw), ctypes.POINTER(struct_ibv_mw_bind)]
except AttributeError:
    pass
try:
    ibv_create_comp_channel = _libraries['libibverbs'].ibv_create_comp_channel
    ibv_create_comp_channel.restype = ctypes.POINTER(struct_ibv_comp_channel)
    ibv_create_comp_channel.argtypes = [ctypes.POINTER(struct_ibv_context)]
except AttributeError:
    pass
try:
    ibv_destroy_comp_channel = _libraries['libibverbs'].ibv_destroy_comp_channel
    ibv_destroy_comp_channel.restype = ctypes.c_int32
    ibv_destroy_comp_channel.argtypes = [ctypes.POINTER(struct_ibv_comp_channel)]
except AttributeError:
    pass
try:
    ibv_advise_mr = _libraries['libibverbs'].ibv_advise_mr
    ibv_advise_mr.restype = ctypes.c_int32
    ibv_advise_mr.argtypes = [ctypes.POINTER(struct_ibv_pd), ib_uverbs_advise_mr_advice, uint32_t, ctypes.POINTER(struct_ibv_sge), uint32_t]
except AttributeError:
    pass
try:
    ibv_alloc_dm = _libraries['libibverbs'].ibv_alloc_dm
    ibv_alloc_dm.restype = ctypes.POINTER(struct_ibv_dm)
    ibv_alloc_dm.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_alloc_dm_attr)]
except AttributeError:
    pass
try:
    ibv_free_dm = _libraries['libibverbs'].ibv_free_dm
    ibv_free_dm.restype = ctypes.c_int32
    ibv_free_dm.argtypes = [ctypes.POINTER(struct_ibv_dm)]
except AttributeError:
    pass
try:
    ibv_memcpy_to_dm = _libraries['libibverbs'].ibv_memcpy_to_dm
    ibv_memcpy_to_dm.restype = ctypes.c_int32
    ibv_memcpy_to_dm.argtypes = [ctypes.POINTER(struct_ibv_dm), uint64_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    ibv_memcpy_from_dm = _libraries['libibverbs'].ibv_memcpy_from_dm
    ibv_memcpy_from_dm.restype = ctypes.c_int32
    ibv_memcpy_from_dm.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_ibv_dm), uint64_t, size_t]
except AttributeError:
    pass
try:
    ibv_alloc_null_mr = _libraries['libibverbs'].ibv_alloc_null_mr
    ibv_alloc_null_mr.restype = ctypes.POINTER(struct_ibv_mr)
    ibv_alloc_null_mr.argtypes = [ctypes.POINTER(struct_ibv_pd)]
except AttributeError:
    pass
try:
    ibv_reg_dm_mr = _libraries['libibverbs'].ibv_reg_dm_mr
    ibv_reg_dm_mr.restype = ctypes.POINTER(struct_ibv_mr)
    ibv_reg_dm_mr.argtypes = [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_dm), uint64_t, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    ibv_create_cq = _libraries['libibverbs'].ibv_create_cq
    ibv_create_cq.restype = ctypes.POINTER(struct_ibv_cq)
    ibv_create_cq.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct_ibv_comp_channel), ctypes.c_int32]
except AttributeError:
    pass
try:
    ibv_create_cq_ex = _libraries['libibverbs'].ibv_create_cq_ex
    ibv_create_cq_ex.restype = ctypes.POINTER(struct_ibv_cq_ex)
    ibv_create_cq_ex.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_cq_init_attr_ex)]
except AttributeError:
    pass
try:
    ibv_resize_cq = _libraries['libibverbs'].ibv_resize_cq
    ibv_resize_cq.restype = ctypes.c_int32
    ibv_resize_cq.argtypes = [ctypes.POINTER(struct_ibv_cq), ctypes.c_int32]
except AttributeError:
    pass
try:
    ibv_destroy_cq = _libraries['libibverbs'].ibv_destroy_cq
    ibv_destroy_cq.restype = ctypes.c_int32
    ibv_destroy_cq.argtypes = [ctypes.POINTER(struct_ibv_cq)]
except AttributeError:
    pass
try:
    ibv_get_cq_event = _libraries['libibverbs'].ibv_get_cq_event
    ibv_get_cq_event.restype = ctypes.c_int32
    ibv_get_cq_event.argtypes = [ctypes.POINTER(struct_ibv_comp_channel), ctypes.POINTER(ctypes.POINTER(struct_ibv_cq)), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    ibv_ack_cq_events = _libraries['libibverbs'].ibv_ack_cq_events
    ibv_ack_cq_events.restype = None
    ibv_ack_cq_events.argtypes = [ctypes.POINTER(struct_ibv_cq), ctypes.c_uint32]
except AttributeError:
    pass
try:
    ibv_poll_cq = _libraries['libibverbs'].ibv_poll_cq
    ibv_poll_cq.restype = ctypes.c_int32
    ibv_poll_cq.argtypes = [ctypes.POINTER(struct_ibv_cq), ctypes.c_int32, ctypes.POINTER(struct_ibv_wc)]
except AttributeError:
    pass
try:
    ibv_req_notify_cq = _libraries['libibverbs'].ibv_req_notify_cq
    ibv_req_notify_cq.restype = ctypes.c_int32
    ibv_req_notify_cq.argtypes = [ctypes.POINTER(struct_ibv_cq), ctypes.c_int32]
except AttributeError:
    pass
try:
    ibv_modify_cq = _libraries['libibverbs'].ibv_modify_cq
    ibv_modify_cq.restype = ctypes.c_int32
    ibv_modify_cq.argtypes = [ctypes.POINTER(struct_ibv_cq), ctypes.POINTER(struct_ibv_modify_cq_attr)]
except AttributeError:
    pass
try:
    ibv_create_srq = _libraries['libibverbs'].ibv_create_srq
    ibv_create_srq.restype = ctypes.POINTER(struct_ibv_srq)
    ibv_create_srq.argtypes = [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_srq_init_attr)]
except AttributeError:
    pass
try:
    ibv_create_srq_ex = _libraries['libibverbs'].ibv_create_srq_ex
    ibv_create_srq_ex.restype = ctypes.POINTER(struct_ibv_srq)
    ibv_create_srq_ex.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_srq_init_attr_ex)]
except AttributeError:
    pass
try:
    ibv_modify_srq = _libraries['libibverbs'].ibv_modify_srq
    ibv_modify_srq.restype = ctypes.c_int32
    ibv_modify_srq.argtypes = [ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_srq_attr), ctypes.c_int32]
except AttributeError:
    pass
try:
    ibv_query_srq = _libraries['libibverbs'].ibv_query_srq
    ibv_query_srq.restype = ctypes.c_int32
    ibv_query_srq.argtypes = [ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_srq_attr)]
except AttributeError:
    pass
try:
    ibv_get_srq_num = _libraries['libibverbs'].ibv_get_srq_num
    ibv_get_srq_num.restype = ctypes.c_int32
    ibv_get_srq_num.argtypes = [ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    ibv_destroy_srq = _libraries['libibverbs'].ibv_destroy_srq
    ibv_destroy_srq.restype = ctypes.c_int32
    ibv_destroy_srq.argtypes = [ctypes.POINTER(struct_ibv_srq)]
except AttributeError:
    pass
try:
    ibv_post_srq_recv = _libraries['libibverbs'].ibv_post_srq_recv
    ibv_post_srq_recv.restype = ctypes.c_int32
    ibv_post_srq_recv.argtypes = [ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_recv_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_recv_wr))]
except AttributeError:
    pass
try:
    ibv_post_srq_ops = _libraries['libibverbs'].ibv_post_srq_ops
    ibv_post_srq_ops.restype = ctypes.c_int32
    ibv_post_srq_ops.argtypes = [ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_ops_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_ops_wr))]
except AttributeError:
    pass
try:
    ibv_create_qp = _libraries['libibverbs'].ibv_create_qp
    ibv_create_qp.restype = ctypes.POINTER(struct_ibv_qp)
    ibv_create_qp.argtypes = [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_qp_init_attr)]
except AttributeError:
    pass
try:
    ibv_create_qp_ex = _libraries['libibverbs'].ibv_create_qp_ex
    ibv_create_qp_ex.restype = ctypes.POINTER(struct_ibv_qp)
    ibv_create_qp_ex.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_qp_init_attr_ex)]
except AttributeError:
    pass
try:
    ibv_alloc_td = _libraries['libibverbs'].ibv_alloc_td
    ibv_alloc_td.restype = ctypes.POINTER(struct_ibv_td)
    ibv_alloc_td.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_td_init_attr)]
except AttributeError:
    pass
try:
    ibv_dealloc_td = _libraries['libibverbs'].ibv_dealloc_td
    ibv_dealloc_td.restype = ctypes.c_int32
    ibv_dealloc_td.argtypes = [ctypes.POINTER(struct_ibv_td)]
except AttributeError:
    pass
try:
    ibv_alloc_parent_domain = _libraries['libibverbs'].ibv_alloc_parent_domain
    ibv_alloc_parent_domain.restype = ctypes.POINTER(struct_ibv_pd)
    ibv_alloc_parent_domain.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_parent_domain_init_attr)]
except AttributeError:
    pass
try:
    ibv_query_rt_values_ex = _libraries['libibverbs'].ibv_query_rt_values_ex
    ibv_query_rt_values_ex.restype = ctypes.c_int32
    ibv_query_rt_values_ex.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_values_ex)]
except AttributeError:
    pass
try:
    ibv_query_device_ex = _libraries['libibverbs'].ibv_query_device_ex
    ibv_query_device_ex.restype = ctypes.c_int32
    ibv_query_device_ex.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_query_device_ex_input), ctypes.POINTER(struct_ibv_device_attr_ex)]
except AttributeError:
    pass
try:
    ibv_open_qp = _libraries['libibverbs'].ibv_open_qp
    ibv_open_qp.restype = ctypes.POINTER(struct_ibv_qp)
    ibv_open_qp.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_qp_open_attr)]
except AttributeError:
    pass
try:
    ibv_modify_qp = _libraries['libibverbs'].ibv_modify_qp
    ibv_modify_qp.restype = ctypes.c_int32
    ibv_modify_qp.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_qp_attr), ctypes.c_int32]
except AttributeError:
    pass
try:
    ibv_modify_qp_rate_limit = _libraries['libibverbs'].ibv_modify_qp_rate_limit
    ibv_modify_qp_rate_limit.restype = ctypes.c_int32
    ibv_modify_qp_rate_limit.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_qp_rate_limit_attr)]
except AttributeError:
    pass
try:
    ibv_query_qp_data_in_order = _libraries['libibverbs'].ibv_query_qp_data_in_order
    ibv_query_qp_data_in_order.restype = ctypes.c_int32
    ibv_query_qp_data_in_order.argtypes = [ctypes.POINTER(struct_ibv_qp), ibv_wr_opcode, uint32_t]
except AttributeError:
    pass
try:
    ibv_query_qp = _libraries['libibverbs'].ibv_query_qp
    ibv_query_qp.restype = ctypes.c_int32
    ibv_query_qp.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_qp_attr), ctypes.c_int32, ctypes.POINTER(struct_ibv_qp_init_attr)]
except AttributeError:
    pass
try:
    ibv_destroy_qp = _libraries['libibverbs'].ibv_destroy_qp
    ibv_destroy_qp.restype = ctypes.c_int32
    ibv_destroy_qp.argtypes = [ctypes.POINTER(struct_ibv_qp)]
except AttributeError:
    pass
try:
    ibv_create_wq = _libraries['libibverbs'].ibv_create_wq
    ibv_create_wq.restype = ctypes.POINTER(struct_ibv_wq)
    ibv_create_wq.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_wq_init_attr)]
except AttributeError:
    pass
try:
    ibv_modify_wq = _libraries['libibverbs'].ibv_modify_wq
    ibv_modify_wq.restype = ctypes.c_int32
    ibv_modify_wq.argtypes = [ctypes.POINTER(struct_ibv_wq), ctypes.POINTER(struct_ibv_wq_attr)]
except AttributeError:
    pass
try:
    ibv_destroy_wq = _libraries['libibverbs'].ibv_destroy_wq
    ibv_destroy_wq.restype = ctypes.c_int32
    ibv_destroy_wq.argtypes = [ctypes.POINTER(struct_ibv_wq)]
except AttributeError:
    pass
try:
    ibv_create_rwq_ind_table = _libraries['libibverbs'].ibv_create_rwq_ind_table
    ibv_create_rwq_ind_table.restype = ctypes.POINTER(struct_ibv_rwq_ind_table)
    ibv_create_rwq_ind_table.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_rwq_ind_table_init_attr)]
except AttributeError:
    pass
try:
    ibv_destroy_rwq_ind_table = _libraries['libibverbs'].ibv_destroy_rwq_ind_table
    ibv_destroy_rwq_ind_table.restype = ctypes.c_int32
    ibv_destroy_rwq_ind_table.argtypes = [ctypes.POINTER(struct_ibv_rwq_ind_table)]
except AttributeError:
    pass
try:
    ibv_post_send = _libraries['libibverbs'].ibv_post_send
    ibv_post_send.restype = ctypes.c_int32
    ibv_post_send.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_send_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_send_wr))]
except AttributeError:
    pass
try:
    ibv_post_recv = _libraries['libibverbs'].ibv_post_recv
    ibv_post_recv.restype = ctypes.c_int32
    ibv_post_recv.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_recv_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_recv_wr))]
except AttributeError:
    pass
try:
    ibv_create_ah = _libraries['libibverbs'].ibv_create_ah
    ibv_create_ah.restype = ctypes.POINTER(struct_ibv_ah)
    ibv_create_ah.argtypes = [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_ah_attr)]
except AttributeError:
    pass
try:
    ibv_init_ah_from_wc = _libraries['libibverbs'].ibv_init_ah_from_wc
    ibv_init_ah_from_wc.restype = ctypes.c_int32
    ibv_init_ah_from_wc.argtypes = [ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.POINTER(struct_ibv_wc), ctypes.POINTER(struct_ibv_grh), ctypes.POINTER(struct_ibv_ah_attr)]
except AttributeError:
    pass
try:
    ibv_create_ah_from_wc = _libraries['libibverbs'].ibv_create_ah_from_wc
    ibv_create_ah_from_wc.restype = ctypes.POINTER(struct_ibv_ah)
    ibv_create_ah_from_wc.argtypes = [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_wc), ctypes.POINTER(struct_ibv_grh), uint8_t]
except AttributeError:
    pass
try:
    ibv_destroy_ah = _libraries['libibverbs'].ibv_destroy_ah
    ibv_destroy_ah.restype = ctypes.c_int32
    ibv_destroy_ah.argtypes = [ctypes.POINTER(struct_ibv_ah)]
except AttributeError:
    pass
try:
    ibv_attach_mcast = _libraries['libibverbs'].ibv_attach_mcast
    ibv_attach_mcast.restype = ctypes.c_int32
    ibv_attach_mcast.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(union_ibv_gid), uint16_t]
except AttributeError:
    pass
try:
    ibv_detach_mcast = _libraries['libibverbs'].ibv_detach_mcast
    ibv_detach_mcast.restype = ctypes.c_int32
    ibv_detach_mcast.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(union_ibv_gid), uint16_t]
except AttributeError:
    pass
try:
    ibv_fork_init = _libraries['libibverbs'].ibv_fork_init
    ibv_fork_init.restype = ctypes.c_int32
    ibv_fork_init.argtypes = []
except AttributeError:
    pass
try:
    ibv_is_fork_initialized = _libraries['libibverbs'].ibv_is_fork_initialized
    ibv_is_fork_initialized.restype = ibv_fork_status
    ibv_is_fork_initialized.argtypes = []
except AttributeError:
    pass
try:
    ibv_node_type_str = _libraries['libibverbs'].ibv_node_type_str
    ibv_node_type_str.restype = ctypes.POINTER(ctypes.c_char)
    ibv_node_type_str.argtypes = [ibv_node_type]
except AttributeError:
    pass
try:
    ibv_port_state_str = _libraries['libibverbs'].ibv_port_state_str
    ibv_port_state_str.restype = ctypes.POINTER(ctypes.c_char)
    ibv_port_state_str.argtypes = [ibv_port_state]
except AttributeError:
    pass
try:
    ibv_event_type_str = _libraries['libibverbs'].ibv_event_type_str
    ibv_event_type_str.restype = ctypes.POINTER(ctypes.c_char)
    ibv_event_type_str.argtypes = [ibv_event_type]
except AttributeError:
    pass
try:
    ibv_resolve_eth_l2_from_gid = _libraries['libibverbs'].ibv_resolve_eth_l2_from_gid
    ibv_resolve_eth_l2_from_gid.restype = ctypes.c_int32
    ibv_resolve_eth_l2_from_gid.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_ah_attr), ctypes.c_ubyte * 6, ctypes.POINTER(ctypes.c_uint16)]
except AttributeError:
    pass
try:
    ibv_is_qpt_supported = _libraries['libibverbs'].ibv_is_qpt_supported
    ibv_is_qpt_supported.restype = ctypes.c_int32
    ibv_is_qpt_supported.argtypes = [uint32_t, ibv_qp_type]
except AttributeError:
    pass
try:
    ibv_create_counters = _libraries['libibverbs'].ibv_create_counters
    ibv_create_counters.restype = ctypes.POINTER(struct_ibv_counters)
    ibv_create_counters.argtypes = [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_counters_init_attr)]
except AttributeError:
    pass
try:
    ibv_destroy_counters = _libraries['libibverbs'].ibv_destroy_counters
    ibv_destroy_counters.restype = ctypes.c_int32
    ibv_destroy_counters.argtypes = [ctypes.POINTER(struct_ibv_counters)]
except AttributeError:
    pass
try:
    ibv_attach_counters_point_flow = _libraries['libibverbs'].ibv_attach_counters_point_flow
    ibv_attach_counters_point_flow.restype = ctypes.c_int32
    ibv_attach_counters_point_flow.argtypes = [ctypes.POINTER(struct_ibv_counters), ctypes.POINTER(struct_ibv_counter_attach_attr), ctypes.POINTER(struct_ibv_flow)]
except AttributeError:
    pass
try:
    ibv_read_counters = _libraries['libibverbs'].ibv_read_counters
    ibv_read_counters.restype = ctypes.c_int32
    ibv_read_counters.argtypes = [ctypes.POINTER(struct_ibv_counters), ctypes.POINTER(ctypes.c_uint64), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    ibv_flow_label_to_udp_sport = _libraries['libibverbs'].ibv_flow_label_to_udp_sport
    ibv_flow_label_to_udp_sport.restype = uint16_t
    ibv_flow_label_to_udp_sport.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    ibv_set_ece = _libraries['libibverbs'].ibv_set_ece
    ibv_set_ece.restype = ctypes.c_int32
    ibv_set_ece.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_ece)]
except AttributeError:
    pass
try:
    ibv_query_ece = _libraries['libibverbs'].ibv_query_ece
    ibv_query_ece.restype = ctypes.c_int32
    ibv_query_ece.argtypes = [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_ece)]
except AttributeError:
    pass
__all__ = \
    ['ETHERNET_LL_SIZE', 'IBV_ACCESS_FLUSH_GLOBAL',
    'IBV_ACCESS_FLUSH_PERSISTENT', 'IBV_ACCESS_HUGETLB',
    'IBV_ACCESS_LOCAL_WRITE', 'IBV_ACCESS_MW_BIND',
    'IBV_ACCESS_ON_DEMAND', 'IBV_ACCESS_OPTIONAL_FIRST',
    'IBV_ACCESS_OPTIONAL_RANGE', 'IBV_ACCESS_RELAXED_ORDERING',
    'IBV_ACCESS_REMOTE_ATOMIC', 'IBV_ACCESS_REMOTE_READ',
    'IBV_ACCESS_REMOTE_WRITE', 'IBV_ACCESS_ZERO_BASED',
    'IBV_ADVISE_MR_ADVICE_PREFETCH',
    'IBV_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT',
    'IBV_ADVISE_MR_ADVICE_PREFETCH_WRITE', 'IBV_ADVISE_MR_FLAG_FLUSH',
    'IBV_ATOMIC_GLOB', 'IBV_ATOMIC_HCA', 'IBV_ATOMIC_NONE',
    'IBV_COUNTER_BYTES', 'IBV_COUNTER_PACKETS',
    'IBV_CQ_ATTR_MODERATE', 'IBV_CQ_ATTR_RESERVED',
    'IBV_CQ_INIT_ATTR_MASK_FLAGS', 'IBV_CQ_INIT_ATTR_MASK_PD',
    'IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN',
    'IBV_CREATE_CQ_ATTR_SINGLE_THREADED',
    'IBV_CREATE_CQ_SUP_WC_FLAGS', 'IBV_CREATE_IND_TABLE_RESERVED',
    'IBV_DEVICE_AUTO_PATH_MIG', 'IBV_DEVICE_BAD_PKEY_CNTR',
    'IBV_DEVICE_BAD_QKEY_CNTR', 'IBV_DEVICE_CHANGE_PHY_PORT',
    'IBV_DEVICE_CURR_QP_STATE_MOD', 'IBV_DEVICE_INIT_TYPE',
    'IBV_DEVICE_MANAGED_FLOW_STEERING',
    'IBV_DEVICE_MEM_MGT_EXTENSIONS', 'IBV_DEVICE_MEM_WINDOW',
    'IBV_DEVICE_MEM_WINDOW_TYPE_2A', 'IBV_DEVICE_MEM_WINDOW_TYPE_2B',
    'IBV_DEVICE_N_NOTIFY_CQ', 'IBV_DEVICE_PCI_WRITE_END_PADDING',
    'IBV_DEVICE_PORT_ACTIVE_EVENT', 'IBV_DEVICE_RAW_IP_CSUM',
    'IBV_DEVICE_RAW_MULTI', 'IBV_DEVICE_RAW_SCATTER_FCS',
    'IBV_DEVICE_RC_IP_CSUM', 'IBV_DEVICE_RC_RNR_NAK_GEN',
    'IBV_DEVICE_RESIZE_MAX_WR', 'IBV_DEVICE_SHUTDOWN_PORT',
    'IBV_DEVICE_SRQ_RESIZE', 'IBV_DEVICE_SYS_IMAGE_GUID',
    'IBV_DEVICE_UD_AV_PORT_ENFORCE', 'IBV_DEVICE_UD_IP_CSUM',
    'IBV_DEVICE_XRC', 'IBV_DM_MASK_HANDLE',
    'IBV_EVENT_CLIENT_REREGISTER', 'IBV_EVENT_COMM_EST',
    'IBV_EVENT_CQ_ERR', 'IBV_EVENT_DEVICE_FATAL',
    'IBV_EVENT_GID_CHANGE', 'IBV_EVENT_LID_CHANGE',
    'IBV_EVENT_PATH_MIG', 'IBV_EVENT_PATH_MIG_ERR',
    'IBV_EVENT_PKEY_CHANGE', 'IBV_EVENT_PORT_ACTIVE',
    'IBV_EVENT_PORT_ERR', 'IBV_EVENT_QP_ACCESS_ERR',
    'IBV_EVENT_QP_FATAL', 'IBV_EVENT_QP_LAST_WQE_REACHED',
    'IBV_EVENT_QP_REQ_ERR', 'IBV_EVENT_SM_CHANGE',
    'IBV_EVENT_SQ_DRAINED', 'IBV_EVENT_SRQ_ERR',
    'IBV_EVENT_SRQ_LIMIT_REACHED', 'IBV_EVENT_WQ_FATAL',
    'IBV_FLOW_ACTION_ESP_FLAGS_DECRYPT',
    'IBV_FLOW_ACTION_ESP_FLAGS_ENCRYPT',
    'IBV_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW',
    'IBV_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD',
    'IBV_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO',
    'IBV_FLOW_ACTION_ESP_FLAGS_TRANSPORT',
    'IBV_FLOW_ACTION_ESP_FLAGS_TUNNEL',
    'IBV_FLOW_ACTION_ESP_KEYMAT_AES_GCM',
    'IBV_FLOW_ACTION_ESP_MASK_ESN', 'IBV_FLOW_ACTION_ESP_REPLAY_BMP',
    'IBV_FLOW_ACTION_ESP_REPLAY_NONE', 'IBV_FLOW_ACTION_IV_ALGO_SEQ',
    'IBV_FLOW_ATTR_ALL_DEFAULT', 'IBV_FLOW_ATTR_FLAGS_DONT_TRAP',
    'IBV_FLOW_ATTR_FLAGS_EGRESS', 'IBV_FLOW_ATTR_MC_DEFAULT',
    'IBV_FLOW_ATTR_NORMAL', 'IBV_FLOW_ATTR_SNIFFER',
    'IBV_FLOW_SPEC_ACTION_COUNT', 'IBV_FLOW_SPEC_ACTION_DROP',
    'IBV_FLOW_SPEC_ACTION_HANDLE', 'IBV_FLOW_SPEC_ACTION_TAG',
    'IBV_FLOW_SPEC_ESP', 'IBV_FLOW_SPEC_ETH', 'IBV_FLOW_SPEC_GRE',
    'IBV_FLOW_SPEC_INNER', 'IBV_FLOW_SPEC_IPV4',
    'IBV_FLOW_SPEC_IPV4_EXT', 'IBV_FLOW_SPEC_IPV6',
    'IBV_FLOW_SPEC_MPLS', 'IBV_FLOW_SPEC_TCP', 'IBV_FLOW_SPEC_UDP',
    'IBV_FLOW_SPEC_VXLAN_TUNNEL', 'IBV_FLUSH_GLOBAL', 'IBV_FLUSH_MR',
    'IBV_FLUSH_PERSISTENT', 'IBV_FLUSH_RANGE', 'IBV_FORK_DISABLED',
    'IBV_FORK_ENABLED', 'IBV_FORK_UNNEEDED', 'IBV_GID_TYPE_IB',
    'IBV_GID_TYPE_ROCE_V1', 'IBV_GID_TYPE_ROCE_V2',
    'IBV_LINK_LAYER_ETHERNET', 'IBV_LINK_LAYER_INFINIBAND',
    'IBV_LINK_LAYER_UNSPECIFIED', 'IBV_MIG_ARMED', 'IBV_MIG_MIGRATED',
    'IBV_MIG_REARM', 'IBV_MTU_1024', 'IBV_MTU_2048', 'IBV_MTU_256',
    'IBV_MTU_4096', 'IBV_MTU_512', 'IBV_MW_TYPE_1', 'IBV_MW_TYPE_2',
    'IBV_NODE_CA', 'IBV_NODE_RNIC', 'IBV_NODE_ROUTER',
    'IBV_NODE_SWITCH', 'IBV_NODE_UNKNOWN', 'IBV_NODE_UNSPECIFIED',
    'IBV_NODE_USNIC', 'IBV_NODE_USNIC_UDP', 'IBV_ODP_SUPPORT',
    'IBV_ODP_SUPPORT_ATOMIC', 'IBV_ODP_SUPPORT_IMPLICIT',
    'IBV_ODP_SUPPORT_READ', 'IBV_ODP_SUPPORT_RECV',
    'IBV_ODP_SUPPORT_SEND', 'IBV_ODP_SUPPORT_SRQ_RECV',
    'IBV_ODP_SUPPORT_WRITE', 'IBV_OPS_SIGNALED', 'IBV_OPS_TM_SYNC',
    'IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS',
    'IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT',
    'IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP',
    'IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP',
    'IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP', 'IBV_PORT_ACTIVE',
    'IBV_PORT_ACTIVE_DEFER', 'IBV_PORT_ARMED',
    'IBV_PORT_AUTO_MIGR_SUP', 'IBV_PORT_BOOT_MGMT_SUP',
    'IBV_PORT_CAP_MASK2_SUP', 'IBV_PORT_CAP_MASK_NOTICE_SUP',
    'IBV_PORT_CLIENT_REG_SUP', 'IBV_PORT_CM_SUP',
    'IBV_PORT_DEVICE_MGMT_SUP', 'IBV_PORT_DOWN',
    'IBV_PORT_DR_NOTICE_SUP', 'IBV_PORT_EXTENDED_SPEEDS_SUP',
    'IBV_PORT_INFO_EXT_SUP', 'IBV_PORT_INIT',
    'IBV_PORT_IP_BASED_GIDS', 'IBV_PORT_LED_INFO_SUP',
    'IBV_PORT_LINK_LATENCY_SUP', 'IBV_PORT_LINK_SPEED_HDR_SUP',
    'IBV_PORT_LINK_SPEED_NDR_SUP', 'IBV_PORT_LINK_SPEED_XDR_SUP',
    'IBV_PORT_LINK_WIDTH_2X_SUP', 'IBV_PORT_MKEY_NVRAM',
    'IBV_PORT_NOP', 'IBV_PORT_NOTICE_SUP', 'IBV_PORT_OPT_IPD_SUP',
    'IBV_PORT_PKEY_NVRAM', 'IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP',
    'IBV_PORT_REINIT_SUP', 'IBV_PORT_SET_NODE_DESC_SUP',
    'IBV_PORT_SL_MAP_SUP', 'IBV_PORT_SM', 'IBV_PORT_SNMP_TUNNEL_SUP',
    'IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP',
    'IBV_PORT_SYS_IMAGE_GUID_SUP', 'IBV_PORT_TRAP_SUP',
    'IBV_PORT_VENDOR_CLASS_SUP', 'IBV_PORT_VIRT_SUP',
    'IBV_QPF_GRH_REQUIRED', 'IBV_QPS_ERR', 'IBV_QPS_INIT',
    'IBV_QPS_RESET', 'IBV_QPS_RTR', 'IBV_QPS_RTS', 'IBV_QPS_SQD',
    'IBV_QPS_SQE', 'IBV_QPS_UNKNOWN', 'IBV_QPT_DRIVER',
    'IBV_QPT_RAW_PACKET', 'IBV_QPT_RC', 'IBV_QPT_UC', 'IBV_QPT_UD',
    'IBV_QPT_XRC_RECV', 'IBV_QPT_XRC_SEND', 'IBV_QP_ACCESS_FLAGS',
    'IBV_QP_ALT_PATH', 'IBV_QP_AV', 'IBV_QP_CAP',
    'IBV_QP_CREATE_BLOCK_SELF_MCAST_LB',
    'IBV_QP_CREATE_CVLAN_STRIPPING',
    'IBV_QP_CREATE_PCI_WRITE_END_PADDING',
    'IBV_QP_CREATE_SCATTER_FCS', 'IBV_QP_CREATE_SOURCE_QPN',
    'IBV_QP_CUR_STATE', 'IBV_QP_DEST_QPN',
    'IBV_QP_EN_SQD_ASYNC_NOTIFY', 'IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP',
    'IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD',
    'IBV_QP_EX_WITH_ATOMIC_WRITE', 'IBV_QP_EX_WITH_BIND_MW',
    'IBV_QP_EX_WITH_FLUSH', 'IBV_QP_EX_WITH_LOCAL_INV',
    'IBV_QP_EX_WITH_RDMA_READ', 'IBV_QP_EX_WITH_RDMA_WRITE',
    'IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM', 'IBV_QP_EX_WITH_SEND',
    'IBV_QP_EX_WITH_SEND_WITH_IMM', 'IBV_QP_EX_WITH_SEND_WITH_INV',
    'IBV_QP_EX_WITH_TSO', 'IBV_QP_INIT_ATTR_CREATE_FLAGS',
    'IBV_QP_INIT_ATTR_IND_TABLE', 'IBV_QP_INIT_ATTR_MAX_TSO_HEADER',
    'IBV_QP_INIT_ATTR_PD', 'IBV_QP_INIT_ATTR_RX_HASH',
    'IBV_QP_INIT_ATTR_SEND_OPS_FLAGS', 'IBV_QP_INIT_ATTR_XRCD',
    'IBV_QP_MAX_DEST_RD_ATOMIC', 'IBV_QP_MAX_QP_RD_ATOMIC',
    'IBV_QP_MIN_RNR_TIMER', 'IBV_QP_OPEN_ATTR_CONTEXT',
    'IBV_QP_OPEN_ATTR_NUM', 'IBV_QP_OPEN_ATTR_RESERVED',
    'IBV_QP_OPEN_ATTR_TYPE', 'IBV_QP_OPEN_ATTR_XRCD',
    'IBV_QP_PATH_MIG_STATE', 'IBV_QP_PATH_MTU', 'IBV_QP_PKEY_INDEX',
    'IBV_QP_PORT', 'IBV_QP_QKEY', 'IBV_QP_RATE_LIMIT',
    'IBV_QP_RETRY_CNT', 'IBV_QP_RNR_RETRY', 'IBV_QP_RQ_PSN',
    'IBV_QP_SQ_PSN', 'IBV_QP_STATE', 'IBV_QP_TIMEOUT',
    'IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES',
    'IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS',
    'IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG', 'IBV_RATE_100_GBPS',
    'IBV_RATE_10_GBPS', 'IBV_RATE_112_GBPS', 'IBV_RATE_1200_GBPS',
    'IBV_RATE_120_GBPS', 'IBV_RATE_14_GBPS', 'IBV_RATE_168_GBPS',
    'IBV_RATE_200_GBPS', 'IBV_RATE_20_GBPS', 'IBV_RATE_25_GBPS',
    'IBV_RATE_28_GBPS', 'IBV_RATE_2_5_GBPS', 'IBV_RATE_300_GBPS',
    'IBV_RATE_30_GBPS', 'IBV_RATE_400_GBPS', 'IBV_RATE_40_GBPS',
    'IBV_RATE_50_GBPS', 'IBV_RATE_56_GBPS', 'IBV_RATE_5_GBPS',
    'IBV_RATE_600_GBPS', 'IBV_RATE_60_GBPS', 'IBV_RATE_800_GBPS',
    'IBV_RATE_80_GBPS', 'IBV_RATE_MAX',
    'IBV_RAW_PACKET_CAP_CVLAN_STRIPPING',
    'IBV_RAW_PACKET_CAP_DELAY_DROP', 'IBV_RAW_PACKET_CAP_IP_CSUM',
    'IBV_RAW_PACKET_CAP_SCATTER_FCS',
    'IBV_READ_COUNTERS_ATTR_PREFER_CACHED',
    'IBV_REREG_MR_CHANGE_ACCESS', 'IBV_REREG_MR_CHANGE_PD',
    'IBV_REREG_MR_CHANGE_TRANSLATION', 'IBV_REREG_MR_ERR_CMD',
    'IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW',
    'IBV_REREG_MR_ERR_DONT_FORK_NEW', 'IBV_REREG_MR_ERR_DO_FORK_OLD',
    'IBV_REREG_MR_ERR_INPUT', 'IBV_REREG_MR_FLAGS_SUPPORTED',
    'IBV_RX_HASH_DST_IPV4', 'IBV_RX_HASH_DST_IPV6',
    'IBV_RX_HASH_DST_PORT_TCP', 'IBV_RX_HASH_DST_PORT_UDP',
    'IBV_RX_HASH_FUNC_TOEPLITZ', 'IBV_RX_HASH_INNER',
    'IBV_RX_HASH_IPSEC_SPI', 'IBV_RX_HASH_SRC_IPV4',
    'IBV_RX_HASH_SRC_IPV6', 'IBV_RX_HASH_SRC_PORT_TCP',
    'IBV_RX_HASH_SRC_PORT_UDP', 'IBV_SEND_FENCE', 'IBV_SEND_INLINE',
    'IBV_SEND_IP_CSUM', 'IBV_SEND_SIGNALED', 'IBV_SEND_SOLICITED',
    'IBV_SRQT_BASIC', 'IBV_SRQT_TM', 'IBV_SRQT_XRC',
    'IBV_SRQ_INIT_ATTR_CQ', 'IBV_SRQ_INIT_ATTR_PD',
    'IBV_SRQ_INIT_ATTR_RESERVED', 'IBV_SRQ_INIT_ATTR_TM',
    'IBV_SRQ_INIT_ATTR_TYPE', 'IBV_SRQ_INIT_ATTR_XRCD',
    'IBV_SRQ_LIMIT', 'IBV_SRQ_MAX_WR', 'IBV_SYSFS_NAME_MAX',
    'IBV_SYSFS_PATH_MAX', 'IBV_TM_CAP_RC', 'IBV_TRANSPORT_IB',
    'IBV_TRANSPORT_IWARP', 'IBV_TRANSPORT_UNKNOWN',
    'IBV_TRANSPORT_UNSPECIFIED', 'IBV_TRANSPORT_USNIC',
    'IBV_TRANSPORT_USNIC_UDP', 'IBV_VALUES_MASK_RAW_CLOCK',
    'IBV_VALUES_MASK_RESERVED', 'IBV_WC_ATOMIC_WRITE',
    'IBV_WC_BAD_RESP_ERR', 'IBV_WC_BIND_MW', 'IBV_WC_COMP_SWAP',
    'IBV_WC_DRIVER1', 'IBV_WC_DRIVER2', 'IBV_WC_DRIVER3',
    'IBV_WC_EX_WITH_BYTE_LEN', 'IBV_WC_EX_WITH_COMPLETION_TIMESTAMP',
    'IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK',
    'IBV_WC_EX_WITH_CVLAN', 'IBV_WC_EX_WITH_DLID_PATH_BITS',
    'IBV_WC_EX_WITH_FLOW_TAG', 'IBV_WC_EX_WITH_IMM',
    'IBV_WC_EX_WITH_QP_NUM', 'IBV_WC_EX_WITH_SL',
    'IBV_WC_EX_WITH_SLID', 'IBV_WC_EX_WITH_SRC_QP',
    'IBV_WC_EX_WITH_TM_INFO', 'IBV_WC_FATAL_ERR', 'IBV_WC_FETCH_ADD',
    'IBV_WC_FLUSH', 'IBV_WC_GENERAL_ERR', 'IBV_WC_GRH',
    'IBV_WC_INV_EECN_ERR', 'IBV_WC_INV_EEC_STATE_ERR',
    'IBV_WC_IP_CSUM_OK', 'IBV_WC_IP_CSUM_OK_SHIFT',
    'IBV_WC_LOCAL_INV', 'IBV_WC_LOC_ACCESS_ERR',
    'IBV_WC_LOC_EEC_OP_ERR', 'IBV_WC_LOC_LEN_ERR',
    'IBV_WC_LOC_PROT_ERR', 'IBV_WC_LOC_QP_OP_ERR',
    'IBV_WC_LOC_RDD_VIOL_ERR', 'IBV_WC_MW_BIND_ERR',
    'IBV_WC_RDMA_READ', 'IBV_WC_RDMA_WRITE', 'IBV_WC_RECV',
    'IBV_WC_RECV_RDMA_WITH_IMM', 'IBV_WC_REM_ABORT_ERR',
    'IBV_WC_REM_ACCESS_ERR', 'IBV_WC_REM_INV_RD_REQ_ERR',
    'IBV_WC_REM_INV_REQ_ERR', 'IBV_WC_REM_OP_ERR',
    'IBV_WC_RESP_TIMEOUT_ERR', 'IBV_WC_RETRY_EXC_ERR',
    'IBV_WC_RNR_RETRY_EXC_ERR', 'IBV_WC_SEND',
    'IBV_WC_STANDARD_FLAGS', 'IBV_WC_SUCCESS', 'IBV_WC_TM_ADD',
    'IBV_WC_TM_DATA_VALID', 'IBV_WC_TM_DEL', 'IBV_WC_TM_ERR',
    'IBV_WC_TM_MATCH', 'IBV_WC_TM_NO_TAG', 'IBV_WC_TM_RECV',
    'IBV_WC_TM_RNDV_INCOMPLETE', 'IBV_WC_TM_SYNC',
    'IBV_WC_TM_SYNC_REQ', 'IBV_WC_TSO', 'IBV_WC_WITH_IMM',
    'IBV_WC_WITH_INV', 'IBV_WC_WR_FLUSH_ERR', 'IBV_WQS_ERR',
    'IBV_WQS_RDY', 'IBV_WQS_RESET', 'IBV_WQS_UNKNOWN', 'IBV_WQT_RQ',
    'IBV_WQ_ATTR_CURR_STATE', 'IBV_WQ_ATTR_FLAGS',
    'IBV_WQ_ATTR_RESERVED', 'IBV_WQ_ATTR_STATE',
    'IBV_WQ_FLAGS_CVLAN_STRIPPING', 'IBV_WQ_FLAGS_DELAY_DROP',
    'IBV_WQ_FLAGS_PCI_WRITE_END_PADDING', 'IBV_WQ_FLAGS_RESERVED',
    'IBV_WQ_FLAGS_SCATTER_FCS', 'IBV_WQ_INIT_ATTR_FLAGS',
    'IBV_WQ_INIT_ATTR_RESERVED', 'IBV_WR_ATOMIC_CMP_AND_SWP',
    'IBV_WR_ATOMIC_FETCH_AND_ADD', 'IBV_WR_ATOMIC_WRITE',
    'IBV_WR_BIND_MW', 'IBV_WR_DRIVER1', 'IBV_WR_FLUSH',
    'IBV_WR_LOCAL_INV', 'IBV_WR_RDMA_READ', 'IBV_WR_RDMA_WRITE',
    'IBV_WR_RDMA_WRITE_WITH_IMM', 'IBV_WR_SEND',
    'IBV_WR_SEND_WITH_IMM', 'IBV_WR_SEND_WITH_INV', 'IBV_WR_TAG_ADD',
    'IBV_WR_TAG_DEL', 'IBV_WR_TAG_SYNC', 'IBV_WR_TSO',
    'IBV_XRCD_INIT_ATTR_FD', 'IBV_XRCD_INIT_ATTR_OFLAGS',
    'IBV_XRCD_INIT_ATTR_RESERVED', 'IB_DEVICE_NAME_MAX',
    'IB_FLUSH_GLOBAL', 'IB_FLUSH_MR', 'IB_FLUSH_PERSISTENT',
    'IB_FLUSH_RANGE', 'IB_GRH_FLOWLABEL_MASK',
    'IB_ROCE_UDP_ENCAP_VALID_PORT_MAX',
    'IB_ROCE_UDP_ENCAP_VALID_PORT_MIN', 'IB_USER_IOCTL_VERBS_H',
    'IB_USER_VERBS_ABI_VERSION', 'IB_USER_VERBS_CMD_ALLOC_MW',
    'IB_USER_VERBS_CMD_ALLOC_PD', 'IB_USER_VERBS_CMD_ATTACH_MCAST',
    'IB_USER_VERBS_CMD_BIND_MW', 'IB_USER_VERBS_CMD_CLOSE_XRCD',
    'IB_USER_VERBS_CMD_COMMAND_MASK', 'IB_USER_VERBS_CMD_CREATE_AH',
    'IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL',
    'IB_USER_VERBS_CMD_CREATE_CQ', 'IB_USER_VERBS_CMD_CREATE_QP',
    'IB_USER_VERBS_CMD_CREATE_SRQ', 'IB_USER_VERBS_CMD_CREATE_XSRQ',
    'IB_USER_VERBS_CMD_DEALLOC_MW', 'IB_USER_VERBS_CMD_DEALLOC_PD',
    'IB_USER_VERBS_CMD_DEREG_MR', 'IB_USER_VERBS_CMD_DESTROY_AH',
    'IB_USER_VERBS_CMD_DESTROY_CQ', 'IB_USER_VERBS_CMD_DESTROY_QP',
    'IB_USER_VERBS_CMD_DESTROY_SRQ', 'IB_USER_VERBS_CMD_DETACH_MCAST',
    'IB_USER_VERBS_CMD_FLAG_EXTENDED',
    'IB_USER_VERBS_CMD_GET_CONTEXT', 'IB_USER_VERBS_CMD_MODIFY_AH',
    'IB_USER_VERBS_CMD_MODIFY_QP', 'IB_USER_VERBS_CMD_MODIFY_SRQ',
    'IB_USER_VERBS_CMD_OPEN_QP', 'IB_USER_VERBS_CMD_OPEN_XRCD',
    'IB_USER_VERBS_CMD_PEEK_CQ', 'IB_USER_VERBS_CMD_POLL_CQ',
    'IB_USER_VERBS_CMD_POST_RECV', 'IB_USER_VERBS_CMD_POST_SEND',
    'IB_USER_VERBS_CMD_POST_SRQ_RECV', 'IB_USER_VERBS_CMD_QUERY_AH',
    'IB_USER_VERBS_CMD_QUERY_DEVICE', 'IB_USER_VERBS_CMD_QUERY_MR',
    'IB_USER_VERBS_CMD_QUERY_PORT', 'IB_USER_VERBS_CMD_QUERY_QP',
    'IB_USER_VERBS_CMD_QUERY_SRQ', 'IB_USER_VERBS_CMD_REG_MR',
    'IB_USER_VERBS_CMD_REG_SMR', 'IB_USER_VERBS_CMD_REQ_NOTIFY_CQ',
    'IB_USER_VERBS_CMD_REREG_MR', 'IB_USER_VERBS_CMD_RESIZE_CQ',
    'IB_USER_VERBS_CMD_THRESHOLD', 'IB_USER_VERBS_EX_CMD_CREATE_CQ',
    'IB_USER_VERBS_EX_CMD_CREATE_FLOW',
    'IB_USER_VERBS_EX_CMD_CREATE_QP',
    'IB_USER_VERBS_EX_CMD_CREATE_RWQ_IND_TBL',
    'IB_USER_VERBS_EX_CMD_CREATE_WQ',
    'IB_USER_VERBS_EX_CMD_DESTROY_FLOW',
    'IB_USER_VERBS_EX_CMD_DESTROY_RWQ_IND_TBL',
    'IB_USER_VERBS_EX_CMD_DESTROY_WQ',
    'IB_USER_VERBS_EX_CMD_MODIFY_CQ',
    'IB_USER_VERBS_EX_CMD_MODIFY_QP',
    'IB_USER_VERBS_EX_CMD_MODIFY_WQ',
    'IB_USER_VERBS_EX_CMD_QUERY_DEVICE', 'IB_USER_VERBS_H',
    'IB_USER_VERBS_MAX_LOG_IND_TBL_SIZE',
    'IB_UVERBS_ACCESS_FLUSH_GLOBAL',
    'IB_UVERBS_ACCESS_FLUSH_PERSISTENT', 'IB_UVERBS_ACCESS_HUGETLB',
    'IB_UVERBS_ACCESS_LOCAL_WRITE', 'IB_UVERBS_ACCESS_MW_BIND',
    'IB_UVERBS_ACCESS_ON_DEMAND', 'IB_UVERBS_ACCESS_OPTIONAL_FIRST',
    'IB_UVERBS_ACCESS_OPTIONAL_LAST',
    'IB_UVERBS_ACCESS_OPTIONAL_RANGE',
    'IB_UVERBS_ACCESS_RELAXED_ORDERING',
    'IB_UVERBS_ACCESS_REMOTE_ATOMIC', 'IB_UVERBS_ACCESS_REMOTE_READ',
    'IB_UVERBS_ACCESS_REMOTE_WRITE', 'IB_UVERBS_ACCESS_ZERO_BASED',
    'IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH',
    'IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT',
    'IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE',
    'IB_UVERBS_ADVISE_MR_FLAG_FLUSH',
    'IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS',
    'IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN',
    'IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION',
    'IB_UVERBS_CREATE_QP_MASK_IND_TABLE',
    'IB_UVERBS_CREATE_QP_SUP_COMP_MASK',
    'IB_UVERBS_DEVICE_ATOMIC_WRITE', 'IB_UVERBS_DEVICE_AUTO_PATH_MIG',
    'IB_UVERBS_DEVICE_BAD_PKEY_CNTR',
    'IB_UVERBS_DEVICE_BAD_QKEY_CNTR',
    'IB_UVERBS_DEVICE_CHANGE_PHY_PORT',
    'IB_UVERBS_DEVICE_CURR_QP_STATE_MOD',
    'IB_UVERBS_DEVICE_FLUSH_GLOBAL',
    'IB_UVERBS_DEVICE_FLUSH_PERSISTENT',
    'IB_UVERBS_DEVICE_MANAGED_FLOW_STEERING',
    'IB_UVERBS_DEVICE_MEM_MGT_EXTENSIONS',
    'IB_UVERBS_DEVICE_MEM_WINDOW',
    'IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2A',
    'IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2B',
    'IB_UVERBS_DEVICE_N_NOTIFY_CQ',
    'IB_UVERBS_DEVICE_PCI_WRITE_END_PADDING',
    'IB_UVERBS_DEVICE_PORT_ACTIVE_EVENT',
    'IB_UVERBS_DEVICE_RAW_IP_CSUM', 'IB_UVERBS_DEVICE_RAW_MULTI',
    'IB_UVERBS_DEVICE_RAW_SCATTER_FCS', 'IB_UVERBS_DEVICE_RC_IP_CSUM',
    'IB_UVERBS_DEVICE_RC_RNR_NAK_GEN',
    'IB_UVERBS_DEVICE_RESIZE_MAX_WR',
    'IB_UVERBS_DEVICE_SHUTDOWN_PORT', 'IB_UVERBS_DEVICE_SRQ_RESIZE',
    'IB_UVERBS_DEVICE_SYS_IMAGE_GUID',
    'IB_UVERBS_DEVICE_UD_AV_PORT_ENFORCE',
    'IB_UVERBS_DEVICE_UD_IP_CSUM', 'IB_UVERBS_DEVICE_XRC',
    'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT',
    'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT',
    'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW',
    'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD',
    'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO',
    'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT',
    'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL',
    'IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM',
    'IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP',
    'IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE',
    'IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ', 'IB_UVERBS_GID_TYPE_IB',
    'IB_UVERBS_GID_TYPE_ROCE_V1', 'IB_UVERBS_GID_TYPE_ROCE_V2',
    'IB_UVERBS_PCF_AUTO_MIGR_SUP', 'IB_UVERBS_PCF_BOOT_MGMT_SUP',
    'IB_UVERBS_PCF_CAP_MASK_NOTICE_SUP',
    'IB_UVERBS_PCF_CLIENT_REG_SUP', 'IB_UVERBS_PCF_CM_SUP',
    'IB_UVERBS_PCF_DEVICE_MGMT_SUP', 'IB_UVERBS_PCF_DR_NOTICE_SUP',
    'IB_UVERBS_PCF_EXTENDED_SPEEDS_SUP',
    'IB_UVERBS_PCF_HIERARCHY_INFO_SUP', 'IB_UVERBS_PCF_IP_BASED_GIDS',
    'IB_UVERBS_PCF_LED_INFO_SUP', 'IB_UVERBS_PCF_LINK_LATENCY_SUP',
    'IB_UVERBS_PCF_LINK_SPEED_WIDTH_TABLE_SUP',
    'IB_UVERBS_PCF_MCAST_FDB_TOP_SUP',
    'IB_UVERBS_PCF_MCAST_PKEY_TRAP_SUPPRESSION_SUP',
    'IB_UVERBS_PCF_MKEY_NVRAM', 'IB_UVERBS_PCF_NOTICE_SUP',
    'IB_UVERBS_PCF_OPT_IPD_SUP', 'IB_UVERBS_PCF_PKEY_NVRAM',
    'IB_UVERBS_PCF_PKEY_SW_EXT_PORT_TRAP_SUP',
    'IB_UVERBS_PCF_REINIT_SUP', 'IB_UVERBS_PCF_SL_MAP_SUP',
    'IB_UVERBS_PCF_SM', 'IB_UVERBS_PCF_SM_DISABLED',
    'IB_UVERBS_PCF_SNMP_TUNNEL_SUP',
    'IB_UVERBS_PCF_SYS_IMAGE_GUID_SUP', 'IB_UVERBS_PCF_TRAP_SUP',
    'IB_UVERBS_PCF_VENDOR_CLASS_SUP',
    'IB_UVERBS_PCF_VENDOR_SPECIFIC_MADS_TABLE_SUP',
    'IB_UVERBS_QPF_GRH_REQUIRED', 'IB_UVERBS_QPT_DRIVER',
    'IB_UVERBS_QPT_RAW_PACKET', 'IB_UVERBS_QPT_RC',
    'IB_UVERBS_QPT_UC', 'IB_UVERBS_QPT_UD', 'IB_UVERBS_QPT_XRC_INI',
    'IB_UVERBS_QPT_XRC_TGT',
    'IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK',
    'IB_UVERBS_QP_CREATE_CVLAN_STRIPPING',
    'IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING',
    'IB_UVERBS_QP_CREATE_SCATTER_FCS',
    'IB_UVERBS_QP_CREATE_SQ_SIG_ALL',
    'IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING',
    'IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP',
    'IB_UVERBS_RAW_PACKET_CAP_IP_CSUM',
    'IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS',
    'IB_UVERBS_READ_COUNTERS_PREFER_CACHED', 'IB_UVERBS_SRQT_BASIC',
    'IB_UVERBS_SRQT_TM', 'IB_UVERBS_SRQT_XRC',
    'IB_UVERBS_WC_ATOMIC_WRITE', 'IB_UVERBS_WC_BIND_MW',
    'IB_UVERBS_WC_COMP_SWAP', 'IB_UVERBS_WC_FETCH_ADD',
    'IB_UVERBS_WC_FLUSH', 'IB_UVERBS_WC_LOCAL_INV',
    'IB_UVERBS_WC_RDMA_READ', 'IB_UVERBS_WC_RDMA_WRITE',
    'IB_UVERBS_WC_SEND', 'IB_UVERBS_WC_TSO', 'IB_UVERBS_WQT_RQ',
    'IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING',
    'IB_UVERBS_WQ_FLAGS_DELAY_DROP',
    'IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING',
    'IB_UVERBS_WQ_FLAGS_SCATTER_FCS',
    'IB_UVERBS_WR_ATOMIC_CMP_AND_SWP',
    'IB_UVERBS_WR_ATOMIC_FETCH_AND_ADD', 'IB_UVERBS_WR_ATOMIC_WRITE',
    'IB_UVERBS_WR_BIND_MW', 'IB_UVERBS_WR_FLUSH',
    'IB_UVERBS_WR_LOCAL_INV',
    'IB_UVERBS_WR_MASKED_ATOMIC_CMP_AND_SWP',
    'IB_UVERBS_WR_MASKED_ATOMIC_FETCH_AND_ADD',
    'IB_UVERBS_WR_RDMA_READ', 'IB_UVERBS_WR_RDMA_READ_WITH_INV',
    'IB_UVERBS_WR_RDMA_WRITE', 'IB_UVERBS_WR_RDMA_WRITE_WITH_IMM',
    'IB_UVERBS_WR_SEND', 'IB_UVERBS_WR_SEND_WITH_IMM',
    'IB_UVERBS_WR_SEND_WITH_INV', 'IB_UVERBS_WR_TSO',
    'INFINIBAND_VERBS_H', 'RDMA_DRIVER_BNXT_RE', 'RDMA_DRIVER_CXGB3',
    'RDMA_DRIVER_CXGB4', 'RDMA_DRIVER_EFA', 'RDMA_DRIVER_ERDMA',
    'RDMA_DRIVER_HFI1', 'RDMA_DRIVER_HNS', 'RDMA_DRIVER_I40IW',
    'RDMA_DRIVER_IRDMA', 'RDMA_DRIVER_MANA', 'RDMA_DRIVER_MLX4',
    'RDMA_DRIVER_MLX5', 'RDMA_DRIVER_MTHCA', 'RDMA_DRIVER_NES',
    'RDMA_DRIVER_OCRDMA', 'RDMA_DRIVER_QEDR', 'RDMA_DRIVER_QIB',
    'RDMA_DRIVER_RXE', 'RDMA_DRIVER_SIW', 'RDMA_DRIVER_UNKNOWN',
    'RDMA_DRIVER_USNIC', 'RDMA_DRIVER_VMW_PVRDMA', 'VERBS_API_H',
    '___ibv_query_port', '__attribute_const', '__be16', '__be32',
    '__be64', '__ibv_reg_mr', '__ibv_reg_mr_iova',
    '_ibv_query_gid_ex', '_ibv_query_gid_table',
    'c__Ea_IBV_CREATE_CQ_SUP_WC_FLAGS',
    'c__Ea_IBV_LINK_LAYER_UNSPECIFIED', 'c__Ea_IBV_SYSFS_NAME_MAX',
    'c__Ea_IBV_WC_IP_CSUM_OK_SHIFT', 'c__Ea_IBV_WC_STANDARD_FLAGS',
    'c__Ea_IB_USER_VERBS_EX_CMD_QUERY_DEVICE',
    'c__Ea_IB_UVERBS_CREATE_QP_SUP_COMP_MASK', 'ib_placement_type',
    'ib_selectivity_level', 'ib_uverbs_access_flags',
    'ib_uverbs_advise_mr_advice', 'ib_uverbs_advise_mr_flag',
    'ib_uverbs_core_support', 'ib_uverbs_create_qp_mask',
    'ib_uverbs_device_cap_flags', 'ib_uverbs_ex_create_cq_flags',
    'ib_uverbs_flow_action_esp_flags',
    'ib_uverbs_flow_action_esp_keymat',
    'ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo',
    'ib_uverbs_flow_action_esp_replay', 'ib_uverbs_gid_type',
    'ib_uverbs_qp_create_flags', 'ib_uverbs_qp_type',
    'ib_uverbs_query_port_cap_flags', 'ib_uverbs_query_port_flags',
    'ib_uverbs_raw_packet_caps', 'ib_uverbs_read_counters_flags',
    'ib_uverbs_srq_type', 'ib_uverbs_wc_opcode', 'ib_uverbs_wq_flags',
    'ib_uverbs_wq_type', 'ib_uverbs_wr_opcode',
    'ib_uverbs_write_cmds', 'ibv_access_flags', 'ibv_ack_async_event',
    'ibv_ack_cq_events', 'ibv_advise_mr', 'ibv_advise_mr_advice',
    'ibv_alloc_dm', 'ibv_alloc_mw', 'ibv_alloc_null_mr',
    'ibv_alloc_parent_domain', 'ibv_alloc_pd', 'ibv_alloc_td',
    'ibv_atomic_cap', 'ibv_attach_counters_point_flow',
    'ibv_attach_mcast', 'ibv_bind_mw', 'ibv_close_device',
    'ibv_close_xrcd', 'ibv_counter_description', 'ibv_cq_attr_mask',
    'ibv_cq_ex_to_cq', 'ibv_cq_init_attr_mask', 'ibv_create_ah',
    'ibv_create_ah_from_wc', 'ibv_create_comp_channel',
    'ibv_create_counters', 'ibv_create_cq',
    'ibv_create_cq_attr_flags', 'ibv_create_cq_ex',
    'ibv_create_cq_wc_flags', 'ibv_create_flow',
    'ibv_create_flow_action_esp', 'ibv_create_qp', 'ibv_create_qp_ex',
    'ibv_create_rwq_ind_table', 'ibv_create_srq', 'ibv_create_srq_ex',
    'ibv_create_wq', 'ibv_dealloc_mw', 'ibv_dealloc_pd',
    'ibv_dealloc_td', 'ibv_dereg_mr', 'ibv_destroy_ah',
    'ibv_destroy_comp_channel', 'ibv_destroy_counters',
    'ibv_destroy_cq', 'ibv_destroy_flow', 'ibv_destroy_flow_action',
    'ibv_destroy_qp', 'ibv_destroy_rwq_ind_table', 'ibv_destroy_srq',
    'ibv_destroy_wq', 'ibv_detach_mcast', 'ibv_device_cap_flags',
    'ibv_dm_mask', 'ibv_end_poll', 'ibv_event_type',
    'ibv_event_type_str', 'ibv_flow_action_esp_flags',
    'ibv_flow_action_esp_keymat',
    'ibv_flow_action_esp_keymat_aes_gcm_iv_algo',
    'ibv_flow_action_esp_mask', 'ibv_flow_action_esp_replay',
    'ibv_flow_attr_type', 'ibv_flow_flags',
    'ibv_flow_label_to_udp_sport', 'ibv_flow_spec_type',
    'ibv_fork_init', 'ibv_fork_status', 'ibv_free_device_list',
    'ibv_free_dm', 'ibv_get_async_event', 'ibv_get_cq_event',
    'ibv_get_device_guid', 'ibv_get_device_index',
    'ibv_get_device_list', 'ibv_get_device_name',
    'ibv_get_pkey_index', 'ibv_get_srq_num', 'ibv_gid_type',
    'ibv_import_device', 'ibv_import_dm', 'ibv_import_mr',
    'ibv_import_pd', 'ibv_inc_rkey', 'ibv_ind_table_init_attr_mask',
    'ibv_init_ah_from_wc', 'ibv_is_fork_initialized',
    'ibv_is_qpt_supported', 'ibv_memcpy_from_dm', 'ibv_memcpy_to_dm',
    'ibv_mig_state', 'ibv_modify_cq', 'ibv_modify_flow_action_esp',
    'ibv_modify_qp', 'ibv_modify_qp_rate_limit', 'ibv_modify_srq',
    'ibv_modify_wq', 'ibv_mtu', 'ibv_mw_type', 'ibv_next_poll',
    'ibv_node_type', 'ibv_node_type_str', 'ibv_odp_general_caps',
    'ibv_odp_transport_cap_bits', 'ibv_open_device', 'ibv_open_qp',
    'ibv_open_xrcd', 'ibv_ops_flags', 'ibv_ops_wr_opcode',
    'ibv_parent_domain_init_attr_mask', 'ibv_pci_atomic_op_size',
    'ibv_placement_type', 'ibv_poll_cq', 'ibv_port_cap_flags',
    'ibv_port_cap_flags2', 'ibv_port_state', 'ibv_port_state_str',
    'ibv_post_recv', 'ibv_post_send', 'ibv_post_srq_ops',
    'ibv_post_srq_recv', 'ibv_post_wq_recv', 'ibv_qp_attr_mask',
    'ibv_qp_create_flags', 'ibv_qp_create_send_ops_flags',
    'ibv_qp_init_attr_mask', 'ibv_qp_open_attr_mask', 'ibv_qp_state',
    'ibv_qp_to_qp_ex', 'ibv_qp_type', 'ibv_query_device',
    'ibv_query_device_ex', 'ibv_query_ece', 'ibv_query_gid',
    'ibv_query_gid_ex', 'ibv_query_gid_table', 'ibv_query_pkey',
    'ibv_query_qp', 'ibv_query_qp_data_in_order',
    'ibv_query_qp_data_in_order_caps',
    'ibv_query_qp_data_in_order_flags', 'ibv_query_rt_values_ex',
    'ibv_query_srq', 'ibv_rate', 'ibv_rate_to_mbps',
    'ibv_rate_to_mult', 'ibv_raw_packet_caps', 'ibv_read_counters',
    'ibv_read_counters_flags', 'ibv_reg_dm_mr', 'ibv_reg_dmabuf_mr',
    'ibv_reg_mr_iova2', 'ibv_req_notify_cq', 'ibv_rereg_mr',
    'ibv_rereg_mr_err_code', 'ibv_rereg_mr_flags', 'ibv_resize_cq',
    'ibv_resolve_eth_l2_from_gid', 'ibv_rx_hash_fields',
    'ibv_rx_hash_function_flags', 'ibv_selectivity_level',
    'ibv_send_flags', 'ibv_set_ece', 'ibv_srq_attr_mask',
    'ibv_srq_init_attr_mask', 'ibv_srq_type', 'ibv_start_poll',
    'ibv_tm_cap_flags', 'ibv_transport_type', 'ibv_unimport_dm',
    'ibv_unimport_mr', 'ibv_unimport_pd', 'ibv_values_mask',
    'ibv_wc_flags', 'ibv_wc_opcode', 'ibv_wc_read_byte_len',
    'ibv_wc_read_completion_ts',
    'ibv_wc_read_completion_wallclock_ns', 'ibv_wc_read_cvlan',
    'ibv_wc_read_dlid_path_bits', 'ibv_wc_read_flow_tag',
    'ibv_wc_read_imm_data', 'ibv_wc_read_invalidated_rkey',
    'ibv_wc_read_opcode', 'ibv_wc_read_qp_num', 'ibv_wc_read_sl',
    'ibv_wc_read_slid', 'ibv_wc_read_src_qp', 'ibv_wc_read_tm_info',
    'ibv_wc_read_vendor_err', 'ibv_wc_read_wc_flags', 'ibv_wc_status',
    'ibv_wc_status_str', 'ibv_wq_attr_mask', 'ibv_wq_flags',
    'ibv_wq_init_attr_mask', 'ibv_wq_state', 'ibv_wq_type',
    'ibv_wr_abort', 'ibv_wr_atomic_cmp_swp',
    'ibv_wr_atomic_fetch_add', 'ibv_wr_atomic_write',
    'ibv_wr_bind_mw', 'ibv_wr_complete', 'ibv_wr_flush',
    'ibv_wr_local_inv', 'ibv_wr_opcode', 'ibv_wr_opcode_str',
    'ibv_wr_rdma_read', 'ibv_wr_rdma_write', 'ibv_wr_rdma_write_imm',
    'ibv_wr_send', 'ibv_wr_send_imm', 'ibv_wr_send_inv',
    'ibv_wr_send_tso', 'ibv_wr_set_inline_data',
    'ibv_wr_set_inline_data_list', 'ibv_wr_set_sge',
    'ibv_wr_set_sge_list', 'ibv_wr_set_ud_addr',
    'ibv_wr_set_xrc_srqn', 'ibv_wr_start', 'ibv_xrcd_init_attr_mask',
    'mbps_to_ibv_rate', 'mult_to_ibv_rate', 'rdma_driver_id',
    'size_t', 'ssize_t', 'struct___pthread_cond_s',
    'struct___pthread_internal_list', 'struct___pthread_mutex_s',
    'struct__compat_ibv_port_attr', 'struct__ibv_device_ops',
    'struct_c__UA___atomic_wide_counter___value32',
    'struct_ib_uverbs_ah_attr', 'struct_ib_uverbs_alloc_mw',
    'struct_ib_uverbs_alloc_mw_resp', 'struct_ib_uverbs_alloc_pd',
    'struct_ib_uverbs_alloc_pd_resp',
    'struct_ib_uverbs_async_event_desc',
    'struct_ib_uverbs_attach_mcast', 'struct_ib_uverbs_close_xrcd',
    'struct_ib_uverbs_cmd_hdr', 'struct_ib_uverbs_comp_event_desc',
    'struct_ib_uverbs_cq_moderation',
    'struct_ib_uverbs_cq_moderation_caps',
    'struct_ib_uverbs_create_ah', 'struct_ib_uverbs_create_ah_resp',
    'struct_ib_uverbs_create_comp_channel',
    'struct_ib_uverbs_create_comp_channel_resp',
    'struct_ib_uverbs_create_cq', 'struct_ib_uverbs_create_cq_resp',
    'struct_ib_uverbs_create_flow',
    'struct_ib_uverbs_create_flow_resp', 'struct_ib_uverbs_create_qp',
    'struct_ib_uverbs_create_qp_resp', 'struct_ib_uverbs_create_srq',
    'struct_ib_uverbs_create_srq_resp',
    'struct_ib_uverbs_create_xsrq', 'struct_ib_uverbs_dealloc_mw',
    'struct_ib_uverbs_dealloc_pd', 'struct_ib_uverbs_dereg_mr',
    'struct_ib_uverbs_destroy_ah', 'struct_ib_uverbs_destroy_cq',
    'struct_ib_uverbs_destroy_cq_resp',
    'struct_ib_uverbs_destroy_flow', 'struct_ib_uverbs_destroy_qp',
    'struct_ib_uverbs_destroy_qp_resp',
    'struct_ib_uverbs_destroy_srq',
    'struct_ib_uverbs_destroy_srq_resp',
    'struct_ib_uverbs_detach_mcast', 'struct_ib_uverbs_ex_cmd_hdr',
    'struct_ib_uverbs_ex_create_cq',
    'struct_ib_uverbs_ex_create_cq_resp',
    'struct_ib_uverbs_ex_create_qp',
    'struct_ib_uverbs_ex_create_qp_resp',
    'struct_ib_uverbs_ex_create_rwq_ind_table',
    'struct_ib_uverbs_ex_create_rwq_ind_table_resp',
    'struct_ib_uverbs_ex_create_wq',
    'struct_ib_uverbs_ex_create_wq_resp',
    'struct_ib_uverbs_ex_destroy_rwq_ind_table',
    'struct_ib_uverbs_ex_destroy_wq',
    'struct_ib_uverbs_ex_destroy_wq_resp',
    'struct_ib_uverbs_ex_modify_cq', 'struct_ib_uverbs_ex_modify_qp',
    'struct_ib_uverbs_ex_modify_qp_resp',
    'struct_ib_uverbs_ex_modify_wq',
    'struct_ib_uverbs_ex_query_device',
    'struct_ib_uverbs_ex_query_device_resp',
    'struct_ib_uverbs_flow_action_esp',
    'struct_ib_uverbs_flow_action_esp_encap',
    'struct_ib_uverbs_flow_action_esp_keymat_aes_gcm',
    'struct_ib_uverbs_flow_action_esp_replay_bmp',
    'struct_ib_uverbs_flow_attr', 'struct_ib_uverbs_flow_eth_filter',
    'struct_ib_uverbs_flow_gre_filter',
    'struct_ib_uverbs_flow_ipv4_filter',
    'struct_ib_uverbs_flow_ipv6_filter',
    'struct_ib_uverbs_flow_mpls_filter',
    'struct_ib_uverbs_flow_spec_action_count',
    'struct_ib_uverbs_flow_spec_action_count_0_0',
    'struct_ib_uverbs_flow_spec_action_drop',
    'struct_ib_uverbs_flow_spec_action_drop_0_0',
    'struct_ib_uverbs_flow_spec_action_handle',
    'struct_ib_uverbs_flow_spec_action_handle_0_0',
    'struct_ib_uverbs_flow_spec_action_tag',
    'struct_ib_uverbs_flow_spec_action_tag_0_0',
    'struct_ib_uverbs_flow_spec_esp',
    'struct_ib_uverbs_flow_spec_esp_0_0',
    'struct_ib_uverbs_flow_spec_esp_filter',
    'struct_ib_uverbs_flow_spec_eth',
    'struct_ib_uverbs_flow_spec_eth_0_0',
    'struct_ib_uverbs_flow_spec_gre',
    'struct_ib_uverbs_flow_spec_gre_0_0',
    'struct_ib_uverbs_flow_spec_hdr',
    'struct_ib_uverbs_flow_spec_ipv4',
    'struct_ib_uverbs_flow_spec_ipv4_0_0',
    'struct_ib_uverbs_flow_spec_ipv6',
    'struct_ib_uverbs_flow_spec_ipv6_0_0',
    'struct_ib_uverbs_flow_spec_mpls',
    'struct_ib_uverbs_flow_spec_mpls_0_0',
    'struct_ib_uverbs_flow_spec_tcp_udp',
    'struct_ib_uverbs_flow_spec_tcp_udp_0_0',
    'struct_ib_uverbs_flow_spec_tunnel',
    'struct_ib_uverbs_flow_spec_tunnel_0_0',
    'struct_ib_uverbs_flow_tcp_udp_filter',
    'struct_ib_uverbs_flow_tunnel_filter',
    'struct_ib_uverbs_get_context',
    'struct_ib_uverbs_get_context_resp', 'struct_ib_uverbs_gid_entry',
    'struct_ib_uverbs_global_route', 'struct_ib_uverbs_modify_qp',
    'struct_ib_uverbs_modify_srq', 'struct_ib_uverbs_odp_caps',
    'struct_ib_uverbs_odp_caps_per_transport_caps',
    'struct_ib_uverbs_open_qp', 'struct_ib_uverbs_open_xrcd',
    'struct_ib_uverbs_open_xrcd_resp', 'struct_ib_uverbs_poll_cq',
    'struct_ib_uverbs_poll_cq_resp', 'struct_ib_uverbs_post_recv',
    'struct_ib_uverbs_post_recv_resp', 'struct_ib_uverbs_post_send',
    'struct_ib_uverbs_post_send_resp',
    'struct_ib_uverbs_post_srq_recv',
    'struct_ib_uverbs_post_srq_recv_resp', 'struct_ib_uverbs_qp_attr',
    'struct_ib_uverbs_qp_cap', 'struct_ib_uverbs_qp_dest',
    'struct_ib_uverbs_query_device',
    'struct_ib_uverbs_query_device_resp',
    'struct_ib_uverbs_query_port', 'struct_ib_uverbs_query_port_resp',
    'struct_ib_uverbs_query_port_resp_ex',
    'struct_ib_uverbs_query_qp', 'struct_ib_uverbs_query_qp_resp',
    'struct_ib_uverbs_query_srq', 'struct_ib_uverbs_query_srq_resp',
    'struct_ib_uverbs_recv_wr', 'struct_ib_uverbs_reg_mr',
    'struct_ib_uverbs_reg_mr_resp', 'struct_ib_uverbs_req_notify_cq',
    'struct_ib_uverbs_rereg_mr', 'struct_ib_uverbs_rereg_mr_resp',
    'struct_ib_uverbs_resize_cq', 'struct_ib_uverbs_resize_cq_resp',
    'struct_ib_uverbs_rss_caps', 'struct_ib_uverbs_send_wr',
    'struct_ib_uverbs_send_wr_1_atomic',
    'struct_ib_uverbs_send_wr_1_rdma',
    'struct_ib_uverbs_send_wr_1_ud', 'struct_ib_uverbs_sge',
    'struct_ib_uverbs_tm_caps', 'struct_ib_uverbs_wc',
    'struct_ibv_ah', 'struct_ibv_ah_attr', 'struct_ibv_alloc_dm_attr',
    'struct_ibv_async_event', 'struct_ibv_comp_channel',
    'struct_ibv_context', 'struct_ibv_context_ops',
    'struct_ibv_counter_attach_attr', 'struct_ibv_counters',
    'struct_ibv_counters_init_attr', 'struct_ibv_cq',
    'struct_ibv_cq_ex', 'struct_ibv_cq_init_attr_ex',
    'struct_ibv_cq_moderation_caps', 'struct_ibv_data_buf',
    'struct_ibv_device', 'struct_ibv_device_attr',
    'struct_ibv_device_attr_ex', 'struct_ibv_dm', 'struct_ibv_ece',
    'struct_ibv_flow', 'struct_ibv_flow_action',
    'struct_ibv_flow_action_esp_attr', 'struct_ibv_flow_attr',
    'struct_ibv_flow_esp_filter', 'struct_ibv_flow_eth_filter',
    'struct_ibv_flow_gre_filter', 'struct_ibv_flow_ipv4_ext_filter',
    'struct_ibv_flow_ipv4_filter', 'struct_ibv_flow_ipv6_filter',
    'struct_ibv_flow_mpls_filter', 'struct_ibv_flow_spec',
    'struct_ibv_flow_spec_0_hdr', 'struct_ibv_flow_spec_action_drop',
    'struct_ibv_flow_spec_action_handle',
    'struct_ibv_flow_spec_action_tag',
    'struct_ibv_flow_spec_counter_action', 'struct_ibv_flow_spec_esp',
    'struct_ibv_flow_spec_eth', 'struct_ibv_flow_spec_gre',
    'struct_ibv_flow_spec_ipv4', 'struct_ibv_flow_spec_ipv4_ext',
    'struct_ibv_flow_spec_ipv6', 'struct_ibv_flow_spec_mpls',
    'struct_ibv_flow_spec_tcp_udp', 'struct_ibv_flow_spec_tunnel',
    'struct_ibv_flow_tcp_udp_filter', 'struct_ibv_flow_tunnel_filter',
    'struct_ibv_gid_entry', 'struct_ibv_gid_global',
    'struct_ibv_global_route', 'struct_ibv_grh',
    'struct_ibv_moderate_cq', 'struct_ibv_modify_cq_attr',
    'struct_ibv_mr', 'struct_ibv_mw', 'struct_ibv_mw_bind',
    'struct_ibv_mw_bind_info', 'struct_ibv_odp_caps',
    'struct_ibv_odp_caps_per_transport_caps', 'struct_ibv_ops_wr',
    'struct_ibv_ops_wr_0_add', 'struct_ibv_ops_wr_tm',
    'struct_ibv_packet_pacing_caps',
    'struct_ibv_parent_domain_init_attr',
    'struct_ibv_pci_atomic_caps', 'struct_ibv_pd',
    'struct_ibv_poll_cq_attr', 'struct_ibv_port_attr',
    'struct_ibv_qp', 'struct_ibv_qp_attr', 'struct_ibv_qp_cap',
    'struct_ibv_qp_ex', 'struct_ibv_qp_init_attr',
    'struct_ibv_qp_init_attr_ex', 'struct_ibv_qp_open_attr',
    'struct_ibv_qp_rate_limit_attr',
    'struct_ibv_query_device_ex_input', 'struct_ibv_recv_wr',
    'struct_ibv_rss_caps', 'struct_ibv_rwq_ind_table',
    'struct_ibv_rwq_ind_table_init_attr', 'struct_ibv_rx_hash_conf',
    'struct_ibv_send_wr', 'struct_ibv_send_wr_1_atomic',
    'struct_ibv_send_wr_1_rdma', 'struct_ibv_send_wr_1_ud',
    'struct_ibv_send_wr_2_xrc', 'struct_ibv_send_wr_3_bind_mw',
    'struct_ibv_send_wr_3_tso', 'struct_ibv_sge', 'struct_ibv_srq',
    'struct_ibv_srq_attr', 'struct_ibv_srq_init_attr',
    'struct_ibv_srq_init_attr_ex', 'struct_ibv_td',
    'struct_ibv_td_init_attr', 'struct_ibv_tm_cap',
    'struct_ibv_tm_caps', 'struct_ibv_tso_caps',
    'struct_ibv_values_ex', 'struct_ibv_wc', 'struct_ibv_wc_tm_info',
    'struct_ibv_wq', 'struct_ibv_wq_attr', 'struct_ibv_wq_init_attr',
    'struct_ibv_xrcd', 'struct_ibv_xrcd_init_attr', 'struct_timespec',
    'struct_verbs_context', 'struct_verbs_ex_private', 'uint16_t',
    'uint32_t', 'uint64_t', 'uint8_t',
    'union_c__UA___atomic_wide_counter', 'union_c__UA_pthread_cond_t',
    'union_c__UA_pthread_mutex_t',
    'union_ib_uverbs_flow_action_esp_encap_0',
    'union_ib_uverbs_flow_action_esp_encap_1',
    'union_ib_uverbs_flow_spec_action_count_0',
    'union_ib_uverbs_flow_spec_action_drop_0',
    'union_ib_uverbs_flow_spec_action_handle_0',
    'union_ib_uverbs_flow_spec_action_tag_0',
    'union_ib_uverbs_flow_spec_esp_0',
    'union_ib_uverbs_flow_spec_eth_0',
    'union_ib_uverbs_flow_spec_gre_0',
    'union_ib_uverbs_flow_spec_ipv4_0',
    'union_ib_uverbs_flow_spec_ipv6_0',
    'union_ib_uverbs_flow_spec_mpls_0',
    'union_ib_uverbs_flow_spec_tcp_udp_0',
    'union_ib_uverbs_flow_spec_tunnel_0',
    'union_ib_uverbs_send_wr_ex', 'union_ib_uverbs_send_wr_wr',
    'union_ib_uverbs_wc_ex', 'union_ibv_async_event_element',
    'union_ibv_flow_spec_0', 'union_ibv_gid', 'union_ibv_send_wr_0',
    'union_ibv_send_wr_3', 'union_ibv_send_wr_qp_type',
    'union_ibv_send_wr_wr', 'union_ibv_wc_0', 'verbs_get_ctx']
