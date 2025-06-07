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

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')
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





LIB_URING_H = True # macro
_XOPEN_SOURCE = 500 # macro
# def uring_unlikely(cond):  # macro
#    return __builtin_expect(!!(cond),0)
# def uring_likely(cond):  # macro
#    return __builtin_expect(!!(cond),1)
# def io_uring_for_each_cqe(ring, head, cqe):  # macro
#    return (head=*(ring)->cq.khead;(cqe=(head!=io_uring_smp_load_acquire((ring)->cq.ktail)?&(ring)->cq.cqes[head&(*(ring)->cq.kring_mask)]:NULL));head++)
class struct_io_uring_sq(Structure):
    pass

class struct_io_uring_sqe(Structure):
    pass

struct_io_uring_sq._pack_ = 1 # source:False
struct_io_uring_sq._fields_ = [
    ('khead', ctypes.POINTER(ctypes.c_uint32)),
    ('ktail', ctypes.POINTER(ctypes.c_uint32)),
    ('kring_mask', ctypes.POINTER(ctypes.c_uint32)),
    ('kring_entries', ctypes.POINTER(ctypes.c_uint32)),
    ('kflags', ctypes.POINTER(ctypes.c_uint32)),
    ('kdropped', ctypes.POINTER(ctypes.c_uint32)),
    ('array', ctypes.POINTER(ctypes.c_uint32)),
    ('sqes', ctypes.POINTER(struct_io_uring_sqe)),
    ('sqe_head', ctypes.c_uint32),
    ('sqe_tail', ctypes.c_uint32),
    ('ring_sz', ctypes.c_uint64),
    ('ring_ptr', ctypes.POINTER(None)),
    ('pad', ctypes.c_uint32 * 4),
]

class union_io_uring_sqe_0(Union):
    pass

union_io_uring_sqe_0._pack_ = 1 # source:False
union_io_uring_sqe_0._fields_ = [
    ('off', ctypes.c_uint64),
    ('addr2', ctypes.c_uint64),
]

class union_io_uring_sqe_1(Union):
    pass

union_io_uring_sqe_1._pack_ = 1 # source:False
union_io_uring_sqe_1._fields_ = [
    ('addr', ctypes.c_uint64),
    ('splice_off_in', ctypes.c_uint64),
]

class union_io_uring_sqe_2(Union):
    pass

union_io_uring_sqe_2._pack_ = 1 # source:False
union_io_uring_sqe_2._fields_ = [
    ('rw_flags', ctypes.c_int32),
    ('fsync_flags', ctypes.c_uint32),
    ('poll_events', ctypes.c_uint16),
    ('poll32_events', ctypes.c_uint32),
    ('sync_range_flags', ctypes.c_uint32),
    ('msg_flags', ctypes.c_uint32),
    ('timeout_flags', ctypes.c_uint32),
    ('accept_flags', ctypes.c_uint32),
    ('cancel_flags', ctypes.c_uint32),
    ('open_flags', ctypes.c_uint32),
    ('statx_flags', ctypes.c_uint32),
    ('fadvise_advice', ctypes.c_uint32),
    ('splice_flags', ctypes.c_uint32),
    ('rename_flags', ctypes.c_uint32),
    ('unlink_flags', ctypes.c_uint32),
    ('hardlink_flags', ctypes.c_uint32),
]

class union_io_uring_sqe_3(Union):
    pass

union_io_uring_sqe_3._pack_ = 1 # source:True
union_io_uring_sqe_3._fields_ = [
    ('buf_index', ctypes.c_uint16),
    ('buf_group', ctypes.c_uint16),
]

class union_io_uring_sqe_4(Union):
    pass

union_io_uring_sqe_4._pack_ = 1 # source:False
union_io_uring_sqe_4._fields_ = [
    ('splice_fd_in', ctypes.c_int32),
    ('file_index', ctypes.c_uint32),
]

struct_io_uring_sqe._pack_ = 1 # source:False
struct_io_uring_sqe._anonymous_ = ('_0', '_1', '_2', '_3', '_4',)
struct_io_uring_sqe._fields_ = [
    ('opcode', ctypes.c_ubyte),
    ('flags', ctypes.c_ubyte),
    ('ioprio', ctypes.c_uint16),
    ('fd', ctypes.c_int32),
    ('_0', union_io_uring_sqe_0),
    ('_1', union_io_uring_sqe_1),
    ('len', ctypes.c_uint32),
    ('_2', union_io_uring_sqe_2),
    ('user_data', ctypes.c_uint64),
    ('_3', union_io_uring_sqe_3),
    ('personality', ctypes.c_uint16),
    ('_4', union_io_uring_sqe_4),
    ('__pad2', ctypes.c_uint64 * 2),
]

class struct_io_uring_cq(Structure):
    pass

class struct_io_uring_cqe(Structure):
    pass

struct_io_uring_cq._pack_ = 1 # source:False
struct_io_uring_cq._fields_ = [
    ('khead', ctypes.POINTER(ctypes.c_uint32)),
    ('ktail', ctypes.POINTER(ctypes.c_uint32)),
    ('kring_mask', ctypes.POINTER(ctypes.c_uint32)),
    ('kring_entries', ctypes.POINTER(ctypes.c_uint32)),
    ('kflags', ctypes.POINTER(ctypes.c_uint32)),
    ('koverflow', ctypes.POINTER(ctypes.c_uint32)),
    ('cqes', ctypes.POINTER(struct_io_uring_cqe)),
    ('ring_sz', ctypes.c_uint64),
    ('ring_ptr', ctypes.POINTER(None)),
    ('pad', ctypes.c_uint32 * 4),
]

struct_io_uring_cqe._pack_ = 1 # source:False
struct_io_uring_cqe._fields_ = [
    ('user_data', ctypes.c_uint64),
    ('res', ctypes.c_int32),
    ('flags', ctypes.c_uint32),
]

class struct_io_uring(Structure):
    pass

struct_io_uring._pack_ = 1 # source:False
struct_io_uring._fields_ = [
    ('sq', struct_io_uring_sq),
    ('cq', struct_io_uring_cq),
    ('flags', ctypes.c_uint32),
    ('ring_fd', ctypes.c_int32),
    ('features', ctypes.c_uint32),
    ('pad', ctypes.c_uint32 * 3),
]

class struct_io_uring_probe(Structure):
    pass

class struct_io_uring_probe_op(Structure):
    pass

struct_io_uring_probe_op._pack_ = 1 # source:False
struct_io_uring_probe_op._fields_ = [
    ('op', ctypes.c_ubyte),
    ('resv', ctypes.c_ubyte),
    ('flags', ctypes.c_uint16),
    ('resv2', ctypes.c_uint32),
]

struct_io_uring_probe._pack_ = 1 # source:False
struct_io_uring_probe._fields_ = [
    ('last_op', ctypes.c_ubyte),
    ('ops_len', ctypes.c_ubyte),
    ('resv', ctypes.c_uint16),
    ('resv2', ctypes.c_uint32 * 3),
    ('ops', struct_io_uring_probe_op * 0),
]

try:
    io_uring_get_probe_ring = _libraries['FIXME_STUB'].io_uring_get_probe_ring
    io_uring_get_probe_ring.restype = ctypes.POINTER(struct_io_uring_probe)
    io_uring_get_probe_ring.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_get_probe = _libraries['FIXME_STUB'].io_uring_get_probe
    io_uring_get_probe.restype = ctypes.POINTER(struct_io_uring_probe)
    io_uring_get_probe.argtypes = []
except AttributeError:
    pass
try:
    io_uring_free_probe = _libraries['FIXME_STUB'].io_uring_free_probe
    io_uring_free_probe.restype = None
    io_uring_free_probe.argtypes = [ctypes.POINTER(struct_io_uring_probe)]
except AttributeError:
    pass
try:
    io_uring_opcode_supported = _libraries['FIXME_STUB'].io_uring_opcode_supported
    io_uring_opcode_supported.restype = ctypes.c_int32
    io_uring_opcode_supported.argtypes = [ctypes.POINTER(struct_io_uring_probe), ctypes.c_int32]
except AttributeError:
    pass
class struct_io_uring_params(Structure):
    pass

class struct_io_sqring_offsets(Structure):
    pass

struct_io_sqring_offsets._pack_ = 1 # source:False
struct_io_sqring_offsets._fields_ = [
    ('head', ctypes.c_uint32),
    ('tail', ctypes.c_uint32),
    ('ring_mask', ctypes.c_uint32),
    ('ring_entries', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('dropped', ctypes.c_uint32),
    ('array', ctypes.c_uint32),
    ('resv1', ctypes.c_uint32),
    ('resv2', ctypes.c_uint64),
]

class struct_io_cqring_offsets(Structure):
    pass

struct_io_cqring_offsets._pack_ = 1 # source:False
struct_io_cqring_offsets._fields_ = [
    ('head', ctypes.c_uint32),
    ('tail', ctypes.c_uint32),
    ('ring_mask', ctypes.c_uint32),
    ('ring_entries', ctypes.c_uint32),
    ('overflow', ctypes.c_uint32),
    ('cqes', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('resv1', ctypes.c_uint32),
    ('resv2', ctypes.c_uint64),
]

struct_io_uring_params._pack_ = 1 # source:False
struct_io_uring_params._fields_ = [
    ('sq_entries', ctypes.c_uint32),
    ('cq_entries', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('sq_thread_cpu', ctypes.c_uint32),
    ('sq_thread_idle', ctypes.c_uint32),
    ('features', ctypes.c_uint32),
    ('wq_fd', ctypes.c_uint32),
    ('resv', ctypes.c_uint32 * 3),
    ('sq_off', struct_io_sqring_offsets),
    ('cq_off', struct_io_cqring_offsets),
]

try:
    io_uring_queue_init_params = _libraries['FIXME_STUB'].io_uring_queue_init_params
    io_uring_queue_init_params.restype = ctypes.c_int32
    io_uring_queue_init_params.argtypes = [ctypes.c_uint32, ctypes.POINTER(struct_io_uring), ctypes.POINTER(struct_io_uring_params)]
except AttributeError:
    pass
try:
    io_uring_queue_init = _libraries['FIXME_STUB'].io_uring_queue_init
    io_uring_queue_init.restype = ctypes.c_int32
    io_uring_queue_init.argtypes = [ctypes.c_uint32, ctypes.POINTER(struct_io_uring), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_queue_mmap = _libraries['FIXME_STUB'].io_uring_queue_mmap
    io_uring_queue_mmap.restype = ctypes.c_int32
    io_uring_queue_mmap.argtypes = [ctypes.c_int32, ctypes.POINTER(struct_io_uring_params), ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_ring_dontfork = _libraries['FIXME_STUB'].io_uring_ring_dontfork
    io_uring_ring_dontfork.restype = ctypes.c_int32
    io_uring_ring_dontfork.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_queue_exit = _libraries['FIXME_STUB'].io_uring_queue_exit
    io_uring_queue_exit.restype = None
    io_uring_queue_exit.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_peek_batch_cqe = _libraries['FIXME_STUB'].io_uring_peek_batch_cqe
    io_uring_peek_batch_cqe.restype = ctypes.c_uint32
    io_uring_peek_batch_cqe.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.POINTER(struct_io_uring_cqe)), ctypes.c_uint32]
except AttributeError:
    pass
class struct___kernel_timespec(Structure):
    pass

struct___kernel_timespec._pack_ = 1 # source:False
struct___kernel_timespec._fields_ = [
    ('tv_sec', ctypes.c_int64),
    ('tv_nsec', ctypes.c_int64),
]

class struct_c__SA___sigset_t(Structure):
    pass

struct_c__SA___sigset_t._pack_ = 1 # source:False
struct_c__SA___sigset_t._fields_ = [
    ('__val', ctypes.c_uint64 * 16),
]

try:
    io_uring_wait_cqes = _libraries['FIXME_STUB'].io_uring_wait_cqes
    io_uring_wait_cqes.restype = ctypes.c_int32
    io_uring_wait_cqes.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.POINTER(struct_io_uring_cqe)), ctypes.c_uint32, ctypes.POINTER(struct___kernel_timespec), ctypes.POINTER(struct_c__SA___sigset_t)]
except AttributeError:
    pass
try:
    io_uring_wait_cqe_timeout = _libraries['FIXME_STUB'].io_uring_wait_cqe_timeout
    io_uring_wait_cqe_timeout.restype = ctypes.c_int32
    io_uring_wait_cqe_timeout.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.POINTER(struct_io_uring_cqe)), ctypes.POINTER(struct___kernel_timespec)]
except AttributeError:
    pass
try:
    io_uring_submit = _libraries['FIXME_STUB'].io_uring_submit
    io_uring_submit.restype = ctypes.c_int32
    io_uring_submit.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_submit_and_wait = _libraries['FIXME_STUB'].io_uring_submit_and_wait
    io_uring_submit_and_wait.restype = ctypes.c_int32
    io_uring_submit_and_wait.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_get_sqe = _libraries['FIXME_STUB'].io_uring_get_sqe
    io_uring_get_sqe.restype = ctypes.POINTER(struct_io_uring_sqe)
    io_uring_get_sqe.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
class struct_iovec(Structure):
    pass

struct_iovec._pack_ = 1 # source:False
struct_iovec._fields_ = [
    ('iov_base', ctypes.POINTER(None)),
    ('iov_len', ctypes.c_uint64),
]

try:
    io_uring_register_buffers = _libraries['FIXME_STUB'].io_uring_register_buffers
    io_uring_register_buffers.restype = ctypes.c_int32
    io_uring_register_buffers.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(struct_iovec), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_register_buffers_tags = _libraries['FIXME_STUB'].io_uring_register_buffers_tags
    io_uring_register_buffers_tags.restype = ctypes.c_int32
    io_uring_register_buffers_tags.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(struct_iovec), ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_register_buffers_update_tag = _libraries['FIXME_STUB'].io_uring_register_buffers_update_tag
    io_uring_register_buffers_update_tag.restype = ctypes.c_int32
    io_uring_register_buffers_update_tag.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.c_uint32, ctypes.POINTER(struct_iovec), ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_unregister_buffers = _libraries['FIXME_STUB'].io_uring_unregister_buffers
    io_uring_unregister_buffers.restype = ctypes.c_int32
    io_uring_unregister_buffers.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_register_files = _libraries['FIXME_STUB'].io_uring_register_files
    io_uring_register_files.restype = ctypes.c_int32
    io_uring_register_files.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_register_files_tags = _libraries['FIXME_STUB'].io_uring_register_files_tags
    io_uring_register_files_tags.restype = ctypes.c_int32
    io_uring_register_files_tags.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_register_files_update_tag = _libraries['FIXME_STUB'].io_uring_register_files_update_tag
    io_uring_register_files_update_tag.restype = ctypes.c_int32
    io_uring_register_files_update_tag.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_unregister_files = _libraries['FIXME_STUB'].io_uring_unregister_files
    io_uring_unregister_files.restype = ctypes.c_int32
    io_uring_unregister_files.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_register_files_update = _libraries['FIXME_STUB'].io_uring_register_files_update
    io_uring_register_files_update.restype = ctypes.c_int32
    io_uring_register_files_update.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_register_eventfd = _libraries['FIXME_STUB'].io_uring_register_eventfd
    io_uring_register_eventfd.restype = ctypes.c_int32
    io_uring_register_eventfd.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_register_eventfd_async = _libraries['FIXME_STUB'].io_uring_register_eventfd_async
    io_uring_register_eventfd_async.restype = ctypes.c_int32
    io_uring_register_eventfd_async.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_unregister_eventfd = _libraries['FIXME_STUB'].io_uring_unregister_eventfd
    io_uring_unregister_eventfd.restype = ctypes.c_int32
    io_uring_unregister_eventfd.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_register_probe = _libraries['FIXME_STUB'].io_uring_register_probe
    io_uring_register_probe.restype = ctypes.c_int32
    io_uring_register_probe.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(struct_io_uring_probe), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_register_personality = _libraries['FIXME_STUB'].io_uring_register_personality
    io_uring_register_personality.restype = ctypes.c_int32
    io_uring_register_personality.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_unregister_personality = _libraries['FIXME_STUB'].io_uring_unregister_personality
    io_uring_unregister_personality.restype = ctypes.c_int32
    io_uring_unregister_personality.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.c_int32]
except AttributeError:
    pass
class struct_io_uring_restriction(Structure):
    pass

class union_io_uring_restriction_0(Union):
    pass

union_io_uring_restriction_0._pack_ = 1 # source:False
union_io_uring_restriction_0._fields_ = [
    ('register_op', ctypes.c_ubyte),
    ('sqe_op', ctypes.c_ubyte),
    ('sqe_flags', ctypes.c_ubyte),
]

struct_io_uring_restriction._pack_ = 1 # source:False
struct_io_uring_restriction._anonymous_ = ('_0',)
struct_io_uring_restriction._fields_ = [
    ('opcode', ctypes.c_uint16),
    ('_0', union_io_uring_restriction_0),
    ('resv', ctypes.c_ubyte),
    ('resv2', ctypes.c_uint32 * 3),
]

try:
    io_uring_register_restrictions = _libraries['FIXME_STUB'].io_uring_register_restrictions
    io_uring_register_restrictions.restype = ctypes.c_int32
    io_uring_register_restrictions.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(struct_io_uring_restriction), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_enable_rings = _libraries['FIXME_STUB'].io_uring_enable_rings
    io_uring_enable_rings.restype = ctypes.c_int32
    io_uring_enable_rings.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    __io_uring_sqring_wait = _libraries['FIXME_STUB'].__io_uring_sqring_wait
    __io_uring_sqring_wait.restype = ctypes.c_int32
    __io_uring_sqring_wait.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
class struct_c__SA_cpu_set_t(Structure):
    pass

struct_c__SA_cpu_set_t._pack_ = 1 # source:False
struct_c__SA_cpu_set_t._fields_ = [
    ('__bits', ctypes.c_uint64 * 16),
]

try:
    io_uring_register_iowq_aff = _libraries['FIXME_STUB'].io_uring_register_iowq_aff
    io_uring_register_iowq_aff.restype = ctypes.c_int32
    io_uring_register_iowq_aff.argtypes = [ctypes.POINTER(struct_io_uring), size_t, ctypes.POINTER(struct_c__SA_cpu_set_t)]
except AttributeError:
    pass
try:
    io_uring_unregister_iowq_aff = _libraries['FIXME_STUB'].io_uring_unregister_iowq_aff
    io_uring_unregister_iowq_aff.restype = ctypes.c_int32
    io_uring_unregister_iowq_aff.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_register_iowq_max_workers = _libraries['FIXME_STUB'].io_uring_register_iowq_max_workers
    io_uring_register_iowq_max_workers.restype = ctypes.c_int32
    io_uring_register_iowq_max_workers.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    __io_uring_get_cqe = _libraries['FIXME_STUB'].__io_uring_get_cqe
    __io_uring_get_cqe.restype = ctypes.c_int32
    __io_uring_get_cqe.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.POINTER(struct_io_uring_cqe)), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(struct_c__SA___sigset_t)]
except AttributeError:
    pass
try:
    io_uring_cq_advance = _libraries['FIXME_STUB'].io_uring_cq_advance
    io_uring_cq_advance.restype = None
    io_uring_cq_advance.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_cqe_seen = _libraries['FIXME_STUB'].io_uring_cqe_seen
    io_uring_cqe_seen.restype = None
    io_uring_cqe_seen.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(struct_io_uring_cqe)]
except AttributeError:
    pass
try:
    io_uring_sqe_set_data = _libraries['FIXME_STUB'].io_uring_sqe_set_data
    io_uring_sqe_set_data.restype = None
    io_uring_sqe_set_data.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    io_uring_cqe_get_data = _libraries['FIXME_STUB'].io_uring_cqe_get_data
    io_uring_cqe_get_data.restype = ctypes.POINTER(None)
    io_uring_cqe_get_data.argtypes = [ctypes.POINTER(struct_io_uring_cqe)]
except AttributeError:
    pass
try:
    io_uring_sqe_set_flags = _libraries['FIXME_STUB'].io_uring_sqe_set_flags
    io_uring_sqe_set_flags.restype = None
    io_uring_sqe_set_flags.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_uint32]
except AttributeError:
    pass
try:
    __io_uring_set_target_fixed_file = _libraries['FIXME_STUB'].__io_uring_set_target_fixed_file
    __io_uring_set_target_fixed_file.restype = None
    __io_uring_set_target_fixed_file.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_uint32]
except AttributeError:
    pass
__u64 = ctypes.c_uint64
# LIBURING_UDATA_TIMEOUT = ((__u64)-1) # macro
try:
    io_uring_prep_rw = _libraries['FIXME_STUB'].io_uring_prep_rw
    io_uring_prep_rw.restype = None
    io_uring_prep_rw.argtypes = [ctypes.c_int32, ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(None), ctypes.c_uint32, __u64]
except AttributeError:
    pass
int64_t = ctypes.c_int64
try:
    io_uring_prep_splice = _libraries['FIXME_STUB'].io_uring_prep_splice
    io_uring_prep_splice.restype = None
    io_uring_prep_splice.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, int64_t, ctypes.c_int32, int64_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_tee = _libraries['FIXME_STUB'].io_uring_prep_tee
    io_uring_prep_tee.restype = None
    io_uring_prep_tee.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_readv = _libraries['FIXME_STUB'].io_uring_prep_readv
    io_uring_prep_readv.restype = None
    io_uring_prep_readv.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(struct_iovec), ctypes.c_uint32, __u64]
except AttributeError:
    pass
try:
    io_uring_prep_read_fixed = _libraries['FIXME_STUB'].io_uring_prep_read_fixed
    io_uring_prep_read_fixed.restype = None
    io_uring_prep_read_fixed.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(None), ctypes.c_uint32, __u64, ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_writev = _libraries['FIXME_STUB'].io_uring_prep_writev
    io_uring_prep_writev.restype = None
    io_uring_prep_writev.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(struct_iovec), ctypes.c_uint32, __u64]
except AttributeError:
    pass
try:
    io_uring_prep_write_fixed = _libraries['FIXME_STUB'].io_uring_prep_write_fixed
    io_uring_prep_write_fixed.restype = None
    io_uring_prep_write_fixed.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(None), ctypes.c_uint32, __u64, ctypes.c_int32]
except AttributeError:
    pass
class struct_msghdr(Structure):
    pass

struct_msghdr._pack_ = 1 # source:False
struct_msghdr._fields_ = [
    ('msg_name', ctypes.POINTER(None)),
    ('msg_namelen', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('msg_iov', ctypes.POINTER(struct_iovec)),
    ('msg_iovlen', ctypes.c_uint64),
    ('msg_control', ctypes.POINTER(None)),
    ('msg_controllen', ctypes.c_uint64),
    ('msg_flags', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

try:
    io_uring_prep_recvmsg = _libraries['FIXME_STUB'].io_uring_prep_recvmsg
    io_uring_prep_recvmsg.restype = None
    io_uring_prep_recvmsg.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(struct_msghdr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_sendmsg = _libraries['FIXME_STUB'].io_uring_prep_sendmsg
    io_uring_prep_sendmsg.restype = None
    io_uring_prep_sendmsg.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(struct_msghdr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    __io_uring_prep_poll_mask = _libraries['FIXME_STUB'].__io_uring_prep_poll_mask
    __io_uring_prep_poll_mask.restype = ctypes.c_uint32
    __io_uring_prep_poll_mask.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_poll_add = _libraries['FIXME_STUB'].io_uring_prep_poll_add
    io_uring_prep_poll_add.restype = None
    io_uring_prep_poll_add.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_poll_multishot = _libraries['FIXME_STUB'].io_uring_prep_poll_multishot
    io_uring_prep_poll_multishot.restype = None
    io_uring_prep_poll_multishot.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_poll_remove = _libraries['FIXME_STUB'].io_uring_prep_poll_remove
    io_uring_prep_poll_remove.restype = None
    io_uring_prep_poll_remove.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    io_uring_prep_poll_update = _libraries['FIXME_STUB'].io_uring_prep_poll_update
    io_uring_prep_poll_update.restype = None
    io_uring_prep_poll_update.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(None), ctypes.POINTER(None), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_fsync = _libraries['FIXME_STUB'].io_uring_prep_fsync
    io_uring_prep_fsync.restype = None
    io_uring_prep_fsync.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_nop = _libraries['FIXME_STUB'].io_uring_prep_nop
    io_uring_prep_nop.restype = None
    io_uring_prep_nop.argtypes = [ctypes.POINTER(struct_io_uring_sqe)]
except AttributeError:
    pass
try:
    io_uring_prep_timeout = _libraries['FIXME_STUB'].io_uring_prep_timeout
    io_uring_prep_timeout.restype = None
    io_uring_prep_timeout.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(struct___kernel_timespec), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_timeout_remove = _libraries['FIXME_STUB'].io_uring_prep_timeout_remove
    io_uring_prep_timeout_remove.restype = None
    io_uring_prep_timeout_remove.argtypes = [ctypes.POINTER(struct_io_uring_sqe), __u64, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_timeout_update = _libraries['FIXME_STUB'].io_uring_prep_timeout_update
    io_uring_prep_timeout_update.restype = None
    io_uring_prep_timeout_update.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(struct___kernel_timespec), __u64, ctypes.c_uint32]
except AttributeError:
    pass
class struct_sockaddr(Structure):
    pass

struct_sockaddr._pack_ = 1 # source:False
struct_sockaddr._fields_ = [
    ('sa_family', ctypes.c_uint16),
    ('sa_data', ctypes.c_char * 14),
]

try:
    io_uring_prep_accept = _libraries['FIXME_STUB'].io_uring_prep_accept
    io_uring_prep_accept.restype = None
    io_uring_prep_accept.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(struct_sockaddr), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_accept_direct = _libraries['FIXME_STUB'].io_uring_prep_accept_direct
    io_uring_prep_accept_direct.restype = None
    io_uring_prep_accept_direct.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(struct_sockaddr), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_cancel = _libraries['FIXME_STUB'].io_uring_prep_cancel
    io_uring_prep_cancel.restype = None
    io_uring_prep_cancel.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(None), ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_link_timeout = _libraries['FIXME_STUB'].io_uring_prep_link_timeout
    io_uring_prep_link_timeout.restype = None
    io_uring_prep_link_timeout.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(struct___kernel_timespec), ctypes.c_uint32]
except AttributeError:
    pass
socklen_t = ctypes.c_uint32
try:
    io_uring_prep_connect = _libraries['FIXME_STUB'].io_uring_prep_connect
    io_uring_prep_connect.restype = None
    io_uring_prep_connect.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(struct_sockaddr), socklen_t]
except AttributeError:
    pass
try:
    io_uring_prep_files_update = _libraries['FIXME_STUB'].io_uring_prep_files_update
    io_uring_prep_files_update.restype = None
    io_uring_prep_files_update.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_int32]
except AttributeError:
    pass
off_t = ctypes.c_int64
try:
    io_uring_prep_fallocate = _libraries['FIXME_STUB'].io_uring_prep_fallocate
    io_uring_prep_fallocate.restype = None
    io_uring_prep_fallocate.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.c_int32, off_t, off_t]
except AttributeError:
    pass
mode_t = ctypes.c_uint32
try:
    io_uring_prep_openat = _libraries['FIXME_STUB'].io_uring_prep_openat
    io_uring_prep_openat.restype = None
    io_uring_prep_openat.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, mode_t]
except AttributeError:
    pass
try:
    io_uring_prep_openat_direct = _libraries['FIXME_STUB'].io_uring_prep_openat_direct
    io_uring_prep_openat_direct.restype = None
    io_uring_prep_openat_direct.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, mode_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_prep_close = _libraries['FIXME_STUB'].io_uring_prep_close
    io_uring_prep_close.restype = None
    io_uring_prep_close.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_read = _libraries['FIXME_STUB'].io_uring_prep_read
    io_uring_prep_read.restype = None
    io_uring_prep_read.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(None), ctypes.c_uint32, __u64]
except AttributeError:
    pass
try:
    io_uring_prep_write = _libraries['FIXME_STUB'].io_uring_prep_write
    io_uring_prep_write.restype = None
    io_uring_prep_write.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(None), ctypes.c_uint32, __u64]
except AttributeError:
    pass
class struct_statx(Structure):
    pass

try:
    io_uring_prep_statx = _libraries['FIXME_STUB'].io_uring_prep_statx
    io_uring_prep_statx.restype = None
    io_uring_prep_statx.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(struct_statx)]
except AttributeError:
    pass
try:
    io_uring_prep_fadvise = _libraries['FIXME_STUB'].io_uring_prep_fadvise
    io_uring_prep_fadvise.restype = None
    io_uring_prep_fadvise.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, __u64, off_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_madvise = _libraries['FIXME_STUB'].io_uring_prep_madvise
    io_uring_prep_madvise.restype = None
    io_uring_prep_madvise.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(None), off_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_send = _libraries['FIXME_STUB'].io_uring_prep_send
    io_uring_prep_send.restype = None
    io_uring_prep_send.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_recv = _libraries['FIXME_STUB'].io_uring_prep_recv
    io_uring_prep_recv.restype = None
    io_uring_prep_recv.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
class struct_open_how(Structure):
    pass

struct_open_how._pack_ = 1 # source:False
struct_open_how._fields_ = [
    ('flags', ctypes.c_uint64),
    ('mode', ctypes.c_uint64),
    ('resolve', ctypes.c_uint64),
]

try:
    io_uring_prep_openat2 = _libraries['FIXME_STUB'].io_uring_prep_openat2
    io_uring_prep_openat2.restype = None
    io_uring_prep_openat2.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_open_how)]
except AttributeError:
    pass
try:
    io_uring_prep_openat2_direct = _libraries['FIXME_STUB'].io_uring_prep_openat2_direct
    io_uring_prep_openat2_direct.restype = None
    io_uring_prep_openat2_direct.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_open_how), ctypes.c_uint32]
except AttributeError:
    pass
class struct_epoll_event(Structure):
    pass

try:
    io_uring_prep_epoll_ctl = _libraries['FIXME_STUB'].io_uring_prep_epoll_ctl
    io_uring_prep_epoll_ctl.restype = None
    io_uring_prep_epoll_ctl.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(struct_epoll_event)]
except AttributeError:
    pass
try:
    io_uring_prep_provide_buffers = _libraries['FIXME_STUB'].io_uring_prep_provide_buffers
    io_uring_prep_provide_buffers.restype = None
    io_uring_prep_provide_buffers.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(None), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_remove_buffers = _libraries['FIXME_STUB'].io_uring_prep_remove_buffers
    io_uring_prep_remove_buffers.restype = None
    io_uring_prep_remove_buffers.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_shutdown = _libraries['FIXME_STUB'].io_uring_prep_shutdown
    io_uring_prep_shutdown.restype = None
    io_uring_prep_shutdown.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_unlinkat = _libraries['FIXME_STUB'].io_uring_prep_unlinkat
    io_uring_prep_unlinkat.restype = None
    io_uring_prep_unlinkat.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_renameat = _libraries['FIXME_STUB'].io_uring_prep_renameat
    io_uring_prep_renameat.restype = None
    io_uring_prep_renameat.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_sync_file_range = _libraries['FIXME_STUB'].io_uring_prep_sync_file_range
    io_uring_prep_sync_file_range.restype = None
    io_uring_prep_sync_file_range.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.c_uint32, __u64, ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_prep_mkdirat = _libraries['FIXME_STUB'].io_uring_prep_mkdirat
    io_uring_prep_mkdirat.restype = None
    io_uring_prep_mkdirat.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), mode_t]
except AttributeError:
    pass
try:
    io_uring_prep_symlinkat = _libraries['FIXME_STUB'].io_uring_prep_symlinkat
    io_uring_prep_symlinkat.restype = None
    io_uring_prep_symlinkat.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    io_uring_prep_linkat = _libraries['FIXME_STUB'].io_uring_prep_linkat
    io_uring_prep_linkat.restype = None
    io_uring_prep_linkat.argtypes = [ctypes.POINTER(struct_io_uring_sqe), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    io_uring_sq_ready = _libraries['FIXME_STUB'].io_uring_sq_ready
    io_uring_sq_ready.restype = ctypes.c_uint32
    io_uring_sq_ready.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_sq_space_left = _libraries['FIXME_STUB'].io_uring_sq_space_left
    io_uring_sq_space_left.restype = ctypes.c_uint32
    io_uring_sq_space_left.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_sqring_wait = _libraries['FIXME_STUB'].io_uring_sqring_wait
    io_uring_sqring_wait.restype = ctypes.c_int32
    io_uring_sqring_wait.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_cq_ready = _libraries['FIXME_STUB'].io_uring_cq_ready
    io_uring_cq_ready.restype = ctypes.c_uint32
    io_uring_cq_ready.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_cq_eventfd_enabled = _libraries['FIXME_STUB'].io_uring_cq_eventfd_enabled
    io_uring_cq_eventfd_enabled.restype = ctypes.c_bool
    io_uring_cq_eventfd_enabled.argtypes = [ctypes.POINTER(struct_io_uring)]
except AttributeError:
    pass
try:
    io_uring_cq_eventfd_toggle = _libraries['FIXME_STUB'].io_uring_cq_eventfd_toggle
    io_uring_cq_eventfd_toggle.restype = ctypes.c_int32
    io_uring_cq_eventfd_toggle.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.c_bool]
except AttributeError:
    pass
try:
    io_uring_wait_cqe_nr = _libraries['FIXME_STUB'].io_uring_wait_cqe_nr
    io_uring_wait_cqe_nr.restype = ctypes.c_int32
    io_uring_wait_cqe_nr.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.POINTER(struct_io_uring_cqe)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_peek_cqe = _libraries['FIXME_STUB'].io_uring_peek_cqe
    io_uring_peek_cqe.restype = ctypes.c_int32
    io_uring_peek_cqe.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.POINTER(struct_io_uring_cqe))]
except AttributeError:
    pass
try:
    io_uring_wait_cqe = _libraries['FIXME_STUB'].io_uring_wait_cqe
    io_uring_wait_cqe.restype = ctypes.c_int32
    io_uring_wait_cqe.argtypes = [ctypes.POINTER(struct_io_uring), ctypes.POINTER(ctypes.POINTER(struct_io_uring_cqe))]
except AttributeError:
    pass
ssize_t = ctypes.c_int64
try:
    io_uring_mlock_size = _libraries['FIXME_STUB'].io_uring_mlock_size
    io_uring_mlock_size.restype = ssize_t
    io_uring_mlock_size.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    io_uring_mlock_size_params = _libraries['FIXME_STUB'].io_uring_mlock_size_params
    io_uring_mlock_size_params.restype = ssize_t
    io_uring_mlock_size_params.argtypes = [ctypes.c_uint32, ctypes.POINTER(struct_io_uring_params)]
except AttributeError:
    pass
LINUX_IO_URING_H = True # macro
IORING_SETUP_IOPOLL = (1<<0) # macro
IORING_SETUP_SQPOLL = (1<<1) # macro
IORING_SETUP_SQ_AFF = (1<<2) # macro
IORING_SETUP_CQSIZE = (1<<3) # macro
IORING_SETUP_CLAMP = (1<<4) # macro
IORING_SETUP_ATTACH_WQ = (1<<5) # macro
IORING_SETUP_R_DISABLED = (1<<6) # macro
IORING_FSYNC_DATASYNC = (1<<0) # macro
IORING_TIMEOUT_ABS = (1<<0) # macro
IORING_TIMEOUT_UPDATE = (1<<1) # macro
IORING_TIMEOUT_BOOTTIME = (1<<2) # macro
IORING_TIMEOUT_REALTIME = (1<<3) # macro
IORING_LINK_TIMEOUT_UPDATE = (1<<4) # macro
IORING_TIMEOUT_CLOCK_MASK = ((1<<2)|(1<<3)) # macro
IORING_TIMEOUT_UPDATE_MASK = ((1<<1)|(1<<4)) # macro
SPLICE_F_FD_IN_FIXED = (1<<31) # macro
IORING_POLL_ADD_MULTI = (1<<0) # macro
IORING_POLL_UPDATE_EVENTS = (1<<1) # macro
IORING_POLL_UPDATE_USER_DATA = (1<<2) # macro
IORING_CQE_F_BUFFER = (1<<0) # macro
IORING_CQE_F_MORE = (1<<1) # macro
IORING_OFF_SQ_RING = 0 # macro
IORING_OFF_CQ_RING = 0x8000000 # macro
IORING_OFF_SQES = 0x10000000 # macro
IORING_SQ_NEED_WAKEUP = (1<<0) # macro
IORING_SQ_CQ_OVERFLOW = (1<<1) # macro
IORING_CQ_EVENTFD_DISABLED = (1<<0) # macro
IORING_ENTER_GETEVENTS = (1<<0) # macro
IORING_ENTER_SQ_WAKEUP = (1<<1) # macro
IORING_ENTER_SQ_WAIT = (1<<2) # macro
IORING_ENTER_EXT_ARG = (1<<3) # macro
IORING_FEAT_SINGLE_MMAP = (1<<0) # macro
IORING_FEAT_NODROP = (1<<1) # macro
IORING_FEAT_SUBMIT_STABLE = (1<<2) # macro
IORING_FEAT_RW_CUR_POS = (1<<3) # macro
IORING_FEAT_CUR_PERSONALITY = (1<<4) # macro
IORING_FEAT_FAST_POLL = (1<<5) # macro
IORING_FEAT_POLL_32BITS = (1<<6) # macro
IORING_FEAT_SQPOLL_NONFIXED = (1<<7) # macro
IORING_FEAT_EXT_ARG = (1<<8) # macro
IORING_FEAT_NATIVE_WORKERS = (1<<9) # macro
IORING_FEAT_RSRC_TAGS = (1<<10) # macro
IORING_REGISTER_FILES_SKIP = (-2) # macro
IO_URING_OP_SUPPORTED = (1<<0) # macro

# values for enumeration 'c__Ea_IOSQE_FIXED_FILE_BIT'
c__Ea_IOSQE_FIXED_FILE_BIT__enumvalues = {
    0: 'IOSQE_FIXED_FILE_BIT',
    1: 'IOSQE_IO_DRAIN_BIT',
    2: 'IOSQE_IO_LINK_BIT',
    3: 'IOSQE_IO_HARDLINK_BIT',
    4: 'IOSQE_ASYNC_BIT',
    5: 'IOSQE_BUFFER_SELECT_BIT',
}
IOSQE_FIXED_FILE_BIT = 0
IOSQE_IO_DRAIN_BIT = 1
IOSQE_IO_LINK_BIT = 2
IOSQE_IO_HARDLINK_BIT = 3
IOSQE_ASYNC_BIT = 4
IOSQE_BUFFER_SELECT_BIT = 5
c__Ea_IOSQE_FIXED_FILE_BIT = ctypes.c_uint32 # enum
IOSQE_FIXED_FILE = (1<<IOSQE_FIXED_FILE_BIT) # macro
IOSQE_IO_DRAIN = (1<<IOSQE_IO_DRAIN_BIT) # macro
IOSQE_IO_LINK = (1<<IOSQE_IO_LINK_BIT) # macro
IOSQE_IO_HARDLINK = (1<<IOSQE_IO_HARDLINK_BIT) # macro
IOSQE_ASYNC = (1<<IOSQE_ASYNC_BIT) # macro
IOSQE_BUFFER_SELECT = (1<<IOSQE_BUFFER_SELECT_BIT) # macro

# values for enumeration 'c__Ea_IORING_OP_NOP'
c__Ea_IORING_OP_NOP__enumvalues = {
    0: 'IORING_OP_NOP',
    1: 'IORING_OP_READV',
    2: 'IORING_OP_WRITEV',
    3: 'IORING_OP_FSYNC',
    4: 'IORING_OP_READ_FIXED',
    5: 'IORING_OP_WRITE_FIXED',
    6: 'IORING_OP_POLL_ADD',
    7: 'IORING_OP_POLL_REMOVE',
    8: 'IORING_OP_SYNC_FILE_RANGE',
    9: 'IORING_OP_SENDMSG',
    10: 'IORING_OP_RECVMSG',
    11: 'IORING_OP_TIMEOUT',
    12: 'IORING_OP_TIMEOUT_REMOVE',
    13: 'IORING_OP_ACCEPT',
    14: 'IORING_OP_ASYNC_CANCEL',
    15: 'IORING_OP_LINK_TIMEOUT',
    16: 'IORING_OP_CONNECT',
    17: 'IORING_OP_FALLOCATE',
    18: 'IORING_OP_OPENAT',
    19: 'IORING_OP_CLOSE',
    20: 'IORING_OP_FILES_UPDATE',
    21: 'IORING_OP_STATX',
    22: 'IORING_OP_READ',
    23: 'IORING_OP_WRITE',
    24: 'IORING_OP_FADVISE',
    25: 'IORING_OP_MADVISE',
    26: 'IORING_OP_SEND',
    27: 'IORING_OP_RECV',
    28: 'IORING_OP_OPENAT2',
    29: 'IORING_OP_EPOLL_CTL',
    30: 'IORING_OP_SPLICE',
    31: 'IORING_OP_PROVIDE_BUFFERS',
    32: 'IORING_OP_REMOVE_BUFFERS',
    33: 'IORING_OP_TEE',
    34: 'IORING_OP_SHUTDOWN',
    35: 'IORING_OP_RENAMEAT',
    36: 'IORING_OP_UNLINKAT',
    37: 'IORING_OP_MKDIRAT',
    38: 'IORING_OP_SYMLINKAT',
    39: 'IORING_OP_LINKAT',
    40: 'IORING_OP_LAST',
}
IORING_OP_NOP = 0
IORING_OP_READV = 1
IORING_OP_WRITEV = 2
IORING_OP_FSYNC = 3
IORING_OP_READ_FIXED = 4
IORING_OP_WRITE_FIXED = 5
IORING_OP_POLL_ADD = 6
IORING_OP_POLL_REMOVE = 7
IORING_OP_SYNC_FILE_RANGE = 8
IORING_OP_SENDMSG = 9
IORING_OP_RECVMSG = 10
IORING_OP_TIMEOUT = 11
IORING_OP_TIMEOUT_REMOVE = 12
IORING_OP_ACCEPT = 13
IORING_OP_ASYNC_CANCEL = 14
IORING_OP_LINK_TIMEOUT = 15
IORING_OP_CONNECT = 16
IORING_OP_FALLOCATE = 17
IORING_OP_OPENAT = 18
IORING_OP_CLOSE = 19
IORING_OP_FILES_UPDATE = 20
IORING_OP_STATX = 21
IORING_OP_READ = 22
IORING_OP_WRITE = 23
IORING_OP_FADVISE = 24
IORING_OP_MADVISE = 25
IORING_OP_SEND = 26
IORING_OP_RECV = 27
IORING_OP_OPENAT2 = 28
IORING_OP_EPOLL_CTL = 29
IORING_OP_SPLICE = 30
IORING_OP_PROVIDE_BUFFERS = 31
IORING_OP_REMOVE_BUFFERS = 32
IORING_OP_TEE = 33
IORING_OP_SHUTDOWN = 34
IORING_OP_RENAMEAT = 35
IORING_OP_UNLINKAT = 36
IORING_OP_MKDIRAT = 37
IORING_OP_SYMLINKAT = 38
IORING_OP_LINKAT = 39
IORING_OP_LAST = 40
c__Ea_IORING_OP_NOP = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_IORING_CQE_BUFFER_SHIFT'
c__Ea_IORING_CQE_BUFFER_SHIFT__enumvalues = {
    16: 'IORING_CQE_BUFFER_SHIFT',
}
IORING_CQE_BUFFER_SHIFT = 16
c__Ea_IORING_CQE_BUFFER_SHIFT = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_IORING_REGISTER_BUFFERS'
c__Ea_IORING_REGISTER_BUFFERS__enumvalues = {
    0: 'IORING_REGISTER_BUFFERS',
    1: 'IORING_UNREGISTER_BUFFERS',
    2: 'IORING_REGISTER_FILES',
    3: 'IORING_UNREGISTER_FILES',
    4: 'IORING_REGISTER_EVENTFD',
    5: 'IORING_UNREGISTER_EVENTFD',
    6: 'IORING_REGISTER_FILES_UPDATE',
    7: 'IORING_REGISTER_EVENTFD_ASYNC',
    8: 'IORING_REGISTER_PROBE',
    9: 'IORING_REGISTER_PERSONALITY',
    10: 'IORING_UNREGISTER_PERSONALITY',
    11: 'IORING_REGISTER_RESTRICTIONS',
    12: 'IORING_REGISTER_ENABLE_RINGS',
    13: 'IORING_REGISTER_FILES2',
    14: 'IORING_REGISTER_FILES_UPDATE2',
    15: 'IORING_REGISTER_BUFFERS2',
    16: 'IORING_REGISTER_BUFFERS_UPDATE',
    17: 'IORING_REGISTER_IOWQ_AFF',
    18: 'IORING_UNREGISTER_IOWQ_AFF',
    19: 'IORING_REGISTER_IOWQ_MAX_WORKERS',
    20: 'IORING_REGISTER_LAST',
}
IORING_REGISTER_BUFFERS = 0
IORING_UNREGISTER_BUFFERS = 1
IORING_REGISTER_FILES = 2
IORING_UNREGISTER_FILES = 3
IORING_REGISTER_EVENTFD = 4
IORING_UNREGISTER_EVENTFD = 5
IORING_REGISTER_FILES_UPDATE = 6
IORING_REGISTER_EVENTFD_ASYNC = 7
IORING_REGISTER_PROBE = 8
IORING_REGISTER_PERSONALITY = 9
IORING_UNREGISTER_PERSONALITY = 10
IORING_REGISTER_RESTRICTIONS = 11
IORING_REGISTER_ENABLE_RINGS = 12
IORING_REGISTER_FILES2 = 13
IORING_REGISTER_FILES_UPDATE2 = 14
IORING_REGISTER_BUFFERS2 = 15
IORING_REGISTER_BUFFERS_UPDATE = 16
IORING_REGISTER_IOWQ_AFF = 17
IORING_UNREGISTER_IOWQ_AFF = 18
IORING_REGISTER_IOWQ_MAX_WORKERS = 19
IORING_REGISTER_LAST = 20
c__Ea_IORING_REGISTER_BUFFERS = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_IO_WQ_BOUND'
c__Ea_IO_WQ_BOUND__enumvalues = {
    0: 'IO_WQ_BOUND',
    1: 'IO_WQ_UNBOUND',
}
IO_WQ_BOUND = 0
IO_WQ_UNBOUND = 1
c__Ea_IO_WQ_BOUND = ctypes.c_uint32 # enum
class struct_io_uring_files_update(Structure):
    pass

struct_io_uring_files_update._pack_ = 1 # source:False
struct_io_uring_files_update._fields_ = [
    ('offset', ctypes.c_uint32),
    ('resv', ctypes.c_uint32),
    ('fds', ctypes.c_uint64),
]

class struct_io_uring_rsrc_register(Structure):
    pass

struct_io_uring_rsrc_register._pack_ = 1 # source:False
struct_io_uring_rsrc_register._fields_ = [
    ('nr', ctypes.c_uint32),
    ('resv', ctypes.c_uint32),
    ('resv2', ctypes.c_uint64),
    ('data', ctypes.c_uint64),
    ('tags', ctypes.c_uint64),
]

class struct_io_uring_rsrc_update(Structure):
    pass

struct_io_uring_rsrc_update._pack_ = 1 # source:False
struct_io_uring_rsrc_update._fields_ = [
    ('offset', ctypes.c_uint32),
    ('resv', ctypes.c_uint32),
    ('data', ctypes.c_uint64),
]

class struct_io_uring_rsrc_update2(Structure):
    pass

struct_io_uring_rsrc_update2._pack_ = 1 # source:False
struct_io_uring_rsrc_update2._fields_ = [
    ('offset', ctypes.c_uint32),
    ('resv', ctypes.c_uint32),
    ('data', ctypes.c_uint64),
    ('tags', ctypes.c_uint64),
    ('nr', ctypes.c_uint32),
    ('resv2', ctypes.c_uint32),
]


# values for enumeration 'c__Ea_IORING_RESTRICTION_REGISTER_OP'
c__Ea_IORING_RESTRICTION_REGISTER_OP__enumvalues = {
    0: 'IORING_RESTRICTION_REGISTER_OP',
    1: 'IORING_RESTRICTION_SQE_OP',
    2: 'IORING_RESTRICTION_SQE_FLAGS_ALLOWED',
    3: 'IORING_RESTRICTION_SQE_FLAGS_REQUIRED',
    4: 'IORING_RESTRICTION_LAST',
}
IORING_RESTRICTION_REGISTER_OP = 0
IORING_RESTRICTION_SQE_OP = 1
IORING_RESTRICTION_SQE_FLAGS_ALLOWED = 2
IORING_RESTRICTION_SQE_FLAGS_REQUIRED = 3
IORING_RESTRICTION_LAST = 4
c__Ea_IORING_RESTRICTION_REGISTER_OP = ctypes.c_uint32 # enum
class struct_io_uring_getevents_arg(Structure):
    pass

struct_io_uring_getevents_arg._pack_ = 1 # source:False
struct_io_uring_getevents_arg._fields_ = [
    ('sigmask', ctypes.c_uint64),
    ('sigmask_sz', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
    ('ts', ctypes.c_uint64),
]

__all__ = \
    ['IORING_CQE_BUFFER_SHIFT', 'IORING_CQE_F_BUFFER',
    'IORING_CQE_F_MORE', 'IORING_CQ_EVENTFD_DISABLED',
    'IORING_ENTER_EXT_ARG', 'IORING_ENTER_GETEVENTS',
    'IORING_ENTER_SQ_WAIT', 'IORING_ENTER_SQ_WAKEUP',
    'IORING_FEAT_CUR_PERSONALITY', 'IORING_FEAT_EXT_ARG',
    'IORING_FEAT_FAST_POLL', 'IORING_FEAT_NATIVE_WORKERS',
    'IORING_FEAT_NODROP', 'IORING_FEAT_POLL_32BITS',
    'IORING_FEAT_RSRC_TAGS', 'IORING_FEAT_RW_CUR_POS',
    'IORING_FEAT_SINGLE_MMAP', 'IORING_FEAT_SQPOLL_NONFIXED',
    'IORING_FEAT_SUBMIT_STABLE', 'IORING_FSYNC_DATASYNC',
    'IORING_LINK_TIMEOUT_UPDATE', 'IORING_OFF_CQ_RING',
    'IORING_OFF_SQES', 'IORING_OFF_SQ_RING', 'IORING_OP_ACCEPT',
    'IORING_OP_ASYNC_CANCEL', 'IORING_OP_CLOSE', 'IORING_OP_CONNECT',
    'IORING_OP_EPOLL_CTL', 'IORING_OP_FADVISE', 'IORING_OP_FALLOCATE',
    'IORING_OP_FILES_UPDATE', 'IORING_OP_FSYNC', 'IORING_OP_LAST',
    'IORING_OP_LINKAT', 'IORING_OP_LINK_TIMEOUT', 'IORING_OP_MADVISE',
    'IORING_OP_MKDIRAT', 'IORING_OP_NOP', 'IORING_OP_OPENAT',
    'IORING_OP_OPENAT2', 'IORING_OP_POLL_ADD',
    'IORING_OP_POLL_REMOVE', 'IORING_OP_PROVIDE_BUFFERS',
    'IORING_OP_READ', 'IORING_OP_READV', 'IORING_OP_READ_FIXED',
    'IORING_OP_RECV', 'IORING_OP_RECVMSG', 'IORING_OP_REMOVE_BUFFERS',
    'IORING_OP_RENAMEAT', 'IORING_OP_SEND', 'IORING_OP_SENDMSG',
    'IORING_OP_SHUTDOWN', 'IORING_OP_SPLICE', 'IORING_OP_STATX',
    'IORING_OP_SYMLINKAT', 'IORING_OP_SYNC_FILE_RANGE',
    'IORING_OP_TEE', 'IORING_OP_TIMEOUT', 'IORING_OP_TIMEOUT_REMOVE',
    'IORING_OP_UNLINKAT', 'IORING_OP_WRITE', 'IORING_OP_WRITEV',
    'IORING_OP_WRITE_FIXED', 'IORING_POLL_ADD_MULTI',
    'IORING_POLL_UPDATE_EVENTS', 'IORING_POLL_UPDATE_USER_DATA',
    'IORING_REGISTER_BUFFERS', 'IORING_REGISTER_BUFFERS2',
    'IORING_REGISTER_BUFFERS_UPDATE', 'IORING_REGISTER_ENABLE_RINGS',
    'IORING_REGISTER_EVENTFD', 'IORING_REGISTER_EVENTFD_ASYNC',
    'IORING_REGISTER_FILES', 'IORING_REGISTER_FILES2',
    'IORING_REGISTER_FILES_SKIP', 'IORING_REGISTER_FILES_UPDATE',
    'IORING_REGISTER_FILES_UPDATE2', 'IORING_REGISTER_IOWQ_AFF',
    'IORING_REGISTER_IOWQ_MAX_WORKERS', 'IORING_REGISTER_LAST',
    'IORING_REGISTER_PERSONALITY', 'IORING_REGISTER_PROBE',
    'IORING_REGISTER_RESTRICTIONS', 'IORING_RESTRICTION_LAST',
    'IORING_RESTRICTION_REGISTER_OP',
    'IORING_RESTRICTION_SQE_FLAGS_ALLOWED',
    'IORING_RESTRICTION_SQE_FLAGS_REQUIRED',
    'IORING_RESTRICTION_SQE_OP', 'IORING_SETUP_ATTACH_WQ',
    'IORING_SETUP_CLAMP', 'IORING_SETUP_CQSIZE',
    'IORING_SETUP_IOPOLL', 'IORING_SETUP_R_DISABLED',
    'IORING_SETUP_SQPOLL', 'IORING_SETUP_SQ_AFF',
    'IORING_SQ_CQ_OVERFLOW', 'IORING_SQ_NEED_WAKEUP',
    'IORING_TIMEOUT_ABS', 'IORING_TIMEOUT_BOOTTIME',
    'IORING_TIMEOUT_CLOCK_MASK', 'IORING_TIMEOUT_REALTIME',
    'IORING_TIMEOUT_UPDATE', 'IORING_TIMEOUT_UPDATE_MASK',
    'IORING_UNREGISTER_BUFFERS', 'IORING_UNREGISTER_EVENTFD',
    'IORING_UNREGISTER_FILES', 'IORING_UNREGISTER_IOWQ_AFF',
    'IORING_UNREGISTER_PERSONALITY', 'IOSQE_ASYNC', 'IOSQE_ASYNC_BIT',
    'IOSQE_BUFFER_SELECT', 'IOSQE_BUFFER_SELECT_BIT',
    'IOSQE_FIXED_FILE', 'IOSQE_FIXED_FILE_BIT', 'IOSQE_IO_DRAIN',
    'IOSQE_IO_DRAIN_BIT', 'IOSQE_IO_HARDLINK',
    'IOSQE_IO_HARDLINK_BIT', 'IOSQE_IO_LINK', 'IOSQE_IO_LINK_BIT',
    'IO_URING_OP_SUPPORTED', 'IO_WQ_BOUND', 'IO_WQ_UNBOUND',
    'LIB_URING_H', 'LINUX_IO_URING_H', 'SPLICE_F_FD_IN_FIXED',
    '_XOPEN_SOURCE', '__io_uring_get_cqe',
    '__io_uring_prep_poll_mask', '__io_uring_set_target_fixed_file',
    '__io_uring_sqring_wait', '__u64',
    'c__Ea_IORING_CQE_BUFFER_SHIFT', 'c__Ea_IORING_OP_NOP',
    'c__Ea_IORING_REGISTER_BUFFERS',
    'c__Ea_IORING_RESTRICTION_REGISTER_OP',
    'c__Ea_IOSQE_FIXED_FILE_BIT', 'c__Ea_IO_WQ_BOUND', 'int64_t',
    'io_uring_cq_advance', 'io_uring_cq_eventfd_enabled',
    'io_uring_cq_eventfd_toggle', 'io_uring_cq_ready',
    'io_uring_cqe_get_data', 'io_uring_cqe_seen',
    'io_uring_enable_rings', 'io_uring_free_probe',
    'io_uring_get_probe', 'io_uring_get_probe_ring',
    'io_uring_get_sqe', 'io_uring_mlock_size',
    'io_uring_mlock_size_params', 'io_uring_opcode_supported',
    'io_uring_peek_batch_cqe', 'io_uring_peek_cqe',
    'io_uring_prep_accept', 'io_uring_prep_accept_direct',
    'io_uring_prep_cancel', 'io_uring_prep_close',
    'io_uring_prep_connect', 'io_uring_prep_epoll_ctl',
    'io_uring_prep_fadvise', 'io_uring_prep_fallocate',
    'io_uring_prep_files_update', 'io_uring_prep_fsync',
    'io_uring_prep_link_timeout', 'io_uring_prep_linkat',
    'io_uring_prep_madvise', 'io_uring_prep_mkdirat',
    'io_uring_prep_nop', 'io_uring_prep_openat',
    'io_uring_prep_openat2', 'io_uring_prep_openat2_direct',
    'io_uring_prep_openat_direct', 'io_uring_prep_poll_add',
    'io_uring_prep_poll_multishot', 'io_uring_prep_poll_remove',
    'io_uring_prep_poll_update', 'io_uring_prep_provide_buffers',
    'io_uring_prep_read', 'io_uring_prep_read_fixed',
    'io_uring_prep_readv', 'io_uring_prep_recv',
    'io_uring_prep_recvmsg', 'io_uring_prep_remove_buffers',
    'io_uring_prep_renameat', 'io_uring_prep_rw',
    'io_uring_prep_send', 'io_uring_prep_sendmsg',
    'io_uring_prep_shutdown', 'io_uring_prep_splice',
    'io_uring_prep_statx', 'io_uring_prep_symlinkat',
    'io_uring_prep_sync_file_range', 'io_uring_prep_tee',
    'io_uring_prep_timeout', 'io_uring_prep_timeout_remove',
    'io_uring_prep_timeout_update', 'io_uring_prep_unlinkat',
    'io_uring_prep_write', 'io_uring_prep_write_fixed',
    'io_uring_prep_writev', 'io_uring_queue_exit',
    'io_uring_queue_init', 'io_uring_queue_init_params',
    'io_uring_queue_mmap', 'io_uring_register_buffers',
    'io_uring_register_buffers_tags',
    'io_uring_register_buffers_update_tag',
    'io_uring_register_eventfd', 'io_uring_register_eventfd_async',
    'io_uring_register_files', 'io_uring_register_files_tags',
    'io_uring_register_files_update',
    'io_uring_register_files_update_tag',
    'io_uring_register_iowq_aff',
    'io_uring_register_iowq_max_workers',
    'io_uring_register_personality', 'io_uring_register_probe',
    'io_uring_register_restrictions', 'io_uring_ring_dontfork',
    'io_uring_sq_ready', 'io_uring_sq_space_left',
    'io_uring_sqe_set_data', 'io_uring_sqe_set_flags',
    'io_uring_sqring_wait', 'io_uring_submit',
    'io_uring_submit_and_wait', 'io_uring_unregister_buffers',
    'io_uring_unregister_eventfd', 'io_uring_unregister_files',
    'io_uring_unregister_iowq_aff', 'io_uring_unregister_personality',
    'io_uring_wait_cqe', 'io_uring_wait_cqe_nr',
    'io_uring_wait_cqe_timeout', 'io_uring_wait_cqes', 'mode_t',
    'off_t', 'size_t', 'socklen_t', 'ssize_t',
    'struct___kernel_timespec', 'struct_c__SA___sigset_t',
    'struct_c__SA_cpu_set_t', 'struct_epoll_event',
    'struct_io_cqring_offsets', 'struct_io_sqring_offsets',
    'struct_io_uring', 'struct_io_uring_cq', 'struct_io_uring_cqe',
    'struct_io_uring_files_update', 'struct_io_uring_getevents_arg',
    'struct_io_uring_params', 'struct_io_uring_probe',
    'struct_io_uring_probe_op', 'struct_io_uring_restriction',
    'struct_io_uring_rsrc_register', 'struct_io_uring_rsrc_update',
    'struct_io_uring_rsrc_update2', 'struct_io_uring_sq',
    'struct_io_uring_sqe', 'struct_iovec', 'struct_msghdr',
    'struct_open_how', 'struct_sockaddr', 'struct_statx',
    'union_io_uring_restriction_0', 'union_io_uring_sqe_0',
    'union_io_uring_sqe_1', 'union_io_uring_sqe_2',
    'union_io_uring_sqe_3', 'union_io_uring_sqe_4']
NR_io_uring_setup = 425
NR_io_uring_enter = 426
NR_io_uring_register = 427
