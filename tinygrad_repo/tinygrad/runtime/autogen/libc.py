# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util, os


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

# libraries['libc'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['libc'] = None if (libc_path := ctypes.util.find_library('c')) is None else ctypes.CDLL(libc_path, use_errno=True) #  ctypes.CDLL('libc')
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





_SYS_MMAN_H = 1 # macro
__need_size_t = True # macro
__off_t_defined = True # macro
__mode_t_defined = True # macro
# MAP_FAILED = ((void*)-1) # macro
off_t = ctypes.c_int64
mode_t = ctypes.c_uint32
size_t = ctypes.c_uint64
__off_t = ctypes.c_int64
try:
    mmap = _libraries['libc'].mmap
    mmap.restype = ctypes.POINTER(None)
    mmap.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, __off_t]
except AttributeError:
    pass
try:
    munmap = _libraries['libc'].munmap
    munmap.restype = ctypes.c_int32
    munmap.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    mprotect = _libraries['libc'].mprotect
    mprotect.restype = ctypes.c_int32
    mprotect.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    msync = _libraries['libc'].msync
    msync.restype = ctypes.c_int32
    msync.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    madvise = _libraries['libc'].madvise
    madvise.restype = ctypes.c_int32
    madvise.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    posix_madvise = _libraries['libc'].posix_madvise
    posix_madvise.restype = ctypes.c_int32
    posix_madvise.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    mlock = _libraries['libc'].mlock
    mlock.restype = ctypes.c_int32
    mlock.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    munlock = _libraries['libc'].munlock
    munlock.restype = ctypes.c_int32
    munlock.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    mlockall = _libraries['libc'].mlockall
    mlockall.restype = ctypes.c_int32
    mlockall.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    munlockall = _libraries['libc'].munlockall
    munlockall.restype = ctypes.c_int32
    munlockall.argtypes = []
except AttributeError:
    pass
try:
    mincore = _libraries['libc'].mincore
    mincore.restype = ctypes.c_int32
    mincore.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    shm_open = _libraries['libc'].shm_open
    shm_open.restype = ctypes.c_int32
    shm_open.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, mode_t]
except AttributeError:
    pass
try:
    shm_unlink = _libraries['libc'].shm_unlink
    shm_unlink.restype = ctypes.c_int32
    shm_unlink.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
_SYSCALL_H = 1 # macro
_STRING_H = 1 # macro
__GLIBC_INTERNAL_STARTING_HEADER_IMPLEMENTATION = True # macro
__need_NULL = True # macro
try:
    memcpy = _libraries['libc'].memcpy
    memcpy.restype = ctypes.POINTER(None)
    memcpy.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    memmove = _libraries['libc'].memmove
    memmove.restype = ctypes.POINTER(None)
    memmove.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    memccpy = _libraries['libc'].memccpy
    memccpy.restype = ctypes.POINTER(None)
    memccpy.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    memset = _libraries['libc'].memset
    memset.restype = ctypes.POINTER(None)
    memset.argtypes = [ctypes.POINTER(None), ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    memcmp = _libraries['libc'].memcmp
    memcmp.restype = ctypes.c_int32
    memcmp.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    __memcmpeq = _libraries['libc'].__memcmpeq
    __memcmpeq.restype = ctypes.c_int32
    __memcmpeq.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    memchr = _libraries['libc'].memchr
    memchr.restype = ctypes.POINTER(None)
    memchr.argtypes = [ctypes.POINTER(None), ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    strcpy = _libraries['libc'].strcpy
    strcpy.restype = ctypes.POINTER(ctypes.c_char)
    strcpy.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strncpy = _libraries['libc'].strncpy
    strncpy.restype = ctypes.POINTER(ctypes.c_char)
    strncpy.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    strcat = _libraries['libc'].strcat
    strcat.restype = ctypes.POINTER(ctypes.c_char)
    strcat.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strncat = _libraries['libc'].strncat
    strncat.restype = ctypes.POINTER(ctypes.c_char)
    strncat.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    strcmp = _libraries['libc'].strcmp
    strcmp.restype = ctypes.c_int32
    strcmp.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strncmp = _libraries['libc'].strncmp
    strncmp.restype = ctypes.c_int32
    strncmp.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    strcoll = _libraries['libc'].strcoll
    strcoll.restype = ctypes.c_int32
    strcoll.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strxfrm = _libraries['libc'].strxfrm
    strxfrm.restype = ctypes.c_uint64
    strxfrm.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
class struct___locale_struct(Structure):
    pass

class struct___locale_data(Structure):
    pass

struct___locale_struct._pack_ = 1 # source:False
struct___locale_struct._fields_ = [
    ('__locales', ctypes.POINTER(struct___locale_data) * 13),
    ('__ctype_b', ctypes.POINTER(ctypes.c_uint16)),
    ('__ctype_tolower', ctypes.POINTER(ctypes.c_int32)),
    ('__ctype_toupper', ctypes.POINTER(ctypes.c_int32)),
    ('__names', ctypes.POINTER(ctypes.c_char) * 13),
]

locale_t = ctypes.POINTER(struct___locale_struct)
try:
    strcoll_l = _libraries['libc'].strcoll_l
    strcoll_l.restype = ctypes.c_int32
    strcoll_l.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), locale_t]
except AttributeError:
    pass
try:
    strxfrm_l = _libraries['libc'].strxfrm_l
    strxfrm_l.restype = size_t
    strxfrm_l.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t, locale_t]
except AttributeError:
    pass
try:
    strdup = _libraries['libc'].strdup
    strdup.restype = ctypes.POINTER(ctypes.c_char)
    strdup.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strndup = _libraries['libc'].strndup
    strndup.restype = ctypes.POINTER(ctypes.c_char)
    strndup.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    strchr = _libraries['libc'].strchr
    strchr.restype = ctypes.POINTER(ctypes.c_char)
    strchr.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    strrchr = _libraries['libc'].strrchr
    strrchr.restype = ctypes.POINTER(ctypes.c_char)
    strrchr.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    strchrnul = _libraries['libc'].strchrnul
    strchrnul.restype = ctypes.POINTER(ctypes.c_char)
    strchrnul.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    strcspn = _libraries['libc'].strcspn
    strcspn.restype = ctypes.c_uint64
    strcspn.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strspn = _libraries['libc'].strspn
    strspn.restype = ctypes.c_uint64
    strspn.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strpbrk = _libraries['libc'].strpbrk
    strpbrk.restype = ctypes.POINTER(ctypes.c_char)
    strpbrk.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strstr = _libraries['libc'].strstr
    strstr.restype = ctypes.POINTER(ctypes.c_char)
    strstr.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strtok = _libraries['libc'].strtok
    strtok.restype = ctypes.POINTER(ctypes.c_char)
    strtok.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    __strtok_r = _libraries['libc'].__strtok_r
    __strtok_r.restype = ctypes.POINTER(ctypes.c_char)
    __strtok_r.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    strtok_r = _libraries['libc'].strtok_r
    strtok_r.restype = ctypes.POINTER(ctypes.c_char)
    strtok_r.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    strcasestr = _libraries['libc'].strcasestr
    strcasestr.restype = ctypes.POINTER(ctypes.c_char)
    strcasestr.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    memmem = _libraries['libc'].memmem
    memmem.restype = ctypes.POINTER(None)
    memmem.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    __mempcpy = _libraries['libc'].__mempcpy
    __mempcpy.restype = ctypes.POINTER(None)
    __mempcpy.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    mempcpy = _libraries['libc'].mempcpy
    mempcpy.restype = ctypes.POINTER(None)
    mempcpy.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    strlen = _libraries['libc'].strlen
    strlen.restype = ctypes.c_uint64
    strlen.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strnlen = _libraries['libc'].strnlen
    strnlen.restype = size_t
    strnlen.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    strerror = _libraries['libc'].strerror
    strerror.restype = ctypes.POINTER(ctypes.c_char)
    strerror.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    strerror_r = _libraries['libc'].strerror_r
    strerror_r.restype = ctypes.c_int32
    strerror_r.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    strerror_l = _libraries['libc'].strerror_l
    strerror_l.restype = ctypes.POINTER(ctypes.c_char)
    strerror_l.argtypes = [ctypes.c_int32, locale_t]
except AttributeError:
    pass
try:
    explicit_bzero = _libraries['libc'].explicit_bzero
    explicit_bzero.restype = None
    explicit_bzero.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    strsep = _libraries['libc'].strsep
    strsep.restype = ctypes.POINTER(ctypes.c_char)
    strsep.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    strsignal = _libraries['libc'].strsignal
    strsignal.restype = ctypes.POINTER(ctypes.c_char)
    strsignal.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    __stpcpy = _libraries['libc'].__stpcpy
    __stpcpy.restype = ctypes.POINTER(ctypes.c_char)
    __stpcpy.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    stpcpy = _libraries['libc'].stpcpy
    stpcpy.restype = ctypes.POINTER(ctypes.c_char)
    stpcpy.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    __stpncpy = _libraries['libc'].__stpncpy
    __stpncpy.restype = ctypes.POINTER(ctypes.c_char)
    __stpncpy.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    stpncpy = _libraries['libc'].stpncpy
    stpncpy.restype = ctypes.POINTER(ctypes.c_char)
    stpncpy.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    strlcpy = _libraries['libc'].strlcpy
    strlcpy.restype = size_t
    strlcpy.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    strlcat = _libraries['libc'].strlcat
    strlcat.restype = size_t
    strlcat.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
_ELF_H = 1 # macro
EI_NIDENT = (16) # macro
EI_MAG0 = 0 # macro
ELFMAG0 = 0x7f # macro
EI_MAG1 = 1 # macro
ELFMAG1 = 'E' # macro
EI_MAG2 = 2 # macro
ELFMAG2 = 'L' # macro
EI_MAG3 = 3 # macro
ELFMAG3 = 'F' # macro
ELFMAG = "\177ELF" # macro
SELFMAG = 4 # macro
EI_CLASS = 4 # macro
ELFCLASSNONE = 0 # macro
ELFCLASS32 = 1 # macro
ELFCLASS64 = 2 # macro
ELFCLASSNUM = 3 # macro
EI_DATA = 5 # macro
ELFDATANONE = 0 # macro
ELFDATA2LSB = 1 # macro
ELFDATA2MSB = 2 # macro
ELFDATANUM = 3 # macro
EI_VERSION = 6 # macro
EI_OSABI = 7 # macro
ELFOSABI_NONE = 0 # macro
ELFOSABI_SYSV = 0 # macro
ELFOSABI_HPUX = 1 # macro
ELFOSABI_NETBSD = 2 # macro
ELFOSABI_GNU = 3 # macro
ELFOSABI_LINUX = 3 # macro
ELFOSABI_SOLARIS = 6 # macro
ELFOSABI_AIX = 7 # macro
ELFOSABI_IRIX = 8 # macro
ELFOSABI_FREEBSD = 9 # macro
ELFOSABI_TRU64 = 10 # macro
ELFOSABI_MODESTO = 11 # macro
ELFOSABI_OPENBSD = 12 # macro
ELFOSABI_ARM_AEABI = 64 # macro
ELFOSABI_ARM = 97 # macro
ELFOSABI_STANDALONE = 255 # macro
EI_ABIVERSION = 8 # macro
EI_PAD = 9 # macro
ET_NONE = 0 # macro
ET_REL = 1 # macro
ET_EXEC = 2 # macro
ET_DYN = 3 # macro
ET_CORE = 4 # macro
ET_NUM = 5 # macro
ET_LOOS = 0xfe00 # macro
ET_HIOS = 0xfeff # macro
ET_LOPROC = 0xff00 # macro
ET_HIPROC = 0xffff # macro
EM_NONE = 0 # macro
EM_M32 = 1 # macro
EM_SPARC = 2 # macro
EM_386 = 3 # macro
EM_68K = 4 # macro
EM_88K = 5 # macro
EM_IAMCU = 6 # macro
EM_860 = 7 # macro
EM_MIPS = 8 # macro
EM_S370 = 9 # macro
EM_MIPS_RS3_LE = 10 # macro
EM_PARISC = 15 # macro
EM_VPP500 = 17 # macro
EM_SPARC32PLUS = 18 # macro
EM_960 = 19 # macro
EM_PPC = 20 # macro
EM_PPC64 = 21 # macro
EM_S390 = 22 # macro
EM_SPU = 23 # macro
EM_V800 = 36 # macro
EM_FR20 = 37 # macro
EM_RH32 = 38 # macro
EM_RCE = 39 # macro
EM_ARM = 40 # macro
EM_FAKE_ALPHA = 41 # macro
EM_SH = 42 # macro
EM_SPARCV9 = 43 # macro
EM_TRICORE = 44 # macro
EM_ARC = 45 # macro
EM_H8_300 = 46 # macro
EM_H8_300H = 47 # macro
EM_H8S = 48 # macro
EM_H8_500 = 49 # macro
EM_IA_64 = 50 # macro
EM_MIPS_X = 51 # macro
EM_COLDFIRE = 52 # macro
EM_68HC12 = 53 # macro
EM_MMA = 54 # macro
EM_PCP = 55 # macro
EM_NCPU = 56 # macro
EM_NDR1 = 57 # macro
EM_STARCORE = 58 # macro
EM_ME16 = 59 # macro
EM_ST100 = 60 # macro
EM_TINYJ = 61 # macro
EM_X86_64 = 62 # macro
EM_PDSP = 63 # macro
EM_PDP10 = 64 # macro
EM_PDP11 = 65 # macro
EM_FX66 = 66 # macro
EM_ST9PLUS = 67 # macro
EM_ST7 = 68 # macro
EM_68HC16 = 69 # macro
EM_68HC11 = 70 # macro
EM_68HC08 = 71 # macro
EM_68HC05 = 72 # macro
EM_SVX = 73 # macro
EM_ST19 = 74 # macro
EM_VAX = 75 # macro
EM_CRIS = 76 # macro
EM_JAVELIN = 77 # macro
EM_FIREPATH = 78 # macro
EM_ZSP = 79 # macro
EM_MMIX = 80 # macro
EM_HUANY = 81 # macro
EM_PRISM = 82 # macro
EM_AVR = 83 # macro
EM_FR30 = 84 # macro
EM_D10V = 85 # macro
EM_D30V = 86 # macro
EM_V850 = 87 # macro
EM_M32R = 88 # macro
EM_MN10300 = 89 # macro
EM_MN10200 = 90 # macro
EM_PJ = 91 # macro
EM_OPENRISC = 92 # macro
EM_ARC_COMPACT = 93 # macro
EM_XTENSA = 94 # macro
EM_VIDEOCORE = 95 # macro
EM_TMM_GPP = 96 # macro
EM_NS32K = 97 # macro
EM_TPC = 98 # macro
EM_SNP1K = 99 # macro
EM_ST200 = 100 # macro
EM_IP2K = 101 # macro
EM_MAX = 102 # macro
EM_CR = 103 # macro
EM_F2MC16 = 104 # macro
EM_MSP430 = 105 # macro
EM_BLACKFIN = 106 # macro
EM_SE_C33 = 107 # macro
EM_SEP = 108 # macro
EM_ARCA = 109 # macro
EM_UNICORE = 110 # macro
EM_EXCESS = 111 # macro
EM_DXP = 112 # macro
EM_ALTERA_NIOS2 = 113 # macro
EM_CRX = 114 # macro
EM_XGATE = 115 # macro
EM_C166 = 116 # macro
EM_M16C = 117 # macro
EM_DSPIC30F = 118 # macro
EM_CE = 119 # macro
EM_M32C = 120 # macro
EM_TSK3000 = 131 # macro
EM_RS08 = 132 # macro
EM_SHARC = 133 # macro
EM_ECOG2 = 134 # macro
EM_SCORE7 = 135 # macro
EM_DSP24 = 136 # macro
EM_VIDEOCORE3 = 137 # macro
EM_LATTICEMICO32 = 138 # macro
EM_SE_C17 = 139 # macro
EM_TI_C6000 = 140 # macro
EM_TI_C2000 = 141 # macro
EM_TI_C5500 = 142 # macro
EM_TI_ARP32 = 143 # macro
EM_TI_PRU = 144 # macro
EM_MMDSP_PLUS = 160 # macro
EM_CYPRESS_M8C = 161 # macro
EM_R32C = 162 # macro
EM_TRIMEDIA = 163 # macro
EM_QDSP6 = 164 # macro
EM_8051 = 165 # macro
EM_STXP7X = 166 # macro
EM_NDS32 = 167 # macro
EM_ECOG1X = 168 # macro
EM_MAXQ30 = 169 # macro
EM_XIMO16 = 170 # macro
EM_MANIK = 171 # macro
EM_CRAYNV2 = 172 # macro
EM_RX = 173 # macro
EM_METAG = 174 # macro
EM_MCST_ELBRUS = 175 # macro
EM_ECOG16 = 176 # macro
EM_CR16 = 177 # macro
EM_ETPU = 178 # macro
EM_SLE9X = 179 # macro
EM_L10M = 180 # macro
EM_K10M = 181 # macro
EM_AARCH64 = 183 # macro
EM_AVR32 = 185 # macro
EM_STM8 = 186 # macro
EM_TILE64 = 187 # macro
EM_TILEPRO = 188 # macro
EM_MICROBLAZE = 189 # macro
EM_CUDA = 190 # macro
EM_TILEGX = 191 # macro
EM_CLOUDSHIELD = 192 # macro
EM_COREA_1ST = 193 # macro
EM_COREA_2ND = 194 # macro
EM_ARCV2 = 195 # macro
EM_OPEN8 = 196 # macro
EM_RL78 = 197 # macro
EM_VIDEOCORE5 = 198 # macro
EM_78KOR = 199 # macro
EM_56800EX = 200 # macro
EM_BA1 = 201 # macro
EM_BA2 = 202 # macro
EM_XCORE = 203 # macro
EM_MCHP_PIC = 204 # macro
EM_INTELGT = 205 # macro
EM_KM32 = 210 # macro
EM_KMX32 = 211 # macro
EM_EMX16 = 212 # macro
EM_EMX8 = 213 # macro
EM_KVARC = 214 # macro
EM_CDP = 215 # macro
EM_COGE = 216 # macro
EM_COOL = 217 # macro
EM_NORC = 218 # macro
EM_CSR_KALIMBA = 219 # macro
EM_Z80 = 220 # macro
EM_VISIUM = 221 # macro
EM_FT32 = 222 # macro
EM_MOXIE = 223 # macro
EM_AMDGPU = 224 # macro
EM_RISCV = 243 # macro
EM_BPF = 247 # macro
EM_CSKY = 252 # macro
EM_LOONGARCH = 258 # macro
EM_NUM = 259 # macro
EM_ARC_A5 = 93 # macro
EM_ALPHA = 0x9026 # macro
EV_NONE = 0 # macro
EV_CURRENT = 1 # macro
EV_NUM = 2 # macro
SHN_UNDEF = 0 # macro
SHN_LORESERVE = 0xff00 # macro
SHN_LOPROC = 0xff00 # macro
SHN_BEFORE = 0xff00 # macro
SHN_AFTER = 0xff01 # macro
SHN_HIPROC = 0xff1f # macro
SHN_LOOS = 0xff20 # macro
SHN_HIOS = 0xff3f # macro
SHN_ABS = 0xfff1 # macro
SHN_COMMON = 0xfff2 # macro
SHN_XINDEX = 0xffff # macro
SHN_HIRESERVE = 0xffff # macro
SHT_NULL = 0 # macro
SHT_PROGBITS = 1 # macro
SHT_SYMTAB = 2 # macro
SHT_STRTAB = 3 # macro
SHT_RELA = 4 # macro
SHT_HASH = 5 # macro
SHT_DYNAMIC = 6 # macro
SHT_NOTE = 7 # macro
SHT_NOBITS = 8 # macro
SHT_REL = 9 # macro
SHT_SHLIB = 10 # macro
SHT_DYNSYM = 11 # macro
SHT_INIT_ARRAY = 14 # macro
SHT_FINI_ARRAY = 15 # macro
SHT_PREINIT_ARRAY = 16 # macro
SHT_GROUP = 17 # macro
SHT_SYMTAB_SHNDX = 18 # macro
SHT_RELR = 19 # macro
SHT_NUM = 20 # macro
SHT_LOOS = 0x60000000 # macro
SHT_GNU_ATTRIBUTES = 0x6ffffff5 # macro
SHT_GNU_HASH = 0x6ffffff6 # macro
SHT_GNU_LIBLIST = 0x6ffffff7 # macro
SHT_CHECKSUM = 0x6ffffff8 # macro
SHT_LOSUNW = 0x6ffffffa # macro
SHT_SUNW_move = 0x6ffffffa # macro
SHT_SUNW_COMDAT = 0x6ffffffb # macro
SHT_SUNW_syminfo = 0x6ffffffc # macro
SHT_GNU_verdef = 0x6ffffffd # macro
SHT_GNU_verneed = 0x6ffffffe # macro
SHT_GNU_versym = 0x6fffffff # macro
SHT_HISUNW = 0x6fffffff # macro
SHT_HIOS = 0x6fffffff # macro
SHT_LOPROC = 0x70000000 # macro
SHT_HIPROC = 0x7fffffff # macro
SHT_LOUSER = 0x80000000 # macro
SHT_HIUSER = 0x8fffffff # macro
SHF_WRITE = (1<<0) # macro
SHF_ALLOC = (1<<1) # macro
SHF_EXECINSTR = (1<<2) # macro
SHF_MERGE = (1<<4) # macro
SHF_STRINGS = (1<<5) # macro
SHF_INFO_LINK = (1<<6) # macro
SHF_LINK_ORDER = (1<<7) # macro
SHF_OS_NONCONFORMING = (1<<8) # macro
SHF_GROUP = (1<<9) # macro
SHF_TLS = (1<<10) # macro
SHF_COMPRESSED = (1<<11) # macro
SHF_MASKOS = 0x0ff00000 # macro
SHF_MASKPROC = 0xf0000000 # macro
SHF_GNU_RETAIN = (1<<21) # macro
SHF_ORDERED = (1<<30) # macro
SHF_EXCLUDE = (1<<31) # macro
ELFCOMPRESS_ZLIB = 1 # macro
ELFCOMPRESS_ZSTD = 2 # macro
ELFCOMPRESS_LOOS = 0x60000000 # macro
ELFCOMPRESS_HIOS = 0x6fffffff # macro
ELFCOMPRESS_LOPROC = 0x70000000 # macro
ELFCOMPRESS_HIPROC = 0x7fffffff # macro
GRP_COMDAT = 0x1 # macro
SYMINFO_BT_SELF = 0xffff # macro
SYMINFO_BT_PARENT = 0xfffe # macro
SYMINFO_BT_LOWRESERVE = 0xff00 # macro
SYMINFO_FLG_DIRECT = 0x0001 # macro
SYMINFO_FLG_PASSTHRU = 0x0002 # macro
SYMINFO_FLG_COPY = 0x0004 # macro
SYMINFO_FLG_LAZYLOAD = 0x0008 # macro
SYMINFO_NONE = 0 # macro
SYMINFO_CURRENT = 1 # macro
SYMINFO_NUM = 2 # macro
def ELF32_ST_BIND(val):  # macro
   return (((val))>>4)
def ELF32_ST_TYPE(val):  # macro
   return ((val)&0xf)
def ELF32_ST_INFO(bind, type):  # macro
   return (((bind)<<4)+((type)&0xf))
def ELF64_ST_BIND(val):  # macro
   return ELF32_ST_BIND(val)
def ELF64_ST_TYPE(val):  # macro
   return ELF32_ST_TYPE(val)
def ELF64_ST_INFO(bind, type):  # macro
   return ELF32_ST_INFO((bind),(type))
STB_LOCAL = 0 # macro
STB_GLOBAL = 1 # macro
STB_WEAK = 2 # macro
STB_NUM = 3 # macro
STB_LOOS = 10 # macro
STB_GNU_UNIQUE = 10 # macro
STB_HIOS = 12 # macro
STB_LOPROC = 13 # macro
STB_HIPROC = 15 # macro
STT_NOTYPE = 0 # macro
STT_OBJECT = 1 # macro
STT_FUNC = 2 # macro
STT_SECTION = 3 # macro
STT_FILE = 4 # macro
STT_COMMON = 5 # macro
STT_TLS = 6 # macro
STT_NUM = 7 # macro
STT_LOOS = 10 # macro
STT_GNU_IFUNC = 10 # macro
STT_HIOS = 12 # macro
STT_LOPROC = 13 # macro
STT_HIPROC = 15 # macro
STN_UNDEF = 0 # macro
def ELF32_ST_VISIBILITY(o):  # macro
   return ((o)&0x03)
def ELF64_ST_VISIBILITY(o):  # macro
   return ELF32_ST_VISIBILITY(o)
STV_DEFAULT = 0 # macro
STV_INTERNAL = 1 # macro
STV_HIDDEN = 2 # macro
STV_PROTECTED = 3 # macro
def ELF32_R_SYM(val):  # macro
   return ((val)>>8)
def ELF32_R_TYPE(val):  # macro
   return ((val)&0xff)
def ELF32_R_INFO(sym, type):  # macro
   return (((sym)<<8)+((type)&0xff))
def ELF64_R_SYM(i):  # macro
   return ((i)>>32)
def ELF64_R_TYPE(i):  # macro
   return ((i)&0xffffffff)
PN_XNUM = 0xffff # macro
PT_NULL = 0 # macro
PT_LOAD = 1 # macro
PT_DYNAMIC = 2 # macro
PT_INTERP = 3 # macro
PT_NOTE = 4 # macro
PT_SHLIB = 5 # macro
PT_PHDR = 6 # macro
PT_TLS = 7 # macro
PT_NUM = 8 # macro
PT_LOOS = 0x60000000 # macro
PT_GNU_EH_FRAME = 0x6474e550 # macro
PT_GNU_STACK = 0x6474e551 # macro
PT_GNU_RELRO = 0x6474e552 # macro
PT_GNU_PROPERTY = 0x6474e553 # macro
PT_GNU_SFRAME = 0x6474e554 # macro
PT_LOSUNW = 0x6ffffffa # macro
PT_SUNWBSS = 0x6ffffffa # macro
PT_SUNWSTACK = 0x6ffffffb # macro
PT_HISUNW = 0x6fffffff # macro
PT_HIOS = 0x6fffffff # macro
PT_LOPROC = 0x70000000 # macro
PT_HIPROC = 0x7fffffff # macro
PF_X = (1<<0) # macro
PF_W = (1<<1) # macro
PF_R = (1<<2) # macro
PF_MASKOS = 0x0ff00000 # macro
PF_MASKPROC = 0xf0000000 # macro
NT_PRSTATUS = 1 # macro
NT_PRFPREG = 2 # macro
NT_FPREGSET = 2 # macro
NT_PRPSINFO = 3 # macro
NT_PRXREG = 4 # macro
NT_TASKSTRUCT = 4 # macro
NT_PLATFORM = 5 # macro
NT_AUXV = 6 # macro
NT_GWINDOWS = 7 # macro
NT_ASRS = 8 # macro
NT_PSTATUS = 10 # macro
NT_PSINFO = 13 # macro
NT_PRCRED = 14 # macro
NT_UTSNAME = 15 # macro
NT_LWPSTATUS = 16 # macro
NT_LWPSINFO = 17 # macro
NT_PRFPXREG = 20 # macro
NT_SIGINFO = 0x53494749 # macro
NT_FILE = 0x46494c45 # macro
NT_PRXFPREG = 0x46e62b7f # macro
NT_PPC_VMX = 0x100 # macro
NT_PPC_SPE = 0x101 # macro
NT_PPC_VSX = 0x102 # macro
NT_PPC_TAR = 0x103 # macro
NT_PPC_PPR = 0x104 # macro
NT_PPC_DSCR = 0x105 # macro
NT_PPC_EBB = 0x106 # macro
NT_PPC_PMU = 0x107 # macro
NT_PPC_TM_CGPR = 0x108 # macro
NT_PPC_TM_CFPR = 0x109 # macro
NT_PPC_TM_CVMX = 0x10a # macro
NT_PPC_TM_CVSX = 0x10b # macro
NT_PPC_TM_SPR = 0x10c # macro
NT_PPC_TM_CTAR = 0x10d # macro
NT_PPC_TM_CPPR = 0x10e # macro
NT_PPC_TM_CDSCR = 0x10f # macro
NT_PPC_PKEY = 0x110 # macro
NT_PPC_DEXCR = 0x111 # macro
NT_PPC_HASHKEYR = 0x112 # macro
NT_386_TLS = 0x200 # macro
NT_386_IOPERM = 0x201 # macro
NT_X86_XSTATE = 0x202 # macro
NT_X86_SHSTK = 0x204 # macro
NT_S390_HIGH_GPRS = 0x300 # macro
NT_S390_TIMER = 0x301 # macro
NT_S390_TODCMP = 0x302 # macro
NT_S390_TODPREG = 0x303 # macro
NT_S390_CTRS = 0x304 # macro
NT_S390_PREFIX = 0x305 # macro
NT_S390_LAST_BREAK = 0x306 # macro
NT_S390_SYSTEM_CALL = 0x307 # macro
NT_S390_TDB = 0x308 # macro
NT_S390_VXRS_LOW = 0x309 # macro
NT_S390_VXRS_HIGH = 0x30a # macro
NT_S390_GS_CB = 0x30b # macro
NT_S390_GS_BC = 0x30c # macro
NT_S390_RI_CB = 0x30d # macro
NT_S390_PV_CPU_DATA = 0x30e # macro
NT_ARM_VFP = 0x400 # macro
NT_ARM_TLS = 0x401 # macro
NT_ARM_HW_BREAK = 0x402 # macro
NT_ARM_HW_WATCH = 0x403 # macro
NT_ARM_SYSTEM_CALL = 0x404 # macro
NT_ARM_SVE = 0x405 # macro
NT_ARM_PAC_MASK = 0x406 # macro
NT_ARM_PACA_KEYS = 0x407 # macro
NT_ARM_PACG_KEYS = 0x408 # macro
NT_ARM_TAGGED_ADDR_CTRL = 0x409 # macro
NT_ARM_PAC_ENABLED_KEYS = 0x40a # macro
NT_VMCOREDD = 0x700 # macro
NT_MIPS_DSP = 0x800 # macro
NT_MIPS_FP_MODE = 0x801 # macro
NT_MIPS_MSA = 0x802 # macro
NT_RISCV_CSR = 0x900 # macro
NT_RISCV_VECTOR = 0x901 # macro
NT_LOONGARCH_CPUCFG = 0xa00 # macro
NT_LOONGARCH_CSR = 0xa01 # macro
NT_LOONGARCH_LSX = 0xa02 # macro
NT_LOONGARCH_LASX = 0xa03 # macro
NT_LOONGARCH_LBT = 0xa04 # macro
NT_LOONGARCH_HW_BREAK = 0xa05 # macro
NT_LOONGARCH_HW_WATCH = 0xa06 # macro
NT_VERSION = 1 # macro
DT_NULL = 0 # macro
DT_NEEDED = 1 # macro
DT_PLTRELSZ = 2 # macro
DT_PLTGOT = 3 # macro
DT_HASH = 4 # macro
DT_STRTAB = 5 # macro
DT_SYMTAB = 6 # macro
DT_RELA = 7 # macro
DT_RELASZ = 8 # macro
DT_RELAENT = 9 # macro
DT_STRSZ = 10 # macro
DT_SYMENT = 11 # macro
DT_INIT = 12 # macro
DT_FINI = 13 # macro
DT_SONAME = 14 # macro
DT_RPATH = 15 # macro
DT_SYMBOLIC = 16 # macro
DT_REL = 17 # macro
DT_RELSZ = 18 # macro
DT_RELENT = 19 # macro
DT_PLTREL = 20 # macro
DT_DEBUG = 21 # macro
DT_TEXTREL = 22 # macro
DT_JMPREL = 23 # macro
DT_BIND_NOW = 24 # macro
DT_INIT_ARRAY = 25 # macro
DT_FINI_ARRAY = 26 # macro
DT_INIT_ARRAYSZ = 27 # macro
DT_FINI_ARRAYSZ = 28 # macro
DT_RUNPATH = 29 # macro
DT_FLAGS = 30 # macro
DT_ENCODING = 32 # macro
DT_PREINIT_ARRAY = 32 # macro
DT_PREINIT_ARRAYSZ = 33 # macro
DT_SYMTAB_SHNDX = 34 # macro
DT_RELRSZ = 35 # macro
DT_RELR = 36 # macro
DT_RELRENT = 37 # macro
DT_NUM = 38 # macro
DT_LOOS = 0x6000000d # macro
DT_HIOS = 0x6ffff000 # macro
DT_LOPROC = 0x70000000 # macro
DT_HIPROC = 0x7fffffff # macro
DT_VALRNGLO = 0x6ffffd00 # macro
DT_GNU_PRELINKED = 0x6ffffdf5 # macro
DT_GNU_CONFLICTSZ = 0x6ffffdf6 # macro
DT_GNU_LIBLISTSZ = 0x6ffffdf7 # macro
DT_CHECKSUM = 0x6ffffdf8 # macro
DT_PLTPADSZ = 0x6ffffdf9 # macro
DT_MOVEENT = 0x6ffffdfa # macro
DT_MOVESZ = 0x6ffffdfb # macro
DT_FEATURE_1 = 0x6ffffdfc # macro
DT_POSFLAG_1 = 0x6ffffdfd # macro
DT_SYMINSZ = 0x6ffffdfe # macro
DT_SYMINENT = 0x6ffffdff # macro
DT_VALRNGHI = 0x6ffffdff # macro
def DT_VALTAGIDX(tag):  # macro
   return (0x6ffffdff-(tag))
DT_VALNUM = 12 # macro
DT_ADDRRNGLO = 0x6ffffe00 # macro
DT_GNU_HASH = 0x6ffffef5 # macro
DT_TLSDESC_PLT = 0x6ffffef6 # macro
DT_TLSDESC_GOT = 0x6ffffef7 # macro
DT_GNU_CONFLICT = 0x6ffffef8 # macro
DT_GNU_LIBLIST = 0x6ffffef9 # macro
DT_CONFIG = 0x6ffffefa # macro
DT_DEPAUDIT = 0x6ffffefb # macro
DT_AUDIT = 0x6ffffefc # macro
DT_PLTPAD = 0x6ffffefd # macro
DT_MOVETAB = 0x6ffffefe # macro
DT_SYMINFO = 0x6ffffeff # macro
DT_ADDRRNGHI = 0x6ffffeff # macro
def DT_ADDRTAGIDX(tag):  # macro
   return (0x6ffffeff-(tag))
DT_ADDRNUM = 11 # macro
DT_VERSYM = 0x6ffffff0 # macro
DT_RELACOUNT = 0x6ffffff9 # macro
DT_RELCOUNT = 0x6ffffffa # macro
DT_FLAGS_1 = 0x6ffffffb # macro
DT_VERDEF = 0x6ffffffc # macro
DT_VERDEFNUM = 0x6ffffffd # macro
DT_VERNEED = 0x6ffffffe # macro
DT_VERNEEDNUM = 0x6fffffff # macro
def DT_VERSIONTAGIDX(tag):  # macro
   return (0x6fffffff-(tag))
DT_VERSIONTAGNUM = 16 # macro
DT_AUXILIARY = 0x7ffffffd # macro
DT_FILTER = 0x7fffffff # macro
DT_EXTRANUM = 3 # macro
DF_ORIGIN = 0x00000001 # macro
DF_SYMBOLIC = 0x00000002 # macro
DF_TEXTREL = 0x00000004 # macro
DF_BIND_NOW = 0x00000008 # macro
DF_STATIC_TLS = 0x00000010 # macro
DF_1_NOW = 0x00000001 # macro
DF_1_GLOBAL = 0x00000002 # macro
DF_1_GROUP = 0x00000004 # macro
DF_1_NODELETE = 0x00000008 # macro
DF_1_LOADFLTR = 0x00000010 # macro
DF_1_INITFIRST = 0x00000020 # macro
DF_1_NOOPEN = 0x00000040 # macro
DF_1_ORIGIN = 0x00000080 # macro
DF_1_DIRECT = 0x00000100 # macro
DF_1_TRANS = 0x00000200 # macro
DF_1_INTERPOSE = 0x00000400 # macro
DF_1_NODEFLIB = 0x00000800 # macro
DF_1_NODUMP = 0x00001000 # macro
DF_1_CONFALT = 0x00002000 # macro
DF_1_ENDFILTEE = 0x00004000 # macro
DF_1_DISPRELDNE = 0x00008000 # macro
DF_1_DISPRELPND = 0x00010000 # macro
DF_1_NODIRECT = 0x00020000 # macro
DF_1_IGNMULDEF = 0x00040000 # macro
DF_1_NOKSYMS = 0x00080000 # macro
DF_1_NOHDR = 0x00100000 # macro
DF_1_EDITED = 0x00200000 # macro
DF_1_NORELOC = 0x00400000 # macro
DF_1_SYMINTPOSE = 0x00800000 # macro
DF_1_GLOBAUDIT = 0x01000000 # macro
DF_1_SINGLETON = 0x02000000 # macro
DF_1_STUB = 0x04000000 # macro
DF_1_PIE = 0x08000000 # macro
DF_1_KMOD = 0x10000000 # macro
DF_1_WEAKFILTER = 0x20000000 # macro
DF_1_NOCOMMON = 0x40000000 # macro
DTF_1_PARINIT = 0x00000001 # macro
DTF_1_CONFEXP = 0x00000002 # macro
DF_P1_LAZYLOAD = 0x00000001 # macro
DF_P1_GROUPPERM = 0x00000002 # macro
VER_DEF_NONE = 0 # macro
VER_DEF_CURRENT = 1 # macro
VER_DEF_NUM = 2 # macro
VER_FLG_BASE = 0x1 # macro
VER_FLG_WEAK = 0x2 # macro
VER_NDX_LOCAL = 0 # macro
VER_NDX_GLOBAL = 1 # macro
VER_NDX_LORESERVE = 0xff00 # macro
VER_NDX_ELIMINATE = 0xff01 # macro
VER_NEED_NONE = 0 # macro
VER_NEED_CURRENT = 1 # macro
VER_NEED_NUM = 2 # macro
AT_NULL = 0 # macro
AT_IGNORE = 1 # macro
AT_EXECFD = 2 # macro
AT_PHDR = 3 # macro
AT_PHENT = 4 # macro
AT_PHNUM = 5 # macro
AT_PAGESZ = 6 # macro
AT_BASE = 7 # macro
AT_FLAGS = 8 # macro
AT_ENTRY = 9 # macro
AT_NOTELF = 10 # macro
AT_UID = 11 # macro
AT_EUID = 12 # macro
AT_GID = 13 # macro
AT_EGID = 14 # macro
AT_CLKTCK = 17 # macro
AT_PLATFORM = 15 # macro
AT_HWCAP = 16 # macro
AT_FPUCW = 18 # macro
AT_DCACHEBSIZE = 19 # macro
AT_ICACHEBSIZE = 20 # macro
AT_UCACHEBSIZE = 21 # macro
AT_IGNOREPPC = 22 # macro
AT_SECURE = 23 # macro
AT_BASE_PLATFORM = 24 # macro
AT_RANDOM = 25 # macro
AT_HWCAP2 = 26 # macro
AT_RSEQ_FEATURE_SIZE = 27 # macro
AT_RSEQ_ALIGN = 28 # macro
AT_HWCAP3 = 29 # macro
AT_HWCAP4 = 30 # macro
AT_EXECFN = 31 # macro
AT_SYSINFO = 32 # macro
AT_SYSINFO_EHDR = 33 # macro
AT_L1I_CACHESHAPE = 34 # macro
AT_L1D_CACHESHAPE = 35 # macro
AT_L2_CACHESHAPE = 36 # macro
AT_L3_CACHESHAPE = 37 # macro
AT_L1I_CACHESIZE = 40 # macro
AT_L1I_CACHEGEOMETRY = 41 # macro
AT_L1D_CACHESIZE = 42 # macro
AT_L1D_CACHEGEOMETRY = 43 # macro
AT_L2_CACHESIZE = 44 # macro
AT_L2_CACHEGEOMETRY = 45 # macro
AT_L3_CACHESIZE = 46 # macro
AT_L3_CACHEGEOMETRY = 47 # macro
AT_MINSIGSTKSZ = 51 # macro
ELF_NOTE_SOLARIS = "SUNW Solaris" # macro
ELF_NOTE_GNU = "GNU" # macro
ELF_NOTE_FDO = "FDO" # macro
ELF_NOTE_PAGESIZE_HINT = 1 # macro
NT_GNU_ABI_TAG = 1 # macro
ELF_NOTE_ABI = 1 # macro
ELF_NOTE_OS_LINUX = 0 # macro
ELF_NOTE_OS_GNU = 1 # macro
ELF_NOTE_OS_SOLARIS2 = 2 # macro
ELF_NOTE_OS_FREEBSD = 3 # macro
NT_GNU_HWCAP = 2 # macro
NT_GNU_BUILD_ID = 3 # macro
NT_GNU_GOLD_VERSION = 4 # macro
NT_GNU_PROPERTY_TYPE_0 = 5 # macro
NT_FDO_PACKAGING_METADATA = 0xcafe1a7e # macro
NOTE_GNU_PROPERTY_SECTION_NAME = ".note.gnu.property" # macro
GNU_PROPERTY_STACK_SIZE = 1 # macro
GNU_PROPERTY_NO_COPY_ON_PROTECTED = 2 # macro
GNU_PROPERTY_UINT32_AND_LO = 0xb0000000 # macro
GNU_PROPERTY_UINT32_AND_HI = 0xb0007fff # macro
GNU_PROPERTY_UINT32_OR_LO = 0xb0008000 # macro
GNU_PROPERTY_UINT32_OR_HI = 0xb000ffff # macro
GNU_PROPERTY_1_NEEDED = 0xb0008000 # macro
GNU_PROPERTY_1_NEEDED_INDIRECT_EXTERN_ACCESS = (1<<0) # macro
GNU_PROPERTY_LOPROC = 0xc0000000 # macro
GNU_PROPERTY_HIPROC = 0xdfffffff # macro
GNU_PROPERTY_LOUSER = 0xe0000000 # macro
GNU_PROPERTY_HIUSER = 0xffffffff # macro
GNU_PROPERTY_AARCH64_FEATURE_1_AND = 0xc0000000 # macro
GNU_PROPERTY_AARCH64_FEATURE_1_BTI = (1<<0) # macro
GNU_PROPERTY_AARCH64_FEATURE_1_PAC = (1<<1) # macro
GNU_PROPERTY_X86_ISA_1_USED = 0xc0010002 # macro
GNU_PROPERTY_X86_ISA_1_NEEDED = 0xc0008002 # macro
GNU_PROPERTY_X86_FEATURE_1_AND = 0xc0000002 # macro
GNU_PROPERTY_X86_ISA_1_BASELINE = (1<<0) # macro
GNU_PROPERTY_X86_ISA_1_V2 = (1<<1) # macro
GNU_PROPERTY_X86_ISA_1_V3 = (1<<2) # macro
GNU_PROPERTY_X86_ISA_1_V4 = (1<<3) # macro
GNU_PROPERTY_X86_FEATURE_1_IBT = (1<<0) # macro
GNU_PROPERTY_X86_FEATURE_1_SHSTK = (1<<1) # macro
def ELF32_M_SYM(info):  # macro
   return ((info)>>8)
def ELF32_M_SIZE(info):  # macro
   return ((info))
def ELF32_M_INFO(sym, size):  # macro
   return (((sym)<<8)+(size))
def ELF64_M_SYM(info):  # macro
   return ELF32_M_SYM(info)
def ELF64_M_SIZE(info):  # macro
   return ELF32_M_SIZE(info)
def ELF64_M_INFO(sym, size):  # macro
   return ELF32_M_INFO(sym,size)
EF_CPU32 = 0x00810000 # macro
R_68K_NONE = 0 # macro
R_68K_32 = 1 # macro
R_68K_16 = 2 # macro
R_68K_8 = 3 # macro
R_68K_PC32 = 4 # macro
R_68K_PC16 = 5 # macro
R_68K_PC8 = 6 # macro
R_68K_GOT32 = 7 # macro
R_68K_GOT16 = 8 # macro
R_68K_GOT8 = 9 # macro
R_68K_GOT32O = 10 # macro
R_68K_GOT16O = 11 # macro
R_68K_GOT8O = 12 # macro
R_68K_PLT32 = 13 # macro
R_68K_PLT16 = 14 # macro
R_68K_PLT8 = 15 # macro
R_68K_PLT32O = 16 # macro
R_68K_PLT16O = 17 # macro
R_68K_PLT8O = 18 # macro
R_68K_COPY = 19 # macro
R_68K_GLOB_DAT = 20 # macro
R_68K_JMP_SLOT = 21 # macro
R_68K_RELATIVE = 22 # macro
R_68K_TLS_GD32 = 25 # macro
R_68K_TLS_GD16 = 26 # macro
R_68K_TLS_GD8 = 27 # macro
R_68K_TLS_LDM32 = 28 # macro
R_68K_TLS_LDM16 = 29 # macro
R_68K_TLS_LDM8 = 30 # macro
R_68K_TLS_LDO32 = 31 # macro
R_68K_TLS_LDO16 = 32 # macro
R_68K_TLS_LDO8 = 33 # macro
R_68K_TLS_IE32 = 34 # macro
R_68K_TLS_IE16 = 35 # macro
R_68K_TLS_IE8 = 36 # macro
R_68K_TLS_LE32 = 37 # macro
R_68K_TLS_LE16 = 38 # macro
R_68K_TLS_LE8 = 39 # macro
R_68K_TLS_DTPMOD32 = 40 # macro
R_68K_TLS_DTPREL32 = 41 # macro
R_68K_TLS_TPREL32 = 42 # macro
R_68K_NUM = 43 # macro
R_386_NONE = 0 # macro
R_386_32 = 1 # macro
R_386_PC32 = 2 # macro
R_386_GOT32 = 3 # macro
R_386_PLT32 = 4 # macro
R_386_COPY = 5 # macro
R_386_GLOB_DAT = 6 # macro
R_386_JMP_SLOT = 7 # macro
R_386_RELATIVE = 8 # macro
R_386_GOTOFF = 9 # macro
R_386_GOTPC = 10 # macro
R_386_32PLT = 11 # macro
R_386_TLS_TPOFF = 14 # macro
R_386_TLS_IE = 15 # macro
R_386_TLS_GOTIE = 16 # macro
R_386_TLS_LE = 17 # macro
R_386_TLS_GD = 18 # macro
R_386_TLS_LDM = 19 # macro
R_386_16 = 20 # macro
R_386_PC16 = 21 # macro
R_386_8 = 22 # macro
R_386_PC8 = 23 # macro
R_386_TLS_GD_32 = 24 # macro
R_386_TLS_GD_PUSH = 25 # macro
R_386_TLS_GD_CALL = 26 # macro
R_386_TLS_GD_POP = 27 # macro
R_386_TLS_LDM_32 = 28 # macro
R_386_TLS_LDM_PUSH = 29 # macro
R_386_TLS_LDM_CALL = 30 # macro
R_386_TLS_LDM_POP = 31 # macro
R_386_TLS_LDO_32 = 32 # macro
R_386_TLS_IE_32 = 33 # macro
R_386_TLS_LE_32 = 34 # macro
R_386_TLS_DTPMOD32 = 35 # macro
R_386_TLS_DTPOFF32 = 36 # macro
R_386_TLS_TPOFF32 = 37 # macro
R_386_SIZE32 = 38 # macro
R_386_TLS_GOTDESC = 39 # macro
R_386_TLS_DESC_CALL = 40 # macro
R_386_TLS_DESC = 41 # macro
R_386_IRELATIVE = 42 # macro
R_386_GOT32X = 43 # macro
R_386_NUM = 44 # macro
STT_SPARC_REGISTER = 13 # macro
EF_SPARCV9_MM = 3 # macro
EF_SPARCV9_TSO = 0 # macro
EF_SPARCV9_PSO = 1 # macro
EF_SPARCV9_RMO = 2 # macro
EF_SPARC_LEDATA = 0x800000 # macro
EF_SPARC_EXT_MASK = 0xFFFF00 # macro
EF_SPARC_32PLUS = 0x000100 # macro
EF_SPARC_SUN_US1 = 0x000200 # macro
EF_SPARC_HAL_R1 = 0x000400 # macro
EF_SPARC_SUN_US3 = 0x000800 # macro
R_SPARC_NONE = 0 # macro
R_SPARC_8 = 1 # macro
R_SPARC_16 = 2 # macro
R_SPARC_32 = 3 # macro
R_SPARC_DISP8 = 4 # macro
R_SPARC_DISP16 = 5 # macro
R_SPARC_DISP32 = 6 # macro
R_SPARC_WDISP30 = 7 # macro
R_SPARC_WDISP22 = 8 # macro
R_SPARC_HI22 = 9 # macro
R_SPARC_22 = 10 # macro
R_SPARC_13 = 11 # macro
R_SPARC_LO10 = 12 # macro
R_SPARC_GOT10 = 13 # macro
R_SPARC_GOT13 = 14 # macro
R_SPARC_GOT22 = 15 # macro
R_SPARC_PC10 = 16 # macro
R_SPARC_PC22 = 17 # macro
R_SPARC_WPLT30 = 18 # macro
R_SPARC_COPY = 19 # macro
R_SPARC_GLOB_DAT = 20 # macro
R_SPARC_JMP_SLOT = 21 # macro
R_SPARC_RELATIVE = 22 # macro
R_SPARC_UA32 = 23 # macro
R_SPARC_PLT32 = 24 # macro
R_SPARC_HIPLT22 = 25 # macro
R_SPARC_LOPLT10 = 26 # macro
R_SPARC_PCPLT32 = 27 # macro
R_SPARC_PCPLT22 = 28 # macro
R_SPARC_PCPLT10 = 29 # macro
R_SPARC_10 = 30 # macro
R_SPARC_11 = 31 # macro
R_SPARC_64 = 32 # macro
R_SPARC_OLO10 = 33 # macro
R_SPARC_HH22 = 34 # macro
R_SPARC_HM10 = 35 # macro
R_SPARC_LM22 = 36 # macro
R_SPARC_PC_HH22 = 37 # macro
R_SPARC_PC_HM10 = 38 # macro
R_SPARC_PC_LM22 = 39 # macro
R_SPARC_WDISP16 = 40 # macro
R_SPARC_WDISP19 = 41 # macro
R_SPARC_GLOB_JMP = 42 # macro
R_SPARC_7 = 43 # macro
R_SPARC_5 = 44 # macro
R_SPARC_6 = 45 # macro
R_SPARC_DISP64 = 46 # macro
R_SPARC_PLT64 = 47 # macro
R_SPARC_HIX22 = 48 # macro
R_SPARC_LOX10 = 49 # macro
R_SPARC_H44 = 50 # macro
R_SPARC_M44 = 51 # macro
R_SPARC_L44 = 52 # macro
R_SPARC_REGISTER = 53 # macro
R_SPARC_UA64 = 54 # macro
R_SPARC_UA16 = 55 # macro
R_SPARC_TLS_GD_HI22 = 56 # macro
R_SPARC_TLS_GD_LO10 = 57 # macro
R_SPARC_TLS_GD_ADD = 58 # macro
R_SPARC_TLS_GD_CALL = 59 # macro
R_SPARC_TLS_LDM_HI22 = 60 # macro
R_SPARC_TLS_LDM_LO10 = 61 # macro
R_SPARC_TLS_LDM_ADD = 62 # macro
R_SPARC_TLS_LDM_CALL = 63 # macro
R_SPARC_TLS_LDO_HIX22 = 64 # macro
R_SPARC_TLS_LDO_LOX10 = 65 # macro
R_SPARC_TLS_LDO_ADD = 66 # macro
R_SPARC_TLS_IE_HI22 = 67 # macro
R_SPARC_TLS_IE_LO10 = 68 # macro
R_SPARC_TLS_IE_LD = 69 # macro
R_SPARC_TLS_IE_LDX = 70 # macro
R_SPARC_TLS_IE_ADD = 71 # macro
R_SPARC_TLS_LE_HIX22 = 72 # macro
R_SPARC_TLS_LE_LOX10 = 73 # macro
R_SPARC_TLS_DTPMOD32 = 74 # macro
R_SPARC_TLS_DTPMOD64 = 75 # macro
R_SPARC_TLS_DTPOFF32 = 76 # macro
R_SPARC_TLS_DTPOFF64 = 77 # macro
R_SPARC_TLS_TPOFF32 = 78 # macro
R_SPARC_TLS_TPOFF64 = 79 # macro
R_SPARC_GOTDATA_HIX22 = 80 # macro
R_SPARC_GOTDATA_LOX10 = 81 # macro
R_SPARC_GOTDATA_OP_HIX22 = 82 # macro
R_SPARC_GOTDATA_OP_LOX10 = 83 # macro
R_SPARC_GOTDATA_OP = 84 # macro
R_SPARC_H34 = 85 # macro
R_SPARC_SIZE32 = 86 # macro
R_SPARC_SIZE64 = 87 # macro
R_SPARC_WDISP10 = 88 # macro
R_SPARC_JMP_IREL = 248 # macro
R_SPARC_IRELATIVE = 249 # macro
R_SPARC_GNU_VTINHERIT = 250 # macro
R_SPARC_GNU_VTENTRY = 251 # macro
R_SPARC_REV32 = 252 # macro
R_SPARC_NUM = 253 # macro
DT_SPARC_REGISTER = 0x70000001 # macro
DT_SPARC_NUM = 2 # macro
EF_MIPS_NOREORDER = 1 # macro
EF_MIPS_PIC = 2 # macro
EF_MIPS_CPIC = 4 # macro
EF_MIPS_XGOT = 8 # macro
EF_MIPS_UCODE = 16 # macro
EF_MIPS_ABI2 = 32 # macro
EF_MIPS_ABI_ON32 = 64 # macro
EF_MIPS_OPTIONS_FIRST = 0x00000080 # macro
EF_MIPS_32BITMODE = 0x00000100 # macro
EF_MIPS_FP64 = 512 # macro
EF_MIPS_NAN2008 = 1024 # macro
EF_MIPS_ARCH_ASE = 0x0f000000 # macro
EF_MIPS_ARCH_ASE_MDMX = 0x08000000 # macro
EF_MIPS_ARCH_ASE_M16 = 0x04000000 # macro
EF_MIPS_ARCH_ASE_MICROMIPS = 0x02000000 # macro
EF_MIPS_ARCH = 0xf0000000 # macro
EF_MIPS_ARCH_1 = 0x00000000 # macro
EF_MIPS_ARCH_2 = 0x10000000 # macro
EF_MIPS_ARCH_3 = 0x20000000 # macro
EF_MIPS_ARCH_4 = 0x30000000 # macro
EF_MIPS_ARCH_5 = 0x40000000 # macro
EF_MIPS_ARCH_32 = 0x50000000 # macro
EF_MIPS_ARCH_64 = 0x60000000 # macro
EF_MIPS_ARCH_32R2 = 0x70000000 # macro
EF_MIPS_ARCH_64R2 = 0x80000000 # macro
EF_MIPS_ARCH_32R6 = 0x90000000 # macro
EF_MIPS_ARCH_64R6 = 0xa0000000 # macro
EF_MIPS_ABI = 0x0000F000 # macro
EF_MIPS_ABI_O32 = 0x00001000 # macro
EF_MIPS_ABI_O64 = 0x00002000 # macro
EF_MIPS_ABI_EABI32 = 0x00003000 # macro
EF_MIPS_ABI_EABI64 = 0x00004000 # macro
EF_MIPS_MACH = 0x00FF0000 # macro
EF_MIPS_MACH_3900 = 0x00810000 # macro
EF_MIPS_MACH_4010 = 0x00820000 # macro
EF_MIPS_MACH_4100 = 0x00830000 # macro
EF_MIPS_MACH_ALLEGREX = 0x00840000 # macro
EF_MIPS_MACH_4650 = 0x00850000 # macro
EF_MIPS_MACH_4120 = 0x00870000 # macro
EF_MIPS_MACH_4111 = 0x00880000 # macro
EF_MIPS_MACH_SB1 = 0x008a0000 # macro
EF_MIPS_MACH_OCTEON = 0x008b0000 # macro
EF_MIPS_MACH_XLR = 0x008c0000 # macro
EF_MIPS_MACH_OCTEON2 = 0x008d0000 # macro
EF_MIPS_MACH_OCTEON3 = 0x008e0000 # macro
EF_MIPS_MACH_5400 = 0x00910000 # macro
EF_MIPS_MACH_5900 = 0x00920000 # macro
EF_MIPS_MACH_IAMR2 = 0x00930000 # macro
EF_MIPS_MACH_5500 = 0x00980000 # macro
EF_MIPS_MACH_9000 = 0x00990000 # macro
EF_MIPS_MACH_LS2E = 0x00A00000 # macro
EF_MIPS_MACH_LS2F = 0x00A10000 # macro
EF_MIPS_MACH_GS464 = 0x00A20000 # macro
EF_MIPS_MACH_GS464E = 0x00A30000 # macro
EF_MIPS_MACH_GS264E = 0x00A40000 # macro
E_MIPS_ARCH_1 = 0x00000000 # macro
E_MIPS_ARCH_2 = 0x10000000 # macro
E_MIPS_ARCH_3 = 0x20000000 # macro
E_MIPS_ARCH_4 = 0x30000000 # macro
E_MIPS_ARCH_5 = 0x40000000 # macro
E_MIPS_ARCH_32 = 0x50000000 # macro
E_MIPS_ARCH_64 = 0x60000000 # macro
SHN_MIPS_ACOMMON = 0xff00 # macro
SHN_MIPS_TEXT = 0xff01 # macro
SHN_MIPS_DATA = 0xff02 # macro
SHN_MIPS_SCOMMON = 0xff03 # macro
SHN_MIPS_SUNDEFINED = 0xff04 # macro
SHT_MIPS_LIBLIST = 0x70000000 # macro
SHT_MIPS_MSYM = 0x70000001 # macro
SHT_MIPS_CONFLICT = 0x70000002 # macro
SHT_MIPS_GPTAB = 0x70000003 # macro
SHT_MIPS_UCODE = 0x70000004 # macro
SHT_MIPS_DEBUG = 0x70000005 # macro
SHT_MIPS_REGINFO = 0x70000006 # macro
SHT_MIPS_PACKAGE = 0x70000007 # macro
SHT_MIPS_PACKSYM = 0x70000008 # macro
SHT_MIPS_RELD = 0x70000009 # macro
SHT_MIPS_IFACE = 0x7000000b # macro
SHT_MIPS_CONTENT = 0x7000000c # macro
SHT_MIPS_OPTIONS = 0x7000000d # macro
SHT_MIPS_SHDR = 0x70000010 # macro
SHT_MIPS_FDESC = 0x70000011 # macro
SHT_MIPS_EXTSYM = 0x70000012 # macro
SHT_MIPS_DENSE = 0x70000013 # macro
SHT_MIPS_PDESC = 0x70000014 # macro
SHT_MIPS_LOCSYM = 0x70000015 # macro
SHT_MIPS_AUXSYM = 0x70000016 # macro
SHT_MIPS_OPTSYM = 0x70000017 # macro
SHT_MIPS_LOCSTR = 0x70000018 # macro
SHT_MIPS_LINE = 0x70000019 # macro
SHT_MIPS_RFDESC = 0x7000001a # macro
SHT_MIPS_DELTASYM = 0x7000001b # macro
SHT_MIPS_DELTAINST = 0x7000001c # macro
SHT_MIPS_DELTACLASS = 0x7000001d # macro
SHT_MIPS_DWARF = 0x7000001e # macro
SHT_MIPS_DELTADECL = 0x7000001f # macro
SHT_MIPS_SYMBOL_LIB = 0x70000020 # macro
SHT_MIPS_EVENTS = 0x70000021 # macro
SHT_MIPS_TRANSLATE = 0x70000022 # macro
SHT_MIPS_PIXIE = 0x70000023 # macro
SHT_MIPS_XLATE = 0x70000024 # macro
SHT_MIPS_XLATE_DEBUG = 0x70000025 # macro
SHT_MIPS_WHIRL = 0x70000026 # macro
SHT_MIPS_EH_REGION = 0x70000027 # macro
SHT_MIPS_XLATE_OLD = 0x70000028 # macro
SHT_MIPS_PDR_EXCEPTION = 0x70000029 # macro
SHT_MIPS_ABIFLAGS = 0x7000002a # macro
SHT_MIPS_XHASH = 0x7000002b # macro
SHF_MIPS_GPREL = 0x10000000 # macro
SHF_MIPS_MERGE = 0x20000000 # macro
SHF_MIPS_ADDR = 0x40000000 # macro
SHF_MIPS_STRINGS = 0x80000000 # macro
SHF_MIPS_NOSTRIP = 0x08000000 # macro
SHF_MIPS_LOCAL = 0x04000000 # macro
SHF_MIPS_NAMES = 0x02000000 # macro
SHF_MIPS_NODUPE = 0x01000000 # macro
STO_MIPS_DEFAULT = 0x0 # macro
STO_MIPS_INTERNAL = 0x1 # macro
STO_MIPS_HIDDEN = 0x2 # macro
STO_MIPS_PROTECTED = 0x3 # macro
STO_MIPS_PLT = 0x8 # macro
STO_MIPS_SC_ALIGN_UNUSED = 0xff # macro
STB_MIPS_SPLIT_COMMON = 13 # macro
ODK_NULL = 0 # macro
ODK_REGINFO = 1 # macro
ODK_EXCEPTIONS = 2 # macro
ODK_PAD = 3 # macro
ODK_HWPATCH = 4 # macro
ODK_FILL = 5 # macro
ODK_TAGS = 6 # macro
ODK_HWAND = 7 # macro
ODK_HWOR = 8 # macro
OEX_FPU_MIN = 0x1f # macro
OEX_FPU_MAX = 0x1f00 # macro
OEX_PAGE0 = 0x10000 # macro
OEX_SMM = 0x20000 # macro
OEX_FPDBUG = 0x40000 # macro
OEX_PRECISEFP = 0x40000 # macro
OEX_DISMISS = 0x80000 # macro
OEX_FPU_INVAL = 0x10 # macro
OEX_FPU_DIV0 = 0x08 # macro
OEX_FPU_OFLO = 0x04 # macro
OEX_FPU_UFLO = 0x02 # macro
OEX_FPU_INEX = 0x01 # macro
OHW_R4KEOP = 0x1 # macro
OHW_R8KPFETCH = 0x2 # macro
OHW_R5KEOP = 0x4 # macro
OHW_R5KCVTL = 0x8 # macro
OPAD_PREFIX = 0x1 # macro
OPAD_POSTFIX = 0x2 # macro
OPAD_SYMBOL = 0x4 # macro
OHWA0_R4KEOP_CHECKED = 0x00000001 # macro
OHWA1_R4KEOP_CLEAN = 0x00000002 # macro
R_MIPS_NONE = 0 # macro
R_MIPS_16 = 1 # macro
R_MIPS_32 = 2 # macro
R_MIPS_REL32 = 3 # macro
R_MIPS_26 = 4 # macro
R_MIPS_HI16 = 5 # macro
R_MIPS_LO16 = 6 # macro
R_MIPS_GPREL16 = 7 # macro
R_MIPS_LITERAL = 8 # macro
R_MIPS_GOT16 = 9 # macro
R_MIPS_PC16 = 10 # macro
R_MIPS_CALL16 = 11 # macro
R_MIPS_GPREL32 = 12 # macro
R_MIPS_SHIFT5 = 16 # macro
R_MIPS_SHIFT6 = 17 # macro
R_MIPS_64 = 18 # macro
R_MIPS_GOT_DISP = 19 # macro
R_MIPS_GOT_PAGE = 20 # macro
R_MIPS_GOT_OFST = 21 # macro
R_MIPS_GOT_HI16 = 22 # macro
R_MIPS_GOT_LO16 = 23 # macro
R_MIPS_SUB = 24 # macro
R_MIPS_INSERT_A = 25 # macro
R_MIPS_INSERT_B = 26 # macro
R_MIPS_DELETE = 27 # macro
R_MIPS_HIGHER = 28 # macro
R_MIPS_HIGHEST = 29 # macro
R_MIPS_CALL_HI16 = 30 # macro
R_MIPS_CALL_LO16 = 31 # macro
R_MIPS_SCN_DISP = 32 # macro
R_MIPS_REL16 = 33 # macro
R_MIPS_ADD_IMMEDIATE = 34 # macro
R_MIPS_PJUMP = 35 # macro
R_MIPS_RELGOT = 36 # macro
R_MIPS_JALR = 37 # macro
R_MIPS_TLS_DTPMOD32 = 38 # macro
R_MIPS_TLS_DTPREL32 = 39 # macro
R_MIPS_TLS_DTPMOD64 = 40 # macro
R_MIPS_TLS_DTPREL64 = 41 # macro
R_MIPS_TLS_GD = 42 # macro
R_MIPS_TLS_LDM = 43 # macro
R_MIPS_TLS_DTPREL_HI16 = 44 # macro
R_MIPS_TLS_DTPREL_LO16 = 45 # macro
R_MIPS_TLS_GOTTPREL = 46 # macro
R_MIPS_TLS_TPREL32 = 47 # macro
R_MIPS_TLS_TPREL64 = 48 # macro
R_MIPS_TLS_TPREL_HI16 = 49 # macro
R_MIPS_TLS_TPREL_LO16 = 50 # macro
R_MIPS_GLOB_DAT = 51 # macro
R_MIPS_PC21_S2 = 60 # macro
R_MIPS_PC26_S2 = 61 # macro
R_MIPS_PC18_S3 = 62 # macro
R_MIPS_PC19_S2 = 63 # macro
R_MIPS_PCHI16 = 64 # macro
R_MIPS_PCLO16 = 65 # macro
R_MIPS16_26 = 100 # macro
R_MIPS16_GPREL = 101 # macro
R_MIPS16_GOT16 = 102 # macro
R_MIPS16_CALL16 = 103 # macro
R_MIPS16_HI16 = 104 # macro
R_MIPS16_LO16 = 105 # macro
R_MIPS16_TLS_GD = 106 # macro
R_MIPS16_TLS_LDM = 107 # macro
R_MIPS16_TLS_DTPREL_HI16 = 108 # macro
R_MIPS16_TLS_DTPREL_LO16 = 109 # macro
R_MIPS16_TLS_GOTTPREL = 110 # macro
R_MIPS16_TLS_TPREL_HI16 = 111 # macro
R_MIPS16_TLS_TPREL_LO16 = 112 # macro
R_MIPS16_PC16_S1 = 113 # macro
R_MIPS_COPY = 126 # macro
R_MIPS_JUMP_SLOT = 127 # macro
R_MIPS_RELATIVE = 128 # macro
R_MICROMIPS_26_S1 = 133 # macro
R_MICROMIPS_HI16 = 134 # macro
R_MICROMIPS_LO16 = 135 # macro
R_MICROMIPS_GPREL16 = 136 # macro
R_MICROMIPS_LITERAL = 137 # macro
R_MICROMIPS_GOT16 = 138 # macro
R_MICROMIPS_PC7_S1 = 139 # macro
R_MICROMIPS_PC10_S1 = 140 # macro
R_MICROMIPS_PC16_S1 = 141 # macro
R_MICROMIPS_CALL16 = 142 # macro
R_MICROMIPS_GOT_DISP = 145 # macro
R_MICROMIPS_GOT_PAGE = 146 # macro
R_MICROMIPS_GOT_OFST = 147 # macro
R_MICROMIPS_GOT_HI16 = 148 # macro
R_MICROMIPS_GOT_LO16 = 149 # macro
R_MICROMIPS_SUB = 150 # macro
R_MICROMIPS_HIGHER = 151 # macro
R_MICROMIPS_HIGHEST = 152 # macro
R_MICROMIPS_CALL_HI16 = 153 # macro
R_MICROMIPS_CALL_LO16 = 154 # macro
R_MICROMIPS_SCN_DISP = 155 # macro
R_MICROMIPS_JALR = 156 # macro
R_MICROMIPS_HI0_LO16 = 157 # macro
R_MICROMIPS_TLS_GD = 162 # macro
R_MICROMIPS_TLS_LDM = 163 # macro
R_MICROMIPS_TLS_DTPREL_HI16 = 164 # macro
R_MICROMIPS_TLS_DTPREL_LO16 = 165 # macro
R_MICROMIPS_TLS_GOTTPREL = 166 # macro
R_MICROMIPS_TLS_TPREL_HI16 = 169 # macro
R_MICROMIPS_TLS_TPREL_LO16 = 170 # macro
R_MICROMIPS_GPREL7_S2 = 172 # macro
R_MICROMIPS_PC23_S2 = 173 # macro
R_MIPS_PC32 = 248 # macro
R_MIPS_EH = 249 # macro
R_MIPS_GNU_REL16_S2 = 250 # macro
R_MIPS_GNU_VTINHERIT = 253 # macro
R_MIPS_GNU_VTENTRY = 254 # macro
R_MIPS_NUM = 255 # macro
PT_MIPS_REGINFO = 0x70000000 # macro
PT_MIPS_RTPROC = 0x70000001 # macro
PT_MIPS_OPTIONS = 0x70000002 # macro
PT_MIPS_ABIFLAGS = 0x70000003 # macro
PF_MIPS_LOCAL = 0x10000000 # macro
DT_MIPS_RLD_VERSION = 0x70000001 # macro
DT_MIPS_TIME_STAMP = 0x70000002 # macro
DT_MIPS_ICHECKSUM = 0x70000003 # macro
DT_MIPS_IVERSION = 0x70000004 # macro
DT_MIPS_FLAGS = 0x70000005 # macro
DT_MIPS_BASE_ADDRESS = 0x70000006 # macro
DT_MIPS_MSYM = 0x70000007 # macro
DT_MIPS_CONFLICT = 0x70000008 # macro
DT_MIPS_LIBLIST = 0x70000009 # macro
DT_MIPS_LOCAL_GOTNO = 0x7000000a # macro
DT_MIPS_CONFLICTNO = 0x7000000b # macro
DT_MIPS_LIBLISTNO = 0x70000010 # macro
DT_MIPS_SYMTABNO = 0x70000011 # macro
DT_MIPS_UNREFEXTNO = 0x70000012 # macro
DT_MIPS_GOTSYM = 0x70000013 # macro
DT_MIPS_HIPAGENO = 0x70000014 # macro
DT_MIPS_RLD_MAP = 0x70000016 # macro
DT_MIPS_DELTA_CLASS = 0x70000017 # macro
DT_MIPS_DELTA_CLASS_NO = 0x70000018 # macro
DT_MIPS_DELTA_INSTANCE = 0x70000019 # macro
DT_MIPS_DELTA_INSTANCE_NO = 0x7000001a # macro
DT_MIPS_DELTA_RELOC = 0x7000001b # macro
DT_MIPS_DELTA_RELOC_NO = 0x7000001c # macro
DT_MIPS_DELTA_SYM = 0x7000001d # macro
DT_MIPS_DELTA_SYM_NO = 0x7000001e # macro
DT_MIPS_DELTA_CLASSSYM = 0x70000020 # macro
DT_MIPS_DELTA_CLASSSYM_NO = 0x70000021 # macro
DT_MIPS_CXX_FLAGS = 0x70000022 # macro
DT_MIPS_PIXIE_INIT = 0x70000023 # macro
DT_MIPS_SYMBOL_LIB = 0x70000024 # macro
DT_MIPS_LOCALPAGE_GOTIDX = 0x70000025 # macro
DT_MIPS_LOCAL_GOTIDX = 0x70000026 # macro
DT_MIPS_HIDDEN_GOTIDX = 0x70000027 # macro
DT_MIPS_PROTECTED_GOTIDX = 0x70000028 # macro
DT_MIPS_OPTIONS = 0x70000029 # macro
DT_MIPS_INTERFACE = 0x7000002a # macro
DT_MIPS_DYNSTR_ALIGN = 0x7000002b # macro
DT_MIPS_INTERFACE_SIZE = 0x7000002c # macro
DT_MIPS_RLD_TEXT_RESOLVE_ADDR = 0x7000002d # macro
DT_MIPS_PERF_SUFFIX = 0x7000002e # macro
DT_MIPS_COMPACT_SIZE = 0x7000002f # macro
DT_MIPS_GP_VALUE = 0x70000030 # macro
DT_MIPS_AUX_DYNAMIC = 0x70000031 # macro
DT_MIPS_PLTGOT = 0x70000032 # macro
DT_MIPS_RWPLT = 0x70000034 # macro
DT_MIPS_RLD_MAP_REL = 0x70000035 # macro
DT_MIPS_XHASH = 0x70000036 # macro
DT_MIPS_NUM = 0x37 # macro
DT_PROCNUM = DT_MIPS_NUM # macro
RHF_NONE = 0 # macro
RHF_QUICKSTART = (1<<0) # macro
RHF_NOTPOT = (1<<1) # macro
RHF_NO_LIBRARY_REPLACEMENT = (1<<2) # macro
RHF_NO_MOVE = (1<<3) # macro
RHF_SGI_ONLY = (1<<4) # macro
RHF_GUARANTEE_INIT = (1<<5) # macro
RHF_DELTA_C_PLUS_PLUS = (1<<6) # macro
RHF_GUARANTEE_START_INIT = (1<<7) # macro
RHF_PIXIE = (1<<8) # macro
RHF_DEFAULT_DELAY_LOAD = (1<<9) # macro
RHF_REQUICKSTART = (1<<10) # macro
RHF_REQUICKSTARTED = (1<<11) # macro
RHF_CORD = (1<<12) # macro
RHF_NO_UNRES_UNDEF = (1<<13) # macro
RHF_RLD_ORDER_SAFE = (1<<14) # macro
LL_NONE = 0 # macro
LL_EXACT_MATCH = (1<<0) # macro
LL_IGNORE_INT_VER = (1<<1) # macro
LL_REQUIRE_MINOR = (1<<2) # macro
LL_EXPORTS = (1<<3) # macro
LL_DELAY_LOAD = (1<<4) # macro
LL_DELTA = (1<<5) # macro
MIPS_AFL_REG_NONE = 0x00 # macro
MIPS_AFL_REG_32 = 0x01 # macro
MIPS_AFL_REG_64 = 0x02 # macro
MIPS_AFL_REG_128 = 0x03 # macro
MIPS_AFL_ASE_DSP = 0x00000001 # macro
MIPS_AFL_ASE_DSPR2 = 0x00000002 # macro
MIPS_AFL_ASE_EVA = 0x00000004 # macro
MIPS_AFL_ASE_MCU = 0x00000008 # macro
MIPS_AFL_ASE_MDMX = 0x00000010 # macro
MIPS_AFL_ASE_MIPS3D = 0x00000020 # macro
MIPS_AFL_ASE_MT = 0x00000040 # macro
MIPS_AFL_ASE_SMARTMIPS = 0x00000080 # macro
MIPS_AFL_ASE_VIRT = 0x00000100 # macro
MIPS_AFL_ASE_MSA = 0x00000200 # macro
MIPS_AFL_ASE_MIPS16 = 0x00000400 # macro
MIPS_AFL_ASE_MICROMIPS = 0x00000800 # macro
MIPS_AFL_ASE_XPA = 0x00001000 # macro
MIPS_AFL_ASE_MASK = 0x00001fff # macro
MIPS_AFL_EXT_XLR = 1 # macro
MIPS_AFL_EXT_OCTEON2 = 2 # macro
MIPS_AFL_EXT_OCTEONP = 3 # macro
MIPS_AFL_EXT_LOONGSON_3A = 4 # macro
MIPS_AFL_EXT_OCTEON = 5 # macro
MIPS_AFL_EXT_5900 = 6 # macro
MIPS_AFL_EXT_4650 = 7 # macro
MIPS_AFL_EXT_4010 = 8 # macro
MIPS_AFL_EXT_4100 = 9 # macro
MIPS_AFL_EXT_3900 = 10 # macro
MIPS_AFL_EXT_10000 = 11 # macro
MIPS_AFL_EXT_SB1 = 12 # macro
MIPS_AFL_EXT_4111 = 13 # macro
MIPS_AFL_EXT_4120 = 14 # macro
MIPS_AFL_EXT_5400 = 15 # macro
MIPS_AFL_EXT_5500 = 16 # macro
MIPS_AFL_EXT_LOONGSON_2E = 17 # macro
MIPS_AFL_EXT_LOONGSON_2F = 18 # macro
MIPS_AFL_FLAGS1_ODDSPREG = 1 # macro
EF_PARISC_TRAPNIL = 0x00010000 # macro
EF_PARISC_EXT = 0x00020000 # macro
EF_PARISC_LSB = 0x00040000 # macro
EF_PARISC_WIDE = 0x00080000 # macro
EF_PARISC_NO_KABP = 0x00100000 # macro
EF_PARISC_LAZYSWAP = 0x00400000 # macro
EF_PARISC_ARCH = 0x0000ffff # macro
EFA_PARISC_1_0 = 0x020b # macro
EFA_PARISC_1_1 = 0x0210 # macro
EFA_PARISC_2_0 = 0x0214 # macro
SHN_PARISC_ANSI_COMMON = 0xff00 # macro
SHN_PARISC_HUGE_COMMON = 0xff01 # macro
SHT_PARISC_EXT = 0x70000000 # macro
SHT_PARISC_UNWIND = 0x70000001 # macro
SHT_PARISC_DOC = 0x70000002 # macro
SHF_PARISC_SHORT = 0x20000000 # macro
SHF_PARISC_HUGE = 0x40000000 # macro
SHF_PARISC_SBP = 0x80000000 # macro
STT_PARISC_MILLICODE = 13 # macro
STT_HP_OPAQUE = (10+0x1) # macro
STT_HP_STUB = (10+0x2) # macro
R_PARISC_NONE = 0 # macro
R_PARISC_DIR32 = 1 # macro
R_PARISC_DIR21L = 2 # macro
R_PARISC_DIR17R = 3 # macro
R_PARISC_DIR17F = 4 # macro
R_PARISC_DIR14R = 6 # macro
R_PARISC_PCREL32 = 9 # macro
R_PARISC_PCREL21L = 10 # macro
R_PARISC_PCREL17R = 11 # macro
R_PARISC_PCREL17F = 12 # macro
R_PARISC_PCREL14R = 14 # macro
R_PARISC_DPREL21L = 18 # macro
R_PARISC_DPREL14R = 22 # macro
R_PARISC_GPREL21L = 26 # macro
R_PARISC_GPREL14R = 30 # macro
R_PARISC_LTOFF21L = 34 # macro
R_PARISC_LTOFF14R = 38 # macro
R_PARISC_SECREL32 = 41 # macro
R_PARISC_SEGBASE = 48 # macro
R_PARISC_SEGREL32 = 49 # macro
R_PARISC_PLTOFF21L = 50 # macro
R_PARISC_PLTOFF14R = 54 # macro
R_PARISC_LTOFF_FPTR32 = 57 # macro
R_PARISC_LTOFF_FPTR21L = 58 # macro
R_PARISC_LTOFF_FPTR14R = 62 # macro
R_PARISC_FPTR64 = 64 # macro
R_PARISC_PLABEL32 = 65 # macro
R_PARISC_PLABEL21L = 66 # macro
R_PARISC_PLABEL14R = 70 # macro
R_PARISC_PCREL64 = 72 # macro
R_PARISC_PCREL22F = 74 # macro
R_PARISC_PCREL14WR = 75 # macro
R_PARISC_PCREL14DR = 76 # macro
R_PARISC_PCREL16F = 77 # macro
R_PARISC_PCREL16WF = 78 # macro
R_PARISC_PCREL16DF = 79 # macro
R_PARISC_DIR64 = 80 # macro
R_PARISC_DIR14WR = 83 # macro
R_PARISC_DIR14DR = 84 # macro
R_PARISC_DIR16F = 85 # macro
R_PARISC_DIR16WF = 86 # macro
R_PARISC_DIR16DF = 87 # macro
R_PARISC_GPREL64 = 88 # macro
R_PARISC_GPREL14WR = 91 # macro
R_PARISC_GPREL14DR = 92 # macro
R_PARISC_GPREL16F = 93 # macro
R_PARISC_GPREL16WF = 94 # macro
R_PARISC_GPREL16DF = 95 # macro
R_PARISC_LTOFF64 = 96 # macro
R_PARISC_LTOFF14WR = 99 # macro
R_PARISC_LTOFF14DR = 100 # macro
R_PARISC_LTOFF16F = 101 # macro
R_PARISC_LTOFF16WF = 102 # macro
R_PARISC_LTOFF16DF = 103 # macro
R_PARISC_SECREL64 = 104 # macro
R_PARISC_SEGREL64 = 112 # macro
R_PARISC_PLTOFF14WR = 115 # macro
R_PARISC_PLTOFF14DR = 116 # macro
R_PARISC_PLTOFF16F = 117 # macro
R_PARISC_PLTOFF16WF = 118 # macro
R_PARISC_PLTOFF16DF = 119 # macro
R_PARISC_LTOFF_FPTR64 = 120 # macro
R_PARISC_LTOFF_FPTR14WR = 123 # macro
R_PARISC_LTOFF_FPTR14DR = 124 # macro
R_PARISC_LTOFF_FPTR16F = 125 # macro
R_PARISC_LTOFF_FPTR16WF = 126 # macro
R_PARISC_LTOFF_FPTR16DF = 127 # macro
R_PARISC_LORESERVE = 128 # macro
R_PARISC_COPY = 128 # macro
R_PARISC_IPLT = 129 # macro
R_PARISC_EPLT = 130 # macro
R_PARISC_TPREL32 = 153 # macro
R_PARISC_TPREL21L = 154 # macro
R_PARISC_TPREL14R = 158 # macro
R_PARISC_LTOFF_TP21L = 162 # macro
R_PARISC_LTOFF_TP14R = 166 # macro
R_PARISC_LTOFF_TP14F = 167 # macro
R_PARISC_TPREL64 = 216 # macro
R_PARISC_TPREL14WR = 219 # macro
R_PARISC_TPREL14DR = 220 # macro
R_PARISC_TPREL16F = 221 # macro
R_PARISC_TPREL16WF = 222 # macro
R_PARISC_TPREL16DF = 223 # macro
R_PARISC_LTOFF_TP64 = 224 # macro
R_PARISC_LTOFF_TP14WR = 227 # macro
R_PARISC_LTOFF_TP14DR = 228 # macro
R_PARISC_LTOFF_TP16F = 229 # macro
R_PARISC_LTOFF_TP16WF = 230 # macro
R_PARISC_LTOFF_TP16DF = 231 # macro
R_PARISC_GNU_VTENTRY = 232 # macro
R_PARISC_GNU_VTINHERIT = 233 # macro
R_PARISC_TLS_GD21L = 234 # macro
R_PARISC_TLS_GD14R = 235 # macro
R_PARISC_TLS_GDCALL = 236 # macro
R_PARISC_TLS_LDM21L = 237 # macro
R_PARISC_TLS_LDM14R = 238 # macro
R_PARISC_TLS_LDMCALL = 239 # macro
R_PARISC_TLS_LDO21L = 240 # macro
R_PARISC_TLS_LDO14R = 241 # macro
R_PARISC_TLS_DTPMOD32 = 242 # macro
R_PARISC_TLS_DTPMOD64 = 243 # macro
R_PARISC_TLS_DTPOFF32 = 244 # macro
R_PARISC_TLS_DTPOFF64 = 245 # macro
R_PARISC_TLS_LE21L = 154 # macro
R_PARISC_TLS_LE14R = 158 # macro
R_PARISC_TLS_IE21L = 162 # macro
R_PARISC_TLS_IE14R = 166 # macro
R_PARISC_TLS_TPREL32 = 153 # macro
R_PARISC_TLS_TPREL64 = 216 # macro
R_PARISC_HIRESERVE = 255 # macro
PT_HP_TLS = (0x60000000+0x0) # macro
PT_HP_CORE_NONE = (0x60000000+0x1) # macro
PT_HP_CORE_VERSION = (0x60000000+0x2) # macro
PT_HP_CORE_KERNEL = (0x60000000+0x3) # macro
PT_HP_CORE_COMM = (0x60000000+0x4) # macro
PT_HP_CORE_PROC = (0x60000000+0x5) # macro
PT_HP_CORE_LOADABLE = (0x60000000+0x6) # macro
PT_HP_CORE_STACK = (0x60000000+0x7) # macro
PT_HP_CORE_SHM = (0x60000000+0x8) # macro
PT_HP_CORE_MMF = (0x60000000+0x9) # macro
PT_HP_PARALLEL = (0x60000000+0x10) # macro
PT_HP_FASTBIND = (0x60000000+0x11) # macro
PT_HP_OPT_ANNOT = (0x60000000+0x12) # macro
PT_HP_HSL_ANNOT = (0x60000000+0x13) # macro
PT_HP_STACK = (0x60000000+0x14) # macro
PT_PARISC_ARCHEXT = 0x70000000 # macro
PT_PARISC_UNWIND = 0x70000001 # macro
PF_PARISC_SBP = 0x08000000 # macro
PF_HP_PAGE_SIZE = 0x00100000 # macro
PF_HP_FAR_SHARED = 0x00200000 # macro
PF_HP_NEAR_SHARED = 0x00400000 # macro
PF_HP_CODE = 0x01000000 # macro
PF_HP_MODIFY = 0x02000000 # macro
PF_HP_LAZYSWAP = 0x04000000 # macro
PF_HP_SBP = 0x08000000 # macro
EF_ALPHA_32BIT = 1 # macro
EF_ALPHA_CANRELAX = 2 # macro
SHT_ALPHA_DEBUG = 0x70000001 # macro
SHT_ALPHA_REGINFO = 0x70000002 # macro
SHF_ALPHA_GPREL = 0x10000000 # macro
STO_ALPHA_NOPV = 0x80 # macro
STO_ALPHA_STD_GPLOAD = 0x88 # macro
R_ALPHA_NONE = 0 # macro
R_ALPHA_REFLONG = 1 # macro
R_ALPHA_REFQUAD = 2 # macro
R_ALPHA_GPREL32 = 3 # macro
R_ALPHA_LITERAL = 4 # macro
R_ALPHA_LITUSE = 5 # macro
R_ALPHA_GPDISP = 6 # macro
R_ALPHA_BRADDR = 7 # macro
R_ALPHA_HINT = 8 # macro
R_ALPHA_SREL16 = 9 # macro
R_ALPHA_SREL32 = 10 # macro
R_ALPHA_SREL64 = 11 # macro
R_ALPHA_GPRELHIGH = 17 # macro
R_ALPHA_GPRELLOW = 18 # macro
R_ALPHA_GPREL16 = 19 # macro
R_ALPHA_COPY = 24 # macro
R_ALPHA_GLOB_DAT = 25 # macro
R_ALPHA_JMP_SLOT = 26 # macro
R_ALPHA_RELATIVE = 27 # macro
R_ALPHA_TLS_GD_HI = 28 # macro
R_ALPHA_TLSGD = 29 # macro
R_ALPHA_TLS_LDM = 30 # macro
R_ALPHA_DTPMOD64 = 31 # macro
R_ALPHA_GOTDTPREL = 32 # macro
R_ALPHA_DTPREL64 = 33 # macro
R_ALPHA_DTPRELHI = 34 # macro
R_ALPHA_DTPRELLO = 35 # macro
R_ALPHA_DTPREL16 = 36 # macro
R_ALPHA_GOTTPREL = 37 # macro
R_ALPHA_TPREL64 = 38 # macro
R_ALPHA_TPRELHI = 39 # macro
R_ALPHA_TPRELLO = 40 # macro
R_ALPHA_TPREL16 = 41 # macro
R_ALPHA_NUM = 46 # macro
LITUSE_ALPHA_ADDR = 0 # macro
LITUSE_ALPHA_BASE = 1 # macro
LITUSE_ALPHA_BYTOFF = 2 # macro
LITUSE_ALPHA_JSR = 3 # macro
LITUSE_ALPHA_TLS_GD = 4 # macro
LITUSE_ALPHA_TLS_LDM = 5 # macro
DT_ALPHA_PLTRO = (0x70000000+0) # macro
DT_ALPHA_NUM = 1 # macro
EF_PPC_EMB = 0x80000000 # macro
EF_PPC_RELOCATABLE = 0x00010000 # macro
EF_PPC_RELOCATABLE_LIB = 0x00008000 # macro
R_PPC_NONE = 0 # macro
R_PPC_ADDR32 = 1 # macro
R_PPC_ADDR24 = 2 # macro
R_PPC_ADDR16 = 3 # macro
R_PPC_ADDR16_LO = 4 # macro
R_PPC_ADDR16_HI = 5 # macro
R_PPC_ADDR16_HA = 6 # macro
R_PPC_ADDR14 = 7 # macro
R_PPC_ADDR14_BRTAKEN = 8 # macro
R_PPC_ADDR14_BRNTAKEN = 9 # macro
R_PPC_REL24 = 10 # macro
R_PPC_REL14 = 11 # macro
R_PPC_REL14_BRTAKEN = 12 # macro
R_PPC_REL14_BRNTAKEN = 13 # macro
R_PPC_GOT16 = 14 # macro
R_PPC_GOT16_LO = 15 # macro
R_PPC_GOT16_HI = 16 # macro
R_PPC_GOT16_HA = 17 # macro
R_PPC_PLTREL24 = 18 # macro
R_PPC_COPY = 19 # macro
R_PPC_GLOB_DAT = 20 # macro
R_PPC_JMP_SLOT = 21 # macro
R_PPC_RELATIVE = 22 # macro
R_PPC_LOCAL24PC = 23 # macro
R_PPC_UADDR32 = 24 # macro
R_PPC_UADDR16 = 25 # macro
R_PPC_REL32 = 26 # macro
R_PPC_PLT32 = 27 # macro
R_PPC_PLTREL32 = 28 # macro
R_PPC_PLT16_LO = 29 # macro
R_PPC_PLT16_HI = 30 # macro
R_PPC_PLT16_HA = 31 # macro
R_PPC_SDAREL16 = 32 # macro
R_PPC_SECTOFF = 33 # macro
R_PPC_SECTOFF_LO = 34 # macro
R_PPC_SECTOFF_HI = 35 # macro
R_PPC_SECTOFF_HA = 36 # macro
R_PPC_TLS = 67 # macro
R_PPC_DTPMOD32 = 68 # macro
R_PPC_TPREL16 = 69 # macro
R_PPC_TPREL16_LO = 70 # macro
R_PPC_TPREL16_HI = 71 # macro
R_PPC_TPREL16_HA = 72 # macro
R_PPC_TPREL32 = 73 # macro
R_PPC_DTPREL16 = 74 # macro
R_PPC_DTPREL16_LO = 75 # macro
R_PPC_DTPREL16_HI = 76 # macro
R_PPC_DTPREL16_HA = 77 # macro
R_PPC_DTPREL32 = 78 # macro
R_PPC_GOT_TLSGD16 = 79 # macro
R_PPC_GOT_TLSGD16_LO = 80 # macro
R_PPC_GOT_TLSGD16_HI = 81 # macro
R_PPC_GOT_TLSGD16_HA = 82 # macro
R_PPC_GOT_TLSLD16 = 83 # macro
R_PPC_GOT_TLSLD16_LO = 84 # macro
R_PPC_GOT_TLSLD16_HI = 85 # macro
R_PPC_GOT_TLSLD16_HA = 86 # macro
R_PPC_GOT_TPREL16 = 87 # macro
R_PPC_GOT_TPREL16_LO = 88 # macro
R_PPC_GOT_TPREL16_HI = 89 # macro
R_PPC_GOT_TPREL16_HA = 90 # macro
R_PPC_GOT_DTPREL16 = 91 # macro
R_PPC_GOT_DTPREL16_LO = 92 # macro
R_PPC_GOT_DTPREL16_HI = 93 # macro
R_PPC_GOT_DTPREL16_HA = 94 # macro
R_PPC_TLSGD = 95 # macro
R_PPC_TLSLD = 96 # macro
R_PPC_EMB_NADDR32 = 101 # macro
R_PPC_EMB_NADDR16 = 102 # macro
R_PPC_EMB_NADDR16_LO = 103 # macro
R_PPC_EMB_NADDR16_HI = 104 # macro
R_PPC_EMB_NADDR16_HA = 105 # macro
R_PPC_EMB_SDAI16 = 106 # macro
R_PPC_EMB_SDA2I16 = 107 # macro
R_PPC_EMB_SDA2REL = 108 # macro
R_PPC_EMB_SDA21 = 109 # macro
R_PPC_EMB_MRKREF = 110 # macro
R_PPC_EMB_RELSEC16 = 111 # macro
R_PPC_EMB_RELST_LO = 112 # macro
R_PPC_EMB_RELST_HI = 113 # macro
R_PPC_EMB_RELST_HA = 114 # macro
R_PPC_EMB_BIT_FLD = 115 # macro
R_PPC_EMB_RELSDA = 116 # macro
R_PPC_DIAB_SDA21_LO = 180 # macro
R_PPC_DIAB_SDA21_HI = 181 # macro
R_PPC_DIAB_SDA21_HA = 182 # macro
R_PPC_DIAB_RELSDA_LO = 183 # macro
R_PPC_DIAB_RELSDA_HI = 184 # macro
R_PPC_DIAB_RELSDA_HA = 185 # macro
R_PPC_IRELATIVE = 248 # macro
R_PPC_REL16 = 249 # macro
R_PPC_REL16_LO = 250 # macro
R_PPC_REL16_HI = 251 # macro
R_PPC_REL16_HA = 252 # macro
R_PPC_TOC16 = 255 # macro
DT_PPC_GOT = (0x70000000+0) # macro
DT_PPC_OPT = (0x70000000+1) # macro
DT_PPC_NUM = 2 # macro
PPC_OPT_TLS = 1 # macro
R_PPC64_NONE = 0 # macro
R_PPC64_ADDR32 = 1 # macro
R_PPC64_ADDR24 = 2 # macro
R_PPC64_ADDR16 = 3 # macro
R_PPC64_ADDR16_LO = 4 # macro
R_PPC64_ADDR16_HI = 5 # macro
R_PPC64_ADDR16_HA = 6 # macro
R_PPC64_ADDR14 = 7 # macro
R_PPC64_ADDR14_BRTAKEN = 8 # macro
R_PPC64_ADDR14_BRNTAKEN = 9 # macro
R_PPC64_REL24 = 10 # macro
R_PPC64_REL14 = 11 # macro
R_PPC64_REL14_BRTAKEN = 12 # macro
R_PPC64_REL14_BRNTAKEN = 13 # macro
R_PPC64_GOT16 = 14 # macro
R_PPC64_GOT16_LO = 15 # macro
R_PPC64_GOT16_HI = 16 # macro
R_PPC64_GOT16_HA = 17 # macro
R_PPC64_COPY = 19 # macro
R_PPC64_GLOB_DAT = 20 # macro
R_PPC64_JMP_SLOT = 21 # macro
R_PPC64_RELATIVE = 22 # macro
R_PPC64_UADDR32 = 24 # macro
R_PPC64_UADDR16 = 25 # macro
R_PPC64_REL32 = 26 # macro
R_PPC64_PLT32 = 27 # macro
R_PPC64_PLTREL32 = 28 # macro
R_PPC64_PLT16_LO = 29 # macro
R_PPC64_PLT16_HI = 30 # macro
R_PPC64_PLT16_HA = 31 # macro
R_PPC64_SECTOFF = 33 # macro
R_PPC64_SECTOFF_LO = 34 # macro
R_PPC64_SECTOFF_HI = 35 # macro
R_PPC64_SECTOFF_HA = 36 # macro
R_PPC64_ADDR30 = 37 # macro
R_PPC64_ADDR64 = 38 # macro
R_PPC64_ADDR16_HIGHER = 39 # macro
R_PPC64_ADDR16_HIGHERA = 40 # macro
R_PPC64_ADDR16_HIGHEST = 41 # macro
R_PPC64_ADDR16_HIGHESTA = 42 # macro
R_PPC64_UADDR64 = 43 # macro
R_PPC64_REL64 = 44 # macro
R_PPC64_PLT64 = 45 # macro
R_PPC64_PLTREL64 = 46 # macro
R_PPC64_TOC16 = 47 # macro
R_PPC64_TOC16_LO = 48 # macro
R_PPC64_TOC16_HI = 49 # macro
R_PPC64_TOC16_HA = 50 # macro
R_PPC64_TOC = 51 # macro
R_PPC64_PLTGOT16 = 52 # macro
R_PPC64_PLTGOT16_LO = 53 # macro
R_PPC64_PLTGOT16_HI = 54 # macro
R_PPC64_PLTGOT16_HA = 55 # macro
R_PPC64_ADDR16_DS = 56 # macro
R_PPC64_ADDR16_LO_DS = 57 # macro
R_PPC64_GOT16_DS = 58 # macro
R_PPC64_GOT16_LO_DS = 59 # macro
R_PPC64_PLT16_LO_DS = 60 # macro
R_PPC64_SECTOFF_DS = 61 # macro
R_PPC64_SECTOFF_LO_DS = 62 # macro
R_PPC64_TOC16_DS = 63 # macro
R_PPC64_TOC16_LO_DS = 64 # macro
R_PPC64_PLTGOT16_DS = 65 # macro
R_PPC64_PLTGOT16_LO_DS = 66 # macro
R_PPC64_TLS = 67 # macro
R_PPC64_DTPMOD64 = 68 # macro
R_PPC64_TPREL16 = 69 # macro
R_PPC64_TPREL16_LO = 70 # macro
R_PPC64_TPREL16_HI = 71 # macro
R_PPC64_TPREL16_HA = 72 # macro
R_PPC64_TPREL64 = 73 # macro
R_PPC64_DTPREL16 = 74 # macro
R_PPC64_DTPREL16_LO = 75 # macro
R_PPC64_DTPREL16_HI = 76 # macro
R_PPC64_DTPREL16_HA = 77 # macro
R_PPC64_DTPREL64 = 78 # macro
R_PPC64_GOT_TLSGD16 = 79 # macro
R_PPC64_GOT_TLSGD16_LO = 80 # macro
R_PPC64_GOT_TLSGD16_HI = 81 # macro
R_PPC64_GOT_TLSGD16_HA = 82 # macro
R_PPC64_GOT_TLSLD16 = 83 # macro
R_PPC64_GOT_TLSLD16_LO = 84 # macro
R_PPC64_GOT_TLSLD16_HI = 85 # macro
R_PPC64_GOT_TLSLD16_HA = 86 # macro
R_PPC64_GOT_TPREL16_DS = 87 # macro
R_PPC64_GOT_TPREL16_LO_DS = 88 # macro
R_PPC64_GOT_TPREL16_HI = 89 # macro
R_PPC64_GOT_TPREL16_HA = 90 # macro
R_PPC64_GOT_DTPREL16_DS = 91 # macro
R_PPC64_GOT_DTPREL16_LO_DS = 92 # macro
R_PPC64_GOT_DTPREL16_HI = 93 # macro
R_PPC64_GOT_DTPREL16_HA = 94 # macro
R_PPC64_TPREL16_DS = 95 # macro
R_PPC64_TPREL16_LO_DS = 96 # macro
R_PPC64_TPREL16_HIGHER = 97 # macro
R_PPC64_TPREL16_HIGHERA = 98 # macro
R_PPC64_TPREL16_HIGHEST = 99 # macro
R_PPC64_TPREL16_HIGHESTA = 100 # macro
R_PPC64_DTPREL16_DS = 101 # macro
R_PPC64_DTPREL16_LO_DS = 102 # macro
R_PPC64_DTPREL16_HIGHER = 103 # macro
R_PPC64_DTPREL16_HIGHERA = 104 # macro
R_PPC64_DTPREL16_HIGHEST = 105 # macro
R_PPC64_DTPREL16_HIGHESTA = 106 # macro
R_PPC64_TLSGD = 107 # macro
R_PPC64_TLSLD = 108 # macro
R_PPC64_TOCSAVE = 109 # macro
R_PPC64_ADDR16_HIGH = 110 # macro
R_PPC64_ADDR16_HIGHA = 111 # macro
R_PPC64_TPREL16_HIGH = 112 # macro
R_PPC64_TPREL16_HIGHA = 113 # macro
R_PPC64_DTPREL16_HIGH = 114 # macro
R_PPC64_DTPREL16_HIGHA = 115 # macro
R_PPC64_JMP_IREL = 247 # macro
R_PPC64_IRELATIVE = 248 # macro
R_PPC64_REL16 = 249 # macro
R_PPC64_REL16_LO = 250 # macro
R_PPC64_REL16_HI = 251 # macro
R_PPC64_REL16_HA = 252 # macro
EF_PPC64_ABI = 3 # macro
DT_PPC64_GLINK = (0x70000000+0) # macro
DT_PPC64_OPD = (0x70000000+1) # macro
DT_PPC64_OPDSZ = (0x70000000+2) # macro
DT_PPC64_OPT = (0x70000000+3) # macro
DT_PPC64_NUM = 4 # macro
PPC64_OPT_TLS = 1 # macro
PPC64_OPT_MULTI_TOC = 2 # macro
PPC64_OPT_LOCALENTRY = 4 # macro
STO_PPC64_LOCAL_BIT = 5 # macro
STO_PPC64_LOCAL_MASK = (7<<5) # macro
def PPC64_LOCAL_ENTRY_OFFSET(other):  # macro
   return (((1<<(((other)&(7<<5))>>5))>>2)<<2)
EF_ARM_RELEXEC = 0x01 # macro
EF_ARM_HASENTRY = 0x02 # macro
EF_ARM_INTERWORK = 0x04 # macro
EF_ARM_APCS_26 = 0x08 # macro
EF_ARM_APCS_FLOAT = 0x10 # macro
EF_ARM_PIC = 0x20 # macro
EF_ARM_ALIGN8 = 0x40 # macro
EF_ARM_NEW_ABI = 0x80 # macro
EF_ARM_OLD_ABI = 0x100 # macro
EF_ARM_SOFT_FLOAT = 0x200 # macro
EF_ARM_VFP_FLOAT = 0x400 # macro
EF_ARM_MAVERICK_FLOAT = 0x800 # macro
EF_ARM_ABI_FLOAT_SOFT = 0x200 # macro
EF_ARM_ABI_FLOAT_HARD = 0x400 # macro
EF_ARM_SYMSARESORTED = 0x04 # macro
EF_ARM_DYNSYMSUSESEGIDX = 0x08 # macro
EF_ARM_MAPSYMSFIRST = 0x10 # macro
EF_ARM_EABIMASK = 0XFF000000 # macro
EF_ARM_BE8 = 0x00800000 # macro
EF_ARM_LE8 = 0x00400000 # macro
def EF_ARM_EABI_VERSION(flags):  # macro
   return ((flags)&0XFF000000)
EF_ARM_EABI_UNKNOWN = 0x00000000 # macro
EF_ARM_EABI_VER1 = 0x01000000 # macro
EF_ARM_EABI_VER2 = 0x02000000 # macro
EF_ARM_EABI_VER3 = 0x03000000 # macro
EF_ARM_EABI_VER4 = 0x04000000 # macro
EF_ARM_EABI_VER5 = 0x05000000 # macro
STT_ARM_TFUNC = 13 # macro
STT_ARM_16BIT = 15 # macro
SHF_ARM_ENTRYSECT = 0x10000000 # macro
SHF_ARM_COMDEF = 0x80000000 # macro
PF_ARM_SB = 0x10000000 # macro
PF_ARM_PI = 0x20000000 # macro
PF_ARM_ABS = 0x40000000 # macro
PT_ARM_EXIDX = (0x70000000+1) # macro
SHT_ARM_EXIDX = (0x70000000+1) # macro
SHT_ARM_PREEMPTMAP = (0x70000000+2) # macro
SHT_ARM_ATTRIBUTES = (0x70000000+3) # macro
R_AARCH64_NONE = 0 # macro
R_AARCH64_P32_ABS32 = 1 # macro
R_AARCH64_P32_COPY = 180 # macro
R_AARCH64_P32_GLOB_DAT = 181 # macro
R_AARCH64_P32_JUMP_SLOT = 182 # macro
R_AARCH64_P32_RELATIVE = 183 # macro
R_AARCH64_P32_TLS_DTPMOD = 184 # macro
R_AARCH64_P32_TLS_DTPREL = 185 # macro
R_AARCH64_P32_TLS_TPREL = 186 # macro
R_AARCH64_P32_TLSDESC = 187 # macro
R_AARCH64_P32_IRELATIVE = 188 # macro
R_AARCH64_ABS64 = 257 # macro
R_AARCH64_ABS32 = 258 # macro
R_AARCH64_ABS16 = 259 # macro
R_AARCH64_PREL64 = 260 # macro
R_AARCH64_PREL32 = 261 # macro
R_AARCH64_PREL16 = 262 # macro
R_AARCH64_MOVW_UABS_G0 = 263 # macro
R_AARCH64_MOVW_UABS_G0_NC = 264 # macro
R_AARCH64_MOVW_UABS_G1 = 265 # macro
R_AARCH64_MOVW_UABS_G1_NC = 266 # macro
R_AARCH64_MOVW_UABS_G2 = 267 # macro
R_AARCH64_MOVW_UABS_G2_NC = 268 # macro
R_AARCH64_MOVW_UABS_G3 = 269 # macro
R_AARCH64_MOVW_SABS_G0 = 270 # macro
R_AARCH64_MOVW_SABS_G1 = 271 # macro
R_AARCH64_MOVW_SABS_G2 = 272 # macro
R_AARCH64_LD_PREL_LO19 = 273 # macro
R_AARCH64_ADR_PREL_LO21 = 274 # macro
R_AARCH64_ADR_PREL_PG_HI21 = 275 # macro
R_AARCH64_ADR_PREL_PG_HI21_NC = 276 # macro
R_AARCH64_ADD_ABS_LO12_NC = 277 # macro
R_AARCH64_LDST8_ABS_LO12_NC = 278 # macro
R_AARCH64_TSTBR14 = 279 # macro
R_AARCH64_CONDBR19 = 280 # macro
R_AARCH64_JUMP26 = 282 # macro
R_AARCH64_CALL26 = 283 # macro
R_AARCH64_LDST16_ABS_LO12_NC = 284 # macro
R_AARCH64_LDST32_ABS_LO12_NC = 285 # macro
R_AARCH64_LDST64_ABS_LO12_NC = 286 # macro
R_AARCH64_MOVW_PREL_G0 = 287 # macro
R_AARCH64_MOVW_PREL_G0_NC = 288 # macro
R_AARCH64_MOVW_PREL_G1 = 289 # macro
R_AARCH64_MOVW_PREL_G1_NC = 290 # macro
R_AARCH64_MOVW_PREL_G2 = 291 # macro
R_AARCH64_MOVW_PREL_G2_NC = 292 # macro
R_AARCH64_MOVW_PREL_G3 = 293 # macro
R_AARCH64_LDST128_ABS_LO12_NC = 299 # macro
R_AARCH64_MOVW_GOTOFF_G0 = 300 # macro
R_AARCH64_MOVW_GOTOFF_G0_NC = 301 # macro
R_AARCH64_MOVW_GOTOFF_G1 = 302 # macro
R_AARCH64_MOVW_GOTOFF_G1_NC = 303 # macro
R_AARCH64_MOVW_GOTOFF_G2 = 304 # macro
R_AARCH64_MOVW_GOTOFF_G2_NC = 305 # macro
R_AARCH64_MOVW_GOTOFF_G3 = 306 # macro
R_AARCH64_GOTREL64 = 307 # macro
R_AARCH64_GOTREL32 = 308 # macro
R_AARCH64_GOT_LD_PREL19 = 309 # macro
R_AARCH64_LD64_GOTOFF_LO15 = 310 # macro
R_AARCH64_ADR_GOT_PAGE = 311 # macro
R_AARCH64_LD64_GOT_LO12_NC = 312 # macro
R_AARCH64_LD64_GOTPAGE_LO15 = 313 # macro
R_AARCH64_TLSGD_ADR_PREL21 = 512 # macro
R_AARCH64_TLSGD_ADR_PAGE21 = 513 # macro
R_AARCH64_TLSGD_ADD_LO12_NC = 514 # macro
R_AARCH64_TLSGD_MOVW_G1 = 515 # macro
R_AARCH64_TLSGD_MOVW_G0_NC = 516 # macro
R_AARCH64_TLSLD_ADR_PREL21 = 517 # macro
R_AARCH64_TLSLD_ADR_PAGE21 = 518 # macro
R_AARCH64_TLSLD_ADD_LO12_NC = 519 # macro
R_AARCH64_TLSLD_MOVW_G1 = 520 # macro
R_AARCH64_TLSLD_MOVW_G0_NC = 521 # macro
R_AARCH64_TLSLD_LD_PREL19 = 522 # macro
R_AARCH64_TLSLD_MOVW_DTPREL_G2 = 523 # macro
R_AARCH64_TLSLD_MOVW_DTPREL_G1 = 524 # macro
R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC = 525 # macro
R_AARCH64_TLSLD_MOVW_DTPREL_G0 = 526 # macro
R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC = 527 # macro
R_AARCH64_TLSLD_ADD_DTPREL_HI12 = 528 # macro
R_AARCH64_TLSLD_ADD_DTPREL_LO12 = 529 # macro
R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC = 530 # macro
R_AARCH64_TLSLD_LDST8_DTPREL_LO12 = 531 # macro
R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC = 532 # macro
R_AARCH64_TLSLD_LDST16_DTPREL_LO12 = 533 # macro
R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC = 534 # macro
R_AARCH64_TLSLD_LDST32_DTPREL_LO12 = 535 # macro
R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC = 536 # macro
R_AARCH64_TLSLD_LDST64_DTPREL_LO12 = 537 # macro
R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC = 538 # macro
R_AARCH64_TLSIE_MOVW_GOTTPREL_G1 = 539 # macro
R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC = 540 # macro
R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 = 541 # macro
R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC = 542 # macro
R_AARCH64_TLSIE_LD_GOTTPREL_PREL19 = 543 # macro
R_AARCH64_TLSLE_MOVW_TPREL_G2 = 544 # macro
R_AARCH64_TLSLE_MOVW_TPREL_G1 = 545 # macro
R_AARCH64_TLSLE_MOVW_TPREL_G1_NC = 546 # macro
R_AARCH64_TLSLE_MOVW_TPREL_G0 = 547 # macro
R_AARCH64_TLSLE_MOVW_TPREL_G0_NC = 548 # macro
R_AARCH64_TLSLE_ADD_TPREL_HI12 = 549 # macro
R_AARCH64_TLSLE_ADD_TPREL_LO12 = 550 # macro
R_AARCH64_TLSLE_ADD_TPREL_LO12_NC = 551 # macro
R_AARCH64_TLSLE_LDST8_TPREL_LO12 = 552 # macro
R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC = 553 # macro
R_AARCH64_TLSLE_LDST16_TPREL_LO12 = 554 # macro
R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC = 555 # macro
R_AARCH64_TLSLE_LDST32_TPREL_LO12 = 556 # macro
R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC = 557 # macro
R_AARCH64_TLSLE_LDST64_TPREL_LO12 = 558 # macro
R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC = 559 # macro
R_AARCH64_TLSDESC_LD_PREL19 = 560 # macro
R_AARCH64_TLSDESC_ADR_PREL21 = 561 # macro
R_AARCH64_TLSDESC_ADR_PAGE21 = 562 # macro
R_AARCH64_TLSDESC_LD64_LO12 = 563 # macro
R_AARCH64_TLSDESC_ADD_LO12 = 564 # macro
R_AARCH64_TLSDESC_OFF_G1 = 565 # macro
R_AARCH64_TLSDESC_OFF_G0_NC = 566 # macro
R_AARCH64_TLSDESC_LDR = 567 # macro
R_AARCH64_TLSDESC_ADD = 568 # macro
R_AARCH64_TLSDESC_CALL = 569 # macro
R_AARCH64_TLSLE_LDST128_TPREL_LO12 = 570 # macro
R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC = 571 # macro
R_AARCH64_TLSLD_LDST128_DTPREL_LO12 = 572 # macro
R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC = 573 # macro
R_AARCH64_COPY = 1024 # macro
R_AARCH64_GLOB_DAT = 1025 # macro
R_AARCH64_JUMP_SLOT = 1026 # macro
R_AARCH64_RELATIVE = 1027 # macro
R_AARCH64_TLS_DTPMOD = 1028 # macro
R_AARCH64_TLS_DTPREL = 1029 # macro
R_AARCH64_TLS_TPREL = 1030 # macro
R_AARCH64_TLSDESC = 1031 # macro
R_AARCH64_IRELATIVE = 1032 # macro
PT_AARCH64_MEMTAG_MTE = (0x70000000+2) # macro
DT_AARCH64_BTI_PLT = (0x70000000+1) # macro
DT_AARCH64_PAC_PLT = (0x70000000+3) # macro
DT_AARCH64_VARIANT_PCS = (0x70000000+5) # macro
DT_AARCH64_NUM = 6 # macro
STO_AARCH64_VARIANT_PCS = 0x80 # macro
R_ARM_NONE = 0 # macro
R_ARM_PC24 = 1 # macro
R_ARM_ABS32 = 2 # macro
R_ARM_REL32 = 3 # macro
R_ARM_PC13 = 4 # macro
R_ARM_ABS16 = 5 # macro
R_ARM_ABS12 = 6 # macro
R_ARM_THM_ABS5 = 7 # macro
R_ARM_ABS8 = 8 # macro
R_ARM_SBREL32 = 9 # macro
R_ARM_THM_PC22 = 10 # macro
R_ARM_THM_PC8 = 11 # macro
R_ARM_AMP_VCALL9 = 12 # macro
R_ARM_SWI24 = 13 # macro
R_ARM_TLS_DESC = 13 # macro
R_ARM_THM_SWI8 = 14 # macro
R_ARM_XPC25 = 15 # macro
R_ARM_THM_XPC22 = 16 # macro
R_ARM_TLS_DTPMOD32 = 17 # macro
R_ARM_TLS_DTPOFF32 = 18 # macro
R_ARM_TLS_TPOFF32 = 19 # macro
R_ARM_COPY = 20 # macro
R_ARM_GLOB_DAT = 21 # macro
R_ARM_JUMP_SLOT = 22 # macro
R_ARM_RELATIVE = 23 # macro
R_ARM_GOTOFF = 24 # macro
R_ARM_GOTPC = 25 # macro
R_ARM_GOT32 = 26 # macro
R_ARM_PLT32 = 27 # macro
R_ARM_CALL = 28 # macro
R_ARM_JUMP24 = 29 # macro
R_ARM_THM_JUMP24 = 30 # macro
R_ARM_BASE_ABS = 31 # macro
R_ARM_ALU_PCREL_7_0 = 32 # macro
R_ARM_ALU_PCREL_15_8 = 33 # macro
R_ARM_ALU_PCREL_23_15 = 34 # macro
R_ARM_LDR_SBREL_11_0 = 35 # macro
R_ARM_ALU_SBREL_19_12 = 36 # macro
R_ARM_ALU_SBREL_27_20 = 37 # macro
R_ARM_TARGET1 = 38 # macro
R_ARM_SBREL31 = 39 # macro
R_ARM_V4BX = 40 # macro
R_ARM_TARGET2 = 41 # macro
R_ARM_PREL31 = 42 # macro
R_ARM_MOVW_ABS_NC = 43 # macro
R_ARM_MOVT_ABS = 44 # macro
R_ARM_MOVW_PREL_NC = 45 # macro
R_ARM_MOVT_PREL = 46 # macro
R_ARM_THM_MOVW_ABS_NC = 47 # macro
R_ARM_THM_MOVT_ABS = 48 # macro
R_ARM_THM_MOVW_PREL_NC = 49 # macro
R_ARM_THM_MOVT_PREL = 50 # macro
R_ARM_THM_JUMP19 = 51 # macro
R_ARM_THM_JUMP6 = 52 # macro
R_ARM_THM_ALU_PREL_11_0 = 53 # macro
R_ARM_THM_PC12 = 54 # macro
R_ARM_ABS32_NOI = 55 # macro
R_ARM_REL32_NOI = 56 # macro
R_ARM_ALU_PC_G0_NC = 57 # macro
R_ARM_ALU_PC_G0 = 58 # macro
R_ARM_ALU_PC_G1_NC = 59 # macro
R_ARM_ALU_PC_G1 = 60 # macro
R_ARM_ALU_PC_G2 = 61 # macro
R_ARM_LDR_PC_G1 = 62 # macro
R_ARM_LDR_PC_G2 = 63 # macro
R_ARM_LDRS_PC_G0 = 64 # macro
R_ARM_LDRS_PC_G1 = 65 # macro
R_ARM_LDRS_PC_G2 = 66 # macro
R_ARM_LDC_PC_G0 = 67 # macro
R_ARM_LDC_PC_G1 = 68 # macro
R_ARM_LDC_PC_G2 = 69 # macro
R_ARM_ALU_SB_G0_NC = 70 # macro
R_ARM_ALU_SB_G0 = 71 # macro
R_ARM_ALU_SB_G1_NC = 72 # macro
R_ARM_ALU_SB_G1 = 73 # macro
R_ARM_ALU_SB_G2 = 74 # macro
R_ARM_LDR_SB_G0 = 75 # macro
R_ARM_LDR_SB_G1 = 76 # macro
R_ARM_LDR_SB_G2 = 77 # macro
R_ARM_LDRS_SB_G0 = 78 # macro
R_ARM_LDRS_SB_G1 = 79 # macro
R_ARM_LDRS_SB_G2 = 80 # macro
R_ARM_LDC_SB_G0 = 81 # macro
R_ARM_LDC_SB_G1 = 82 # macro
R_ARM_LDC_SB_G2 = 83 # macro
R_ARM_MOVW_BREL_NC = 84 # macro
R_ARM_MOVT_BREL = 85 # macro
R_ARM_MOVW_BREL = 86 # macro
R_ARM_THM_MOVW_BREL_NC = 87 # macro
R_ARM_THM_MOVT_BREL = 88 # macro
R_ARM_THM_MOVW_BREL = 89 # macro
R_ARM_TLS_GOTDESC = 90 # macro
R_ARM_TLS_CALL = 91 # macro
R_ARM_TLS_DESCSEQ = 92 # macro
R_ARM_THM_TLS_CALL = 93 # macro
R_ARM_PLT32_ABS = 94 # macro
R_ARM_GOT_ABS = 95 # macro
R_ARM_GOT_PREL = 96 # macro
R_ARM_GOT_BREL12 = 97 # macro
R_ARM_GOTOFF12 = 98 # macro
R_ARM_GOTRELAX = 99 # macro
R_ARM_GNU_VTENTRY = 100 # macro
R_ARM_GNU_VTINHERIT = 101 # macro
R_ARM_THM_PC11 = 102 # macro
R_ARM_THM_PC9 = 103 # macro
R_ARM_TLS_GD32 = 104 # macro
R_ARM_TLS_LDM32 = 105 # macro
R_ARM_TLS_LDO32 = 106 # macro
R_ARM_TLS_IE32 = 107 # macro
R_ARM_TLS_LE32 = 108 # macro
R_ARM_TLS_LDO12 = 109 # macro
R_ARM_TLS_LE12 = 110 # macro
R_ARM_TLS_IE12GP = 111 # macro
R_ARM_ME_TOO = 128 # macro
R_ARM_THM_TLS_DESCSEQ = 129 # macro
R_ARM_THM_TLS_DESCSEQ16 = 129 # macro
R_ARM_THM_TLS_DESCSEQ32 = 130 # macro
R_ARM_THM_GOT_BREL12 = 131 # macro
R_ARM_IRELATIVE = 160 # macro
R_ARM_RXPC25 = 249 # macro
R_ARM_RSBREL32 = 250 # macro
R_ARM_THM_RPC22 = 251 # macro
R_ARM_RREL32 = 252 # macro
R_ARM_RABS22 = 253 # macro
R_ARM_RPC24 = 254 # macro
R_ARM_RBASE = 255 # macro
R_ARM_NUM = 256 # macro
R_CKCORE_NONE = 0 # macro
R_CKCORE_ADDR32 = 1 # macro
R_CKCORE_PCRELIMM8BY4 = 2 # macro
R_CKCORE_PCRELIMM11BY2 = 3 # macro
R_CKCORE_PCREL32 = 5 # macro
R_CKCORE_PCRELJSR_IMM11BY2 = 6 # macro
R_CKCORE_RELATIVE = 9 # macro
R_CKCORE_COPY = 10 # macro
R_CKCORE_GLOB_DAT = 11 # macro
R_CKCORE_JUMP_SLOT = 12 # macro
R_CKCORE_GOTOFF = 13 # macro
R_CKCORE_GOTPC = 14 # macro
R_CKCORE_GOT32 = 15 # macro
R_CKCORE_PLT32 = 16 # macro
R_CKCORE_ADDRGOT = 17 # macro
R_CKCORE_ADDRPLT = 18 # macro
R_CKCORE_PCREL_IMM26BY2 = 19 # macro
R_CKCORE_PCREL_IMM16BY2 = 20 # macro
R_CKCORE_PCREL_IMM16BY4 = 21 # macro
R_CKCORE_PCREL_IMM10BY2 = 22 # macro
R_CKCORE_PCREL_IMM10BY4 = 23 # macro
R_CKCORE_ADDR_HI16 = 24 # macro
R_CKCORE_ADDR_LO16 = 25 # macro
R_CKCORE_GOTPC_HI16 = 26 # macro
R_CKCORE_GOTPC_LO16 = 27 # macro
R_CKCORE_GOTOFF_HI16 = 28 # macro
R_CKCORE_GOTOFF_LO16 = 29 # macro
R_CKCORE_GOT12 = 30 # macro
R_CKCORE_GOT_HI16 = 31 # macro
R_CKCORE_GOT_LO16 = 32 # macro
R_CKCORE_PLT12 = 33 # macro
R_CKCORE_PLT_HI16 = 34 # macro
R_CKCORE_PLT_LO16 = 35 # macro
R_CKCORE_ADDRGOT_HI16 = 36 # macro
R_CKCORE_ADDRGOT_LO16 = 37 # macro
R_CKCORE_ADDRPLT_HI16 = 38 # macro
R_CKCORE_ADDRPLT_LO16 = 39 # macro
R_CKCORE_PCREL_JSR_IMM26BY2 = 40 # macro
R_CKCORE_TOFFSET_LO16 = 41 # macro
R_CKCORE_DOFFSET_LO16 = 42 # macro
R_CKCORE_PCREL_IMM18BY2 = 43 # macro
R_CKCORE_DOFFSET_IMM18 = 44 # macro
R_CKCORE_DOFFSET_IMM18BY2 = 45 # macro
R_CKCORE_DOFFSET_IMM18BY4 = 46 # macro
R_CKCORE_GOT_IMM18BY4 = 48 # macro
R_CKCORE_PLT_IMM18BY4 = 49 # macro
R_CKCORE_PCREL_IMM7BY4 = 50 # macro
R_CKCORE_TLS_LE32 = 51 # macro
R_CKCORE_TLS_IE32 = 52 # macro
R_CKCORE_TLS_GD32 = 53 # macro
R_CKCORE_TLS_LDM32 = 54 # macro
R_CKCORE_TLS_LDO32 = 55 # macro
R_CKCORE_TLS_DTPMOD32 = 56 # macro
R_CKCORE_TLS_DTPOFF32 = 57 # macro
R_CKCORE_TLS_TPOFF32 = 58 # macro
EF_CSKY_ABIMASK = 0XF0000000 # macro
EF_CSKY_OTHER = 0X0FFF0000 # macro
EF_CSKY_PROCESSOR = 0X0000FFFF # macro
EF_CSKY_ABIV1 = 0X10000000 # macro
EF_CSKY_ABIV2 = 0X20000000 # macro
SHT_CSKY_ATTRIBUTES = (0x70000000+1) # macro
EF_IA_64_MASKOS = 0x0000000f # macro
EF_IA_64_ABI64 = 0x00000010 # macro
EF_IA_64_ARCH = 0xff000000 # macro
PT_IA_64_ARCHEXT = (0x70000000+0) # macro
PT_IA_64_UNWIND = (0x70000000+1) # macro
PT_IA_64_HP_OPT_ANOT = (0x60000000+0x12) # macro
PT_IA_64_HP_HSL_ANOT = (0x60000000+0x13) # macro
PT_IA_64_HP_STACK = (0x60000000+0x14) # macro
PF_IA_64_NORECOV = 0x80000000 # macro
SHT_IA_64_EXT = (0x70000000+0) # macro
SHT_IA_64_UNWIND = (0x70000000+1) # macro
SHF_IA_64_SHORT = 0x10000000 # macro
SHF_IA_64_NORECOV = 0x20000000 # macro
DT_IA_64_PLT_RESERVE = (0x70000000+0) # macro
DT_IA_64_NUM = 1 # macro
R_IA64_NONE = 0x00 # macro
R_IA64_IMM14 = 0x21 # macro
R_IA64_IMM22 = 0x22 # macro
R_IA64_IMM64 = 0x23 # macro
R_IA64_DIR32MSB = 0x24 # macro
R_IA64_DIR32LSB = 0x25 # macro
R_IA64_DIR64MSB = 0x26 # macro
R_IA64_DIR64LSB = 0x27 # macro
R_IA64_GPREL22 = 0x2a # macro
R_IA64_GPREL64I = 0x2b # macro
R_IA64_GPREL32MSB = 0x2c # macro
R_IA64_GPREL32LSB = 0x2d # macro
R_IA64_GPREL64MSB = 0x2e # macro
R_IA64_GPREL64LSB = 0x2f # macro
R_IA64_LTOFF22 = 0x32 # macro
R_IA64_LTOFF64I = 0x33 # macro
R_IA64_PLTOFF22 = 0x3a # macro
R_IA64_PLTOFF64I = 0x3b # macro
R_IA64_PLTOFF64MSB = 0x3e # macro
R_IA64_PLTOFF64LSB = 0x3f # macro
R_IA64_FPTR64I = 0x43 # macro
R_IA64_FPTR32MSB = 0x44 # macro
R_IA64_FPTR32LSB = 0x45 # macro
R_IA64_FPTR64MSB = 0x46 # macro
R_IA64_FPTR64LSB = 0x47 # macro
R_IA64_PCREL60B = 0x48 # macro
R_IA64_PCREL21B = 0x49 # macro
R_IA64_PCREL21M = 0x4a # macro
R_IA64_PCREL21F = 0x4b # macro
R_IA64_PCREL32MSB = 0x4c # macro
R_IA64_PCREL32LSB = 0x4d # macro
R_IA64_PCREL64MSB = 0x4e # macro
R_IA64_PCREL64LSB = 0x4f # macro
R_IA64_LTOFF_FPTR22 = 0x52 # macro
R_IA64_LTOFF_FPTR64I = 0x53 # macro
R_IA64_LTOFF_FPTR32MSB = 0x54 # macro
R_IA64_LTOFF_FPTR32LSB = 0x55 # macro
R_IA64_LTOFF_FPTR64MSB = 0x56 # macro
R_IA64_LTOFF_FPTR64LSB = 0x57 # macro
R_IA64_SEGREL32MSB = 0x5c # macro
R_IA64_SEGREL32LSB = 0x5d # macro
R_IA64_SEGREL64MSB = 0x5e # macro
R_IA64_SEGREL64LSB = 0x5f # macro
R_IA64_SECREL32MSB = 0x64 # macro
R_IA64_SECREL32LSB = 0x65 # macro
R_IA64_SECREL64MSB = 0x66 # macro
R_IA64_SECREL64LSB = 0x67 # macro
R_IA64_REL32MSB = 0x6c # macro
R_IA64_REL32LSB = 0x6d # macro
R_IA64_REL64MSB = 0x6e # macro
R_IA64_REL64LSB = 0x6f # macro
R_IA64_LTV32MSB = 0x74 # macro
R_IA64_LTV32LSB = 0x75 # macro
R_IA64_LTV64MSB = 0x76 # macro
R_IA64_LTV64LSB = 0x77 # macro
R_IA64_PCREL21BI = 0x79 # macro
R_IA64_PCREL22 = 0x7a # macro
R_IA64_PCREL64I = 0x7b # macro
R_IA64_IPLTMSB = 0x80 # macro
R_IA64_IPLTLSB = 0x81 # macro
R_IA64_COPY = 0x84 # macro
R_IA64_SUB = 0x85 # macro
R_IA64_LTOFF22X = 0x86 # macro
R_IA64_LDXMOV = 0x87 # macro
R_IA64_TPREL14 = 0x91 # macro
R_IA64_TPREL22 = 0x92 # macro
R_IA64_TPREL64I = 0x93 # macro
R_IA64_TPREL64MSB = 0x96 # macro
R_IA64_TPREL64LSB = 0x97 # macro
R_IA64_LTOFF_TPREL22 = 0x9a # macro
R_IA64_DTPMOD64MSB = 0xa6 # macro
R_IA64_DTPMOD64LSB = 0xa7 # macro
R_IA64_LTOFF_DTPMOD22 = 0xaa # macro
R_IA64_DTPREL14 = 0xb1 # macro
R_IA64_DTPREL22 = 0xb2 # macro
R_IA64_DTPREL64I = 0xb3 # macro
R_IA64_DTPREL32MSB = 0xb4 # macro
R_IA64_DTPREL32LSB = 0xb5 # macro
R_IA64_DTPREL64MSB = 0xb6 # macro
R_IA64_DTPREL64LSB = 0xb7 # macro
R_IA64_LTOFF_DTPREL22 = 0xba # macro
EF_SH_MACH_MASK = 0x1f # macro
EF_SH_UNKNOWN = 0x0 # macro
EF_SH1 = 0x1 # macro
EF_SH2 = 0x2 # macro
EF_SH3 = 0x3 # macro
EF_SH_DSP = 0x4 # macro
EF_SH3_DSP = 0x5 # macro
EF_SH4AL_DSP = 0x6 # macro
EF_SH3E = 0x8 # macro
EF_SH4 = 0x9 # macro
EF_SH2E = 0xb # macro
EF_SH4A = 0xc # macro
EF_SH2A = 0xd # macro
EF_SH4_NOFPU = 0x10 # macro
EF_SH4A_NOFPU = 0x11 # macro
EF_SH4_NOMMU_NOFPU = 0x12 # macro
EF_SH2A_NOFPU = 0x13 # macro
EF_SH3_NOMMU = 0x14 # macro
EF_SH2A_SH4_NOFPU = 0x15 # macro
EF_SH2A_SH3_NOFPU = 0x16 # macro
EF_SH2A_SH4 = 0x17 # macro
EF_SH2A_SH3E = 0x18 # macro
R_SH_NONE = 0 # macro
R_SH_DIR32 = 1 # macro
R_SH_REL32 = 2 # macro
R_SH_DIR8WPN = 3 # macro
R_SH_IND12W = 4 # macro
R_SH_DIR8WPL = 5 # macro
R_SH_DIR8WPZ = 6 # macro
R_SH_DIR8BP = 7 # macro
R_SH_DIR8W = 8 # macro
R_SH_DIR8L = 9 # macro
R_SH_SWITCH16 = 25 # macro
R_SH_SWITCH32 = 26 # macro
R_SH_USES = 27 # macro
R_SH_COUNT = 28 # macro
R_SH_ALIGN = 29 # macro
R_SH_CODE = 30 # macro
R_SH_DATA = 31 # macro
R_SH_LABEL = 32 # macro
R_SH_SWITCH8 = 33 # macro
R_SH_GNU_VTINHERIT = 34 # macro
R_SH_GNU_VTENTRY = 35 # macro
R_SH_TLS_GD_32 = 144 # macro
R_SH_TLS_LD_32 = 145 # macro
R_SH_TLS_LDO_32 = 146 # macro
R_SH_TLS_IE_32 = 147 # macro
R_SH_TLS_LE_32 = 148 # macro
R_SH_TLS_DTPMOD32 = 149 # macro
R_SH_TLS_DTPOFF32 = 150 # macro
R_SH_TLS_TPOFF32 = 151 # macro
R_SH_GOT32 = 160 # macro
R_SH_PLT32 = 161 # macro
R_SH_COPY = 162 # macro
R_SH_GLOB_DAT = 163 # macro
R_SH_JMP_SLOT = 164 # macro
R_SH_RELATIVE = 165 # macro
R_SH_GOTOFF = 166 # macro
R_SH_GOTPC = 167 # macro
R_SH_NUM = 256 # macro
EF_S390_HIGH_GPRS = 0x00000001 # macro
R_390_NONE = 0 # macro
R_390_8 = 1 # macro
R_390_12 = 2 # macro
R_390_16 = 3 # macro
R_390_32 = 4 # macro
R_390_PC32 = 5 # macro
R_390_GOT12 = 6 # macro
R_390_GOT32 = 7 # macro
R_390_PLT32 = 8 # macro
R_390_COPY = 9 # macro
R_390_GLOB_DAT = 10 # macro
R_390_JMP_SLOT = 11 # macro
R_390_RELATIVE = 12 # macro
R_390_GOTOFF32 = 13 # macro
R_390_GOTPC = 14 # macro
R_390_GOT16 = 15 # macro
R_390_PC16 = 16 # macro
R_390_PC16DBL = 17 # macro
R_390_PLT16DBL = 18 # macro
R_390_PC32DBL = 19 # macro
R_390_PLT32DBL = 20 # macro
R_390_GOTPCDBL = 21 # macro
R_390_64 = 22 # macro
R_390_PC64 = 23 # macro
R_390_GOT64 = 24 # macro
R_390_PLT64 = 25 # macro
R_390_GOTENT = 26 # macro
R_390_GOTOFF16 = 27 # macro
R_390_GOTOFF64 = 28 # macro
R_390_GOTPLT12 = 29 # macro
R_390_GOTPLT16 = 30 # macro
R_390_GOTPLT32 = 31 # macro
R_390_GOTPLT64 = 32 # macro
R_390_GOTPLTENT = 33 # macro
R_390_PLTOFF16 = 34 # macro
R_390_PLTOFF32 = 35 # macro
R_390_PLTOFF64 = 36 # macro
R_390_TLS_LOAD = 37 # macro
R_390_TLS_GDCALL = 38 # macro
R_390_TLS_LDCALL = 39 # macro
R_390_TLS_GD32 = 40 # macro
R_390_TLS_GD64 = 41 # macro
R_390_TLS_GOTIE12 = 42 # macro
R_390_TLS_GOTIE32 = 43 # macro
R_390_TLS_GOTIE64 = 44 # macro
R_390_TLS_LDM32 = 45 # macro
R_390_TLS_LDM64 = 46 # macro
R_390_TLS_IE32 = 47 # macro
R_390_TLS_IE64 = 48 # macro
R_390_TLS_IEENT = 49 # macro
R_390_TLS_LE32 = 50 # macro
R_390_TLS_LE64 = 51 # macro
R_390_TLS_LDO32 = 52 # macro
R_390_TLS_LDO64 = 53 # macro
R_390_TLS_DTPMOD = 54 # macro
R_390_TLS_DTPOFF = 55 # macro
R_390_TLS_TPOFF = 56 # macro
R_390_20 = 57 # macro
R_390_GOT20 = 58 # macro
R_390_GOTPLT20 = 59 # macro
R_390_TLS_GOTIE20 = 60 # macro
R_390_IRELATIVE = 61 # macro
R_390_NUM = 62 # macro
R_CRIS_NONE = 0 # macro
R_CRIS_8 = 1 # macro
R_CRIS_16 = 2 # macro
R_CRIS_32 = 3 # macro
R_CRIS_8_PCREL = 4 # macro
R_CRIS_16_PCREL = 5 # macro
R_CRIS_32_PCREL = 6 # macro
R_CRIS_GNU_VTINHERIT = 7 # macro
R_CRIS_GNU_VTENTRY = 8 # macro
R_CRIS_COPY = 9 # macro
R_CRIS_GLOB_DAT = 10 # macro
R_CRIS_JUMP_SLOT = 11 # macro
R_CRIS_RELATIVE = 12 # macro
R_CRIS_16_GOT = 13 # macro
R_CRIS_32_GOT = 14 # macro
R_CRIS_16_GOTPLT = 15 # macro
R_CRIS_32_GOTPLT = 16 # macro
R_CRIS_32_GOTREL = 17 # macro
R_CRIS_32_PLT_GOTREL = 18 # macro
R_CRIS_32_PLT_PCREL = 19 # macro
R_CRIS_NUM = 20 # macro
R_X86_64_NONE = 0 # macro
R_X86_64_64 = 1 # macro
R_X86_64_PC32 = 2 # macro
R_X86_64_GOT32 = 3 # macro
R_X86_64_PLT32 = 4 # macro
R_X86_64_COPY = 5 # macro
R_X86_64_GLOB_DAT = 6 # macro
R_X86_64_JUMP_SLOT = 7 # macro
R_X86_64_RELATIVE = 8 # macro
R_X86_64_GOTPCREL = 9 # macro
R_X86_64_32 = 10 # macro
R_X86_64_32S = 11 # macro
R_X86_64_16 = 12 # macro
R_X86_64_PC16 = 13 # macro
R_X86_64_8 = 14 # macro
R_X86_64_PC8 = 15 # macro
R_X86_64_DTPMOD64 = 16 # macro
R_X86_64_DTPOFF64 = 17 # macro
R_X86_64_TPOFF64 = 18 # macro
R_X86_64_TLSGD = 19 # macro
R_X86_64_TLSLD = 20 # macro
R_X86_64_DTPOFF32 = 21 # macro
R_X86_64_GOTTPOFF = 22 # macro
R_X86_64_TPOFF32 = 23 # macro
R_X86_64_PC64 = 24 # macro
R_X86_64_GOTOFF64 = 25 # macro
R_X86_64_GOTPC32 = 26 # macro
R_X86_64_GOT64 = 27 # macro
R_X86_64_GOTPCREL64 = 28 # macro
R_X86_64_GOTPC64 = 29 # macro
R_X86_64_GOTPLT64 = 30 # macro
R_X86_64_PLTOFF64 = 31 # macro
R_X86_64_SIZE32 = 32 # macro
R_X86_64_SIZE64 = 33 # macro
R_X86_64_GOTPC32_TLSDESC = 34 # macro
R_X86_64_TLSDESC_CALL = 35 # macro
R_X86_64_TLSDESC = 36 # macro
R_X86_64_IRELATIVE = 37 # macro
R_X86_64_RELATIVE64 = 38 # macro
R_X86_64_GOTPCRELX = 41 # macro
R_X86_64_REX_GOTPCRELX = 42 # macro
R_X86_64_NUM = 43 # macro
SHT_X86_64_UNWIND = 0x70000001 # macro
DT_X86_64_PLT = (0x70000000+0) # macro
DT_X86_64_PLTSZ = (0x70000000+1) # macro
DT_X86_64_PLTENT = (0x70000000+3) # macro
DT_X86_64_NUM = 4 # macro
R_MN10300_NONE = 0 # macro
R_MN10300_32 = 1 # macro
R_MN10300_16 = 2 # macro
R_MN10300_8 = 3 # macro
R_MN10300_PCREL32 = 4 # macro
R_MN10300_PCREL16 = 5 # macro
R_MN10300_PCREL8 = 6 # macro
R_MN10300_GNU_VTINHERIT = 7 # macro
R_MN10300_GNU_VTENTRY = 8 # macro
R_MN10300_24 = 9 # macro
R_MN10300_GOTPC32 = 10 # macro
R_MN10300_GOTPC16 = 11 # macro
R_MN10300_GOTOFF32 = 12 # macro
R_MN10300_GOTOFF24 = 13 # macro
R_MN10300_GOTOFF16 = 14 # macro
R_MN10300_PLT32 = 15 # macro
R_MN10300_PLT16 = 16 # macro
R_MN10300_GOT32 = 17 # macro
R_MN10300_GOT24 = 18 # macro
R_MN10300_GOT16 = 19 # macro
R_MN10300_COPY = 20 # macro
R_MN10300_GLOB_DAT = 21 # macro
R_MN10300_JMP_SLOT = 22 # macro
R_MN10300_RELATIVE = 23 # macro
R_MN10300_TLS_GD = 24 # macro
R_MN10300_TLS_LD = 25 # macro
R_MN10300_TLS_LDO = 26 # macro
R_MN10300_TLS_GOTIE = 27 # macro
R_MN10300_TLS_IE = 28 # macro
R_MN10300_TLS_LE = 29 # macro
R_MN10300_TLS_DTPMOD = 30 # macro
R_MN10300_TLS_DTPOFF = 31 # macro
R_MN10300_TLS_TPOFF = 32 # macro
R_MN10300_SYM_DIFF = 33 # macro
R_MN10300_ALIGN = 34 # macro
R_MN10300_NUM = 35 # macro
R_M32R_NONE = 0 # macro
R_M32R_16 = 1 # macro
R_M32R_32 = 2 # macro
R_M32R_24 = 3 # macro
R_M32R_10_PCREL = 4 # macro
R_M32R_18_PCREL = 5 # macro
R_M32R_26_PCREL = 6 # macro
R_M32R_HI16_ULO = 7 # macro
R_M32R_HI16_SLO = 8 # macro
R_M32R_LO16 = 9 # macro
R_M32R_SDA16 = 10 # macro
R_M32R_GNU_VTINHERIT = 11 # macro
R_M32R_GNU_VTENTRY = 12 # macro
R_M32R_16_RELA = 33 # macro
R_M32R_32_RELA = 34 # macro
R_M32R_24_RELA = 35 # macro
R_M32R_10_PCREL_RELA = 36 # macro
R_M32R_18_PCREL_RELA = 37 # macro
R_M32R_26_PCREL_RELA = 38 # macro
R_M32R_HI16_ULO_RELA = 39 # macro
R_M32R_HI16_SLO_RELA = 40 # macro
R_M32R_LO16_RELA = 41 # macro
R_M32R_SDA16_RELA = 42 # macro
R_M32R_RELA_GNU_VTINHERIT = 43 # macro
R_M32R_RELA_GNU_VTENTRY = 44 # macro
R_M32R_REL32 = 45 # macro
R_M32R_GOT24 = 48 # macro
R_M32R_26_PLTREL = 49 # macro
R_M32R_COPY = 50 # macro
R_M32R_GLOB_DAT = 51 # macro
R_M32R_JMP_SLOT = 52 # macro
R_M32R_RELATIVE = 53 # macro
R_M32R_GOTOFF = 54 # macro
R_M32R_GOTPC24 = 55 # macro
R_M32R_GOT16_HI_ULO = 56 # macro
R_M32R_GOT16_HI_SLO = 57 # macro
R_M32R_GOT16_LO = 58 # macro
R_M32R_GOTPC_HI_ULO = 59 # macro
R_M32R_GOTPC_HI_SLO = 60 # macro
R_M32R_GOTPC_LO = 61 # macro
R_M32R_GOTOFF_HI_ULO = 62 # macro
R_M32R_GOTOFF_HI_SLO = 63 # macro
R_M32R_GOTOFF_LO = 64 # macro
R_M32R_NUM = 256 # macro
R_MICROBLAZE_NONE = 0 # macro
R_MICROBLAZE_32 = 1 # macro
R_MICROBLAZE_32_PCREL = 2 # macro
R_MICROBLAZE_64_PCREL = 3 # macro
R_MICROBLAZE_32_PCREL_LO = 4 # macro
R_MICROBLAZE_64 = 5 # macro
R_MICROBLAZE_32_LO = 6 # macro
R_MICROBLAZE_SRO32 = 7 # macro
R_MICROBLAZE_SRW32 = 8 # macro
R_MICROBLAZE_64_NONE = 9 # macro
R_MICROBLAZE_32_SYM_OP_SYM = 10 # macro
R_MICROBLAZE_GNU_VTINHERIT = 11 # macro
R_MICROBLAZE_GNU_VTENTRY = 12 # macro
R_MICROBLAZE_GOTPC_64 = 13 # macro
R_MICROBLAZE_GOT_64 = 14 # macro
R_MICROBLAZE_PLT_64 = 15 # macro
R_MICROBLAZE_REL = 16 # macro
R_MICROBLAZE_JUMP_SLOT = 17 # macro
R_MICROBLAZE_GLOB_DAT = 18 # macro
R_MICROBLAZE_GOTOFF_64 = 19 # macro
R_MICROBLAZE_GOTOFF_32 = 20 # macro
R_MICROBLAZE_COPY = 21 # macro
R_MICROBLAZE_TLS = 22 # macro
R_MICROBLAZE_TLSGD = 23 # macro
R_MICROBLAZE_TLSLD = 24 # macro
R_MICROBLAZE_TLSDTPMOD32 = 25 # macro
R_MICROBLAZE_TLSDTPREL32 = 26 # macro
R_MICROBLAZE_TLSDTPREL64 = 27 # macro
R_MICROBLAZE_TLSGOTTPREL32 = 28 # macro
R_MICROBLAZE_TLSTPREL32 = 29 # macro
DT_NIOS2_GP = 0x70000002 # macro
R_NIOS2_NONE = 0 # macro
R_NIOS2_S16 = 1 # macro
R_NIOS2_U16 = 2 # macro
R_NIOS2_PCREL16 = 3 # macro
R_NIOS2_CALL26 = 4 # macro
R_NIOS2_IMM5 = 5 # macro
R_NIOS2_CACHE_OPX = 6 # macro
R_NIOS2_IMM6 = 7 # macro
R_NIOS2_IMM8 = 8 # macro
R_NIOS2_HI16 = 9 # macro
R_NIOS2_LO16 = 10 # macro
R_NIOS2_HIADJ16 = 11 # macro
R_NIOS2_BFD_RELOC_32 = 12 # macro
R_NIOS2_BFD_RELOC_16 = 13 # macro
R_NIOS2_BFD_RELOC_8 = 14 # macro
R_NIOS2_GPREL = 15 # macro
R_NIOS2_GNU_VTINHERIT = 16 # macro
R_NIOS2_GNU_VTENTRY = 17 # macro
R_NIOS2_UJMP = 18 # macro
R_NIOS2_CJMP = 19 # macro
R_NIOS2_CALLR = 20 # macro
R_NIOS2_ALIGN = 21 # macro
R_NIOS2_GOT16 = 22 # macro
R_NIOS2_CALL16 = 23 # macro
R_NIOS2_GOTOFF_LO = 24 # macro
R_NIOS2_GOTOFF_HA = 25 # macro
R_NIOS2_PCREL_LO = 26 # macro
R_NIOS2_PCREL_HA = 27 # macro
R_NIOS2_TLS_GD16 = 28 # macro
R_NIOS2_TLS_LDM16 = 29 # macro
R_NIOS2_TLS_LDO16 = 30 # macro
R_NIOS2_TLS_IE16 = 31 # macro
R_NIOS2_TLS_LE16 = 32 # macro
R_NIOS2_TLS_DTPMOD = 33 # macro
R_NIOS2_TLS_DTPREL = 34 # macro
R_NIOS2_TLS_TPREL = 35 # macro
R_NIOS2_COPY = 36 # macro
R_NIOS2_GLOB_DAT = 37 # macro
R_NIOS2_JUMP_SLOT = 38 # macro
R_NIOS2_RELATIVE = 39 # macro
R_NIOS2_GOTOFF = 40 # macro
R_NIOS2_CALL26_NOAT = 41 # macro
R_NIOS2_GOT_LO = 42 # macro
R_NIOS2_GOT_HA = 43 # macro
R_NIOS2_CALL_LO = 44 # macro
R_NIOS2_CALL_HA = 45 # macro
R_TILEPRO_NONE = 0 # macro
R_TILEPRO_32 = 1 # macro
R_TILEPRO_16 = 2 # macro
R_TILEPRO_8 = 3 # macro
R_TILEPRO_32_PCREL = 4 # macro
R_TILEPRO_16_PCREL = 5 # macro
R_TILEPRO_8_PCREL = 6 # macro
R_TILEPRO_LO16 = 7 # macro
R_TILEPRO_HI16 = 8 # macro
R_TILEPRO_HA16 = 9 # macro
R_TILEPRO_COPY = 10 # macro
R_TILEPRO_GLOB_DAT = 11 # macro
R_TILEPRO_JMP_SLOT = 12 # macro
R_TILEPRO_RELATIVE = 13 # macro
R_TILEPRO_BROFF_X1 = 14 # macro
R_TILEPRO_JOFFLONG_X1 = 15 # macro
R_TILEPRO_JOFFLONG_X1_PLT = 16 # macro
R_TILEPRO_IMM8_X0 = 17 # macro
R_TILEPRO_IMM8_Y0 = 18 # macro
R_TILEPRO_IMM8_X1 = 19 # macro
R_TILEPRO_IMM8_Y1 = 20 # macro
R_TILEPRO_MT_IMM15_X1 = 21 # macro
R_TILEPRO_MF_IMM15_X1 = 22 # macro
R_TILEPRO_IMM16_X0 = 23 # macro
R_TILEPRO_IMM16_X1 = 24 # macro
R_TILEPRO_IMM16_X0_LO = 25 # macro
R_TILEPRO_IMM16_X1_LO = 26 # macro
R_TILEPRO_IMM16_X0_HI = 27 # macro
R_TILEPRO_IMM16_X1_HI = 28 # macro
R_TILEPRO_IMM16_X0_HA = 29 # macro
R_TILEPRO_IMM16_X1_HA = 30 # macro
R_TILEPRO_IMM16_X0_PCREL = 31 # macro
R_TILEPRO_IMM16_X1_PCREL = 32 # macro
R_TILEPRO_IMM16_X0_LO_PCREL = 33 # macro
R_TILEPRO_IMM16_X1_LO_PCREL = 34 # macro
R_TILEPRO_IMM16_X0_HI_PCREL = 35 # macro
R_TILEPRO_IMM16_X1_HI_PCREL = 36 # macro
R_TILEPRO_IMM16_X0_HA_PCREL = 37 # macro
R_TILEPRO_IMM16_X1_HA_PCREL = 38 # macro
R_TILEPRO_IMM16_X0_GOT = 39 # macro
R_TILEPRO_IMM16_X1_GOT = 40 # macro
R_TILEPRO_IMM16_X0_GOT_LO = 41 # macro
R_TILEPRO_IMM16_X1_GOT_LO = 42 # macro
R_TILEPRO_IMM16_X0_GOT_HI = 43 # macro
R_TILEPRO_IMM16_X1_GOT_HI = 44 # macro
R_TILEPRO_IMM16_X0_GOT_HA = 45 # macro
R_TILEPRO_IMM16_X1_GOT_HA = 46 # macro
R_TILEPRO_MMSTART_X0 = 47 # macro
R_TILEPRO_MMEND_X0 = 48 # macro
R_TILEPRO_MMSTART_X1 = 49 # macro
R_TILEPRO_MMEND_X1 = 50 # macro
R_TILEPRO_SHAMT_X0 = 51 # macro
R_TILEPRO_SHAMT_X1 = 52 # macro
R_TILEPRO_SHAMT_Y0 = 53 # macro
R_TILEPRO_SHAMT_Y1 = 54 # macro
R_TILEPRO_DEST_IMM8_X1 = 55 # macro
R_TILEPRO_TLS_GD_CALL = 60 # macro
R_TILEPRO_IMM8_X0_TLS_GD_ADD = 61 # macro
R_TILEPRO_IMM8_X1_TLS_GD_ADD = 62 # macro
R_TILEPRO_IMM8_Y0_TLS_GD_ADD = 63 # macro
R_TILEPRO_IMM8_Y1_TLS_GD_ADD = 64 # macro
R_TILEPRO_TLS_IE_LOAD = 65 # macro
R_TILEPRO_IMM16_X0_TLS_GD = 66 # macro
R_TILEPRO_IMM16_X1_TLS_GD = 67 # macro
R_TILEPRO_IMM16_X0_TLS_GD_LO = 68 # macro
R_TILEPRO_IMM16_X1_TLS_GD_LO = 69 # macro
R_TILEPRO_IMM16_X0_TLS_GD_HI = 70 # macro
R_TILEPRO_IMM16_X1_TLS_GD_HI = 71 # macro
R_TILEPRO_IMM16_X0_TLS_GD_HA = 72 # macro
R_TILEPRO_IMM16_X1_TLS_GD_HA = 73 # macro
R_TILEPRO_IMM16_X0_TLS_IE = 74 # macro
R_TILEPRO_IMM16_X1_TLS_IE = 75 # macro
R_TILEPRO_IMM16_X0_TLS_IE_LO = 76 # macro
R_TILEPRO_IMM16_X1_TLS_IE_LO = 77 # macro
R_TILEPRO_IMM16_X0_TLS_IE_HI = 78 # macro
R_TILEPRO_IMM16_X1_TLS_IE_HI = 79 # macro
R_TILEPRO_IMM16_X0_TLS_IE_HA = 80 # macro
R_TILEPRO_IMM16_X1_TLS_IE_HA = 81 # macro
R_TILEPRO_TLS_DTPMOD32 = 82 # macro
R_TILEPRO_TLS_DTPOFF32 = 83 # macro
R_TILEPRO_TLS_TPOFF32 = 84 # macro
R_TILEPRO_IMM16_X0_TLS_LE = 85 # macro
R_TILEPRO_IMM16_X1_TLS_LE = 86 # macro
R_TILEPRO_IMM16_X0_TLS_LE_LO = 87 # macro
R_TILEPRO_IMM16_X1_TLS_LE_LO = 88 # macro
R_TILEPRO_IMM16_X0_TLS_LE_HI = 89 # macro
R_TILEPRO_IMM16_X1_TLS_LE_HI = 90 # macro
R_TILEPRO_IMM16_X0_TLS_LE_HA = 91 # macro
R_TILEPRO_IMM16_X1_TLS_LE_HA = 92 # macro
R_TILEPRO_GNU_VTINHERIT = 128 # macro
R_TILEPRO_GNU_VTENTRY = 129 # macro
R_TILEPRO_NUM = 130 # macro
R_TILEGX_NONE = 0 # macro
R_TILEGX_64 = 1 # macro
R_TILEGX_32 = 2 # macro
R_TILEGX_16 = 3 # macro
R_TILEGX_8 = 4 # macro
R_TILEGX_64_PCREL = 5 # macro
R_TILEGX_32_PCREL = 6 # macro
R_TILEGX_16_PCREL = 7 # macro
R_TILEGX_8_PCREL = 8 # macro
R_TILEGX_HW0 = 9 # macro
R_TILEGX_HW1 = 10 # macro
R_TILEGX_HW2 = 11 # macro
R_TILEGX_HW3 = 12 # macro
R_TILEGX_HW0_LAST = 13 # macro
R_TILEGX_HW1_LAST = 14 # macro
R_TILEGX_HW2_LAST = 15 # macro
R_TILEGX_COPY = 16 # macro
R_TILEGX_GLOB_DAT = 17 # macro
R_TILEGX_JMP_SLOT = 18 # macro
R_TILEGX_RELATIVE = 19 # macro
R_TILEGX_BROFF_X1 = 20 # macro
R_TILEGX_JUMPOFF_X1 = 21 # macro
R_TILEGX_JUMPOFF_X1_PLT = 22 # macro
R_TILEGX_IMM8_X0 = 23 # macro
R_TILEGX_IMM8_Y0 = 24 # macro
R_TILEGX_IMM8_X1 = 25 # macro
R_TILEGX_IMM8_Y1 = 26 # macro
R_TILEGX_DEST_IMM8_X1 = 27 # macro
R_TILEGX_MT_IMM14_X1 = 28 # macro
R_TILEGX_MF_IMM14_X1 = 29 # macro
R_TILEGX_MMSTART_X0 = 30 # macro
R_TILEGX_MMEND_X0 = 31 # macro
R_TILEGX_SHAMT_X0 = 32 # macro
R_TILEGX_SHAMT_X1 = 33 # macro
R_TILEGX_SHAMT_Y0 = 34 # macro
R_TILEGX_SHAMT_Y1 = 35 # macro
R_TILEGX_IMM16_X0_HW0 = 36 # macro
R_TILEGX_IMM16_X1_HW0 = 37 # macro
R_TILEGX_IMM16_X0_HW1 = 38 # macro
R_TILEGX_IMM16_X1_HW1 = 39 # macro
R_TILEGX_IMM16_X0_HW2 = 40 # macro
R_TILEGX_IMM16_X1_HW2 = 41 # macro
R_TILEGX_IMM16_X0_HW3 = 42 # macro
R_TILEGX_IMM16_X1_HW3 = 43 # macro
R_TILEGX_IMM16_X0_HW0_LAST = 44 # macro
R_TILEGX_IMM16_X1_HW0_LAST = 45 # macro
R_TILEGX_IMM16_X0_HW1_LAST = 46 # macro
R_TILEGX_IMM16_X1_HW1_LAST = 47 # macro
R_TILEGX_IMM16_X0_HW2_LAST = 48 # macro
R_TILEGX_IMM16_X1_HW2_LAST = 49 # macro
R_TILEGX_IMM16_X0_HW0_PCREL = 50 # macro
R_TILEGX_IMM16_X1_HW0_PCREL = 51 # macro
R_TILEGX_IMM16_X0_HW1_PCREL = 52 # macro
R_TILEGX_IMM16_X1_HW1_PCREL = 53 # macro
R_TILEGX_IMM16_X0_HW2_PCREL = 54 # macro
R_TILEGX_IMM16_X1_HW2_PCREL = 55 # macro
R_TILEGX_IMM16_X0_HW3_PCREL = 56 # macro
R_TILEGX_IMM16_X1_HW3_PCREL = 57 # macro
R_TILEGX_IMM16_X0_HW0_LAST_PCREL = 58 # macro
R_TILEGX_IMM16_X1_HW0_LAST_PCREL = 59 # macro
R_TILEGX_IMM16_X0_HW1_LAST_PCREL = 60 # macro
R_TILEGX_IMM16_X1_HW1_LAST_PCREL = 61 # macro
R_TILEGX_IMM16_X0_HW2_LAST_PCREL = 62 # macro
R_TILEGX_IMM16_X1_HW2_LAST_PCREL = 63 # macro
R_TILEGX_IMM16_X0_HW0_GOT = 64 # macro
R_TILEGX_IMM16_X1_HW0_GOT = 65 # macro
R_TILEGX_IMM16_X0_HW0_PLT_PCREL = 66 # macro
R_TILEGX_IMM16_X1_HW0_PLT_PCREL = 67 # macro
R_TILEGX_IMM16_X0_HW1_PLT_PCREL = 68 # macro
R_TILEGX_IMM16_X1_HW1_PLT_PCREL = 69 # macro
R_TILEGX_IMM16_X0_HW2_PLT_PCREL = 70 # macro
R_TILEGX_IMM16_X1_HW2_PLT_PCREL = 71 # macro
R_TILEGX_IMM16_X0_HW0_LAST_GOT = 72 # macro
R_TILEGX_IMM16_X1_HW0_LAST_GOT = 73 # macro
R_TILEGX_IMM16_X0_HW1_LAST_GOT = 74 # macro
R_TILEGX_IMM16_X1_HW1_LAST_GOT = 75 # macro
R_TILEGX_IMM16_X0_HW3_PLT_PCREL = 76 # macro
R_TILEGX_IMM16_X1_HW3_PLT_PCREL = 77 # macro
R_TILEGX_IMM16_X0_HW0_TLS_GD = 78 # macro
R_TILEGX_IMM16_X1_HW0_TLS_GD = 79 # macro
R_TILEGX_IMM16_X0_HW0_TLS_LE = 80 # macro
R_TILEGX_IMM16_X1_HW0_TLS_LE = 81 # macro
R_TILEGX_IMM16_X0_HW0_LAST_TLS_LE = 82 # macro
R_TILEGX_IMM16_X1_HW0_LAST_TLS_LE = 83 # macro
R_TILEGX_IMM16_X0_HW1_LAST_TLS_LE = 84 # macro
R_TILEGX_IMM16_X1_HW1_LAST_TLS_LE = 85 # macro
R_TILEGX_IMM16_X0_HW0_LAST_TLS_GD = 86 # macro
R_TILEGX_IMM16_X1_HW0_LAST_TLS_GD = 87 # macro
R_TILEGX_IMM16_X0_HW1_LAST_TLS_GD = 88 # macro
R_TILEGX_IMM16_X1_HW1_LAST_TLS_GD = 89 # macro
R_TILEGX_IMM16_X0_HW0_TLS_IE = 92 # macro
R_TILEGX_IMM16_X1_HW0_TLS_IE = 93 # macro
R_TILEGX_IMM16_X0_HW0_LAST_PLT_PCREL = 94 # macro
R_TILEGX_IMM16_X1_HW0_LAST_PLT_PCREL = 95 # macro
R_TILEGX_IMM16_X0_HW1_LAST_PLT_PCREL = 96 # macro
R_TILEGX_IMM16_X1_HW1_LAST_PLT_PCREL = 97 # macro
R_TILEGX_IMM16_X0_HW2_LAST_PLT_PCREL = 98 # macro
R_TILEGX_IMM16_X1_HW2_LAST_PLT_PCREL = 99 # macro
R_TILEGX_IMM16_X0_HW0_LAST_TLS_IE = 100 # macro
R_TILEGX_IMM16_X1_HW0_LAST_TLS_IE = 101 # macro
R_TILEGX_IMM16_X0_HW1_LAST_TLS_IE = 102 # macro
R_TILEGX_IMM16_X1_HW1_LAST_TLS_IE = 103 # macro
R_TILEGX_TLS_DTPMOD64 = 106 # macro
R_TILEGX_TLS_DTPOFF64 = 107 # macro
R_TILEGX_TLS_TPOFF64 = 108 # macro
R_TILEGX_TLS_DTPMOD32 = 109 # macro
R_TILEGX_TLS_DTPOFF32 = 110 # macro
R_TILEGX_TLS_TPOFF32 = 111 # macro
R_TILEGX_TLS_GD_CALL = 112 # macro
R_TILEGX_IMM8_X0_TLS_GD_ADD = 113 # macro
R_TILEGX_IMM8_X1_TLS_GD_ADD = 114 # macro
R_TILEGX_IMM8_Y0_TLS_GD_ADD = 115 # macro
R_TILEGX_IMM8_Y1_TLS_GD_ADD = 116 # macro
R_TILEGX_TLS_IE_LOAD = 117 # macro
R_TILEGX_IMM8_X0_TLS_ADD = 118 # macro
R_TILEGX_IMM8_X1_TLS_ADD = 119 # macro
R_TILEGX_IMM8_Y0_TLS_ADD = 120 # macro
R_TILEGX_IMM8_Y1_TLS_ADD = 121 # macro
R_TILEGX_GNU_VTINHERIT = 128 # macro
R_TILEGX_GNU_VTENTRY = 129 # macro
R_TILEGX_NUM = 130 # macro
EF_RISCV_RVC = 0x0001 # macro
EF_RISCV_FLOAT_ABI = 0x0006 # macro
EF_RISCV_FLOAT_ABI_SOFT = 0x0000 # macro
EF_RISCV_FLOAT_ABI_SINGLE = 0x0002 # macro
EF_RISCV_FLOAT_ABI_DOUBLE = 0x0004 # macro
EF_RISCV_FLOAT_ABI_QUAD = 0x0006 # macro
EF_RISCV_RVE = 0x0008 # macro
EF_RISCV_TSO = 0x0010 # macro
R_RISCV_NONE = 0 # macro
R_RISCV_32 = 1 # macro
R_RISCV_64 = 2 # macro
R_RISCV_RELATIVE = 3 # macro
R_RISCV_COPY = 4 # macro
R_RISCV_JUMP_SLOT = 5 # macro
R_RISCV_TLS_DTPMOD32 = 6 # macro
R_RISCV_TLS_DTPMOD64 = 7 # macro
R_RISCV_TLS_DTPREL32 = 8 # macro
R_RISCV_TLS_DTPREL64 = 9 # macro
R_RISCV_TLS_TPREL32 = 10 # macro
R_RISCV_TLS_TPREL64 = 11 # macro
R_RISCV_BRANCH = 16 # macro
R_RISCV_JAL = 17 # macro
R_RISCV_CALL = 18 # macro
R_RISCV_CALL_PLT = 19 # macro
R_RISCV_GOT_HI20 = 20 # macro
R_RISCV_TLS_GOT_HI20 = 21 # macro
R_RISCV_TLS_GD_HI20 = 22 # macro
R_RISCV_PCREL_HI20 = 23 # macro
R_RISCV_PCREL_LO12_I = 24 # macro
R_RISCV_PCREL_LO12_S = 25 # macro
R_RISCV_HI20 = 26 # macro
R_RISCV_LO12_I = 27 # macro
R_RISCV_LO12_S = 28 # macro
R_RISCV_TPREL_HI20 = 29 # macro
R_RISCV_TPREL_LO12_I = 30 # macro
R_RISCV_TPREL_LO12_S = 31 # macro
R_RISCV_TPREL_ADD = 32 # macro
R_RISCV_ADD8 = 33 # macro
R_RISCV_ADD16 = 34 # macro
R_RISCV_ADD32 = 35 # macro
R_RISCV_ADD64 = 36 # macro
R_RISCV_SUB8 = 37 # macro
R_RISCV_SUB16 = 38 # macro
R_RISCV_SUB32 = 39 # macro
R_RISCV_SUB64 = 40 # macro
R_RISCV_GNU_VTINHERIT = 41 # macro
R_RISCV_GNU_VTENTRY = 42 # macro
R_RISCV_ALIGN = 43 # macro
R_RISCV_RVC_BRANCH = 44 # macro
R_RISCV_RVC_JUMP = 45 # macro
R_RISCV_RVC_LUI = 46 # macro
R_RISCV_GPREL_I = 47 # macro
R_RISCV_GPREL_S = 48 # macro
R_RISCV_TPREL_I = 49 # macro
R_RISCV_TPREL_S = 50 # macro
R_RISCV_RELAX = 51 # macro
R_RISCV_SUB6 = 52 # macro
R_RISCV_SET6 = 53 # macro
R_RISCV_SET8 = 54 # macro
R_RISCV_SET16 = 55 # macro
R_RISCV_SET32 = 56 # macro
R_RISCV_32_PCREL = 57 # macro
R_RISCV_IRELATIVE = 58 # macro
R_RISCV_PLT32 = 59 # macro
R_RISCV_SET_ULEB128 = 60 # macro
R_RISCV_SUB_ULEB128 = 61 # macro
R_RISCV_NUM = 62 # macro
STO_RISCV_VARIANT_CC = 0x80 # macro
SHT_RISCV_ATTRIBUTES = (0x70000000+3) # macro
PT_RISCV_ATTRIBUTES = (0x70000000+3) # macro
DT_RISCV_VARIANT_CC = (0x70000000+1) # macro
R_BPF_NONE = 0 # macro
R_BPF_64_64 = 1 # macro
R_BPF_64_32 = 10 # macro
R_METAG_HIADDR16 = 0 # macro
R_METAG_LOADDR16 = 1 # macro
R_METAG_ADDR32 = 2 # macro
R_METAG_NONE = 3 # macro
R_METAG_RELBRANCH = 4 # macro
R_METAG_GETSETOFF = 5 # macro
R_METAG_REG32OP1 = 6 # macro
R_METAG_REG32OP2 = 7 # macro
R_METAG_REG32OP3 = 8 # macro
R_METAG_REG16OP1 = 9 # macro
R_METAG_REG16OP2 = 10 # macro
R_METAG_REG16OP3 = 11 # macro
R_METAG_REG32OP4 = 12 # macro
R_METAG_HIOG = 13 # macro
R_METAG_LOOG = 14 # macro
R_METAG_REL8 = 15 # macro
R_METAG_REL16 = 16 # macro
R_METAG_GNU_VTINHERIT = 30 # macro
R_METAG_GNU_VTENTRY = 31 # macro
R_METAG_HI16_GOTOFF = 32 # macro
R_METAG_LO16_GOTOFF = 33 # macro
R_METAG_GETSET_GOTOFF = 34 # macro
R_METAG_GETSET_GOT = 35 # macro
R_METAG_HI16_GOTPC = 36 # macro
R_METAG_LO16_GOTPC = 37 # macro
R_METAG_HI16_PLT = 38 # macro
R_METAG_LO16_PLT = 39 # macro
R_METAG_RELBRANCH_PLT = 40 # macro
R_METAG_GOTOFF = 41 # macro
R_METAG_PLT = 42 # macro
R_METAG_COPY = 43 # macro
R_METAG_JMP_SLOT = 44 # macro
R_METAG_RELATIVE = 45 # macro
R_METAG_GLOB_DAT = 46 # macro
R_METAG_TLS_GD = 47 # macro
R_METAG_TLS_LDM = 48 # macro
R_METAG_TLS_LDO_HI16 = 49 # macro
R_METAG_TLS_LDO_LO16 = 50 # macro
R_METAG_TLS_LDO = 51 # macro
R_METAG_TLS_IE = 52 # macro
R_METAG_TLS_IENONPIC = 53 # macro
R_METAG_TLS_IENONPIC_HI16 = 54 # macro
R_METAG_TLS_IENONPIC_LO16 = 55 # macro
R_METAG_TLS_TPOFF = 56 # macro
R_METAG_TLS_DTPMOD = 57 # macro
R_METAG_TLS_DTPOFF = 58 # macro
R_METAG_TLS_LE = 59 # macro
R_METAG_TLS_LE_HI16 = 60 # macro
R_METAG_TLS_LE_LO16 = 61 # macro
R_NDS32_NONE = 0 # macro
R_NDS32_32_RELA = 20 # macro
R_NDS32_COPY = 39 # macro
R_NDS32_GLOB_DAT = 40 # macro
R_NDS32_JMP_SLOT = 41 # macro
R_NDS32_RELATIVE = 42 # macro
R_NDS32_TLS_TPOFF = 102 # macro
R_NDS32_TLS_DESC = 119 # macro
EF_LARCH_ABI_MODIFIER_MASK = 0x07 # macro
EF_LARCH_ABI_SOFT_FLOAT = 0x01 # macro
EF_LARCH_ABI_SINGLE_FLOAT = 0x02 # macro
EF_LARCH_ABI_DOUBLE_FLOAT = 0x03 # macro
EF_LARCH_OBJABI_V1 = 0x40 # macro
R_LARCH_NONE = 0 # macro
R_LARCH_32 = 1 # macro
R_LARCH_64 = 2 # macro
R_LARCH_RELATIVE = 3 # macro
R_LARCH_COPY = 4 # macro
R_LARCH_JUMP_SLOT = 5 # macro
R_LARCH_TLS_DTPMOD32 = 6 # macro
R_LARCH_TLS_DTPMOD64 = 7 # macro
R_LARCH_TLS_DTPREL32 = 8 # macro
R_LARCH_TLS_DTPREL64 = 9 # macro
R_LARCH_TLS_TPREL32 = 10 # macro
R_LARCH_TLS_TPREL64 = 11 # macro
R_LARCH_IRELATIVE = 12 # macro
R_LARCH_MARK_LA = 20 # macro
R_LARCH_MARK_PCREL = 21 # macro
R_LARCH_SOP_PUSH_PCREL = 22 # macro
R_LARCH_SOP_PUSH_ABSOLUTE = 23 # macro
R_LARCH_SOP_PUSH_DUP = 24 # macro
R_LARCH_SOP_PUSH_GPREL = 25 # macro
R_LARCH_SOP_PUSH_TLS_TPREL = 26 # macro
R_LARCH_SOP_PUSH_TLS_GOT = 27 # macro
R_LARCH_SOP_PUSH_TLS_GD = 28 # macro
R_LARCH_SOP_PUSH_PLT_PCREL = 29 # macro
R_LARCH_SOP_ASSERT = 30 # macro
R_LARCH_SOP_NOT = 31 # macro
R_LARCH_SOP_SUB = 32 # macro
R_LARCH_SOP_SL = 33 # macro
R_LARCH_SOP_SR = 34 # macro
R_LARCH_SOP_ADD = 35 # macro
R_LARCH_SOP_AND = 36 # macro
R_LARCH_SOP_IF_ELSE = 37 # macro
R_LARCH_SOP_POP_32_S_10_5 = 38 # macro
R_LARCH_SOP_POP_32_U_10_12 = 39 # macro
R_LARCH_SOP_POP_32_S_10_12 = 40 # macro
R_LARCH_SOP_POP_32_S_10_16 = 41 # macro
R_LARCH_SOP_POP_32_S_10_16_S2 = 42 # macro
R_LARCH_SOP_POP_32_S_5_20 = 43 # macro
R_LARCH_SOP_POP_32_S_0_5_10_16_S2 = 44 # macro
R_LARCH_SOP_POP_32_S_0_10_10_16_S2 = 45 # macro
R_LARCH_SOP_POP_32_U = 46 # macro
R_LARCH_ADD8 = 47 # macro
R_LARCH_ADD16 = 48 # macro
R_LARCH_ADD24 = 49 # macro
R_LARCH_ADD32 = 50 # macro
R_LARCH_ADD64 = 51 # macro
R_LARCH_SUB8 = 52 # macro
R_LARCH_SUB16 = 53 # macro
R_LARCH_SUB24 = 54 # macro
R_LARCH_SUB32 = 55 # macro
R_LARCH_SUB64 = 56 # macro
R_LARCH_GNU_VTINHERIT = 57 # macro
R_LARCH_GNU_VTENTRY = 58 # macro
R_LARCH_B16 = 64 # macro
R_LARCH_B21 = 65 # macro
R_LARCH_B26 = 66 # macro
R_LARCH_ABS_HI20 = 67 # macro
R_LARCH_ABS_LO12 = 68 # macro
R_LARCH_ABS64_LO20 = 69 # macro
R_LARCH_ABS64_HI12 = 70 # macro
R_LARCH_PCALA_HI20 = 71 # macro
R_LARCH_PCALA_LO12 = 72 # macro
R_LARCH_PCALA64_LO20 = 73 # macro
R_LARCH_PCALA64_HI12 = 74 # macro
R_LARCH_GOT_PC_HI20 = 75 # macro
R_LARCH_GOT_PC_LO12 = 76 # macro
R_LARCH_GOT64_PC_LO20 = 77 # macro
R_LARCH_GOT64_PC_HI12 = 78 # macro
R_LARCH_GOT_HI20 = 79 # macro
R_LARCH_GOT_LO12 = 80 # macro
R_LARCH_GOT64_LO20 = 81 # macro
R_LARCH_GOT64_HI12 = 82 # macro
R_LARCH_TLS_LE_HI20 = 83 # macro
R_LARCH_TLS_LE_LO12 = 84 # macro
R_LARCH_TLS_LE64_LO20 = 85 # macro
R_LARCH_TLS_LE64_HI12 = 86 # macro
R_LARCH_TLS_IE_PC_HI20 = 87 # macro
R_LARCH_TLS_IE_PC_LO12 = 88 # macro
R_LARCH_TLS_IE64_PC_LO20 = 89 # macro
R_LARCH_TLS_IE64_PC_HI12 = 90 # macro
R_LARCH_TLS_IE_HI20 = 91 # macro
R_LARCH_TLS_IE_LO12 = 92 # macro
R_LARCH_TLS_IE64_LO20 = 93 # macro
R_LARCH_TLS_IE64_HI12 = 94 # macro
R_LARCH_TLS_LD_PC_HI20 = 95 # macro
R_LARCH_TLS_LD_HI20 = 96 # macro
R_LARCH_TLS_GD_PC_HI20 = 97 # macro
R_LARCH_TLS_GD_HI20 = 98 # macro
R_LARCH_32_PCREL = 99 # macro
R_LARCH_RELAX = 100 # macro
R_LARCH_DELETE = 101 # macro
R_LARCH_ALIGN = 102 # macro
R_LARCH_PCREL20_S2 = 103 # macro
R_LARCH_CFA = 104 # macro
R_LARCH_ADD6 = 105 # macro
R_LARCH_SUB6 = 106 # macro
R_LARCH_ADD_ULEB128 = 107 # macro
R_LARCH_SUB_ULEB128 = 108 # macro
R_LARCH_64_PCREL = 109 # macro
EF_ARC_MACH_MSK = 0x000000ff # macro
EF_ARC_OSABI_MSK = 0x00000f00 # macro
EF_ARC_ALL_MSK = (0x000000ff|0x00000f00) # macro
SHT_ARC_ATTRIBUTES = (0x70000000+1) # macro
R_ARC_NONE = 0x0 # macro
R_ARC_8 = 0x1 # macro
R_ARC_16 = 0x2 # macro
R_ARC_24 = 0x3 # macro
R_ARC_32 = 0x4 # macro
R_ARC_B22_PCREL = 0x6 # macro
R_ARC_H30 = 0x7 # macro
R_ARC_N8 = 0x8 # macro
R_ARC_N16 = 0x9 # macro
R_ARC_N24 = 0xA # macro
R_ARC_N32 = 0xB # macro
R_ARC_SDA = 0xC # macro
R_ARC_SECTOFF = 0xD # macro
R_ARC_S21H_PCREL = 0xE # macro
R_ARC_S21W_PCREL = 0xF # macro
R_ARC_S25H_PCREL = 0x10 # macro
R_ARC_S25W_PCREL = 0x11 # macro
R_ARC_SDA32 = 0x12 # macro
R_ARC_SDA_LDST = 0x13 # macro
R_ARC_SDA_LDST1 = 0x14 # macro
R_ARC_SDA_LDST2 = 0x15 # macro
R_ARC_SDA16_LD = 0x16 # macro
R_ARC_SDA16_LD1 = 0x17 # macro
R_ARC_SDA16_LD2 = 0x18 # macro
R_ARC_S13_PCREL = 0x19 # macro
R_ARC_W = 0x1A # macro
R_ARC_32_ME = 0x1B # macro
R_ARC_N32_ME = 0x1C # macro
R_ARC_SECTOFF_ME = 0x1D # macro
R_ARC_SDA32_ME = 0x1E # macro
R_ARC_W_ME = 0x1F # macro
R_ARC_H30_ME = 0x20 # macro
R_ARC_SECTOFF_U8 = 0x21 # macro
R_ARC_SECTOFF_S9 = 0x22 # macro
R_AC_SECTOFF_U8 = 0x23 # macro
R_AC_SECTOFF_U8_1 = 0x24 # macro
R_AC_SECTOFF_U8_2 = 0x25 # macro
R_AC_SECTOFF_S9 = 0x26 # macro
R_AC_SECTOFF_S9_1 = 0x27 # macro
R_AC_SECTOFF_S9_2 = 0x28 # macro
R_ARC_SECTOFF_ME_1 = 0x29 # macro
R_ARC_SECTOFF_ME_2 = 0x2A # macro
R_ARC_SECTOFF_1 = 0x2B # macro
R_ARC_SECTOFF_2 = 0x2C # macro
R_ARC_SDA_12 = 0x2D # macro
R_ARC_SDA16_ST2 = 0x30 # macro
R_ARC_32_PCREL = 0x31 # macro
R_ARC_PC32 = 0x32 # macro
R_ARC_GOTPC32 = 0x33 # macro
R_ARC_PLT32 = 0x34 # macro
R_ARC_COPY = 0x35 # macro
R_ARC_GLOB_DAT = 0x36 # macro
R_ARC_JMP_SLOT = 0x37 # macro
R_ARC_RELATIVE = 0x38 # macro
R_ARC_GOTOFF = 0x39 # macro
R_ARC_GOTPC = 0x3A # macro
R_ARC_GOT32 = 0x3B # macro
R_ARC_S21W_PCREL_PLT = 0x3C # macro
R_ARC_S25H_PCREL_PLT = 0x3D # macro
R_ARC_JLI_SECTOFF = 0x3F # macro
R_ARC_TLS_DTPMOD = 0x42 # macro
R_ARC_TLS_DTPOFF = 0x43 # macro
R_ARC_TLS_TPOFF = 0x44 # macro
R_ARC_TLS_GD_GOT = 0x45 # macro
R_ARC_TLS_GD_LD = 0x46 # macro
R_ARC_TLS_GD_CALL = 0x47 # macro
R_ARC_TLS_IE_GOT = 0x48 # macro
R_ARC_TLS_DTPOFF_S9 = 0x49 # macro
R_ARC_TLS_LE_S9 = 0x4A # macro
R_ARC_TLS_LE_32 = 0x4B # macro
R_ARC_S25W_PCREL_PLT = 0x4C # macro
R_ARC_S21H_PCREL_PLT = 0x4D # macro
R_ARC_NPS_CMEM16 = 0x4E # macro
R_OR1K_NONE = 0 # macro
R_OR1K_32 = 1 # macro
R_OR1K_16 = 2 # macro
R_OR1K_8 = 3 # macro
R_OR1K_LO_16_IN_INSN = 4 # macro
R_OR1K_HI_16_IN_INSN = 5 # macro
R_OR1K_INSN_REL_26 = 6 # macro
R_OR1K_GNU_VTENTRY = 7 # macro
R_OR1K_GNU_VTINHERIT = 8 # macro
R_OR1K_32_PCREL = 9 # macro
R_OR1K_16_PCREL = 10 # macro
R_OR1K_8_PCREL = 11 # macro
R_OR1K_GOTPC_HI16 = 12 # macro
R_OR1K_GOTPC_LO16 = 13 # macro
R_OR1K_GOT16 = 14 # macro
R_OR1K_PLT26 = 15 # macro
R_OR1K_GOTOFF_HI16 = 16 # macro
R_OR1K_GOTOFF_LO16 = 17 # macro
R_OR1K_COPY = 18 # macro
R_OR1K_GLOB_DAT = 19 # macro
R_OR1K_JMP_SLOT = 20 # macro
R_OR1K_RELATIVE = 21 # macro
R_OR1K_TLS_GD_HI16 = 22 # macro
R_OR1K_TLS_GD_LO16 = 23 # macro
R_OR1K_TLS_LDM_HI16 = 24 # macro
R_OR1K_TLS_LDM_LO16 = 25 # macro
R_OR1K_TLS_LDO_HI16 = 26 # macro
R_OR1K_TLS_LDO_LO16 = 27 # macro
R_OR1K_TLS_IE_HI16 = 28 # macro
R_OR1K_TLS_IE_LO16 = 29 # macro
R_OR1K_TLS_LE_HI16 = 30 # macro
R_OR1K_TLS_LE_LO16 = 31 # macro
R_OR1K_TLS_TPOFF = 32 # macro
R_OR1K_TLS_DTPOFF = 33 # macro
R_OR1K_TLS_DTPMOD = 34 # macro
Elf32_Half = ctypes.c_uint16
Elf64_Half = ctypes.c_uint16
Elf32_Word = ctypes.c_uint32
Elf32_Sword = ctypes.c_int32
def DT_EXTRATAGIDX(tag):  # macro
   return ((Elf32_Word)-((Elf32_Sword)(tag)<<1>>1)-1)
Elf64_Word = ctypes.c_uint32
Elf64_Sword = ctypes.c_int32
Elf32_Xword = ctypes.c_uint64
Elf32_Sxword = ctypes.c_int64
Elf64_Xword = ctypes.c_uint64
def ELF64_R_INFO(sym, type):  # macro
   return ((((Elf64_Xword)(sym))<<32)+(type))
Elf64_Sxword = ctypes.c_int64
Elf32_Addr = ctypes.c_uint32
Elf64_Addr = ctypes.c_uint64
Elf32_Off = ctypes.c_uint32
Elf64_Off = ctypes.c_uint64
Elf32_Section = ctypes.c_uint16
Elf64_Section = ctypes.c_uint16
Elf32_Versym = ctypes.c_uint16
Elf64_Versym = ctypes.c_uint16
class struct_c__SA_Elf32_Ehdr(Structure):
    pass

struct_c__SA_Elf32_Ehdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Ehdr._fields_ = [
    ('e_ident', ctypes.c_ubyte * 16),
    ('e_type', ctypes.c_uint16),
    ('e_machine', ctypes.c_uint16),
    ('e_version', ctypes.c_uint32),
    ('e_entry', ctypes.c_uint32),
    ('e_phoff', ctypes.c_uint32),
    ('e_shoff', ctypes.c_uint32),
    ('e_flags', ctypes.c_uint32),
    ('e_ehsize', ctypes.c_uint16),
    ('e_phentsize', ctypes.c_uint16),
    ('e_phnum', ctypes.c_uint16),
    ('e_shentsize', ctypes.c_uint16),
    ('e_shnum', ctypes.c_uint16),
    ('e_shstrndx', ctypes.c_uint16),
]

Elf32_Ehdr = struct_c__SA_Elf32_Ehdr
class struct_c__SA_Elf64_Ehdr(Structure):
    pass

struct_c__SA_Elf64_Ehdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Ehdr._fields_ = [
    ('e_ident', ctypes.c_ubyte * 16),
    ('e_type', ctypes.c_uint16),
    ('e_machine', ctypes.c_uint16),
    ('e_version', ctypes.c_uint32),
    ('e_entry', ctypes.c_uint64),
    ('e_phoff', ctypes.c_uint64),
    ('e_shoff', ctypes.c_uint64),
    ('e_flags', ctypes.c_uint32),
    ('e_ehsize', ctypes.c_uint16),
    ('e_phentsize', ctypes.c_uint16),
    ('e_phnum', ctypes.c_uint16),
    ('e_shentsize', ctypes.c_uint16),
    ('e_shnum', ctypes.c_uint16),
    ('e_shstrndx', ctypes.c_uint16),
]

Elf64_Ehdr = struct_c__SA_Elf64_Ehdr
class struct_c__SA_Elf32_Shdr(Structure):
    pass

struct_c__SA_Elf32_Shdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Shdr._fields_ = [
    ('sh_name', ctypes.c_uint32),
    ('sh_type', ctypes.c_uint32),
    ('sh_flags', ctypes.c_uint32),
    ('sh_addr', ctypes.c_uint32),
    ('sh_offset', ctypes.c_uint32),
    ('sh_size', ctypes.c_uint32),
    ('sh_link', ctypes.c_uint32),
    ('sh_info', ctypes.c_uint32),
    ('sh_addralign', ctypes.c_uint32),
    ('sh_entsize', ctypes.c_uint32),
]

Elf32_Shdr = struct_c__SA_Elf32_Shdr
class struct_c__SA_Elf64_Shdr(Structure):
    pass

struct_c__SA_Elf64_Shdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Shdr._fields_ = [
    ('sh_name', ctypes.c_uint32),
    ('sh_type', ctypes.c_uint32),
    ('sh_flags', ctypes.c_uint64),
    ('sh_addr', ctypes.c_uint64),
    ('sh_offset', ctypes.c_uint64),
    ('sh_size', ctypes.c_uint64),
    ('sh_link', ctypes.c_uint32),
    ('sh_info', ctypes.c_uint32),
    ('sh_addralign', ctypes.c_uint64),
    ('sh_entsize', ctypes.c_uint64),
]

Elf64_Shdr = struct_c__SA_Elf64_Shdr
class struct_c__SA_Elf32_Chdr(Structure):
    pass

struct_c__SA_Elf32_Chdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Chdr._fields_ = [
    ('ch_type', ctypes.c_uint32),
    ('ch_size', ctypes.c_uint32),
    ('ch_addralign', ctypes.c_uint32),
]

Elf32_Chdr = struct_c__SA_Elf32_Chdr
class struct_c__SA_Elf64_Chdr(Structure):
    pass

struct_c__SA_Elf64_Chdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Chdr._fields_ = [
    ('ch_type', ctypes.c_uint32),
    ('ch_reserved', ctypes.c_uint32),
    ('ch_size', ctypes.c_uint64),
    ('ch_addralign', ctypes.c_uint64),
]

Elf64_Chdr = struct_c__SA_Elf64_Chdr
class struct_c__SA_Elf32_Sym(Structure):
    pass

struct_c__SA_Elf32_Sym._pack_ = 1 # source:False
struct_c__SA_Elf32_Sym._fields_ = [
    ('st_name', ctypes.c_uint32),
    ('st_value', ctypes.c_uint32),
    ('st_size', ctypes.c_uint32),
    ('st_info', ctypes.c_ubyte),
    ('st_other', ctypes.c_ubyte),
    ('st_shndx', ctypes.c_uint16),
]

Elf32_Sym = struct_c__SA_Elf32_Sym
class struct_c__SA_Elf64_Sym(Structure):
    pass

struct_c__SA_Elf64_Sym._pack_ = 1 # source:False
struct_c__SA_Elf64_Sym._fields_ = [
    ('st_name', ctypes.c_uint32),
    ('st_info', ctypes.c_ubyte),
    ('st_other', ctypes.c_ubyte),
    ('st_shndx', ctypes.c_uint16),
    ('st_value', ctypes.c_uint64),
    ('st_size', ctypes.c_uint64),
]

Elf64_Sym = struct_c__SA_Elf64_Sym
class struct_c__SA_Elf32_Syminfo(Structure):
    pass

struct_c__SA_Elf32_Syminfo._pack_ = 1 # source:False
struct_c__SA_Elf32_Syminfo._fields_ = [
    ('si_boundto', ctypes.c_uint16),
    ('si_flags', ctypes.c_uint16),
]

Elf32_Syminfo = struct_c__SA_Elf32_Syminfo
class struct_c__SA_Elf64_Syminfo(Structure):
    pass

struct_c__SA_Elf64_Syminfo._pack_ = 1 # source:False
struct_c__SA_Elf64_Syminfo._fields_ = [
    ('si_boundto', ctypes.c_uint16),
    ('si_flags', ctypes.c_uint16),
]

Elf64_Syminfo = struct_c__SA_Elf64_Syminfo
class struct_c__SA_Elf32_Rel(Structure):
    pass

struct_c__SA_Elf32_Rel._pack_ = 1 # source:False
struct_c__SA_Elf32_Rel._fields_ = [
    ('r_offset', ctypes.c_uint32),
    ('r_info', ctypes.c_uint32),
]

Elf32_Rel = struct_c__SA_Elf32_Rel
class struct_c__SA_Elf64_Rel(Structure):
    pass

struct_c__SA_Elf64_Rel._pack_ = 1 # source:False
struct_c__SA_Elf64_Rel._fields_ = [
    ('r_offset', ctypes.c_uint64),
    ('r_info', ctypes.c_uint64),
]

Elf64_Rel = struct_c__SA_Elf64_Rel
class struct_c__SA_Elf32_Rela(Structure):
    pass

struct_c__SA_Elf32_Rela._pack_ = 1 # source:False
struct_c__SA_Elf32_Rela._fields_ = [
    ('r_offset', ctypes.c_uint32),
    ('r_info', ctypes.c_uint32),
    ('r_addend', ctypes.c_int32),
]

Elf32_Rela = struct_c__SA_Elf32_Rela
class struct_c__SA_Elf64_Rela(Structure):
    pass

struct_c__SA_Elf64_Rela._pack_ = 1 # source:False
struct_c__SA_Elf64_Rela._fields_ = [
    ('r_offset', ctypes.c_uint64),
    ('r_info', ctypes.c_uint64),
    ('r_addend', ctypes.c_int64),
]

Elf64_Rela = struct_c__SA_Elf64_Rela
Elf32_Relr = ctypes.c_uint32
Elf64_Relr = ctypes.c_uint64
class struct_c__SA_Elf32_Phdr(Structure):
    pass

struct_c__SA_Elf32_Phdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Phdr._fields_ = [
    ('p_type', ctypes.c_uint32),
    ('p_offset', ctypes.c_uint32),
    ('p_vaddr', ctypes.c_uint32),
    ('p_paddr', ctypes.c_uint32),
    ('p_filesz', ctypes.c_uint32),
    ('p_memsz', ctypes.c_uint32),
    ('p_flags', ctypes.c_uint32),
    ('p_align', ctypes.c_uint32),
]

Elf32_Phdr = struct_c__SA_Elf32_Phdr
class struct_c__SA_Elf64_Phdr(Structure):
    pass

struct_c__SA_Elf64_Phdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Phdr._fields_ = [
    ('p_type', ctypes.c_uint32),
    ('p_flags', ctypes.c_uint32),
    ('p_offset', ctypes.c_uint64),
    ('p_vaddr', ctypes.c_uint64),
    ('p_paddr', ctypes.c_uint64),
    ('p_filesz', ctypes.c_uint64),
    ('p_memsz', ctypes.c_uint64),
    ('p_align', ctypes.c_uint64),
]

Elf64_Phdr = struct_c__SA_Elf64_Phdr
class struct_c__SA_Elf32_Dyn(Structure):
    pass

class union_c__SA_Elf32_Dyn_d_un(Union):
    pass

union_c__SA_Elf32_Dyn_d_un._pack_ = 1 # source:False
union_c__SA_Elf32_Dyn_d_un._fields_ = [
    ('d_val', ctypes.c_uint32),
    ('d_ptr', ctypes.c_uint32),
]

struct_c__SA_Elf32_Dyn._pack_ = 1 # source:False
struct_c__SA_Elf32_Dyn._fields_ = [
    ('d_tag', ctypes.c_int32),
    ('d_un', union_c__SA_Elf32_Dyn_d_un),
]

Elf32_Dyn = struct_c__SA_Elf32_Dyn
class struct_c__SA_Elf64_Dyn(Structure):
    pass

class union_c__SA_Elf64_Dyn_d_un(Union):
    pass

union_c__SA_Elf64_Dyn_d_un._pack_ = 1 # source:False
union_c__SA_Elf64_Dyn_d_un._fields_ = [
    ('d_val', ctypes.c_uint64),
    ('d_ptr', ctypes.c_uint64),
]

struct_c__SA_Elf64_Dyn._pack_ = 1 # source:False
struct_c__SA_Elf64_Dyn._fields_ = [
    ('d_tag', ctypes.c_int64),
    ('d_un', union_c__SA_Elf64_Dyn_d_un),
]

Elf64_Dyn = struct_c__SA_Elf64_Dyn
class struct_c__SA_Elf32_Verdef(Structure):
    pass

struct_c__SA_Elf32_Verdef._pack_ = 1 # source:False
struct_c__SA_Elf32_Verdef._fields_ = [
    ('vd_version', ctypes.c_uint16),
    ('vd_flags', ctypes.c_uint16),
    ('vd_ndx', ctypes.c_uint16),
    ('vd_cnt', ctypes.c_uint16),
    ('vd_hash', ctypes.c_uint32),
    ('vd_aux', ctypes.c_uint32),
    ('vd_next', ctypes.c_uint32),
]

Elf32_Verdef = struct_c__SA_Elf32_Verdef
class struct_c__SA_Elf64_Verdef(Structure):
    pass

struct_c__SA_Elf64_Verdef._pack_ = 1 # source:False
struct_c__SA_Elf64_Verdef._fields_ = [
    ('vd_version', ctypes.c_uint16),
    ('vd_flags', ctypes.c_uint16),
    ('vd_ndx', ctypes.c_uint16),
    ('vd_cnt', ctypes.c_uint16),
    ('vd_hash', ctypes.c_uint32),
    ('vd_aux', ctypes.c_uint32),
    ('vd_next', ctypes.c_uint32),
]

Elf64_Verdef = struct_c__SA_Elf64_Verdef
class struct_c__SA_Elf32_Verdaux(Structure):
    pass

struct_c__SA_Elf32_Verdaux._pack_ = 1 # source:False
struct_c__SA_Elf32_Verdaux._fields_ = [
    ('vda_name', ctypes.c_uint32),
    ('vda_next', ctypes.c_uint32),
]

Elf32_Verdaux = struct_c__SA_Elf32_Verdaux
class struct_c__SA_Elf64_Verdaux(Structure):
    pass

struct_c__SA_Elf64_Verdaux._pack_ = 1 # source:False
struct_c__SA_Elf64_Verdaux._fields_ = [
    ('vda_name', ctypes.c_uint32),
    ('vda_next', ctypes.c_uint32),
]

Elf64_Verdaux = struct_c__SA_Elf64_Verdaux
class struct_c__SA_Elf32_Verneed(Structure):
    pass

struct_c__SA_Elf32_Verneed._pack_ = 1 # source:False
struct_c__SA_Elf32_Verneed._fields_ = [
    ('vn_version', ctypes.c_uint16),
    ('vn_cnt', ctypes.c_uint16),
    ('vn_file', ctypes.c_uint32),
    ('vn_aux', ctypes.c_uint32),
    ('vn_next', ctypes.c_uint32),
]

Elf32_Verneed = struct_c__SA_Elf32_Verneed
class struct_c__SA_Elf64_Verneed(Structure):
    pass

struct_c__SA_Elf64_Verneed._pack_ = 1 # source:False
struct_c__SA_Elf64_Verneed._fields_ = [
    ('vn_version', ctypes.c_uint16),
    ('vn_cnt', ctypes.c_uint16),
    ('vn_file', ctypes.c_uint32),
    ('vn_aux', ctypes.c_uint32),
    ('vn_next', ctypes.c_uint32),
]

Elf64_Verneed = struct_c__SA_Elf64_Verneed
class struct_c__SA_Elf32_Vernaux(Structure):
    pass

struct_c__SA_Elf32_Vernaux._pack_ = 1 # source:False
struct_c__SA_Elf32_Vernaux._fields_ = [
    ('vna_hash', ctypes.c_uint32),
    ('vna_flags', ctypes.c_uint16),
    ('vna_other', ctypes.c_uint16),
    ('vna_name', ctypes.c_uint32),
    ('vna_next', ctypes.c_uint32),
]

Elf32_Vernaux = struct_c__SA_Elf32_Vernaux
class struct_c__SA_Elf64_Vernaux(Structure):
    pass

struct_c__SA_Elf64_Vernaux._pack_ = 1 # source:False
struct_c__SA_Elf64_Vernaux._fields_ = [
    ('vna_hash', ctypes.c_uint32),
    ('vna_flags', ctypes.c_uint16),
    ('vna_other', ctypes.c_uint16),
    ('vna_name', ctypes.c_uint32),
    ('vna_next', ctypes.c_uint32),
]

Elf64_Vernaux = struct_c__SA_Elf64_Vernaux
class struct_c__SA_Elf32_auxv_t(Structure):
    pass

class union_c__SA_Elf32_auxv_t_a_un(Union):
    pass

union_c__SA_Elf32_auxv_t_a_un._pack_ = 1 # source:False
union_c__SA_Elf32_auxv_t_a_un._fields_ = [
    ('a_val', ctypes.c_uint32),
]

struct_c__SA_Elf32_auxv_t._pack_ = 1 # source:False
struct_c__SA_Elf32_auxv_t._fields_ = [
    ('a_type', ctypes.c_uint32),
    ('a_un', union_c__SA_Elf32_auxv_t_a_un),
]

Elf32_auxv_t = struct_c__SA_Elf32_auxv_t
class struct_c__SA_Elf64_auxv_t(Structure):
    pass

class union_c__SA_Elf64_auxv_t_a_un(Union):
    pass

union_c__SA_Elf64_auxv_t_a_un._pack_ = 1 # source:False
union_c__SA_Elf64_auxv_t_a_un._fields_ = [
    ('a_val', ctypes.c_uint64),
]

struct_c__SA_Elf64_auxv_t._pack_ = 1 # source:False
struct_c__SA_Elf64_auxv_t._fields_ = [
    ('a_type', ctypes.c_uint64),
    ('a_un', union_c__SA_Elf64_auxv_t_a_un),
]

Elf64_auxv_t = struct_c__SA_Elf64_auxv_t
class struct_c__SA_Elf32_Nhdr(Structure):
    pass

struct_c__SA_Elf32_Nhdr._pack_ = 1 # source:False
struct_c__SA_Elf32_Nhdr._fields_ = [
    ('n_namesz', ctypes.c_uint32),
    ('n_descsz', ctypes.c_uint32),
    ('n_type', ctypes.c_uint32),
]

Elf32_Nhdr = struct_c__SA_Elf32_Nhdr
class struct_c__SA_Elf64_Nhdr(Structure):
    pass

struct_c__SA_Elf64_Nhdr._pack_ = 1 # source:False
struct_c__SA_Elf64_Nhdr._fields_ = [
    ('n_namesz', ctypes.c_uint32),
    ('n_descsz', ctypes.c_uint32),
    ('n_type', ctypes.c_uint32),
]

Elf64_Nhdr = struct_c__SA_Elf64_Nhdr
class struct_c__SA_Elf32_Move(Structure):
    pass

struct_c__SA_Elf32_Move._pack_ = 1 # source:False
struct_c__SA_Elf32_Move._fields_ = [
    ('m_value', ctypes.c_uint64),
    ('m_info', ctypes.c_uint32),
    ('m_poffset', ctypes.c_uint32),
    ('m_repeat', ctypes.c_uint16),
    ('m_stride', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

Elf32_Move = struct_c__SA_Elf32_Move
class struct_c__SA_Elf64_Move(Structure):
    pass

struct_c__SA_Elf64_Move._pack_ = 1 # source:False
struct_c__SA_Elf64_Move._fields_ = [
    ('m_value', ctypes.c_uint64),
    ('m_info', ctypes.c_uint64),
    ('m_poffset', ctypes.c_uint64),
    ('m_repeat', ctypes.c_uint16),
    ('m_stride', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

Elf64_Move = struct_c__SA_Elf64_Move
class union_c__UA_Elf32_gptab(Union):
    pass

class struct_c__UA_Elf32_gptab_gt_header(Structure):
    pass

struct_c__UA_Elf32_gptab_gt_header._pack_ = 1 # source:False
struct_c__UA_Elf32_gptab_gt_header._fields_ = [
    ('gt_current_g_value', ctypes.c_uint32),
    ('gt_unused', ctypes.c_uint32),
]

class struct_c__UA_Elf32_gptab_gt_entry(Structure):
    pass

struct_c__UA_Elf32_gptab_gt_entry._pack_ = 1 # source:False
struct_c__UA_Elf32_gptab_gt_entry._fields_ = [
    ('gt_g_value', ctypes.c_uint32),
    ('gt_bytes', ctypes.c_uint32),
]

union_c__UA_Elf32_gptab._pack_ = 1 # source:False
union_c__UA_Elf32_gptab._fields_ = [
    ('gt_header', struct_c__UA_Elf32_gptab_gt_header),
    ('gt_entry', struct_c__UA_Elf32_gptab_gt_entry),
]

Elf32_gptab = union_c__UA_Elf32_gptab
class struct_c__SA_Elf32_RegInfo(Structure):
    pass

struct_c__SA_Elf32_RegInfo._pack_ = 1 # source:False
struct_c__SA_Elf32_RegInfo._fields_ = [
    ('ri_gprmask', ctypes.c_uint32),
    ('ri_cprmask', ctypes.c_uint32 * 4),
    ('ri_gp_value', ctypes.c_int32),
]

Elf32_RegInfo = struct_c__SA_Elf32_RegInfo
class struct_c__SA_Elf_Options(Structure):
    pass

struct_c__SA_Elf_Options._pack_ = 1 # source:False
struct_c__SA_Elf_Options._fields_ = [
    ('kind', ctypes.c_ubyte),
    ('size', ctypes.c_ubyte),
    ('section', ctypes.c_uint16),
    ('info', ctypes.c_uint32),
]

Elf_Options = struct_c__SA_Elf_Options
class struct_c__SA_Elf_Options_Hw(Structure):
    pass

struct_c__SA_Elf_Options_Hw._pack_ = 1 # source:False
struct_c__SA_Elf_Options_Hw._fields_ = [
    ('hwp_flags1', ctypes.c_uint32),
    ('hwp_flags2', ctypes.c_uint32),
]

Elf_Options_Hw = struct_c__SA_Elf_Options_Hw
class struct_c__SA_Elf32_Lib(Structure):
    pass

struct_c__SA_Elf32_Lib._pack_ = 1 # source:False
struct_c__SA_Elf32_Lib._fields_ = [
    ('l_name', ctypes.c_uint32),
    ('l_time_stamp', ctypes.c_uint32),
    ('l_checksum', ctypes.c_uint32),
    ('l_version', ctypes.c_uint32),
    ('l_flags', ctypes.c_uint32),
]

Elf32_Lib = struct_c__SA_Elf32_Lib
class struct_c__SA_Elf64_Lib(Structure):
    pass

struct_c__SA_Elf64_Lib._pack_ = 1 # source:False
struct_c__SA_Elf64_Lib._fields_ = [
    ('l_name', ctypes.c_uint32),
    ('l_time_stamp', ctypes.c_uint32),
    ('l_checksum', ctypes.c_uint32),
    ('l_version', ctypes.c_uint32),
    ('l_flags', ctypes.c_uint32),
]

Elf64_Lib = struct_c__SA_Elf64_Lib
Elf32_Conflict = ctypes.c_uint32
class struct_c__SA_Elf_MIPS_ABIFlags_v0(Structure):
    pass

struct_c__SA_Elf_MIPS_ABIFlags_v0._pack_ = 1 # source:False
struct_c__SA_Elf_MIPS_ABIFlags_v0._fields_ = [
    ('version', ctypes.c_uint16),
    ('isa_level', ctypes.c_ubyte),
    ('isa_rev', ctypes.c_ubyte),
    ('gpr_size', ctypes.c_ubyte),
    ('cpr1_size', ctypes.c_ubyte),
    ('cpr2_size', ctypes.c_ubyte),
    ('fp_abi', ctypes.c_ubyte),
    ('isa_ext', ctypes.c_uint32),
    ('ases', ctypes.c_uint32),
    ('flags1', ctypes.c_uint32),
    ('flags2', ctypes.c_uint32),
]

Elf_MIPS_ABIFlags_v0 = struct_c__SA_Elf_MIPS_ABIFlags_v0

# values for enumeration 'c__Ea_Val_GNU_MIPS_ABI_FP_ANY'
c__Ea_Val_GNU_MIPS_ABI_FP_ANY__enumvalues = {
    0: 'Val_GNU_MIPS_ABI_FP_ANY',
    1: 'Val_GNU_MIPS_ABI_FP_DOUBLE',
    2: 'Val_GNU_MIPS_ABI_FP_SINGLE',
    3: 'Val_GNU_MIPS_ABI_FP_SOFT',
    4: 'Val_GNU_MIPS_ABI_FP_OLD_64',
    5: 'Val_GNU_MIPS_ABI_FP_XX',
    6: 'Val_GNU_MIPS_ABI_FP_64',
    7: 'Val_GNU_MIPS_ABI_FP_64A',
    7: 'Val_GNU_MIPS_ABI_FP_MAX',
}
Val_GNU_MIPS_ABI_FP_ANY = 0
Val_GNU_MIPS_ABI_FP_DOUBLE = 1
Val_GNU_MIPS_ABI_FP_SINGLE = 2
Val_GNU_MIPS_ABI_FP_SOFT = 3
Val_GNU_MIPS_ABI_FP_OLD_64 = 4
Val_GNU_MIPS_ABI_FP_XX = 5
Val_GNU_MIPS_ABI_FP_64 = 6
Val_GNU_MIPS_ABI_FP_64A = 7
Val_GNU_MIPS_ABI_FP_MAX = 7
c__Ea_Val_GNU_MIPS_ABI_FP_ANY = ctypes.c_uint32 # enum
_UNISTD_H = 1 # macro
_POSIX_VERSION = 200809 # macro
__POSIX2_THIS_VERSION = 200809 # macro
_POSIX2_VERSION = 200809 # macro
_POSIX2_C_VERSION = 200809 # macro
_POSIX2_C_BIND = 200809 # macro
_POSIX2_C_DEV = 200809 # macro
_POSIX2_SW_DEV = 200809 # macro
_POSIX2_LOCALEDEF = 200809 # macro
_XOPEN_VERSION = 700 # macro
_XOPEN_XCU_VERSION = 4 # macro
_XOPEN_XPG2 = 1 # macro
_XOPEN_XPG3 = 1 # macro
_XOPEN_XPG4 = 1 # macro
_XOPEN_UNIX = 1 # macro
_XOPEN_ENH_I18N = 1 # macro
_XOPEN_LEGACY = 1 # macro
STDIN_FILENO = 0 # macro
STDOUT_FILENO = 1 # macro
STDERR_FILENO = 2 # macro
__ssize_t_defined = True # macro
__gid_t_defined = True # macro
__uid_t_defined = True # macro
__useconds_t_defined = True # macro
__pid_t_defined = True # macro
__intptr_t_defined = True # macro
__socklen_t_defined = True # macro
R_OK = 4 # macro
W_OK = 2 # macro
X_OK = 1 # macro
F_OK = 0 # macro
SEEK_SET = 0 # macro
SEEK_CUR = 1 # macro
SEEK_END = 2 # macro
L_SET = 0 # macro
L_INCR = 1 # macro
L_XTND = 2 # macro
F_ULOCK = 0 # macro
F_LOCK = 1 # macro
F_TLOCK = 2 # macro
F_TEST = 3 # macro
ssize_t = ctypes.c_int64
gid_t = ctypes.c_uint32
uid_t = ctypes.c_uint32
useconds_t = ctypes.c_uint32
pid_t = ctypes.c_int32
intptr_t = ctypes.c_int64
socklen_t = ctypes.c_uint32
try:
    access = _libraries['libc'].access
    access.restype = ctypes.c_int32
    access.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    faccessat = _libraries['libc'].faccessat
    faccessat.restype = ctypes.c_int32
    faccessat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    lseek = _libraries['libc'].lseek
    lseek.restype = __off_t
    lseek.argtypes = [ctypes.c_int32, __off_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    close = _libraries['libc'].close
    close.restype = ctypes.c_int32
    close.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    closefrom = _libraries['libc'].closefrom
    closefrom.restype = None
    closefrom.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    read = _libraries['libc'].read
    read.restype = ssize_t
    read.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    write = _libraries['libc'].write
    write.restype = ssize_t
    write.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    pread = _libraries['libc'].pread
    pread.restype = ssize_t
    pread.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t, __off_t]
except AttributeError:
    pass
try:
    pwrite = _libraries['libc'].pwrite
    pwrite.restype = ssize_t
    pwrite.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t, __off_t]
except AttributeError:
    pass
try:
    pipe = _libraries['libc'].pipe
    pipe.restype = ctypes.c_int32
    pipe.argtypes = [ctypes.c_int32 * 2]
except AttributeError:
    pass
try:
    alarm = _libraries['libc'].alarm
    alarm.restype = ctypes.c_uint32
    alarm.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    sleep = _libraries['libc'].sleep
    sleep.restype = ctypes.c_uint32
    sleep.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
__useconds_t = ctypes.c_uint32
try:
    ualarm = _libraries['libc'].ualarm
    ualarm.restype = __useconds_t
    ualarm.argtypes = [__useconds_t, __useconds_t]
except AttributeError:
    pass
try:
    usleep = _libraries['libc'].usleep
    usleep.restype = ctypes.c_int32
    usleep.argtypes = [__useconds_t]
except AttributeError:
    pass
try:
    pause = _libraries['libc'].pause
    pause.restype = ctypes.c_int32
    pause.argtypes = []
except AttributeError:
    pass
__uid_t = ctypes.c_uint32
__gid_t = ctypes.c_uint32
try:
    chown = _libraries['libc'].chown
    chown.restype = ctypes.c_int32
    chown.argtypes = [ctypes.POINTER(ctypes.c_char), __uid_t, __gid_t]
except AttributeError:
    pass
try:
    fchown = _libraries['libc'].fchown
    fchown.restype = ctypes.c_int32
    fchown.argtypes = [ctypes.c_int32, __uid_t, __gid_t]
except AttributeError:
    pass
try:
    lchown = _libraries['libc'].lchown
    lchown.restype = ctypes.c_int32
    lchown.argtypes = [ctypes.POINTER(ctypes.c_char), __uid_t, __gid_t]
except AttributeError:
    pass
try:
    fchownat = _libraries['libc'].fchownat
    fchownat.restype = ctypes.c_int32
    fchownat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), __uid_t, __gid_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    chdir = _libraries['libc'].chdir
    chdir.restype = ctypes.c_int32
    chdir.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    fchdir = _libraries['libc'].fchdir
    fchdir.restype = ctypes.c_int32
    fchdir.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    getcwd = _libraries['libc'].getcwd
    getcwd.restype = ctypes.POINTER(ctypes.c_char)
    getcwd.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    getwd = _libraries['libc'].getwd
    getwd.restype = ctypes.POINTER(ctypes.c_char)
    getwd.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    dup = _libraries['libc'].dup
    dup.restype = ctypes.c_int32
    dup.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    dup2 = _libraries['libc'].dup2
    dup2.restype = ctypes.c_int32
    dup2.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
__environ = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))() # Variable ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
try:
    execve = _libraries['libc'].execve
    execve.restype = ctypes.c_int32
    execve.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char) * 0, ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    fexecve = _libraries['libc'].fexecve
    fexecve.restype = ctypes.c_int32
    fexecve.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char) * 0, ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    execv = _libraries['libc'].execv
    execv.restype = ctypes.c_int32
    execv.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    execle = _libraries['libc'].execle
    execle.restype = ctypes.c_int32
    execle.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    execl = _libraries['libc'].execl
    execl.restype = ctypes.c_int32
    execl.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    execvp = _libraries['libc'].execvp
    execvp.restype = ctypes.c_int32
    execvp.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    execlp = _libraries['libc'].execlp
    execlp.restype = ctypes.c_int32
    execlp.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nice = _libraries['libc'].nice
    nice.restype = ctypes.c_int32
    nice.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    _exit = _libraries['libc']._exit
    _exit.restype = None
    _exit.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    pathconf = _libraries['libc'].pathconf
    pathconf.restype = ctypes.c_int64
    pathconf.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    fpathconf = _libraries['libc'].fpathconf
    fpathconf.restype = ctypes.c_int64
    fpathconf.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    sysconf = _libraries['libc'].sysconf
    sysconf.restype = ctypes.c_int64
    sysconf.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    confstr = _libraries['libc'].confstr
    confstr.restype = size_t
    confstr.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
__pid_t = ctypes.c_int32
try:
    getpid = _libraries['libc'].getpid
    getpid.restype = __pid_t
    getpid.argtypes = []
except AttributeError:
    pass
try:
    getppid = _libraries['libc'].getppid
    getppid.restype = __pid_t
    getppid.argtypes = []
except AttributeError:
    pass
try:
    getpgrp = _libraries['libc'].getpgrp
    getpgrp.restype = __pid_t
    getpgrp.argtypes = []
except AttributeError:
    pass
try:
    __getpgid = _libraries['libc'].__getpgid
    __getpgid.restype = __pid_t
    __getpgid.argtypes = [__pid_t]
except AttributeError:
    pass
try:
    getpgid = _libraries['libc'].getpgid
    getpgid.restype = __pid_t
    getpgid.argtypes = [__pid_t]
except AttributeError:
    pass
try:
    setpgid = _libraries['libc'].setpgid
    setpgid.restype = ctypes.c_int32
    setpgid.argtypes = [__pid_t, __pid_t]
except AttributeError:
    pass
try:
    setpgrp = _libraries['libc'].setpgrp
    setpgrp.restype = ctypes.c_int32
    setpgrp.argtypes = []
except AttributeError:
    pass
try:
    setsid = _libraries['libc'].setsid
    setsid.restype = __pid_t
    setsid.argtypes = []
except AttributeError:
    pass
try:
    getsid = _libraries['libc'].getsid
    getsid.restype = __pid_t
    getsid.argtypes = [__pid_t]
except AttributeError:
    pass
try:
    getuid = _libraries['libc'].getuid
    getuid.restype = __uid_t
    getuid.argtypes = []
except AttributeError:
    pass
try:
    geteuid = _libraries['libc'].geteuid
    geteuid.restype = __uid_t
    geteuid.argtypes = []
except AttributeError:
    pass
try:
    getgid = _libraries['libc'].getgid
    getgid.restype = __gid_t
    getgid.argtypes = []
except AttributeError:
    pass
try:
    getegid = _libraries['libc'].getegid
    getegid.restype = __gid_t
    getegid.argtypes = []
except AttributeError:
    pass
try:
    getgroups = _libraries['libc'].getgroups
    getgroups.restype = ctypes.c_int32
    getgroups.argtypes = [ctypes.c_int32, ctypes.c_uint32 * 0]
except AttributeError:
    pass
try:
    setuid = _libraries['libc'].setuid
    setuid.restype = ctypes.c_int32
    setuid.argtypes = [__uid_t]
except AttributeError:
    pass
try:
    setreuid = _libraries['libc'].setreuid
    setreuid.restype = ctypes.c_int32
    setreuid.argtypes = [__uid_t, __uid_t]
except AttributeError:
    pass
try:
    seteuid = _libraries['libc'].seteuid
    seteuid.restype = ctypes.c_int32
    seteuid.argtypes = [__uid_t]
except AttributeError:
    pass
try:
    setgid = _libraries['libc'].setgid
    setgid.restype = ctypes.c_int32
    setgid.argtypes = [__gid_t]
except AttributeError:
    pass
try:
    setregid = _libraries['libc'].setregid
    setregid.restype = ctypes.c_int32
    setregid.argtypes = [__gid_t, __gid_t]
except AttributeError:
    pass
try:
    setegid = _libraries['libc'].setegid
    setegid.restype = ctypes.c_int32
    setegid.argtypes = [__gid_t]
except AttributeError:
    pass
try:
    fork = _libraries['libc'].fork
    fork.restype = __pid_t
    fork.argtypes = []
except AttributeError:
    pass
try:
    vfork = _libraries['libc'].vfork
    vfork.restype = ctypes.c_int32
    vfork.argtypes = []
except AttributeError:
    pass
try:
    ttyname = _libraries['libc'].ttyname
    ttyname.restype = ctypes.POINTER(ctypes.c_char)
    ttyname.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    ttyname_r = _libraries['libc'].ttyname_r
    ttyname_r.restype = ctypes.c_int32
    ttyname_r.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    isatty = _libraries['libc'].isatty
    isatty.restype = ctypes.c_int32
    isatty.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    ttyslot = _libraries['libc'].ttyslot
    ttyslot.restype = ctypes.c_int32
    ttyslot.argtypes = []
except AttributeError:
    pass
try:
    link = _libraries['libc'].link
    link.restype = ctypes.c_int32
    link.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    linkat = _libraries['libc'].linkat
    linkat.restype = ctypes.c_int32
    linkat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    symlink = _libraries['libc'].symlink
    symlink.restype = ctypes.c_int32
    symlink.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    readlink = _libraries['libc'].readlink
    readlink.restype = ssize_t
    readlink.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    symlinkat = _libraries['libc'].symlinkat
    symlinkat.restype = ctypes.c_int32
    symlinkat.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    readlinkat = _libraries['libc'].readlinkat
    readlinkat.restype = ssize_t
    readlinkat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    unlink = _libraries['libc'].unlink
    unlink.restype = ctypes.c_int32
    unlink.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    unlinkat = _libraries['libc'].unlinkat
    unlinkat.restype = ctypes.c_int32
    unlinkat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    rmdir = _libraries['libc'].rmdir
    rmdir.restype = ctypes.c_int32
    rmdir.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    tcgetpgrp = _libraries['libc'].tcgetpgrp
    tcgetpgrp.restype = __pid_t
    tcgetpgrp.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    tcsetpgrp = _libraries['libc'].tcsetpgrp
    tcsetpgrp.restype = ctypes.c_int32
    tcsetpgrp.argtypes = [ctypes.c_int32, __pid_t]
except AttributeError:
    pass
try:
    getlogin = _libraries['libc'].getlogin
    getlogin.restype = ctypes.POINTER(ctypes.c_char)
    getlogin.argtypes = []
except AttributeError:
    pass
try:
    getlogin_r = _libraries['libc'].getlogin_r
    getlogin_r.restype = ctypes.c_int32
    getlogin_r.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    setlogin = _libraries['libc'].setlogin
    setlogin.restype = ctypes.c_int32
    setlogin.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    gethostname = _libraries['libc'].gethostname
    gethostname.restype = ctypes.c_int32
    gethostname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    sethostname = _libraries['libc'].sethostname
    sethostname.restype = ctypes.c_int32
    sethostname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    sethostid = _libraries['libc'].sethostid
    sethostid.restype = ctypes.c_int32
    sethostid.argtypes = [ctypes.c_int64]
except AttributeError:
    pass
try:
    getdomainname = _libraries['libc'].getdomainname
    getdomainname.restype = ctypes.c_int32
    getdomainname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    setdomainname = _libraries['libc'].setdomainname
    setdomainname.restype = ctypes.c_int32
    setdomainname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    vhangup = _libraries['libc'].vhangup
    vhangup.restype = ctypes.c_int32
    vhangup.argtypes = []
except AttributeError:
    pass
try:
    revoke = _libraries['libc'].revoke
    revoke.restype = ctypes.c_int32
    revoke.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    profil = _libraries['libc'].profil
    profil.restype = ctypes.c_int32
    profil.argtypes = [ctypes.POINTER(ctypes.c_uint16), size_t, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    acct = _libraries['libc'].acct
    acct.restype = ctypes.c_int32
    acct.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    getusershell = _libraries['libc'].getusershell
    getusershell.restype = ctypes.POINTER(ctypes.c_char)
    getusershell.argtypes = []
except AttributeError:
    pass
try:
    endusershell = _libraries['libc'].endusershell
    endusershell.restype = None
    endusershell.argtypes = []
except AttributeError:
    pass
try:
    setusershell = _libraries['libc'].setusershell
    setusershell.restype = None
    setusershell.argtypes = []
except AttributeError:
    pass
try:
    daemon = _libraries['libc'].daemon
    daemon.restype = ctypes.c_int32
    daemon.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    chroot = _libraries['libc'].chroot
    chroot.restype = ctypes.c_int32
    chroot.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    getpass = _libraries['libc'].getpass
    getpass.restype = ctypes.POINTER(ctypes.c_char)
    getpass.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    fsync = _libraries['libc'].fsync
    fsync.restype = ctypes.c_int32
    fsync.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    gethostid = _libraries['libc'].gethostid
    gethostid.restype = ctypes.c_int64
    gethostid.argtypes = []
except AttributeError:
    pass
try:
    sync = _libraries['libc'].sync
    sync.restype = None
    sync.argtypes = []
except AttributeError:
    pass
try:
    getpagesize = _libraries['libc'].getpagesize
    getpagesize.restype = ctypes.c_int32
    getpagesize.argtypes = []
except AttributeError:
    pass
try:
    getdtablesize = _libraries['libc'].getdtablesize
    getdtablesize.restype = ctypes.c_int32
    getdtablesize.argtypes = []
except AttributeError:
    pass
try:
    truncate = _libraries['libc'].truncate
    truncate.restype = ctypes.c_int32
    truncate.argtypes = [ctypes.POINTER(ctypes.c_char), __off_t]
except AttributeError:
    pass
try:
    ftruncate = _libraries['libc'].ftruncate
    ftruncate.restype = ctypes.c_int32
    ftruncate.argtypes = [ctypes.c_int32, __off_t]
except AttributeError:
    pass
try:
    brk = _libraries['libc'].brk
    brk.restype = ctypes.c_int32
    brk.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    sbrk = _libraries['libc'].sbrk
    sbrk.restype = ctypes.POINTER(None)
    sbrk.argtypes = [intptr_t]
except AttributeError:
    pass
try:
    syscall = _libraries['libc'].syscall
    syscall.restype = ctypes.c_int64
    syscall.argtypes = [ctypes.c_int64]
except AttributeError:
    pass
try:
    lockf = _libraries['libc'].lockf
    lockf.restype = ctypes.c_int32
    lockf.argtypes = [ctypes.c_int32, ctypes.c_int32, __off_t]
except AttributeError:
    pass
try:
    fdatasync = _libraries['libc'].fdatasync
    fdatasync.restype = ctypes.c_int32
    fdatasync.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    crypt = _libraries['libc'].crypt
    crypt.restype = ctypes.POINTER(ctypes.c_char)
    crypt.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    getentropy = _libraries['libc'].getentropy
    getentropy.restype = ctypes.c_int32
    getentropy.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
__ASM_GENERIC_MMAN_COMMON_H = True # macro
PROT_READ = 0x1 # macro
PROT_WRITE = 0x2 # macro
PROT_EXEC = 0x4 # macro
PROT_SEM = 0x8 # macro
PROT_NONE = 0x0 # macro
PROT_GROWSDOWN = 0x01000000 # macro
PROT_GROWSUP = 0x02000000 # macro
MAP_TYPE = 0x0f # macro
MAP_FIXED = 0x10 # macro
MAP_ANONYMOUS = 0x20 # macro
MAP_POPULATE = 0x008000 # macro
MAP_NONBLOCK = 0x010000 # macro
MAP_STACK = 0x020000 # macro
MAP_HUGETLB = 0x040000 # macro
MAP_SYNC = 0x080000 # macro
MAP_FIXED_NOREPLACE = 0x100000 # macro
MAP_UNINITIALIZED = 0x4000000 # macro
MLOCK_ONFAULT = 0x01 # macro
MS_ASYNC = 1 # macro
MS_INVALIDATE = 2 # macro
MS_SYNC = 4 # macro
MADV_NORMAL = 0 # macro
MADV_RANDOM = 1 # macro
MADV_SEQUENTIAL = 2 # macro
MADV_WILLNEED = 3 # macro
MADV_DONTNEED = 4 # macro
MADV_FREE = 8 # macro
MADV_REMOVE = 9 # macro
MADV_DONTFORK = 10 # macro
MADV_DOFORK = 11 # macro
MADV_HWPOISON = 100 # macro
MADV_SOFT_OFFLINE = 101 # macro
MADV_MERGEABLE = 12 # macro
MADV_UNMERGEABLE = 13 # macro
MADV_HUGEPAGE = 14 # macro
MADV_NOHUGEPAGE = 15 # macro
MADV_DONTDUMP = 16 # macro
MADV_DODUMP = 17 # macro
MADV_WIPEONFORK = 18 # macro
MADV_KEEPONFORK = 19 # macro
MADV_COLD = 20 # macro
MADV_PAGEOUT = 21 # macro
MADV_POPULATE_READ = 22 # macro
MADV_POPULATE_WRITE = 23 # macro
MADV_DONTNEED_LOCKED = 24 # macro
MADV_COLLAPSE = 25 # macro
MAP_FILE = 0 # macro
PKEY_DISABLE_ACCESS = 0x1 # macro
PKEY_DISABLE_WRITE = 0x2 # macro
PKEY_ACCESS_MASK = (0x1|0x2) # macro
__all__ = \
    ['AT_BASE', 'AT_BASE_PLATFORM', 'AT_CLKTCK', 'AT_DCACHEBSIZE',
    'AT_EGID', 'AT_ENTRY', 'AT_EUID', 'AT_EXECFD', 'AT_EXECFN',
    'AT_FLAGS', 'AT_FPUCW', 'AT_GID', 'AT_HWCAP', 'AT_HWCAP2',
    'AT_HWCAP3', 'AT_HWCAP4', 'AT_ICACHEBSIZE', 'AT_IGNORE',
    'AT_IGNOREPPC', 'AT_L1D_CACHEGEOMETRY', 'AT_L1D_CACHESHAPE',
    'AT_L1D_CACHESIZE', 'AT_L1I_CACHEGEOMETRY', 'AT_L1I_CACHESHAPE',
    'AT_L1I_CACHESIZE', 'AT_L2_CACHEGEOMETRY', 'AT_L2_CACHESHAPE',
    'AT_L2_CACHESIZE', 'AT_L3_CACHEGEOMETRY', 'AT_L3_CACHESHAPE',
    'AT_L3_CACHESIZE', 'AT_MINSIGSTKSZ', 'AT_NOTELF', 'AT_NULL',
    'AT_PAGESZ', 'AT_PHDR', 'AT_PHENT', 'AT_PHNUM', 'AT_PLATFORM',
    'AT_RANDOM', 'AT_RSEQ_ALIGN', 'AT_RSEQ_FEATURE_SIZE', 'AT_SECURE',
    'AT_SYSINFO', 'AT_SYSINFO_EHDR', 'AT_UCACHEBSIZE', 'AT_UID',
    'DF_1_CONFALT', 'DF_1_DIRECT', 'DF_1_DISPRELDNE',
    'DF_1_DISPRELPND', 'DF_1_EDITED', 'DF_1_ENDFILTEE', 'DF_1_GLOBAL',
    'DF_1_GLOBAUDIT', 'DF_1_GROUP', 'DF_1_IGNMULDEF',
    'DF_1_INITFIRST', 'DF_1_INTERPOSE', 'DF_1_KMOD', 'DF_1_LOADFLTR',
    'DF_1_NOCOMMON', 'DF_1_NODEFLIB', 'DF_1_NODELETE',
    'DF_1_NODIRECT', 'DF_1_NODUMP', 'DF_1_NOHDR', 'DF_1_NOKSYMS',
    'DF_1_NOOPEN', 'DF_1_NORELOC', 'DF_1_NOW', 'DF_1_ORIGIN',
    'DF_1_PIE', 'DF_1_SINGLETON', 'DF_1_STUB', 'DF_1_SYMINTPOSE',
    'DF_1_TRANS', 'DF_1_WEAKFILTER', 'DF_BIND_NOW', 'DF_ORIGIN',
    'DF_P1_GROUPPERM', 'DF_P1_LAZYLOAD', 'DF_STATIC_TLS',
    'DF_SYMBOLIC', 'DF_TEXTREL', 'DTF_1_CONFEXP', 'DTF_1_PARINIT',
    'DT_AARCH64_BTI_PLT', 'DT_AARCH64_NUM', 'DT_AARCH64_PAC_PLT',
    'DT_AARCH64_VARIANT_PCS', 'DT_ADDRNUM', 'DT_ADDRRNGHI',
    'DT_ADDRRNGLO', 'DT_ALPHA_NUM', 'DT_ALPHA_PLTRO', 'DT_AUDIT',
    'DT_AUXILIARY', 'DT_BIND_NOW', 'DT_CHECKSUM', 'DT_CONFIG',
    'DT_DEBUG', 'DT_DEPAUDIT', 'DT_ENCODING', 'DT_EXTRANUM',
    'DT_FEATURE_1', 'DT_FILTER', 'DT_FINI', 'DT_FINI_ARRAY',
    'DT_FINI_ARRAYSZ', 'DT_FLAGS', 'DT_FLAGS_1', 'DT_GNU_CONFLICT',
    'DT_GNU_CONFLICTSZ', 'DT_GNU_HASH', 'DT_GNU_LIBLIST',
    'DT_GNU_LIBLISTSZ', 'DT_GNU_PRELINKED', 'DT_HASH', 'DT_HIOS',
    'DT_HIPROC', 'DT_IA_64_NUM', 'DT_IA_64_PLT_RESERVE', 'DT_INIT',
    'DT_INIT_ARRAY', 'DT_INIT_ARRAYSZ', 'DT_JMPREL', 'DT_LOOS',
    'DT_LOPROC', 'DT_MIPS_AUX_DYNAMIC', 'DT_MIPS_BASE_ADDRESS',
    'DT_MIPS_COMPACT_SIZE', 'DT_MIPS_CONFLICT', 'DT_MIPS_CONFLICTNO',
    'DT_MIPS_CXX_FLAGS', 'DT_MIPS_DELTA_CLASS',
    'DT_MIPS_DELTA_CLASSSYM', 'DT_MIPS_DELTA_CLASSSYM_NO',
    'DT_MIPS_DELTA_CLASS_NO', 'DT_MIPS_DELTA_INSTANCE',
    'DT_MIPS_DELTA_INSTANCE_NO', 'DT_MIPS_DELTA_RELOC',
    'DT_MIPS_DELTA_RELOC_NO', 'DT_MIPS_DELTA_SYM',
    'DT_MIPS_DELTA_SYM_NO', 'DT_MIPS_DYNSTR_ALIGN', 'DT_MIPS_FLAGS',
    'DT_MIPS_GOTSYM', 'DT_MIPS_GP_VALUE', 'DT_MIPS_HIDDEN_GOTIDX',
    'DT_MIPS_HIPAGENO', 'DT_MIPS_ICHECKSUM', 'DT_MIPS_INTERFACE',
    'DT_MIPS_INTERFACE_SIZE', 'DT_MIPS_IVERSION', 'DT_MIPS_LIBLIST',
    'DT_MIPS_LIBLISTNO', 'DT_MIPS_LOCALPAGE_GOTIDX',
    'DT_MIPS_LOCAL_GOTIDX', 'DT_MIPS_LOCAL_GOTNO', 'DT_MIPS_MSYM',
    'DT_MIPS_NUM', 'DT_MIPS_OPTIONS', 'DT_MIPS_PERF_SUFFIX',
    'DT_MIPS_PIXIE_INIT', 'DT_MIPS_PLTGOT',
    'DT_MIPS_PROTECTED_GOTIDX', 'DT_MIPS_RLD_MAP',
    'DT_MIPS_RLD_MAP_REL', 'DT_MIPS_RLD_TEXT_RESOLVE_ADDR',
    'DT_MIPS_RLD_VERSION', 'DT_MIPS_RWPLT', 'DT_MIPS_SYMBOL_LIB',
    'DT_MIPS_SYMTABNO', 'DT_MIPS_TIME_STAMP', 'DT_MIPS_UNREFEXTNO',
    'DT_MIPS_XHASH', 'DT_MOVEENT', 'DT_MOVESZ', 'DT_MOVETAB',
    'DT_NEEDED', 'DT_NIOS2_GP', 'DT_NULL', 'DT_NUM', 'DT_PLTGOT',
    'DT_PLTPAD', 'DT_PLTPADSZ', 'DT_PLTREL', 'DT_PLTRELSZ',
    'DT_POSFLAG_1', 'DT_PPC64_GLINK', 'DT_PPC64_NUM', 'DT_PPC64_OPD',
    'DT_PPC64_OPDSZ', 'DT_PPC64_OPT', 'DT_PPC_GOT', 'DT_PPC_NUM',
    'DT_PPC_OPT', 'DT_PREINIT_ARRAY', 'DT_PREINIT_ARRAYSZ',
    'DT_PROCNUM', 'DT_REL', 'DT_RELA', 'DT_RELACOUNT', 'DT_RELAENT',
    'DT_RELASZ', 'DT_RELCOUNT', 'DT_RELENT', 'DT_RELR', 'DT_RELRENT',
    'DT_RELRSZ', 'DT_RELSZ', 'DT_RISCV_VARIANT_CC', 'DT_RPATH',
    'DT_RUNPATH', 'DT_SONAME', 'DT_SPARC_NUM', 'DT_SPARC_REGISTER',
    'DT_STRSZ', 'DT_STRTAB', 'DT_SYMBOLIC', 'DT_SYMENT',
    'DT_SYMINENT', 'DT_SYMINFO', 'DT_SYMINSZ', 'DT_SYMTAB',
    'DT_SYMTAB_SHNDX', 'DT_TEXTREL', 'DT_TLSDESC_GOT',
    'DT_TLSDESC_PLT', 'DT_VALNUM', 'DT_VALRNGHI', 'DT_VALRNGLO',
    'DT_VERDEF', 'DT_VERDEFNUM', 'DT_VERNEED', 'DT_VERNEEDNUM',
    'DT_VERSIONTAGNUM', 'DT_VERSYM', 'DT_X86_64_NUM', 'DT_X86_64_PLT',
    'DT_X86_64_PLTENT', 'DT_X86_64_PLTSZ', 'EFA_PARISC_1_0',
    'EFA_PARISC_1_1', 'EFA_PARISC_2_0', 'EF_ALPHA_32BIT',
    'EF_ALPHA_CANRELAX', 'EF_ARC_ALL_MSK', 'EF_ARC_MACH_MSK',
    'EF_ARC_OSABI_MSK', 'EF_ARM_ABI_FLOAT_HARD',
    'EF_ARM_ABI_FLOAT_SOFT', 'EF_ARM_ALIGN8', 'EF_ARM_APCS_26',
    'EF_ARM_APCS_FLOAT', 'EF_ARM_BE8', 'EF_ARM_DYNSYMSUSESEGIDX',
    'EF_ARM_EABIMASK', 'EF_ARM_EABI_UNKNOWN', 'EF_ARM_EABI_VER1',
    'EF_ARM_EABI_VER2', 'EF_ARM_EABI_VER3', 'EF_ARM_EABI_VER4',
    'EF_ARM_EABI_VER5', 'EF_ARM_HASENTRY', 'EF_ARM_INTERWORK',
    'EF_ARM_LE8', 'EF_ARM_MAPSYMSFIRST', 'EF_ARM_MAVERICK_FLOAT',
    'EF_ARM_NEW_ABI', 'EF_ARM_OLD_ABI', 'EF_ARM_PIC',
    'EF_ARM_RELEXEC', 'EF_ARM_SOFT_FLOAT', 'EF_ARM_SYMSARESORTED',
    'EF_ARM_VFP_FLOAT', 'EF_CPU32', 'EF_CSKY_ABIMASK',
    'EF_CSKY_ABIV1', 'EF_CSKY_ABIV2', 'EF_CSKY_OTHER',
    'EF_CSKY_PROCESSOR', 'EF_IA_64_ABI64', 'EF_IA_64_ARCH',
    'EF_IA_64_MASKOS', 'EF_LARCH_ABI_DOUBLE_FLOAT',
    'EF_LARCH_ABI_MODIFIER_MASK', 'EF_LARCH_ABI_SINGLE_FLOAT',
    'EF_LARCH_ABI_SOFT_FLOAT', 'EF_LARCH_OBJABI_V1',
    'EF_MIPS_32BITMODE', 'EF_MIPS_ABI', 'EF_MIPS_ABI2',
    'EF_MIPS_ABI_EABI32', 'EF_MIPS_ABI_EABI64', 'EF_MIPS_ABI_O32',
    'EF_MIPS_ABI_O64', 'EF_MIPS_ABI_ON32', 'EF_MIPS_ARCH',
    'EF_MIPS_ARCH_1', 'EF_MIPS_ARCH_2', 'EF_MIPS_ARCH_3',
    'EF_MIPS_ARCH_32', 'EF_MIPS_ARCH_32R2', 'EF_MIPS_ARCH_32R6',
    'EF_MIPS_ARCH_4', 'EF_MIPS_ARCH_5', 'EF_MIPS_ARCH_64',
    'EF_MIPS_ARCH_64R2', 'EF_MIPS_ARCH_64R6', 'EF_MIPS_ARCH_ASE',
    'EF_MIPS_ARCH_ASE_M16', 'EF_MIPS_ARCH_ASE_MDMX',
    'EF_MIPS_ARCH_ASE_MICROMIPS', 'EF_MIPS_CPIC', 'EF_MIPS_FP64',
    'EF_MIPS_MACH', 'EF_MIPS_MACH_3900', 'EF_MIPS_MACH_4010',
    'EF_MIPS_MACH_4100', 'EF_MIPS_MACH_4111', 'EF_MIPS_MACH_4120',
    'EF_MIPS_MACH_4650', 'EF_MIPS_MACH_5400', 'EF_MIPS_MACH_5500',
    'EF_MIPS_MACH_5900', 'EF_MIPS_MACH_9000', 'EF_MIPS_MACH_ALLEGREX',
    'EF_MIPS_MACH_GS264E', 'EF_MIPS_MACH_GS464',
    'EF_MIPS_MACH_GS464E', 'EF_MIPS_MACH_IAMR2', 'EF_MIPS_MACH_LS2E',
    'EF_MIPS_MACH_LS2F', 'EF_MIPS_MACH_OCTEON',
    'EF_MIPS_MACH_OCTEON2', 'EF_MIPS_MACH_OCTEON3',
    'EF_MIPS_MACH_SB1', 'EF_MIPS_MACH_XLR', 'EF_MIPS_NAN2008',
    'EF_MIPS_NOREORDER', 'EF_MIPS_OPTIONS_FIRST', 'EF_MIPS_PIC',
    'EF_MIPS_UCODE', 'EF_MIPS_XGOT', 'EF_PARISC_ARCH',
    'EF_PARISC_EXT', 'EF_PARISC_LAZYSWAP', 'EF_PARISC_LSB',
    'EF_PARISC_NO_KABP', 'EF_PARISC_TRAPNIL', 'EF_PARISC_WIDE',
    'EF_PPC64_ABI', 'EF_PPC_EMB', 'EF_PPC_RELOCATABLE',
    'EF_PPC_RELOCATABLE_LIB', 'EF_RISCV_FLOAT_ABI',
    'EF_RISCV_FLOAT_ABI_DOUBLE', 'EF_RISCV_FLOAT_ABI_QUAD',
    'EF_RISCV_FLOAT_ABI_SINGLE', 'EF_RISCV_FLOAT_ABI_SOFT',
    'EF_RISCV_RVC', 'EF_RISCV_RVE', 'EF_RISCV_TSO',
    'EF_S390_HIGH_GPRS', 'EF_SH1', 'EF_SH2', 'EF_SH2A',
    'EF_SH2A_NOFPU', 'EF_SH2A_SH3E', 'EF_SH2A_SH3_NOFPU',
    'EF_SH2A_SH4', 'EF_SH2A_SH4_NOFPU', 'EF_SH2E', 'EF_SH3',
    'EF_SH3E', 'EF_SH3_DSP', 'EF_SH3_NOMMU', 'EF_SH4', 'EF_SH4A',
    'EF_SH4AL_DSP', 'EF_SH4A_NOFPU', 'EF_SH4_NOFPU',
    'EF_SH4_NOMMU_NOFPU', 'EF_SH_DSP', 'EF_SH_MACH_MASK',
    'EF_SH_UNKNOWN', 'EF_SPARCV9_MM', 'EF_SPARCV9_PSO',
    'EF_SPARCV9_RMO', 'EF_SPARCV9_TSO', 'EF_SPARC_32PLUS',
    'EF_SPARC_EXT_MASK', 'EF_SPARC_HAL_R1', 'EF_SPARC_LEDATA',
    'EF_SPARC_SUN_US1', 'EF_SPARC_SUN_US3', 'EI_ABIVERSION',
    'EI_CLASS', 'EI_DATA', 'EI_MAG0', 'EI_MAG1', 'EI_MAG2', 'EI_MAG3',
    'EI_NIDENT', 'EI_OSABI', 'EI_PAD', 'EI_VERSION', 'ELFCLASS32',
    'ELFCLASS64', 'ELFCLASSNONE', 'ELFCLASSNUM', 'ELFCOMPRESS_HIOS',
    'ELFCOMPRESS_HIPROC', 'ELFCOMPRESS_LOOS', 'ELFCOMPRESS_LOPROC',
    'ELFCOMPRESS_ZLIB', 'ELFCOMPRESS_ZSTD', 'ELFDATA2LSB',
    'ELFDATA2MSB', 'ELFDATANONE', 'ELFDATANUM', 'ELFMAG', 'ELFMAG0',
    'ELFMAG1', 'ELFMAG2', 'ELFMAG3', 'ELFOSABI_AIX', 'ELFOSABI_ARM',
    'ELFOSABI_ARM_AEABI', 'ELFOSABI_FREEBSD', 'ELFOSABI_GNU',
    'ELFOSABI_HPUX', 'ELFOSABI_IRIX', 'ELFOSABI_LINUX',
    'ELFOSABI_MODESTO', 'ELFOSABI_NETBSD', 'ELFOSABI_NONE',
    'ELFOSABI_OPENBSD', 'ELFOSABI_SOLARIS', 'ELFOSABI_STANDALONE',
    'ELFOSABI_SYSV', 'ELFOSABI_TRU64', 'ELF_NOTE_ABI', 'ELF_NOTE_FDO',
    'ELF_NOTE_GNU', 'ELF_NOTE_OS_FREEBSD', 'ELF_NOTE_OS_GNU',
    'ELF_NOTE_OS_LINUX', 'ELF_NOTE_OS_SOLARIS2',
    'ELF_NOTE_PAGESIZE_HINT', 'ELF_NOTE_SOLARIS', 'EM_386',
    'EM_56800EX', 'EM_68HC05', 'EM_68HC08', 'EM_68HC11', 'EM_68HC12',
    'EM_68HC16', 'EM_68K', 'EM_78KOR', 'EM_8051', 'EM_860', 'EM_88K',
    'EM_960', 'EM_AARCH64', 'EM_ALPHA', 'EM_ALTERA_NIOS2',
    'EM_AMDGPU', 'EM_ARC', 'EM_ARCA', 'EM_ARCV2', 'EM_ARC_A5',
    'EM_ARC_COMPACT', 'EM_ARM', 'EM_AVR', 'EM_AVR32', 'EM_BA1',
    'EM_BA2', 'EM_BLACKFIN', 'EM_BPF', 'EM_C166', 'EM_CDP', 'EM_CE',
    'EM_CLOUDSHIELD', 'EM_COGE', 'EM_COLDFIRE', 'EM_COOL',
    'EM_COREA_1ST', 'EM_COREA_2ND', 'EM_CR', 'EM_CR16', 'EM_CRAYNV2',
    'EM_CRIS', 'EM_CRX', 'EM_CSKY', 'EM_CSR_KALIMBA', 'EM_CUDA',
    'EM_CYPRESS_M8C', 'EM_D10V', 'EM_D30V', 'EM_DSP24', 'EM_DSPIC30F',
    'EM_DXP', 'EM_ECOG16', 'EM_ECOG1X', 'EM_ECOG2', 'EM_EMX16',
    'EM_EMX8', 'EM_ETPU', 'EM_EXCESS', 'EM_F2MC16', 'EM_FAKE_ALPHA',
    'EM_FIREPATH', 'EM_FR20', 'EM_FR30', 'EM_FT32', 'EM_FX66',
    'EM_H8S', 'EM_H8_300', 'EM_H8_300H', 'EM_H8_500', 'EM_HUANY',
    'EM_IAMCU', 'EM_IA_64', 'EM_INTELGT', 'EM_IP2K', 'EM_JAVELIN',
    'EM_K10M', 'EM_KM32', 'EM_KMX32', 'EM_KVARC', 'EM_L10M',
    'EM_LATTICEMICO32', 'EM_LOONGARCH', 'EM_M16C', 'EM_M32',
    'EM_M32C', 'EM_M32R', 'EM_MANIK', 'EM_MAX', 'EM_MAXQ30',
    'EM_MCHP_PIC', 'EM_MCST_ELBRUS', 'EM_ME16', 'EM_METAG',
    'EM_MICROBLAZE', 'EM_MIPS', 'EM_MIPS_RS3_LE', 'EM_MIPS_X',
    'EM_MMA', 'EM_MMDSP_PLUS', 'EM_MMIX', 'EM_MN10200', 'EM_MN10300',
    'EM_MOXIE', 'EM_MSP430', 'EM_NCPU', 'EM_NDR1', 'EM_NDS32',
    'EM_NONE', 'EM_NORC', 'EM_NS32K', 'EM_NUM', 'EM_OPEN8',
    'EM_OPENRISC', 'EM_PARISC', 'EM_PCP', 'EM_PDP10', 'EM_PDP11',
    'EM_PDSP', 'EM_PJ', 'EM_PPC', 'EM_PPC64', 'EM_PRISM', 'EM_QDSP6',
    'EM_R32C', 'EM_RCE', 'EM_RH32', 'EM_RISCV', 'EM_RL78', 'EM_RS08',
    'EM_RX', 'EM_S370', 'EM_S390', 'EM_SCORE7', 'EM_SEP', 'EM_SE_C17',
    'EM_SE_C33', 'EM_SH', 'EM_SHARC', 'EM_SLE9X', 'EM_SNP1K',
    'EM_SPARC', 'EM_SPARC32PLUS', 'EM_SPARCV9', 'EM_SPU', 'EM_ST100',
    'EM_ST19', 'EM_ST200', 'EM_ST7', 'EM_ST9PLUS', 'EM_STARCORE',
    'EM_STM8', 'EM_STXP7X', 'EM_SVX', 'EM_TILE64', 'EM_TILEGX',
    'EM_TILEPRO', 'EM_TINYJ', 'EM_TI_ARP32', 'EM_TI_C2000',
    'EM_TI_C5500', 'EM_TI_C6000', 'EM_TI_PRU', 'EM_TMM_GPP', 'EM_TPC',
    'EM_TRICORE', 'EM_TRIMEDIA', 'EM_TSK3000', 'EM_UNICORE',
    'EM_V800', 'EM_V850', 'EM_VAX', 'EM_VIDEOCORE', 'EM_VIDEOCORE3',
    'EM_VIDEOCORE5', 'EM_VISIUM', 'EM_VPP500', 'EM_X86_64',
    'EM_XCORE', 'EM_XGATE', 'EM_XIMO16', 'EM_XTENSA', 'EM_Z80',
    'EM_ZSP', 'ET_CORE', 'ET_DYN', 'ET_EXEC', 'ET_HIOS', 'ET_HIPROC',
    'ET_LOOS', 'ET_LOPROC', 'ET_NONE', 'ET_NUM', 'ET_REL',
    'EV_CURRENT', 'EV_NONE', 'EV_NUM', 'E_MIPS_ARCH_1',
    'E_MIPS_ARCH_2', 'E_MIPS_ARCH_3', 'E_MIPS_ARCH_32',
    'E_MIPS_ARCH_4', 'E_MIPS_ARCH_5', 'E_MIPS_ARCH_64', 'Elf32_Addr',
    'Elf32_Chdr', 'Elf32_Conflict', 'Elf32_Dyn', 'Elf32_Ehdr',
    'Elf32_Half', 'Elf32_Lib', 'Elf32_Move', 'Elf32_Nhdr',
    'Elf32_Off', 'Elf32_Phdr', 'Elf32_RegInfo', 'Elf32_Rel',
    'Elf32_Rela', 'Elf32_Relr', 'Elf32_Section', 'Elf32_Shdr',
    'Elf32_Sword', 'Elf32_Sxword', 'Elf32_Sym', 'Elf32_Syminfo',
    'Elf32_Verdaux', 'Elf32_Verdef', 'Elf32_Vernaux', 'Elf32_Verneed',
    'Elf32_Versym', 'Elf32_Word', 'Elf32_Xword', 'Elf32_auxv_t',
    'Elf32_gptab', 'Elf64_Addr', 'Elf64_Chdr', 'Elf64_Dyn',
    'Elf64_Ehdr', 'Elf64_Half', 'Elf64_Lib', 'Elf64_Move',
    'Elf64_Nhdr', 'Elf64_Off', 'Elf64_Phdr', 'Elf64_Rel',
    'Elf64_Rela', 'Elf64_Relr', 'Elf64_Section', 'Elf64_Shdr',
    'Elf64_Sword', 'Elf64_Sxword', 'Elf64_Sym', 'Elf64_Syminfo',
    'Elf64_Verdaux', 'Elf64_Verdef', 'Elf64_Vernaux', 'Elf64_Verneed',
    'Elf64_Versym', 'Elf64_Word', 'Elf64_Xword', 'Elf64_auxv_t',
    'Elf_MIPS_ABIFlags_v0', 'Elf_Options', 'Elf_Options_Hw', 'F_LOCK',
    'F_OK', 'F_TEST', 'F_TLOCK', 'F_ULOCK', 'GNU_PROPERTY_1_NEEDED',
    'GNU_PROPERTY_1_NEEDED_INDIRECT_EXTERN_ACCESS',
    'GNU_PROPERTY_AARCH64_FEATURE_1_AND',
    'GNU_PROPERTY_AARCH64_FEATURE_1_BTI',
    'GNU_PROPERTY_AARCH64_FEATURE_1_PAC', 'GNU_PROPERTY_HIPROC',
    'GNU_PROPERTY_HIUSER', 'GNU_PROPERTY_LOPROC',
    'GNU_PROPERTY_LOUSER', 'GNU_PROPERTY_NO_COPY_ON_PROTECTED',
    'GNU_PROPERTY_STACK_SIZE', 'GNU_PROPERTY_UINT32_AND_HI',
    'GNU_PROPERTY_UINT32_AND_LO', 'GNU_PROPERTY_UINT32_OR_HI',
    'GNU_PROPERTY_UINT32_OR_LO', 'GNU_PROPERTY_X86_FEATURE_1_AND',
    'GNU_PROPERTY_X86_FEATURE_1_IBT',
    'GNU_PROPERTY_X86_FEATURE_1_SHSTK',
    'GNU_PROPERTY_X86_ISA_1_BASELINE',
    'GNU_PROPERTY_X86_ISA_1_NEEDED', 'GNU_PROPERTY_X86_ISA_1_USED',
    'GNU_PROPERTY_X86_ISA_1_V2', 'GNU_PROPERTY_X86_ISA_1_V3',
    'GNU_PROPERTY_X86_ISA_1_V4', 'GRP_COMDAT', 'LITUSE_ALPHA_ADDR',
    'LITUSE_ALPHA_BASE', 'LITUSE_ALPHA_BYTOFF', 'LITUSE_ALPHA_JSR',
    'LITUSE_ALPHA_TLS_GD', 'LITUSE_ALPHA_TLS_LDM', 'LL_DELAY_LOAD',
    'LL_DELTA', 'LL_EXACT_MATCH', 'LL_EXPORTS', 'LL_IGNORE_INT_VER',
    'LL_NONE', 'LL_REQUIRE_MINOR', 'L_INCR', 'L_SET', 'L_XTND',
    'MADV_COLD', 'MADV_COLLAPSE', 'MADV_DODUMP', 'MADV_DOFORK',
    'MADV_DONTDUMP', 'MADV_DONTFORK', 'MADV_DONTNEED',
    'MADV_DONTNEED_LOCKED', 'MADV_FREE', 'MADV_HUGEPAGE',
    'MADV_HWPOISON', 'MADV_KEEPONFORK', 'MADV_MERGEABLE',
    'MADV_NOHUGEPAGE', 'MADV_NORMAL', 'MADV_PAGEOUT',
    'MADV_POPULATE_READ', 'MADV_POPULATE_WRITE', 'MADV_RANDOM',
    'MADV_REMOVE', 'MADV_SEQUENTIAL', 'MADV_SOFT_OFFLINE',
    'MADV_UNMERGEABLE', 'MADV_WILLNEED', 'MADV_WIPEONFORK',
    'MAP_ANONYMOUS', 'MAP_FILE', 'MAP_FIXED', 'MAP_FIXED_NOREPLACE',
    'MAP_HUGETLB', 'MAP_NONBLOCK', 'MAP_POPULATE', 'MAP_STACK',
    'MAP_SYNC', 'MAP_TYPE', 'MAP_UNINITIALIZED', 'MIPS_AFL_ASE_DSP',
    'MIPS_AFL_ASE_DSPR2', 'MIPS_AFL_ASE_EVA', 'MIPS_AFL_ASE_MASK',
    'MIPS_AFL_ASE_MCU', 'MIPS_AFL_ASE_MDMX', 'MIPS_AFL_ASE_MICROMIPS',
    'MIPS_AFL_ASE_MIPS16', 'MIPS_AFL_ASE_MIPS3D', 'MIPS_AFL_ASE_MSA',
    'MIPS_AFL_ASE_MT', 'MIPS_AFL_ASE_SMARTMIPS', 'MIPS_AFL_ASE_VIRT',
    'MIPS_AFL_ASE_XPA', 'MIPS_AFL_EXT_10000', 'MIPS_AFL_EXT_3900',
    'MIPS_AFL_EXT_4010', 'MIPS_AFL_EXT_4100', 'MIPS_AFL_EXT_4111',
    'MIPS_AFL_EXT_4120', 'MIPS_AFL_EXT_4650', 'MIPS_AFL_EXT_5400',
    'MIPS_AFL_EXT_5500', 'MIPS_AFL_EXT_5900',
    'MIPS_AFL_EXT_LOONGSON_2E', 'MIPS_AFL_EXT_LOONGSON_2F',
    'MIPS_AFL_EXT_LOONGSON_3A', 'MIPS_AFL_EXT_OCTEON',
    'MIPS_AFL_EXT_OCTEON2', 'MIPS_AFL_EXT_OCTEONP',
    'MIPS_AFL_EXT_SB1', 'MIPS_AFL_EXT_XLR',
    'MIPS_AFL_FLAGS1_ODDSPREG', 'MIPS_AFL_REG_128', 'MIPS_AFL_REG_32',
    'MIPS_AFL_REG_64', 'MIPS_AFL_REG_NONE', 'MLOCK_ONFAULT',
    'MS_ASYNC', 'MS_INVALIDATE', 'MS_SYNC',
    'NOTE_GNU_PROPERTY_SECTION_NAME', 'NT_386_IOPERM', 'NT_386_TLS',
    'NT_ARM_HW_BREAK', 'NT_ARM_HW_WATCH', 'NT_ARM_PACA_KEYS',
    'NT_ARM_PACG_KEYS', 'NT_ARM_PAC_ENABLED_KEYS', 'NT_ARM_PAC_MASK',
    'NT_ARM_SVE', 'NT_ARM_SYSTEM_CALL', 'NT_ARM_TAGGED_ADDR_CTRL',
    'NT_ARM_TLS', 'NT_ARM_VFP', 'NT_ASRS', 'NT_AUXV',
    'NT_FDO_PACKAGING_METADATA', 'NT_FILE', 'NT_FPREGSET',
    'NT_GNU_ABI_TAG', 'NT_GNU_BUILD_ID', 'NT_GNU_GOLD_VERSION',
    'NT_GNU_HWCAP', 'NT_GNU_PROPERTY_TYPE_0', 'NT_GWINDOWS',
    'NT_LOONGARCH_CPUCFG', 'NT_LOONGARCH_CSR',
    'NT_LOONGARCH_HW_BREAK', 'NT_LOONGARCH_HW_WATCH',
    'NT_LOONGARCH_LASX', 'NT_LOONGARCH_LBT', 'NT_LOONGARCH_LSX',
    'NT_LWPSINFO', 'NT_LWPSTATUS', 'NT_MIPS_DSP', 'NT_MIPS_FP_MODE',
    'NT_MIPS_MSA', 'NT_PLATFORM', 'NT_PPC_DEXCR', 'NT_PPC_DSCR',
    'NT_PPC_EBB', 'NT_PPC_HASHKEYR', 'NT_PPC_PKEY', 'NT_PPC_PMU',
    'NT_PPC_PPR', 'NT_PPC_SPE', 'NT_PPC_TAR', 'NT_PPC_TM_CDSCR',
    'NT_PPC_TM_CFPR', 'NT_PPC_TM_CGPR', 'NT_PPC_TM_CPPR',
    'NT_PPC_TM_CTAR', 'NT_PPC_TM_CVMX', 'NT_PPC_TM_CVSX',
    'NT_PPC_TM_SPR', 'NT_PPC_VMX', 'NT_PPC_VSX', 'NT_PRCRED',
    'NT_PRFPREG', 'NT_PRFPXREG', 'NT_PRPSINFO', 'NT_PRSTATUS',
    'NT_PRXFPREG', 'NT_PRXREG', 'NT_PSINFO', 'NT_PSTATUS',
    'NT_RISCV_CSR', 'NT_RISCV_VECTOR', 'NT_S390_CTRS',
    'NT_S390_GS_BC', 'NT_S390_GS_CB', 'NT_S390_HIGH_GPRS',
    'NT_S390_LAST_BREAK', 'NT_S390_PREFIX', 'NT_S390_PV_CPU_DATA',
    'NT_S390_RI_CB', 'NT_S390_SYSTEM_CALL', 'NT_S390_TDB',
    'NT_S390_TIMER', 'NT_S390_TODCMP', 'NT_S390_TODPREG',
    'NT_S390_VXRS_HIGH', 'NT_S390_VXRS_LOW', 'NT_SIGINFO',
    'NT_TASKSTRUCT', 'NT_UTSNAME', 'NT_VERSION', 'NT_VMCOREDD',
    'NT_X86_SHSTK', 'NT_X86_XSTATE', 'ODK_EXCEPTIONS', 'ODK_FILL',
    'ODK_HWAND', 'ODK_HWOR', 'ODK_HWPATCH', 'ODK_NULL', 'ODK_PAD',
    'ODK_REGINFO', 'ODK_TAGS', 'OEX_DISMISS', 'OEX_FPDBUG',
    'OEX_FPU_DIV0', 'OEX_FPU_INEX', 'OEX_FPU_INVAL', 'OEX_FPU_MAX',
    'OEX_FPU_MIN', 'OEX_FPU_OFLO', 'OEX_FPU_UFLO', 'OEX_PAGE0',
    'OEX_PRECISEFP', 'OEX_SMM', 'OHWA0_R4KEOP_CHECKED',
    'OHWA1_R4KEOP_CLEAN', 'OHW_R4KEOP', 'OHW_R5KCVTL', 'OHW_R5KEOP',
    'OHW_R8KPFETCH', 'OPAD_POSTFIX', 'OPAD_PREFIX', 'OPAD_SYMBOL',
    'PF_ARM_ABS', 'PF_ARM_PI', 'PF_ARM_SB', 'PF_HP_CODE',
    'PF_HP_FAR_SHARED', 'PF_HP_LAZYSWAP', 'PF_HP_MODIFY',
    'PF_HP_NEAR_SHARED', 'PF_HP_PAGE_SIZE', 'PF_HP_SBP',
    'PF_IA_64_NORECOV', 'PF_MASKOS', 'PF_MASKPROC', 'PF_MIPS_LOCAL',
    'PF_PARISC_SBP', 'PF_R', 'PF_W', 'PF_X', 'PKEY_ACCESS_MASK',
    'PKEY_DISABLE_ACCESS', 'PKEY_DISABLE_WRITE', 'PN_XNUM',
    'PPC64_OPT_LOCALENTRY', 'PPC64_OPT_MULTI_TOC', 'PPC64_OPT_TLS',
    'PPC_OPT_TLS', 'PROT_EXEC', 'PROT_GROWSDOWN', 'PROT_GROWSUP',
    'PROT_NONE', 'PROT_READ', 'PROT_SEM', 'PROT_WRITE',
    'PT_AARCH64_MEMTAG_MTE', 'PT_ARM_EXIDX', 'PT_DYNAMIC',
    'PT_GNU_EH_FRAME', 'PT_GNU_PROPERTY', 'PT_GNU_RELRO',
    'PT_GNU_SFRAME', 'PT_GNU_STACK', 'PT_HIOS', 'PT_HIPROC',
    'PT_HISUNW', 'PT_HP_CORE_COMM', 'PT_HP_CORE_KERNEL',
    'PT_HP_CORE_LOADABLE', 'PT_HP_CORE_MMF', 'PT_HP_CORE_NONE',
    'PT_HP_CORE_PROC', 'PT_HP_CORE_SHM', 'PT_HP_CORE_STACK',
    'PT_HP_CORE_VERSION', 'PT_HP_FASTBIND', 'PT_HP_HSL_ANNOT',
    'PT_HP_OPT_ANNOT', 'PT_HP_PARALLEL', 'PT_HP_STACK', 'PT_HP_TLS',
    'PT_IA_64_ARCHEXT', 'PT_IA_64_HP_HSL_ANOT',
    'PT_IA_64_HP_OPT_ANOT', 'PT_IA_64_HP_STACK', 'PT_IA_64_UNWIND',
    'PT_INTERP', 'PT_LOAD', 'PT_LOOS', 'PT_LOPROC', 'PT_LOSUNW',
    'PT_MIPS_ABIFLAGS', 'PT_MIPS_OPTIONS', 'PT_MIPS_REGINFO',
    'PT_MIPS_RTPROC', 'PT_NOTE', 'PT_NULL', 'PT_NUM',
    'PT_PARISC_ARCHEXT', 'PT_PARISC_UNWIND', 'PT_PHDR',
    'PT_RISCV_ATTRIBUTES', 'PT_SHLIB', 'PT_SUNWBSS', 'PT_SUNWSTACK',
    'PT_TLS', 'RHF_CORD', 'RHF_DEFAULT_DELAY_LOAD',
    'RHF_DELTA_C_PLUS_PLUS', 'RHF_GUARANTEE_INIT',
    'RHF_GUARANTEE_START_INIT', 'RHF_NONE', 'RHF_NOTPOT',
    'RHF_NO_LIBRARY_REPLACEMENT', 'RHF_NO_MOVE', 'RHF_NO_UNRES_UNDEF',
    'RHF_PIXIE', 'RHF_QUICKSTART', 'RHF_REQUICKSTART',
    'RHF_REQUICKSTARTED', 'RHF_RLD_ORDER_SAFE', 'RHF_SGI_ONLY',
    'R_386_16', 'R_386_32', 'R_386_32PLT', 'R_386_8', 'R_386_COPY',
    'R_386_GLOB_DAT', 'R_386_GOT32', 'R_386_GOT32X', 'R_386_GOTOFF',
    'R_386_GOTPC', 'R_386_IRELATIVE', 'R_386_JMP_SLOT', 'R_386_NONE',
    'R_386_NUM', 'R_386_PC16', 'R_386_PC32', 'R_386_PC8',
    'R_386_PLT32', 'R_386_RELATIVE', 'R_386_SIZE32', 'R_386_TLS_DESC',
    'R_386_TLS_DESC_CALL', 'R_386_TLS_DTPMOD32', 'R_386_TLS_DTPOFF32',
    'R_386_TLS_GD', 'R_386_TLS_GD_32', 'R_386_TLS_GD_CALL',
    'R_386_TLS_GD_POP', 'R_386_TLS_GD_PUSH', 'R_386_TLS_GOTDESC',
    'R_386_TLS_GOTIE', 'R_386_TLS_IE', 'R_386_TLS_IE_32',
    'R_386_TLS_LDM', 'R_386_TLS_LDM_32', 'R_386_TLS_LDM_CALL',
    'R_386_TLS_LDM_POP', 'R_386_TLS_LDM_PUSH', 'R_386_TLS_LDO_32',
    'R_386_TLS_LE', 'R_386_TLS_LE_32', 'R_386_TLS_TPOFF',
    'R_386_TLS_TPOFF32', 'R_390_12', 'R_390_16', 'R_390_20',
    'R_390_32', 'R_390_64', 'R_390_8', 'R_390_COPY', 'R_390_GLOB_DAT',
    'R_390_GOT12', 'R_390_GOT16', 'R_390_GOT20', 'R_390_GOT32',
    'R_390_GOT64', 'R_390_GOTENT', 'R_390_GOTOFF16', 'R_390_GOTOFF32',
    'R_390_GOTOFF64', 'R_390_GOTPC', 'R_390_GOTPCDBL',
    'R_390_GOTPLT12', 'R_390_GOTPLT16', 'R_390_GOTPLT20',
    'R_390_GOTPLT32', 'R_390_GOTPLT64', 'R_390_GOTPLTENT',
    'R_390_IRELATIVE', 'R_390_JMP_SLOT', 'R_390_NONE', 'R_390_NUM',
    'R_390_PC16', 'R_390_PC16DBL', 'R_390_PC32', 'R_390_PC32DBL',
    'R_390_PC64', 'R_390_PLT16DBL', 'R_390_PLT32', 'R_390_PLT32DBL',
    'R_390_PLT64', 'R_390_PLTOFF16', 'R_390_PLTOFF32',
    'R_390_PLTOFF64', 'R_390_RELATIVE', 'R_390_TLS_DTPMOD',
    'R_390_TLS_DTPOFF', 'R_390_TLS_GD32', 'R_390_TLS_GD64',
    'R_390_TLS_GDCALL', 'R_390_TLS_GOTIE12', 'R_390_TLS_GOTIE20',
    'R_390_TLS_GOTIE32', 'R_390_TLS_GOTIE64', 'R_390_TLS_IE32',
    'R_390_TLS_IE64', 'R_390_TLS_IEENT', 'R_390_TLS_LDCALL',
    'R_390_TLS_LDM32', 'R_390_TLS_LDM64', 'R_390_TLS_LDO32',
    'R_390_TLS_LDO64', 'R_390_TLS_LE32', 'R_390_TLS_LE64',
    'R_390_TLS_LOAD', 'R_390_TLS_TPOFF', 'R_68K_16', 'R_68K_32',
    'R_68K_8', 'R_68K_COPY', 'R_68K_GLOB_DAT', 'R_68K_GOT16',
    'R_68K_GOT16O', 'R_68K_GOT32', 'R_68K_GOT32O', 'R_68K_GOT8',
    'R_68K_GOT8O', 'R_68K_JMP_SLOT', 'R_68K_NONE', 'R_68K_NUM',
    'R_68K_PC16', 'R_68K_PC32', 'R_68K_PC8', 'R_68K_PLT16',
    'R_68K_PLT16O', 'R_68K_PLT32', 'R_68K_PLT32O', 'R_68K_PLT8',
    'R_68K_PLT8O', 'R_68K_RELATIVE', 'R_68K_TLS_DTPMOD32',
    'R_68K_TLS_DTPREL32', 'R_68K_TLS_GD16', 'R_68K_TLS_GD32',
    'R_68K_TLS_GD8', 'R_68K_TLS_IE16', 'R_68K_TLS_IE32',
    'R_68K_TLS_IE8', 'R_68K_TLS_LDM16', 'R_68K_TLS_LDM32',
    'R_68K_TLS_LDM8', 'R_68K_TLS_LDO16', 'R_68K_TLS_LDO32',
    'R_68K_TLS_LDO8', 'R_68K_TLS_LE16', 'R_68K_TLS_LE32',
    'R_68K_TLS_LE8', 'R_68K_TLS_TPREL32', 'R_AARCH64_ABS16',
    'R_AARCH64_ABS32', 'R_AARCH64_ABS64', 'R_AARCH64_ADD_ABS_LO12_NC',
    'R_AARCH64_ADR_GOT_PAGE', 'R_AARCH64_ADR_PREL_LO21',
    'R_AARCH64_ADR_PREL_PG_HI21', 'R_AARCH64_ADR_PREL_PG_HI21_NC',
    'R_AARCH64_CALL26', 'R_AARCH64_CONDBR19', 'R_AARCH64_COPY',
    'R_AARCH64_GLOB_DAT', 'R_AARCH64_GOTREL32', 'R_AARCH64_GOTREL64',
    'R_AARCH64_GOT_LD_PREL19', 'R_AARCH64_IRELATIVE',
    'R_AARCH64_JUMP26', 'R_AARCH64_JUMP_SLOT',
    'R_AARCH64_LD64_GOTOFF_LO15', 'R_AARCH64_LD64_GOTPAGE_LO15',
    'R_AARCH64_LD64_GOT_LO12_NC', 'R_AARCH64_LDST128_ABS_LO12_NC',
    'R_AARCH64_LDST16_ABS_LO12_NC', 'R_AARCH64_LDST32_ABS_LO12_NC',
    'R_AARCH64_LDST64_ABS_LO12_NC', 'R_AARCH64_LDST8_ABS_LO12_NC',
    'R_AARCH64_LD_PREL_LO19', 'R_AARCH64_MOVW_GOTOFF_G0',
    'R_AARCH64_MOVW_GOTOFF_G0_NC', 'R_AARCH64_MOVW_GOTOFF_G1',
    'R_AARCH64_MOVW_GOTOFF_G1_NC', 'R_AARCH64_MOVW_GOTOFF_G2',
    'R_AARCH64_MOVW_GOTOFF_G2_NC', 'R_AARCH64_MOVW_GOTOFF_G3',
    'R_AARCH64_MOVW_PREL_G0', 'R_AARCH64_MOVW_PREL_G0_NC',
    'R_AARCH64_MOVW_PREL_G1', 'R_AARCH64_MOVW_PREL_G1_NC',
    'R_AARCH64_MOVW_PREL_G2', 'R_AARCH64_MOVW_PREL_G2_NC',
    'R_AARCH64_MOVW_PREL_G3', 'R_AARCH64_MOVW_SABS_G0',
    'R_AARCH64_MOVW_SABS_G1', 'R_AARCH64_MOVW_SABS_G2',
    'R_AARCH64_MOVW_UABS_G0', 'R_AARCH64_MOVW_UABS_G0_NC',
    'R_AARCH64_MOVW_UABS_G1', 'R_AARCH64_MOVW_UABS_G1_NC',
    'R_AARCH64_MOVW_UABS_G2', 'R_AARCH64_MOVW_UABS_G2_NC',
    'R_AARCH64_MOVW_UABS_G3', 'R_AARCH64_NONE', 'R_AARCH64_P32_ABS32',
    'R_AARCH64_P32_COPY', 'R_AARCH64_P32_GLOB_DAT',
    'R_AARCH64_P32_IRELATIVE', 'R_AARCH64_P32_JUMP_SLOT',
    'R_AARCH64_P32_RELATIVE', 'R_AARCH64_P32_TLSDESC',
    'R_AARCH64_P32_TLS_DTPMOD', 'R_AARCH64_P32_TLS_DTPREL',
    'R_AARCH64_P32_TLS_TPREL', 'R_AARCH64_PREL16', 'R_AARCH64_PREL32',
    'R_AARCH64_PREL64', 'R_AARCH64_RELATIVE', 'R_AARCH64_TLSDESC',
    'R_AARCH64_TLSDESC_ADD', 'R_AARCH64_TLSDESC_ADD_LO12',
    'R_AARCH64_TLSDESC_ADR_PAGE21', 'R_AARCH64_TLSDESC_ADR_PREL21',
    'R_AARCH64_TLSDESC_CALL', 'R_AARCH64_TLSDESC_LD64_LO12',
    'R_AARCH64_TLSDESC_LDR', 'R_AARCH64_TLSDESC_LD_PREL19',
    'R_AARCH64_TLSDESC_OFF_G0_NC', 'R_AARCH64_TLSDESC_OFF_G1',
    'R_AARCH64_TLSGD_ADD_LO12_NC', 'R_AARCH64_TLSGD_ADR_PAGE21',
    'R_AARCH64_TLSGD_ADR_PREL21', 'R_AARCH64_TLSGD_MOVW_G0_NC',
    'R_AARCH64_TLSGD_MOVW_G1', 'R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21',
    'R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC',
    'R_AARCH64_TLSIE_LD_GOTTPREL_PREL19',
    'R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC',
    'R_AARCH64_TLSIE_MOVW_GOTTPREL_G1',
    'R_AARCH64_TLSLD_ADD_DTPREL_HI12',
    'R_AARCH64_TLSLD_ADD_DTPREL_LO12',
    'R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC',
    'R_AARCH64_TLSLD_ADD_LO12_NC', 'R_AARCH64_TLSLD_ADR_PAGE21',
    'R_AARCH64_TLSLD_ADR_PREL21',
    'R_AARCH64_TLSLD_LDST128_DTPREL_LO12',
    'R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC',
    'R_AARCH64_TLSLD_LDST16_DTPREL_LO12',
    'R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC',
    'R_AARCH64_TLSLD_LDST32_DTPREL_LO12',
    'R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC',
    'R_AARCH64_TLSLD_LDST64_DTPREL_LO12',
    'R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC',
    'R_AARCH64_TLSLD_LDST8_DTPREL_LO12',
    'R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC',
    'R_AARCH64_TLSLD_LD_PREL19', 'R_AARCH64_TLSLD_MOVW_DTPREL_G0',
    'R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC',
    'R_AARCH64_TLSLD_MOVW_DTPREL_G1',
    'R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC',
    'R_AARCH64_TLSLD_MOVW_DTPREL_G2', 'R_AARCH64_TLSLD_MOVW_G0_NC',
    'R_AARCH64_TLSLD_MOVW_G1', 'R_AARCH64_TLSLE_ADD_TPREL_HI12',
    'R_AARCH64_TLSLE_ADD_TPREL_LO12',
    'R_AARCH64_TLSLE_ADD_TPREL_LO12_NC',
    'R_AARCH64_TLSLE_LDST128_TPREL_LO12',
    'R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC',
    'R_AARCH64_TLSLE_LDST16_TPREL_LO12',
    'R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC',
    'R_AARCH64_TLSLE_LDST32_TPREL_LO12',
    'R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC',
    'R_AARCH64_TLSLE_LDST64_TPREL_LO12',
    'R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC',
    'R_AARCH64_TLSLE_LDST8_TPREL_LO12',
    'R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC',
    'R_AARCH64_TLSLE_MOVW_TPREL_G0',
    'R_AARCH64_TLSLE_MOVW_TPREL_G0_NC',
    'R_AARCH64_TLSLE_MOVW_TPREL_G1',
    'R_AARCH64_TLSLE_MOVW_TPREL_G1_NC',
    'R_AARCH64_TLSLE_MOVW_TPREL_G2', 'R_AARCH64_TLS_DTPMOD',
    'R_AARCH64_TLS_DTPREL', 'R_AARCH64_TLS_TPREL',
    'R_AARCH64_TSTBR14', 'R_AC_SECTOFF_S9', 'R_AC_SECTOFF_S9_1',
    'R_AC_SECTOFF_S9_2', 'R_AC_SECTOFF_U8', 'R_AC_SECTOFF_U8_1',
    'R_AC_SECTOFF_U8_2', 'R_ALPHA_BRADDR', 'R_ALPHA_COPY',
    'R_ALPHA_DTPMOD64', 'R_ALPHA_DTPREL16', 'R_ALPHA_DTPREL64',
    'R_ALPHA_DTPRELHI', 'R_ALPHA_DTPRELLO', 'R_ALPHA_GLOB_DAT',
    'R_ALPHA_GOTDTPREL', 'R_ALPHA_GOTTPREL', 'R_ALPHA_GPDISP',
    'R_ALPHA_GPREL16', 'R_ALPHA_GPREL32', 'R_ALPHA_GPRELHIGH',
    'R_ALPHA_GPRELLOW', 'R_ALPHA_HINT', 'R_ALPHA_JMP_SLOT',
    'R_ALPHA_LITERAL', 'R_ALPHA_LITUSE', 'R_ALPHA_NONE',
    'R_ALPHA_NUM', 'R_ALPHA_REFLONG', 'R_ALPHA_REFQUAD',
    'R_ALPHA_RELATIVE', 'R_ALPHA_SREL16', 'R_ALPHA_SREL32',
    'R_ALPHA_SREL64', 'R_ALPHA_TLSGD', 'R_ALPHA_TLS_GD_HI',
    'R_ALPHA_TLS_LDM', 'R_ALPHA_TPREL16', 'R_ALPHA_TPREL64',
    'R_ALPHA_TPRELHI', 'R_ALPHA_TPRELLO', 'R_ARC_16', 'R_ARC_24',
    'R_ARC_32', 'R_ARC_32_ME', 'R_ARC_32_PCREL', 'R_ARC_8',
    'R_ARC_B22_PCREL', 'R_ARC_COPY', 'R_ARC_GLOB_DAT', 'R_ARC_GOT32',
    'R_ARC_GOTOFF', 'R_ARC_GOTPC', 'R_ARC_GOTPC32', 'R_ARC_H30',
    'R_ARC_H30_ME', 'R_ARC_JLI_SECTOFF', 'R_ARC_JMP_SLOT',
    'R_ARC_N16', 'R_ARC_N24', 'R_ARC_N32', 'R_ARC_N32_ME', 'R_ARC_N8',
    'R_ARC_NONE', 'R_ARC_NPS_CMEM16', 'R_ARC_PC32', 'R_ARC_PLT32',
    'R_ARC_RELATIVE', 'R_ARC_S13_PCREL', 'R_ARC_S21H_PCREL',
    'R_ARC_S21H_PCREL_PLT', 'R_ARC_S21W_PCREL',
    'R_ARC_S21W_PCREL_PLT', 'R_ARC_S25H_PCREL',
    'R_ARC_S25H_PCREL_PLT', 'R_ARC_S25W_PCREL',
    'R_ARC_S25W_PCREL_PLT', 'R_ARC_SDA', 'R_ARC_SDA16_LD',
    'R_ARC_SDA16_LD1', 'R_ARC_SDA16_LD2', 'R_ARC_SDA16_ST2',
    'R_ARC_SDA32', 'R_ARC_SDA32_ME', 'R_ARC_SDA_12', 'R_ARC_SDA_LDST',
    'R_ARC_SDA_LDST1', 'R_ARC_SDA_LDST2', 'R_ARC_SECTOFF',
    'R_ARC_SECTOFF_1', 'R_ARC_SECTOFF_2', 'R_ARC_SECTOFF_ME',
    'R_ARC_SECTOFF_ME_1', 'R_ARC_SECTOFF_ME_2', 'R_ARC_SECTOFF_S9',
    'R_ARC_SECTOFF_U8', 'R_ARC_TLS_DTPMOD', 'R_ARC_TLS_DTPOFF',
    'R_ARC_TLS_DTPOFF_S9', 'R_ARC_TLS_GD_CALL', 'R_ARC_TLS_GD_GOT',
    'R_ARC_TLS_GD_LD', 'R_ARC_TLS_IE_GOT', 'R_ARC_TLS_LE_32',
    'R_ARC_TLS_LE_S9', 'R_ARC_TLS_TPOFF', 'R_ARC_W', 'R_ARC_W_ME',
    'R_ARM_ABS12', 'R_ARM_ABS16', 'R_ARM_ABS32', 'R_ARM_ABS32_NOI',
    'R_ARM_ABS8', 'R_ARM_ALU_PCREL_15_8', 'R_ARM_ALU_PCREL_23_15',
    'R_ARM_ALU_PCREL_7_0', 'R_ARM_ALU_PC_G0', 'R_ARM_ALU_PC_G0_NC',
    'R_ARM_ALU_PC_G1', 'R_ARM_ALU_PC_G1_NC', 'R_ARM_ALU_PC_G2',
    'R_ARM_ALU_SBREL_19_12', 'R_ARM_ALU_SBREL_27_20',
    'R_ARM_ALU_SB_G0', 'R_ARM_ALU_SB_G0_NC', 'R_ARM_ALU_SB_G1',
    'R_ARM_ALU_SB_G1_NC', 'R_ARM_ALU_SB_G2', 'R_ARM_AMP_VCALL9',
    'R_ARM_BASE_ABS', 'R_ARM_CALL', 'R_ARM_COPY', 'R_ARM_GLOB_DAT',
    'R_ARM_GNU_VTENTRY', 'R_ARM_GNU_VTINHERIT', 'R_ARM_GOT32',
    'R_ARM_GOTOFF', 'R_ARM_GOTOFF12', 'R_ARM_GOTPC', 'R_ARM_GOTRELAX',
    'R_ARM_GOT_ABS', 'R_ARM_GOT_BREL12', 'R_ARM_GOT_PREL',
    'R_ARM_IRELATIVE', 'R_ARM_JUMP24', 'R_ARM_JUMP_SLOT',
    'R_ARM_LDC_PC_G0', 'R_ARM_LDC_PC_G1', 'R_ARM_LDC_PC_G2',
    'R_ARM_LDC_SB_G0', 'R_ARM_LDC_SB_G1', 'R_ARM_LDC_SB_G2',
    'R_ARM_LDRS_PC_G0', 'R_ARM_LDRS_PC_G1', 'R_ARM_LDRS_PC_G2',
    'R_ARM_LDRS_SB_G0', 'R_ARM_LDRS_SB_G1', 'R_ARM_LDRS_SB_G2',
    'R_ARM_LDR_PC_G1', 'R_ARM_LDR_PC_G2', 'R_ARM_LDR_SBREL_11_0',
    'R_ARM_LDR_SB_G0', 'R_ARM_LDR_SB_G1', 'R_ARM_LDR_SB_G2',
    'R_ARM_ME_TOO', 'R_ARM_MOVT_ABS', 'R_ARM_MOVT_BREL',
    'R_ARM_MOVT_PREL', 'R_ARM_MOVW_ABS_NC', 'R_ARM_MOVW_BREL',
    'R_ARM_MOVW_BREL_NC', 'R_ARM_MOVW_PREL_NC', 'R_ARM_NONE',
    'R_ARM_NUM', 'R_ARM_PC13', 'R_ARM_PC24', 'R_ARM_PLT32',
    'R_ARM_PLT32_ABS', 'R_ARM_PREL31', 'R_ARM_RABS22', 'R_ARM_RBASE',
    'R_ARM_REL32', 'R_ARM_REL32_NOI', 'R_ARM_RELATIVE', 'R_ARM_RPC24',
    'R_ARM_RREL32', 'R_ARM_RSBREL32', 'R_ARM_RXPC25', 'R_ARM_SBREL31',
    'R_ARM_SBREL32', 'R_ARM_SWI24', 'R_ARM_TARGET1', 'R_ARM_TARGET2',
    'R_ARM_THM_ABS5', 'R_ARM_THM_ALU_PREL_11_0',
    'R_ARM_THM_GOT_BREL12', 'R_ARM_THM_JUMP19', 'R_ARM_THM_JUMP24',
    'R_ARM_THM_JUMP6', 'R_ARM_THM_MOVT_ABS', 'R_ARM_THM_MOVT_BREL',
    'R_ARM_THM_MOVT_PREL', 'R_ARM_THM_MOVW_ABS_NC',
    'R_ARM_THM_MOVW_BREL', 'R_ARM_THM_MOVW_BREL_NC',
    'R_ARM_THM_MOVW_PREL_NC', 'R_ARM_THM_PC11', 'R_ARM_THM_PC12',
    'R_ARM_THM_PC22', 'R_ARM_THM_PC8', 'R_ARM_THM_PC9',
    'R_ARM_THM_RPC22', 'R_ARM_THM_SWI8', 'R_ARM_THM_TLS_CALL',
    'R_ARM_THM_TLS_DESCSEQ', 'R_ARM_THM_TLS_DESCSEQ16',
    'R_ARM_THM_TLS_DESCSEQ32', 'R_ARM_THM_XPC22', 'R_ARM_TLS_CALL',
    'R_ARM_TLS_DESC', 'R_ARM_TLS_DESCSEQ', 'R_ARM_TLS_DTPMOD32',
    'R_ARM_TLS_DTPOFF32', 'R_ARM_TLS_GD32', 'R_ARM_TLS_GOTDESC',
    'R_ARM_TLS_IE12GP', 'R_ARM_TLS_IE32', 'R_ARM_TLS_LDM32',
    'R_ARM_TLS_LDO12', 'R_ARM_TLS_LDO32', 'R_ARM_TLS_LE12',
    'R_ARM_TLS_LE32', 'R_ARM_TLS_TPOFF32', 'R_ARM_V4BX',
    'R_ARM_XPC25', 'R_BPF_64_32', 'R_BPF_64_64', 'R_BPF_NONE',
    'R_CKCORE_ADDR32', 'R_CKCORE_ADDRGOT', 'R_CKCORE_ADDRGOT_HI16',
    'R_CKCORE_ADDRGOT_LO16', 'R_CKCORE_ADDRPLT',
    'R_CKCORE_ADDRPLT_HI16', 'R_CKCORE_ADDRPLT_LO16',
    'R_CKCORE_ADDR_HI16', 'R_CKCORE_ADDR_LO16', 'R_CKCORE_COPY',
    'R_CKCORE_DOFFSET_IMM18', 'R_CKCORE_DOFFSET_IMM18BY2',
    'R_CKCORE_DOFFSET_IMM18BY4', 'R_CKCORE_DOFFSET_LO16',
    'R_CKCORE_GLOB_DAT', 'R_CKCORE_GOT12', 'R_CKCORE_GOT32',
    'R_CKCORE_GOTOFF', 'R_CKCORE_GOTOFF_HI16', 'R_CKCORE_GOTOFF_LO16',
    'R_CKCORE_GOTPC', 'R_CKCORE_GOTPC_HI16', 'R_CKCORE_GOTPC_LO16',
    'R_CKCORE_GOT_HI16', 'R_CKCORE_GOT_IMM18BY4', 'R_CKCORE_GOT_LO16',
    'R_CKCORE_JUMP_SLOT', 'R_CKCORE_NONE', 'R_CKCORE_PCREL32',
    'R_CKCORE_PCRELIMM11BY2', 'R_CKCORE_PCRELIMM8BY4',
    'R_CKCORE_PCRELJSR_IMM11BY2', 'R_CKCORE_PCREL_IMM10BY2',
    'R_CKCORE_PCREL_IMM10BY4', 'R_CKCORE_PCREL_IMM16BY2',
    'R_CKCORE_PCREL_IMM16BY4', 'R_CKCORE_PCREL_IMM18BY2',
    'R_CKCORE_PCREL_IMM26BY2', 'R_CKCORE_PCREL_IMM7BY4',
    'R_CKCORE_PCREL_JSR_IMM26BY2', 'R_CKCORE_PLT12', 'R_CKCORE_PLT32',
    'R_CKCORE_PLT_HI16', 'R_CKCORE_PLT_IMM18BY4', 'R_CKCORE_PLT_LO16',
    'R_CKCORE_RELATIVE', 'R_CKCORE_TLS_DTPMOD32',
    'R_CKCORE_TLS_DTPOFF32', 'R_CKCORE_TLS_GD32', 'R_CKCORE_TLS_IE32',
    'R_CKCORE_TLS_LDM32', 'R_CKCORE_TLS_LDO32', 'R_CKCORE_TLS_LE32',
    'R_CKCORE_TLS_TPOFF32', 'R_CKCORE_TOFFSET_LO16', 'R_CRIS_16',
    'R_CRIS_16_GOT', 'R_CRIS_16_GOTPLT', 'R_CRIS_16_PCREL',
    'R_CRIS_32', 'R_CRIS_32_GOT', 'R_CRIS_32_GOTPLT',
    'R_CRIS_32_GOTREL', 'R_CRIS_32_PCREL', 'R_CRIS_32_PLT_GOTREL',
    'R_CRIS_32_PLT_PCREL', 'R_CRIS_8', 'R_CRIS_8_PCREL',
    'R_CRIS_COPY', 'R_CRIS_GLOB_DAT', 'R_CRIS_GNU_VTENTRY',
    'R_CRIS_GNU_VTINHERIT', 'R_CRIS_JUMP_SLOT', 'R_CRIS_NONE',
    'R_CRIS_NUM', 'R_CRIS_RELATIVE', 'R_IA64_COPY', 'R_IA64_DIR32LSB',
    'R_IA64_DIR32MSB', 'R_IA64_DIR64LSB', 'R_IA64_DIR64MSB',
    'R_IA64_DTPMOD64LSB', 'R_IA64_DTPMOD64MSB', 'R_IA64_DTPREL14',
    'R_IA64_DTPREL22', 'R_IA64_DTPREL32LSB', 'R_IA64_DTPREL32MSB',
    'R_IA64_DTPREL64I', 'R_IA64_DTPREL64LSB', 'R_IA64_DTPREL64MSB',
    'R_IA64_FPTR32LSB', 'R_IA64_FPTR32MSB', 'R_IA64_FPTR64I',
    'R_IA64_FPTR64LSB', 'R_IA64_FPTR64MSB', 'R_IA64_GPREL22',
    'R_IA64_GPREL32LSB', 'R_IA64_GPREL32MSB', 'R_IA64_GPREL64I',
    'R_IA64_GPREL64LSB', 'R_IA64_GPREL64MSB', 'R_IA64_IMM14',
    'R_IA64_IMM22', 'R_IA64_IMM64', 'R_IA64_IPLTLSB',
    'R_IA64_IPLTMSB', 'R_IA64_LDXMOV', 'R_IA64_LTOFF22',
    'R_IA64_LTOFF22X', 'R_IA64_LTOFF64I', 'R_IA64_LTOFF_DTPMOD22',
    'R_IA64_LTOFF_DTPREL22', 'R_IA64_LTOFF_FPTR22',
    'R_IA64_LTOFF_FPTR32LSB', 'R_IA64_LTOFF_FPTR32MSB',
    'R_IA64_LTOFF_FPTR64I', 'R_IA64_LTOFF_FPTR64LSB',
    'R_IA64_LTOFF_FPTR64MSB', 'R_IA64_LTOFF_TPREL22',
    'R_IA64_LTV32LSB', 'R_IA64_LTV32MSB', 'R_IA64_LTV64LSB',
    'R_IA64_LTV64MSB', 'R_IA64_NONE', 'R_IA64_PCREL21B',
    'R_IA64_PCREL21BI', 'R_IA64_PCREL21F', 'R_IA64_PCREL21M',
    'R_IA64_PCREL22', 'R_IA64_PCREL32LSB', 'R_IA64_PCREL32MSB',
    'R_IA64_PCREL60B', 'R_IA64_PCREL64I', 'R_IA64_PCREL64LSB',
    'R_IA64_PCREL64MSB', 'R_IA64_PLTOFF22', 'R_IA64_PLTOFF64I',
    'R_IA64_PLTOFF64LSB', 'R_IA64_PLTOFF64MSB', 'R_IA64_REL32LSB',
    'R_IA64_REL32MSB', 'R_IA64_REL64LSB', 'R_IA64_REL64MSB',
    'R_IA64_SECREL32LSB', 'R_IA64_SECREL32MSB', 'R_IA64_SECREL64LSB',
    'R_IA64_SECREL64MSB', 'R_IA64_SEGREL32LSB', 'R_IA64_SEGREL32MSB',
    'R_IA64_SEGREL64LSB', 'R_IA64_SEGREL64MSB', 'R_IA64_SUB',
    'R_IA64_TPREL14', 'R_IA64_TPREL22', 'R_IA64_TPREL64I',
    'R_IA64_TPREL64LSB', 'R_IA64_TPREL64MSB', 'R_LARCH_32',
    'R_LARCH_32_PCREL', 'R_LARCH_64', 'R_LARCH_64_PCREL',
    'R_LARCH_ABS64_HI12', 'R_LARCH_ABS64_LO20', 'R_LARCH_ABS_HI20',
    'R_LARCH_ABS_LO12', 'R_LARCH_ADD16', 'R_LARCH_ADD24',
    'R_LARCH_ADD32', 'R_LARCH_ADD6', 'R_LARCH_ADD64', 'R_LARCH_ADD8',
    'R_LARCH_ADD_ULEB128', 'R_LARCH_ALIGN', 'R_LARCH_B16',
    'R_LARCH_B21', 'R_LARCH_B26', 'R_LARCH_CFA', 'R_LARCH_COPY',
    'R_LARCH_DELETE', 'R_LARCH_GNU_VTENTRY', 'R_LARCH_GNU_VTINHERIT',
    'R_LARCH_GOT64_HI12', 'R_LARCH_GOT64_LO20',
    'R_LARCH_GOT64_PC_HI12', 'R_LARCH_GOT64_PC_LO20',
    'R_LARCH_GOT_HI20', 'R_LARCH_GOT_LO12', 'R_LARCH_GOT_PC_HI20',
    'R_LARCH_GOT_PC_LO12', 'R_LARCH_IRELATIVE', 'R_LARCH_JUMP_SLOT',
    'R_LARCH_MARK_LA', 'R_LARCH_MARK_PCREL', 'R_LARCH_NONE',
    'R_LARCH_PCALA64_HI12', 'R_LARCH_PCALA64_LO20',
    'R_LARCH_PCALA_HI20', 'R_LARCH_PCALA_LO12', 'R_LARCH_PCREL20_S2',
    'R_LARCH_RELATIVE', 'R_LARCH_RELAX', 'R_LARCH_SOP_ADD',
    'R_LARCH_SOP_AND', 'R_LARCH_SOP_ASSERT', 'R_LARCH_SOP_IF_ELSE',
    'R_LARCH_SOP_NOT', 'R_LARCH_SOP_POP_32_S_0_10_10_16_S2',
    'R_LARCH_SOP_POP_32_S_0_5_10_16_S2', 'R_LARCH_SOP_POP_32_S_10_12',
    'R_LARCH_SOP_POP_32_S_10_16', 'R_LARCH_SOP_POP_32_S_10_16_S2',
    'R_LARCH_SOP_POP_32_S_10_5', 'R_LARCH_SOP_POP_32_S_5_20',
    'R_LARCH_SOP_POP_32_U', 'R_LARCH_SOP_POP_32_U_10_12',
    'R_LARCH_SOP_PUSH_ABSOLUTE', 'R_LARCH_SOP_PUSH_DUP',
    'R_LARCH_SOP_PUSH_GPREL', 'R_LARCH_SOP_PUSH_PCREL',
    'R_LARCH_SOP_PUSH_PLT_PCREL', 'R_LARCH_SOP_PUSH_TLS_GD',
    'R_LARCH_SOP_PUSH_TLS_GOT', 'R_LARCH_SOP_PUSH_TLS_TPREL',
    'R_LARCH_SOP_SL', 'R_LARCH_SOP_SR', 'R_LARCH_SOP_SUB',
    'R_LARCH_SUB16', 'R_LARCH_SUB24', 'R_LARCH_SUB32', 'R_LARCH_SUB6',
    'R_LARCH_SUB64', 'R_LARCH_SUB8', 'R_LARCH_SUB_ULEB128',
    'R_LARCH_TLS_DTPMOD32', 'R_LARCH_TLS_DTPMOD64',
    'R_LARCH_TLS_DTPREL32', 'R_LARCH_TLS_DTPREL64',
    'R_LARCH_TLS_GD_HI20', 'R_LARCH_TLS_GD_PC_HI20',
    'R_LARCH_TLS_IE64_HI12', 'R_LARCH_TLS_IE64_LO20',
    'R_LARCH_TLS_IE64_PC_HI12', 'R_LARCH_TLS_IE64_PC_LO20',
    'R_LARCH_TLS_IE_HI20', 'R_LARCH_TLS_IE_LO12',
    'R_LARCH_TLS_IE_PC_HI20', 'R_LARCH_TLS_IE_PC_LO12',
    'R_LARCH_TLS_LD_HI20', 'R_LARCH_TLS_LD_PC_HI20',
    'R_LARCH_TLS_LE64_HI12', 'R_LARCH_TLS_LE64_LO20',
    'R_LARCH_TLS_LE_HI20', 'R_LARCH_TLS_LE_LO12',
    'R_LARCH_TLS_TPREL32', 'R_LARCH_TLS_TPREL64', 'R_M32R_10_PCREL',
    'R_M32R_10_PCREL_RELA', 'R_M32R_16', 'R_M32R_16_RELA',
    'R_M32R_18_PCREL', 'R_M32R_18_PCREL_RELA', 'R_M32R_24',
    'R_M32R_24_RELA', 'R_M32R_26_PCREL', 'R_M32R_26_PCREL_RELA',
    'R_M32R_26_PLTREL', 'R_M32R_32', 'R_M32R_32_RELA', 'R_M32R_COPY',
    'R_M32R_GLOB_DAT', 'R_M32R_GNU_VTENTRY', 'R_M32R_GNU_VTINHERIT',
    'R_M32R_GOT16_HI_SLO', 'R_M32R_GOT16_HI_ULO', 'R_M32R_GOT16_LO',
    'R_M32R_GOT24', 'R_M32R_GOTOFF', 'R_M32R_GOTOFF_HI_SLO',
    'R_M32R_GOTOFF_HI_ULO', 'R_M32R_GOTOFF_LO', 'R_M32R_GOTPC24',
    'R_M32R_GOTPC_HI_SLO', 'R_M32R_GOTPC_HI_ULO', 'R_M32R_GOTPC_LO',
    'R_M32R_HI16_SLO', 'R_M32R_HI16_SLO_RELA', 'R_M32R_HI16_ULO',
    'R_M32R_HI16_ULO_RELA', 'R_M32R_JMP_SLOT', 'R_M32R_LO16',
    'R_M32R_LO16_RELA', 'R_M32R_NONE', 'R_M32R_NUM', 'R_M32R_REL32',
    'R_M32R_RELATIVE', 'R_M32R_RELA_GNU_VTENTRY',
    'R_M32R_RELA_GNU_VTINHERIT', 'R_M32R_SDA16', 'R_M32R_SDA16_RELA',
    'R_METAG_ADDR32', 'R_METAG_COPY', 'R_METAG_GETSETOFF',
    'R_METAG_GETSET_GOT', 'R_METAG_GETSET_GOTOFF', 'R_METAG_GLOB_DAT',
    'R_METAG_GNU_VTENTRY', 'R_METAG_GNU_VTINHERIT', 'R_METAG_GOTOFF',
    'R_METAG_HI16_GOTOFF', 'R_METAG_HI16_GOTPC', 'R_METAG_HI16_PLT',
    'R_METAG_HIADDR16', 'R_METAG_HIOG', 'R_METAG_JMP_SLOT',
    'R_METAG_LO16_GOTOFF', 'R_METAG_LO16_GOTPC', 'R_METAG_LO16_PLT',
    'R_METAG_LOADDR16', 'R_METAG_LOOG', 'R_METAG_NONE', 'R_METAG_PLT',
    'R_METAG_REG16OP1', 'R_METAG_REG16OP2', 'R_METAG_REG16OP3',
    'R_METAG_REG32OP1', 'R_METAG_REG32OP2', 'R_METAG_REG32OP3',
    'R_METAG_REG32OP4', 'R_METAG_REL16', 'R_METAG_REL8',
    'R_METAG_RELATIVE', 'R_METAG_RELBRANCH', 'R_METAG_RELBRANCH_PLT',
    'R_METAG_TLS_DTPMOD', 'R_METAG_TLS_DTPOFF', 'R_METAG_TLS_GD',
    'R_METAG_TLS_IE', 'R_METAG_TLS_IENONPIC',
    'R_METAG_TLS_IENONPIC_HI16', 'R_METAG_TLS_IENONPIC_LO16',
    'R_METAG_TLS_LDM', 'R_METAG_TLS_LDO', 'R_METAG_TLS_LDO_HI16',
    'R_METAG_TLS_LDO_LO16', 'R_METAG_TLS_LE', 'R_METAG_TLS_LE_HI16',
    'R_METAG_TLS_LE_LO16', 'R_METAG_TLS_TPOFF', 'R_MICROBLAZE_32',
    'R_MICROBLAZE_32_LO', 'R_MICROBLAZE_32_PCREL',
    'R_MICROBLAZE_32_PCREL_LO', 'R_MICROBLAZE_32_SYM_OP_SYM',
    'R_MICROBLAZE_64', 'R_MICROBLAZE_64_NONE',
    'R_MICROBLAZE_64_PCREL', 'R_MICROBLAZE_COPY',
    'R_MICROBLAZE_GLOB_DAT', 'R_MICROBLAZE_GNU_VTENTRY',
    'R_MICROBLAZE_GNU_VTINHERIT', 'R_MICROBLAZE_GOTOFF_32',
    'R_MICROBLAZE_GOTOFF_64', 'R_MICROBLAZE_GOTPC_64',
    'R_MICROBLAZE_GOT_64', 'R_MICROBLAZE_JUMP_SLOT',
    'R_MICROBLAZE_NONE', 'R_MICROBLAZE_PLT_64', 'R_MICROBLAZE_REL',
    'R_MICROBLAZE_SRO32', 'R_MICROBLAZE_SRW32', 'R_MICROBLAZE_TLS',
    'R_MICROBLAZE_TLSDTPMOD32', 'R_MICROBLAZE_TLSDTPREL32',
    'R_MICROBLAZE_TLSDTPREL64', 'R_MICROBLAZE_TLSGD',
    'R_MICROBLAZE_TLSGOTTPREL32', 'R_MICROBLAZE_TLSLD',
    'R_MICROBLAZE_TLSTPREL32', 'R_MICROMIPS_26_S1',
    'R_MICROMIPS_CALL16', 'R_MICROMIPS_CALL_HI16',
    'R_MICROMIPS_CALL_LO16', 'R_MICROMIPS_GOT16',
    'R_MICROMIPS_GOT_DISP', 'R_MICROMIPS_GOT_HI16',
    'R_MICROMIPS_GOT_LO16', 'R_MICROMIPS_GOT_OFST',
    'R_MICROMIPS_GOT_PAGE', 'R_MICROMIPS_GPREL16',
    'R_MICROMIPS_GPREL7_S2', 'R_MICROMIPS_HI0_LO16',
    'R_MICROMIPS_HI16', 'R_MICROMIPS_HIGHER', 'R_MICROMIPS_HIGHEST',
    'R_MICROMIPS_JALR', 'R_MICROMIPS_LITERAL', 'R_MICROMIPS_LO16',
    'R_MICROMIPS_PC10_S1', 'R_MICROMIPS_PC16_S1',
    'R_MICROMIPS_PC23_S2', 'R_MICROMIPS_PC7_S1',
    'R_MICROMIPS_SCN_DISP', 'R_MICROMIPS_SUB',
    'R_MICROMIPS_TLS_DTPREL_HI16', 'R_MICROMIPS_TLS_DTPREL_LO16',
    'R_MICROMIPS_TLS_GD', 'R_MICROMIPS_TLS_GOTTPREL',
    'R_MICROMIPS_TLS_LDM', 'R_MICROMIPS_TLS_TPREL_HI16',
    'R_MICROMIPS_TLS_TPREL_LO16', 'R_MIPS16_26', 'R_MIPS16_CALL16',
    'R_MIPS16_GOT16', 'R_MIPS16_GPREL', 'R_MIPS16_HI16',
    'R_MIPS16_LO16', 'R_MIPS16_PC16_S1', 'R_MIPS16_TLS_DTPREL_HI16',
    'R_MIPS16_TLS_DTPREL_LO16', 'R_MIPS16_TLS_GD',
    'R_MIPS16_TLS_GOTTPREL', 'R_MIPS16_TLS_LDM',
    'R_MIPS16_TLS_TPREL_HI16', 'R_MIPS16_TLS_TPREL_LO16', 'R_MIPS_16',
    'R_MIPS_26', 'R_MIPS_32', 'R_MIPS_64', 'R_MIPS_ADD_IMMEDIATE',
    'R_MIPS_CALL16', 'R_MIPS_CALL_HI16', 'R_MIPS_CALL_LO16',
    'R_MIPS_COPY', 'R_MIPS_DELETE', 'R_MIPS_EH', 'R_MIPS_GLOB_DAT',
    'R_MIPS_GNU_REL16_S2', 'R_MIPS_GNU_VTENTRY',
    'R_MIPS_GNU_VTINHERIT', 'R_MIPS_GOT16', 'R_MIPS_GOT_DISP',
    'R_MIPS_GOT_HI16', 'R_MIPS_GOT_LO16', 'R_MIPS_GOT_OFST',
    'R_MIPS_GOT_PAGE', 'R_MIPS_GPREL16', 'R_MIPS_GPREL32',
    'R_MIPS_HI16', 'R_MIPS_HIGHER', 'R_MIPS_HIGHEST',
    'R_MIPS_INSERT_A', 'R_MIPS_INSERT_B', 'R_MIPS_JALR',
    'R_MIPS_JUMP_SLOT', 'R_MIPS_LITERAL', 'R_MIPS_LO16',
    'R_MIPS_NONE', 'R_MIPS_NUM', 'R_MIPS_PC16', 'R_MIPS_PC18_S3',
    'R_MIPS_PC19_S2', 'R_MIPS_PC21_S2', 'R_MIPS_PC26_S2',
    'R_MIPS_PC32', 'R_MIPS_PCHI16', 'R_MIPS_PCLO16', 'R_MIPS_PJUMP',
    'R_MIPS_REL16', 'R_MIPS_REL32', 'R_MIPS_RELATIVE',
    'R_MIPS_RELGOT', 'R_MIPS_SCN_DISP', 'R_MIPS_SHIFT5',
    'R_MIPS_SHIFT6', 'R_MIPS_SUB', 'R_MIPS_TLS_DTPMOD32',
    'R_MIPS_TLS_DTPMOD64', 'R_MIPS_TLS_DTPREL32',
    'R_MIPS_TLS_DTPREL64', 'R_MIPS_TLS_DTPREL_HI16',
    'R_MIPS_TLS_DTPREL_LO16', 'R_MIPS_TLS_GD', 'R_MIPS_TLS_GOTTPREL',
    'R_MIPS_TLS_LDM', 'R_MIPS_TLS_TPREL32', 'R_MIPS_TLS_TPREL64',
    'R_MIPS_TLS_TPREL_HI16', 'R_MIPS_TLS_TPREL_LO16', 'R_MN10300_16',
    'R_MN10300_24', 'R_MN10300_32', 'R_MN10300_8', 'R_MN10300_ALIGN',
    'R_MN10300_COPY', 'R_MN10300_GLOB_DAT', 'R_MN10300_GNU_VTENTRY',
    'R_MN10300_GNU_VTINHERIT', 'R_MN10300_GOT16', 'R_MN10300_GOT24',
    'R_MN10300_GOT32', 'R_MN10300_GOTOFF16', 'R_MN10300_GOTOFF24',
    'R_MN10300_GOTOFF32', 'R_MN10300_GOTPC16', 'R_MN10300_GOTPC32',
    'R_MN10300_JMP_SLOT', 'R_MN10300_NONE', 'R_MN10300_NUM',
    'R_MN10300_PCREL16', 'R_MN10300_PCREL32', 'R_MN10300_PCREL8',
    'R_MN10300_PLT16', 'R_MN10300_PLT32', 'R_MN10300_RELATIVE',
    'R_MN10300_SYM_DIFF', 'R_MN10300_TLS_DTPMOD',
    'R_MN10300_TLS_DTPOFF', 'R_MN10300_TLS_GD', 'R_MN10300_TLS_GOTIE',
    'R_MN10300_TLS_IE', 'R_MN10300_TLS_LD', 'R_MN10300_TLS_LDO',
    'R_MN10300_TLS_LE', 'R_MN10300_TLS_TPOFF', 'R_NDS32_32_RELA',
    'R_NDS32_COPY', 'R_NDS32_GLOB_DAT', 'R_NDS32_JMP_SLOT',
    'R_NDS32_NONE', 'R_NDS32_RELATIVE', 'R_NDS32_TLS_DESC',
    'R_NDS32_TLS_TPOFF', 'R_NIOS2_ALIGN', 'R_NIOS2_BFD_RELOC_16',
    'R_NIOS2_BFD_RELOC_32', 'R_NIOS2_BFD_RELOC_8',
    'R_NIOS2_CACHE_OPX', 'R_NIOS2_CALL16', 'R_NIOS2_CALL26',
    'R_NIOS2_CALL26_NOAT', 'R_NIOS2_CALLR', 'R_NIOS2_CALL_HA',
    'R_NIOS2_CALL_LO', 'R_NIOS2_CJMP', 'R_NIOS2_COPY',
    'R_NIOS2_GLOB_DAT', 'R_NIOS2_GNU_VTENTRY',
    'R_NIOS2_GNU_VTINHERIT', 'R_NIOS2_GOT16', 'R_NIOS2_GOTOFF',
    'R_NIOS2_GOTOFF_HA', 'R_NIOS2_GOTOFF_LO', 'R_NIOS2_GOT_HA',
    'R_NIOS2_GOT_LO', 'R_NIOS2_GPREL', 'R_NIOS2_HI16',
    'R_NIOS2_HIADJ16', 'R_NIOS2_IMM5', 'R_NIOS2_IMM6', 'R_NIOS2_IMM8',
    'R_NIOS2_JUMP_SLOT', 'R_NIOS2_LO16', 'R_NIOS2_NONE',
    'R_NIOS2_PCREL16', 'R_NIOS2_PCREL_HA', 'R_NIOS2_PCREL_LO',
    'R_NIOS2_RELATIVE', 'R_NIOS2_S16', 'R_NIOS2_TLS_DTPMOD',
    'R_NIOS2_TLS_DTPREL', 'R_NIOS2_TLS_GD16', 'R_NIOS2_TLS_IE16',
    'R_NIOS2_TLS_LDM16', 'R_NIOS2_TLS_LDO16', 'R_NIOS2_TLS_LE16',
    'R_NIOS2_TLS_TPREL', 'R_NIOS2_U16', 'R_NIOS2_UJMP', 'R_OK',
    'R_OR1K_16', 'R_OR1K_16_PCREL', 'R_OR1K_32', 'R_OR1K_32_PCREL',
    'R_OR1K_8', 'R_OR1K_8_PCREL', 'R_OR1K_COPY', 'R_OR1K_GLOB_DAT',
    'R_OR1K_GNU_VTENTRY', 'R_OR1K_GNU_VTINHERIT', 'R_OR1K_GOT16',
    'R_OR1K_GOTOFF_HI16', 'R_OR1K_GOTOFF_LO16', 'R_OR1K_GOTPC_HI16',
    'R_OR1K_GOTPC_LO16', 'R_OR1K_HI_16_IN_INSN', 'R_OR1K_INSN_REL_26',
    'R_OR1K_JMP_SLOT', 'R_OR1K_LO_16_IN_INSN', 'R_OR1K_NONE',
    'R_OR1K_PLT26', 'R_OR1K_RELATIVE', 'R_OR1K_TLS_DTPMOD',
    'R_OR1K_TLS_DTPOFF', 'R_OR1K_TLS_GD_HI16', 'R_OR1K_TLS_GD_LO16',
    'R_OR1K_TLS_IE_HI16', 'R_OR1K_TLS_IE_LO16', 'R_OR1K_TLS_LDM_HI16',
    'R_OR1K_TLS_LDM_LO16', 'R_OR1K_TLS_LDO_HI16',
    'R_OR1K_TLS_LDO_LO16', 'R_OR1K_TLS_LE_HI16', 'R_OR1K_TLS_LE_LO16',
    'R_OR1K_TLS_TPOFF', 'R_PARISC_COPY', 'R_PARISC_DIR14DR',
    'R_PARISC_DIR14R', 'R_PARISC_DIR14WR', 'R_PARISC_DIR16DF',
    'R_PARISC_DIR16F', 'R_PARISC_DIR16WF', 'R_PARISC_DIR17F',
    'R_PARISC_DIR17R', 'R_PARISC_DIR21L', 'R_PARISC_DIR32',
    'R_PARISC_DIR64', 'R_PARISC_DPREL14R', 'R_PARISC_DPREL21L',
    'R_PARISC_EPLT', 'R_PARISC_FPTR64', 'R_PARISC_GNU_VTENTRY',
    'R_PARISC_GNU_VTINHERIT', 'R_PARISC_GPREL14DR',
    'R_PARISC_GPREL14R', 'R_PARISC_GPREL14WR', 'R_PARISC_GPREL16DF',
    'R_PARISC_GPREL16F', 'R_PARISC_GPREL16WF', 'R_PARISC_GPREL21L',
    'R_PARISC_GPREL64', 'R_PARISC_HIRESERVE', 'R_PARISC_IPLT',
    'R_PARISC_LORESERVE', 'R_PARISC_LTOFF14DR', 'R_PARISC_LTOFF14R',
    'R_PARISC_LTOFF14WR', 'R_PARISC_LTOFF16DF', 'R_PARISC_LTOFF16F',
    'R_PARISC_LTOFF16WF', 'R_PARISC_LTOFF21L', 'R_PARISC_LTOFF64',
    'R_PARISC_LTOFF_FPTR14DR', 'R_PARISC_LTOFF_FPTR14R',
    'R_PARISC_LTOFF_FPTR14WR', 'R_PARISC_LTOFF_FPTR16DF',
    'R_PARISC_LTOFF_FPTR16F', 'R_PARISC_LTOFF_FPTR16WF',
    'R_PARISC_LTOFF_FPTR21L', 'R_PARISC_LTOFF_FPTR32',
    'R_PARISC_LTOFF_FPTR64', 'R_PARISC_LTOFF_TP14DR',
    'R_PARISC_LTOFF_TP14F', 'R_PARISC_LTOFF_TP14R',
    'R_PARISC_LTOFF_TP14WR', 'R_PARISC_LTOFF_TP16DF',
    'R_PARISC_LTOFF_TP16F', 'R_PARISC_LTOFF_TP16WF',
    'R_PARISC_LTOFF_TP21L', 'R_PARISC_LTOFF_TP64', 'R_PARISC_NONE',
    'R_PARISC_PCREL14DR', 'R_PARISC_PCREL14R', 'R_PARISC_PCREL14WR',
    'R_PARISC_PCREL16DF', 'R_PARISC_PCREL16F', 'R_PARISC_PCREL16WF',
    'R_PARISC_PCREL17F', 'R_PARISC_PCREL17R', 'R_PARISC_PCREL21L',
    'R_PARISC_PCREL22F', 'R_PARISC_PCREL32', 'R_PARISC_PCREL64',
    'R_PARISC_PLABEL14R', 'R_PARISC_PLABEL21L', 'R_PARISC_PLABEL32',
    'R_PARISC_PLTOFF14DR', 'R_PARISC_PLTOFF14R',
    'R_PARISC_PLTOFF14WR', 'R_PARISC_PLTOFF16DF',
    'R_PARISC_PLTOFF16F', 'R_PARISC_PLTOFF16WF', 'R_PARISC_PLTOFF21L',
    'R_PARISC_SECREL32', 'R_PARISC_SECREL64', 'R_PARISC_SEGBASE',
    'R_PARISC_SEGREL32', 'R_PARISC_SEGREL64', 'R_PARISC_TLS_DTPMOD32',
    'R_PARISC_TLS_DTPMOD64', 'R_PARISC_TLS_DTPOFF32',
    'R_PARISC_TLS_DTPOFF64', 'R_PARISC_TLS_GD14R',
    'R_PARISC_TLS_GD21L', 'R_PARISC_TLS_GDCALL', 'R_PARISC_TLS_IE14R',
    'R_PARISC_TLS_IE21L', 'R_PARISC_TLS_LDM14R',
    'R_PARISC_TLS_LDM21L', 'R_PARISC_TLS_LDMCALL',
    'R_PARISC_TLS_LDO14R', 'R_PARISC_TLS_LDO21L',
    'R_PARISC_TLS_LE14R', 'R_PARISC_TLS_LE21L',
    'R_PARISC_TLS_TPREL32', 'R_PARISC_TLS_TPREL64',
    'R_PARISC_TPREL14DR', 'R_PARISC_TPREL14R', 'R_PARISC_TPREL14WR',
    'R_PARISC_TPREL16DF', 'R_PARISC_TPREL16F', 'R_PARISC_TPREL16WF',
    'R_PARISC_TPREL21L', 'R_PARISC_TPREL32', 'R_PARISC_TPREL64',
    'R_PPC64_ADDR14', 'R_PPC64_ADDR14_BRNTAKEN',
    'R_PPC64_ADDR14_BRTAKEN', 'R_PPC64_ADDR16', 'R_PPC64_ADDR16_DS',
    'R_PPC64_ADDR16_HA', 'R_PPC64_ADDR16_HI', 'R_PPC64_ADDR16_HIGH',
    'R_PPC64_ADDR16_HIGHA', 'R_PPC64_ADDR16_HIGHER',
    'R_PPC64_ADDR16_HIGHERA', 'R_PPC64_ADDR16_HIGHEST',
    'R_PPC64_ADDR16_HIGHESTA', 'R_PPC64_ADDR16_LO',
    'R_PPC64_ADDR16_LO_DS', 'R_PPC64_ADDR24', 'R_PPC64_ADDR30',
    'R_PPC64_ADDR32', 'R_PPC64_ADDR64', 'R_PPC64_COPY',
    'R_PPC64_DTPMOD64', 'R_PPC64_DTPREL16', 'R_PPC64_DTPREL16_DS',
    'R_PPC64_DTPREL16_HA', 'R_PPC64_DTPREL16_HI',
    'R_PPC64_DTPREL16_HIGH', 'R_PPC64_DTPREL16_HIGHA',
    'R_PPC64_DTPREL16_HIGHER', 'R_PPC64_DTPREL16_HIGHERA',
    'R_PPC64_DTPREL16_HIGHEST', 'R_PPC64_DTPREL16_HIGHESTA',
    'R_PPC64_DTPREL16_LO', 'R_PPC64_DTPREL16_LO_DS',
    'R_PPC64_DTPREL64', 'R_PPC64_GLOB_DAT', 'R_PPC64_GOT16',
    'R_PPC64_GOT16_DS', 'R_PPC64_GOT16_HA', 'R_PPC64_GOT16_HI',
    'R_PPC64_GOT16_LO', 'R_PPC64_GOT16_LO_DS',
    'R_PPC64_GOT_DTPREL16_DS', 'R_PPC64_GOT_DTPREL16_HA',
    'R_PPC64_GOT_DTPREL16_HI', 'R_PPC64_GOT_DTPREL16_LO_DS',
    'R_PPC64_GOT_TLSGD16', 'R_PPC64_GOT_TLSGD16_HA',
    'R_PPC64_GOT_TLSGD16_HI', 'R_PPC64_GOT_TLSGD16_LO',
    'R_PPC64_GOT_TLSLD16', 'R_PPC64_GOT_TLSLD16_HA',
    'R_PPC64_GOT_TLSLD16_HI', 'R_PPC64_GOT_TLSLD16_LO',
    'R_PPC64_GOT_TPREL16_DS', 'R_PPC64_GOT_TPREL16_HA',
    'R_PPC64_GOT_TPREL16_HI', 'R_PPC64_GOT_TPREL16_LO_DS',
    'R_PPC64_IRELATIVE', 'R_PPC64_JMP_IREL', 'R_PPC64_JMP_SLOT',
    'R_PPC64_NONE', 'R_PPC64_PLT16_HA', 'R_PPC64_PLT16_HI',
    'R_PPC64_PLT16_LO', 'R_PPC64_PLT16_LO_DS', 'R_PPC64_PLT32',
    'R_PPC64_PLT64', 'R_PPC64_PLTGOT16', 'R_PPC64_PLTGOT16_DS',
    'R_PPC64_PLTGOT16_HA', 'R_PPC64_PLTGOT16_HI',
    'R_PPC64_PLTGOT16_LO', 'R_PPC64_PLTGOT16_LO_DS',
    'R_PPC64_PLTREL32', 'R_PPC64_PLTREL64', 'R_PPC64_REL14',
    'R_PPC64_REL14_BRNTAKEN', 'R_PPC64_REL14_BRTAKEN',
    'R_PPC64_REL16', 'R_PPC64_REL16_HA', 'R_PPC64_REL16_HI',
    'R_PPC64_REL16_LO', 'R_PPC64_REL24', 'R_PPC64_REL32',
    'R_PPC64_REL64', 'R_PPC64_RELATIVE', 'R_PPC64_SECTOFF',
    'R_PPC64_SECTOFF_DS', 'R_PPC64_SECTOFF_HA', 'R_PPC64_SECTOFF_HI',
    'R_PPC64_SECTOFF_LO', 'R_PPC64_SECTOFF_LO_DS', 'R_PPC64_TLS',
    'R_PPC64_TLSGD', 'R_PPC64_TLSLD', 'R_PPC64_TOC', 'R_PPC64_TOC16',
    'R_PPC64_TOC16_DS', 'R_PPC64_TOC16_HA', 'R_PPC64_TOC16_HI',
    'R_PPC64_TOC16_LO', 'R_PPC64_TOC16_LO_DS', 'R_PPC64_TOCSAVE',
    'R_PPC64_TPREL16', 'R_PPC64_TPREL16_DS', 'R_PPC64_TPREL16_HA',
    'R_PPC64_TPREL16_HI', 'R_PPC64_TPREL16_HIGH',
    'R_PPC64_TPREL16_HIGHA', 'R_PPC64_TPREL16_HIGHER',
    'R_PPC64_TPREL16_HIGHERA', 'R_PPC64_TPREL16_HIGHEST',
    'R_PPC64_TPREL16_HIGHESTA', 'R_PPC64_TPREL16_LO',
    'R_PPC64_TPREL16_LO_DS', 'R_PPC64_TPREL64', 'R_PPC64_UADDR16',
    'R_PPC64_UADDR32', 'R_PPC64_UADDR64', 'R_PPC_ADDR14',
    'R_PPC_ADDR14_BRNTAKEN', 'R_PPC_ADDR14_BRTAKEN', 'R_PPC_ADDR16',
    'R_PPC_ADDR16_HA', 'R_PPC_ADDR16_HI', 'R_PPC_ADDR16_LO',
    'R_PPC_ADDR24', 'R_PPC_ADDR32', 'R_PPC_COPY',
    'R_PPC_DIAB_RELSDA_HA', 'R_PPC_DIAB_RELSDA_HI',
    'R_PPC_DIAB_RELSDA_LO', 'R_PPC_DIAB_SDA21_HA',
    'R_PPC_DIAB_SDA21_HI', 'R_PPC_DIAB_SDA21_LO', 'R_PPC_DTPMOD32',
    'R_PPC_DTPREL16', 'R_PPC_DTPREL16_HA', 'R_PPC_DTPREL16_HI',
    'R_PPC_DTPREL16_LO', 'R_PPC_DTPREL32', 'R_PPC_EMB_BIT_FLD',
    'R_PPC_EMB_MRKREF', 'R_PPC_EMB_NADDR16', 'R_PPC_EMB_NADDR16_HA',
    'R_PPC_EMB_NADDR16_HI', 'R_PPC_EMB_NADDR16_LO',
    'R_PPC_EMB_NADDR32', 'R_PPC_EMB_RELSDA', 'R_PPC_EMB_RELSEC16',
    'R_PPC_EMB_RELST_HA', 'R_PPC_EMB_RELST_HI', 'R_PPC_EMB_RELST_LO',
    'R_PPC_EMB_SDA21', 'R_PPC_EMB_SDA2I16', 'R_PPC_EMB_SDA2REL',
    'R_PPC_EMB_SDAI16', 'R_PPC_GLOB_DAT', 'R_PPC_GOT16',
    'R_PPC_GOT16_HA', 'R_PPC_GOT16_HI', 'R_PPC_GOT16_LO',
    'R_PPC_GOT_DTPREL16', 'R_PPC_GOT_DTPREL16_HA',
    'R_PPC_GOT_DTPREL16_HI', 'R_PPC_GOT_DTPREL16_LO',
    'R_PPC_GOT_TLSGD16', 'R_PPC_GOT_TLSGD16_HA',
    'R_PPC_GOT_TLSGD16_HI', 'R_PPC_GOT_TLSGD16_LO',
    'R_PPC_GOT_TLSLD16', 'R_PPC_GOT_TLSLD16_HA',
    'R_PPC_GOT_TLSLD16_HI', 'R_PPC_GOT_TLSLD16_LO',
    'R_PPC_GOT_TPREL16', 'R_PPC_GOT_TPREL16_HA',
    'R_PPC_GOT_TPREL16_HI', 'R_PPC_GOT_TPREL16_LO', 'R_PPC_IRELATIVE',
    'R_PPC_JMP_SLOT', 'R_PPC_LOCAL24PC', 'R_PPC_NONE',
    'R_PPC_PLT16_HA', 'R_PPC_PLT16_HI', 'R_PPC_PLT16_LO',
    'R_PPC_PLT32', 'R_PPC_PLTREL24', 'R_PPC_PLTREL32', 'R_PPC_REL14',
    'R_PPC_REL14_BRNTAKEN', 'R_PPC_REL14_BRTAKEN', 'R_PPC_REL16',
    'R_PPC_REL16_HA', 'R_PPC_REL16_HI', 'R_PPC_REL16_LO',
    'R_PPC_REL24', 'R_PPC_REL32', 'R_PPC_RELATIVE', 'R_PPC_SDAREL16',
    'R_PPC_SECTOFF', 'R_PPC_SECTOFF_HA', 'R_PPC_SECTOFF_HI',
    'R_PPC_SECTOFF_LO', 'R_PPC_TLS', 'R_PPC_TLSGD', 'R_PPC_TLSLD',
    'R_PPC_TOC16', 'R_PPC_TPREL16', 'R_PPC_TPREL16_HA',
    'R_PPC_TPREL16_HI', 'R_PPC_TPREL16_LO', 'R_PPC_TPREL32',
    'R_PPC_UADDR16', 'R_PPC_UADDR32', 'R_RISCV_32',
    'R_RISCV_32_PCREL', 'R_RISCV_64', 'R_RISCV_ADD16',
    'R_RISCV_ADD32', 'R_RISCV_ADD64', 'R_RISCV_ADD8', 'R_RISCV_ALIGN',
    'R_RISCV_BRANCH', 'R_RISCV_CALL', 'R_RISCV_CALL_PLT',
    'R_RISCV_COPY', 'R_RISCV_GNU_VTENTRY', 'R_RISCV_GNU_VTINHERIT',
    'R_RISCV_GOT_HI20', 'R_RISCV_GPREL_I', 'R_RISCV_GPREL_S',
    'R_RISCV_HI20', 'R_RISCV_IRELATIVE', 'R_RISCV_JAL',
    'R_RISCV_JUMP_SLOT', 'R_RISCV_LO12_I', 'R_RISCV_LO12_S',
    'R_RISCV_NONE', 'R_RISCV_NUM', 'R_RISCV_PCREL_HI20',
    'R_RISCV_PCREL_LO12_I', 'R_RISCV_PCREL_LO12_S', 'R_RISCV_PLT32',
    'R_RISCV_RELATIVE', 'R_RISCV_RELAX', 'R_RISCV_RVC_BRANCH',
    'R_RISCV_RVC_JUMP', 'R_RISCV_RVC_LUI', 'R_RISCV_SET16',
    'R_RISCV_SET32', 'R_RISCV_SET6', 'R_RISCV_SET8',
    'R_RISCV_SET_ULEB128', 'R_RISCV_SUB16', 'R_RISCV_SUB32',
    'R_RISCV_SUB6', 'R_RISCV_SUB64', 'R_RISCV_SUB8',
    'R_RISCV_SUB_ULEB128', 'R_RISCV_TLS_DTPMOD32',
    'R_RISCV_TLS_DTPMOD64', 'R_RISCV_TLS_DTPREL32',
    'R_RISCV_TLS_DTPREL64', 'R_RISCV_TLS_GD_HI20',
    'R_RISCV_TLS_GOT_HI20', 'R_RISCV_TLS_TPREL32',
    'R_RISCV_TLS_TPREL64', 'R_RISCV_TPREL_ADD', 'R_RISCV_TPREL_HI20',
    'R_RISCV_TPREL_I', 'R_RISCV_TPREL_LO12_I', 'R_RISCV_TPREL_LO12_S',
    'R_RISCV_TPREL_S', 'R_SH_ALIGN', 'R_SH_CODE', 'R_SH_COPY',
    'R_SH_COUNT', 'R_SH_DATA', 'R_SH_DIR32', 'R_SH_DIR8BP',
    'R_SH_DIR8L', 'R_SH_DIR8W', 'R_SH_DIR8WPL', 'R_SH_DIR8WPN',
    'R_SH_DIR8WPZ', 'R_SH_GLOB_DAT', 'R_SH_GNU_VTENTRY',
    'R_SH_GNU_VTINHERIT', 'R_SH_GOT32', 'R_SH_GOTOFF', 'R_SH_GOTPC',
    'R_SH_IND12W', 'R_SH_JMP_SLOT', 'R_SH_LABEL', 'R_SH_NONE',
    'R_SH_NUM', 'R_SH_PLT32', 'R_SH_REL32', 'R_SH_RELATIVE',
    'R_SH_SWITCH16', 'R_SH_SWITCH32', 'R_SH_SWITCH8',
    'R_SH_TLS_DTPMOD32', 'R_SH_TLS_DTPOFF32', 'R_SH_TLS_GD_32',
    'R_SH_TLS_IE_32', 'R_SH_TLS_LDO_32', 'R_SH_TLS_LD_32',
    'R_SH_TLS_LE_32', 'R_SH_TLS_TPOFF32', 'R_SH_USES', 'R_SPARC_10',
    'R_SPARC_11', 'R_SPARC_13', 'R_SPARC_16', 'R_SPARC_22',
    'R_SPARC_32', 'R_SPARC_5', 'R_SPARC_6', 'R_SPARC_64', 'R_SPARC_7',
    'R_SPARC_8', 'R_SPARC_COPY', 'R_SPARC_DISP16', 'R_SPARC_DISP32',
    'R_SPARC_DISP64', 'R_SPARC_DISP8', 'R_SPARC_GLOB_DAT',
    'R_SPARC_GLOB_JMP', 'R_SPARC_GNU_VTENTRY',
    'R_SPARC_GNU_VTINHERIT', 'R_SPARC_GOT10', 'R_SPARC_GOT13',
    'R_SPARC_GOT22', 'R_SPARC_GOTDATA_HIX22', 'R_SPARC_GOTDATA_LOX10',
    'R_SPARC_GOTDATA_OP', 'R_SPARC_GOTDATA_OP_HIX22',
    'R_SPARC_GOTDATA_OP_LOX10', 'R_SPARC_H34', 'R_SPARC_H44',
    'R_SPARC_HH22', 'R_SPARC_HI22', 'R_SPARC_HIPLT22',
    'R_SPARC_HIX22', 'R_SPARC_HM10', 'R_SPARC_IRELATIVE',
    'R_SPARC_JMP_IREL', 'R_SPARC_JMP_SLOT', 'R_SPARC_L44',
    'R_SPARC_LM22', 'R_SPARC_LO10', 'R_SPARC_LOPLT10',
    'R_SPARC_LOX10', 'R_SPARC_M44', 'R_SPARC_NONE', 'R_SPARC_NUM',
    'R_SPARC_OLO10', 'R_SPARC_PC10', 'R_SPARC_PC22',
    'R_SPARC_PCPLT10', 'R_SPARC_PCPLT22', 'R_SPARC_PCPLT32',
    'R_SPARC_PC_HH22', 'R_SPARC_PC_HM10', 'R_SPARC_PC_LM22',
    'R_SPARC_PLT32', 'R_SPARC_PLT64', 'R_SPARC_REGISTER',
    'R_SPARC_RELATIVE', 'R_SPARC_REV32', 'R_SPARC_SIZE32',
    'R_SPARC_SIZE64', 'R_SPARC_TLS_DTPMOD32', 'R_SPARC_TLS_DTPMOD64',
    'R_SPARC_TLS_DTPOFF32', 'R_SPARC_TLS_DTPOFF64',
    'R_SPARC_TLS_GD_ADD', 'R_SPARC_TLS_GD_CALL',
    'R_SPARC_TLS_GD_HI22', 'R_SPARC_TLS_GD_LO10',
    'R_SPARC_TLS_IE_ADD', 'R_SPARC_TLS_IE_HI22', 'R_SPARC_TLS_IE_LD',
    'R_SPARC_TLS_IE_LDX', 'R_SPARC_TLS_IE_LO10',
    'R_SPARC_TLS_LDM_ADD', 'R_SPARC_TLS_LDM_CALL',
    'R_SPARC_TLS_LDM_HI22', 'R_SPARC_TLS_LDM_LO10',
    'R_SPARC_TLS_LDO_ADD', 'R_SPARC_TLS_LDO_HIX22',
    'R_SPARC_TLS_LDO_LOX10', 'R_SPARC_TLS_LE_HIX22',
    'R_SPARC_TLS_LE_LOX10', 'R_SPARC_TLS_TPOFF32',
    'R_SPARC_TLS_TPOFF64', 'R_SPARC_UA16', 'R_SPARC_UA32',
    'R_SPARC_UA64', 'R_SPARC_WDISP10', 'R_SPARC_WDISP16',
    'R_SPARC_WDISP19', 'R_SPARC_WDISP22', 'R_SPARC_WDISP30',
    'R_SPARC_WPLT30', 'R_TILEGX_16', 'R_TILEGX_16_PCREL',
    'R_TILEGX_32', 'R_TILEGX_32_PCREL', 'R_TILEGX_64',
    'R_TILEGX_64_PCREL', 'R_TILEGX_8', 'R_TILEGX_8_PCREL',
    'R_TILEGX_BROFF_X1', 'R_TILEGX_COPY', 'R_TILEGX_DEST_IMM8_X1',
    'R_TILEGX_GLOB_DAT', 'R_TILEGX_GNU_VTENTRY',
    'R_TILEGX_GNU_VTINHERIT', 'R_TILEGX_HW0', 'R_TILEGX_HW0_LAST',
    'R_TILEGX_HW1', 'R_TILEGX_HW1_LAST', 'R_TILEGX_HW2',
    'R_TILEGX_HW2_LAST', 'R_TILEGX_HW3', 'R_TILEGX_IMM16_X0_HW0',
    'R_TILEGX_IMM16_X0_HW0_GOT', 'R_TILEGX_IMM16_X0_HW0_LAST',
    'R_TILEGX_IMM16_X0_HW0_LAST_GOT',
    'R_TILEGX_IMM16_X0_HW0_LAST_PCREL',
    'R_TILEGX_IMM16_X0_HW0_LAST_PLT_PCREL',
    'R_TILEGX_IMM16_X0_HW0_LAST_TLS_GD',
    'R_TILEGX_IMM16_X0_HW0_LAST_TLS_IE',
    'R_TILEGX_IMM16_X0_HW0_LAST_TLS_LE',
    'R_TILEGX_IMM16_X0_HW0_PCREL', 'R_TILEGX_IMM16_X0_HW0_PLT_PCREL',
    'R_TILEGX_IMM16_X0_HW0_TLS_GD', 'R_TILEGX_IMM16_X0_HW0_TLS_IE',
    'R_TILEGX_IMM16_X0_HW0_TLS_LE', 'R_TILEGX_IMM16_X0_HW1',
    'R_TILEGX_IMM16_X0_HW1_LAST', 'R_TILEGX_IMM16_X0_HW1_LAST_GOT',
    'R_TILEGX_IMM16_X0_HW1_LAST_PCREL',
    'R_TILEGX_IMM16_X0_HW1_LAST_PLT_PCREL',
    'R_TILEGX_IMM16_X0_HW1_LAST_TLS_GD',
    'R_TILEGX_IMM16_X0_HW1_LAST_TLS_IE',
    'R_TILEGX_IMM16_X0_HW1_LAST_TLS_LE',
    'R_TILEGX_IMM16_X0_HW1_PCREL', 'R_TILEGX_IMM16_X0_HW1_PLT_PCREL',
    'R_TILEGX_IMM16_X0_HW2', 'R_TILEGX_IMM16_X0_HW2_LAST',
    'R_TILEGX_IMM16_X0_HW2_LAST_PCREL',
    'R_TILEGX_IMM16_X0_HW2_LAST_PLT_PCREL',
    'R_TILEGX_IMM16_X0_HW2_PCREL', 'R_TILEGX_IMM16_X0_HW2_PLT_PCREL',
    'R_TILEGX_IMM16_X0_HW3', 'R_TILEGX_IMM16_X0_HW3_PCREL',
    'R_TILEGX_IMM16_X0_HW3_PLT_PCREL', 'R_TILEGX_IMM16_X1_HW0',
    'R_TILEGX_IMM16_X1_HW0_GOT', 'R_TILEGX_IMM16_X1_HW0_LAST',
    'R_TILEGX_IMM16_X1_HW0_LAST_GOT',
    'R_TILEGX_IMM16_X1_HW0_LAST_PCREL',
    'R_TILEGX_IMM16_X1_HW0_LAST_PLT_PCREL',
    'R_TILEGX_IMM16_X1_HW0_LAST_TLS_GD',
    'R_TILEGX_IMM16_X1_HW0_LAST_TLS_IE',
    'R_TILEGX_IMM16_X1_HW0_LAST_TLS_LE',
    'R_TILEGX_IMM16_X1_HW0_PCREL', 'R_TILEGX_IMM16_X1_HW0_PLT_PCREL',
    'R_TILEGX_IMM16_X1_HW0_TLS_GD', 'R_TILEGX_IMM16_X1_HW0_TLS_IE',
    'R_TILEGX_IMM16_X1_HW0_TLS_LE', 'R_TILEGX_IMM16_X1_HW1',
    'R_TILEGX_IMM16_X1_HW1_LAST', 'R_TILEGX_IMM16_X1_HW1_LAST_GOT',
    'R_TILEGX_IMM16_X1_HW1_LAST_PCREL',
    'R_TILEGX_IMM16_X1_HW1_LAST_PLT_PCREL',
    'R_TILEGX_IMM16_X1_HW1_LAST_TLS_GD',
    'R_TILEGX_IMM16_X1_HW1_LAST_TLS_IE',
    'R_TILEGX_IMM16_X1_HW1_LAST_TLS_LE',
    'R_TILEGX_IMM16_X1_HW1_PCREL', 'R_TILEGX_IMM16_X1_HW1_PLT_PCREL',
    'R_TILEGX_IMM16_X1_HW2', 'R_TILEGX_IMM16_X1_HW2_LAST',
    'R_TILEGX_IMM16_X1_HW2_LAST_PCREL',
    'R_TILEGX_IMM16_X1_HW2_LAST_PLT_PCREL',
    'R_TILEGX_IMM16_X1_HW2_PCREL', 'R_TILEGX_IMM16_X1_HW2_PLT_PCREL',
    'R_TILEGX_IMM16_X1_HW3', 'R_TILEGX_IMM16_X1_HW3_PCREL',
    'R_TILEGX_IMM16_X1_HW3_PLT_PCREL', 'R_TILEGX_IMM8_X0',
    'R_TILEGX_IMM8_X0_TLS_ADD', 'R_TILEGX_IMM8_X0_TLS_GD_ADD',
    'R_TILEGX_IMM8_X1', 'R_TILEGX_IMM8_X1_TLS_ADD',
    'R_TILEGX_IMM8_X1_TLS_GD_ADD', 'R_TILEGX_IMM8_Y0',
    'R_TILEGX_IMM8_Y0_TLS_ADD', 'R_TILEGX_IMM8_Y0_TLS_GD_ADD',
    'R_TILEGX_IMM8_Y1', 'R_TILEGX_IMM8_Y1_TLS_ADD',
    'R_TILEGX_IMM8_Y1_TLS_GD_ADD', 'R_TILEGX_JMP_SLOT',
    'R_TILEGX_JUMPOFF_X1', 'R_TILEGX_JUMPOFF_X1_PLT',
    'R_TILEGX_MF_IMM14_X1', 'R_TILEGX_MMEND_X0',
    'R_TILEGX_MMSTART_X0', 'R_TILEGX_MT_IMM14_X1', 'R_TILEGX_NONE',
    'R_TILEGX_NUM', 'R_TILEGX_RELATIVE', 'R_TILEGX_SHAMT_X0',
    'R_TILEGX_SHAMT_X1', 'R_TILEGX_SHAMT_Y0', 'R_TILEGX_SHAMT_Y1',
    'R_TILEGX_TLS_DTPMOD32', 'R_TILEGX_TLS_DTPMOD64',
    'R_TILEGX_TLS_DTPOFF32', 'R_TILEGX_TLS_DTPOFF64',
    'R_TILEGX_TLS_GD_CALL', 'R_TILEGX_TLS_IE_LOAD',
    'R_TILEGX_TLS_TPOFF32', 'R_TILEGX_TLS_TPOFF64', 'R_TILEPRO_16',
    'R_TILEPRO_16_PCREL', 'R_TILEPRO_32', 'R_TILEPRO_32_PCREL',
    'R_TILEPRO_8', 'R_TILEPRO_8_PCREL', 'R_TILEPRO_BROFF_X1',
    'R_TILEPRO_COPY', 'R_TILEPRO_DEST_IMM8_X1', 'R_TILEPRO_GLOB_DAT',
    'R_TILEPRO_GNU_VTENTRY', 'R_TILEPRO_GNU_VTINHERIT',
    'R_TILEPRO_HA16', 'R_TILEPRO_HI16', 'R_TILEPRO_IMM16_X0',
    'R_TILEPRO_IMM16_X0_GOT', 'R_TILEPRO_IMM16_X0_GOT_HA',
    'R_TILEPRO_IMM16_X0_GOT_HI', 'R_TILEPRO_IMM16_X0_GOT_LO',
    'R_TILEPRO_IMM16_X0_HA', 'R_TILEPRO_IMM16_X0_HA_PCREL',
    'R_TILEPRO_IMM16_X0_HI', 'R_TILEPRO_IMM16_X0_HI_PCREL',
    'R_TILEPRO_IMM16_X0_LO', 'R_TILEPRO_IMM16_X0_LO_PCREL',
    'R_TILEPRO_IMM16_X0_PCREL', 'R_TILEPRO_IMM16_X0_TLS_GD',
    'R_TILEPRO_IMM16_X0_TLS_GD_HA', 'R_TILEPRO_IMM16_X0_TLS_GD_HI',
    'R_TILEPRO_IMM16_X0_TLS_GD_LO', 'R_TILEPRO_IMM16_X0_TLS_IE',
    'R_TILEPRO_IMM16_X0_TLS_IE_HA', 'R_TILEPRO_IMM16_X0_TLS_IE_HI',
    'R_TILEPRO_IMM16_X0_TLS_IE_LO', 'R_TILEPRO_IMM16_X0_TLS_LE',
    'R_TILEPRO_IMM16_X0_TLS_LE_HA', 'R_TILEPRO_IMM16_X0_TLS_LE_HI',
    'R_TILEPRO_IMM16_X0_TLS_LE_LO', 'R_TILEPRO_IMM16_X1',
    'R_TILEPRO_IMM16_X1_GOT', 'R_TILEPRO_IMM16_X1_GOT_HA',
    'R_TILEPRO_IMM16_X1_GOT_HI', 'R_TILEPRO_IMM16_X1_GOT_LO',
    'R_TILEPRO_IMM16_X1_HA', 'R_TILEPRO_IMM16_X1_HA_PCREL',
    'R_TILEPRO_IMM16_X1_HI', 'R_TILEPRO_IMM16_X1_HI_PCREL',
    'R_TILEPRO_IMM16_X1_LO', 'R_TILEPRO_IMM16_X1_LO_PCREL',
    'R_TILEPRO_IMM16_X1_PCREL', 'R_TILEPRO_IMM16_X1_TLS_GD',
    'R_TILEPRO_IMM16_X1_TLS_GD_HA', 'R_TILEPRO_IMM16_X1_TLS_GD_HI',
    'R_TILEPRO_IMM16_X1_TLS_GD_LO', 'R_TILEPRO_IMM16_X1_TLS_IE',
    'R_TILEPRO_IMM16_X1_TLS_IE_HA', 'R_TILEPRO_IMM16_X1_TLS_IE_HI',
    'R_TILEPRO_IMM16_X1_TLS_IE_LO', 'R_TILEPRO_IMM16_X1_TLS_LE',
    'R_TILEPRO_IMM16_X1_TLS_LE_HA', 'R_TILEPRO_IMM16_X1_TLS_LE_HI',
    'R_TILEPRO_IMM16_X1_TLS_LE_LO', 'R_TILEPRO_IMM8_X0',
    'R_TILEPRO_IMM8_X0_TLS_GD_ADD', 'R_TILEPRO_IMM8_X1',
    'R_TILEPRO_IMM8_X1_TLS_GD_ADD', 'R_TILEPRO_IMM8_Y0',
    'R_TILEPRO_IMM8_Y0_TLS_GD_ADD', 'R_TILEPRO_IMM8_Y1',
    'R_TILEPRO_IMM8_Y1_TLS_GD_ADD', 'R_TILEPRO_JMP_SLOT',
    'R_TILEPRO_JOFFLONG_X1', 'R_TILEPRO_JOFFLONG_X1_PLT',
    'R_TILEPRO_LO16', 'R_TILEPRO_MF_IMM15_X1', 'R_TILEPRO_MMEND_X0',
    'R_TILEPRO_MMEND_X1', 'R_TILEPRO_MMSTART_X0',
    'R_TILEPRO_MMSTART_X1', 'R_TILEPRO_MT_IMM15_X1', 'R_TILEPRO_NONE',
    'R_TILEPRO_NUM', 'R_TILEPRO_RELATIVE', 'R_TILEPRO_SHAMT_X0',
    'R_TILEPRO_SHAMT_X1', 'R_TILEPRO_SHAMT_Y0', 'R_TILEPRO_SHAMT_Y1',
    'R_TILEPRO_TLS_DTPMOD32', 'R_TILEPRO_TLS_DTPOFF32',
    'R_TILEPRO_TLS_GD_CALL', 'R_TILEPRO_TLS_IE_LOAD',
    'R_TILEPRO_TLS_TPOFF32', 'R_X86_64_16', 'R_X86_64_32',
    'R_X86_64_32S', 'R_X86_64_64', 'R_X86_64_8', 'R_X86_64_COPY',
    'R_X86_64_DTPMOD64', 'R_X86_64_DTPOFF32', 'R_X86_64_DTPOFF64',
    'R_X86_64_GLOB_DAT', 'R_X86_64_GOT32', 'R_X86_64_GOT64',
    'R_X86_64_GOTOFF64', 'R_X86_64_GOTPC32',
    'R_X86_64_GOTPC32_TLSDESC', 'R_X86_64_GOTPC64',
    'R_X86_64_GOTPCREL', 'R_X86_64_GOTPCREL64', 'R_X86_64_GOTPCRELX',
    'R_X86_64_GOTPLT64', 'R_X86_64_GOTTPOFF', 'R_X86_64_IRELATIVE',
    'R_X86_64_JUMP_SLOT', 'R_X86_64_NONE', 'R_X86_64_NUM',
    'R_X86_64_PC16', 'R_X86_64_PC32', 'R_X86_64_PC64', 'R_X86_64_PC8',
    'R_X86_64_PLT32', 'R_X86_64_PLTOFF64', 'R_X86_64_RELATIVE',
    'R_X86_64_RELATIVE64', 'R_X86_64_REX_GOTPCRELX',
    'R_X86_64_SIZE32', 'R_X86_64_SIZE64', 'R_X86_64_TLSDESC',
    'R_X86_64_TLSDESC_CALL', 'R_X86_64_TLSGD', 'R_X86_64_TLSLD',
    'R_X86_64_TPOFF32', 'R_X86_64_TPOFF64', 'SEEK_CUR', 'SEEK_END',
    'SEEK_SET', 'SELFMAG', 'SHF_ALLOC', 'SHF_ALPHA_GPREL',
    'SHF_ARM_COMDEF', 'SHF_ARM_ENTRYSECT', 'SHF_COMPRESSED',
    'SHF_EXCLUDE', 'SHF_EXECINSTR', 'SHF_GNU_RETAIN', 'SHF_GROUP',
    'SHF_IA_64_NORECOV', 'SHF_IA_64_SHORT', 'SHF_INFO_LINK',
    'SHF_LINK_ORDER', 'SHF_MASKOS', 'SHF_MASKPROC', 'SHF_MERGE',
    'SHF_MIPS_ADDR', 'SHF_MIPS_GPREL', 'SHF_MIPS_LOCAL',
    'SHF_MIPS_MERGE', 'SHF_MIPS_NAMES', 'SHF_MIPS_NODUPE',
    'SHF_MIPS_NOSTRIP', 'SHF_MIPS_STRINGS', 'SHF_ORDERED',
    'SHF_OS_NONCONFORMING', 'SHF_PARISC_HUGE', 'SHF_PARISC_SBP',
    'SHF_PARISC_SHORT', 'SHF_STRINGS', 'SHF_TLS', 'SHF_WRITE',
    'SHN_ABS', 'SHN_AFTER', 'SHN_BEFORE', 'SHN_COMMON', 'SHN_HIOS',
    'SHN_HIPROC', 'SHN_HIRESERVE', 'SHN_LOOS', 'SHN_LOPROC',
    'SHN_LORESERVE', 'SHN_MIPS_ACOMMON', 'SHN_MIPS_DATA',
    'SHN_MIPS_SCOMMON', 'SHN_MIPS_SUNDEFINED', 'SHN_MIPS_TEXT',
    'SHN_PARISC_ANSI_COMMON', 'SHN_PARISC_HUGE_COMMON', 'SHN_UNDEF',
    'SHN_XINDEX', 'SHT_ALPHA_DEBUG', 'SHT_ALPHA_REGINFO',
    'SHT_ARC_ATTRIBUTES', 'SHT_ARM_ATTRIBUTES', 'SHT_ARM_EXIDX',
    'SHT_ARM_PREEMPTMAP', 'SHT_CHECKSUM', 'SHT_CSKY_ATTRIBUTES',
    'SHT_DYNAMIC', 'SHT_DYNSYM', 'SHT_FINI_ARRAY',
    'SHT_GNU_ATTRIBUTES', 'SHT_GNU_HASH', 'SHT_GNU_LIBLIST',
    'SHT_GNU_verdef', 'SHT_GNU_verneed', 'SHT_GNU_versym',
    'SHT_GROUP', 'SHT_HASH', 'SHT_HIOS', 'SHT_HIPROC', 'SHT_HISUNW',
    'SHT_HIUSER', 'SHT_IA_64_EXT', 'SHT_IA_64_UNWIND',
    'SHT_INIT_ARRAY', 'SHT_LOOS', 'SHT_LOPROC', 'SHT_LOSUNW',
    'SHT_LOUSER', 'SHT_MIPS_ABIFLAGS', 'SHT_MIPS_AUXSYM',
    'SHT_MIPS_CONFLICT', 'SHT_MIPS_CONTENT', 'SHT_MIPS_DEBUG',
    'SHT_MIPS_DELTACLASS', 'SHT_MIPS_DELTADECL', 'SHT_MIPS_DELTAINST',
    'SHT_MIPS_DELTASYM', 'SHT_MIPS_DENSE', 'SHT_MIPS_DWARF',
    'SHT_MIPS_EH_REGION', 'SHT_MIPS_EVENTS', 'SHT_MIPS_EXTSYM',
    'SHT_MIPS_FDESC', 'SHT_MIPS_GPTAB', 'SHT_MIPS_IFACE',
    'SHT_MIPS_LIBLIST', 'SHT_MIPS_LINE', 'SHT_MIPS_LOCSTR',
    'SHT_MIPS_LOCSYM', 'SHT_MIPS_MSYM', 'SHT_MIPS_OPTIONS',
    'SHT_MIPS_OPTSYM', 'SHT_MIPS_PACKAGE', 'SHT_MIPS_PACKSYM',
    'SHT_MIPS_PDESC', 'SHT_MIPS_PDR_EXCEPTION', 'SHT_MIPS_PIXIE',
    'SHT_MIPS_REGINFO', 'SHT_MIPS_RELD', 'SHT_MIPS_RFDESC',
    'SHT_MIPS_SHDR', 'SHT_MIPS_SYMBOL_LIB', 'SHT_MIPS_TRANSLATE',
    'SHT_MIPS_UCODE', 'SHT_MIPS_WHIRL', 'SHT_MIPS_XHASH',
    'SHT_MIPS_XLATE', 'SHT_MIPS_XLATE_DEBUG', 'SHT_MIPS_XLATE_OLD',
    'SHT_NOBITS', 'SHT_NOTE', 'SHT_NULL', 'SHT_NUM', 'SHT_PARISC_DOC',
    'SHT_PARISC_EXT', 'SHT_PARISC_UNWIND', 'SHT_PREINIT_ARRAY',
    'SHT_PROGBITS', 'SHT_REL', 'SHT_RELA', 'SHT_RELR',
    'SHT_RISCV_ATTRIBUTES', 'SHT_SHLIB', 'SHT_STRTAB',
    'SHT_SUNW_COMDAT', 'SHT_SUNW_move', 'SHT_SUNW_syminfo',
    'SHT_SYMTAB', 'SHT_SYMTAB_SHNDX', 'SHT_X86_64_UNWIND',
    'STB_GLOBAL', 'STB_GNU_UNIQUE', 'STB_HIOS', 'STB_HIPROC',
    'STB_LOCAL', 'STB_LOOS', 'STB_LOPROC', 'STB_MIPS_SPLIT_COMMON',
    'STB_NUM', 'STB_WEAK', 'STDERR_FILENO', 'STDIN_FILENO',
    'STDOUT_FILENO', 'STN_UNDEF', 'STO_AARCH64_VARIANT_PCS',
    'STO_ALPHA_NOPV', 'STO_ALPHA_STD_GPLOAD', 'STO_MIPS_DEFAULT',
    'STO_MIPS_HIDDEN', 'STO_MIPS_INTERNAL', 'STO_MIPS_PLT',
    'STO_MIPS_PROTECTED', 'STO_MIPS_SC_ALIGN_UNUSED',
    'STO_PPC64_LOCAL_BIT', 'STO_PPC64_LOCAL_MASK',
    'STO_RISCV_VARIANT_CC', 'STT_ARM_16BIT', 'STT_ARM_TFUNC',
    'STT_COMMON', 'STT_FILE', 'STT_FUNC', 'STT_GNU_IFUNC', 'STT_HIOS',
    'STT_HIPROC', 'STT_HP_OPAQUE', 'STT_HP_STUB', 'STT_LOOS',
    'STT_LOPROC', 'STT_NOTYPE', 'STT_NUM', 'STT_OBJECT',
    'STT_PARISC_MILLICODE', 'STT_SECTION', 'STT_SPARC_REGISTER',
    'STT_TLS', 'STV_DEFAULT', 'STV_HIDDEN', 'STV_INTERNAL',
    'STV_PROTECTED', 'SYMINFO_BT_LOWRESERVE', 'SYMINFO_BT_PARENT',
    'SYMINFO_BT_SELF', 'SYMINFO_CURRENT', 'SYMINFO_FLG_COPY',
    'SYMINFO_FLG_DIRECT', 'SYMINFO_FLG_LAZYLOAD',
    'SYMINFO_FLG_PASSTHRU', 'SYMINFO_NONE', 'SYMINFO_NUM',
    'VER_DEF_CURRENT', 'VER_DEF_NONE', 'VER_DEF_NUM', 'VER_FLG_BASE',
    'VER_FLG_WEAK', 'VER_NDX_ELIMINATE', 'VER_NDX_GLOBAL',
    'VER_NDX_LOCAL', 'VER_NDX_LORESERVE', 'VER_NEED_CURRENT',
    'VER_NEED_NONE', 'VER_NEED_NUM', 'Val_GNU_MIPS_ABI_FP_64',
    'Val_GNU_MIPS_ABI_FP_64A', 'Val_GNU_MIPS_ABI_FP_ANY',
    'Val_GNU_MIPS_ABI_FP_DOUBLE', 'Val_GNU_MIPS_ABI_FP_MAX',
    'Val_GNU_MIPS_ABI_FP_OLD_64', 'Val_GNU_MIPS_ABI_FP_SINGLE',
    'Val_GNU_MIPS_ABI_FP_SOFT', 'Val_GNU_MIPS_ABI_FP_XX', 'W_OK',
    'X_OK', '_ELF_H', '_POSIX2_C_BIND', '_POSIX2_C_DEV',
    '_POSIX2_C_VERSION', '_POSIX2_LOCALEDEF', '_POSIX2_SW_DEV',
    '_POSIX2_VERSION', '_POSIX_VERSION', '_STRING_H', '_SYSCALL_H',
    '_SYS_MMAN_H', '_UNISTD_H', '_XOPEN_ENH_I18N', '_XOPEN_LEGACY',
    '_XOPEN_UNIX', '_XOPEN_VERSION', '_XOPEN_XCU_VERSION',
    '_XOPEN_XPG2', '_XOPEN_XPG3', '_XOPEN_XPG4',
    '__ASM_GENERIC_MMAN_COMMON_H',
    '__GLIBC_INTERNAL_STARTING_HEADER_IMPLEMENTATION',
    '__POSIX2_THIS_VERSION', '__environ', '__getpgid', '__gid_t',
    '__gid_t_defined', '__intptr_t_defined', '__memcmpeq',
    '__mempcpy', '__mode_t_defined', '__need_NULL', '__need_size_t',
    '__off_t', '__off_t_defined', '__pid_t', '__pid_t_defined',
    '__socklen_t_defined', '__ssize_t_defined', '__stpcpy',
    '__stpncpy', '__strtok_r', '__uid_t', '__uid_t_defined',
    '__useconds_t', '__useconds_t_defined', '_exit', 'access', 'acct',
    'alarm', 'brk', 'c__Ea_Val_GNU_MIPS_ABI_FP_ANY', 'chdir', 'chown',
    'chroot', 'close', 'closefrom', 'confstr', 'crypt', 'daemon',
    'dup', 'dup2', 'endusershell', 'execl', 'execle', 'execlp',
    'execv', 'execve', 'execvp', 'explicit_bzero', 'faccessat',
    'fchdir', 'fchown', 'fchownat', 'fdatasync', 'fexecve', 'fork',
    'fpathconf', 'fsync', 'ftruncate', 'getcwd', 'getdomainname',
    'getdtablesize', 'getegid', 'getentropy', 'geteuid', 'getgid',
    'getgroups', 'gethostid', 'gethostname', 'getlogin', 'getlogin_r',
    'getpagesize', 'getpass', 'getpgid', 'getpgrp', 'getpid',
    'getppid', 'getsid', 'getuid', 'getusershell', 'getwd', 'gid_t',
    'intptr_t', 'isatty', 'lchown', 'link', 'linkat', 'locale_t',
    'lockf', 'lseek', 'madvise', 'memccpy', 'memchr', 'memcmp',
    'memcpy', 'memmem', 'memmove', 'mempcpy', 'memset', 'mincore',
    'mlock', 'mlockall', 'mmap', 'mode_t', 'mprotect', 'msync',
    'munlock', 'munlockall', 'munmap', 'nice', 'off_t', 'pathconf',
    'pause', 'pid_t', 'pipe', 'posix_madvise', 'pread', 'profil',
    'pwrite', 'read', 'readlink', 'readlinkat', 'revoke', 'rmdir',
    'sbrk', 'setdomainname', 'setegid', 'seteuid', 'setgid',
    'sethostid', 'sethostname', 'setlogin', 'setpgid', 'setpgrp',
    'setregid', 'setreuid', 'setsid', 'setuid', 'setusershell',
    'shm_open', 'shm_unlink', 'size_t', 'sleep', 'socklen_t',
    'ssize_t', 'stpcpy', 'stpncpy', 'strcasestr', 'strcat', 'strchr',
    'strchrnul', 'strcmp', 'strcoll', 'strcoll_l', 'strcpy',
    'strcspn', 'strdup', 'strerror', 'strerror_l', 'strerror_r',
    'strlcat', 'strlcpy', 'strlen', 'strncat', 'strncmp', 'strncpy',
    'strndup', 'strnlen', 'strpbrk', 'strrchr', 'strsep', 'strsignal',
    'strspn', 'strstr', 'strtok', 'strtok_r', 'struct___locale_data',
    'struct___locale_struct', 'struct_c__SA_Elf32_Chdr',
    'struct_c__SA_Elf32_Dyn', 'struct_c__SA_Elf32_Ehdr',
    'struct_c__SA_Elf32_Lib', 'struct_c__SA_Elf32_Move',
    'struct_c__SA_Elf32_Nhdr', 'struct_c__SA_Elf32_Phdr',
    'struct_c__SA_Elf32_RegInfo', 'struct_c__SA_Elf32_Rel',
    'struct_c__SA_Elf32_Rela', 'struct_c__SA_Elf32_Shdr',
    'struct_c__SA_Elf32_Sym', 'struct_c__SA_Elf32_Syminfo',
    'struct_c__SA_Elf32_Verdaux', 'struct_c__SA_Elf32_Verdef',
    'struct_c__SA_Elf32_Vernaux', 'struct_c__SA_Elf32_Verneed',
    'struct_c__SA_Elf32_auxv_t', 'struct_c__SA_Elf64_Chdr',
    'struct_c__SA_Elf64_Dyn', 'struct_c__SA_Elf64_Ehdr',
    'struct_c__SA_Elf64_Lib', 'struct_c__SA_Elf64_Move',
    'struct_c__SA_Elf64_Nhdr', 'struct_c__SA_Elf64_Phdr',
    'struct_c__SA_Elf64_Rel', 'struct_c__SA_Elf64_Rela',
    'struct_c__SA_Elf64_Shdr', 'struct_c__SA_Elf64_Sym',
    'struct_c__SA_Elf64_Syminfo', 'struct_c__SA_Elf64_Verdaux',
    'struct_c__SA_Elf64_Verdef', 'struct_c__SA_Elf64_Vernaux',
    'struct_c__SA_Elf64_Verneed', 'struct_c__SA_Elf64_auxv_t',
    'struct_c__SA_Elf_MIPS_ABIFlags_v0', 'struct_c__SA_Elf_Options',
    'struct_c__SA_Elf_Options_Hw',
    'struct_c__UA_Elf32_gptab_gt_entry',
    'struct_c__UA_Elf32_gptab_gt_header', 'strxfrm', 'strxfrm_l',
    'symlink', 'symlinkat', 'sync', 'syscall', 'sysconf', 'tcgetpgrp',
    'tcsetpgrp', 'truncate', 'ttyname', 'ttyname_r', 'ttyslot',
    'ualarm', 'uid_t', 'union_c__SA_Elf32_Dyn_d_un',
    'union_c__SA_Elf32_auxv_t_a_un', 'union_c__SA_Elf64_Dyn_d_un',
    'union_c__SA_Elf64_auxv_t_a_un', 'union_c__UA_Elf32_gptab',
    'unlink', 'unlinkat', 'useconds_t', 'usleep', 'vfork', 'vhangup',
    'write']
