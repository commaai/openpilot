# mypy: ignore-errors
import ctypes
from tinygrad.helpers import unwrap
from tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR
from ctypes.util import find_library
def dll():
  try: return ctypes.CDLL(unwrap(find_library('c')), use_errno=True)
  except: pass
  return None
dll = dll()

off_t = ctypes.c_int64
mode_t = ctypes.c_uint32
size_t = ctypes.c_uint64
__off_t = ctypes.c_int64
# extern void *mmap(void *__addr, size_t __len, int __prot, int __flags, int __fd, __off_t __offset) __attribute__((nothrow))
try: (mmap:=dll.mmap).restype, mmap.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int64]
except AttributeError: pass

# extern int munmap(void *__addr, size_t __len) __attribute__((nothrow))
try: (munmap:=dll.munmap).restype, munmap.argtypes = ctypes.c_int32, [ctypes.c_void_p, size_t]
except AttributeError: pass

# extern int mprotect(void *__addr, size_t __len, int __prot) __attribute__((nothrow))
try: (mprotect:=dll.mprotect).restype, mprotect.argtypes = ctypes.c_int32, [ctypes.c_void_p, size_t, ctypes.c_int32]
except AttributeError: pass

# extern int msync(void *__addr, size_t __len, int __flags)
try: (msync:=dll.msync).restype, msync.argtypes = ctypes.c_int32, [ctypes.c_void_p, size_t, ctypes.c_int32]
except AttributeError: pass

# extern int madvise(void *__addr, size_t __len, int __advice) __attribute__((nothrow))
try: (madvise:=dll.madvise).restype, madvise.argtypes = ctypes.c_int32, [ctypes.c_void_p, size_t, ctypes.c_int32]
except AttributeError: pass

# extern int posix_madvise(void *__addr, size_t __len, int __advice) __attribute__((nothrow))
try: (posix_madvise:=dll.posix_madvise).restype, posix_madvise.argtypes = ctypes.c_int32, [ctypes.c_void_p, size_t, ctypes.c_int32]
except AttributeError: pass

# extern int mlock(const void *__addr, size_t __len) __attribute__((nothrow))
try: (mlock:=dll.mlock).restype, mlock.argtypes = ctypes.c_int32, [ctypes.c_void_p, size_t]
except AttributeError: pass

# extern int munlock(const void *__addr, size_t __len) __attribute__((nothrow))
try: (munlock:=dll.munlock).restype, munlock.argtypes = ctypes.c_int32, [ctypes.c_void_p, size_t]
except AttributeError: pass

# extern int mlockall(int __flags) __attribute__((nothrow))
try: (mlockall:=dll.mlockall).restype, mlockall.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern int munlockall(void) __attribute__((nothrow))
try: (munlockall:=dll.munlockall).restype, munlockall.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern int mincore(void *__start, size_t __len, unsigned char *__vec) __attribute__((nothrow))
try: (mincore:=dll.mincore).restype, mincore.argtypes = ctypes.c_int32, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError: pass

# extern int shm_open(const char *__name, int __oflag, mode_t __mode)
try: (shm_open:=dll.shm_open).restype, shm_open.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, mode_t]
except AttributeError: pass

# extern int shm_unlink(const char *__name)
try: (shm_unlink:=dll.shm_unlink).restype, shm_unlink.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern void *memcpy(void *restrict __dest, const void *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (memcpy:=dll.memcpy).restype, memcpy.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# extern void *memmove(void *__dest, const void *__src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (memmove:=dll.memmove).restype, memmove.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# extern void *memccpy(void *restrict __dest, const void *restrict __src, int __c, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (memccpy:=dll.memccpy).restype, memccpy.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, size_t]
except AttributeError: pass

# extern void *memset(void *__s, int __c, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (memset:=dll.memset).restype, memset.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int32, size_t]
except AttributeError: pass

# extern int memcmp(const void *__s1, const void *__s2, size_t __n) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (memcmp:=dll.memcmp).restype, memcmp.argtypes = ctypes.c_int32, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# extern int __memcmpeq(const void *__s1, const void *__s2, size_t __n) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (__memcmpeq:=dll.__memcmpeq).restype, __memcmpeq.argtypes = ctypes.c_int32, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# extern void *memchr(const void *__s, int __c, size_t __n) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1)))
try: (memchr:=dll.memchr).restype, memchr.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int32, size_t]
except AttributeError: pass

# extern char *strcpy(char *restrict __dest, const char *restrict __src) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (strcpy:=dll.strcpy).restype, strcpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *strncpy(char *restrict __dest, const char *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (strncpy:=dll.strncpy).restype, strncpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern char *strcat(char *restrict __dest, const char *restrict __src) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (strcat:=dll.strcat).restype, strcat.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *strncat(char *restrict __dest, const char *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (strncat:=dll.strncat).restype, strncat.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int strcmp(const char *__s1, const char *__s2) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (strcmp:=dll.strcmp).restype, strcmp.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int strncmp(const char *__s1, const char *__s2, size_t __n) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (strncmp:=dll.strncmp).restype, strncmp.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int strcoll(const char *__s1, const char *__s2) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (strcoll:=dll.strcoll).restype, strcoll.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern unsigned long strxfrm(char *restrict __dest, const char *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(2)))
try: (strxfrm:=dll.strxfrm).restype, strxfrm.argtypes = ctypes.c_uint64, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

class struct___locale_struct(Struct): pass
class struct___locale_data(Struct): pass
struct___locale_struct._fields_ = [
  ('__locales', (ctypes.POINTER(struct___locale_data) * 13)),
  ('__ctype_b', ctypes.POINTER(ctypes.c_uint16)),
  ('__ctype_tolower', ctypes.POINTER(ctypes.c_int32)),
  ('__ctype_toupper', ctypes.POINTER(ctypes.c_int32)),
  ('__names', (ctypes.POINTER(ctypes.c_char) * 13)),
]
locale_t = ctypes.POINTER(struct___locale_struct)
# extern int strcoll_l(const char *__s1, const char *__s2, locale_t __l) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2, 3)))
try: (strcoll_l:=dll.strcoll_l).restype, strcoll_l.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), locale_t]
except AttributeError: pass

# extern size_t strxfrm_l(char *__dest, const char *__src, size_t __n, locale_t __l) __attribute__((nothrow)) __attribute__((nonnull(2, 4)))
try: (strxfrm_l:=dll.strxfrm_l).restype, strxfrm_l.argtypes = size_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t, locale_t]
except AttributeError: pass

# extern char *strdup(const char *__s) __attribute__((nothrow)) __attribute__((malloc)) __attribute__((nonnull(1)))
try: (strdup:=dll.strdup).restype, strdup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *strndup(const char *__string, size_t __n) __attribute__((nothrow)) __attribute__((malloc)) __attribute__((nonnull(1)))
try: (strndup:=dll.strndup).restype, strndup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern char *strchr(const char *__s, int __c) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1)))
try: (strchr:=dll.strchr).restype, strchr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern char *strrchr(const char *__s, int __c) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1)))
try: (strrchr:=dll.strrchr).restype, strrchr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern char *strchrnul(const char *__s, int __c) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1)))
try: (strchrnul:=dll.strchrnul).restype, strchrnul.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern unsigned long strcspn(const char *__s, const char *__reject) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (strcspn:=dll.strcspn).restype, strcspn.argtypes = ctypes.c_uint64, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern unsigned long strspn(const char *__s, const char *__accept) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (strspn:=dll.strspn).restype, strspn.argtypes = ctypes.c_uint64, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *strpbrk(const char *__s, const char *__accept) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (strpbrk:=dll.strpbrk).restype, strpbrk.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *strstr(const char *__haystack, const char *__needle) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (strstr:=dll.strstr).restype, strstr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *strtok(char *restrict __s, const char *restrict __delim) __attribute__((nothrow)) __attribute__((nonnull(2)))
try: (strtok:=dll.strtok).restype, strtok.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *__strtok_r(char *restrict __s, const char *restrict __delim, char **restrict __save_ptr) __attribute__((nothrow)) __attribute__((nonnull(2, 3)))
try: (__strtok_r:=dll.__strtok_r).restype, __strtok_r.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# extern char *strtok_r(char *restrict __s, const char *restrict __delim, char **restrict __save_ptr) __attribute__((nothrow)) __attribute__((nonnull(2, 3)))
try: (strtok_r:=dll.strtok_r).restype, strtok_r.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# extern char *strcasestr(const char *__haystack, const char *__needle) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 2)))
try: (strcasestr:=dll.strcasestr).restype, strcasestr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern void *memmem(const void *__haystack, size_t __haystacklen, const void *__needle, size_t __needlelen) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1, 3)))
try: (memmem:=dll.memmem).restype, memmem.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

# extern void *__mempcpy(void *restrict __dest, const void *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (__mempcpy:=dll.__mempcpy).restype, __mempcpy.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# extern void *mempcpy(void *restrict __dest, const void *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (mempcpy:=dll.mempcpy).restype, mempcpy.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# extern unsigned long strlen(const char *__s) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1)))
try: (strlen:=dll.strlen).restype, strlen.argtypes = ctypes.c_uint64, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern size_t strnlen(const char *__string, size_t __maxlen) __attribute__((nothrow)) __attribute__((pure)) __attribute__((nonnull(1)))
try: (strnlen:=dll.strnlen).restype, strnlen.argtypes = size_t, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern char *strerror(int __errnum) __attribute__((nothrow))
try: (strerror:=dll.strerror).restype, strerror.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int32]
except AttributeError: pass

# extern int strerror_r(int __errnum, char *__buf, size_t __buflen) asm("__xpg_strerror_r") __attribute__((nothrow)) __attribute__((nonnull(2)))
try: (strerror_r:=dll.strerror_r).restype, strerror_r.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern char *strerror_l(int __errnum, locale_t __l) __attribute__((nothrow))
try: (strerror_l:=dll.strerror_l).restype, strerror_l.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int32, locale_t]
except AttributeError: pass

# extern void explicit_bzero(void *__s, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (explicit_bzero:=dll.explicit_bzero).restype, explicit_bzero.argtypes = None, [ctypes.c_void_p, size_t]
except AttributeError: pass

# extern char *strsep(char **restrict __stringp, const char *restrict __delim) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (strsep:=dll.strsep).restype, strsep.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *strsignal(int __sig) __attribute__((nothrow))
try: (strsignal:=dll.strsignal).restype, strsignal.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int32]
except AttributeError: pass

# extern char *__stpcpy(char *restrict __dest, const char *restrict __src) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (__stpcpy:=dll.__stpcpy).restype, __stpcpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *stpcpy(char *restrict __dest, const char *restrict __src) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (stpcpy:=dll.stpcpy).restype, stpcpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *__stpncpy(char *restrict __dest, const char *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (__stpncpy:=dll.__stpncpy).restype, __stpncpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern char *stpncpy(char *restrict __dest, const char *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (stpncpy:=dll.stpncpy).restype, stpncpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern size_t strlcpy(char *restrict __dest, const char *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (strlcpy:=dll.strlcpy).restype, strlcpy.argtypes = size_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern size_t strlcat(char *restrict __dest, const char *restrict __src, size_t __n) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (strlcat:=dll.strlcat).restype, strlcat.argtypes = size_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

Elf32_Half = ctypes.c_uint16
Elf64_Half = ctypes.c_uint16
Elf32_Word = ctypes.c_uint32
Elf32_Sword = ctypes.c_int32
Elf64_Word = ctypes.c_uint32
Elf64_Sword = ctypes.c_int32
Elf32_Xword = ctypes.c_uint64
Elf32_Sxword = ctypes.c_int64
Elf64_Xword = ctypes.c_uint64
Elf64_Sxword = ctypes.c_int64
Elf32_Addr = ctypes.c_uint32
Elf64_Addr = ctypes.c_uint64
Elf32_Off = ctypes.c_uint32
Elf64_Off = ctypes.c_uint64
Elf32_Section = ctypes.c_uint16
Elf64_Section = ctypes.c_uint16
Elf32_Versym = ctypes.c_uint16
Elf64_Versym = ctypes.c_uint16
class Elf32_Ehdr(Struct): pass
Elf32_Ehdr._fields_ = [
  ('e_ident', (ctypes.c_ubyte * 16)),
  ('e_type', Elf32_Half),
  ('e_machine', Elf32_Half),
  ('e_version', Elf32_Word),
  ('e_entry', Elf32_Addr),
  ('e_phoff', Elf32_Off),
  ('e_shoff', Elf32_Off),
  ('e_flags', Elf32_Word),
  ('e_ehsize', Elf32_Half),
  ('e_phentsize', Elf32_Half),
  ('e_phnum', Elf32_Half),
  ('e_shentsize', Elf32_Half),
  ('e_shnum', Elf32_Half),
  ('e_shstrndx', Elf32_Half),
]
class Elf64_Ehdr(Struct): pass
Elf64_Ehdr._fields_ = [
  ('e_ident', (ctypes.c_ubyte * 16)),
  ('e_type', Elf64_Half),
  ('e_machine', Elf64_Half),
  ('e_version', Elf64_Word),
  ('e_entry', Elf64_Addr),
  ('e_phoff', Elf64_Off),
  ('e_shoff', Elf64_Off),
  ('e_flags', Elf64_Word),
  ('e_ehsize', Elf64_Half),
  ('e_phentsize', Elf64_Half),
  ('e_phnum', Elf64_Half),
  ('e_shentsize', Elf64_Half),
  ('e_shnum', Elf64_Half),
  ('e_shstrndx', Elf64_Half),
]
class Elf32_Shdr(Struct): pass
Elf32_Shdr._fields_ = [
  ('sh_name', Elf32_Word),
  ('sh_type', Elf32_Word),
  ('sh_flags', Elf32_Word),
  ('sh_addr', Elf32_Addr),
  ('sh_offset', Elf32_Off),
  ('sh_size', Elf32_Word),
  ('sh_link', Elf32_Word),
  ('sh_info', Elf32_Word),
  ('sh_addralign', Elf32_Word),
  ('sh_entsize', Elf32_Word),
]
class Elf64_Shdr(Struct): pass
Elf64_Shdr._fields_ = [
  ('sh_name', Elf64_Word),
  ('sh_type', Elf64_Word),
  ('sh_flags', Elf64_Xword),
  ('sh_addr', Elf64_Addr),
  ('sh_offset', Elf64_Off),
  ('sh_size', Elf64_Xword),
  ('sh_link', Elf64_Word),
  ('sh_info', Elf64_Word),
  ('sh_addralign', Elf64_Xword),
  ('sh_entsize', Elf64_Xword),
]
class Elf32_Chdr(Struct): pass
Elf32_Chdr._fields_ = [
  ('ch_type', Elf32_Word),
  ('ch_size', Elf32_Word),
  ('ch_addralign', Elf32_Word),
]
class Elf64_Chdr(Struct): pass
Elf64_Chdr._fields_ = [
  ('ch_type', Elf64_Word),
  ('ch_reserved', Elf64_Word),
  ('ch_size', Elf64_Xword),
  ('ch_addralign', Elf64_Xword),
]
class Elf32_Sym(Struct): pass
Elf32_Sym._fields_ = [
  ('st_name', Elf32_Word),
  ('st_value', Elf32_Addr),
  ('st_size', Elf32_Word),
  ('st_info', ctypes.c_ubyte),
  ('st_other', ctypes.c_ubyte),
  ('st_shndx', Elf32_Section),
]
class Elf64_Sym(Struct): pass
Elf64_Sym._fields_ = [
  ('st_name', Elf64_Word),
  ('st_info', ctypes.c_ubyte),
  ('st_other', ctypes.c_ubyte),
  ('st_shndx', Elf64_Section),
  ('st_value', Elf64_Addr),
  ('st_size', Elf64_Xword),
]
class Elf32_Syminfo(Struct): pass
Elf32_Syminfo._fields_ = [
  ('si_boundto', Elf32_Half),
  ('si_flags', Elf32_Half),
]
class Elf64_Syminfo(Struct): pass
Elf64_Syminfo._fields_ = [
  ('si_boundto', Elf64_Half),
  ('si_flags', Elf64_Half),
]
class Elf32_Rel(Struct): pass
Elf32_Rel._fields_ = [
  ('r_offset', Elf32_Addr),
  ('r_info', Elf32_Word),
]
class Elf64_Rel(Struct): pass
Elf64_Rel._fields_ = [
  ('r_offset', Elf64_Addr),
  ('r_info', Elf64_Xword),
]
class Elf32_Rela(Struct): pass
Elf32_Rela._fields_ = [
  ('r_offset', Elf32_Addr),
  ('r_info', Elf32_Word),
  ('r_addend', Elf32_Sword),
]
class Elf64_Rela(Struct): pass
Elf64_Rela._fields_ = [
  ('r_offset', Elf64_Addr),
  ('r_info', Elf64_Xword),
  ('r_addend', Elf64_Sxword),
]
Elf32_Relr = ctypes.c_uint32
Elf64_Relr = ctypes.c_uint64
class Elf32_Phdr(Struct): pass
Elf32_Phdr._fields_ = [
  ('p_type', Elf32_Word),
  ('p_offset', Elf32_Off),
  ('p_vaddr', Elf32_Addr),
  ('p_paddr', Elf32_Addr),
  ('p_filesz', Elf32_Word),
  ('p_memsz', Elf32_Word),
  ('p_flags', Elf32_Word),
  ('p_align', Elf32_Word),
]
class Elf64_Phdr(Struct): pass
Elf64_Phdr._fields_ = [
  ('p_type', Elf64_Word),
  ('p_flags', Elf64_Word),
  ('p_offset', Elf64_Off),
  ('p_vaddr', Elf64_Addr),
  ('p_paddr', Elf64_Addr),
  ('p_filesz', Elf64_Xword),
  ('p_memsz', Elf64_Xword),
  ('p_align', Elf64_Xword),
]
class Elf32_Dyn(Struct): pass
class Elf32_Dyn_d_un(ctypes.Union): pass
Elf32_Dyn_d_un._fields_ = [
  ('d_val', Elf32_Word),
  ('d_ptr', Elf32_Addr),
]
Elf32_Dyn._fields_ = [
  ('d_tag', Elf32_Sword),
  ('d_un', Elf32_Dyn_d_un),
]
class Elf64_Dyn(Struct): pass
class Elf64_Dyn_d_un(ctypes.Union): pass
Elf64_Dyn_d_un._fields_ = [
  ('d_val', Elf64_Xword),
  ('d_ptr', Elf64_Addr),
]
Elf64_Dyn._fields_ = [
  ('d_tag', Elf64_Sxword),
  ('d_un', Elf64_Dyn_d_un),
]
class Elf32_Verdef(Struct): pass
Elf32_Verdef._fields_ = [
  ('vd_version', Elf32_Half),
  ('vd_flags', Elf32_Half),
  ('vd_ndx', Elf32_Half),
  ('vd_cnt', Elf32_Half),
  ('vd_hash', Elf32_Word),
  ('vd_aux', Elf32_Word),
  ('vd_next', Elf32_Word),
]
class Elf64_Verdef(Struct): pass
Elf64_Verdef._fields_ = [
  ('vd_version', Elf64_Half),
  ('vd_flags', Elf64_Half),
  ('vd_ndx', Elf64_Half),
  ('vd_cnt', Elf64_Half),
  ('vd_hash', Elf64_Word),
  ('vd_aux', Elf64_Word),
  ('vd_next', Elf64_Word),
]
class Elf32_Verdaux(Struct): pass
Elf32_Verdaux._fields_ = [
  ('vda_name', Elf32_Word),
  ('vda_next', Elf32_Word),
]
class Elf64_Verdaux(Struct): pass
Elf64_Verdaux._fields_ = [
  ('vda_name', Elf64_Word),
  ('vda_next', Elf64_Word),
]
class Elf32_Verneed(Struct): pass
Elf32_Verneed._fields_ = [
  ('vn_version', Elf32_Half),
  ('vn_cnt', Elf32_Half),
  ('vn_file', Elf32_Word),
  ('vn_aux', Elf32_Word),
  ('vn_next', Elf32_Word),
]
class Elf64_Verneed(Struct): pass
Elf64_Verneed._fields_ = [
  ('vn_version', Elf64_Half),
  ('vn_cnt', Elf64_Half),
  ('vn_file', Elf64_Word),
  ('vn_aux', Elf64_Word),
  ('vn_next', Elf64_Word),
]
class Elf32_Vernaux(Struct): pass
Elf32_Vernaux._fields_ = [
  ('vna_hash', Elf32_Word),
  ('vna_flags', Elf32_Half),
  ('vna_other', Elf32_Half),
  ('vna_name', Elf32_Word),
  ('vna_next', Elf32_Word),
]
class Elf64_Vernaux(Struct): pass
Elf64_Vernaux._fields_ = [
  ('vna_hash', Elf64_Word),
  ('vna_flags', Elf64_Half),
  ('vna_other', Elf64_Half),
  ('vna_name', Elf64_Word),
  ('vna_next', Elf64_Word),
]
class Elf32_auxv_t(Struct): pass
uint32_t = ctypes.c_uint32
class Elf32_auxv_t_a_un(ctypes.Union): pass
Elf32_auxv_t_a_un._fields_ = [
  ('a_val', uint32_t),
]
Elf32_auxv_t._fields_ = [
  ('a_type', uint32_t),
  ('a_un', Elf32_auxv_t_a_un),
]
class Elf64_auxv_t(Struct): pass
uint64_t = ctypes.c_uint64
class Elf64_auxv_t_a_un(ctypes.Union): pass
Elf64_auxv_t_a_un._fields_ = [
  ('a_val', uint64_t),
]
Elf64_auxv_t._fields_ = [
  ('a_type', uint64_t),
  ('a_un', Elf64_auxv_t_a_un),
]
class Elf32_Nhdr(Struct): pass
Elf32_Nhdr._fields_ = [
  ('n_namesz', Elf32_Word),
  ('n_descsz', Elf32_Word),
  ('n_type', Elf32_Word),
]
class Elf64_Nhdr(Struct): pass
Elf64_Nhdr._fields_ = [
  ('n_namesz', Elf64_Word),
  ('n_descsz', Elf64_Word),
  ('n_type', Elf64_Word),
]
class Elf32_Move(Struct): pass
Elf32_Move._fields_ = [
  ('m_value', Elf32_Xword),
  ('m_info', Elf32_Word),
  ('m_poffset', Elf32_Word),
  ('m_repeat', Elf32_Half),
  ('m_stride', Elf32_Half),
]
class Elf64_Move(Struct): pass
Elf64_Move._fields_ = [
  ('m_value', Elf64_Xword),
  ('m_info', Elf64_Xword),
  ('m_poffset', Elf64_Xword),
  ('m_repeat', Elf64_Half),
  ('m_stride', Elf64_Half),
]
class Elf32_gptab(ctypes.Union): pass
class Elf32_gptab_gt_header(Struct): pass
Elf32_gptab_gt_header._fields_ = [
  ('gt_current_g_value', Elf32_Word),
  ('gt_unused', Elf32_Word),
]
class Elf32_gptab_gt_entry(Struct): pass
Elf32_gptab_gt_entry._fields_ = [
  ('gt_g_value', Elf32_Word),
  ('gt_bytes', Elf32_Word),
]
Elf32_gptab._fields_ = [
  ('gt_header', Elf32_gptab_gt_header),
  ('gt_entry', Elf32_gptab_gt_entry),
]
class Elf32_RegInfo(Struct): pass
Elf32_RegInfo._fields_ = [
  ('ri_gprmask', Elf32_Word),
  ('ri_cprmask', (Elf32_Word * 4)),
  ('ri_gp_value', Elf32_Sword),
]
class Elf_Options(Struct): pass
Elf_Options._fields_ = [
  ('kind', ctypes.c_ubyte),
  ('size', ctypes.c_ubyte),
  ('section', Elf32_Section),
  ('info', Elf32_Word),
]
class Elf_Options_Hw(Struct): pass
Elf_Options_Hw._fields_ = [
  ('hwp_flags1', Elf32_Word),
  ('hwp_flags2', Elf32_Word),
]
class Elf32_Lib(Struct): pass
Elf32_Lib._fields_ = [
  ('l_name', Elf32_Word),
  ('l_time_stamp', Elf32_Word),
  ('l_checksum', Elf32_Word),
  ('l_version', Elf32_Word),
  ('l_flags', Elf32_Word),
]
class Elf64_Lib(Struct): pass
Elf64_Lib._fields_ = [
  ('l_name', Elf64_Word),
  ('l_time_stamp', Elf64_Word),
  ('l_checksum', Elf64_Word),
  ('l_version', Elf64_Word),
  ('l_flags', Elf64_Word),
]
Elf32_Conflict = ctypes.c_uint32
class Elf_MIPS_ABIFlags_v0(Struct): pass
Elf_MIPS_ABIFlags_v0._fields_ = [
  ('version', Elf32_Half),
  ('isa_level', ctypes.c_ubyte),
  ('isa_rev', ctypes.c_ubyte),
  ('gpr_size', ctypes.c_ubyte),
  ('cpr1_size', ctypes.c_ubyte),
  ('cpr2_size', ctypes.c_ubyte),
  ('fp_abi', ctypes.c_ubyte),
  ('isa_ext', Elf32_Word),
  ('ases', Elf32_Word),
  ('flags1', Elf32_Word),
  ('flags2', Elf32_Word),
]
_anonenum0 = CEnum(ctypes.c_uint32)
Val_GNU_MIPS_ABI_FP_ANY = _anonenum0.define('Val_GNU_MIPS_ABI_FP_ANY', 0)
Val_GNU_MIPS_ABI_FP_DOUBLE = _anonenum0.define('Val_GNU_MIPS_ABI_FP_DOUBLE', 1)
Val_GNU_MIPS_ABI_FP_SINGLE = _anonenum0.define('Val_GNU_MIPS_ABI_FP_SINGLE', 2)
Val_GNU_MIPS_ABI_FP_SOFT = _anonenum0.define('Val_GNU_MIPS_ABI_FP_SOFT', 3)
Val_GNU_MIPS_ABI_FP_OLD_64 = _anonenum0.define('Val_GNU_MIPS_ABI_FP_OLD_64', 4)
Val_GNU_MIPS_ABI_FP_XX = _anonenum0.define('Val_GNU_MIPS_ABI_FP_XX', 5)
Val_GNU_MIPS_ABI_FP_64 = _anonenum0.define('Val_GNU_MIPS_ABI_FP_64', 6)
Val_GNU_MIPS_ABI_FP_64A = _anonenum0.define('Val_GNU_MIPS_ABI_FP_64A', 7)
Val_GNU_MIPS_ABI_FP_MAX = _anonenum0.define('Val_GNU_MIPS_ABI_FP_MAX', 7)

ssize_t = ctypes.c_int64
gid_t = ctypes.c_uint32
uid_t = ctypes.c_uint32
useconds_t = ctypes.c_uint32
pid_t = ctypes.c_int32
intptr_t = ctypes.c_int64
socklen_t = ctypes.c_uint32
# extern int access(const char *__name, int __type) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (access:=dll.access).restype, access.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern int faccessat(int __fd, const char *__file, int __type, int __flag) __attribute__((nothrow)) __attribute__((nonnull(2)))
try: (faccessat:=dll.faccessat).restype, faccessat.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

# extern __off_t lseek(int __fd, __off_t __offset, int __whence) __attribute__((nothrow))
try: (lseek:=dll.lseek).restype, lseek.argtypes = ctypes.c_int64, [ctypes.c_int32, ctypes.c_int64, ctypes.c_int32]
except AttributeError: pass

# extern int close(int __fd)
try: (close:=dll.close).restype, close.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern void closefrom(int __lowfd) __attribute__((nothrow))
try: (closefrom:=dll.closefrom).restype, closefrom.argtypes = None, [ctypes.c_int32]
except AttributeError: pass

# extern ssize_t read(int __fd, void *__buf, size_t __nbytes)
try: (read:=dll.read).restype, read.argtypes = ssize_t, [ctypes.c_int32, ctypes.c_void_p, size_t]
except AttributeError: pass

# extern ssize_t write(int __fd, const void *__buf, size_t __n)
try: (write:=dll.write).restype, write.argtypes = ssize_t, [ctypes.c_int32, ctypes.c_void_p, size_t]
except AttributeError: pass

# extern ssize_t pread(int __fd, void *__buf, size_t __nbytes, __off_t __offset)
try: (pread:=dll.pread).restype, pread.argtypes = ssize_t, [ctypes.c_int32, ctypes.c_void_p, size_t, ctypes.c_int64]
except AttributeError: pass

# extern ssize_t pwrite(int __fd, const void *__buf, size_t __n, __off_t __offset)
try: (pwrite:=dll.pwrite).restype, pwrite.argtypes = ssize_t, [ctypes.c_int32, ctypes.c_void_p, size_t, ctypes.c_int64]
except AttributeError: pass

# extern int pipe(int __pipedes[2]) __attribute__((nothrow))
try: (pipe:=dll.pipe).restype, pipe.argtypes = ctypes.c_int32, [(ctypes.c_int32 * 2)]
except AttributeError: pass

# extern unsigned int alarm(unsigned int __seconds) __attribute__((nothrow))
try: (alarm:=dll.alarm).restype, alarm.argtypes = ctypes.c_uint32, [ctypes.c_uint32]
except AttributeError: pass

# extern unsigned int sleep(unsigned int __seconds)
try: (sleep:=dll.sleep).restype, sleep.argtypes = ctypes.c_uint32, [ctypes.c_uint32]
except AttributeError: pass

__useconds_t = ctypes.c_uint32
# extern __useconds_t ualarm(__useconds_t __value, __useconds_t __interval) __attribute__((nothrow))
try: (ualarm:=dll.ualarm).restype, ualarm.argtypes = ctypes.c_uint32, [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

# extern int usleep(__useconds_t __useconds)
try: (usleep:=dll.usleep).restype, usleep.argtypes = ctypes.c_int32, [ctypes.c_uint32]
except AttributeError: pass

# extern int pause(void)
try: (pause:=dll.pause).restype, pause.argtypes = ctypes.c_int32, []
except AttributeError: pass

__uid_t = ctypes.c_uint32
__gid_t = ctypes.c_uint32
# extern int chown(const char *__file, __uid_t __owner, __gid_t __group) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (chown:=dll.chown).restype, chown.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

# extern int fchown(int __fd, __uid_t __owner, __gid_t __group) __attribute__((nothrow))
try: (fchown:=dll.fchown).restype, fchown.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

# extern int lchown(const char *__file, __uid_t __owner, __gid_t __group) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (lchown:=dll.lchown).restype, lchown.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

# extern int fchownat(int __fd, const char *__file, __uid_t __owner, __gid_t __group, int __flag) __attribute__((nothrow)) __attribute__((nonnull(2)))
try: (fchownat:=dll.fchownat).restype, fchownat.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32]
except AttributeError: pass

# extern int chdir(const char *__path) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (chdir:=dll.chdir).restype, chdir.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int fchdir(int __fd) __attribute__((nothrow))
try: (fchdir:=dll.fchdir).restype, fchdir.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern char *getcwd(char *__buf, size_t __size) __attribute__((nothrow))
try: (getcwd:=dll.getcwd).restype, getcwd.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern char *getwd(char *__buf) __attribute__((nothrow)) __attribute__((nonnull(1))) __attribute__((deprecated("")))
try: (getwd:=dll.getwd).restype, getwd.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int dup(int __fd) __attribute__((nothrow))
try: (dup:=dll.dup).restype, dup.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern int dup2(int __fd, int __fd2) __attribute__((nothrow))
try: (dup2:=dll.dup2).restype, dup2.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

try: __environ = ctypes.POINTER(ctypes.POINTER(ctypes.c_char)).in_dll(dll, '__environ')
except (ValueError,AttributeError): pass
# extern int execve(const char *__path, char *const __argv[], char *const __envp[]) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (execve:=dll.execve).restype, execve.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), (ctypes.POINTER(ctypes.c_char) * 0), (ctypes.POINTER(ctypes.c_char) * 0)]
except AttributeError: pass

# extern int fexecve(int __fd, char *const __argv[], char *const __envp[]) __attribute__((nothrow)) __attribute__((nonnull(2)))
try: (fexecve:=dll.fexecve).restype, fexecve.argtypes = ctypes.c_int32, [ctypes.c_int32, (ctypes.POINTER(ctypes.c_char) * 0), (ctypes.POINTER(ctypes.c_char) * 0)]
except AttributeError: pass

# extern int execv(const char *__path, char *const __argv[]) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (execv:=dll.execv).restype, execv.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), (ctypes.POINTER(ctypes.c_char) * 0)]
except AttributeError: pass

# extern int execle(const char *__path, const char *__arg, ...) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (execle:=dll.execle).restype, execle.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int execl(const char *__path, const char *__arg, ...) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (execl:=dll.execl).restype, execl.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int execvp(const char *__file, char *const __argv[]) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (execvp:=dll.execvp).restype, execvp.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), (ctypes.POINTER(ctypes.c_char) * 0)]
except AttributeError: pass

# extern int execlp(const char *__file, const char *__arg, ...) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (execlp:=dll.execlp).restype, execlp.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int nice(int __inc) __attribute__((nothrow))
try: (nice:=dll.nice).restype, nice.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern void _exit(int __status)
try: (_exit:=dll._exit).restype, _exit.argtypes = None, [ctypes.c_int32]
except AttributeError: pass

# extern long pathconf(const char *__path, int __name) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (pathconf:=dll.pathconf).restype, pathconf.argtypes = ctypes.c_int64, [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern long fpathconf(int __fd, int __name) __attribute__((nothrow))
try: (fpathconf:=dll.fpathconf).restype, fpathconf.argtypes = ctypes.c_int64, [ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

# extern long sysconf(int __name) __attribute__((nothrow))
try: (sysconf:=dll.sysconf).restype, sysconf.argtypes = ctypes.c_int64, [ctypes.c_int32]
except AttributeError: pass

# extern size_t confstr(int __name, char *__buf, size_t __len) __attribute__((nothrow))
try: (confstr:=dll.confstr).restype, confstr.argtypes = size_t, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

__pid_t = ctypes.c_int32
# extern __pid_t getpid(void) __attribute__((nothrow))
try: (getpid:=dll.getpid).restype, getpid.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern __pid_t getppid(void) __attribute__((nothrow))
try: (getppid:=dll.getppid).restype, getppid.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern __pid_t getpgrp(void) __attribute__((nothrow))
try: (getpgrp:=dll.getpgrp).restype, getpgrp.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern __pid_t __getpgid(__pid_t __pid) __attribute__((nothrow))
try: (__getpgid:=dll.__getpgid).restype, __getpgid.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern __pid_t getpgid(__pid_t __pid) __attribute__((nothrow))
try: (getpgid:=dll.getpgid).restype, getpgid.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern int setpgid(__pid_t __pid, __pid_t __pgid) __attribute__((nothrow))
try: (setpgid:=dll.setpgid).restype, setpgid.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

# extern int setpgrp(void) __attribute__((nothrow))
try: (setpgrp:=dll.setpgrp).restype, setpgrp.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern __pid_t setsid(void) __attribute__((nothrow))
try: (setsid:=dll.setsid).restype, setsid.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern __pid_t getsid(__pid_t __pid) __attribute__((nothrow))
try: (getsid:=dll.getsid).restype, getsid.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern __uid_t getuid(void) __attribute__((nothrow))
try: (getuid:=dll.getuid).restype, getuid.argtypes = ctypes.c_uint32, []
except AttributeError: pass

# extern __uid_t geteuid(void) __attribute__((nothrow))
try: (geteuid:=dll.geteuid).restype, geteuid.argtypes = ctypes.c_uint32, []
except AttributeError: pass

# extern __gid_t getgid(void) __attribute__((nothrow))
try: (getgid:=dll.getgid).restype, getgid.argtypes = ctypes.c_uint32, []
except AttributeError: pass

# extern __gid_t getegid(void) __attribute__((nothrow))
try: (getegid:=dll.getegid).restype, getegid.argtypes = ctypes.c_uint32, []
except AttributeError: pass

# extern int getgroups(int __size, __gid_t __list[]) __attribute__((nothrow))
try: (getgroups:=dll.getgroups).restype, getgroups.argtypes = ctypes.c_int32, [ctypes.c_int32, (ctypes.c_uint32 * 0)]
except AttributeError: pass

# extern int setuid(__uid_t __uid) __attribute__((nothrow))
try: (setuid:=dll.setuid).restype, setuid.argtypes = ctypes.c_int32, [ctypes.c_uint32]
except AttributeError: pass

# extern int setreuid(__uid_t __ruid, __uid_t __euid) __attribute__((nothrow))
try: (setreuid:=dll.setreuid).restype, setreuid.argtypes = ctypes.c_int32, [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

# extern int seteuid(__uid_t __uid) __attribute__((nothrow))
try: (seteuid:=dll.seteuid).restype, seteuid.argtypes = ctypes.c_int32, [ctypes.c_uint32]
except AttributeError: pass

# extern int setgid(__gid_t __gid) __attribute__((nothrow))
try: (setgid:=dll.setgid).restype, setgid.argtypes = ctypes.c_int32, [ctypes.c_uint32]
except AttributeError: pass

# extern int setregid(__gid_t __rgid, __gid_t __egid) __attribute__((nothrow))
try: (setregid:=dll.setregid).restype, setregid.argtypes = ctypes.c_int32, [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

# extern int setegid(__gid_t __gid) __attribute__((nothrow))
try: (setegid:=dll.setegid).restype, setegid.argtypes = ctypes.c_int32, [ctypes.c_uint32]
except AttributeError: pass

# extern __pid_t fork(void) __attribute__((nothrow))
try: (fork:=dll.fork).restype, fork.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern int vfork(void) __attribute__((nothrow))
try: (vfork:=dll.vfork).restype, vfork.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern char *ttyname(int __fd) __attribute__((nothrow))
try: (ttyname:=dll.ttyname).restype, ttyname.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int32]
except AttributeError: pass

# extern int ttyname_r(int __fd, char *__buf, size_t __buflen) __attribute__((nothrow)) __attribute__((nonnull(2)))
try: (ttyname_r:=dll.ttyname_r).restype, ttyname_r.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int isatty(int __fd) __attribute__((nothrow))
try: (isatty:=dll.isatty).restype, isatty.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern int ttyslot(void) __attribute__((nothrow))
try: (ttyslot:=dll.ttyslot).restype, ttyslot.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern int link(const char *__from, const char *__to) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (link:=dll.link).restype, link.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int linkat(int __fromfd, const char *__from, int __tofd, const char *__to, int __flags) __attribute__((nothrow)) __attribute__((nonnull(2, 4)))
try: (linkat:=dll.linkat).restype, linkat.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern int symlink(const char *__from, const char *__to) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (symlink:=dll.symlink).restype, symlink.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern ssize_t readlink(const char *restrict __path, char *restrict __buf, size_t __len) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (readlink:=dll.readlink).restype, readlink.argtypes = ssize_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int symlinkat(const char *__from, int __tofd, const char *__to) __attribute__((nothrow)) __attribute__((nonnull(1, 3)))
try: (symlinkat:=dll.symlinkat).restype, symlinkat.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern ssize_t readlinkat(int __fd, const char *restrict __path, char *restrict __buf, size_t __len) __attribute__((nothrow)) __attribute__((nonnull(2, 3)))
try: (readlinkat:=dll.readlinkat).restype, readlinkat.argtypes = ssize_t, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int unlink(const char *__name) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (unlink:=dll.unlink).restype, unlink.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int unlinkat(int __fd, const char *__name, int __flag) __attribute__((nothrow)) __attribute__((nonnull(2)))
try: (unlinkat:=dll.unlinkat).restype, unlinkat.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern int rmdir(const char *__path) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (rmdir:=dll.rmdir).restype, rmdir.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern __pid_t tcgetpgrp(int __fd) __attribute__((nothrow))
try: (tcgetpgrp:=dll.tcgetpgrp).restype, tcgetpgrp.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern int tcsetpgrp(int __fd, __pid_t __pgrp_id) __attribute__((nothrow))
try: (tcsetpgrp:=dll.tcsetpgrp).restype, tcsetpgrp.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

# extern char *getlogin(void)
try: (getlogin:=dll.getlogin).restype, getlogin.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# extern int getlogin_r(char *__name, size_t __name_len) __attribute__((nonnull(1)))
try: (getlogin_r:=dll.getlogin_r).restype, getlogin_r.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int setlogin(const char *__name) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (setlogin:=dll.setlogin).restype, setlogin.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int gethostname(char *__name, size_t __len) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (gethostname:=dll.gethostname).restype, gethostname.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int sethostname(const char *__name, size_t __len) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (sethostname:=dll.sethostname).restype, sethostname.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int sethostid(long __id) __attribute__((nothrow))
try: (sethostid:=dll.sethostid).restype, sethostid.argtypes = ctypes.c_int32, [ctypes.c_int64]
except AttributeError: pass

# extern int getdomainname(char *__name, size_t __len) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (getdomainname:=dll.getdomainname).restype, getdomainname.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int setdomainname(const char *__name, size_t __len) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (setdomainname:=dll.setdomainname).restype, setdomainname.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern int vhangup(void) __attribute__((nothrow))
try: (vhangup:=dll.vhangup).restype, vhangup.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern int revoke(const char *__file) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (revoke:=dll.revoke).restype, revoke.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int profil(unsigned short *__sample_buffer, size_t __size, size_t __offset, unsigned int __scale) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (profil:=dll.profil).restype, profil.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_uint16), size_t, size_t, ctypes.c_uint32]
except AttributeError: pass

# extern int acct(const char *__name) __attribute__((nothrow))
try: (acct:=dll.acct).restype, acct.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *getusershell(void) __attribute__((nothrow))
try: (getusershell:=dll.getusershell).restype, getusershell.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# extern void endusershell(void) __attribute__((nothrow))
try: (endusershell:=dll.endusershell).restype, endusershell.argtypes = None, []
except AttributeError: pass

# extern void setusershell(void) __attribute__((nothrow))
try: (setusershell:=dll.setusershell).restype, setusershell.argtypes = None, []
except AttributeError: pass

# extern int daemon(int __nochdir, int __noclose) __attribute__((nothrow))
try: (daemon:=dll.daemon).restype, daemon.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

# extern int chroot(const char *__path) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (chroot:=dll.chroot).restype, chroot.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern char *getpass(const char *__prompt) __attribute__((nonnull(1)))
try: (getpass:=dll.getpass).restype, getpass.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern int fsync(int __fd)
try: (fsync:=dll.fsync).restype, fsync.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern long gethostid(void)
try: (gethostid:=dll.gethostid).restype, gethostid.argtypes = ctypes.c_int64, []
except AttributeError: pass

# extern void sync(void) __attribute__((nothrow))
try: (sync:=dll.sync).restype, sync.argtypes = None, []
except AttributeError: pass

# extern int getpagesize(void) __attribute__((nothrow)) __attribute__((const))
try: (getpagesize:=dll.getpagesize).restype, getpagesize.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern int getdtablesize(void) __attribute__((nothrow))
try: (getdtablesize:=dll.getdtablesize).restype, getdtablesize.argtypes = ctypes.c_int32, []
except AttributeError: pass

# extern int truncate(const char *__file, __off_t __length) __attribute__((nothrow)) __attribute__((nonnull(1)))
try: (truncate:=dll.truncate).restype, truncate.argtypes = ctypes.c_int32, [ctypes.POINTER(ctypes.c_char), ctypes.c_int64]
except AttributeError: pass

# extern int ftruncate(int __fd, __off_t __length) __attribute__((nothrow))
try: (ftruncate:=dll.ftruncate).restype, ftruncate.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.c_int64]
except AttributeError: pass

# extern int brk(void *__addr) __attribute__((nothrow))
try: (brk:=dll.brk).restype, brk.argtypes = ctypes.c_int32, [ctypes.c_void_p]
except AttributeError: pass

# extern void *sbrk(intptr_t __delta) __attribute__((nothrow))
try: (sbrk:=dll.sbrk).restype, sbrk.argtypes = ctypes.c_void_p, [intptr_t]
except AttributeError: pass

# extern long syscall(long __sysno, ...) __attribute__((nothrow))
try: (syscall:=dll.syscall).restype, syscall.argtypes = ctypes.c_int64, [ctypes.c_int64]
except AttributeError: pass

# extern int lockf(int __fd, int __cmd, __off_t __len)
try: (lockf:=dll.lockf).restype, lockf.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.c_int32, ctypes.c_int64]
except AttributeError: pass

# extern int fdatasync(int __fildes)
try: (fdatasync:=dll.fdatasync).restype, fdatasync.argtypes = ctypes.c_int32, [ctypes.c_int32]
except AttributeError: pass

# extern char *crypt(const char *__key, const char *__salt) __attribute__((nothrow)) __attribute__((nonnull(1, 2)))
try: (crypt:=dll.crypt).restype, crypt.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int getentropy(void *__buffer, size_t __length)
try: (getentropy:=dll.getentropy).restype, getentropy.argtypes = ctypes.c_int32, [ctypes.c_void_p, size_t]
except AttributeError: pass

_SYS_MMAN_H = 1
_SYSCALL_H = 1
_STRING_H = 1
_ELF_H = 1
EI_NIDENT = (16)
EI_MAG0 = 0
ELFMAG0 = 0x7f
EI_MAG1 = 1
ELFMAG1 = 'E'
EI_MAG2 = 2
ELFMAG2 = 'L'
EI_MAG3 = 3
ELFMAG3 = 'F'
ELFMAG = "\177ELF"
SELFMAG = 4
EI_CLASS = 4
ELFCLASSNONE = 0
ELFCLASS32 = 1
ELFCLASS64 = 2
ELFCLASSNUM = 3
EI_DATA = 5
ELFDATANONE = 0
ELFDATA2LSB = 1
ELFDATA2MSB = 2
ELFDATANUM = 3
EI_VERSION = 6
EI_OSABI = 7
ELFOSABI_NONE = 0
ELFOSABI_SYSV = 0
ELFOSABI_HPUX = 1
ELFOSABI_NETBSD = 2
ELFOSABI_GNU = 3
ELFOSABI_LINUX = ELFOSABI_GNU
ELFOSABI_SOLARIS = 6
ELFOSABI_AIX = 7
ELFOSABI_IRIX = 8
ELFOSABI_FREEBSD = 9
ELFOSABI_TRU64 = 10
ELFOSABI_MODESTO = 11
ELFOSABI_OPENBSD = 12
ELFOSABI_ARM_AEABI = 64
ELFOSABI_ARM = 97
ELFOSABI_STANDALONE = 255
EI_ABIVERSION = 8
EI_PAD = 9
ET_NONE = 0
ET_REL = 1
ET_EXEC = 2
ET_DYN = 3
ET_CORE = 4
ET_NUM = 5
ET_LOOS = 0xfe00
ET_HIOS = 0xfeff
ET_LOPROC = 0xff00
ET_HIPROC = 0xffff
EM_NONE = 0
EM_M32 = 1
EM_SPARC = 2
EM_386 = 3
EM_68K = 4
EM_88K = 5
EM_IAMCU = 6
EM_860 = 7
EM_MIPS = 8
EM_S370 = 9
EM_MIPS_RS3_LE = 10
EM_PARISC = 15
EM_VPP500 = 17
EM_SPARC32PLUS = 18
EM_960 = 19
EM_PPC = 20
EM_PPC64 = 21
EM_S390 = 22
EM_SPU = 23
EM_V800 = 36
EM_FR20 = 37
EM_RH32 = 38
EM_RCE = 39
EM_ARM = 40
EM_FAKE_ALPHA = 41
EM_SH = 42
EM_SPARCV9 = 43
EM_TRICORE = 44
EM_ARC = 45
EM_H8_300 = 46
EM_H8_300H = 47
EM_H8S = 48
EM_H8_500 = 49
EM_IA_64 = 50
EM_MIPS_X = 51
EM_COLDFIRE = 52
EM_68HC12 = 53
EM_MMA = 54
EM_PCP = 55
EM_NCPU = 56
EM_NDR1 = 57
EM_STARCORE = 58
EM_ME16 = 59
EM_ST100 = 60
EM_TINYJ = 61
EM_X86_64 = 62
EM_PDSP = 63
EM_PDP10 = 64
EM_PDP11 = 65
EM_FX66 = 66
EM_ST9PLUS = 67
EM_ST7 = 68
EM_68HC16 = 69
EM_68HC11 = 70
EM_68HC08 = 71
EM_68HC05 = 72
EM_SVX = 73
EM_ST19 = 74
EM_VAX = 75
EM_CRIS = 76
EM_JAVELIN = 77
EM_FIREPATH = 78
EM_ZSP = 79
EM_MMIX = 80
EM_HUANY = 81
EM_PRISM = 82
EM_AVR = 83
EM_FR30 = 84
EM_D10V = 85
EM_D30V = 86
EM_V850 = 87
EM_M32R = 88
EM_MN10300 = 89
EM_MN10200 = 90
EM_PJ = 91
EM_OPENRISC = 92
EM_ARC_COMPACT = 93
EM_XTENSA = 94
EM_VIDEOCORE = 95
EM_TMM_GPP = 96
EM_NS32K = 97
EM_TPC = 98
EM_SNP1K = 99
EM_ST200 = 100
EM_IP2K = 101
EM_MAX = 102
EM_CR = 103
EM_F2MC16 = 104
EM_MSP430 = 105
EM_BLACKFIN = 106
EM_SE_C33 = 107
EM_SEP = 108
EM_ARCA = 109
EM_UNICORE = 110
EM_EXCESS = 111
EM_DXP = 112
EM_ALTERA_NIOS2 = 113
EM_CRX = 114
EM_XGATE = 115
EM_C166 = 116
EM_M16C = 117
EM_DSPIC30F = 118
EM_CE = 119
EM_M32C = 120
EM_TSK3000 = 131
EM_RS08 = 132
EM_SHARC = 133
EM_ECOG2 = 134
EM_SCORE7 = 135
EM_DSP24 = 136
EM_VIDEOCORE3 = 137
EM_LATTICEMICO32 = 138
EM_SE_C17 = 139
EM_TI_C6000 = 140
EM_TI_C2000 = 141
EM_TI_C5500 = 142
EM_TI_ARP32 = 143
EM_TI_PRU = 144
EM_MMDSP_PLUS = 160
EM_CYPRESS_M8C = 161
EM_R32C = 162
EM_TRIMEDIA = 163
EM_QDSP6 = 164
EM_8051 = 165
EM_STXP7X = 166
EM_NDS32 = 167
EM_ECOG1X = 168
EM_MAXQ30 = 169
EM_XIMO16 = 170
EM_MANIK = 171
EM_CRAYNV2 = 172
EM_RX = 173
EM_METAG = 174
EM_MCST_ELBRUS = 175
EM_ECOG16 = 176
EM_CR16 = 177
EM_ETPU = 178
EM_SLE9X = 179
EM_L10M = 180
EM_K10M = 181
EM_AARCH64 = 183
EM_AVR32 = 185
EM_STM8 = 186
EM_TILE64 = 187
EM_TILEPRO = 188
EM_MICROBLAZE = 189
EM_CUDA = 190
EM_TILEGX = 191
EM_CLOUDSHIELD = 192
EM_COREA_1ST = 193
EM_COREA_2ND = 194
EM_ARCV2 = 195
EM_OPEN8 = 196
EM_RL78 = 197
EM_VIDEOCORE5 = 198
EM_78KOR = 199
EM_56800EX = 200
EM_BA1 = 201
EM_BA2 = 202
EM_XCORE = 203
EM_MCHP_PIC = 204
EM_INTELGT = 205
EM_KM32 = 210
EM_KMX32 = 211
EM_EMX16 = 212
EM_EMX8 = 213
EM_KVARC = 214
EM_CDP = 215
EM_COGE = 216
EM_COOL = 217
EM_NORC = 218
EM_CSR_KALIMBA = 219
EM_Z80 = 220
EM_VISIUM = 221
EM_FT32 = 222
EM_MOXIE = 223
EM_AMDGPU = 224
EM_RISCV = 243
EM_BPF = 247
EM_CSKY = 252
EM_LOONGARCH = 258
EM_NUM = 259
EM_ARC_A5 = EM_ARC_COMPACT
EM_ALPHA = 0x9026
EV_NONE = 0
EV_CURRENT = 1
EV_NUM = 2
SHN_UNDEF = 0
SHN_LORESERVE = 0xff00
SHN_LOPROC = 0xff00
SHN_BEFORE = 0xff00
SHN_AFTER = 0xff01
SHN_HIPROC = 0xff1f
SHN_LOOS = 0xff20
SHN_HIOS = 0xff3f
SHN_ABS = 0xfff1
SHN_COMMON = 0xfff2
SHN_XINDEX = 0xffff
SHN_HIRESERVE = 0xffff
SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_HASH = 5
SHT_DYNAMIC = 6
SHT_NOTE = 7
SHT_NOBITS = 8
SHT_REL = 9
SHT_SHLIB = 10
SHT_DYNSYM = 11
SHT_INIT_ARRAY = 14
SHT_FINI_ARRAY = 15
SHT_PREINIT_ARRAY = 16
SHT_GROUP = 17
SHT_SYMTAB_SHNDX = 18
SHT_RELR = 19
SHT_NUM = 20
SHT_LOOS = 0x60000000
SHT_GNU_ATTRIBUTES = 0x6ffffff5
SHT_GNU_HASH = 0x6ffffff6
SHT_GNU_LIBLIST = 0x6ffffff7
SHT_CHECKSUM = 0x6ffffff8
SHT_LOSUNW = 0x6ffffffa
SHT_SUNW_move = 0x6ffffffa
SHT_SUNW_COMDAT = 0x6ffffffb
SHT_SUNW_syminfo = 0x6ffffffc
SHT_GNU_verdef = 0x6ffffffd
SHT_GNU_verneed = 0x6ffffffe
SHT_GNU_versym = 0x6fffffff
SHT_HISUNW = 0x6fffffff
SHT_HIOS = 0x6fffffff
SHT_LOPROC = 0x70000000
SHT_HIPROC = 0x7fffffff
SHT_LOUSER = 0x80000000
SHT_HIUSER = 0x8fffffff
SHF_WRITE = (1 << 0)
SHF_ALLOC = (1 << 1)
SHF_EXECINSTR = (1 << 2)
SHF_MERGE = (1 << 4)
SHF_STRINGS = (1 << 5)
SHF_INFO_LINK = (1 << 6)
SHF_LINK_ORDER = (1 << 7)
SHF_OS_NONCONFORMING = (1 << 8)
SHF_GROUP = (1 << 9)
SHF_TLS = (1 << 10)
SHF_COMPRESSED = (1 << 11)
SHF_MASKOS = 0x0ff00000
SHF_MASKPROC = 0xf0000000
SHF_GNU_RETAIN = (1 << 21)
SHF_ORDERED = (1 << 30)
SHF_EXCLUDE = (1 << 31)
ELFCOMPRESS_ZLIB = 1
ELFCOMPRESS_ZSTD = 2
ELFCOMPRESS_LOOS = 0x60000000
ELFCOMPRESS_HIOS = 0x6fffffff
ELFCOMPRESS_LOPROC = 0x70000000
ELFCOMPRESS_HIPROC = 0x7fffffff
GRP_COMDAT = 0x1
SYMINFO_BT_SELF = 0xffff
SYMINFO_BT_PARENT = 0xfffe
SYMINFO_BT_LOWRESERVE = 0xff00
SYMINFO_FLG_DIRECT = 0x0001
SYMINFO_FLG_PASSTHRU = 0x0002
SYMINFO_FLG_COPY = 0x0004
SYMINFO_FLG_LAZYLOAD = 0x0008
SYMINFO_NONE = 0
SYMINFO_CURRENT = 1
SYMINFO_NUM = 2
ELF32_ST_BIND = lambda val: (( (val)) >> 4)
ELF32_ST_TYPE = lambda val: ((val) & 0xf)
ELF32_ST_INFO = lambda bind,type: (((bind) << 4) + ((type) & 0xf))
ELF64_ST_BIND = lambda val: ELF32_ST_BIND (val)
ELF64_ST_TYPE = lambda val: ELF32_ST_TYPE (val)
ELF64_ST_INFO = lambda bind,type: ELF32_ST_INFO ((bind), (type))
STB_LOCAL = 0
STB_GLOBAL = 1
STB_WEAK = 2
STB_NUM = 3
STB_LOOS = 10
STB_GNU_UNIQUE = 10
STB_HIOS = 12
STB_LOPROC = 13
STB_HIPROC = 15
STT_NOTYPE = 0
STT_OBJECT = 1
STT_FUNC = 2
STT_SECTION = 3
STT_FILE = 4
STT_COMMON = 5
STT_TLS = 6
STT_NUM = 7
STT_LOOS = 10
STT_GNU_IFUNC = 10
STT_HIOS = 12
STT_LOPROC = 13
STT_HIPROC = 15
STN_UNDEF = 0
ELF32_ST_VISIBILITY = lambda o: ((o) & 0x03)
ELF64_ST_VISIBILITY = lambda o: ELF32_ST_VISIBILITY (o)
STV_DEFAULT = 0
STV_INTERNAL = 1
STV_HIDDEN = 2
STV_PROTECTED = 3
ELF32_R_SYM = lambda val: ((val) >> 8)
ELF32_R_TYPE = lambda val: ((val) & 0xff)
ELF32_R_INFO = lambda sym,type: (((sym) << 8) + ((type) & 0xff))
ELF64_R_SYM = lambda i: ((i) >> 32)
ELF64_R_TYPE = lambda i: ((i) & 0xffffffff)
ELF64_R_INFO = lambda sym,type: ((((Elf64_Xword) (sym)) << 32) + (type))
PN_XNUM = 0xffff
PT_NULL = 0
PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_NOTE = 4
PT_SHLIB = 5
PT_PHDR = 6
PT_TLS = 7
PT_NUM = 8
PT_LOOS = 0x60000000
PT_GNU_EH_FRAME = 0x6474e550
PT_GNU_STACK = 0x6474e551
PT_GNU_RELRO = 0x6474e552
PT_GNU_PROPERTY = 0x6474e553
PT_GNU_SFRAME = 0x6474e554
PT_LOSUNW = 0x6ffffffa
PT_SUNWBSS = 0x6ffffffa
PT_SUNWSTACK = 0x6ffffffb
PT_HISUNW = 0x6fffffff
PT_HIOS = 0x6fffffff
PT_LOPROC = 0x70000000
PT_HIPROC = 0x7fffffff
PF_X = (1 << 0)
PF_W = (1 << 1)
PF_R = (1 << 2)
PF_MASKOS = 0x0ff00000
PF_MASKPROC = 0xf0000000
NT_PRSTATUS = 1
NT_PRFPREG = 2
NT_FPREGSET = 2
NT_PRPSINFO = 3
NT_PRXREG = 4
NT_TASKSTRUCT = 4
NT_PLATFORM = 5
NT_AUXV = 6
NT_GWINDOWS = 7
NT_ASRS = 8
NT_PSTATUS = 10
NT_PSINFO = 13
NT_PRCRED = 14
NT_UTSNAME = 15
NT_LWPSTATUS = 16
NT_LWPSINFO = 17
NT_PRFPXREG = 20
NT_SIGINFO = 0x53494749
NT_FILE = 0x46494c45
NT_PRXFPREG = 0x46e62b7f
NT_PPC_VMX = 0x100
NT_PPC_SPE = 0x101
NT_PPC_VSX = 0x102
NT_PPC_TAR = 0x103
NT_PPC_PPR = 0x104
NT_PPC_DSCR = 0x105
NT_PPC_EBB = 0x106
NT_PPC_PMU = 0x107
NT_PPC_TM_CGPR = 0x108
NT_PPC_TM_CFPR = 0x109
NT_PPC_TM_CVMX = 0x10a
NT_PPC_TM_CVSX = 0x10b
NT_PPC_TM_SPR = 0x10c
NT_PPC_TM_CTAR = 0x10d
NT_PPC_TM_CPPR = 0x10e
NT_PPC_TM_CDSCR = 0x10f
NT_PPC_PKEY = 0x110
NT_PPC_DEXCR = 0x111
NT_PPC_HASHKEYR = 0x112
NT_386_TLS = 0x200
NT_386_IOPERM = 0x201
NT_X86_XSTATE = 0x202
NT_X86_SHSTK = 0x204
NT_S390_HIGH_GPRS = 0x300
NT_S390_TIMER = 0x301
NT_S390_TODCMP = 0x302
NT_S390_TODPREG = 0x303
NT_S390_CTRS = 0x304
NT_S390_PREFIX = 0x305
NT_S390_LAST_BREAK = 0x306
NT_S390_SYSTEM_CALL = 0x307
NT_S390_TDB = 0x308
NT_S390_VXRS_LOW = 0x309
NT_S390_VXRS_HIGH = 0x30a
NT_S390_GS_CB = 0x30b
NT_S390_GS_BC = 0x30c
NT_S390_RI_CB = 0x30d
NT_S390_PV_CPU_DATA = 0x30e
NT_ARM_VFP = 0x400
NT_ARM_TLS = 0x401
NT_ARM_HW_BREAK = 0x402
NT_ARM_HW_WATCH = 0x403
NT_ARM_SYSTEM_CALL = 0x404
NT_ARM_SVE = 0x405
NT_ARM_PAC_MASK = 0x406
NT_ARM_PACA_KEYS = 0x407
NT_ARM_PACG_KEYS = 0x408
NT_ARM_TAGGED_ADDR_CTRL = 0x409
NT_ARM_PAC_ENABLED_KEYS = 0x40a
NT_VMCOREDD = 0x700
NT_MIPS_DSP = 0x800
NT_MIPS_FP_MODE = 0x801
NT_MIPS_MSA = 0x802
NT_RISCV_CSR = 0x900
NT_RISCV_VECTOR = 0x901
NT_LOONGARCH_CPUCFG = 0xa00
NT_LOONGARCH_CSR = 0xa01
NT_LOONGARCH_LSX = 0xa02
NT_LOONGARCH_LASX = 0xa03
NT_LOONGARCH_LBT = 0xa04
NT_LOONGARCH_HW_BREAK = 0xa05
NT_LOONGARCH_HW_WATCH = 0xa06
NT_VERSION = 1
DT_NULL = 0
DT_NEEDED = 1
DT_PLTRELSZ = 2
DT_PLTGOT = 3
DT_HASH = 4
DT_STRTAB = 5
DT_SYMTAB = 6
DT_RELA = 7
DT_RELASZ = 8
DT_RELAENT = 9
DT_STRSZ = 10
DT_SYMENT = 11
DT_INIT = 12
DT_FINI = 13
DT_SONAME = 14
DT_RPATH = 15
DT_SYMBOLIC = 16
DT_REL = 17
DT_RELSZ = 18
DT_RELENT = 19
DT_PLTREL = 20
DT_DEBUG = 21
DT_TEXTREL = 22
DT_JMPREL = 23
DT_BIND_NOW = 24
DT_INIT_ARRAY = 25
DT_FINI_ARRAY = 26
DT_INIT_ARRAYSZ = 27
DT_FINI_ARRAYSZ = 28
DT_RUNPATH = 29
DT_FLAGS = 30
DT_ENCODING = 32
DT_PREINIT_ARRAY = 32
DT_PREINIT_ARRAYSZ = 33
DT_SYMTAB_SHNDX = 34
DT_RELRSZ = 35
DT_RELR = 36
DT_RELRENT = 37
DT_NUM = 38
DT_LOOS = 0x6000000d
DT_HIOS = 0x6ffff000
DT_LOPROC = 0x70000000
DT_HIPROC = 0x7fffffff
DT_VALRNGLO = 0x6ffffd00
DT_GNU_PRELINKED = 0x6ffffdf5
DT_GNU_CONFLICTSZ = 0x6ffffdf6
DT_GNU_LIBLISTSZ = 0x6ffffdf7
DT_CHECKSUM = 0x6ffffdf8
DT_PLTPADSZ = 0x6ffffdf9
DT_MOVEENT = 0x6ffffdfa
DT_MOVESZ = 0x6ffffdfb
DT_FEATURE_1 = 0x6ffffdfc
DT_POSFLAG_1 = 0x6ffffdfd
DT_SYMINSZ = 0x6ffffdfe
DT_SYMINENT = 0x6ffffdff
DT_VALRNGHI = 0x6ffffdff
DT_VALTAGIDX = lambda tag: (DT_VALRNGHI - (tag))
DT_VALNUM = 12
DT_ADDRRNGLO = 0x6ffffe00
DT_GNU_HASH = 0x6ffffef5
DT_TLSDESC_PLT = 0x6ffffef6
DT_TLSDESC_GOT = 0x6ffffef7
DT_GNU_CONFLICT = 0x6ffffef8
DT_GNU_LIBLIST = 0x6ffffef9
DT_CONFIG = 0x6ffffefa
DT_DEPAUDIT = 0x6ffffefb
DT_AUDIT = 0x6ffffefc
DT_PLTPAD = 0x6ffffefd
DT_MOVETAB = 0x6ffffefe
DT_SYMINFO = 0x6ffffeff
DT_ADDRRNGHI = 0x6ffffeff
DT_ADDRTAGIDX = lambda tag: (DT_ADDRRNGHI - (tag))
DT_ADDRNUM = 11
DT_VERSYM = 0x6ffffff0
DT_RELACOUNT = 0x6ffffff9
DT_RELCOUNT = 0x6ffffffa
DT_FLAGS_1 = 0x6ffffffb
DT_VERDEF = 0x6ffffffc
DT_VERDEFNUM = 0x6ffffffd
DT_VERNEED = 0x6ffffffe
DT_VERNEEDNUM = 0x6fffffff
DT_VERSIONTAGIDX = lambda tag: (DT_VERNEEDNUM - (tag))
DT_VERSIONTAGNUM = 16
DT_AUXILIARY = 0x7ffffffd
DT_FILTER = 0x7fffffff
DT_EXTRATAGIDX = lambda tag: ((Elf32_Word)-((Elf32_Sword) (tag) <<1>>1)-1)
DT_EXTRANUM = 3
DF_ORIGIN = 0x00000001
DF_SYMBOLIC = 0x00000002
DF_TEXTREL = 0x00000004
DF_BIND_NOW = 0x00000008
DF_STATIC_TLS = 0x00000010
DF_1_NOW = 0x00000001
DF_1_GLOBAL = 0x00000002
DF_1_GROUP = 0x00000004
DF_1_NODELETE = 0x00000008
DF_1_LOADFLTR = 0x00000010
DF_1_INITFIRST = 0x00000020
DF_1_NOOPEN = 0x00000040
DF_1_ORIGIN = 0x00000080
DF_1_DIRECT = 0x00000100
DF_1_TRANS = 0x00000200
DF_1_INTERPOSE = 0x00000400
DF_1_NODEFLIB = 0x00000800
DF_1_NODUMP = 0x00001000
DF_1_CONFALT = 0x00002000
DF_1_ENDFILTEE = 0x00004000
DF_1_DISPRELDNE = 0x00008000
DF_1_DISPRELPND = 0x00010000
DF_1_NODIRECT = 0x00020000
DF_1_IGNMULDEF = 0x00040000
DF_1_NOKSYMS = 0x00080000
DF_1_NOHDR = 0x00100000
DF_1_EDITED = 0x00200000
DF_1_NORELOC = 0x00400000
DF_1_SYMINTPOSE = 0x00800000
DF_1_GLOBAUDIT = 0x01000000
DF_1_SINGLETON = 0x02000000
DF_1_STUB = 0x04000000
DF_1_PIE = 0x08000000
DF_1_KMOD = 0x10000000
DF_1_WEAKFILTER = 0x20000000
DF_1_NOCOMMON = 0x40000000
DTF_1_PARINIT = 0x00000001
DTF_1_CONFEXP = 0x00000002
DF_P1_LAZYLOAD = 0x00000001
DF_P1_GROUPPERM = 0x00000002
VER_DEF_NONE = 0
VER_DEF_CURRENT = 1
VER_DEF_NUM = 2
VER_FLG_BASE = 0x1
VER_FLG_WEAK = 0x2
VER_NDX_LOCAL = 0
VER_NDX_GLOBAL = 1
VER_NDX_LORESERVE = 0xff00
VER_NDX_ELIMINATE = 0xff01
VER_NEED_NONE = 0
VER_NEED_CURRENT = 1
VER_NEED_NUM = 2
AT_NULL = 0
AT_IGNORE = 1
AT_EXECFD = 2
AT_PHDR = 3
AT_PHENT = 4
AT_PHNUM = 5
AT_PAGESZ = 6
AT_BASE = 7
AT_FLAGS = 8
AT_ENTRY = 9
AT_NOTELF = 10
AT_UID = 11
AT_EUID = 12
AT_GID = 13
AT_EGID = 14
AT_CLKTCK = 17
AT_PLATFORM = 15
AT_HWCAP = 16
AT_FPUCW = 18
AT_DCACHEBSIZE = 19
AT_ICACHEBSIZE = 20
AT_UCACHEBSIZE = 21
AT_IGNOREPPC = 22
AT_SECURE = 23
AT_BASE_PLATFORM = 24
AT_RANDOM = 25
AT_HWCAP2 = 26
AT_RSEQ_FEATURE_SIZE = 27
AT_RSEQ_ALIGN = 28
AT_HWCAP3 = 29
AT_HWCAP4 = 30
AT_EXECFN = 31
AT_SYSINFO = 32
AT_SYSINFO_EHDR = 33
AT_L1I_CACHESHAPE = 34
AT_L1D_CACHESHAPE = 35
AT_L2_CACHESHAPE = 36
AT_L3_CACHESHAPE = 37
AT_L1I_CACHESIZE = 40
AT_L1I_CACHEGEOMETRY = 41
AT_L1D_CACHESIZE = 42
AT_L1D_CACHEGEOMETRY = 43
AT_L2_CACHESIZE = 44
AT_L2_CACHEGEOMETRY = 45
AT_L3_CACHESIZE = 46
AT_L3_CACHEGEOMETRY = 47
AT_MINSIGSTKSZ = 51
ELF_NOTE_SOLARIS = "SUNW Solaris"
ELF_NOTE_GNU = "GNU"
ELF_NOTE_FDO = "FDO"
ELF_NOTE_PAGESIZE_HINT = 1
NT_GNU_ABI_TAG = 1
ELF_NOTE_ABI = NT_GNU_ABI_TAG
ELF_NOTE_OS_LINUX = 0
ELF_NOTE_OS_GNU = 1
ELF_NOTE_OS_SOLARIS2 = 2
ELF_NOTE_OS_FREEBSD = 3
NT_GNU_HWCAP = 2
NT_GNU_BUILD_ID = 3
NT_GNU_GOLD_VERSION = 4
NT_GNU_PROPERTY_TYPE_0 = 5
NT_FDO_PACKAGING_METADATA = 0xcafe1a7e
NOTE_GNU_PROPERTY_SECTION_NAME = ".note.gnu.property"
GNU_PROPERTY_STACK_SIZE = 1
GNU_PROPERTY_NO_COPY_ON_PROTECTED = 2
GNU_PROPERTY_UINT32_AND_LO = 0xb0000000
GNU_PROPERTY_UINT32_AND_HI = 0xb0007fff
GNU_PROPERTY_UINT32_OR_LO = 0xb0008000
GNU_PROPERTY_UINT32_OR_HI = 0xb000ffff
GNU_PROPERTY_1_NEEDED = GNU_PROPERTY_UINT32_OR_LO
GNU_PROPERTY_1_NEEDED_INDIRECT_EXTERN_ACCESS = (1 << 0)
GNU_PROPERTY_LOPROC = 0xc0000000
GNU_PROPERTY_HIPROC = 0xdfffffff
GNU_PROPERTY_LOUSER = 0xe0000000
GNU_PROPERTY_HIUSER = 0xffffffff
GNU_PROPERTY_AARCH64_FEATURE_1_AND = 0xc0000000
GNU_PROPERTY_AARCH64_FEATURE_1_BTI = (1 << 0)
GNU_PROPERTY_AARCH64_FEATURE_1_PAC = (1 << 1)
GNU_PROPERTY_X86_ISA_1_USED = 0xc0010002
GNU_PROPERTY_X86_ISA_1_NEEDED = 0xc0008002
GNU_PROPERTY_X86_FEATURE_1_AND = 0xc0000002
GNU_PROPERTY_X86_ISA_1_BASELINE = (1 << 0)
GNU_PROPERTY_X86_ISA_1_V2 = (1 << 1)
GNU_PROPERTY_X86_ISA_1_V3 = (1 << 2)
GNU_PROPERTY_X86_ISA_1_V4 = (1 << 3)
GNU_PROPERTY_X86_FEATURE_1_IBT = (1 << 0)
GNU_PROPERTY_X86_FEATURE_1_SHSTK = (1 << 1)
ELF32_M_SYM = lambda info: ((info) >> 8)
ELF32_M_SIZE = lambda info: ( (info))
ELF32_M_INFO = lambda sym,size: (((sym) << 8) +  (size))
ELF64_M_SYM = lambda info: ELF32_M_SYM (info)
ELF64_M_SIZE = lambda info: ELF32_M_SIZE (info)
ELF64_M_INFO = lambda sym,size: ELF32_M_INFO (sym, size)
EF_CPU32 = 0x00810000
R_68K_NONE = 0
R_68K_32 = 1
R_68K_16 = 2
R_68K_8 = 3
R_68K_PC32 = 4
R_68K_PC16 = 5
R_68K_PC8 = 6
R_68K_GOT32 = 7
R_68K_GOT16 = 8
R_68K_GOT8 = 9
R_68K_GOT32O = 10
R_68K_GOT16O = 11
R_68K_GOT8O = 12
R_68K_PLT32 = 13
R_68K_PLT16 = 14
R_68K_PLT8 = 15
R_68K_PLT32O = 16
R_68K_PLT16O = 17
R_68K_PLT8O = 18
R_68K_COPY = 19
R_68K_GLOB_DAT = 20
R_68K_JMP_SLOT = 21
R_68K_RELATIVE = 22
R_68K_TLS_GD32 = 25
R_68K_TLS_GD16 = 26
R_68K_TLS_GD8 = 27
R_68K_TLS_LDM32 = 28
R_68K_TLS_LDM16 = 29
R_68K_TLS_LDM8 = 30
R_68K_TLS_LDO32 = 31
R_68K_TLS_LDO16 = 32
R_68K_TLS_LDO8 = 33
R_68K_TLS_IE32 = 34
R_68K_TLS_IE16 = 35
R_68K_TLS_IE8 = 36
R_68K_TLS_LE32 = 37
R_68K_TLS_LE16 = 38
R_68K_TLS_LE8 = 39
R_68K_TLS_DTPMOD32 = 40
R_68K_TLS_DTPREL32 = 41
R_68K_TLS_TPREL32 = 42
R_68K_NUM = 43
R_386_NONE = 0
R_386_32 = 1
R_386_PC32 = 2
R_386_GOT32 = 3
R_386_PLT32 = 4
R_386_COPY = 5
R_386_GLOB_DAT = 6
R_386_JMP_SLOT = 7
R_386_RELATIVE = 8
R_386_GOTOFF = 9
R_386_GOTPC = 10
R_386_32PLT = 11
R_386_TLS_TPOFF = 14
R_386_TLS_IE = 15
R_386_TLS_GOTIE = 16
R_386_TLS_LE = 17
R_386_TLS_GD = 18
R_386_TLS_LDM = 19
R_386_16 = 20
R_386_PC16 = 21
R_386_8 = 22
R_386_PC8 = 23
R_386_TLS_GD_32 = 24
R_386_TLS_GD_PUSH = 25
R_386_TLS_GD_CALL = 26
R_386_TLS_GD_POP = 27
R_386_TLS_LDM_32 = 28
R_386_TLS_LDM_PUSH = 29
R_386_TLS_LDM_CALL = 30
R_386_TLS_LDM_POP = 31
R_386_TLS_LDO_32 = 32
R_386_TLS_IE_32 = 33
R_386_TLS_LE_32 = 34
R_386_TLS_DTPMOD32 = 35
R_386_TLS_DTPOFF32 = 36
R_386_TLS_TPOFF32 = 37
R_386_SIZE32 = 38
R_386_TLS_GOTDESC = 39
R_386_TLS_DESC_CALL = 40
R_386_TLS_DESC = 41
R_386_IRELATIVE = 42
R_386_GOT32X = 43
R_386_NUM = 44
STT_SPARC_REGISTER = 13
EF_SPARCV9_MM = 3
EF_SPARCV9_TSO = 0
EF_SPARCV9_PSO = 1
EF_SPARCV9_RMO = 2
EF_SPARC_LEDATA = 0x800000
EF_SPARC_EXT_MASK = 0xFFFF00
EF_SPARC_32PLUS = 0x000100
EF_SPARC_SUN_US1 = 0x000200
EF_SPARC_HAL_R1 = 0x000400
EF_SPARC_SUN_US3 = 0x000800
R_SPARC_NONE = 0
R_SPARC_8 = 1
R_SPARC_16 = 2
R_SPARC_32 = 3
R_SPARC_DISP8 = 4
R_SPARC_DISP16 = 5
R_SPARC_DISP32 = 6
R_SPARC_WDISP30 = 7
R_SPARC_WDISP22 = 8
R_SPARC_HI22 = 9
R_SPARC_22 = 10
R_SPARC_13 = 11
R_SPARC_LO10 = 12
R_SPARC_GOT10 = 13
R_SPARC_GOT13 = 14
R_SPARC_GOT22 = 15
R_SPARC_PC10 = 16
R_SPARC_PC22 = 17
R_SPARC_WPLT30 = 18
R_SPARC_COPY = 19
R_SPARC_GLOB_DAT = 20
R_SPARC_JMP_SLOT = 21
R_SPARC_RELATIVE = 22
R_SPARC_UA32 = 23
R_SPARC_PLT32 = 24
R_SPARC_HIPLT22 = 25
R_SPARC_LOPLT10 = 26
R_SPARC_PCPLT32 = 27
R_SPARC_PCPLT22 = 28
R_SPARC_PCPLT10 = 29
R_SPARC_10 = 30
R_SPARC_11 = 31
R_SPARC_64 = 32
R_SPARC_OLO10 = 33
R_SPARC_HH22 = 34
R_SPARC_HM10 = 35
R_SPARC_LM22 = 36
R_SPARC_PC_HH22 = 37
R_SPARC_PC_HM10 = 38
R_SPARC_PC_LM22 = 39
R_SPARC_WDISP16 = 40
R_SPARC_WDISP19 = 41
R_SPARC_GLOB_JMP = 42
R_SPARC_7 = 43
R_SPARC_5 = 44
R_SPARC_6 = 45
R_SPARC_DISP64 = 46
R_SPARC_PLT64 = 47
R_SPARC_HIX22 = 48
R_SPARC_LOX10 = 49
R_SPARC_H44 = 50
R_SPARC_M44 = 51
R_SPARC_L44 = 52
R_SPARC_REGISTER = 53
R_SPARC_UA64 = 54
R_SPARC_UA16 = 55
R_SPARC_TLS_GD_HI22 = 56
R_SPARC_TLS_GD_LO10 = 57
R_SPARC_TLS_GD_ADD = 58
R_SPARC_TLS_GD_CALL = 59
R_SPARC_TLS_LDM_HI22 = 60
R_SPARC_TLS_LDM_LO10 = 61
R_SPARC_TLS_LDM_ADD = 62
R_SPARC_TLS_LDM_CALL = 63
R_SPARC_TLS_LDO_HIX22 = 64
R_SPARC_TLS_LDO_LOX10 = 65
R_SPARC_TLS_LDO_ADD = 66
R_SPARC_TLS_IE_HI22 = 67
R_SPARC_TLS_IE_LO10 = 68
R_SPARC_TLS_IE_LD = 69
R_SPARC_TLS_IE_LDX = 70
R_SPARC_TLS_IE_ADD = 71
R_SPARC_TLS_LE_HIX22 = 72
R_SPARC_TLS_LE_LOX10 = 73
R_SPARC_TLS_DTPMOD32 = 74
R_SPARC_TLS_DTPMOD64 = 75
R_SPARC_TLS_DTPOFF32 = 76
R_SPARC_TLS_DTPOFF64 = 77
R_SPARC_TLS_TPOFF32 = 78
R_SPARC_TLS_TPOFF64 = 79
R_SPARC_GOTDATA_HIX22 = 80
R_SPARC_GOTDATA_LOX10 = 81
R_SPARC_GOTDATA_OP_HIX22 = 82
R_SPARC_GOTDATA_OP_LOX10 = 83
R_SPARC_GOTDATA_OP = 84
R_SPARC_H34 = 85
R_SPARC_SIZE32 = 86
R_SPARC_SIZE64 = 87
R_SPARC_WDISP10 = 88
R_SPARC_JMP_IREL = 248
R_SPARC_IRELATIVE = 249
R_SPARC_GNU_VTINHERIT = 250
R_SPARC_GNU_VTENTRY = 251
R_SPARC_REV32 = 252
R_SPARC_NUM = 253
DT_SPARC_REGISTER = 0x70000001
DT_SPARC_NUM = 2
EF_MIPS_NOREORDER = 1
EF_MIPS_PIC = 2
EF_MIPS_CPIC = 4
EF_MIPS_XGOT = 8
EF_MIPS_UCODE = 16
EF_MIPS_ABI2 = 32
EF_MIPS_ABI_ON32 = 64
EF_MIPS_OPTIONS_FIRST = 0x00000080
EF_MIPS_32BITMODE = 0x00000100
EF_MIPS_FP64 = 512
EF_MIPS_NAN2008 = 1024
EF_MIPS_ARCH_ASE = 0x0f000000
EF_MIPS_ARCH_ASE_MDMX = 0x08000000
EF_MIPS_ARCH_ASE_M16 = 0x04000000
EF_MIPS_ARCH_ASE_MICROMIPS = 0x02000000
EF_MIPS_ARCH = 0xf0000000
EF_MIPS_ARCH_1 = 0x00000000
EF_MIPS_ARCH_2 = 0x10000000
EF_MIPS_ARCH_3 = 0x20000000
EF_MIPS_ARCH_4 = 0x30000000
EF_MIPS_ARCH_5 = 0x40000000
EF_MIPS_ARCH_32 = 0x50000000
EF_MIPS_ARCH_64 = 0x60000000
EF_MIPS_ARCH_32R2 = 0x70000000
EF_MIPS_ARCH_64R2 = 0x80000000
EF_MIPS_ARCH_32R6 = 0x90000000
EF_MIPS_ARCH_64R6 = 0xa0000000
EF_MIPS_ABI = 0x0000F000
EF_MIPS_ABI_O32 = 0x00001000
EF_MIPS_ABI_O64 = 0x00002000
EF_MIPS_ABI_EABI32 = 0x00003000
EF_MIPS_ABI_EABI64 = 0x00004000
EF_MIPS_MACH = 0x00FF0000
EF_MIPS_MACH_3900 = 0x00810000
EF_MIPS_MACH_4010 = 0x00820000
EF_MIPS_MACH_4100 = 0x00830000
EF_MIPS_MACH_ALLEGREX = 0x00840000
EF_MIPS_MACH_4650 = 0x00850000
EF_MIPS_MACH_4120 = 0x00870000
EF_MIPS_MACH_4111 = 0x00880000
EF_MIPS_MACH_SB1 = 0x008a0000
EF_MIPS_MACH_OCTEON = 0x008b0000
EF_MIPS_MACH_XLR = 0x008c0000
EF_MIPS_MACH_OCTEON2 = 0x008d0000
EF_MIPS_MACH_OCTEON3 = 0x008e0000
EF_MIPS_MACH_5400 = 0x00910000
EF_MIPS_MACH_5900 = 0x00920000
EF_MIPS_MACH_IAMR2 = 0x00930000
EF_MIPS_MACH_5500 = 0x00980000
EF_MIPS_MACH_9000 = 0x00990000
EF_MIPS_MACH_LS2E = 0x00A00000
EF_MIPS_MACH_LS2F = 0x00A10000
EF_MIPS_MACH_GS464 = 0x00A20000
EF_MIPS_MACH_GS464E = 0x00A30000
EF_MIPS_MACH_GS264E = 0x00A40000
E_MIPS_ARCH_1 = EF_MIPS_ARCH_1
E_MIPS_ARCH_2 = EF_MIPS_ARCH_2
E_MIPS_ARCH_3 = EF_MIPS_ARCH_3
E_MIPS_ARCH_4 = EF_MIPS_ARCH_4
E_MIPS_ARCH_5 = EF_MIPS_ARCH_5
E_MIPS_ARCH_32 = EF_MIPS_ARCH_32
E_MIPS_ARCH_64 = EF_MIPS_ARCH_64
SHN_MIPS_ACOMMON = 0xff00
SHN_MIPS_TEXT = 0xff01
SHN_MIPS_DATA = 0xff02
SHN_MIPS_SCOMMON = 0xff03
SHN_MIPS_SUNDEFINED = 0xff04
SHT_MIPS_LIBLIST = 0x70000000
SHT_MIPS_MSYM = 0x70000001
SHT_MIPS_CONFLICT = 0x70000002
SHT_MIPS_GPTAB = 0x70000003
SHT_MIPS_UCODE = 0x70000004
SHT_MIPS_DEBUG = 0x70000005
SHT_MIPS_REGINFO = 0x70000006
SHT_MIPS_PACKAGE = 0x70000007
SHT_MIPS_PACKSYM = 0x70000008
SHT_MIPS_RELD = 0x70000009
SHT_MIPS_IFACE = 0x7000000b
SHT_MIPS_CONTENT = 0x7000000c
SHT_MIPS_OPTIONS = 0x7000000d
SHT_MIPS_SHDR = 0x70000010
SHT_MIPS_FDESC = 0x70000011
SHT_MIPS_EXTSYM = 0x70000012
SHT_MIPS_DENSE = 0x70000013
SHT_MIPS_PDESC = 0x70000014
SHT_MIPS_LOCSYM = 0x70000015
SHT_MIPS_AUXSYM = 0x70000016
SHT_MIPS_OPTSYM = 0x70000017
SHT_MIPS_LOCSTR = 0x70000018
SHT_MIPS_LINE = 0x70000019
SHT_MIPS_RFDESC = 0x7000001a
SHT_MIPS_DELTASYM = 0x7000001b
SHT_MIPS_DELTAINST = 0x7000001c
SHT_MIPS_DELTACLASS = 0x7000001d
SHT_MIPS_DWARF = 0x7000001e
SHT_MIPS_DELTADECL = 0x7000001f
SHT_MIPS_SYMBOL_LIB = 0x70000020
SHT_MIPS_EVENTS = 0x70000021
SHT_MIPS_TRANSLATE = 0x70000022
SHT_MIPS_PIXIE = 0x70000023
SHT_MIPS_XLATE = 0x70000024
SHT_MIPS_XLATE_DEBUG = 0x70000025
SHT_MIPS_WHIRL = 0x70000026
SHT_MIPS_EH_REGION = 0x70000027
SHT_MIPS_XLATE_OLD = 0x70000028
SHT_MIPS_PDR_EXCEPTION = 0x70000029
SHT_MIPS_ABIFLAGS = 0x7000002a
SHT_MIPS_XHASH = 0x7000002b
SHF_MIPS_GPREL = 0x10000000
SHF_MIPS_MERGE = 0x20000000
SHF_MIPS_ADDR = 0x40000000
SHF_MIPS_STRINGS = 0x80000000
SHF_MIPS_NOSTRIP = 0x08000000
SHF_MIPS_LOCAL = 0x04000000
SHF_MIPS_NAMES = 0x02000000
SHF_MIPS_NODUPE = 0x01000000
STO_MIPS_DEFAULT = 0x0
STO_MIPS_INTERNAL = 0x1
STO_MIPS_HIDDEN = 0x2
STO_MIPS_PROTECTED = 0x3
STO_MIPS_PLT = 0x8
STO_MIPS_SC_ALIGN_UNUSED = 0xff
STB_MIPS_SPLIT_COMMON = 13
ODK_NULL = 0
ODK_REGINFO = 1
ODK_EXCEPTIONS = 2
ODK_PAD = 3
ODK_HWPATCH = 4
ODK_FILL = 5
ODK_TAGS = 6
ODK_HWAND = 7
ODK_HWOR = 8
OEX_FPU_MIN = 0x1f
OEX_FPU_MAX = 0x1f00
OEX_PAGE0 = 0x10000
OEX_SMM = 0x20000
OEX_FPDBUG = 0x40000
OEX_PRECISEFP = OEX_FPDBUG
OEX_DISMISS = 0x80000
OEX_FPU_INVAL = 0x10
OEX_FPU_DIV0 = 0x08
OEX_FPU_OFLO = 0x04
OEX_FPU_UFLO = 0x02
OEX_FPU_INEX = 0x01
OHW_R4KEOP = 0x1
OHW_R8KPFETCH = 0x2
OHW_R5KEOP = 0x4
OHW_R5KCVTL = 0x8
OPAD_PREFIX = 0x1
OPAD_POSTFIX = 0x2
OPAD_SYMBOL = 0x4
OHWA0_R4KEOP_CHECKED = 0x00000001
OHWA1_R4KEOP_CLEAN = 0x00000002
R_MIPS_NONE = 0
R_MIPS_16 = 1
R_MIPS_32 = 2
R_MIPS_REL32 = 3
R_MIPS_26 = 4
R_MIPS_HI16 = 5
R_MIPS_LO16 = 6
R_MIPS_GPREL16 = 7
R_MIPS_LITERAL = 8
R_MIPS_GOT16 = 9
R_MIPS_PC16 = 10
R_MIPS_CALL16 = 11
R_MIPS_GPREL32 = 12
R_MIPS_SHIFT5 = 16
R_MIPS_SHIFT6 = 17
R_MIPS_64 = 18
R_MIPS_GOT_DISP = 19
R_MIPS_GOT_PAGE = 20
R_MIPS_GOT_OFST = 21
R_MIPS_GOT_HI16 = 22
R_MIPS_GOT_LO16 = 23
R_MIPS_SUB = 24
R_MIPS_INSERT_A = 25
R_MIPS_INSERT_B = 26
R_MIPS_DELETE = 27
R_MIPS_HIGHER = 28
R_MIPS_HIGHEST = 29
R_MIPS_CALL_HI16 = 30
R_MIPS_CALL_LO16 = 31
R_MIPS_SCN_DISP = 32
R_MIPS_REL16 = 33
R_MIPS_ADD_IMMEDIATE = 34
R_MIPS_PJUMP = 35
R_MIPS_RELGOT = 36
R_MIPS_JALR = 37
R_MIPS_TLS_DTPMOD32 = 38
R_MIPS_TLS_DTPREL32 = 39
R_MIPS_TLS_DTPMOD64 = 40
R_MIPS_TLS_DTPREL64 = 41
R_MIPS_TLS_GD = 42
R_MIPS_TLS_LDM = 43
R_MIPS_TLS_DTPREL_HI16 = 44
R_MIPS_TLS_DTPREL_LO16 = 45
R_MIPS_TLS_GOTTPREL = 46
R_MIPS_TLS_TPREL32 = 47
R_MIPS_TLS_TPREL64 = 48
R_MIPS_TLS_TPREL_HI16 = 49
R_MIPS_TLS_TPREL_LO16 = 50
R_MIPS_GLOB_DAT = 51
R_MIPS_PC21_S2 = 60
R_MIPS_PC26_S2 = 61
R_MIPS_PC18_S3 = 62
R_MIPS_PC19_S2 = 63
R_MIPS_PCHI16 = 64
R_MIPS_PCLO16 = 65
R_MIPS16_26 = 100
R_MIPS16_GPREL = 101
R_MIPS16_GOT16 = 102
R_MIPS16_CALL16 = 103
R_MIPS16_HI16 = 104
R_MIPS16_LO16 = 105
R_MIPS16_TLS_GD = 106
R_MIPS16_TLS_LDM = 107
R_MIPS16_TLS_DTPREL_HI16 = 108
R_MIPS16_TLS_DTPREL_LO16 = 109
R_MIPS16_TLS_GOTTPREL = 110
R_MIPS16_TLS_TPREL_HI16 = 111
R_MIPS16_TLS_TPREL_LO16 = 112
R_MIPS16_PC16_S1 = 113
R_MIPS_COPY = 126
R_MIPS_JUMP_SLOT = 127
R_MIPS_RELATIVE = 128
R_MICROMIPS_26_S1 = 133
R_MICROMIPS_HI16 = 134
R_MICROMIPS_LO16 = 135
R_MICROMIPS_GPREL16 = 136
R_MICROMIPS_LITERAL = 137
R_MICROMIPS_GOT16 = 138
R_MICROMIPS_PC7_S1 = 139
R_MICROMIPS_PC10_S1 = 140
R_MICROMIPS_PC16_S1 = 141
R_MICROMIPS_CALL16 = 142
R_MICROMIPS_GOT_DISP = 145
R_MICROMIPS_GOT_PAGE = 146
R_MICROMIPS_GOT_OFST = 147
R_MICROMIPS_GOT_HI16 = 148
R_MICROMIPS_GOT_LO16 = 149
R_MICROMIPS_SUB = 150
R_MICROMIPS_HIGHER = 151
R_MICROMIPS_HIGHEST = 152
R_MICROMIPS_CALL_HI16 = 153
R_MICROMIPS_CALL_LO16 = 154
R_MICROMIPS_SCN_DISP = 155
R_MICROMIPS_JALR = 156
R_MICROMIPS_HI0_LO16 = 157
R_MICROMIPS_TLS_GD = 162
R_MICROMIPS_TLS_LDM = 163
R_MICROMIPS_TLS_DTPREL_HI16 = 164
R_MICROMIPS_TLS_DTPREL_LO16 = 165
R_MICROMIPS_TLS_GOTTPREL = 166
R_MICROMIPS_TLS_TPREL_HI16 = 169
R_MICROMIPS_TLS_TPREL_LO16 = 170
R_MICROMIPS_GPREL7_S2 = 172
R_MICROMIPS_PC23_S2 = 173
R_MIPS_PC32 = 248
R_MIPS_EH = 249
R_MIPS_GNU_REL16_S2 = 250
R_MIPS_GNU_VTINHERIT = 253
R_MIPS_GNU_VTENTRY = 254
R_MIPS_NUM = 255
PT_MIPS_REGINFO = 0x70000000
PT_MIPS_RTPROC = 0x70000001
PT_MIPS_OPTIONS = 0x70000002
PT_MIPS_ABIFLAGS = 0x70000003
PF_MIPS_LOCAL = 0x10000000
DT_MIPS_RLD_VERSION = 0x70000001
DT_MIPS_TIME_STAMP = 0x70000002
DT_MIPS_ICHECKSUM = 0x70000003
DT_MIPS_IVERSION = 0x70000004
DT_MIPS_FLAGS = 0x70000005
DT_MIPS_BASE_ADDRESS = 0x70000006
DT_MIPS_MSYM = 0x70000007
DT_MIPS_CONFLICT = 0x70000008
DT_MIPS_LIBLIST = 0x70000009
DT_MIPS_LOCAL_GOTNO = 0x7000000a
DT_MIPS_CONFLICTNO = 0x7000000b
DT_MIPS_LIBLISTNO = 0x70000010
DT_MIPS_SYMTABNO = 0x70000011
DT_MIPS_UNREFEXTNO = 0x70000012
DT_MIPS_GOTSYM = 0x70000013
DT_MIPS_HIPAGENO = 0x70000014
DT_MIPS_RLD_MAP = 0x70000016
DT_MIPS_DELTA_CLASS = 0x70000017
DT_MIPS_DELTA_CLASS_NO = 0x70000018
DT_MIPS_DELTA_INSTANCE = 0x70000019
DT_MIPS_DELTA_INSTANCE_NO = 0x7000001a
DT_MIPS_DELTA_RELOC = 0x7000001b
DT_MIPS_DELTA_RELOC_NO = 0x7000001c
DT_MIPS_DELTA_SYM = 0x7000001d
DT_MIPS_DELTA_SYM_NO = 0x7000001e
DT_MIPS_DELTA_CLASSSYM = 0x70000020
DT_MIPS_DELTA_CLASSSYM_NO = 0x70000021
DT_MIPS_CXX_FLAGS = 0x70000022
DT_MIPS_PIXIE_INIT = 0x70000023
DT_MIPS_SYMBOL_LIB = 0x70000024
DT_MIPS_LOCALPAGE_GOTIDX = 0x70000025
DT_MIPS_LOCAL_GOTIDX = 0x70000026
DT_MIPS_HIDDEN_GOTIDX = 0x70000027
DT_MIPS_PROTECTED_GOTIDX = 0x70000028
DT_MIPS_OPTIONS = 0x70000029
DT_MIPS_INTERFACE = 0x7000002a
DT_MIPS_DYNSTR_ALIGN = 0x7000002b
DT_MIPS_INTERFACE_SIZE = 0x7000002c
DT_MIPS_RLD_TEXT_RESOLVE_ADDR = 0x7000002d
DT_MIPS_PERF_SUFFIX = 0x7000002e
DT_MIPS_COMPACT_SIZE = 0x7000002f
DT_MIPS_GP_VALUE = 0x70000030
DT_MIPS_AUX_DYNAMIC = 0x70000031
DT_MIPS_PLTGOT = 0x70000032
DT_MIPS_RWPLT = 0x70000034
DT_MIPS_RLD_MAP_REL = 0x70000035
DT_MIPS_XHASH = 0x70000036
DT_MIPS_NUM = 0x37
RHF_NONE = 0
RHF_QUICKSTART = (1 << 0)
RHF_NOTPOT = (1 << 1)
RHF_NO_LIBRARY_REPLACEMENT = (1 << 2)
RHF_NO_MOVE = (1 << 3)
RHF_SGI_ONLY = (1 << 4)
RHF_GUARANTEE_INIT = (1 << 5)
RHF_DELTA_C_PLUS_PLUS = (1 << 6)
RHF_GUARANTEE_START_INIT = (1 << 7)
RHF_PIXIE = (1 << 8)
RHF_DEFAULT_DELAY_LOAD = (1 << 9)
RHF_REQUICKSTART = (1 << 10)
RHF_REQUICKSTARTED = (1 << 11)
RHF_CORD = (1 << 12)
RHF_NO_UNRES_UNDEF = (1 << 13)
RHF_RLD_ORDER_SAFE = (1 << 14)
LL_NONE = 0
LL_EXACT_MATCH = (1 << 0)
LL_IGNORE_INT_VER = (1 << 1)
LL_REQUIRE_MINOR = (1 << 2)
LL_EXPORTS = (1 << 3)
LL_DELAY_LOAD = (1 << 4)
LL_DELTA = (1 << 5)
MIPS_AFL_REG_NONE = 0x00
MIPS_AFL_REG_32 = 0x01
MIPS_AFL_REG_64 = 0x02
MIPS_AFL_REG_128 = 0x03
MIPS_AFL_ASE_DSP = 0x00000001
MIPS_AFL_ASE_DSPR2 = 0x00000002
MIPS_AFL_ASE_EVA = 0x00000004
MIPS_AFL_ASE_MCU = 0x00000008
MIPS_AFL_ASE_MDMX = 0x00000010
MIPS_AFL_ASE_MIPS3D = 0x00000020
MIPS_AFL_ASE_MT = 0x00000040
MIPS_AFL_ASE_SMARTMIPS = 0x00000080
MIPS_AFL_ASE_VIRT = 0x00000100
MIPS_AFL_ASE_MSA = 0x00000200
MIPS_AFL_ASE_MIPS16 = 0x00000400
MIPS_AFL_ASE_MICROMIPS = 0x00000800
MIPS_AFL_ASE_XPA = 0x00001000
MIPS_AFL_ASE_MASK = 0x00001fff
MIPS_AFL_EXT_XLR = 1
MIPS_AFL_EXT_OCTEON2 = 2
MIPS_AFL_EXT_OCTEONP = 3
MIPS_AFL_EXT_LOONGSON_3A = 4
MIPS_AFL_EXT_OCTEON = 5
MIPS_AFL_EXT_5900 = 6
MIPS_AFL_EXT_4650 = 7
MIPS_AFL_EXT_4010 = 8
MIPS_AFL_EXT_4100 = 9
MIPS_AFL_EXT_3900 = 10
MIPS_AFL_EXT_10000 = 11
MIPS_AFL_EXT_SB1 = 12
MIPS_AFL_EXT_4111 = 13
MIPS_AFL_EXT_4120 = 14
MIPS_AFL_EXT_5400 = 15
MIPS_AFL_EXT_5500 = 16
MIPS_AFL_EXT_LOONGSON_2E = 17
MIPS_AFL_EXT_LOONGSON_2F = 18
MIPS_AFL_FLAGS1_ODDSPREG = 1
EF_PARISC_TRAPNIL = 0x00010000
EF_PARISC_EXT = 0x00020000
EF_PARISC_LSB = 0x00040000
EF_PARISC_WIDE = 0x00080000
EF_PARISC_NO_KABP = 0x00100000
EF_PARISC_LAZYSWAP = 0x00400000
EF_PARISC_ARCH = 0x0000ffff
EFA_PARISC_1_0 = 0x020b
EFA_PARISC_1_1 = 0x0210
EFA_PARISC_2_0 = 0x0214
SHN_PARISC_ANSI_COMMON = 0xff00
SHN_PARISC_HUGE_COMMON = 0xff01
SHT_PARISC_EXT = 0x70000000
SHT_PARISC_UNWIND = 0x70000001
SHT_PARISC_DOC = 0x70000002
SHF_PARISC_SHORT = 0x20000000
SHF_PARISC_HUGE = 0x40000000
SHF_PARISC_SBP = 0x80000000
STT_PARISC_MILLICODE = 13
STT_HP_OPAQUE = (STT_LOOS + 0x1)
STT_HP_STUB = (STT_LOOS + 0x2)
R_PARISC_NONE = 0
R_PARISC_DIR32 = 1
R_PARISC_DIR21L = 2
R_PARISC_DIR17R = 3
R_PARISC_DIR17F = 4
R_PARISC_DIR14R = 6
R_PARISC_PCREL32 = 9
R_PARISC_PCREL21L = 10
R_PARISC_PCREL17R = 11
R_PARISC_PCREL17F = 12
R_PARISC_PCREL14R = 14
R_PARISC_DPREL21L = 18
R_PARISC_DPREL14R = 22
R_PARISC_GPREL21L = 26
R_PARISC_GPREL14R = 30
R_PARISC_LTOFF21L = 34
R_PARISC_LTOFF14R = 38
R_PARISC_SECREL32 = 41
R_PARISC_SEGBASE = 48
R_PARISC_SEGREL32 = 49
R_PARISC_PLTOFF21L = 50
R_PARISC_PLTOFF14R = 54
R_PARISC_LTOFF_FPTR32 = 57
R_PARISC_LTOFF_FPTR21L = 58
R_PARISC_LTOFF_FPTR14R = 62
R_PARISC_FPTR64 = 64
R_PARISC_PLABEL32 = 65
R_PARISC_PLABEL21L = 66
R_PARISC_PLABEL14R = 70
R_PARISC_PCREL64 = 72
R_PARISC_PCREL22F = 74
R_PARISC_PCREL14WR = 75
R_PARISC_PCREL14DR = 76
R_PARISC_PCREL16F = 77
R_PARISC_PCREL16WF = 78
R_PARISC_PCREL16DF = 79
R_PARISC_DIR64 = 80
R_PARISC_DIR14WR = 83
R_PARISC_DIR14DR = 84
R_PARISC_DIR16F = 85
R_PARISC_DIR16WF = 86
R_PARISC_DIR16DF = 87
R_PARISC_GPREL64 = 88
R_PARISC_GPREL14WR = 91
R_PARISC_GPREL14DR = 92
R_PARISC_GPREL16F = 93
R_PARISC_GPREL16WF = 94
R_PARISC_GPREL16DF = 95
R_PARISC_LTOFF64 = 96
R_PARISC_LTOFF14WR = 99
R_PARISC_LTOFF14DR = 100
R_PARISC_LTOFF16F = 101
R_PARISC_LTOFF16WF = 102
R_PARISC_LTOFF16DF = 103
R_PARISC_SECREL64 = 104
R_PARISC_SEGREL64 = 112
R_PARISC_PLTOFF14WR = 115
R_PARISC_PLTOFF14DR = 116
R_PARISC_PLTOFF16F = 117
R_PARISC_PLTOFF16WF = 118
R_PARISC_PLTOFF16DF = 119
R_PARISC_LTOFF_FPTR64 = 120
R_PARISC_LTOFF_FPTR14WR = 123
R_PARISC_LTOFF_FPTR14DR = 124
R_PARISC_LTOFF_FPTR16F = 125
R_PARISC_LTOFF_FPTR16WF = 126
R_PARISC_LTOFF_FPTR16DF = 127
R_PARISC_LORESERVE = 128
R_PARISC_COPY = 128
R_PARISC_IPLT = 129
R_PARISC_EPLT = 130
R_PARISC_TPREL32 = 153
R_PARISC_TPREL21L = 154
R_PARISC_TPREL14R = 158
R_PARISC_LTOFF_TP21L = 162
R_PARISC_LTOFF_TP14R = 166
R_PARISC_LTOFF_TP14F = 167
R_PARISC_TPREL64 = 216
R_PARISC_TPREL14WR = 219
R_PARISC_TPREL14DR = 220
R_PARISC_TPREL16F = 221
R_PARISC_TPREL16WF = 222
R_PARISC_TPREL16DF = 223
R_PARISC_LTOFF_TP64 = 224
R_PARISC_LTOFF_TP14WR = 227
R_PARISC_LTOFF_TP14DR = 228
R_PARISC_LTOFF_TP16F = 229
R_PARISC_LTOFF_TP16WF = 230
R_PARISC_LTOFF_TP16DF = 231
R_PARISC_GNU_VTENTRY = 232
R_PARISC_GNU_VTINHERIT = 233
R_PARISC_TLS_GD21L = 234
R_PARISC_TLS_GD14R = 235
R_PARISC_TLS_GDCALL = 236
R_PARISC_TLS_LDM21L = 237
R_PARISC_TLS_LDM14R = 238
R_PARISC_TLS_LDMCALL = 239
R_PARISC_TLS_LDO21L = 240
R_PARISC_TLS_LDO14R = 241
R_PARISC_TLS_DTPMOD32 = 242
R_PARISC_TLS_DTPMOD64 = 243
R_PARISC_TLS_DTPOFF32 = 244
R_PARISC_TLS_DTPOFF64 = 245
R_PARISC_TLS_LE21L = R_PARISC_TPREL21L
R_PARISC_TLS_LE14R = R_PARISC_TPREL14R
R_PARISC_TLS_IE21L = R_PARISC_LTOFF_TP21L
R_PARISC_TLS_IE14R = R_PARISC_LTOFF_TP14R
R_PARISC_TLS_TPREL32 = R_PARISC_TPREL32
R_PARISC_TLS_TPREL64 = R_PARISC_TPREL64
R_PARISC_HIRESERVE = 255
PT_HP_TLS = (PT_LOOS + 0x0)
PT_HP_CORE_NONE = (PT_LOOS + 0x1)
PT_HP_CORE_VERSION = (PT_LOOS + 0x2)
PT_HP_CORE_KERNEL = (PT_LOOS + 0x3)
PT_HP_CORE_COMM = (PT_LOOS + 0x4)
PT_HP_CORE_PROC = (PT_LOOS + 0x5)
PT_HP_CORE_LOADABLE = (PT_LOOS + 0x6)
PT_HP_CORE_STACK = (PT_LOOS + 0x7)
PT_HP_CORE_SHM = (PT_LOOS + 0x8)
PT_HP_CORE_MMF = (PT_LOOS + 0x9)
PT_HP_PARALLEL = (PT_LOOS + 0x10)
PT_HP_FASTBIND = (PT_LOOS + 0x11)
PT_HP_OPT_ANNOT = (PT_LOOS + 0x12)
PT_HP_HSL_ANNOT = (PT_LOOS + 0x13)
PT_HP_STACK = (PT_LOOS + 0x14)
PT_PARISC_ARCHEXT = 0x70000000
PT_PARISC_UNWIND = 0x70000001
PF_PARISC_SBP = 0x08000000
PF_HP_PAGE_SIZE = 0x00100000
PF_HP_FAR_SHARED = 0x00200000
PF_HP_NEAR_SHARED = 0x00400000
PF_HP_CODE = 0x01000000
PF_HP_MODIFY = 0x02000000
PF_HP_LAZYSWAP = 0x04000000
PF_HP_SBP = 0x08000000
EF_ALPHA_32BIT = 1
EF_ALPHA_CANRELAX = 2
SHT_ALPHA_DEBUG = 0x70000001
SHT_ALPHA_REGINFO = 0x70000002
SHF_ALPHA_GPREL = 0x10000000
STO_ALPHA_NOPV = 0x80
STO_ALPHA_STD_GPLOAD = 0x88
R_ALPHA_NONE = 0
R_ALPHA_REFLONG = 1
R_ALPHA_REFQUAD = 2
R_ALPHA_GPREL32 = 3
R_ALPHA_LITERAL = 4
R_ALPHA_LITUSE = 5
R_ALPHA_GPDISP = 6
R_ALPHA_BRADDR = 7
R_ALPHA_HINT = 8
R_ALPHA_SREL16 = 9
R_ALPHA_SREL32 = 10
R_ALPHA_SREL64 = 11
R_ALPHA_GPRELHIGH = 17
R_ALPHA_GPRELLOW = 18
R_ALPHA_GPREL16 = 19
R_ALPHA_COPY = 24
R_ALPHA_GLOB_DAT = 25
R_ALPHA_JMP_SLOT = 26
R_ALPHA_RELATIVE = 27
R_ALPHA_TLS_GD_HI = 28
R_ALPHA_TLSGD = 29
R_ALPHA_TLS_LDM = 30
R_ALPHA_DTPMOD64 = 31
R_ALPHA_GOTDTPREL = 32
R_ALPHA_DTPREL64 = 33
R_ALPHA_DTPRELHI = 34
R_ALPHA_DTPRELLO = 35
R_ALPHA_DTPREL16 = 36
R_ALPHA_GOTTPREL = 37
R_ALPHA_TPREL64 = 38
R_ALPHA_TPRELHI = 39
R_ALPHA_TPRELLO = 40
R_ALPHA_TPREL16 = 41
R_ALPHA_NUM = 46
LITUSE_ALPHA_ADDR = 0
LITUSE_ALPHA_BASE = 1
LITUSE_ALPHA_BYTOFF = 2
LITUSE_ALPHA_JSR = 3
LITUSE_ALPHA_TLS_GD = 4
LITUSE_ALPHA_TLS_LDM = 5
DT_ALPHA_PLTRO = (DT_LOPROC + 0)
DT_ALPHA_NUM = 1
EF_PPC_EMB = 0x80000000
EF_PPC_RELOCATABLE = 0x00010000
EF_PPC_RELOCATABLE_LIB = 0x00008000
R_PPC_NONE = 0
R_PPC_ADDR32 = 1
R_PPC_ADDR24 = 2
R_PPC_ADDR16 = 3
R_PPC_ADDR16_LO = 4
R_PPC_ADDR16_HI = 5
R_PPC_ADDR16_HA = 6
R_PPC_ADDR14 = 7
R_PPC_ADDR14_BRTAKEN = 8
R_PPC_ADDR14_BRNTAKEN = 9
R_PPC_REL24 = 10
R_PPC_REL14 = 11
R_PPC_REL14_BRTAKEN = 12
R_PPC_REL14_BRNTAKEN = 13
R_PPC_GOT16 = 14
R_PPC_GOT16_LO = 15
R_PPC_GOT16_HI = 16
R_PPC_GOT16_HA = 17
R_PPC_PLTREL24 = 18
R_PPC_COPY = 19
R_PPC_GLOB_DAT = 20
R_PPC_JMP_SLOT = 21
R_PPC_RELATIVE = 22
R_PPC_LOCAL24PC = 23
R_PPC_UADDR32 = 24
R_PPC_UADDR16 = 25
R_PPC_REL32 = 26
R_PPC_PLT32 = 27
R_PPC_PLTREL32 = 28
R_PPC_PLT16_LO = 29
R_PPC_PLT16_HI = 30
R_PPC_PLT16_HA = 31
R_PPC_SDAREL16 = 32
R_PPC_SECTOFF = 33
R_PPC_SECTOFF_LO = 34
R_PPC_SECTOFF_HI = 35
R_PPC_SECTOFF_HA = 36
R_PPC_TLS = 67
R_PPC_DTPMOD32 = 68
R_PPC_TPREL16 = 69
R_PPC_TPREL16_LO = 70
R_PPC_TPREL16_HI = 71
R_PPC_TPREL16_HA = 72
R_PPC_TPREL32 = 73
R_PPC_DTPREL16 = 74
R_PPC_DTPREL16_LO = 75
R_PPC_DTPREL16_HI = 76
R_PPC_DTPREL16_HA = 77
R_PPC_DTPREL32 = 78
R_PPC_GOT_TLSGD16 = 79
R_PPC_GOT_TLSGD16_LO = 80
R_PPC_GOT_TLSGD16_HI = 81
R_PPC_GOT_TLSGD16_HA = 82
R_PPC_GOT_TLSLD16 = 83
R_PPC_GOT_TLSLD16_LO = 84
R_PPC_GOT_TLSLD16_HI = 85
R_PPC_GOT_TLSLD16_HA = 86
R_PPC_GOT_TPREL16 = 87
R_PPC_GOT_TPREL16_LO = 88
R_PPC_GOT_TPREL16_HI = 89
R_PPC_GOT_TPREL16_HA = 90
R_PPC_GOT_DTPREL16 = 91
R_PPC_GOT_DTPREL16_LO = 92
R_PPC_GOT_DTPREL16_HI = 93
R_PPC_GOT_DTPREL16_HA = 94
R_PPC_TLSGD = 95
R_PPC_TLSLD = 96
R_PPC_EMB_NADDR32 = 101
R_PPC_EMB_NADDR16 = 102
R_PPC_EMB_NADDR16_LO = 103
R_PPC_EMB_NADDR16_HI = 104
R_PPC_EMB_NADDR16_HA = 105
R_PPC_EMB_SDAI16 = 106
R_PPC_EMB_SDA2I16 = 107
R_PPC_EMB_SDA2REL = 108
R_PPC_EMB_SDA21 = 109
R_PPC_EMB_MRKREF = 110
R_PPC_EMB_RELSEC16 = 111
R_PPC_EMB_RELST_LO = 112
R_PPC_EMB_RELST_HI = 113
R_PPC_EMB_RELST_HA = 114
R_PPC_EMB_BIT_FLD = 115
R_PPC_EMB_RELSDA = 116
R_PPC_DIAB_SDA21_LO = 180
R_PPC_DIAB_SDA21_HI = 181
R_PPC_DIAB_SDA21_HA = 182
R_PPC_DIAB_RELSDA_LO = 183
R_PPC_DIAB_RELSDA_HI = 184
R_PPC_DIAB_RELSDA_HA = 185
R_PPC_IRELATIVE = 248
R_PPC_REL16 = 249
R_PPC_REL16_LO = 250
R_PPC_REL16_HI = 251
R_PPC_REL16_HA = 252
R_PPC_TOC16 = 255
DT_PPC_GOT = (DT_LOPROC + 0)
DT_PPC_OPT = (DT_LOPROC + 1)
DT_PPC_NUM = 2
PPC_OPT_TLS = 1
R_PPC64_NONE = R_PPC_NONE
R_PPC64_ADDR32 = R_PPC_ADDR32
R_PPC64_ADDR24 = R_PPC_ADDR24
R_PPC64_ADDR16 = R_PPC_ADDR16
R_PPC64_ADDR16_LO = R_PPC_ADDR16_LO
R_PPC64_ADDR16_HI = R_PPC_ADDR16_HI
R_PPC64_ADDR16_HA = R_PPC_ADDR16_HA
R_PPC64_ADDR14 = R_PPC_ADDR14
R_PPC64_ADDR14_BRTAKEN = R_PPC_ADDR14_BRTAKEN
R_PPC64_ADDR14_BRNTAKEN = R_PPC_ADDR14_BRNTAKEN
R_PPC64_REL24 = R_PPC_REL24
R_PPC64_REL14 = R_PPC_REL14
R_PPC64_REL14_BRTAKEN = R_PPC_REL14_BRTAKEN
R_PPC64_REL14_BRNTAKEN = R_PPC_REL14_BRNTAKEN
R_PPC64_GOT16 = R_PPC_GOT16
R_PPC64_GOT16_LO = R_PPC_GOT16_LO
R_PPC64_GOT16_HI = R_PPC_GOT16_HI
R_PPC64_GOT16_HA = R_PPC_GOT16_HA
R_PPC64_COPY = R_PPC_COPY
R_PPC64_GLOB_DAT = R_PPC_GLOB_DAT
R_PPC64_JMP_SLOT = R_PPC_JMP_SLOT
R_PPC64_RELATIVE = R_PPC_RELATIVE
R_PPC64_UADDR32 = R_PPC_UADDR32
R_PPC64_UADDR16 = R_PPC_UADDR16
R_PPC64_REL32 = R_PPC_REL32
R_PPC64_PLT32 = R_PPC_PLT32
R_PPC64_PLTREL32 = R_PPC_PLTREL32
R_PPC64_PLT16_LO = R_PPC_PLT16_LO
R_PPC64_PLT16_HI = R_PPC_PLT16_HI
R_PPC64_PLT16_HA = R_PPC_PLT16_HA
R_PPC64_SECTOFF = R_PPC_SECTOFF
R_PPC64_SECTOFF_LO = R_PPC_SECTOFF_LO
R_PPC64_SECTOFF_HI = R_PPC_SECTOFF_HI
R_PPC64_SECTOFF_HA = R_PPC_SECTOFF_HA
R_PPC64_ADDR30 = 37
R_PPC64_ADDR64 = 38
R_PPC64_ADDR16_HIGHER = 39
R_PPC64_ADDR16_HIGHERA = 40
R_PPC64_ADDR16_HIGHEST = 41
R_PPC64_ADDR16_HIGHESTA = 42
R_PPC64_UADDR64 = 43
R_PPC64_REL64 = 44
R_PPC64_PLT64 = 45
R_PPC64_PLTREL64 = 46
R_PPC64_TOC16 = 47
R_PPC64_TOC16_LO = 48
R_PPC64_TOC16_HI = 49
R_PPC64_TOC16_HA = 50
R_PPC64_TOC = 51
R_PPC64_PLTGOT16 = 52
R_PPC64_PLTGOT16_LO = 53
R_PPC64_PLTGOT16_HI = 54
R_PPC64_PLTGOT16_HA = 55
R_PPC64_ADDR16_DS = 56
R_PPC64_ADDR16_LO_DS = 57
R_PPC64_GOT16_DS = 58
R_PPC64_GOT16_LO_DS = 59
R_PPC64_PLT16_LO_DS = 60
R_PPC64_SECTOFF_DS = 61
R_PPC64_SECTOFF_LO_DS = 62
R_PPC64_TOC16_DS = 63
R_PPC64_TOC16_LO_DS = 64
R_PPC64_PLTGOT16_DS = 65
R_PPC64_PLTGOT16_LO_DS = 66
R_PPC64_TLS = 67
R_PPC64_DTPMOD64 = 68
R_PPC64_TPREL16 = 69
R_PPC64_TPREL16_LO = 70
R_PPC64_TPREL16_HI = 71
R_PPC64_TPREL16_HA = 72
R_PPC64_TPREL64 = 73
R_PPC64_DTPREL16 = 74
R_PPC64_DTPREL16_LO = 75
R_PPC64_DTPREL16_HI = 76
R_PPC64_DTPREL16_HA = 77
R_PPC64_DTPREL64 = 78
R_PPC64_GOT_TLSGD16 = 79
R_PPC64_GOT_TLSGD16_LO = 80
R_PPC64_GOT_TLSGD16_HI = 81
R_PPC64_GOT_TLSGD16_HA = 82
R_PPC64_GOT_TLSLD16 = 83
R_PPC64_GOT_TLSLD16_LO = 84
R_PPC64_GOT_TLSLD16_HI = 85
R_PPC64_GOT_TLSLD16_HA = 86
R_PPC64_GOT_TPREL16_DS = 87
R_PPC64_GOT_TPREL16_LO_DS = 88
R_PPC64_GOT_TPREL16_HI = 89
R_PPC64_GOT_TPREL16_HA = 90
R_PPC64_GOT_DTPREL16_DS = 91
R_PPC64_GOT_DTPREL16_LO_DS = 92
R_PPC64_GOT_DTPREL16_HI = 93
R_PPC64_GOT_DTPREL16_HA = 94
R_PPC64_TPREL16_DS = 95
R_PPC64_TPREL16_LO_DS = 96
R_PPC64_TPREL16_HIGHER = 97
R_PPC64_TPREL16_HIGHERA = 98
R_PPC64_TPREL16_HIGHEST = 99
R_PPC64_TPREL16_HIGHESTA = 100
R_PPC64_DTPREL16_DS = 101
R_PPC64_DTPREL16_LO_DS = 102
R_PPC64_DTPREL16_HIGHER = 103
R_PPC64_DTPREL16_HIGHERA = 104
R_PPC64_DTPREL16_HIGHEST = 105
R_PPC64_DTPREL16_HIGHESTA = 106
R_PPC64_TLSGD = 107
R_PPC64_TLSLD = 108
R_PPC64_TOCSAVE = 109
R_PPC64_ADDR16_HIGH = 110
R_PPC64_ADDR16_HIGHA = 111
R_PPC64_TPREL16_HIGH = 112
R_PPC64_TPREL16_HIGHA = 113
R_PPC64_DTPREL16_HIGH = 114
R_PPC64_DTPREL16_HIGHA = 115
R_PPC64_JMP_IREL = 247
R_PPC64_IRELATIVE = 248
R_PPC64_REL16 = 249
R_PPC64_REL16_LO = 250
R_PPC64_REL16_HI = 251
R_PPC64_REL16_HA = 252
EF_PPC64_ABI = 3
DT_PPC64_GLINK = (DT_LOPROC + 0)
DT_PPC64_OPD = (DT_LOPROC + 1)
DT_PPC64_OPDSZ = (DT_LOPROC + 2)
DT_PPC64_OPT = (DT_LOPROC + 3)
DT_PPC64_NUM = 4
PPC64_OPT_TLS = 1
PPC64_OPT_MULTI_TOC = 2
PPC64_OPT_LOCALENTRY = 4
STO_PPC64_LOCAL_BIT = 5
STO_PPC64_LOCAL_MASK = (7 << STO_PPC64_LOCAL_BIT)
PPC64_LOCAL_ENTRY_OFFSET = lambda other: (((1 << (((other) & STO_PPC64_LOCAL_MASK) >> STO_PPC64_LOCAL_BIT)) >> 2) << 2)
EF_ARM_RELEXEC = 0x01
EF_ARM_HASENTRY = 0x02
EF_ARM_INTERWORK = 0x04
EF_ARM_APCS_26 = 0x08
EF_ARM_APCS_FLOAT = 0x10
EF_ARM_PIC = 0x20
EF_ARM_ALIGN8 = 0x40
EF_ARM_NEW_ABI = 0x80
EF_ARM_OLD_ABI = 0x100
EF_ARM_SOFT_FLOAT = 0x200
EF_ARM_VFP_FLOAT = 0x400
EF_ARM_MAVERICK_FLOAT = 0x800
EF_ARM_ABI_FLOAT_SOFT = 0x200
EF_ARM_ABI_FLOAT_HARD = 0x400
EF_ARM_SYMSARESORTED = 0x04
EF_ARM_DYNSYMSUSESEGIDX = 0x08
EF_ARM_MAPSYMSFIRST = 0x10
EF_ARM_EABIMASK = 0XFF000000
EF_ARM_BE8 = 0x00800000
EF_ARM_LE8 = 0x00400000
EF_ARM_EABI_VERSION = lambda flags: ((flags) & EF_ARM_EABIMASK)
EF_ARM_EABI_UNKNOWN = 0x00000000
EF_ARM_EABI_VER1 = 0x01000000
EF_ARM_EABI_VER2 = 0x02000000
EF_ARM_EABI_VER3 = 0x03000000
EF_ARM_EABI_VER4 = 0x04000000
EF_ARM_EABI_VER5 = 0x05000000
STT_ARM_TFUNC = STT_LOPROC
STT_ARM_16BIT = STT_HIPROC
SHF_ARM_ENTRYSECT = 0x10000000
SHF_ARM_COMDEF = 0x80000000
PF_ARM_SB = 0x10000000
PF_ARM_PI = 0x20000000
PF_ARM_ABS = 0x40000000
PT_ARM_EXIDX = (PT_LOPROC + 1)
SHT_ARM_EXIDX = (SHT_LOPROC + 1)
SHT_ARM_PREEMPTMAP = (SHT_LOPROC + 2)
SHT_ARM_ATTRIBUTES = (SHT_LOPROC + 3)
R_AARCH64_NONE = 0
R_AARCH64_P32_ABS32 = 1
R_AARCH64_P32_COPY = 180
R_AARCH64_P32_GLOB_DAT = 181
R_AARCH64_P32_JUMP_SLOT = 182
R_AARCH64_P32_RELATIVE = 183
R_AARCH64_P32_TLS_DTPMOD = 184
R_AARCH64_P32_TLS_DTPREL = 185
R_AARCH64_P32_TLS_TPREL = 186
R_AARCH64_P32_TLSDESC = 187
R_AARCH64_P32_IRELATIVE = 188
R_AARCH64_ABS64 = 257
R_AARCH64_ABS32 = 258
R_AARCH64_ABS16 = 259
R_AARCH64_PREL64 = 260
R_AARCH64_PREL32 = 261
R_AARCH64_PREL16 = 262
R_AARCH64_MOVW_UABS_G0 = 263
R_AARCH64_MOVW_UABS_G0_NC = 264
R_AARCH64_MOVW_UABS_G1 = 265
R_AARCH64_MOVW_UABS_G1_NC = 266
R_AARCH64_MOVW_UABS_G2 = 267
R_AARCH64_MOVW_UABS_G2_NC = 268
R_AARCH64_MOVW_UABS_G3 = 269
R_AARCH64_MOVW_SABS_G0 = 270
R_AARCH64_MOVW_SABS_G1 = 271
R_AARCH64_MOVW_SABS_G2 = 272
R_AARCH64_LD_PREL_LO19 = 273
R_AARCH64_ADR_PREL_LO21 = 274
R_AARCH64_ADR_PREL_PG_HI21 = 275
R_AARCH64_ADR_PREL_PG_HI21_NC = 276
R_AARCH64_ADD_ABS_LO12_NC = 277
R_AARCH64_LDST8_ABS_LO12_NC = 278
R_AARCH64_TSTBR14 = 279
R_AARCH64_CONDBR19 = 280
R_AARCH64_JUMP26 = 282
R_AARCH64_CALL26 = 283
R_AARCH64_LDST16_ABS_LO12_NC = 284
R_AARCH64_LDST32_ABS_LO12_NC = 285
R_AARCH64_LDST64_ABS_LO12_NC = 286
R_AARCH64_MOVW_PREL_G0 = 287
R_AARCH64_MOVW_PREL_G0_NC = 288
R_AARCH64_MOVW_PREL_G1 = 289
R_AARCH64_MOVW_PREL_G1_NC = 290
R_AARCH64_MOVW_PREL_G2 = 291
R_AARCH64_MOVW_PREL_G2_NC = 292
R_AARCH64_MOVW_PREL_G3 = 293
R_AARCH64_LDST128_ABS_LO12_NC = 299
R_AARCH64_MOVW_GOTOFF_G0 = 300
R_AARCH64_MOVW_GOTOFF_G0_NC = 301
R_AARCH64_MOVW_GOTOFF_G1 = 302
R_AARCH64_MOVW_GOTOFF_G1_NC = 303
R_AARCH64_MOVW_GOTOFF_G2 = 304
R_AARCH64_MOVW_GOTOFF_G2_NC = 305
R_AARCH64_MOVW_GOTOFF_G3 = 306
R_AARCH64_GOTREL64 = 307
R_AARCH64_GOTREL32 = 308
R_AARCH64_GOT_LD_PREL19 = 309
R_AARCH64_LD64_GOTOFF_LO15 = 310
R_AARCH64_ADR_GOT_PAGE = 311
R_AARCH64_LD64_GOT_LO12_NC = 312
R_AARCH64_LD64_GOTPAGE_LO15 = 313
R_AARCH64_TLSGD_ADR_PREL21 = 512
R_AARCH64_TLSGD_ADR_PAGE21 = 513
R_AARCH64_TLSGD_ADD_LO12_NC = 514
R_AARCH64_TLSGD_MOVW_G1 = 515
R_AARCH64_TLSGD_MOVW_G0_NC = 516
R_AARCH64_TLSLD_ADR_PREL21 = 517
R_AARCH64_TLSLD_ADR_PAGE21 = 518
R_AARCH64_TLSLD_ADD_LO12_NC = 519
R_AARCH64_TLSLD_MOVW_G1 = 520
R_AARCH64_TLSLD_MOVW_G0_NC = 521
R_AARCH64_TLSLD_LD_PREL19 = 522
R_AARCH64_TLSLD_MOVW_DTPREL_G2 = 523
R_AARCH64_TLSLD_MOVW_DTPREL_G1 = 524
R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC = 525
R_AARCH64_TLSLD_MOVW_DTPREL_G0 = 526
R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC = 527
R_AARCH64_TLSLD_ADD_DTPREL_HI12 = 528
R_AARCH64_TLSLD_ADD_DTPREL_LO12 = 529
R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC = 530
R_AARCH64_TLSLD_LDST8_DTPREL_LO12 = 531
R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC = 532
R_AARCH64_TLSLD_LDST16_DTPREL_LO12 = 533
R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC = 534
R_AARCH64_TLSLD_LDST32_DTPREL_LO12 = 535
R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC = 536
R_AARCH64_TLSLD_LDST64_DTPREL_LO12 = 537
R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC = 538
R_AARCH64_TLSIE_MOVW_GOTTPREL_G1 = 539
R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC = 540
R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 = 541
R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC = 542
R_AARCH64_TLSIE_LD_GOTTPREL_PREL19 = 543
R_AARCH64_TLSLE_MOVW_TPREL_G2 = 544
R_AARCH64_TLSLE_MOVW_TPREL_G1 = 545
R_AARCH64_TLSLE_MOVW_TPREL_G1_NC = 546
R_AARCH64_TLSLE_MOVW_TPREL_G0 = 547
R_AARCH64_TLSLE_MOVW_TPREL_G0_NC = 548
R_AARCH64_TLSLE_ADD_TPREL_HI12 = 549
R_AARCH64_TLSLE_ADD_TPREL_LO12 = 550
R_AARCH64_TLSLE_ADD_TPREL_LO12_NC = 551
R_AARCH64_TLSLE_LDST8_TPREL_LO12 = 552
R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC = 553
R_AARCH64_TLSLE_LDST16_TPREL_LO12 = 554
R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC = 555
R_AARCH64_TLSLE_LDST32_TPREL_LO12 = 556
R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC = 557
R_AARCH64_TLSLE_LDST64_TPREL_LO12 = 558
R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC = 559
R_AARCH64_TLSDESC_LD_PREL19 = 560
R_AARCH64_TLSDESC_ADR_PREL21 = 561
R_AARCH64_TLSDESC_ADR_PAGE21 = 562
R_AARCH64_TLSDESC_LD64_LO12 = 563
R_AARCH64_TLSDESC_ADD_LO12 = 564
R_AARCH64_TLSDESC_OFF_G1 = 565
R_AARCH64_TLSDESC_OFF_G0_NC = 566
R_AARCH64_TLSDESC_LDR = 567
R_AARCH64_TLSDESC_ADD = 568
R_AARCH64_TLSDESC_CALL = 569
R_AARCH64_TLSLE_LDST128_TPREL_LO12 = 570
R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC = 571
R_AARCH64_TLSLD_LDST128_DTPREL_LO12 = 572
R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC = 573
R_AARCH64_COPY = 1024
R_AARCH64_GLOB_DAT = 1025
R_AARCH64_JUMP_SLOT = 1026
R_AARCH64_RELATIVE = 1027
R_AARCH64_TLS_DTPMOD = 1028
R_AARCH64_TLS_DTPREL = 1029
R_AARCH64_TLS_TPREL = 1030
R_AARCH64_TLSDESC = 1031
R_AARCH64_IRELATIVE = 1032
PT_AARCH64_MEMTAG_MTE = (PT_LOPROC + 2)
DT_AARCH64_BTI_PLT = (DT_LOPROC + 1)
DT_AARCH64_PAC_PLT = (DT_LOPROC + 3)
DT_AARCH64_VARIANT_PCS = (DT_LOPROC + 5)
DT_AARCH64_NUM = 6
STO_AARCH64_VARIANT_PCS = 0x80
R_ARM_NONE = 0
R_ARM_PC24 = 1
R_ARM_ABS32 = 2
R_ARM_REL32 = 3
R_ARM_PC13 = 4
R_ARM_ABS16 = 5
R_ARM_ABS12 = 6
R_ARM_THM_ABS5 = 7
R_ARM_ABS8 = 8
R_ARM_SBREL32 = 9
R_ARM_THM_PC22 = 10
R_ARM_THM_PC8 = 11
R_ARM_AMP_VCALL9 = 12
R_ARM_SWI24 = 13
R_ARM_TLS_DESC = 13
R_ARM_THM_SWI8 = 14
R_ARM_XPC25 = 15
R_ARM_THM_XPC22 = 16
R_ARM_TLS_DTPMOD32 = 17
R_ARM_TLS_DTPOFF32 = 18
R_ARM_TLS_TPOFF32 = 19
R_ARM_COPY = 20
R_ARM_GLOB_DAT = 21
R_ARM_JUMP_SLOT = 22
R_ARM_RELATIVE = 23
R_ARM_GOTOFF = 24
R_ARM_GOTPC = 25
R_ARM_GOT32 = 26
R_ARM_PLT32 = 27
R_ARM_CALL = 28
R_ARM_JUMP24 = 29
R_ARM_THM_JUMP24 = 30
R_ARM_BASE_ABS = 31
R_ARM_ALU_PCREL_7_0 = 32
R_ARM_ALU_PCREL_15_8 = 33
R_ARM_ALU_PCREL_23_15 = 34
R_ARM_LDR_SBREL_11_0 = 35
R_ARM_ALU_SBREL_19_12 = 36
R_ARM_ALU_SBREL_27_20 = 37
R_ARM_TARGET1 = 38
R_ARM_SBREL31 = 39
R_ARM_V4BX = 40
R_ARM_TARGET2 = 41
R_ARM_PREL31 = 42
R_ARM_MOVW_ABS_NC = 43
R_ARM_MOVT_ABS = 44
R_ARM_MOVW_PREL_NC = 45
R_ARM_MOVT_PREL = 46
R_ARM_THM_MOVW_ABS_NC = 47
R_ARM_THM_MOVT_ABS = 48
R_ARM_THM_MOVW_PREL_NC = 49
R_ARM_THM_MOVT_PREL = 50
R_ARM_THM_JUMP19 = 51
R_ARM_THM_JUMP6 = 52
R_ARM_THM_ALU_PREL_11_0 = 53
R_ARM_THM_PC12 = 54
R_ARM_ABS32_NOI = 55
R_ARM_REL32_NOI = 56
R_ARM_ALU_PC_G0_NC = 57
R_ARM_ALU_PC_G0 = 58
R_ARM_ALU_PC_G1_NC = 59
R_ARM_ALU_PC_G1 = 60
R_ARM_ALU_PC_G2 = 61
R_ARM_LDR_PC_G1 = 62
R_ARM_LDR_PC_G2 = 63
R_ARM_LDRS_PC_G0 = 64
R_ARM_LDRS_PC_G1 = 65
R_ARM_LDRS_PC_G2 = 66
R_ARM_LDC_PC_G0 = 67
R_ARM_LDC_PC_G1 = 68
R_ARM_LDC_PC_G2 = 69
R_ARM_ALU_SB_G0_NC = 70
R_ARM_ALU_SB_G0 = 71
R_ARM_ALU_SB_G1_NC = 72
R_ARM_ALU_SB_G1 = 73
R_ARM_ALU_SB_G2 = 74
R_ARM_LDR_SB_G0 = 75
R_ARM_LDR_SB_G1 = 76
R_ARM_LDR_SB_G2 = 77
R_ARM_LDRS_SB_G0 = 78
R_ARM_LDRS_SB_G1 = 79
R_ARM_LDRS_SB_G2 = 80
R_ARM_LDC_SB_G0 = 81
R_ARM_LDC_SB_G1 = 82
R_ARM_LDC_SB_G2 = 83
R_ARM_MOVW_BREL_NC = 84
R_ARM_MOVT_BREL = 85
R_ARM_MOVW_BREL = 86
R_ARM_THM_MOVW_BREL_NC = 87
R_ARM_THM_MOVT_BREL = 88
R_ARM_THM_MOVW_BREL = 89
R_ARM_TLS_GOTDESC = 90
R_ARM_TLS_CALL = 91
R_ARM_TLS_DESCSEQ = 92
R_ARM_THM_TLS_CALL = 93
R_ARM_PLT32_ABS = 94
R_ARM_GOT_ABS = 95
R_ARM_GOT_PREL = 96
R_ARM_GOT_BREL12 = 97
R_ARM_GOTOFF12 = 98
R_ARM_GOTRELAX = 99
R_ARM_GNU_VTENTRY = 100
R_ARM_GNU_VTINHERIT = 101
R_ARM_THM_PC11 = 102
R_ARM_THM_PC9 = 103
R_ARM_TLS_GD32 = 104
R_ARM_TLS_LDM32 = 105
R_ARM_TLS_LDO32 = 106
R_ARM_TLS_IE32 = 107
R_ARM_TLS_LE32 = 108
R_ARM_TLS_LDO12 = 109
R_ARM_TLS_LE12 = 110
R_ARM_TLS_IE12GP = 111
R_ARM_ME_TOO = 128
R_ARM_THM_TLS_DESCSEQ = 129
R_ARM_THM_TLS_DESCSEQ16 = 129
R_ARM_THM_TLS_DESCSEQ32 = 130
R_ARM_THM_GOT_BREL12 = 131
R_ARM_IRELATIVE = 160
R_ARM_RXPC25 = 249
R_ARM_RSBREL32 = 250
R_ARM_THM_RPC22 = 251
R_ARM_RREL32 = 252
R_ARM_RABS22 = 253
R_ARM_RPC24 = 254
R_ARM_RBASE = 255
R_ARM_NUM = 256
R_CKCORE_NONE = 0
R_CKCORE_ADDR32 = 1
R_CKCORE_PCRELIMM8BY4 = 2
R_CKCORE_PCRELIMM11BY2 = 3
R_CKCORE_PCREL32 = 5
R_CKCORE_PCRELJSR_IMM11BY2 = 6
R_CKCORE_RELATIVE = 9
R_CKCORE_COPY = 10
R_CKCORE_GLOB_DAT = 11
R_CKCORE_JUMP_SLOT = 12
R_CKCORE_GOTOFF = 13
R_CKCORE_GOTPC = 14
R_CKCORE_GOT32 = 15
R_CKCORE_PLT32 = 16
R_CKCORE_ADDRGOT = 17
R_CKCORE_ADDRPLT = 18
R_CKCORE_PCREL_IMM26BY2 = 19
R_CKCORE_PCREL_IMM16BY2 = 20
R_CKCORE_PCREL_IMM16BY4 = 21
R_CKCORE_PCREL_IMM10BY2 = 22
R_CKCORE_PCREL_IMM10BY4 = 23
R_CKCORE_ADDR_HI16 = 24
R_CKCORE_ADDR_LO16 = 25
R_CKCORE_GOTPC_HI16 = 26
R_CKCORE_GOTPC_LO16 = 27
R_CKCORE_GOTOFF_HI16 = 28
R_CKCORE_GOTOFF_LO16 = 29
R_CKCORE_GOT12 = 30
R_CKCORE_GOT_HI16 = 31
R_CKCORE_GOT_LO16 = 32
R_CKCORE_PLT12 = 33
R_CKCORE_PLT_HI16 = 34
R_CKCORE_PLT_LO16 = 35
R_CKCORE_ADDRGOT_HI16 = 36
R_CKCORE_ADDRGOT_LO16 = 37
R_CKCORE_ADDRPLT_HI16 = 38
R_CKCORE_ADDRPLT_LO16 = 39
R_CKCORE_PCREL_JSR_IMM26BY2 = 40
R_CKCORE_TOFFSET_LO16 = 41
R_CKCORE_DOFFSET_LO16 = 42
R_CKCORE_PCREL_IMM18BY2 = 43
R_CKCORE_DOFFSET_IMM18 = 44
R_CKCORE_DOFFSET_IMM18BY2 = 45
R_CKCORE_DOFFSET_IMM18BY4 = 46
R_CKCORE_GOT_IMM18BY4 = 48
R_CKCORE_PLT_IMM18BY4 = 49
R_CKCORE_PCREL_IMM7BY4 = 50
R_CKCORE_TLS_LE32 = 51
R_CKCORE_TLS_IE32 = 52
R_CKCORE_TLS_GD32 = 53
R_CKCORE_TLS_LDM32 = 54
R_CKCORE_TLS_LDO32 = 55
R_CKCORE_TLS_DTPMOD32 = 56
R_CKCORE_TLS_DTPOFF32 = 57
R_CKCORE_TLS_TPOFF32 = 58
EF_CSKY_ABIMASK = 0XF0000000
EF_CSKY_OTHER = 0X0FFF0000
EF_CSKY_PROCESSOR = 0X0000FFFF
EF_CSKY_ABIV1 = 0X10000000
EF_CSKY_ABIV2 = 0X20000000
SHT_CSKY_ATTRIBUTES = (SHT_LOPROC + 1)
EF_IA_64_MASKOS = 0x0000000f
EF_IA_64_ABI64 = 0x00000010
EF_IA_64_ARCH = 0xff000000
PT_IA_64_ARCHEXT = (PT_LOPROC + 0)
PT_IA_64_UNWIND = (PT_LOPROC + 1)
PT_IA_64_HP_OPT_ANOT = (PT_LOOS + 0x12)
PT_IA_64_HP_HSL_ANOT = (PT_LOOS + 0x13)
PT_IA_64_HP_STACK = (PT_LOOS + 0x14)
PF_IA_64_NORECOV = 0x80000000
SHT_IA_64_EXT = (SHT_LOPROC + 0)
SHT_IA_64_UNWIND = (SHT_LOPROC + 1)
SHF_IA_64_SHORT = 0x10000000
SHF_IA_64_NORECOV = 0x20000000
DT_IA_64_PLT_RESERVE = (DT_LOPROC + 0)
DT_IA_64_NUM = 1
R_IA64_NONE = 0x00
R_IA64_IMM14 = 0x21
R_IA64_IMM22 = 0x22
R_IA64_IMM64 = 0x23
R_IA64_DIR32MSB = 0x24
R_IA64_DIR32LSB = 0x25
R_IA64_DIR64MSB = 0x26
R_IA64_DIR64LSB = 0x27
R_IA64_GPREL22 = 0x2a
R_IA64_GPREL64I = 0x2b
R_IA64_GPREL32MSB = 0x2c
R_IA64_GPREL32LSB = 0x2d
R_IA64_GPREL64MSB = 0x2e
R_IA64_GPREL64LSB = 0x2f
R_IA64_LTOFF22 = 0x32
R_IA64_LTOFF64I = 0x33
R_IA64_PLTOFF22 = 0x3a
R_IA64_PLTOFF64I = 0x3b
R_IA64_PLTOFF64MSB = 0x3e
R_IA64_PLTOFF64LSB = 0x3f
R_IA64_FPTR64I = 0x43
R_IA64_FPTR32MSB = 0x44
R_IA64_FPTR32LSB = 0x45
R_IA64_FPTR64MSB = 0x46
R_IA64_FPTR64LSB = 0x47
R_IA64_PCREL60B = 0x48
R_IA64_PCREL21B = 0x49
R_IA64_PCREL21M = 0x4a
R_IA64_PCREL21F = 0x4b
R_IA64_PCREL32MSB = 0x4c
R_IA64_PCREL32LSB = 0x4d
R_IA64_PCREL64MSB = 0x4e
R_IA64_PCREL64LSB = 0x4f
R_IA64_LTOFF_FPTR22 = 0x52
R_IA64_LTOFF_FPTR64I = 0x53
R_IA64_LTOFF_FPTR32MSB = 0x54
R_IA64_LTOFF_FPTR32LSB = 0x55
R_IA64_LTOFF_FPTR64MSB = 0x56
R_IA64_LTOFF_FPTR64LSB = 0x57
R_IA64_SEGREL32MSB = 0x5c
R_IA64_SEGREL32LSB = 0x5d
R_IA64_SEGREL64MSB = 0x5e
R_IA64_SEGREL64LSB = 0x5f
R_IA64_SECREL32MSB = 0x64
R_IA64_SECREL32LSB = 0x65
R_IA64_SECREL64MSB = 0x66
R_IA64_SECREL64LSB = 0x67
R_IA64_REL32MSB = 0x6c
R_IA64_REL32LSB = 0x6d
R_IA64_REL64MSB = 0x6e
R_IA64_REL64LSB = 0x6f
R_IA64_LTV32MSB = 0x74
R_IA64_LTV32LSB = 0x75
R_IA64_LTV64MSB = 0x76
R_IA64_LTV64LSB = 0x77
R_IA64_PCREL21BI = 0x79
R_IA64_PCREL22 = 0x7a
R_IA64_PCREL64I = 0x7b
R_IA64_IPLTMSB = 0x80
R_IA64_IPLTLSB = 0x81
R_IA64_COPY = 0x84
R_IA64_SUB = 0x85
R_IA64_LTOFF22X = 0x86
R_IA64_LDXMOV = 0x87
R_IA64_TPREL14 = 0x91
R_IA64_TPREL22 = 0x92
R_IA64_TPREL64I = 0x93
R_IA64_TPREL64MSB = 0x96
R_IA64_TPREL64LSB = 0x97
R_IA64_LTOFF_TPREL22 = 0x9a
R_IA64_DTPMOD64MSB = 0xa6
R_IA64_DTPMOD64LSB = 0xa7
R_IA64_LTOFF_DTPMOD22 = 0xaa
R_IA64_DTPREL14 = 0xb1
R_IA64_DTPREL22 = 0xb2
R_IA64_DTPREL64I = 0xb3
R_IA64_DTPREL32MSB = 0xb4
R_IA64_DTPREL32LSB = 0xb5
R_IA64_DTPREL64MSB = 0xb6
R_IA64_DTPREL64LSB = 0xb7
R_IA64_LTOFF_DTPREL22 = 0xba
EF_SH_MACH_MASK = 0x1f
EF_SH_UNKNOWN = 0x0
EF_SH1 = 0x1
EF_SH2 = 0x2
EF_SH3 = 0x3
EF_SH_DSP = 0x4
EF_SH3_DSP = 0x5
EF_SH4AL_DSP = 0x6
EF_SH3E = 0x8
EF_SH4 = 0x9
EF_SH2E = 0xb
EF_SH4A = 0xc
EF_SH2A = 0xd
EF_SH4_NOFPU = 0x10
EF_SH4A_NOFPU = 0x11
EF_SH4_NOMMU_NOFPU = 0x12
EF_SH2A_NOFPU = 0x13
EF_SH3_NOMMU = 0x14
EF_SH2A_SH4_NOFPU = 0x15
EF_SH2A_SH3_NOFPU = 0x16
EF_SH2A_SH4 = 0x17
EF_SH2A_SH3E = 0x18
R_SH_NONE = 0
R_SH_DIR32 = 1
R_SH_REL32 = 2
R_SH_DIR8WPN = 3
R_SH_IND12W = 4
R_SH_DIR8WPL = 5
R_SH_DIR8WPZ = 6
R_SH_DIR8BP = 7
R_SH_DIR8W = 8
R_SH_DIR8L = 9
R_SH_SWITCH16 = 25
R_SH_SWITCH32 = 26
R_SH_USES = 27
R_SH_COUNT = 28
R_SH_ALIGN = 29
R_SH_CODE = 30
R_SH_DATA = 31
R_SH_LABEL = 32
R_SH_SWITCH8 = 33
R_SH_GNU_VTINHERIT = 34
R_SH_GNU_VTENTRY = 35
R_SH_TLS_GD_32 = 144
R_SH_TLS_LD_32 = 145
R_SH_TLS_LDO_32 = 146
R_SH_TLS_IE_32 = 147
R_SH_TLS_LE_32 = 148
R_SH_TLS_DTPMOD32 = 149
R_SH_TLS_DTPOFF32 = 150
R_SH_TLS_TPOFF32 = 151
R_SH_GOT32 = 160
R_SH_PLT32 = 161
R_SH_COPY = 162
R_SH_GLOB_DAT = 163
R_SH_JMP_SLOT = 164
R_SH_RELATIVE = 165
R_SH_GOTOFF = 166
R_SH_GOTPC = 167
R_SH_NUM = 256
EF_S390_HIGH_GPRS = 0x00000001
R_390_NONE = 0
R_390_8 = 1
R_390_12 = 2
R_390_16 = 3
R_390_32 = 4
R_390_PC32 = 5
R_390_GOT12 = 6
R_390_GOT32 = 7
R_390_PLT32 = 8
R_390_COPY = 9
R_390_GLOB_DAT = 10
R_390_JMP_SLOT = 11
R_390_RELATIVE = 12
R_390_GOTOFF32 = 13
R_390_GOTPC = 14
R_390_GOT16 = 15
R_390_PC16 = 16
R_390_PC16DBL = 17
R_390_PLT16DBL = 18
R_390_PC32DBL = 19
R_390_PLT32DBL = 20
R_390_GOTPCDBL = 21
R_390_64 = 22
R_390_PC64 = 23
R_390_GOT64 = 24
R_390_PLT64 = 25
R_390_GOTENT = 26
R_390_GOTOFF16 = 27
R_390_GOTOFF64 = 28
R_390_GOTPLT12 = 29
R_390_GOTPLT16 = 30
R_390_GOTPLT32 = 31
R_390_GOTPLT64 = 32
R_390_GOTPLTENT = 33
R_390_PLTOFF16 = 34
R_390_PLTOFF32 = 35
R_390_PLTOFF64 = 36
R_390_TLS_LOAD = 37
R_390_TLS_GDCALL = 38
R_390_TLS_LDCALL = 39
R_390_TLS_GD32 = 40
R_390_TLS_GD64 = 41
R_390_TLS_GOTIE12 = 42
R_390_TLS_GOTIE32 = 43
R_390_TLS_GOTIE64 = 44
R_390_TLS_LDM32 = 45
R_390_TLS_LDM64 = 46
R_390_TLS_IE32 = 47
R_390_TLS_IE64 = 48
R_390_TLS_IEENT = 49
R_390_TLS_LE32 = 50
R_390_TLS_LE64 = 51
R_390_TLS_LDO32 = 52
R_390_TLS_LDO64 = 53
R_390_TLS_DTPMOD = 54
R_390_TLS_DTPOFF = 55
R_390_TLS_TPOFF = 56
R_390_20 = 57
R_390_GOT20 = 58
R_390_GOTPLT20 = 59
R_390_TLS_GOTIE20 = 60
R_390_IRELATIVE = 61
R_390_NUM = 62
R_CRIS_NONE = 0
R_CRIS_8 = 1
R_CRIS_16 = 2
R_CRIS_32 = 3
R_CRIS_8_PCREL = 4
R_CRIS_16_PCREL = 5
R_CRIS_32_PCREL = 6
R_CRIS_GNU_VTINHERIT = 7
R_CRIS_GNU_VTENTRY = 8
R_CRIS_COPY = 9
R_CRIS_GLOB_DAT = 10
R_CRIS_JUMP_SLOT = 11
R_CRIS_RELATIVE = 12
R_CRIS_16_GOT = 13
R_CRIS_32_GOT = 14
R_CRIS_16_GOTPLT = 15
R_CRIS_32_GOTPLT = 16
R_CRIS_32_GOTREL = 17
R_CRIS_32_PLT_GOTREL = 18
R_CRIS_32_PLT_PCREL = 19
R_CRIS_NUM = 20
R_X86_64_NONE = 0
R_X86_64_64 = 1
R_X86_64_PC32 = 2
R_X86_64_GOT32 = 3
R_X86_64_PLT32 = 4
R_X86_64_COPY = 5
R_X86_64_GLOB_DAT = 6
R_X86_64_JUMP_SLOT = 7
R_X86_64_RELATIVE = 8
R_X86_64_GOTPCREL = 9
R_X86_64_32 = 10
R_X86_64_32S = 11
R_X86_64_16 = 12
R_X86_64_PC16 = 13
R_X86_64_8 = 14
R_X86_64_PC8 = 15
R_X86_64_DTPMOD64 = 16
R_X86_64_DTPOFF64 = 17
R_X86_64_TPOFF64 = 18
R_X86_64_TLSGD = 19
R_X86_64_TLSLD = 20
R_X86_64_DTPOFF32 = 21
R_X86_64_GOTTPOFF = 22
R_X86_64_TPOFF32 = 23
R_X86_64_PC64 = 24
R_X86_64_GOTOFF64 = 25
R_X86_64_GOTPC32 = 26
R_X86_64_GOT64 = 27
R_X86_64_GOTPCREL64 = 28
R_X86_64_GOTPC64 = 29
R_X86_64_GOTPLT64 = 30
R_X86_64_PLTOFF64 = 31
R_X86_64_SIZE32 = 32
R_X86_64_SIZE64 = 33
R_X86_64_GOTPC32_TLSDESC = 34
R_X86_64_TLSDESC_CALL = 35
R_X86_64_TLSDESC = 36
R_X86_64_IRELATIVE = 37
R_X86_64_RELATIVE64 = 38
R_X86_64_GOTPCRELX = 41
R_X86_64_REX_GOTPCRELX = 42
R_X86_64_NUM = 43
SHT_X86_64_UNWIND = 0x70000001
DT_X86_64_PLT = (DT_LOPROC + 0)
DT_X86_64_PLTSZ = (DT_LOPROC + 1)
DT_X86_64_PLTENT = (DT_LOPROC + 3)
DT_X86_64_NUM = 4
R_MN10300_NONE = 0
R_MN10300_32 = 1
R_MN10300_16 = 2
R_MN10300_8 = 3
R_MN10300_PCREL32 = 4
R_MN10300_PCREL16 = 5
R_MN10300_PCREL8 = 6
R_MN10300_GNU_VTINHERIT = 7
R_MN10300_GNU_VTENTRY = 8
R_MN10300_24 = 9
R_MN10300_GOTPC32 = 10
R_MN10300_GOTPC16 = 11
R_MN10300_GOTOFF32 = 12
R_MN10300_GOTOFF24 = 13
R_MN10300_GOTOFF16 = 14
R_MN10300_PLT32 = 15
R_MN10300_PLT16 = 16
R_MN10300_GOT32 = 17
R_MN10300_GOT24 = 18
R_MN10300_GOT16 = 19
R_MN10300_COPY = 20
R_MN10300_GLOB_DAT = 21
R_MN10300_JMP_SLOT = 22
R_MN10300_RELATIVE = 23
R_MN10300_TLS_GD = 24
R_MN10300_TLS_LD = 25
R_MN10300_TLS_LDO = 26
R_MN10300_TLS_GOTIE = 27
R_MN10300_TLS_IE = 28
R_MN10300_TLS_LE = 29
R_MN10300_TLS_DTPMOD = 30
R_MN10300_TLS_DTPOFF = 31
R_MN10300_TLS_TPOFF = 32
R_MN10300_SYM_DIFF = 33
R_MN10300_ALIGN = 34
R_MN10300_NUM = 35
R_M32R_NONE = 0
R_M32R_16 = 1
R_M32R_32 = 2
R_M32R_24 = 3
R_M32R_10_PCREL = 4
R_M32R_18_PCREL = 5
R_M32R_26_PCREL = 6
R_M32R_HI16_ULO = 7
R_M32R_HI16_SLO = 8
R_M32R_LO16 = 9
R_M32R_SDA16 = 10
R_M32R_GNU_VTINHERIT = 11
R_M32R_GNU_VTENTRY = 12
R_M32R_16_RELA = 33
R_M32R_32_RELA = 34
R_M32R_24_RELA = 35
R_M32R_10_PCREL_RELA = 36
R_M32R_18_PCREL_RELA = 37
R_M32R_26_PCREL_RELA = 38
R_M32R_HI16_ULO_RELA = 39
R_M32R_HI16_SLO_RELA = 40
R_M32R_LO16_RELA = 41
R_M32R_SDA16_RELA = 42
R_M32R_RELA_GNU_VTINHERIT = 43
R_M32R_RELA_GNU_VTENTRY = 44
R_M32R_REL32 = 45
R_M32R_GOT24 = 48
R_M32R_26_PLTREL = 49
R_M32R_COPY = 50
R_M32R_GLOB_DAT = 51
R_M32R_JMP_SLOT = 52
R_M32R_RELATIVE = 53
R_M32R_GOTOFF = 54
R_M32R_GOTPC24 = 55
R_M32R_GOT16_HI_ULO = 56
R_M32R_GOT16_HI_SLO = 57
R_M32R_GOT16_LO = 58
R_M32R_GOTPC_HI_ULO = 59
R_M32R_GOTPC_HI_SLO = 60
R_M32R_GOTPC_LO = 61
R_M32R_GOTOFF_HI_ULO = 62
R_M32R_GOTOFF_HI_SLO = 63
R_M32R_GOTOFF_LO = 64
R_M32R_NUM = 256
R_MICROBLAZE_NONE = 0
R_MICROBLAZE_32 = 1
R_MICROBLAZE_32_PCREL = 2
R_MICROBLAZE_64_PCREL = 3
R_MICROBLAZE_32_PCREL_LO = 4
R_MICROBLAZE_64 = 5
R_MICROBLAZE_32_LO = 6
R_MICROBLAZE_SRO32 = 7
R_MICROBLAZE_SRW32 = 8
R_MICROBLAZE_64_NONE = 9
R_MICROBLAZE_32_SYM_OP_SYM = 10
R_MICROBLAZE_GNU_VTINHERIT = 11
R_MICROBLAZE_GNU_VTENTRY = 12
R_MICROBLAZE_GOTPC_64 = 13
R_MICROBLAZE_GOT_64 = 14
R_MICROBLAZE_PLT_64 = 15
R_MICROBLAZE_REL = 16
R_MICROBLAZE_JUMP_SLOT = 17
R_MICROBLAZE_GLOB_DAT = 18
R_MICROBLAZE_GOTOFF_64 = 19
R_MICROBLAZE_GOTOFF_32 = 20
R_MICROBLAZE_COPY = 21
R_MICROBLAZE_TLS = 22
R_MICROBLAZE_TLSGD = 23
R_MICROBLAZE_TLSLD = 24
R_MICROBLAZE_TLSDTPMOD32 = 25
R_MICROBLAZE_TLSDTPREL32 = 26
R_MICROBLAZE_TLSDTPREL64 = 27
R_MICROBLAZE_TLSGOTTPREL32 = 28
R_MICROBLAZE_TLSTPREL32 = 29
DT_NIOS2_GP = 0x70000002
R_NIOS2_NONE = 0
R_NIOS2_S16 = 1
R_NIOS2_U16 = 2
R_NIOS2_PCREL16 = 3
R_NIOS2_CALL26 = 4
R_NIOS2_IMM5 = 5
R_NIOS2_CACHE_OPX = 6
R_NIOS2_IMM6 = 7
R_NIOS2_IMM8 = 8
R_NIOS2_HI16 = 9
R_NIOS2_LO16 = 10
R_NIOS2_HIADJ16 = 11
R_NIOS2_BFD_RELOC_32 = 12
R_NIOS2_BFD_RELOC_16 = 13
R_NIOS2_BFD_RELOC_8 = 14
R_NIOS2_GPREL = 15
R_NIOS2_GNU_VTINHERIT = 16
R_NIOS2_GNU_VTENTRY = 17
R_NIOS2_UJMP = 18
R_NIOS2_CJMP = 19
R_NIOS2_CALLR = 20
R_NIOS2_ALIGN = 21
R_NIOS2_GOT16 = 22
R_NIOS2_CALL16 = 23
R_NIOS2_GOTOFF_LO = 24
R_NIOS2_GOTOFF_HA = 25
R_NIOS2_PCREL_LO = 26
R_NIOS2_PCREL_HA = 27
R_NIOS2_TLS_GD16 = 28
R_NIOS2_TLS_LDM16 = 29
R_NIOS2_TLS_LDO16 = 30
R_NIOS2_TLS_IE16 = 31
R_NIOS2_TLS_LE16 = 32
R_NIOS2_TLS_DTPMOD = 33
R_NIOS2_TLS_DTPREL = 34
R_NIOS2_TLS_TPREL = 35
R_NIOS2_COPY = 36
R_NIOS2_GLOB_DAT = 37
R_NIOS2_JUMP_SLOT = 38
R_NIOS2_RELATIVE = 39
R_NIOS2_GOTOFF = 40
R_NIOS2_CALL26_NOAT = 41
R_NIOS2_GOT_LO = 42
R_NIOS2_GOT_HA = 43
R_NIOS2_CALL_LO = 44
R_NIOS2_CALL_HA = 45
R_TILEPRO_NONE = 0
R_TILEPRO_32 = 1
R_TILEPRO_16 = 2
R_TILEPRO_8 = 3
R_TILEPRO_32_PCREL = 4
R_TILEPRO_16_PCREL = 5
R_TILEPRO_8_PCREL = 6
R_TILEPRO_LO16 = 7
R_TILEPRO_HI16 = 8
R_TILEPRO_HA16 = 9
R_TILEPRO_COPY = 10
R_TILEPRO_GLOB_DAT = 11
R_TILEPRO_JMP_SLOT = 12
R_TILEPRO_RELATIVE = 13
R_TILEPRO_BROFF_X1 = 14
R_TILEPRO_JOFFLONG_X1 = 15
R_TILEPRO_JOFFLONG_X1_PLT = 16
R_TILEPRO_IMM8_X0 = 17
R_TILEPRO_IMM8_Y0 = 18
R_TILEPRO_IMM8_X1 = 19
R_TILEPRO_IMM8_Y1 = 20
R_TILEPRO_MT_IMM15_X1 = 21
R_TILEPRO_MF_IMM15_X1 = 22
R_TILEPRO_IMM16_X0 = 23
R_TILEPRO_IMM16_X1 = 24
R_TILEPRO_IMM16_X0_LO = 25
R_TILEPRO_IMM16_X1_LO = 26
R_TILEPRO_IMM16_X0_HI = 27
R_TILEPRO_IMM16_X1_HI = 28
R_TILEPRO_IMM16_X0_HA = 29
R_TILEPRO_IMM16_X1_HA = 30
R_TILEPRO_IMM16_X0_PCREL = 31
R_TILEPRO_IMM16_X1_PCREL = 32
R_TILEPRO_IMM16_X0_LO_PCREL = 33
R_TILEPRO_IMM16_X1_LO_PCREL = 34
R_TILEPRO_IMM16_X0_HI_PCREL = 35
R_TILEPRO_IMM16_X1_HI_PCREL = 36
R_TILEPRO_IMM16_X0_HA_PCREL = 37
R_TILEPRO_IMM16_X1_HA_PCREL = 38
R_TILEPRO_IMM16_X0_GOT = 39
R_TILEPRO_IMM16_X1_GOT = 40
R_TILEPRO_IMM16_X0_GOT_LO = 41
R_TILEPRO_IMM16_X1_GOT_LO = 42
R_TILEPRO_IMM16_X0_GOT_HI = 43
R_TILEPRO_IMM16_X1_GOT_HI = 44
R_TILEPRO_IMM16_X0_GOT_HA = 45
R_TILEPRO_IMM16_X1_GOT_HA = 46
R_TILEPRO_MMSTART_X0 = 47
R_TILEPRO_MMEND_X0 = 48
R_TILEPRO_MMSTART_X1 = 49
R_TILEPRO_MMEND_X1 = 50
R_TILEPRO_SHAMT_X0 = 51
R_TILEPRO_SHAMT_X1 = 52
R_TILEPRO_SHAMT_Y0 = 53
R_TILEPRO_SHAMT_Y1 = 54
R_TILEPRO_DEST_IMM8_X1 = 55
R_TILEPRO_TLS_GD_CALL = 60
R_TILEPRO_IMM8_X0_TLS_GD_ADD = 61
R_TILEPRO_IMM8_X1_TLS_GD_ADD = 62
R_TILEPRO_IMM8_Y0_TLS_GD_ADD = 63
R_TILEPRO_IMM8_Y1_TLS_GD_ADD = 64
R_TILEPRO_TLS_IE_LOAD = 65
R_TILEPRO_IMM16_X0_TLS_GD = 66
R_TILEPRO_IMM16_X1_TLS_GD = 67
R_TILEPRO_IMM16_X0_TLS_GD_LO = 68
R_TILEPRO_IMM16_X1_TLS_GD_LO = 69
R_TILEPRO_IMM16_X0_TLS_GD_HI = 70
R_TILEPRO_IMM16_X1_TLS_GD_HI = 71
R_TILEPRO_IMM16_X0_TLS_GD_HA = 72
R_TILEPRO_IMM16_X1_TLS_GD_HA = 73
R_TILEPRO_IMM16_X0_TLS_IE = 74
R_TILEPRO_IMM16_X1_TLS_IE = 75
R_TILEPRO_IMM16_X0_TLS_IE_LO = 76
R_TILEPRO_IMM16_X1_TLS_IE_LO = 77
R_TILEPRO_IMM16_X0_TLS_IE_HI = 78
R_TILEPRO_IMM16_X1_TLS_IE_HI = 79
R_TILEPRO_IMM16_X0_TLS_IE_HA = 80
R_TILEPRO_IMM16_X1_TLS_IE_HA = 81
R_TILEPRO_TLS_DTPMOD32 = 82
R_TILEPRO_TLS_DTPOFF32 = 83
R_TILEPRO_TLS_TPOFF32 = 84
R_TILEPRO_IMM16_X0_TLS_LE = 85
R_TILEPRO_IMM16_X1_TLS_LE = 86
R_TILEPRO_IMM16_X0_TLS_LE_LO = 87
R_TILEPRO_IMM16_X1_TLS_LE_LO = 88
R_TILEPRO_IMM16_X0_TLS_LE_HI = 89
R_TILEPRO_IMM16_X1_TLS_LE_HI = 90
R_TILEPRO_IMM16_X0_TLS_LE_HA = 91
R_TILEPRO_IMM16_X1_TLS_LE_HA = 92
R_TILEPRO_GNU_VTINHERIT = 128
R_TILEPRO_GNU_VTENTRY = 129
R_TILEPRO_NUM = 130
R_TILEGX_NONE = 0
R_TILEGX_64 = 1
R_TILEGX_32 = 2
R_TILEGX_16 = 3
R_TILEGX_8 = 4
R_TILEGX_64_PCREL = 5
R_TILEGX_32_PCREL = 6
R_TILEGX_16_PCREL = 7
R_TILEGX_8_PCREL = 8
R_TILEGX_HW0 = 9
R_TILEGX_HW1 = 10
R_TILEGX_HW2 = 11
R_TILEGX_HW3 = 12
R_TILEGX_HW0_LAST = 13
R_TILEGX_HW1_LAST = 14
R_TILEGX_HW2_LAST = 15
R_TILEGX_COPY = 16
R_TILEGX_GLOB_DAT = 17
R_TILEGX_JMP_SLOT = 18
R_TILEGX_RELATIVE = 19
R_TILEGX_BROFF_X1 = 20
R_TILEGX_JUMPOFF_X1 = 21
R_TILEGX_JUMPOFF_X1_PLT = 22
R_TILEGX_IMM8_X0 = 23
R_TILEGX_IMM8_Y0 = 24
R_TILEGX_IMM8_X1 = 25
R_TILEGX_IMM8_Y1 = 26
R_TILEGX_DEST_IMM8_X1 = 27
R_TILEGX_MT_IMM14_X1 = 28
R_TILEGX_MF_IMM14_X1 = 29
R_TILEGX_MMSTART_X0 = 30
R_TILEGX_MMEND_X0 = 31
R_TILEGX_SHAMT_X0 = 32
R_TILEGX_SHAMT_X1 = 33
R_TILEGX_SHAMT_Y0 = 34
R_TILEGX_SHAMT_Y1 = 35
R_TILEGX_IMM16_X0_HW0 = 36
R_TILEGX_IMM16_X1_HW0 = 37
R_TILEGX_IMM16_X0_HW1 = 38
R_TILEGX_IMM16_X1_HW1 = 39
R_TILEGX_IMM16_X0_HW2 = 40
R_TILEGX_IMM16_X1_HW2 = 41
R_TILEGX_IMM16_X0_HW3 = 42
R_TILEGX_IMM16_X1_HW3 = 43
R_TILEGX_IMM16_X0_HW0_LAST = 44
R_TILEGX_IMM16_X1_HW0_LAST = 45
R_TILEGX_IMM16_X0_HW1_LAST = 46
R_TILEGX_IMM16_X1_HW1_LAST = 47
R_TILEGX_IMM16_X0_HW2_LAST = 48
R_TILEGX_IMM16_X1_HW2_LAST = 49
R_TILEGX_IMM16_X0_HW0_PCREL = 50
R_TILEGX_IMM16_X1_HW0_PCREL = 51
R_TILEGX_IMM16_X0_HW1_PCREL = 52
R_TILEGX_IMM16_X1_HW1_PCREL = 53
R_TILEGX_IMM16_X0_HW2_PCREL = 54
R_TILEGX_IMM16_X1_HW2_PCREL = 55
R_TILEGX_IMM16_X0_HW3_PCREL = 56
R_TILEGX_IMM16_X1_HW3_PCREL = 57
R_TILEGX_IMM16_X0_HW0_LAST_PCREL = 58
R_TILEGX_IMM16_X1_HW0_LAST_PCREL = 59
R_TILEGX_IMM16_X0_HW1_LAST_PCREL = 60
R_TILEGX_IMM16_X1_HW1_LAST_PCREL = 61
R_TILEGX_IMM16_X0_HW2_LAST_PCREL = 62
R_TILEGX_IMM16_X1_HW2_LAST_PCREL = 63
R_TILEGX_IMM16_X0_HW0_GOT = 64
R_TILEGX_IMM16_X1_HW0_GOT = 65
R_TILEGX_IMM16_X0_HW0_PLT_PCREL = 66
R_TILEGX_IMM16_X1_HW0_PLT_PCREL = 67
R_TILEGX_IMM16_X0_HW1_PLT_PCREL = 68
R_TILEGX_IMM16_X1_HW1_PLT_PCREL = 69
R_TILEGX_IMM16_X0_HW2_PLT_PCREL = 70
R_TILEGX_IMM16_X1_HW2_PLT_PCREL = 71
R_TILEGX_IMM16_X0_HW0_LAST_GOT = 72
R_TILEGX_IMM16_X1_HW0_LAST_GOT = 73
R_TILEGX_IMM16_X0_HW1_LAST_GOT = 74
R_TILEGX_IMM16_X1_HW1_LAST_GOT = 75
R_TILEGX_IMM16_X0_HW3_PLT_PCREL = 76
R_TILEGX_IMM16_X1_HW3_PLT_PCREL = 77
R_TILEGX_IMM16_X0_HW0_TLS_GD = 78
R_TILEGX_IMM16_X1_HW0_TLS_GD = 79
R_TILEGX_IMM16_X0_HW0_TLS_LE = 80
R_TILEGX_IMM16_X1_HW0_TLS_LE = 81
R_TILEGX_IMM16_X0_HW0_LAST_TLS_LE = 82
R_TILEGX_IMM16_X1_HW0_LAST_TLS_LE = 83
R_TILEGX_IMM16_X0_HW1_LAST_TLS_LE = 84
R_TILEGX_IMM16_X1_HW1_LAST_TLS_LE = 85
R_TILEGX_IMM16_X0_HW0_LAST_TLS_GD = 86
R_TILEGX_IMM16_X1_HW0_LAST_TLS_GD = 87
R_TILEGX_IMM16_X0_HW1_LAST_TLS_GD = 88
R_TILEGX_IMM16_X1_HW1_LAST_TLS_GD = 89
R_TILEGX_IMM16_X0_HW0_TLS_IE = 92
R_TILEGX_IMM16_X1_HW0_TLS_IE = 93
R_TILEGX_IMM16_X0_HW0_LAST_PLT_PCREL = 94
R_TILEGX_IMM16_X1_HW0_LAST_PLT_PCREL = 95
R_TILEGX_IMM16_X0_HW1_LAST_PLT_PCREL = 96
R_TILEGX_IMM16_X1_HW1_LAST_PLT_PCREL = 97
R_TILEGX_IMM16_X0_HW2_LAST_PLT_PCREL = 98
R_TILEGX_IMM16_X1_HW2_LAST_PLT_PCREL = 99
R_TILEGX_IMM16_X0_HW0_LAST_TLS_IE = 100
R_TILEGX_IMM16_X1_HW0_LAST_TLS_IE = 101
R_TILEGX_IMM16_X0_HW1_LAST_TLS_IE = 102
R_TILEGX_IMM16_X1_HW1_LAST_TLS_IE = 103
R_TILEGX_TLS_DTPMOD64 = 106
R_TILEGX_TLS_DTPOFF64 = 107
R_TILEGX_TLS_TPOFF64 = 108
R_TILEGX_TLS_DTPMOD32 = 109
R_TILEGX_TLS_DTPOFF32 = 110
R_TILEGX_TLS_TPOFF32 = 111
R_TILEGX_TLS_GD_CALL = 112
R_TILEGX_IMM8_X0_TLS_GD_ADD = 113
R_TILEGX_IMM8_X1_TLS_GD_ADD = 114
R_TILEGX_IMM8_Y0_TLS_GD_ADD = 115
R_TILEGX_IMM8_Y1_TLS_GD_ADD = 116
R_TILEGX_TLS_IE_LOAD = 117
R_TILEGX_IMM8_X0_TLS_ADD = 118
R_TILEGX_IMM8_X1_TLS_ADD = 119
R_TILEGX_IMM8_Y0_TLS_ADD = 120
R_TILEGX_IMM8_Y1_TLS_ADD = 121
R_TILEGX_GNU_VTINHERIT = 128
R_TILEGX_GNU_VTENTRY = 129
R_TILEGX_NUM = 130
EF_RISCV_RVC = 0x0001
EF_RISCV_FLOAT_ABI = 0x0006
EF_RISCV_FLOAT_ABI_SOFT = 0x0000
EF_RISCV_FLOAT_ABI_SINGLE = 0x0002
EF_RISCV_FLOAT_ABI_DOUBLE = 0x0004
EF_RISCV_FLOAT_ABI_QUAD = 0x0006
EF_RISCV_RVE = 0x0008
EF_RISCV_TSO = 0x0010
R_RISCV_NONE = 0
R_RISCV_32 = 1
R_RISCV_64 = 2
R_RISCV_RELATIVE = 3
R_RISCV_COPY = 4
R_RISCV_JUMP_SLOT = 5
R_RISCV_TLS_DTPMOD32 = 6
R_RISCV_TLS_DTPMOD64 = 7
R_RISCV_TLS_DTPREL32 = 8
R_RISCV_TLS_DTPREL64 = 9
R_RISCV_TLS_TPREL32 = 10
R_RISCV_TLS_TPREL64 = 11
R_RISCV_BRANCH = 16
R_RISCV_JAL = 17
R_RISCV_CALL = 18
R_RISCV_CALL_PLT = 19
R_RISCV_GOT_HI20 = 20
R_RISCV_TLS_GOT_HI20 = 21
R_RISCV_TLS_GD_HI20 = 22
R_RISCV_PCREL_HI20 = 23
R_RISCV_PCREL_LO12_I = 24
R_RISCV_PCREL_LO12_S = 25
R_RISCV_HI20 = 26
R_RISCV_LO12_I = 27
R_RISCV_LO12_S = 28
R_RISCV_TPREL_HI20 = 29
R_RISCV_TPREL_LO12_I = 30
R_RISCV_TPREL_LO12_S = 31
R_RISCV_TPREL_ADD = 32
R_RISCV_ADD8 = 33
R_RISCV_ADD16 = 34
R_RISCV_ADD32 = 35
R_RISCV_ADD64 = 36
R_RISCV_SUB8 = 37
R_RISCV_SUB16 = 38
R_RISCV_SUB32 = 39
R_RISCV_SUB64 = 40
R_RISCV_GNU_VTINHERIT = 41
R_RISCV_GNU_VTENTRY = 42
R_RISCV_ALIGN = 43
R_RISCV_RVC_BRANCH = 44
R_RISCV_RVC_JUMP = 45
R_RISCV_RVC_LUI = 46
R_RISCV_GPREL_I = 47
R_RISCV_GPREL_S = 48
R_RISCV_TPREL_I = 49
R_RISCV_TPREL_S = 50
R_RISCV_RELAX = 51
R_RISCV_SUB6 = 52
R_RISCV_SET6 = 53
R_RISCV_SET8 = 54
R_RISCV_SET16 = 55
R_RISCV_SET32 = 56
R_RISCV_32_PCREL = 57
R_RISCV_IRELATIVE = 58
R_RISCV_PLT32 = 59
R_RISCV_SET_ULEB128 = 60
R_RISCV_SUB_ULEB128 = 61
R_RISCV_NUM = 62
STO_RISCV_VARIANT_CC = 0x80
SHT_RISCV_ATTRIBUTES = (SHT_LOPROC + 3)
PT_RISCV_ATTRIBUTES = (PT_LOPROC + 3)
DT_RISCV_VARIANT_CC = (DT_LOPROC + 1)
R_BPF_NONE = 0
R_BPF_64_64 = 1
R_BPF_64_32 = 10
R_METAG_HIADDR16 = 0
R_METAG_LOADDR16 = 1
R_METAG_ADDR32 = 2
R_METAG_NONE = 3
R_METAG_RELBRANCH = 4
R_METAG_GETSETOFF = 5
R_METAG_REG32OP1 = 6
R_METAG_REG32OP2 = 7
R_METAG_REG32OP3 = 8
R_METAG_REG16OP1 = 9
R_METAG_REG16OP2 = 10
R_METAG_REG16OP3 = 11
R_METAG_REG32OP4 = 12
R_METAG_HIOG = 13
R_METAG_LOOG = 14
R_METAG_REL8 = 15
R_METAG_REL16 = 16
R_METAG_GNU_VTINHERIT = 30
R_METAG_GNU_VTENTRY = 31
R_METAG_HI16_GOTOFF = 32
R_METAG_LO16_GOTOFF = 33
R_METAG_GETSET_GOTOFF = 34
R_METAG_GETSET_GOT = 35
R_METAG_HI16_GOTPC = 36
R_METAG_LO16_GOTPC = 37
R_METAG_HI16_PLT = 38
R_METAG_LO16_PLT = 39
R_METAG_RELBRANCH_PLT = 40
R_METAG_GOTOFF = 41
R_METAG_PLT = 42
R_METAG_COPY = 43
R_METAG_JMP_SLOT = 44
R_METAG_RELATIVE = 45
R_METAG_GLOB_DAT = 46
R_METAG_TLS_GD = 47
R_METAG_TLS_LDM = 48
R_METAG_TLS_LDO_HI16 = 49
R_METAG_TLS_LDO_LO16 = 50
R_METAG_TLS_LDO = 51
R_METAG_TLS_IE = 52
R_METAG_TLS_IENONPIC = 53
R_METAG_TLS_IENONPIC_HI16 = 54
R_METAG_TLS_IENONPIC_LO16 = 55
R_METAG_TLS_TPOFF = 56
R_METAG_TLS_DTPMOD = 57
R_METAG_TLS_DTPOFF = 58
R_METAG_TLS_LE = 59
R_METAG_TLS_LE_HI16 = 60
R_METAG_TLS_LE_LO16 = 61
R_NDS32_NONE = 0
R_NDS32_32_RELA = 20
R_NDS32_COPY = 39
R_NDS32_GLOB_DAT = 40
R_NDS32_JMP_SLOT = 41
R_NDS32_RELATIVE = 42
R_NDS32_TLS_TPOFF = 102
R_NDS32_TLS_DESC = 119
EF_LARCH_ABI_MODIFIER_MASK = 0x07
EF_LARCH_ABI_SOFT_FLOAT = 0x01
EF_LARCH_ABI_SINGLE_FLOAT = 0x02
EF_LARCH_ABI_DOUBLE_FLOAT = 0x03
EF_LARCH_OBJABI_V1 = 0x40
R_LARCH_NONE = 0
R_LARCH_32 = 1
R_LARCH_64 = 2
R_LARCH_RELATIVE = 3
R_LARCH_COPY = 4
R_LARCH_JUMP_SLOT = 5
R_LARCH_TLS_DTPMOD32 = 6
R_LARCH_TLS_DTPMOD64 = 7
R_LARCH_TLS_DTPREL32 = 8
R_LARCH_TLS_DTPREL64 = 9
R_LARCH_TLS_TPREL32 = 10
R_LARCH_TLS_TPREL64 = 11
R_LARCH_IRELATIVE = 12
R_LARCH_MARK_LA = 20
R_LARCH_MARK_PCREL = 21
R_LARCH_SOP_PUSH_PCREL = 22
R_LARCH_SOP_PUSH_ABSOLUTE = 23
R_LARCH_SOP_PUSH_DUP = 24
R_LARCH_SOP_PUSH_GPREL = 25
R_LARCH_SOP_PUSH_TLS_TPREL = 26
R_LARCH_SOP_PUSH_TLS_GOT = 27
R_LARCH_SOP_PUSH_TLS_GD = 28
R_LARCH_SOP_PUSH_PLT_PCREL = 29
R_LARCH_SOP_ASSERT = 30
R_LARCH_SOP_NOT = 31
R_LARCH_SOP_SUB = 32
R_LARCH_SOP_SL = 33
R_LARCH_SOP_SR = 34
R_LARCH_SOP_ADD = 35
R_LARCH_SOP_AND = 36
R_LARCH_SOP_IF_ELSE = 37
R_LARCH_SOP_POP_32_S_10_5 = 38
R_LARCH_SOP_POP_32_U_10_12 = 39
R_LARCH_SOP_POP_32_S_10_12 = 40
R_LARCH_SOP_POP_32_S_10_16 = 41
R_LARCH_SOP_POP_32_S_10_16_S2 = 42
R_LARCH_SOP_POP_32_S_5_20 = 43
R_LARCH_SOP_POP_32_S_0_5_10_16_S2 = 44
R_LARCH_SOP_POP_32_S_0_10_10_16_S2 = 45
R_LARCH_SOP_POP_32_U = 46
R_LARCH_ADD8 = 47
R_LARCH_ADD16 = 48
R_LARCH_ADD24 = 49
R_LARCH_ADD32 = 50
R_LARCH_ADD64 = 51
R_LARCH_SUB8 = 52
R_LARCH_SUB16 = 53
R_LARCH_SUB24 = 54
R_LARCH_SUB32 = 55
R_LARCH_SUB64 = 56
R_LARCH_GNU_VTINHERIT = 57
R_LARCH_GNU_VTENTRY = 58
R_LARCH_B16 = 64
R_LARCH_B21 = 65
R_LARCH_B26 = 66
R_LARCH_ABS_HI20 = 67
R_LARCH_ABS_LO12 = 68
R_LARCH_ABS64_LO20 = 69
R_LARCH_ABS64_HI12 = 70
R_LARCH_PCALA_HI20 = 71
R_LARCH_PCALA_LO12 = 72
R_LARCH_PCALA64_LO20 = 73
R_LARCH_PCALA64_HI12 = 74
R_LARCH_GOT_PC_HI20 = 75
R_LARCH_GOT_PC_LO12 = 76
R_LARCH_GOT64_PC_LO20 = 77
R_LARCH_GOT64_PC_HI12 = 78
R_LARCH_GOT_HI20 = 79
R_LARCH_GOT_LO12 = 80
R_LARCH_GOT64_LO20 = 81
R_LARCH_GOT64_HI12 = 82
R_LARCH_TLS_LE_HI20 = 83
R_LARCH_TLS_LE_LO12 = 84
R_LARCH_TLS_LE64_LO20 = 85
R_LARCH_TLS_LE64_HI12 = 86
R_LARCH_TLS_IE_PC_HI20 = 87
R_LARCH_TLS_IE_PC_LO12 = 88
R_LARCH_TLS_IE64_PC_LO20 = 89
R_LARCH_TLS_IE64_PC_HI12 = 90
R_LARCH_TLS_IE_HI20 = 91
R_LARCH_TLS_IE_LO12 = 92
R_LARCH_TLS_IE64_LO20 = 93
R_LARCH_TLS_IE64_HI12 = 94
R_LARCH_TLS_LD_PC_HI20 = 95
R_LARCH_TLS_LD_HI20 = 96
R_LARCH_TLS_GD_PC_HI20 = 97
R_LARCH_TLS_GD_HI20 = 98
R_LARCH_32_PCREL = 99
R_LARCH_RELAX = 100
R_LARCH_DELETE = 101
R_LARCH_ALIGN = 102
R_LARCH_PCREL20_S2 = 103
R_LARCH_CFA = 104
R_LARCH_ADD6 = 105
R_LARCH_SUB6 = 106
R_LARCH_ADD_ULEB128 = 107
R_LARCH_SUB_ULEB128 = 108
R_LARCH_64_PCREL = 109
EF_ARC_MACH_MSK = 0x000000ff
EF_ARC_OSABI_MSK = 0x00000f00
EF_ARC_ALL_MSK = (EF_ARC_MACH_MSK | EF_ARC_OSABI_MSK)
SHT_ARC_ATTRIBUTES = (SHT_LOPROC + 1)
R_ARC_NONE = 0x0
R_ARC_8 = 0x1
R_ARC_16 = 0x2
R_ARC_24 = 0x3
R_ARC_32 = 0x4
R_ARC_B22_PCREL = 0x6
R_ARC_H30 = 0x7
R_ARC_N8 = 0x8
R_ARC_N16 = 0x9
R_ARC_N24 = 0xA
R_ARC_N32 = 0xB
R_ARC_SDA = 0xC
R_ARC_SECTOFF = 0xD
R_ARC_S21H_PCREL = 0xE
R_ARC_S21W_PCREL = 0xF
R_ARC_S25H_PCREL = 0x10
R_ARC_S25W_PCREL = 0x11
R_ARC_SDA32 = 0x12
R_ARC_SDA_LDST = 0x13
R_ARC_SDA_LDST1 = 0x14
R_ARC_SDA_LDST2 = 0x15
R_ARC_SDA16_LD = 0x16
R_ARC_SDA16_LD1 = 0x17
R_ARC_SDA16_LD2 = 0x18
R_ARC_S13_PCREL = 0x19
R_ARC_W = 0x1A
R_ARC_32_ME = 0x1B
R_ARC_N32_ME = 0x1C
R_ARC_SECTOFF_ME = 0x1D
R_ARC_SDA32_ME = 0x1E
R_ARC_W_ME = 0x1F
R_ARC_H30_ME = 0x20
R_ARC_SECTOFF_U8 = 0x21
R_ARC_SECTOFF_S9 = 0x22
R_AC_SECTOFF_U8 = 0x23
R_AC_SECTOFF_U8_1 = 0x24
R_AC_SECTOFF_U8_2 = 0x25
R_AC_SECTOFF_S9 = 0x26
R_AC_SECTOFF_S9_1 = 0x27
R_AC_SECTOFF_S9_2 = 0x28
R_ARC_SECTOFF_ME_1 = 0x29
R_ARC_SECTOFF_ME_2 = 0x2A
R_ARC_SECTOFF_1 = 0x2B
R_ARC_SECTOFF_2 = 0x2C
R_ARC_SDA_12 = 0x2D
R_ARC_SDA16_ST2 = 0x30
R_ARC_32_PCREL = 0x31
R_ARC_PC32 = 0x32
R_ARC_GOTPC32 = 0x33
R_ARC_PLT32 = 0x34
R_ARC_COPY = 0x35
R_ARC_GLOB_DAT = 0x36
R_ARC_JMP_SLOT = 0x37
R_ARC_RELATIVE = 0x38
R_ARC_GOTOFF = 0x39
R_ARC_GOTPC = 0x3A
R_ARC_GOT32 = 0x3B
R_ARC_S21W_PCREL_PLT = 0x3C
R_ARC_S25H_PCREL_PLT = 0x3D
R_ARC_JLI_SECTOFF = 0x3F
R_ARC_TLS_DTPMOD = 0x42
R_ARC_TLS_DTPOFF = 0x43
R_ARC_TLS_TPOFF = 0x44
R_ARC_TLS_GD_GOT = 0x45
R_ARC_TLS_GD_LD = 0x46
R_ARC_TLS_GD_CALL = 0x47
R_ARC_TLS_IE_GOT = 0x48
R_ARC_TLS_DTPOFF_S9 = 0x49
R_ARC_TLS_LE_S9 = 0x4A
R_ARC_TLS_LE_32 = 0x4B
R_ARC_S25W_PCREL_PLT = 0x4C
R_ARC_S21H_PCREL_PLT = 0x4D
R_ARC_NPS_CMEM16 = 0x4E
R_OR1K_NONE = 0
R_OR1K_32 = 1
R_OR1K_16 = 2
R_OR1K_8 = 3
R_OR1K_LO_16_IN_INSN = 4
R_OR1K_HI_16_IN_INSN = 5
R_OR1K_INSN_REL_26 = 6
R_OR1K_GNU_VTENTRY = 7
R_OR1K_GNU_VTINHERIT = 8
R_OR1K_32_PCREL = 9
R_OR1K_16_PCREL = 10
R_OR1K_8_PCREL = 11
R_OR1K_GOTPC_HI16 = 12
R_OR1K_GOTPC_LO16 = 13
R_OR1K_GOT16 = 14
R_OR1K_PLT26 = 15
R_OR1K_GOTOFF_HI16 = 16
R_OR1K_GOTOFF_LO16 = 17
R_OR1K_COPY = 18
R_OR1K_GLOB_DAT = 19
R_OR1K_JMP_SLOT = 20
R_OR1K_RELATIVE = 21
R_OR1K_TLS_GD_HI16 = 22
R_OR1K_TLS_GD_LO16 = 23
R_OR1K_TLS_LDM_HI16 = 24
R_OR1K_TLS_LDM_LO16 = 25
R_OR1K_TLS_LDO_HI16 = 26
R_OR1K_TLS_LDO_LO16 = 27
R_OR1K_TLS_IE_HI16 = 28
R_OR1K_TLS_IE_LO16 = 29
R_OR1K_TLS_LE_HI16 = 30
R_OR1K_TLS_LE_LO16 = 31
R_OR1K_TLS_TPOFF = 32
R_OR1K_TLS_DTPOFF = 33
R_OR1K_TLS_DTPMOD = 34
_UNISTD_H = 1
_POSIX_VERSION = 200809
__POSIX2_THIS_VERSION = 200809
_POSIX2_VERSION = __POSIX2_THIS_VERSION
_POSIX2_C_VERSION = __POSIX2_THIS_VERSION
_POSIX2_C_BIND = __POSIX2_THIS_VERSION
_POSIX2_C_DEV = __POSIX2_THIS_VERSION
_POSIX2_SW_DEV = __POSIX2_THIS_VERSION
_POSIX2_LOCALEDEF = __POSIX2_THIS_VERSION
_XOPEN_VERSION = 700
_XOPEN_XCU_VERSION = 4
_XOPEN_XPG2 = 1
_XOPEN_XPG3 = 1
_XOPEN_XPG4 = 1
_XOPEN_UNIX = 1
_XOPEN_ENH_I18N = 1
_XOPEN_LEGACY = 1
STDIN_FILENO = 0
STDOUT_FILENO = 1
STDERR_FILENO = 2
R_OK = 4
W_OK = 2
X_OK = 1
F_OK = 0
SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2
L_SET = SEEK_SET
L_INCR = SEEK_CUR
L_XTND = SEEK_END
F_ULOCK = 0
F_LOCK = 1
F_TLOCK = 2
F_TEST = 3
PROT_READ = 0x1
PROT_WRITE = 0x2
PROT_EXEC = 0x4
PROT_SEM = 0x8
PROT_NONE = 0x0
PROT_GROWSDOWN = 0x01000000
PROT_GROWSUP = 0x02000000
MAP_TYPE = 0x0f
MAP_FIXED = 0x10
MAP_ANONYMOUS = 0x20
MAP_POPULATE = 0x008000
MAP_NONBLOCK = 0x010000
MAP_STACK = 0x020000
MAP_HUGETLB = 0x040000
MAP_SYNC = 0x080000
MAP_FIXED_NOREPLACE = 0x100000
MAP_UNINITIALIZED = 0x4000000
MLOCK_ONFAULT = 0x01
MS_ASYNC = 1
MS_INVALIDATE = 2
MS_SYNC = 4
MADV_NORMAL = 0
MADV_RANDOM = 1
MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3
MADV_DONTNEED = 4
MADV_FREE = 8
MADV_REMOVE = 9
MADV_DONTFORK = 10
MADV_DOFORK = 11
MADV_HWPOISON = 100
MADV_SOFT_OFFLINE = 101
MADV_MERGEABLE = 12
MADV_UNMERGEABLE = 13
MADV_HUGEPAGE = 14
MADV_NOHUGEPAGE = 15
MADV_DONTDUMP = 16
MADV_DODUMP = 17
MADV_WIPEONFORK = 18
MADV_KEEPONFORK = 19
MADV_COLD = 20
MADV_PAGEOUT = 21
MADV_POPULATE_READ = 22
MADV_POPULATE_WRITE = 23
MADV_DONTNEED_LOCKED = 24
MADV_COLLAPSE = 25
MAP_FILE = 0
PKEY_DISABLE_ACCESS = 0x1
PKEY_DISABLE_WRITE = 0x2
PKEY_ACCESS_MASK = (PKEY_DISABLE_ACCESS | PKEY_DISABLE_WRITE)