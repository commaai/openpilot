# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('libc', 'c', use_errno=True)
off_t: TypeAlias = Annotated[int, ctypes.c_int64]
mode_t: TypeAlias = Annotated[int, ctypes.c_uint32]
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
__off_t: TypeAlias = Annotated[int, ctypes.c_int64]
@dll.bind
def mmap(__addr:ctypes.c_void_p, __len:size_t, __prot:Annotated[int, ctypes.c_int32], __flags:Annotated[int, ctypes.c_int32], __fd:Annotated[int, ctypes.c_int32], __offset:Annotated[int, ctypes.c_int64]) -> ctypes.c_void_p: ...
@dll.bind
def munmap(__addr:ctypes.c_void_p, __len:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def mprotect(__addr:ctypes.c_void_p, __len:size_t, __prot:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def msync(__addr:ctypes.c_void_p, __len:size_t, __flags:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def madvise(__addr:ctypes.c_void_p, __len:size_t, __advice:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def posix_madvise(__addr:ctypes.c_void_p, __len:size_t, __advice:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def mlock(__addr:ctypes.c_void_p, __len:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def munlock(__addr:ctypes.c_void_p, __len:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def mlockall(__flags:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def munlockall() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def mincore(__start:ctypes.c_void_p, __len:size_t, __vec:c.POINTER[Annotated[int, ctypes.c_ubyte]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def shm_open(__name:c.POINTER[Annotated[bytes, ctypes.c_char]], __oflag:Annotated[int, ctypes.c_int32], __mode:mode_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def shm_unlink(__name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def memcpy(__dest:ctypes.c_void_p, __src:ctypes.c_void_p, __n:size_t) -> ctypes.c_void_p: ...
@dll.bind
def memmove(__dest:ctypes.c_void_p, __src:ctypes.c_void_p, __n:size_t) -> ctypes.c_void_p: ...
@dll.bind
def memccpy(__dest:ctypes.c_void_p, __src:ctypes.c_void_p, __c:Annotated[int, ctypes.c_int32], __n:size_t) -> ctypes.c_void_p: ...
@dll.bind
def memset(__s:ctypes.c_void_p, __c:Annotated[int, ctypes.c_int32], __n:size_t) -> ctypes.c_void_p: ...
@dll.bind
def memcmp(__s1:ctypes.c_void_p, __s2:ctypes.c_void_p, __n:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def __memcmpeq(__s1:ctypes.c_void_p, __s2:ctypes.c_void_p, __n:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def memchr(__s:ctypes.c_void_p, __c:Annotated[int, ctypes.c_int32], __n:size_t) -> ctypes.c_void_p: ...
@dll.bind
def strcpy(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strncpy(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strcat(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strncat(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strcmp(__s1:c.POINTER[Annotated[bytes, ctypes.c_char]], __s2:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def strncmp(__s1:c.POINTER[Annotated[bytes, ctypes.c_char]], __s2:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def strcoll(__s1:c.POINTER[Annotated[bytes, ctypes.c_char]], __s2:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def strxfrm(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t) -> Annotated[int, ctypes.c_uint64]: ...
@c.record
class struct___locale_struct(c.Struct):
  SIZE = 232
  __locales: Annotated[c.Array[c.POINTER[struct___locale_data], Literal[13]], 0]
  __ctype_b: Annotated[c.POINTER[Annotated[int, ctypes.c_uint16]], 104]
  __ctype_tolower: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 112]
  __ctype_toupper: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 120]
  __names: Annotated[c.Array[c.POINTER[Annotated[bytes, ctypes.c_char]], Literal[13]], 128]
class struct___locale_data(ctypes.Structure): pass
locale_t: TypeAlias = c.POINTER[struct___locale_struct]
@dll.bind
def strcoll_l(__s1:c.POINTER[Annotated[bytes, ctypes.c_char]], __s2:c.POINTER[Annotated[bytes, ctypes.c_char]], __l:locale_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def strxfrm_l(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t, __l:locale_t) -> size_t: ...
@dll.bind
def strdup(__s:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strndup(__string:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strchr(__s:c.POINTER[Annotated[bytes, ctypes.c_char]], __c:Annotated[int, ctypes.c_int32]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strrchr(__s:c.POINTER[Annotated[bytes, ctypes.c_char]], __c:Annotated[int, ctypes.c_int32]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strchrnul(__s:c.POINTER[Annotated[bytes, ctypes.c_char]], __c:Annotated[int, ctypes.c_int32]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strcspn(__s:c.POINTER[Annotated[bytes, ctypes.c_char]], __reject:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def strspn(__s:c.POINTER[Annotated[bytes, ctypes.c_char]], __accept:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def strpbrk(__s:c.POINTER[Annotated[bytes, ctypes.c_char]], __accept:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strstr(__haystack:c.POINTER[Annotated[bytes, ctypes.c_char]], __needle:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strtok(__s:c.POINTER[Annotated[bytes, ctypes.c_char]], __delim:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def __strtok_r(__s:c.POINTER[Annotated[bytes, ctypes.c_char]], __delim:c.POINTER[Annotated[bytes, ctypes.c_char]], __save_ptr:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strtok_r(__s:c.POINTER[Annotated[bytes, ctypes.c_char]], __delim:c.POINTER[Annotated[bytes, ctypes.c_char]], __save_ptr:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strcasestr(__haystack:c.POINTER[Annotated[bytes, ctypes.c_char]], __needle:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def memmem(__haystack:ctypes.c_void_p, __haystacklen:size_t, __needle:ctypes.c_void_p, __needlelen:size_t) -> ctypes.c_void_p: ...
@dll.bind
def __mempcpy(__dest:ctypes.c_void_p, __src:ctypes.c_void_p, __n:size_t) -> ctypes.c_void_p: ...
@dll.bind
def mempcpy(__dest:ctypes.c_void_p, __src:ctypes.c_void_p, __n:size_t) -> ctypes.c_void_p: ...
@dll.bind
def strlen(__s:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def strnlen(__string:c.POINTER[Annotated[bytes, ctypes.c_char]], __maxlen:size_t) -> size_t: ...
@dll.bind
def strerror(__errnum:Annotated[int, ctypes.c_int32]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strerror_r(__errnum:Annotated[int, ctypes.c_int32], __buf:c.POINTER[Annotated[bytes, ctypes.c_char]], __buflen:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def strerror_l(__errnum:Annotated[int, ctypes.c_int32], __l:locale_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def explicit_bzero(__s:ctypes.c_void_p, __n:size_t) -> None: ...
@dll.bind
def strsep(__stringp:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], __delim:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strsignal(__sig:Annotated[int, ctypes.c_int32]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def __stpcpy(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def stpcpy(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def __stpncpy(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def stpncpy(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def strlcpy(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t) -> size_t: ...
@dll.bind
def strlcat(__dest:c.POINTER[Annotated[bytes, ctypes.c_char]], __src:c.POINTER[Annotated[bytes, ctypes.c_char]], __n:size_t) -> size_t: ...
Elf32_Half: TypeAlias = Annotated[int, ctypes.c_uint16]
Elf64_Half: TypeAlias = Annotated[int, ctypes.c_uint16]
Elf32_Word: TypeAlias = Annotated[int, ctypes.c_uint32]
Elf32_Sword: TypeAlias = Annotated[int, ctypes.c_int32]
Elf64_Word: TypeAlias = Annotated[int, ctypes.c_uint32]
Elf64_Sword: TypeAlias = Annotated[int, ctypes.c_int32]
Elf32_Xword: TypeAlias = Annotated[int, ctypes.c_uint64]
Elf32_Sxword: TypeAlias = Annotated[int, ctypes.c_int64]
Elf64_Xword: TypeAlias = Annotated[int, ctypes.c_uint64]
Elf64_Sxword: TypeAlias = Annotated[int, ctypes.c_int64]
Elf32_Addr: TypeAlias = Annotated[int, ctypes.c_uint32]
Elf64_Addr: TypeAlias = Annotated[int, ctypes.c_uint64]
Elf32_Off: TypeAlias = Annotated[int, ctypes.c_uint32]
Elf64_Off: TypeAlias = Annotated[int, ctypes.c_uint64]
Elf32_Section: TypeAlias = Annotated[int, ctypes.c_uint16]
Elf64_Section: TypeAlias = Annotated[int, ctypes.c_uint16]
Elf32_Versym: TypeAlias = Annotated[int, ctypes.c_uint16]
Elf64_Versym: TypeAlias = Annotated[int, ctypes.c_uint16]
@c.record
class Elf32_Ehdr(c.Struct):
  SIZE = 52
  e_ident: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  e_type: Annotated[Elf32_Half, 16]
  e_machine: Annotated[Elf32_Half, 18]
  e_version: Annotated[Elf32_Word, 20]
  e_entry: Annotated[Elf32_Addr, 24]
  e_phoff: Annotated[Elf32_Off, 28]
  e_shoff: Annotated[Elf32_Off, 32]
  e_flags: Annotated[Elf32_Word, 36]
  e_ehsize: Annotated[Elf32_Half, 40]
  e_phentsize: Annotated[Elf32_Half, 42]
  e_phnum: Annotated[Elf32_Half, 44]
  e_shentsize: Annotated[Elf32_Half, 46]
  e_shnum: Annotated[Elf32_Half, 48]
  e_shstrndx: Annotated[Elf32_Half, 50]
@c.record
class Elf64_Ehdr(c.Struct):
  SIZE = 64
  e_ident: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  e_type: Annotated[Elf64_Half, 16]
  e_machine: Annotated[Elf64_Half, 18]
  e_version: Annotated[Elf64_Word, 20]
  e_entry: Annotated[Elf64_Addr, 24]
  e_phoff: Annotated[Elf64_Off, 32]
  e_shoff: Annotated[Elf64_Off, 40]
  e_flags: Annotated[Elf64_Word, 48]
  e_ehsize: Annotated[Elf64_Half, 52]
  e_phentsize: Annotated[Elf64_Half, 54]
  e_phnum: Annotated[Elf64_Half, 56]
  e_shentsize: Annotated[Elf64_Half, 58]
  e_shnum: Annotated[Elf64_Half, 60]
  e_shstrndx: Annotated[Elf64_Half, 62]
@c.record
class Elf32_Shdr(c.Struct):
  SIZE = 40
  sh_name: Annotated[Elf32_Word, 0]
  sh_type: Annotated[Elf32_Word, 4]
  sh_flags: Annotated[Elf32_Word, 8]
  sh_addr: Annotated[Elf32_Addr, 12]
  sh_offset: Annotated[Elf32_Off, 16]
  sh_size: Annotated[Elf32_Word, 20]
  sh_link: Annotated[Elf32_Word, 24]
  sh_info: Annotated[Elf32_Word, 28]
  sh_addralign: Annotated[Elf32_Word, 32]
  sh_entsize: Annotated[Elf32_Word, 36]
@c.record
class Elf64_Shdr(c.Struct):
  SIZE = 64
  sh_name: Annotated[Elf64_Word, 0]
  sh_type: Annotated[Elf64_Word, 4]
  sh_flags: Annotated[Elf64_Xword, 8]
  sh_addr: Annotated[Elf64_Addr, 16]
  sh_offset: Annotated[Elf64_Off, 24]
  sh_size: Annotated[Elf64_Xword, 32]
  sh_link: Annotated[Elf64_Word, 40]
  sh_info: Annotated[Elf64_Word, 44]
  sh_addralign: Annotated[Elf64_Xword, 48]
  sh_entsize: Annotated[Elf64_Xword, 56]
@c.record
class Elf32_Chdr(c.Struct):
  SIZE = 12
  ch_type: Annotated[Elf32_Word, 0]
  ch_size: Annotated[Elf32_Word, 4]
  ch_addralign: Annotated[Elf32_Word, 8]
@c.record
class Elf64_Chdr(c.Struct):
  SIZE = 24
  ch_type: Annotated[Elf64_Word, 0]
  ch_reserved: Annotated[Elf64_Word, 4]
  ch_size: Annotated[Elf64_Xword, 8]
  ch_addralign: Annotated[Elf64_Xword, 16]
@c.record
class Elf32_Sym(c.Struct):
  SIZE = 16
  st_name: Annotated[Elf32_Word, 0]
  st_value: Annotated[Elf32_Addr, 4]
  st_size: Annotated[Elf32_Word, 8]
  st_info: Annotated[Annotated[int, ctypes.c_ubyte], 12]
  st_other: Annotated[Annotated[int, ctypes.c_ubyte], 13]
  st_shndx: Annotated[Elf32_Section, 14]
@c.record
class Elf64_Sym(c.Struct):
  SIZE = 24
  st_name: Annotated[Elf64_Word, 0]
  st_info: Annotated[Annotated[int, ctypes.c_ubyte], 4]
  st_other: Annotated[Annotated[int, ctypes.c_ubyte], 5]
  st_shndx: Annotated[Elf64_Section, 6]
  st_value: Annotated[Elf64_Addr, 8]
  st_size: Annotated[Elf64_Xword, 16]
@c.record
class Elf32_Syminfo(c.Struct):
  SIZE = 4
  si_boundto: Annotated[Elf32_Half, 0]
  si_flags: Annotated[Elf32_Half, 2]
@c.record
class Elf64_Syminfo(c.Struct):
  SIZE = 4
  si_boundto: Annotated[Elf64_Half, 0]
  si_flags: Annotated[Elf64_Half, 2]
@c.record
class Elf32_Rel(c.Struct):
  SIZE = 8
  r_offset: Annotated[Elf32_Addr, 0]
  r_info: Annotated[Elf32_Word, 4]
@c.record
class Elf64_Rel(c.Struct):
  SIZE = 16
  r_offset: Annotated[Elf64_Addr, 0]
  r_info: Annotated[Elf64_Xword, 8]
@c.record
class Elf32_Rela(c.Struct):
  SIZE = 12
  r_offset: Annotated[Elf32_Addr, 0]
  r_info: Annotated[Elf32_Word, 4]
  r_addend: Annotated[Elf32_Sword, 8]
@c.record
class Elf64_Rela(c.Struct):
  SIZE = 24
  r_offset: Annotated[Elf64_Addr, 0]
  r_info: Annotated[Elf64_Xword, 8]
  r_addend: Annotated[Elf64_Sxword, 16]
Elf32_Relr: TypeAlias = Annotated[int, ctypes.c_uint32]
Elf64_Relr: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class Elf32_Phdr(c.Struct):
  SIZE = 32
  p_type: Annotated[Elf32_Word, 0]
  p_offset: Annotated[Elf32_Off, 4]
  p_vaddr: Annotated[Elf32_Addr, 8]
  p_paddr: Annotated[Elf32_Addr, 12]
  p_filesz: Annotated[Elf32_Word, 16]
  p_memsz: Annotated[Elf32_Word, 20]
  p_flags: Annotated[Elf32_Word, 24]
  p_align: Annotated[Elf32_Word, 28]
@c.record
class Elf64_Phdr(c.Struct):
  SIZE = 56
  p_type: Annotated[Elf64_Word, 0]
  p_flags: Annotated[Elf64_Word, 4]
  p_offset: Annotated[Elf64_Off, 8]
  p_vaddr: Annotated[Elf64_Addr, 16]
  p_paddr: Annotated[Elf64_Addr, 24]
  p_filesz: Annotated[Elf64_Xword, 32]
  p_memsz: Annotated[Elf64_Xword, 40]
  p_align: Annotated[Elf64_Xword, 48]
@c.record
class Elf32_Dyn(c.Struct):
  SIZE = 8
  d_tag: Annotated[Elf32_Sword, 0]
  d_un: Annotated[Elf32_Dyn_d_un, 4]
@c.record
class Elf32_Dyn_d_un(c.Struct):
  SIZE = 4
  d_val: Annotated[Elf32_Word, 0]
  d_ptr: Annotated[Elf32_Addr, 0]
@c.record
class Elf64_Dyn(c.Struct):
  SIZE = 16
  d_tag: Annotated[Elf64_Sxword, 0]
  d_un: Annotated[Elf64_Dyn_d_un, 8]
@c.record
class Elf64_Dyn_d_un(c.Struct):
  SIZE = 8
  d_val: Annotated[Elf64_Xword, 0]
  d_ptr: Annotated[Elf64_Addr, 0]
@c.record
class Elf32_Verdef(c.Struct):
  SIZE = 20
  vd_version: Annotated[Elf32_Half, 0]
  vd_flags: Annotated[Elf32_Half, 2]
  vd_ndx: Annotated[Elf32_Half, 4]
  vd_cnt: Annotated[Elf32_Half, 6]
  vd_hash: Annotated[Elf32_Word, 8]
  vd_aux: Annotated[Elf32_Word, 12]
  vd_next: Annotated[Elf32_Word, 16]
@c.record
class Elf64_Verdef(c.Struct):
  SIZE = 20
  vd_version: Annotated[Elf64_Half, 0]
  vd_flags: Annotated[Elf64_Half, 2]
  vd_ndx: Annotated[Elf64_Half, 4]
  vd_cnt: Annotated[Elf64_Half, 6]
  vd_hash: Annotated[Elf64_Word, 8]
  vd_aux: Annotated[Elf64_Word, 12]
  vd_next: Annotated[Elf64_Word, 16]
@c.record
class Elf32_Verdaux(c.Struct):
  SIZE = 8
  vda_name: Annotated[Elf32_Word, 0]
  vda_next: Annotated[Elf32_Word, 4]
@c.record
class Elf64_Verdaux(c.Struct):
  SIZE = 8
  vda_name: Annotated[Elf64_Word, 0]
  vda_next: Annotated[Elf64_Word, 4]
@c.record
class Elf32_Verneed(c.Struct):
  SIZE = 16
  vn_version: Annotated[Elf32_Half, 0]
  vn_cnt: Annotated[Elf32_Half, 2]
  vn_file: Annotated[Elf32_Word, 4]
  vn_aux: Annotated[Elf32_Word, 8]
  vn_next: Annotated[Elf32_Word, 12]
@c.record
class Elf64_Verneed(c.Struct):
  SIZE = 16
  vn_version: Annotated[Elf64_Half, 0]
  vn_cnt: Annotated[Elf64_Half, 2]
  vn_file: Annotated[Elf64_Word, 4]
  vn_aux: Annotated[Elf64_Word, 8]
  vn_next: Annotated[Elf64_Word, 12]
@c.record
class Elf32_Vernaux(c.Struct):
  SIZE = 16
  vna_hash: Annotated[Elf32_Word, 0]
  vna_flags: Annotated[Elf32_Half, 4]
  vna_other: Annotated[Elf32_Half, 6]
  vna_name: Annotated[Elf32_Word, 8]
  vna_next: Annotated[Elf32_Word, 12]
@c.record
class Elf64_Vernaux(c.Struct):
  SIZE = 16
  vna_hash: Annotated[Elf64_Word, 0]
  vna_flags: Annotated[Elf64_Half, 4]
  vna_other: Annotated[Elf64_Half, 6]
  vna_name: Annotated[Elf64_Word, 8]
  vna_next: Annotated[Elf64_Word, 12]
@c.record
class Elf32_auxv_t(c.Struct):
  SIZE = 8
  a_type: Annotated[uint32_t, 0]
  a_un: Annotated[Elf32_auxv_t_a_un, 4]
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class Elf32_auxv_t_a_un(c.Struct):
  SIZE = 4
  a_val: Annotated[uint32_t, 0]
@c.record
class Elf64_auxv_t(c.Struct):
  SIZE = 16
  a_type: Annotated[uint64_t, 0]
  a_un: Annotated[Elf64_auxv_t_a_un, 8]
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class Elf64_auxv_t_a_un(c.Struct):
  SIZE = 8
  a_val: Annotated[uint64_t, 0]
@c.record
class Elf32_Nhdr(c.Struct):
  SIZE = 12
  n_namesz: Annotated[Elf32_Word, 0]
  n_descsz: Annotated[Elf32_Word, 4]
  n_type: Annotated[Elf32_Word, 8]
@c.record
class Elf64_Nhdr(c.Struct):
  SIZE = 12
  n_namesz: Annotated[Elf64_Word, 0]
  n_descsz: Annotated[Elf64_Word, 4]
  n_type: Annotated[Elf64_Word, 8]
@c.record
class Elf32_Move(c.Struct):
  SIZE = 24
  m_value: Annotated[Elf32_Xword, 0]
  m_info: Annotated[Elf32_Word, 8]
  m_poffset: Annotated[Elf32_Word, 12]
  m_repeat: Annotated[Elf32_Half, 16]
  m_stride: Annotated[Elf32_Half, 18]
@c.record
class Elf64_Move(c.Struct):
  SIZE = 32
  m_value: Annotated[Elf64_Xword, 0]
  m_info: Annotated[Elf64_Xword, 8]
  m_poffset: Annotated[Elf64_Xword, 16]
  m_repeat: Annotated[Elf64_Half, 24]
  m_stride: Annotated[Elf64_Half, 26]
@c.record
class Elf32_gptab(c.Struct):
  SIZE = 8
  gt_header: Annotated[Elf32_gptab_gt_header, 0]
  gt_entry: Annotated[Elf32_gptab_gt_entry, 0]
@c.record
class Elf32_gptab_gt_header(c.Struct):
  SIZE = 8
  gt_current_g_value: Annotated[Elf32_Word, 0]
  gt_unused: Annotated[Elf32_Word, 4]
@c.record
class Elf32_gptab_gt_entry(c.Struct):
  SIZE = 8
  gt_g_value: Annotated[Elf32_Word, 0]
  gt_bytes: Annotated[Elf32_Word, 4]
@c.record
class Elf32_RegInfo(c.Struct):
  SIZE = 24
  ri_gprmask: Annotated[Elf32_Word, 0]
  ri_cprmask: Annotated[c.Array[Elf32_Word, Literal[4]], 4]
  ri_gp_value: Annotated[Elf32_Sword, 20]
@c.record
class Elf_Options(c.Struct):
  SIZE = 8
  kind: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  size: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  section: Annotated[Elf32_Section, 2]
  info: Annotated[Elf32_Word, 4]
@c.record
class Elf_Options_Hw(c.Struct):
  SIZE = 8
  hwp_flags1: Annotated[Elf32_Word, 0]
  hwp_flags2: Annotated[Elf32_Word, 4]
@c.record
class Elf32_Lib(c.Struct):
  SIZE = 20
  l_name: Annotated[Elf32_Word, 0]
  l_time_stamp: Annotated[Elf32_Word, 4]
  l_checksum: Annotated[Elf32_Word, 8]
  l_version: Annotated[Elf32_Word, 12]
  l_flags: Annotated[Elf32_Word, 16]
@c.record
class Elf64_Lib(c.Struct):
  SIZE = 20
  l_name: Annotated[Elf64_Word, 0]
  l_time_stamp: Annotated[Elf64_Word, 4]
  l_checksum: Annotated[Elf64_Word, 8]
  l_version: Annotated[Elf64_Word, 12]
  l_flags: Annotated[Elf64_Word, 16]
Elf32_Conflict: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class Elf_MIPS_ABIFlags_v0(c.Struct):
  SIZE = 24
  version: Annotated[Elf32_Half, 0]
  isa_level: Annotated[Annotated[int, ctypes.c_ubyte], 2]
  isa_rev: Annotated[Annotated[int, ctypes.c_ubyte], 3]
  gpr_size: Annotated[Annotated[int, ctypes.c_ubyte], 4]
  cpr1_size: Annotated[Annotated[int, ctypes.c_ubyte], 5]
  cpr2_size: Annotated[Annotated[int, ctypes.c_ubyte], 6]
  fp_abi: Annotated[Annotated[int, ctypes.c_ubyte], 7]
  isa_ext: Annotated[Elf32_Word, 8]
  ases: Annotated[Elf32_Word, 12]
  flags1: Annotated[Elf32_Word, 16]
  flags2: Annotated[Elf32_Word, 20]
class _anonenum0(Annotated[int, ctypes.c_uint32], c.Enum): pass
Val_GNU_MIPS_ABI_FP_ANY = _anonenum0.define('Val_GNU_MIPS_ABI_FP_ANY', 0)
Val_GNU_MIPS_ABI_FP_DOUBLE = _anonenum0.define('Val_GNU_MIPS_ABI_FP_DOUBLE', 1)
Val_GNU_MIPS_ABI_FP_SINGLE = _anonenum0.define('Val_GNU_MIPS_ABI_FP_SINGLE', 2)
Val_GNU_MIPS_ABI_FP_SOFT = _anonenum0.define('Val_GNU_MIPS_ABI_FP_SOFT', 3)
Val_GNU_MIPS_ABI_FP_OLD_64 = _anonenum0.define('Val_GNU_MIPS_ABI_FP_OLD_64', 4)
Val_GNU_MIPS_ABI_FP_XX = _anonenum0.define('Val_GNU_MIPS_ABI_FP_XX', 5)
Val_GNU_MIPS_ABI_FP_64 = _anonenum0.define('Val_GNU_MIPS_ABI_FP_64', 6)
Val_GNU_MIPS_ABI_FP_64A = _anonenum0.define('Val_GNU_MIPS_ABI_FP_64A', 7)
Val_GNU_MIPS_ABI_FP_MAX = _anonenum0.define('Val_GNU_MIPS_ABI_FP_MAX', 7)

ssize_t: TypeAlias = Annotated[int, ctypes.c_int64]
gid_t: TypeAlias = Annotated[int, ctypes.c_uint32]
uid_t: TypeAlias = Annotated[int, ctypes.c_uint32]
useconds_t: TypeAlias = Annotated[int, ctypes.c_uint32]
pid_t: TypeAlias = Annotated[int, ctypes.c_int32]
intptr_t: TypeAlias = Annotated[int, ctypes.c_int64]
socklen_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@dll.bind
def access(__name:c.POINTER[Annotated[bytes, ctypes.c_char]], __type:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def faccessat(__fd:Annotated[int, ctypes.c_int32], __file:c.POINTER[Annotated[bytes, ctypes.c_char]], __type:Annotated[int, ctypes.c_int32], __flag:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def lseek(__fd:Annotated[int, ctypes.c_int32], __offset:Annotated[int, ctypes.c_int64], __whence:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def close(__fd:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def closefrom(__lowfd:Annotated[int, ctypes.c_int32]) -> None: ...
@dll.bind
def read(__fd:Annotated[int, ctypes.c_int32], __buf:ctypes.c_void_p, __nbytes:size_t) -> ssize_t: ...
@dll.bind
def write(__fd:Annotated[int, ctypes.c_int32], __buf:ctypes.c_void_p, __n:size_t) -> ssize_t: ...
@dll.bind
def pread(__fd:Annotated[int, ctypes.c_int32], __buf:ctypes.c_void_p, __nbytes:size_t, __offset:Annotated[int, ctypes.c_int64]) -> ssize_t: ...
@dll.bind
def pwrite(__fd:Annotated[int, ctypes.c_int32], __buf:ctypes.c_void_p, __n:size_t, __offset:Annotated[int, ctypes.c_int64]) -> ssize_t: ...
@dll.bind
def pipe(__pipedes:c.Array[Annotated[int, ctypes.c_int32], Literal[2]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def alarm(__seconds:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def sleep(__seconds:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_uint32]: ...
__useconds_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@dll.bind
def ualarm(__value:Annotated[int, ctypes.c_uint32], __interval:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def usleep(__useconds:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def pause() -> Annotated[int, ctypes.c_int32]: ...
__uid_t: TypeAlias = Annotated[int, ctypes.c_uint32]
__gid_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@dll.bind
def chown(__file:c.POINTER[Annotated[bytes, ctypes.c_char]], __owner:Annotated[int, ctypes.c_uint32], __group:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def fchown(__fd:Annotated[int, ctypes.c_int32], __owner:Annotated[int, ctypes.c_uint32], __group:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def lchown(__file:c.POINTER[Annotated[bytes, ctypes.c_char]], __owner:Annotated[int, ctypes.c_uint32], __group:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def fchownat(__fd:Annotated[int, ctypes.c_int32], __file:c.POINTER[Annotated[bytes, ctypes.c_char]], __owner:Annotated[int, ctypes.c_uint32], __group:Annotated[int, ctypes.c_uint32], __flag:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def chdir(__path:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def fchdir(__fd:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getcwd(__buf:c.POINTER[Annotated[bytes, ctypes.c_char]], __size:size_t) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def getwd(__buf:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def dup(__fd:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def dup2(__fd:Annotated[int, ctypes.c_int32], __fd2:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
try: __environ = c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]].in_dll(dll, '__environ') # type: ignore
except (ValueError,AttributeError): pass
@dll.bind
def execve(__path:c.POINTER[Annotated[bytes, ctypes.c_char]], __argv:c.Array[c.POINTER[Annotated[bytes, ctypes.c_char]], Literal[0]], __envp:c.Array[c.POINTER[Annotated[bytes, ctypes.c_char]], Literal[0]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def fexecve(__fd:Annotated[int, ctypes.c_int32], __argv:c.Array[c.POINTER[Annotated[bytes, ctypes.c_char]], Literal[0]], __envp:c.Array[c.POINTER[Annotated[bytes, ctypes.c_char]], Literal[0]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def execv(__path:c.POINTER[Annotated[bytes, ctypes.c_char]], __argv:c.Array[c.POINTER[Annotated[bytes, ctypes.c_char]], Literal[0]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def execle(__path:c.POINTER[Annotated[bytes, ctypes.c_char]], __arg:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def execl(__path:c.POINTER[Annotated[bytes, ctypes.c_char]], __arg:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def execvp(__file:c.POINTER[Annotated[bytes, ctypes.c_char]], __argv:c.Array[c.POINTER[Annotated[bytes, ctypes.c_char]], Literal[0]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def execlp(__file:c.POINTER[Annotated[bytes, ctypes.c_char]], __arg:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def nice(__inc:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def _exit(__status:Annotated[int, ctypes.c_int32]) -> None: ...
@dll.bind
def pathconf(__path:c.POINTER[Annotated[bytes, ctypes.c_char]], __name:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def fpathconf(__fd:Annotated[int, ctypes.c_int32], __name:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def sysconf(__name:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def confstr(__name:Annotated[int, ctypes.c_int32], __buf:c.POINTER[Annotated[bytes, ctypes.c_char]], __len:size_t) -> size_t: ...
__pid_t: TypeAlias = Annotated[int, ctypes.c_int32]
@dll.bind
def getpid() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getppid() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getpgrp() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def __getpgid(__pid:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getpgid(__pid:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setpgid(__pid:Annotated[int, ctypes.c_int32], __pgid:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setpgrp() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setsid() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getsid(__pid:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getuid() -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def geteuid() -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def getgid() -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def getegid() -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def getgroups(__size:Annotated[int, ctypes.c_int32], __list:c.Array[Annotated[int, ctypes.c_uint32], Literal[0]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setuid(__uid:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setreuid(__ruid:Annotated[int, ctypes.c_uint32], __euid:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def seteuid(__uid:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setgid(__gid:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setregid(__rgid:Annotated[int, ctypes.c_uint32], __egid:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setegid(__gid:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def fork() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def vfork() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ttyname(__fd:Annotated[int, ctypes.c_int32]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def ttyname_r(__fd:Annotated[int, ctypes.c_int32], __buf:c.POINTER[Annotated[bytes, ctypes.c_char]], __buflen:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def isatty(__fd:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ttyslot() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def link(__from:c.POINTER[Annotated[bytes, ctypes.c_char]], __to:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def linkat(__fromfd:Annotated[int, ctypes.c_int32], __from:c.POINTER[Annotated[bytes, ctypes.c_char]], __tofd:Annotated[int, ctypes.c_int32], __to:c.POINTER[Annotated[bytes, ctypes.c_char]], __flags:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def symlink(__from:c.POINTER[Annotated[bytes, ctypes.c_char]], __to:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def readlink(__path:c.POINTER[Annotated[bytes, ctypes.c_char]], __buf:c.POINTER[Annotated[bytes, ctypes.c_char]], __len:size_t) -> ssize_t: ...
@dll.bind
def symlinkat(__from:c.POINTER[Annotated[bytes, ctypes.c_char]], __tofd:Annotated[int, ctypes.c_int32], __to:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def readlinkat(__fd:Annotated[int, ctypes.c_int32], __path:c.POINTER[Annotated[bytes, ctypes.c_char]], __buf:c.POINTER[Annotated[bytes, ctypes.c_char]], __len:size_t) -> ssize_t: ...
@dll.bind
def unlink(__name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def unlinkat(__fd:Annotated[int, ctypes.c_int32], __name:c.POINTER[Annotated[bytes, ctypes.c_char]], __flag:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def rmdir(__path:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def tcgetpgrp(__fd:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def tcsetpgrp(__fd:Annotated[int, ctypes.c_int32], __pgrp_id:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getlogin() -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def getlogin_r(__name:c.POINTER[Annotated[bytes, ctypes.c_char]], __name_len:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setlogin(__name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def gethostname(__name:c.POINTER[Annotated[bytes, ctypes.c_char]], __len:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def sethostname(__name:c.POINTER[Annotated[bytes, ctypes.c_char]], __len:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def sethostid(__id:Annotated[int, ctypes.c_int64]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getdomainname(__name:c.POINTER[Annotated[bytes, ctypes.c_char]], __len:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def setdomainname(__name:c.POINTER[Annotated[bytes, ctypes.c_char]], __len:size_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def vhangup() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def revoke(__file:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def profil(__sample_buffer:c.POINTER[Annotated[int, ctypes.c_uint16]], __size:size_t, __offset:size_t, __scale:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def acct(__name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getusershell() -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def endusershell() -> None: ...
@dll.bind
def setusershell() -> None: ...
@dll.bind
def daemon(__nochdir:Annotated[int, ctypes.c_int32], __noclose:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def chroot(__path:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getpass(__prompt:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def fsync(__fd:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def gethostid() -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def sync() -> None: ...
@dll.bind
def getpagesize() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def getdtablesize() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def truncate(__file:c.POINTER[Annotated[bytes, ctypes.c_char]], __length:Annotated[int, ctypes.c_int64]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ftruncate(__fd:Annotated[int, ctypes.c_int32], __length:Annotated[int, ctypes.c_int64]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def brk(__addr:ctypes.c_void_p) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def sbrk(__delta:intptr_t) -> ctypes.c_void_p: ...
@dll.bind
def syscall(__sysno:Annotated[int, ctypes.c_int64]) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def lockf(__fd:Annotated[int, ctypes.c_int32], __cmd:Annotated[int, ctypes.c_int32], __len:Annotated[int, ctypes.c_int64]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def fdatasync(__fildes:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def crypt(__key:c.POINTER[Annotated[bytes, ctypes.c_char]], __salt:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def getentropy(__buffer:ctypes.c_void_p, __length:size_t) -> Annotated[int, ctypes.c_int32]: ...
c.init_records()
_SYS_MMAN_H = 1 # type: ignore
_SYSCALL_H = 1 # type: ignore
_STRING_H = 1 # type: ignore
_ELF_H = 1 # type: ignore
EI_NIDENT = (16) # type: ignore
EI_MAG0 = 0 # type: ignore
ELFMAG0 = 0x7f # type: ignore
EI_MAG1 = 1 # type: ignore
ELFMAG1 = 'E' # type: ignore
EI_MAG2 = 2 # type: ignore
ELFMAG2 = 'L' # type: ignore
EI_MAG3 = 3 # type: ignore
ELFMAG3 = 'F' # type: ignore
ELFMAG = "\177ELF" # type: ignore
SELFMAG = 4 # type: ignore
EI_CLASS = 4 # type: ignore
ELFCLASSNONE = 0 # type: ignore
ELFCLASS32 = 1 # type: ignore
ELFCLASS64 = 2 # type: ignore
ELFCLASSNUM = 3 # type: ignore
EI_DATA = 5 # type: ignore
ELFDATANONE = 0 # type: ignore
ELFDATA2LSB = 1 # type: ignore
ELFDATA2MSB = 2 # type: ignore
ELFDATANUM = 3 # type: ignore
EI_VERSION = 6 # type: ignore
EI_OSABI = 7 # type: ignore
ELFOSABI_NONE = 0 # type: ignore
ELFOSABI_SYSV = 0 # type: ignore
ELFOSABI_HPUX = 1 # type: ignore
ELFOSABI_NETBSD = 2 # type: ignore
ELFOSABI_GNU = 3 # type: ignore
ELFOSABI_LINUX = ELFOSABI_GNU # type: ignore
ELFOSABI_SOLARIS = 6 # type: ignore
ELFOSABI_AIX = 7 # type: ignore
ELFOSABI_IRIX = 8 # type: ignore
ELFOSABI_FREEBSD = 9 # type: ignore
ELFOSABI_TRU64 = 10 # type: ignore
ELFOSABI_MODESTO = 11 # type: ignore
ELFOSABI_OPENBSD = 12 # type: ignore
ELFOSABI_ARM_AEABI = 64 # type: ignore
ELFOSABI_ARM = 97 # type: ignore
ELFOSABI_STANDALONE = 255 # type: ignore
EI_ABIVERSION = 8 # type: ignore
EI_PAD = 9 # type: ignore
ET_NONE = 0 # type: ignore
ET_REL = 1 # type: ignore
ET_EXEC = 2 # type: ignore
ET_DYN = 3 # type: ignore
ET_CORE = 4 # type: ignore
ET_NUM = 5 # type: ignore
ET_LOOS = 0xfe00 # type: ignore
ET_HIOS = 0xfeff # type: ignore
ET_LOPROC = 0xff00 # type: ignore
ET_HIPROC = 0xffff # type: ignore
EM_NONE = 0 # type: ignore
EM_M32 = 1 # type: ignore
EM_SPARC = 2 # type: ignore
EM_386 = 3 # type: ignore
EM_68K = 4 # type: ignore
EM_88K = 5 # type: ignore
EM_IAMCU = 6 # type: ignore
EM_860 = 7 # type: ignore
EM_MIPS = 8 # type: ignore
EM_S370 = 9 # type: ignore
EM_MIPS_RS3_LE = 10 # type: ignore
EM_PARISC = 15 # type: ignore
EM_VPP500 = 17 # type: ignore
EM_SPARC32PLUS = 18 # type: ignore
EM_960 = 19 # type: ignore
EM_PPC = 20 # type: ignore
EM_PPC64 = 21 # type: ignore
EM_S390 = 22 # type: ignore
EM_SPU = 23 # type: ignore
EM_V800 = 36 # type: ignore
EM_FR20 = 37 # type: ignore
EM_RH32 = 38 # type: ignore
EM_RCE = 39 # type: ignore
EM_ARM = 40 # type: ignore
EM_FAKE_ALPHA = 41 # type: ignore
EM_SH = 42 # type: ignore
EM_SPARCV9 = 43 # type: ignore
EM_TRICORE = 44 # type: ignore
EM_ARC = 45 # type: ignore
EM_H8_300 = 46 # type: ignore
EM_H8_300H = 47 # type: ignore
EM_H8S = 48 # type: ignore
EM_H8_500 = 49 # type: ignore
EM_IA_64 = 50 # type: ignore
EM_MIPS_X = 51 # type: ignore
EM_COLDFIRE = 52 # type: ignore
EM_68HC12 = 53 # type: ignore
EM_MMA = 54 # type: ignore
EM_PCP = 55 # type: ignore
EM_NCPU = 56 # type: ignore
EM_NDR1 = 57 # type: ignore
EM_STARCORE = 58 # type: ignore
EM_ME16 = 59 # type: ignore
EM_ST100 = 60 # type: ignore
EM_TINYJ = 61 # type: ignore
EM_X86_64 = 62 # type: ignore
EM_PDSP = 63 # type: ignore
EM_PDP10 = 64 # type: ignore
EM_PDP11 = 65 # type: ignore
EM_FX66 = 66 # type: ignore
EM_ST9PLUS = 67 # type: ignore
EM_ST7 = 68 # type: ignore
EM_68HC16 = 69 # type: ignore
EM_68HC11 = 70 # type: ignore
EM_68HC08 = 71 # type: ignore
EM_68HC05 = 72 # type: ignore
EM_SVX = 73 # type: ignore
EM_ST19 = 74 # type: ignore
EM_VAX = 75 # type: ignore
EM_CRIS = 76 # type: ignore
EM_JAVELIN = 77 # type: ignore
EM_FIREPATH = 78 # type: ignore
EM_ZSP = 79 # type: ignore
EM_MMIX = 80 # type: ignore
EM_HUANY = 81 # type: ignore
EM_PRISM = 82 # type: ignore
EM_AVR = 83 # type: ignore
EM_FR30 = 84 # type: ignore
EM_D10V = 85 # type: ignore
EM_D30V = 86 # type: ignore
EM_V850 = 87 # type: ignore
EM_M32R = 88 # type: ignore
EM_MN10300 = 89 # type: ignore
EM_MN10200 = 90 # type: ignore
EM_PJ = 91 # type: ignore
EM_OPENRISC = 92 # type: ignore
EM_ARC_COMPACT = 93 # type: ignore
EM_XTENSA = 94 # type: ignore
EM_VIDEOCORE = 95 # type: ignore
EM_TMM_GPP = 96 # type: ignore
EM_NS32K = 97 # type: ignore
EM_TPC = 98 # type: ignore
EM_SNP1K = 99 # type: ignore
EM_ST200 = 100 # type: ignore
EM_IP2K = 101 # type: ignore
EM_MAX = 102 # type: ignore
EM_CR = 103 # type: ignore
EM_F2MC16 = 104 # type: ignore
EM_MSP430 = 105 # type: ignore
EM_BLACKFIN = 106 # type: ignore
EM_SE_C33 = 107 # type: ignore
EM_SEP = 108 # type: ignore
EM_ARCA = 109 # type: ignore
EM_UNICORE = 110 # type: ignore
EM_EXCESS = 111 # type: ignore
EM_DXP = 112 # type: ignore
EM_ALTERA_NIOS2 = 113 # type: ignore
EM_CRX = 114 # type: ignore
EM_XGATE = 115 # type: ignore
EM_C166 = 116 # type: ignore
EM_M16C = 117 # type: ignore
EM_DSPIC30F = 118 # type: ignore
EM_CE = 119 # type: ignore
EM_M32C = 120 # type: ignore
EM_TSK3000 = 131 # type: ignore
EM_RS08 = 132 # type: ignore
EM_SHARC = 133 # type: ignore
EM_ECOG2 = 134 # type: ignore
EM_SCORE7 = 135 # type: ignore
EM_DSP24 = 136 # type: ignore
EM_VIDEOCORE3 = 137 # type: ignore
EM_LATTICEMICO32 = 138 # type: ignore
EM_SE_C17 = 139 # type: ignore
EM_TI_C6000 = 140 # type: ignore
EM_TI_C2000 = 141 # type: ignore
EM_TI_C5500 = 142 # type: ignore
EM_TI_ARP32 = 143 # type: ignore
EM_TI_PRU = 144 # type: ignore
EM_MMDSP_PLUS = 160 # type: ignore
EM_CYPRESS_M8C = 161 # type: ignore
EM_R32C = 162 # type: ignore
EM_TRIMEDIA = 163 # type: ignore
EM_QDSP6 = 164 # type: ignore
EM_8051 = 165 # type: ignore
EM_STXP7X = 166 # type: ignore
EM_NDS32 = 167 # type: ignore
EM_ECOG1X = 168 # type: ignore
EM_MAXQ30 = 169 # type: ignore
EM_XIMO16 = 170 # type: ignore
EM_MANIK = 171 # type: ignore
EM_CRAYNV2 = 172 # type: ignore
EM_RX = 173 # type: ignore
EM_METAG = 174 # type: ignore
EM_MCST_ELBRUS = 175 # type: ignore
EM_ECOG16 = 176 # type: ignore
EM_CR16 = 177 # type: ignore
EM_ETPU = 178 # type: ignore
EM_SLE9X = 179 # type: ignore
EM_L10M = 180 # type: ignore
EM_K10M = 181 # type: ignore
EM_AARCH64 = 183 # type: ignore
EM_AVR32 = 185 # type: ignore
EM_STM8 = 186 # type: ignore
EM_TILE64 = 187 # type: ignore
EM_TILEPRO = 188 # type: ignore
EM_MICROBLAZE = 189 # type: ignore
EM_CUDA = 190 # type: ignore
EM_TILEGX = 191 # type: ignore
EM_CLOUDSHIELD = 192 # type: ignore
EM_COREA_1ST = 193 # type: ignore
EM_COREA_2ND = 194 # type: ignore
EM_ARCV2 = 195 # type: ignore
EM_OPEN8 = 196 # type: ignore
EM_RL78 = 197 # type: ignore
EM_VIDEOCORE5 = 198 # type: ignore
EM_78KOR = 199 # type: ignore
EM_56800EX = 200 # type: ignore
EM_BA1 = 201 # type: ignore
EM_BA2 = 202 # type: ignore
EM_XCORE = 203 # type: ignore
EM_MCHP_PIC = 204 # type: ignore
EM_INTELGT = 205 # type: ignore
EM_KM32 = 210 # type: ignore
EM_KMX32 = 211 # type: ignore
EM_EMX16 = 212 # type: ignore
EM_EMX8 = 213 # type: ignore
EM_KVARC = 214 # type: ignore
EM_CDP = 215 # type: ignore
EM_COGE = 216 # type: ignore
EM_COOL = 217 # type: ignore
EM_NORC = 218 # type: ignore
EM_CSR_KALIMBA = 219 # type: ignore
EM_Z80 = 220 # type: ignore
EM_VISIUM = 221 # type: ignore
EM_FT32 = 222 # type: ignore
EM_MOXIE = 223 # type: ignore
EM_AMDGPU = 224 # type: ignore
EM_RISCV = 243 # type: ignore
EM_BPF = 247 # type: ignore
EM_CSKY = 252 # type: ignore
EM_LOONGARCH = 258 # type: ignore
EM_NUM = 259 # type: ignore
EM_ARC_A5 = EM_ARC_COMPACT # type: ignore
EM_ALPHA = 0x9026 # type: ignore
EV_NONE = 0 # type: ignore
EV_CURRENT = 1 # type: ignore
EV_NUM = 2 # type: ignore
SHN_UNDEF = 0 # type: ignore
SHN_LORESERVE = 0xff00 # type: ignore
SHN_LOPROC = 0xff00 # type: ignore
SHN_BEFORE = 0xff00 # type: ignore
SHN_AFTER = 0xff01 # type: ignore
SHN_HIPROC = 0xff1f # type: ignore
SHN_LOOS = 0xff20 # type: ignore
SHN_HIOS = 0xff3f # type: ignore
SHN_ABS = 0xfff1 # type: ignore
SHN_COMMON = 0xfff2 # type: ignore
SHN_XINDEX = 0xffff # type: ignore
SHN_HIRESERVE = 0xffff # type: ignore
SHT_NULL = 0 # type: ignore
SHT_PROGBITS = 1 # type: ignore
SHT_SYMTAB = 2 # type: ignore
SHT_STRTAB = 3 # type: ignore
SHT_RELA = 4 # type: ignore
SHT_HASH = 5 # type: ignore
SHT_DYNAMIC = 6 # type: ignore
SHT_NOTE = 7 # type: ignore
SHT_NOBITS = 8 # type: ignore
SHT_REL = 9 # type: ignore
SHT_SHLIB = 10 # type: ignore
SHT_DYNSYM = 11 # type: ignore
SHT_INIT_ARRAY = 14 # type: ignore
SHT_FINI_ARRAY = 15 # type: ignore
SHT_PREINIT_ARRAY = 16 # type: ignore
SHT_GROUP = 17 # type: ignore
SHT_SYMTAB_SHNDX = 18 # type: ignore
SHT_RELR = 19 # type: ignore
SHT_NUM = 20 # type: ignore
SHT_LOOS = 0x60000000 # type: ignore
SHT_GNU_ATTRIBUTES = 0x6ffffff5 # type: ignore
SHT_GNU_HASH = 0x6ffffff6 # type: ignore
SHT_GNU_LIBLIST = 0x6ffffff7 # type: ignore
SHT_CHECKSUM = 0x6ffffff8 # type: ignore
SHT_LOSUNW = 0x6ffffffa # type: ignore
SHT_SUNW_move = 0x6ffffffa # type: ignore
SHT_SUNW_COMDAT = 0x6ffffffb # type: ignore
SHT_SUNW_syminfo = 0x6ffffffc # type: ignore
SHT_GNU_verdef = 0x6ffffffd # type: ignore
SHT_GNU_verneed = 0x6ffffffe # type: ignore
SHT_GNU_versym = 0x6fffffff # type: ignore
SHT_HISUNW = 0x6fffffff # type: ignore
SHT_HIOS = 0x6fffffff # type: ignore
SHT_LOPROC = 0x70000000 # type: ignore
SHT_HIPROC = 0x7fffffff # type: ignore
SHT_LOUSER = 0x80000000 # type: ignore
SHT_HIUSER = 0x8fffffff # type: ignore
SHF_WRITE = (1 << 0) # type: ignore
SHF_ALLOC = (1 << 1) # type: ignore
SHF_EXECINSTR = (1 << 2) # type: ignore
SHF_MERGE = (1 << 4) # type: ignore
SHF_STRINGS = (1 << 5) # type: ignore
SHF_INFO_LINK = (1 << 6) # type: ignore
SHF_LINK_ORDER = (1 << 7) # type: ignore
SHF_OS_NONCONFORMING = (1 << 8) # type: ignore
SHF_GROUP = (1 << 9) # type: ignore
SHF_TLS = (1 << 10) # type: ignore
SHF_COMPRESSED = (1 << 11) # type: ignore
SHF_MASKOS = 0x0ff00000 # type: ignore
SHF_MASKPROC = 0xf0000000 # type: ignore
SHF_GNU_RETAIN = (1 << 21) # type: ignore
SHF_ORDERED = (1 << 30) # type: ignore
SHF_EXCLUDE = (1 << 31) # type: ignore
ELFCOMPRESS_ZLIB = 1 # type: ignore
ELFCOMPRESS_ZSTD = 2 # type: ignore
ELFCOMPRESS_LOOS = 0x60000000 # type: ignore
ELFCOMPRESS_HIOS = 0x6fffffff # type: ignore
ELFCOMPRESS_LOPROC = 0x70000000 # type: ignore
ELFCOMPRESS_HIPROC = 0x7fffffff # type: ignore
GRP_COMDAT = 0x1 # type: ignore
SYMINFO_BT_SELF = 0xffff # type: ignore
SYMINFO_BT_PARENT = 0xfffe # type: ignore
SYMINFO_BT_LOWRESERVE = 0xff00 # type: ignore
SYMINFO_FLG_DIRECT = 0x0001 # type: ignore
SYMINFO_FLG_PASSTHRU = 0x0002 # type: ignore
SYMINFO_FLG_COPY = 0x0004 # type: ignore
SYMINFO_FLG_LAZYLOAD = 0x0008 # type: ignore
SYMINFO_NONE = 0 # type: ignore
SYMINFO_CURRENT = 1 # type: ignore
SYMINFO_NUM = 2 # type: ignore
ELF32_ST_BIND = lambda val: (( (val)) >> 4) # type: ignore
ELF32_ST_TYPE = lambda val: ((val) & 0xf) # type: ignore
ELF32_ST_INFO = lambda bind,type: (((bind) << 4) + ((type) & 0xf)) # type: ignore
ELF64_ST_BIND = lambda val: ELF32_ST_BIND (val) # type: ignore
ELF64_ST_TYPE = lambda val: ELF32_ST_TYPE (val) # type: ignore
ELF64_ST_INFO = lambda bind,type: ELF32_ST_INFO ((bind), (type)) # type: ignore
STB_LOCAL = 0 # type: ignore
STB_GLOBAL = 1 # type: ignore
STB_WEAK = 2 # type: ignore
STB_NUM = 3 # type: ignore
STB_LOOS = 10 # type: ignore
STB_GNU_UNIQUE = 10 # type: ignore
STB_HIOS = 12 # type: ignore
STB_LOPROC = 13 # type: ignore
STB_HIPROC = 15 # type: ignore
STT_NOTYPE = 0 # type: ignore
STT_OBJECT = 1 # type: ignore
STT_FUNC = 2 # type: ignore
STT_SECTION = 3 # type: ignore
STT_FILE = 4 # type: ignore
STT_COMMON = 5 # type: ignore
STT_TLS = 6 # type: ignore
STT_NUM = 7 # type: ignore
STT_LOOS = 10 # type: ignore
STT_GNU_IFUNC = 10 # type: ignore
STT_HIOS = 12 # type: ignore
STT_LOPROC = 13 # type: ignore
STT_HIPROC = 15 # type: ignore
STN_UNDEF = 0 # type: ignore
ELF32_ST_VISIBILITY = lambda o: ((o) & 0x03) # type: ignore
ELF64_ST_VISIBILITY = lambda o: ELF32_ST_VISIBILITY (o) # type: ignore
STV_DEFAULT = 0 # type: ignore
STV_INTERNAL = 1 # type: ignore
STV_HIDDEN = 2 # type: ignore
STV_PROTECTED = 3 # type: ignore
ELF32_R_SYM = lambda val: ((val) >> 8) # type: ignore
ELF32_R_TYPE = lambda val: ((val) & 0xff) # type: ignore
ELF32_R_INFO = lambda sym,type: (((sym) << 8) + ((type) & 0xff)) # type: ignore
ELF64_R_SYM = lambda i: ((i) >> 32) # type: ignore
ELF64_R_TYPE = lambda i: ((i) & 0xffffffff) # type: ignore
ELF64_R_INFO = lambda sym,type: ((((Elf64_Xword) (sym)) << 32) + (type)) # type: ignore
PN_XNUM = 0xffff # type: ignore
PT_NULL = 0 # type: ignore
PT_LOAD = 1 # type: ignore
PT_DYNAMIC = 2 # type: ignore
PT_INTERP = 3 # type: ignore
PT_NOTE = 4 # type: ignore
PT_SHLIB = 5 # type: ignore
PT_PHDR = 6 # type: ignore
PT_TLS = 7 # type: ignore
PT_NUM = 8 # type: ignore
PT_LOOS = 0x60000000 # type: ignore
PT_GNU_EH_FRAME = 0x6474e550 # type: ignore
PT_GNU_STACK = 0x6474e551 # type: ignore
PT_GNU_RELRO = 0x6474e552 # type: ignore
PT_GNU_PROPERTY = 0x6474e553 # type: ignore
PT_GNU_SFRAME = 0x6474e554 # type: ignore
PT_LOSUNW = 0x6ffffffa # type: ignore
PT_SUNWBSS = 0x6ffffffa # type: ignore
PT_SUNWSTACK = 0x6ffffffb # type: ignore
PT_HISUNW = 0x6fffffff # type: ignore
PT_HIOS = 0x6fffffff # type: ignore
PT_LOPROC = 0x70000000 # type: ignore
PT_HIPROC = 0x7fffffff # type: ignore
PF_X = (1 << 0) # type: ignore
PF_W = (1 << 1) # type: ignore
PF_R = (1 << 2) # type: ignore
PF_MASKOS = 0x0ff00000 # type: ignore
PF_MASKPROC = 0xf0000000 # type: ignore
NT_PRSTATUS = 1 # type: ignore
NT_PRFPREG = 2 # type: ignore
NT_FPREGSET = 2 # type: ignore
NT_PRPSINFO = 3 # type: ignore
NT_PRXREG = 4 # type: ignore
NT_TASKSTRUCT = 4 # type: ignore
NT_PLATFORM = 5 # type: ignore
NT_AUXV = 6 # type: ignore
NT_GWINDOWS = 7 # type: ignore
NT_ASRS = 8 # type: ignore
NT_PSTATUS = 10 # type: ignore
NT_PSINFO = 13 # type: ignore
NT_PRCRED = 14 # type: ignore
NT_UTSNAME = 15 # type: ignore
NT_LWPSTATUS = 16 # type: ignore
NT_LWPSINFO = 17 # type: ignore
NT_PRFPXREG = 20 # type: ignore
NT_SIGINFO = 0x53494749 # type: ignore
NT_FILE = 0x46494c45 # type: ignore
NT_PRXFPREG = 0x46e62b7f # type: ignore
NT_PPC_VMX = 0x100 # type: ignore
NT_PPC_SPE = 0x101 # type: ignore
NT_PPC_VSX = 0x102 # type: ignore
NT_PPC_TAR = 0x103 # type: ignore
NT_PPC_PPR = 0x104 # type: ignore
NT_PPC_DSCR = 0x105 # type: ignore
NT_PPC_EBB = 0x106 # type: ignore
NT_PPC_PMU = 0x107 # type: ignore
NT_PPC_TM_CGPR = 0x108 # type: ignore
NT_PPC_TM_CFPR = 0x109 # type: ignore
NT_PPC_TM_CVMX = 0x10a # type: ignore
NT_PPC_TM_CVSX = 0x10b # type: ignore
NT_PPC_TM_SPR = 0x10c # type: ignore
NT_PPC_TM_CTAR = 0x10d # type: ignore
NT_PPC_TM_CPPR = 0x10e # type: ignore
NT_PPC_TM_CDSCR = 0x10f # type: ignore
NT_PPC_PKEY = 0x110 # type: ignore
NT_PPC_DEXCR = 0x111 # type: ignore
NT_PPC_HASHKEYR = 0x112 # type: ignore
NT_386_TLS = 0x200 # type: ignore
NT_386_IOPERM = 0x201 # type: ignore
NT_X86_XSTATE = 0x202 # type: ignore
NT_X86_SHSTK = 0x204 # type: ignore
NT_S390_HIGH_GPRS = 0x300 # type: ignore
NT_S390_TIMER = 0x301 # type: ignore
NT_S390_TODCMP = 0x302 # type: ignore
NT_S390_TODPREG = 0x303 # type: ignore
NT_S390_CTRS = 0x304 # type: ignore
NT_S390_PREFIX = 0x305 # type: ignore
NT_S390_LAST_BREAK = 0x306 # type: ignore
NT_S390_SYSTEM_CALL = 0x307 # type: ignore
NT_S390_TDB = 0x308 # type: ignore
NT_S390_VXRS_LOW = 0x309 # type: ignore
NT_S390_VXRS_HIGH = 0x30a # type: ignore
NT_S390_GS_CB = 0x30b # type: ignore
NT_S390_GS_BC = 0x30c # type: ignore
NT_S390_RI_CB = 0x30d # type: ignore
NT_S390_PV_CPU_DATA = 0x30e # type: ignore
NT_ARM_VFP = 0x400 # type: ignore
NT_ARM_TLS = 0x401 # type: ignore
NT_ARM_HW_BREAK = 0x402 # type: ignore
NT_ARM_HW_WATCH = 0x403 # type: ignore
NT_ARM_SYSTEM_CALL = 0x404 # type: ignore
NT_ARM_SVE = 0x405 # type: ignore
NT_ARM_PAC_MASK = 0x406 # type: ignore
NT_ARM_PACA_KEYS = 0x407 # type: ignore
NT_ARM_PACG_KEYS = 0x408 # type: ignore
NT_ARM_TAGGED_ADDR_CTRL = 0x409 # type: ignore
NT_ARM_PAC_ENABLED_KEYS = 0x40a # type: ignore
NT_VMCOREDD = 0x700 # type: ignore
NT_MIPS_DSP = 0x800 # type: ignore
NT_MIPS_FP_MODE = 0x801 # type: ignore
NT_MIPS_MSA = 0x802 # type: ignore
NT_RISCV_CSR = 0x900 # type: ignore
NT_RISCV_VECTOR = 0x901 # type: ignore
NT_LOONGARCH_CPUCFG = 0xa00 # type: ignore
NT_LOONGARCH_CSR = 0xa01 # type: ignore
NT_LOONGARCH_LSX = 0xa02 # type: ignore
NT_LOONGARCH_LASX = 0xa03 # type: ignore
NT_LOONGARCH_LBT = 0xa04 # type: ignore
NT_LOONGARCH_HW_BREAK = 0xa05 # type: ignore
NT_LOONGARCH_HW_WATCH = 0xa06 # type: ignore
NT_VERSION = 1 # type: ignore
DT_NULL = 0 # type: ignore
DT_NEEDED = 1 # type: ignore
DT_PLTRELSZ = 2 # type: ignore
DT_PLTGOT = 3 # type: ignore
DT_HASH = 4 # type: ignore
DT_STRTAB = 5 # type: ignore
DT_SYMTAB = 6 # type: ignore
DT_RELA = 7 # type: ignore
DT_RELASZ = 8 # type: ignore
DT_RELAENT = 9 # type: ignore
DT_STRSZ = 10 # type: ignore
DT_SYMENT = 11 # type: ignore
DT_INIT = 12 # type: ignore
DT_FINI = 13 # type: ignore
DT_SONAME = 14 # type: ignore
DT_RPATH = 15 # type: ignore
DT_SYMBOLIC = 16 # type: ignore
DT_REL = 17 # type: ignore
DT_RELSZ = 18 # type: ignore
DT_RELENT = 19 # type: ignore
DT_PLTREL = 20 # type: ignore
DT_DEBUG = 21 # type: ignore
DT_TEXTREL = 22 # type: ignore
DT_JMPREL = 23 # type: ignore
DT_BIND_NOW = 24 # type: ignore
DT_INIT_ARRAY = 25 # type: ignore
DT_FINI_ARRAY = 26 # type: ignore
DT_INIT_ARRAYSZ = 27 # type: ignore
DT_FINI_ARRAYSZ = 28 # type: ignore
DT_RUNPATH = 29 # type: ignore
DT_FLAGS = 30 # type: ignore
DT_ENCODING = 32 # type: ignore
DT_PREINIT_ARRAY = 32 # type: ignore
DT_PREINIT_ARRAYSZ = 33 # type: ignore
DT_SYMTAB_SHNDX = 34 # type: ignore
DT_RELRSZ = 35 # type: ignore
DT_RELR = 36 # type: ignore
DT_RELRENT = 37 # type: ignore
DT_NUM = 38 # type: ignore
DT_LOOS = 0x6000000d # type: ignore
DT_HIOS = 0x6ffff000 # type: ignore
DT_LOPROC = 0x70000000 # type: ignore
DT_HIPROC = 0x7fffffff # type: ignore
DT_VALRNGLO = 0x6ffffd00 # type: ignore
DT_GNU_PRELINKED = 0x6ffffdf5 # type: ignore
DT_GNU_CONFLICTSZ = 0x6ffffdf6 # type: ignore
DT_GNU_LIBLISTSZ = 0x6ffffdf7 # type: ignore
DT_CHECKSUM = 0x6ffffdf8 # type: ignore
DT_PLTPADSZ = 0x6ffffdf9 # type: ignore
DT_MOVEENT = 0x6ffffdfa # type: ignore
DT_MOVESZ = 0x6ffffdfb # type: ignore
DT_FEATURE_1 = 0x6ffffdfc # type: ignore
DT_POSFLAG_1 = 0x6ffffdfd # type: ignore
DT_SYMINSZ = 0x6ffffdfe # type: ignore
DT_SYMINENT = 0x6ffffdff # type: ignore
DT_VALRNGHI = 0x6ffffdff # type: ignore
DT_VALTAGIDX = lambda tag: (DT_VALRNGHI - (tag)) # type: ignore
DT_VALNUM = 12 # type: ignore
DT_ADDRRNGLO = 0x6ffffe00 # type: ignore
DT_GNU_HASH = 0x6ffffef5 # type: ignore
DT_TLSDESC_PLT = 0x6ffffef6 # type: ignore
DT_TLSDESC_GOT = 0x6ffffef7 # type: ignore
DT_GNU_CONFLICT = 0x6ffffef8 # type: ignore
DT_GNU_LIBLIST = 0x6ffffef9 # type: ignore
DT_CONFIG = 0x6ffffefa # type: ignore
DT_DEPAUDIT = 0x6ffffefb # type: ignore
DT_AUDIT = 0x6ffffefc # type: ignore
DT_PLTPAD = 0x6ffffefd # type: ignore
DT_MOVETAB = 0x6ffffefe # type: ignore
DT_SYMINFO = 0x6ffffeff # type: ignore
DT_ADDRRNGHI = 0x6ffffeff # type: ignore
DT_ADDRTAGIDX = lambda tag: (DT_ADDRRNGHI - (tag)) # type: ignore
DT_ADDRNUM = 11 # type: ignore
DT_VERSYM = 0x6ffffff0 # type: ignore
DT_RELACOUNT = 0x6ffffff9 # type: ignore
DT_RELCOUNT = 0x6ffffffa # type: ignore
DT_FLAGS_1 = 0x6ffffffb # type: ignore
DT_VERDEF = 0x6ffffffc # type: ignore
DT_VERDEFNUM = 0x6ffffffd # type: ignore
DT_VERNEED = 0x6ffffffe # type: ignore
DT_VERNEEDNUM = 0x6fffffff # type: ignore
DT_VERSIONTAGIDX = lambda tag: (DT_VERNEEDNUM - (tag)) # type: ignore
DT_VERSIONTAGNUM = 16 # type: ignore
DT_AUXILIARY = 0x7ffffffd # type: ignore
DT_FILTER = 0x7fffffff # type: ignore
DT_EXTRATAGIDX = lambda tag: ((Elf32_Word)-((Elf32_Sword) (tag) <<1>>1)-1) # type: ignore
DT_EXTRANUM = 3 # type: ignore
DF_ORIGIN = 0x00000001 # type: ignore
DF_SYMBOLIC = 0x00000002 # type: ignore
DF_TEXTREL = 0x00000004 # type: ignore
DF_BIND_NOW = 0x00000008 # type: ignore
DF_STATIC_TLS = 0x00000010 # type: ignore
DF_1_NOW = 0x00000001 # type: ignore
DF_1_GLOBAL = 0x00000002 # type: ignore
DF_1_GROUP = 0x00000004 # type: ignore
DF_1_NODELETE = 0x00000008 # type: ignore
DF_1_LOADFLTR = 0x00000010 # type: ignore
DF_1_INITFIRST = 0x00000020 # type: ignore
DF_1_NOOPEN = 0x00000040 # type: ignore
DF_1_ORIGIN = 0x00000080 # type: ignore
DF_1_DIRECT = 0x00000100 # type: ignore
DF_1_TRANS = 0x00000200 # type: ignore
DF_1_INTERPOSE = 0x00000400 # type: ignore
DF_1_NODEFLIB = 0x00000800 # type: ignore
DF_1_NODUMP = 0x00001000 # type: ignore
DF_1_CONFALT = 0x00002000 # type: ignore
DF_1_ENDFILTEE = 0x00004000 # type: ignore
DF_1_DISPRELDNE = 0x00008000 # type: ignore
DF_1_DISPRELPND = 0x00010000 # type: ignore
DF_1_NODIRECT = 0x00020000 # type: ignore
DF_1_IGNMULDEF = 0x00040000 # type: ignore
DF_1_NOKSYMS = 0x00080000 # type: ignore
DF_1_NOHDR = 0x00100000 # type: ignore
DF_1_EDITED = 0x00200000 # type: ignore
DF_1_NORELOC = 0x00400000 # type: ignore
DF_1_SYMINTPOSE = 0x00800000 # type: ignore
DF_1_GLOBAUDIT = 0x01000000 # type: ignore
DF_1_SINGLETON = 0x02000000 # type: ignore
DF_1_STUB = 0x04000000 # type: ignore
DF_1_PIE = 0x08000000 # type: ignore
DF_1_KMOD = 0x10000000 # type: ignore
DF_1_WEAKFILTER = 0x20000000 # type: ignore
DF_1_NOCOMMON = 0x40000000 # type: ignore
DTF_1_PARINIT = 0x00000001 # type: ignore
DTF_1_CONFEXP = 0x00000002 # type: ignore
DF_P1_LAZYLOAD = 0x00000001 # type: ignore
DF_P1_GROUPPERM = 0x00000002 # type: ignore
VER_DEF_NONE = 0 # type: ignore
VER_DEF_CURRENT = 1 # type: ignore
VER_DEF_NUM = 2 # type: ignore
VER_FLG_BASE = 0x1 # type: ignore
VER_FLG_WEAK = 0x2 # type: ignore
VER_NDX_LOCAL = 0 # type: ignore
VER_NDX_GLOBAL = 1 # type: ignore
VER_NDX_LORESERVE = 0xff00 # type: ignore
VER_NDX_ELIMINATE = 0xff01 # type: ignore
VER_NEED_NONE = 0 # type: ignore
VER_NEED_CURRENT = 1 # type: ignore
VER_NEED_NUM = 2 # type: ignore
AT_NULL = 0 # type: ignore
AT_IGNORE = 1 # type: ignore
AT_EXECFD = 2 # type: ignore
AT_PHDR = 3 # type: ignore
AT_PHENT = 4 # type: ignore
AT_PHNUM = 5 # type: ignore
AT_PAGESZ = 6 # type: ignore
AT_BASE = 7 # type: ignore
AT_FLAGS = 8 # type: ignore
AT_ENTRY = 9 # type: ignore
AT_NOTELF = 10 # type: ignore
AT_UID = 11 # type: ignore
AT_EUID = 12 # type: ignore
AT_GID = 13 # type: ignore
AT_EGID = 14 # type: ignore
AT_CLKTCK = 17 # type: ignore
AT_PLATFORM = 15 # type: ignore
AT_HWCAP = 16 # type: ignore
AT_FPUCW = 18 # type: ignore
AT_DCACHEBSIZE = 19 # type: ignore
AT_ICACHEBSIZE = 20 # type: ignore
AT_UCACHEBSIZE = 21 # type: ignore
AT_IGNOREPPC = 22 # type: ignore
AT_SECURE = 23 # type: ignore
AT_BASE_PLATFORM = 24 # type: ignore
AT_RANDOM = 25 # type: ignore
AT_HWCAP2 = 26 # type: ignore
AT_RSEQ_FEATURE_SIZE = 27 # type: ignore
AT_RSEQ_ALIGN = 28 # type: ignore
AT_HWCAP3 = 29 # type: ignore
AT_HWCAP4 = 30 # type: ignore
AT_EXECFN = 31 # type: ignore
AT_SYSINFO = 32 # type: ignore
AT_SYSINFO_EHDR = 33 # type: ignore
AT_L1I_CACHESHAPE = 34 # type: ignore
AT_L1D_CACHESHAPE = 35 # type: ignore
AT_L2_CACHESHAPE = 36 # type: ignore
AT_L3_CACHESHAPE = 37 # type: ignore
AT_L1I_CACHESIZE = 40 # type: ignore
AT_L1I_CACHEGEOMETRY = 41 # type: ignore
AT_L1D_CACHESIZE = 42 # type: ignore
AT_L1D_CACHEGEOMETRY = 43 # type: ignore
AT_L2_CACHESIZE = 44 # type: ignore
AT_L2_CACHEGEOMETRY = 45 # type: ignore
AT_L3_CACHESIZE = 46 # type: ignore
AT_L3_CACHEGEOMETRY = 47 # type: ignore
AT_MINSIGSTKSZ = 51 # type: ignore
ELF_NOTE_SOLARIS = "SUNW Solaris" # type: ignore
ELF_NOTE_GNU = "GNU" # type: ignore
ELF_NOTE_FDO = "FDO" # type: ignore
ELF_NOTE_PAGESIZE_HINT = 1 # type: ignore
NT_GNU_ABI_TAG = 1 # type: ignore
ELF_NOTE_ABI = NT_GNU_ABI_TAG # type: ignore
ELF_NOTE_OS_LINUX = 0 # type: ignore
ELF_NOTE_OS_GNU = 1 # type: ignore
ELF_NOTE_OS_SOLARIS2 = 2 # type: ignore
ELF_NOTE_OS_FREEBSD = 3 # type: ignore
NT_GNU_HWCAP = 2 # type: ignore
NT_GNU_BUILD_ID = 3 # type: ignore
NT_GNU_GOLD_VERSION = 4 # type: ignore
NT_GNU_PROPERTY_TYPE_0 = 5 # type: ignore
NT_FDO_PACKAGING_METADATA = 0xcafe1a7e # type: ignore
NOTE_GNU_PROPERTY_SECTION_NAME = ".note.gnu.property" # type: ignore
GNU_PROPERTY_STACK_SIZE = 1 # type: ignore
GNU_PROPERTY_NO_COPY_ON_PROTECTED = 2 # type: ignore
GNU_PROPERTY_UINT32_AND_LO = 0xb0000000 # type: ignore
GNU_PROPERTY_UINT32_AND_HI = 0xb0007fff # type: ignore
GNU_PROPERTY_UINT32_OR_LO = 0xb0008000 # type: ignore
GNU_PROPERTY_UINT32_OR_HI = 0xb000ffff # type: ignore
GNU_PROPERTY_1_NEEDED = GNU_PROPERTY_UINT32_OR_LO # type: ignore
GNU_PROPERTY_1_NEEDED_INDIRECT_EXTERN_ACCESS = (1 << 0) # type: ignore
GNU_PROPERTY_LOPROC = 0xc0000000 # type: ignore
GNU_PROPERTY_HIPROC = 0xdfffffff # type: ignore
GNU_PROPERTY_LOUSER = 0xe0000000 # type: ignore
GNU_PROPERTY_HIUSER = 0xffffffff # type: ignore
GNU_PROPERTY_AARCH64_FEATURE_1_AND = 0xc0000000 # type: ignore
GNU_PROPERTY_AARCH64_FEATURE_1_BTI = (1 << 0) # type: ignore
GNU_PROPERTY_AARCH64_FEATURE_1_PAC = (1 << 1) # type: ignore
GNU_PROPERTY_X86_ISA_1_USED = 0xc0010002 # type: ignore
GNU_PROPERTY_X86_ISA_1_NEEDED = 0xc0008002 # type: ignore
GNU_PROPERTY_X86_FEATURE_1_AND = 0xc0000002 # type: ignore
GNU_PROPERTY_X86_ISA_1_BASELINE = (1 << 0) # type: ignore
GNU_PROPERTY_X86_ISA_1_V2 = (1 << 1) # type: ignore
GNU_PROPERTY_X86_ISA_1_V3 = (1 << 2) # type: ignore
GNU_PROPERTY_X86_ISA_1_V4 = (1 << 3) # type: ignore
GNU_PROPERTY_X86_FEATURE_1_IBT = (1 << 0) # type: ignore
GNU_PROPERTY_X86_FEATURE_1_SHSTK = (1 << 1) # type: ignore
ELF32_M_SYM = lambda info: ((info) >> 8) # type: ignore
ELF32_M_SIZE = lambda info: ( (info)) # type: ignore
ELF32_M_INFO = lambda sym,size: (((sym) << 8) +  (size)) # type: ignore
ELF64_M_SYM = lambda info: ELF32_M_SYM (info) # type: ignore
ELF64_M_SIZE = lambda info: ELF32_M_SIZE (info) # type: ignore
ELF64_M_INFO = lambda sym,size: ELF32_M_INFO (sym, size) # type: ignore
EF_CPU32 = 0x00810000 # type: ignore
R_68K_NONE = 0 # type: ignore
R_68K_32 = 1 # type: ignore
R_68K_16 = 2 # type: ignore
R_68K_8 = 3 # type: ignore
R_68K_PC32 = 4 # type: ignore
R_68K_PC16 = 5 # type: ignore
R_68K_PC8 = 6 # type: ignore
R_68K_GOT32 = 7 # type: ignore
R_68K_GOT16 = 8 # type: ignore
R_68K_GOT8 = 9 # type: ignore
R_68K_GOT32O = 10 # type: ignore
R_68K_GOT16O = 11 # type: ignore
R_68K_GOT8O = 12 # type: ignore
R_68K_PLT32 = 13 # type: ignore
R_68K_PLT16 = 14 # type: ignore
R_68K_PLT8 = 15 # type: ignore
R_68K_PLT32O = 16 # type: ignore
R_68K_PLT16O = 17 # type: ignore
R_68K_PLT8O = 18 # type: ignore
R_68K_COPY = 19 # type: ignore
R_68K_GLOB_DAT = 20 # type: ignore
R_68K_JMP_SLOT = 21 # type: ignore
R_68K_RELATIVE = 22 # type: ignore
R_68K_TLS_GD32 = 25 # type: ignore
R_68K_TLS_GD16 = 26 # type: ignore
R_68K_TLS_GD8 = 27 # type: ignore
R_68K_TLS_LDM32 = 28 # type: ignore
R_68K_TLS_LDM16 = 29 # type: ignore
R_68K_TLS_LDM8 = 30 # type: ignore
R_68K_TLS_LDO32 = 31 # type: ignore
R_68K_TLS_LDO16 = 32 # type: ignore
R_68K_TLS_LDO8 = 33 # type: ignore
R_68K_TLS_IE32 = 34 # type: ignore
R_68K_TLS_IE16 = 35 # type: ignore
R_68K_TLS_IE8 = 36 # type: ignore
R_68K_TLS_LE32 = 37 # type: ignore
R_68K_TLS_LE16 = 38 # type: ignore
R_68K_TLS_LE8 = 39 # type: ignore
R_68K_TLS_DTPMOD32 = 40 # type: ignore
R_68K_TLS_DTPREL32 = 41 # type: ignore
R_68K_TLS_TPREL32 = 42 # type: ignore
R_68K_NUM = 43 # type: ignore
R_386_NONE = 0 # type: ignore
R_386_32 = 1 # type: ignore
R_386_PC32 = 2 # type: ignore
R_386_GOT32 = 3 # type: ignore
R_386_PLT32 = 4 # type: ignore
R_386_COPY = 5 # type: ignore
R_386_GLOB_DAT = 6 # type: ignore
R_386_JMP_SLOT = 7 # type: ignore
R_386_RELATIVE = 8 # type: ignore
R_386_GOTOFF = 9 # type: ignore
R_386_GOTPC = 10 # type: ignore
R_386_32PLT = 11 # type: ignore
R_386_TLS_TPOFF = 14 # type: ignore
R_386_TLS_IE = 15 # type: ignore
R_386_TLS_GOTIE = 16 # type: ignore
R_386_TLS_LE = 17 # type: ignore
R_386_TLS_GD = 18 # type: ignore
R_386_TLS_LDM = 19 # type: ignore
R_386_16 = 20 # type: ignore
R_386_PC16 = 21 # type: ignore
R_386_8 = 22 # type: ignore
R_386_PC8 = 23 # type: ignore
R_386_TLS_GD_32 = 24 # type: ignore
R_386_TLS_GD_PUSH = 25 # type: ignore
R_386_TLS_GD_CALL = 26 # type: ignore
R_386_TLS_GD_POP = 27 # type: ignore
R_386_TLS_LDM_32 = 28 # type: ignore
R_386_TLS_LDM_PUSH = 29 # type: ignore
R_386_TLS_LDM_CALL = 30 # type: ignore
R_386_TLS_LDM_POP = 31 # type: ignore
R_386_TLS_LDO_32 = 32 # type: ignore
R_386_TLS_IE_32 = 33 # type: ignore
R_386_TLS_LE_32 = 34 # type: ignore
R_386_TLS_DTPMOD32 = 35 # type: ignore
R_386_TLS_DTPOFF32 = 36 # type: ignore
R_386_TLS_TPOFF32 = 37 # type: ignore
R_386_SIZE32 = 38 # type: ignore
R_386_TLS_GOTDESC = 39 # type: ignore
R_386_TLS_DESC_CALL = 40 # type: ignore
R_386_TLS_DESC = 41 # type: ignore
R_386_IRELATIVE = 42 # type: ignore
R_386_GOT32X = 43 # type: ignore
R_386_NUM = 44 # type: ignore
STT_SPARC_REGISTER = 13 # type: ignore
EF_SPARCV9_MM = 3 # type: ignore
EF_SPARCV9_TSO = 0 # type: ignore
EF_SPARCV9_PSO = 1 # type: ignore
EF_SPARCV9_RMO = 2 # type: ignore
EF_SPARC_LEDATA = 0x800000 # type: ignore
EF_SPARC_EXT_MASK = 0xFFFF00 # type: ignore
EF_SPARC_32PLUS = 0x000100 # type: ignore
EF_SPARC_SUN_US1 = 0x000200 # type: ignore
EF_SPARC_HAL_R1 = 0x000400 # type: ignore
EF_SPARC_SUN_US3 = 0x000800 # type: ignore
R_SPARC_NONE = 0 # type: ignore
R_SPARC_8 = 1 # type: ignore
R_SPARC_16 = 2 # type: ignore
R_SPARC_32 = 3 # type: ignore
R_SPARC_DISP8 = 4 # type: ignore
R_SPARC_DISP16 = 5 # type: ignore
R_SPARC_DISP32 = 6 # type: ignore
R_SPARC_WDISP30 = 7 # type: ignore
R_SPARC_WDISP22 = 8 # type: ignore
R_SPARC_HI22 = 9 # type: ignore
R_SPARC_22 = 10 # type: ignore
R_SPARC_13 = 11 # type: ignore
R_SPARC_LO10 = 12 # type: ignore
R_SPARC_GOT10 = 13 # type: ignore
R_SPARC_GOT13 = 14 # type: ignore
R_SPARC_GOT22 = 15 # type: ignore
R_SPARC_PC10 = 16 # type: ignore
R_SPARC_PC22 = 17 # type: ignore
R_SPARC_WPLT30 = 18 # type: ignore
R_SPARC_COPY = 19 # type: ignore
R_SPARC_GLOB_DAT = 20 # type: ignore
R_SPARC_JMP_SLOT = 21 # type: ignore
R_SPARC_RELATIVE = 22 # type: ignore
R_SPARC_UA32 = 23 # type: ignore
R_SPARC_PLT32 = 24 # type: ignore
R_SPARC_HIPLT22 = 25 # type: ignore
R_SPARC_LOPLT10 = 26 # type: ignore
R_SPARC_PCPLT32 = 27 # type: ignore
R_SPARC_PCPLT22 = 28 # type: ignore
R_SPARC_PCPLT10 = 29 # type: ignore
R_SPARC_10 = 30 # type: ignore
R_SPARC_11 = 31 # type: ignore
R_SPARC_64 = 32 # type: ignore
R_SPARC_OLO10 = 33 # type: ignore
R_SPARC_HH22 = 34 # type: ignore
R_SPARC_HM10 = 35 # type: ignore
R_SPARC_LM22 = 36 # type: ignore
R_SPARC_PC_HH22 = 37 # type: ignore
R_SPARC_PC_HM10 = 38 # type: ignore
R_SPARC_PC_LM22 = 39 # type: ignore
R_SPARC_WDISP16 = 40 # type: ignore
R_SPARC_WDISP19 = 41 # type: ignore
R_SPARC_GLOB_JMP = 42 # type: ignore
R_SPARC_7 = 43 # type: ignore
R_SPARC_5 = 44 # type: ignore
R_SPARC_6 = 45 # type: ignore
R_SPARC_DISP64 = 46 # type: ignore
R_SPARC_PLT64 = 47 # type: ignore
R_SPARC_HIX22 = 48 # type: ignore
R_SPARC_LOX10 = 49 # type: ignore
R_SPARC_H44 = 50 # type: ignore
R_SPARC_M44 = 51 # type: ignore
R_SPARC_L44 = 52 # type: ignore
R_SPARC_REGISTER = 53 # type: ignore
R_SPARC_UA64 = 54 # type: ignore
R_SPARC_UA16 = 55 # type: ignore
R_SPARC_TLS_GD_HI22 = 56 # type: ignore
R_SPARC_TLS_GD_LO10 = 57 # type: ignore
R_SPARC_TLS_GD_ADD = 58 # type: ignore
R_SPARC_TLS_GD_CALL = 59 # type: ignore
R_SPARC_TLS_LDM_HI22 = 60 # type: ignore
R_SPARC_TLS_LDM_LO10 = 61 # type: ignore
R_SPARC_TLS_LDM_ADD = 62 # type: ignore
R_SPARC_TLS_LDM_CALL = 63 # type: ignore
R_SPARC_TLS_LDO_HIX22 = 64 # type: ignore
R_SPARC_TLS_LDO_LOX10 = 65 # type: ignore
R_SPARC_TLS_LDO_ADD = 66 # type: ignore
R_SPARC_TLS_IE_HI22 = 67 # type: ignore
R_SPARC_TLS_IE_LO10 = 68 # type: ignore
R_SPARC_TLS_IE_LD = 69 # type: ignore
R_SPARC_TLS_IE_LDX = 70 # type: ignore
R_SPARC_TLS_IE_ADD = 71 # type: ignore
R_SPARC_TLS_LE_HIX22 = 72 # type: ignore
R_SPARC_TLS_LE_LOX10 = 73 # type: ignore
R_SPARC_TLS_DTPMOD32 = 74 # type: ignore
R_SPARC_TLS_DTPMOD64 = 75 # type: ignore
R_SPARC_TLS_DTPOFF32 = 76 # type: ignore
R_SPARC_TLS_DTPOFF64 = 77 # type: ignore
R_SPARC_TLS_TPOFF32 = 78 # type: ignore
R_SPARC_TLS_TPOFF64 = 79 # type: ignore
R_SPARC_GOTDATA_HIX22 = 80 # type: ignore
R_SPARC_GOTDATA_LOX10 = 81 # type: ignore
R_SPARC_GOTDATA_OP_HIX22 = 82 # type: ignore
R_SPARC_GOTDATA_OP_LOX10 = 83 # type: ignore
R_SPARC_GOTDATA_OP = 84 # type: ignore
R_SPARC_H34 = 85 # type: ignore
R_SPARC_SIZE32 = 86 # type: ignore
R_SPARC_SIZE64 = 87 # type: ignore
R_SPARC_WDISP10 = 88 # type: ignore
R_SPARC_JMP_IREL = 248 # type: ignore
R_SPARC_IRELATIVE = 249 # type: ignore
R_SPARC_GNU_VTINHERIT = 250 # type: ignore
R_SPARC_GNU_VTENTRY = 251 # type: ignore
R_SPARC_REV32 = 252 # type: ignore
R_SPARC_NUM = 253 # type: ignore
DT_SPARC_REGISTER = 0x70000001 # type: ignore
DT_SPARC_NUM = 2 # type: ignore
EF_MIPS_NOREORDER = 1 # type: ignore
EF_MIPS_PIC = 2 # type: ignore
EF_MIPS_CPIC = 4 # type: ignore
EF_MIPS_XGOT = 8 # type: ignore
EF_MIPS_UCODE = 16 # type: ignore
EF_MIPS_ABI2 = 32 # type: ignore
EF_MIPS_ABI_ON32 = 64 # type: ignore
EF_MIPS_OPTIONS_FIRST = 0x00000080 # type: ignore
EF_MIPS_32BITMODE = 0x00000100 # type: ignore
EF_MIPS_FP64 = 512 # type: ignore
EF_MIPS_NAN2008 = 1024 # type: ignore
EF_MIPS_ARCH_ASE = 0x0f000000 # type: ignore
EF_MIPS_ARCH_ASE_MDMX = 0x08000000 # type: ignore
EF_MIPS_ARCH_ASE_M16 = 0x04000000 # type: ignore
EF_MIPS_ARCH_ASE_MICROMIPS = 0x02000000 # type: ignore
EF_MIPS_ARCH = 0xf0000000 # type: ignore
EF_MIPS_ARCH_1 = 0x00000000 # type: ignore
EF_MIPS_ARCH_2 = 0x10000000 # type: ignore
EF_MIPS_ARCH_3 = 0x20000000 # type: ignore
EF_MIPS_ARCH_4 = 0x30000000 # type: ignore
EF_MIPS_ARCH_5 = 0x40000000 # type: ignore
EF_MIPS_ARCH_32 = 0x50000000 # type: ignore
EF_MIPS_ARCH_64 = 0x60000000 # type: ignore
EF_MIPS_ARCH_32R2 = 0x70000000 # type: ignore
EF_MIPS_ARCH_64R2 = 0x80000000 # type: ignore
EF_MIPS_ARCH_32R6 = 0x90000000 # type: ignore
EF_MIPS_ARCH_64R6 = 0xa0000000 # type: ignore
EF_MIPS_ABI = 0x0000F000 # type: ignore
EF_MIPS_ABI_O32 = 0x00001000 # type: ignore
EF_MIPS_ABI_O64 = 0x00002000 # type: ignore
EF_MIPS_ABI_EABI32 = 0x00003000 # type: ignore
EF_MIPS_ABI_EABI64 = 0x00004000 # type: ignore
EF_MIPS_MACH = 0x00FF0000 # type: ignore
EF_MIPS_MACH_3900 = 0x00810000 # type: ignore
EF_MIPS_MACH_4010 = 0x00820000 # type: ignore
EF_MIPS_MACH_4100 = 0x00830000 # type: ignore
EF_MIPS_MACH_ALLEGREX = 0x00840000 # type: ignore
EF_MIPS_MACH_4650 = 0x00850000 # type: ignore
EF_MIPS_MACH_4120 = 0x00870000 # type: ignore
EF_MIPS_MACH_4111 = 0x00880000 # type: ignore
EF_MIPS_MACH_SB1 = 0x008a0000 # type: ignore
EF_MIPS_MACH_OCTEON = 0x008b0000 # type: ignore
EF_MIPS_MACH_XLR = 0x008c0000 # type: ignore
EF_MIPS_MACH_OCTEON2 = 0x008d0000 # type: ignore
EF_MIPS_MACH_OCTEON3 = 0x008e0000 # type: ignore
EF_MIPS_MACH_5400 = 0x00910000 # type: ignore
EF_MIPS_MACH_5900 = 0x00920000 # type: ignore
EF_MIPS_MACH_IAMR2 = 0x00930000 # type: ignore
EF_MIPS_MACH_5500 = 0x00980000 # type: ignore
EF_MIPS_MACH_9000 = 0x00990000 # type: ignore
EF_MIPS_MACH_LS2E = 0x00A00000 # type: ignore
EF_MIPS_MACH_LS2F = 0x00A10000 # type: ignore
EF_MIPS_MACH_GS464 = 0x00A20000 # type: ignore
EF_MIPS_MACH_GS464E = 0x00A30000 # type: ignore
EF_MIPS_MACH_GS264E = 0x00A40000 # type: ignore
E_MIPS_ARCH_1 = EF_MIPS_ARCH_1 # type: ignore
E_MIPS_ARCH_2 = EF_MIPS_ARCH_2 # type: ignore
E_MIPS_ARCH_3 = EF_MIPS_ARCH_3 # type: ignore
E_MIPS_ARCH_4 = EF_MIPS_ARCH_4 # type: ignore
E_MIPS_ARCH_5 = EF_MIPS_ARCH_5 # type: ignore
E_MIPS_ARCH_32 = EF_MIPS_ARCH_32 # type: ignore
E_MIPS_ARCH_64 = EF_MIPS_ARCH_64 # type: ignore
SHN_MIPS_ACOMMON = 0xff00 # type: ignore
SHN_MIPS_TEXT = 0xff01 # type: ignore
SHN_MIPS_DATA = 0xff02 # type: ignore
SHN_MIPS_SCOMMON = 0xff03 # type: ignore
SHN_MIPS_SUNDEFINED = 0xff04 # type: ignore
SHT_MIPS_LIBLIST = 0x70000000 # type: ignore
SHT_MIPS_MSYM = 0x70000001 # type: ignore
SHT_MIPS_CONFLICT = 0x70000002 # type: ignore
SHT_MIPS_GPTAB = 0x70000003 # type: ignore
SHT_MIPS_UCODE = 0x70000004 # type: ignore
SHT_MIPS_DEBUG = 0x70000005 # type: ignore
SHT_MIPS_REGINFO = 0x70000006 # type: ignore
SHT_MIPS_PACKAGE = 0x70000007 # type: ignore
SHT_MIPS_PACKSYM = 0x70000008 # type: ignore
SHT_MIPS_RELD = 0x70000009 # type: ignore
SHT_MIPS_IFACE = 0x7000000b # type: ignore
SHT_MIPS_CONTENT = 0x7000000c # type: ignore
SHT_MIPS_OPTIONS = 0x7000000d # type: ignore
SHT_MIPS_SHDR = 0x70000010 # type: ignore
SHT_MIPS_FDESC = 0x70000011 # type: ignore
SHT_MIPS_EXTSYM = 0x70000012 # type: ignore
SHT_MIPS_DENSE = 0x70000013 # type: ignore
SHT_MIPS_PDESC = 0x70000014 # type: ignore
SHT_MIPS_LOCSYM = 0x70000015 # type: ignore
SHT_MIPS_AUXSYM = 0x70000016 # type: ignore
SHT_MIPS_OPTSYM = 0x70000017 # type: ignore
SHT_MIPS_LOCSTR = 0x70000018 # type: ignore
SHT_MIPS_LINE = 0x70000019 # type: ignore
SHT_MIPS_RFDESC = 0x7000001a # type: ignore
SHT_MIPS_DELTASYM = 0x7000001b # type: ignore
SHT_MIPS_DELTAINST = 0x7000001c # type: ignore
SHT_MIPS_DELTACLASS = 0x7000001d # type: ignore
SHT_MIPS_DWARF = 0x7000001e # type: ignore
SHT_MIPS_DELTADECL = 0x7000001f # type: ignore
SHT_MIPS_SYMBOL_LIB = 0x70000020 # type: ignore
SHT_MIPS_EVENTS = 0x70000021 # type: ignore
SHT_MIPS_TRANSLATE = 0x70000022 # type: ignore
SHT_MIPS_PIXIE = 0x70000023 # type: ignore
SHT_MIPS_XLATE = 0x70000024 # type: ignore
SHT_MIPS_XLATE_DEBUG = 0x70000025 # type: ignore
SHT_MIPS_WHIRL = 0x70000026 # type: ignore
SHT_MIPS_EH_REGION = 0x70000027 # type: ignore
SHT_MIPS_XLATE_OLD = 0x70000028 # type: ignore
SHT_MIPS_PDR_EXCEPTION = 0x70000029 # type: ignore
SHT_MIPS_ABIFLAGS = 0x7000002a # type: ignore
SHT_MIPS_XHASH = 0x7000002b # type: ignore
SHF_MIPS_GPREL = 0x10000000 # type: ignore
SHF_MIPS_MERGE = 0x20000000 # type: ignore
SHF_MIPS_ADDR = 0x40000000 # type: ignore
SHF_MIPS_STRINGS = 0x80000000 # type: ignore
SHF_MIPS_NOSTRIP = 0x08000000 # type: ignore
SHF_MIPS_LOCAL = 0x04000000 # type: ignore
SHF_MIPS_NAMES = 0x02000000 # type: ignore
SHF_MIPS_NODUPE = 0x01000000 # type: ignore
STO_MIPS_DEFAULT = 0x0 # type: ignore
STO_MIPS_INTERNAL = 0x1 # type: ignore
STO_MIPS_HIDDEN = 0x2 # type: ignore
STO_MIPS_PROTECTED = 0x3 # type: ignore
STO_MIPS_PLT = 0x8 # type: ignore
STO_MIPS_SC_ALIGN_UNUSED = 0xff # type: ignore
STB_MIPS_SPLIT_COMMON = 13 # type: ignore
ODK_NULL = 0 # type: ignore
ODK_REGINFO = 1 # type: ignore
ODK_EXCEPTIONS = 2 # type: ignore
ODK_PAD = 3 # type: ignore
ODK_HWPATCH = 4 # type: ignore
ODK_FILL = 5 # type: ignore
ODK_TAGS = 6 # type: ignore
ODK_HWAND = 7 # type: ignore
ODK_HWOR = 8 # type: ignore
OEX_FPU_MIN = 0x1f # type: ignore
OEX_FPU_MAX = 0x1f00 # type: ignore
OEX_PAGE0 = 0x10000 # type: ignore
OEX_SMM = 0x20000 # type: ignore
OEX_FPDBUG = 0x40000 # type: ignore
OEX_PRECISEFP = OEX_FPDBUG # type: ignore
OEX_DISMISS = 0x80000 # type: ignore
OEX_FPU_INVAL = 0x10 # type: ignore
OEX_FPU_DIV0 = 0x08 # type: ignore
OEX_FPU_OFLO = 0x04 # type: ignore
OEX_FPU_UFLO = 0x02 # type: ignore
OEX_FPU_INEX = 0x01 # type: ignore
OHW_R4KEOP = 0x1 # type: ignore
OHW_R8KPFETCH = 0x2 # type: ignore
OHW_R5KEOP = 0x4 # type: ignore
OHW_R5KCVTL = 0x8 # type: ignore
OPAD_PREFIX = 0x1 # type: ignore
OPAD_POSTFIX = 0x2 # type: ignore
OPAD_SYMBOL = 0x4 # type: ignore
OHWA0_R4KEOP_CHECKED = 0x00000001 # type: ignore
OHWA1_R4KEOP_CLEAN = 0x00000002 # type: ignore
R_MIPS_NONE = 0 # type: ignore
R_MIPS_16 = 1 # type: ignore
R_MIPS_32 = 2 # type: ignore
R_MIPS_REL32 = 3 # type: ignore
R_MIPS_26 = 4 # type: ignore
R_MIPS_HI16 = 5 # type: ignore
R_MIPS_LO16 = 6 # type: ignore
R_MIPS_GPREL16 = 7 # type: ignore
R_MIPS_LITERAL = 8 # type: ignore
R_MIPS_GOT16 = 9 # type: ignore
R_MIPS_PC16 = 10 # type: ignore
R_MIPS_CALL16 = 11 # type: ignore
R_MIPS_GPREL32 = 12 # type: ignore
R_MIPS_SHIFT5 = 16 # type: ignore
R_MIPS_SHIFT6 = 17 # type: ignore
R_MIPS_64 = 18 # type: ignore
R_MIPS_GOT_DISP = 19 # type: ignore
R_MIPS_GOT_PAGE = 20 # type: ignore
R_MIPS_GOT_OFST = 21 # type: ignore
R_MIPS_GOT_HI16 = 22 # type: ignore
R_MIPS_GOT_LO16 = 23 # type: ignore
R_MIPS_SUB = 24 # type: ignore
R_MIPS_INSERT_A = 25 # type: ignore
R_MIPS_INSERT_B = 26 # type: ignore
R_MIPS_DELETE = 27 # type: ignore
R_MIPS_HIGHER = 28 # type: ignore
R_MIPS_HIGHEST = 29 # type: ignore
R_MIPS_CALL_HI16 = 30 # type: ignore
R_MIPS_CALL_LO16 = 31 # type: ignore
R_MIPS_SCN_DISP = 32 # type: ignore
R_MIPS_REL16 = 33 # type: ignore
R_MIPS_ADD_IMMEDIATE = 34 # type: ignore
R_MIPS_PJUMP = 35 # type: ignore
R_MIPS_RELGOT = 36 # type: ignore
R_MIPS_JALR = 37 # type: ignore
R_MIPS_TLS_DTPMOD32 = 38 # type: ignore
R_MIPS_TLS_DTPREL32 = 39 # type: ignore
R_MIPS_TLS_DTPMOD64 = 40 # type: ignore
R_MIPS_TLS_DTPREL64 = 41 # type: ignore
R_MIPS_TLS_GD = 42 # type: ignore
R_MIPS_TLS_LDM = 43 # type: ignore
R_MIPS_TLS_DTPREL_HI16 = 44 # type: ignore
R_MIPS_TLS_DTPREL_LO16 = 45 # type: ignore
R_MIPS_TLS_GOTTPREL = 46 # type: ignore
R_MIPS_TLS_TPREL32 = 47 # type: ignore
R_MIPS_TLS_TPREL64 = 48 # type: ignore
R_MIPS_TLS_TPREL_HI16 = 49 # type: ignore
R_MIPS_TLS_TPREL_LO16 = 50 # type: ignore
R_MIPS_GLOB_DAT = 51 # type: ignore
R_MIPS_PC21_S2 = 60 # type: ignore
R_MIPS_PC26_S2 = 61 # type: ignore
R_MIPS_PC18_S3 = 62 # type: ignore
R_MIPS_PC19_S2 = 63 # type: ignore
R_MIPS_PCHI16 = 64 # type: ignore
R_MIPS_PCLO16 = 65 # type: ignore
R_MIPS16_26 = 100 # type: ignore
R_MIPS16_GPREL = 101 # type: ignore
R_MIPS16_GOT16 = 102 # type: ignore
R_MIPS16_CALL16 = 103 # type: ignore
R_MIPS16_HI16 = 104 # type: ignore
R_MIPS16_LO16 = 105 # type: ignore
R_MIPS16_TLS_GD = 106 # type: ignore
R_MIPS16_TLS_LDM = 107 # type: ignore
R_MIPS16_TLS_DTPREL_HI16 = 108 # type: ignore
R_MIPS16_TLS_DTPREL_LO16 = 109 # type: ignore
R_MIPS16_TLS_GOTTPREL = 110 # type: ignore
R_MIPS16_TLS_TPREL_HI16 = 111 # type: ignore
R_MIPS16_TLS_TPREL_LO16 = 112 # type: ignore
R_MIPS16_PC16_S1 = 113 # type: ignore
R_MIPS_COPY = 126 # type: ignore
R_MIPS_JUMP_SLOT = 127 # type: ignore
R_MIPS_RELATIVE = 128 # type: ignore
R_MICROMIPS_26_S1 = 133 # type: ignore
R_MICROMIPS_HI16 = 134 # type: ignore
R_MICROMIPS_LO16 = 135 # type: ignore
R_MICROMIPS_GPREL16 = 136 # type: ignore
R_MICROMIPS_LITERAL = 137 # type: ignore
R_MICROMIPS_GOT16 = 138 # type: ignore
R_MICROMIPS_PC7_S1 = 139 # type: ignore
R_MICROMIPS_PC10_S1 = 140 # type: ignore
R_MICROMIPS_PC16_S1 = 141 # type: ignore
R_MICROMIPS_CALL16 = 142 # type: ignore
R_MICROMIPS_GOT_DISP = 145 # type: ignore
R_MICROMIPS_GOT_PAGE = 146 # type: ignore
R_MICROMIPS_GOT_OFST = 147 # type: ignore
R_MICROMIPS_GOT_HI16 = 148 # type: ignore
R_MICROMIPS_GOT_LO16 = 149 # type: ignore
R_MICROMIPS_SUB = 150 # type: ignore
R_MICROMIPS_HIGHER = 151 # type: ignore
R_MICROMIPS_HIGHEST = 152 # type: ignore
R_MICROMIPS_CALL_HI16 = 153 # type: ignore
R_MICROMIPS_CALL_LO16 = 154 # type: ignore
R_MICROMIPS_SCN_DISP = 155 # type: ignore
R_MICROMIPS_JALR = 156 # type: ignore
R_MICROMIPS_HI0_LO16 = 157 # type: ignore
R_MICROMIPS_TLS_GD = 162 # type: ignore
R_MICROMIPS_TLS_LDM = 163 # type: ignore
R_MICROMIPS_TLS_DTPREL_HI16 = 164 # type: ignore
R_MICROMIPS_TLS_DTPREL_LO16 = 165 # type: ignore
R_MICROMIPS_TLS_GOTTPREL = 166 # type: ignore
R_MICROMIPS_TLS_TPREL_HI16 = 169 # type: ignore
R_MICROMIPS_TLS_TPREL_LO16 = 170 # type: ignore
R_MICROMIPS_GPREL7_S2 = 172 # type: ignore
R_MICROMIPS_PC23_S2 = 173 # type: ignore
R_MIPS_PC32 = 248 # type: ignore
R_MIPS_EH = 249 # type: ignore
R_MIPS_GNU_REL16_S2 = 250 # type: ignore
R_MIPS_GNU_VTINHERIT = 253 # type: ignore
R_MIPS_GNU_VTENTRY = 254 # type: ignore
R_MIPS_NUM = 255 # type: ignore
PT_MIPS_REGINFO = 0x70000000 # type: ignore
PT_MIPS_RTPROC = 0x70000001 # type: ignore
PT_MIPS_OPTIONS = 0x70000002 # type: ignore
PT_MIPS_ABIFLAGS = 0x70000003 # type: ignore
PF_MIPS_LOCAL = 0x10000000 # type: ignore
DT_MIPS_RLD_VERSION = 0x70000001 # type: ignore
DT_MIPS_TIME_STAMP = 0x70000002 # type: ignore
DT_MIPS_ICHECKSUM = 0x70000003 # type: ignore
DT_MIPS_IVERSION = 0x70000004 # type: ignore
DT_MIPS_FLAGS = 0x70000005 # type: ignore
DT_MIPS_BASE_ADDRESS = 0x70000006 # type: ignore
DT_MIPS_MSYM = 0x70000007 # type: ignore
DT_MIPS_CONFLICT = 0x70000008 # type: ignore
DT_MIPS_LIBLIST = 0x70000009 # type: ignore
DT_MIPS_LOCAL_GOTNO = 0x7000000a # type: ignore
DT_MIPS_CONFLICTNO = 0x7000000b # type: ignore
DT_MIPS_LIBLISTNO = 0x70000010 # type: ignore
DT_MIPS_SYMTABNO = 0x70000011 # type: ignore
DT_MIPS_UNREFEXTNO = 0x70000012 # type: ignore
DT_MIPS_GOTSYM = 0x70000013 # type: ignore
DT_MIPS_HIPAGENO = 0x70000014 # type: ignore
DT_MIPS_RLD_MAP = 0x70000016 # type: ignore
DT_MIPS_DELTA_CLASS = 0x70000017 # type: ignore
DT_MIPS_DELTA_CLASS_NO = 0x70000018 # type: ignore
DT_MIPS_DELTA_INSTANCE = 0x70000019 # type: ignore
DT_MIPS_DELTA_INSTANCE_NO = 0x7000001a # type: ignore
DT_MIPS_DELTA_RELOC = 0x7000001b # type: ignore
DT_MIPS_DELTA_RELOC_NO = 0x7000001c # type: ignore
DT_MIPS_DELTA_SYM = 0x7000001d # type: ignore
DT_MIPS_DELTA_SYM_NO = 0x7000001e # type: ignore
DT_MIPS_DELTA_CLASSSYM = 0x70000020 # type: ignore
DT_MIPS_DELTA_CLASSSYM_NO = 0x70000021 # type: ignore
DT_MIPS_CXX_FLAGS = 0x70000022 # type: ignore
DT_MIPS_PIXIE_INIT = 0x70000023 # type: ignore
DT_MIPS_SYMBOL_LIB = 0x70000024 # type: ignore
DT_MIPS_LOCALPAGE_GOTIDX = 0x70000025 # type: ignore
DT_MIPS_LOCAL_GOTIDX = 0x70000026 # type: ignore
DT_MIPS_HIDDEN_GOTIDX = 0x70000027 # type: ignore
DT_MIPS_PROTECTED_GOTIDX = 0x70000028 # type: ignore
DT_MIPS_OPTIONS = 0x70000029 # type: ignore
DT_MIPS_INTERFACE = 0x7000002a # type: ignore
DT_MIPS_DYNSTR_ALIGN = 0x7000002b # type: ignore
DT_MIPS_INTERFACE_SIZE = 0x7000002c # type: ignore
DT_MIPS_RLD_TEXT_RESOLVE_ADDR = 0x7000002d # type: ignore
DT_MIPS_PERF_SUFFIX = 0x7000002e # type: ignore
DT_MIPS_COMPACT_SIZE = 0x7000002f # type: ignore
DT_MIPS_GP_VALUE = 0x70000030 # type: ignore
DT_MIPS_AUX_DYNAMIC = 0x70000031 # type: ignore
DT_MIPS_PLTGOT = 0x70000032 # type: ignore
DT_MIPS_RWPLT = 0x70000034 # type: ignore
DT_MIPS_RLD_MAP_REL = 0x70000035 # type: ignore
DT_MIPS_XHASH = 0x70000036 # type: ignore
DT_MIPS_NUM = 0x37 # type: ignore
RHF_NONE = 0 # type: ignore
RHF_QUICKSTART = (1 << 0) # type: ignore
RHF_NOTPOT = (1 << 1) # type: ignore
RHF_NO_LIBRARY_REPLACEMENT = (1 << 2) # type: ignore
RHF_NO_MOVE = (1 << 3) # type: ignore
RHF_SGI_ONLY = (1 << 4) # type: ignore
RHF_GUARANTEE_INIT = (1 << 5) # type: ignore
RHF_DELTA_C_PLUS_PLUS = (1 << 6) # type: ignore
RHF_GUARANTEE_START_INIT = (1 << 7) # type: ignore
RHF_PIXIE = (1 << 8) # type: ignore
RHF_DEFAULT_DELAY_LOAD = (1 << 9) # type: ignore
RHF_REQUICKSTART = (1 << 10) # type: ignore
RHF_REQUICKSTARTED = (1 << 11) # type: ignore
RHF_CORD = (1 << 12) # type: ignore
RHF_NO_UNRES_UNDEF = (1 << 13) # type: ignore
RHF_RLD_ORDER_SAFE = (1 << 14) # type: ignore
LL_NONE = 0 # type: ignore
LL_EXACT_MATCH = (1 << 0) # type: ignore
LL_IGNORE_INT_VER = (1 << 1) # type: ignore
LL_REQUIRE_MINOR = (1 << 2) # type: ignore
LL_EXPORTS = (1 << 3) # type: ignore
LL_DELAY_LOAD = (1 << 4) # type: ignore
LL_DELTA = (1 << 5) # type: ignore
MIPS_AFL_REG_NONE = 0x00 # type: ignore
MIPS_AFL_REG_32 = 0x01 # type: ignore
MIPS_AFL_REG_64 = 0x02 # type: ignore
MIPS_AFL_REG_128 = 0x03 # type: ignore
MIPS_AFL_ASE_DSP = 0x00000001 # type: ignore
MIPS_AFL_ASE_DSPR2 = 0x00000002 # type: ignore
MIPS_AFL_ASE_EVA = 0x00000004 # type: ignore
MIPS_AFL_ASE_MCU = 0x00000008 # type: ignore
MIPS_AFL_ASE_MDMX = 0x00000010 # type: ignore
MIPS_AFL_ASE_MIPS3D = 0x00000020 # type: ignore
MIPS_AFL_ASE_MT = 0x00000040 # type: ignore
MIPS_AFL_ASE_SMARTMIPS = 0x00000080 # type: ignore
MIPS_AFL_ASE_VIRT = 0x00000100 # type: ignore
MIPS_AFL_ASE_MSA = 0x00000200 # type: ignore
MIPS_AFL_ASE_MIPS16 = 0x00000400 # type: ignore
MIPS_AFL_ASE_MICROMIPS = 0x00000800 # type: ignore
MIPS_AFL_ASE_XPA = 0x00001000 # type: ignore
MIPS_AFL_ASE_MASK = 0x00001fff # type: ignore
MIPS_AFL_EXT_XLR = 1 # type: ignore
MIPS_AFL_EXT_OCTEON2 = 2 # type: ignore
MIPS_AFL_EXT_OCTEONP = 3 # type: ignore
MIPS_AFL_EXT_LOONGSON_3A = 4 # type: ignore
MIPS_AFL_EXT_OCTEON = 5 # type: ignore
MIPS_AFL_EXT_5900 = 6 # type: ignore
MIPS_AFL_EXT_4650 = 7 # type: ignore
MIPS_AFL_EXT_4010 = 8 # type: ignore
MIPS_AFL_EXT_4100 = 9 # type: ignore
MIPS_AFL_EXT_3900 = 10 # type: ignore
MIPS_AFL_EXT_10000 = 11 # type: ignore
MIPS_AFL_EXT_SB1 = 12 # type: ignore
MIPS_AFL_EXT_4111 = 13 # type: ignore
MIPS_AFL_EXT_4120 = 14 # type: ignore
MIPS_AFL_EXT_5400 = 15 # type: ignore
MIPS_AFL_EXT_5500 = 16 # type: ignore
MIPS_AFL_EXT_LOONGSON_2E = 17 # type: ignore
MIPS_AFL_EXT_LOONGSON_2F = 18 # type: ignore
MIPS_AFL_FLAGS1_ODDSPREG = 1 # type: ignore
EF_PARISC_TRAPNIL = 0x00010000 # type: ignore
EF_PARISC_EXT = 0x00020000 # type: ignore
EF_PARISC_LSB = 0x00040000 # type: ignore
EF_PARISC_WIDE = 0x00080000 # type: ignore
EF_PARISC_NO_KABP = 0x00100000 # type: ignore
EF_PARISC_LAZYSWAP = 0x00400000 # type: ignore
EF_PARISC_ARCH = 0x0000ffff # type: ignore
EFA_PARISC_1_0 = 0x020b # type: ignore
EFA_PARISC_1_1 = 0x0210 # type: ignore
EFA_PARISC_2_0 = 0x0214 # type: ignore
SHN_PARISC_ANSI_COMMON = 0xff00 # type: ignore
SHN_PARISC_HUGE_COMMON = 0xff01 # type: ignore
SHT_PARISC_EXT = 0x70000000 # type: ignore
SHT_PARISC_UNWIND = 0x70000001 # type: ignore
SHT_PARISC_DOC = 0x70000002 # type: ignore
SHF_PARISC_SHORT = 0x20000000 # type: ignore
SHF_PARISC_HUGE = 0x40000000 # type: ignore
SHF_PARISC_SBP = 0x80000000 # type: ignore
STT_PARISC_MILLICODE = 13 # type: ignore
STT_HP_OPAQUE = (STT_LOOS + 0x1) # type: ignore
STT_HP_STUB = (STT_LOOS + 0x2) # type: ignore
R_PARISC_NONE = 0 # type: ignore
R_PARISC_DIR32 = 1 # type: ignore
R_PARISC_DIR21L = 2 # type: ignore
R_PARISC_DIR17R = 3 # type: ignore
R_PARISC_DIR17F = 4 # type: ignore
R_PARISC_DIR14R = 6 # type: ignore
R_PARISC_PCREL32 = 9 # type: ignore
R_PARISC_PCREL21L = 10 # type: ignore
R_PARISC_PCREL17R = 11 # type: ignore
R_PARISC_PCREL17F = 12 # type: ignore
R_PARISC_PCREL14R = 14 # type: ignore
R_PARISC_DPREL21L = 18 # type: ignore
R_PARISC_DPREL14R = 22 # type: ignore
R_PARISC_GPREL21L = 26 # type: ignore
R_PARISC_GPREL14R = 30 # type: ignore
R_PARISC_LTOFF21L = 34 # type: ignore
R_PARISC_LTOFF14R = 38 # type: ignore
R_PARISC_SECREL32 = 41 # type: ignore
R_PARISC_SEGBASE = 48 # type: ignore
R_PARISC_SEGREL32 = 49 # type: ignore
R_PARISC_PLTOFF21L = 50 # type: ignore
R_PARISC_PLTOFF14R = 54 # type: ignore
R_PARISC_LTOFF_FPTR32 = 57 # type: ignore
R_PARISC_LTOFF_FPTR21L = 58 # type: ignore
R_PARISC_LTOFF_FPTR14R = 62 # type: ignore
R_PARISC_FPTR64 = 64 # type: ignore
R_PARISC_PLABEL32 = 65 # type: ignore
R_PARISC_PLABEL21L = 66 # type: ignore
R_PARISC_PLABEL14R = 70 # type: ignore
R_PARISC_PCREL64 = 72 # type: ignore
R_PARISC_PCREL22F = 74 # type: ignore
R_PARISC_PCREL14WR = 75 # type: ignore
R_PARISC_PCREL14DR = 76 # type: ignore
R_PARISC_PCREL16F = 77 # type: ignore
R_PARISC_PCREL16WF = 78 # type: ignore
R_PARISC_PCREL16DF = 79 # type: ignore
R_PARISC_DIR64 = 80 # type: ignore
R_PARISC_DIR14WR = 83 # type: ignore
R_PARISC_DIR14DR = 84 # type: ignore
R_PARISC_DIR16F = 85 # type: ignore
R_PARISC_DIR16WF = 86 # type: ignore
R_PARISC_DIR16DF = 87 # type: ignore
R_PARISC_GPREL64 = 88 # type: ignore
R_PARISC_GPREL14WR = 91 # type: ignore
R_PARISC_GPREL14DR = 92 # type: ignore
R_PARISC_GPREL16F = 93 # type: ignore
R_PARISC_GPREL16WF = 94 # type: ignore
R_PARISC_GPREL16DF = 95 # type: ignore
R_PARISC_LTOFF64 = 96 # type: ignore
R_PARISC_LTOFF14WR = 99 # type: ignore
R_PARISC_LTOFF14DR = 100 # type: ignore
R_PARISC_LTOFF16F = 101 # type: ignore
R_PARISC_LTOFF16WF = 102 # type: ignore
R_PARISC_LTOFF16DF = 103 # type: ignore
R_PARISC_SECREL64 = 104 # type: ignore
R_PARISC_SEGREL64 = 112 # type: ignore
R_PARISC_PLTOFF14WR = 115 # type: ignore
R_PARISC_PLTOFF14DR = 116 # type: ignore
R_PARISC_PLTOFF16F = 117 # type: ignore
R_PARISC_PLTOFF16WF = 118 # type: ignore
R_PARISC_PLTOFF16DF = 119 # type: ignore
R_PARISC_LTOFF_FPTR64 = 120 # type: ignore
R_PARISC_LTOFF_FPTR14WR = 123 # type: ignore
R_PARISC_LTOFF_FPTR14DR = 124 # type: ignore
R_PARISC_LTOFF_FPTR16F = 125 # type: ignore
R_PARISC_LTOFF_FPTR16WF = 126 # type: ignore
R_PARISC_LTOFF_FPTR16DF = 127 # type: ignore
R_PARISC_LORESERVE = 128 # type: ignore
R_PARISC_COPY = 128 # type: ignore
R_PARISC_IPLT = 129 # type: ignore
R_PARISC_EPLT = 130 # type: ignore
R_PARISC_TPREL32 = 153 # type: ignore
R_PARISC_TPREL21L = 154 # type: ignore
R_PARISC_TPREL14R = 158 # type: ignore
R_PARISC_LTOFF_TP21L = 162 # type: ignore
R_PARISC_LTOFF_TP14R = 166 # type: ignore
R_PARISC_LTOFF_TP14F = 167 # type: ignore
R_PARISC_TPREL64 = 216 # type: ignore
R_PARISC_TPREL14WR = 219 # type: ignore
R_PARISC_TPREL14DR = 220 # type: ignore
R_PARISC_TPREL16F = 221 # type: ignore
R_PARISC_TPREL16WF = 222 # type: ignore
R_PARISC_TPREL16DF = 223 # type: ignore
R_PARISC_LTOFF_TP64 = 224 # type: ignore
R_PARISC_LTOFF_TP14WR = 227 # type: ignore
R_PARISC_LTOFF_TP14DR = 228 # type: ignore
R_PARISC_LTOFF_TP16F = 229 # type: ignore
R_PARISC_LTOFF_TP16WF = 230 # type: ignore
R_PARISC_LTOFF_TP16DF = 231 # type: ignore
R_PARISC_GNU_VTENTRY = 232 # type: ignore
R_PARISC_GNU_VTINHERIT = 233 # type: ignore
R_PARISC_TLS_GD21L = 234 # type: ignore
R_PARISC_TLS_GD14R = 235 # type: ignore
R_PARISC_TLS_GDCALL = 236 # type: ignore
R_PARISC_TLS_LDM21L = 237 # type: ignore
R_PARISC_TLS_LDM14R = 238 # type: ignore
R_PARISC_TLS_LDMCALL = 239 # type: ignore
R_PARISC_TLS_LDO21L = 240 # type: ignore
R_PARISC_TLS_LDO14R = 241 # type: ignore
R_PARISC_TLS_DTPMOD32 = 242 # type: ignore
R_PARISC_TLS_DTPMOD64 = 243 # type: ignore
R_PARISC_TLS_DTPOFF32 = 244 # type: ignore
R_PARISC_TLS_DTPOFF64 = 245 # type: ignore
R_PARISC_TLS_LE21L = R_PARISC_TPREL21L # type: ignore
R_PARISC_TLS_LE14R = R_PARISC_TPREL14R # type: ignore
R_PARISC_TLS_IE21L = R_PARISC_LTOFF_TP21L # type: ignore
R_PARISC_TLS_IE14R = R_PARISC_LTOFF_TP14R # type: ignore
R_PARISC_TLS_TPREL32 = R_PARISC_TPREL32 # type: ignore
R_PARISC_TLS_TPREL64 = R_PARISC_TPREL64 # type: ignore
R_PARISC_HIRESERVE = 255 # type: ignore
PT_HP_TLS = (PT_LOOS + 0x0) # type: ignore
PT_HP_CORE_NONE = (PT_LOOS + 0x1) # type: ignore
PT_HP_CORE_VERSION = (PT_LOOS + 0x2) # type: ignore
PT_HP_CORE_KERNEL = (PT_LOOS + 0x3) # type: ignore
PT_HP_CORE_COMM = (PT_LOOS + 0x4) # type: ignore
PT_HP_CORE_PROC = (PT_LOOS + 0x5) # type: ignore
PT_HP_CORE_LOADABLE = (PT_LOOS + 0x6) # type: ignore
PT_HP_CORE_STACK = (PT_LOOS + 0x7) # type: ignore
PT_HP_CORE_SHM = (PT_LOOS + 0x8) # type: ignore
PT_HP_CORE_MMF = (PT_LOOS + 0x9) # type: ignore
PT_HP_PARALLEL = (PT_LOOS + 0x10) # type: ignore
PT_HP_FASTBIND = (PT_LOOS + 0x11) # type: ignore
PT_HP_OPT_ANNOT = (PT_LOOS + 0x12) # type: ignore
PT_HP_HSL_ANNOT = (PT_LOOS + 0x13) # type: ignore
PT_HP_STACK = (PT_LOOS + 0x14) # type: ignore
PT_PARISC_ARCHEXT = 0x70000000 # type: ignore
PT_PARISC_UNWIND = 0x70000001 # type: ignore
PF_PARISC_SBP = 0x08000000 # type: ignore
PF_HP_PAGE_SIZE = 0x00100000 # type: ignore
PF_HP_FAR_SHARED = 0x00200000 # type: ignore
PF_HP_NEAR_SHARED = 0x00400000 # type: ignore
PF_HP_CODE = 0x01000000 # type: ignore
PF_HP_MODIFY = 0x02000000 # type: ignore
PF_HP_LAZYSWAP = 0x04000000 # type: ignore
PF_HP_SBP = 0x08000000 # type: ignore
EF_ALPHA_32BIT = 1 # type: ignore
EF_ALPHA_CANRELAX = 2 # type: ignore
SHT_ALPHA_DEBUG = 0x70000001 # type: ignore
SHT_ALPHA_REGINFO = 0x70000002 # type: ignore
SHF_ALPHA_GPREL = 0x10000000 # type: ignore
STO_ALPHA_NOPV = 0x80 # type: ignore
STO_ALPHA_STD_GPLOAD = 0x88 # type: ignore
R_ALPHA_NONE = 0 # type: ignore
R_ALPHA_REFLONG = 1 # type: ignore
R_ALPHA_REFQUAD = 2 # type: ignore
R_ALPHA_GPREL32 = 3 # type: ignore
R_ALPHA_LITERAL = 4 # type: ignore
R_ALPHA_LITUSE = 5 # type: ignore
R_ALPHA_GPDISP = 6 # type: ignore
R_ALPHA_BRADDR = 7 # type: ignore
R_ALPHA_HINT = 8 # type: ignore
R_ALPHA_SREL16 = 9 # type: ignore
R_ALPHA_SREL32 = 10 # type: ignore
R_ALPHA_SREL64 = 11 # type: ignore
R_ALPHA_GPRELHIGH = 17 # type: ignore
R_ALPHA_GPRELLOW = 18 # type: ignore
R_ALPHA_GPREL16 = 19 # type: ignore
R_ALPHA_COPY = 24 # type: ignore
R_ALPHA_GLOB_DAT = 25 # type: ignore
R_ALPHA_JMP_SLOT = 26 # type: ignore
R_ALPHA_RELATIVE = 27 # type: ignore
R_ALPHA_TLS_GD_HI = 28 # type: ignore
R_ALPHA_TLSGD = 29 # type: ignore
R_ALPHA_TLS_LDM = 30 # type: ignore
R_ALPHA_DTPMOD64 = 31 # type: ignore
R_ALPHA_GOTDTPREL = 32 # type: ignore
R_ALPHA_DTPREL64 = 33 # type: ignore
R_ALPHA_DTPRELHI = 34 # type: ignore
R_ALPHA_DTPRELLO = 35 # type: ignore
R_ALPHA_DTPREL16 = 36 # type: ignore
R_ALPHA_GOTTPREL = 37 # type: ignore
R_ALPHA_TPREL64 = 38 # type: ignore
R_ALPHA_TPRELHI = 39 # type: ignore
R_ALPHA_TPRELLO = 40 # type: ignore
R_ALPHA_TPREL16 = 41 # type: ignore
R_ALPHA_NUM = 46 # type: ignore
LITUSE_ALPHA_ADDR = 0 # type: ignore
LITUSE_ALPHA_BASE = 1 # type: ignore
LITUSE_ALPHA_BYTOFF = 2 # type: ignore
LITUSE_ALPHA_JSR = 3 # type: ignore
LITUSE_ALPHA_TLS_GD = 4 # type: ignore
LITUSE_ALPHA_TLS_LDM = 5 # type: ignore
DT_ALPHA_PLTRO = (DT_LOPROC + 0) # type: ignore
DT_ALPHA_NUM = 1 # type: ignore
EF_PPC_EMB = 0x80000000 # type: ignore
EF_PPC_RELOCATABLE = 0x00010000 # type: ignore
EF_PPC_RELOCATABLE_LIB = 0x00008000 # type: ignore
R_PPC_NONE = 0 # type: ignore
R_PPC_ADDR32 = 1 # type: ignore
R_PPC_ADDR24 = 2 # type: ignore
R_PPC_ADDR16 = 3 # type: ignore
R_PPC_ADDR16_LO = 4 # type: ignore
R_PPC_ADDR16_HI = 5 # type: ignore
R_PPC_ADDR16_HA = 6 # type: ignore
R_PPC_ADDR14 = 7 # type: ignore
R_PPC_ADDR14_BRTAKEN = 8 # type: ignore
R_PPC_ADDR14_BRNTAKEN = 9 # type: ignore
R_PPC_REL24 = 10 # type: ignore
R_PPC_REL14 = 11 # type: ignore
R_PPC_REL14_BRTAKEN = 12 # type: ignore
R_PPC_REL14_BRNTAKEN = 13 # type: ignore
R_PPC_GOT16 = 14 # type: ignore
R_PPC_GOT16_LO = 15 # type: ignore
R_PPC_GOT16_HI = 16 # type: ignore
R_PPC_GOT16_HA = 17 # type: ignore
R_PPC_PLTREL24 = 18 # type: ignore
R_PPC_COPY = 19 # type: ignore
R_PPC_GLOB_DAT = 20 # type: ignore
R_PPC_JMP_SLOT = 21 # type: ignore
R_PPC_RELATIVE = 22 # type: ignore
R_PPC_LOCAL24PC = 23 # type: ignore
R_PPC_UADDR32 = 24 # type: ignore
R_PPC_UADDR16 = 25 # type: ignore
R_PPC_REL32 = 26 # type: ignore
R_PPC_PLT32 = 27 # type: ignore
R_PPC_PLTREL32 = 28 # type: ignore
R_PPC_PLT16_LO = 29 # type: ignore
R_PPC_PLT16_HI = 30 # type: ignore
R_PPC_PLT16_HA = 31 # type: ignore
R_PPC_SDAREL16 = 32 # type: ignore
R_PPC_SECTOFF = 33 # type: ignore
R_PPC_SECTOFF_LO = 34 # type: ignore
R_PPC_SECTOFF_HI = 35 # type: ignore
R_PPC_SECTOFF_HA = 36 # type: ignore
R_PPC_TLS = 67 # type: ignore
R_PPC_DTPMOD32 = 68 # type: ignore
R_PPC_TPREL16 = 69 # type: ignore
R_PPC_TPREL16_LO = 70 # type: ignore
R_PPC_TPREL16_HI = 71 # type: ignore
R_PPC_TPREL16_HA = 72 # type: ignore
R_PPC_TPREL32 = 73 # type: ignore
R_PPC_DTPREL16 = 74 # type: ignore
R_PPC_DTPREL16_LO = 75 # type: ignore
R_PPC_DTPREL16_HI = 76 # type: ignore
R_PPC_DTPREL16_HA = 77 # type: ignore
R_PPC_DTPREL32 = 78 # type: ignore
R_PPC_GOT_TLSGD16 = 79 # type: ignore
R_PPC_GOT_TLSGD16_LO = 80 # type: ignore
R_PPC_GOT_TLSGD16_HI = 81 # type: ignore
R_PPC_GOT_TLSGD16_HA = 82 # type: ignore
R_PPC_GOT_TLSLD16 = 83 # type: ignore
R_PPC_GOT_TLSLD16_LO = 84 # type: ignore
R_PPC_GOT_TLSLD16_HI = 85 # type: ignore
R_PPC_GOT_TLSLD16_HA = 86 # type: ignore
R_PPC_GOT_TPREL16 = 87 # type: ignore
R_PPC_GOT_TPREL16_LO = 88 # type: ignore
R_PPC_GOT_TPREL16_HI = 89 # type: ignore
R_PPC_GOT_TPREL16_HA = 90 # type: ignore
R_PPC_GOT_DTPREL16 = 91 # type: ignore
R_PPC_GOT_DTPREL16_LO = 92 # type: ignore
R_PPC_GOT_DTPREL16_HI = 93 # type: ignore
R_PPC_GOT_DTPREL16_HA = 94 # type: ignore
R_PPC_TLSGD = 95 # type: ignore
R_PPC_TLSLD = 96 # type: ignore
R_PPC_EMB_NADDR32 = 101 # type: ignore
R_PPC_EMB_NADDR16 = 102 # type: ignore
R_PPC_EMB_NADDR16_LO = 103 # type: ignore
R_PPC_EMB_NADDR16_HI = 104 # type: ignore
R_PPC_EMB_NADDR16_HA = 105 # type: ignore
R_PPC_EMB_SDAI16 = 106 # type: ignore
R_PPC_EMB_SDA2I16 = 107 # type: ignore
R_PPC_EMB_SDA2REL = 108 # type: ignore
R_PPC_EMB_SDA21 = 109 # type: ignore
R_PPC_EMB_MRKREF = 110 # type: ignore
R_PPC_EMB_RELSEC16 = 111 # type: ignore
R_PPC_EMB_RELST_LO = 112 # type: ignore
R_PPC_EMB_RELST_HI = 113 # type: ignore
R_PPC_EMB_RELST_HA = 114 # type: ignore
R_PPC_EMB_BIT_FLD = 115 # type: ignore
R_PPC_EMB_RELSDA = 116 # type: ignore
R_PPC_DIAB_SDA21_LO = 180 # type: ignore
R_PPC_DIAB_SDA21_HI = 181 # type: ignore
R_PPC_DIAB_SDA21_HA = 182 # type: ignore
R_PPC_DIAB_RELSDA_LO = 183 # type: ignore
R_PPC_DIAB_RELSDA_HI = 184 # type: ignore
R_PPC_DIAB_RELSDA_HA = 185 # type: ignore
R_PPC_IRELATIVE = 248 # type: ignore
R_PPC_REL16 = 249 # type: ignore
R_PPC_REL16_LO = 250 # type: ignore
R_PPC_REL16_HI = 251 # type: ignore
R_PPC_REL16_HA = 252 # type: ignore
R_PPC_TOC16 = 255 # type: ignore
DT_PPC_GOT = (DT_LOPROC + 0) # type: ignore
DT_PPC_OPT = (DT_LOPROC + 1) # type: ignore
DT_PPC_NUM = 2 # type: ignore
PPC_OPT_TLS = 1 # type: ignore
R_PPC64_NONE = R_PPC_NONE # type: ignore
R_PPC64_ADDR32 = R_PPC_ADDR32 # type: ignore
R_PPC64_ADDR24 = R_PPC_ADDR24 # type: ignore
R_PPC64_ADDR16 = R_PPC_ADDR16 # type: ignore
R_PPC64_ADDR16_LO = R_PPC_ADDR16_LO # type: ignore
R_PPC64_ADDR16_HI = R_PPC_ADDR16_HI # type: ignore
R_PPC64_ADDR16_HA = R_PPC_ADDR16_HA # type: ignore
R_PPC64_ADDR14 = R_PPC_ADDR14 # type: ignore
R_PPC64_ADDR14_BRTAKEN = R_PPC_ADDR14_BRTAKEN # type: ignore
R_PPC64_ADDR14_BRNTAKEN = R_PPC_ADDR14_BRNTAKEN # type: ignore
R_PPC64_REL24 = R_PPC_REL24 # type: ignore
R_PPC64_REL14 = R_PPC_REL14 # type: ignore
R_PPC64_REL14_BRTAKEN = R_PPC_REL14_BRTAKEN # type: ignore
R_PPC64_REL14_BRNTAKEN = R_PPC_REL14_BRNTAKEN # type: ignore
R_PPC64_GOT16 = R_PPC_GOT16 # type: ignore
R_PPC64_GOT16_LO = R_PPC_GOT16_LO # type: ignore
R_PPC64_GOT16_HI = R_PPC_GOT16_HI # type: ignore
R_PPC64_GOT16_HA = R_PPC_GOT16_HA # type: ignore
R_PPC64_COPY = R_PPC_COPY # type: ignore
R_PPC64_GLOB_DAT = R_PPC_GLOB_DAT # type: ignore
R_PPC64_JMP_SLOT = R_PPC_JMP_SLOT # type: ignore
R_PPC64_RELATIVE = R_PPC_RELATIVE # type: ignore
R_PPC64_UADDR32 = R_PPC_UADDR32 # type: ignore
R_PPC64_UADDR16 = R_PPC_UADDR16 # type: ignore
R_PPC64_REL32 = R_PPC_REL32 # type: ignore
R_PPC64_PLT32 = R_PPC_PLT32 # type: ignore
R_PPC64_PLTREL32 = R_PPC_PLTREL32 # type: ignore
R_PPC64_PLT16_LO = R_PPC_PLT16_LO # type: ignore
R_PPC64_PLT16_HI = R_PPC_PLT16_HI # type: ignore
R_PPC64_PLT16_HA = R_PPC_PLT16_HA # type: ignore
R_PPC64_SECTOFF = R_PPC_SECTOFF # type: ignore
R_PPC64_SECTOFF_LO = R_PPC_SECTOFF_LO # type: ignore
R_PPC64_SECTOFF_HI = R_PPC_SECTOFF_HI # type: ignore
R_PPC64_SECTOFF_HA = R_PPC_SECTOFF_HA # type: ignore
R_PPC64_ADDR30 = 37 # type: ignore
R_PPC64_ADDR64 = 38 # type: ignore
R_PPC64_ADDR16_HIGHER = 39 # type: ignore
R_PPC64_ADDR16_HIGHERA = 40 # type: ignore
R_PPC64_ADDR16_HIGHEST = 41 # type: ignore
R_PPC64_ADDR16_HIGHESTA = 42 # type: ignore
R_PPC64_UADDR64 = 43 # type: ignore
R_PPC64_REL64 = 44 # type: ignore
R_PPC64_PLT64 = 45 # type: ignore
R_PPC64_PLTREL64 = 46 # type: ignore
R_PPC64_TOC16 = 47 # type: ignore
R_PPC64_TOC16_LO = 48 # type: ignore
R_PPC64_TOC16_HI = 49 # type: ignore
R_PPC64_TOC16_HA = 50 # type: ignore
R_PPC64_TOC = 51 # type: ignore
R_PPC64_PLTGOT16 = 52 # type: ignore
R_PPC64_PLTGOT16_LO = 53 # type: ignore
R_PPC64_PLTGOT16_HI = 54 # type: ignore
R_PPC64_PLTGOT16_HA = 55 # type: ignore
R_PPC64_ADDR16_DS = 56 # type: ignore
R_PPC64_ADDR16_LO_DS = 57 # type: ignore
R_PPC64_GOT16_DS = 58 # type: ignore
R_PPC64_GOT16_LO_DS = 59 # type: ignore
R_PPC64_PLT16_LO_DS = 60 # type: ignore
R_PPC64_SECTOFF_DS = 61 # type: ignore
R_PPC64_SECTOFF_LO_DS = 62 # type: ignore
R_PPC64_TOC16_DS = 63 # type: ignore
R_PPC64_TOC16_LO_DS = 64 # type: ignore
R_PPC64_PLTGOT16_DS = 65 # type: ignore
R_PPC64_PLTGOT16_LO_DS = 66 # type: ignore
R_PPC64_TLS = 67 # type: ignore
R_PPC64_DTPMOD64 = 68 # type: ignore
R_PPC64_TPREL16 = 69 # type: ignore
R_PPC64_TPREL16_LO = 70 # type: ignore
R_PPC64_TPREL16_HI = 71 # type: ignore
R_PPC64_TPREL16_HA = 72 # type: ignore
R_PPC64_TPREL64 = 73 # type: ignore
R_PPC64_DTPREL16 = 74 # type: ignore
R_PPC64_DTPREL16_LO = 75 # type: ignore
R_PPC64_DTPREL16_HI = 76 # type: ignore
R_PPC64_DTPREL16_HA = 77 # type: ignore
R_PPC64_DTPREL64 = 78 # type: ignore
R_PPC64_GOT_TLSGD16 = 79 # type: ignore
R_PPC64_GOT_TLSGD16_LO = 80 # type: ignore
R_PPC64_GOT_TLSGD16_HI = 81 # type: ignore
R_PPC64_GOT_TLSGD16_HA = 82 # type: ignore
R_PPC64_GOT_TLSLD16 = 83 # type: ignore
R_PPC64_GOT_TLSLD16_LO = 84 # type: ignore
R_PPC64_GOT_TLSLD16_HI = 85 # type: ignore
R_PPC64_GOT_TLSLD16_HA = 86 # type: ignore
R_PPC64_GOT_TPREL16_DS = 87 # type: ignore
R_PPC64_GOT_TPREL16_LO_DS = 88 # type: ignore
R_PPC64_GOT_TPREL16_HI = 89 # type: ignore
R_PPC64_GOT_TPREL16_HA = 90 # type: ignore
R_PPC64_GOT_DTPREL16_DS = 91 # type: ignore
R_PPC64_GOT_DTPREL16_LO_DS = 92 # type: ignore
R_PPC64_GOT_DTPREL16_HI = 93 # type: ignore
R_PPC64_GOT_DTPREL16_HA = 94 # type: ignore
R_PPC64_TPREL16_DS = 95 # type: ignore
R_PPC64_TPREL16_LO_DS = 96 # type: ignore
R_PPC64_TPREL16_HIGHER = 97 # type: ignore
R_PPC64_TPREL16_HIGHERA = 98 # type: ignore
R_PPC64_TPREL16_HIGHEST = 99 # type: ignore
R_PPC64_TPREL16_HIGHESTA = 100 # type: ignore
R_PPC64_DTPREL16_DS = 101 # type: ignore
R_PPC64_DTPREL16_LO_DS = 102 # type: ignore
R_PPC64_DTPREL16_HIGHER = 103 # type: ignore
R_PPC64_DTPREL16_HIGHERA = 104 # type: ignore
R_PPC64_DTPREL16_HIGHEST = 105 # type: ignore
R_PPC64_DTPREL16_HIGHESTA = 106 # type: ignore
R_PPC64_TLSGD = 107 # type: ignore
R_PPC64_TLSLD = 108 # type: ignore
R_PPC64_TOCSAVE = 109 # type: ignore
R_PPC64_ADDR16_HIGH = 110 # type: ignore
R_PPC64_ADDR16_HIGHA = 111 # type: ignore
R_PPC64_TPREL16_HIGH = 112 # type: ignore
R_PPC64_TPREL16_HIGHA = 113 # type: ignore
R_PPC64_DTPREL16_HIGH = 114 # type: ignore
R_PPC64_DTPREL16_HIGHA = 115 # type: ignore
R_PPC64_JMP_IREL = 247 # type: ignore
R_PPC64_IRELATIVE = 248 # type: ignore
R_PPC64_REL16 = 249 # type: ignore
R_PPC64_REL16_LO = 250 # type: ignore
R_PPC64_REL16_HI = 251 # type: ignore
R_PPC64_REL16_HA = 252 # type: ignore
EF_PPC64_ABI = 3 # type: ignore
DT_PPC64_GLINK = (DT_LOPROC + 0) # type: ignore
DT_PPC64_OPD = (DT_LOPROC + 1) # type: ignore
DT_PPC64_OPDSZ = (DT_LOPROC + 2) # type: ignore
DT_PPC64_OPT = (DT_LOPROC + 3) # type: ignore
DT_PPC64_NUM = 4 # type: ignore
PPC64_OPT_TLS = 1 # type: ignore
PPC64_OPT_MULTI_TOC = 2 # type: ignore
PPC64_OPT_LOCALENTRY = 4 # type: ignore
STO_PPC64_LOCAL_BIT = 5 # type: ignore
STO_PPC64_LOCAL_MASK = (7 << STO_PPC64_LOCAL_BIT) # type: ignore
PPC64_LOCAL_ENTRY_OFFSET = lambda other: (((1 << (((other) & STO_PPC64_LOCAL_MASK) >> STO_PPC64_LOCAL_BIT)) >> 2) << 2) # type: ignore
EF_ARM_RELEXEC = 0x01 # type: ignore
EF_ARM_HASENTRY = 0x02 # type: ignore
EF_ARM_INTERWORK = 0x04 # type: ignore
EF_ARM_APCS_26 = 0x08 # type: ignore
EF_ARM_APCS_FLOAT = 0x10 # type: ignore
EF_ARM_PIC = 0x20 # type: ignore
EF_ARM_ALIGN8 = 0x40 # type: ignore
EF_ARM_NEW_ABI = 0x80 # type: ignore
EF_ARM_OLD_ABI = 0x100 # type: ignore
EF_ARM_SOFT_FLOAT = 0x200 # type: ignore
EF_ARM_VFP_FLOAT = 0x400 # type: ignore
EF_ARM_MAVERICK_FLOAT = 0x800 # type: ignore
EF_ARM_ABI_FLOAT_SOFT = 0x200 # type: ignore
EF_ARM_ABI_FLOAT_HARD = 0x400 # type: ignore
EF_ARM_SYMSARESORTED = 0x04 # type: ignore
EF_ARM_DYNSYMSUSESEGIDX = 0x08 # type: ignore
EF_ARM_MAPSYMSFIRST = 0x10 # type: ignore
EF_ARM_EABIMASK = 0XFF000000 # type: ignore
EF_ARM_BE8 = 0x00800000 # type: ignore
EF_ARM_LE8 = 0x00400000 # type: ignore
EF_ARM_EABI_VERSION = lambda flags: ((flags) & EF_ARM_EABIMASK) # type: ignore
EF_ARM_EABI_UNKNOWN = 0x00000000 # type: ignore
EF_ARM_EABI_VER1 = 0x01000000 # type: ignore
EF_ARM_EABI_VER2 = 0x02000000 # type: ignore
EF_ARM_EABI_VER3 = 0x03000000 # type: ignore
EF_ARM_EABI_VER4 = 0x04000000 # type: ignore
EF_ARM_EABI_VER5 = 0x05000000 # type: ignore
STT_ARM_TFUNC = STT_LOPROC # type: ignore
STT_ARM_16BIT = STT_HIPROC # type: ignore
SHF_ARM_ENTRYSECT = 0x10000000 # type: ignore
SHF_ARM_COMDEF = 0x80000000 # type: ignore
PF_ARM_SB = 0x10000000 # type: ignore
PF_ARM_PI = 0x20000000 # type: ignore
PF_ARM_ABS = 0x40000000 # type: ignore
PT_ARM_EXIDX = (PT_LOPROC + 1) # type: ignore
SHT_ARM_EXIDX = (SHT_LOPROC + 1) # type: ignore
SHT_ARM_PREEMPTMAP = (SHT_LOPROC + 2) # type: ignore
SHT_ARM_ATTRIBUTES = (SHT_LOPROC + 3) # type: ignore
R_AARCH64_NONE = 0 # type: ignore
R_AARCH64_P32_ABS32 = 1 # type: ignore
R_AARCH64_P32_COPY = 180 # type: ignore
R_AARCH64_P32_GLOB_DAT = 181 # type: ignore
R_AARCH64_P32_JUMP_SLOT = 182 # type: ignore
R_AARCH64_P32_RELATIVE = 183 # type: ignore
R_AARCH64_P32_TLS_DTPMOD = 184 # type: ignore
R_AARCH64_P32_TLS_DTPREL = 185 # type: ignore
R_AARCH64_P32_TLS_TPREL = 186 # type: ignore
R_AARCH64_P32_TLSDESC = 187 # type: ignore
R_AARCH64_P32_IRELATIVE = 188 # type: ignore
R_AARCH64_ABS64 = 257 # type: ignore
R_AARCH64_ABS32 = 258 # type: ignore
R_AARCH64_ABS16 = 259 # type: ignore
R_AARCH64_PREL64 = 260 # type: ignore
R_AARCH64_PREL32 = 261 # type: ignore
R_AARCH64_PREL16 = 262 # type: ignore
R_AARCH64_MOVW_UABS_G0 = 263 # type: ignore
R_AARCH64_MOVW_UABS_G0_NC = 264 # type: ignore
R_AARCH64_MOVW_UABS_G1 = 265 # type: ignore
R_AARCH64_MOVW_UABS_G1_NC = 266 # type: ignore
R_AARCH64_MOVW_UABS_G2 = 267 # type: ignore
R_AARCH64_MOVW_UABS_G2_NC = 268 # type: ignore
R_AARCH64_MOVW_UABS_G3 = 269 # type: ignore
R_AARCH64_MOVW_SABS_G0 = 270 # type: ignore
R_AARCH64_MOVW_SABS_G1 = 271 # type: ignore
R_AARCH64_MOVW_SABS_G2 = 272 # type: ignore
R_AARCH64_LD_PREL_LO19 = 273 # type: ignore
R_AARCH64_ADR_PREL_LO21 = 274 # type: ignore
R_AARCH64_ADR_PREL_PG_HI21 = 275 # type: ignore
R_AARCH64_ADR_PREL_PG_HI21_NC = 276 # type: ignore
R_AARCH64_ADD_ABS_LO12_NC = 277 # type: ignore
R_AARCH64_LDST8_ABS_LO12_NC = 278 # type: ignore
R_AARCH64_TSTBR14 = 279 # type: ignore
R_AARCH64_CONDBR19 = 280 # type: ignore
R_AARCH64_JUMP26 = 282 # type: ignore
R_AARCH64_CALL26 = 283 # type: ignore
R_AARCH64_LDST16_ABS_LO12_NC = 284 # type: ignore
R_AARCH64_LDST32_ABS_LO12_NC = 285 # type: ignore
R_AARCH64_LDST64_ABS_LO12_NC = 286 # type: ignore
R_AARCH64_MOVW_PREL_G0 = 287 # type: ignore
R_AARCH64_MOVW_PREL_G0_NC = 288 # type: ignore
R_AARCH64_MOVW_PREL_G1 = 289 # type: ignore
R_AARCH64_MOVW_PREL_G1_NC = 290 # type: ignore
R_AARCH64_MOVW_PREL_G2 = 291 # type: ignore
R_AARCH64_MOVW_PREL_G2_NC = 292 # type: ignore
R_AARCH64_MOVW_PREL_G3 = 293 # type: ignore
R_AARCH64_LDST128_ABS_LO12_NC = 299 # type: ignore
R_AARCH64_MOVW_GOTOFF_G0 = 300 # type: ignore
R_AARCH64_MOVW_GOTOFF_G0_NC = 301 # type: ignore
R_AARCH64_MOVW_GOTOFF_G1 = 302 # type: ignore
R_AARCH64_MOVW_GOTOFF_G1_NC = 303 # type: ignore
R_AARCH64_MOVW_GOTOFF_G2 = 304 # type: ignore
R_AARCH64_MOVW_GOTOFF_G2_NC = 305 # type: ignore
R_AARCH64_MOVW_GOTOFF_G3 = 306 # type: ignore
R_AARCH64_GOTREL64 = 307 # type: ignore
R_AARCH64_GOTREL32 = 308 # type: ignore
R_AARCH64_GOT_LD_PREL19 = 309 # type: ignore
R_AARCH64_LD64_GOTOFF_LO15 = 310 # type: ignore
R_AARCH64_ADR_GOT_PAGE = 311 # type: ignore
R_AARCH64_LD64_GOT_LO12_NC = 312 # type: ignore
R_AARCH64_LD64_GOTPAGE_LO15 = 313 # type: ignore
R_AARCH64_TLSGD_ADR_PREL21 = 512 # type: ignore
R_AARCH64_TLSGD_ADR_PAGE21 = 513 # type: ignore
R_AARCH64_TLSGD_ADD_LO12_NC = 514 # type: ignore
R_AARCH64_TLSGD_MOVW_G1 = 515 # type: ignore
R_AARCH64_TLSGD_MOVW_G0_NC = 516 # type: ignore
R_AARCH64_TLSLD_ADR_PREL21 = 517 # type: ignore
R_AARCH64_TLSLD_ADR_PAGE21 = 518 # type: ignore
R_AARCH64_TLSLD_ADD_LO12_NC = 519 # type: ignore
R_AARCH64_TLSLD_MOVW_G1 = 520 # type: ignore
R_AARCH64_TLSLD_MOVW_G0_NC = 521 # type: ignore
R_AARCH64_TLSLD_LD_PREL19 = 522 # type: ignore
R_AARCH64_TLSLD_MOVW_DTPREL_G2 = 523 # type: ignore
R_AARCH64_TLSLD_MOVW_DTPREL_G1 = 524 # type: ignore
R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC = 525 # type: ignore
R_AARCH64_TLSLD_MOVW_DTPREL_G0 = 526 # type: ignore
R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC = 527 # type: ignore
R_AARCH64_TLSLD_ADD_DTPREL_HI12 = 528 # type: ignore
R_AARCH64_TLSLD_ADD_DTPREL_LO12 = 529 # type: ignore
R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC = 530 # type: ignore
R_AARCH64_TLSLD_LDST8_DTPREL_LO12 = 531 # type: ignore
R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC = 532 # type: ignore
R_AARCH64_TLSLD_LDST16_DTPREL_LO12 = 533 # type: ignore
R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC = 534 # type: ignore
R_AARCH64_TLSLD_LDST32_DTPREL_LO12 = 535 # type: ignore
R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC = 536 # type: ignore
R_AARCH64_TLSLD_LDST64_DTPREL_LO12 = 537 # type: ignore
R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC = 538 # type: ignore
R_AARCH64_TLSIE_MOVW_GOTTPREL_G1 = 539 # type: ignore
R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC = 540 # type: ignore
R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 = 541 # type: ignore
R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC = 542 # type: ignore
R_AARCH64_TLSIE_LD_GOTTPREL_PREL19 = 543 # type: ignore
R_AARCH64_TLSLE_MOVW_TPREL_G2 = 544 # type: ignore
R_AARCH64_TLSLE_MOVW_TPREL_G1 = 545 # type: ignore
R_AARCH64_TLSLE_MOVW_TPREL_G1_NC = 546 # type: ignore
R_AARCH64_TLSLE_MOVW_TPREL_G0 = 547 # type: ignore
R_AARCH64_TLSLE_MOVW_TPREL_G0_NC = 548 # type: ignore
R_AARCH64_TLSLE_ADD_TPREL_HI12 = 549 # type: ignore
R_AARCH64_TLSLE_ADD_TPREL_LO12 = 550 # type: ignore
R_AARCH64_TLSLE_ADD_TPREL_LO12_NC = 551 # type: ignore
R_AARCH64_TLSLE_LDST8_TPREL_LO12 = 552 # type: ignore
R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC = 553 # type: ignore
R_AARCH64_TLSLE_LDST16_TPREL_LO12 = 554 # type: ignore
R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC = 555 # type: ignore
R_AARCH64_TLSLE_LDST32_TPREL_LO12 = 556 # type: ignore
R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC = 557 # type: ignore
R_AARCH64_TLSLE_LDST64_TPREL_LO12 = 558 # type: ignore
R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC = 559 # type: ignore
R_AARCH64_TLSDESC_LD_PREL19 = 560 # type: ignore
R_AARCH64_TLSDESC_ADR_PREL21 = 561 # type: ignore
R_AARCH64_TLSDESC_ADR_PAGE21 = 562 # type: ignore
R_AARCH64_TLSDESC_LD64_LO12 = 563 # type: ignore
R_AARCH64_TLSDESC_ADD_LO12 = 564 # type: ignore
R_AARCH64_TLSDESC_OFF_G1 = 565 # type: ignore
R_AARCH64_TLSDESC_OFF_G0_NC = 566 # type: ignore
R_AARCH64_TLSDESC_LDR = 567 # type: ignore
R_AARCH64_TLSDESC_ADD = 568 # type: ignore
R_AARCH64_TLSDESC_CALL = 569 # type: ignore
R_AARCH64_TLSLE_LDST128_TPREL_LO12 = 570 # type: ignore
R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC = 571 # type: ignore
R_AARCH64_TLSLD_LDST128_DTPREL_LO12 = 572 # type: ignore
R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC = 573 # type: ignore
R_AARCH64_COPY = 1024 # type: ignore
R_AARCH64_GLOB_DAT = 1025 # type: ignore
R_AARCH64_JUMP_SLOT = 1026 # type: ignore
R_AARCH64_RELATIVE = 1027 # type: ignore
R_AARCH64_TLS_DTPMOD = 1028 # type: ignore
R_AARCH64_TLS_DTPREL = 1029 # type: ignore
R_AARCH64_TLS_TPREL = 1030 # type: ignore
R_AARCH64_TLSDESC = 1031 # type: ignore
R_AARCH64_IRELATIVE = 1032 # type: ignore
PT_AARCH64_MEMTAG_MTE = (PT_LOPROC + 2) # type: ignore
DT_AARCH64_BTI_PLT = (DT_LOPROC + 1) # type: ignore
DT_AARCH64_PAC_PLT = (DT_LOPROC + 3) # type: ignore
DT_AARCH64_VARIANT_PCS = (DT_LOPROC + 5) # type: ignore
DT_AARCH64_NUM = 6 # type: ignore
STO_AARCH64_VARIANT_PCS = 0x80 # type: ignore
R_ARM_NONE = 0 # type: ignore
R_ARM_PC24 = 1 # type: ignore
R_ARM_ABS32 = 2 # type: ignore
R_ARM_REL32 = 3 # type: ignore
R_ARM_PC13 = 4 # type: ignore
R_ARM_ABS16 = 5 # type: ignore
R_ARM_ABS12 = 6 # type: ignore
R_ARM_THM_ABS5 = 7 # type: ignore
R_ARM_ABS8 = 8 # type: ignore
R_ARM_SBREL32 = 9 # type: ignore
R_ARM_THM_PC22 = 10 # type: ignore
R_ARM_THM_PC8 = 11 # type: ignore
R_ARM_AMP_VCALL9 = 12 # type: ignore
R_ARM_SWI24 = 13 # type: ignore
R_ARM_TLS_DESC = 13 # type: ignore
R_ARM_THM_SWI8 = 14 # type: ignore
R_ARM_XPC25 = 15 # type: ignore
R_ARM_THM_XPC22 = 16 # type: ignore
R_ARM_TLS_DTPMOD32 = 17 # type: ignore
R_ARM_TLS_DTPOFF32 = 18 # type: ignore
R_ARM_TLS_TPOFF32 = 19 # type: ignore
R_ARM_COPY = 20 # type: ignore
R_ARM_GLOB_DAT = 21 # type: ignore
R_ARM_JUMP_SLOT = 22 # type: ignore
R_ARM_RELATIVE = 23 # type: ignore
R_ARM_GOTOFF = 24 # type: ignore
R_ARM_GOTPC = 25 # type: ignore
R_ARM_GOT32 = 26 # type: ignore
R_ARM_PLT32 = 27 # type: ignore
R_ARM_CALL = 28 # type: ignore
R_ARM_JUMP24 = 29 # type: ignore
R_ARM_THM_JUMP24 = 30 # type: ignore
R_ARM_BASE_ABS = 31 # type: ignore
R_ARM_ALU_PCREL_7_0 = 32 # type: ignore
R_ARM_ALU_PCREL_15_8 = 33 # type: ignore
R_ARM_ALU_PCREL_23_15 = 34 # type: ignore
R_ARM_LDR_SBREL_11_0 = 35 # type: ignore
R_ARM_ALU_SBREL_19_12 = 36 # type: ignore
R_ARM_ALU_SBREL_27_20 = 37 # type: ignore
R_ARM_TARGET1 = 38 # type: ignore
R_ARM_SBREL31 = 39 # type: ignore
R_ARM_V4BX = 40 # type: ignore
R_ARM_TARGET2 = 41 # type: ignore
R_ARM_PREL31 = 42 # type: ignore
R_ARM_MOVW_ABS_NC = 43 # type: ignore
R_ARM_MOVT_ABS = 44 # type: ignore
R_ARM_MOVW_PREL_NC = 45 # type: ignore
R_ARM_MOVT_PREL = 46 # type: ignore
R_ARM_THM_MOVW_ABS_NC = 47 # type: ignore
R_ARM_THM_MOVT_ABS = 48 # type: ignore
R_ARM_THM_MOVW_PREL_NC = 49 # type: ignore
R_ARM_THM_MOVT_PREL = 50 # type: ignore
R_ARM_THM_JUMP19 = 51 # type: ignore
R_ARM_THM_JUMP6 = 52 # type: ignore
R_ARM_THM_ALU_PREL_11_0 = 53 # type: ignore
R_ARM_THM_PC12 = 54 # type: ignore
R_ARM_ABS32_NOI = 55 # type: ignore
R_ARM_REL32_NOI = 56 # type: ignore
R_ARM_ALU_PC_G0_NC = 57 # type: ignore
R_ARM_ALU_PC_G0 = 58 # type: ignore
R_ARM_ALU_PC_G1_NC = 59 # type: ignore
R_ARM_ALU_PC_G1 = 60 # type: ignore
R_ARM_ALU_PC_G2 = 61 # type: ignore
R_ARM_LDR_PC_G1 = 62 # type: ignore
R_ARM_LDR_PC_G2 = 63 # type: ignore
R_ARM_LDRS_PC_G0 = 64 # type: ignore
R_ARM_LDRS_PC_G1 = 65 # type: ignore
R_ARM_LDRS_PC_G2 = 66 # type: ignore
R_ARM_LDC_PC_G0 = 67 # type: ignore
R_ARM_LDC_PC_G1 = 68 # type: ignore
R_ARM_LDC_PC_G2 = 69 # type: ignore
R_ARM_ALU_SB_G0_NC = 70 # type: ignore
R_ARM_ALU_SB_G0 = 71 # type: ignore
R_ARM_ALU_SB_G1_NC = 72 # type: ignore
R_ARM_ALU_SB_G1 = 73 # type: ignore
R_ARM_ALU_SB_G2 = 74 # type: ignore
R_ARM_LDR_SB_G0 = 75 # type: ignore
R_ARM_LDR_SB_G1 = 76 # type: ignore
R_ARM_LDR_SB_G2 = 77 # type: ignore
R_ARM_LDRS_SB_G0 = 78 # type: ignore
R_ARM_LDRS_SB_G1 = 79 # type: ignore
R_ARM_LDRS_SB_G2 = 80 # type: ignore
R_ARM_LDC_SB_G0 = 81 # type: ignore
R_ARM_LDC_SB_G1 = 82 # type: ignore
R_ARM_LDC_SB_G2 = 83 # type: ignore
R_ARM_MOVW_BREL_NC = 84 # type: ignore
R_ARM_MOVT_BREL = 85 # type: ignore
R_ARM_MOVW_BREL = 86 # type: ignore
R_ARM_THM_MOVW_BREL_NC = 87 # type: ignore
R_ARM_THM_MOVT_BREL = 88 # type: ignore
R_ARM_THM_MOVW_BREL = 89 # type: ignore
R_ARM_TLS_GOTDESC = 90 # type: ignore
R_ARM_TLS_CALL = 91 # type: ignore
R_ARM_TLS_DESCSEQ = 92 # type: ignore
R_ARM_THM_TLS_CALL = 93 # type: ignore
R_ARM_PLT32_ABS = 94 # type: ignore
R_ARM_GOT_ABS = 95 # type: ignore
R_ARM_GOT_PREL = 96 # type: ignore
R_ARM_GOT_BREL12 = 97 # type: ignore
R_ARM_GOTOFF12 = 98 # type: ignore
R_ARM_GOTRELAX = 99 # type: ignore
R_ARM_GNU_VTENTRY = 100 # type: ignore
R_ARM_GNU_VTINHERIT = 101 # type: ignore
R_ARM_THM_PC11 = 102 # type: ignore
R_ARM_THM_PC9 = 103 # type: ignore
R_ARM_TLS_GD32 = 104 # type: ignore
R_ARM_TLS_LDM32 = 105 # type: ignore
R_ARM_TLS_LDO32 = 106 # type: ignore
R_ARM_TLS_IE32 = 107 # type: ignore
R_ARM_TLS_LE32 = 108 # type: ignore
R_ARM_TLS_LDO12 = 109 # type: ignore
R_ARM_TLS_LE12 = 110 # type: ignore
R_ARM_TLS_IE12GP = 111 # type: ignore
R_ARM_ME_TOO = 128 # type: ignore
R_ARM_THM_TLS_DESCSEQ = 129 # type: ignore
R_ARM_THM_TLS_DESCSEQ16 = 129 # type: ignore
R_ARM_THM_TLS_DESCSEQ32 = 130 # type: ignore
R_ARM_THM_GOT_BREL12 = 131 # type: ignore
R_ARM_IRELATIVE = 160 # type: ignore
R_ARM_RXPC25 = 249 # type: ignore
R_ARM_RSBREL32 = 250 # type: ignore
R_ARM_THM_RPC22 = 251 # type: ignore
R_ARM_RREL32 = 252 # type: ignore
R_ARM_RABS22 = 253 # type: ignore
R_ARM_RPC24 = 254 # type: ignore
R_ARM_RBASE = 255 # type: ignore
R_ARM_NUM = 256 # type: ignore
R_CKCORE_NONE = 0 # type: ignore
R_CKCORE_ADDR32 = 1 # type: ignore
R_CKCORE_PCRELIMM8BY4 = 2 # type: ignore
R_CKCORE_PCRELIMM11BY2 = 3 # type: ignore
R_CKCORE_PCREL32 = 5 # type: ignore
R_CKCORE_PCRELJSR_IMM11BY2 = 6 # type: ignore
R_CKCORE_RELATIVE = 9 # type: ignore
R_CKCORE_COPY = 10 # type: ignore
R_CKCORE_GLOB_DAT = 11 # type: ignore
R_CKCORE_JUMP_SLOT = 12 # type: ignore
R_CKCORE_GOTOFF = 13 # type: ignore
R_CKCORE_GOTPC = 14 # type: ignore
R_CKCORE_GOT32 = 15 # type: ignore
R_CKCORE_PLT32 = 16 # type: ignore
R_CKCORE_ADDRGOT = 17 # type: ignore
R_CKCORE_ADDRPLT = 18 # type: ignore
R_CKCORE_PCREL_IMM26BY2 = 19 # type: ignore
R_CKCORE_PCREL_IMM16BY2 = 20 # type: ignore
R_CKCORE_PCREL_IMM16BY4 = 21 # type: ignore
R_CKCORE_PCREL_IMM10BY2 = 22 # type: ignore
R_CKCORE_PCREL_IMM10BY4 = 23 # type: ignore
R_CKCORE_ADDR_HI16 = 24 # type: ignore
R_CKCORE_ADDR_LO16 = 25 # type: ignore
R_CKCORE_GOTPC_HI16 = 26 # type: ignore
R_CKCORE_GOTPC_LO16 = 27 # type: ignore
R_CKCORE_GOTOFF_HI16 = 28 # type: ignore
R_CKCORE_GOTOFF_LO16 = 29 # type: ignore
R_CKCORE_GOT12 = 30 # type: ignore
R_CKCORE_GOT_HI16 = 31 # type: ignore
R_CKCORE_GOT_LO16 = 32 # type: ignore
R_CKCORE_PLT12 = 33 # type: ignore
R_CKCORE_PLT_HI16 = 34 # type: ignore
R_CKCORE_PLT_LO16 = 35 # type: ignore
R_CKCORE_ADDRGOT_HI16 = 36 # type: ignore
R_CKCORE_ADDRGOT_LO16 = 37 # type: ignore
R_CKCORE_ADDRPLT_HI16 = 38 # type: ignore
R_CKCORE_ADDRPLT_LO16 = 39 # type: ignore
R_CKCORE_PCREL_JSR_IMM26BY2 = 40 # type: ignore
R_CKCORE_TOFFSET_LO16 = 41 # type: ignore
R_CKCORE_DOFFSET_LO16 = 42 # type: ignore
R_CKCORE_PCREL_IMM18BY2 = 43 # type: ignore
R_CKCORE_DOFFSET_IMM18 = 44 # type: ignore
R_CKCORE_DOFFSET_IMM18BY2 = 45 # type: ignore
R_CKCORE_DOFFSET_IMM18BY4 = 46 # type: ignore
R_CKCORE_GOT_IMM18BY4 = 48 # type: ignore
R_CKCORE_PLT_IMM18BY4 = 49 # type: ignore
R_CKCORE_PCREL_IMM7BY4 = 50 # type: ignore
R_CKCORE_TLS_LE32 = 51 # type: ignore
R_CKCORE_TLS_IE32 = 52 # type: ignore
R_CKCORE_TLS_GD32 = 53 # type: ignore
R_CKCORE_TLS_LDM32 = 54 # type: ignore
R_CKCORE_TLS_LDO32 = 55 # type: ignore
R_CKCORE_TLS_DTPMOD32 = 56 # type: ignore
R_CKCORE_TLS_DTPOFF32 = 57 # type: ignore
R_CKCORE_TLS_TPOFF32 = 58 # type: ignore
EF_CSKY_ABIMASK = 0XF0000000 # type: ignore
EF_CSKY_OTHER = 0X0FFF0000 # type: ignore
EF_CSKY_PROCESSOR = 0X0000FFFF # type: ignore
EF_CSKY_ABIV1 = 0X10000000 # type: ignore
EF_CSKY_ABIV2 = 0X20000000 # type: ignore
SHT_CSKY_ATTRIBUTES = (SHT_LOPROC + 1) # type: ignore
EF_IA_64_MASKOS = 0x0000000f # type: ignore
EF_IA_64_ABI64 = 0x00000010 # type: ignore
EF_IA_64_ARCH = 0xff000000 # type: ignore
PT_IA_64_ARCHEXT = (PT_LOPROC + 0) # type: ignore
PT_IA_64_UNWIND = (PT_LOPROC + 1) # type: ignore
PT_IA_64_HP_OPT_ANOT = (PT_LOOS + 0x12) # type: ignore
PT_IA_64_HP_HSL_ANOT = (PT_LOOS + 0x13) # type: ignore
PT_IA_64_HP_STACK = (PT_LOOS + 0x14) # type: ignore
PF_IA_64_NORECOV = 0x80000000 # type: ignore
SHT_IA_64_EXT = (SHT_LOPROC + 0) # type: ignore
SHT_IA_64_UNWIND = (SHT_LOPROC + 1) # type: ignore
SHF_IA_64_SHORT = 0x10000000 # type: ignore
SHF_IA_64_NORECOV = 0x20000000 # type: ignore
DT_IA_64_PLT_RESERVE = (DT_LOPROC + 0) # type: ignore
DT_IA_64_NUM = 1 # type: ignore
R_IA64_NONE = 0x00 # type: ignore
R_IA64_IMM14 = 0x21 # type: ignore
R_IA64_IMM22 = 0x22 # type: ignore
R_IA64_IMM64 = 0x23 # type: ignore
R_IA64_DIR32MSB = 0x24 # type: ignore
R_IA64_DIR32LSB = 0x25 # type: ignore
R_IA64_DIR64MSB = 0x26 # type: ignore
R_IA64_DIR64LSB = 0x27 # type: ignore
R_IA64_GPREL22 = 0x2a # type: ignore
R_IA64_GPREL64I = 0x2b # type: ignore
R_IA64_GPREL32MSB = 0x2c # type: ignore
R_IA64_GPREL32LSB = 0x2d # type: ignore
R_IA64_GPREL64MSB = 0x2e # type: ignore
R_IA64_GPREL64LSB = 0x2f # type: ignore
R_IA64_LTOFF22 = 0x32 # type: ignore
R_IA64_LTOFF64I = 0x33 # type: ignore
R_IA64_PLTOFF22 = 0x3a # type: ignore
R_IA64_PLTOFF64I = 0x3b # type: ignore
R_IA64_PLTOFF64MSB = 0x3e # type: ignore
R_IA64_PLTOFF64LSB = 0x3f # type: ignore
R_IA64_FPTR64I = 0x43 # type: ignore
R_IA64_FPTR32MSB = 0x44 # type: ignore
R_IA64_FPTR32LSB = 0x45 # type: ignore
R_IA64_FPTR64MSB = 0x46 # type: ignore
R_IA64_FPTR64LSB = 0x47 # type: ignore
R_IA64_PCREL60B = 0x48 # type: ignore
R_IA64_PCREL21B = 0x49 # type: ignore
R_IA64_PCREL21M = 0x4a # type: ignore
R_IA64_PCREL21F = 0x4b # type: ignore
R_IA64_PCREL32MSB = 0x4c # type: ignore
R_IA64_PCREL32LSB = 0x4d # type: ignore
R_IA64_PCREL64MSB = 0x4e # type: ignore
R_IA64_PCREL64LSB = 0x4f # type: ignore
R_IA64_LTOFF_FPTR22 = 0x52 # type: ignore
R_IA64_LTOFF_FPTR64I = 0x53 # type: ignore
R_IA64_LTOFF_FPTR32MSB = 0x54 # type: ignore
R_IA64_LTOFF_FPTR32LSB = 0x55 # type: ignore
R_IA64_LTOFF_FPTR64MSB = 0x56 # type: ignore
R_IA64_LTOFF_FPTR64LSB = 0x57 # type: ignore
R_IA64_SEGREL32MSB = 0x5c # type: ignore
R_IA64_SEGREL32LSB = 0x5d # type: ignore
R_IA64_SEGREL64MSB = 0x5e # type: ignore
R_IA64_SEGREL64LSB = 0x5f # type: ignore
R_IA64_SECREL32MSB = 0x64 # type: ignore
R_IA64_SECREL32LSB = 0x65 # type: ignore
R_IA64_SECREL64MSB = 0x66 # type: ignore
R_IA64_SECREL64LSB = 0x67 # type: ignore
R_IA64_REL32MSB = 0x6c # type: ignore
R_IA64_REL32LSB = 0x6d # type: ignore
R_IA64_REL64MSB = 0x6e # type: ignore
R_IA64_REL64LSB = 0x6f # type: ignore
R_IA64_LTV32MSB = 0x74 # type: ignore
R_IA64_LTV32LSB = 0x75 # type: ignore
R_IA64_LTV64MSB = 0x76 # type: ignore
R_IA64_LTV64LSB = 0x77 # type: ignore
R_IA64_PCREL21BI = 0x79 # type: ignore
R_IA64_PCREL22 = 0x7a # type: ignore
R_IA64_PCREL64I = 0x7b # type: ignore
R_IA64_IPLTMSB = 0x80 # type: ignore
R_IA64_IPLTLSB = 0x81 # type: ignore
R_IA64_COPY = 0x84 # type: ignore
R_IA64_SUB = 0x85 # type: ignore
R_IA64_LTOFF22X = 0x86 # type: ignore
R_IA64_LDXMOV = 0x87 # type: ignore
R_IA64_TPREL14 = 0x91 # type: ignore
R_IA64_TPREL22 = 0x92 # type: ignore
R_IA64_TPREL64I = 0x93 # type: ignore
R_IA64_TPREL64MSB = 0x96 # type: ignore
R_IA64_TPREL64LSB = 0x97 # type: ignore
R_IA64_LTOFF_TPREL22 = 0x9a # type: ignore
R_IA64_DTPMOD64MSB = 0xa6 # type: ignore
R_IA64_DTPMOD64LSB = 0xa7 # type: ignore
R_IA64_LTOFF_DTPMOD22 = 0xaa # type: ignore
R_IA64_DTPREL14 = 0xb1 # type: ignore
R_IA64_DTPREL22 = 0xb2 # type: ignore
R_IA64_DTPREL64I = 0xb3 # type: ignore
R_IA64_DTPREL32MSB = 0xb4 # type: ignore
R_IA64_DTPREL32LSB = 0xb5 # type: ignore
R_IA64_DTPREL64MSB = 0xb6 # type: ignore
R_IA64_DTPREL64LSB = 0xb7 # type: ignore
R_IA64_LTOFF_DTPREL22 = 0xba # type: ignore
EF_SH_MACH_MASK = 0x1f # type: ignore
EF_SH_UNKNOWN = 0x0 # type: ignore
EF_SH1 = 0x1 # type: ignore
EF_SH2 = 0x2 # type: ignore
EF_SH3 = 0x3 # type: ignore
EF_SH_DSP = 0x4 # type: ignore
EF_SH3_DSP = 0x5 # type: ignore
EF_SH4AL_DSP = 0x6 # type: ignore
EF_SH3E = 0x8 # type: ignore
EF_SH4 = 0x9 # type: ignore
EF_SH2E = 0xb # type: ignore
EF_SH4A = 0xc # type: ignore
EF_SH2A = 0xd # type: ignore
EF_SH4_NOFPU = 0x10 # type: ignore
EF_SH4A_NOFPU = 0x11 # type: ignore
EF_SH4_NOMMU_NOFPU = 0x12 # type: ignore
EF_SH2A_NOFPU = 0x13 # type: ignore
EF_SH3_NOMMU = 0x14 # type: ignore
EF_SH2A_SH4_NOFPU = 0x15 # type: ignore
EF_SH2A_SH3_NOFPU = 0x16 # type: ignore
EF_SH2A_SH4 = 0x17 # type: ignore
EF_SH2A_SH3E = 0x18 # type: ignore
R_SH_NONE = 0 # type: ignore
R_SH_DIR32 = 1 # type: ignore
R_SH_REL32 = 2 # type: ignore
R_SH_DIR8WPN = 3 # type: ignore
R_SH_IND12W = 4 # type: ignore
R_SH_DIR8WPL = 5 # type: ignore
R_SH_DIR8WPZ = 6 # type: ignore
R_SH_DIR8BP = 7 # type: ignore
R_SH_DIR8W = 8 # type: ignore
R_SH_DIR8L = 9 # type: ignore
R_SH_SWITCH16 = 25 # type: ignore
R_SH_SWITCH32 = 26 # type: ignore
R_SH_USES = 27 # type: ignore
R_SH_COUNT = 28 # type: ignore
R_SH_ALIGN = 29 # type: ignore
R_SH_CODE = 30 # type: ignore
R_SH_DATA = 31 # type: ignore
R_SH_LABEL = 32 # type: ignore
R_SH_SWITCH8 = 33 # type: ignore
R_SH_GNU_VTINHERIT = 34 # type: ignore
R_SH_GNU_VTENTRY = 35 # type: ignore
R_SH_TLS_GD_32 = 144 # type: ignore
R_SH_TLS_LD_32 = 145 # type: ignore
R_SH_TLS_LDO_32 = 146 # type: ignore
R_SH_TLS_IE_32 = 147 # type: ignore
R_SH_TLS_LE_32 = 148 # type: ignore
R_SH_TLS_DTPMOD32 = 149 # type: ignore
R_SH_TLS_DTPOFF32 = 150 # type: ignore
R_SH_TLS_TPOFF32 = 151 # type: ignore
R_SH_GOT32 = 160 # type: ignore
R_SH_PLT32 = 161 # type: ignore
R_SH_COPY = 162 # type: ignore
R_SH_GLOB_DAT = 163 # type: ignore
R_SH_JMP_SLOT = 164 # type: ignore
R_SH_RELATIVE = 165 # type: ignore
R_SH_GOTOFF = 166 # type: ignore
R_SH_GOTPC = 167 # type: ignore
R_SH_NUM = 256 # type: ignore
EF_S390_HIGH_GPRS = 0x00000001 # type: ignore
R_390_NONE = 0 # type: ignore
R_390_8 = 1 # type: ignore
R_390_12 = 2 # type: ignore
R_390_16 = 3 # type: ignore
R_390_32 = 4 # type: ignore
R_390_PC32 = 5 # type: ignore
R_390_GOT12 = 6 # type: ignore
R_390_GOT32 = 7 # type: ignore
R_390_PLT32 = 8 # type: ignore
R_390_COPY = 9 # type: ignore
R_390_GLOB_DAT = 10 # type: ignore
R_390_JMP_SLOT = 11 # type: ignore
R_390_RELATIVE = 12 # type: ignore
R_390_GOTOFF32 = 13 # type: ignore
R_390_GOTPC = 14 # type: ignore
R_390_GOT16 = 15 # type: ignore
R_390_PC16 = 16 # type: ignore
R_390_PC16DBL = 17 # type: ignore
R_390_PLT16DBL = 18 # type: ignore
R_390_PC32DBL = 19 # type: ignore
R_390_PLT32DBL = 20 # type: ignore
R_390_GOTPCDBL = 21 # type: ignore
R_390_64 = 22 # type: ignore
R_390_PC64 = 23 # type: ignore
R_390_GOT64 = 24 # type: ignore
R_390_PLT64 = 25 # type: ignore
R_390_GOTENT = 26 # type: ignore
R_390_GOTOFF16 = 27 # type: ignore
R_390_GOTOFF64 = 28 # type: ignore
R_390_GOTPLT12 = 29 # type: ignore
R_390_GOTPLT16 = 30 # type: ignore
R_390_GOTPLT32 = 31 # type: ignore
R_390_GOTPLT64 = 32 # type: ignore
R_390_GOTPLTENT = 33 # type: ignore
R_390_PLTOFF16 = 34 # type: ignore
R_390_PLTOFF32 = 35 # type: ignore
R_390_PLTOFF64 = 36 # type: ignore
R_390_TLS_LOAD = 37 # type: ignore
R_390_TLS_GDCALL = 38 # type: ignore
R_390_TLS_LDCALL = 39 # type: ignore
R_390_TLS_GD32 = 40 # type: ignore
R_390_TLS_GD64 = 41 # type: ignore
R_390_TLS_GOTIE12 = 42 # type: ignore
R_390_TLS_GOTIE32 = 43 # type: ignore
R_390_TLS_GOTIE64 = 44 # type: ignore
R_390_TLS_LDM32 = 45 # type: ignore
R_390_TLS_LDM64 = 46 # type: ignore
R_390_TLS_IE32 = 47 # type: ignore
R_390_TLS_IE64 = 48 # type: ignore
R_390_TLS_IEENT = 49 # type: ignore
R_390_TLS_LE32 = 50 # type: ignore
R_390_TLS_LE64 = 51 # type: ignore
R_390_TLS_LDO32 = 52 # type: ignore
R_390_TLS_LDO64 = 53 # type: ignore
R_390_TLS_DTPMOD = 54 # type: ignore
R_390_TLS_DTPOFF = 55 # type: ignore
R_390_TLS_TPOFF = 56 # type: ignore
R_390_20 = 57 # type: ignore
R_390_GOT20 = 58 # type: ignore
R_390_GOTPLT20 = 59 # type: ignore
R_390_TLS_GOTIE20 = 60 # type: ignore
R_390_IRELATIVE = 61 # type: ignore
R_390_NUM = 62 # type: ignore
R_CRIS_NONE = 0 # type: ignore
R_CRIS_8 = 1 # type: ignore
R_CRIS_16 = 2 # type: ignore
R_CRIS_32 = 3 # type: ignore
R_CRIS_8_PCREL = 4 # type: ignore
R_CRIS_16_PCREL = 5 # type: ignore
R_CRIS_32_PCREL = 6 # type: ignore
R_CRIS_GNU_VTINHERIT = 7 # type: ignore
R_CRIS_GNU_VTENTRY = 8 # type: ignore
R_CRIS_COPY = 9 # type: ignore
R_CRIS_GLOB_DAT = 10 # type: ignore
R_CRIS_JUMP_SLOT = 11 # type: ignore
R_CRIS_RELATIVE = 12 # type: ignore
R_CRIS_16_GOT = 13 # type: ignore
R_CRIS_32_GOT = 14 # type: ignore
R_CRIS_16_GOTPLT = 15 # type: ignore
R_CRIS_32_GOTPLT = 16 # type: ignore
R_CRIS_32_GOTREL = 17 # type: ignore
R_CRIS_32_PLT_GOTREL = 18 # type: ignore
R_CRIS_32_PLT_PCREL = 19 # type: ignore
R_CRIS_NUM = 20 # type: ignore
R_X86_64_NONE = 0 # type: ignore
R_X86_64_64 = 1 # type: ignore
R_X86_64_PC32 = 2 # type: ignore
R_X86_64_GOT32 = 3 # type: ignore
R_X86_64_PLT32 = 4 # type: ignore
R_X86_64_COPY = 5 # type: ignore
R_X86_64_GLOB_DAT = 6 # type: ignore
R_X86_64_JUMP_SLOT = 7 # type: ignore
R_X86_64_RELATIVE = 8 # type: ignore
R_X86_64_GOTPCREL = 9 # type: ignore
R_X86_64_32 = 10 # type: ignore
R_X86_64_32S = 11 # type: ignore
R_X86_64_16 = 12 # type: ignore
R_X86_64_PC16 = 13 # type: ignore
R_X86_64_8 = 14 # type: ignore
R_X86_64_PC8 = 15 # type: ignore
R_X86_64_DTPMOD64 = 16 # type: ignore
R_X86_64_DTPOFF64 = 17 # type: ignore
R_X86_64_TPOFF64 = 18 # type: ignore
R_X86_64_TLSGD = 19 # type: ignore
R_X86_64_TLSLD = 20 # type: ignore
R_X86_64_DTPOFF32 = 21 # type: ignore
R_X86_64_GOTTPOFF = 22 # type: ignore
R_X86_64_TPOFF32 = 23 # type: ignore
R_X86_64_PC64 = 24 # type: ignore
R_X86_64_GOTOFF64 = 25 # type: ignore
R_X86_64_GOTPC32 = 26 # type: ignore
R_X86_64_GOT64 = 27 # type: ignore
R_X86_64_GOTPCREL64 = 28 # type: ignore
R_X86_64_GOTPC64 = 29 # type: ignore
R_X86_64_GOTPLT64 = 30 # type: ignore
R_X86_64_PLTOFF64 = 31 # type: ignore
R_X86_64_SIZE32 = 32 # type: ignore
R_X86_64_SIZE64 = 33 # type: ignore
R_X86_64_GOTPC32_TLSDESC = 34 # type: ignore
R_X86_64_TLSDESC_CALL = 35 # type: ignore
R_X86_64_TLSDESC = 36 # type: ignore
R_X86_64_IRELATIVE = 37 # type: ignore
R_X86_64_RELATIVE64 = 38 # type: ignore
R_X86_64_GOTPCRELX = 41 # type: ignore
R_X86_64_REX_GOTPCRELX = 42 # type: ignore
R_X86_64_NUM = 43 # type: ignore
SHT_X86_64_UNWIND = 0x70000001 # type: ignore
DT_X86_64_PLT = (DT_LOPROC + 0) # type: ignore
DT_X86_64_PLTSZ = (DT_LOPROC + 1) # type: ignore
DT_X86_64_PLTENT = (DT_LOPROC + 3) # type: ignore
DT_X86_64_NUM = 4 # type: ignore
R_MN10300_NONE = 0 # type: ignore
R_MN10300_32 = 1 # type: ignore
R_MN10300_16 = 2 # type: ignore
R_MN10300_8 = 3 # type: ignore
R_MN10300_PCREL32 = 4 # type: ignore
R_MN10300_PCREL16 = 5 # type: ignore
R_MN10300_PCREL8 = 6 # type: ignore
R_MN10300_GNU_VTINHERIT = 7 # type: ignore
R_MN10300_GNU_VTENTRY = 8 # type: ignore
R_MN10300_24 = 9 # type: ignore
R_MN10300_GOTPC32 = 10 # type: ignore
R_MN10300_GOTPC16 = 11 # type: ignore
R_MN10300_GOTOFF32 = 12 # type: ignore
R_MN10300_GOTOFF24 = 13 # type: ignore
R_MN10300_GOTOFF16 = 14 # type: ignore
R_MN10300_PLT32 = 15 # type: ignore
R_MN10300_PLT16 = 16 # type: ignore
R_MN10300_GOT32 = 17 # type: ignore
R_MN10300_GOT24 = 18 # type: ignore
R_MN10300_GOT16 = 19 # type: ignore
R_MN10300_COPY = 20 # type: ignore
R_MN10300_GLOB_DAT = 21 # type: ignore
R_MN10300_JMP_SLOT = 22 # type: ignore
R_MN10300_RELATIVE = 23 # type: ignore
R_MN10300_TLS_GD = 24 # type: ignore
R_MN10300_TLS_LD = 25 # type: ignore
R_MN10300_TLS_LDO = 26 # type: ignore
R_MN10300_TLS_GOTIE = 27 # type: ignore
R_MN10300_TLS_IE = 28 # type: ignore
R_MN10300_TLS_LE = 29 # type: ignore
R_MN10300_TLS_DTPMOD = 30 # type: ignore
R_MN10300_TLS_DTPOFF = 31 # type: ignore
R_MN10300_TLS_TPOFF = 32 # type: ignore
R_MN10300_SYM_DIFF = 33 # type: ignore
R_MN10300_ALIGN = 34 # type: ignore
R_MN10300_NUM = 35 # type: ignore
R_M32R_NONE = 0 # type: ignore
R_M32R_16 = 1 # type: ignore
R_M32R_32 = 2 # type: ignore
R_M32R_24 = 3 # type: ignore
R_M32R_10_PCREL = 4 # type: ignore
R_M32R_18_PCREL = 5 # type: ignore
R_M32R_26_PCREL = 6 # type: ignore
R_M32R_HI16_ULO = 7 # type: ignore
R_M32R_HI16_SLO = 8 # type: ignore
R_M32R_LO16 = 9 # type: ignore
R_M32R_SDA16 = 10 # type: ignore
R_M32R_GNU_VTINHERIT = 11 # type: ignore
R_M32R_GNU_VTENTRY = 12 # type: ignore
R_M32R_16_RELA = 33 # type: ignore
R_M32R_32_RELA = 34 # type: ignore
R_M32R_24_RELA = 35 # type: ignore
R_M32R_10_PCREL_RELA = 36 # type: ignore
R_M32R_18_PCREL_RELA = 37 # type: ignore
R_M32R_26_PCREL_RELA = 38 # type: ignore
R_M32R_HI16_ULO_RELA = 39 # type: ignore
R_M32R_HI16_SLO_RELA = 40 # type: ignore
R_M32R_LO16_RELA = 41 # type: ignore
R_M32R_SDA16_RELA = 42 # type: ignore
R_M32R_RELA_GNU_VTINHERIT = 43 # type: ignore
R_M32R_RELA_GNU_VTENTRY = 44 # type: ignore
R_M32R_REL32 = 45 # type: ignore
R_M32R_GOT24 = 48 # type: ignore
R_M32R_26_PLTREL = 49 # type: ignore
R_M32R_COPY = 50 # type: ignore
R_M32R_GLOB_DAT = 51 # type: ignore
R_M32R_JMP_SLOT = 52 # type: ignore
R_M32R_RELATIVE = 53 # type: ignore
R_M32R_GOTOFF = 54 # type: ignore
R_M32R_GOTPC24 = 55 # type: ignore
R_M32R_GOT16_HI_ULO = 56 # type: ignore
R_M32R_GOT16_HI_SLO = 57 # type: ignore
R_M32R_GOT16_LO = 58 # type: ignore
R_M32R_GOTPC_HI_ULO = 59 # type: ignore
R_M32R_GOTPC_HI_SLO = 60 # type: ignore
R_M32R_GOTPC_LO = 61 # type: ignore
R_M32R_GOTOFF_HI_ULO = 62 # type: ignore
R_M32R_GOTOFF_HI_SLO = 63 # type: ignore
R_M32R_GOTOFF_LO = 64 # type: ignore
R_M32R_NUM = 256 # type: ignore
R_MICROBLAZE_NONE = 0 # type: ignore
R_MICROBLAZE_32 = 1 # type: ignore
R_MICROBLAZE_32_PCREL = 2 # type: ignore
R_MICROBLAZE_64_PCREL = 3 # type: ignore
R_MICROBLAZE_32_PCREL_LO = 4 # type: ignore
R_MICROBLAZE_64 = 5 # type: ignore
R_MICROBLAZE_32_LO = 6 # type: ignore
R_MICROBLAZE_SRO32 = 7 # type: ignore
R_MICROBLAZE_SRW32 = 8 # type: ignore
R_MICROBLAZE_64_NONE = 9 # type: ignore
R_MICROBLAZE_32_SYM_OP_SYM = 10 # type: ignore
R_MICROBLAZE_GNU_VTINHERIT = 11 # type: ignore
R_MICROBLAZE_GNU_VTENTRY = 12 # type: ignore
R_MICROBLAZE_GOTPC_64 = 13 # type: ignore
R_MICROBLAZE_GOT_64 = 14 # type: ignore
R_MICROBLAZE_PLT_64 = 15 # type: ignore
R_MICROBLAZE_REL = 16 # type: ignore
R_MICROBLAZE_JUMP_SLOT = 17 # type: ignore
R_MICROBLAZE_GLOB_DAT = 18 # type: ignore
R_MICROBLAZE_GOTOFF_64 = 19 # type: ignore
R_MICROBLAZE_GOTOFF_32 = 20 # type: ignore
R_MICROBLAZE_COPY = 21 # type: ignore
R_MICROBLAZE_TLS = 22 # type: ignore
R_MICROBLAZE_TLSGD = 23 # type: ignore
R_MICROBLAZE_TLSLD = 24 # type: ignore
R_MICROBLAZE_TLSDTPMOD32 = 25 # type: ignore
R_MICROBLAZE_TLSDTPREL32 = 26 # type: ignore
R_MICROBLAZE_TLSDTPREL64 = 27 # type: ignore
R_MICROBLAZE_TLSGOTTPREL32 = 28 # type: ignore
R_MICROBLAZE_TLSTPREL32 = 29 # type: ignore
DT_NIOS2_GP = 0x70000002 # type: ignore
R_NIOS2_NONE = 0 # type: ignore
R_NIOS2_S16 = 1 # type: ignore
R_NIOS2_U16 = 2 # type: ignore
R_NIOS2_PCREL16 = 3 # type: ignore
R_NIOS2_CALL26 = 4 # type: ignore
R_NIOS2_IMM5 = 5 # type: ignore
R_NIOS2_CACHE_OPX = 6 # type: ignore
R_NIOS2_IMM6 = 7 # type: ignore
R_NIOS2_IMM8 = 8 # type: ignore
R_NIOS2_HI16 = 9 # type: ignore
R_NIOS2_LO16 = 10 # type: ignore
R_NIOS2_HIADJ16 = 11 # type: ignore
R_NIOS2_BFD_RELOC_32 = 12 # type: ignore
R_NIOS2_BFD_RELOC_16 = 13 # type: ignore
R_NIOS2_BFD_RELOC_8 = 14 # type: ignore
R_NIOS2_GPREL = 15 # type: ignore
R_NIOS2_GNU_VTINHERIT = 16 # type: ignore
R_NIOS2_GNU_VTENTRY = 17 # type: ignore
R_NIOS2_UJMP = 18 # type: ignore
R_NIOS2_CJMP = 19 # type: ignore
R_NIOS2_CALLR = 20 # type: ignore
R_NIOS2_ALIGN = 21 # type: ignore
R_NIOS2_GOT16 = 22 # type: ignore
R_NIOS2_CALL16 = 23 # type: ignore
R_NIOS2_GOTOFF_LO = 24 # type: ignore
R_NIOS2_GOTOFF_HA = 25 # type: ignore
R_NIOS2_PCREL_LO = 26 # type: ignore
R_NIOS2_PCREL_HA = 27 # type: ignore
R_NIOS2_TLS_GD16 = 28 # type: ignore
R_NIOS2_TLS_LDM16 = 29 # type: ignore
R_NIOS2_TLS_LDO16 = 30 # type: ignore
R_NIOS2_TLS_IE16 = 31 # type: ignore
R_NIOS2_TLS_LE16 = 32 # type: ignore
R_NIOS2_TLS_DTPMOD = 33 # type: ignore
R_NIOS2_TLS_DTPREL = 34 # type: ignore
R_NIOS2_TLS_TPREL = 35 # type: ignore
R_NIOS2_COPY = 36 # type: ignore
R_NIOS2_GLOB_DAT = 37 # type: ignore
R_NIOS2_JUMP_SLOT = 38 # type: ignore
R_NIOS2_RELATIVE = 39 # type: ignore
R_NIOS2_GOTOFF = 40 # type: ignore
R_NIOS2_CALL26_NOAT = 41 # type: ignore
R_NIOS2_GOT_LO = 42 # type: ignore
R_NIOS2_GOT_HA = 43 # type: ignore
R_NIOS2_CALL_LO = 44 # type: ignore
R_NIOS2_CALL_HA = 45 # type: ignore
R_TILEPRO_NONE = 0 # type: ignore
R_TILEPRO_32 = 1 # type: ignore
R_TILEPRO_16 = 2 # type: ignore
R_TILEPRO_8 = 3 # type: ignore
R_TILEPRO_32_PCREL = 4 # type: ignore
R_TILEPRO_16_PCREL = 5 # type: ignore
R_TILEPRO_8_PCREL = 6 # type: ignore
R_TILEPRO_LO16 = 7 # type: ignore
R_TILEPRO_HI16 = 8 # type: ignore
R_TILEPRO_HA16 = 9 # type: ignore
R_TILEPRO_COPY = 10 # type: ignore
R_TILEPRO_GLOB_DAT = 11 # type: ignore
R_TILEPRO_JMP_SLOT = 12 # type: ignore
R_TILEPRO_RELATIVE = 13 # type: ignore
R_TILEPRO_BROFF_X1 = 14 # type: ignore
R_TILEPRO_JOFFLONG_X1 = 15 # type: ignore
R_TILEPRO_JOFFLONG_X1_PLT = 16 # type: ignore
R_TILEPRO_IMM8_X0 = 17 # type: ignore
R_TILEPRO_IMM8_Y0 = 18 # type: ignore
R_TILEPRO_IMM8_X1 = 19 # type: ignore
R_TILEPRO_IMM8_Y1 = 20 # type: ignore
R_TILEPRO_MT_IMM15_X1 = 21 # type: ignore
R_TILEPRO_MF_IMM15_X1 = 22 # type: ignore
R_TILEPRO_IMM16_X0 = 23 # type: ignore
R_TILEPRO_IMM16_X1 = 24 # type: ignore
R_TILEPRO_IMM16_X0_LO = 25 # type: ignore
R_TILEPRO_IMM16_X1_LO = 26 # type: ignore
R_TILEPRO_IMM16_X0_HI = 27 # type: ignore
R_TILEPRO_IMM16_X1_HI = 28 # type: ignore
R_TILEPRO_IMM16_X0_HA = 29 # type: ignore
R_TILEPRO_IMM16_X1_HA = 30 # type: ignore
R_TILEPRO_IMM16_X0_PCREL = 31 # type: ignore
R_TILEPRO_IMM16_X1_PCREL = 32 # type: ignore
R_TILEPRO_IMM16_X0_LO_PCREL = 33 # type: ignore
R_TILEPRO_IMM16_X1_LO_PCREL = 34 # type: ignore
R_TILEPRO_IMM16_X0_HI_PCREL = 35 # type: ignore
R_TILEPRO_IMM16_X1_HI_PCREL = 36 # type: ignore
R_TILEPRO_IMM16_X0_HA_PCREL = 37 # type: ignore
R_TILEPRO_IMM16_X1_HA_PCREL = 38 # type: ignore
R_TILEPRO_IMM16_X0_GOT = 39 # type: ignore
R_TILEPRO_IMM16_X1_GOT = 40 # type: ignore
R_TILEPRO_IMM16_X0_GOT_LO = 41 # type: ignore
R_TILEPRO_IMM16_X1_GOT_LO = 42 # type: ignore
R_TILEPRO_IMM16_X0_GOT_HI = 43 # type: ignore
R_TILEPRO_IMM16_X1_GOT_HI = 44 # type: ignore
R_TILEPRO_IMM16_X0_GOT_HA = 45 # type: ignore
R_TILEPRO_IMM16_X1_GOT_HA = 46 # type: ignore
R_TILEPRO_MMSTART_X0 = 47 # type: ignore
R_TILEPRO_MMEND_X0 = 48 # type: ignore
R_TILEPRO_MMSTART_X1 = 49 # type: ignore
R_TILEPRO_MMEND_X1 = 50 # type: ignore
R_TILEPRO_SHAMT_X0 = 51 # type: ignore
R_TILEPRO_SHAMT_X1 = 52 # type: ignore
R_TILEPRO_SHAMT_Y0 = 53 # type: ignore
R_TILEPRO_SHAMT_Y1 = 54 # type: ignore
R_TILEPRO_DEST_IMM8_X1 = 55 # type: ignore
R_TILEPRO_TLS_GD_CALL = 60 # type: ignore
R_TILEPRO_IMM8_X0_TLS_GD_ADD = 61 # type: ignore
R_TILEPRO_IMM8_X1_TLS_GD_ADD = 62 # type: ignore
R_TILEPRO_IMM8_Y0_TLS_GD_ADD = 63 # type: ignore
R_TILEPRO_IMM8_Y1_TLS_GD_ADD = 64 # type: ignore
R_TILEPRO_TLS_IE_LOAD = 65 # type: ignore
R_TILEPRO_IMM16_X0_TLS_GD = 66 # type: ignore
R_TILEPRO_IMM16_X1_TLS_GD = 67 # type: ignore
R_TILEPRO_IMM16_X0_TLS_GD_LO = 68 # type: ignore
R_TILEPRO_IMM16_X1_TLS_GD_LO = 69 # type: ignore
R_TILEPRO_IMM16_X0_TLS_GD_HI = 70 # type: ignore
R_TILEPRO_IMM16_X1_TLS_GD_HI = 71 # type: ignore
R_TILEPRO_IMM16_X0_TLS_GD_HA = 72 # type: ignore
R_TILEPRO_IMM16_X1_TLS_GD_HA = 73 # type: ignore
R_TILEPRO_IMM16_X0_TLS_IE = 74 # type: ignore
R_TILEPRO_IMM16_X1_TLS_IE = 75 # type: ignore
R_TILEPRO_IMM16_X0_TLS_IE_LO = 76 # type: ignore
R_TILEPRO_IMM16_X1_TLS_IE_LO = 77 # type: ignore
R_TILEPRO_IMM16_X0_TLS_IE_HI = 78 # type: ignore
R_TILEPRO_IMM16_X1_TLS_IE_HI = 79 # type: ignore
R_TILEPRO_IMM16_X0_TLS_IE_HA = 80 # type: ignore
R_TILEPRO_IMM16_X1_TLS_IE_HA = 81 # type: ignore
R_TILEPRO_TLS_DTPMOD32 = 82 # type: ignore
R_TILEPRO_TLS_DTPOFF32 = 83 # type: ignore
R_TILEPRO_TLS_TPOFF32 = 84 # type: ignore
R_TILEPRO_IMM16_X0_TLS_LE = 85 # type: ignore
R_TILEPRO_IMM16_X1_TLS_LE = 86 # type: ignore
R_TILEPRO_IMM16_X0_TLS_LE_LO = 87 # type: ignore
R_TILEPRO_IMM16_X1_TLS_LE_LO = 88 # type: ignore
R_TILEPRO_IMM16_X0_TLS_LE_HI = 89 # type: ignore
R_TILEPRO_IMM16_X1_TLS_LE_HI = 90 # type: ignore
R_TILEPRO_IMM16_X0_TLS_LE_HA = 91 # type: ignore
R_TILEPRO_IMM16_X1_TLS_LE_HA = 92 # type: ignore
R_TILEPRO_GNU_VTINHERIT = 128 # type: ignore
R_TILEPRO_GNU_VTENTRY = 129 # type: ignore
R_TILEPRO_NUM = 130 # type: ignore
R_TILEGX_NONE = 0 # type: ignore
R_TILEGX_64 = 1 # type: ignore
R_TILEGX_32 = 2 # type: ignore
R_TILEGX_16 = 3 # type: ignore
R_TILEGX_8 = 4 # type: ignore
R_TILEGX_64_PCREL = 5 # type: ignore
R_TILEGX_32_PCREL = 6 # type: ignore
R_TILEGX_16_PCREL = 7 # type: ignore
R_TILEGX_8_PCREL = 8 # type: ignore
R_TILEGX_HW0 = 9 # type: ignore
R_TILEGX_HW1 = 10 # type: ignore
R_TILEGX_HW2 = 11 # type: ignore
R_TILEGX_HW3 = 12 # type: ignore
R_TILEGX_HW0_LAST = 13 # type: ignore
R_TILEGX_HW1_LAST = 14 # type: ignore
R_TILEGX_HW2_LAST = 15 # type: ignore
R_TILEGX_COPY = 16 # type: ignore
R_TILEGX_GLOB_DAT = 17 # type: ignore
R_TILEGX_JMP_SLOT = 18 # type: ignore
R_TILEGX_RELATIVE = 19 # type: ignore
R_TILEGX_BROFF_X1 = 20 # type: ignore
R_TILEGX_JUMPOFF_X1 = 21 # type: ignore
R_TILEGX_JUMPOFF_X1_PLT = 22 # type: ignore
R_TILEGX_IMM8_X0 = 23 # type: ignore
R_TILEGX_IMM8_Y0 = 24 # type: ignore
R_TILEGX_IMM8_X1 = 25 # type: ignore
R_TILEGX_IMM8_Y1 = 26 # type: ignore
R_TILEGX_DEST_IMM8_X1 = 27 # type: ignore
R_TILEGX_MT_IMM14_X1 = 28 # type: ignore
R_TILEGX_MF_IMM14_X1 = 29 # type: ignore
R_TILEGX_MMSTART_X0 = 30 # type: ignore
R_TILEGX_MMEND_X0 = 31 # type: ignore
R_TILEGX_SHAMT_X0 = 32 # type: ignore
R_TILEGX_SHAMT_X1 = 33 # type: ignore
R_TILEGX_SHAMT_Y0 = 34 # type: ignore
R_TILEGX_SHAMT_Y1 = 35 # type: ignore
R_TILEGX_IMM16_X0_HW0 = 36 # type: ignore
R_TILEGX_IMM16_X1_HW0 = 37 # type: ignore
R_TILEGX_IMM16_X0_HW1 = 38 # type: ignore
R_TILEGX_IMM16_X1_HW1 = 39 # type: ignore
R_TILEGX_IMM16_X0_HW2 = 40 # type: ignore
R_TILEGX_IMM16_X1_HW2 = 41 # type: ignore
R_TILEGX_IMM16_X0_HW3 = 42 # type: ignore
R_TILEGX_IMM16_X1_HW3 = 43 # type: ignore
R_TILEGX_IMM16_X0_HW0_LAST = 44 # type: ignore
R_TILEGX_IMM16_X1_HW0_LAST = 45 # type: ignore
R_TILEGX_IMM16_X0_HW1_LAST = 46 # type: ignore
R_TILEGX_IMM16_X1_HW1_LAST = 47 # type: ignore
R_TILEGX_IMM16_X0_HW2_LAST = 48 # type: ignore
R_TILEGX_IMM16_X1_HW2_LAST = 49 # type: ignore
R_TILEGX_IMM16_X0_HW0_PCREL = 50 # type: ignore
R_TILEGX_IMM16_X1_HW0_PCREL = 51 # type: ignore
R_TILEGX_IMM16_X0_HW1_PCREL = 52 # type: ignore
R_TILEGX_IMM16_X1_HW1_PCREL = 53 # type: ignore
R_TILEGX_IMM16_X0_HW2_PCREL = 54 # type: ignore
R_TILEGX_IMM16_X1_HW2_PCREL = 55 # type: ignore
R_TILEGX_IMM16_X0_HW3_PCREL = 56 # type: ignore
R_TILEGX_IMM16_X1_HW3_PCREL = 57 # type: ignore
R_TILEGX_IMM16_X0_HW0_LAST_PCREL = 58 # type: ignore
R_TILEGX_IMM16_X1_HW0_LAST_PCREL = 59 # type: ignore
R_TILEGX_IMM16_X0_HW1_LAST_PCREL = 60 # type: ignore
R_TILEGX_IMM16_X1_HW1_LAST_PCREL = 61 # type: ignore
R_TILEGX_IMM16_X0_HW2_LAST_PCREL = 62 # type: ignore
R_TILEGX_IMM16_X1_HW2_LAST_PCREL = 63 # type: ignore
R_TILEGX_IMM16_X0_HW0_GOT = 64 # type: ignore
R_TILEGX_IMM16_X1_HW0_GOT = 65 # type: ignore
R_TILEGX_IMM16_X0_HW0_PLT_PCREL = 66 # type: ignore
R_TILEGX_IMM16_X1_HW0_PLT_PCREL = 67 # type: ignore
R_TILEGX_IMM16_X0_HW1_PLT_PCREL = 68 # type: ignore
R_TILEGX_IMM16_X1_HW1_PLT_PCREL = 69 # type: ignore
R_TILEGX_IMM16_X0_HW2_PLT_PCREL = 70 # type: ignore
R_TILEGX_IMM16_X1_HW2_PLT_PCREL = 71 # type: ignore
R_TILEGX_IMM16_X0_HW0_LAST_GOT = 72 # type: ignore
R_TILEGX_IMM16_X1_HW0_LAST_GOT = 73 # type: ignore
R_TILEGX_IMM16_X0_HW1_LAST_GOT = 74 # type: ignore
R_TILEGX_IMM16_X1_HW1_LAST_GOT = 75 # type: ignore
R_TILEGX_IMM16_X0_HW3_PLT_PCREL = 76 # type: ignore
R_TILEGX_IMM16_X1_HW3_PLT_PCREL = 77 # type: ignore
R_TILEGX_IMM16_X0_HW0_TLS_GD = 78 # type: ignore
R_TILEGX_IMM16_X1_HW0_TLS_GD = 79 # type: ignore
R_TILEGX_IMM16_X0_HW0_TLS_LE = 80 # type: ignore
R_TILEGX_IMM16_X1_HW0_TLS_LE = 81 # type: ignore
R_TILEGX_IMM16_X0_HW0_LAST_TLS_LE = 82 # type: ignore
R_TILEGX_IMM16_X1_HW0_LAST_TLS_LE = 83 # type: ignore
R_TILEGX_IMM16_X0_HW1_LAST_TLS_LE = 84 # type: ignore
R_TILEGX_IMM16_X1_HW1_LAST_TLS_LE = 85 # type: ignore
R_TILEGX_IMM16_X0_HW0_LAST_TLS_GD = 86 # type: ignore
R_TILEGX_IMM16_X1_HW0_LAST_TLS_GD = 87 # type: ignore
R_TILEGX_IMM16_X0_HW1_LAST_TLS_GD = 88 # type: ignore
R_TILEGX_IMM16_X1_HW1_LAST_TLS_GD = 89 # type: ignore
R_TILEGX_IMM16_X0_HW0_TLS_IE = 92 # type: ignore
R_TILEGX_IMM16_X1_HW0_TLS_IE = 93 # type: ignore
R_TILEGX_IMM16_X0_HW0_LAST_PLT_PCREL = 94 # type: ignore
R_TILEGX_IMM16_X1_HW0_LAST_PLT_PCREL = 95 # type: ignore
R_TILEGX_IMM16_X0_HW1_LAST_PLT_PCREL = 96 # type: ignore
R_TILEGX_IMM16_X1_HW1_LAST_PLT_PCREL = 97 # type: ignore
R_TILEGX_IMM16_X0_HW2_LAST_PLT_PCREL = 98 # type: ignore
R_TILEGX_IMM16_X1_HW2_LAST_PLT_PCREL = 99 # type: ignore
R_TILEGX_IMM16_X0_HW0_LAST_TLS_IE = 100 # type: ignore
R_TILEGX_IMM16_X1_HW0_LAST_TLS_IE = 101 # type: ignore
R_TILEGX_IMM16_X0_HW1_LAST_TLS_IE = 102 # type: ignore
R_TILEGX_IMM16_X1_HW1_LAST_TLS_IE = 103 # type: ignore
R_TILEGX_TLS_DTPMOD64 = 106 # type: ignore
R_TILEGX_TLS_DTPOFF64 = 107 # type: ignore
R_TILEGX_TLS_TPOFF64 = 108 # type: ignore
R_TILEGX_TLS_DTPMOD32 = 109 # type: ignore
R_TILEGX_TLS_DTPOFF32 = 110 # type: ignore
R_TILEGX_TLS_TPOFF32 = 111 # type: ignore
R_TILEGX_TLS_GD_CALL = 112 # type: ignore
R_TILEGX_IMM8_X0_TLS_GD_ADD = 113 # type: ignore
R_TILEGX_IMM8_X1_TLS_GD_ADD = 114 # type: ignore
R_TILEGX_IMM8_Y0_TLS_GD_ADD = 115 # type: ignore
R_TILEGX_IMM8_Y1_TLS_GD_ADD = 116 # type: ignore
R_TILEGX_TLS_IE_LOAD = 117 # type: ignore
R_TILEGX_IMM8_X0_TLS_ADD = 118 # type: ignore
R_TILEGX_IMM8_X1_TLS_ADD = 119 # type: ignore
R_TILEGX_IMM8_Y0_TLS_ADD = 120 # type: ignore
R_TILEGX_IMM8_Y1_TLS_ADD = 121 # type: ignore
R_TILEGX_GNU_VTINHERIT = 128 # type: ignore
R_TILEGX_GNU_VTENTRY = 129 # type: ignore
R_TILEGX_NUM = 130 # type: ignore
EF_RISCV_RVC = 0x0001 # type: ignore
EF_RISCV_FLOAT_ABI = 0x0006 # type: ignore
EF_RISCV_FLOAT_ABI_SOFT = 0x0000 # type: ignore
EF_RISCV_FLOAT_ABI_SINGLE = 0x0002 # type: ignore
EF_RISCV_FLOAT_ABI_DOUBLE = 0x0004 # type: ignore
EF_RISCV_FLOAT_ABI_QUAD = 0x0006 # type: ignore
EF_RISCV_RVE = 0x0008 # type: ignore
EF_RISCV_TSO = 0x0010 # type: ignore
R_RISCV_NONE = 0 # type: ignore
R_RISCV_32 = 1 # type: ignore
R_RISCV_64 = 2 # type: ignore
R_RISCV_RELATIVE = 3 # type: ignore
R_RISCV_COPY = 4 # type: ignore
R_RISCV_JUMP_SLOT = 5 # type: ignore
R_RISCV_TLS_DTPMOD32 = 6 # type: ignore
R_RISCV_TLS_DTPMOD64 = 7 # type: ignore
R_RISCV_TLS_DTPREL32 = 8 # type: ignore
R_RISCV_TLS_DTPREL64 = 9 # type: ignore
R_RISCV_TLS_TPREL32 = 10 # type: ignore
R_RISCV_TLS_TPREL64 = 11 # type: ignore
R_RISCV_BRANCH = 16 # type: ignore
R_RISCV_JAL = 17 # type: ignore
R_RISCV_CALL = 18 # type: ignore
R_RISCV_CALL_PLT = 19 # type: ignore
R_RISCV_GOT_HI20 = 20 # type: ignore
R_RISCV_TLS_GOT_HI20 = 21 # type: ignore
R_RISCV_TLS_GD_HI20 = 22 # type: ignore
R_RISCV_PCREL_HI20 = 23 # type: ignore
R_RISCV_PCREL_LO12_I = 24 # type: ignore
R_RISCV_PCREL_LO12_S = 25 # type: ignore
R_RISCV_HI20 = 26 # type: ignore
R_RISCV_LO12_I = 27 # type: ignore
R_RISCV_LO12_S = 28 # type: ignore
R_RISCV_TPREL_HI20 = 29 # type: ignore
R_RISCV_TPREL_LO12_I = 30 # type: ignore
R_RISCV_TPREL_LO12_S = 31 # type: ignore
R_RISCV_TPREL_ADD = 32 # type: ignore
R_RISCV_ADD8 = 33 # type: ignore
R_RISCV_ADD16 = 34 # type: ignore
R_RISCV_ADD32 = 35 # type: ignore
R_RISCV_ADD64 = 36 # type: ignore
R_RISCV_SUB8 = 37 # type: ignore
R_RISCV_SUB16 = 38 # type: ignore
R_RISCV_SUB32 = 39 # type: ignore
R_RISCV_SUB64 = 40 # type: ignore
R_RISCV_GNU_VTINHERIT = 41 # type: ignore
R_RISCV_GNU_VTENTRY = 42 # type: ignore
R_RISCV_ALIGN = 43 # type: ignore
R_RISCV_RVC_BRANCH = 44 # type: ignore
R_RISCV_RVC_JUMP = 45 # type: ignore
R_RISCV_RVC_LUI = 46 # type: ignore
R_RISCV_GPREL_I = 47 # type: ignore
R_RISCV_GPREL_S = 48 # type: ignore
R_RISCV_TPREL_I = 49 # type: ignore
R_RISCV_TPREL_S = 50 # type: ignore
R_RISCV_RELAX = 51 # type: ignore
R_RISCV_SUB6 = 52 # type: ignore
R_RISCV_SET6 = 53 # type: ignore
R_RISCV_SET8 = 54 # type: ignore
R_RISCV_SET16 = 55 # type: ignore
R_RISCV_SET32 = 56 # type: ignore
R_RISCV_32_PCREL = 57 # type: ignore
R_RISCV_IRELATIVE = 58 # type: ignore
R_RISCV_PLT32 = 59 # type: ignore
R_RISCV_SET_ULEB128 = 60 # type: ignore
R_RISCV_SUB_ULEB128 = 61 # type: ignore
R_RISCV_NUM = 62 # type: ignore
STO_RISCV_VARIANT_CC = 0x80 # type: ignore
SHT_RISCV_ATTRIBUTES = (SHT_LOPROC + 3) # type: ignore
PT_RISCV_ATTRIBUTES = (PT_LOPROC + 3) # type: ignore
DT_RISCV_VARIANT_CC = (DT_LOPROC + 1) # type: ignore
R_BPF_NONE = 0 # type: ignore
R_BPF_64_64 = 1 # type: ignore
R_BPF_64_32 = 10 # type: ignore
R_METAG_HIADDR16 = 0 # type: ignore
R_METAG_LOADDR16 = 1 # type: ignore
R_METAG_ADDR32 = 2 # type: ignore
R_METAG_NONE = 3 # type: ignore
R_METAG_RELBRANCH = 4 # type: ignore
R_METAG_GETSETOFF = 5 # type: ignore
R_METAG_REG32OP1 = 6 # type: ignore
R_METAG_REG32OP2 = 7 # type: ignore
R_METAG_REG32OP3 = 8 # type: ignore
R_METAG_REG16OP1 = 9 # type: ignore
R_METAG_REG16OP2 = 10 # type: ignore
R_METAG_REG16OP3 = 11 # type: ignore
R_METAG_REG32OP4 = 12 # type: ignore
R_METAG_HIOG = 13 # type: ignore
R_METAG_LOOG = 14 # type: ignore
R_METAG_REL8 = 15 # type: ignore
R_METAG_REL16 = 16 # type: ignore
R_METAG_GNU_VTINHERIT = 30 # type: ignore
R_METAG_GNU_VTENTRY = 31 # type: ignore
R_METAG_HI16_GOTOFF = 32 # type: ignore
R_METAG_LO16_GOTOFF = 33 # type: ignore
R_METAG_GETSET_GOTOFF = 34 # type: ignore
R_METAG_GETSET_GOT = 35 # type: ignore
R_METAG_HI16_GOTPC = 36 # type: ignore
R_METAG_LO16_GOTPC = 37 # type: ignore
R_METAG_HI16_PLT = 38 # type: ignore
R_METAG_LO16_PLT = 39 # type: ignore
R_METAG_RELBRANCH_PLT = 40 # type: ignore
R_METAG_GOTOFF = 41 # type: ignore
R_METAG_PLT = 42 # type: ignore
R_METAG_COPY = 43 # type: ignore
R_METAG_JMP_SLOT = 44 # type: ignore
R_METAG_RELATIVE = 45 # type: ignore
R_METAG_GLOB_DAT = 46 # type: ignore
R_METAG_TLS_GD = 47 # type: ignore
R_METAG_TLS_LDM = 48 # type: ignore
R_METAG_TLS_LDO_HI16 = 49 # type: ignore
R_METAG_TLS_LDO_LO16 = 50 # type: ignore
R_METAG_TLS_LDO = 51 # type: ignore
R_METAG_TLS_IE = 52 # type: ignore
R_METAG_TLS_IENONPIC = 53 # type: ignore
R_METAG_TLS_IENONPIC_HI16 = 54 # type: ignore
R_METAG_TLS_IENONPIC_LO16 = 55 # type: ignore
R_METAG_TLS_TPOFF = 56 # type: ignore
R_METAG_TLS_DTPMOD = 57 # type: ignore
R_METAG_TLS_DTPOFF = 58 # type: ignore
R_METAG_TLS_LE = 59 # type: ignore
R_METAG_TLS_LE_HI16 = 60 # type: ignore
R_METAG_TLS_LE_LO16 = 61 # type: ignore
R_NDS32_NONE = 0 # type: ignore
R_NDS32_32_RELA = 20 # type: ignore
R_NDS32_COPY = 39 # type: ignore
R_NDS32_GLOB_DAT = 40 # type: ignore
R_NDS32_JMP_SLOT = 41 # type: ignore
R_NDS32_RELATIVE = 42 # type: ignore
R_NDS32_TLS_TPOFF = 102 # type: ignore
R_NDS32_TLS_DESC = 119 # type: ignore
EF_LARCH_ABI_MODIFIER_MASK = 0x07 # type: ignore
EF_LARCH_ABI_SOFT_FLOAT = 0x01 # type: ignore
EF_LARCH_ABI_SINGLE_FLOAT = 0x02 # type: ignore
EF_LARCH_ABI_DOUBLE_FLOAT = 0x03 # type: ignore
EF_LARCH_OBJABI_V1 = 0x40 # type: ignore
R_LARCH_NONE = 0 # type: ignore
R_LARCH_32 = 1 # type: ignore
R_LARCH_64 = 2 # type: ignore
R_LARCH_RELATIVE = 3 # type: ignore
R_LARCH_COPY = 4 # type: ignore
R_LARCH_JUMP_SLOT = 5 # type: ignore
R_LARCH_TLS_DTPMOD32 = 6 # type: ignore
R_LARCH_TLS_DTPMOD64 = 7 # type: ignore
R_LARCH_TLS_DTPREL32 = 8 # type: ignore
R_LARCH_TLS_DTPREL64 = 9 # type: ignore
R_LARCH_TLS_TPREL32 = 10 # type: ignore
R_LARCH_TLS_TPREL64 = 11 # type: ignore
R_LARCH_IRELATIVE = 12 # type: ignore
R_LARCH_MARK_LA = 20 # type: ignore
R_LARCH_MARK_PCREL = 21 # type: ignore
R_LARCH_SOP_PUSH_PCREL = 22 # type: ignore
R_LARCH_SOP_PUSH_ABSOLUTE = 23 # type: ignore
R_LARCH_SOP_PUSH_DUP = 24 # type: ignore
R_LARCH_SOP_PUSH_GPREL = 25 # type: ignore
R_LARCH_SOP_PUSH_TLS_TPREL = 26 # type: ignore
R_LARCH_SOP_PUSH_TLS_GOT = 27 # type: ignore
R_LARCH_SOP_PUSH_TLS_GD = 28 # type: ignore
R_LARCH_SOP_PUSH_PLT_PCREL = 29 # type: ignore
R_LARCH_SOP_ASSERT = 30 # type: ignore
R_LARCH_SOP_NOT = 31 # type: ignore
R_LARCH_SOP_SUB = 32 # type: ignore
R_LARCH_SOP_SL = 33 # type: ignore
R_LARCH_SOP_SR = 34 # type: ignore
R_LARCH_SOP_ADD = 35 # type: ignore
R_LARCH_SOP_AND = 36 # type: ignore
R_LARCH_SOP_IF_ELSE = 37 # type: ignore
R_LARCH_SOP_POP_32_S_10_5 = 38 # type: ignore
R_LARCH_SOP_POP_32_U_10_12 = 39 # type: ignore
R_LARCH_SOP_POP_32_S_10_12 = 40 # type: ignore
R_LARCH_SOP_POP_32_S_10_16 = 41 # type: ignore
R_LARCH_SOP_POP_32_S_10_16_S2 = 42 # type: ignore
R_LARCH_SOP_POP_32_S_5_20 = 43 # type: ignore
R_LARCH_SOP_POP_32_S_0_5_10_16_S2 = 44 # type: ignore
R_LARCH_SOP_POP_32_S_0_10_10_16_S2 = 45 # type: ignore
R_LARCH_SOP_POP_32_U = 46 # type: ignore
R_LARCH_ADD8 = 47 # type: ignore
R_LARCH_ADD16 = 48 # type: ignore
R_LARCH_ADD24 = 49 # type: ignore
R_LARCH_ADD32 = 50 # type: ignore
R_LARCH_ADD64 = 51 # type: ignore
R_LARCH_SUB8 = 52 # type: ignore
R_LARCH_SUB16 = 53 # type: ignore
R_LARCH_SUB24 = 54 # type: ignore
R_LARCH_SUB32 = 55 # type: ignore
R_LARCH_SUB64 = 56 # type: ignore
R_LARCH_GNU_VTINHERIT = 57 # type: ignore
R_LARCH_GNU_VTENTRY = 58 # type: ignore
R_LARCH_B16 = 64 # type: ignore
R_LARCH_B21 = 65 # type: ignore
R_LARCH_B26 = 66 # type: ignore
R_LARCH_ABS_HI20 = 67 # type: ignore
R_LARCH_ABS_LO12 = 68 # type: ignore
R_LARCH_ABS64_LO20 = 69 # type: ignore
R_LARCH_ABS64_HI12 = 70 # type: ignore
R_LARCH_PCALA_HI20 = 71 # type: ignore
R_LARCH_PCALA_LO12 = 72 # type: ignore
R_LARCH_PCALA64_LO20 = 73 # type: ignore
R_LARCH_PCALA64_HI12 = 74 # type: ignore
R_LARCH_GOT_PC_HI20 = 75 # type: ignore
R_LARCH_GOT_PC_LO12 = 76 # type: ignore
R_LARCH_GOT64_PC_LO20 = 77 # type: ignore
R_LARCH_GOT64_PC_HI12 = 78 # type: ignore
R_LARCH_GOT_HI20 = 79 # type: ignore
R_LARCH_GOT_LO12 = 80 # type: ignore
R_LARCH_GOT64_LO20 = 81 # type: ignore
R_LARCH_GOT64_HI12 = 82 # type: ignore
R_LARCH_TLS_LE_HI20 = 83 # type: ignore
R_LARCH_TLS_LE_LO12 = 84 # type: ignore
R_LARCH_TLS_LE64_LO20 = 85 # type: ignore
R_LARCH_TLS_LE64_HI12 = 86 # type: ignore
R_LARCH_TLS_IE_PC_HI20 = 87 # type: ignore
R_LARCH_TLS_IE_PC_LO12 = 88 # type: ignore
R_LARCH_TLS_IE64_PC_LO20 = 89 # type: ignore
R_LARCH_TLS_IE64_PC_HI12 = 90 # type: ignore
R_LARCH_TLS_IE_HI20 = 91 # type: ignore
R_LARCH_TLS_IE_LO12 = 92 # type: ignore
R_LARCH_TLS_IE64_LO20 = 93 # type: ignore
R_LARCH_TLS_IE64_HI12 = 94 # type: ignore
R_LARCH_TLS_LD_PC_HI20 = 95 # type: ignore
R_LARCH_TLS_LD_HI20 = 96 # type: ignore
R_LARCH_TLS_GD_PC_HI20 = 97 # type: ignore
R_LARCH_TLS_GD_HI20 = 98 # type: ignore
R_LARCH_32_PCREL = 99 # type: ignore
R_LARCH_RELAX = 100 # type: ignore
R_LARCH_DELETE = 101 # type: ignore
R_LARCH_ALIGN = 102 # type: ignore
R_LARCH_PCREL20_S2 = 103 # type: ignore
R_LARCH_CFA = 104 # type: ignore
R_LARCH_ADD6 = 105 # type: ignore
R_LARCH_SUB6 = 106 # type: ignore
R_LARCH_ADD_ULEB128 = 107 # type: ignore
R_LARCH_SUB_ULEB128 = 108 # type: ignore
R_LARCH_64_PCREL = 109 # type: ignore
EF_ARC_MACH_MSK = 0x000000ff # type: ignore
EF_ARC_OSABI_MSK = 0x00000f00 # type: ignore
EF_ARC_ALL_MSK = (EF_ARC_MACH_MSK | EF_ARC_OSABI_MSK) # type: ignore
SHT_ARC_ATTRIBUTES = (SHT_LOPROC + 1) # type: ignore
R_ARC_NONE = 0x0 # type: ignore
R_ARC_8 = 0x1 # type: ignore
R_ARC_16 = 0x2 # type: ignore
R_ARC_24 = 0x3 # type: ignore
R_ARC_32 = 0x4 # type: ignore
R_ARC_B22_PCREL = 0x6 # type: ignore
R_ARC_H30 = 0x7 # type: ignore
R_ARC_N8 = 0x8 # type: ignore
R_ARC_N16 = 0x9 # type: ignore
R_ARC_N24 = 0xA # type: ignore
R_ARC_N32 = 0xB # type: ignore
R_ARC_SDA = 0xC # type: ignore
R_ARC_SECTOFF = 0xD # type: ignore
R_ARC_S21H_PCREL = 0xE # type: ignore
R_ARC_S21W_PCREL = 0xF # type: ignore
R_ARC_S25H_PCREL = 0x10 # type: ignore
R_ARC_S25W_PCREL = 0x11 # type: ignore
R_ARC_SDA32 = 0x12 # type: ignore
R_ARC_SDA_LDST = 0x13 # type: ignore
R_ARC_SDA_LDST1 = 0x14 # type: ignore
R_ARC_SDA_LDST2 = 0x15 # type: ignore
R_ARC_SDA16_LD = 0x16 # type: ignore
R_ARC_SDA16_LD1 = 0x17 # type: ignore
R_ARC_SDA16_LD2 = 0x18 # type: ignore
R_ARC_S13_PCREL = 0x19 # type: ignore
R_ARC_W = 0x1A # type: ignore
R_ARC_32_ME = 0x1B # type: ignore
R_ARC_N32_ME = 0x1C # type: ignore
R_ARC_SECTOFF_ME = 0x1D # type: ignore
R_ARC_SDA32_ME = 0x1E # type: ignore
R_ARC_W_ME = 0x1F # type: ignore
R_ARC_H30_ME = 0x20 # type: ignore
R_ARC_SECTOFF_U8 = 0x21 # type: ignore
R_ARC_SECTOFF_S9 = 0x22 # type: ignore
R_AC_SECTOFF_U8 = 0x23 # type: ignore
R_AC_SECTOFF_U8_1 = 0x24 # type: ignore
R_AC_SECTOFF_U8_2 = 0x25 # type: ignore
R_AC_SECTOFF_S9 = 0x26 # type: ignore
R_AC_SECTOFF_S9_1 = 0x27 # type: ignore
R_AC_SECTOFF_S9_2 = 0x28 # type: ignore
R_ARC_SECTOFF_ME_1 = 0x29 # type: ignore
R_ARC_SECTOFF_ME_2 = 0x2A # type: ignore
R_ARC_SECTOFF_1 = 0x2B # type: ignore
R_ARC_SECTOFF_2 = 0x2C # type: ignore
R_ARC_SDA_12 = 0x2D # type: ignore
R_ARC_SDA16_ST2 = 0x30 # type: ignore
R_ARC_32_PCREL = 0x31 # type: ignore
R_ARC_PC32 = 0x32 # type: ignore
R_ARC_GOTPC32 = 0x33 # type: ignore
R_ARC_PLT32 = 0x34 # type: ignore
R_ARC_COPY = 0x35 # type: ignore
R_ARC_GLOB_DAT = 0x36 # type: ignore
R_ARC_JMP_SLOT = 0x37 # type: ignore
R_ARC_RELATIVE = 0x38 # type: ignore
R_ARC_GOTOFF = 0x39 # type: ignore
R_ARC_GOTPC = 0x3A # type: ignore
R_ARC_GOT32 = 0x3B # type: ignore
R_ARC_S21W_PCREL_PLT = 0x3C # type: ignore
R_ARC_S25H_PCREL_PLT = 0x3D # type: ignore
R_ARC_JLI_SECTOFF = 0x3F # type: ignore
R_ARC_TLS_DTPMOD = 0x42 # type: ignore
R_ARC_TLS_DTPOFF = 0x43 # type: ignore
R_ARC_TLS_TPOFF = 0x44 # type: ignore
R_ARC_TLS_GD_GOT = 0x45 # type: ignore
R_ARC_TLS_GD_LD = 0x46 # type: ignore
R_ARC_TLS_GD_CALL = 0x47 # type: ignore
R_ARC_TLS_IE_GOT = 0x48 # type: ignore
R_ARC_TLS_DTPOFF_S9 = 0x49 # type: ignore
R_ARC_TLS_LE_S9 = 0x4A # type: ignore
R_ARC_TLS_LE_32 = 0x4B # type: ignore
R_ARC_S25W_PCREL_PLT = 0x4C # type: ignore
R_ARC_S21H_PCREL_PLT = 0x4D # type: ignore
R_ARC_NPS_CMEM16 = 0x4E # type: ignore
R_OR1K_NONE = 0 # type: ignore
R_OR1K_32 = 1 # type: ignore
R_OR1K_16 = 2 # type: ignore
R_OR1K_8 = 3 # type: ignore
R_OR1K_LO_16_IN_INSN = 4 # type: ignore
R_OR1K_HI_16_IN_INSN = 5 # type: ignore
R_OR1K_INSN_REL_26 = 6 # type: ignore
R_OR1K_GNU_VTENTRY = 7 # type: ignore
R_OR1K_GNU_VTINHERIT = 8 # type: ignore
R_OR1K_32_PCREL = 9 # type: ignore
R_OR1K_16_PCREL = 10 # type: ignore
R_OR1K_8_PCREL = 11 # type: ignore
R_OR1K_GOTPC_HI16 = 12 # type: ignore
R_OR1K_GOTPC_LO16 = 13 # type: ignore
R_OR1K_GOT16 = 14 # type: ignore
R_OR1K_PLT26 = 15 # type: ignore
R_OR1K_GOTOFF_HI16 = 16 # type: ignore
R_OR1K_GOTOFF_LO16 = 17 # type: ignore
R_OR1K_COPY = 18 # type: ignore
R_OR1K_GLOB_DAT = 19 # type: ignore
R_OR1K_JMP_SLOT = 20 # type: ignore
R_OR1K_RELATIVE = 21 # type: ignore
R_OR1K_TLS_GD_HI16 = 22 # type: ignore
R_OR1K_TLS_GD_LO16 = 23 # type: ignore
R_OR1K_TLS_LDM_HI16 = 24 # type: ignore
R_OR1K_TLS_LDM_LO16 = 25 # type: ignore
R_OR1K_TLS_LDO_HI16 = 26 # type: ignore
R_OR1K_TLS_LDO_LO16 = 27 # type: ignore
R_OR1K_TLS_IE_HI16 = 28 # type: ignore
R_OR1K_TLS_IE_LO16 = 29 # type: ignore
R_OR1K_TLS_LE_HI16 = 30 # type: ignore
R_OR1K_TLS_LE_LO16 = 31 # type: ignore
R_OR1K_TLS_TPOFF = 32 # type: ignore
R_OR1K_TLS_DTPOFF = 33 # type: ignore
R_OR1K_TLS_DTPMOD = 34 # type: ignore
_UNISTD_H = 1 # type: ignore
_POSIX_VERSION = 200809 # type: ignore
__POSIX2_THIS_VERSION = 200809 # type: ignore
_POSIX2_VERSION = __POSIX2_THIS_VERSION # type: ignore
_POSIX2_C_VERSION = __POSIX2_THIS_VERSION # type: ignore
_POSIX2_C_BIND = __POSIX2_THIS_VERSION # type: ignore
_POSIX2_C_DEV = __POSIX2_THIS_VERSION # type: ignore
_POSIX2_SW_DEV = __POSIX2_THIS_VERSION # type: ignore
_POSIX2_LOCALEDEF = __POSIX2_THIS_VERSION # type: ignore
_XOPEN_VERSION = 700 # type: ignore
_XOPEN_XCU_VERSION = 4 # type: ignore
_XOPEN_XPG2 = 1 # type: ignore
_XOPEN_XPG3 = 1 # type: ignore
_XOPEN_XPG4 = 1 # type: ignore
_XOPEN_UNIX = 1 # type: ignore
_XOPEN_ENH_I18N = 1 # type: ignore
_XOPEN_LEGACY = 1 # type: ignore
STDIN_FILENO = 0 # type: ignore
STDOUT_FILENO = 1 # type: ignore
STDERR_FILENO = 2 # type: ignore
R_OK = 4 # type: ignore
W_OK = 2 # type: ignore
X_OK = 1 # type: ignore
F_OK = 0 # type: ignore
SEEK_SET = 0 # type: ignore
SEEK_CUR = 1 # type: ignore
SEEK_END = 2 # type: ignore
L_SET = SEEK_SET # type: ignore
L_INCR = SEEK_CUR # type: ignore
L_XTND = SEEK_END # type: ignore
F_ULOCK = 0 # type: ignore
F_LOCK = 1 # type: ignore
F_TLOCK = 2 # type: ignore
F_TEST = 3 # type: ignore
PROT_READ = 0x1 # type: ignore
PROT_WRITE = 0x2 # type: ignore
PROT_EXEC = 0x4 # type: ignore
PROT_SEM = 0x8 # type: ignore
PROT_NONE = 0x0 # type: ignore
PROT_GROWSDOWN = 0x01000000 # type: ignore
PROT_GROWSUP = 0x02000000 # type: ignore
MAP_TYPE = 0x0f # type: ignore
MAP_FIXED = 0x10 # type: ignore
MAP_ANONYMOUS = 0x20 # type: ignore
MAP_POPULATE = 0x008000 # type: ignore
MAP_NONBLOCK = 0x010000 # type: ignore
MAP_STACK = 0x020000 # type: ignore
MAP_HUGETLB = 0x040000 # type: ignore
MAP_SYNC = 0x080000 # type: ignore
MAP_FIXED_NOREPLACE = 0x100000 # type: ignore
MAP_UNINITIALIZED = 0x4000000 # type: ignore
MLOCK_ONFAULT = 0x01 # type: ignore
MS_ASYNC = 1 # type: ignore
MS_INVALIDATE = 2 # type: ignore
MS_SYNC = 4 # type: ignore
MADV_NORMAL = 0 # type: ignore
MADV_RANDOM = 1 # type: ignore
MADV_SEQUENTIAL = 2 # type: ignore
MADV_WILLNEED = 3 # type: ignore
MADV_DONTNEED = 4 # type: ignore
MADV_FREE = 8 # type: ignore
MADV_REMOVE = 9 # type: ignore
MADV_DONTFORK = 10 # type: ignore
MADV_DOFORK = 11 # type: ignore
MADV_HWPOISON = 100 # type: ignore
MADV_SOFT_OFFLINE = 101 # type: ignore
MADV_MERGEABLE = 12 # type: ignore
MADV_UNMERGEABLE = 13 # type: ignore
MADV_HUGEPAGE = 14 # type: ignore
MADV_NOHUGEPAGE = 15 # type: ignore
MADV_DONTDUMP = 16 # type: ignore
MADV_DODUMP = 17 # type: ignore
MADV_WIPEONFORK = 18 # type: ignore
MADV_KEEPONFORK = 19 # type: ignore
MADV_COLD = 20 # type: ignore
MADV_PAGEOUT = 21 # type: ignore
MADV_POPULATE_READ = 22 # type: ignore
MADV_POPULATE_WRITE = 23 # type: ignore
MADV_DONTNEED_LOCKED = 24 # type: ignore
MADV_COLLAPSE = 25 # type: ignore
MAP_FILE = 0 # type: ignore
PKEY_DISABLE_ACCESS = 0x1 # type: ignore
PKEY_DISABLE_WRITE = 0x2 # type: ignore
PKEY_ACCESS_MASK = (PKEY_DISABLE_ACCESS | PKEY_DISABLE_WRITE) # type: ignore