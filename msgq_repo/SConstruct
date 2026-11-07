import os
import platform
import subprocess

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"

# shm_open/shm_unlink live in librt on older glibc versions (including manylinux).
common = [] if arch == "Darwin" else ["rt"]

cpppath = [
  "#/",
  '#msgq/',
  '/usr/lib/include',
]

AddOption('--minimal',
          action='store_false',
          dest='extras',
          default=True,
          help='the minimum build. no tests, tools, etc.')

AddOption('--asan',
          action='store_true',
          help='turn on ASAN')

AddOption('--ubsan',
          action='store_true',
          help='turn on UBSan')

ccflags = []
ldflags = []
if GetOption('ubsan'):
  flags = [
    "-fsanitize=undefined",
    "-fno-sanitize-recover=undefined",
  ]
  ccflags += flags
  ldflags += flags
elif GetOption('asan'):
  ccflags += ["-fsanitize=address", "-fno-omit-frame-pointer"]
  ldflags += ["-fsanitize=address"]

env = Environment(
  ENV=os.environ,
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Wunused",
    "-Werror",
    "-Wshadow" if arch == "Darwin" else "-Wshadow=local",
    "-Wno-vla-cxx-extension",
    "-Wno-unknown-warning-option",
  ] + ccflags,
  LDFLAGS=ldflags,
  LINKFLAGS=ldflags,

  CFLAGS="-std=gnu11",
  CXXFLAGS="-std=c++1z",
  CPPPATH=cpppath,
  tools=["default"]
)

Export('env', 'arch', 'common')

SConscript(['SConscript'])
