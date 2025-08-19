import os
import subprocess
import platform

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"

cpppath = [
  '#',
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

# safety options
AddOption('--ubsan',
          action='store_true',
          help='turn on UBSan')

AddOption('--mutation',
          action='store_true',
          help='generate mutation-ready code')

ccflags_asan = ["-fsanitize=address", "-fno-omit-frame-pointer"] if GetOption('asan') else []
ldflags_asan = ["-fsanitize=address"] if GetOption('asan') else []

env = Environment(
  ENV=os.environ,
  CC='gcc',
  CXX='g++',
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Wunused",
    "-Werror",
    "-Wshadow",
    "-Wno-vla-cxx-extension",
    "-Wno-unknown-warning-option",  # for compatibility across compiler versions
  ] + ccflags_asan,
  LDFLAGS=ldflags_asan,
  LINKFLAGS=ldflags_asan,
  CFLAGS="-std=gnu11",
  CXXFLAGS=["-std=c++1z"],
  CPPPATH=cpppath,
  tools=["default", "compilation_db"]
)
if arch != "Darwin":
  env.Append(CCFLAGS=["-fmax-errors=1", ])

env.CompilationDatabase('compile_commands.json')

Export('env', 'arch')

SConscript(['SConscript'])
