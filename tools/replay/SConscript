Import('env', 'arch', 'common', 'messaging', 'visionipc', 'cereal')

replay_env = env.Clone()
replay_env['CCFLAGS'] += ['-Wno-deprecated-declarations']

base_frameworks = []
base_libs = [common, messaging, cereal, visionipc, 'm', 'ssl', 'crypto', 'pthread']

if arch == "Darwin":
  base_frameworks.append('OpenCL')
else:
  base_libs.append('OpenCL')

replay_lib_src = ["replay.cc", "consoleui.cc", "camera.cc", "filereader.cc", "logreader.cc", "framereader.cc",
                  "route.cc", "util.cc", "seg_mgr.cc", "timeline.cc", "api.cc", "qcom_decoder.cc"]
replay_lib = replay_env.Library("replay", replay_lib_src, LIBS=base_libs, FRAMEWORKS=base_frameworks)
Export('replay_lib')
replay_libs = [replay_lib, 'avutil', 'avcodec', 'avformat', 'bz2', 'zstd', 'curl', 'yuv', 'ncurses'] + base_libs
replay_env.Program("replay", ["main.cc"], LIBS=replay_libs, FRAMEWORKS=base_frameworks)

if GetOption('extras'):
  replay_env.Program('tests/test_replay', ['tests/test_replay.cc'], LIBS=replay_libs)
