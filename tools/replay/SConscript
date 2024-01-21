Import('env', 'qt_env', 'arch', 'common', 'messaging', 'visionipc', 'cereal')

base_frameworks = qt_env['FRAMEWORKS']
base_libs = [common, messaging, cereal, visionipc, 'zmq',
             'capnp', 'kj', 'm', 'ssl', 'crypto', 'pthread', 'qt_util'] + qt_env["LIBS"]

if arch == "Darwin":
  base_frameworks.append('OpenCL')
else:
  base_libs.append('OpenCL')

qt_env['CXXFLAGS'] += ["-Wno-deprecated-declarations"]

replay_lib_src = ["replay.cc", "consoleui.cc", "camera.cc", "filereader.cc", "logreader.cc", "framereader.cc", "route.cc", "util.cc"]
replay_lib = qt_env.Library("qt_replay", replay_lib_src, LIBS=base_libs, FRAMEWORKS=base_frameworks)
Export('replay_lib')
replay_libs = [replay_lib, 'avutil', 'avcodec', 'avformat', 'bz2', 'curl', 'yuv', 'ncurses'] + base_libs
qt_env.Program("replay", ["main.cc"], LIBS=replay_libs, FRAMEWORKS=base_frameworks)

if GetOption('extras'):
  qt_env.Program('tests/test_replay', ['tests/test_runner.cc', 'tests/test_replay.cc'], LIBS=[replay_libs, base_libs])
