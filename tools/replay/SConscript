import os
Import('env', 'qt_env', 'arch', 'common', 'messaging', 'visionipc',
       'cereal', 'transformations')

base_libs = [common, messaging, cereal, visionipc, transformations, 'zmq',
             'capnp', 'kj', 'm', 'OpenCL', 'ssl', 'crypto', 'pthread'] + qt_env["LIBS"]

qt_libs = ['qt_util'] + base_libs
if arch in ['x86_64', 'Darwin'] or GetOption('extras'):
  qt_env['CXXFLAGS'] += ["-Wno-deprecated-declarations"]

  replay_lib_src = ["replay.cc", "consoleui.cc", "camera.cc", "filereader.cc", "logreader.cc", "framereader.cc", "route.cc", "util.cc"]

  replay_lib = qt_env.Library("qt_replay", replay_lib_src, LIBS=qt_libs)
  replay_libs = [replay_lib, 'avutil', 'avcodec', 'avformat', 'bz2', 'curl', 'yuv', 'ncurses'] + qt_libs
  qt_env.Program("replay", ["main.cc"], LIBS=replay_libs)

  if GetOption('test'):
    qt_env.Program('tests/test_replay', ['tests/test_runner.cc', 'tests/test_replay.cc'], LIBS=[replay_libs])
