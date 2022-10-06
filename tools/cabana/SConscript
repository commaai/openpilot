import os
Import('env', 'qt_env', 'arch', 'common', 'messaging', 'visionipc',
       'cereal', 'transformations', 'widgets', 'opendbc')

base_frameworks = qt_env['FRAMEWORKS']
base_libs = [common, messaging, cereal, visionipc, transformations, 'zmq',
             'capnp', 'kj', 'm', 'ssl', 'crypto', 'pthread'] + qt_env["LIBS"]

if arch == "Darwin":
  base_frameworks.append('OpenCL')
else:
  base_libs.append('OpenCL')

qt_libs = ['qt_util', 'Qt5Charts'] + base_libs
if arch in ['x86_64', 'Darwin'] and GetOption('extras'):
  qt_env['CXXFLAGS'] += ["-Wno-deprecated-declarations"]

  Import('replay_lib')
  cabana_libs = [widgets, cereal, messaging, visionipc, replay_lib, opendbc,'avutil', 'avcodec', 'avformat', 'bz2', 'curl', 'yuv'] + qt_libs
  qt_env.Program('_cabana', ['cabana.cc', 'mainwin.cc', 'chartswidget.cc', 'videowidget.cc', 'signaledit.cc', 'parser.cc', 'messageswidget.cc', 'detailwidget.cc'], LIBS=cabana_libs, FRAMEWORKS=base_frameworks)
