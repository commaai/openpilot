import os
Import('env', 'qt_env', 'arch', 'common', 'messaging', 'visionipc', 'replay_lib',
       'cereal', 'transformations', 'widgets', 'opendbc', 'asset_obj')

base_frameworks = qt_env['FRAMEWORKS']
base_libs = [common, messaging, cereal, visionipc, transformations, 'zmq',
             'capnp', 'kj', 'm', 'ssl', 'crypto', 'pthread'] + qt_env["LIBS"]

if arch == "Darwin":
  base_frameworks.append('OpenCL')
  base_frameworks.append('QtCharts')
else:
  base_libs.append('OpenCL')
  base_libs.append('Qt5Charts')

qt_libs = ['qt_util'] + base_libs

cabana_libs = [widgets, cereal, messaging, visionipc, replay_lib, opendbc,'avutil', 'avcodec', 'avformat', 'bz2', 'curl', 'yuv'] + qt_libs
cabana_env = qt_env.Clone()

prev_moc_path = cabana_env['QT_MOCHPREFIX']
cabana_env['QT_MOCHPREFIX'] = os.path.dirname(prev_moc_path) + '/cabana/moc_'
cabana_lib = cabana_env.Library("cabana_lib", ['mainwin.cc', 'streams/livestream.cc', 'streams/abstractstream.cc', 'streams/replaystream.cc', 'binaryview.cc', 'chartswidget.cc', 'historylog.cc', 'videowidget.cc', 'signaledit.cc', 'dbcmanager.cc',
                                               'commands.cc', 'messageswidget.cc', 'settings.cc', 'util.cc', 'detailwidget.cc', 'tools/findsimilarbits.cc'], LIBS=cabana_libs, FRAMEWORKS=base_frameworks)
cabana_env.Program('_cabana', ['cabana.cc', cabana_lib, asset_obj], LIBS=cabana_libs, FRAMEWORKS=base_frameworks)

if arch == "Darwin":
  cabana_env.Execute('install_name_tool -change opendbc/can/libdbc.dylib @loader_path/../../opendbc/can/libdbc.dylib ./_cabana')

if GetOption('test'):
  cabana_env.Program('tests/_test_cabana', ['tests/test_runner.cc', 'tests/test_cabana.cc', cabana_lib], LIBS=[cabana_libs])

def generate_dbc_json(target, source, env):
  env.Execute('tools/cabana/generate_dbc_json.py --out tools/cabana/car_fingerprint_to_dbc.json')
cabana_env.Command('generate_dbc_json', [], generate_dbc_json)
