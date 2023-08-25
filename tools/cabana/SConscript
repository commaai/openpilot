import os
Import('env', 'qt_env', 'arch', 'common', 'messaging', 'visionipc', 'replay_lib',
       'cereal', 'transformations', 'widgets')

base_frameworks = qt_env['FRAMEWORKS']
base_libs = [common, messaging, cereal, visionipc, transformations, 'zmq',
             'capnp', 'kj', 'm', 'ssl', 'crypto', 'pthread'] + qt_env["LIBS"]

if arch == "Darwin":
  base_frameworks.append('OpenCL')
  base_frameworks.append('QtCharts')
  base_frameworks.append('QtSerialBus')
else:
  base_libs.append('OpenCL')
  base_libs.append('Qt5Charts')
  base_libs.append('Qt5SerialBus')

qt_libs = ['qt_util'] + base_libs

cabana_env = qt_env.Clone()
cabana_env["LIBPATH"] += ['../../opendbc/can']
cabana_libs = [widgets, cereal, messaging, visionipc, replay_lib, 'panda', 'libdbc_static', 'avutil', 'avcodec', 'avformat', 'bz2', 'curl', 'yuv', 'usb-1.0'] + qt_libs
opendbc_path = '-DOPENDBC_FILE_PATH=\'"%s"\'' % (cabana_env.Dir("../../opendbc").abspath)
cabana_env['CXXFLAGS'] += [opendbc_path]

# build assets
assets = "assets/assets.cc"
assets_src = "assets/assets.qrc"
cabana_env.Command(assets, assets_src, f"rcc $SOURCES -o $TARGET")
cabana_env.Depends(assets, Glob('/assets/*', exclude=[assets, assets_src, "assets/assets.o"]))

cabana_lib = cabana_env.Library("cabana_lib", ['mainwin.cc', 'streams/socketcanstream.cc', 'streams/pandastream.cc', 'streams/devicestream.cc', 'streams/livestream.cc', 'streams/abstractstream.cc', 'streams/replaystream.cc', 'binaryview.cc', 'historylog.cc', 'videowidget.cc', 'signalview.cc',
                                               'dbc/dbc.cc', 'dbc/dbcfile.cc', 'dbc/dbcmanager.cc',
                                               'chart/chartswidget.cc', 'chart/chart.cc', 'chart/signalselector.cc', 'chart/tiplabel.cc', 'chart/sparkline.cc',
                                               'commands.cc', 'messageswidget.cc', 'streamselector.cc', 'settings.cc', 'util.cc', 'detailwidget.cc', 'tools/findsimilarbits.cc', 'tools/findsignal.cc'], LIBS=cabana_libs, FRAMEWORKS=base_frameworks)
cabana_env.Program('cabana', ['cabana.cc', cabana_lib, assets], LIBS=cabana_libs, FRAMEWORKS=base_frameworks)

if GetOption('extras'):
  cabana_env.Program('tests/test_cabana', ['tests/test_runner.cc', 'tests/test_cabana.cc', cabana_lib], LIBS=[cabana_libs])

generate_dbc = cabana_env.Command('generate_dbc_json',
                                   [],
                                   "python3 tools/cabana/dbc/generate_dbc_json.py --out tools/cabana/dbc/car_fingerprint_to_dbc.json")
cabana_env.Depends(generate_dbc, ["#common", "#selfdrive/boardd", "#opendbc", "#cereal"])
