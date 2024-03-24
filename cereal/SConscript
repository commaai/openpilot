Import('env', 'envCython', 'arch', 'common')

import shutil

cereal_dir = Dir('.')
gen_dir = Dir('gen')
messaging_dir = Dir('messaging')

# Build cereal

schema_files = ['log.capnp', 'car.capnp', 'legacy.capnp', 'custom.capnp']
env.Command(["gen/c/include/c++.capnp.h"], [], "mkdir -p " + gen_dir.path + "/c/include && touch $TARGETS")
env.Command([f'gen/cpp/{s}.c++' for s in schema_files] + [f'gen/cpp/{s}.h' for s in schema_files],
            schema_files,
            f"capnpc --src-prefix={cereal_dir.path} $SOURCES -o c++:{gen_dir.path}/cpp/")

# TODO: remove non shared cereal and messaging
cereal_objects = env.SharedObject([f'gen/cpp/{s}.c++' for s in schema_files])

cereal = env.Library('cereal', cereal_objects)
env.SharedLibrary('cereal_shared', cereal_objects)

# Build messaging

services_h = env.Command(['services.h'], ['services.py'], 'python3 ' + cereal_dir.path + '/services.py > $TARGET')

messaging_objects = env.SharedObject([
  'messaging/messaging.cc',
  'messaging/event.cc',
  'messaging/impl_zmq.cc',
  'messaging/impl_msgq.cc',
  'messaging/impl_fake.cc',
  'messaging/msgq.cc',
  'messaging/socketmaster.cc',
])

messaging = env.Library('messaging', messaging_objects)
Depends('messaging/impl_zmq.cc', services_h)

env.Program('messaging/bridge', ['messaging/bridge.cc'], LIBS=[messaging, 'zmq', common])
Depends('messaging/bridge.cc', services_h)

messaging_python = envCython.Program('messaging/messaging_pyx.so', 'messaging/messaging_pyx.pyx', LIBS=envCython["LIBS"]+[messaging, "zmq", common])

# Build Vision IPC
vipc_sources = [
  'visionipc/ipc.cc',
  'visionipc/visionipc_server.cc',
  'visionipc/visionipc_client.cc',
  'visionipc/visionbuf.cc',
]

if arch == "larch64":
  vipc_sources += ['visionipc/visionbuf_ion.cc']
else:
  vipc_sources += ['visionipc/visionbuf_cl.cc']

vipc_objects = env.SharedObject(vipc_sources)
visionipc = env.Library('visionipc', vipc_objects)


vipc_frameworks = []
vipc_libs = envCython["LIBS"] + [visionipc, messaging, common, "zmq"]
if arch == "Darwin":
  vipc_frameworks.append('OpenCL')
else:
  vipc_libs.append('OpenCL')
envCython.Program('visionipc/visionipc_pyx.so', 'visionipc/visionipc_pyx.pyx',
                  LIBS=vipc_libs, FRAMEWORKS=vipc_frameworks)

if GetOption('extras'):
  env.Program('messaging/test_runner', ['messaging/test_runner.cc', 'messaging/msgq_tests.cc'], LIBS=[messaging, common])

  env.Program('visionipc/test_runner', ['visionipc/test_runner.cc', 'visionipc/visionipc_tests.cc'],
              LIBS=['pthread'] + vipc_libs, FRAMEWORKS=vipc_frameworks)


Export('cereal', 'messaging', 'messaging_python', 'visionipc')