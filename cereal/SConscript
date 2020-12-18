Import('env', 'envCython', 'arch', 'zmq')

import shutil

cereal_dir = Dir('.')
gen_dir = Dir('gen')
messaging_dir = Dir('messaging')

# TODO: remove src-prefix and cereal from command string. can we set working directory?
env.Command(["gen/c/include/c++.capnp.h", "gen/c/include/java.capnp.h"], [], "mkdir -p " + gen_dir.path + "/c/include && touch $TARGETS")
env.Command(['gen/cpp/car.capnp.c++', 'gen/cpp/log.capnp.c++', 'gen/cpp/car.capnp.h', 'gen/cpp/log.capnp.h'],
            ['car.capnp', 'log.capnp'],
            f"capnpc --src-prefix={cereal_dir.path} $SOURCES -o c++:{gen_dir.path}/cpp/")

if shutil.which('capnpc-java'):
  env.Command(['gen/java/Car.java', 'gen/java/Log.java'],
              ['car.capnp', 'log.capnp'],
              f"capnpc $SOURCES --src-prefix={cereal_dir.path} -o java:{gen_dir.path}/java/")

# TODO: remove non shared cereal and messaging
cereal_objects = env.SharedObject([
  'gen/cpp/car.capnp.c++',
  'gen/cpp/log.capnp.c++',
])

env.Library('cereal', cereal_objects)
env.SharedLibrary('cereal_shared', cereal_objects)

cereal_dir = Dir('.')
services_h = env.Command(['services.h'],
                          ['service_list.yaml', 'services.py'],
                          'python3 ' + cereal_dir.path + '/services.py > $TARGET')

messaging_objects = env.SharedObject([
  'messaging/messaging.cc',
  'messaging/impl_zmq.cc',
  'messaging/impl_msgq.cc',
  'messaging/msgq.cc',
  'messaging/socketmaster.cc',
])

messaging_lib = env.Library('messaging', messaging_objects)
Depends('messaging/impl_zmq.cc', services_h)

# note, this rebuilds the deps shared, zmq is statically linked to make APK happy
# TODO: get APK to load system zmq to remove the static link
if arch == "aarch64":
  zmq_static = FindFile("libzmq.a", "/usr/lib")
  shared_lib_shared_lib = [zmq_static, 'm', 'stdc++', "gnustl_shared", "kj", "capnp"]
  env.SharedLibrary('messaging_shared', messaging_objects, LIBS=shared_lib_shared_lib)

env.Program('messaging/bridge', ['messaging/bridge.cc'], LIBS=[messaging_lib, 'zmq'])
Depends('messaging/bridge.cc', services_h)

# different target?
#env.Program('messaging/demo', ['messaging/demo.cc'], LIBS=[messaging_lib, 'zmq'])

envCython.Program('messaging/messaging_pyx.so', 'messaging/messaging_pyx.pyx', LIBS=envCython["LIBS"]+[messaging_lib, "zmq"])

if GetOption('test'):
  env.Program('messaging/test_runner', ['messaging/test_runner.cc', 'messaging/msgq_tests.cc'], LIBS=[messaging_lib])
