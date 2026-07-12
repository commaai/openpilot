import os

Import('env', 'envCython', 'common', 'msgq_root')

def src(*parts):
  return os.path.join(msgq_root, *parts)

msgq_objects = env.SharedObject([src(f) for f in (
  'ipc.cc', 'event.cc', 'impl_msgq.cc', 'impl_fake.cc', 'msgq.cc',
)])
msgq = env.Library('msgq', msgq_objects)
msgq_python = envCython.Program(src('ipc_pyx.so'), src('ipc_pyx.pyx'), LIBS=envCython['LIBS'] + [msgq] + common)

visionipc_dir = src('visionipc')
vipc_files = ['visionipc.cc', 'visionipc_server.cc', 'visionipc_client.cc']
vipc_files += ['visionbuf_ion.cc'] if File('/dev/ion').exists() else ['visionbuf.cc']
visionipc = env.Library('visionipc', env.SharedObject([os.path.join(visionipc_dir, f) for f in vipc_files]))
envCython.Program(os.path.join(visionipc_dir, 'visionipc_pyx.so'), os.path.join(visionipc_dir, 'visionipc_pyx.pyx'),
                  LIBS=envCython['LIBS'] + [visionipc, msgq] + common)

Export('visionipc', 'msgq', 'msgq_python')
