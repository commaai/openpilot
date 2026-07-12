import os

Import('env', 'envCython', 'common', 'rednose_root')

helpers = os.path.join(rednose_root, 'helpers')
objects = env.SharedObject([os.path.join(helpers, f) for f in ('ekf_load.cc', 'ekf_sym.cc')])
libs = ['dl'] + ([common, 'zmq'] if common != '' else [])
rednose = env.Library('rednose', objects, LIBS=libs)
rednose_python = envCython.Program(os.path.join(helpers, 'ekf_sym_pyx.so'),
                                  [os.path.join(helpers, 'ekf_sym_pyx.pyx'), objects],
                                  LIBS=libs + envCython['LIBS'])
Export('rednose', 'rednose_python')
