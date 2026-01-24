Import('env', 'common', 'messaging')

libs = ['usb-1.0', common, messaging, 'pthread']
panda = env.Library('panda', ['panda.cc', 'panda_comms.cc', 'spi.cc'])

env.Program('pandad', ['main.cc', 'pandad.cc', 'panda_safety.cc'], LIBS=[panda] + libs)

if GetOption('extras'):
  env.Program('tests/test_pandad_usbprotocol', ['tests/test_pandad_usbprotocol.cc'], LIBS=[panda] + libs)
