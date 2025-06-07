Import("env")

SConscript(['opendbc/can/SConscript'], exports={'env': env})
SConscript(['opendbc/dbc/SConscript'], exports={'env': env})

# test files
if GetOption('extras'):
  SConscript('opendbc/safety/tests/libsafety/SConscript')
