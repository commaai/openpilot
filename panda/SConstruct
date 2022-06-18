AddOption('--test',
          action='store_true',
          help='build test files')

SConscript('board/SConscript')

if GetOption('test'):
  SConscript('tests/safety/SConscript')
