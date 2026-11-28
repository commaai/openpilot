#!/bin/sh
python3 -c "
try:
    from tinygrad.runtime.support.system import APLRemotePCIDevice
    APLRemotePCIDevice.ensure_app()
except Exception as e:
    print('Your tinygrad is too old. Please clone the latest tinygrad: git clone https://github.com/tinygrad/tinygrad.git && cd tinygrad && python3 -m pip install -e .')
    print(e)
    exit(1)
"
