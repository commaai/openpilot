#!/bin/bash
PYTHON_PATH=$(readlink -f $(which python3))
sudo setcap 'cap_dac_override,cap_sys_rawio,cap_sys_admin,cap_ipc_lock=ep' $PYTHON_PATH
