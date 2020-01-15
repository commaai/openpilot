Installs the panda linux kernel driver using DKMS.

This will allow the panda to work with tools such as `can-utils`

prerequisites:
 - `apt-get install dkms gcc linux-headers-$(uname -r) make sudo`

installation:
 - `make link` (only needed the first time. It will report an error on subsequent attempts to link)
 - `make all`
 - `make install`

uninstall:
 - `make uninstall`

usage:

You will need to bring it up using `sudo ifconfig can0 up` or
`sudo ip link set dev can0 up`, depending on your platform.
