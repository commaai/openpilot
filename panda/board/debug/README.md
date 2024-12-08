# In-circuit debugging using openocd and gdb

## Hardware
Connect an ST-Link V2 programmer to the SWD pins on the board. The pins that need to be connected are:
- GND
- VTref
- SWDIO
- SWCLK
- NRST

Make sure you're using a genuine one for boards that do not have a 3.3V panda power rail. For example, the tres runs at 1.8V, which is not supported by the clones.

## Openocd
Install openocd. For Ubuntu 24.04, the one in the package manager works fine: `sudo apt install openocd`.

To run, use `./debug_f4.sh (TODO)` or `./debug_h7.sh` depending on the panda.

## GDB
You need `gdb-multiarch`.

Once openocd is running, you can connect from gdb as follows:
```
$ gdb-multiarch
(gdb) target ext :3333
```
To reset and break, use `monitor reset halt`.
