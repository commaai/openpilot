# GPS test setup
Testing the GPS receiver using GPS spoofing. At the moment only
static location relpay is supported.

# Usage
```
# on host, start gps signal simulation
./run_static_lime.py
```

`run_static_lime.py` downloads the latest ephemeris file from
https://cddis.nasa.gov/archive/gnss/data/daily/20xx/brdc/.


# Hardware Setup
* [LimeSDR USB](https://wiki.myriadrf.org/LimeSDR-USB)
* Asus AX58BT antenna

# Software Setup
* https://github.com/myriadrf/LimeSuite
To communicate with LimeSDR the LimeSuite is needed it abstracts the direct
communication. It also contains examples for a quick start.

The latest stable version(22.09) does not have the corresponding firmware
download available at https://downloads.myriadrf.org/project/limesuite. Therefore
version 20.10 was chosen.

A successull build should give something like:
```
./LimeUtil --info
######################################################
## LimeSuite information summary
######################################################

Version information:
  Library version:	v20.10.0-g1480bfea
  Build timestamp:	2022-09-22
  Interface version:	v2020.10.0
  Binary interface:	20.10-1

System resources:
  Installation root:	/usr/local
  User home directory:	/home/batman
  App data directory:	/home/batman/.local/share/LimeSuite
  Config directory:	/home/batman/.limesuite
  Image search paths:
     - /home/batman/.local/share/LimeSuite/images
     - /usr/local/share/LimeSuite/images

Supported connections:
   * FT601
   * FX3
   * PCIEXillybus
```

* https://github.com/osqzss/LimeGPS
Build on top of LimeSuite (libLimeSuite.so.20.10-1), generates the GPS signal.

With a successful GPS signals can be spoofed.

```
./LimeGPS -e <ephemeris file> -l <location coordinates>

# Example
./LimeGPS -e /pathTo/brdc2660.22n -l 47.202028,15.740394,100
```

# NOTE
GPS spoofing is illegal, be cautious when using it.
