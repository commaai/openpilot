# GPS test setup
Testing the GPS receiver using GPS spoofing. At the moment only
static location relpay is supported.

# Usage
on C3 run `rpc_server.py`, on host PC run `fuzzy_testing.py`

`simulate_gps_signal.py` downloads the latest ephemeris file from
https://cddis.nasa.gov/archive/gnss/data/daily/20xx/brdc/.


# Hardware Setup
* [LimeSDR USB](https://wiki.myriadrf.org/LimeSDR-USB)
* Asus AX58BT antenna

# Software Setup
* https://github.com/myriadrf/LimeSuite
To communicate with LimeSDR the LimeSuite is needed it abstracts the direct
communication. It also contains examples for a quick start.

The latest stable version (22.09) does not have the corresponding firmware
download available at https://downloads.myriadrf.org/project/limesuite. Therefore
version 20.10 was chosen.

* https://github.com/osqzss/LimeGPS
Built on top of LimeSuite (libLimeSuite.so.20.10-1), generates the GPS signal.

```
./LimeGPS -e <ephemeris file> -l <location coordinates>

# Example
./LimeGPS -e /pathTo/brdc2660.22n -l 47.202028,15.740394,100
```
