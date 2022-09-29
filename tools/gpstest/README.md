# GPS test setup

# Usage
```
# replaying a static location
./gpstest.sh -e <ephemeris file> -s <static location>

# replaying a prerecorded route (NMEA cvs file)
./gpstest.sh -e <ephemeris file> -d <dynamic location>
```

If `-e` is not provided the latest ephemeris file will be downloaded from
https://cddis.nasa.gov/archive/gnss/data/daily/20xx/brdc/.
(TODO: add auto downloader)

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
