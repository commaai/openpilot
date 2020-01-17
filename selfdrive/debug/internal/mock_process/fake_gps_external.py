#!/usr/bin/env python3
import time
import zmq

from cereal import log
import cereal.messaging as messaging
from cereal.services import service_list


if __name__ == '__main__':
    gpsLocationExternal = messaging.pub_sock('gpsLocationExternal')

    while True:
        dat = messaging.new_message()
        dat.init('gpsLocationExternal')
        dat.gpsLocationExternal.latitude = 37.6513687
        dat.gpsLocationExternal.longitude = -122.4535056
        dat.gpsLocationExternal.speed = 28.2
        dat.gpsLocationExternal.flags = 1
        dat.gpsLocationExternal.altitude = 75.
        dat.gpsLocationExternal.bearing = 145.5
        dat.gpsLocationExternal.accuracy = 1.
        dat.gpsLocationExternal.timestamp = int(time.time() * 1000)
        dat.gpsLocationExternal.source = log.GpsLocationData.SensorSource.ublox

        gpsLocationExternal.send(dat.to_bytes())
        time.sleep(.1)
