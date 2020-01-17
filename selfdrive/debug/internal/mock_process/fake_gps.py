# mock_gps.py: Publishes a generated path moving at 15m/s to gpsLocation
# USAGE: python mock_gps.py
# Then start manager

from itertools import cycle
import time
import zmq

from cereal import log
import cereal.messaging as messaging
from cereal.services import service_list

degrees_per_meter = 0.000009000009 # approximation
start_lat = 43.64199141443989
start_lng = -94.97520411931725

def gen_path(length_seconds, speed=15):
    return [{"lat": start_lat,
             "lng": start_lng + speed * i * degrees_per_meter, # moving along longitudinal axis at speed m/s
             "speed": speed}
             for i in range(1, length_seconds + 1)]

if __name__ == '__main__':
    gpsLocation = messaging.pub_sock('gpsLocation')

    path_stopped_5s = [{"lat": start_lat, "lng": start_lng, "speed": 0}] * 5
    path_moving = gen_path(30, speed=15)
    path_stopped_5s_then_moving = path_stopped_5s + path_moving

    for point in cycle(path_stopped_5s_then_moving):
        print('sending gpsLocation from point: {}'.format(str(point)))
        dat = messaging.new_message()
        dat.init('gpsLocation')
        dat.gpsLocation.latitude = point['lat']
        dat.gpsLocation.longitude = point['lng']
        dat.gpsLocation.speed = point['speed']
        dat.gpsLocation.flags = 0
        dat.gpsLocation.altitude = 0
        dat.gpsLocation.bearing = 0 # todo we can mock this
        dat.gpsLocation.accuracy = 1
        dat.gpsLocation.timestamp = int(time.time() * 1000)
        dat.gpsLocation.source = log.GpsLocationData.SensorSource.android

        gpsLocation.send(dat.to_bytes())
        time.sleep(1)

