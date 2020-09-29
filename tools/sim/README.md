openpilot in simulator
=====================


## Running the simulator
First, start the CARLA server.
```
./start_carla.sh
```

Then start bridge and openpilot.
```
./start_openpilot_docker.sh
``` 

To engage openpilot press 1 a few times while focused on bridge.py to increase the cruise speed. 
## Controls

You can control openpilot driving in the simulation with the following keys

|  key  |   functionality   |
| :---: | :---------------: |
|   1   |  Cruise up 5 mph  |
|   2   | Cruise down 5 mph |
|   3   |   Cruise cancel   |
|   q   |     Exit all      |

## Arguments
Arguments for `start_openpilot_docker.sh`:
```
  -h, --help            show this help message and exit
  --joystick
  --town TOWN
  --spawn_point NUM_SELECTED_SPAWN_POINT
  --cloudyness CLOUDYNESS
  --precipitation PRECIPITATION
  --precipitation_deposits PRECIPITATION_DEPOSITS
  --wind_intensity WIND_INTENSITY
  --sun_azimuth_angle SUN_AZIMUTH_ANGLE
  --sun_altitude_angle SUN_ALTITUDE_ANGLE
```
