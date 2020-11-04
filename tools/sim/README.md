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

Or

Start Carla a background container and openpilot in a foreground container with an interactive terminal running tmux.
```
./start_sim.sh
```

Clean up with:
```
./stop_sim.sh
```
(shuts down Carla and cleans up the containers)

To engage openpilot press 1 a few times while focused on bridge.py to increase the cruise speed. 
## Controls

You can control openpilot driving in the simulation with the following keys

|  key  |   functionality   |
| :---: | :---------------: |
|   1   |  Cruise up 5 mph  |
|   2   | Cruise down 5 mph |
|   3   |   Cruise cancel   |
|   q   |     Exit all      |

To see the options for changing the environment, such as the town, spawn point or precipitation, you can run `./start_openpilot_docker.sh --help`.
This will print the help output inside the docker container. You need to exit the docker container before running `./start_openpilot_docker.sh` again.
