openpilot in simulator
=====================


## Building the openpilot container
Until a more seamless sim script with all the deps included, these are the instructions for onnxruntime-gpu==1.5.3 with CUDA 10.2 + CuDNN 8:
1. You'll need the `cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb` file from here: https://developer.nvidia.com/cuda-10.2-download-archive (Linux -> x86_64 -> Ubuntu -> 18.04 -> deb (local)
2. You'll also need the CuDNN runtime library file `libcudnn8_8.0.4.30-1+cuda10.2_amd64.deb` from here: https://developer.nvidia.com/rdp/cudnn-archive , (version v8.0.3 for CUDA 10 -> cuDNN Runtime Library for Ubuntu18.04 (Deb) )
3. Download and place both files above in `openpilot/tools/sim/assets` folder (create it if it doesnt exist).
4. Run `build_container.sh` script.
5. Use the instructions below to run the simulator.


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

To see the options for changing the environment, such as the town, spawn point or precipitation, you can run `./start_openpilot_docker.sh --help`.
This will print the help output inside the docker container. You need to exit the docker container before running `./start_openpilot_docker.sh` again.

## Further Reading
The following resources contain more details and troubleshooting tips.
* [CARLA on the openpilot wiki](https://github.com/commaai/openpilot/wiki/CARLA) 
