# webcamerad
Follow only step #2 from tools/sim/README.md, which is:
```
cd ~/openpilot/selfdrive/
PASSIVE=0 NOBOARD=1 ./manager.py
```
then in a different shell run
```
cd ~/openpilot/tools/webcam
LD_LIBRARY_PATH=$HOME/openpilot/cereal:$LD_LIBRARY_PATH ./webcamerad
```
and then you should see the openpilot UI display your webcam feed
