In order to run any of the scripts in this folder, one needs first to set the PYTHONPATH in their environment to point to /data/openpilot. 

To do so just run the command below in your bash session before running any other script

```export PYTHONPATH="/data/openpilot/"```

Also, please make sure at least boardd is not running (I recommend killing the whole OP by doing a ```tmux a``` and then press ```Ctrl + C``` once, wait for ```KeyboardInterrupt``` to appear and then press ```Ctrl + C``` again).
