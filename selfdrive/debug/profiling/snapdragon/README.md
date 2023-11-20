snapdragon profiler
--------


* download from https://developer.qualcomm.com/software/snapdragon-profiler/tools-archive (need a qc developer account)
  * choose v2021.5 (verified working with 20.04)
* unzip to selfdrive/debug/profiling/snapdragon/SnapdragonProfiler
* run ```./setup-profiler.sh```
* run ```./setup-agnos.sh```
* run ```selfdrive/debug/adb.sh``` on device
* run the ```adb connect xxx``` command that was given to you on local pc (if you changed adb path in previous step, run that version of adb)
* cd to SnapdragonProfiler and run ```./run_sdp.sh```
* connect to device -> choose device you just setup