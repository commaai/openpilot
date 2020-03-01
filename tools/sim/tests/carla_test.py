#!/usr/bin/env python3
import os
import time
import subprocess


def run_carla():
	# Run carla with rendering disabled 
	comm = subprocess.Popen(["carla/CarlaUE4.sh -disable-rendering"],stdout=subprocess.PIPE, shell=True)
	time.sleep(10)
	comm.terminate()	
	print("******* Success ********")

run_carla()



