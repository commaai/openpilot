#!/usr/bin/env python

# custom service run tasks every SLEEP_TIMER second(s)

import time
import subprocess

POWER_OFF_TIMER = 1 # shut down after POWER_OFF_TIMER hours of no USB connection, set to 0 to disable this.
SLEEP_TIMER = 5
BAT_TEMP_LIMIT = 460
BAT_CAP_LIMIT = 35
POWER_OFF_BAT_LIMIT = 35
RESTORE_CPU_FREQ_WHEN_BAT_CAP_AT = 5

def main(gctx=None):

  # when first execute
  set_max_cpu_freq(True)

  # allow charging
  set_charging_status(True)

  prev_usb_status = get_usb_status()
  sec_counter = 0
  power_off_sec = 0

  # calculate power_off_sec value
  if POWER_OFF_TIMER > 0:
    power_off_sec = 60*60*POWER_OFF_TIMER/SLEEP_TIMER

  # main logic
  while True:
    battery_temp = get_battery_temp()
    battery_cap_now = get_battery_capacity()
    charging_status = get_charging_status()

    # if temp is high AND we still have enough battery capacity, then we stop charging the battery
    if battery_temp > BAT_TEMP_LIMIT and battery_cap_now > BAT_CAP_LIMIT:
      allow_charge = False
    else:
      allow_charge = True

    # set battery charging status only when it's different then previous.
    if charging_status != allow_charge:
      set_charging_status(allow_charge)

    # current usb status
    cur_usb_status = get_usb_status()

    # we set cpu back to original freq when usb is not charging
    # and bat is less than the limit
    # so when next time we boot up, it boot faster.
    if battery_cap_now <= RESTORE_CPU_FREQ_WHEN_BAT_CAP_AT and prev_usb_status != 0:
      prev_usb_status = 1
      set_max_cpu_freq(True)
    else:
      # if USB status changed, we update CPU frequency accordingly.
      if cur_usb_status != prev_usb_status:
        set_max_cpu_freq(cur_usb_status == 1)
        prev_usb_status = cur_usb_status

    if power_off_sec > 0 and sec_counter >= power_off_sec:
      set_max_cpu_freq(True)
      shutdown()

    if cur_usb_status == 0:
      sec_counter = sec_counter + 1
    else:
      sec_counter = 0

    time.sleep(SLEEP_TIMER)


def shutdown():
  os.system('LD_LIBRARY_PATH="" svc power shutdown')

def get_battery_temp():
  with open("/sys/class/power_supply/battery/temp") as f:
    return int(f.read())

def get_battery_capacity():
  with open("/sys/class/power_supply/battery/capacity") as f:
    return int(f.read())

def get_charging_status():
  with open("/sys/class/power_supply/battery/charging_enabled") as f:
    return int(f.read())

def get_usb_status():
  with open("/sys/class/power_supply/usb/present") as f:
    return int(f.read())

def set_charging_status(status):
  if status == True:
    val = "1"
  else:
    val = "0"
  subprocess.call(["echo", val, ">", "/sys/class/power_supply/battery/charging_enabled"])

def set_max_cpu_freq(ismax=True):
  param = "{print $NF}"
  if not ismax:
    param = "{print $1}"

  for i in range(0, 4):
    cpuid = str(i)
    try:
      # fetch cpu's available max/min freq
      proc = subprocess.Popen(
        ["awk", param, "/sys/devices/system/cpu/cpu" + cpuid + "/cpufreq/scaling_available_frequencies"],
        stdout=subprocess.PIPE)
      freq = proc.stdout.read()
      # set cpu's max freq to max/min available
      print "Set cpu" , cpuid , " to " , freq
      subprocess.call(["echo", freq, ">", "/sys/devices/system/cpu/cpu" + cpuid + "/freq/scaling_max_freq"])

    except subprocess.CalledProcessError, e:
      print "Unable to extract cpu freq for cpu ", cpuid

if __name__ == "__main__":
  main()
