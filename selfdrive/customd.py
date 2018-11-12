#!/usr/bin/env python
import os
import time

# custom service run tasks every SLEEP_TIMER second(s)

POWER_OFF_TIMER = 1 # shut down after POWER_OFF_TIMER hours of no USB connection, set to 0 to disable this.
SLEEP_TIMER = 5 # sleep timer, in seconds, how often you would like the script to query the system status.
BAT_TEMP_LIMIT = 460 # temp limit - 46 degree, match thermald.py, it stop charging when reach this temp level.
BAT_CAP_LIMIT = 35 # battery limit (percentage), if battery capacity is lower than this then it will keep charging.
RESTORE_CPU_FREQ_WHEN_BAT_CAP_AT = 5 # when battery reach this capacity, we turn cpu freq back on prepare for a shutdown

def main(gctx=None):
  # a few system optimisation, may only effect from next reboot
  battery_optimisation()

  # when first execute
  set_max_cpu_freq(True)

  # allow charging
  set_charging_status(True)

  prev_usb_status = get_usb_status()
  count = 0
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

    if power_off_sec > 0 and count >= power_off_sec:
      set_max_cpu_freq(True)
      shutdown()

    if cur_usb_status == 0:
      count = count + 1
    else:
      count = 0

    time.sleep(SLEEP_TIMER)

def battery_optimisation():
  # a few system optimisation, may only effect from next reboot
  # Wi-Fi (scanning always available) off
  os.system('settings put global wifi_scan_always_enabled 0')
  # disable notify the user of open networks.
  os.system('settings put global wifi_networks_available_notification_on 0')
  # keep wifi on during sleep only when plugged in
  os.system('settings put global wifi_sleep_policy 1')
  # disable nfc
  os.system('LD_LIBRARY_PATH="" svc nfc disable')

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
  val = "1" if status == True else "0"
  os.system("echo " + val + " > " + "/sys/class/power_supply/battery/charging_enabled")

def set_max_cpu_freq(ismax=True):
  for i in range(0, 4):
    cpuid = str(i)
    if ismax:
      file = '/sys/devices/system/cpu/cpu' + cpuid + '/cpufreq/cpuinfo_max_freq'
    else:
      file = '/sys/devices/system/cpu/cpu' + cpuid + '/cpufreq/cpuinfo_min_freq'

    with open(file) as f:
      freq = f.read().rstrip()
      os.system("echo " + freq + " > " + "/sys/devices/system/cpu/cpu" + cpuid + "/cpufreq/scaling_max_freq")

if __name__ == "__main__":
  main()
