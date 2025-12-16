#!/usr/bin/env bash
set -u

pkill -f watch3
pkill -f compressed_vipc

touch /tmp/new_cam
BIG=1 /home/batman/openpilot/selfdrive/ui/watch3.py &

on_device() {
  local serial="$1"

  adb -s "$serial" shell 'su - comma -c "source /etc/profile && tmux kill-server"'

  case "$serial" in
    cdea8611) n=1 ;;
    e3a64917)  n=2 ;;
    63c61c84)  n=3 ;;
    *)        n=0 ;;
  esac

  PORTS="51336 57332 42305"
  for p in $PORTS; do
    adb -s $serial forward tcp:$((p + n)) tcp:$p
  done

  adb -s "$serial" push camera.sh /data
  adb -s "$serial" shell 'su - comma -c "source /etc/profile && sudo chown comma: /data/camera.sh && chmod +x /data/camera.sh"'

  camera_kill() {
    pkill -f "focusing_$n"
    adb -s "$serial" shell 'pkill -9 -f camerad'
    adb -s "$serial" shell 'pkill -9 -f encoderd'
    adb -s "$serial" shell 'pkill -9 -f bridge'
    adb -s "$serial" shell 'pkill -9 -f camera.sh'
  }

  camera_start() {
    camera_kill
    PORT_OFFSET=$n /home/batman/openpilot/tools/camerastream/compressed_vipc.py 127.0.0.1 --server="focusing_$n" &
    adb -s "$serial" shell 'su - comma -c "source /etc/profile && /data/camera.sh"' &
  }

  camera_start

  while :; do
    if [ -e "/tmp/kill_camera_$n" ]; then
      rm -f "/tmp/kill_camera_$n"
      echo "KILLING CAMERA $n"
      camera_kill
    fi
    if [ -e "/tmp/restart_camera_$n" ]; then
      rm -f "/tmp/restart_camera_$n"
      echo "STARTING CAMERA $n"
      camera_start
    fi
    sleep 0.2
  done

  pkill -f "focusing_$n"
}

declare -A connected=()

while true; do

  # connected
  declare -A now=()
  while read -r serial state; do
    [[ -z "${serial:-}" ]] && continue
    [[ "${state:-}" != "device" ]] && continue
    now["$serial"]=1
  done < <(adb devices | tail -n +2)

  # disconnected
  for serial in "${!connected[@]}"; do
    [[ -z "${now[$serial]+x}" ]] && unset "connected[$serial]" && echo "Disconnected: $serial"
  done

  # new connected
  for serial in "${!now[@]}"; do
    if [[ -z "${connected[$serial]+x}" ]]; then
      connected["$serial"]=1
      echo "Connected: $serial"
      on_device "$serial" &
    fi
  done

  sleep 0.1
done
