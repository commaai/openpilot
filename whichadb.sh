#!/usr/bin/env bash

if [ $# -ne 3 ]; then
  echo "Requires 3 serials";
  echo "Example Usage: ./whichadb.sh serial1 serial2 serial3"
  exit 1;
fi;

declare -a device_ordered=("$1" "$2" "$3")
declare -A devices=(["$1"]="(Orange) Device 1" ["$2"]="(Blue) Device 2" ["$3"]="(Green) Device 3")

while true; do
  mapfile -t connected < <(adb devices | tail -n +2 | cut -sf 1 | tr -d $'\n');
  for serial in "${device_ordered[@]}"; do
    if [[ ! " ${connected[*]} " =~ [[:space:]]${serial}[[:space:]] ]]; then
        echo "Restart ${devices[$serial]}"
    fi
  done;
  sleep 1;
  clear;
done;
