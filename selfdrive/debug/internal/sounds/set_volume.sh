#!/usr/bin/bash
while true
do
  service call audio 3 i32 3 i32 $1 i32 1
  sleep 1
done
