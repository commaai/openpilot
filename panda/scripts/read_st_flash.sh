#!/usr/bin/env bash
rm -f /tmp/dump_bootstub
rm -f /tmp/dump_main
dfu-util -a 0 -s 0x08000000 -U /tmp/dump_bootstub
dfu-util -a 0 -s 0x08004000 -U /tmp/dump_main

