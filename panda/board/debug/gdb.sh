#!/usr/bin/env bash

gdb-multiarch  --eval-command="target extended-remote localhost:3333"
