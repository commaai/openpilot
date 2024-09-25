#!/usr/bin/env bash

while :
do
  rm selfdrive/modeld/models/supercombo_tinygrad.pkl
  ./system/manager/build.py 
done

