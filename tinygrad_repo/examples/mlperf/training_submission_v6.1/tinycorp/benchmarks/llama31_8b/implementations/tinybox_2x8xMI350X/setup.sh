#!/usr/bin/env bash

for p in $(lspci -D -d ::0604 | cut -d" " -f1); do sudo -n setpci -s $p ECAP_ACS+0x6.w=0000:000c 2>/dev/null || true; done
