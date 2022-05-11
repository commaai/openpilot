#!/usr/bin/bash

nmcli connection modify --temporary lte gsm.home-only yes
nmcli connection modify --temporary lte gsm.auto-config yes
nmcli connection modify --temporary lte connection.autoconnect-retries 20

sudo systemctl stop ModemManager

# full restart
#/usr/comma/lte/lte.sh stop_blocking
#sudo systemctl restart lte

# quick shutdown
/usr/comma/lte/lte.sh stop
nmcli con down lte

#sudo systemctl restart ModemManager
sudo ModemManager --debug
