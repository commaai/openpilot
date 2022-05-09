#!/usr/bin/bash

nmcli connection modify --temporary lte gsm.auto-config yes
nmcli connection modify --temporary lte gsm.home-only yes
nmcli connection modify --temporary lte connection.autoconnect-retries 20

# restart modem
sudo systemctl stop ModemManager
/usr/comma/lte/lte.sh stop_blocking
sudo systemctl restart lte

#sudo systemctl restart ModemManager
sudo ModemManager --debug
