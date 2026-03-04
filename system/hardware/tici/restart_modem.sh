#!/usr/bin/env bash

#nmcli connection modify --temporary lte gsm.home-only yes
#nmcli connection modify --temporary lte gsm.auto-config yes
#nmcli connection modify --temporary lte connection.autoconnect-retries 20
sudo nmcli connection reload

nmcli con down lte
nmcli con down blue-prime

# power cycle modem
/usr/comma/lte/lte.sh stop_blocking
/usr/comma/lte/lte.sh start

sudo systemctl restart NetworkManager
sudo systemctl restart lte
