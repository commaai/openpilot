#!/usr/bin/env bash

# Restart modem without ModemManager — uses AT commands directly

# power cycle modem
/usr/comma/lte/lte.sh stop_blocking
/usr/comma/lte/lte.sh start

# restart NetworkManager (still used for WiFi)
sudo systemctl restart NetworkManager

# modem daemon (in hardwared) will automatically re-initialise
echo "Modem restarted. modem.py daemon will re-initialise."
