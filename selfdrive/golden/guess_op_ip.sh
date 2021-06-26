#!/bin/bash

OFFICE_WIFI_1="yes:CNSVWA1024"
OFFICE_WIFI_2="yes:DESKTOP-Golden"
OFFICE_WIFI_3="yes:DESKTOP-Rad"
HOME_WIFI=" fang_5G"
HOME_WIFI_2=" fang"
PHONE_WIFI="yes:golden"
OS=$(uname)
MAC_OS="Darwin"

if [ -z "$OP_IP" ]; then
    echo 'OP_IP not set'

    if [[ "$OS" == "$MAC_OS" ]]; then
        WIFI=$(/System/Library/PrivateFrameworks/Apple80211.framework/Resources/airport -I | awk -F: '/ SSID/{print $2}')
    else
        ifconfig | grep 192.168.5.10
        if [ $? -eq 0 ]; then
            echo "ethernet OP"
            export OP_IP=192.168.5.11
        else
            WIFI=$(nmcli -t -f active,ssid dev wifi | egrep '^yes' | cut -d\' -f2)
        fi

        ifconfig | grep 192.168.0.
        if [ $? -eq 0 ]; then
            echo "Router_SJ version"
            export OP_IP=192.168.0.138
        else
            WIFI=$(nmcli -t -f active,ssid dev wifi | egrep '^yes' | cut -d\' -f2)
        fi
    fi

    if [ -z "$OP_IP" ]; then
        echo $WIFI
        if [[ "$WIFI" == "$OFFICE_WIFI_1" ]]; then
            echo "office wifi"
            export OP_IP=192.168.137.138
        fi

        if [[ "$WIFI" == "$OFFICE_WIFI_2" ]]; then
            echo "office wifi"
            export OP_IP=192.168.137.138
        fi

        if [[ "$WIFI" == "$OFFICE_WIFI_3" ]]; then
            echo "office wifi"
            export OP_IP=192.168.137.138
        fi

        if [[ "$WIFI" == "$PHONE_WIFI" ]]; then
            echo "phone wifi"
            export OP_IP=192.168.43.138
        fi

        if [[ "$WIFI" == "$HOME_WIFI" ]]; then
            echo "home wifi"
            export OP_IP=192.168.3.138
            #launchctl setenv OP_IP "192.168.3.138"
        fi

        if [[ "$WIFI" == "$HOME_WIFI_2" ]]; then
            echo "home wifi"
            export OP_IP=192.168.3.138
            #launchctl setenv OP_IP "192.168.3.138"
        fi
    fi
else
    #echo 'OP_IP set ' $OP_IP
    #IP=$OP_IP
    echo ''
fi
