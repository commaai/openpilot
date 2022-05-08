#!/usr/bin/bash

nmcli connection modify --temporary lte ipv4.route-metric 1
nmcli connection modify --temporary lte ipv6.route-metric 1
nmcli con up lte
