#!/usr/bin/env bash
nmcli connection modify --temporary esim ipv4.route-metric 1 ipv6.route-metric 1
nmcli con up esim
