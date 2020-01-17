#!/bin/sh
echo 1 > /proc/sys/net/ipv4/ip_forward
#iptables -t nat --delete-chain
iptables --flush
iptables -t nat --flush

# could be either one
iptables -t nat -A POSTROUTING -o v4-rmnet_data0 -j MASQUERADE
iptables -t nat -A POSTROUTING -o rmnet_data0 -j MASQUERADE

#iptables --delete-chain
#iptables -A INPUT -i eth0 -j ACCEPT
#iptables -A INPUT -i v4-rmnet_data0 -m state --state RELATED,ESTABLISHED -j ACCEPT
#iptables -A OUTPUT -j ACCEPT
#iptables -A FORWARD -i rmnet_data0 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT
#iptables -A FORWARD -i eth0 -o v4-rmnet_data0 -j ACCEPT

