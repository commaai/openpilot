route add default gw 192.168.5.1 && ndc network create 100 && ndc network interface add 100 eth0 && ndc resolver setnetdns 100 localdomain 8.8.8.8 8.8.4.4 && ndc network default set 100
