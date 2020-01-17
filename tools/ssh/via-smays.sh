# enp5s0 is the smays network name. Change it appropriately if you are using an ethernet adapter (type ifconfig to get the proper network name)
sudo ifconfig enp5s0 192.168.5.1 netmask 255.255.255.0
ssh -F config EON-smays
