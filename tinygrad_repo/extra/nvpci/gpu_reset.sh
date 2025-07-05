GPU="$1"
echo 1 | sudo tee /sys/bus/pci/devices/$GPU/reset 2>/dev/null
