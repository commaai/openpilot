#!/bin/sh

echo "Creating /data/cleardata/cleardata.sh"
# Create deletion script
mkdir /data/cleardata
echo "#!/bin/sh
# get the available space left on the device
size=\$(df -k /storage/emulated/0/Android/data | tail -1 | awk '{print \$4}')
# check if the available space is smaller than 10GB (10000000kB)
if ((\$size<10000000)); then
  # find all files currently saved by openpilot
  find /data/media/0/realdata/* -mmin +480 -exec rm -f {} \;
fi" > /data/cleardata/cleardata.sh

chmod +x /data/cleardata/cleardata.sh

# Create job file
echo "Creating /data/crontab/root"
mkdir /data/crontab
echo "
00 * * * * /data/cleardata/cleardata.sh" > /data/crontab/root

#start job
echo "Starting cron job"
crond -b -c /data/crontab


### For persistance after reboot
echo "Persisting cron job after reboot"
# Remount /system to be read/write
mount -o rw,remount /dev/block/bootdevice/by-name/system /system

# Use Busybox crond
echo "
crond -b -c /data/crontab" >> /system/etc/init.d/crond
chmod +x /system/etc/init.d/crond

# Remount /system to be read only
mount -o ro,remount /dev/block/bootdevice/by-name/system /system

echo "Done!"
