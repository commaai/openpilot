#!/bin/env bash
set -e

# install via curl
# curl -sL https://gist.github.com/chasebolt/fd5210b4c2a44a2b0db383162a66632c/raw/install_eon_purge_data.sh | bash

# create purge script
mkdir -p /data/local
cat <<'EOF' > /data/local/purge-data.sh
#!/bin/sh
set -e
# OP is disabled if free space is less than 15%
MAX_USED_PERCENT=80
# update $PATH to contain all the binaries we need access to when running from cron
PATH=/data/data/com.termux/files/usr/bin:/data/data/com.termux/files/usr/bin/applets:$PATH
echo -e "\t\tUsed\tMax"
# delete oldest entries until we are under the $MAX_USED_PERCENT limit
while true; do
  # get the used space on the device
  USED_PERCENT=$(df -h /data/media/0/realdata | tail -1 | awk '{print $5}' | sed 's/%$//')
  # check if the available space is lower than the max allowed
  if [[ USED_PERCENT -gt MAX_USED_PERCENT ]]; then
    echo -e "Deleting\t${USED_PERCENT}%\t${MAX_USED_PERCENT}%"
    # TODO: prioritize deleting driver monitoring videos first
    find /data/media/0/realdata/* -type f -not -name 'rlog.bz2' -print0 2>/dev/null | \
      xargs -r -0 ls -tr 2>/dev/null | \
      head -1 | \
      xargs rm -rf
  else
    echo -e "Complete\t${USED_PERCENT}%\t${MAX_USED_PERCENT}%"
    break
  fi
done
# delete empty directories
# find doesnt have -empty param so we have to use shell
find /data/media/0/realdata/* -type d -exec bash -c \
  'shopt -s nullglob; shopt -s dotglob; \
  a=("$1"/*); [[ ${a[@]} ]] || printf "%s\n" "$1"' sh {} \; 2>/dev/null | \
  xargs -r rmdir
EOF
chmod 0755 /data/local/purge-data.sh

# create cron job to run every 10 minutes.
mkdir -p /data/local/crontab
echo '*/10 * * * * /data/local/purge-data.sh' > /data/local/crontab/root

# install crond into userinit
cat <<EOF > /data/local/userinit.sh
#!/bin/env sh
/data/data/com.termux/files/usr/bin/applets/crond -c /data/local/crontab
EOF
chmod 0755 /data/local/userinit.sh

# cleanup previous versions
if [[ -f /system/etc/init.d/crond ]]; then
  mount -o rw,remount /dev/block/bootdevice/by-name/system /system
fi

rm -rf \
  /data/cleardata \
  /data/crontab \
  /system/etc/init.d/crond

if grep '\srw[\s,]' /proc/mounts | grep -q '\s/system\s'; then
  mount -o ro,remount /dev/block/bootdevice/by-name/system /system
fi

echo 'Install complete! Please reboot to start crond.'
