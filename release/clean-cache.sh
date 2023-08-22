#!/usr/bin/env bash
set -e

# delete outdated moc files
scons --dry-run --taskmastertrace /tmp/tasktrace.log >/dev/null
MOC_EXISTING_PROD="$(find /tmp/scons_cache/moc_files -type f | sort)"
MOC_CURRENT_PROD="$(egrep -o "'[^']*moc_files[^']*'" /tmp/tasktrace.log | sed  "s/'//g" | sort | uniq)"
MOC_JUNK="$(comm -23 <(echo "$MOC_EXISTING_PROD") <(echo "$MOC_CURRENT_PROD"))"
echo "$MOC_JUNK" | xargs -I{} rm {}
rm /tmp/tasktrace.log

# delete cache except for moc files
rm -rf $(find /tmp/scons_cache -maxdepth 1 ! -name moc_files ! -name scons_cache)

# repopulate cache 
scons --dry-run --cache-populate >/dev/null
