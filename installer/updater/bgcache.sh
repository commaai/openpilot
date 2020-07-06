#!/usr/bin/bash

# Start background download of NEOS. Invoked by selfdrive/updated.py with
# a CWD inside the finalized copy of the update overlay. The NEOS updater
# takes care of handling corrupt files or resuming download of partials.

source ../../launch_env.sh
CURRENT_NEOS_VERSION="$(< /VERSION)"

update_status() {
  python -c "from common.params import Params; params = Params(); params.put('Offroad_NeosUpdate', '$UPDATING')"
}

if [ "$CURRENT_NEOS_VERSION" != "$REQUIRED_NEOS_VERSION" ]; then
  echo -n "update from $CURRENT_NEOS_VERSION to $REQUIRED_NEOS_VERSION; "
  if [ -f "/data/neoupdate/cache_success_marker" ]; then
    echo "already cached"
  else
    echo "starting download"
    UPDATING=1 update_status
    ./updater bgcache "file://update.json"
    UPDATING=0 update_status
  fi
else
  echo "no update required"
fi
