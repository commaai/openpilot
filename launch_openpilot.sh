#!/usr/bin/bash

rm -rf /data/media/0/realdata/*

echo -n 1 > /data/params/d/UploadRaw
echo -n 1 > /data/params/d/SshEnabled
echo -n 1 > /data/params/d/RecordFront
echo -n 1 > /data/params/d/RecordFrontLock
echo -n 1 > /data/params/d/CommunityFeaturesToggle
echo -n 2 > /data/params/d/HasAcceptedTerms
echo -n "0.2.0" > /data/params/d/CompletedTrainingVersion

rm -f /data/params/d/LastUpdate*
rm -f /data/params/d/UpdateFailedCount

setprop persist.neos.ssh 1
tools/scripts/setup_ssh_keys.py commaci2

export SKIP_FW_QUERY="1"
export FINGERPRINT="TOYOTA COROLLA TSS2 2019"
export LOG_TIMESTAMPS="1"

export PASSIVE="0"
exec ./launch_chffrplus.sh
