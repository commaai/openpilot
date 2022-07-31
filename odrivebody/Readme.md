## Start body without fingerprinting:
`export SKIP_FW_QUERY=1 && export FINGERPRINT="COMMA BODY" && python selfdrive/manager/manager.py`

The following processes are disabled in the manager and have to be started manually:

## Start controlsd
`export SKIP_FW_QUERY=1 && export FINGERPRINT="COMMA BODY" && python selfdrive/controls/controlsd.py`

## Start canseriald
`cd selfdrive/canseriald`
`python canseriald.py`

## If Permission denied on /dev/ttyACM0: 
`sudo chmod 666 /dev/ttyACM0`