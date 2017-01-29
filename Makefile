.PHONY: all

# TODO: Add a global build system to openpilot
all:
	cd /data/openpilot/selfdrive && PYTHONPATH=/data/openpilot PREPAREONLY=1 /data/openpilot/selfdrive/manager.py

