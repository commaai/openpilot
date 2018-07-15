
code_dir := $(shell pwd)

# TODO: Add a global build system

.PHONY: all
all:
	cd selfdrive && PYTHONPATH=$(code_dir) PREPAREONLY=1 ./manager.py

