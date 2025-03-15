#!/bin/bash -e
g++ compile.mm -F /System/Library/PrivateFrameworks/ -framework ANECompiler -framework CoreFoundation
rm -f model.hwx
./a.out net.plist debug
rm -f context_switch_log.txt
log show --process a.out --last 1m --info --debug

