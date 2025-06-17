#!/bin/bash
clang++ ane.mm --shared -F /System/Library/PrivateFrameworks/ -framework ANEServices -framework IOSurface -framework Foundation -framework IOKit -framework ANECompiler -o libane.dylib

