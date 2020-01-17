#!/bin/bash
#
# Copyright (C) 2007 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is a wrapper for smali.jar, so you can simply call "smali",
# instead of java -jar smali.jar. It is heavily based on the "dx" script
# from the Android SDK

# Set up prog to be the path of this script, including following symlinks,
# and set up progdir to be the fully-qualified pathname of its directory.
prog="$0"
while [ -h "${prog}" ]; do
    newProg=`/bin/ls -ld "${prog}"`

    newProg=`expr "${newProg}" : ".* -> \(.*\)$"`
    if expr "x${newProg}" : 'x/' >/dev/null; then
        prog="${newProg}"
    else
        progdir=`dirname "${prog}"`
        prog="${progdir}/${newProg}"
    fi
done
oldwd=`pwd`
progdir=`dirname "${prog}"`
cd "${progdir}"
progdir=`pwd`
prog="${progdir}"/`basename "${prog}"`
cd "${oldwd}"

jarfile=apktool.jar
libdir="$progdir"
if [ ! -r "$libdir/$jarfile" ]
then
    echo `basename "$prog"`": can't find $jarfile"
    exit 1
fi

javaOpts=""

# If you want DX to have more memory when executing, uncomment the following
# line and adjust the value accordingly. Use "java -X" for a list of options
# you can pass here.
# 
javaOpts="-Xmx512M -Dfile.encoding=utf-8"

# Alternatively, this will extract any parameter "-Jxxx" from the command line
# and pass them to Java (instead of to dx). This makes it possible for you to
# add a command-line parameter such as "-JXmx256M" in your ant scripts, for
# example.
while expr "x$1" : 'x-J' >/dev/null; do
    opt=`expr "$1" : '-J\(.*\)'`
    javaOpts="${javaOpts} -${opt}"
    shift
done

if [ "$OSTYPE" = "cygwin" ] ; then
	jarpath=`cygpath -w  "$libdir/$jarfile"`
else
	jarpath="$libdir/$jarfile"
fi

# add current location to path for aapt
PATH=$PATH:`pwd`;
export PATH;
exec java $javaOpts -jar "$jarpath" "$@"