#!/usr/bin/env  bash

SRC=$1
OUTPUT=$2
IGNORE_PATH=$3

echo "-- Copying docs & config to sphinx build folder..."
mkdir -p $OUTPUT
cd $SRC && \
    find . -type f \( -name "*.md" -o -name "*.rst" -o -name "*.png" -o -name "*.jpg" \) \
        -not -path "*/.*" \
        -not -path "./docs/build/*" \
        -not -path "./docs/sphinx/*" \
        -not -path "./xx/*" \
        -not -path "./yy/*" \
        -exec cp --parents "{}" $OUTPUT \;