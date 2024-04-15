#!/bin/bash

SOURCE_DIR=$1
TARGET_DIR=$2

if [ -f /TICI ]; then
  FILES_SRC="release/files_tici"
else
  FILES_SRC="release/files_pc"
fi

cd $SOURCE_DIR
cp -pR --parents $(cat release/files_common) $TARGET_DIR/
cp -pR --parents $(cat $FILES_SRC) $TARGET_DIR/
