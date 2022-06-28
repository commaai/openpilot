#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR/..

TEST_TEXT="(WRAPPED_SOURCE_TEXT)"
TEST_TS_FILE=$DIR/../translations/main_test_en.ts

UNFINISHED="<translation type=\"unfinished\"><\/translation>"
TRANSLATED="<translation>$TEST_TEXT<\/translation>"

mkdir -p $DIR/../translations
rm -f $TEST_TS_FILE
lupdate -extensions cc,h -recursive '.' -ts $TEST_TS_FILE
sed -i "s/$UNFINISHED/$TRANSLATED/" $TEST_TS_FILE
lrelease $TEST_TS_FILE
